import mlflow
import pandas as pd
import peft
import torch
import transformers
from pathlib import Path
from huggingface_hub import HfApi
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature

from src.seq2seq.utils import load_config
from src.serving.lora_pyfunc import LoRATranslator


def latest_eval_metrics(experiment_name): # LoRA 평가 run 중 가장 최근 것의 지표 (run ID를 코드에 박지 않음)
    runs = mlflow.search_runs(
        experiment_names = [experiment_name],
        filter_string = "tags.run_type = 'eval' AND tags.method = 'lora'", # LLM 평가 run은 method 태그로 zero-shot/lora 구분
        order_by = ["attributes.start_time DESC"],
        max_results = 1,
    )
    if runs.empty:
        return None, {}
    row = runs.iloc[0]
    keys = ["test_bleu", "test_chrf", "test_comet", "latency_p50_ms"]
    return row["run_id"], {k: round(float(row[f"metrics.{k}"]), 4) for k in keys if f"metrics.{k}" in row}


if __name__ == "__main__":
    config = load_config("config/config.yaml")
    llm_cfg = config["llm"]
    lora_serving = config["serving"]["lora"]
    model_name = lora_serving["registered_model_name"]
    experiment_name = llm_cfg["mlflow_experiment"]
    adapter_dir = Path(lora_serving["adapter_dir"])
    base_model = llm_cfg["lora"]["base_model"]
    train_run_id = (adapter_dir / "mlflow_run_id.txt").read_text().strip() # LoRA 학습 run

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(experiment_name)

    # base 모델의 현재 커밋 해시 → 불러올 때 이 버전으로 고정 (Hub 모델이 바뀌어도 같은 가중치)
    base_model_revision = HfApi().model_info(base_model).sha

    model_config = {
        "base_model": base_model,
        "base_model_revision": base_model_revision,
        "max_new_tokens": lora_serving["max_new_token"],
        "prompt": {"system": llm_cfg["prompt"]["system"], "user": llm_cfg["prompt"]["user"]},
        "enable_thinking": llm_cfg["generation"]["enable_thinking"],
    }

    input_example = pd.DataFrame({"text": ["나는 오늘 도서관에서 책을 읽었다.", "서울은 대한민국의 수도이다."]})
    signature = infer_signature(
        input_example,
        pd.DataFrame({"translation": ["I read a book at the library today.", "Seoul is the capital of South Korea."]}),
    )

    with mlflow.start_run(
        run_name = "package-qwen3-1.7b-lora",
        parent_run_id = train_run_id, # 학습 run 아래에 포장 run을 붙임 (학습 → 평가 / 포장)
        tags = {"run_type": "package", "model_family": "llm", "method": "lora", "model_name": base_model},
    ):
        model_info = mlflow.pyfunc.log_model(
            name = "model",
            python_model = LoRATranslator(),
            artifacts = {"adapter": str(adapter_dir)}, # adapter + 토크나이저만 (base 가중치는 불러올 때 Hub에서)
            model_config = model_config,
            code_paths = ["src"], # 모델 안에서도 'from src.llm ...' import가 그대로 동작
            signature = signature,
            input_example = input_example, # 저장 직후 실제로 불러와 번역해 보며 포장을 검증
            pip_requirements = [
                f"torch=={torch.__version__}",
                f"transformers=={transformers.__version__}",
                f"peft=={peft.__version__}",
                "pandas",
            ],
            registered_model_name = model_name, # 저장과 동시에 Registry에 새 버전으로 등록
        )
        mlflow.log_params({k: v for k, v in model_config.items() if k != "prompt"}) # 프롬프트 전문은 아래 텍스트로
        mlflow.log_text(model_config["prompt"]["system"] + "\n\n" + model_config["prompt"]["user"], "prompt.txt")

    # 등록된 버전에 평가 결과를 태그로 남기고 @champion 지정
    client = MlflowClient()
    
    # @champion을 옮기기 전에 등록된 모델을 실제로 불러와 번역해 봄 (input_example 검증은 실패해도 경고만 남기므로)
    smoke = mlflow.pyfunc.load_model(model_info.model_uri).predict(input_example.head(1))
    if not smoke["translation"][0].strip():
        raise RuntimeError("등록된 모델의 번역 결과가 비어 있습니다. @champion을 옮기지 않습니다.")
    logger.info(f"스모크 테스트 통과: {smoke['translation'][0]}")
    
    version = model_info.registered_model_version
    eval_run_id, metrics = latest_eval_metrics(experiment_name)
    if eval_run_id is None:
        logger.warning("LoRA 평가 run을 찾지 못해 태그를 생략합니다.")
    else:
        client.set_model_version_tag(model_name, version, "eval_run", eval_run_id)
        for k, v in metrics.items():
            client.set_model_version_tag(model_name, version, k, str(v))
    client.set_registered_model_alias(model_name, "champion", version)
    logger.info(f"등록 완료: models:/{model_name}@champion (version {version})")