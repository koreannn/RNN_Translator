import tempfile
import mlflow
import pandas as pd
import torch
import transformers
from pathlib import Path
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature
from transformers import AutoTokenizer

from src.seq2seq.utils import load_config
from src.serving.rnn_pyfunc import RNNTranslator


def latest_eval_metrics(experiment_name, decoding): # 해당 디코딩의 최신 평가 run 지표 (run ID를 코드에 박지 않음)
    runs = mlflow.search_runs(
        experiment_names = [experiment_name],
        filter_string = f"tags.run_type = 'eval' AND tags.decoding = '{decoding}'",
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
    serving_cfg = config["serving"]
    model_name = serving_cfg["registered_model_name"]
    experiment_name = config["mlflow"]["experiment_name"]

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(experiment_name)

    # 모델 안에 함께 저장할 디코딩 설정 (평가 때와 같은 값)
    model_config = {
        "max_length": config["inference"]["max_length"],
        "max_new_tokens": config["inference"]["max_new_token"],
        "beam_size": config["inference"]["beam"]["beam_size"],
        "beam_alpha": config["inference"]["beam"]["alpha"],
        "default_decoding": serving_cfg["default_decoding"],
    }

    # signature: 입력·출력 형식과 params(디코딩 선택, 기본값)를 명시 → 서빙 시 형식이 다른 요청을 걸러냄
    input_example = pd.DataFrame({"text": ["나는 오늘 도서관에서 책을 읽었다.", "서울은 대한민국의 수도이다."]})
    signature = infer_signature(
        input_example,
        pd.DataFrame({"translation": ["i read a book at the library today.", "seoul is the capital of korea."]}),
        params = {"decoding": serving_cfg["default_decoding"]},
    )

    with tempfile.TemporaryDirectory() as tmp:
        # 토크나이저를 파일로 저장해 모델에 포함 → 서빙 환경에서 인터넷 없이 로드, 버전도 고정
        tokenizer_dirs = {}
        for key, name in (("kor_tokenizer", config["model"]["kor_tokenizer"]), ("en_tokenizer", config["model"]["en_tokenizer"])):
            tokenizer_dirs[key] = str(Path(tmp) / key)
            AutoTokenizer.from_pretrained(name).save_pretrained(tokenizer_dirs[key])

        with mlflow.start_run(
            run_name = "package-rnn",
            tags = {"run_type": "package", "model_family": "rnn", "source_train_run_id": serving_cfg["source_train_run_id"]},
        ):
            model_info = mlflow.pyfunc.log_model(
                name = "model",
                python_model = RNNTranslator(),
                artifacts = {"checkpoint": serving_cfg["checkpoint_path"], **tokenizer_dirs},
                model_config = model_config,
                code_paths = ["src"], # 모델 안에서도 'from src.seq2seq ...' import가 그대로 동작
                signature = signature,
                input_example = input_example, # 저장 직후 이 예시로 실제 predict를 돌려 포장이 동작하는지 검증
                pip_requirements = [f"torch=={torch.__version__}", f"transformers=={transformers.__version__}", "pandas"],
                registered_model_name = model_name, # 저장과 동시에 Registry에 새 버전으로 등록
            )
            mlflow.log_params(model_config)

    # 등록된 버전에 평가 결과를 태그로 남기고 @champion 지정
    client = MlflowClient()
    version = model_info.registered_model_version
    for decoding in ("greedy", "beam"):
        eval_run_id, metrics = latest_eval_metrics(experiment_name, decoding)
        if eval_run_id is None:
            logger.warning(f"{decoding} 평가 run을 찾지 못해 태그를 생략합니다.")
            continue
        client.set_model_version_tag(model_name, version, f"eval_run_{decoding}", eval_run_id)
        for k, v in metrics.items():
            client.set_model_version_tag(model_name, version, f"{decoding}.{k}", str(v))
    client.set_registered_model_alias(model_name, "champion", version)
    logger.info(f"등록 완료: models:/{model_name}@champion (version {version})")