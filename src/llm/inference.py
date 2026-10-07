import gc
import time
import torch
import mlflow
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, __version__ as vllm_version
from loguru import logger

from src.seq2seq.utils import load_config, set_seed
from src.common.splits import load_splits, get_test_examples
from src.llm.prompt import build_prompt
from src.llm.postprocess import clean_translation
from src.evaluation.metrics import evaluate, save_predictions


def generate_predictions(llm, tokenizer, examples, llm_cfg): # test 예시 → records (RNN과 같은 형식 + LLM 전용 필드)
    prompts = [build_prompt(e["source"], tokenizer, llm_cfg) for e in examples]
    params = SamplingParams(
        temperature = llm_cfg["generation"]["temperature"],
        max_tokens = llm_cfg["generation"]["max_tokens"],
    )
    outputs = llm.generate(prompts, params) # 입력 순서대로 결과가 돌아옴

    records = []
    for example, out in zip(examples, outputs):
        completion = out.outputs[0]
        records.append({
            "id": example["id"],
            "source": example["source"],
            "reference": example["reference"],
            "hypothesis": clean_translation(completion.text),
            "raw_output": completion.text, # 후처리 전 원본 (후처리가 무엇을 바꿨는지 추적용)
            "num_output_tokens": len(completion.token_ids),
            "finish_reason": completion.finish_reason, # "length"면 max_tokens에서 잘림 (반복 루프 등)
        })
    return records


if __name__ == "__main__":
    config = load_config("config/config.yaml")
    llm_cfg = config["llm"]
    set_seed(config["seed"])
    model_name = llm_cfg["model_name"]
    run_tag = f"{model_name.split('/')[-1].lower()}-zeroshot" # 예: qwen3-0.6b-zeroshot

    examples = get_test_examples(load_splits(config["data"], config["seed"]), llm_cfg["sample_size"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(
        model = model_name,
        dtype = llm_cfg["vllm"]["dtype"],
        max_model_len = llm_cfg["vllm"]["max_model_len"],
        gpu_memory_utilization = llm_cfg["vllm"]["gpu_memory_utilization"],
        seed = config["seed"],
    )

    records = generate_predictions(llm, tokenizer, examples, llm_cfg)
    predictions_path = f"outputs/predictions/{run_tag}.jsonl"
    save_predictions(records, predictions_path)

    del llm # COMET이 GPU를 쓸 수 있도록 vLLM이 잡아둔 메모리 해제
    gc.collect()
    torch.cuda.empty_cache()
    free_gb, total_gb = (x / 1024**3 for x in torch.cuda.mem_get_info())
    logger.info(f"vLLM 해제 후 GPU 여유 메모리: {free_gb:.1f} / {total_gb:.1f} GB")

    use_comet = llm_cfg.get("evaluation", {}).get("use_comet", False)
    metrics = evaluate(records, use_comet = use_comet)
    truncated_rate = sum(r["finish_reason"] == "length" for r in records) / len(records)
    logger.info(f"Test 평가 결과({run_tag}): " + ", ".join(f"{k} = {v:.4f}" for k, v in metrics.items()) + f", 잘린 비율 = {truncated_rate:.1%}")

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(llm_cfg["mlflow_experiment"])
    with mlflow.start_run(
        run_name = f"eval-{run_tag}",
        tags = {
            "run_type": "eval",
            "model_family": "llm",
            "method": "zero-shot",
            "decoding": "greedy",
            "model_name": model_name,
        },
    ):
        mlflow.log_params({
            "model_name": model_name,
            "prompt_version": llm_cfg["prompt"]["version"],
            "sample_size": llm_cfg["sample_size"],
            "max_tokens": llm_cfg["generation"]["max_tokens"],
            "max_model_len": llm_cfg["vllm"]["max_model_len"],
            "vllm_version": vllm_version,
        })
        mlflow.log_metrics({
            **{f"test_{k}": v for k, v in metrics.items()}, # RNN 평가 run과 같은 이름
            "truncated_rate": truncated_rate,
        })
        mlflow.log_text(llm_cfg["prompt"]["system"] + "\n\n" + llm_cfg["prompt"]["user"], "prompt.txt") # 프롬프트 전문
        mlflow.log_artifact(predictions_path, artifact_path = "predictions")