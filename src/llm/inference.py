import gc
import time
import torch
import mlflow
import json

from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, __version__ as vllm_version
from loguru import logger

from src.common.utils import load_config, set_seed
from src.data_pipeline.loader import get_eval_examples, TAG_COLUMNS
from src.llm.prompt import build_prompt
from src.llm.postprocess import clean_translation
from src.evaluation.metrics import evaluate, save_predictions, get_target_lang, bleu_tokenize, SLICES, evaluate_by_slice, flatten_slice_metrics, format_slice_breakdown
from src.evaluation.efficiency import summarize_latency, hf_model_stats


def generate_predictions(llm, tokenizer, examples, llm_cfg, direction): # test 예시 → records (RNN과 같은 형식 + LLM 전용 필드)
    prompts = [build_prompt(e["source"], tokenizer, llm_cfg, direction) for e in examples]
    params = SamplingParams(
        temperature = llm_cfg["inference"]["generation"]["temperature"],
        max_tokens = llm_cfg["inference"]["generation"]["max_tokens"],
    )
    
    start = time.perf_counter()
    outputs = llm.generate(prompts, params) # 입력 순서대로 결과가 돌아옴
    batch_elapsed = time.perf_counter() - start # throughput 계산용

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
            **{k: example[k] for k in TAG_COLUMNS},
        })
    return records, batch_elapsed


def measure_latency(llm, tokenizer, examples, llm_cfg, direction): # 한 건씩 생성 → 요청 하나의 응답 시간(ms)
    params = SamplingParams(
        temperature = llm_cfg["inference"]["generation"]["temperature"],
        max_tokens = llm_cfg["inference"]["generation"]["max_tokens"],
    )
    latencies = {}
    for example in examples:
        prompt = build_prompt(example["source"], tokenizer, llm_cfg, direction)
        start = time.perf_counter()
        llm.generate([prompt], params, use_tqdm = False) # 결과는 버리고 시간만 사용 (품질 평가는 일괄 생성 결과로)
        latencies[example["id"]] = (time.perf_counter() - start) * 1000
    return latencies


if __name__ == "__main__":
    config = load_config("config/config.yaml")
    llm_config = config["model"]["llm"]
    
    test_set = config["evaluation"]["test_set"]
    direction = llm_config["languages"][config["data"]["direction"]] # ("Korean", "English") 등 → 프롬프트의 원문·번역 언어
    target_lang = get_target_lang(config) # BLEU 토큰화 선택용 ("en" | "ko")
    set_seed(config["seed"])

    checkpoint = llm_config["inference"]["checkpoint"]
    model_name = checkpoint or llm_config["base_model"] # null일 경우 base모델 그대로
    lora_info = None
    
    if checkpoint:
        lora_info_path = Path(checkpoint) / "lora_info.json"
        lora_info = json.loads(lora_info_path.read_text()) if lora_info_path.exists() else None
    
    method = "lora" if lora_info else "zero-shot"
    run_tag = f"{model_name.split('/')[-1].lower()}-{'lora' if lora_info else 'zeroshot'}-{test_set}-{config['data']['direction']}" # 예: qwen3-1.7b-lora-natural-ko2en | qwen3-1.7b-natural-ko2en

    examples = get_eval_examples(config) # test 전체
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(
        model = model_name,
        dtype = llm_config["inference"]["vllm"]["dtype"],
        max_model_len = llm_config["inference"]["vllm"]["max_model_len"],
        gpu_memory_utilization = llm_config["inference"]["vllm"]["gpu_memory_utilization"],
        seed = config["seed"],
        enable_prefix_caching = llm_config["inference"]["vllm"]["enable_prefix_caching"],
    )

    records, batch_elapsed = generate_predictions(llm, tokenizer, examples, llm_config, direction)
    
    # latency: test 전체를 한 건씩 생성해 측정 (seq2seq·소형 Transformer와 같은 문장 전체로 비교, 구간별 지연도 계산 가능)
    # 프롬프트 수정 중 품질만 빠르게 볼 때는 config의 inference.measure_latency: false로 끔
    do_latency = llm_config["inference"]["measure_latency"]
    latencies = measure_latency(llm, tokenizer, examples, llm_config, direction) if do_latency else {}
    for r in records:
        r["latency_ms"] = round(latencies[r["id"]], 3) if r["id"] in latencies else None # 측정 안 한 문장은 None
        
    predictions_path = f"outputs/predictions/{run_tag}.jsonl"
    save_predictions(records, predictions_path) # 생성 비용이 크므로 평가(COMET) 전에 먼저 저장해 둠

    del llm # COMET이 GPU를 쓸 수 있도록 vLLM이 잡아둔 메모리 해제
    gc.collect()
    torch.cuda.empty_cache()
    free_gb, total_gb = (x / 1024**3 for x in torch.cuda.mem_get_info())
    logger.info(f"vLLM 해제 후 GPU 여유 메모리: {free_gb:.1f} / {total_gb:.1f} GB")

    use_comet = config["evaluation"].get("use_comet", False)
    metrics = evaluate(records, target_lang, use_comet = use_comet) # use_comet이면 records에 문장별 COMET 점수도 추가됨
    # 구간별 지표: 서브도메인·길이(en_words)·숫자/부정/조건 (evaluate 다음에 호출해야 문장별 COMET 평균이 들어감)
    slice_breakdowns = {name: evaluate_by_slice(records, key_fn, target_lang) for name, key_fn in SLICES.items()}
    if use_comet:
        save_predictions(records, predictions_path) # 문장별 COMET 점수를 포함해 다시 저장
    
    # 효율 지표 계산 및 기록
    measured = [r["latency_ms"] for r in records if r["latency_ms"] is not None]
    latency_stats = summarize_latency(measured) if measured else {} # 측정을 끈 실행은 지연 지표 없이 기록
    total_output_tokens = sum(r["num_output_tokens"] for r in records)
    efficiency = {
        **latency_stats, # latency_p50_ms, latency_p95_ms, latency_mean_ms (RNN과 같은 이름)
        "throughput_sent_per_sec": len(records) / batch_elapsed, # RNN과 같은 이름
        "output_tokens_per_sec": total_output_tokens / batch_elapsed, # LLM 전용: 출력 길이 차이를 감안한 속도
        **hf_model_stats(model_name, llm_config["inference"]["vllm"]["dtype"]), # num_params, weight_vram_mb
        "num_trainable_params": lora_info["num_trainable_params"] if lora_info else 0, # zero-shot은 학습한 파라미터 없음
    }
    logger.info("효율: " + ", ".join(f"{k} = {v:,.2f}" for k, v in efficiency.items()))
    
    truncated_rate = sum(r["finish_reason"] == "length" for r in records) / len(records)
    logger.info(f"Test 평가 결과({run_tag}): " + ", ".join(f"{k} = {v:.4f}" for k, v in metrics.items()) + f", 잘린 비율 = {truncated_rate:.1%}")
    logger.info(format_slice_breakdown(slice_breakdowns))

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(llm_config["mlflow_experiment"])
    
    with mlflow.start_run(
        run_name = f"eval-{run_tag}",
        parent_run_id = lora_info["train_run_id"] if lora_info else None, # LoRA는 학습 run의 자식으로 (RNN과 같은 구조)
        tags = {
            "run_type": "eval",
            "model_family": "llm",
            "method": method,
            "decoding": "greedy",
            "model_name": model_name,
            "test_set": test_set, # natural / balanced
        },
    ):
        mlflow.log_params({
            "model_name": model_name,
            "prompt_version": llm_config["prompt"]["version"],
            "num_examples": len(examples), # 평가한 문장 수 (test 전체)
            "max_tokens": llm_config["inference"]["generation"]["max_tokens"],
            "max_model_len": llm_config["inference"]["vllm"]["max_model_len"],
            "vllm_version": vllm_version,
            "measure_latency": do_latency, # 한 건씩 지연을 측정했는지 (true면 test 전체)
            "enable_prefix_caching": llm_config["inference"]["vllm"]["enable_prefix_caching"],
            "slices": list(SLICES), # 구간별 지표를 낸 기준 (subdomain, length, has_number 등)
            "direction": config["data"]["direction"],
            "bleu_tokenize": bleu_tokenize(target_lang), # 토큰화가 다르면 BLEU 숫자끼리 비교 불가
            "model_path": model_name, # 실제로 로드한 경로 (병합 모델이면 로컬 폴더)
            "gpu_memory_utilization": llm_config["inference"]["vllm"]["gpu_memory_utilization"], # throughput에 영향
        })
        mlflow.log_metrics({
            **{f"test_{k}": v for k, v in metrics.items()}, # RNN 평가 run과 같은 이름
            "truncated_rate": truncated_rate,
            **efficiency, # latency_p50/p95/mean_ms, throughput_sent_per_sec, output_tokens_per_sec, num_params, weight_vram_mb, num_trainable_params
            **flatten_slice_metrics(slice_breakdowns), # test_chrf/subdomain_금융_보험, test_chrf/length_short 등
        })
        mlflow.log_dict(slice_breakdowns, "slice_breakdown.json") # 구간별 전체 표 (구간 이름은 원래 이름)
        mlflow.log_text(llm_config["prompt"]["system"] + "\n\n" + llm_config["prompt"]["user"], "prompt.txt") # 프롬프트 전문
        mlflow.log_artifact(predictions_path, artifact_path = "predictions")