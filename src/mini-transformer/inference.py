import re
import time
import mlflow
import transformers
from loguru import logger
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.seq2seq.utils import load_config, resolve_device, set_seed
from src.common.splits import load_splits, get_test_examples
from src.evaluation.metrics import evaluate, save_predictions, get_length_bins, evaluate_by_length, flatten_length_metrics, format_length_breakdown
from src.evaluation.efficiency import count_parameters, reset_peak_vram, get_peak_vram_mb, measure_ms, summarize_latency


def get_generation_kwargs(mt_cfg): # 디코딩 설정을 명시 (모델 기본값은 num_beams = 6이라 생략하면 의도와 다르게 beam으로 동작)
    strategy = mt_cfg["decoding_strategy"]
    if strategy not in ("greedy", "beam"):
        raise ValueError(f"알 수 없는 디코딩 전략: {strategy} (greedy | beam)")
    return {
        "num_beams": 1 if strategy == "greedy" else mt_cfg["beam"]["num_beams"],
        "max_new_tokens": mt_cfg["max_new_tokens"],
    }


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[.!?][\"'”’)\]])\s+") # 문장부호(+닫는 따옴표·괄호) 뒤 공백에서 분리


def split_segments(text, split_sentences): # 원문 → 줄별 번역 단위 [[문장, ...], ...] (줄바꿈은 번역 후 복원)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not split_sentences:
        return [[text.strip()]]
    return [[s for s in _SENTENCE_END.split(line) if s.strip()] for line in lines]


def generate_predictions(model, tokenizer, examples, device, mt_cfg): # test 예시 → records (RNN·LLM과 같은 형식)
    # opus-mt는 문장 단위로 학습돼, 문단을 통째로 넣으면 앞부분만 번역하고 끝냄 (30문단 확인: 출력 길이가 정답의 60%, beam은 36%)
    # → 실제 사용 방식대로 문장으로 나눠 번역한 뒤 다시 이어 붙임. 한 문단의 문장들은 한 번의 generate로 묶어 처리 (요청 1건 = 문단 1개)
    batch_size = mt_cfg["batch_size"] # 한 번에 처리할 문단 수
    max_length = mt_cfg["max_length"]
    split_sentences = mt_cfg.get("split_sentences", True)
    gen_kwargs = get_generation_kwargs(mt_cfg)
    records = []

    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        structures = [split_segments(e["source"], split_sentences) for e in batch]
        segments = [s for lines in structures for line in lines for s in line] # 배치 전체의 문장을 한 줄로 펼침

        enc = tokenizer(segments, truncation = True, max_length = max_length, padding = True, return_tensors = "pt")
        # 입력 토큰 수가 max_length에 닿았으면 잘린 입력 (길이별 평가에서 성능 하락 원인 구분용)
        segment_truncated = (enc["attention_mask"].sum(dim = 1) >= max_length).tolist()
        enc = enc.to(device)

        gen_ids, batch_ms = measure_ms(lambda: model.generate(**enc, **gen_kwargs), device)
        translations = [s.strip() for s in tokenizer.batch_decode(gen_ids, skip_special_tokens = True)]
        # 첫 토큰(decoder_start)을 뺀 출력에 EOS가 없으면 max_new_tokens에서 잘린 것 (LLM의 finish_reason과 같은 의미)
        segment_stopped = [bool((row[1:] == tokenizer.eos_token_id).any()) for row in gen_ids]

        cursor = 0
        for example, lines in zip(batch, structures):
            n = sum(len(line) for line in lines)
            outputs = iter(translations[cursor : cursor + n])
            hypothesis = "\n".join(" ".join(next(outputs) for _ in line) for line in lines) # 문장은 공백, 줄은 줄바꿈으로 복원
            records.append({
                "id": example["id"],
                "source": example["source"],
                "reference": example["reference"],
                "hypothesis": hypothesis,
                "latency_ms": round(batch_ms, 3), # batch_size > 1이면 배치 전체 시간
                "num_segments": n, # 문장 단위로 나눠 번역한 개수
                "source_truncated": any(segment_truncated[cursor : cursor + n]), # 한 문장이라도 잘렸으면 True
                "finish_reason": "stop" if all(segment_stopped[cursor : cursor + n]) else "length",
            })
            cursor += n

    return records


if __name__ == "__main__":
    device = resolve_device()
    logger.info(f"device: {device}")
    config = load_config("config/config.yaml")
    mt_cfg = config["mini_transformer"]
    set_seed(config["seed"])

    model_name = mt_cfg["model_name"]
    strategy = mt_cfg["decoding_strategy"]
    run_tag = f"{model_name.split('/')[-1].lower()}-{strategy}" # 예: opus-mt-ko-en-greedy

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    param_stats = {
        "num_params": count_parameters(model)["num_params"],
        "num_trainable_params": 0, # 사전학습 모델을 그대로 사용 → 이 과제를 위해 학습한 파라미터 없음 (LLM zero-shot과 같은 기준)
    }
    logger.info(f"# of model param: {param_stats['num_params']:,}")

    examples = get_test_examples(load_splits(config["data"], config["seed"]), mt_cfg["sample_size"])
    use_comet = mt_cfg.get("evaluation", {}).get("use_comet", False)
    length_bins = get_length_bins(config) # 원문 길이 구간 경계 (RNN·LLM 평가와 공유)

    reset_peak_vram(device)
    start_time = time.time()
    records = generate_predictions(model, tokenizer, examples, device, mt_cfg)
    elapsed = time.time() - start_time # 디코딩 시간만 측정 (저장·평가 제외)
    peak_vram_mb = get_peak_vram_mb(device)
    if peak_vram_mb is not None:
        logger.info(f"Peak VRAM: {peak_vram_mb:.1f} MB")

    latency_stats = summarize_latency([r["latency_ms"] for r in records])
    throughput = len(records) / elapsed # 초당 문장 처리 수
    logger.info(f"Latency p50 = {latency_stats['latency_p50_ms']:.1f} ms, p95 = {latency_stats['latency_p95_ms']:.1f} ms, throughput = {throughput:.2f} 문장/초")

    metrics = evaluate(records, use_comet = use_comet) # use_comet이면 records에 문장별 COMET 점수도 추가됨
    truncated_rate = sum(r["finish_reason"] == "length" for r in records) / len(records)
    source_truncated_rate = sum(r["source_truncated"] for r in records) / len(records)
    logger.info(
        f"Test 평가 결과({run_tag}): " + ", ".join(f"{k} = {v:.4f}" for k, v in metrics.items())
        + f", 출력 잘림 비율 = {truncated_rate:.1%}, 입력 잘림 비율 = {source_truncated_rate:.1%}"
    )
    length_breakdown = evaluate_by_length(records, length_bins)
    logger.info(format_length_breakdown(length_breakdown))

    predictions_path = f"outputs/predictions/{run_tag}.jsonl"
    save_predictions(records, predictions_path) # 평가 후 저장 → 문장별 COMET 점수까지 포함

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(mt_cfg["mlflow_experiment"])
    with mlflow.start_run(
        run_name = f"eval-{run_tag}",
        tags = {
            "run_type": "eval",
            "model_family": "mini-transformer", # rnn / mini-transformer / llm
            "method": "pretrained", # scratch / pretrained / zero-shot / lora
            "decoding": strategy,
            "model_name": model_name,
        },
    ):
        mlflow.log_params({
            "model_name": model_name,
            "sample_size": mt_cfg["sample_size"],
            "batch_size": mt_cfg["batch_size"],
            "max_length": mt_cfg["max_length"],
            "split_sentences": mt_cfg.get("split_sentences", True),
            "use_comet": use_comet,
            "length_bins": length_bins,
            "transformers_version": transformers.__version__,
            **{f"decode.{k}": v for k, v in get_generation_kwargs(mt_cfg).items()},
        })
        mlflow.log_metrics({
            "inference_time_sec": elapsed,
            **{f"test_{k}": v for k, v in metrics.items()}, # RNN·LLM 평가 run과 같은 이름
            **param_stats, # num_params, num_trainable_params
            **({"peak_vram_mb": peak_vram_mb} if peak_vram_mb is not None else {}), # cuda에서만 기록
            **latency_stats, # latency_p50_ms, latency_p95_ms, latency_mean_ms
            "throughput_sent_per_sec": throughput,
            "truncated_rate": truncated_rate, # LLM과 같은 이름: 출력이 max_new_tokens에서 잘린 비율
            "source_truncated_rate": source_truncated_rate,
            **flatten_length_metrics(length_breakdown), # test_chrf/len_201-300 등
        })
        mlflow.log_dict(length_breakdown, "length_breakdown.json") # 구간별 전체 표
        mlflow.log_artifact(predictions_path, artifact_path = "predictions")
    logger.info(f"Total Inference Time: {elapsed:.2f}초")
