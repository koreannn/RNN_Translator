import time
import random
import json
import torch
import wandb
import mlflow

from loguru import logger
from pathlib import Path
from transformers import AutoTokenizer
from src.seq2seq.model import build_model
from src.seq2seq.utils import load_config, resolve_device, set_seed
from src.seq2seq.decoding import get_special_token_ids, greedy_decoding, beam_decoding, sampling_decoding
from src.evaluation.metrics import evaluate
from src.evaluation.efficiency import count_parameters, reset_peak_vram, get_peak_vram_mb, measure_ms, summarize_latency
from src.seq2seq.dataloader import CustomDataLoader

def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location = "cpu")
    get_model_config(checkpoint) # 모델 복원에 필요한 키가 모두 있는지 미리 검증

    if "seq2seq_state_dict" not in checkpoint:
        raise KeyError("Missing keys in checkpoint: ['seq2seq_state_dict']")
    return checkpoint

def get_model_config(checkpoint):
    if "model_config" in checkpoint:
        return checkpoint["model_config"]

    # 구버전 체크포인트 호환 (모델 구조 정보가 최상위 키로 흩어져 저장되던 형식)
    required = ["embedding_dim", "hidden_dim", "kor_vocab_size", "en_vocab_size"]
    missing = [k for k in required if k not in checkpoint]
    if missing:
        raise KeyError(f"Missing keys in checkpoint: {missing}")

    return {
        "kor_vocab_size": int(checkpoint["kor_vocab_size"]),
        "en_vocab_size": int(checkpoint["en_vocab_size"]),
        "embedding_dim": int(checkpoint["embedding_dim"]),
        "hidden_dim": int(checkpoint["hidden_dim"]),
        "init_scheme": checkpoint.get("init_scheme", "default"),
        "use_layer_norm": checkpoint.get("use_layer_norm", False),
        "padding_id": int(checkpoint.get("pad_token_id", 0)),
    }

def get_model_from_checkpoint(checkpoint, device):
    model = build_model(**get_model_config(checkpoint))
    model.load_state_dict(checkpoint["seq2seq_state_dict"], strict = True)
    model.to(device)
    model.eval()
    return model

def translate_sentence( # Streamlit 대시보드용
    model,
    kor_tokenizer,
    en_tokenizer,
    device,
    text,
    max_length,
    max_new_tokens,
):
    ids = get_special_token_ids(kor_tokenizer, en_tokenizer)
    model.eval()
    
    src_ids = kor_tokenizer(
        text,
        truncation = True,
        max_length = max_length,
        return_tensors = "pt",
    )["input_ids"].to(device)
    gen_ids = greedy_decoding(
        model,
        src_ids,
        max_new_tokens = max_new_tokens,
        **ids,
    )
    
    return en_tokenizer.decode(gen_ids[0], skip_special_tokens = True).strip()


def generate_predictions( # test 로더를 돌며 번역 결과를 records로 수집 (점수 계산은 하지 않음)
    model,
    kor_tokenizer,
    en_tokenizer,
    device,
    test_dataloader,
    strategy, # "greedy" | "beam" | "hybrid"
    max_length,
    max_new_tokens,
    decode_kwargs = None, # 전략별 하이퍼파라미터 (beam: beam_size, alpha / hybrid: temperature, top_k, top_p)
    sample_size = None,
):
    if strategy not in ("greedy", "beam", "hybrid"):
        raise ValueError(f"알 수 없는 디코딩 전략: {strategy} (greedy | beam | hybrid)")

    ids = get_special_token_ids(kor_tokenizer, en_tokenizer)
    decode_kwargs = decode_kwargs or {}
    records = [] # [{"id", "source", "reference", "hypothesis"}, ...]

    for src_ids, _, _, src_text, tgt_text in test_dataloader:
        src_ids = src_ids.to(device)

        if strategy == "greedy": # 배치 단위
            gen_ids, batch_ms = measure_ms(
                lambda: greedy_decoding(model, src_ids, max_new_tokens = max_new_tokens, **ids),
                device
            )
            hypotheses = [s.strip() for s in en_tokenizer.batch_decode(gen_ids, skip_special_tokens = True)]
            latencies = [batch_ms] * len(hypotheses)
        else: # beam / hybrid는 문장 단위
            decode_fn = beam_decoding if strategy == "beam" else sampling_decoding
            hypotheses, latencies = [], []
            for i in range(src_ids.size(0)):
                if sample_size is not None and len(records) + len(hypotheses) >= sample_size:
                    break
                gen_ids, ms = measure_ms(
                    lambda: decode_fn(
                        model,
                        src_ids[i : i + 1],
                        max_new_tokens = max_new_tokens,
                        max_length = max_length,
                        **ids,
                        **decode_kwargs,
                    ),
                    device,
                )
                
                hypotheses.append(en_tokenizer.decode(gen_ids, skip_special_tokens = True).strip())
                latencies.append(ms)

        if sample_size is not None: # greedy는 배치 단위라 sample_size를 넘칠 수 있으므로 잘라냄
            hypotheses = hypotheses[: sample_size - len(records)]

        for source, reference, hypothesis, latency_ms in zip(src_text, tgt_text, hypotheses, latencies):
            records.append({
                "id": len(records), # test split 내 순번 (test 로더는 shuffle = False)
                "source": source,
                "reference": reference,
                "hypothesis": hypothesis,
                "latency_ms": round(latency_ms, 3),
            })

        if hypotheses:
            sample_idx = random.randrange(len(hypotheses))
            logger.info(f"번역 전 문장(예시): {src_text[sample_idx]}")
            logger.info(f"번역된 문장(예시): {hypotheses[sample_idx]}")

        if sample_size is not None and len(records) >= sample_size:
            break

    return records


def save_predictions(records, path):
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    
    with open(path, "w", encoding = "utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii = False) + "\n")
    
    logger.info(f"예측 결과 저장: {path} ({len(records)} 문장)")


if __name__ == "__main__":
    device = resolve_device()
    logger.info(f"device: {device}")
    # logger.add(f"logs/{wandb_exp_name}", encoding = "utf-8")
    config = load_config("config/config.yaml")
    set_seed(config["seed"])

    # h_param
    max_length = config["inference"]["max_length"]
    max_n_token = config["inference"]["max_new_token"]
    batch_size = config["inference"]["batch_size"]
    
    # config
    model_checkpoint_path = config["inference"]["checkpoint_path"]
    kor_tokenizer_name = config["model"]["kor_tokenizer"]
    en_tokenizer_name = config["model"]["en_tokenizer"]
    
    # wandb 세팅
    wandb.init(
        entity = config["wandb"]["wandb_entity"],
        project = config["wandb"]["wandb_project"],
        name = f"inference-{Path(model_checkpoint_path).stem}",
        job_type = "inference",
        config = {
            "architecture": config["wandb"]["wandb_architecture"],
            "checkpoint_path": model_checkpoint_path,
            "batch_size": batch_size,
        },
    )
    
    kor_tokenizer = AutoTokenizer.from_pretrained(kor_tokenizer_name)
    en_tokenizer = AutoTokenizer.from_pretrained(en_tokenizer_name)

    checkpoint = load_checkpoint(model_checkpoint_path, device = device)
    logger.info(f"Loaded checkpoint from {model_checkpoint_path}")

    # 평가 run을 붙일 부모(학습) run: config 값 우선, 없으면 체크포인트에 저장된 학습 run ID
    config_run_id = config["inference"].get("source_run_id")
    ckpt_run_id = checkpoint.get("mlflow_run_id")
    if config_run_id and ckpt_run_id and config_run_id != ckpt_run_id:
        logger.warning(f"config의 source_run_id({config_run_id})와 체크포인트의 run ID({ckpt_run_id})가 다릅니다. config 값을 사용합니다.")
    source_run_id = config_run_id or ckpt_run_id
    if source_run_id is None:
        logger.warning("학습 run ID를 찾지 못해 평가 결과를 독립 run으로 기록합니다.")

    model = get_model_from_checkpoint(checkpoint, device = device)
    del checkpoint # CPU 메모리 해제용(GPU에는 변화 없음)
    param_stats = count_parameters(model)
    logger.info(f"# of model param: {param_stats['num_params']:,}")
    
    dataloader = CustomDataLoader(kor_tokenizer, en_tokenizer, max_length = max_length, batch_size = batch_size)
    _, _, test_dataloader = dataloader.get_data_loader() # test의 데이터로더는 1개씩 들어가도록 고정되어있음
    
    strategy = config["inference"]["decoding_strategy"]
    decode_kwargs = config["inference"].get(strategy, {}) # greedy는 하이퍼파라미터 섹션이 없으므로 {}
    sample_size = config["inference"]["sample_size"]
    use_comet = config["inference"].get("evaluation", {}).get("use_comet", False) # COMET은 GPU 권장 (CPU에선 매우 느림)
    
    reset_peak_vram(device)
    start_time = time.time()
    records = generate_predictions(
        model,
        kor_tokenizer,
        en_tokenizer,
        device,
        test_dataloader,
        strategy = strategy,
        max_length = max_length,
        max_new_tokens = max_n_token,
        decode_kwargs = decode_kwargs,
        sample_size = sample_size,
    )
    elapsed = time.time() - start_time # 디코딩 시간만 측정 (저장·평가 제외)
    peak_vram_mb = get_peak_vram_mb(device)
    if peak_vram_mb is not None:
        logger.info(f"Peak VRAM: {peak_vram_mb:.1f} MB")

    # 문장 추론 latency
    latency_stats = summarize_latency([r["latency_ms"] for r in records])
    throughput = len(records) / elapsed # 초당 문장 처리 수
    logger.info(f"Latency p50 = {latency_stats['latency_p50_ms']:.1f} ms, p95 = {latency_stats['latency_p95_ms']:.1f} ms, throughput = {throughput:.2f} 문장/초")

    predictions_path = f"outputs/predictions/{Path(model_checkpoint_path).stem}-{strategy}.jsonl"
    save_predictions(records, predictions_path)
    metrics = evaluate(records, use_comet = use_comet)
    logger.info(f"Test 평가 결과({strategy}): " + ", ".join(f"{k} = {v:.4f}" for k, v in metrics.items()))

    # wandb·MLflow 공통 metric (이름은 이후 LLM 평가 run과 동일하게 사용)
    eval_metrics = {
        "inference_time_sec": elapsed,
        **{f"test_{k}": v for k, v in metrics.items()}, # test_bleu, test_chrf, (test_comet)
        **param_stats, # num_params, num_trainable_params
        **({"peak_vram_mb": peak_vram_mb} if peak_vram_mb is not None else {}), # cuda에서만 기록
        **latency_stats, # latency_p50_ms, latency_p95_ms, latency_mean_ms
        "throughput_sent_per_sec": throughput,
    }

    wandb.log(eval_metrics)
    wandb.finish()

    # MLflow: 학습 run(부모) 아래에 평가 run(자식)으로 기록 → 모델별 평가 결과가 학습 run 하위에 쌓임
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"]) # 부모(학습 run)와 같은 experiment여야 함
    with mlflow.start_run(
        run_name = f"eval-{strategy}",
        parent_run_id = source_run_id, # None이면 독립 run
        tags = {
            "run_type": "eval",
            "model_family": "rnn", # rnn / llm
            "method": "scratch", # scratch / zero-shot / lora
            "decoding": strategy,
        },
    ):
        mlflow.log_params({
            "checkpoint_path": model_checkpoint_path,
            "batch_size": batch_size,
            "sample_size": sample_size,
            "use_comet": use_comet,
            **{f"decode.{k}": v for k, v in decode_kwargs.items()},
        })
        mlflow.log_metrics(eval_metrics)
        mlflow.log_artifact(predictions_path, artifact_path = "predictions")
    logger.info(f"Total Inference Time: {elapsed:.2f}초")
    