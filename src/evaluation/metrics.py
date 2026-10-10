import sys
import json
import bisect
import sacrebleu
import torch
import functools
import numpy as np

from sacrebleu.metrics import CHRF
from loguru import logger
from pathlib import Path

COMET_MODEL_NAME = "Unbabel/wmt22-comet-da"
DEFAULT_LENGTH_BINS = [50, 200, 300, 400] # config에 evaluation.length_bins가 없을 때 사용하는 원문 길이 구간 경계

def compute_bleu(hypotheses, references):
    # valid(train)/test(inference) 공통 BLEU 기준: 정답은 원문 텍스트, 대소문자 무시 (en_tokenizer가 uncased이므로)
    return sacrebleu.corpus_bleu(hypotheses, [references], lowercase = True).score


def compute_chrf(hypotheses, references):
    # BLEU와 동일하게 대소문자 무시 (en_tokenizer가 uncased이므로)
    return CHRF(lowercase = True).corpus_score(hypotheses, [references]).score


@functools.lru_cache(maxsize = 1) # 같은 프로세스에서는 한 번만 로드하도록
def _load_comet_model():
    from comet import download_model, load_from_checkpoint
    return load_from_checkpoint(download_model(COMET_MODEL_NAME))

def compute_comet(sources, hypotheses, references, batch_size = 32):
    # 대소문자를 그대로 둠: 의미 기반 지표이고, 대문자를 못 쓰는 것도 실제 번역 품질의 일부이므로

    model = _load_comet_model() # 첫 호출때만 로드하고, 이후엔 캐시된 모델 재사용
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references)]
    output = model.predict(data, batch_size = batch_size, gpus = 1 if torch.cuda.is_available() else 0)
    return output.system_score, output.scores # 전체 점수(0~1), 문장별 점수


def evaluate(records, use_comet = False) -> dict: # 모델 종류와 무관하게 records(jsonl)만 보고 평가
    sources = [r["source"] for r in records]
    hypotheses = [r["hypothesis"] for r in records]
    references = [r["reference"] for r in records]

    metrics = {
        "bleu": compute_bleu(hypotheses, references),
        "chrf": compute_chrf(hypotheses, references),
    }
    if use_comet:
        metrics["comet"], sentence_scores = compute_comet(sources, hypotheses, references)
        for record, score in zip(records, sentence_scores): # 문장별 점수를 records에 저장 (길이별 평가·오류 분석용)
            record["comet"] = score
    return metrics


def get_length_bins(config):
    # RNN·LLM이 같은 구간으로 비교되도록 config 최상위 evaluation.length_bins를 공유
    return (config.get("evaluation") or {}).get("length_bins") or DEFAULT_LENGTH_BINS


def source_length(text):
    # 원문 길이 = 공백을 뺀 글자 수
    # - 토큰 수는 모델마다 토크나이저가 달라 기준이 흔들리므로 사용하지 않음
    # - 공백 제외: 데이터셋을 추가해도 띄어쓰기 습관 차이에 영향받지 않도록
    return len("".join(text.split()))


def _length_labels(bins): # [50, 200] → ["0-50", "51-200", "201-inf"] (MLflow metric 이름에 쓸 수 있는 문자만 사용)
    if not bins or any(b <= 0 for b in bins) or any(a >= b for a, b in zip(bins, bins[1:])):
        raise ValueError(f"length_bins는 양수이고 오름차순이어야 합니다: {bins}")
    lows = [0] + [b + 1 for b in bins]
    highs = [str(b) for b in bins] + ["inf"]
    return [f"{low}-{high}" for low, high in zip(lows, highs)]


def evaluate_by_length(records, bins, latency_warmup = 5) -> dict:
    # 원문 길이 구간별 지표. records(jsonl)만 보고 계산하므로 RNN·LLM 공통이고, 저장된 jsonl로 사후 계산도 가능
    # 구간 경계를 바꾸면 구간 이름(=metric 이름)도 바뀌어, 서로 다른 구간을 같은 지표로 착각해 비교하는 일을 막음
    labels = _length_labels(bins)
    groups = {label: [] for label in labels}
    for r in records:
        groups[labels[bisect.bisect_left(bins, source_length(r["source"]))]].append(r) # 길이 <= 경계인 첫 구간

    # latency는 전체 통계(summarize_latency)와 같은 기준: 측정한 문장 중 앞 latency_warmup개 제외
    timed = [r for r in records if r.get("latency_ms") is not None]
    warmup_ids = {r["id"] for r in timed[:latency_warmup]}

    breakdown = {}
    for label, group in groups.items():
        stats = {"num_sentences": len(group)}
        if not group: # 데이터 구성에 따라 빈 구간이 생길 수 있음 → 문장 수(0)만 기록
            breakdown[label] = stats
            continue

        hypotheses = [r["hypothesis"] for r in group]
        references = [r["reference"] for r in group]
        stats["bleu"] = compute_bleu(hypotheses, references)
        stats["chrf"] = compute_chrf(hypotheses, references)
        if all("comet" in r for r in group): # evaluate(use_comet = True)가 넣어둔 문장별 점수의 평균
            stats["comet"] = float(np.mean([r["comet"] for r in group]))

        latencies = [r["latency_ms"] for r in group if r.get("latency_ms") is not None and r["id"] not in warmup_ids]
        if latencies:
            stats["latency_p50_ms"] = float(np.percentile(latencies, 50))
        if "finish_reason" in group[0]: # LLM: max_tokens에서 잘린 비율 (반복 루프 등)
            stats["truncated_rate"] = sum(r["finish_reason"] == "length" for r in group) / len(group)
        if "source_truncated" in group[0]: # RNN: 입력이 max_length에서 잘린 비율
            stats["source_truncated_rate"] = sum(r["source_truncated"] for r in group) / len(group)

        breakdown[label] = stats
    return breakdown


def flatten_length_metrics(breakdown) -> dict:
    # MLflow·wandb 기록용: "test_chrf/len_201-300" 형태 → UI에서 같은 지표의 구간들이 한 그룹으로 묶임
    quality = {"bleu", "chrf", "comet"} # 전체 지표와 같은 test_ 접두사
    return {
        f"{'test_' + name if name in quality else name}/len_{label}": value
        for label, stats in breakdown.items()
        for name, value in stats.items()
    }


def format_length_breakdown(breakdown) -> str: # 로그 출력용 표
    lines = ["원문 길이(공백 제외 글자 수)별 지표:"]
    for label, stats in breakdown.items():
        values = ", ".join(f"{k} = {v:.4f}" if isinstance(v, float) else f"{k} = {v}" for k, v in stats.items())
        lines.append(f"  [{label}] {values}")
    return "\n".join(lines)


def load_predictions(path):
    with open(path, encoding = "utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
    

def save_predictions(records, path):
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    
    with open(path, "w", encoding = "utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii = False) + "\n")
    
    logger.info(f"예측 결과 저장: {path} ({len(records)} 문장)")


if __name__ == "__main__":
    # 저장된 예측 jsonl을 디코딩 없이 다시 평가 (구간 경계를 바꾸거나 데이터가 바뀌었을 때 과거 결과도 같은 기준으로 재계산)
    # 사용법: python src/evaluation/metrics.py outputs/predictions/best-greedy.jsonl [...] [--no-comet] [--length-bins=50,200,300,400]
    # 구간 경계는 --length-bins > config/config.yaml의 evaluation.length_bins > DEFAULT_LENGTH_BINS 순으로 사용
    usage = "사용법: python src/evaluation/metrics.py <예측 jsonl 경로> [...] [--no-comet] [--length-bins=50,200,300,400]"
    use_comet = "--no-comet" not in sys.argv
    bins_arg = next((arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--length-bins=")), None)
    paths = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    if not paths:
        sys.exit(usage)

    if bins_arg:
        length_bins = [int(b) for b in bins_arg.split(",")]
    else:
        config_path = Path("config/config.yaml")
        if config_path.exists():
            import yaml
            length_bins = get_length_bins(yaml.safe_load(config_path.read_text(encoding = "utf-8")) or {})
        else:
            length_bins = DEFAULT_LENGTH_BINS

    for path in paths:
        records = load_predictions(path)
        metrics = evaluate(records, use_comet = use_comet)
        print(f"{path} ({len(records)}문장): " + ", ".join(f"{k} = {v:.4f}" for k, v in metrics.items()))
        print(format_length_breakdown(evaluate_by_length(records, length_bins)))
