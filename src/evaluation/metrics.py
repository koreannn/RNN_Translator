import sys
import json
import sacrebleu
import torch
import functools

from sacrebleu.metrics import CHRF
from loguru import logger
from pathlib import Path

COMET_MODEL_NAME = "Unbabel/wmt22-comet-da"

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
    return output.system_score # 0~1 사이 (문장별 점수는 output.scores)


def evaluate(records, use_comet = False) -> dict: # 모델 종류와 무관하게 records(jsonl)만 보고 평가
    sources = [r["source"] for r in records]
    hypotheses = [r["hypothesis"] for r in records]
    references = [r["reference"] for r in records]

    metrics = {
        "bleu": compute_bleu(hypotheses, references),
        "chrf": compute_chrf(hypotheses, references),
    }
    if use_comet:
        metrics["comet"] = compute_comet(sources, hypotheses, references)
    return metrics


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
    # 저장된 예측 jsonl을 디코딩 없이 다시 평가
    # 사용법: python src/evaluation/metrics.py outputs/predictions/best-greedy.jsonl [...] [--no-comet]
    use_comet = "--no-comet" not in sys.argv
    paths = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    if not paths:
        sys.exit("사용법: python src/evaluation/metrics.py <예측 jsonl 경로> [...] [--no-comet]")

    for path in paths:
        records = load_predictions(path)
        metrics = evaluate(records, use_comet = use_comet)
        print(f"{path} ({len(records)}문장): " + ", ".join(f"{k} = {v:.4f}" for k, v in metrics.items()))
