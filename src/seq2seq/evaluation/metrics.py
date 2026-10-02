import sacrebleu

def compute_bleu(hypotheses, references):
    # valid(train)/test(inference) 공통 BLEU 기준: 정답은 원문 텍스트, 대소문자 무시 (en_tokenizer가 uncased이므로)
    return sacrebleu.corpus_bleu(hypotheses, [references], lowercase = True).score

def evaluate(records) -> dict: # 모델 종류와 무관하게 records(jsonl)만 보고 평가
    hypotheses = [r["hypothesis"] for r in records]
    references = [r["reference"] for r in records]
    return {
        "bleu": compute_bleu(hypotheses, references),
    }