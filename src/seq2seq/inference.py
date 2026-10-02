import time
import random
import json
import torch
import wandb

from loguru import logger
from pathlib import Path
from transformers import AutoTokenizer
from src.seq2seq.model import build_model
from src.seq2seq.utils import load_config, resolve_device, set_seed
from src.seq2seq.decoding import get_special_token_ids, greedy_decoding, beam_decoding, sampling_decoding
from src.evaluation.metrics import evaluate
from src.seq2seq.dataloader import CustomDataLoader

def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location = device)
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
            gen_ids = greedy_decoding(model, src_ids, max_new_tokens = max_new_tokens, **ids)
            hypotheses = [s.strip() for s in en_tokenizer.batch_decode(gen_ids, skip_special_tokens = True)]
        else: # beam / hybrid는 문장 단위
            decode_fn = beam_decoding if strategy == "beam" else sampling_decoding
            hypotheses = []
            for i in range(src_ids.size(0)):
                if sample_size is not None and len(records) + len(hypotheses) >= sample_size:
                    break
                gen_ids = decode_fn(
                    model,
                    src_ids[i : i + 1],
                    max_new_tokens = max_new_tokens,
                    max_length = max_length,
                    **ids,
                    **decode_kwargs,
                )
                hypotheses.append(en_tokenizer.decode(gen_ids, skip_special_tokens = True).strip())

        if sample_size is not None: # greedy는 배치 단위라 sample_size를 넘칠 수 있으므로 잘라냄
            hypotheses = hypotheses[: sample_size - len(records)]

        for source, reference, hypothesis in zip(src_text, tgt_text, hypotheses):
            records.append({
                "id": len(records), # test split 내 순번 (test 로더는 shuffle = False)
                "source": source,
                "reference": reference,
                "hypothesis": hypothesis,
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

    model = load_checkpoint(model_checkpoint_path, device = device)
    
    # logger.info(f"Loaded checkpoint from {model_checkpoint_path} (epoch = {model.get("epoch")} & validation loss = {model.get("valid_loss")})")
    logger.info(f"Loaded checkpoint from {model_checkpoint_path}")
    
    model = get_model_from_checkpoint(model, device = device)
    
    dataloader = CustomDataLoader(kor_tokenizer, en_tokenizer, max_length = max_length, batch_size = batch_size)
    _, _, test_dataloader = dataloader.get_data_loader() # test의 데이터로더는 1개씩 들어가도록 고정되어있음
    
    strategy = config["inference"]["decoding_strategy"]
    decode_kwargs = config["inference"].get(strategy, {}) # greedy는 하이퍼파라미터 섹션이 없으므로 {}
    sample_size = config["inference"]["sample_size"]

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

    save_predictions(records, f"outputs/predictions/{Path(model_checkpoint_path).stem}-{strategy}.jsonl")
    metrics = evaluate(records)
    bleu_score = metrics["bleu"]
    logger.info(f"Test 평가 결과({strategy}): {metrics}")
    
    wandb.log(
        {
            "inference_time_sec": elapsed,
            "inference_bleu": bleu_score,
        }
    )
    wandb.finish()
    logger.info(f"최종 BLEU Score: {bleu_score:.2f}")   
    logger.info(f"Total Inference Time: {elapsed:.2f}초")
    