import time
import random
import torch
import wandb

from loguru import logger
from pathlib import Path
from transformers import AutoTokenizer
from src.seq2seq.model import build_model
from src.seq2seq.utils import load_config, resolve_device, set_seed
from src.seq2seq.decoding import get_special_token_ids, greedy_decoding
from src.seq2seq.evaluation import compute_bleu
from dataloader import CustomDataLoader

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


def greedy_search( # greedy방식으로 하나씩 추론
    model,
    kor_tokenizer,
    en_tokenizer,
    device,
    test_dataloader,
    max_length,
    max_new_tokens,
    sample_size = None,
):
    
    ids = get_special_token_ids(kor_tokenizer, en_tokenizer)
    all_yhat = [] # 번역된 문장 전체를 담고있는 리스트
    all_ground_truth = [] # 원본 정답 문장(영어)
    all_source = [] # 원본 입력 문장(한국어)
    
    for batch_idx, (src_ids, _, _, src_text, tgt_text) in enumerate(test_dataloader): # 학습할때는 (src_ids, tgt_input, tgt_label) / 추론 시에는 오직 자신이 만든 토큰으로 다음 토큰을 예측해야함 -> (src_ids, _, _)
        src_ids = src_ids.to(device)
        gen_ids = greedy_decoding(
            model,
            src_ids,
            max_new_tokens = max_new_tokens,
            **ids,
        )
        batch_translated = [s.strip() for s in en_tokenizer.batch_decode(gen_ids, skip_special_tokens = True)]

        all_yhat.extend(batch_translated)
        all_ground_truth.extend(tgt_text)
        all_source.extend(src_text)

        sample_idx = random.randrange(len(batch_translated))
        logger.info(f"번역 전 문장(예시): {src_text[sample_idx]}")
        logger.info(f"번역된 문장(예시): {batch_translated[sample_idx]}")
        
        if sample_size is not None and len(all_yhat) >= sample_size:
            break
                
    bleu_score = compute_bleu(all_yhat, all_ground_truth)
    logger.info(f"Test corpus BLEU 점수: {bleu_score:.2f}")
    
    return bleu_score

def beam_search(
    model,
    kor_tokenizer,
    en_tokenizer,
    device,
    test_dataloader,
    max_length,
    beam_size = 4,
    max_new_tokens = 400,
    alpha = 0.6,  # 길이 페널티 하이퍼파라미터 (표준값 0.6 ~ 0.7)
    sample_size = None,
):
    sos_token_id = en_tokenizer.cls_token_id
    eos_token_id = en_tokenizer.sep_token_id
    vocab_size = en_tokenizer.vocab_size

    if sos_token_id is None or eos_token_id is None:
        raise ValueError("영어 토크나이저는 반드시 cls_token과 sep_token이 있어야합니다.")

    all_yhat = []
    all_ground_truth = []
    all_source = []

    for batch_idx, (src_ids, _, _, src_text, tgt_text) in enumerate(test_dataloader):
        src_ids = src_ids.to(device)
        bs = src_ids.size(0)

        for i in range(bs):  # 배치 내 문장 하나씩 빔서치
            if sample_size is not None and len(all_yhat) >= sample_size:
                break
            sent_src_ids = src_ids[i : i + 1]  # (1, src_len)
            logger.info(f"번역 전 문장: {kor_tokenizer.decode(sent_src_ids[0].tolist(), skip_special_tokens = True)}")

            with torch.no_grad():
                encoder_outputs, enc_hidden = model.encoder(sent_src_ids)   # (1, src_len, hidden_dim) / (1, 1, hidden_dim)
                src_mask = (sent_src_ids != kor_tokenizer.pad_token_id)     # (1, src_len)

                dec_input = torch.tensor([[sos_token_id]], dtype = torch.long, device = device)
                logits_step, step_hidden, _ = model.decoder(dec_input, enc_hidden, encoder_outputs, src_mask)
                log_probs = torch.log_softmax(logits_step[:, -1, :], dim = -1)
                top_scores, top_tokens = torch.topk(log_probs[0], beam_size)
                beam_seqs = [[sos_token_id, top_tokens[b].item()] for b in range(beam_size)]
                beam_scores = top_scores.clone()
                beam_hidden = step_hidden.expand(-1, beam_size, -1).contiguous()

                # attention이 참조할 encoder_outputs/src_mask도 빔 개수만큼 복제 (같은 소스 문장이라 재정렬 불필요)
                beam_encoder_outputs = encoder_outputs.expand(beam_size, -1, -1).contiguous()
                beam_src_mask = src_mask.expand(beam_size, -1).contiguous()

                completed_beams = []

                active_mask = []
                for b in range(beam_size):
                    if beam_seqs[b][-1] == eos_token_id:
                        lp = ((5 + len(beam_seqs[b])) ** alpha) / ((5 + 1) ** alpha)
                        completed_beams.append((beam_scores[b].item() / lp, beam_seqs[b]))
                        active_mask.append(False)
                    else:
                        active_mask.append(True)
                active_mask = torch.tensor(active_mask, device = device)

                for time_step in range(max_new_tokens - 1):
                    if not active_mask.any():
                        break

                    last_tokens = torch.tensor([[seq[-1]] for seq in beam_seqs], dtype = torch.long, device = device)
                    logits_batch, beam_hidden, _ = model.decoder(last_tokens, beam_hidden, beam_encoder_outputs, beam_src_mask)
                    log_probs = torch.log_softmax(logits_batch[:, -1, :], dim = -1)

                    candidate_scores = beam_scores.unsqueeze(1) + log_probs
                    candidate_scores[~active_mask] = -float('inf')

                    flat = candidate_scores.view(-1)
                    top_scores_new, top_flat_idx = torch.topk(flat, beam_size)

                    parent_beams = top_flat_idx // vocab_size
                    next_tokens = top_flat_idx % vocab_size

                    beam_hidden = beam_hidden[:, parent_beams, :].contiguous()
                    # beam_encoder_outputs / beam_src_mask는 전부 같은 소스 문장이라 그대로 유지 (재정렬 불필요)

                    new_seqs, new_active = [], []
                    parent_list = parent_beams.tolist()
                    token_list = next_tokens.tolist()
                    
                    for b in range(beam_size):
                        parent = parent_beams[b]
                        token  = next_tokens[b]
                        new_seq = beam_seqs[parent] + [token]
                        new_seqs.append(new_seq)

                        if token == eos_token_id or len(new_seq) > max_length:
                            lp = ((5 + len(new_seq)) ** alpha) / ((5 + 1) ** alpha)
                            completed_beams.append((top_scores_new[b].item() / lp, new_seq))
                            new_active.append(False)
                        else:
                            new_active.append(True)

                    beam_seqs   = new_seqs
                    beam_scores = top_scores_new
                    active_mask = torch.tensor(new_active, device = device)

                    if len(completed_beams) >= beam_size:
                        completed_beams.sort(key = lambda x: x[0], reverse = True)
                        best_done = completed_beams[0][0]
                        active_idx = active_mask.nonzero(as_tuple = True)[0]
                        if len(active_idx) > 0:
                            best_ongoing = beam_scores[active_idx[0]].item()
                            best_len     = len(beam_seqs[active_idx[0].item()])
                            best_lp      = ((5 + best_len) ** alpha) / ((5 + 1) ** alpha)
                            if best_done >= best_ongoing / best_lp:
                                break
                        else:
                            break

                for b, seq in enumerate(beam_seqs):
                    if active_mask[b]:
                        lp = ((5 + len(seq)) ** alpha) / ((5 + 1) ** alpha)
                        completed_beams.append((beam_scores[b].item() / lp, seq))

                completed_beams.sort(key = lambda x: x[0], reverse = True)
                best_seq     = completed_beams[0][1]
                translated   = en_tokenizer.decode(best_seq, skip_special_tokens = True).strip()
                ground_truth = tgt_text[i]

                all_yhat.append(translated)
                all_ground_truth.append(ground_truth)
                all_source.append(src_text[i])
                logger.info(f"번역된 문장(1위, Normalized Score: {completed_beams[0][0]:.3f}): {translated}")

    bleu_score = compute_bleu(all_yhat, all_ground_truth)
    logger.info(f"Test corpus BLEU 점수: {bleu_score:.2f}")

    return bleu_score


def hybrid_sampling(
    model,
    kor_tokenizer,
    en_tokenizer,
    device,
    test_dataloader,
    max_length,
    max_new_tokens,
    temperature,
    top_k,
    top_p,
    sample_size = None,
):
    sos_token_id = en_tokenizer.cls_token_id
    eos_token_id = en_tokenizer.sep_token_id

    if sos_token_id is None or eos_token_id is None:
        raise ValueError("영어 토크나이저는 반드시 cls_token과 sep_token이 있어야합니다.")

    all_yhat = []
    all_ground_truth = []
    all_source = []

    for batch_idx, (src_ids, _, _, src_text, tgt_text) in enumerate(test_dataloader):
        src_ids = src_ids.to(device)
        bs = src_ids.size(0)

        for i in range(bs): 
            if sample_size is not None and len(all_yhat) >= sample_size:
                break

            sent_src_ids = src_ids[i : i + 1]
            logger.info(f"번역 전 문장: {kor_tokenizer.decode(sent_src_ids[0].tolist(), skip_special_tokens = True)}")
            generated_ids = [sos_token_id]

            with torch.no_grad():
                encoder_outputs, enc_hidden = model.encoder(sent_src_ids)
                src_mask = (sent_src_ids != kor_tokenizer.pad_token_id)
                dec_hidden = enc_hidden
                dec_input = torch.tensor([[sos_token_id]], dtype = torch.long, device = device)

                for _ in range(max_new_tokens):
                    logits, dec_hidden, _ = model.decoder(dec_input, dec_hidden, encoder_outputs, src_mask)
                    next_token_logits = logits[:, -1, :].squeeze(0)
                    scaled_logits = next_token_logits / temperature

                    # 1. Top-K 필터링
                    if top_k > 0:
                        criteria_logit = torch.topk(scaled_logits, top_k)[0][-1]
                        indices_to_removed = scaled_logits < criteria_logit
                        scaled_logits[indices_to_removed] = -float('inf')

                    # 2. Top-P 필터링
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending = True)
                        sorted_probs = torch.softmax(sorted_logits, dim = -1)
                        cumulative_probs = torch.cumsum(sorted_probs, dim = -1)

                        sorted_indices_to_removed = cumulative_probs > top_p
                        cloned_sorted_indices_to_removed = sorted_indices_to_removed.clone()
                        cloned_sorted_indices_to_removed[1:] = sorted_indices_to_removed[:-1]
                        cloned_sorted_indices_to_removed[0] = False

                        indices_to_removed = sorted_indices[cloned_sorted_indices_to_removed]
                        scaled_logits[indices_to_removed] = -float('inf')

                    probs = torch.softmax(scaled_logits, dim = -1)
                    next_id = int(torch.multinomial(probs, num_samples = 1).item())

                    generated_ids.append(next_id)
                    if next_id == eos_token_id or len(generated_ids) > max_length:
                        break

                    dec_input = torch.tensor([[next_id]], dtype = torch.long, device = device)

                translated = en_tokenizer.decode(generated_ids, skip_special_tokens = True).strip()
                ground_truth = tgt_text[i]
                all_yhat.append(translated)
                all_ground_truth.append(ground_truth)
                all_source.append(src_text[i])

                logger.info(f"번역된 문장: {translated}")

        if sample_size is not None and len(all_yhat) >= sample_size:
            break

    bleu_score = compute_bleu(all_yhat, all_ground_truth)
    logger.info(f"Test corpus BLEU 점수: {bleu_score:.2f}")

    return bleu_score


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
    
    start_time = time.time()
    # greedy search
    bleu_score = greedy_search(
        model,
        kor_tokenizer,
        en_tokenizer,
        device,
        test_dataloader,
        max_length,
        max_n_token,
        sample_size = 1000,
    )
    
    # # beam_search
    # bleu_score = beam_search(
    #     model,
    #     kor_tokenizer,
    #     en_tokenizer,
    #     device,
    #     test_dataloader,
    #     max_length,
    #     beam_size = 4,
    #     max_new_tokens = max_n_token,
    #     alpha = 0.6,
    #     sample_size = 1000,
    # )
    
    # hybrid sampling
    bleu_score = hybrid_sampling(
        model,
        kor_tokenizer,
        en_tokenizer,
        device,
        test_dataloader,
        max_length,
        max_new_tokens = max_n_token,
        temperature = 0.8,
        top_k = 50,
        top_p = 0.9,
        sample_size = 1000,
    )
    
    elapsed = time.time() - start_time
    
    wandb.log(
        {
            "inference_time_sec": elapsed,
            "inference_bleu": bleu_score,
        }
    )
    wandb.finish()
    logger.info(f"최종 BLEU Score: {bleu_score:.2f}")   
    logger.info(f"Total Inference Time: {elapsed:.2f}초")
    