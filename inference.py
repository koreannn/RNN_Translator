import torch
import pandas as pd
import sacrebleu

from loguru import logger
from utils import load_config
from transformers import AutoTokenizer
from model import Encoder, Decoder, Seq2Seq
from dataloader import CustomDataLoader

def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location = device)
    
    required = ["seq2seq_state_dict", "embedding_dim", "hidden_dim", "kor_vocab_size", "en_vocab_size"]
    missing = [k for k in required if k not in checkpoint]
    
    if missing:
        raise KeyError(f"Missiong keys in checkpoint: {missing}")
    return checkpoint
    
def get_model_from_checkpoint(checkpoint, device):
    embedding_dim = int(checkpoint["embedding_dim"])
    hidden_dim = int(checkpoint["hidden_dim"])
    kor_vocab_size = int(checkpoint["kor_vocab_size"])
    en_vocab_size = int(checkpoint["en_vocab_size"])
    
    encoder = Encoder(kor_vocab_size, embedding_dim, hidden_dim)
    decoder = Decoder(en_vocab_size, embedding_dim, hidden_dim)
    model = Seq2Seq(encoder, decoder, padding_id = int(checkpoint.get("pad_token_id", 0)))
    
    model.load_state_dict(checkpoint["seq2seq_state_dict"], strict = True)
    model.to(device)
    model.eval()
    return model

def greedy_search( # greedy방식으로 하나씩 추론
    model,
    kor_tokenizer,
    en_tokenizer,
    device,
    test_dataloader,
    max_length,
    max_new_tokens = 50,
):
    sos_token_id = en_tokenizer.cls_token_id
    eos_token_id = en_tokenizer.sep_token_id
    
    if sos_token_id is None or eos_token_id is None:
        raise ValueError("영어 토크나이저는 반드시 cls_token과 sep_token이 있어야합니다.")
    
    all_yhat = []
    all_ground_truth = []
    
    for idx, (src_ids, _, tgt_label) in enumerate(test_dataloader): # 학습할때는 (src_ids, tgt_input, tgt_label) / 추론 시에는 오직 자신이 만든 토큰으로 다음 토큰을 예측해야함 -> (src_ids, _, _)
        
        logger.info(f"번역 전 문장: {kor_tokenizer.decode(src_ids[0].tolist(), skip_special_tokens = True)}")
        generated_ids = [sos_token_id]
        with torch.no_grad():
            src_ids = src_ids.to(device)
            _, enc_hidden = model.encoder(src_ids)
            dec_hidden = enc_hidden
            dec_input = torch.tensor([[sos_token_id]], dtype = torch.long, device = device)
            
            for _ in range(max_new_tokens):
                logits, dec_hidden = model.decoder(dec_input, dec_hidden)
                next_token_logits = logits[:, -1, :] # 배치 내 모든 hidden 중 마지막 hidden / (bs(1), vocab_size)
                next_id = int(torch.argmax(next_token_logits, dim = -1).item())
                
                generated_ids.append(next_id)
                if next_id == eos_token_id or len(generated_ids) > max_length:
                    break
                
                dec_input = torch.tensor([[next_id]], dtype = torch.long, device = device)
            translated = en_tokenizer.decode(generated_ids, skip_special_tokens = True).strip()
            
            # BLEU
            ground_truth = en_tokenizer.decode(tgt_label[0].tolist(), skip_special_tokens = True).strip()
            all_yhat.append(translated)
            all_ground_truth.append(ground_truth)
            
            logger.info(f"번역된 문장: {translated}")
    
    bleu_result = sacrebleu.corpus_bleu(all_yhat, [all_ground_truth])
    logger.info(f"Test corpus BLEU 점수: {bleu_result.score:.2f}")
    
    return bleu_result.score

def beam_search(
    model,
    kor_tokenizer,
    en_tokenizer,
    device,
    test_dataloader,
    max_length,
    beam_size = 4,
    max_new_tokens = 50,
    alpha = 0.6  # 길이 페널티 하이퍼파라미터 (표준값 0.6 ~ 0.7)
):
    sos_token_id = en_tokenizer.cls_token_id
    eos_token_id = en_tokenizer.sep_token_id
    
    if sos_token_id is None or eos_token_id is None:
        raise ValueError("영어 토크나이자는 반드시 cls_token과 sep_token이 있어야합니다.")
    
    all_yhat = []
    all_ground_truth = []
    
    for idx, (src_ids, _, tgt_label) in enumerate(test_dataloader):
        logger.info(f"번역 전 문장: {kor_tokenizer.decode(src_ids[0].tolist(), skip_special_tokens = True)}")
        
        with torch.no_grad():
            src_ids = src_ids.to(device)
            _, enc_hidden = model.encoder(src_ids)
            
            # beams 구조: (누적 로그 확률, 토큰 리스트, 히든 스테이트)
            beams = [(0.0, [sos_token_id], enc_hidden)]
            completed_beams = [] # 저장 구조: (정규화된 스코어, 토큰 리스트)
            
            for time_step in range(max_new_tokens):
                all_candidates = []
                
                for score, seq, hidden in beams:
                    # 1. EOS를 만난 빔 처리 (정규화 스코어 적용하여 완료 목록 저장)
                    if seq[-1] == eos_token_id:
                        # 정석적인 빔서치 길이 페널티 수식 분모: ((5 + L)^alpha) / ((5 + 1)^alpha)
                        lp = ((5 + len(seq)) ** alpha) / ((5 + 1) ** alpha)
                        normalized_score = score / lp
                        completed_beams.append((normalized_score, seq))
                        continue
                    
                    # 2. 미완료 빔 확장
                    dec_input = torch.tensor([[seq[-1]]], dtype=torch.long, device=device)
                    logits, new_hidden = model.decoder(dec_input, hidden)
                    log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
                    
                    topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)
                    
                    for j in range(beam_size):
                        next_score = score + topk_log_probs[0, j].item()
                        next_seq = seq + [topk_ids[0, j].item()]
                        all_candidates.append((next_score, next_seq, new_hidden.clone()))
                
                # 살아남은 후보가 없다면 종료
                if not all_candidates:
                    break
                
                # 누적 로그 확률(Raw Score) 기준으로 정렬하여 다음 빔 상위 K개 선택
                # (확장하는 단계에서는 Raw Score로 잘라야 수학적 탐색이 왜곡되지 않음)
                all_candidates.sort(key=lambda x: x[0], reverse=True)
                beams = all_candidates[:beam_size]
                
                # 3. 정규화 기반의 조기 종료(Early Stopping) 조건 검사
                if len(completed_beams) >= beam_size:
                    completed_beams.sort(key=lambda x: x[0], reverse=True)
                    best_completed_score = completed_beams[0][0] # 완료된 최선책의 정규화 점수
                    
                    # 현재 진행 중인 빔들의 '잠재적 최대 정규화 점수' 계산
                    # 진행 중인 빔은 앞으로 토큰이 붙으면 점수가 더 낮아질 것이므로, 
                    # 현재 시점의 점수를 현재 길이로 정규화한 것이 이론상 가질 수 있는 최댓값임
                    current_best_lp = ((5 + len(beams[0][1])) ** alpha) / ((5 + 1) ** alpha)
                    current_best_normalized = beams[0][0] / current_best_lp
                    
                    # 완료된 최선책의 점수가 진행 중인 어떤 빔의 최선 예측치보다도 높다면 더 돎 필요 없음
                    if best_completed_score >= current_best_normalized:
                        break
            
            # 4. 루프 종료 후 미완료 빔들도 정규화 점수 매겨서 최종 취합
            for score, seq, _ in beams:
                if seq[-1] != eos_token_id:
                    lp = ((5 + len(seq)) ** alpha) / ((5 + 1) ** alpha)
                    completed_beams.append((score / lp, seq))
            # 최종 정규화 점수 기준으로 정렬
            completed_beams.sort(key=lambda x: x[0], reverse=True)
            
            
            # 결과 저장 및 출력
            translated_sentences = [en_tokenizer.decode(b[1], skip_special_tokens=True).strip() for b in completed_beams[:beam_size]]
            ground_truth = en_tokenizer.decode(tgt_label[0].tolist(), skip_special_tokens=True).strip()
            
            all_yhat.append(translated_sentences[0]) 
            all_ground_truth.append(ground_truth)
            
            # for i in range(min(beam_size, len(translated_sentences))):
            #     logger.info(f"번역된 문장({i + 1}위, Normalized Score: {completed_beams[i][0]:.3f}): {translated_sentences[i]}")
            if translated_sentences:
                logger.info(f"번역된 문장(1위, Normalized Score: {completed_beams[0][0]:.3f}): {translated_sentences[0]}")
    
    bleu_result = sacrebleu.corpus_bleu(all_yhat, [all_ground_truth])
    logger.info(f"Test corpus BLEU 점수: {bleu_result.score:.2f}")
    
    return bleu_result.score


def hybrid_sampling(
    model,
    kor_tokenizer,
    en_tokenizer,
    device,
    test_dataloader,
    max_length,
    max_new_tokens = 50,
    temperature = 0.8,
    top_k = 50,
    top_p = 0.9
):
    sos_token_id = en_tokenizer.cls_token_id
    eos_token_id = en_tokenizer.sep_token_id
    
    if sos_token_id is None or eos_token_id is None:
        raise ValueError("영어 토크나이저는 반드시 cls_token과 sep_token이 있어야합니다.")
    
    all_yhat = []
    all_ground_truth = []
    
    for idx, (src_ids, _, tgt_label) in enumerate(test_dataloader): # 학습할때는 (src_ids, tgt_input, tgt_label) / 추론 시에는 오직 자신이 만든 토큰으로 다음 토큰을 예측해야함 -> (src_ids, _, _)
        
        logger.info(f"번역 전 문장: {kor_tokenizer.decode(src_ids[0].tolist(), skip_special_tokens = True)}")
        generated_ids = [sos_token_id]
        with torch.no_grad():
            src_ids = src_ids.to(device)
            _, enc_hidden = model.encoder(src_ids)
            dec_hidden = enc_hidden
            dec_input = torch.tensor([[sos_token_id]], dtype = torch.long, device = device)
            
            for _ in range(max_new_tokens):
                logits, dec_hidden = model.decoder(dec_input, dec_hidden) # logits.shape() = (bs, seq_len, vocab_size)
                next_token_logits = logits[:, -1, :].squeeze(0) # 배치 차원 버림(next_token_logits.shape() = (vocab_size,))
                scaled_logits = next_token_logits / temperature
                
                # 1. Top-K 필터링
                if top_k > 0:
                    criteria_logit = torch.topk(scaled_logits, top_k)[0][-1]
                    indices_to_removed = scaled_logits < criteria_logit
                    scaled_logits[indices_to_removed] = -float('inf')
                
                # 2. Top-P 필터링
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending = True)
                    sorted_probs = torch.softmax(sorted_logits, dim = -1) # 어차피 1차원 텐서지만 축을 지정해야 경고 메시지가 안뜸
                    cumulative_probs = torch.cumsum(sorted_probs, dim = -1)
                    
                    sorted_indices_to_removed = cumulative_probs > top_p # sorted_indices_to_removed: [False, False, ... , True] (shape: (vocab_size, ))
                    cloned_sorted_indices_to_removed = sorted_indices_to_removed.clone()
                    cloned_sorted_indices_to_removed[1:] = sorted_indices_to_removed[:-1]
                    cloned_sorted_indices_to_removed[0] = False # 맨 첫 번째 토큰은 탈락하지 않도록
                    
                    indices_to_removed = sorted_indices[cloned_sorted_indices_to_removed]
                    scaled_logits[indices_to_removed] = -float('inf')
                    
                probs = torch.softmax(scaled_logits, dim = -1)
                next_id = int(torch.multinomial(probs, num_samples = 1).item())
                
                generated_ids.append(next_id)
                if next_id == eos_token_id or len(generated_ids) > max_length:
                    break
                
                dec_input = torch.tensor([[next_id]], dtype = torch.long, device = device)
            
            translated = en_tokenizer.decode(generated_ids, skip_special_tokens = True).strip()
            
            # BLEU
            ground_truth = en_tokenizer.decode(tgt_label[0].tolist(), skip_special_tokens = True).strip()
            all_yhat.append(translated)
            all_ground_truth.append(ground_truth)
            
            logger.info(f"번역된 문장: {translated}")
    
    bleu_result = sacrebleu.corpus_bleu(all_yhat, [all_ground_truth])
    logger.info(f"Test corpus BLEU 점수: {bleu_result.score:.2f}")
    
    return bleu_result.score



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps"
    logger.info(f"device: {device}")
    # logger.add(f"logs/{wandb_exp_name}", encoding = "utf-8")
    config = load_config("config.yaml")
    # h_param
    max_length = config["inference"]["max_length"]
    max_n_token = config["inference"]["max_new_token"]
    batch_size = config["inference"]["batch_size"]
    
    kor_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    en_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    
    model_checkpoint_path = "checkpoints/best.pt"
    model = load_checkpoint(model_checkpoint_path, device = device)
    
    # logger.info(f"Loaded checkpoint from {model_checkpoint_path} (epoch = {model.get("epoch")} & validation loss = {model.get("valid_loss")})")
    logger.info(f"Loaded checkpoint from {model_checkpoint_path}")
    
    model = get_model_from_checkpoint(model, device = device)
    
    dataloader = CustomDataLoader(kor_tokenizer, en_tokenizer, max_length = max_length, batch_size = batch_size)
    _, _, test_dataloader = dataloader.get_data_loader() # test의 데이터로더는 1개씩 들어가도록 고정되어있음
    
    blue_score = hybrid_sampling(
        model,
        kor_tokenizer,
        en_tokenizer,
        device,
        test_dataloader,
        max_length,
        max_n_token,
    )
    logger.info(f"최종 BLEU Score: {blue_score:.2f}")
    
