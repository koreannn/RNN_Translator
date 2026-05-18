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
):
    sos_token_id = en_tokenizer.cls_token_id
    eos_token_id = en_tokenizer.sep_token_id
    
    if sos_token_id is None or eos_token_id is None:
        raise ValueError("영어 토크나이저는 반드시 cls_token과 sep_token이 있어야합니다.")
    
    all_yhat = []
    all_ground_truth = []
    
    for idx, (src_ids, _, tgt_label) in enumerate(test_dataloader): # 학습할때는 (src_ids, tgt_input, tgt_label) / 추론 시에는 오직 자신이 만든 토큰으로 다음 토큰을 예측해야함 -> (src_ids, _, _)
        
        logger.info(f"번역 전 문장: {kor_tokenizer.decode(src_ids[0].tolist(), skip_special_tokens = True)}")
        generated_ids = [[sos_token_id] for _ in range(beam_size)]
        
        with torch.no_grad():
            src_ids = src_ids.to(device)
            _, enc_hidden = model.encoder(src_ids)
            
            beams = [(0.0, [sos_token_id], enc_hidden)] # (누적 로그 확률, 토큰 리스트, hidden state)
            completed_beams = []
            
            for time_step in range(max_new_tokens):
                all_candidates = []
                
                for score, seq, hidden in beams:
                    if seq[-1] == eos_token_id:
                        completed_beams.append((score, seq))
                        continue
                    
                    # 마지막 토큰을 입력으로 사용
                    dec_input = torch.tensor([[seq[-1]]], dtype = torch.long, device = device)
                    logits, new_hidden = model.decoder(dec_input, hidden)
                    log_probs = torch.log_softmax(logits[:, -1, :], dim = -1)
                    
                    topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)
                    
                    for j in range(beam_size):
                        next_score = score + topk_log_probs[0, j].item()
                        next_seq = seq + [topk_ids[0, j].item()]
                        all_candidates.append((next_score, next_seq, new_hidden))
            
            dec_hiddens = [enc_hidden.clone() for _ in range(beam_size)]
            dec_inputs = [torch.tensor([[sos_token_id]], dtype = torch.long, device = device) for _ in range(beam_size)]
            beam_scores = [0.0 for _ in range(beam_size)]
            
            for time_step in range(max_new_tokens):
                all_candidates = []
                
                for i in range(beam_size):
                    logits, new_hidden = model.decoder(dec_inputs[i], dec_hiddens[i])
                    log_probs = torch.log_softmax(logits[:, -1, :], dim = -1) # (1, vocab_size)
                    
                    if i == 0: # 누적 확률 찍어보기
                        top4 = torch.topk(log_probs, beam_size)
                        logger.info(f"step={time_step} beam=0 scores={[f'{v:.3f}' for v in top4.values[0].tolist()]}")
                        
                    topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)
                    
                    for j in range(beam_size):
                        new_score = beam_scores[i] + topk_log_probs[0, j].item()
                        new_seq = generated_ids[i] + [topk_ids[0, j].item()]
                        all_candidates.append((new_score, new_seq, new_hidden.clone()))
                        
                
                # 살아남은 가설이 없거나, 완료된 빔이 충분히 쌓였다면 루프 종료 조건 검사
                if not all_candidates:
                    break
                    
                # 전체 후보군 중 상위 beam_size 개 선택
                all_candidates.sort(key=lambda x: x[0], reverse=True)
                beams = all_candidates[:beam_size]
                
                # 효율적인 종료를 위해 완료된 빔과 합쳐서 상위 K개가 모두 완료 상태이면 조기 종료
                temp_all = completed_beams + [(b[0], b[1]) for b in beams]
                temp_all.sort(key=lambda x: x[0], reverse=True)
                if len(completed_beams) >= beam_size and temp_all[beam_size-1] in completed_beams:
                    break
            
            # 최종 후보 취합 (완료된 빔 + 미완료 빔)
            final_beams = completed_beams + [(b[0], b[1]) for b in beams]
            final_beams.sort(key=lambda x: x[0], reverse=True)
            
            # Top-K 출력 및 저장
            translatedes = [en_tokenizer.decode(b[1], skip_special_tokens=True).strip() for b in final_beams[:beam_size]]
            ground_truth = en_tokenizer.decode(tgt_label[0].tolist(), skip_special_tokens=True).strip()
            
            all_yhat.append(translatedes[0]) # 가장 확률이 높은 1위 결과 저장
            all_ground_truth.append(ground_truth)
            
            for i in range(min(beam_size, len(translatedes))):
                logger.info(f"번역된 문장({i + 1}위, Score: {final_beams[i][0]:.3f}): {translatedes[i]}")
    
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
    
    blue_score = beam_search(
        model,
        kor_tokenizer,
        en_tokenizer,
        device,
        test_dataloader,
        max_length = 50,
        max_new_tokens = 50,
    )
    logger.info(f"최종 BLEU Score: {blue_score:.2f}")
    
