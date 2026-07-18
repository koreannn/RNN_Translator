"""
한국어 -> 영어 번역기
"""
import time
import random
import torch
import torch.nn.functional as F
import wandb
import sacrebleu
import numpy as np

from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from torch.optim import Adam
from loguru import logger
from dataloader import CustomDataLoader
from model import Encoder, Decoder, Seq2Seq
from utils import load_config

def greedy_decode_batch( # valid 배치에 대한 BLEU 집계용
    seq2seq_model,
    src_ids,
    sos_token_id,
    eos_token_id,
    pad_token_id,
    max_new_token,
    device,
):
    _, dec_hidden = seq2seq_model.encoder(src_ids) # (1, bs, hidden)
    batch_size = src_ids.size(0)
    
    dec_input = torch.full((batch_size, 1), sos_token_id, dtype = torch.long, device = device)
    finished = torch.zeros(batch_size, dtype = torch.bool, device = device) # 배치 내의 각 샘플이 EOS에 도달했는지 체크하기 위한 용도
    generated = []
    
    for _ in range(max_new_token):
        logits_step, dec_hidden = seq2seq_model.decoder(dec_input, dec_hidden)
        next_ids = torch.argmax(logits_step[:, -1, :], dim = -1)
        next_ids = next_ids.masked_fill(finished, pad_token_id)
        generated.append(next_ids)
        
        finished = finished | (next_ids == eos_token_id)
        if finished.all(): # 배치 내 모든 문장이 EOS에 도달했을 경우
            break
        dec_input = next_ids.unsqueeze(1) # (bs, 1)
        
    return torch.stack(generated, dim = 1) # (bs, gen_len)


def train(
    epochs, patience, min_delta, lr, batch_size, embedding_dim, hidden_dim,
    train_loader, valid_loader, valid_bleu_sample_size,
    kor_vocab_size, en_vocab_size, en_tokenizer, max_new_token,
    encoder, decoder, seq2seq_model,
    device, wandb_project_name,
    wandb_entity, wandb_project, wandb_architecture,
    checkpoint_dir = "checkpoints",
):
    optimizer = Adam([
        {"params": seq2seq_model.encoder.embedding.parameters(), "lr": 1e-5},
        {"params": seq2seq_model.decoder.embedding.parameters(), "lr": 1e-5},
        {"params": seq2seq_model.encoder.rnn.parameters(), "lr": 1e-3},
        {"params": seq2seq_model.decoder.rnn.parameters(), "lr": 1e-3},
        {"params": seq2seq_model.decoder.fc.parameters(), "lr": 1e-3},
    ])
    checkpoint_dir = Path(checkpoint_dir)
    best_valid_loss = float("inf")
    epochs_no_improve = 0
    pad_token_id = 0

    def save_checkpoint(epoch, train_loss, valid_loss):
        nonlocal best_valid_loss, epochs_no_improve
        checkpoint_dir.mkdir(parents = True, exist_ok = True)
        payload = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            # "encoder_state_dict": encoder.state_dict(), 
            # "decoder_state_dict": decoder.state_dict(), -> seq2seq_state_dict에 중복
            "seq2seq_state_dict": seq2seq_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "kor_vocab_size": kor_vocab_size,
            "en_vocab_size": en_vocab_size,
            "pad_token_id": pad_token_id,
        }
        torch.save(payload, checkpoint_dir / "last.pt")

        if valid_loss < best_valid_loss - min_delta:
            best_valid_loss = valid_loss
            epochs_no_improve = 0
            torch.save(payload, checkpoint_dir / "best.pt")
            logger.info(f"Best updated in epoch {epoch + 1}.")
        else:
            epochs_no_improve += 1

    wandb.init(
        entity = wandb_entity,
        project = wandb_project,
        name = wandb_project_name,
        config = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "architecture": wandb_architecture,
        }
    )

    sos_token_id= en_tokenizer.cls_token_id
    eos_token_id = en_tokenizer.sep_token_id
    max_n_token = max_new_token # 새로 생성할 토큰의 최대 개수
    
    train_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        logger.info(f"Epoch {epoch + 1} / {epochs}")
        seq2seq_model.train()
        train_loss_sum = 0.0
        train_steps = 0

        for _, (src_ids, tgt_input, tgt_label) in enumerate(train_loader):
            src_ids = src_ids.to(device) # (bs, (한국어)seq_len)
            tgt_input = tgt_input.to(device) # (bs, (영어)seq_len - 1)
            tgt_label = tgt_label.to(device) # (bs, (영어)seq_len - 1)

            logits = seq2seq_model(src_ids, tgt_input)  # (bs, seq_len, vocab_size)
            logits_flat = logits.reshape(-1, logits.size(-1))  # (bs*seq_len, vocab_size)
            tgt_label_flat = tgt_label.reshape(-1) # (bs*seq_len,)
            loss = F.cross_entropy(logits_flat, tgt_label_flat, ignore_index = pad_token_id)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_steps += 1

        train_avg_loss = train_loss_sum / max(1, train_steps)

        # Validation Loss & Validation BLEU
        seq2seq_model.eval()
        valid_loss_sum = 0.0
        valid_steps = 0
        
        # BLEU Score
        all_yhat = []
        all_ground_truth = []

        with torch.no_grad():
            for src_ids, tgt_input, tgt_label in valid_loader:
                src_ids = src_ids.to(device)
                tgt_input = tgt_input.to(device)
                tgt_label = tgt_label.to(device)

                logits = seq2seq_model(src_ids, tgt_input)
                logits_flat = logits.reshape(-1, logits.size(-1))
                tgt_label_flat = tgt_label.reshape(-1)

                loss = F.cross_entropy(logits_flat, tgt_label_flat, ignore_index = pad_token_id)
                valid_loss_sum += loss.item()
                valid_steps += 1
                
            for src_ids, _, tgt_label in valid_loader:
                src_ids = src_ids.to(device)
                gen_ids = greedy_decode_batch(
                    seq2seq_model, src_ids,
                    sos_token_id, eos_token_id, pad_token_id,
                    max_n_token, device,
                )
                all_yhat.extend(s.strip() for s in en_tokenizer.batch_decode(gen_ids, skip_special_tokens = True))
                all_ground_truth.extend(s.strip() for s in en_tokenizer.batch_decode(tgt_label, skip_special_tokens = True))
                if len(all_yhat) >= valid_bleu_sample_size:
                    break
                # # BLEU 집계
                # _, enc_hidden = seq2seq_model.encoder(src_ids)
                # for i in range(src_ids.size(0)):
                #     dec_hidden = enc_hidden[:, i:i + 1, :]
                #     dec_input = torch.tensor([[sos_token_id]], device = device)
                    
                #     generated_ids = [sos_token_id]
                    
                #     for _ in range(max_n_token):
                #         logits_step, dec_hidden = seq2seq_model.decoder(dec_input, dec_hidden)
                #         next_id = int(torch.argmax(logits_step[:, -1, :], dim = -1).item())
                #         generated_ids.append(next_id)
                #         if next_id == eos_token_id:
                #             break
                #         dec_input = torch.tensor([[next_id]], device = device)
                        
                #     hyp = en_tokenizer.decode(generated_ids, skip_special_tokens = True).strip()
                #     ref = en_tokenizer.decode(tgt_label[i].tolist(), skip_special_tokens = True).strip()
                #     all_yhat.append(hyp)
                #     all_ground_truth.append(ref)

        valid_avg_loss = valid_loss_sum / max(1, valid_steps)
        
        bleu_result = sacrebleu.corpus_bleu(all_yhat, [all_ground_truth])
        valid_bleu = bleu_result.score

        logger.info(f"epoch = {epoch + 1} train_loss = {train_avg_loss:.4f} valid_loss = {valid_avg_loss:.4f} valid_bleu = {valid_bleu:.2f}")
        epoch_elapsed = time.time() - epoch_start
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_avg_loss,
            "valid_loss": valid_avg_loss,
            "valid_bleu": valid_bleu,
            "epoch_time_sec": epoch_elapsed,
        })
        save_checkpoint(epoch = epoch, train_loss = train_avg_loss, valid_loss = valid_avg_loss)
        
        if epochs_no_improve >= patience:
            logger.info(f"[Early Stopping] Epoch {epoch + 1}에서 valid loss가 {patience}회 연속으로 개선되지 않아 조기 종료합니다")
            break
        
    # 조기 종료용 에포크 카운트
    actual_epochs = epoch + 1
    if actual_epochs != epochs:
        wandb.run.name = wandb_project_name.replace(f"-ep{epochs}-", f"-ep{actual_epochs}-")
        
    total_train_time = time.time() - train_start_time
    wandb.summary["total_train_time_sec"] = total_train_time
    return actual_epochs # 로그 이름 바꾸기 위한 반환


if __name__ == "__main__":
    config = load_config("config.yaml")
    device = "cuda" if torch.cuda.is_available() else "mps"

    # 난수 고정
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # h_param
    model_architecture = config["train"]["model_architecture"]
    epochs = config["train"]["h_param"]["epochs"]
    learning_rate = config["train"]["h_param"]["learning_rate"]
    batch_size = config["train"]["h_param"]["batch_size"]
    embedding_dim = config["train"]["h_param"]["embedding_dim"]
    hidden_dim = config["train"]["h_param"]["hidden_dim"]
    max_length = config["train"]["h_param"]["max_length"] # 생성 시퀀스가 이 길이를 초과할 경우 강제 종료
    max_new_token = config["train"]["h_param"]["max_new_token"] # 새로 생성할 토큰 개수의 상한선
    valid_bleu_sample_size = config["train"]["h_param"]["valid_bleu_sample_size"] # 검증 단계 BLEU 점수 측정 문장 개수
    patience = config["train"]["h_param"]["early_stopping"]["patience"]
    min_delta = config["train"]["h_param"]["early_stopping"]["min_delta"]

    # tokenizer
    kor_tokenizer_name = config["model"]["kor_tokenizer"]
    en_tokenizer_name = config["model"]["en_tokenizer"]
    kor_tokenizer = AutoTokenizer.from_pretrained(kor_tokenizer_name)
    en_tokenizer = AutoTokenizer.from_pretrained(en_tokenizer_name)
    kor_pretrained_weight = AutoModel.from_pretrained(kor_tokenizer_name).embeddings.word_embeddings.weight.detach()
    en_pretrained_weight = AutoModel.from_pretrained(en_tokenizer_name).embeddings.word_embeddings.weight.detach()
    kor_vocab_size = kor_tokenizer.vocab_size
    en_vocab_size = en_tokenizer.vocab_size

    # wandb
    wandb_project = config["wandb"]["wandb_project"]
    wandb_entity = config["wandb"]["wandb_entity"]
    wandb_architecture = config["wandb"]["wandb_architecture"]
    wandb_exp_name = f"architecture{model_architecture}-ep{epochs}-lr{learning_rate}-bs{batch_size}-emb{embedding_dim}-hid{hidden_dim}" # 실험 로그 네이밍 컨벤션: <모델구조(이름 및 특징)-주요변수(hp)-그외특징>
    
    # 로그 기록
    logger.add(f"logs/{wandb_exp_name}", encoding = "utf-8")
    
    data_loader = CustomDataLoader(kor_tokenizer, en_tokenizer, max_length = max_length, batch_size = batch_size)
    train_dataloader, valid_dataloader, _ = data_loader.get_data_loader()

    logger.info(f"device: {device}")

    encoder = Encoder(vocab_size = kor_vocab_size, embedding_dim = embedding_dim, hidden_dim = hidden_dim, pretrained_weight = kor_pretrained_weight).to(device)
    decoder = Decoder(vocab_size = en_vocab_size, embedding_dim = embedding_dim, hidden_dim = hidden_dim, pretrained_weight = en_pretrained_weight).to(device)
    seq2seq = Seq2Seq(encoder, decoder).to(device)

    start_time = time.time()
    actual_epoch = train(
        epochs = epochs,
        patience = patience, # 조기 종료용 h param
        lr = learning_rate,
        batch_size = batch_size,
        embedding_dim = embedding_dim,
        hidden_dim = hidden_dim,
        train_loader = train_dataloader,
        valid_loader = valid_dataloader,
        valid_bleu_sample_size = valid_bleu_sample_size,
        kor_vocab_size = kor_vocab_size,
        en_vocab_size = en_vocab_size,
        en_tokenizer = en_tokenizer,
        max_new_token = max_new_token,
        encoder = encoder,
        decoder = decoder,
        seq2seq_model = seq2seq,
        device = device,
        wandb_project = wandb_project,
        wandb_entity = wandb_entity,
        wandb_architecture = wandb_architecture,
        wandb_project_name = wandb_exp_name,
    )
    elapsed = time.time() - start_time
    logger.info(f"학습 소요 시간: {elapsed:.2f}초")
