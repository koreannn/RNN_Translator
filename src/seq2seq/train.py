"""
한국어 -> 영어 번역기
"""
import time
import tempfile
import random
import torch
import torch.nn.functional as F
import wandb
import mlflow
import sacrebleu
import numpy as np

from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from torch.optim import Adam
from loguru import logger
from dataloader import CustomDataLoader
from model import Encoder, Decoder, Seq2Seq
from utils import load_config
from decoding import greedy_decoding

def train(
    epochs, patience, min_delta, lr, embedding_lr, batch_size, embedding_dim, hidden_dim, grad_clip_max_norm,
    train_loader, valid_loader, valid_bleu_sample_size,
    use_layer_norm, init_scheme,
    kor_vocab_size, en_vocab_size, en_tokenizer, max_new_token,
    seq2seq_model,
    device, wandb_project_name,
    wandb_entity, wandb_project, wandb_architecture,
    checkpoint_dir = "checkpoints",
):
    optimizer = Adam([
        {"params": seq2seq_model.encoder.embedding.parameters(), "lr": embedding_lr},
        {"params": seq2seq_model.decoder.embedding.parameters(), "lr": embedding_lr},
        {"params": seq2seq_model.encoder.rnn.parameters(), "lr": lr},
        {"params": seq2seq_model.decoder.rnn.parameters(), "lr": lr},
        {"params": seq2seq_model.decoder.attention.parameters(), "lr": lr},
        {"params": seq2seq_model.decoder.fc.parameters(), "lr": lr},
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
            "use_layer_norm": use_layer_norm,
            "init_scheme": init_scheme,
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
    wandb.define_metric('train/step')
    wandb.define_metric('train/grad_norm_*', step_metric = 'train/step')

    sos_token_id= en_tokenizer.cls_token_id
    eos_token_id = en_tokenizer.sep_token_id
    max_n_token = max_new_token # 새로 생성할 토큰의 최대 개수
    
    train_start_time = time.time()
    global_step = 0 # 전체 에포크의 배치 스텝
    for epoch in range(epochs):
        epoch_start = time.time()
        logger.info(f"Epoch {epoch + 1} / {epochs}")
        seq2seq_model.train()
        train_loss_sum = 0.0
        train_steps = 0 # 배치의 스텝

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
            
            # Gradient 노름 계측
            enc_grad_norm = torch.nn.utils.clip_grad_norm_(seq2seq_model.encoder.rnn.parameters(), max_norm = float('inf'))
            dec_grad_norm = torch.nn.utils.clip_grad_norm_(seq2seq_model.decoder.rnn.parameters(), max_norm = float('inf'))
            total_grad_norm = torch.nn.utils.clip_grad_norm_(seq2seq_model.parameters(), max_norm = grad_clip_max_norm)
            
            optimizer.step()

            train_loss_sum += loss.item()
            train_steps += 1
            global_step += 1
            wandb.log({
                'train/step': global_step,
                'train/grad_norm_total': total_grad_norm.item(),
                'train/grad_norm_encoder_rnn': enc_grad_norm.item(),
                'train/grad_norm_decoder_rnn': dec_grad_norm.item(),
            })
            mlflow.log_metrics({
                'train/grad_norm_total': total_grad_norm.item(),
                'train/grad_norm_encoder_rnn': enc_grad_norm.item(),
                'train/grad_norm_decoder_rnn': dec_grad_norm.item(),
            }, step = global_step, synchronous = False)

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
                gen_ids = greedy_decoding(
                    seq2seq_model, src_ids,
                    max_new_tokens = max_n_token,
                    **special_ids,
                )
                all_yhat.extend(s.strip() for s in en_tokenizer.batch_decode(gen_ids, skip_special_tokens = True))
                all_ground_truth.extend(s.strip() for s in en_tokenizer.batch_decode(tgt_label, skip_special_tokens = True))
                if len(all_yhat) >= valid_bleu_sample_size:
                    break

        valid_avg_loss = valid_loss_sum / max(1, valid_steps)
        
        bleu_result = sacrebleu.corpus_bleu(all_yhat, [all_ground_truth])
        valid_bleu = bleu_result.score
        
        samples = "\n\n".join(
            f"[REF] {ref}\n[HYP] {hyp}"
            for ref, hyp in zip(all_ground_truth[:20], all_yhat[:20])
        )
        mlflow.log_text(samples, f"samples/epoch_{epoch + 1:02d}.txt")

        logger.info(f"epoch = {epoch + 1} train_loss = {train_avg_loss:.4f} valid_loss = {valid_avg_loss:.4f} valid_bleu = {valid_bleu:.2f}")
        epoch_elapsed = time.time() - epoch_start
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_avg_loss,
            "valid_loss": valid_avg_loss,
            "valid_bleu": valid_bleu,
            "epoch_time_sec": epoch_elapsed,
        })
        
        mlflow.log_metrics({
            "train_loss": train_avg_loss,
            "valid_loss": valid_avg_loss,
            "valid_bleu": valid_bleu,
            "epoch_time_sec": epoch_elapsed,
        }, step = epoch + 1)
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
    mlflow.log_metric("total_train_time_sec", total_train_time)
    mlflow.log_metric("actual_epochs", actual_epochs)
    return actual_epochs # 로그 이름 바꾸기 위한 반환


if __name__ == "__main__":
    config = load_config("config/config.yaml")
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
    batch_size = config["train"]["h_param"]["batch_size"]
    embedding_dim = config["train"]["h_param"]["embedding_dim"]
    hidden_dim = config["train"]["h_param"]["hidden_dim"]
    embedding_lr = config["train"]["h_param"]["embedding_lr"]
    rnn_attn_fc_lr = config["train"]["h_param"]["rnn_attn_fc_lr"]
    grad_clip_max_norm = config["train"]["h_param"].get("grad_clip_max_norm", None)
    init_scheme = config["train"]["h_param"].get("init_scheme", "default")
    use_layer_norm = config["train"]["h_param"].get("use_layer_norm", False)
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
    wandb_exp_name = f"architecture{model_architecture}-ep{epochs}-lr{rnn_attn_fc_lr}-bs{batch_size}-emb{embedding_dim}-hid{hidden_dim}-init{init_scheme}-ln{use_layer_norm}" # 실험 로그 네이밍 컨벤션: <모델구조(이름 및 특징)-주요변수(hp)-그외특징>
    
    # mlflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    # 로그 기록
    log_path = f"logs/{wandb_exp_name}-{time.strftime('%Y%m%d-%H%M%S')}.log"
    logger.add(log_path, encoding = "utf-8")
    
    data_loader = CustomDataLoader(kor_tokenizer, en_tokenizer, max_length = max_length, batch_size = batch_size)
    train_dataloader, valid_dataloader, _ = data_loader.get_data_loader()

    logger.info(f"device: {device}")

    encoder = Encoder(
        vocab_size = kor_vocab_size,
        embedding_dim = embedding_dim,
        hidden_dim = hidden_dim,
        pretrained_weight = kor_pretrained_weight,
        init_scheme = init_scheme,
        use_layer_norm = use_layer_norm
        ).to(device)
    
    decoder = Decoder(
        vocab_size = en_vocab_size, 
        embedding_dim = embedding_dim,
        hidden_dim = hidden_dim,
        pretrained_weight = en_pretrained_weight,
        init_scheme = init_scheme,
        use_layer_norm = use_layer_norm
        ).to(device)
    
    seq2seq = Seq2Seq(encoder, decoder).to(device)

    
    with mlflow.start_run(
        run_name = config["mlflow"].get("run_name"),
        description = config["mlflow"].get("run_description"),
        tags = {
            "run_type": "train",
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else device,
            **config["mlflow"].get("tags", {}),
        }
        ):
        start_time = time.time()
        
        mlflow.log_params({
            "model_architecture": model_architecture,
            "epochs": epochs,
            "batch_size": batch_size,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "init_scheme": init_scheme,
            "use_layer_norm": use_layer_norm,
            "grad_clip_max_norm": grad_clip_max_norm,
            "max_length": max_length,
            "max_new_token": max_new_token,
            "valid_bleu_sample_size": valid_bleu_sample_size,
            "patience": patience,
            "min_delta": min_delta,
            "seed": config["seed"],
            "kor_tokenizer": kor_tokenizer_name,
            "en_tokenizer": en_tokenizer_name,
            "lr_embedding": embedding_lr,
            "lr_rnn_attn_fc": rnn_attn_fc_lr,
        })
        mlflow.log_artifact("config/config.yaml", artifact_path = "config")
        mlflow.log_params({f"data.{k}": v for k, v in config["data"].items()})
        
        actual_epoch = train(
            epochs = epochs,
            patience = patience, 
            min_delta = min_delta, # 조기 종료용 h param
            lr = rnn_attn_fc_lr,
            embedding_lr = embedding_lr,
            batch_size = batch_size,
            embedding_dim = embedding_dim,
            hidden_dim = hidden_dim,
            grad_clip_max_norm = grad_clip_max_norm,
            train_loader = train_dataloader,
            valid_loader = valid_dataloader,
            valid_bleu_sample_size = valid_bleu_sample_size,
            use_layer_norm = use_layer_norm,
            init_scheme = init_scheme,
            kor_vocab_size = kor_vocab_size,
            en_vocab_size = en_vocab_size,
            en_tokenizer = en_tokenizer,
            max_new_token = max_new_token,
            seq2seq_model = seq2seq,
            device = device,
            wandb_project = wandb_project,
            wandb_entity = wandb_entity,
            wandb_architecture = wandb_architecture,
            wandb_project_name = wandb_exp_name,
            )
        elapsed = time.time() - start_time
        logger.info(f"학습 소요 시간: {elapsed:.2f}초")
        
        if config["mlflow"].get("log_model_weights", True):
            best_ckpt = torch.load("checkpoints/best.pt", map_location = "cpu")
            best_ckpt.pop("optimizer_state_dict")  # 추론에는 불필요 (용량의 약 2/3)
            with tempfile.TemporaryDirectory() as tmp_dir:
                weights_path = Path(tmp_dir) / "best_weights.pt"
                torch.save(best_ckpt, weights_path)
                mlflow.log_artifact(str(weights_path), artifact_path = "checkpoints")

        mlflow.log_artifact(log_path, artifact_path = "logs")
    

