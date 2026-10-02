"""
한국어 -> 영어 번역기
"""
import time
import tempfile
import torch
import torch.nn.functional as F
import wandb
import mlflow

from dataclasses import dataclass, asdict
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from torch.optim import Adam
from loguru import logger
from src.seq2seq.dataloader import CustomDataLoader
from src.seq2seq.model import build_model
from src.seq2seq.utils import load_config, resolve_device, set_seed
from src.seq2seq.decoding import get_special_token_ids, greedy_decoding
from src.seq2seq.evaluation.metrics import compute_bleu


@dataclass
class TrainConfig:
    model_architecture: str
    epochs: int
    batch_size: int
    embedding_dim: int
    hidden_dim: int
    embedding_lr: float
    rnn_attn_fc_lr: float
    max_length: int # 토크나이징 시 시퀀스 최대 길이 (초과분은 truncation)
    max_new_token: int # 새로 생성할 토큰 개수의 상한선
    valid_bleu_sample_size: int # 검증 단계 BLEU 점수 측정 문장 개수
    patience: int # valid loss가 이 횟수만큼 연속으로 개선되지 않을 경우 early stopping
    min_delta: float
    grad_clip_max_norm: float | None = None
    init_scheme: str = "default"
    use_layer_norm: bool = False

    @classmethod
    def from_config(cls, config):
        h_param = config["train"]["h_param"]
        return cls(
            model_architecture = config["train"]["model_architecture"],
            epochs = h_param["epochs"],
            batch_size = h_param["batch_size"],
            embedding_dim = h_param["embedding_dim"],
            hidden_dim = h_param["hidden_dim"],
            embedding_lr = h_param["embedding_lr"],
            rnn_attn_fc_lr = h_param["rnn_attn_fc_lr"],
            max_length = h_param["max_length"],
            max_new_token = h_param["max_new_token"],
            valid_bleu_sample_size = h_param["valid_bleu_sample_size"],
            patience = h_param["early_stopping"]["patience"],
            min_delta = h_param["early_stopping"]["min_delta"],
            grad_clip_max_norm = h_param.get("grad_clip_max_norm", None),
            init_scheme = h_param.get("init_scheme", "default"),
            use_layer_norm = h_param.get("use_layer_norm", False),
        )

    @property
    def exp_name(self): # 실험 로그 네이밍 컨벤션: <모델구조(이름 및 특징)-주요변수(hp)-그외특징>
        return (
            f"architecture{self.model_architecture}-ep{self.epochs}-lr{self.rnn_attn_fc_lr}-bs{self.batch_size}"
            f"-emb{self.embedding_dim}-hid{self.hidden_dim}-init{self.init_scheme}-ln{self.use_layer_norm}"
        )


def train(
    cfg, seq2seq_model, model_config,
    train_loader, valid_loader,
    kor_tokenizer, en_tokenizer,
    device, wandb_config, wandb_exp_name,
    checkpoint_dir = "checkpoints",
):
    optimizer = Adam([
        {"params": seq2seq_model.encoder.embedding.parameters(), "lr": cfg.embedding_lr},
        {"params": seq2seq_model.decoder.embedding.parameters(), "lr": cfg.embedding_lr},
        {"params": seq2seq_model.encoder.rnn.parameters(), "lr": cfg.rnn_attn_fc_lr},
        {"params": seq2seq_model.decoder.rnn.parameters(), "lr": cfg.rnn_attn_fc_lr},
        {"params": seq2seq_model.decoder.attention.parameters(), "lr": cfg.rnn_attn_fc_lr},
        {"params": seq2seq_model.decoder.fc.parameters(), "lr": cfg.rnn_attn_fc_lr},
    ])
    checkpoint_dir = Path(checkpoint_dir)
    best_valid_loss = float("inf")
    epochs_no_improve = 0
    tgt_pad_id = en_tokenizer.pad_token_id # loss 계산 시 무시할 정답(영어) pad

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
            "model_config": model_config, # 추론 시 build_model(**model_config)로 그대로 복원
        }
        torch.save(payload, checkpoint_dir / "last.pt")

        if valid_loss < best_valid_loss - cfg.min_delta:
            best_valid_loss = valid_loss
            epochs_no_improve = 0
            torch.save(payload, checkpoint_dir / "best.pt")
            logger.info(f"Best updated in epoch {epoch + 1}.")
        else:
            epochs_no_improve += 1

    wandb.init(
        entity = wandb_config["wandb_entity"],
        project = wandb_config["wandb_project"],
        name = wandb_exp_name,
        config = {
            "learning_rate": cfg.rnn_attn_fc_lr,
            "batch_size": cfg.batch_size,
            "architecture": wandb_config["wandb_architecture"],
        }
    )
    wandb.define_metric('train/step')
    wandb.define_metric('train/grad_norm_*', step_metric = 'train/step')

    special_ids = get_special_token_ids(kor_tokenizer, en_tokenizer)

    train_start_time = time.time()
    global_step = 0 # 전체 에포크의 배치 스텝
    for epoch in range(cfg.epochs):
        epoch_start = time.time()
        logger.info(f"Epoch {epoch + 1} / {cfg.epochs}")
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
            loss = F.cross_entropy(logits_flat, tgt_label_flat, ignore_index = tgt_pad_id)
            optimizer.zero_grad()
            loss.backward()

            # Gradient 노름 계측
            enc_grad_norm = torch.nn.utils.clip_grad_norm_(seq2seq_model.encoder.rnn.parameters(), max_norm = float('inf'))
            dec_grad_norm = torch.nn.utils.clip_grad_norm_(seq2seq_model.decoder.rnn.parameters(), max_norm = float('inf'))
            total_grad_norm = torch.nn.utils.clip_grad_norm_(seq2seq_model.parameters(), max_norm = cfg.grad_clip_max_norm)

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
            for src_ids, tgt_input, tgt_label, _, _ in valid_loader:
                src_ids = src_ids.to(device)
                tgt_input = tgt_input.to(device)
                tgt_label = tgt_label.to(device)

                logits = seq2seq_model(src_ids, tgt_input)
                logits_flat = logits.reshape(-1, logits.size(-1))
                tgt_label_flat = tgt_label.reshape(-1)

                loss = F.cross_entropy(logits_flat, tgt_label_flat, ignore_index = tgt_pad_id)
                valid_loss_sum += loss.item()
                valid_steps += 1

            for src_ids, _, _, _, tgt_text in valid_loader:
                src_ids = src_ids.to(device)
                gen_ids = greedy_decoding(
                    seq2seq_model, src_ids,
                    max_new_tokens = cfg.max_new_token,
                    **special_ids,
                )
                all_yhat.extend(s.strip() for s in en_tokenizer.batch_decode(gen_ids, skip_special_tokens = True))
                all_ground_truth.extend(tgt_text) # test와 동일하게 원문 정답 사용
                if len(all_yhat) >= cfg.valid_bleu_sample_size:
                    break

        valid_avg_loss = valid_loss_sum / max(1, valid_steps)
        valid_bleu = compute_bleu(all_yhat, all_ground_truth)

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

        if epochs_no_improve >= cfg.patience:
            logger.info(f"[Early Stopping] Epoch {epoch + 1}에서 valid loss가 {cfg.patience}회 연속으로 개선되지 않아 조기 종료합니다")
            break

    # 조기 종료용 에포크 카운트
    actual_epochs = epoch + 1
    if actual_epochs != cfg.epochs:
        wandb.run.name = wandb_exp_name.replace(f"-ep{cfg.epochs}-", f"-ep{actual_epochs}-")

    total_train_time = time.time() - train_start_time
    wandb.summary["total_train_time_sec"] = total_train_time
    mlflow.log_metric("total_train_time_sec", total_train_time)
    mlflow.log_metric("actual_epochs", actual_epochs)
    return actual_epochs # 로그 이름 바꾸기 위한 반환


if __name__ == "__main__":
    config = load_config("config/config.yaml")
    device = resolve_device()

    # 난수 고정
    set_seed(config["seed"])

    # h_param
    cfg = TrainConfig.from_config(config)

    # tokenizer
    kor_tokenizer_name = config["model"]["kor_tokenizer"]
    en_tokenizer_name = config["model"]["en_tokenizer"]
    kor_tokenizer = AutoTokenizer.from_pretrained(kor_tokenizer_name)
    en_tokenizer = AutoTokenizer.from_pretrained(en_tokenizer_name)
    kor_pretrained_weight = AutoModel.from_pretrained(kor_tokenizer_name).embeddings.word_embeddings.weight.detach()
    en_pretrained_weight = AutoModel.from_pretrained(en_tokenizer_name).embeddings.word_embeddings.weight.detach()

    # wandb
    wandb_exp_name = cfg.exp_name

    # mlflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # 로그 기록
    log_path = f"logs/{wandb_exp_name}-{time.strftime('%Y%m%d-%H%M%S')}.log"
    logger.add(log_path, encoding = "utf-8")

    data_loader = CustomDataLoader(kor_tokenizer, en_tokenizer, max_length = cfg.max_length, batch_size = cfg.batch_size)
    train_dataloader, valid_dataloader, _ = data_loader.get_data_loader()

    logger.info(f"device: {device}")

    # 모델 구조 정보 (체크포인트에 그대로 저장되어 추론 시 모델 복원에 사용)
    model_config = {
        "kor_vocab_size": kor_tokenizer.vocab_size,
        "en_vocab_size": en_tokenizer.vocab_size,
        "embedding_dim": cfg.embedding_dim,
        "hidden_dim": cfg.hidden_dim,
        "init_scheme": cfg.init_scheme,
        "use_layer_norm": cfg.use_layer_norm,
        "padding_id": kor_tokenizer.pad_token_id, # src(한국어) pad -> Seq2Seq의 src_mask용
    }
    seq2seq = build_model(
        **model_config,
        kor_pretrained_weight = kor_pretrained_weight,
        en_pretrained_weight = en_pretrained_weight,
    ).to(device)


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

        params = asdict(cfg)
        params["lr_embedding"] = params.pop("embedding_lr") # 기존 MLflow run들과 파라미터 이름 유지
        params["lr_rnn_attn_fc"] = params.pop("rnn_attn_fc_lr")
        mlflow.log_params({
            **params,
            "seed": config["seed"],
            "kor_tokenizer": kor_tokenizer_name,
            "en_tokenizer": en_tokenizer_name,
        })
        mlflow.log_artifact("config/config.yaml", artifact_path = "config")
        mlflow.log_params({f"data.{k}": v for k, v in config["data"].items()})

        actual_epoch = train(
            cfg = cfg,
            seq2seq_model = seq2seq,
            model_config = model_config,
            train_loader = train_dataloader,
            valid_loader = valid_dataloader,
            kor_tokenizer = kor_tokenizer,
            en_tokenizer = en_tokenizer,
            device = device,
            wandb_config = config["wandb"],
            wandb_exp_name = wandb_exp_name,
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

