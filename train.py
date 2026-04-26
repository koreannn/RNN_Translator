"""
한국어 -> 영어 번역기
"""
import torch
import torch.nn.functional as F
import wandb
import sacrebleu

from pathlib import Path
from transformers import AutoTokenizer
from torch.optim import Adam
from loguru import logger
from dataloader import CustomDataLoader
from model import Encoder, Decoder, Seq2Seq
from utils import load_config


def train(
    epochs, lr, batch_size, embedding_dim, hidden_dim,
    train_loader, valid_loader,
    kor_vocab_size, en_vocab_size, en_tokenizer,
    encoder, decoder, seq2seq_model,
    device, wandb_project_name,
    wandb_entity, wandb_project, wandb_architecture,
    checkpoint_dir = "checkpoints",
):
    optimizer = Adam(seq2seq_model.parameters(), lr = lr)
    checkpoint_dir = Path(checkpoint_dir)
    best_valid_loss = float("inf")
    pad_token_id = 0

    def save_checkpoint(epoch, train_loss, valid_loss):
        nonlocal best_valid_loss
        checkpoint_dir.mkdir(parents = True, exist_ok = True)
        payload = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "seq2seq_state_dict": seq2seq_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "kor_vocab_size": kor_vocab_size,
            "en_vocab_size": en_vocab_size,
            "pad_token_id": pad_token_id,
        }
        torch.save(payload, checkpoint_dir / "last.pt")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(payload, checkpoint_dir / "best.pt")
            logger.info(f"Best updated in epoch {epoch + 1}.")

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

    for epoch in range(epochs):
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

        seq2seq_model.eval()
        valid_loss_sum = 0.0
        valid_steps = 0
        
        # BLEU Score
        valid_steps = 0
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
                
                # BLEU 집계
                pred_ids = torch.argmax(logits, dim = -1) # (bs, seq_len)
                for i in range(pred_ids.size(0)):
                    hyp = en_tokenizer.decode(pred_ids[i].tolist(), skip_special_tokens = True).strip()
                    ref = en_tokenizer.decode(tgt_label[i].tolist(), skip_special_tokens = True).strip()
                    all_yhat.append(hyp)
                    all_ground_truth.append(ref)

        valid_avg_loss = valid_loss_sum / max(1, valid_steps)
        
        bleu_result = sacrebleu.corpus_bleu(all_yhat, [all_ground_truth])
        valid_bleu = bleu_result.score

        logger.info(f"epoch = {epoch + 1} train_loss = {train_avg_loss:.4f} valid_loss = {valid_avg_loss:.4f}"
                    f"valid_bleu = {valid_bleu:.2f}")
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_avg_loss,
            "valid_loss": valid_avg_loss,
            "valid_bleu": valid_bleu,
        })
        save_checkpoint(epoch = epoch, train_loss = train_avg_loss, valid_loss = valid_avg_loss)


if __name__ == "__main__":
    config = load_config("config.yaml")
    device = "cuda" if torch.cuda.is_available() else "mps"

    # h_param
    epochs = config["train"]["h_param"]["epochs"]
    learning_rate = config["train"]["h_param"]["learning_rate"]
    batch_size = config["train"]["h_param"]["batch_size"]
    embedding_dim = config["train"]["h_param"]["embedding_dim"]
    hidden_dim = config["train"]["h_param"]["hidden_dim"]
    max_length = config["train"]["h_param"]["max_length"]

    # tokenizer
    kor_tokenizer_name = config["model"]["kor_tokenizer"]
    en_tokenizer_name = config["model"]["en_tokenizer"]
    kor_tokenizer = AutoTokenizer.from_pretrained(kor_tokenizer_name)
    en_tokenizer = AutoTokenizer.from_pretrained(en_tokenizer_name)
    kor_vocab_size = kor_tokenizer.vocab_size
    en_vocab_size = en_tokenizer.vocab_size

    # wandb
    wandb_project_name = config["train"]["wandb_exp_name"]
    wandb_project = config["wandb"]["wandb_project"]
    wandb_entity = config["wandb"]["wandb_entity"]
    wandb_architecture = config["wandb"]["wandb_architecture"]

    data_loader = CustomDataLoader(kor_tokenizer, en_tokenizer, max_length = max_length, batch_size = batch_size)
    train_dataloader, valid_dataloader, _ = data_loader.get_data_loader()

    logger.info(f"device: {device}")

    encoder = Encoder(vocab_size = kor_vocab_size, embedding_dim = embedding_dim, hidden_dim = hidden_dim).to(device)
    decoder = Decoder(vocab_size = en_vocab_size, embedding_dim = embedding_dim, hidden_dim = hidden_dim).to(device)
    seq2seq = Seq2Seq(encoder, decoder).to(device)

    train(
        epochs = epochs,
        lr = learning_rate,
        batch_size = batch_size,
        embedding_dim = embedding_dim,
        hidden_dim = hidden_dim,
        train_loader = train_dataloader,
        valid_loader = valid_dataloader,
        kor_vocab_size = kor_vocab_size,
        en_vocab_size = en_vocab_size,
        en_tokenizer = en_tokenizer,
        encoder = encoder,
        decoder = decoder,
        seq2seq_model = seq2seq,
        device = device,
        wandb_project = wandb_project,
        wandb_entity = wandb_entity,
        wandb_architecture = wandb_architecture,
        wandb_project_name = wandb_project_name,
    )
