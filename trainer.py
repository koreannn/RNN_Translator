import os
import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from model import Encoder, Decoder, Seq2Seq
from torch.optim import SGD
from loguru import logger
from dataloader import CustomDataLoader

class Trainer:
    def __init__(
        self, epochs, embedding_dim, hidden_dim, 
        train_dataloader, valid_dataloader, 
        encoder_model, decoder_model, seq2seq_model, 
        device, checkpoint_dir = "checkpoints"):
        self.epochs = epochs
        self.train_loader = train_dataloader
        self.valid_loader = valid_dataloader
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.encoder = encoder_model
        self.decoder = decoder_model
        self.seq2seq_model = seq2seq_model
        self.optimizer = SGD(self.seq2seq_model.parameters(), lr = 1e-3)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.best_valid_loss = float("inf")
        self.pad_token_id = 0
        self.device = device
    
    def _checkpoint_payload(self, epoch, train_loss, valid_loss):
        return {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "encoder_state_dict": self.encoder.state_dict(),
            "decoder_state_dict": self.decoder.state_dict(),
            "seq2seq_state_dict": self.seq2seq_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "kor_vocab_size": 32000,
            "en_vocab_size": 30522,
            "pad_token_id": self.pad_token_id,
        }
    
    def save_checkpoint(self, epoch, train_loss, valid_loss):
        last_path = self.checkpoint_dir / "last.pt" # 마지막 체크포인트
        torch.save(self._checkpoint_payload(epoch, train_loss, valid_loss), last_path)
        
        epoch_path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        torch.save(self._checkpoint_payload(epoch, train_loss, valid_loss), epoch_path)
        
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(self._checkpoint_payload(epoch, train_loss, valid_loss), best_path)
            logger.info(f"Best updated in epoch {epoch}.")
    
    def train(self):
        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch + 1} / {self.epochs}")
            self.seq2seq_model.train()
            train_loss_sum = 0.0
            train_steps = 0
            
            for step, (src_ids, tgt_input, tgt_label) in enumerate(self.train_loader):
                src_ids = src_ids.to(self.device) # (bs, (한국어)seq_len)
                tgt_input = tgt_input.to(self.device) # (bs, (영어)seq_len - 1)
                tgt_label = tgt_label.to(self.device) # (bs, (영어)seq_len - 1)
                
                logits = self.seq2seq_model(src_ids, tgt_input) # (bs, seq_len, vocab_size)
                logits_flat = logits.reshape(-1, logits.size(-1)) # (bs*seq_len, vocab_size)
                tgt_label_flat = tgt_label.reshape(-1) # (bs*seq_len, )
                loss = F.cross_entropy(logits_flat, tgt_label_flat, ignore_index = self.pad_token_id)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss_sum += loss.item()
                train_steps += 1
            
            train_avg_loss = train_loss_sum / max(1, train_steps)
            
            # valid
            self.seq2seq_model.eval()
            valid_loss_sum = 0.0
            valid_steps = 0
            
            with torch.no_grad():
                for src_ids, tgt_input, tgt_label in self.valid_loader:
                    src_ids = src_ids.to(self.device)
                    tgt_input = tgt_input.to(self.device)
                    tgt_label = tgt_label.to(self.device)
                    
                    logits = self.seq2seq_model(src_ids, tgt_input) # (bs, seq_len, vocab_size)
                    logits_flat = logits.reshape(-1, logits.size(-1)) # (bs*seq_len, vocab_size)
                    tgt_label_flat = tgt_label.reshape(-1) 
                    
                    loss = F.cross_entropy(logits_flat, tgt_label_flat, ignore_index = self.pad_token_id)
                    valid_loss_sum += loss.item()
                    valid_steps += 1
            
            valid_avg_loss = valid_loss_sum / max(1, valid_steps)
            
            logger.info(f"epoch={epoch} train_loss={train_avg_loss:.4f} valid_loss={valid_avg_loss:.4f}")
            self.save_checkpoint(epoch = epoch, train_loss = train_avg_loss, valid_loss = valid_avg_loss)