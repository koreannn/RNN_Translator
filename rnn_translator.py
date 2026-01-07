"""
한국어 -> 영어 번역기
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import SGD
from transformers import AutoTokenizer
from loguru import logger
from dataloader import CustomDataLoader

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embedding_dim)
        self.rnn = nn.RNN(input_size = embedding_dim, hidden_size = hidden_dim, batch_first = True)
        
    def forward(self, src_ids):
        embedded = self.embedding(src_ids)
        outputs, hidden = self.rnn(embedded)
        
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(input_size = embedding_dim, hidden_size = hidden_dim, batch_first = True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, target_ids, enc_last_hidden):
        embedded = self.embedding(target_ids)
        outputs, dec_last_hidden = self.rnn(embedded, enc_last_hidden)
        logits = self.fc(outputs)
        return logits, dec_last_hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, padding_id = 0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = padding_id
    
    def forward(self, src_ids, target_ids):
        _, encoder_hidden = self.encoder(src_ids)
        logits, _ = self.decoder(target_ids, encoder_hidden)
        return logits


def inference(model, dataloader):
    model.eval()
    with torch.no_grad():
        for src_ids, tgt_input, tgt_label in dataloader:
            src_ids = src_ids.to(device)
            tgt_input = tgt_input.to(device)
            tgt_label = tgt_label.to(device)
            
            logits = seq2seq_model(src_ids, tgt_input)
            logits_flat = logits.reshape(-1, logits.size(-1))
            
            

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps"
    
    data = pd.read_csv("kor2en.csv")
    data_loader = CustomDataLoader(data, max_length = 50, batch_size = 32)
    train_dataloader, valid_dataloader, test_dataloader = data_loader.get_data_loader()
    
    logger.info(f"device: {device}")
    
    # Train loop
    epochs = 3
    encoder = Encoder(kor_vocab_size, embedding_dim = embedding_dim, hidden_dim = hidden_dim).to(device)
    decoder = Decoder(en_vocab_size, embedding_dim = embedding_dim, hidden_dim = hidden_dim).to(device)
    seq2seq_model = Seq2Seq(encoder, decoder).to(device)
    optimizer = SGD(seq2seq_model.parameters(), lr = 1e-3)
    
    
    
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch} / {epochs}")
        seq2seq_model.train()
        train_loss_sum = 0.0
        train_steps = 0
        
        # logits = seq2seq_model(input_tokenized, tgt_label)
        for step, (src_ids, tgt_input, tgt_label) in enumerate(train_dataloader):
            src_ids = src_ids.to(device)
            tgt_input = tgt_input.to(device)
            tgt_label = tgt_label.to(device)
            
            logits = seq2seq_model(src_ids, tgt_input)
            logits_flat = logits.reshape(-1, logits.size(-1))
            tgt_label_flat = tgt_label.reshape(-1)
            loss = F.cross_entropy(logits_flat, tgt_label_flat, ignore_index = en_pad_token_id)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_steps += 1
        
        train_avg_loss = train_loss_sum / max(1, train_steps)
        
        # valid
        seq2seq_model.eval()
        valid_loss_sum = 0.0
        valid_steps = 0
        
        with torch.no_grad():
            for src_ids, tgt_input, tgt_label in valid_dataloader:
                src_ids = src_ids.to(device)
                tgt_input = tgt_input.to(device)
                tgt_label = tgt_label.to(device)
                
                logits = seq2seq_model(src_ids, tgt_input)
                logits_flat = logits.reshape(-1, logits.size(-1))
                tgt_label_flat = tgt_label.reshape(-1)
                
                loss = F.cross_entropy(logits_flat, tgt_label_flat, ignore_index = en_pad_token_id)
                valid_loss_sum = loss.item()
                valid_steps += 1
        
        valid_avg_loss = valid_loss_sum / max(1, valid_steps)
        
        logger.info(f"epoch={epoch} train_loss={train_avg_loss:.4f} valid_loss={valid_avg_loss:.4f}")
        
        inference(seq2seq_model, test_dataloader)