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

kor_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
en_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
en_vocab_size, kor_vocab_size = en_tokenizer.vocab_size, kor_tokenizer.vocab_size # 30522, 32000
data = pd.read_csv("kor2en.csv")

kor_data = data["원문"]
en_data = data["번역문"]
kor_vocab = kor_tokenizer.vocab
en_vocab = en_tokenizer.vocab
embedding_dim = 512
hidden_dim = 512
device = "cuda" if torch.cuda.is_available() else "mps"


kor_pad_token_id = kor_tokenizer.pad_token_id
en_pad_token_id = en_tokenizer.pad_token_id
max_length = 50 # 문장의 최대 길이

class CustomDataset(Dataset):
    def __init__(self, src_series, tgt_series):
        self.src_series = src_series.reset_index(drop = True)
        self.tgt_series = tgt_series.reset_index(drop = True)
    
    def __len__(self):
        return len(self.src_series)
    
    def __getitem__(self, idx):
        src = str(self.src_series.iloc[idx]).strip()
        tgt = str(self.tgt_series.iloc[idx]).strip()
        return src, tgt

dataset = CustomDataset(kor_data, en_data)
train_size = int(0.8 * len(dataset))
valid_size = int((len(dataset)) * 0.16) # 전체 데이터셋중 16%
test_size = len(dataset) - train_size - valid_size # 전체 데이터셋중 4%

train_dataset, valid_dataset, test_dataset = random_split(
    dataset,
    [train_size, valid_size, test_size],
    generator = torch.Generator().manual_seed(123),
)

def collate_fn(batch: list[tuple[str, str]]):
    src_text = [src for src, _ in batch]
    tgt_text = [tgt for _, tgt in batch]
    
    src_enc = kor_tokenizer(
        src_text,
        padding = "longest",
        max_length = max_length,
        truncation = True,
        return_tensors = "pt",
    )
    
    tgt_enc = en_tokenizer(
        tgt_text,
        padding = "longest",
        max_length = max_length,
        truncation = True,
        return_tensors = "pt",
    )
    
    src_ids = src_enc["input_ids"].to(torch.long)
    tgt_ids = tgt_enc["input_ids"].to(torch.long)
    tgt_input = tgt_ids[:, :-1].contiguous()
    tgt_label = tgt_ids[:, 1:].contiguous()
    
    return src_ids, tgt_input, tgt_label


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

def encoder_smoke_test(): # Encoder 단위 테스트
    logger.info("=====Encoder Smoke Test 입니다.=====")
    src_text = "나는 자연어 처리를 좋아합니다."
    
    input_ids = torch.tensor(
        kor_tokenizer(src_text)["input_ids"],
        dtype = torch.long
    ).to(device).unsqueeze(0) # [bs, seq_len]
    
    encoder = Encoder(kor_vocab_size, embedding_dim, hidden_dim).to(device)
    outputs, last_hidden = encoder(input_ids) # outputs: [bs, seq_len, hidden_dim] / last_hidden(Context Vector): [bs, 1, hidden_dim]
    logger.info(f"outputs.shape: {outputs.shape}")
    logger.info(f"last_hidden.shape: {last_hidden.shape}")
    logger.info("=====Encoder Smoke Test 종료.=====")
    return

def decoder_smoke_test(): # Decoder 단위 테스트
    logger.info("=====Decoder Smoke Test 시작=====")
    # 1. 인코더 입력 만들기
    src_text = "나는 자연어 처리를 좋아합니다."
    input_ids = torch.tensor(
        kor_tokenizer(src_text)["input_ids"],
        dtype = torch.long
    ).to(device).unsqueeze(0) # [bs, seq_len]
    
    encoder = Encoder(kor_vocab_size, embedding_dim, hidden_dim).to(device)
    _, encoder_hidden = encoder(input_ids)
    
    # 2. 디코더 입력 준비(Teacher Forcing)
    tgt_text = "I love natural language processing"
    
    tgt_ids = torch.tensor(
        en_tokenizer(tgt_text)["input_ids"], 
        dtype = torch.long
    ).to(device).unsqueeze(0) # [bs, seq_len]
    # Teacher Forcing
    
    tgt_input = tgt_ids[:, :-1] # 현재 예측해야하는 토큰
    # tgt_label = tgt_ids[:, 1:] # 현재 예측해야하는 ground truth
    
    # 디코더 forward
    decoder = Decoder(en_vocab_size, embedding_dim, hidden_dim).to(device)
    
    logits, dec_last_hidden = decoder(tgt_input, encoder_hidden) # logits: [bs, seq_len, vocab_size] / dec_last_hidden: [bs, 1, hidden_dim]
    logger.info(f"(Decoder)logits.shape: {logits.shape}")
    logger.info(f"(Decoder)dec_last_hidden.shape: {dec_last_hidden.shape}")
    logger.info("=====Decoder Smoke Test 종료=====")

def inference():
    pass

if __name__ == "__main__":
    # encoder_smoke_test()
    # decoder_smoke_test()
    logger.info(f"device: {device}")
    
    # Train loop
    epochs = 3
    encoder = Encoder(kor_vocab_size, embedding_dim = embedding_dim, hidden_dim = hidden_dim).to(device)
    decoder = Decoder(en_vocab_size, embedding_dim = embedding_dim, hidden_dim = hidden_dim).to(device)
    seq2seq_model = Seq2Seq(encoder, decoder).to(device)
    optimizer = SGD(seq2seq_model.parameters(), lr = 1e-3)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size = 32,
        shuffle = True,
        num_workers = 0,
        collate_fn = collate_fn,
        drop_last = True,
        pin_memory = torch.cuda.is_available(),
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size = 32,
        shuffle = False,
        collate_fn = collate_fn,
    )
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch} / {epochs}")
        seq2seq_model.train()
        train_loss_sum = 0.0
        train_steps = 0
        
        # logits = seq2seq_model(input_tokenized, tgt_label)
        for step, (src_ids, tgt_input, tgt_label) in enumerate(train_loader):
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
            for src_ids, tgt_input, tgt_label in valid_loader:
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
        