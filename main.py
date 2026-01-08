"""
한국어 -> 영어 번역기
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer
from loguru import logger
from dataloader import CustomDataLoader
from model import Encoder, Decoder, Seq2Seq
from trainer import Trainer
from inference import inference

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps"
    embedding_dim = 512
    hidden_dim = 512
    epochs = 100
    max_length = 50
    batch_size = 32
    
    data = pd.read_csv("kor2en.csv")
    kor_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    en_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    data_loader = CustomDataLoader(data, kor_tokenizer, en_tokenizer, max_length = max_length, batch_size = batch_size)
    train_dataloader, valid_dataloader, test_dataloader = data_loader.get_data_loader()
    
    logger.info(f"device: {device}")
    
    encoder = Encoder(vocab_size = kor_tokenizer.vocab_size, embedding_dim = embedding_dim, hidden_dim = hidden_dim).to(device)
    decoder = Decoder(vocab_size = en_tokenizer.vocab_size, embedding_dim = embedding_dim, hidden_dim = hidden_dim).to(device)
    seq2seq = Seq2Seq(encoder, decoder).to(device)
    
    breakpoint()
    trainer = Trainer(
        epochs = epochs,
        embedding_dim = embedding_dim,
        hidden_dim = hidden_dim,
        train_dataloader = train_dataloader,
        valid_dataloader = valid_dataloader,
        encoder_model = encoder,
        decoder_model = decoder,
        seq2seq_model = seq2seq,
        device = device
    )
    trainer.train()
    # inference(
    #     seq2seq,
    #     kor_tokenizer,
    #     en_tokenizer,
    #     device,
    #     test_dataloader,
    #     max_length,
    #     max_new_tokens = 50
    # )