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
from utils import load_config

if __name__ == "__main__":
    config = load_config("config.yaml")
    device = "cuda" if torch.cuda.is_available() else "mps"
    embedding_dim = config["train"]["h_param"]["embedding_dim"]
    hidden_dim = config["train"]["h_param"]["hidden_dim"]
    epochs = config["train"]["h_param"]["epochs"]
    max_length = config["train"]["h_param"]["max_length"]
    batch_size = config["train"]["h_param"]["batch_size"]
    data_path = config["data"]["data_path"]
    kor_tokenizer_name = config["model"]["kor_tokenizer"]
    en_tokenizer_name = config["model"]["en_tokenizer"]
    
    data = pd.read_csv(data_path)
    kor_tokenizer = AutoTokenizer.from_pretrained(kor_tokenizer_name)
    en_tokenizer = AutoTokenizer.from_pretrained(en_tokenizer_name)
    data_loader = CustomDataLoader(data, kor_tokenizer, en_tokenizer, max_length = max_length, batch_size = batch_size)
    train_dataloader, valid_dataloader, test_dataloader = data_loader.get_data_loader()
    
    logger.info(f"device: {device}")
    
    encoder = Encoder(vocab_size = kor_tokenizer.vocab_size, embedding_dim = embedding_dim, hidden_dim = hidden_dim).to(device)
    decoder = Decoder(vocab_size = en_tokenizer.vocab_size, embedding_dim = embedding_dim, hidden_dim = hidden_dim).to(device)
    seq2seq = Seq2Seq(encoder, decoder).to(device)
    
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