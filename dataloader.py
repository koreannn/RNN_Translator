import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from utils import load_config
from datasets import load_dataset


class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample["translation"]["ko"], sample["translation"]["en"]


class CustomDataLoader:
    def __init__(self, 
                kor_tokenizer, en_tokenizer,
                max_length, batch_size,
            ):
        dataset = load_dataset("Helsinki-NLP/opus-100", "en-ko")
        self.config = load_config("config.yaml")
        self.train_data = TranslationDataset(dataset["train"])
        self.valid_data = TranslationDataset(dataset["validation"])
        self.test_data = TranslationDataset(dataset["test"])
        
        self.kor_tokenizer = kor_tokenizer
        self.en_tokenizer = en_tokenizer
        self.en_vocab_size = self.en_tokenizer.vocab_size
        self.kor_vocab_size = self.kor_tokenizer.vocab_size # 30522, 32000
        
        self.kor_pad_token_id = self.kor_tokenizer.pad_token_id
        self.en_pad_token_id = self.en_tokenizer.pad_token_id
        self.batch_size = batch_size
        self.max_length = max_length
        self.sos_token = self.en_tokenizer
    
    def _collate_fn(self, batch):
        src_text = [src for src, _ in batch]
        tgt_text = [tgt for _, tgt in batch]
        
        src_enc = self.kor_tokenizer(
            src_text,
            padding = "longest",
            truncation = True,
            max_length = self.max_length,
            return_tensors = "pt",
        )
        tgt_enc = self.en_tokenizer(
            tgt_text,
            padding = "longest",
            truncation = True,
            max_length = self.max_length,
            return_tensors = "pt",
        )

        src_ids = src_enc["input_ids"].to(torch.long)
        tgt_ids = tgt_enc["input_ids"].to(torch.long)

        # teacher forcing shift
        tgt_input = tgt_ids[:, :-1].contiguous()
        tgt_label = tgt_ids[:, 1:].contiguous()
        
        return src_ids, tgt_input, tgt_label # (bs, seq_len(logest)) / (bs, seq_len(logest)) / (bs, seq_len(logest))
    
    def get_data_loader(self):
        train_dataloader = DataLoader(
            self.train_data,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = 4,
            collate_fn = self._collate_fn,
            drop_last = True,
            pin_memory = torch.cuda.is_available(),
        )
        
        valid_dataloader = DataLoader(
            self.valid_data,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = 4,
            collate_fn = self._collate_fn,
            drop_last = True,
            pin_memory = torch.cuda.is_available(),
        )
        
        test_dataloader = DataLoader(
            self.test_data,
            batch_size = 1,
            shuffle = True,
            num_workers = 0,
            collate_fn = self._collate_fn,
            drop_last = True,
            pin_memory = torch.cuda.is_available(),
        )
        return train_dataloader, valid_dataloader, test_dataloader
