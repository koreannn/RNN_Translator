import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class CustomDataLoader:
    class _TranslateDataset(Dataset):
        def __init__(self, src_series, tgt_series):
            self.src_series = src_series.reset_index(drop = True)
            self.tgt_series = tgt_series.reset_index(drop = True)
        def __len__(self):
            return len(self.src_series)
        def __getitem__(self, idx):
            src = str(self.src_series.iloc[idx]).strip()
            tgt = str(self.tgt_series.iloc[idx]).strip()
            return src, tgt
        
    def __init__(self, data, 
                kor_tokenizer, en_tokenizer,
                max_length = 50, batch_size = 32,
            ):
        self.data = data
        self.kor_tokenizer = kor_tokenizer
        self.en_tokenizer = en_tokenizer
        self.en_vocab_size = self.en_tokenizer.vocab_size
        self.kor_vocab_size = self.kor_tokenizer.vocab_size # 30522, 32000
        self.kor_data = self.data["원문"]
        self.en_data = self.data["번역문"]
        self.kor_pad_token_id = self.kor_tokenizer.pad_token_id
        self.en_pad_token_id = self.en_tokenizer.pad_token_id
        self.train_ratio = 0.8
        self.valid_ratio = 0.16
        self.seed = 123
        self.batch_size = batch_size
        self.max_length = max_length
        self.sos_token = self.en_tokenizer
    
    def _build_dataset(self):
        src = self.kor_data
        tgt = self.en_data
        return self._TranslateDataset(src, tgt)
    
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
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        src_ids = src_enc["input_ids"].to(torch.long)
        tgt_ids = tgt_enc["input_ids"].to(torch.long)

        # teacher forcing shift
        tgt_input = tgt_ids[:, :-1].contiguous()
        tgt_label = tgt_ids[:, 1:].contiguous()
        
        return src_ids, tgt_input, tgt_label # (bs, seq_len(logest)) / (bs, seq_len(logest)) / (bs, seq_len(logest))

    def build_data_split(self):
        dataset = self._build_dataset()
        
        train_size = int(len(dataset) * self.train_ratio)
        valid_size = int(len(dataset) * self.valid_ratio)
        test_size = len(dataset) - train_size - valid_size
        
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset,
            [train_size, valid_size, test_size],
            generator = torch.Generator().manual_seed(self.seed)
        )
        return train_dataset, valid_dataset, test_dataset
    
    def get_data_loader(self):
        train_dataset, valid_dataset, test_dataset = self.build_data_split()

        train_loader = DataLoader(
            train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = 0,
            collate_fn = self._collate_fn,
            drop_last = True,
            pin_memory = torch.cuda.is_available(),
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = 0,
            collate_fn = self._collate_fn,
            drop_last = False,
            pin_memory = torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size = 1,
            shuffle = False,
            num_workers = 0,
            collate_fn = self._collate_fn,
            drop_last = False,
            pin_memory = torch.cuda.is_available(),
        )

        return train_loader, valid_loader, test_loader
