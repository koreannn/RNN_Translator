import torch
from torch.utils.data import Dataset, DataLoader
from src.seq2seq.utils import load_config
from datasets import load_dataset


class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample["korean"], sample["english"] # "lemon-mint/korean_english_parallel_wiki_augmented_v1" 데이터셋의 스키마


class CustomDataLoader:
    def __init__(self, 
                kor_tokenizer, en_tokenizer,
                max_length, batch_size,
            ):
        self.config = load_config("config/config.yaml")
        data_config = self.config["data"]
        
        dataset = load_dataset(data_config["dataset2"])["train"]
        
        # 테스트셋 설정
        dataset_split = dataset.train_test_split(
            test_size = data_config["dataset2_test_ratio"], seed = self.config["seed"]
        )
        dataset_train_valid, dataset_test = dataset_split["train"], dataset_split["test"]

        # 검증셋 설정
        dataset_split = dataset_train_valid.train_test_split(
            test_size = data_config["dataset2_valid_ratio"], seed = self.config["seed"]
        )
        dataset_train, dataset_valid = dataset_split["train"], dataset_split["test"]
        
        self.train_data = TranslationDataset(dataset_train)
        self.valid_data = TranslationDataset(dataset_valid)
        self.test_data = TranslationDataset(dataset_test)
        
        
        self.kor_tokenizer = kor_tokenizer
        self.en_tokenizer = en_tokenizer
        self.en_vocab_size = self.en_tokenizer.vocab_size
        self.kor_vocab_size = self.kor_tokenizer.vocab_size # 30522, 32000
        
        self.kor_pad_token_id = self.kor_tokenizer.pad_token_id
        self.en_pad_token_id = self.en_tokenizer.pad_token_id
        self.batch_size = batch_size
        self.max_length = max_length
        self.sos_token = self.en_tokenizer.cls_token_id
        
        # 시드 고정
        self.shuffle_generator = torch.Generator()
        self.shuffle_generator.manual_seed(self.config["seed"])
        
    
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
    
    def _collate_fn_with_text(self, batch): # test용(평가에 쓸 원본 텍스트도 함께 반환
        src_ids, tgt_input, tgt_label = self._collate_fn(batch)
        src_text = [src for src, _ in batch]
        tgt_text = [tgt for _, tgt in batch]
        return src_ids, tgt_input, tgt_label, src_text, tgt_text
    
    
    def get_data_loader(self):
        train_dataloader = DataLoader(
            self.train_data,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = 1,
            collate_fn = self._collate_fn,
            drop_last = True,
            pin_memory = torch.cuda.is_available(),
            generator = self.shuffle_generator,
        )
        
        valid_dataloader = DataLoader(
            self.valid_data,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = 1,
            collate_fn = self._collate_fn,
            drop_last = True,
            pin_memory = torch.cuda.is_available(),
            generator = self.shuffle_generator,
        )
        
        test_dataloader = DataLoader(
            self.test_data,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = 1,
            collate_fn = self._collate_fn_with_text,
            drop_last = False,
            pin_memory = torch.cuda.is_available(),
        )
        return train_dataloader, valid_dataloader, test_dataloader
