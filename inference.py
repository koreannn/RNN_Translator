import torch
import pandas as pd
from loguru import logger
from transformers import AutoTokenizer
from trainer import Trainer
from model import Encoder, Decoder, Seq2Seq
from dataloader import CustomDataLoader

def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location = device)
    
    required = ["seq2seq_state_dict", "embedding_dim", "hidden_dim", "kor_vocab_size", "en_vocab_size"]
    missing = [k for k in required if k not in checkpoint]
    
    if missing:
        raise KeyError(f"Missiong keys in checkpoint: {missing}")
    return checkpoint
    
def get_model_from_checkpoint(checkpoint, device):
    embedding_dim = int(checkpoint["embedding_dim"])
    hidden_dim = int(checkpoint["hidden_dim"])
    kor_vocab_size = int(checkpoint["kor_vocab_size"])
    en_vocab_size = int(checkpoint["en_vocab_size"])
    
    encoder = Encoder(kor_vocab_size, embedding_dim, hidden_dim)
    decoder = Decoder(en_vocab_size, embedding_dim, hidden_dim)
    model = Seq2Seq(encoder, decoder, padding_id = int(checkpoint.get("pad_token_id", 0)))
    
    model.load_state_dict(checkpoint["seq2seq_state_dict"], strict = True)
    model.to(device)
    model.eval()
    return model

def inference( # greedy방식으로 하나씩 추론
    model,
    kor_tokenizer,
    en_tokenizer,
    device,
    test_dataloader,
    max_length,
    max_new_tokens = 50,
):
    sos_token_id = en_tokenizer.cls_token_id
    eos_token_id = en_tokenizer.sep_token_id
    
    if sos_token_id is None or eos_token_id is None:
        raise ValueError("영어 토크나이저는 반드시 cls_token과 sep_token이 있어야합니다.")
    
    for idx, (src_ids, _, _) in enumerate(test_dataloader): # 학습할때는 (src_ids, tgt_input, tgt_label) / 추론 시에는 오직 자신이 만든 토큰으로 다음 토큰을 예측해야함 -> (src_ids, _, _)
        
        logger.info(f"번역 전 문장: {kor_tokenizer.decode(src_ids[0].tolist(), skip_special_tokens = True)}")
        generated_ids = [sos_token_id]
        with torch.no_grad():
            _, enc_hidden = model.encoder(src_ids)
            dec_hidden = enc_hidden
            dec_input = torch.tensor([[sos_token_id]], dtype = torch.long, device = device)
            
            for _ in range(max_new_tokens):
                logits, dec_hidden = model.decoder(dec_input, dec_hidden)
                next_token_logits = logits[:, -1, :] # 배치 내 모든 hidden 중 마지막 hidden / (bs(1), vocab_size)
                next_id = int(torch.argmax(next_token_logits, dim = -1).item())
                
                generated_ids.append(next_id)
                if next_id == eos_token_id or len(generated_ids) > max_length:
                    break
                
                dec_input = torch.tensor([[next_id]], dtype = torch.long, device = device)
            translated = en_tokenizer.decode(generated_ids, skip_special_token = True).strip()
            logger.info(f"번역된 문장: {translated}")
    return
                
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device: {device}")
    
    kor_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    en_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    
    model_checkpoint_path = "checkpoints/best.pt"
    model = load_checkpoint(model_checkpoint_path, device = device)
    # logger.info(f"Loaded checkpoint from {model_checkpoint_path} (epoch = {model.get("epoch")} & validation loss = {model.get("valid_loss")})")
    logger.info(f"Loaded checkpoint from {model_checkpoint_path}")
    
    model = get_model_from_checkpoint(model, device = device)
    
    data = pd.read_csv("kor2en.csv")
    dataloader = CustomDataLoader(data, kor_tokenizer, en_tokenizer, max_length = 50, batch_size = 32)
    _, _, test_dataloader = dataloader.get_data_loader() # test의 데이터로더는 1개씩 들어가도록 고정되어있음
    
    pred = inference(
        model,
        kor_tokenizer,
        en_tokenizer,
        device,
        test_dataloader,
        max_length = 50,
        max_new_tokens = 50,
    )
    
    