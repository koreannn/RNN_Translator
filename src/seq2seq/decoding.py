import torch

def get_special_token_ids(kor_tokenizer, en_tokenizer):
    sos_token_id = en_tokenizer.cls_token_id
    eos_token_id = en_tokenizer.sep_token_id

    if sos_token_id is None or eos_token_id is None:
        raise ValueError("영어 토크나이저는 반드시 cls_token과 sep_token이 있어야합니다.")

    return {
        "sos_token_id": sos_token_id,
        "eos_token_id": eos_token_id,
        "src_pad_id": kor_tokenizer.pad_token_id, # src_mask용 (한국어)
        "tgt_pad_id": en_tokenizer.pad_token_id if en_tokenizer.pad_token_id is not None else eos_token_id, # EOS 이후 채움용 (영어)
    }

@torch.no_grad()
def greedy_decoding(
    model,
    src_ids, # (bs, src_len), device에 올라간 상태
    sos_token_id,
    eos_token_id,
    src_pad_id,
    tgt_pad_id,
    max_new_tokens,
):
    device = src_ids.device
    batch_size = src_ids.size(0)

    encoder_outputs, dec_hidden = model.encoder(src_ids)
    src_mask = (src_ids != src_pad_id)

    dec_input = torch.full((batch_size, 1), sos_token_id, dtype = torch.long, device = device)
    finished = torch.zeros(batch_size, dtype = torch.bool, device = device) # 배치 내의 각 샘플이 EOS에 도달했는지 체크하기 위한 용도
    generated = []

    for _ in range(max_new_tokens):
        logits, dec_hidden, _ = model.decoder(dec_input, dec_hidden, encoder_outputs, src_mask) # (bs, 1, vocab_size)
        next_ids = torch.argmax(logits[:, -1, :], dim = -1) # (bs,)
        next_ids = next_ids.masked_fill(finished, tgt_pad_id)
        generated.append(next_ids)

        finished = finished | (next_ids == eos_token_id)
        if finished.all(): # 배치 내 모든 문장이 EOS에 도달했을 경우
            break
        dec_input = next_ids.unsqueeze(1) # (bs, 1)

    return torch.stack(generated, dim = 1) # (bs, gen_len), SOS 미포함
