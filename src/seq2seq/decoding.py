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


def _length_penalty(length, alpha): # GNMT 길이 페널티
    return ((5 + length) ** alpha) / ((5 + 1) ** alpha)

@torch.no_grad()
def beam_decoding(
    model,
    src_ids, # (1, src_len), 문장 하나씩 디코딩
    sos_token_id,
    eos_token_id,
    src_pad_id,
    tgt_pad_id, # 미사용 (greedy_decoding과 인자 통일용)
    max_new_tokens,
    max_length,
    beam_size = 4,
    alpha = 0.6, # 길이 페널티 하이퍼파라미터 (표준값 0.6 ~ 0.7)
):
    device = src_ids.device

    encoder_outputs, enc_hidden = model.encoder(src_ids) # (1, src_len, hidden_dim) / (1, 1, hidden_dim)
    src_mask = (src_ids != src_pad_id) # (1, src_len)

    dec_input = torch.tensor([[sos_token_id]], dtype = torch.long, device = device)
    logits_step, step_hidden, _ = model.decoder(dec_input, enc_hidden, encoder_outputs, src_mask)
    log_probs = torch.log_softmax(logits_step[:, -1, :], dim = -1)
    vocab_size = log_probs.size(-1)

    top_scores, top_tokens = torch.topk(log_probs[0], beam_size)
    beam_seqs = [[sos_token_id, top_tokens[b].item()] for b in range(beam_size)]
    beam_scores = top_scores.clone()
    beam_hidden = step_hidden.expand(-1, beam_size, -1).contiguous()

    # attention이 참조할 encoder_outputs/src_mask도 빔 개수만큼 복제 (같은 소스 문장이라 재정렬 불필요)
    beam_encoder_outputs = encoder_outputs.expand(beam_size, -1, -1).contiguous()
    beam_src_mask = src_mask.expand(beam_size, -1).contiguous()

    completed_beams = []

    active_mask = []
    for b in range(beam_size):
        if beam_seqs[b][-1] == eos_token_id:
            completed_beams.append((beam_scores[b].item() / _length_penalty(len(beam_seqs[b]), alpha), beam_seqs[b]))
            active_mask.append(False)
        else:
            active_mask.append(True)
    active_mask = torch.tensor(active_mask, device = device)

    for _ in range(max_new_tokens - 1):
        if not active_mask.any():
            break

        last_tokens = torch.tensor([[seq[-1]] for seq in beam_seqs], dtype = torch.long, device = device)
        logits_batch, beam_hidden, _ = model.decoder(last_tokens, beam_hidden, beam_encoder_outputs, beam_src_mask)
        log_probs = torch.log_softmax(logits_batch[:, -1, :], dim = -1)

        candidate_scores = beam_scores.unsqueeze(1) + log_probs
        candidate_scores[~active_mask] = -float('inf')

        top_scores_new, top_flat_idx = torch.topk(candidate_scores.view(-1), beam_size)
        parent_beams = top_flat_idx // vocab_size
        next_tokens = top_flat_idx % vocab_size

        beam_hidden = beam_hidden[:, parent_beams, :].contiguous()
        # beam_encoder_outputs / beam_src_mask는 전부 같은 소스 문장이라 그대로 유지 (재정렬 불필요)

        new_seqs, new_active = [], []
        for parent, token, score in zip(parent_beams.tolist(), next_tokens.tolist(), top_scores_new.tolist()):
            new_seq = beam_seqs[parent] + [token]
            new_seqs.append(new_seq)

            if token == eos_token_id or len(new_seq) > max_length:
                completed_beams.append((score / _length_penalty(len(new_seq), alpha), new_seq))
                new_active.append(False)
            else:
                new_active.append(True)

        beam_seqs = new_seqs
        beam_scores = top_scores_new
        active_mask = torch.tensor(new_active, device = device)

        if len(completed_beams) >= beam_size: # 완료된 최고 점수를 진행 중인 빔이 더 이상 넘을 수 없으면 조기 종료
            completed_beams.sort(key = lambda x: x[0], reverse = True)
            best_done = completed_beams[0][0]
            active_idx = active_mask.nonzero(as_tuple = True)[0]
            if len(active_idx) > 0:
                first_active = active_idx[0].item()
                best_ongoing = beam_scores[first_active].item() / _length_penalty(len(beam_seqs[first_active]), alpha)
                if best_done >= best_ongoing:
                    break
            else:
                break

    for b, seq in enumerate(beam_seqs): # max_new_tokens에 도달할 때까지 끝나지 않은 빔도 후보에 포함
        if active_mask[b]:
            completed_beams.append((beam_scores[b].item() / _length_penalty(len(seq), alpha), seq))

    completed_beams.sort(key = lambda x: x[0], reverse = True)
    return completed_beams[0][1] # list[int], SOS 포함

@torch.no_grad()
def sampling_decoding( # temperature + top-k + top-p 샘플링
    model,
    src_ids, # (1, src_len), 문장 하나씩 디코딩
    sos_token_id,
    eos_token_id,
    src_pad_id,
    tgt_pad_id, # 미사용 (greedy_decoding과 인자 통일용)
    max_new_tokens,
    max_length,
    temperature,
    top_k,
    top_p,
):
    device = src_ids.device

    encoder_outputs, dec_hidden = model.encoder(src_ids)
    src_mask = (src_ids != src_pad_id)
    dec_input = torch.tensor([[sos_token_id]], dtype = torch.long, device = device)
    generated_ids = [sos_token_id]

    for _ in range(max_new_tokens):
        logits, dec_hidden, _ = model.decoder(dec_input, dec_hidden, encoder_outputs, src_mask)
        scaled_logits = logits[:, -1, :].squeeze(0) / temperature

        # 1. Top-K 필터링
        if top_k > 0:
            criteria_logit = torch.topk(scaled_logits, top_k)[0][-1]
            scaled_logits[scaled_logits < criteria_logit] = -float('inf')

        # 2. Top-P 필터링
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending = True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim = -1), dim = -1)

            # 누적확률이 top_p를 처음 넘는 토큰까지는 남기도록 한 칸 오른쪽으로 밀기
            sorted_to_remove = cumulative_probs > top_p
            shifted_to_remove = sorted_to_remove.clone()
            shifted_to_remove[1:] = sorted_to_remove[:-1]
            shifted_to_remove[0] = False

            scaled_logits[sorted_indices[shifted_to_remove]] = -float('inf')

        probs = torch.softmax(scaled_logits, dim = -1)
        next_id = int(torch.multinomial(probs, num_samples = 1).item())

        generated_ids.append(next_id)
        if next_id == eos_token_id or len(generated_ids) > max_length:
            break

        dec_input = torch.tensor([[next_id]], dtype = torch.long, device = device)

    return generated_ids # list[int], SOS 포함
