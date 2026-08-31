def greedy_search( # greedy방식으로 하나씩 추론
    model,
    kor_tokenizer,
    en_tokenizer,
    device,
    test_dataloader,
    max_length,
    max_new_tokens,
):
    sos_token_id = en_tokenizer.cls_token_id
    eos_token_id = en_tokenizer.sep_token_id
    pad_token_id = en_tokenizer.pad_token_id if en_tokenizer.pad_token_id is not None else eos_token_id
    
    if sos_token_id is None or eos_token_id is None:
        raise ValueError("영어 토크나이저는 반드시 cls_token과 sep_token이 있어야합니다.")
    
    all_yhat = [] # 번역된 문장 전체를 담고있는 리스트
    all_ground_truth = [] # 
    all_source = [] 
    
    for batch_idx, (src_ids, _, tgt_label) in enumerate(test_dataloader): # 학습할때는 (src_ids, tgt_input, tgt_label) / 추론 시에는 오직 자신이 만든 토큰으로 다음 토큰을 예측해야함 -> (src_ids, _, _)
        with torch.no_grad():
            src_ids = src_ids.to(device)
            bs = src_ids.size(0)
            
            encoder_outputs, enc_hidden = model.encoder(src_ids) # (1, bs, hidden_dim)
            src_mask = (src_ids != kor_tokenizer.pad_token_id)
            dec_hidden = enc_hidden
            dec_input = torch.full((bs, 1), sos_token_id, dtype = torch.long, device = device)
            
            generated = dec_input.clone()
            finished = torch.zeros(bs, dtype = torch.bool, device = device)
            
            for _ in range(max_new_tokens):
                logits, dec_hidden, _ = model.decoder(dec_input, dec_hidden, encoder_outputs, src_mask) # (bs, 1, vocab_size)
                next_ids = torch.argmax(logits[:, -1, :], dim = -1) # (bs, )
                next_ids = torch.where(finished, torch.full_like(next_ids, pad_token_id), next_ids)
                
                generated = torch.cat([generated, next_ids.unsqueeze(1)], dim = 1)
                finished = finished | (next_ids == eos_token_id)
                
                if finished.all() or generated.size(1) > max_length:
                    break
                dec_input = next_ids.unsqueeze(1)
            
            batch_translated = []
            for i in range(bs):
                translated = en_tokenizer.decode(generated[i].tolist(), skip_special_tokens = True).strip()
                ground_truth = en_tokenizer.decode(tgt_label[i].tolist(), skip_special_tokens = True).strip()
                all_yhat.append(translated)
                all_ground_truth.append(ground_truth)
                batch_translated.append(translated)
        
        sample_idx = random.randrange(bs)        
        logger.info(f"번역 전 문장(예시): {kor_tokenizer.decode(src_ids[sample_idx].tolist(), skip_special_tokens = True)}")
        logger.info(f"번역된 문장(예시): {all_yhat[sample_idx]}")
                
    bleu_result = sacrebleu.corpus_bleu(all_yhat, [all_ground_truth])
    logger.info(f"Test corpus BLEU 점수: {bleu_result.score:.2f}")
    
    return bleu_result.score