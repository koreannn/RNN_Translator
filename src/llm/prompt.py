def build_prompt(source, tokenizer, llm_cfg, direction): # config의 프롬프트로 모델 입력 문자열 생성
    _input, _output = direction
    propmt_template: str = llm_cfg["prompt"]["user"]
    formatted_prompt = propmt_template.format(src_lang = _input, tgt_lang = _output, source = source)
    messages = [
        {"role": "system", "content": llm_cfg["prompt"]["system"].format(src_lang = _input, tgt_lang = _output)},
        {"role": "user", "content": formatted_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize = False, add_generation_prompt = True,
        enable_thinking = llm_cfg["inference"]["generation"]["enable_thinking"],
    )
