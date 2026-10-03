def build_prompt(source, tokenizer, llm_cfg): # config의 프롬프트로 모델 입력 문자열 생성
    messages = [
        {"role": "system", "content": llm_cfg["prompt"]["system"]},
        {"role": "user", "content": llm_cfg["prompt"]["user"].format(source = source)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize = False, add_generation_prompt = True,
        enable_thinking = llm_cfg["generation"]["enable_thinking"],
    )