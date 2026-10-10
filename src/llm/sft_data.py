from datasets import Dataset
from loguru import logger

from src.llm.prompt import build_prompt


def _sample(split, size, seed): # size가 null이거나 split보다 크면 전체 사용
    if size is None or size >= len(split):
        return split
    return split.shuffle(seed = seed).select(range(size))


def to_prompt_completion(split, tokenizer, llm_cfg):
    # prompt: zero-shot 추론과 같은 입력(build_prompt), completion: 정답 번역 + 종료 토큰
    eos = tokenizer.eos_token # Qwen3: <|im_end|> → 번역이 끝나면 멈추는 법을 학습
    return Dataset.from_list([
        {"prompt": build_prompt(ex["korean"], tokenizer, llm_cfg), "completion": ex["english"] + eos}
        for ex in split
    ])


def filter_by_length(dataset, tokenizer, max_seq_length):
    # 잘라서 학습하면 번역을 중간에 끊는 법을 배우므로, 상한을 넘는 예시는 자르지 않고 제외
    def fits(ex):
        return len(tokenizer(ex["prompt"] + ex["completion"], add_special_tokens = False)["input_ids"]) <= max_seq_length

    before = len(dataset)
    dataset = dataset.filter(fits)
    logger.info(f"max_seq_length({max_seq_length}) 초과로 제외: {before - len(dataset)} / {before}")
    return dataset


def build_sft_datasets(splits, tokenizer, llm_cfg, seed): # RNN과 같은 split에서 샘플링 (test는 사용하지 않음)
    lora_cfg = llm_cfg["lora"]
    datasets = {}
    for name, size in (("train", lora_cfg["train_size"]), ("valid", lora_cfg["valid_size"])):
        sampled = _sample(splits[name], size, seed)
        datasets[name] = filter_by_length(to_prompt_completion(sampled, tokenizer, llm_cfg), tokenizer, lora_cfg["max_seq_length"])
    return datasets["train"], datasets["valid"]