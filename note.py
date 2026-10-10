from transformers import AutoTokenizer
from src.seq2seq.utils import load_config
from src.common.splits import load_splits
from src.llm.sft_data import build_sft_datasets

config = load_config("config/config.yaml")
llm_cfg = config["llm"]
tokenizer = AutoTokenizer.from_pretrained(llm_cfg["lora"]["base_model"])
train, valid = build_sft_datasets(load_splits(config["data"], config["seed"]), tokenizer, llm_cfg, config["seed"])

print(len(train), len(valid))
print(repr(train[0]["prompt"] + train[0]["completion"])) # <think>\n\n</think>\n\n 뒤에 번역, 끝에 <|im_end|> 하나인지