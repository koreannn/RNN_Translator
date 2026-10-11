import json
import torch
from pathlib import Path
from loguru import logger
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.seq2seq.utils import load_config


if __name__ == "__main__":
    config = load_config("config/config.yaml")
    lora_cfg = config["llm"]["lora"]
    adapter_dir = Path(lora_cfg["output_dir"]) / "final"
    merged_dir = Path(lora_cfg["output_dir"]) / "merged"

    model = AutoModelForCausalLM.from_pretrained(lora_cfg["base_model"], dtype = torch.bfloat16) # CPU에서 병합 (GPU 불필요)
    model = PeftModel.from_pretrained(model, adapter_dir)
    num_trainable_params = sum(p.numel() for name, p in model.named_parameters() if "lora_" in name) # 학습 run의 값과 같아야 함
    model = model.merge_and_unload() # W + BA를 원래 가중치에 더해 일반 모델로 → vLLM이 그대로 로드 가능, 추론 속도도 base와 같음
    model.save_pretrained(merged_dir)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(merged_dir) # 학습 때와 같은 토크나이저·채팅 템플릿

    # 평가 스크립트가 LoRA 모델임을 알아보고, 평가 run을 학습 run의 자식으로 붙이는 데 사용
    (merged_dir / "lora_info.json").write_text(json.dumps({
        "base_model": lora_cfg["base_model"],
        "train_run_id": (adapter_dir / "mlflow_run_id.txt").read_text().strip(),
        "num_trainable_params": num_trainable_params,
    }, indent = 2))
    logger.info(f"병합 완료: {merged_dir} (LoRA 파라미터 {num_trainable_params:,}개)")