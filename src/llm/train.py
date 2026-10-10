import torch
import mlflow
from pathlib import Path
from loguru import logger
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from src.seq2seq.utils import load_config, set_seed
from src.common.splits import load_splits
from src.llm.sft_data import build_sft_datasets


class AdapterUploadCallback(TrainerCallback):
    # 체크포인트를 저장할 때마다 adapter만 MLflow(S3)에 업로드 → GPU 서버가 초기화돼도 학습 결과 보존
    # optimizer 상태는 크기가 커서 제외 (이어서 학습은 못 하지만 해당 시점 모델로 평가는 가능)
    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            mlflow.log_artifact(str(ckpt_dir / name), artifact_path = f"checkpoints/step-{state.global_step}")


if __name__ == "__main__":
    config = load_config("config/config.yaml")
    llm_cfg = config["llm"]
    lora_cfg = llm_cfg["lora"]
    set_seed(config["seed"])
    base_model = lora_cfg["base_model"]
    run_tag = f"{base_model.split('/')[-1].lower()}-lora" # 예: qwen3-1.7b-lora

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    train_dataset, valid_dataset = build_sft_datasets(load_splits(config["data"], config["seed"]), tokenizer, llm_cfg, config["seed"])
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype = torch.bfloat16)

    peft_config = LoraConfig(
        r = lora_cfg["r"],
        lora_alpha = lora_cfg["alpha"],
        lora_dropout = lora_cfg["dropout"],
        target_modules = lora_cfg["target_modules"],
        task_type = "CAUSAL_LM",
    )
    sft_config = SFTConfig(
        output_dir = lora_cfg["output_dir"],
        max_length = lora_cfg["max_seq_length"], # 이미 길이로 필터링해서 실제로 잘리는 예시는 없음
        completion_only_loss = True, # 번역문(completion)에만 loss
        num_train_epochs = lora_cfg["num_train_epochs"],
        max_steps = lora_cfg.get("max_steps", -1),
        per_device_train_batch_size = lora_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size = lora_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps = lora_cfg["gradient_accumulation_steps"],
        learning_rate = lora_cfg["learning_rate"],
        lr_scheduler_type = "cosine",
        warmup_ratio = lora_cfg["warmup_ratio"],
        bf16 = True,
        gradient_checkpointing = True, # 활성값을 저장하지 않고 역전파 때 다시 계산 → 메모리 절약, 속도는 약 20~30% 느려짐
        gradient_checkpointing_kwargs = {"use_reentrant": False}, # LoRA처럼 일부만 학습할 때 권장 방식
        eval_strategy = "steps",
        eval_steps = lora_cfg["eval_steps"],
        save_strategy = "steps",
        save_steps = lora_cfg["save_steps"],
        save_total_limit = 2, # 로컬 디스크에는 최근 2개만 (S3에는 전부)
        logging_steps = lora_cfg["logging_steps"],
        report_to = ["mlflow"], # 아래에서 연 run에 loss·eval_loss가 기록됨
        seed = config["seed"],
    )

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(llm_cfg["mlflow_experiment"])
    with mlflow.start_run(
        run_name = f"train-{run_tag}",
        tags = {"run_type": "train", "model_family": "llm", "method": "lora", "model_name": base_model},
    ) as run:
        trainer = SFTTrainer(
            model = model,
            args = sft_config,
            train_dataset = train_dataset,
            eval_dataset = valid_dataset,
            processing_class = tokenizer,
            peft_config = peft_config,
            callbacks = [AdapterUploadCallback()],
        )

        # loss가 실제로 번역문에만 걸리는지 확인 (마지막 200자만 출력)
        sample = trainer.train_dataset[0]
        if "completion_mask" in sample:
            target = tokenizer.decode([t for t, m in zip(sample["input_ids"], sample["completion_mask"]) if m])
            logger.info(f"loss 대상 토큰(끝부분): {target[-200:]!r}")

        num_trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        logger.info(f"학습 파라미터 수: {num_trainable_params:,}")
        # 이름은 Trainer가 자동 기록하는 TrainingArguments와 겹치지 않게 (같은 이름에 다른 값이면 MLflow 에러)
        mlflow.log_params({
            "base_model": base_model,
            "prompt_version": llm_cfg["prompt"]["version"],
            "train_size": len(train_dataset), # 길이 필터 후 실제 학습 예시 수
            "valid_size": len(valid_dataset),
            "lora_r": lora_cfg["r"],
            "lora_alpha": lora_cfg["alpha"],
            "lora_dropout": lora_cfg["dropout"],
            "lora_target_modules": lora_cfg["target_modules"],
        })
        mlflow.log_metric("num_trainable_params", num_trainable_params)

        trainer.train()

        final_dir = Path(lora_cfg["output_dir"]) / "final"
        trainer.save_model(str(final_dir)) # adapter만 저장 (base 모델은 Hub에서 다시 받음)
        (final_dir / "mlflow_run_id.txt").write_text(run.info.run_id) # 평가 run을 이 학습 run의 자식으로 붙일 때 사용
        mlflow.log_artifacts(str(final_dir), artifact_path = "adapter")
        logger.info(f"학습 완료: {final_dir} (MLflow run {run.info.run_id})")