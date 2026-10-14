import torch
import mlflow
from pathlib import Path
from loguru import logger
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.seq2seq.utils import load_config, set_seed
from src.common.splits import load_splits


def _sample(split, size, seed): # LoRA(src/llm/sft_data.py)와 같은 샘플링 → 같은 2만 쌍
    if size is None or size >= len(split):
        return split
    return split.shuffle(seed = seed).select(range(size))


def tokenize_and_filter(split, tokenizer, max_length):
    # 원문 → input_ids, 정답 → labels. 길이를 넘는 쌍은 자르지 않고 제외
    # (잘라서 학습하면 번역을 중간에 끊거나 입력에 없는 내용을 지어내는 법을 배움)
    def tokenize(batch):
        return tokenizer(batch["korean"], text_target = batch["english"], truncation = False)

    tokenized = split.map(tokenize, batched = True, remove_columns = split.column_names)
    before = len(tokenized)
    tokenized = tokenized.filter(lambda ex: len(ex["input_ids"]) <= max_length and len(ex["labels"]) <= max_length)
    logger.info(f"max_length({max_length}) 초과로 제외: {before - len(tokenized)} / {before}")
    return tokenized


if __name__ == "__main__":
    config = load_config("config/config.yaml")
    ft_cfg = config["mini_transformer"]["finetune"]
    set_seed(config["seed"])
    base_model = ft_cfg["base_model"]
    run_tag = f"{base_model.split('/')[-1].lower()}-ft" # 예: opus-mt-ko-en-ft

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

    splits = load_splits(config["data"], config["seed"])
    train_dataset = tokenize_and_filter(_sample(splits["train"], ft_cfg["train_size"], config["seed"]), tokenizer, ft_cfg["max_length"])
    valid_dataset = tokenize_and_filter(_sample(splits["valid"], ft_cfg["valid_size"], config["seed"]), tokenizer, ft_cfg["max_length"])

    args = Seq2SeqTrainingArguments(
        output_dir = ft_cfg["output_dir"],
        num_train_epochs = ft_cfg["num_train_epochs"],
        max_steps = ft_cfg.get("max_steps", -1),
        per_device_train_batch_size = ft_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size = ft_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps = ft_cfg["gradient_accumulation_steps"],
        learning_rate = ft_cfg["learning_rate"],
        lr_scheduler_type = "linear",
        warmup_ratio = ft_cfg["warmup_ratio"],
        weight_decay = 0.01,
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(), # GPU면 bf16, Mac(MPS)·CPU는 fp32
        eval_strategy = "steps",
        eval_steps = ft_cfg["eval_steps"],
        save_strategy = "steps",
        save_steps = ft_cfg["save_steps"],
        save_total_limit = 2,
        load_best_model_at_end = True, # valid loss가 가장 낮은 체크포인트로 끝냄
        metric_for_best_model = "eval_loss",
        greater_is_better = False,
        logging_steps = ft_cfg["logging_steps"],
        report_to = ["mlflow"], # 아래에서 연 run에 loss·eval_loss 기록
        seed = config["seed"],
    )

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mini_transformer"]["mlflow_experiment"])
    with mlflow.start_run(
        run_name = f"train-{run_tag}",
        tags = {"run_type": "train", "model_family": "mini-transformer", "method": "fine-tuned", "model_name": base_model},
    ) as run:
        trainer = Seq2SeqTrainer(
            model = model,
            args = args,
            train_dataset = train_dataset,
            eval_dataset = valid_dataset,
            data_collator = DataCollatorForSeq2Seq(tokenizer, model = model), # 배치별 padding, labels의 padding은 -100(loss 제외)
            processing_class = tokenizer,
        )
        # 이름은 Trainer가 자동 기록하는 TrainingArguments와 겹치지 않게
        mlflow.log_params({
            "base_model": base_model,
            "train_size": len(train_dataset), # 길이 필터 후 실제 학습 쌍 수
            "valid_size": len(valid_dataset),
            "filter_max_length": ft_cfg["max_length"],
        })
        mlflow.log_metric("num_trainable_params", sum(p.numel() for p in model.parameters() if p.requires_grad))

        trainer.train()

        final_dir = Path(ft_cfg["output_dir"]) / "final"
        trainer.save_model(str(final_dir)) # 가장 좋은 체크포인트(load_best_model_at_end)
        tokenizer.save_pretrained(str(final_dir))
        (final_dir / "mlflow_run_id.txt").write_text(run.info.run_id) # 평가 run을 이 학습 run의 자식으로 붙일 때 사용
        mlflow.log_artifacts(str(final_dir), artifact_path = "model") # 약 300MB → 프록시로 업로드
        logger.info(f"학습 완료: {final_dir} (MLflow run {run.info.run_id})")