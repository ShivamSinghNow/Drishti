from __future__ import annotations

import argparse
import importlib.metadata
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from build_dataloader import (
    DEFAULT_DATA_DIR,
    QwenVlDataCollator,
    load_jsonl_dataset,
    tensor_shapes,
    validate_batch,
)
from preprocess_samples import MODEL_NAME


PROJECT_NAME = "tbx11k-qwen-vl-finetuning"
DEFAULT_OUTPUT_DIR = Path("outputs/qlora-run")
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass(frozen=True)
class TrainingComponents:
    processor: Any
    train_dataset: Any
    eval_dataset: Any
    data_collator: QwenVlDataCollator


def build_quantization_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def build_lora_config(rank: int, alpha: int, dropout: float) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(LORA_TARGET_MODULES),
        bias="none",
        inference_mode=False,
    )


def build_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    report_to = [] if args.wandb_mode == "disabled" else ["wandb"]
    return TrainingArguments(
        output_dir=str(args.output_dir),
        run_name=args.run_name,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        bf16=args.bf16,
        fp16=False,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        report_to=report_to,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )


def configure_wandb(args: argparse.Namespace) -> None:
    if args.wandb_mode == "disabled":
        os.environ["WANDB_DISABLED"] = "true"
        return
    os.environ.setdefault("WANDB_PROJECT", args.project_name)
    os.environ.setdefault("WANDB_MODE", args.wandb_mode)


def package_available(distribution_name: str) -> bool:
    try:
        importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def require_full_training_environment() -> None:
    missing = [
        name
        for name in ("bitsandbytes", "optimum-amd")
        if not package_available(name)
    ]
    if missing:
        raise RuntimeError(f"Missing required training package(s): {', '.join(missing)}")
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm/CUDA is not available. Full QLoRA training must run on the AMD GPU host.")


def load_training_components(args: argparse.Namespace) -> TrainingComponents:
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    train_dataset = load_jsonl_dataset("train", args.data_dir, args.train_limit)
    eval_dataset = load_jsonl_dataset("val", args.data_dir, args.eval_limit)
    data_collator = QwenVlDataCollator(processor)
    return TrainingComponents(processor, train_dataset, eval_dataset, data_collator)


def dry_run_batch_summary(args: argparse.Namespace, components: TrainingComponents) -> dict[str, tuple[int, ...]]:
    dataloader = DataLoader(
        components.train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=components.data_collator,
    )
    batch = next(iter(dataloader))
    validate_batch("train", batch)
    return tensor_shapes(batch)


def run_dry_run(args: argparse.Namespace) -> int:
    quantization_config = build_quantization_config()
    lora_config = build_lora_config(args.rank, args.alpha, args.lora_dropout)
    training_args = build_training_arguments(args)
    components = load_training_components(args)
    shapes = dry_run_batch_summary(args, components)

    print("Dry run complete. Full model was not loaded and training was not started.")
    print(f"model_name={args.model_name}")
    print(f"train_records={len(components.train_dataset)} eval_records={len(components.eval_dataset)}")
    print(f"quantization=4bit:{quantization_config.load_in_4bit} type:{quantization_config.bnb_4bit_quant_type}")
    print(f"lora_rank={lora_config.r} lora_alpha={lora_config.lora_alpha} targets={list(lora_config.target_modules)}")
    print(f"trainer_output_dir={training_args.output_dir}")
    for key, shape in sorted(shapes.items()):
        print(f"batch_{key}_shape={shape}")
    return 0


def train(args: argparse.Namespace) -> int:
    require_full_training_environment()
    configure_wandb(args)

    quantization_config = build_quantization_config()
    lora_config = build_lora_config(args.rank, args.alpha, args.lora_dropout)
    training_args = build_training_arguments(args)
    components = load_training_components(args)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=components.train_dataset,
        eval_dataset=components.eval_dataset,
        data_collator=components.data_collator,
        processing_class=components.processor,
    )
    trainer.train()
    trainer.save_model()
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Drishti Qwen2-VL with QLoRA.")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Base Qwen2-VL model name.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, type=Path, help="Directory with train.jsonl and val.jsonl.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path, help="Checkpoint output directory.")
    parser.add_argument("--project-name", default=PROJECT_NAME, help="W&B project name.")
    parser.add_argument("--run-name", default="drishti-qlora-run", help="Trainer/W&B run name.")
    parser.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"))
    parser.add_argument("--lr", default=2e-4, type=float, help="Learning rate.")
    parser.add_argument("--rank", default=16, type=int, help="LoRA rank.")
    parser.add_argument("--alpha", default=32, type=int, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", default=0.05, type=float, help="LoRA dropout.")
    parser.add_argument("--batch-size", default=1, type=int, help="Per-device batch size.")
    parser.add_argument("--grad-accum", default=4, type=int, help="Gradient accumulation steps.")
    parser.add_argument("--epochs", default=2.0, type=float, help="Training epochs.")
    parser.add_argument("--max-steps", default=-1, type=int, help="Override epoch-based training when > 0.")
    parser.add_argument("--warmup-steps", default=100, type=int, help="Warmup steps.")
    parser.add_argument("--save-steps", default=100, type=int, help="Checkpoint save interval.")
    parser.add_argument("--eval-steps", default=100, type=int, help="Evaluation interval.")
    parser.add_argument("--logging-steps", default=10, type=int, help="Logging interval.")
    parser.add_argument("--lr-scheduler-type", default="cosine", help="Trainer LR scheduler type.")
    parser.add_argument("--optim", default="paged_adamw_8bit", help="Trainer optimizer.")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True, help="Use bf16 training.")
    parser.add_argument("--train-limit", default=None, type=int, help="Optional train split limit.")
    parser.add_argument("--eval-limit", default=None, type=int, help="Optional val split limit.")
    parser.add_argument("--dry-run", action="store_true", help="Validate configs and tiny batches without loading the model.")
    args = parser.parse_args(argv)
    if args.dry_run:
        args.train_limit = args.train_limit or 2
        args.eval_limit = args.eval_limit or 2
        if args.wandb_mode == "online":
            args.wandb_mode = "disabled"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.dry_run:
            return run_dry_run(args)
        return train(args)
    except (ImportError, RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
