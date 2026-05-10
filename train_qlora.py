from __future__ import annotations

import argparse
import importlib.metadata
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, WeightedRandomSampler
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
from format_samples import CLASS_LABELS
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


@dataclass(frozen=True)
class TrainerComponents:
    trainer: Trainer
    components: TrainingComponents
    model: Any


def dataset_labels(dataset: Any) -> list[str]:
    if "label" in getattr(dataset, "column_names", []):
        return [str(label) for label in dataset["label"]]

    labels = []
    for index in range(len(dataset)):
        feature = dataset[index]
        if "label" not in feature:
            raise ValueError("Balanced sampling requires a label column on the train dataset.")
        labels.append(str(feature["label"]))
    return labels


def build_class_balanced_weights(
    labels: list[str],
    boost_class: str | None = None,
    boost_multiplier: float = 1.0,
    weight_exponent: float = 1.0,
) -> torch.Tensor:
    if not labels:
        raise ValueError("Balanced sampling requires at least one training sample.")
    if boost_multiplier <= 0:
        raise ValueError("--boost-multiplier must be greater than 0.")
    if weight_exponent <= 0:
        raise ValueError("--sampler-weight-exponent must be greater than 0.")

    counts = Counter(labels)
    weights = []
    for label in labels:
        weight = (1.0 / counts[label]) ** weight_exponent
        if boost_class is not None and label == boost_class:
            weight *= boost_multiplier
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.double)


class ClassBalancedTrainer(Trainer):
    def __init__(
        self,
        *args,
        boost_class: str | None = None,
        boost_multiplier: float = 1.0,
        weight_exponent: float = 1.0,
        sampler_seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.boost_class = boost_class
        self.boost_multiplier = boost_multiplier
        self.weight_exponent = weight_exponent
        self.sampler_seed = sampler_seed

    def _get_train_sampler(self, train_dataset: Any | None = None):
        labels = dataset_labels(train_dataset or self.train_dataset)
        weights = build_class_balanced_weights(
            labels,
            self.boost_class,
            self.boost_multiplier,
            self.weight_exponent,
        )
        generator = torch.Generator()
        generator.manual_seed(self.sampler_seed)
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )


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
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        fp16=False,
        seed=args.seed,
        data_seed=args.seed,
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


def package_version(distribution_name: str) -> str:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def accelerator_backend() -> str:
    if not torch.cuda.is_available():
        return "none"
    if getattr(torch.version, "hip", None):
        return "rocm"
    if getattr(torch.version, "cuda", None):
        return "cuda"
    return "cuda"


def required_training_packages(backend: str) -> tuple[str, ...]:
    if backend == "rocm":
        return ("bitsandbytes", "optimum", "optimum-amd")
    if backend == "cuda":
        return ("bitsandbytes",)
    return ()


def require_full_training_environment() -> None:
    backend = accelerator_backend()
    if backend == "none":
        raise RuntimeError("CUDA/ROCm is not available. Full QLoRA training must run on a GPU host.")

    missing = [
        name
        for name in required_training_packages(backend)
        if not package_available(name)
    ]
    if missing:
        raise RuntimeError(f"Missing required training package(s): {', '.join(missing)}")


def sampler_weight_exponent(args: argparse.Namespace) -> float:
    if args.sampler_weight_exponent is not None:
        return args.sampler_weight_exponent
    if args.sampling_strategy == "soft-balanced":
        return 0.5
    return 1.0


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


def trainable_parameter_counts(model: Any) -> tuple[int, int]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return trainable, total


def load_lora_model(args: argparse.Namespace, quantization_config: BitsAndBytesConfig, lora_config: LoraConfig):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    return get_peft_model(model, lora_config)


def build_trainer_components(args: argparse.Namespace) -> TrainerComponents:
    quantization_config = build_quantization_config()
    lora_config = build_lora_config(args.rank, args.alpha, args.lora_dropout)
    training_args = build_training_arguments(args)
    components = load_training_components(args)
    model = load_lora_model(args, quantization_config, lora_config)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": components.train_dataset,
        "eval_dataset": components.eval_dataset,
        "data_collator": components.data_collator,
        "processing_class": components.processor,
    }
    if args.sampling_strategy == "balanced":
        weight_exponent = sampler_weight_exponent(args)
        trainer = ClassBalancedTrainer(
            **trainer_kwargs,
            boost_class=args.boost_class,
            boost_multiplier=args.boost_multiplier,
            weight_exponent=weight_exponent,
            sampler_seed=args.seed,
        )
    elif args.sampling_strategy == "soft-balanced":
        weight_exponent = sampler_weight_exponent(args)
        trainer = ClassBalancedTrainer(
            **trainer_kwargs,
            boost_class=args.boost_class,
            boost_multiplier=args.boost_multiplier,
            weight_exponent=weight_exponent,
            sampler_seed=args.seed,
        )
    else:
        trainer = Trainer(
            **trainer_kwargs,
        )
    return TrainerComponents(trainer=trainer, components=components, model=model)


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
    print(f"sampling_strategy={args.sampling_strategy} boost_class={args.boost_class} boost_multiplier={args.boost_multiplier}")
    print(f"sampler_weight_exponent={sampler_weight_exponent(args)}")
    print(f"trainer_output_dir={training_args.output_dir}")
    for key, shape in sorted(shapes.items()):
        print(f"batch_{key}_shape={shape}")
    return 0


def run_setup_only(args: argparse.Namespace) -> int:
    require_full_training_environment()
    configure_wandb(args)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    trainer_components = build_trainer_components(args)
    shapes = dry_run_batch_summary(args, trainer_components.components)
    trainable, total = trainable_parameter_counts(trainer_components.model)

    print("Setup-only validation complete. Trainer was built; training was not started.")
    print(f"model_name={args.model_name}")
    print(f"torch={torch.__version__}")
    print(f"accelerator_backend={accelerator_backend()}")
    print(f"torch_cuda={getattr(torch.version, 'cuda', None)}")
    print(f"torch_hip={getattr(torch.version, 'hip', None)}")
    print(f"transformers={package_version('transformers')}")
    print(f"peft={package_version('peft')}")
    print(f"bitsandbytes={package_version('bitsandbytes')}")
    print(f"optimum={package_version('optimum')}")
    print(f"optimum_amd={package_version('optimum-amd')}")
    print(f"train_records={len(trainer_components.components.train_dataset)}")
    print(f"eval_records={len(trainer_components.components.eval_dataset)}")
    print(f"trainer_output_dir={trainer_components.trainer.args.output_dir}")
    print(f"trainer_remove_unused_columns={trainer_components.trainer.args.remove_unused_columns}")
    print(f"sampling_strategy={args.sampling_strategy}")
    print(f"boost_class={args.boost_class}")
    print(f"boost_multiplier={args.boost_multiplier}")
    print(f"sampler_weight_exponent={sampler_weight_exponent(args)}")
    print(f"trainable_parameters={trainable}")
    print(f"total_parameters={total}")
    if total:
        print(f"trainable_parameter_percent={(trainable / total) * 100:.4f}")
    for key, shape in sorted(shapes.items()):
        print(f"batch_{key}_shape={shape}")
    if torch.cuda.is_available():
        print(f"peak_allocated_gb={torch.cuda.max_memory_allocated() / 1e9}")
        print(f"peak_reserved_gb={torch.cuda.max_memory_reserved() / 1e9}")
    return 0


def train(args: argparse.Namespace) -> int:
    require_full_training_environment()
    configure_wandb(args)

    trainer_components = build_trainer_components(args)
    resume_from_checkpoint = str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    trainer_components.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer_components.trainer.save_model()
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
    parser.add_argument("--weight-decay", default=0.0, type=float, help="Trainer weight decay.")
    parser.add_argument("--seed", default=42, type=int, help="Trainer and sampler seed.")
    parser.add_argument("--sampling-strategy", default="natural", choices=("natural", "balanced", "soft-balanced"))
    parser.add_argument("--boost-class", default=None, choices=CLASS_LABELS, help="Optional class to upweight under balanced sampling.")
    parser.add_argument("--boost-multiplier", default=1.0, type=float, help="Multiplier for --boost-class under balanced sampling.")
    parser.add_argument(
        "--sampler-weight-exponent",
        default=None,
        type=float,
        help="Exponent applied to inverse class counts. Defaults to 1.0 for balanced and 0.5 for soft-balanced.",
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True, help="Use bf16 training.")
    parser.add_argument("--train-limit", default=None, type=int, help="Optional train split limit.")
    parser.add_argument("--eval-limit", default=None, type=int, help="Optional val split limit.")
    parser.add_argument("--resume-from-checkpoint", default=None, type=Path, help="Optional Trainer checkpoint to resume from.")
    parser.add_argument("--dry-run", action="store_true", help="Validate configs and tiny batches without loading the model.")
    parser.add_argument("--setup-only", action="store_true", help="Load the real 4-bit LoRA model and build Trainer without training.")
    args = parser.parse_args(argv)
    if args.dry_run and args.setup_only:
        parser.error("--dry-run and --setup-only are mutually exclusive.")
    if args.boost_multiplier <= 0:
        parser.error("--boost-multiplier must be greater than 0.")
    if args.sampler_weight_exponent is not None and args.sampler_weight_exponent <= 0:
        parser.error("--sampler-weight-exponent must be greater than 0.")
    if args.dry_run or args.setup_only:
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
        if args.setup_only:
            return run_setup_only(args)
        return train(args)
    except (ImportError, RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
