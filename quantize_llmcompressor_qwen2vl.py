from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from quantization_utils import (
    FULL_PRECISION_BASELINE,
    directory_size_bytes,
    format_size_gb,
    qwen_vl_messages_for_sample,
    select_calibration_samples,
    write_json,
)


DEFAULT_MERGED_MODEL_DIR = Path("outputs/dri23-run4-merged-fp16")
DEFAULT_OUTPUT_DIR = Path("outputs/dri23-run4-llmcompressor-gptq-int4")
DEFAULT_IGNORE = ("lm_head", "re:visual.*", "re:model.visual.*")


def load_calibration_samples(data_dir: Path, split: str, total: int) -> list[Any]:
    from evaluate_checkpoint import load_eval_samples

    samples = load_eval_samples(data_dir, split)
    return select_calibration_samples(samples, total=total)


def build_calibration_dataset(
    samples: list[Any],
    processor: Any,
    max_sequence_length: int,
) -> Any:
    from datasets import Dataset
    from qwen_vl_utils import process_vision_info

    rows = [
        {
            "index": sample.index,
            "true_label": sample.true_label,
            "messages": qwen_vl_messages_for_sample(sample, include_answer=True),
        }
        for sample in samples
    ]
    dataset = Dataset.from_list(rows)

    def preprocess_and_tokenize(example: dict[str, Any]) -> dict[str, Any]:
        messages = example["messages"]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        return processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            max_length=max_sequence_length,
            truncation=True,
        )

    return dataset.map(preprocess_and_tokenize, remove_columns=dataset.column_names)


def oneshot_collator(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    if len(batch) != 1:
        raise ValueError("LLM Compressor Qwen2-VL GPTQ calibration expects batch size 1.")
    return {
        key: value if torch.is_tensor(value) else torch.tensor(value)
        for key, value in batch[0].items()
    }


def load_model_and_processor(model_dir: Path) -> tuple[Any, Any]:
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_dir,
            dtype="auto",
            trust_remote_code=True,
        )
    except TypeError:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype="auto",
            trust_remote_code=True,
        )
    if not hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, processor


def quantize_model(args: argparse.Namespace) -> dict[str, Any]:
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier

    started_at = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model_and_processor(args.merged_model_dir)
    calibration_samples = load_calibration_samples(
        args.data_dir,
        args.split,
        args.calibration_samples,
    )
    calibration_dataset = build_calibration_dataset(
        calibration_samples,
        processor,
        args.max_sequence_length,
    )

    recipe = [
        GPTQModifier(
            targets=args.targets,
            scheme=args.scheme,
            ignore=list(args.ignore),
            block_size=args.block_size,
            dampening_frac=args.dampening_frac,
            offload_hessians=args.offload_hessians,
        )
    ]

    oneshot(
        model=model,
        tokenizer=str(args.merged_model_dir),
        dataset=calibration_dataset,
        recipe=recipe,
        max_seq_length=args.max_sequence_length,
        num_calibration_samples=len(calibration_samples),
        trust_remote_code_model=True,
        data_collator=oneshot_collator,
        sequential_targets=[args.sequential_target],
    )

    model.save_pretrained(
        args.output_dir,
        save_compressed=True,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    processor.save_pretrained(args.output_dir)

    size_bytes = directory_size_bytes(args.output_dir)
    size_gb = format_size_gb(size_bytes)
    metadata = {
        "quantization_backend": "llmcompressor",
        "quantization_method": "gptq",
        "scheme": args.scheme,
        "targets": args.targets,
        "ignore": list(args.ignore),
        "sequential_target": args.sequential_target,
        "offload_hessians": args.offload_hessians,
        "block_size": args.block_size,
        "dampening_frac": args.dampening_frac,
        "merged_model_dir": str(args.merged_model_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "split": args.split,
        "calibration_samples": len(calibration_samples),
        "max_sequence_length": args.max_sequence_length,
        "max_shard_size": args.max_shard_size,
        "size_bytes": size_bytes,
        "size_gb": size_gb,
        "size_limit_gb": args.size_limit_gb,
        "full_precision_baseline": FULL_PRECISION_BASELINE,
        "runtime_seconds": time.time() - started_at,
    }
    metadata_path = write_json(args.output_dir / "quantization_metadata.json", metadata)
    metadata["metadata_path"] = str(metadata_path)

    if args.size_limit_gb is not None and size_gb > args.size_limit_gb:
        raise ValueError(
            f"Quantized model is {size_gb:.3f} GB, above limit {args.size_limit_gb:.3f} GB. "
            "The LLM Compressor multimodal recipe leaves the vision tower unquantized."
        )
    return metadata


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize the merged Drishti Qwen2-VL run #4 checkpoint with LLM Compressor GPTQ."
    )
    parser.add_argument("--merged-model-dir", default=DEFAULT_MERGED_MODEL_DIR, type=Path)
    parser.add_argument("--data-dir", default=Path("data/processed"), type=Path)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--calibration-samples", default=128, type=int)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--scheme", default="W4A16", help="LLM Compressor preset scheme. W4A16 uses INT4 weights, group size 128.")
    parser.add_argument("--targets", default="Linear", help="Module class/name target passed to GPTQModifier.")
    parser.add_argument(
        "--ignore",
        default=DEFAULT_IGNORE,
        nargs="+",
        help="Modules to leave unquantized. Defaults match the official Qwen2-VL multimodal recipe.",
    )
    parser.add_argument("--sequential-target", default="Qwen2VLDecoderLayer")
    parser.add_argument("--max-sequence-length", default=2048, type=int)
    parser.add_argument("--block-size", default=128, type=int)
    parser.add_argument("--dampening-frac", default=0.01, type=float)
    parser.add_argument("--no-offload-hessians", dest="offload_hessians", action="store_false")
    parser.set_defaults(offload_hessians=True)
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--size-limit-gb", default=None, type=float)
    parser.add_argument("--debug-traceback", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        metadata = quantize_model(args)
    except Exception as exc:  # noqa: BLE001 - CLI should surface concise Colab errors.
        print(f"ERROR: {exc}")
        if args.debug_traceback:
            traceback.print_exc()
        return 1

    print(f"Quantized model saved to {metadata['output_dir']}")
    print(f"size_gb={metadata['size_gb']:.3f}")
    print(f"metadata={metadata['metadata_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
