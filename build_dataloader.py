from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from preprocess_samples import MODEL_NAME


DEFAULT_DATA_DIR = Path("data/processed")
IGNORE_INDEX = -100
REQUIRED_BATCH_KEYS = {
    "input_ids",
    "attention_mask",
    "mm_token_type_ids",
    "pixel_values",
    "image_grid_thw",
    "labels",
}


@dataclass(frozen=True)
class BatchValidationSummary:
    split: str
    batch_size: int
    tensor_shapes: dict[str, tuple[int, ...]]
    label_token_counts: tuple[int, ...]
    masked_token_counts: tuple[int, ...]


def load_jsonl_dataset(
    split: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    limit: int | None = None,
) -> Dataset:
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run generate_jsonl.py --output-dir {data_dir} first."
        )

    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line)["messages"])
                if limit is not None and len(rows) >= limit:
                    break
    return Dataset.from_dict({"messages_json": [json.dumps(messages) for messages in rows]})


def mask_prompt_labels(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: list[int] | torch.Tensor,
    padding_side: str = "left",
) -> torch.Tensor:
    labels = input_ids.clone()
    labels[attention_mask == 0] = IGNORE_INDEX

    if isinstance(prompt_lengths, torch.Tensor):
        prompt_lengths = [int(length) for length in prompt_lengths.tolist()]

    sequence_length = input_ids.shape[1]
    actual_lengths = attention_mask.sum(dim=1).tolist()
    for row, prompt_length in enumerate(prompt_lengths):
        actual_length = int(actual_lengths[row])
        if prompt_length > actual_length:
            raise ValueError(f"Prompt length {prompt_length} exceeds sequence length {actual_length}.")
        prompt_start = sequence_length - actual_length if padding_side == "left" else 0
        labels[row, prompt_start : prompt_start + prompt_length] = IGNORE_INDEX
    return labels


class QwenVlDataCollator:
    def __init__(self, processor) -> None:
        self.processor = processor
        self.padding_side = getattr(getattr(processor, "tokenizer", None), "padding_side", "left")

    @staticmethod
    def feature_messages(feature: dict[str, Any]) -> list[dict[str, Any]]:
        if "messages" in feature:
            return feature["messages"]
        return json.loads(feature["messages_json"])

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        conversations = [self.feature_messages(feature) for feature in features]
        prompt_conversations = [conversation[:1] for conversation in conversations]

        batch = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        prompt_batch = self.processor.apply_chat_template(
            prompt_conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1)
        batch["labels"] = mask_prompt_labels(
            batch["input_ids"],
            batch["attention_mask"],
            prompt_lengths,
            self.padding_side,
        )
        return batch


def build_dataloader(
    split: str,
    batch_size: int = 2,
    data_dir: Path = DEFAULT_DATA_DIR,
    limit: int | None = None,
    processor=None,
    model_name: str = MODEL_NAME,
) -> DataLoader:
    dataset = load_jsonl_dataset(split, data_dir, limit)
    if processor is None:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=QwenVlDataCollator(processor),
    )


def tensor_shapes(batch: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    return {
        key: tuple(int(dimension) for dimension in value.shape)
        for key, value in batch.items()
        if hasattr(value, "shape")
    }


def validate_batch(split: str, batch: dict[str, torch.Tensor]) -> BatchValidationSummary:
    missing = REQUIRED_BATCH_KEYS.difference(batch)
    if missing:
        raise ValueError(f"Batch is missing required keys: {sorted(missing)}")
    if batch["labels"].shape != batch["input_ids"].shape:
        raise ValueError("labels must have the same shape as input_ids")

    label_token_counts = (batch["labels"] != IGNORE_INDEX).sum(dim=1)
    masked_token_counts = (batch["labels"] == IGNORE_INDEX).sum(dim=1)
    if torch.any(label_token_counts <= 0):
        raise ValueError("Every sample must have at least one assistant label token")
    if torch.any(masked_token_counts <= 0):
        raise ValueError("Every sample must mask prompt or padding tokens")

    return BatchValidationSummary(
        split=split,
        batch_size=int(batch["input_ids"].shape[0]),
        tensor_shapes=tensor_shapes(batch),
        label_token_counts=tuple(int(value) for value in label_token_counts.tolist()),
        masked_token_counts=tuple(int(value) for value in masked_token_counts.tolist()),
    )


def print_summary(summary: BatchValidationSummary) -> None:
    print(f"{summary.split}: validated batch_size={summary.batch_size}")
    for key, shape in sorted(summary.tensor_shapes.items()):
        print(f"{summary.split}: {key} shape={shape}")
    print(f"{summary.split}: assistant label tokens={summary.label_token_counts}")
    print(f"{summary.split}: masked prompt/pad tokens={summary.masked_token_counts}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Qwen-VL train/val DataLoader batches.")
    parser.add_argument("--split", default="train", choices=("train", "val"), help="JSONL split to load.")
    parser.add_argument("--batch-size", default=2, type=int, help="Batch size to validate.")
    parser.add_argument("--limit", default=2, type=int, help="Limit records loaded from the split.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, type=Path, help="Directory containing train.jsonl and val.jsonl.")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Hugging Face processor name.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        dataloader = build_dataloader(
            split=args.split,
            batch_size=args.batch_size,
            data_dir=args.data_dir,
            limit=args.limit,
            model_name=args.model_name,
        )
        batch = next(iter(dataloader))
        summary = validate_batch(args.split, batch)
    except (FileNotFoundError, ImportError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
