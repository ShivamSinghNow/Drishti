from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoProcessor

from format_samples import PROMPT, assistant_response
from tbx11k_utils import DATA_DIR, ImageRecord, category_from_path, is_image, load_records


MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
IMAGE_SIZE = (512, 512)
OFFICIAL_ROOT_NAME = "TBX11K"


@dataclass(frozen=True)
class PreprocessedSample:
    source_path: Path
    split: str
    category: str
    messages: list[dict[str, Any]]
    image_mode: str
    image_size: tuple[int, int]
    processor_output_keys: list[str]
    processor_output_shapes: dict[str, tuple[int, ...]]


def load_rgb_image(path: Path, image_size: tuple[int, int] = IMAGE_SIZE) -> Image.Image:
    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        if rgb_image.size != image_size:
            rgb_image = rgb_image.resize(image_size, Image.Resampling.BICUBIC)
        return rgb_image


def training_messages(image_path: Path, category: str) -> list[dict[str, Any]]:
    try:
        response = assistant_response(category)
    except KeyError as exc:
        raise ValueError(f"Unsupported category for formatted response: {category}") from exc

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": str(image_path.resolve())},
                {"type": "text", "text": PROMPT},
            ],
        },
        {
            "role": "assistant",
            "content": response,
        },
    ]


def _processor_messages(image: Image.Image, category: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        },
        {
            "role": "assistant",
            "content": assistant_response(category),
        },
    ]


def _processor_output_shapes(processor_outputs: Any) -> dict[str, tuple[int, ...]]:
    shapes = {}
    for key, value in processor_outputs.items():
        shape = getattr(value, "shape", None)
        if shape is not None:
            shapes[key] = tuple(int(dimension) for dimension in shape)
    return shapes


def preprocess_record(
    record: ImageRecord,
    processor,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> PreprocessedSample:
    image = load_rgb_image(record.path, image_size)
    processor_outputs = processor.apply_chat_template(
        _processor_messages(image, record.category),
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    return PreprocessedSample(
        source_path=record.path.resolve(),
        split=record.split,
        category=record.category,
        messages=training_messages(record.path, record.category),
        image_mode=image.mode,
        image_size=image.size,
        processor_output_keys=list(processor_outputs.keys()),
        processor_output_shapes=_processor_output_shapes(processor_outputs),
    )


def validation_subset(records: list[ImageRecord], limit: int) -> list[ImageRecord]:
    by_category: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in sorted(records, key=lambda item: (item.category, str(item.path))):
        by_category[record.category].append(record)

    selected = []
    categories = sorted(by_category)
    while len(selected) < limit and any(by_category.values()):
        for category in categories:
            if by_category[category]:
                selected.append(by_category[category].pop(0))
                if len(selected) == limit:
                    break
    return selected


def validate_preprocessing(
    records: list[ImageRecord],
    processor,
    limit: int = 10,
) -> list[PreprocessedSample]:
    return [
        preprocess_record(record, processor)
        for record in validation_subset(records, limit)
    ]


def official_split_records(split: str, root: Path = DATA_DIR) -> list[ImageRecord]:
    official_root = root / OFFICIAL_ROOT_NAME
    split_file = official_root / "lists" / f"TBX11K_{split}.txt"
    image_root = official_root / "imgs"
    if not split_file.exists():
        return []

    records = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry:
            continue
        image_path = image_root / entry
        if not is_image(image_path):
            continue
        category = category_from_path(image_path, root)
        if category is None:
            continue
        records.append(ImageRecord(image_path.resolve(), split, category))
    return records


def split_records(split: str, root: Path = DATA_DIR) -> list[ImageRecord]:
    records = official_split_records(split, root)
    if records:
        return records

    all_records, _, _ = load_records(root)
    return [record for record in all_records if record.split == split]


def _print_summary(samples: list[PreprocessedSample], model_name: str) -> None:
    print(f"Validated {len(samples)} samples through {model_name}.")
    for index, sample in enumerate(samples, start=1):
        shapes = {
            key: list(shape)
            for key, shape in sample.processor_output_shapes.items()
        }
        print(
            f"{index}. {sample.source_path} | split={sample.split} | "
            f"label={sample.category} | mode={sample.image_mode} | "
            f"size={sample.image_size[0]}x{sample.image_size[1]} | "
            f"keys={', '.join(sample.processor_output_keys)}"
        )
        print(f"   shapes={json.dumps(shapes, sort_keys=True)}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the TBX11K Qwen-VL preprocessing pipeline.")
    parser.add_argument("--split", default="train", choices=("train", "val"), help="Dataset split to validate.")
    parser.add_argument("--limit", default=10, type=int, help="Number of samples to preprocess.")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Hugging Face processor name.")
    parser.add_argument("--data-dir", default=DATA_DIR, type=Path, help="TBX11K dataset root.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    records = split_records(args.split, args.data_dir)
    if not records:
        print(f"ERROR: No {args.split} records found under {args.data_dir}. Run download_dataset.py first.")
        return 1

    print(f"Loading processor: {args.model_name}")
    try:
        processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    except ImportError as exc:
        print("ERROR: Qwen-VL processor dependencies are missing.")
        print("Install torch, torchvision, and torchaudio together for the target host, then retry.")
        print(f"Details: {exc}")
        return 1

    samples = validate_preprocessing(records, processor, args.limit)
    if len(samples) != min(args.limit, len(records)):
        print(f"ERROR: Expected {min(args.limit, len(records))} samples, got {len(samples)}.")
        return 1

    _print_summary(samples, args.model_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
