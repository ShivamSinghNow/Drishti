from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from preprocess_samples import split_records, training_messages
from tbx11k_utils import DATA_DIR, ImageRecord


DEFAULT_OUTPUT_DIR = Path("data/processed")
SPLITS = ("train", "val")
EXPECTED_COUNTS = {
    "train": Counter({"active_tb": 600, "healthy": 3000, "sick_but_non_tb": 3000}),
    "val": Counter({"active_tb": 200, "healthy": 800, "sick_but_non_tb": 800}),
}


@dataclass(frozen=True)
class JsonlSummary:
    path: Path
    total_records: int
    category_counts: Counter
    missing_paths: tuple[Path, ...]
    non_absolute_paths: tuple[Path, ...]

    @property
    def is_valid(self) -> bool:
        return not self.missing_paths and not self.non_absolute_paths


def jsonl_record(record: ImageRecord) -> dict[str, Any]:
    return {"messages": training_messages(record.path, record.category)}


def _assistant_category(payload: dict[str, Any]) -> str | None:
    for message in payload.get("messages", []):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        for line in content.splitlines():
            if line.startswith("Classification: "):
                return line.removeprefix("Classification: ").strip()
    return None


def _image_path(payload: dict[str, Any]) -> Path | None:
    for message in payload.get("messages", []):
        if message.get("role") != "user":
            continue
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") == "image" and item.get("path"):
                return Path(item["path"])
    return None


def write_jsonl(records: list[ImageRecord], output_path: Path) -> JsonlSummary:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(jsonl_record(record), ensure_ascii=False))
            handle.write("\n")
    return validate_jsonl(output_path)


def validate_jsonl(output_path: Path) -> JsonlSummary:
    category_counts = Counter()
    missing_paths = []
    non_absolute_paths = []
    total_records = 0

    with output_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            total_records += 1

            image_path = _image_path(payload)
            if image_path is None:
                raise ValueError(f"{output_path}:{line_number} is missing a user image path")
            if not image_path.is_absolute():
                non_absolute_paths.append(image_path)
            if not image_path.exists():
                missing_paths.append(image_path)

            category = _assistant_category(payload)
            if category is None:
                raise ValueError(f"{output_path}:{line_number} is missing assistant classification")
            category_counts[category] += 1

    return JsonlSummary(
        path=output_path,
        total_records=total_records,
        category_counts=category_counts,
        missing_paths=tuple(missing_paths),
        non_absolute_paths=tuple(non_absolute_paths),
    )


def generate_splits(
    output_dir: Path,
    data_dir: Path = DATA_DIR,
) -> dict[str, JsonlSummary]:
    summaries = {}
    for split in SPLITS:
        records = split_records(split, data_dir)
        summaries[split] = write_jsonl(records, output_dir / f"{split}.jsonl")
    return summaries


def deterministic_spot_checks(records: list[ImageRecord], count: int = 5, seed: int = 42) -> list[ImageRecord]:
    if len(records) <= count:
        return records
    return random.Random(seed).sample(records, count)


def _print_split_summary(split: str, summary: JsonlSummary, records: list[ImageRecord]) -> None:
    print(f"{split}: wrote {summary.total_records} records to {summary.path}")
    print(f"{split}: class distribution {dict(sorted(summary.category_counts.items()))}")
    print(f"{split}: missing paths={len(summary.missing_paths)} non-absolute paths={len(summary.non_absolute_paths)}")
    print(f"{split}: deterministic spot checks")
    for record in deterministic_spot_checks(records):
        response_preview = training_messages(record.path, record.category)[1]["content"].splitlines()[0]
        print(f"  - {record.category}: {record.path.resolve()} | {response_preview}")


def _validate_summary(split: str, summary: JsonlSummary) -> list[str]:
    errors = []
    expected = EXPECTED_COUNTS[split]
    if summary.category_counts != expected:
        errors.append(f"{split}: expected class distribution {dict(expected)}, got {dict(summary.category_counts)}")
    expected_total = sum(expected.values())
    if summary.total_records != expected_total:
        errors.append(f"{split}: expected {expected_total} records, got {summary.total_records}")
    if summary.missing_paths:
        errors.append(f"{split}: found {len(summary.missing_paths)} missing image paths")
    if summary.non_absolute_paths:
        errors.append(f"{split}: found {len(summary.non_absolute_paths)} non-absolute image paths")
    return errors


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TBX11K Qwen-VL train/val JSONL files.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path, help="Directory for train.jsonl and val.jsonl.")
    parser.add_argument("--data-dir", default=DATA_DIR, type=Path, help="TBX11K dataset root.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    summaries = generate_splits(args.output_dir, args.data_dir)

    errors = []
    for split, summary in summaries.items():
        records = split_records(split, args.data_dir)
        _print_split_summary(split, summary, records)
        errors.extend(_validate_summary(split, summary))

    if errors:
        print("\nValidation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nJSONL generation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
