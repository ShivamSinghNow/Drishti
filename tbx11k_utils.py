from __future__ import annotations

import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DATA_DIR = Path("data/tbx11k")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TARGET_CATEGORIES = ("healthy", "active_tb", "sick_but_non_tb")
SPLITS = ("train", "val", "test")

CATEGORY_ALIASES = {
    "healthy": {"healthy", "health", "normal", "no_tb", "no_tuberculosis"},
    "active_tb": {"active_tb", "activetb", "active", "tb", "tuberculosis"},
    "latent_tb": {"latent_tb", "latenttb", "latent"},
    "sick_but_non_tb": {
        "sick_but_non_tb",
        "sickbutnontb",
        "sick_non_tb",
        "sicknontb",
        "sick_but_non-tb",
        "non_tb",
        "nontb",
        "sick",
    },
}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    split: str
    category: str


def normalize_token(value: str | Path | None) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def canonical_category(value: str | Path | None) -> str | None:
    token = normalize_token(value)
    if not token:
        return None
    for category, aliases in CATEGORY_ALIASES.items():
        candidates = _category_candidates(category, aliases)
        if token in candidates:
            return category

    best_category = None
    best_specificity = 0
    for category, aliases in CATEGORY_ALIASES.items():
        for candidate in _category_candidates(category, aliases):
            specificity = _token_sequence_specificity(token, candidate)
            if specificity > best_specificity:
                best_category = category
                best_specificity = specificity
    if best_category:
        return best_category
    return None


def _category_candidates(category: str, aliases: set[str]) -> set[str]:
    return {
        normalized
        for value in {category, *aliases}
        if (normalized := normalize_token(value))
    }


def _token_sequence_specificity(token: str, candidate: str) -> int:
    token_parts = token.split("_")
    candidate_parts = candidate.split("_")
    if not token_parts or not candidate_parts or len(candidate_parts) > len(token_parts):
        return 0
    for index in range(len(token_parts) - len(candidate_parts) + 1):
        if token_parts[index : index + len(candidate_parts)] == candidate_parts:
            return len(candidate_parts)
    return 0


def canonical_split(value: str | Path | None) -> str | None:
    token = normalize_token(value)
    if token in {"validation", "valid", "val"}:
        return "val"
    if token in SPLITS:
        return token
    if "validation" in token or re.search(r"(^|_)val(id)?(_|$)", token):
        return "val"
    for split in SPLITS:
        if re.search(rf"(^|_){split}(_|$)", token):
            return split
    return None


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file()


def iter_images(root: Path = DATA_DIR) -> Iterable[Path]:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if is_image(path):
            yield path


def resolve_image_path(root: Path, raw_path: str) -> Path | None:
    candidate = Path(raw_path)
    checks = []
    if candidate.is_absolute():
        checks.append(candidate)
    else:
        checks.extend([root / candidate, root / candidate.name])
    for check in checks:
        if is_image(check):
            return check

    name_matches = list(root.rglob(candidate.name))
    for match in name_matches:
        if is_image(match):
            return match
    return None


def _annotation_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    suffixes = {".csv", ".tsv", ".json", ".jsonl", ".txt"}
    ignored_names = {"readme", "license", "requirements"}
    files = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            if path.stem.lower() not in ignored_names:
                files.append(path)
    return sorted(files)


def _column_lookup(fieldnames: Iterable[str] | None, candidates: set[str]) -> str | None:
    if not fieldnames:
        return None
    normalized = {normalize_token(name): name for name in fieldnames}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for norm_name, original_name in normalized.items():
        if any(candidate in norm_name for candidate in candidates):
            return original_name
    return None


def _records_from_rows(root: Path, source: Path, rows: list[dict]) -> list[ImageRecord]:
    if not rows:
        return []

    fieldnames = rows[0].keys()
    image_col = _column_lookup(
        fieldnames,
        {
            "image",
            "image_path",
            "img",
            "img_path",
            "file",
            "file_name",
            "filename",
            "path",
            "name",
        },
    )
    label_col = _column_lookup(fieldnames, {"label", "labels", "class", "category", "finding", "target"})
    split_col = _column_lookup(fieldnames, {"split", "set", "subset", "phase"})
    source_split = canonical_split(source.stem)

    records = []
    for row in rows:
        if not image_col:
            continue
        image_path = resolve_image_path(root, str(row.get(image_col, "")))
        if image_path is None:
            continue

        category = canonical_category(row.get(label_col, "")) if label_col else None
        split = canonical_split(row.get(split_col, "")) if split_col else None
        if category is None:
            category = category_from_path(image_path, root)
        if split is None:
            split = source_split or split_from_path(image_path, root)
        if category and split:
            records.append(ImageRecord(image_path, split, category))
    return records


def _records_from_delimited_file(root: Path, source: Path) -> list[ImageRecord]:
    delimiter = "\t" if source.suffix.lower() == ".tsv" else ","
    try:
        with source.open(newline="", encoding="utf-8-sig") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            if source.suffix.lower() == ".txt":
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",\t ")
                    delimiter = dialect.delimiter
                except csv.Error:
                    delimiter = None
            if delimiter:
                reader = csv.DictReader(handle, delimiter=delimiter)
                rows = list(reader)
                if reader.fieldnames and len(reader.fieldnames) > 1:
                    parsed_records = _records_from_rows(root, source, rows)
                    if parsed_records:
                        return parsed_records
    except UnicodeDecodeError:
        return []

    records = []
    source_split = canonical_split(source.stem)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return records

    for line in lines:
        if not line.strip():
            continue
        parts = re.split(r"[\s,]+", line.strip())
        image_path = resolve_image_path(root, parts[0])
        if image_path is None:
            continue
        category = None
        split = source_split
        for part in parts[1:]:
            category = category or canonical_category(part)
            split = split or canonical_split(part)
        category = category or category_from_path(image_path, root)
        split = split or split_from_path(image_path, root)
        if category and split:
            records.append(ImageRecord(image_path, split, category))
    return records


def _flatten_json_rows(payload) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("annotations", "images", "data", "records", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _records_from_json_file(root: Path, source: Path) -> list[ImageRecord]:
    records = []
    try:
        if source.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            rows = _flatten_json_rows(json.loads(source.read_text(encoding="utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return records
    return _records_from_rows(root, source, rows)


def category_from_path(path: Path, root: Path = DATA_DIR) -> str | None:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    for part in reversed(parts[:-1]):
        category = canonical_category(part)
        if category:
            return category
    stem = normalize_token(path.stem)
    if re.fullmatch(r"h\d+", stem):
        return "healthy"
    if re.fullmatch(r"tb\d+", stem):
        return "active_tb"
    if re.fullmatch(r"s\d+", stem):
        return "sick_but_non_tb"
    return canonical_category(path.name)


def split_from_path(path: Path, root: Path = DATA_DIR) -> str | None:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    for part in reversed(parts[:-1]):
        split = canonical_split(part)
        if split:
            return split
    return canonical_split(path.name)


def records_from_annotations(root: Path = DATA_DIR) -> tuple[list[ImageRecord], list[Path]]:
    records = []
    used_files = []
    for source in _annotation_files(root):
        if source.suffix.lower() in {".json", ".jsonl"}:
            source_records = _records_from_json_file(root, source)
        else:
            source_records = _records_from_delimited_file(root, source)
        if source_records:
            records.extend(source_records)
            used_files.append(source)

    unique = {}
    for record in records:
        unique[(record.path.resolve(), record.split, record.category)] = record
    return list(unique.values()), used_files


def records_from_folders(root: Path = DATA_DIR) -> list[ImageRecord]:
    records = []
    for image_path in iter_images(root):
        category = category_from_path(image_path, root)
        split = split_from_path(image_path, root)
        if category is None:
            continue
        if split is None:
            split = "unsplit"
        records.append(ImageRecord(image_path, split, category))
    return records


def load_records(root: Path = DATA_DIR) -> tuple[list[ImageRecord], list[Path], str]:
    annotation_records, used_files = records_from_annotations(root)
    if annotation_records:
        return annotation_records, used_files, "annotations"
    return records_from_folders(root), [], "folders"


def split_counts(records: Iterable[ImageRecord]) -> Counter:
    return Counter(record.split for record in records)


def class_distribution(records: Iterable[ImageRecord]) -> dict[str, Counter]:
    distribution = defaultdict(Counter)
    for record in records:
        distribution[record.split][record.category] += 1
    return dict(distribution)


def sample_by_category(records: Iterable[ImageRecord], category: str, split: str | None = None) -> ImageRecord | None:
    matches = [
        record
        for record in records
        if record.category == category and (split is None or record.split == split)
    ]
    if not matches and split is not None:
        matches = [record for record in records if record.category == category]
    return sorted(matches, key=lambda record: str(record.path))[0] if matches else None


def random_records(records: Iterable[ImageRecord], split: str, count: int, seed: int = 42) -> list[ImageRecord]:
    matches = [record for record in records if record.split == split]
    rng = random.Random(seed)
    if len(matches) <= count:
        return matches
    return rng.sample(matches, count)


def count_direct_files(path: Path) -> int:
    try:
        return sum(1 for child in path.iterdir() if child.is_file())
    except OSError:
        return 0


def print_folder_structure(root: Path = DATA_DIR) -> None:
    if not root.exists():
        print(f"Dataset folder does not exist: {root}")
        return
    for dirpath, dirnames, _ in os.walk(root):
        dirnames.sort()
        current = Path(dirpath)
        depth = len(current.relative_to(root).parts)
        indent = "  " * depth
        print(f"{indent}{current.name}/ ({count_direct_files(current)} files)")
