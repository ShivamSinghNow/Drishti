from __future__ import annotations

import sys
from collections import Counter

import numpy as np
from PIL import Image
from tqdm import tqdm

from tbx11k_utils import DATA_DIR, TARGET_CATEGORIES, SPLITS, class_distribution, load_records, random_records


def main() -> int:
    records, used_files, source = load_records(DATA_DIR)
    if not records:
        print(f"ERROR: No TBX11K records found under {DATA_DIR}. Run download_dataset.py first.")
        return 1

    print(f"Loaded {len(records)} records from {source}.")
    if used_files:
        print("Annotation files loaded:")
        for path in used_files:
            print(f"  {path}")
    elif source == "folders":
        print("No parseable annotation files were found; using folder names as labels/splits.")

    split_counts = Counter(record.split for record in records)
    print("\nTotal image count per split:")
    for split in SPLITS:
        print(f"  {split}: {split_counts.get(split, 0)}")
    extra_splits = sorted(split for split in split_counts if split not in SPLITS)
    for split in extra_splits:
        print(f"  {split}: {split_counts[split]}")

    print("\nClass distribution across each split:")
    distribution = class_distribution(records)
    for split in [*SPLITS, *extra_splits]:
        print(f"  {split}:")
        for category in TARGET_CATEGORIES:
            print(f"    {category}: {distribution.get(split, Counter()).get(category, 0)}")

    train_samples = random_records(records, "train", 3)
    if not train_samples:
        print("\nERROR: No training samples found.")
        return 1

    print("\nThree random training images:")
    for record in train_samples:
        with Image.open(record.path) as image:
            array = np.asarray(image)
            print(
                f"  {record.path} | category={record.category} | "
                f"shape={array.shape} | pixel_range=({array.min()}, {array.max()})"
            )

    print("\nChecking all image dimensions...")
    invalid_sizes = []
    for record in tqdm(records, desc="Verifying 512x512"):
        try:
            with Image.open(record.path) as image:
                if image.size != (512, 512):
                    invalid_sizes.append((record.path, image.size))
        except Exception as exc:
            invalid_sizes.append((record.path, f"unreadable: {exc}"))

    if invalid_sizes:
        print("ERROR: Not all images are 512x512.")
        for path, size in invalid_sizes[:20]:
            print(f"  {path}: {size}")
        if len(invalid_sizes) > 20:
            print(f"  ... {len(invalid_sizes) - 20} more")
        return 1

    print("Confirmed: all images are 512x512.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
