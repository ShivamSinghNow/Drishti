from __future__ import annotations

import sys
from pathlib import Path

from tbx11k_utils import DATA_DIR, TARGET_CATEGORIES, iter_images, load_records, print_folder_structure, sample_by_category


DATASET_SLUG = "vbookshelf/tbx11k-simplified"


def download_dataset(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError("kaggle package is not installed. Install it with `pip install kaggle`.") from exc

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET_SLUG, path=str(destination), unzip=True, quiet=False)


def main() -> int:
    try:
        print(f"Downloading Kaggle dataset {DATASET_SLUG} into {DATA_DIR} ...")
        download_dataset(DATA_DIR)
    except SystemExit as exc:
        print("ERROR: Kaggle dataset download failed.")
        print("Make sure your Kaggle credentials are configured at ~/.kaggle/kaggle.json or via KAGGLE_USERNAME/KAGGLE_KEY.")
        print(f"Details: Kaggle API exited with status {exc.code}")
        return int(exc.code) if isinstance(exc.code, int) else 1
    except Exception as exc:
        print("ERROR: Kaggle dataset download failed.")
        print("Make sure your Kaggle credentials are configured at ~/.kaggle/kaggle.json or via KAGGLE_USERNAME/KAGGLE_KEY.")
        print(f"Details: {exc}")
        return 1

    print("\nFolder structure:")
    print_folder_structure(DATA_DIR)

    records, used_files, source = load_records(DATA_DIR)
    print(f"\nDiscovered {len(list(iter_images(DATA_DIR)))} image files.")
    print(f"Metadata source: {source}")
    if used_files:
        print("Annotation files used:")
        for path in used_files:
            print(f"  {path}")

    print("\nSample image path by category:")
    for category in TARGET_CATEGORIES:
        record = sample_by_category(records, category)
        if record:
            print(f"  {category}: {record.path}")
        else:
            print(f"  {category}: NOT FOUND")

    missing = [category for category in TARGET_CATEGORIES if sample_by_category(records, category) is None]
    if missing:
        print(f"\nERROR: Missing required categories: {', '.join(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
