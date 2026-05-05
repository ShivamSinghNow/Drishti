from __future__ import annotations

import json
import sys
from pathlib import Path

from tbx11k_utils import DATA_DIR, load_records, sample_by_category


OUTPUT_PATH = Path("formatted_samples.json")
OFFICIAL_ROOT = DATA_DIR / "TBX11K"
OFFICIAL_IMAGE_ROOT = OFFICIAL_ROOT / "imgs"
OFFICIAL_TRAIN_LIST = OFFICIAL_ROOT / "lists" / "TBX11K_train.txt"
OFFICIAL_TRAIN_ANNOTATIONS = OFFICIAL_ROOT / "annotations" / "json" / "TBX11K_train.json"
PROMPT = (
    "Analyze this chest X-ray image for tuberculosis screening. "
    "Return the classification, visual finding, confidence, and whether referral is recommended."
)

RESPONSES = {
    "healthy": {
        "finding": "No focal lung opacity is visible on this screening image.",
        "confidence": "High",
        "referral": "No",
    },
    "active_tb": {
        "finding": "Patchy upper-lung opacities are present, suspicious for active tuberculosis.",
        "confidence": "High",
        "referral": "Yes",
    },
    "sick_but_non_tb": {
        "finding": "Abnormal lung opacity is present but the pattern is not specific for tuberculosis.",
        "confidence": "Medium",
        "referral": "Yes",
    },
}


def assistant_response(category: str) -> str:
    response = RESPONSES[category]
    return (
        f"Classification: {category}\n"
        f"Finding: {response['finding']}\n"
        f"Confidence: {response['confidence']}\n"
        f"Referral recommended: {response['referral']}"
    )


def official_train_entries() -> list[str]:
    if not OFFICIAL_TRAIN_LIST.exists():
        return []
    return [line.strip() for line in OFFICIAL_TRAIN_LIST.read_text(encoding="utf-8").splitlines() if line.strip()]


def first_official_train_path(prefix: str) -> Path | None:
    for entry in official_train_entries():
        if entry.startswith(f"{prefix}/"):
            path = OFFICIAL_IMAGE_ROOT / entry
            if path.exists():
                return path.resolve()
    return None


def active_tb_official_train_path() -> Path | None:
    if not OFFICIAL_TRAIN_ANNOTATIONS.exists():
        return None

    payload = json.loads(OFFICIAL_TRAIN_ANNOTATIONS.read_text(encoding="utf-8"))
    active_category_ids = {
        category["id"]
        for category in payload.get("categories", [])
        if category.get("name") == "ActiveTuberculosis"
    }
    active_image_ids = {
        annotation["image_id"]
        for annotation in payload.get("annotations", [])
        if annotation.get("category_id") in active_category_ids
    }
    train_entries = set(official_train_entries())
    for image in payload.get("images", []):
        file_name = image.get("file_name")
        if image.get("id") in active_image_ids and file_name in train_entries:
            path = OFFICIAL_IMAGE_ROOT / file_name
            if path.exists():
                return path.resolve()
    return None


def official_samples() -> dict[str, Path] | None:
    paths = {
        "healthy": first_official_train_path("health"),
        "active_tb": active_tb_official_train_path(),
        "sick_but_non_tb": first_official_train_path("sick"),
    }
    if all(paths.values()):
        return {category: path for category, path in paths.items() if path is not None}
    return None


def fallback_samples() -> dict[str, Path] | None:
    records, _, _ = load_records(DATA_DIR)
    if not records:
        return None
    paths = {}
    for category in ("healthy", "active_tb", "sick_but_non_tb"):
        record = sample_by_category(records, category, split="train")
        if record is None:
            return None
        paths[category] = record.path.resolve()
    return paths


def main() -> int:
    samples = official_samples() or fallback_samples()
    if not samples:
        print(f"ERROR: Could not find required TBX11K samples under {DATA_DIR}. Run download_dataset.py first.")
        return 1

    formatted = []
    for category in ("healthy", "active_tb", "sick_but_non_tb"):
        path = samples[category]
        formatted.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "path": str(path)},
                            {"type": "text", "text": PROMPT},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": assistant_response(category),
                    },
                ]
            }
        )

    OUTPUT_PATH.write_text(json.dumps(formatted, indent=2), encoding="utf-8")
    print(json.dumps(formatted, indent=2))
    print(f"\nSaved {len(formatted)} formatted samples to {OUTPUT_PATH}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
