from __future__ import annotations

import json
import sys
from pathlib import Path

from tbx11k_utils import DATA_DIR, load_records, sample_by_category


OUTPUT_PATH = Path("formatted_samples.json")
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
        f"Classification: {category}. "
        f"Finding: {response['finding']} "
        f"Confidence: {response['confidence']}. "
        f"Referral recommended: {response['referral']}."
    )


def main() -> int:
    records, _, _ = load_records(DATA_DIR)
    if not records:
        print(f"ERROR: No TBX11K records found under {DATA_DIR}. Run download_dataset.py first.")
        return 1

    formatted = []
    for category in ("healthy", "active_tb", "sick_but_non_tb"):
        record = sample_by_category(records, category, split="train")
        if record is None:
            print(f"ERROR: Could not find a sample for category {category}.")
            return 1
        formatted.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(record.path)},
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
