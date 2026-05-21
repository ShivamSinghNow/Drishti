from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


CANONICAL_BASE_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
CANONICAL_ADAPTER_REPO_ID = "ShivSingh123/drishti-qlora-run4-vision-lora-ablation"
CANONICAL_ADAPTER_REPO_PATH = "checkpoints/checkpoint-4950"
CANONICAL_GPTQ_REPO_ID = "ShivSingh123/drishti-qwen2vl-run4-gptq-int4"
CLASS_LABELS = ("active_tb", "healthy", "sick_but_non_tb")
FULL_PRECISION_BASELINE = {
    "checkpoint": f"{CANONICAL_ADAPTER_REPO_ID}/{CANONICAL_ADAPTER_REPO_PATH}",
    "accuracy": 0.987222,
    "macro_f1": 0.978787,
    "per_class_f1": {
        "active_tb": 0.953317,
        "healthy": 0.996881,
        "sick_but_non_tb": 0.986164,
    },
}


def structured_classification_label(text: str) -> str | None:
    stripped = text.strip()
    if "\n" in stripped:
        return None
    prefix = "Classification: "
    if not stripped.startswith(prefix):
        return None
    label = stripped.removeprefix(prefix).strip()
    return label if label in CLASS_LABELS else None


def is_structured_classification(text: str) -> bool:
    return structured_classification_label(text) is not None


def select_calibration_samples(samples: list[Any], total: int = 128) -> list[Any]:
    if total <= 0:
        raise ValueError("total calibration samples must be greater than 0.")
    if not samples:
        raise ValueError("At least one sample is required for calibration.")

    by_label: dict[str, list[Any]] = defaultdict(list)
    for sample in samples:
        if sample.true_label in CLASS_LABELS:
            by_label[sample.true_label].append(sample)

    missing_labels = [label for label in CLASS_LABELS if not by_label[label]]
    if missing_labels:
        raise ValueError(f"Calibration split is missing label(s): {', '.join(missing_labels)}")

    selected: list[Any] = []
    label_index = 0
    labels = list(CLASS_LABELS)
    while len(selected) < min(total, len(samples)):
        label = labels[label_index % len(labels)]
        bucket = by_label[label]
        bucket_offset = label_index // len(labels)
        if bucket_offset < len(bucket):
            selected.append(bucket[bucket_offset])
        label_index += 1
        if label_index > len(samples) * len(labels) and len(selected) < total:
            break

    if len(selected) < total:
        selected_ids = {sample.index for sample in selected}
        for sample in samples:
            if sample.index not in selected_ids:
                selected.append(sample)
                if len(selected) >= total:
                    break

    return [
        type(sample)(
            index=index,
            messages=sample.messages,
            true_label=sample.true_label,
            image_path=sample.image_path,
        )
        for index, sample in enumerate(selected[:total])
    ]


def _local_image_uri(path: str) -> str:
    return Path(path).expanduser().resolve().as_uri()


def qwen_vl_messages_for_sample(sample: Any, include_answer: bool = False) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for message in sample.messages:
        role = message.get("role")
        if role == "assistant" and not include_answer:
            continue
        content = message.get("content")
        if not isinstance(content, list):
            messages.append({"role": role, "content": content})
            continue

        converted_content = []
        for item in content:
            if item.get("type") == "image":
                if item.get("image"):
                    converted_content.append(dict(item))
                elif item.get("path"):
                    converted_content.append({"type": "image", "image": _local_image_uri(str(item["path"]))})
                else:
                    converted_content.append(dict(item))
            else:
                converted_content.append(dict(item))
        messages.append({"role": role, "content": converted_content})
    return messages


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        return path.stat().st_size
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def format_size_gb(size_bytes: int) -> float:
    return size_bytes / (1024**3)


def metric_delta_report(
    quantized_metrics: dict[str, Any],
    baseline: dict[str, Any] = FULL_PRECISION_BASELINE,
    max_macro_f1_drop: float = 0.02,
) -> dict[str, Any]:
    quantized_macro_f1 = float(quantized_metrics["macro_f1"])
    baseline_macro_f1 = float(baseline["macro_f1"])
    macro_f1_delta = quantized_macro_f1 - baseline_macro_f1
    macro_f1_drop = max(0.0, baseline_macro_f1 - quantized_macro_f1)

    per_class = {}
    for label, baseline_f1 in baseline["per_class_f1"].items():
        quantized_f1 = float(quantized_metrics.get("per_class", {}).get(label, {}).get("f1", 0.0))
        per_class[label] = {
            "baseline_f1": float(baseline_f1),
            "quantized_f1": quantized_f1,
            "delta": quantized_f1 - float(baseline_f1),
        }

    return {
        "baseline_checkpoint": baseline["checkpoint"],
        "baseline_accuracy": float(baseline["accuracy"]),
        "quantized_accuracy": float(quantized_metrics["accuracy"]),
        "accuracy_delta": float(quantized_metrics["accuracy"]) - float(baseline["accuracy"]),
        "baseline_macro_f1": baseline_macro_f1,
        "quantized_macro_f1": quantized_macro_f1,
        "macro_f1_delta": macro_f1_delta,
        "macro_f1_drop": macro_f1_drop,
        "max_allowed_macro_f1_drop": max_macro_f1_drop,
        "passed": macro_f1_drop <= max_macro_f1_drop,
        "per_class": per_class,
    }


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
