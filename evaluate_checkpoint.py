from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm

from build_dataloader import DEFAULT_DATA_DIR, IGNORE_INDEX, mask_prompt_labels
from format_samples import assistant_response
from preprocess_samples import MODEL_NAME


CLASS_LABELS = ("active_tb", "healthy", "sick_but_non_tb")
TB_POSITIVE_LABELS = ("active_tb", "latent_tb")
DEFAULT_OUTPUT_DIR = Path("outputs/eval/checkpoint")
RUN3_FULL_VAL_SIZE = 1800


@dataclass(frozen=True)
class EvalSample:
    index: int
    messages: list[dict[str, Any]]
    true_label: str
    image_path: str | None


@dataclass(frozen=True)
class PredictionRecord:
    index: int
    image_path: str | None
    true_label: str
    predicted_label: str
    binary_tb_score: float
    class_probabilities: dict[str, float]
    class_log_likelihoods: dict[str, float]


def extract_true_label(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        for line in content.splitlines():
            if line.startswith("Classification: "):
                return line.removeprefix("Classification: ").strip()
    raise ValueError("Messages are missing a locked assistant Classification line.")


def extract_image_path(messages: list[dict[str, Any]]) -> str | None:
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") == "image" and item.get("path"):
                return str(item["path"])
    return None


def load_eval_samples(data_dir: Path, split: str, limit: int | None = None) -> list[EvalSample]:
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run generate_jsonl.py first.")

    samples = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            messages = json.loads(line)["messages"]
            samples.append(
                EvalSample(
                    index=len(samples),
                    messages=messages,
                    true_label=extract_true_label(messages),
                    image_path=extract_image_path(messages),
                )
            )
            if limit is not None and len(samples) >= limit:
                break
    return samples


def stratified_limit_samples(samples: list[EvalSample], limit_per_class: int) -> list[EvalSample]:
    if limit_per_class <= 0:
        raise ValueError("--limit-per-class must be greater than 0.")

    by_label = {label: [] for label in CLASS_LABELS}
    for sample in samples:
        if sample.true_label not in by_label:
            continue
        if len(by_label[sample.true_label]) < limit_per_class:
            by_label[sample.true_label].append(sample)

    missing = {
        label: limit_per_class - len(label_samples)
        for label, label_samples in by_label.items()
        if len(label_samples) < limit_per_class
    }
    if missing:
        details = ", ".join(f"{label}: missing {count}" for label, count in sorted(missing.items()))
        raise ValueError(f"Not enough samples for stratified limit. {details}")

    limited = []
    for label in CLASS_LABELS:
        limited.extend(by_label[label])
    return [
        EvalSample(
            index=index,
            messages=sample.messages,
            true_label=sample.true_label,
            image_path=sample.image_path,
        )
        for index, sample in enumerate(limited)
    ]


def candidate_messages(sample: EvalSample, candidate_label: str) -> list[dict[str, Any]]:
    if candidate_label not in CLASS_LABELS:
        raise ValueError(f"Unsupported candidate label: {candidate_label}")
    return [
        copy.deepcopy(sample.messages[0]),
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": assistant_response(candidate_label),
                }
            ],
        },
    ]


def softmax_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    max_score = max(scores)
    exp_scores = [math.exp(score - max_score) for score in scores]
    total = sum(exp_scores)
    return [score / total for score in exp_scores]


def predict_label(probabilities: dict[str, float]) -> str:
    return max(probabilities.items(), key=lambda item: item[1])[0]


def average_label_log_likelihood(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    label_mask = shifted_labels != IGNORE_INDEX
    safe_labels = shifted_labels.masked_fill(~label_mask, 0)

    token_log_probs = torch.log_softmax(shifted_logits, dim=-1)
    token_log_probs = token_log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs.masked_fill(~label_mask, 0.0)

    token_counts = label_mask.sum(dim=1)
    if torch.any(token_counts <= 0):
        raise ValueError("Every candidate response must contain at least one scored label token.")
    return token_log_probs.sum(dim=1) / token_counts


def model_device(model: Any) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_tensors(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def score_sample_batch(
    model: Any,
    processor: Any,
    samples: list[EvalSample],
    candidate_labels: tuple[str, ...] = CLASS_LABELS,
) -> list[PredictionRecord]:
    conversations = []
    prompt_conversations = []
    for sample in samples:
        for label in candidate_labels:
            messages = candidate_messages(sample, label)
            conversations.append(messages)
            prompt_conversations.append(messages[:1])

    batch = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    prompt_batch = processor.apply_chat_template(
        prompt_conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    padding_side = getattr(getattr(processor, "tokenizer", None), "padding_side", "left")
    labels = mask_prompt_labels(
        batch["input_ids"],
        batch["attention_mask"],
        prompt_batch["attention_mask"].sum(dim=1),
        padding_side,
    )

    device = model_device(model)
    batch = move_tensors(batch, device)
    labels = labels.to(device)
    with torch.inference_mode():
        outputs = model(**batch)
        scores = average_label_log_likelihood(outputs.logits, labels).detach().cpu().tolist()

    predictions = []
    width = len(candidate_labels)
    for sample_index, sample in enumerate(samples):
        offset = sample_index * width
        sample_scores = scores[offset : offset + width]
        probabilities = softmax_scores(sample_scores)
        probability_by_label = dict(zip(candidate_labels, probabilities, strict=True))
        score_by_label = dict(zip(candidate_labels, sample_scores, strict=True))
        predicted_label = predict_label(probability_by_label)
        binary_tb_score = sum(probability_by_label.get(label, 0.0) for label in TB_POSITIVE_LABELS)
        predictions.append(
            PredictionRecord(
                index=sample.index,
                image_path=sample.image_path,
                true_label=sample.true_label,
                predicted_label=predicted_label,
                binary_tb_score=binary_tb_score,
                class_probabilities=probability_by_label,
                class_log_likelihoods=score_by_label,
            )
        )
    return predictions


def evaluate_samples(
    model: Any,
    processor: Any,
    samples: list[EvalSample],
    batch_size: int,
) -> list[PredictionRecord]:
    predictions = []
    for start in tqdm(range(0, len(samples), batch_size), desc="scoring", unit="batch"):
        predictions.extend(score_sample_batch(model, processor, samples[start : start + batch_size]))
    return predictions


def metric_labels(predictions: list[PredictionRecord]) -> tuple[str, ...]:
    labels = list(CLASS_LABELS)
    for prediction in predictions:
        for label in (prediction.true_label, prediction.predicted_label):
            if label not in labels:
                labels.append(label)
    return tuple(labels)


def compute_metrics(predictions: list[PredictionRecord]) -> dict[str, Any]:
    if not predictions:
        raise ValueError("Cannot compute metrics for zero predictions.")

    labels = metric_labels(predictions)
    y_true = [prediction.true_label for prediction in predictions]
    y_pred = [prediction.predicted_label for prediction in predictions]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(labels),
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }

    y_true_binary = [1 if label in TB_POSITIVE_LABELS else 0 for label in y_true]
    y_score_binary = [prediction.binary_tb_score for prediction in predictions]
    binary_roc_auc = None
    if len(set(y_true_binary)) > 1:
        binary_roc_auc = float(roc_auc_score(y_true_binary, y_score_binary))

    return {
        "total_samples": len(predictions),
        "labels": list(labels),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
        "per_class": per_class,
        "binary_tb_positive_labels": list(TB_POSITIVE_LABELS),
        "binary_roc_auc": binary_roc_auc,
        "confusion_matrix": {
            "labels": list(labels),
            "matrix": confusion_matrix(y_true, y_pred, labels=list(labels)).astype(int).tolist(),
        },
        "class_distribution": dict(sorted(Counter(y_true).items())),
        "prediction_distribution": dict(sorted(Counter(y_pred).items())),
    }


def _prediction_count(metrics: dict[str, Any], label: str) -> int:
    return int(metrics.get("prediction_distribution", {}).get(label, 0))


def _zero_recall_labels(metrics: dict[str, Any]) -> list[str]:
    per_class = metrics.get("per_class", {})
    labels = []
    for label in CLASS_LABELS:
        class_metrics = per_class.get(label, {})
        if class_metrics.get("support", 0) > 0 and class_metrics.get("recall", 0.0) <= 0.0:
            labels.append(label)
    return labels


def _scaled_count(full_val_count: int, total_samples: int, full_val_size: int) -> int:
    if full_val_size <= 0:
        raise ValueError("full_val_size must be greater than 0.")
    return max(1, math.ceil((full_val_count * total_samples) / full_val_size))


def evaluate_run3_diagnostic_gate(
    metrics: dict[str, Any],
    full_val_size: int = RUN3_FULL_VAL_SIZE,
) -> dict[str, Any]:
    total_samples = int(metrics["total_samples"])
    max_prediction_count = max(metrics.get("prediction_distribution", {}).values(), default=0)
    active_tb_minimum = _scaled_count(50, total_samples, full_val_size)
    zero_recall_labels = _zero_recall_labels(metrics)
    checks = {
        "no_zero_recall": {
            "passed": not zero_recall_labels,
            "zero_recall_labels": zero_recall_labels,
        },
        "no_prediction_monopoly": {
            "passed": max_prediction_count <= (0.80 * total_samples),
            "max_prediction_count": int(max_prediction_count),
            "max_allowed_count": int(math.floor(0.80 * total_samples)),
        },
        "active_tb_predictions_minimum": {
            "passed": _prediction_count(metrics, "active_tb") >= active_tb_minimum,
            "actual": _prediction_count(metrics, "active_tb"),
            "minimum": active_tb_minimum,
        },
        "healthy_predictions_nonzero": {
            "passed": _prediction_count(metrics, "healthy") > 0,
            "actual": _prediction_count(metrics, "healthy"),
        },
    }
    return {
        "name": "run3_diagnostic",
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def evaluate_run4_diagnostic_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    total_samples = int(metrics["total_samples"])
    max_prediction_count = max(metrics.get("prediction_distribution", {}).values(), default=0)
    zero_recall_labels = _zero_recall_labels(metrics)
    minimum_predictions_by_class = {
        label: {
            "passed": _prediction_count(metrics, label) >= 50,
            "actual": _prediction_count(metrics, label),
            "minimum": 50,
        }
        for label in CLASS_LABELS
    }
    checks = {
        "no_zero_recall": {
            "passed": not zero_recall_labels,
            "zero_recall_labels": zero_recall_labels,
        },
        "no_prediction_monopoly": {
            "passed": max_prediction_count <= (0.80 * total_samples),
            "max_prediction_count": int(max_prediction_count),
            "max_allowed_count": int(math.floor(0.80 * total_samples)),
        },
        "minimum_predictions_per_class": {
            "passed": all(check["passed"] for check in minimum_predictions_by_class.values()),
            "minimum": 50,
            "by_class": minimum_predictions_by_class,
        },
    }
    return {
        "name": "run4_diagnostic",
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def evaluate_run3_success_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    zero_recall_labels = _zero_recall_labels(metrics)
    checks = {
        "full_val_size": {
            "passed": int(metrics["total_samples"]) == RUN3_FULL_VAL_SIZE,
            "actual": int(metrics["total_samples"]),
            "expected": RUN3_FULL_VAL_SIZE,
        },
        "macro_f1": {
            "passed": float(metrics["macro_f1"]) >= 0.30,
            "actual": float(metrics["macro_f1"]),
            "minimum": 0.30,
        },
        "accuracy": {
            "passed": float(metrics["accuracy"]) >= 0.50,
            "actual": float(metrics["accuracy"]),
            "minimum": 0.50,
        },
        "sick_but_non_tb_predictions_range": {
            "passed": 400 <= _prediction_count(metrics, "sick_but_non_tb") <= 1000,
            "actual": _prediction_count(metrics, "sick_but_non_tb"),
            "minimum": 400,
            "maximum": 1000,
        },
        "healthy_predictions_range": {
            "passed": 400 <= _prediction_count(metrics, "healthy") <= 1000,
            "actual": _prediction_count(metrics, "healthy"),
            "minimum": 400,
            "maximum": 1000,
        },
        "active_tb_predictions_minimum": {
            "passed": _prediction_count(metrics, "active_tb") >= 50,
            "actual": _prediction_count(metrics, "active_tb"),
            "minimum": 50,
        },
        "no_zero_recall": {
            "passed": not zero_recall_labels,
            "zero_recall_labels": zero_recall_labels,
        },
    }
    return {
        "name": "run3_full",
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def evaluate_gate(metrics: dict[str, Any], gate: str, full_val_size: int = RUN3_FULL_VAL_SIZE) -> dict[str, Any] | None:
    if gate == "none":
        return None
    if gate == "run3-diagnostic":
        return evaluate_run3_diagnostic_gate(metrics, full_val_size)
    if gate == "run4-diagnostic":
        return evaluate_run4_diagnostic_gate(metrics)
    if gate == "run3-full":
        return evaluate_run3_success_gate(metrics)
    raise ValueError(f"Unsupported gate: {gate}")


def write_outputs(
    output_dir: Path,
    metrics: dict[str, Any],
    predictions: list[PredictionRecord],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "eval_results.json"
    predictions_path = output_dir / "predictions.jsonl"

    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    with predictions_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(asdict(prediction), sort_keys=True))
            handle.write("\n")
    return metrics_path, predictions_path


def load_model_and_processor(model_name: str, adapter_dir: Path | None) -> tuple[Any, Any]:
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    from train_qlora import build_quantization_config, require_full_training_environment

    require_full_training_environment()
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=build_quantization_config(),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.eval()
    return model, processor


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen2-VL/Drishti checkpoints on TBX11K JSONL splits.")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Base model name.")
    parser.add_argument("--adapter-dir", default=None, type=Path, help="Optional LoRA adapter checkpoint directory.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, type=Path, help="Directory containing split JSONL files.")
    parser.add_argument("--split", default="val", choices=("train", "val"), help="Split to evaluate.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path, help="Directory for eval_results.json and predictions.jsonl.")
    parser.add_argument("--batch-size", default=3, type=int, help="Number of source samples per scoring batch.")
    parser.add_argument("--limit", default=None, type=int, help="Optional sample limit for smoke tests.")
    parser.add_argument("--limit-per-class", default=None, type=int, help="Optional stratified sample limit per class.")
    parser.add_argument("--gate", default="none", choices=("none", "run3-diagnostic", "run4-diagnostic", "run3-full"), help="Optional success gate to attach to eval results.")
    parser.add_argument("--gate-full-val-size", default=RUN3_FULL_VAL_SIZE, type=int, help="Full validation size used for proportional diagnostic gates.")
    parser.add_argument("--fail-on-gate-fail", action="store_true", help="Return exit code 2 when the selected gate fails.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    started_at = time.time()
    try:
        samples = load_eval_samples(args.data_dir, args.split)
        if args.limit_per_class is not None:
            samples = stratified_limit_samples(samples, args.limit_per_class)
        elif args.limit is not None:
            samples = samples[: args.limit]
        model, processor = load_model_and_processor(args.model_name, args.adapter_dir)
        predictions = evaluate_samples(model, processor, samples, args.batch_size)
        metrics = compute_metrics(predictions)
        metrics["metadata"] = {
            "model_name": args.model_name,
            "adapter_dir": str(args.adapter_dir.resolve()) if args.adapter_dir else None,
            "data_dir": str(args.data_dir.resolve()),
            "split": args.split,
            "limit": args.limit,
            "limit_per_class": args.limit_per_class,
            "batch_size": args.batch_size,
            "candidate_labels": list(CLASS_LABELS),
            "runtime_seconds": time.time() - started_at,
        }
        gate_result = evaluate_gate(metrics, args.gate, args.gate_full_val_size)
        if gate_result is not None:
            metrics["gate"] = gate_result
        metrics_path, predictions_path = write_outputs(args.output_dir, metrics, predictions)
    except (ImportError, RuntimeError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Evaluated {metrics['total_samples']} samples.")
    print(f"accuracy={metrics['accuracy']:.6f}")
    print(f"macro_f1={metrics['macro_f1']:.6f}")
    print(f"binary_roc_auc={metrics['binary_roc_auc']}")
    if "gate" in metrics:
        print(f"gate={metrics['gate']['name']} passed={metrics['gate']['passed']}")
    print(f"metrics={metrics_path}")
    print(f"predictions={predictions_path}")
    if args.fail_on_gate_fail and metrics.get("gate") and not metrics["gate"]["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
