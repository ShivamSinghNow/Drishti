from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch

from quantization_utils import (
    FULL_PRECISION_BASELINE,
    is_structured_classification,
    metric_delta_report,
    qwen_vl_messages_for_sample,
)
from quantize_autogptq_qwen2vl import load_qwen2vl_gptq_class


DEFAULT_MODEL_DIR = Path("outputs/dri23-run4-gptq-int4")
DEFAULT_EVAL_OUTPUT_DIR = Path("outputs/eval/dri23-run4-gptq-int4")


def load_quantized_model_and_processor(model_dir: Path, device_map: str = "auto") -> tuple[Any, Any]:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    try:
        model_class = load_qwen2vl_gptq_class()
        model = model_class.from_quantized(str(model_dir), trust_remote_code=True, device_map=device_map)
    except (ImportError, AttributeError, RuntimeError, TypeError):
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir,
            torch_dtype="auto",
            device_map=device_map,
            trust_remote_code=True,
        )
    model.eval()
    return model, processor


def generate_locked_response(model: Any, processor: Any, sample: Any, max_new_tokens: int = 12) -> str:
    from qwen_vl_utils import process_vision_info

    messages = qwen_vl_messages_for_sample(sample, include_answer=False)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    prompt_length = int(inputs["input_ids"].shape[1])
    generated_only = generated_ids[:, prompt_length:]
    return processor.batch_decode(generated_only, skip_special_tokens=True)[0].strip()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a merged GPTQ INT4 Drishti Qwen2-VL model.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, type=Path, help="Quantized GPTQ model directory.")
    parser.add_argument("--data-dir", default=Path("data/processed"), type=Path, help="Directory containing split JSONL files.")
    parser.add_argument("--split", default="val", choices=("train", "val"), help="Split to evaluate.")
    parser.add_argument("--output-dir", default=DEFAULT_EVAL_OUTPUT_DIR, type=Path, help="Output directory for metrics and predictions.")
    parser.add_argument("--batch-size", default=3, type=int, help="Number of source samples per scoring batch.")
    parser.add_argument("--limit", default=None, type=int, help="Optional sample limit for smoke tests.")
    parser.add_argument("--limit-per-class", default=None, type=int, help="Optional stratified sample limit per class.")
    parser.add_argument("--gate", default="run3-full", choices=("none", "run3-diagnostic", "run4-diagnostic", "run3-full"), help="Optional eval gate.")
    parser.add_argument("--fail-on-gate-fail", action="store_true", help="Return exit code 2 when the selected gate fails.")
    parser.add_argument("--max-macro-f1-drop", default=0.02, type=float, help="Maximum allowed macro-F1 drop vs run #4 full-precision baseline.")
    parser.add_argument("--generation-smoke", action="store_true", help="Generate one response and verify the locked Classification format.")
    parser.add_argument("--device-map", default="auto", help="Device map passed to the model loader.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    started_at = time.time()
    try:
        from evaluate_checkpoint import (
            compute_metrics,
            evaluate_gate,
            evaluate_samples,
            load_eval_samples,
            stratified_limit_samples,
            write_outputs,
        )

        samples = load_eval_samples(args.data_dir, args.split)
        if args.limit_per_class is not None:
            samples = stratified_limit_samples(samples, args.limit_per_class)
        elif args.limit is not None:
            samples = samples[: args.limit]
        model, processor = load_quantized_model_and_processor(args.model_dir, args.device_map)
        structured_output = None
        if args.generation_smoke:
            generated = generate_locked_response(model, processor, samples[0])
            structured_output = {
                "sample_index": samples[0].index,
                "generated_text": generated,
                "passed": is_structured_classification(generated),
            }

        predictions = evaluate_samples(model, processor, samples, args.batch_size)
        metrics = compute_metrics(predictions)
        metrics["metadata"] = {
            "model_dir": str(args.model_dir.resolve()),
            "data_dir": str(args.data_dir.resolve()),
            "split": args.split,
            "limit": args.limit,
            "limit_per_class": args.limit_per_class,
            "batch_size": args.batch_size,
            "runtime_seconds": time.time() - started_at,
        }
        metrics["full_precision_baseline"] = FULL_PRECISION_BASELINE
        metrics["quantization_delta"] = metric_delta_report(metrics, max_macro_f1_drop=args.max_macro_f1_drop)
        if structured_output is not None:
            metrics["structured_output_smoke"] = structured_output

        gate_result = evaluate_gate(metrics, args.gate)
        if gate_result is not None:
            metrics["gate"] = gate_result
        metrics_path, predictions_path = write_outputs(args.output_dir, metrics, predictions)
    except (ImportError, RuntimeError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Evaluated {metrics['total_samples']} samples.")
    print(f"accuracy={metrics['accuracy']:.6f}")
    print(f"macro_f1={metrics['macro_f1']:.6f}")
    print(f"macro_f1_delta={metrics['quantization_delta']['macro_f1_delta']:.6f}")
    print(f"macro_f1_drop={metrics['quantization_delta']['macro_f1_drop']:.6f}")
    print(f"delta_gate_passed={metrics['quantization_delta']['passed']}")
    if "structured_output_smoke" in metrics:
        print(f"structured_output_passed={metrics['structured_output_smoke']['passed']}")
        print(f"structured_output={metrics['structured_output_smoke']['generated_text']}")
    if "gate" in metrics:
        print(f"gate={metrics['gate']['name']} passed={metrics['gate']['passed']}")
    print(f"metrics={metrics_path}")
    print(f"predictions={predictions_path}")

    if args.fail_on_gate_fail:
        failed_eval_gate = bool(metrics.get("gate")) and not metrics["gate"]["passed"]
        failed_delta_gate = not metrics["quantization_delta"]["passed"]
        failed_structured_gate = (
            "structured_output_smoke" in metrics and not metrics["structured_output_smoke"]["passed"]
        )
        if failed_eval_gate or failed_delta_gate or failed_structured_gate:
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
