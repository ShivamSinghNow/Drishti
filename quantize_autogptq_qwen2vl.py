from __future__ import annotations

import argparse
import contextlib
import importlib
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from quantization_utils import (
    directory_size_bytes,
    format_size_gb,
    qwen_vl_messages_for_sample,
    select_calibration_samples,
    write_json,
)


DEFAULT_MERGED_MODEL_DIR = Path("outputs/dri23-run4-merged-fp16")
DEFAULT_OUTPUT_DIR = Path("outputs/dri23-run4-gptq-int4")
DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_SIZE_LIMIT_GB = 4.0


def ensure_autogptq_transformers_compat() -> None:
    import torch
    import transformers.modeling_utils as modeling_utils

    try:
        import peft.mapping as peft_mapping
        import peft.peft_model as peft_model

        if not hasattr(peft_model, "PEFT_TYPE_TO_MODEL_MAPPING") and hasattr(
            peft_mapping,
            "PEFT_TYPE_TO_MODEL_MAPPING",
        ):
            peft_model.PEFT_TYPE_TO_MODEL_MAPPING = peft_mapping.PEFT_TYPE_TO_MODEL_MAPPING
        elif not hasattr(peft_model, "PEFT_TYPE_TO_MODEL_MAPPING"):
            peft_model.PEFT_TYPE_TO_MODEL_MAPPING = {}
    except ImportError:
        pass

    try:
        from transformers import Qwen2VLForConditionalGeneration
    except ImportError:
        Qwen2VLForConditionalGeneration = None

    if Qwen2VLForConditionalGeneration is not None and not getattr(
        Qwen2VLForConditionalGeneration,
        "_drishti_autogptq_init_patched",
        False,
    ):
        original_init = Qwen2VLForConditionalGeneration.__init__
        stale_hub_kwargs = {
            "cache_dir",
            "force_download",
            "local_files_only",
            "mirror",
            "proxies",
            "resume_download",
            "revision",
            "subfolder",
            "token",
            "use_auth_token",
            "_commit_hash",
        }

        def patched_init(self, *args, **kwargs):
            for key in stale_hub_kwargs:
                kwargs.pop(key, None)
            return original_init(self, *args, **kwargs)

        Qwen2VLForConditionalGeneration.__init__ = patched_init
        Qwen2VLForConditionalGeneration._drishti_autogptq_init_patched = True

    try:
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDecoderLayer
    except ImportError:
        Qwen2VLDecoderLayer = None

    if Qwen2VLDecoderLayer is not None and not getattr(
        Qwen2VLDecoderLayer,
        "_drishti_autogptq_forward_patched",
        False,
    ):
        original_decoder_forward = Qwen2VLDecoderLayer.forward

        def patched_decoder_forward(self, hidden_states, *args, **kwargs):
            if torch.is_tensor(hidden_states) and hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)
            if torch.is_tensor(kwargs.get("hidden_states")) and kwargs["hidden_states"].dim() == 2:
                kwargs["hidden_states"] = kwargs["hidden_states"].unsqueeze(0)
            return original_decoder_forward(self, hidden_states, *args, **kwargs)

        Qwen2VLDecoderLayer.forward = patched_decoder_forward
        Qwen2VLDecoderLayer._drishti_autogptq_forward_patched = True

    if hasattr(modeling_utils, "no_init_weights"):
        return

    init_function_names = (
        "uniform_",
        "normal_",
        "trunc_normal_",
        "constant_",
        "xavier_uniform_",
        "xavier_normal_",
        "kaiming_uniform_",
        "kaiming_normal_",
        "orthogonal_",
        "sparse_",
    )

    @contextlib.contextmanager
    def no_init_weights(_enable: bool = True):
        old_init_weights = getattr(modeling_utils, "_init_weights", True)
        originals = {}
        if _enable:
            modeling_utils._init_weights = False

            def _skip_init(*_args, **_kwargs):
                return None

            for name in init_function_names:
                if hasattr(torch.nn.init, name):
                    originals[name] = getattr(torch.nn.init, name)
                    setattr(torch.nn.init, name, _skip_init)
        try:
            yield
        finally:
            modeling_utils._init_weights = old_init_weights
            for name, original in originals.items():
                setattr(torch.nn.init, name, original)

    modeling_utils.no_init_weights = no_init_weights


def load_qwen2vl_gptq_class() -> Any:
    ensure_autogptq_transformers_compat()
    candidates = (
        ("auto_gptq.modeling.qwen2_vl", "Qwen2VLGPTQForConditionalGeneration"),
        ("auto_gptq.modeling.qwen2vl", "Qwen2VLGPTQForConditionalGeneration"),
        ("auto_gptq.modeling.qwen2_vl_gptq", "Qwen2VLGPTQForConditionalGeneration"),
    )
    errors = []
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as exc:
            errors.append(f"{module_name}.{class_name}: {exc}")
    details = "\n".join(errors)
    raise ImportError(
        "Could not import Qwen2-VL AutoGPTQ support. Install the Qwen2-VL AutoGPTQ fork in Colab, "
        "for example: python -m pip install git+https://github.com/kq-chen/AutoGPTQ.git\n"
        f"Tried:\n{details}"
    )


def prepare_calibration_examples(processor: Any, samples: list[Any]) -> list[dict[str, Any]]:
    from qwen_vl_utils import process_vision_info

    examples = []
    for sample in samples:
        messages = qwen_vl_messages_for_sample(sample, include_answer=True)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        examples.append(dict(inputs))
    return examples


def module_by_path(root: Any, path: str) -> Any | None:
    current = root
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def patch_qwen2vl_gptq_layout(gptq_model: Any) -> None:
    import torch

    root_model = gptq_model.model
    existing_layers = module_by_path(root_model, gptq_model.layers_block_name)
    if existing_layers is not None:
        return

    layer_type = getattr(gptq_model, "layer_type", "Qwen2VLDecoderLayer")
    for module_name, module in root_model.named_modules():
        if not isinstance(module, torch.nn.ModuleList) or len(module) == 0:
            continue
        if module[0].__class__.__name__ != layer_type:
            continue
        gptq_model.layers_block_name = module_name
        prefix = module_name.removesuffix(".layers")
        gptq_model.outside_layer_modules = [
            f"{prefix}.embed_tokens",
            f"{prefix}.norm",
            "visual",
        ]
        return

    sample_names = [
        name
        for name, module in root_model.named_modules()
        if isinstance(module, torch.nn.ModuleList)
    ][:20]
    raise ValueError(
        "Could not locate Qwen2-VL decoder layers for AutoGPTQ. "
        f"Tried {gptq_model.layers_block_name!r}; found ModuleLists: {sample_names}"
    )


def quantize_model(args: argparse.Namespace) -> dict[str, Any]:
    ensure_autogptq_transformers_compat()
    import torch
    from auto_gptq import BaseQuantizeConfig
    from transformers import AutoProcessor
    from evaluate_checkpoint import load_eval_samples

    started_at = time.time()
    samples = load_eval_samples(args.data_dir, args.split)
    calibration_samples = select_calibration_samples(samples, args.calibration_samples)
    processor = AutoProcessor.from_pretrained(args.merged_model_dir, trust_remote_code=True)
    calibration_examples = prepare_calibration_examples(processor, calibration_samples)

    quantize_config = BaseQuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=args.desc_act,
    )
    model_class = load_qwen2vl_gptq_class()
    model = model_class.from_pretrained(
        str(args.merged_model_dir),
        quantize_config=quantize_config,
        trust_remote_code=True,
        device_map=args.device_map,
    )
    if not hasattr(model.model.config, "use_cache"):
        model.model.config.use_cache = False
    patch_qwen2vl_gptq_layout(model)
    if args.force_model_cuda and torch.cuda.is_available():
        model.model.to("cuda:0")
    model.quantize(calibration_examples, batch_size=args.batch_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save_quantized(str(args.output_dir), use_safetensors=True)
    except TypeError:
        model.save_quantized(str(args.output_dir))
    processor.save_pretrained(args.output_dir)

    size_bytes = directory_size_bytes(args.output_dir)
    size_gb = format_size_gb(size_bytes)
    label_counts: dict[str, int] = {}
    for sample in calibration_samples:
        label_counts[sample.true_label] = label_counts.get(sample.true_label, 0) + 1
    metadata = {
        "merged_model_dir": str(args.merged_model_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "split": args.split,
        "calibration_samples": len(calibration_samples),
        "calibration_label_counts": dict(sorted(label_counts.items())),
        "bits": args.bits,
        "group_size": args.group_size,
        "desc_act": args.desc_act,
        "batch_size": args.batch_size,
        "device_map": args.device_map,
        "size_bytes": size_bytes,
        "size_gb": size_gb,
        "size_limit_gb": args.size_limit_gb,
        "size_gate_passed": size_gb < args.size_limit_gb,
        "runtime_seconds": time.time() - started_at,
    }
    write_json(args.output_dir / "quantization_metadata.json", metadata)
    return metadata


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize a merged Qwen2-VL Drishti checkpoint to INT4 with AutoGPTQ."
    )
    parser.add_argument("--merged-model-dir", default=DEFAULT_MERGED_MODEL_DIR, type=Path, help="Merged full-precision model directory.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, type=Path, help="Directory containing generated JSONL splits.")
    parser.add_argument("--split", default="val", choices=("train", "val"), help="Split used for calibration.")
    parser.add_argument("--calibration-samples", default=128, type=int, help="Number of stratified calibration samples.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path, help="Directory for the GPTQ INT4 model.")
    parser.add_argument("--bits", default=4, type=int, choices=(2, 3, 4, 8), help="GPTQ bit width.")
    parser.add_argument("--group-size", default=128, type=int, help="GPTQ group size.")
    parser.add_argument("--desc-act", action="store_true", help="Enable activation-order GPTQ quantization.")
    parser.add_argument("--batch-size", default=1, type=int, help="Calibration batch size.")
    parser.add_argument("--device-map", default="auto", help="Device map passed to the GPTQ model loader.")
    parser.add_argument("--debug-traceback", action="store_true", help="Print full traceback on script-level failures.")
    parser.add_argument("--no-force-model-cuda", dest="force_model_cuda", action="store_false", help="Do not move the full merged model to cuda:0 before quantization.")
    parser.add_argument("--size-limit-gb", default=DEFAULT_SIZE_LIMIT_GB, type=float, help="Acceptance threshold for quantized model size.")
    parser.set_defaults(force_model_cuda=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        metadata = quantize_model(args)
    except (ImportError, RuntimeError, FileNotFoundError, ValueError, KeyError) as exc:
        if args.debug_traceback:
            traceback.print_exc()
        print(f"ERROR: {exc}")
        return 1

    print(f"Quantized model saved to {metadata['output_dir']}")
    print(f"size_gb={metadata['size_gb']:.3f}")
    print(f"size_gate_passed={metadata['size_gate_passed']}")
    print(f"metadata={Path(metadata['output_dir']) / 'quantization_metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
