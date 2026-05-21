from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch

from quantization_utils import (
    CANONICAL_ADAPTER_REPO_ID,
    CANONICAL_ADAPTER_REPO_PATH,
    CANONICAL_BASE_MODEL,
    directory_size_bytes,
    format_size_gb,
    write_json,
)


DEFAULT_OUTPUT_DIR = Path("outputs/dri23-run4-merged-fp16")


def resolve_torch_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported torch dtype: {name}")


def download_adapter_checkpoint(repo_id: str, repo_path: str) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=[f"{repo_path.rstrip('/')}/*"],
        )
    )
    adapter_dir = snapshot_dir / repo_path
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Downloaded snapshot did not contain {repo_path!r}.")
    return adapter_dir


def merge_lora_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    started_at = time.time()
    adapter_dir = args.adapter_dir or download_adapter_checkpoint(args.adapter_repo_id, args.adapter_repo_path)
    torch_dtype = resolve_torch_dtype(args.torch_dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    merged_model = model.merge_and_unload()
    merged_model.eval()

    merged_model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    processor.save_pretrained(args.output_dir)

    size_bytes = directory_size_bytes(args.output_dir)
    metadata = {
        "base_model": args.base_model,
        "adapter_dir": str(adapter_dir.resolve()),
        "adapter_repo_id": args.adapter_repo_id,
        "adapter_repo_path": args.adapter_repo_path,
        "output_dir": str(args.output_dir.resolve()),
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "max_shard_size": args.max_shard_size,
        "size_bytes": size_bytes,
        "size_gb": format_size_gb(size_bytes),
        "runtime_seconds": time.time() - started_at,
    }
    write_json(args.output_dir / "merge_metadata.json", metadata)
    return metadata


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge the selected Drishti LoRA adapter into the Qwen2-VL base model.")
    parser.add_argument("--base-model", default=CANONICAL_BASE_MODEL, help="Base Qwen2-VL model ID or local path.")
    parser.add_argument("--adapter-dir", default=None, type=Path, help="Local LoRA adapter checkpoint directory. If omitted, downloads from Hugging Face.")
    parser.add_argument("--adapter-repo-id", default=CANONICAL_ADAPTER_REPO_ID, help="Hugging Face adapter repo ID.")
    parser.add_argument("--adapter-repo-path", default=CANONICAL_ADAPTER_REPO_PATH, help="Path inside the adapter repo.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path, help="Directory for the merged full-precision model.")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=("bfloat16", "float16"), help="Dtype used when loading and saving the merged model.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map for loading the base model.")
    parser.add_argument("--max-shard-size", default="4GB", help="Maximum shard size passed to save_pretrained.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        metadata = merge_lora_checkpoint(args)
    except (ImportError, RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Merged adapter into {metadata['output_dir']}")
    print(f"size_gb={metadata['size_gb']:.3f}")
    print(f"metadata={Path(metadata['output_dir']) / 'merge_metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
