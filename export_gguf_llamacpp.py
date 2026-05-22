from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from quantization_utils import (
    CANONICAL_ADAPTER_REPO_ID,
    CANONICAL_ADAPTER_REPO_PATH,
    CANONICAL_BASE_MODEL,
    directory_size_bytes,
    format_size_gb,
    write_json,
)


LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"
DEFAULT_QUANTIZED_REPO_ID = "ShivSingh123/drishti-qwen2vl-run4-llmcompressor-gptq-int4"
DEFAULT_GGUF_REPO_ID = "ShivSingh123/drishti-qwen2vl-run4-gguf"
DEFAULT_OUTPUT_DIR = Path("outputs/dri24-gguf")
DEFAULT_LLAMA_CPP_DIR = Path("external/llama.cpp")
DEFAULT_QUANTIZED_DIR = Path("outputs/dri23-run4-llmcompressor-gptq-int4")
DEFAULT_MERGED_DIR = Path("outputs/dri23-run4-merged-fp16")


@dataclass
class CommandResult:
    command: list[str]
    cwd: str | None
    returncode: int
    stdout_tail: str
    stderr_tail: str
    runtime_seconds: float


def _tail(text: str, limit: int = 6000) -> str:
    return text[-limit:]


def run_command(command: list[str], cwd: Path | None = None, check: bool = True) -> CommandResult:
    started_at = time.time()
    print("\n$ " + " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        capture_output=True,
    )
    if result.stdout:
        print(_tail(result.stdout), flush=True)
    if result.stderr:
        print(_tail(result.stderr), file=sys.stderr, flush=True)
    command_result = CommandResult(
        command=command,
        cwd=str(cwd) if cwd is not None else None,
        returncode=result.returncode,
        stdout_tail=_tail(result.stdout),
        stderr_tail=_tail(result.stderr),
        runtime_seconds=time.time() - started_at,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")
    return command_result


def ensure_llama_cpp(llama_cpp_dir: Path, build: bool = True, jobs: int = 2) -> list[CommandResult]:
    results: list[CommandResult] = []
    if llama_cpp_dir.exists():
        results.append(run_command(["git", "pull", "--ff-only"], cwd=llama_cpp_dir))
    else:
        llama_cpp_dir.parent.mkdir(parents=True, exist_ok=True)
        results.append(
            run_command(
                ["git", "clone", "--depth", "1", LLAMA_CPP_REPO, str(llama_cpp_dir)],
                cwd=None,
            )
        )

    results.append(run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=llama_cpp_dir))
    if build:
        results.append(run_command(["cmake", "-B", "build", "-DLLAMA_CURL=ON"], cwd=llama_cpp_dir))
        results.append(
            run_command(
                [
                    "cmake",
                    "--build",
                    "build",
                    "--config",
                    "Release",
                    "-j",
                    str(jobs),
                    "--target",
                    "llama-quantize",
                    "llama-mtmd-cli",
                ],
                cwd=llama_cpp_dir,
            )
        )
    return results


def find_llama_binary(llama_cpp_dir: Path, name: str) -> Path:
    candidates = [
        llama_cpp_dir / "build" / "bin" / name,
        llama_cpp_dir / "build" / "bin" / f"{name}.exe",
        llama_cpp_dir / "build" / "tools" / "mtmd" / name,
        llama_cpp_dir / "build" / "tools" / "mtmd" / f"{name}.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find built llama.cpp binary {name!r} under {llama_cpp_dir}.")


def download_hf_snapshot(repo_id: str, output_dir: Path, allow_patterns: list[str] | None = None) -> Path:
    from huggingface_hub import snapshot_download

    if output_dir.exists() and any(output_dir.iterdir()):
        return output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )
    return output_dir


def source_model_dir(args: argparse.Namespace) -> Path:
    if args.source == "quantized":
        return download_hf_snapshot(args.quantized_repo_id, args.quantized_dir)
    if args.source == "merged":
        if args.merged_dir.exists():
            return args.merged_dir
        run_command(
            [
                sys.executable,
                "merge_lora_checkpoint.py",
                "--base-model",
                args.base_model,
                "--adapter-repo-id",
                args.adapter_repo_id,
                "--adapter-repo-path",
                args.adapter_repo_path,
                "--output-dir",
                str(args.merged_dir),
                "--torch-dtype",
                "bfloat16",
            ]
        )
        return args.merged_dir
    raise ValueError(f"Unsupported source: {args.source}")


def file_metadata(path: Path) -> dict[str, Any]:
    size_bytes = directory_size_bytes(path)
    return {
        "path": str(path.resolve()),
        "size_bytes": size_bytes,
        "size_gb": format_size_gb(size_bytes),
    }


def convert_to_gguf(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command_results: list[CommandResult] = []
    command_results.extend(ensure_llama_cpp(args.llama_cpp_dir, build=not args.skip_build, jobs=args.jobs))

    model_dir = source_model_dir(args).resolve()
    convert_script = (args.llama_cpp_dir / "convert_hf_to_gguf.py").resolve()
    quantize_binary = find_llama_binary(args.llama_cpp_dir, "llama-quantize").resolve()

    intermediate_gguf = (args.output_dir / f"{args.output_prefix}-{args.intermediate_outtype}.gguf").resolve()
    quantized_gguf = (args.output_dir / f"{args.output_prefix}-{args.quantize_type.lower()}.gguf").resolve()
    mmproj_gguf = (args.output_dir / f"mmproj-{args.output_prefix}-{args.mmproj_outtype}.gguf").resolve()

    command_results.append(
        run_command(
            [
                sys.executable,
                str(convert_script),
                "--outfile",
                str(intermediate_gguf),
                "--outtype",
                args.intermediate_outtype,
                str(model_dir),
            ],
            cwd=args.llama_cpp_dir,
        )
    )
    command_results.append(
        run_command(
            [
                str(quantize_binary),
                str(intermediate_gguf),
                str(quantized_gguf),
                args.quantize_type.upper(),
            ]
        )
    )
    command_results.append(
        run_command(
            [
                sys.executable,
                str(convert_script),
                "--mmproj",
                "--outfile",
                str(mmproj_gguf),
                "--outtype",
                args.mmproj_outtype,
                str(model_dir),
            ],
            cwd=args.llama_cpp_dir,
        )
    )

    validation: dict[str, Any] | None = None
    if args.validate:
        if not args.sample_image:
            raise ValueError("--validate requires --sample-image.")
        mtmd_binary = find_llama_binary(args.llama_cpp_dir, "llama-mtmd-cli")
        validation_result = run_command(
            [
                str(mtmd_binary),
                "-m",
                str(quantized_gguf),
                "--mmproj",
                str(mmproj_gguf),
                "--image",
                str(args.sample_image),
                "-p",
                args.prompt,
                "-n",
                str(args.max_new_tokens),
                "--temp",
                "0",
            ],
            check=False,
        )
        command_results.append(validation_result)
        validation = {
            "passed": validation_result.returncode == 0,
            "sample_image": str(args.sample_image.resolve()),
            "prompt": args.prompt,
            "returncode": validation_result.returncode,
            "stdout_tail": validation_result.stdout_tail,
            "stderr_tail": validation_result.stderr_tail,
        }
        if args.fail_on_validation_error and validation_result.returncode != 0:
            raise RuntimeError("llama-mtmd-cli validation failed.")

    report = {
        "source": args.source,
        "source_model_dir": str(model_dir.resolve()),
        "quantized_repo_id": args.quantized_repo_id,
        "output_dir": str(args.output_dir.resolve()),
        "intermediate_gguf": file_metadata(intermediate_gguf),
        "text_gguf": file_metadata(quantized_gguf),
        "mmproj_gguf": file_metadata(mmproj_gguf),
        "quantize_type": args.quantize_type.upper(),
        "intermediate_outtype": args.intermediate_outtype,
        "mmproj_outtype": args.mmproj_outtype,
        "validation": validation,
        "commands": [asdict(result) for result in command_results],
        "runtime_seconds": time.time() - started_at,
    }
    write_json(args.output_dir / "gguf_export_report.json", report)
    return report


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Drishti Qwen2-VL run #4 artifacts to llama.cpp GGUF.")
    parser.add_argument("--source", default="quantized", choices=("quantized", "merged"), help="HF checkpoint source to convert.")
    parser.add_argument("--quantized-repo-id", default=DEFAULT_QUANTIZED_REPO_ID)
    parser.add_argument("--quantized-dir", default=DEFAULT_QUANTIZED_DIR, type=Path)
    parser.add_argument("--merged-dir", default=DEFAULT_MERGED_DIR, type=Path)
    parser.add_argument("--base-model", default=CANONICAL_BASE_MODEL)
    parser.add_argument("--adapter-repo-id", default=CANONICAL_ADAPTER_REPO_ID)
    parser.add_argument("--adapter-repo-path", default=CANONICAL_ADAPTER_REPO_PATH)
    parser.add_argument("--llama-cpp-dir", default=DEFAULT_LLAMA_CPP_DIR, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--output-prefix", default="drishti-qwen2vl-run4")
    parser.add_argument("--intermediate-outtype", default="f16", choices=("f32", "f16", "bf16", "q8_0", "auto"))
    parser.add_argument("--quantize-type", default="Q4_K_M")
    parser.add_argument("--mmproj-outtype", default="f16", choices=("f32", "f16", "bf16", "q8_0", "auto"))
    parser.add_argument("--skip-build", action="store_true", help="Use an existing llama.cpp build.")
    parser.add_argument("--jobs", default=max((os.cpu_count() or 2) // 2, 1), type=int)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--sample-image", type=Path, default=None)
    parser.add_argument("--prompt", default="Classify this chest X-ray. Respond exactly: Classification: <label>")
    parser.add_argument("--max-new-tokens", default=16, type=int)
    parser.add_argument("--fail-on-validation-error", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        report = convert_to_gguf(args)
    except Exception as exc:  # noqa: BLE001 - Colab CLI should print concise errors.
        print(f"ERROR: {exc}")
        return 1

    print("GGUF export complete.")
    print(f"text_gguf={report['text_gguf']['path']}")
    print(f"text_size_gb={report['text_gguf']['size_gb']:.3f}")
    print(f"mmproj_gguf={report['mmproj_gguf']['path']}")
    print(f"mmproj_size_gb={report['mmproj_gguf']['size_gb']:.3f}")
    if report["validation"] is not None:
        print(f"validation_passed={report['validation']['passed']}")
    print(f"report={Path(report['output_dir']) / 'gguf_export_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
