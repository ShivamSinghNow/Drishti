from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from quantization_utils import directory_size_bytes, format_size_gb, write_json


DEFAULT_OUTPUT_DIR = Path("outputs/dri25-offline-llamacpp")
DEFAULT_PROMPT = "Analyze this chest X-ray and respond exactly: Classification: <label>"
LABEL_PATTERN = r"Classification:\s*(active_tb|healthy|sick_but_non_tb)"
GENERIC_TOKENS_PER_SECOND_PATTERN = re.compile(r"(?P<tps>[0-9]+(?:\.[0-9]+)?)\s+tokens per second", re.IGNORECASE)
PERF_LABEL_PATTERN = re.compile(r"(?::|^)\s*(?P<label>[A-Za-z _/-]*?time)\s*=", re.IGNORECASE)


@dataclass
class OfflineCheck:
    checked: bool
    passed: bool | None
    host: str
    port: int
    error: str | None = None


@dataclass
class LlamaCppOfflineResult:
    command: list[str]
    returncode: int
    wall_clock_seconds: float
    latency_threshold_seconds: float
    latency_passed: bool
    structured_output: str | None
    structured_output_passed: bool
    tokens_per_second: float | None
    performance_entries: list[dict[str, Any]]
    stdout_path: str
    stderr_path: str
    sample_output_path: str
    report_path: str
    passed: bool


def find_mtmd_binary(llama_cpp_dir: Path, explicit_binary: Path | None = None) -> Path:
    if explicit_binary is not None:
        return explicit_binary

    candidates = [
        llama_cpp_dir / "build" / "bin" / "llama-mtmd-cli",
        llama_cpp_dir / "build" / "bin" / "llama-mtmd-cli.exe",
        llama_cpp_dir / "build" / "tools" / "mtmd" / "llama-mtmd-cli",
        llama_cpp_dir / "build" / "tools" / "mtmd" / "llama-mtmd-cli.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find llama-mtmd-cli. Pass --mtmd-binary or build llama.cpp with the llama-mtmd-cli target."
    )


def require_existing_file(path: Path, name: str) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{name} does not exist or is not a file: {path}")
    return path.resolve()


def check_network_offline(host: str = "1.1.1.1", port: int = 53, timeout_seconds: float = 2.0) -> OfflineCheck:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return OfflineCheck(checked=True, passed=False, host=host, port=port)
    except OSError as exc:
        return OfflineCheck(checked=True, passed=True, host=host, port=port, error=str(exc))


def extract_structured_output(text: str, pattern: str = LABEL_PATTERN) -> str | None:
    match = re.search(pattern, text)
    if match is None:
        return None
    return match.group(0).strip()


def parse_performance_entries(text: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line in text.splitlines():
        tps_match = GENERIC_TOKENS_PER_SECOND_PATTERN.search(line)
        if tps_match is None:
            continue
        label_match = PERF_LABEL_PATTERN.search(line)
        label = "unlabeled" if label_match is None else " ".join(label_match.group("label").strip().split())
        entries.append({"label": label, "tokens_per_second": float(tps_match.group("tps"))})
    return entries


def select_generation_tokens_per_second(entries: list[dict[str, Any]]) -> float | None:
    if not entries:
        return None
    for entry in reversed(entries):
        if entry["label"].lower() == "eval time":
            return float(entry["tokens_per_second"])
    return float(entries[-1]["tokens_per_second"])


def build_llama_command(args: argparse.Namespace, mtmd_binary: Path, model_path: Path, mmproj_path: Path, image_path: Path) -> list[str]:
    command = [
        str(mtmd_binary),
        "-m",
        str(model_path),
        "--mmproj",
        str(mmproj_path),
        args.image_flag,
        str(image_path),
        "-p",
        args.prompt,
        "-n",
        str(args.max_new_tokens),
        "--temp",
        str(args.temperature),
    ]
    if args.cpu_only:
        command.extend(["-ngl", "0", "--no-mmproj-offload"])
    if args.threads is not None:
        command.extend(["-t", str(args.threads)])
    if args.context_size is not None:
        command.extend(["-c", str(args.context_size)])
    command.extend(args.extra_llama_arg)
    return command


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_model_card_snippet(report: dict[str, Any]) -> str:
    result = report["result"]
    artifacts = report["artifacts"]
    return "\n".join(
        [
            "## DRI-25 Offline llama.cpp Validation",
            "",
            f"- Text GGUF: `{artifacts['model_path']}` ({artifacts['model_size_gb']:.3f} GB)",
            f"- mmproj GGUF: `{artifacts['mmproj_path']}` ({artifacts['mmproj_size_gb']:.3f} GB)",
            f"- CPU-only: `{report['runtime']['cpu_only']}`",
            f"- Wall-clock latency: `{result['wall_clock_seconds']:.3f}s`",
            f"- Latency gate: `{result['latency_threshold_seconds']:.3f}s`, passed `{result['latency_passed']}`",
            f"- Tokens/sec: `{result['tokens_per_second']}`",
            f"- Structured output passed: `{result['structured_output_passed']}`",
            f"- Sample output: `{result['structured_output']}`",
            f"- Offline network check: `{report['offline_check']['passed']}`",
            "",
        ]
    )


def verify_offline_llamacpp(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = require_existing_file(args.model, "Text GGUF")
    mmproj_path = require_existing_file(args.mmproj, "mmproj GGUF")
    image_path = require_existing_file(args.image, "Sample image")
    mtmd_binary = require_existing_file(find_mtmd_binary(args.llama_cpp_dir, args.mtmd_binary), "llama-mtmd-cli")

    offline_check = OfflineCheck(checked=False, passed=None, host=args.offline_check_host, port=args.offline_check_port)
    if args.require_offline:
        offline_check = check_network_offline(args.offline_check_host, args.offline_check_port, args.offline_check_timeout)
        if offline_check.passed is False:
            raise RuntimeError(
                f"Network appears reachable at {offline_check.host}:{offline_check.port}. "
                "Turn off Wi-Fi / enable airplane mode before the acceptance run."
            )

    command = build_llama_command(args, mtmd_binary, model_path, mmproj_path, image_path)
    if args.dry_run:
        return {
            "dry_run": True,
            "command": command,
            "artifacts": {
                "model_path": str(model_path),
                "mmproj_path": str(mmproj_path),
                "image_path": str(image_path),
            },
        }

    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "NO_PROXY": "*",
        }
    )

    started_at = time.time()
    result = subprocess.run(command, text=True, capture_output=True, timeout=args.timeout_seconds, env=env)
    wall_clock_seconds = time.time() - started_at

    stdout_path = args.output_dir / "llamacpp_stdout.txt"
    stderr_path = args.output_dir / "llamacpp_stderr.txt"
    sample_output_path = args.output_dir / "sample_output.txt"
    report_path = args.output_dir / "offline_llamacpp_report.json"
    model_card_path = args.output_dir / "model_card_metrics.md"

    combined = f"{result.stdout}\n{result.stderr}"
    structured_output = extract_structured_output(combined, args.structured_regex)
    performance_entries = parse_performance_entries(combined)
    tokens_per_second = select_generation_tokens_per_second(performance_entries)
    latency_passed = wall_clock_seconds <= args.latency_threshold_seconds
    structured_output_passed = structured_output is not None
    passed = result.returncode == 0 and latency_passed and structured_output_passed

    write_text(stdout_path, result.stdout)
    write_text(stderr_path, result.stderr)
    write_text(sample_output_path, structured_output or result.stdout.strip())

    payload = {
        "artifacts": {
            "model_path": str(model_path),
            "model_size_bytes": directory_size_bytes(model_path),
            "model_size_gb": format_size_gb(directory_size_bytes(model_path)),
            "mmproj_path": str(mmproj_path),
            "mmproj_size_bytes": directory_size_bytes(mmproj_path),
            "mmproj_size_gb": format_size_gb(directory_size_bytes(mmproj_path)),
            "image_path": str(image_path),
        },
        "runtime": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_only": args.cpu_only,
            "threads": args.threads,
            "context_size": args.context_size,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        },
        "offline_check": asdict(offline_check),
        "result": asdict(
            LlamaCppOfflineResult(
                command=command,
                returncode=result.returncode,
                wall_clock_seconds=wall_clock_seconds,
                latency_threshold_seconds=args.latency_threshold_seconds,
                latency_passed=latency_passed,
                structured_output=structured_output,
                structured_output_passed=structured_output_passed,
                tokens_per_second=tokens_per_second,
                performance_entries=performance_entries,
                stdout_path=str(stdout_path.resolve()),
                stderr_path=str(stderr_path.resolve()),
                sample_output_path=str(sample_output_path.resolve()),
                report_path=str(report_path.resolve()),
                passed=passed,
            )
        ),
    }
    write_json(report_path, payload)
    write_text(model_card_path, render_model_card_snippet(payload))

    if args.fail_on_gate_fail and not passed:
        raise RuntimeError(
            "Offline llama.cpp validation gate failed. "
            f"returncode={result.returncode}, latency_passed={latency_passed}, "
            f"structured_output_passed={structured_output_passed}."
        )
    return payload


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DRI-25 offline llama.cpp GGUF validation.")
    parser.add_argument("--model", type=Path, required=True, help="Path to the quantized text GGUF.")
    parser.add_argument("--mmproj", type=Path, required=True, help="Path to the Qwen2-VL mmproj GGUF.")
    parser.add_argument("--image", type=Path, required=True, help="Local X-ray image used for the offline proof.")
    parser.add_argument("--llama-cpp-dir", type=Path, default=Path("external/llama.cpp"))
    parser.add_argument("--mtmd-binary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--image-flag", default="--image", help="llama-mtmd-cli image argument name.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument("--cpu-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--latency-threshold-seconds", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--structured-regex", default=LABEL_PATTERN)
    parser.add_argument("--require-offline", action="store_true")
    parser.add_argument("--offline-check-host", default="1.1.1.1")
    parser.add_argument("--offline-check-port", type=int, default=53)
    parser.add_argument("--offline-check-timeout", type=float, default=2.0)
    parser.add_argument("--extra-llama-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-gate-fail", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        report = verify_offline_llamacpp(args)
    except Exception as exc:  # noqa: BLE001 - this is a user-facing validation CLI.
        print(f"ERROR: {exc}")
        return 1

    if report.get("dry_run"):
        print("Dry run command:")
        print(" ".join(report["command"]))
        return 0

    result = report["result"]
    print(f"passed={result['passed']}")
    print(f"wall_clock_seconds={result['wall_clock_seconds']:.3f}")
    print(f"tokens_per_second={result['tokens_per_second']}")
    print(f"structured_output_passed={result['structured_output_passed']}")
    print(f"structured_output={result['structured_output']}")
    print(f"report={result['report_path']}")
    print(f"sample_output={result['sample_output_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
