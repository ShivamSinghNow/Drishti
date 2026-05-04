from __future__ import annotations

import os
from pathlib import Path

import torch
import wandb

from tbx11k_utils import DATA_DIR, load_records


PROJECT_NAME = "tbx11k-qwen-vl-finetuning"
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
DATASET_NAME = "vbookshelf/tbx11k-simplified"


def wandb_mode() -> str:
    if os.getenv("WANDB_MODE"):
        return os.environ["WANDB_MODE"]
    if os.getenv("WANDB_API_KEY"):
        return "online"
    netrc = Path.home() / ".netrc"
    if netrc.exists() and "api.wandb.ai" in netrc.read_text(encoding="utf-8", errors="ignore"):
        return "online"
    return "offline"


def main() -> int:
    records, _, _ = load_records(DATA_DIR)
    num_train_images = sum(1 for record in records if record.split == "train")
    num_val_images = sum(1 for record in records if record.split == "val")
    gpu_type = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA/ROCm not available"

    config = {
        "model_name": MODEL_NAME,
        "dataset": DATASET_NAME,
        "num_train_images": num_train_images,
        "num_val_images": num_val_images,
        "gpu_type": gpu_type,
    }

    mode = wandb_mode()
    run = wandb.init(project=PROJECT_NAME, config=config, mode=mode)
    wandb.log({"setup_complete": 1})
    run.finish()

    print(f"W&B project: {PROJECT_NAME}")
    print(f"W&B mode: {mode}")
    print("Logged config:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("Logged dummy metric: setup_complete=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
