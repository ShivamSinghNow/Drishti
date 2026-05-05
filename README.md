# Drishti

Drishti is a medical AI hackathon setup for tuberculosis screening experiments with TBX11K chest X-rays and Qwen2-VL-7B-Instruct. The project includes environment checks, Kaggle dataset download and exploration, Qwen2-VL inference verification, sample prompt formatting, and Weights & Biases setup.

This repository contains setup utilities only. It does not include the TBX11K dataset, W&B logs, virtual environments, or model weights.

## Files

- `verify_gpu.py`: verifies PyTorch ROCm/CUDA availability, GPU name, total memory, and ROCm version.
- `download_dataset.py`: downloads `vbookshelf/tbx11k-simplified` from Kaggle into `data/tbx11k`.
- `explore_dataset.py`: reports split counts, class distribution, image shapes, pixel ranges, and verifies 512x512 image size.
- `verify_model.py`: loads `Qwen/Qwen2-VL-7B-Instruct`, runs one chest X-ray inference, and reports peak GPU memory.
- `format_samples.py`: writes three Qwen2-VL-style fine-tuning examples to `formatted_samples.json`.
- `preprocess_samples.py`: validates the RGB resize and Qwen-VL processor path on a small TBX11K batch.
- `generate_jsonl.py`: writes local Qwen-VL conversation JSONL files for the TBX11K train and val splits.
- `setup_wandb.py`: initializes the `tbx11k-qwen-vl-finetuning` W&B project and logs a setup metric.
- `tbx11k_utils.py`: shared dataset discovery and annotation parsing helpers.
- `test_tbx11k_utils.py`: regression tests for TBX11K category and split parsing.

## Setup On AMD MI300X With ROCm

Create and activate a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch for ROCm first. Pick the ROCm index URL that matches the host:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
```

Then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Configure Kaggle credentials before downloading the dataset:

```bash
mkdir -p ~/.kaggle
# place kaggle.json in ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

For W&B online logging, run `wandb login` or set `WANDB_API_KEY`. Without credentials, `setup_wandb.py` uses offline mode.

## Run Order

```bash
python -m unittest -v
python verify_gpu.py
python download_dataset.py
python explore_dataset.py
python verify_model.py
python format_samples.py
python preprocess_samples.py --split train --limit 10
python generate_jsonl.py --output-dir data/processed
python setup_wandb.py
```

## Local Validation Notes

The initial setup was created on macOS arm64, where ROCm wheels and MI300X GPU access are not available. The scripts include clear failure messages for missing ROCm/CUDA, missing Kaggle credentials, and missing dataset files, then run fully on the ROCm host once those prerequisites are present.
