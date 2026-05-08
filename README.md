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
- `build_dataloader.py`: validates Qwen-VL tokenized train/val DataLoader batches with masked labels.
- `train_qlora.py`: configures the AMD-ready QLoRA training run and supports dry-run plus setup-only validation.
- `evaluate_checkpoint.py`: scores checkpoint predictions and writes accuracy, F1, AUC, and confusion matrix metrics.
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

On prebuilt ROCm containers that already include AMD PyTorch, preserve the existing ROCm torch build and install only the missing DRI-10 packages:

```bash
python -m pip install transformers==5.7.0 peft==0.19.1
python -m pip install --no-deps bitsandbytes==0.49.2 optimum==2.1.0 optimum-amd==0.1.0
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
python build_dataloader.py --split train --batch-size 2 --limit 2
python build_dataloader.py --split val --batch-size 2 --limit 2
python train_qlora.py --dry-run --train-limit 2 --eval-limit 2
python train_qlora.py --setup-only --train-limit 2 --eval-limit 2 --wandb-mode disabled --output-dir outputs/dri10-setup-validation
python evaluate_checkpoint.py --adapter-dir outputs/dri12-run1/checkpoint-3300 --data-dir data/processed --split val --output-dir outputs/eval/dri12-run1-checkpoint-3300 --batch-size 3
python setup_wandb.py
```

## DRI-17 Run 2: Classification Recovery

Regenerate the processed JSONL after pulling this branch so training uses the shortened `Classification: <label>` assistant response:

```bash
python generate_jsonl.py --output-dir data/processed
```

Launch the balanced recovery run:

```bash
WANDB_TAGS=dri-17,run2,balanced-sick-recovery \
WANDB_NOTES="Run #2 vs run #1: rank 32, alpha 64, dropout 0.05, lr 1.5e-4, 3 epochs, weight_decay 0.01, balanced sampler, sick_but_non_tb boost 2.0, simplified response format (Classification only), explicit label taxonomy prompt." \
python train_qlora.py \
  --run-name drishti-qlora-run2-balanced-sick-recovery \
  --output-dir outputs/dri17-run2-balanced-sick-recovery \
  --rank 32 \
  --alpha 64 \
  --lora-dropout 0.05 \
  --lr 1.5e-4 \
  --epochs 3 \
  --batch-size 1 \
  --grad-accum 4 \
  --warmup-steps 150 \
  --weight-decay 0.01 \
  --lr-scheduler-type cosine \
  --sampling-strategy balanced \
  --boost-class sick_but_non_tb \
  --boost-multiplier 2.0 \
  --save-steps 200 \
  --eval-steps 200 \
  --logging-steps 10 \
  --seed 42
```

Evaluate the saved adapter:

```bash
python evaluate_checkpoint.py \
  --adapter-dir outputs/dri17-run2-balanced-sick-recovery \
  --data-dir data/processed \
  --split val \
  --output-dir outputs/eval/dri17-run2-balanced-sick-recovery \
  --batch-size 3
```

Run 2 is a real improvement only if val has at least 200 `sick_but_non_tb` predictions, `macro_f1 >= 0.30`, and `accuracy >= 0.50`.

## Local Validation Notes

The initial setup was created on macOS arm64, where ROCm wheels and MI300X GPU access are not available. The scripts include clear failure messages for missing ROCm/CUDA, missing Kaggle credentials, and missing dataset files, then run fully on the ROCm host once those prerequisites are present.
