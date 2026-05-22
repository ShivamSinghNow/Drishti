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
- `train_qlora.py`: configures QLoRA training on CUDA or ROCm and supports dry-run plus setup-only validation.
- `evaluate_checkpoint.py`: scores checkpoint predictions and writes accuracy, F1, AUC, and confusion matrix metrics.
- `merge_lora_checkpoint.py`: merges the selected run #4 LoRA adapter into the Qwen2-VL base model before INT4 quantization.
- `quantize_autogptq_qwen2vl.py`: runs Qwen2-VL AutoGPTQ INT4 quantization with stratified TBX11K calibration samples.
- `evaluate_quantized_checkpoint.py`: evaluates a merged GPTQ model and reports deltas against the run #4 full-precision baseline.
- `export_gguf_llamacpp.py`: exports the selected run #4 checkpoint into llama.cpp text GGUF plus Qwen2-VL mmproj artifacts.
- `verify_llamacpp_offline.py`: runs the DRI-25 local offline llama.cpp proof and saves latency, tokens/sec, sample output, and model-card notes.
- `quantization_utils.py`: shared DRI-23 helpers for calibration selection, locked-format checks, model-size reporting, and metric deltas.
- `generate_gradcam.py`: generates Grad-CAM heatmaps and overlays from the Qwen2-VL vision encoder.
- `heatmap_rendering.py`: renders Grad-CAM maps as readable overlays and standalone demo panels.
- `setup_wandb.py`: initializes the `tbx11k-qwen-vl-finetuning` W&B project and logs a setup metric.
- `tbx11k_utils.py`: shared dataset discovery and annotation parsing helpers.
- `test_tbx11k_utils.py`: regression tests for TBX11K category and split parsing.

## Setup On Colab Pro With CUDA

Use [notebooks/dri18_run3_colab.ipynb](notebooks/dri18_run3_colab.ipynb) for the DRI-18 run #3 workflow. The notebook installs CUDA-compatible dependencies, downloads TBX11K from Kaggle, regenerates JSONL, runs preflight checks, trains in diagnostic segments, evaluates early confusion-matrix gates, and uploads artifacts to Hugging Face.

Use [notebooks/dri19_run4_vision_lora_colab.ipynb](notebooks/dri19_run4_vision_lora_colab.ipynb) for the DRI-19 run #4 vision-LoRA ablation. It reuses the run #3 recipe with seed `42`, adds exact vision-attention LoRA targets, runs a setup-only forward/backward OOM check, and falls back to vision rank `16` / alpha `32` only if rank `32` OOMs.

Use [notebooks/dri23_int4_quantize_colab.ipynb](notebooks/dri23_int4_quantize_colab.ipynb) for DRI-23 INT4 quantization. It installs the Qwen2-VL AutoGPTQ fork, downloads the selected run #4 checkpoint from Hugging Face, merges LoRA into the base model, quantizes with 128 stratified val calibration samples, evaluates the quantized model, checks the `<0.02` macro-F1 drop gate, and uploads GPTQ artifacts.

The Colab path should install the CUDA dependency set:

```bash
python -m pip install -r requirements-colab.txt
```

Colab uses NVIDIA/CUDA. Do not install `optimum-amd` or the ROCm PyTorch wheel in Colab.

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
python -m pip install -r requirements-rocm.txt
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

## DRI-18 Run 3: Colab Diagnostic Recovery

Run #3 should start in Colab with softened sampling instead of the run #2 `2.0` sick boost:

```bash
WANDB_TAGS=dri-18,run3,colab,diagnostic,soft-balanced \
WANDB_NOTES="Run #3 diagnostic: CUDA Colab, softened balanced sampler, no sick boost, segmented early gates." \
python train_qlora.py \
  --run-name drishti-qlora-run3-colab-soft-balanced-diagnostic \
  --output-dir outputs/dri18-run3-colab-soft-balanced-diagnostic \
  --rank 32 \
  --alpha 64 \
  --lora-dropout 0.05 \
  --lr 1.5e-4 \
  --max-steps 800 \
  --batch-size 1 \
  --grad-accum 4 \
  --warmup-steps 150 \
  --weight-decay 0.01 \
  --lr-scheduler-type cosine \
  --sampling-strategy soft-balanced \
  --save-steps 200 \
  --eval-steps 200 \
  --logging-steps 10 \
  --seed 42
```

Evaluate diagnostic checkpoints with the run #3 gate:

```bash
python evaluate_checkpoint.py \
  --adapter-dir outputs/dri18-run3-colab-soft-balanced-diagnostic/checkpoint-800 \
  --data-dir data/processed \
  --split val \
  --output-dir outputs/eval/dri18-run3-colab-soft-balanced-diagnostic/checkpoint-800 \
  --batch-size 3 \
  --limit-per-class 150 \
  --gate run3-diagnostic \
  --fail-on-gate-fail
```

Continue only if no class has zero recall, no class owns more than 80% of predictions, and active TB predictions meet the proportional minimum.

## DRI-19 Run 4: Vision-LoRA Ablation

Run #4 keeps run #3 fixed and changes only the LoRA target scope:

```bash
python train_qlora.py \
  --setup-only \
  --setup-backward-check \
  --train-limit 2 \
  --eval-limit 2 \
  --wandb-mode disabled \
  --lora-target-scope language-vision-attn \
  --rank 32 \
  --alpha 64 \
  --seed 42
```

If that setup check OOMs on Colab, retry with `--vision-rank 16 --vision-alpha 32`. The run #4 notebook evaluates diagnostic checkpoints at steps `400` and `800`; diagnostic class collapse means fewer than `50` predictions for any class on the stratified diagnostic subset.

## Grad-CAM Vision Encoder Overlays

Generate a demo heatmap from a fine-tuned adapter:

```bash
python generate_gradcam.py \
  --adapter-dir outputs/dri19-run4-vision-lora-ablation/checkpoint-4950 \
  --data-dir data/processed \
  --split val \
  --sample-index 0 \
  --target-label predicted \
  --colormap viridis \
  --overlay-alpha 0.38 \
  --output-dir outputs/gradcam/run4-checkpoint-4950
```

## DRI-23 INT4 Quantization

DRI-23 should run on a CUDA Colab runtime. The selected full-precision checkpoint is:

```text
ShivSingh123/drishti-qlora-run4-vision-lora-ablation/checkpoints/checkpoint-4950
```

Merge the run #4 adapter into the base model:

```bash
python merge_lora_checkpoint.py \
  --base-model Qwen/Qwen2-VL-7B-Instruct \
  --adapter-repo-id ShivSingh123/drishti-qlora-run4-vision-lora-ablation \
  --adapter-repo-path checkpoints/checkpoint-4950 \
  --output-dir outputs/dri23-run4-merged-fp16
```

Quantize the merged model:

```bash
python quantize_autogptq_qwen2vl.py \
  --merged-model-dir outputs/dri23-run4-merged-fp16 \
  --data-dir data/processed \
  --split val \
  --calibration-samples 128 \
  --output-dir outputs/dri23-run4-gptq-int4 \
  --bits 4 \
  --group-size 128
```

Evaluate the quantized model against the run #4 baseline:

```bash
python evaluate_quantized_checkpoint.py \
  --model-dir outputs/dri23-run4-gptq-int4 \
  --data-dir data/processed \
  --split val \
  --output-dir outputs/eval/dri23-run4-gptq-int4/full-val \
  --batch-size 3 \
  --gate run3-full \
  --generation-smoke \
  --fail-on-gate-fail
```

Acceptance requires the quantized model-size report to pass the configured `4.0GB` gate, the generation smoke check to return the locked `Classification: <label>` format, and macro-F1 to drop by no more than `0.02` from the run #4 full-precision baseline.

## DRI-24 GGUF Export

Use [notebooks/dri24_gguf_export_colab.ipynb](notebooks/dri24_gguf_export_colab.ipynb) or the CLI to produce llama.cpp artifacts:

```bash
python export_gguf_llamacpp.py \
  --source quantized \
  --output-dir outputs/dri24-gguf \
  --output-prefix drishti-qwen2vl-run4
```

This writes a deployable text GGUF, a separate Qwen2-VL `mmproj` GGUF, and `outputs/dri24-gguf/gguf_export_report.json`.

## DRI-25 Offline llama.cpp Proof

Run [notebooks/dri25_offline_llamacpp_validation.ipynb](notebooks/dri25_offline_llamacpp_validation.ipynb) locally on the laptop. Stage the GGUF, `mmproj`, `llama-mtmd-cli`, and a sample X-ray while online, then turn off Wi-Fi / enable airplane mode for the acceptance cell.

The direct CLI form is:

```bash
python verify_llamacpp_offline.py \
  --model outputs/dri24-gguf/drishti-qwen2vl-run4-q4_k_m.gguf \
  --mmproj outputs/dri24-gguf/mmproj-drishti-qwen2vl-run4-f16.gguf \
  --image outputs/dri25-offline-llamacpp/sample_xray.png \
  --llama-cpp-dir external/llama.cpp \
  --require-offline \
  --cpu-only \
  --latency-threshold-seconds 10 \
  --fail-on-gate-fail
```

The verifier sets offline environment flags, checks that a network socket is unreachable when `--require-offline` is passed, runs `llama-mtmd-cli` with local files only, and saves:

- `offline_llamacpp_report.json`
- `sample_output.txt`
- `llamacpp_stdout.txt`
- `llamacpp_stderr.txt`
- `model_card_metrics.md`

Acceptance passes when the command exits cleanly, returns `Classification: <label>`, records tokens/sec from llama.cpp output, and finishes within the configured CPU latency gate.

The script hooks the selected Qwen2-VL vision transformer block, scores the three locked classification responses, backprops from the selected class log-likelihood, and writes a heatmap PNG, readable overlay PNG, demo panel PNG with side legend, and metadata JSON. It uses the last vision block by default, renders with conservative viridis defaults, suppresses black X-ray borders, and falls back to a gradient-activation map if vanilla Grad-CAM is flat.

## Local Validation Notes

The initial setup was created on macOS arm64, where GPU training is not available. The scripts include clear failure messages for missing CUDA/ROCm, missing Kaggle credentials, and missing dataset files, then run fully on Colab CUDA or the ROCm host once those prerequisites are present.
