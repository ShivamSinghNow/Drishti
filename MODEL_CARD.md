---
language:
  - en
library_name: transformers
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen2-VL-7B-Instruct
license: apache-2.0
tags:
  - medical-imaging
  - chest-xray
  - tuberculosis
  - qwen2-vl
  - qlora
  - gptq
  - gguf
  - llama.cpp
---

# Drishti Qwen2-VL TBX11K Classifier

Drishti is a chest X-ray tuberculosis screening research model built for the TBX11K simplified dataset. It fine-tunes Qwen2-VL-7B-Instruct to return one locked classification line:

```text
Classification: <label>
```

Allowed labels:

- `active_tb`
- `healthy`
- `sick_but_non_tb`

This model is a hackathon/research artifact, not a clinical diagnostic system. It should not be used to make medical decisions without clinical validation, physician review, and appropriate regulatory clearance.

## Model Summary

- Base model: `Qwen/Qwen2-VL-7B-Instruct`
- Best checkpoint: run #4, step `4950`
- Training method: QLoRA
- Winning ablation: language LoRA plus Qwen2-VL vision-attention LoRA
- Dataset: TBX11K simplified chest X-rays
- Output contract: single-line `Classification: <label>`
- Deployment targets:
  - Full precision / adapter checkpoint for analysis
  - LLM Compressor GPTQ INT4 checkpoint for compact GPU inference
  - llama.cpp GGUF + Qwen2-VL `mmproj` export for local edge testing

## Intended Use

Drishti is intended for:

- Demonstrating a tuberculosis screening workflow on chest X-rays
- Comparing full-precision, quantized, and GGUF deployment paths
- Producing interpretable Grad-CAM-style heatmaps for demo and model-card review
- Offline proof-of-concept inference experiments with llama.cpp

Drishti is not intended for:

- Clinical diagnosis
- Triage without human oversight
- Deployment on unseen hospital populations without external validation
- Use as a replacement for radiology review, sputum testing, or clinical workup

## Dataset

The training workflow uses the Kaggle `vbookshelf/tbx11k-simplified` mirror of TBX11K.

Processed split used for the run #3/#4 workflow:

| Split | active_tb | healthy | sick_but_non_tb | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 600 | 3000 | 3000 | 6600 |
| Validation | 200 | 800 | 800 | 1800 |

Training examples use a shortened assistant target to avoid wasting loss on predictable boilerplate:

```text
Classification: active_tb
```

The explicit label taxonomy remains in the prompt so the model knows the allowed output space.

## Training Recipe

The selected checkpoint is run #4, a clean ablation against run #3.

Run #3 fixed the class-collapse failure from earlier runs with:

- CUDA Colab Pro training
- Seed `42`
- QLoRA 4-bit NF4 loading
- Rank `32`, alpha `64`, LoRA dropout `0.05`
- Learning rate `1.5e-4`
- Cosine scheduler
- Weight decay `0.01`
- Batch size `1`, gradient accumulation `4`
- Soft-balanced sampling, no class boost
- Three epochs / `4950` optimizer steps

Run #4 kept those settings fixed and changed only the LoRA target scope:

- Language attention LoRA: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Vision-attention LoRA: `visual.blocks.*.attn.qkv`, `visual.blocks.*.attn.proj`
- Patch embedding, vision MLP, and broad `proj` suffix matches were excluded

This ablation produced the best validation metrics and became the selected checkpoint.

## Validation Results

All validation results below are on the 1,800-sample TBX11K validation split unless otherwise stated.

| Model / checkpoint | Accuracy | Macro-F1 | active_tb F1 | healthy F1 | sick_but_non_tb F1 | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Run #3, step 4950 | 0.942222 | 0.910004 | 0.809195 | 0.954399 | 0.966418 | Passed |
| Run #4 vision-LoRA, step 4950 | 0.987222 | 0.978787 | 0.953317 | 0.996881 | 0.986164 | Passed |
| INT4 LLM Compressor quantized | 0.985000 | 0.975194 | 0.945107 | 0.994364 | 0.986111 | Passed |

Quantized model delta vs run #4 full precision:

- Accuracy delta: `-0.002222`
- Macro-F1 delta: `-0.003593`
- Macro-F1 drop gate: passed, max allowed drop `0.02`
- Structured output smoke test: passed

Quantized validation prediction distribution:

| Label | Predictions |
| --- | ---: |
| active_tb | 219 |
| healthy | 797 |
| sick_but_non_tb | 784 |

Quantized per-class metrics:

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| active_tb | 0.904110 | 0.990000 | 0.945107 | 200 |
| healthy | 0.996236 | 0.992500 | 0.994364 | 800 |
| sick_but_non_tb | 0.996173 | 0.976250 | 0.986111 | 800 |

## Confusion Matrix: Run #4 Full Precision

Rows are true labels. Columns are predicted labels.

|  | active_tb | healthy | sick_but_non_tb |
| --- | ---: | ---: | ---: |
| active_tb | 194 | 1 | 5 |
| healthy | 0 | 799 | 1 |
| sick_but_non_tb | 13 | 3 | 784 |

## INT4 Quantized Evaluation Notes

The INT4 checkpoint preserved the locked response format and stayed very close to run #4 full precision. Its macro-F1 drop was `0.003593`, comfortably inside the `0.02` acceptance threshold. The exported evaluation artifact should be kept alongside the model repo so the exact prediction JSONL and confusion matrix are reproducible.

## Interpretability

Drishti includes a Grad-CAM-style vision encoder hook and rendering pipeline:

- Hook layer: final Qwen2-VL vision block, typically `visual.blocks.31`
- Target score: selected classification label likelihood
- Output: raw heatmap, readable overlay, and standalone panel PNG with legend
- Renderer: conservative viridis overlay with percentile clipping, smoothing, gamma control, border suppression, and side legend

Smoke-tested examples from the selected run #4 checkpoint produced correct class predictions for representative `active_tb`, `healthy`, and `sick_but_non_tb` validation samples.

Known interpretability caveat: Grad-CAM on transformer vision encoders is approximate. The heatmap should be presented as model-attention evidence, not as a medical lesion segmentation. Quantitative pointing-game validation against TBX11K bounding boxes is planned separately and should be added to this card once measured.

## Quantization And Edge Export

Two compact deployment formats are tracked:

1. LLM Compressor GPTQ INT4
   - Quantized from the merged run #4 checkpoint
   - Calibration: 128 stratified TBX11K validation samples
   - Acceptance: macro-F1 drop below `0.02`, structured output preserved

2. llama.cpp GGUF
   - Text GGUF exported with llama.cpp conversion tooling
   - Vision `mmproj` exported separately for Qwen2-VL image input
   - Text model quantized to `Q4_K_M`
   - Offline validation harness: `verify_llamacpp_offline.py`

Public GGUF download files:

| File | Purpose | Size |
| --- | --- | ---: |
| `drishti-qwen2vl-run4-quantized-q4_k_m.gguf` | Main text model, Q4_K_M | 4.36 GiB |
| `mmproj-drishti-qwen2vl-run4-quantized-f16.gguf` | Qwen2-VL multimodal projector | 1.26 GiB |
| `drishti-qwen2vl-run4-quantized-f16.gguf` | Intermediate f16 text GGUF, retained for reproducibility | 14.19 GiB |

The GGUF artifact may display a different parameter count than the compressed-tensors INT4 checkpoint because GGUF metadata reports logical architecture parameters, while some Hugging Face quantized displays count packed/compressed tensor storage. Quantization changes storage precision and file size; it does not change the model architecture.

## Offline llama.cpp Proof

DRI-25 adds a local proof harness for airplane-mode inference:

```bash
python verify_llamacpp_offline.py \
  --model outputs/dri24-gguf/drishti-qwen2vl-run4-quantized-q4_k_m.gguf \
  --mmproj outputs/dri24-gguf/mmproj-drishti-qwen2vl-run4-quantized-f16.gguf \
  --image outputs/dri25-offline-llamacpp/sample_xray.png \
  --llama-cpp-dir external/llama.cpp \
  --require-offline \
  --cpu-only \
  --latency-threshold-seconds 10 \
  --fail-on-gate-fail
```

This step saves:

- `offline_llamacpp_report.json`
- `sample_output.txt`
- `llamacpp_stdout.txt`
- `llamacpp_stderr.txt`
- `model_card_metrics.md`

Offline acceptance status:

| Check | Status |
| --- | --- |
| Local GGUF + mmproj staged | Pending final local run |
| Airplane-mode / network-unreachable check | Pending final local run |
| Structured output within 10 seconds on CPU | Pending final local run |
| Tokens/sec logged | Pending final local run |
| Sample output saved | Pending final local run |

After the local run, paste the generated `model_card_metrics.md` section here.

## Example Output

Expected locked format:

```text
Classification: active_tb
```

No additional prose should be emitted by the classifier path.

## Limitations

- The validation split is from the same dataset source and distribution as training; external validation is required.
- The model predicts one of three coarse classes and does not provide a radiology report.
- `sick_but_non_tb` is a broad non-TB abnormal class; it should not be interpreted as a specific differential diagnosis.
- The model has not been clinically calibrated for prevalence, uncertainty, or downstream triage thresholds.
- The model may be sensitive to acquisition differences, patient positioning, markers, cropping, and dataset artifacts.
- Heatmaps are explanatory aids, not ground-truth lesion localization.
- Offline llama.cpp latency depends heavily on laptop CPU, memory bandwidth, quantization type, and llama.cpp build flags.

## Ethical And Safety Notes

This project should be described as a screening research demo. Any real-world medical workflow would require:

- External validation across sites and devices
- Bias and subgroup analysis
- Radiologist review
- Calibration and uncertainty analysis
- Prospective evaluation
- Privacy/security review for image handling
- Regulatory assessment before clinical use

## Reproducibility Pointers

Key scripts and notebooks:

- `notebooks/dri19_run4_vision_lora_colab.ipynb`
- `notebooks/dri23_llmcompressor_int4_quantize_colab.ipynb`
- `notebooks/dri24_gguf_export_colab.ipynb`
- `notebooks/dri25_offline_llamacpp_validation.ipynb`
- `train_qlora.py`
- `evaluate_checkpoint.py`
- `evaluate_quantized_checkpoint.py`
- `generate_gradcam.py`
- `heatmap_rendering.py`
- `merge_lora_checkpoint.py`
- `quantize_llmcompressor_qwen2vl.py`
- `export_gguf_llamacpp.py`
- `verify_llamacpp_offline.py`

Canonical selected full-precision checkpoint:

```text
ShivSingh123/drishti-qlora-run4-vision-lora-ablation/checkpoints/checkpoint-4950
```

Quantized checkpoint:

```text
ShivSingh123/drishti-qwen2vl-run4-llmcompressor-gptq-int4
```

GGUF checkpoint:

```text
ShivSingh123/drishti-qwen2vl-run4-gguf
```

## Citation / Attribution

This model builds on Qwen2-VL-7B-Instruct and the TBX11K chest X-ray dataset. Cite the original base model and dataset sources in any public write-up or demo submission.
