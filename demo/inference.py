"""Self-contained Drishti demo inference for Modal / API serving."""

from __future__ import annotations

import copy
import math
import tempfile
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image

PROMPT = (
    "Analyze this chest X-ray image for tuberculosis screening. "
    "Use exactly one of these labels: active_tb, healthy, sick_but_non_tb. "
    "Return only the classification line in the form `Classification: <label>`."
)

CLASS_LABELS = ("active_tb", "healthy", "sick_but_non_tb")
TB_POSITIVE_LABELS = ("active_tb", "latent_tb")
IGNORE_INDEX = -100

MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
ADAPTER_REPO = "ShivSingh123/drishti-qlora-run4-vision-lora-ablation"
ADAPTER_SUBPATH = "checkpoints/checkpoint-4950"
GGUF_REPO = "ShivSingh123/drishti-qwen2vl-run4-gguf"
GGUF_FILE = "drishti-qwen2vl-run4-quantized-q4_k_m.gguf"
IMAGE_SIZE = (512, 512)

LABEL_DISPLAY = {
    "active_tb": "Active TB",
    "healthy": "Healthy",
    "sick_but_non_tb": "Sick (non-TB)",
}

LABEL_GUIDANCE = {
    "active_tb": "Refer to a specialist for confirmatory testing (e.g. sputum smear).",
    "healthy": "No TB indicators detected in this screening pass.",
    "sick_but_non_tb": "Chest abnormality detected, but pattern is not consistent with active TB.",
}


@dataclass(frozen=True)
class EvalSample:
    index: int
    messages: list[dict[str, Any]]
    true_label: str
    image_path: str | None


def assistant_response(category: str) -> str:
    if category not in CLASS_LABELS:
        raise ValueError(f"Unsupported label: {category}")
    return f"Classification: {category}"


def load_rgb_image(source: bytes | Path) -> Image.Image:
    if isinstance(source, bytes):
        with Image.open(BytesIO(source)) as image:
            rgb = image.convert("RGB")
    else:
        with Image.open(source) as image:
            rgb = image.convert("RGB")
    if rgb.size != IMAGE_SIZE:
        rgb = rgb.resize(IMAGE_SIZE, Image.Resampling.BICUBIC)
    return rgb


def build_user_messages(image_path: Path) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": str(image_path.resolve())},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]


def candidate_messages(sample: EvalSample, candidate_label: str) -> list[dict[str, Any]]:
    return [
        copy.deepcopy(sample.messages[0]),
        {"role": "assistant", "content": assistant_response(candidate_label)},
    ]


def softmax_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    max_score = max(scores)
    exp_scores = [math.exp(score - max_score) for score in scores]
    total = sum(exp_scores)
    return [score / total for score in exp_scores]


def mask_prompt_labels(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: list[int] | torch.Tensor,
    padding_side: str = "left",
) -> torch.Tensor:
    labels = input_ids.clone()
    labels[attention_mask == 0] = IGNORE_INDEX

    if isinstance(prompt_lengths, torch.Tensor):
        prompt_lengths = [int(length) for length in prompt_lengths.tolist()]

    sequence_length = input_ids.shape[1]
    actual_lengths = attention_mask.sum(dim=1).tolist()
    for row, prompt_length in enumerate(prompt_lengths):
        actual_length = int(actual_lengths[row])
        if prompt_length > actual_length:
            raise ValueError(f"Prompt length {prompt_length} exceeds sequence length {actual_length}.")
        prompt_start = sequence_length - actual_length if padding_side == "left" else 0
        labels[row, prompt_start : prompt_start + prompt_length] = IGNORE_INDEX
    return labels


def average_label_log_likelihood(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    label_mask = shifted_labels != IGNORE_INDEX
    safe_labels = shifted_labels.masked_fill(~label_mask, 0)

    token_log_probs = torch.log_softmax(shifted_logits, dim=-1)
    token_log_probs = token_log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs.masked_fill(~label_mask, 0.0)

    token_counts = label_mask.sum(dim=1)
    if torch.any(token_counts <= 0):
        raise ValueError("Every candidate response must contain at least one scored label token.")
    return token_log_probs.sum(dim=1) / token_counts


def model_device(model: Any) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_tensors(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def score_sample_batch(
    model: Any,
    processor: Any,
    samples: list[EvalSample],
    candidate_labels: tuple[str, ...] = CLASS_LABELS,
) -> list[dict[str, Any]]:
    conversations = []
    prompt_conversations = []
    for sample in samples:
        for label in candidate_labels:
            messages = candidate_messages(sample, label)
            conversations.append(messages)
            prompt_conversations.append(messages[:1])

    batch = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": True},
    )
    prompt_batch = processor.apply_chat_template(
        prompt_conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": True},
    )
    padding_side = getattr(getattr(processor, "tokenizer", None), "padding_side", "left")
    labels = mask_prompt_labels(
        batch["input_ids"],
        batch["attention_mask"],
        prompt_batch["attention_mask"].sum(dim=1),
        padding_side,
    )

    device = model_device(model)
    batch = move_tensors(batch, device)
    labels = labels.to(device)
    with torch.inference_mode():
        outputs = model(**batch)
        scores = average_label_log_likelihood(outputs.logits, labels).detach().cpu().tolist()

    predictions = []
    width = len(candidate_labels)
    for sample_index, sample in enumerate(samples):
        offset = sample_index * width
        sample_scores = scores[offset : offset + width]
        probabilities = softmax_scores(sample_scores)
        probability_by_label = dict(zip(candidate_labels, probabilities, strict=True))
        predicted_label = max(probability_by_label.items(), key=lambda item: item[1])[0]
        confidence = float(probability_by_label[predicted_label])
        binary_tb_score = sum(probability_by_label.get(label, 0.0) for label in TB_POSITIVE_LABELS)
        predictions.append(
            {
                "predicted_label": predicted_label,
                "confidence": confidence,
                "binary_tb_score": float(binary_tb_score),
                "class_probabilities": {key: float(value) for key, value in probability_by_label.items()},
                "image_path": sample.image_path,
            }
        )
    return predictions


def load_model_and_processor() -> tuple[Any, Any]:
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Drishti demo inference.")

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    snapshot_dir = snapshot_download(
        repo_id=ADAPTER_REPO,
        repo_type="model",
        allow_patterns=[f"{ADAPTER_SUBPATH}/*"],
    )
    adapter_dir = Path(snapshot_dir) / ADAPTER_SUBPATH
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.eval()
    return model, processor


def format_api_response(prediction: dict[str, Any], latency_ms: int) -> dict[str, Any]:
    label = prediction["predicted_label"]
    return {
        "label": label,
        "label_display": LABEL_DISPLAY.get(label, label),
        "confidence": prediction["confidence"],
        "binary_tb_score": prediction["binary_tb_score"],
        "class_probabilities": prediction["class_probabilities"],
        "guidance": LABEL_GUIDANCE.get(label, ""),
        "latency_ms": latency_ms,
        "model": {
            "checkpoint": "drishti run4 vision-lora step 4950",
            "base": MODEL_NAME,
            "adapter": f"{ADAPTER_REPO}/{ADAPTER_SUBPATH}",
            "gguf_artifact": f"{GGUF_REPO} ({GGUF_FILE})",
            "hosted_on": "modal",
        },
        "disclaimer": "Research demo only. Not for clinical diagnosis.",
    }


def analyze_image_bytes(model: Any, processor: Any, image_bytes: bytes) -> dict[str, Any]:
    started = time.perf_counter()
    rgb = load_rgb_image(image_bytes)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        temp_path = Path(handle.name)
        rgb.save(temp_path, format="PNG")

    try:
        sample = EvalSample(
            index=0,
            messages=build_user_messages(temp_path),
            true_label="unknown",
            image_path=str(temp_path),
        )
        prediction = score_sample_batch(model, processor, [sample])[0]
    finally:
        temp_path.unlink(missing_ok=True)

    latency_ms = int((time.perf_counter() - started) * 1000)
    return format_api_response(prediction, latency_ms)
