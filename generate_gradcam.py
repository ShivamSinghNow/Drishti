from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from build_dataloader import DEFAULT_DATA_DIR, mask_prompt_labels
from evaluate_checkpoint import (
    CLASS_LABELS,
    EvalSample,
    average_label_log_likelihood,
    candidate_messages,
    load_eval_samples,
    load_model_and_processor,
    model_device,
    move_tensors,
    score_sample_batch,
)
from preprocess_samples import MODEL_NAME


DEFAULT_OUTPUT_DIR = Path("outputs/gradcam")
TARGET_LABEL_CHOICES = ("predicted",) + CLASS_LABELS


@dataclass(frozen=True)
class GradCamMetadata:
    sample_index: int
    image_path: str
    true_label: str
    predicted_label: str
    target_label: str
    target_score: float
    class_probabilities: dict[str, float]
    class_log_likelihoods: dict[str, float]
    hook_layer_name: str
    hook_layer_index: int
    vision_grid_thw: tuple[int, int, int]
    token_count: int
    spatial_shape: tuple[int, int]
    output_heatmap: str
    output_overlay: str
    output_metadata: str
    cam_method: str


class ActivationGradientCapture:
    def __init__(self, module: torch.nn.Module) -> None:
        self.module = module
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def __enter__(self) -> "ActivationGradientCapture":
        self._handle = self.module.register_forward_hook(self._capture)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            self._handle.remove()

    def _capture(self, _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        activation = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(activation):
            raise RuntimeError("Grad-CAM hook expected the vision block to return a tensor.")
        if not activation.requires_grad:
            raise RuntimeError("Vision activations do not require gradients. Ensure pixel_values requires grad.")
        self.activations = activation
        activation.retain_grad()
        activation.register_hook(self._save_gradient)

    def _save_gradient(self, gradient: torch.Tensor) -> None:
        self.gradients = gradient


def unwrap_base_model(model: Any) -> Any:
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def find_vision_transformer(model: Any) -> Any:
    base_model = unwrap_base_model(model)
    if hasattr(base_model, "model") and hasattr(base_model.model, "visual"):
        return base_model.model.visual
    if hasattr(base_model, "visual"):
        return base_model.visual
    raise ValueError("Could not locate Qwen2-VL vision transformer at model.visual or model.model.visual.")


def find_vision_block(model: Any, layer_index: int) -> tuple[str, torch.nn.Module]:
    vision = find_vision_transformer(model)
    blocks = getattr(vision, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise ValueError("Qwen2-VL vision transformer does not expose non-empty blocks.")
    resolved_index = layer_index if layer_index >= 0 else len(blocks) + layer_index
    if resolved_index < 0 or resolved_index >= len(blocks):
        raise ValueError(f"Vision layer index {layer_index} resolves outside 0..{len(blocks) - 1}.")
    return f"visual.blocks.{resolved_index}", blocks[resolved_index]


def build_single_candidate_batch(processor: Any, sample: EvalSample, target_label: str) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    messages = candidate_messages(sample, target_label)
    batch = processor.apply_chat_template(
        [messages],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": True},
    )
    prompt_batch = processor.apply_chat_template(
        [messages[:1]],
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
    return batch, labels


def score_target_label_with_grad(
    model: Any,
    processor: Any,
    sample: EvalSample,
    target_label: str,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    batch, labels = build_single_candidate_batch(processor, sample, target_label)
    device = model_device(model)
    batch = move_tensors(batch, device)
    labels = labels.to(device)

    if "pixel_values" not in batch:
        raise ValueError("Grad-CAM requires image pixel_values in the processor batch.")
    batch["pixel_values"] = batch["pixel_values"].detach().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    outputs = model(**batch)
    scores = average_label_log_likelihood(outputs.logits, labels)
    if scores.numel() != 1:
        raise RuntimeError(f"Expected one candidate score, got {scores.numel()}.")

    grid = batch["image_grid_thw"][0].detach().cpu().tolist()
    return scores[0], (int(grid[0]), int(grid[1]), int(grid[2]))


def compute_token_gradcam(
    activations: torch.Tensor,
    gradients: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, str]:
    if activations.shape != gradients.shape:
        raise ValueError(f"Activation and gradient shapes must match, got {activations.shape} and {gradients.shape}.")
    if activations.ndim != 2:
        raise ValueError(f"Expected vision block activations shaped (tokens, channels), got {tuple(activations.shape)}.")

    activations = activations.float()
    gradients = gradients.float()
    weights = gradients.mean(dim=0)
    cam = torch.relu((activations * weights).sum(dim=-1))
    method = "grad_cam"
    if float(cam.max().detach().cpu()) <= eps:
        cam = (activations * gradients).sum(dim=-1).abs()
        method = "gradient_activation_fallback"
    return normalize_tensor(cam, eps), method


def normalize_tensor(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    values = values - values.min()
    max_value = values.max()
    if float(max_value.detach().cpu()) <= eps:
        return torch.zeros_like(values)
    return values / max_value


def token_cam_to_spatial_map(token_cam: torch.Tensor, grid_thw: tuple[int, int, int]) -> torch.Tensor:
    grid_t, grid_h, grid_w = grid_thw
    expected_tokens = grid_t * grid_h * grid_w
    if token_cam.numel() != expected_tokens:
        raise ValueError(f"Grad-CAM token count {token_cam.numel()} does not match image grid {grid_thw}.")
    spatial = token_cam.reshape(grid_t, grid_h, grid_w).mean(dim=0)
    return normalize_tensor(spatial)


def resize_heatmap(heatmap: torch.Tensor, image_size: tuple[int, int]) -> np.ndarray:
    tensor = heatmap.detach().float().cpu().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=(image_size[1], image_size[0]), mode="bilinear", align_corners=False)
    return resized.squeeze().clamp(0, 1).numpy()


def heatmap_to_rgb(heatmap: np.ndarray) -> Image.Image:
    heatmap = np.clip(heatmap, 0.0, 1.0)
    red = (255 * heatmap).astype(np.uint8)
    green = (255 * np.sqrt(heatmap)).astype(np.uint8)
    blue = (80 * (1.0 - heatmap)).astype(np.uint8)
    return Image.fromarray(np.stack([red, green, blue], axis=-1), mode="RGB")


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float) -> Image.Image:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("--overlay-alpha must be between 0 and 1.")
    heatmap_image = heatmap_to_rgb(heatmap).resize(image.size)
    return Image.blend(image.convert("RGB"), heatmap_image, alpha=alpha)


def generate_gradcam(
    model: Any,
    processor: Any,
    sample: EvalSample,
    output_dir: Path,
    target_label: str = "predicted",
    layer_index: int = -1,
    overlay_alpha: float = 0.45,
) -> GradCamMetadata:
    if sample.image_path is None:
        raise ValueError("Selected sample has no image path.")

    image_path = Path(sample.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    prediction = score_sample_batch(model, processor, [sample])[0]
    resolved_target_label = prediction.predicted_label if target_label == "predicted" else target_label
    if resolved_target_label not in CLASS_LABELS:
        raise ValueError(f"Unsupported target label: {resolved_target_label}")

    hook_layer_name, hook_layer = find_vision_block(model, layer_index)
    hook_layer_index = int(hook_layer_name.rsplit(".", maxsplit=1)[-1])
    with ActivationGradientCapture(hook_layer) as capture:
        target_score, grid_thw = score_target_label_with_grad(model, processor, sample, resolved_target_label)
        target_score.backward()

    if capture.activations is None or capture.gradients is None:
        raise RuntimeError("Grad-CAM hook did not capture activations and gradients.")

    token_cam, cam_method = compute_token_gradcam(
        capture.activations.detach(),
        capture.gradients.detach(),
    )
    spatial_map = token_cam_to_spatial_map(token_cam, grid_thw)

    original_image = Image.open(image_path).convert("RGB")
    resized_heatmap = resize_heatmap(spatial_map, original_image.size)
    heatmap_image = heatmap_to_rgb(resized_heatmap)
    overlay_image = overlay_heatmap(original_image, resized_heatmap, overlay_alpha)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{sample.index:04d}_{resolved_target_label}"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    overlay_path = output_dir / f"{stem}_overlay.png"
    metadata_path = output_dir / f"{stem}_metadata.json"
    heatmap_image.save(heatmap_path)
    overlay_image.save(overlay_path)

    metadata = GradCamMetadata(
        sample_index=sample.index,
        image_path=str(image_path.resolve()),
        true_label=sample.true_label,
        predicted_label=prediction.predicted_label,
        target_label=resolved_target_label,
        target_score=float(target_score.detach().cpu()),
        class_probabilities=prediction.class_probabilities,
        class_log_likelihoods=prediction.class_log_likelihoods,
        hook_layer_name=hook_layer_name,
        hook_layer_index=hook_layer_index,
        vision_grid_thw=grid_thw,
        token_count=int(token_cam.numel()),
        spatial_shape=(int(spatial_map.shape[0]), int(spatial_map.shape[1])),
        output_heatmap=str(heatmap_path),
        output_overlay=str(overlay_path),
        output_metadata=str(metadata_path),
        cam_method=cam_method,
    )
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM overlays from the Qwen2-VL vision encoder.")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Base Qwen2-VL model name.")
    parser.add_argument("--adapter-dir", default=None, type=Path, help="Optional LoRA adapter checkpoint directory.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, type=Path, help="Directory containing split JSONL files.")
    parser.add_argument("--split", default="val", choices=("train", "val"), help="Dataset split to sample from.")
    parser.add_argument("--sample-index", default=0, type=int, help="Zero-based sample index in the selected split.")
    parser.add_argument("--target-label", default="predicted", choices=TARGET_LABEL_CHOICES, help="Class to explain.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path, help="Directory for heatmap, overlay, and metadata.")
    parser.add_argument("--layer-index", default=-1, type=int, help="Vision block index to hook; -1 means last block.")
    parser.add_argument("--overlay-alpha", default=0.45, type=float, help="Heatmap opacity over the source image.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        samples = load_eval_samples(args.data_dir, args.split)
        if args.sample_index < 0 or args.sample_index >= len(samples):
            raise ValueError(f"--sample-index must be between 0 and {len(samples) - 1}.")
        model, processor = load_model_and_processor(args.model_name, args.adapter_dir)
        metadata = generate_gradcam(
            model,
            processor,
            samples[args.sample_index],
            args.output_dir,
            target_label=args.target_label,
            layer_index=args.layer_index,
            overlay_alpha=args.overlay_alpha,
        )
    except (ImportError, RuntimeError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"sample_index={metadata.sample_index}")
    print(f"true_label={metadata.true_label}")
    print(f"predicted_label={metadata.predicted_label}")
    print(f"target_label={metadata.target_label}")
    print(f"cam_method={metadata.cam_method}")
    print(f"hook_layer={metadata.hook_layer_name}")
    print(f"heatmap={metadata.output_heatmap}")
    print(f"overlay={metadata.output_overlay}")
    print(f"metadata={metadata.output_metadata}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
