from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


SUPPORTED_COLORMAPS = ("viridis", "jet", "magma")
_OPENCV_COLORMAPS = {
    "viridis": cv2.COLORMAP_VIRIDIS,
    "jet": cv2.COLORMAP_JET,
    "magma": cv2.COLORMAP_MAGMA,
}


@dataclass(frozen=True)
class HeatmapRenderConfig:
    colormap: str = "viridis"
    alpha: float = 0.38
    clip_low_percentile: float = 55.0
    clip_high_percentile: float = 99.5
    smoothing_sigma: float = 0.85
    gamma: float = 0.8
    suppress_borders: bool = True
    border_threshold: int = 12
    legend: bool = True
    legend_width: int = 72
    legend_gap: int = 16
    label_width: int = 74
    panel_background: tuple[int, int, int] = (9, 10, 12)
    text_color: tuple[int, int, int] = (238, 242, 245)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_render_config(config: HeatmapRenderConfig) -> None:
    if config.colormap not in SUPPORTED_COLORMAPS:
        raise ValueError(f"Unsupported colormap: {config.colormap}. Choose from {SUPPORTED_COLORMAPS}.")
    if not 0.0 <= config.alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    if not 0.0 <= config.clip_low_percentile < config.clip_high_percentile <= 100.0:
        raise ValueError("clip percentiles must satisfy 0 <= low < high <= 100.")
    if config.smoothing_sigma < 0:
        raise ValueError("smoothing_sigma must be >= 0.")
    if config.gamma <= 0:
        raise ValueError("gamma must be > 0.")
    if config.border_threshold < 0 or config.border_threshold > 255:
        raise ValueError("border_threshold must be between 0 and 255.")
    if config.legend_width <= 0 or config.legend_gap < 0 or config.label_width < 0:
        raise ValueError("legend dimensions must be positive.")


def image_to_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"image must be HxW, HxWx3, or HxWx4; got shape {array.shape}.")
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if np.issubdtype(array.dtype, np.floating):
        max_value = 1.0 if float(np.nanmax(array)) <= 1.0 else 255.0
        array = np.clip(array, 0.0, max_value) / max_value * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def heatmap_to_float_array(heatmap: Any) -> np.ndarray:
    if hasattr(heatmap, "detach"):
        heatmap = heatmap.detach().float().cpu().numpy()
    array = np.asarray(heatmap, dtype=np.float32)
    if array.ndim == 3 and 1 in array.shape:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"heatmap must be 2D; got shape {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("heatmap contains NaN or infinite values.")
    return array


def normalize_heatmap(heatmap: np.ndarray, config: HeatmapRenderConfig) -> np.ndarray:
    validate_render_config(config)
    heatmap = heatmap_to_float_array(heatmap)
    low = float(np.percentile(heatmap, config.clip_low_percentile))
    high = float(np.percentile(heatmap, config.clip_high_percentile))
    if high <= low:
        low = float(np.min(heatmap))
        high = float(np.max(heatmap))
    if high <= low:
        return np.zeros_like(heatmap, dtype=np.float32)

    normalized = np.clip((heatmap - low) / (high - low), 0.0, 1.0).astype(np.float32)
    if config.smoothing_sigma > 0:
        normalized = cv2.GaussianBlur(normalized, (0, 0), sigmaX=config.smoothing_sigma, sigmaY=config.smoothing_sigma)
    normalized = np.clip(normalized, 0.0, 1.0)
    if config.gamma != 1.0:
        normalized = np.power(normalized, config.gamma).astype(np.float32)
    return np.clip(normalized, 0.0, 1.0)


def resize_heatmap_to_image(heatmap: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    resized = cv2.resize(heatmap.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def foreground_mask(rgb_image: np.ndarray, config: HeatmapRenderConfig) -> np.ndarray:
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    mask = (gray > config.border_threshold).astype(np.float32)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3.0, sigmaY=3.0)
    return np.clip(mask, 0.0, 1.0)


def prepare_heatmap_for_image(
    image: Image.Image | np.ndarray,
    heatmap: Any,
    config: HeatmapRenderConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    config = config or HeatmapRenderConfig()
    rgb = image_to_rgb_array(image)
    normalized = normalize_heatmap(heatmap_to_float_array(heatmap), config)
    resized = resize_heatmap_to_image(normalized, rgb.shape[:2])
    if config.suppress_borders:
        resized = np.clip(resized * foreground_mask(rgb, config), 0.0, 1.0)
    return rgb, resized.astype(np.float32)


def colorize_heatmap(heatmap: Any, config: HeatmapRenderConfig | None = None) -> np.ndarray:
    config = config or HeatmapRenderConfig()
    validate_render_config(config)
    normalized = heatmap_to_float_array(heatmap)
    normalized = np.clip(normalized, 0.0, 1.0)
    encoded = (normalized * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(encoded, _OPENCV_COLORMAPS[config.colormap])
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def render_heatmap_overlay(
    image: Image.Image | np.ndarray,
    heatmap: Any,
    config: HeatmapRenderConfig | None = None,
) -> np.ndarray:
    config = config or HeatmapRenderConfig()
    rgb, prepared_heatmap = prepare_heatmap_for_image(image, heatmap, config)
    heatmap_rgb = colorize_heatmap(prepared_heatmap, config).astype(np.float32)
    alpha_map = (prepared_heatmap * config.alpha)[:, :, None]
    overlay = (rgb.astype(np.float32) * (1.0 - alpha_map)) + (heatmap_rgb * alpha_map)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def render_heatmap_image(
    image: Image.Image | np.ndarray,
    heatmap: Any,
    config: HeatmapRenderConfig | None = None,
) -> np.ndarray:
    config = config or HeatmapRenderConfig()
    _rgb, prepared_heatmap = prepare_heatmap_for_image(image, heatmap, config)
    return colorize_heatmap(prepared_heatmap, config)


def legend_array(height: int, config: HeatmapRenderConfig | None = None) -> np.ndarray:
    config = config or HeatmapRenderConfig()
    validate_render_config(config)
    gradient = np.linspace(1.0, 0.0, height, dtype=np.float32)[:, None]
    color_strip = colorize_heatmap(np.repeat(gradient, config.legend_width, axis=1), config)
    label_width = config.label_width
    if label_width <= 0:
        return color_strip

    canvas = Image.new(
        "RGB",
        (config.legend_width + label_width, height),
        color=tuple(config.panel_background),
    )
    canvas.paste(Image.fromarray(color_strip), (0, 0))
    draw = ImageDraw.Draw(canvas)
    text_x = config.legend_width + 8
    draw.text((text_x, 8), "High", fill=tuple(config.text_color))
    draw.text((text_x, max(8, height - 22)), "Low", fill=tuple(config.text_color))
    draw.text((text_x, max(28, height // 2 - 8)), "Attention", fill=tuple(config.text_color))
    return np.asarray(canvas, dtype=np.uint8)


def render_heatmap_panel(
    image: Image.Image | np.ndarray,
    heatmap: Any,
    metadata: dict[str, Any] | None = None,
    config: HeatmapRenderConfig | None = None,
) -> np.ndarray:
    config = config or HeatmapRenderConfig()
    overlay = render_heatmap_overlay(image, heatmap, config)
    if not config.legend:
        return overlay

    height, width = overlay.shape[:2]
    legend = legend_array(height, config)
    panel_width = width + config.legend_gap + legend.shape[1]
    panel = np.zeros((height, panel_width, 3), dtype=np.uint8)
    panel[:, :] = np.asarray(config.panel_background, dtype=np.uint8)
    panel[:, :width] = overlay
    panel[:, width + config.legend_gap :] = legend
    return panel
