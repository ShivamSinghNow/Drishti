from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from heatmap_rendering import (
    HeatmapRenderConfig,
    colorize_heatmap,
    normalize_heatmap,
    prepare_heatmap_for_image,
    render_heatmap_overlay,
    render_heatmap_panel,
)


class HeatmapRenderingTests(unittest.TestCase):
    def test_overlay_returns_rgb_uint8_at_original_size(self) -> None:
        image = Image.new("RGB", (8, 6), color=(120, 120, 120))
        heatmap = np.arange(24, dtype=np.float32).reshape(4, 6)

        overlay = render_heatmap_overlay(image, heatmap)

        self.assertEqual(overlay.shape, (6, 8, 3))
        self.assertEqual(overlay.dtype, np.uint8)

    def test_zero_alpha_preserves_readable_source_image(self) -> None:
        image = np.full((6, 8, 3), 128, dtype=np.uint8)
        heatmap = np.arange(48, dtype=np.float32).reshape(6, 8)

        overlay = render_heatmap_overlay(image, heatmap, HeatmapRenderConfig(alpha=0.0))

        np.testing.assert_array_equal(overlay, image)

    def test_invalid_alpha_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "alpha"):
            render_heatmap_overlay(
                Image.new("RGB", (8, 6)),
                np.ones((3, 3), dtype=np.float32),
                HeatmapRenderConfig(alpha=1.5),
            )

    def test_supported_colormaps_produce_different_rgb_maps(self) -> None:
        heatmap = np.linspace(0, 1, 25, dtype=np.float32).reshape(5, 5)

        viridis = colorize_heatmap(heatmap, HeatmapRenderConfig(colormap="viridis"))
        jet = colorize_heatmap(heatmap, HeatmapRenderConfig(colormap="jet"))

        self.assertEqual(viridis.shape, (5, 5, 3))
        self.assertFalse(np.array_equal(viridis, jet))

    def test_percentile_clipping_normalizes_conservative_range(self) -> None:
        heatmap = np.array([0.0, 0.0, 1.0, 2.0, 100.0], dtype=np.float32).reshape(1, 5)

        normalized = normalize_heatmap(
            heatmap,
            HeatmapRenderConfig(clip_low_percentile=20.0, clip_high_percentile=80.0, smoothing_sigma=0.0, gamma=1.0),
        )

        self.assertAlmostEqual(float(normalized.min()), 0.0)
        self.assertAlmostEqual(float(normalized.max()), 1.0)

    def test_panel_adds_side_legend_width(self) -> None:
        image = Image.new("RGB", (8, 6), color=(120, 120, 120))
        heatmap = np.arange(24, dtype=np.float32).reshape(4, 6)
        config = HeatmapRenderConfig(legend_width=12, legend_gap=3, label_width=20)

        panel = render_heatmap_panel(image, heatmap, metadata={}, config=config)

        self.assertEqual(panel.shape, (6, 8 + 3 + 12 + 20, 3))
        self.assertEqual(panel.dtype, np.uint8)

    def test_black_border_suppression_attenuates_non_anatomy_regions(self) -> None:
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        image[:, 8:] = 140
        heatmap = np.tile(np.linspace(0, 1, 16, dtype=np.float32), (16, 1))
        config = HeatmapRenderConfig(
            suppress_borders=True,
            smoothing_sigma=0.0,
            gamma=1.0,
            clip_low_percentile=0.0,
            clip_high_percentile=100.0,
            border_threshold=12,
        )

        _rgb, prepared = prepare_heatmap_for_image(image, heatmap, config)

        self.assertLess(float(prepared[:, :6].mean()), 0.05)
        self.assertGreater(float(prepared[:, 10:].mean()), 0.35)


if __name__ == "__main__":
    unittest.main()
