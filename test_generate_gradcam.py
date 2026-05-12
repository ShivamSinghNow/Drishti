from __future__ import annotations

import types
import unittest

import numpy as np
import torch
from PIL import Image

from generate_gradcam import (
    ActivationGradientCapture,
    compute_token_gradcam,
    find_vision_block,
    heatmap_to_rgb,
    overlay_heatmap,
    resize_heatmap,
    token_cam_to_spatial_map,
)


class FakeQwenWrapper:
    def __init__(self) -> None:
        self.model = types.SimpleNamespace(
            visual=types.SimpleNamespace(
                blocks=torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity()])
            )
        )


class FakePeftWrapper:
    def __init__(self) -> None:
        self._base = FakeQwenWrapper()

    def get_base_model(self):
        return self._base


class GradCamTests(unittest.TestCase):
    def test_find_vision_block_resolves_last_block_through_peft_wrapper(self) -> None:
        name, block = find_vision_block(FakePeftWrapper(), -1)

        self.assertEqual(name, "visual.blocks.2")
        self.assertIsInstance(block, torch.nn.Identity)

    def test_activation_gradient_capture_records_forward_output_and_backward_gradient(self) -> None:
        block = torch.nn.Linear(2, 2, bias=False)
        inputs = torch.ones((3, 2), requires_grad=True)

        with ActivationGradientCapture(block) as capture:
            loss = block(inputs).sum()
            loss.backward()

        self.assertIsNotNone(capture.activations)
        self.assertIsNotNone(capture.gradients)
        self.assertEqual(tuple(capture.activations.shape), (3, 2))
        self.assertEqual(tuple(capture.gradients.shape), (3, 2))

    def test_compute_token_gradcam_uses_gradient_weighted_activations(self) -> None:
        activations = torch.tensor([[2.0, 0.0], [0.0, 1.0], [0.0, 0.5]])
        gradients = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

        token_cam, method = compute_token_gradcam(activations, gradients)

        self.assertEqual(method, "grad_cam")
        self.assertAlmostEqual(float(token_cam.max()), 1.0)
        self.assertGreater(float(token_cam[0]), float(token_cam[-1]))

    def test_compute_token_gradcam_falls_back_when_standard_map_is_flat(self) -> None:
        activations = torch.ones((3, 2))
        gradients = torch.zeros((3, 2))

        token_cam, method = compute_token_gradcam(activations, gradients)

        self.assertEqual(method, "gradient_activation_fallback")
        self.assertTrue(torch.equal(token_cam, torch.zeros_like(token_cam)))

    def test_token_cam_to_spatial_map_averages_temporal_grid(self) -> None:
        token_cam = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        spatial = token_cam_to_spatial_map(token_cam, (2, 2, 2))

        self.assertEqual(tuple(spatial.shape), (2, 2))
        self.assertAlmostEqual(float(spatial.max()), 1.0)
        self.assertAlmostEqual(float(spatial.min()), 0.0)

    def test_heatmap_render_helpers_preserve_image_size(self) -> None:
        heatmap = torch.tensor([[0.0, 1.0], [0.5, 0.25]])
        resized = resize_heatmap(heatmap, (6, 4))
        heatmap_image = heatmap_to_rgb(resized)
        overlay = overlay_heatmap(Image.new("RGB", (6, 4), "white"), resized, alpha=0.4)

        self.assertEqual(resized.shape, (4, 6))
        self.assertEqual(heatmap_image.size, (6, 4))
        self.assertEqual(overlay.size, (6, 4))
        self.assertEqual(np.asarray(overlay).shape, (4, 6, 3))


if __name__ == "__main__":
    unittest.main()
