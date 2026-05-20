from __future__ import annotations

import types
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from evaluate_checkpoint import EvalSample, PredictionRecord
from generate_gradcam import (
    ActivationGradientCapture,
    compute_token_gradcam,
    find_vision_block,
    generate_gradcam,
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


class FakeGradCamModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(use_cache=True)
        self.block = torch.nn.Linear(2, 2, bias=False)

    def forward_hook_score(self) -> torch.Tensor:
        inputs = torch.ones((4, 2), requires_grad=True)
        return self.block(inputs).sum()


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

    def test_generate_gradcam_writes_heatmap_overlay_panel_and_metadata(self) -> None:
        fake_model = FakeGradCamModel()
        prediction = PredictionRecord(
            0,
            "/tmp/xray.png",
            "active_tb",
            "active_tb",
            0.90,
            {"active_tb": 0.90, "healthy": 0.05, "sick_but_non_tb": 0.05},
            {"active_tb": -0.10, "healthy": -4.0, "sick_but_non_tb": -3.5},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "xray.png"
            Image.new("RGB", (12, 10), color=(120, 120, 120)).save(image_path)
            sample = EvalSample(7, [], "active_tb", str(image_path))

            def score_with_hook(_model, _processor, _sample, _target_label):
                return fake_model.forward_hook_score(), (1, 2, 2)

            with (
                patch("generate_gradcam.score_sample_batch", return_value=[prediction]),
                patch("generate_gradcam.find_vision_block", return_value=("visual.blocks.31", fake_model.block)),
                patch("generate_gradcam.score_target_label_with_grad", side_effect=score_with_hook),
            ):
                metadata = generate_gradcam(fake_model, object(), sample, tmp_path / "gradcam")

            self.assertTrue(Path(metadata.output_heatmap).exists())
            self.assertTrue(Path(metadata.output_overlay).exists())
            self.assertTrue(Path(metadata.output_panel).exists())
            self.assertTrue(Path(metadata.output_metadata).exists())
            self.assertIn("colormap", metadata.render_config)
            self.assertEqual(np.asarray(Image.open(metadata.output_panel)).shape[2], 3)


if __name__ == "__main__":
    unittest.main()
