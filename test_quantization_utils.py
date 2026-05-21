from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from quantization_utils import (
    CLASS_LABELS,
    directory_size_bytes,
    format_size_gb,
    is_structured_classification,
    metric_delta_report,
    qwen_vl_messages_for_sample,
    select_calibration_samples,
    structured_classification_label,
)


@dataclass(frozen=True)
class LightweightEvalSample:
    index: int
    messages: list[dict]
    true_label: str
    image_path: str | None


def sample_for_label(index: int, label: str) -> LightweightEvalSample:
    image_path = str(Path(f"/tmp/{label}-{index}.png").resolve())
    return LightweightEvalSample(
        index=index,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": image_path},
                    {"type": "text", "text": "Analyze this chest X-ray image."},
                ],
            },
            {"role": "assistant", "content": f"Classification: {label}"},
        ],
        true_label=label,
        image_path=image_path,
    )


class QuantizationUtilsTests(unittest.TestCase):
    def test_structured_classification_parser_accepts_locked_format_only(self) -> None:
        self.assertEqual(structured_classification_label("Classification: active_tb"), "active_tb")
        self.assertTrue(is_structured_classification("Classification: sick_but_non_tb"))
        self.assertIsNone(structured_classification_label("Classification: latent_tb"))
        self.assertIsNone(structured_classification_label("Finding: active_tb"))
        self.assertIsNone(structured_classification_label("Classification: healthy\nConfidence: high"))

    def test_select_calibration_samples_round_robins_all_classes(self) -> None:
        samples = []
        for label in CLASS_LABELS:
            for index in range(50):
                samples.append(sample_for_label(len(samples), label))

        selected = select_calibration_samples(samples, total=128)

        self.assertEqual(len(selected), 128)
        selected_labels = [sample.true_label for sample in selected]
        for label in CLASS_LABELS:
            self.assertIn(label, selected_labels)
            self.assertGreaterEqual(selected_labels.count(label), 42)

    def test_select_calibration_samples_requires_all_classes(self) -> None:
        samples = [sample_for_label(index, "healthy") for index in range(3)]

        with self.assertRaisesRegex(ValueError, "missing label"):
            select_calibration_samples(samples, total=2)

    def test_qwen_vl_messages_convert_local_image_path_to_file_uri(self) -> None:
        sample = sample_for_label(0, "active_tb")

        messages = qwen_vl_messages_for_sample(sample, include_answer=False)

        image_item = messages[0]["content"][0]
        self.assertEqual(image_item["type"], "image")
        self.assertTrue(image_item["image"].startswith("file://"))
        self.assertNotIn("path", image_item)
        self.assertEqual(len(messages), 1)

    def test_qwen_vl_messages_can_include_answer_for_calibration(self) -> None:
        sample = sample_for_label(0, "healthy")

        messages = qwen_vl_messages_for_sample(sample, include_answer=True)

        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["content"], "Classification: healthy")

    def test_metric_delta_report_enforces_macro_f1_drop_limit(self) -> None:
        metrics = {
            "accuracy": 0.98,
            "macro_f1": 0.965,
            "per_class": {
                "active_tb": {"f1": 0.94},
                "healthy": {"f1": 0.99},
                "sick_but_non_tb": {"f1": 0.98},
            },
        }

        report = metric_delta_report(metrics, max_macro_f1_drop=0.02)

        self.assertTrue(report["passed"])
        self.assertAlmostEqual(report["macro_f1_drop"], 0.013787)
        self.assertIn("active_tb", report["per_class"])

    def test_directory_size_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "a.bin").write_bytes(b"a" * 1024)
            (path / "nested").mkdir()
            (path / "nested" / "b.bin").write_bytes(b"b" * 1024)

            size_bytes = directory_size_bytes(path)

        self.assertEqual(size_bytes, 2048)
        self.assertAlmostEqual(format_size_gb(size_bytes), 2048 / (1024**3))


if __name__ == "__main__":
    unittest.main()
