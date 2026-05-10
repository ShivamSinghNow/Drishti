from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from preprocess_samples import (
    IMAGE_SIZE,
    load_rgb_image,
    preprocess_record,
    simplified_split_records,
    training_messages,
    validate_preprocessing,
)
from tbx11k_utils import ImageRecord


class FakeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class FakeProcessor:
    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(self, conversation, **kwargs):
        self.calls.append((conversation, kwargs))
        return {
            "input_ids": FakeTensor((1, 8)),
            "attention_mask": FakeTensor((1, 8)),
            "pixel_values": FakeTensor((1, 3, 512, 512)),
        }


class PreprocessingTests(unittest.TestCase):
    def test_load_rgb_image_converts_grayscale_and_resizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "gray.png"
            Image.new("L", (64, 80), color=128).save(image_path)

            image = load_rgb_image(image_path)

        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, IMAGE_SIZE)

    def test_training_messages_include_locked_assistant_response(self) -> None:
        messages = training_messages(Path("/tmp/sample.png"), "active_tb")

        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"][0]["type"], "image")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["content"], "Classification: active_tb")

    def test_preprocess_record_calls_processor_with_resized_rgb_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("L", (32, 32), color=64).save(image_path)
            processor = FakeProcessor()
            record = ImageRecord(image_path, "train", "healthy")

            sample = preprocess_record(record, processor)

        conversation, kwargs = processor.calls[0]
        processor_image = conversation[0]["content"][0]["image"]
        self.assertEqual(processor_image.mode, "RGB")
        self.assertEqual(processor_image.size, IMAGE_SIZE)
        self.assertTrue(kwargs["tokenize"])
        self.assertFalse(kwargs["add_generation_prompt"])
        self.assertEqual(kwargs["return_tensors"], "pt")
        self.assertEqual(sample.processor_output_keys, ["input_ids", "attention_mask", "pixel_values"])
        self.assertEqual(sample.processor_output_shapes["pixel_values"], (1, 3, 512, 512))

    def test_validate_preprocessing_limits_and_balances_categories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            records = []
            for category in ("healthy", "active_tb", "sick_but_non_tb"):
                for index in range(3):
                    image_path = root / f"{category}_{index}.png"
                    Image.new("RGB", (16, 16), color=(index, index, index)).save(image_path)
                    records.append(ImageRecord(image_path, "train", category))

            samples = validate_preprocessing(records, FakeProcessor(), limit=5)

        self.assertEqual(len(samples), 5)
        self.assertEqual(
            [sample.category for sample in samples[:3]],
            ["active_tb", "healthy", "sick_but_non_tb"],
        )

    def test_simplified_split_records_use_filename_prefixes_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_root = root / "tbx11k-simplified" / "images"
            image_root.mkdir(parents=True)
            for name in ("tb0001.png", "tb0002.png", "h0001.png", "h0002.png", "s0001.png", "s0002.png"):
                Image.new("RGB", (16, 16), color=(1, 1, 1)).save(image_root / name)

            split_counts = {
                "train": {"active_tb": 1, "healthy": 1, "sick_but_non_tb": 1},
                "val": {"active_tb": 1, "healthy": 1, "sick_but_non_tb": 1},
            }

            train_records = simplified_split_records("train", root, split_counts)
            val_records = simplified_split_records("val", root, split_counts)

        self.assertEqual([record.path.name for record in train_records], ["tb0001.png", "h0001.png", "s0001.png"])
        self.assertEqual([record.path.name for record in val_records], ["tb0002.png", "h0002.png", "s0002.png"])
        self.assertEqual([record.category for record in train_records], ["active_tb", "healthy", "sick_but_non_tb"])


if __name__ == "__main__":
    unittest.main()
