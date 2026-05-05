from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from build_dataloader import (
    IGNORE_INDEX,
    QwenVlDataCollator,
    load_jsonl_dataset,
    mask_prompt_labels,
    validate_batch,
)
from generate_jsonl import jsonl_record
from tbx11k_utils import ImageRecord


class FakeTokenizer:
    padding_side = "left"


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(self, conversations, **kwargs):
        self.calls.append((conversations, kwargs))
        if kwargs["add_generation_prompt"]:
            return {
                "input_ids": torch.tensor([[101, 102, 103], [101, 102, 103]]),
                "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
            }
        return {
            "input_ids": torch.tensor([[101, 102, 103, 201, 202], [101, 102, 103, 203, 204]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
            "mm_token_type_ids": torch.tensor([[0, 0, 0, 1, 1], [0, 0, 0, 1, 1]]),
            "pixel_values": torch.zeros((2, 3, 4, 4)),
            "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
        }


def sample_messages(category: str = "healthy") -> list[dict]:
    return jsonl_record(ImageRecord(Path("/tmp/sample.png"), "train", category))["messages"]


class LabelMaskingTests(unittest.TestCase):
    def test_mask_prompt_labels_handles_left_padding(self) -> None:
        input_ids = torch.tensor(
            [
                [0, 10, 11, 12, 13],
                [20, 21, 22, 23, 24],
            ]
        )
        attention_mask = torch.tensor(
            [
                [0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]
        )

        labels = mask_prompt_labels(input_ids, attention_mask, [2, 3], padding_side="left")

        self.assertEqual(labels.tolist()[0], [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 12, 13])
        self.assertEqual(labels.tolist()[1], [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 23, 24])

    def test_collator_returns_required_keys_and_masks_prompt(self) -> None:
        collator = QwenVlDataCollator(FakeProcessor())
        batch = collator(
            [
                {"messages": sample_messages("healthy")},
                {"messages": sample_messages("active_tb")},
            ]
        )

        self.assertIn("labels", batch)
        self.assertEqual(batch["labels"].shape, batch["input_ids"].shape)
        self.assertEqual(batch["labels"][:, :3].tolist(), [[IGNORE_INDEX] * 3, [IGNORE_INDEX] * 3])
        self.assertEqual(batch["labels"][:, 3:].tolist(), [[201, 202], [203, 204]])

    def test_validate_batch_rejects_missing_label_tokens(self) -> None:
        batch = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "mm_token_type_ids": torch.tensor([[0, 0]]),
            "pixel_values": torch.zeros((1, 3, 4, 4)),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "labels": torch.tensor([[IGNORE_INDEX, IGNORE_INDEX]]),
        }

        with self.assertRaises(ValueError):
            validate_batch("train", batch)


class DatasetLoadingTests(unittest.TestCase):
    def test_load_jsonl_dataset_from_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            payload = {"messages": sample_messages("sick_but_non_tb")}
            (data_dir / "train.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")

            dataset = load_jsonl_dataset("train", data_dir)

        self.assertEqual(len(dataset), 1)
        self.assertEqual(json.loads(dataset[0]["messages_json"])[1]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
