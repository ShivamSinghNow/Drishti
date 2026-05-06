from __future__ import annotations

import json
import types
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from generate_jsonl import jsonl_record
from tbx11k_utils import ImageRecord
from train_qlora import (
    LORA_TARGET_MODULES,
    build_lora_config,
    build_quantization_config,
    build_training_arguments,
    parse_args,
    package_available,
    run_dry_run,
    run_setup_only,
)


class FakeTokenizer:
    padding_side = "left"


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def apply_chat_template(self, conversations, **kwargs):
        if kwargs["add_generation_prompt"]:
            return {
                "input_ids": torch.tensor([[101, 102, 103]] * len(conversations)),
                "attention_mask": torch.tensor([[1, 1, 1]] * len(conversations)),
            }
        return {
            "input_ids": torch.tensor([[101, 102, 103, 201, 202]] * len(conversations)),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]] * len(conversations)),
            "mm_token_type_ids": torch.tensor([[0, 0, 0, 1, 1]] * len(conversations)),
            "pixel_values": torch.zeros((len(conversations), 3, 4, 4)),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * len(conversations)),
        }


class FakeModel:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace(use_cache=True)
        self._frozen = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
        self._trainable = torch.nn.Parameter(torch.ones(3, 2), requires_grad=True)

    def parameters(self):
        return iter((self._frozen, self._trainable))


class FakeTrainer:
    train_called = False

    def __init__(self, *, args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def train(self) -> None:
        FakeTrainer.train_called = True
        raise AssertionError("setup-only must not start training")

    def save_model(self) -> None:
        raise AssertionError("setup-only must not save checkpoints")


def write_split(data_dir: Path, split: str, category: str) -> None:
    payload = jsonl_record(ImageRecord(Path("/tmp/sample.png"), split, category))
    (data_dir / f"{split}.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")


class TrainingScriptTests(unittest.TestCase):
    def test_default_cli_hyperparameters_match_run_one_plan(self) -> None:
        args = parse_args([])

        self.assertEqual(args.lr, 2e-4)
        self.assertEqual(args.rank, 16)
        self.assertEqual(args.alpha, 32)
        self.assertEqual(args.batch_size, 1)
        self.assertEqual(args.grad_accum, 4)
        self.assertEqual(args.epochs, 2.0)
        self.assertEqual(args.warmup_steps, 100)

    def test_quantization_config_is_4bit_nf4_bf16(self) -> None:
        config = build_quantization_config()

        self.assertTrue(config.load_in_4bit)
        self.assertEqual(config.bnb_4bit_quant_type, "nf4")
        self.assertEqual(config.bnb_4bit_compute_dtype, torch.bfloat16)
        self.assertTrue(config.bnb_4bit_use_double_quant)

    def test_lora_config_targets_language_attention_only(self) -> None:
        config = build_lora_config(rank=16, alpha=32, dropout=0.05)

        self.assertEqual(config.r, 16)
        self.assertEqual(config.lora_alpha, 32)
        self.assertEqual(set(config.target_modules), set(LORA_TARGET_MODULES))
        self.assertNotIn("vision", set(config.target_modules))

    def test_training_arguments_include_checkpointing_wandb_and_bf16(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = parse_args(["--output-dir", tmpdir, "--wandb-mode", "offline"])
            training_args = build_training_arguments(args)

        self.assertEqual(training_args.per_device_train_batch_size, 1)
        self.assertEqual(training_args.gradient_accumulation_steps, 4)
        self.assertEqual(training_args.learning_rate, 2e-4)
        self.assertEqual(training_args.save_steps, 100)
        self.assertEqual(training_args.report_to, ["wandb"])
        self.assertTrue(training_args.bf16)
        self.assertEqual(training_args.lr_scheduler_type.value, "cosine")
        self.assertFalse(training_args.remove_unused_columns)

    def test_dry_run_validates_without_loading_full_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            write_split(data_dir, "train", "healthy")
            write_split(data_dir, "val", "active_tb")
            args = parse_args(
                [
                    "--dry-run",
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(data_dir / "outputs"),
                    "--batch-size",
                    "1",
                    "--train-limit",
                    "1",
                    "--eval-limit",
                    "1",
                ]
            )

            with patch("train_qlora.AutoProcessor.from_pretrained", return_value=FakeProcessor()):
                exit_code = run_dry_run(args)

        self.assertEqual(exit_code, 0)

    def test_setup_only_builds_trainer_without_training(self) -> None:
        FakeTrainer.train_called = False
        fake_model = FakeModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            write_split(data_dir, "train", "healthy")
            write_split(data_dir, "val", "active_tb")
            args = parse_args(
                [
                    "--setup-only",
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(data_dir / "outputs"),
                    "--batch-size",
                    "1",
                    "--train-limit",
                    "1",
                    "--eval-limit",
                    "1",
                ]
            )

            with (
                patch("train_qlora.require_full_training_environment"),
                patch("train_qlora.AutoProcessor.from_pretrained", return_value=FakeProcessor()),
                patch("train_qlora.Qwen2VLForConditionalGeneration.from_pretrained", return_value=fake_model),
                patch("train_qlora.prepare_model_for_kbit_training", return_value=fake_model) as prepare,
                patch("train_qlora.get_peft_model", return_value=fake_model) as get_peft,
                patch("train_qlora.Trainer", FakeTrainer),
            ):
                exit_code = run_setup_only(args)

        self.assertEqual(exit_code, 0)
        self.assertFalse(FakeTrainer.train_called)
        self.assertFalse(fake_model.config.use_cache)
        prepare.assert_called_once_with(fake_model)
        get_peft.assert_called_once()

    def test_setup_only_defaults_to_disabled_wandb_and_tiny_limits(self) -> None:
        args = parse_args(["--setup-only"])

        self.assertEqual(args.train_limit, 2)
        self.assertEqual(args.eval_limit, 2)
        self.assertEqual(args.wandb_mode, "disabled")

    def test_package_available_handles_missing_package(self) -> None:
        self.assertFalse(package_available("definitely-not-installed-drishti-package"))


if __name__ == "__main__":
    unittest.main()
