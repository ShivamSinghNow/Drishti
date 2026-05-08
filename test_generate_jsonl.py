from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from generate_jsonl import jsonl_record, validate_jsonl, write_jsonl
from tbx11k_utils import ImageRecord


class JsonlGenerationTests(unittest.TestCase):
    def test_jsonl_record_uses_absolute_existing_image_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (8, 8)).save(image_path)
            record = ImageRecord(image_path, "train", "healthy")

            payload = jsonl_record(record)

        image_entry = payload["messages"][0]["content"][0]
        self.assertEqual(image_entry["type"], "image")
        self.assertTrue(Path(image_entry["path"]).is_absolute())

    def test_jsonl_record_contains_locked_assistant_response(self) -> None:
        payload = jsonl_record(ImageRecord(Path("/tmp/sample.png"), "train", "active_tb"))

        prompt = payload["messages"][0]["content"][1]["text"]
        self.assertIn("active_tb, healthy, sick_but_non_tb", prompt)
        self.assertEqual(payload["messages"][1]["content"], "Classification: active_tb")

    def test_write_jsonl_creates_one_object_per_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_paths = [root / "healthy.png", root / "active.png"]
            for image_path in image_paths:
                Image.new("RGB", (8, 8)).save(image_path)
            records = [
                ImageRecord(image_paths[0], "train", "healthy"),
                ImageRecord(image_paths[1], "train", "active_tb"),
            ]
            output_path = root / "train.jsonl"

            summary = write_jsonl(records, output_path)
            lines = output_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 2)
        self.assertEqual(summary.total_records, 2)
        self.assertEqual(summary.category_counts["healthy"], 1)
        self.assertEqual(summary.category_counts["active_tb"], 1)
        self.assertTrue(all(json.loads(line)["messages"] for line in lines))

    def test_validate_jsonl_reports_missing_paths_and_category_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            missing_path = root / "missing.png"
            output_path = root / "train.jsonl"
            payload = jsonl_record(ImageRecord(missing_path, "train", "sick_but_non_tb"))
            output_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

            summary = validate_jsonl(output_path)

        self.assertEqual(summary.total_records, 1)
        self.assertEqual(summary.category_counts["sick_but_non_tb"], 1)
        self.assertEqual(summary.missing_paths, (missing_path.resolve(),))
        self.assertEqual(summary.non_absolute_paths, ())


if __name__ == "__main__":
    unittest.main()
