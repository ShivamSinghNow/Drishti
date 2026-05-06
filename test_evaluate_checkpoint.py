from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluate_checkpoint import (
    CLASS_LABELS,
    EvalSample,
    PredictionRecord,
    candidate_messages,
    compute_metrics,
    extract_true_label,
    load_eval_samples,
    predict_label,
    softmax_scores,
    write_outputs,
)
from generate_jsonl import jsonl_record
from tbx11k_utils import ImageRecord


class CheckpointEvaluationTests(unittest.TestCase):
    def test_extract_true_label_from_locked_assistant_response(self) -> None:
        payload = jsonl_record(ImageRecord(Path("/tmp/sample.png"), "val", "active_tb"))

        self.assertEqual(extract_true_label(payload["messages"]), "active_tb")

    def test_load_eval_samples_reads_labels_and_image_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            payload = jsonl_record(ImageRecord(Path("/tmp/sample.png"), "val", "healthy"))
            (data_dir / "val.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")

            samples = load_eval_samples(data_dir, "val")

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].true_label, "healthy")
        self.assertEqual(samples[0].image_path, str(Path("/tmp/sample.png").resolve()))

    def test_candidate_messages_use_locked_dri6_responses(self) -> None:
        payload = jsonl_record(ImageRecord(Path("/tmp/sample.png"), "val", "healthy"))
        sample = EvalSample(0, payload["messages"], "healthy", "/tmp/sample.png")

        messages = candidate_messages(sample, "sick_but_non_tb")

        self.assertEqual(messages[0], payload["messages"][0])
        self.assertIn("Classification: sick_but_non_tb", messages[1]["content"])
        self.assertIn("Referral recommended: Yes", messages[1]["content"])

    def test_softmax_scores_and_prediction_selection(self) -> None:
        probabilities = dict(zip(CLASS_LABELS, softmax_scores([2.0, 1.0, 0.0]), strict=True))

        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        self.assertEqual(predict_label(probabilities), "active_tb")
        self.assertGreater(probabilities["active_tb"], probabilities["healthy"])

    def test_compute_metrics_includes_f1_auc_and_confusion_matrix(self) -> None:
        predictions = [
            PredictionRecord(0, "/tmp/0.png", "active_tb", "active_tb", 0.90, {"active_tb": 0.90, "healthy": 0.05, "sick_but_non_tb": 0.05}, {}),
            PredictionRecord(1, "/tmp/1.png", "healthy", "healthy", 0.10, {"active_tb": 0.10, "healthy": 0.80, "sick_but_non_tb": 0.10}, {}),
            PredictionRecord(2, "/tmp/2.png", "sick_but_non_tb", "healthy", 0.20, {"active_tb": 0.20, "healthy": 0.50, "sick_but_non_tb": 0.30}, {}),
            PredictionRecord(3, "/tmp/3.png", "active_tb", "active_tb", 0.85, {"active_tb": 0.85, "healthy": 0.10, "sick_but_non_tb": 0.05}, {}),
        ]

        metrics = compute_metrics(predictions)

        self.assertEqual(metrics["total_samples"], 4)
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertIn("macro_f1", metrics)
        self.assertAlmostEqual(metrics["binary_roc_auc"], 1.0)
        self.assertEqual(metrics["confusion_matrix"]["labels"], list(CLASS_LABELS))
        self.assertEqual(metrics["class_distribution"]["active_tb"], 2)
        self.assertEqual(metrics["prediction_distribution"]["healthy"], 2)

    def test_write_outputs_creates_metrics_json_and_predictions_jsonl(self) -> None:
        prediction = PredictionRecord(
            0,
            "/tmp/0.png",
            "healthy",
            "healthy",
            0.05,
            {"active_tb": 0.05, "healthy": 0.90, "sick_but_non_tb": 0.05},
            {"active_tb": -5.0, "healthy": -1.0, "sick_but_non_tb": -4.5},
        )
        metrics = compute_metrics([prediction])
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path, predictions_path = write_outputs(Path(tmpdir), metrics, [prediction])

            saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            saved_predictions = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(saved_metrics["total_samples"], 1)
        self.assertEqual(saved_predictions[0]["predicted_label"], "healthy")


if __name__ == "__main__":
    unittest.main()
