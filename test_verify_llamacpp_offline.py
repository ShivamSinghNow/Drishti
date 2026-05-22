from __future__ import annotations

import unittest

from verify_llamacpp_offline import (
    extract_structured_output,
    parse_performance_entries,
    select_generation_tokens_per_second,
)


class VerifyLlamaCppOfflineTests(unittest.TestCase):
    def test_extract_structured_output_from_noisy_llamacpp_output(self) -> None:
        output = """
        llama_model_loader: loaded meta data
        Analyze this chest X-ray.
        Classification: active_tb
        llama_perf_context_print: total time = 1200.00 ms
        """

        self.assertEqual(extract_structured_output(output), "Classification: active_tb")

    def test_extract_structured_output_rejects_unknown_label(self) -> None:
        output = "Classification: maybe_tb"

        self.assertIsNone(extract_structured_output(output))

    def test_parse_labeled_tokens_per_second_entries(self) -> None:
        output = """
        llama_perf_context_print: prompt eval time = 1234.56 ms / 16 tokens (77.16 ms per token, 12.96 tokens per second)
        llama_perf_context_print:        eval time = 2345.67 ms / 8 runs   (293.21 ms per token, 3.41 tokens per second)
        """

        entries = parse_performance_entries(output)

        self.assertEqual(
            entries,
            [
                {"label": "prompt eval time", "tokens_per_second": 12.96},
                {"label": "eval time", "tokens_per_second": 3.41},
            ],
        )

    def test_parse_unlabeled_tokens_per_second_fallback(self) -> None:
        output = "timing: 9.25 tokens per second"

        self.assertEqual(parse_performance_entries(output), [{"label": "unlabeled", "tokens_per_second": 9.25}])

    def test_select_generation_tokens_per_second_prefers_eval_time(self) -> None:
        entries = [
            {"label": "prompt eval time", "tokens_per_second": 12.96},
            {"label": "eval time", "tokens_per_second": 3.41},
            {"label": "total time", "tokens_per_second": 4.25},
        ]

        self.assertEqual(select_generation_tokens_per_second(entries), 3.41)


if __name__ == "__main__":
    unittest.main()
