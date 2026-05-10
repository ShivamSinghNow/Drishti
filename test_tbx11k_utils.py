from __future__ import annotations

import unittest
from pathlib import Path

from tbx11k_utils import category_from_path, canonical_category, canonical_split


class CategoryParsingTests(unittest.TestCase):
    def test_exact_categories_and_aliases(self) -> None:
        self.assertEqual(canonical_category("healthy"), "healthy")
        self.assertEqual(canonical_category("active_tb"), "active_tb")
        self.assertEqual(canonical_category("latent_tb"), "latent_tb")
        self.assertEqual(canonical_category("sick-but-non-tb"), "sick_but_non_tb")

    def test_phrase_aliases_beat_short_tb_alias(self) -> None:
        self.assertEqual(canonical_category("no_tb_case"), "healthy")
        self.assertEqual(canonical_category("latent_tb_case"), "latent_tb")
        self.assertEqual(canonical_category("sick_non_tb_case"), "sick_but_non_tb")

    def test_substrings_inside_words_are_not_labels(self) -> None:
        self.assertIsNone(canonical_category("abnormal"))
        self.assertIsNone(canonical_category("unhealthy"))
        self.assertIsNone(canonical_category("tbx11k"))

    def test_simplified_kaggle_filename_prefixes_are_categories(self) -> None:
        root = Path("data/tbx11k")

        self.assertEqual(category_from_path(root / "tbx11k-simplified/images/h0001.png", root), "healthy")
        self.assertEqual(category_from_path(root / "tbx11k-simplified/images/tb0003.png", root), "active_tb")
        self.assertEqual(category_from_path(root / "tbx11k-simplified/images/s0001.png", root), "sick_but_non_tb")


class SplitParsingTests(unittest.TestCase):
    def test_split_aliases(self) -> None:
        self.assertEqual(canonical_split("train"), "train")
        self.assertEqual(canonical_split("validation"), "val")
        self.assertEqual(canonical_split("valid"), "val")
        self.assertEqual(canonical_split("test"), "test")


if __name__ == "__main__":
    unittest.main()
