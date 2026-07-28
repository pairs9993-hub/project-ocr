from __future__ import annotations

import unittest

from ocr_roi_validator.compare import normalize_ocr_ui_text, normalize_ui_text
from ocr_roi_validator.ocr_engine import _drop_left_gutter_icons
from ocr_roi_validator.scroll_merge import ScrollTextAccumulator


class UIImageTokenTests(unittest.TestCase):
    def test_expected_start_token_is_canonicalized(self) -> None:
        self.assertEqual(normalize_ui_text("{0:img_start}"), "▶Ⅱ")

    def test_icon_only_ocr_artifact_completes_short_expected_text(self) -> None:
        expected = "{0:img_start}"
        actual = normalize_ocr_ui_text("ⅡI", expected)
        accumulator = ScrollTextAccumulator(min_length=2, expected_text=expected)

        self.assertTrue(accumulator.add(actual, 0.67))
        self.assertEqual(accumulator.coverage, 1.0)
        self.assertTrue(accumulator.cycle_complete)
        self.assertEqual(accumulator.final_text, "▶Ⅱ")

    def test_start_key_artifact_requires_explicit_expected_token(self) -> None:
        self.assertEqual(normalize_ocr_ui_text("Il", "Il"), "Il")
        self.assertEqual(normalize_ocr_ui_text("Il", "{0:img_start}"), "▶Ⅱ")

    def test_start_key_artifact_is_repaired_in_instruction_context(self) -> None:
        expected = "Appuyez sur {0:img_start} pour redémarrer."
        actual = "Appuyz sur Il pour redémarrer."
        self.assertEqual(
            normalize_ocr_ui_text(actual, expected),
            "Appuyz sur ▶Ⅱ pour redémarrer.",
        )

    def test_small_left_box_is_preserved_only_for_token_aware_roi(self) -> None:
        icon = (10.0, 10.0, 2.0, 2.0, 18.0, 20.0, "Il", 0.7)
        text = (10.0, 100.0, 50.0, 2.0, 150.0, 20.0, "Start cycle", 0.9)

        self.assertEqual(_drop_left_gutter_icons([icon, text]), [text])
        self.assertEqual(
            _drop_left_gutter_icons([icon, text], preserve_small_left_noise=True),
            [icon, text],
        )


if __name__ == "__main__":
    unittest.main()