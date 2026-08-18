"""Tests for verdict-to-action mapping and the single-character rewrite."""

import unittest

import numpy as np

from ocr_roi_validator.runtime_action import (
    ACCENT_PRESENT, APPLY_E_CORRECTION, BARE_E, KEEP_BASELINE,
    PREMODEL_UNKNOWN, UNKNOWN, action_of, apply_action,
)


class ActionMappingTests(unittest.TestCase):
    def test_only_bare_e_acts(self) -> None:
        actions = action_of(np.array([ACCENT_PRESENT, BARE_E, UNKNOWN,
                                      PREMODEL_UNKNOWN]))
        self.assertEqual(list(actions), [KEEP_BASELINE, APPLY_E_CORRECTION,
                                         KEEP_BASELINE, KEEP_BASELINE])

    def test_accent_and_unknown_are_action_equivalent(self) -> None:
        """The mismatch that survived parity: different verdict, same action."""
        self.assertEqual(action_of(np.array([ACCENT_PRESENT]))[0],
                         action_of(np.array([UNKNOWN]))[0])

    def test_premodel_unknown_never_corrects(self) -> None:
        self.assertEqual(action_of(np.array([PREMODEL_UNKNOWN]))[0],
                         KEEP_BASELINE)


class RewriteTests(unittest.TestCase):
    def test_single_character_is_replaced(self) -> None:
        self.assertEqual(
            apply_action("Véuillez allumer", APPLY_E_CORRECTION, 1),
            "Veuillez allumer")

    def test_keep_baseline_changes_nothing(self) -> None:
        text = "Véuillez allumer"
        self.assertEqual(apply_action(text, KEEP_BASELINE, 1), text)

    def test_only_the_target_position_changes(self) -> None:
        before = "Vérifiez la préssion"
        after = apply_action(before, APPLY_E_CORRECTION, 1)
        differing = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
        self.assertEqual(differing, [1])

    def test_non_accent_target_is_left_alone(self) -> None:
        text = "Veuillez"
        self.assertEqual(apply_action(text, APPLY_E_CORRECTION, 1), text)

    def test_out_of_range_position_is_safe(self) -> None:
        text = "Veuillez"
        for position in (-1, 99):
            self.assertEqual(apply_action(text, APPLY_E_CORRECTION, position),
                             text)

    def test_uppercase_accent_maps_to_uppercase(self) -> None:
        self.assertEqual(apply_action("État", APPLY_E_CORRECTION, 0),
                         "Etat")

    def test_length_is_preserved(self) -> None:
        text = "Véuillez"
        self.assertEqual(len(apply_action(text, APPLY_E_CORRECTION, 1)),
                         len(text))


if __name__ == "__main__":
    unittest.main()
