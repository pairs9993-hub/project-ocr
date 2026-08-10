"""Unit tests for the French confusable specialist router."""

import inspect
import unicodedata
import unittest

from ocr_roi_validator.fr_specialist_router import (
    ROUTE_ALLOWED_SUBSTITUTION,
    ROUTE_BLOCKED_LENGTH_CHANGE,
    ROUTE_BLOCKED_MULTIPLE_CHANGES,
    ROUTE_BLOCKED_NON_CONFUSABLE_CHANGE,
    ROUTE_IDENTICAL,
    route_specialist_text,
)


class RequiredScenarioTests(unittest.TestCase):
    """The scenarios named explicitly in the specification."""

    def test_veuillez_accent_repair_applies_specialist(self) -> None:
        result = route_specialist_text("Véuillez allumer l'eau.", "Veuillez allumer l'eau.")
        self.assertEqual(result.route, ROUTE_ALLOWED_SUBSTITUTION)
        self.assertTrue(result.specialist_applied)
        self.assertEqual(result.final_text, "Veuillez allumer l'eau.")

    def test_numeric_insertion_keeps_baseline(self) -> None:
        """`1.5` -> `1.s5` is the known specialist regression on 3-142/3-181."""
        result = route_specialist_text("1.5", "1.s5")
        self.assertEqual(result.route, ROUTE_BLOCKED_LENGTH_CHANGE)
        self.assertFalse(result.specialist_applied)
        self.assertEqual(result.final_text, "1.5")

    def test_timer_deletion_keeps_baseline(self) -> None:
        result = route_specialist_text("1 hr 30 min", "1 hr 30 mn")
        self.assertEqual(result.route, ROUTE_BLOCKED_LENGTH_CHANGE)
        self.assertFalse(result.specialist_applied)
        self.assertEqual(result.final_text, "1 hr 30 min")

    def test_i_to_l_single_substitution_is_allowed(self) -> None:
        result = route_specialist_text("disponible", "disponlble")
        self.assertEqual(result.route, ROUTE_ALLOWED_SUBSTITUTION)
        self.assertTrue(result.specialist_applied)
        self.assertEqual(result.final_text, "disponlble")

    def test_accent_fix_plus_numeric_regression_keeps_baseline(self) -> None:
        result = route_specialist_text("Véuillez 1.5", "Veuillez 1.s5")
        self.assertFalse(result.specialist_applied)
        self.assertEqual(result.final_text, "Véuillez 1.5")
        # The inserted `s` changes the length, which is checked first.
        self.assertEqual(result.route, ROUTE_BLOCKED_LENGTH_CHANGE)


class MultipleChangeTests(unittest.TestCase):
    def test_two_substitutions_same_length_keep_baseline(self) -> None:
        result = route_specialist_text("Véuillez disponible", "Veuillez disponlble")
        self.assertEqual(result.route, ROUTE_BLOCKED_MULTIPLE_CHANGES)
        self.assertFalse(result.specialist_applied)
        self.assertEqual(result.final_text, "Véuillez disponible")

    def test_three_substitutions_keep_baseline(self) -> None:
        result = route_specialist_text("eee", "ééé")
        self.assertEqual(result.route, ROUTE_BLOCKED_MULTIPLE_CHANGES)
        self.assertEqual(result.final_text, "eee")


class InsertionDeletionTests(unittest.TestCase):
    def test_insertion_blocked(self) -> None:
        result = route_specialist_text("Lavage", "Lavagee")
        self.assertEqual(result.route, ROUTE_BLOCKED_LENGTH_CHANGE)
        self.assertEqual(result.final_text, "Lavage")

    def test_deletion_blocked(self) -> None:
        result = route_specialist_text("Lavage", "Lavag")
        self.assertEqual(result.route, ROUTE_BLOCKED_LENGTH_CHANGE)
        self.assertEqual(result.final_text, "Lavage")

    def test_insertion_of_allowed_character_still_blocked(self) -> None:
        """Adding an `é` is an insertion, not a substitution."""
        result = route_specialist_text("Veuillez", "Veuilleéz")
        self.assertEqual(result.route, ROUTE_BLOCKED_LENGTH_CHANGE)
        self.assertEqual(result.final_text, "Veuillez")


class ProtectedCharacterClassTests(unittest.TestCase):
    def test_digit_change_blocked(self) -> None:
        result = route_specialist_text("1.5", "1.6")
        self.assertEqual(result.route, ROUTE_BLOCKED_NON_CONFUSABLE_CHANGE)
        self.assertEqual(result.final_text, "1.5")

    def test_digit_to_letter_blocked(self) -> None:
        """`1` -> `l` is visually confusable but must never be routed."""
        result = route_specialist_text("1.5", "l.5")
        self.assertEqual(result.route, ROUTE_BLOCKED_NON_CONFUSABLE_CHANGE)
        self.assertEqual(result.final_text, "1.5")

    def test_whitespace_removal_blocked(self) -> None:
        result = route_specialist_text("1 hr 30 min", "1 hr30 min")
        self.assertFalse(result.specialist_applied)
        self.assertEqual(result.final_text, "1 hr 30 min")

    def test_space_substituted_for_letter_blocked(self) -> None:
        result = route_specialist_text("Lavage en cours", "Lavage_en cours")
        self.assertEqual(result.route, ROUTE_BLOCKED_NON_CONFUSABLE_CHANGE)
        self.assertEqual(result.final_text, "Lavage en cours")

    def test_newline_change_blocked(self) -> None:
        result = route_specialist_text("Dist.\n1.5", "Dist. 1.5")
        self.assertEqual(result.route, ROUTE_BLOCKED_NON_CONFUSABLE_CHANGE)
        self.assertEqual(result.final_text, "Dist.\n1.5")

    def test_punctuation_change_blocked(self) -> None:
        result = route_specialist_text("l'eau.", "l'eau,")
        self.assertEqual(result.route, ROUTE_BLOCKED_NON_CONFUSABLE_CHANGE)
        self.assertEqual(result.final_text, "l'eau.")

    def test_apostrophe_style_change_blocked(self) -> None:
        """A typographic apostrophe swap must not be normalized away."""
        result = route_specialist_text("l'eau.", "l’eau.")
        self.assertEqual(result.route, ROUTE_BLOCKED_NON_CONFUSABLE_CHANGE)
        self.assertEqual(result.final_text, "l'eau.")

    def test_case_change_blocked(self) -> None:
        result = route_specialist_text("Eau", "eau")
        self.assertEqual(result.route, ROUTE_BLOCKED_NON_CONFUSABLE_CHANGE)
        self.assertEqual(result.final_text, "Eau")

    def test_uppercase_accent_not_in_allowed_pairs(self) -> None:
        """ALLOWED_PAIRS is lowercase-only, so `E`/`É` must not route."""
        result = route_specialist_text("Eau", "Éau")
        self.assertEqual(result.route, ROUTE_BLOCKED_NON_CONFUSABLE_CHANGE)
        self.assertEqual(result.final_text, "Eau")

    def test_other_diacritic_blocked(self) -> None:
        """Only e/é is allowed; e/è must keep the baseline."""
        result = route_specialist_text("eau", "èau")
        self.assertEqual(result.route, ROUTE_BLOCKED_NON_CONFUSABLE_CHANGE)
        self.assertEqual(result.final_text, "eau")


class IdentityAndNormalizationTests(unittest.TestCase):
    def test_identical_keeps_baseline(self) -> None:
        result = route_specialist_text("Veuillez allumer l'eau.", "Veuillez allumer l'eau.")
        self.assertEqual(result.route, ROUTE_IDENTICAL)
        self.assertFalse(result.specialist_applied)
        self.assertEqual(result.final_text, "Veuillez allumer l'eau.")

    def test_empty_strings_are_identical(self) -> None:
        result = route_specialist_text("", "")
        self.assertEqual(result.route, ROUTE_IDENTICAL)
        self.assertFalse(result.specialist_applied)

    def test_nfc_decomposed_input_is_treated_as_identical(self) -> None:
        composed = "Véuillez"
        decomposed = unicodedata.normalize("NFD", composed)
        self.assertNotEqual(composed, decomposed)
        result = route_specialist_text(composed, decomposed)
        self.assertEqual(result.route, ROUTE_IDENTICAL)
        self.assertFalse(result.specialist_applied)

    def test_decomposed_specialist_input_routes_after_normalization(self) -> None:
        specialist = unicodedata.normalize("NFD", "léau")
        result = route_specialist_text("leau", specialist)
        self.assertEqual(result.route, ROUTE_ALLOWED_SUBSTITUTION)
        self.assertTrue(result.specialist_applied)
        self.assertEqual(result.final_text, unicodedata.normalize("NFC", "léau"))

    def test_outputs_are_nfc_normalized(self) -> None:
        result = route_specialist_text(unicodedata.normalize("NFD", "Véu"), "Veu")
        self.assertEqual(result.baseline_text, unicodedata.normalize("NFC", "Véu"))
        self.assertEqual(result.route, ROUTE_ALLOWED_SUBSTITUTION)


class ApiSafetyTests(unittest.TestCase):
    def test_router_api_has_no_expected_text_parameter(self) -> None:
        """Expected text must not be reachable from the routing decision."""
        parameters = inspect.signature(route_specialist_text).parameters
        self.assertEqual(list(parameters), ["baseline_text", "specialist_text"])
        for name in parameters:
            self.assertNotIn("expected", name.lower())
            self.assertNotIn("reference", name.lower())

    def test_result_type_is_immutable(self) -> None:
        result = route_specialist_text("1.5", "1.s5")
        with self.assertRaises(Exception):
            result.final_text = "mutated"  # type: ignore[misc]

    def test_result_carries_both_texts_for_diagnostics(self) -> None:
        result = route_specialist_text("Véuillez", "Veuillez")
        self.assertEqual(result.baseline_text, "Véuillez")
        self.assertEqual(result.specialist_text, "Veuillez")


if __name__ == "__main__":
    unittest.main()
