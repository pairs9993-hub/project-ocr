"""Unit tests for the French confusable specialist router."""

import inspect
import unicodedata
import unittest

from ocr_roi_validator.fr_specialist_router import (
    ALLOWED_PAIRS,
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

    def test_accent_fix_plus_numeric_regression_keeps_baseline(self) -> None:
        result = route_specialist_text("Véuillez 1.5", "Veuillez 1.s5")
        self.assertFalse(result.specialist_applied)
        self.assertEqual(result.final_text, "Véuillez 1.5")
        # The inserted `s` changes the length, which is checked first.
        self.assertEqual(result.route, ROUTE_BLOCKED_LENGTH_CHANGE)


class AllowedPairScopeTests(unittest.TestCase):
    """ALLOWED_PAIRS is scoped to the single accent-removal direction."""

    def test_allowed_pairs_is_only_acute_removal(self) -> None:
        self.assertEqual(set(ALLOWED_PAIRS), {("é", "e")})

    def test_accent_addition_keeps_baseline(self) -> None:
        """`e` -> `é` is not allowed: adding an accent is never a repair here."""
        result = route_specialist_text("Veuillez", "Véuillez")
        self.assertEqual(result.route, ROUTE_BLOCKED_NON_CONFUSABLE_CHANGE)
        self.assertFalse(result.specialist_applied)
        self.assertEqual(result.final_text, "Veuillez")

    def test_i_to_l_keeps_baseline(self) -> None:
        """i/l was removed from the allowed set; it must now keep the baseline."""
        result = route_specialist_text("disponible", "disponlble")
        self.assertEqual(result.route, ROUTE_BLOCKED_NON_CONFUSABLE_CHANGE)
        self.assertFalse(result.specialist_applied)
        self.assertEqual(result.final_text, "disponible")

    def test_l_to_i_keeps_baseline(self) -> None:
        result = route_specialist_text("disponlble", "disponible")
        self.assertEqual(result.route, ROUTE_BLOCKED_NON_CONFUSABLE_CHANGE)
        self.assertFalse(result.specialist_applied)
        self.assertEqual(result.final_text, "disponlble")


class MultipleChangeTests(unittest.TestCase):
    def test_two_substitutions_same_length_keep_baseline(self) -> None:
        result = route_specialist_text("Véuillez déteste", "Veuillez deteste")
        self.assertEqual(result.route, ROUTE_BLOCKED_MULTIPLE_CHANGES)
        self.assertFalse(result.specialist_applied)
        self.assertEqual(result.final_text, "Véuillez déteste")

    def test_three_substitutions_keep_baseline(self) -> None:
        result = route_specialist_text("ééé", "eee")
        self.assertEqual(result.route, ROUTE_BLOCKED_MULTIPLE_CHANGES)
        self.assertEqual(result.final_text, "ééé")


class InsertionDeletionTests(unittest.TestCase):
    def test_insertion_blocked(self) -> None:
        result = route_specialist_text("Lavage", "Lavagee")
        self.assertEqual(result.route, ROUTE_BLOCKED_LENGTH_CHANGE)
        self.assertEqual(result.final_text, "Lavage")

    def test_deletion_blocked(self) -> None:
        result = route_specialist_text("Lavage", "Lavag")
        self.assertEqual(result.route, ROUTE_BLOCKED_LENGTH_CHANGE)
        self.assertEqual(result.final_text, "Lavage")

    def test_accent_deletion_by_insertion_still_blocked(self) -> None:
        """Removing `é` by shortening the string is a deletion, not a swap."""
        result = route_specialist_text("Véuillez", "Vuillez")
        self.assertEqual(result.route, ROUTE_BLOCKED_LENGTH_CHANGE)
        self.assertEqual(result.final_text, "Véuillez")


class ProtectedCharacterClassTests(unittest.TestCase):
    def test_digit_change_blocked(self) -> None:
        result = route_specialist_text("1.5", "1.6")
        self.assertEqual(result.route, ROUTE_BLOCKED_NON_CONFUSABLE_CHANGE)
        self.assertEqual(result.final_text, "1.5")

    def test_digit_to_letter_blocked(self) -> None:
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

    def test_uppercase_accent_removal_blocked(self) -> None:
        """ALLOWED_PAIRS is lowercase-only, so `É` -> `E` must not route."""
        result = route_specialist_text("Éau", "Eau")
        self.assertEqual(result.route, ROUTE_BLOCKED_NON_CONFUSABLE_CHANGE)
        self.assertEqual(result.final_text, "Éau")

    def test_other_diacritic_removal_blocked(self) -> None:
        """Only é -> e is allowed; è -> e must keep the baseline."""
        result = route_specialist_text("èau", "eau")
        self.assertEqual(result.route, ROUTE_BLOCKED_NON_CONFUSABLE_CHANGE)
        self.assertEqual(result.final_text, "èau")


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

    def test_nfd_input_is_treated_as_identical(self) -> None:
        """Composed and decomposed `é` are the same text, not a substitution."""
        composed = "Véuillez"
        decomposed = unicodedata.normalize("NFD", composed)
        self.assertNotEqual(composed, decomposed)
        result = route_specialist_text(composed, decomposed)
        self.assertEqual(result.route, ROUTE_IDENTICAL)
        self.assertFalse(result.specialist_applied)


class BaselineCodepointPreservationTests(unittest.TestCase):
    """A declined route must return the baseline unchanged, codepoint for codepoint."""

    def test_nfd_baseline_preserved_when_identical(self) -> None:
        decomposed = unicodedata.normalize("NFD", "Véuillez")
        result = route_specialist_text(decomposed, "Véuillez")
        self.assertEqual(result.route, ROUTE_IDENTICAL)
        self.assertEqual(result.final_text, decomposed)
        self.assertEqual(result.baseline_text, decomposed)
        # Guard against a silent NFC rewrite.
        self.assertNotEqual(result.final_text, unicodedata.normalize("NFC", decomposed))

    def test_nfd_baseline_preserved_when_blocked(self) -> None:
        decomposed = unicodedata.normalize("NFD", "Véuillez 1.5")
        result = route_specialist_text(decomposed, "Veuillez 1.s5")
        self.assertFalse(result.specialist_applied)
        self.assertEqual(result.final_text, decomposed)
        self.assertNotEqual(result.final_text, unicodedata.normalize("NFC", decomposed))

    def test_nfd_specialist_returned_verbatim_when_applied(self) -> None:
        specialist = unicodedata.normalize("NFD", "Veuillez déjà")
        baseline = "Véuillez déjà"
        result = route_specialist_text(baseline, specialist)
        # One accent removed on `Veuillez`; `déjà` is unchanged between them.
        self.assertEqual(result.route, ROUTE_ALLOWED_SUBSTITUTION)
        self.assertTrue(result.specialist_applied)
        self.assertEqual(result.final_text, specialist)

    def test_declined_result_is_identical_object_text(self) -> None:
        for baseline, specialist in (
            ("1.5", "1.s5"),
            ("Veuillez", "Véuillez"),
            ("disponible", "disponlble"),
            ("l'eau.", "l’eau."),
        ):
            with self.subTest(baseline=baseline):
                result = route_specialist_text(baseline, specialist)
                self.assertFalse(result.specialist_applied)
                self.assertEqual(result.final_text, baseline)
                self.assertEqual(result.baseline_text, baseline)
                self.assertEqual(result.specialist_text, specialist)


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
