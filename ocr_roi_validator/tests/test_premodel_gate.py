"""Tests for the pre-model fail-closed path.

The property that matters is negative: on a malformed query the model must not
run, no correction may be applied, and the baseline string must survive
untouched. A gate that returned UNKNOWN but still let a correction through
would be worse than no gate, so the mock counts invocations rather than
trusting the code path.
"""

import inspect
import unittest

from ocr_roi_validator.premodel_gate import (
    PREMODEL_REASONS,
    check_premodel,
)


class SpyModel:
    """Stands in for the verifier; records whether it was ever called."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return "BARE_E"


def run(ordinal, decoded_length, baseline="Il reglage lla", input_built=True):
    """Apply the gate, then the model only if the gate allows it."""
    model = SpyModel()
    verdict = check_premodel(ordinal, decoded_length, input_built)
    text = baseline
    if not verdict.rejected:
        prediction = model(ordinal, decoded_length)
        if prediction == "BARE_E" and verdict.correction_allowed:
            text = baseline.replace("e", "é", 1)
    return verdict, model.calls, text


class RejectionTests(unittest.TestCase):
    def test_negative_ordinal_rejects_without_inference(self) -> None:
        verdict, calls, text = run(-1, 8)
        self.assertEqual(verdict.verdict, "UNKNOWN")
        self.assertEqual(verdict.reason, "ORDINAL_NEGATIVE")
        self.assertEqual(calls, 0)
        self.assertEqual(text, "Il reglage lla")

    def test_ordinal_at_or_past_length_rejects(self) -> None:
        for ordinal in (8, 9, 100):
            verdict, calls, _ = run(ordinal, 8)
            self.assertEqual(verdict.reason, "ORDINAL_OUT_OF_RANGE")
            self.assertEqual(calls, 0)

    def test_non_positive_decoded_length_rejects(self) -> None:
        for length in (0, -1):
            verdict, calls, _ = run(0, length)
            self.assertEqual(verdict.reason, "DECODED_LENGTH_NOT_POSITIVE")
            self.assertEqual(calls, 0)

    def test_failed_input_build_rejects(self) -> None:
        verdict, calls, _ = run(3, 8, input_built=False)
        self.assertEqual(verdict.reason, "INPUT_BUILD_FAILED")
        self.assertEqual(calls, 0)

    def test_none_inputs_reject(self) -> None:
        self.assertTrue(check_premodel(None, 8).rejected)
        self.assertTrue(check_premodel(3, None).rejected)

    def test_rejection_forbids_correction(self) -> None:
        for ordinal, length in ((-1, 8), (9, 8), (0, 0)):
            verdict, _, text = run(ordinal, length)
            self.assertFalse(verdict.correction_allowed)
            self.assertEqual(text, "Il reglage lla")

    def test_rejection_never_invokes_the_network(self) -> None:
        for ordinal, length in ((-5, 8), (8, 8), (12, 3), (0, 0)):
            self.assertFalse(check_premodel(ordinal, length).network_invoked)

    def test_every_reason_is_declared(self) -> None:
        seen = {check_premodel(-1, 8).reason, check_premodel(9, 8).reason,
                check_premodel(0, 0).reason,
                check_premodel(3, 8, input_built=False).reason}
        self.assertEqual(seen, set(PREMODEL_REASONS))


class PassThroughTests(unittest.TestCase):
    def test_valid_query_proceeds_to_the_model(self) -> None:
        verdict, calls, text = run(3, 8)
        self.assertEqual(verdict.verdict, "PROCEED")
        self.assertIsNone(verdict.reason)
        self.assertEqual(calls, 1)
        self.assertNotEqual(text, "Il reglage lla")

    def test_boundary_ordinal_is_allowed(self) -> None:
        """The last valid position is length - 1, and it must pass."""
        verdict, calls, _ = run(7, 8)
        self.assertEqual(verdict.verdict, "PROCEED")
        self.assertEqual(calls, 1)

    def test_first_position_is_allowed(self) -> None:
        self.assertFalse(check_premodel(0, 1).rejected)


class ContractTests(unittest.TestCase):
    def test_signature_takes_no_expected_text(self) -> None:
        """No Expected argument can reach this gate."""
        parameters = inspect.signature(check_premodel).parameters
        for name in parameters:
            for banned in ("expected", "text", "decoded_text", "truth", "word"):
                self.assertNotIn(banned, name.lower())

    def test_signature_is_positions_only(self) -> None:
        self.assertEqual(list(inspect.signature(check_premodel).parameters),
                         ["target_ordinal", "decoded_length", "input_built"])

    def test_ordinal_shifted_inside_range_is_not_rejected(self) -> None:
        """A wrong-but-in-range ordinal needs the model; the gate must not
        pre-empt it, or the trainable UNKNOWN case would never reach training."""
        self.assertFalse(check_premodel(5, 8).rejected)
        self.assertFalse(check_premodel(0, 8).rejected)


if __name__ == "__main__":
    unittest.main()
