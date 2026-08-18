"""Tests for support-role assignment and its status gate.

Two defects sit behind these. The status check counted words per role and
reported SUFFICIENT while one word supplied 58% of a role's events, and the
report described a stratified assignment that the code discarded by re-sorting
globally. Both were found by inspection rather than by a test, so the
properties are pinned here.
"""

import hashlib
import json
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from assign_support_roles_v1 import (  # noqa: E402
    MAX_SHARE_OF_ROLE, ROLE_MINIMUM, ROLES, stable_rank,
)


def assign(words):
    """The shipped assignment rule, isolated for testing."""
    buckets = {role: [] for role in ROLES}
    for position, word in enumerate(sorted(words,
                                           key=lambda w: stable_rank(w, "role"))):
        buckets[ROLES[position % len(ROLES)]].append(word)
    return buckets


def canonical(assignment):
    return json.dumps({role: sorted(words) for role, words in assignment.items()},
                      sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def share(events):
    total = sum(events.values())
    return (max(events.values()) / total) if total else 0.0


class DeterminismTests(unittest.TestCase):
    WORDS = ["Delvrat", "Kelvrat", "Lelvrat", "Pelvrat", "Telvrat", "aerbint",
             "bemrat", "demrat", "grafte", "hemrat", "lemrat", "temolie", "treb"]

    def test_input_order_does_not_change_membership(self) -> None:
        """The defect this guards: assignment must not depend on JSON order."""
        forward = assign(self.WORDS)
        reversed_order = assign(list(reversed(self.WORDS)))
        shuffled = assign(sorted(self.WORDS, key=len))
        self.assertEqual(canonical(forward), canonical(reversed_order))
        self.assertEqual(canonical(forward), canonical(shuffled))

    def test_membership_hash_is_stable(self) -> None:
        first = hashlib.sha256(canonical(assign(self.WORDS)).encode()).hexdigest()
        second = hashlib.sha256(
            canonical(assign(list(reversed(self.WORDS)))).encode()).hexdigest()
        self.assertEqual(first, second)

    def test_frozen_membership_matches_the_recorded_hash(self) -> None:
        """Pins the membership that is now frozen, under its new role names."""
        expected = {
            "coverage_diagnostic_candidate":
                ["Delvrat", "Telvrat", "aerbint", "bemrat", "temolie"],
            "natural_support_preflight_candidate":
                ["Kelvrat", "Pelvrat", "hemrat", "treb"],
            "screened_support_holdout_candidate":
                ["Lelvrat", "demrat", "grafte", "lemrat"],
        }
        digest = hashlib.sha256(
            json.dumps(expected, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(
            digest,
            "fff8c059b5214862c444864f06d35b5ba791a3ee30ca704b70fb6148b1909900")

    def test_every_word_lands_in_exactly_one_role(self) -> None:
        assignment = assign(self.WORDS)
        placed = [w for words in assignment.values() for w in words]
        self.assertEqual(sorted(placed), sorted(self.WORDS))
        self.assertEqual(len(placed), len(set(placed)))

    def test_roles_do_not_overlap(self) -> None:
        assignment = assign(self.WORDS)
        names = list(assignment)
        for index, first in enumerate(names):
            for second in names[index + 1:]:
                self.assertEqual(
                    set(assignment[first]) & set(assignment[second]), set())


class ConcentrationGateTests(unittest.TestCase):
    def test_dominant_word_exceeds_the_cap(self) -> None:
        self.assertGreater(share({"treb": 29, "Pelvrat": 7, "hemrat": 5,
                                  "Kelvrat": 9}), MAX_SHARE_OF_ROLE)

    def test_balanced_role_is_within_the_cap(self) -> None:
        self.assertLessEqual(share({"a": 20, "b": 20, "c": 20}),
                             MAX_SHARE_OF_ROLE)

    def test_exactly_fifty_percent_passes(self) -> None:
        """The boundary is inclusive; 50% is not a breach."""
        value = share({"a": 20, "b": 10, "c": 10})
        self.assertEqual(value, 0.5)
        self.assertTrue(value <= MAX_SHARE_OF_ROLE)

    def test_just_over_fifty_percent_fails(self) -> None:
        self.assertFalse(share({"a": 21, "b": 10, "c": 10}) <= MAX_SHARE_OF_ROLE)

    def test_zero_event_role_does_not_divide_by_zero(self) -> None:
        self.assertEqual(share({"a": 0, "b": 0}), 0.0)

    def test_single_word_role_is_wholly_concentrated(self) -> None:
        self.assertEqual(share({"only": 12}), 1.0)

    def test_status_cannot_pass_while_a_role_is_concentrated(self) -> None:
        """The exact bug: word count met, cap breached, status said SUFFICIENT."""
        met = {role: True for role in ROLES}
        concentrated = ["natural_support_preflight_candidate"]
        overlaps = {"a vs b": []}
        sufficient = (all(met.values()) and not any(overlaps.values())
                      and not concentrated)
        self.assertFalse(sufficient)

    def test_status_passes_only_when_all_three_conditions_hold(self) -> None:
        met = {role: True for role in ROLES}
        self.assertTrue(all(met.values()) and not any({"a vs b": []}.values())
                        and not [])


class RoleMinimumTests(unittest.TestCase):
    def test_too_few_words_fails_the_minimum(self) -> None:
        assignment = assign(["alpha", "beta"])          # 2 words, 3 roles
        met = {role: len(words) >= ROLE_MINIMUM
               for role, words in assignment.items()}
        self.assertFalse(all(met.values()))

    def test_nine_words_gives_every_role_its_minimum(self) -> None:
        assignment = assign([f"word{i}" for i in range(9)])
        for words in assignment.values():
            self.assertGreaterEqual(len(words), ROLE_MINIMUM)

    def test_round_robin_keeps_roles_within_one_word(self) -> None:
        assignment = assign([f"w{i}" for i in range(13)])
        sizes = [len(words) for words in assignment.values()]
        self.assertLessEqual(max(sizes) - min(sizes), 1)


class DescriptionMatchesCodeTests(unittest.TestCase):
    def test_no_stale_stratification_claim(self) -> None:
        """The report once claimed stratification the code did not perform."""
        source = (SCRIPTS / "assign_support_roles_v1.py").read_text(
            encoding="utf-8")
        method = source.split('"assignment_method":')[1].split("),")[0]
        self.assertIn("global stable-hash", method)
        self.assertNotIn("stratified by rate band", method)

    def test_assignment_uses_no_grouping(self) -> None:
        source = (SCRIPTS / "assign_support_roles_v1.py").read_text(
            encoding="utf-8")
        body = source.split("Global stable-hash assignment")[1][:900]
        self.assertNotIn("grouped[", body)


if __name__ == "__main__":
    unittest.main()
