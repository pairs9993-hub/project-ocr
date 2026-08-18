"""Seal the natural-support challenge: an equal event quota per word.

The concentration breach came from giving every word the same *exposure*, which
hands the role's event count to whichever word triggers most often -- treb
supplied 58% of preflight's events that way. Scaling the others' exposure by
1.38x would place the point estimate exactly on the 50% cap with no room for
sampling variation, so it is not used.

Instead each word contributes an equal number of *accepted events*: the first
20 clean hallucinations found in its own deterministic stream, and no more.
With every word capped at 20, one word cannot exceed 50% of a role unless the
role has fewer than two words, which the role minimum already forbids. The
concentration property becomes structural rather than something to be hit by
tuning exposure.

Each word gets its own hard cap, sized so that 20 events arrive with
probability 0.95 under a conservative lower bound on its measured rate. A word
that fails its cap fails the recipe; its shortfall is never covered by another
word's surplus, because that is exactly the substitution the equal-quota design
exists to prevent.

This seals the recipe and the budget. It generates nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
for extra in (VALIDATOR_ROOT, VALIDATOR_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from ocr_roi_validator.diagnostic_runner import atomic_write_json  # noqa: E402
from size_v2_budget import binomial_at_least, renderings_for_quota  # noqa: E402

ACCEPTED_EVENTS_PER_WORD = 20
CHALLENGE_SEED = 60100000
ROLE_RENAMES = {
    "coverage_calibration_candidate": "coverage_diagnostic_candidate",
    "preflight_candidate": "natural_support_preflight_candidate",
    "support_holdout_candidate": "screened_support_holdout_candidate",
}


def wilson_lower(successes: int, trials: int, z: float = 1.6449) -> float:
    """One-sided 95% lower bound on a proportion."""
    if trials == 0 or successes == 0:
        return 0.0
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    spread = z * math.sqrt(p * (1 - p) / trials
                           + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - spread)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--erratum", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    roles_report = json.loads(args.roles.read_text(encoding="utf-8"))
    erratum = json.loads(args.erratum.read_text(encoding="utf-8"))
    composite = roles_report["composite_registry"]
    membership = erratum["canonical_membership"]

    per_word = {}
    infeasible = []
    for role, words in membership.items():
        for word in words:
            entry = composite[word]
            events = entry["clean_hallucination"]
            denominator = entry["clean_bare_denominator"]
            # The rate is per clean-comparable occurrence, but a rendering only
            # sometimes reaches a clean comparison, so the cap is expressed in
            # renderings using the observed reach.
            reach = denominator / entry["renderings"] if entry["renderings"] else 0.0
            lower = wilson_lower(events, denominator)
            per_rendering_lower = lower * reach
            cap = (renderings_for_quota(per_rendering_lower,
                                        ACCEPTED_EVENTS_PER_WORD)
                   if per_rendering_lower > 0 else None)
            if cap is None:
                infeasible.append(word)
            per_word[word] = {
                "role": role,
                "observed_events": events,
                "observed_clean_denominator": denominator,
                "observed_renderings": entry["renderings"],
                "clean_reach": round(reach, 6),
                "rate_per_clean_lower95": round(lower, 8),
                "rate_per_rendering_lower95": round(per_rendering_lower, 8),
                "accepted_event_quota": ACCEPTED_EVENTS_PER_WORD,
                "hard_cap_renderings": cap,
                "success_probability_at_cap": (
                    round(binomial_at_least(cap, per_rendering_lower,
                                            ACCEPTED_EVENTS_PER_WORD), 6)
                    if cap else None),
            }

    feasible = not infeasible
    total = (sum(v["hard_cap_renderings"] for v in per_word.values())
             if feasible else None)

    # With every word capped at the same accepted count, a role's largest share
    # is 1/len(words) once complete -- structural, not tuned.
    projected = {}
    for role, words in membership.items():
        count = len(words)
        projected[role] = {
            "words": count,
            "accepted_events_when_complete": count * ACCEPTED_EVENTS_PER_WORD,
            "largest_word_share_when_complete": round(1.0 / count, 4),
            "within_50pct_cap": (1.0 / count) <= 0.50,
        }

    recipe = {
        "dataset": "natural_support_challenge_v1",
        "status": "SEALED_NOT_GENERATED",
        "role_source_erratum_sha256": hashlib.sha256(
            args.erratum.read_bytes()).hexdigest(),
        "canonical_membership_sha256": erratum["canonical_membership_sha256"],
        "membership": membership,
        "role_renames_applied": ROLE_RENAMES,
        "seed": CHALLENGE_SEED,
        "accepted_events_per_word": ACCEPTED_EVENTS_PER_WORD,
        "acceptance_rule": (
            "the first 20 clean hallucinations in each word's own deterministic "
            "stream are accepted; later ones are recorded but not accepted"),
        "why_not_exposure_scaling": (
            "scaling the other words' exposure by 1.38x puts the point estimate "
            "exactly on the 50% cap with no margin for sampling variation. An "
            "equal accepted-event quota makes the share structural: with n "
            "words each capped at 20, the largest share is 1/n"),
        "substitution_prohibited": (
            "a word short of its quota is never covered by another word's "
            "surplus; the whole recipe reports QUOTA_NOT_MET"),
        "per_word": per_word,
        "projected_concentration": projected,
        "total_hard_cap_renderings": total,
        "infeasible_words": infeasible,
        "prevalence_claim_prohibited": (
            "exposure here is chosen to collect a fixed event count, so no "
            "natural prevalence may be estimated from this data"),
        "usage_constraints": {
            "coverage_diagnostic_candidate":
                "correction coverage measurement only; must not drive weights, "
                "threshold, guard, crop or preprocessing",
            "natural_support_preflight_candidate":
                "frozen-model pre-gate only",
            "screened_support_holdout_candidate":
                "NOT a final holdout; screened during discovery",
            "threshold_source":
                "counterfactual calibration and legitimate-accent preservation "
                "data only",
        },
        "STATUS": "READY" if feasible else "INFEASIBLE",
    }
    digest = atomic_write_json(args.out, recipe)

    print(f"{'word':12s} {'role':36s} {'ev':>4s} {'rate/rend':>11s} {'cap':>9s} {'P':>7s}")
    for word, entry in sorted(per_word.items(),
                              key=lambda item: item[1]["hard_cap_renderings"] or 0):
        cap = entry["hard_cap_renderings"]
        print(f"{word:12s} {entry['role']:36s} {entry['observed_events']:4d} "
              f"{entry['rate_per_rendering_lower95']:11.6f} "
              f"{(f'{cap:,}' if cap else 'INFEAS'):>9s} "
              f"{(entry['success_probability_at_cap'] or 0):7.4f}")
    print("\nprojected concentration when complete:")
    for role, entry in projected.items():
        print(f"  {role:38s} {entry['words']} words, "
              f"{entry['accepted_events_when_complete']} events, "
              f"largest share {entry['largest_word_share_when_complete']:.1%} "
              f"-> {'ok' if entry['within_50pct_cap'] else 'EXCEEDS'}")
    if total:
        print(f"\ntotal hard cap {total:,} renderings "
              f"({total * 0.1784 / 3600:.1f}h)")
    print(f"STATUS {recipe['STATUS']}")
    print(f"recipe sha256 {digest}")
    return 0 if feasible else 1


if __name__ == "__main__":
    raise SystemExit(main())
