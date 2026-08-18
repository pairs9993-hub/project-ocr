"""Reclassify word support on the composite and assign the three roles.

The extension gave each of the sixteen sparse words 1,600 further renderings,
so support is recomputed on all 2,000 rather than on the original 400. That
matters: Telvrat produced one event in discovery and sixteen in the extension,
which means its earlier SPARSE label reflected exposure, not the word.

Roles replace the abandoned 12/8/8 word gate. Natural hallucination words are
no longer the training set -- counterfactual pairs are -- so these words carry
narrower jobs: measuring correction coverage, gating a frozen model, and a
held-back set. The last is called support-holdout rather than final holdout
precisely because its behaviour has already been observed here; a genuine
untouched pool is reserved separately and left unscreened.

Assignment is a global stable hash over the spelling, dealt round-robin.
Nothing is chosen by looking at the numbers. It is deliberately not stratified:
an earlier revision built a stratified pool, then re-sorted it globally and
threw the stratification away while still describing itself as stratified. The
membership that produced is frozen and is not redrawn here, because correcting
a description must not become a licence to reassign roles after seeing results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
for extra in (VALIDATOR_ROOT, VALIDATOR_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from discover_word_support_v1 import ROBUST_MIN_BREADTH, ROBUST_MIN_EVENTS  # noqa: E402
from ocr_roi_validator.diagnostic_runner import atomic_write_json  # noqa: E402

CLEAN_BARE = {"CLEAN_CORRECT_BARE_E", "CLEAN_HALLUCINATION"}
ROLES = ("coverage_calibration_candidate", "preflight_candidate",
         "support_holdout_candidate")
ROLE_MINIMUM = 3
MAX_SHARE_OF_ROLE = 0.50


def wilson(successes: int, trials: int, z: float = 1.96):
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    spread = z * math.sqrt(p * (1 - p) / trials
                           + z * z / (4 * trials * trials)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def upper_bound_zero(trials: int, confidence: float = 0.95) -> float:
    return 1.0 - (1.0 - confidence) ** (1.0 / trials) if trials else 1.0


def stable_rank(word: str, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}|{word}".encode()).hexdigest()[:16], 16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base = json.loads(args.registry.read_text(encoding="utf-8"))
    registry = base["registry"]
    extension_rows = [json.loads(line) for line
                      in (args.extension / "checkpoint.jsonl").read_text(
                          encoding="utf-8").splitlines() if line.strip()]

    added: dict[str, dict] = defaultdict(
        lambda: {"renderings": 0, "clean_bare": 0, "events": 0,
                 "preservation": 0, "fonts": set(), "strata": set()})
    for row in extension_rows:
        entry = added[row["word"]]
        entry["renderings"] += 1
        if row["visual_target"] == "e" and row["terminal_reason"] in CLEAN_BARE:
            entry["clean_bare"] += 1
        if row["clean_hallucination"]:
            entry["events"] += 1
            entry["fonts"].add(row["font"])
            if row.get("measured_stratum"):
                entry["strata"].add(row["measured_stratum"])
        if row["clean_preservation"]:
            entry["preservation"] += 1

    composite: dict[str, dict] = {}
    promoted, still_sparse = [], []
    for word, entry in registry.items():
        extra = added.get(word)
        events = entry["clean_hallucination"] + (extra["events"] if extra else 0)
        denominator = entry["clean_bare_denominator"] + (
            extra["clean_bare"] if extra else 0)
        fonts = set(entry["fonts_with_events"]) | (
            extra["fonts"] if extra else set())
        strata = set(entry["strata_with_events"]) | (
            extra["strata"] if extra else set())
        support = ("ROBUST_SUPPORT"
                   if events >= ROBUST_MIN_EVENTS
                   and (len(fonts) >= ROBUST_MIN_BREADTH
                        or len(strata) >= ROBUST_MIN_BREADTH)
                   else "SPARSE_SUPPORT" if events >= 1 else "NOT_OBSERVED")
        composite[word] = {
            "group": entry["group"],
            "preceding_class": entry["preceding_class"],
            "following_class": entry["following_class"],
            "word_length": entry["word_length"],
            "renderings": entry["renderings"] + (extra["renderings"] if extra else 0),
            "clean_bare_denominator": denominator,
            "clean_hallucination": events,
            "clean_preservation": entry["clean_preservation"] + (
                extra["preservation"] if extra else 0),
            "fonts_with_events": sorted(fonts),
            "strata_with_events": sorted(strata),
            "support_rate": events / denominator if denominator else None,
            "support_rate_ci95": wilson(events, denominator),
            "zero_event_upper_bound_95": (upper_bound_zero(denominator)
                                          if events == 0 and denominator else None),
            "support": support,
            "extended": bool(extra),
            "support_before_extension": entry["support"],
        }
        if extra and entry["support"] == "SPARSE_SUPPORT":
            (promoted if support == "ROBUST_SUPPORT" else still_sparse).append(word)

    robust = sorted(w for w, v in composite.items()
                    if v["support"] == "ROBUST_SUPPORT")
    observed = sorted(w for w, v in composite.items()
                      if v["support"] != "NOT_OBSERVED")

    # Global stable-hash assignment: words are ordered by a hash of their
    # spelling and dealt round-robin.
    #
    # An earlier version built a pool grouped by rate band and preceding class
    # and described itself as stratified, but then re-sorted that pool globally
    # by the same hash, which discarded the grouping entirely. Computing the
    # assignment with and without the grouping step gives byte-identical
    # membership, so the description was wrong rather than the code. The
    # grouping is removed instead of repaired: membership is already frozen,
    # and making it real now would redraw the roles after seeing results.
    assignment: dict[str, list[str]] = {role: [] for role in ROLES}
    for position, word in enumerate(sorted(robust,
                                           key=lambda w: stable_rank(w, "role"))):
        assignment[ROLES[position % len(ROLES)]].append(word)

    canonical = json.dumps({role: sorted(words)
                            for role, words in assignment.items()},
                           sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"))
    membership_digest = hashlib.sha256(canonical.encode()).hexdigest()

    # A role whose events come mostly from one word measures that word.
    concentration = {}
    for role, words in assignment.items():
        events = {w: composite[w]["clean_hallucination"] for w in words}
        total = sum(events.values())
        worst = max(events.values()) / total if total else 0.0
        concentration[role] = {
            "words": len(words), "projected_events": total,
            "largest_word_share": round(worst, 4),
            "within_50pct_cap": worst <= MAX_SHARE_OF_ROLE,
            "per_word_events": events,
        }

    met = {role: len(words) >= ROLE_MINIMUM
           for role, words in assignment.items()}
    overlaps = {}
    names = list(assignment)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            overlaps[f"{first} vs {second}"] = sorted(
                set(assignment[first]) & set(assignment[second]))

    # Word count alone is not the gate. A role whose events come mostly from
    # one word measures that word rather than the role, so the concentration
    # cap has to bind too -- an earlier version checked only the count and
    # reported SUFFICIENT while preflight sat at 58%.
    #
    # The share is computed on events observed during discovery and extension,
    # where exposure per word was equal. Generation can rebalance it by giving
    # low-rate words more renderings, so the breach is reported as a constraint
    # on how the role must be built, not as a property that cannot change.
    concentrated = [role for role, entry in concentration.items()
                    if not entry["within_50pct_cap"]]
    required_exposure = {}
    for role in concentrated:
        entry = concentration[role]
        events = entry["per_word_events"]
        dominant = max(events, key=events.get)
        others = sum(v for k, v in events.items() if k != dominant)
        # To bring the dominant word to 50%, the rest must supply as many
        # events as it does; scale their exposure by the shortfall.
        needed = events[dominant]
        required_exposure[role] = {
            "dominant_word": dominant,
            "dominant_events": events[dominant],
            "other_events": others,
            "other_events_needed_for_50pct": needed,
            "exposure_multiplier_for_others": (round(needed / others, 2)
                                               if others else None),
        }

    sufficient = (all(met.values()) and not any(overlaps.values())
                  and not concentrated)
    counts = Counter(v["support"] for v in composite.values())

    report = {
        "analysis": "word_support_roles_v1",
        "base_registry_sha256": hashlib.sha256(
            args.registry.read_bytes()).hexdigest(),
        "extension_rows": len(extension_rows),
        "support_counts_composite": dict(counts),
        "promoted_to_robust": sorted(promoted),
        "remaining_sparse_after_extension": sorted(still_sparse),
        "total_observed_support_words": len(observed),
        "composite_registry": composite,
        "role_assignment": assignment,
        "role_minimum": ROLE_MINIMUM,
        "role_targets_met": met,
        "role_overlaps": overlaps,
        "role_concentration": concentration,
        "roles_exceeding_concentration_cap": concentrated,
        "exposure_rebalance_required": required_exposure,
        "assignment_method": ("global stable-hash assignment: words ordered "
                              "by sha256 of the spelling and dealt round-robin. "
                              "Not stratified -- an earlier description claimed "
                              "stratification that the code discarded"),
        "canonical_membership_sha256": membership_digest,
        "canonical_membership": canonical,
        "exposure_lesson": (
            "several words labelled SPARSE at 400 renderings became ROBUST at "
            "2,000, so that label reflected exposure rather than the word. The "
            "130 NOT_OBSERVED words were not extended and remain undetermined "
            "rather than shown to have zero rate."),
        "support_holdout_naming": (
            "called support-holdout, not final holdout: its baseline behaviour "
            "has been observed here, so it cannot serve as untouched final "
            "evidence"),
        "unseen_final_pool": "RESERVED_UNSCREENED -- not generated, not inferred",
        "STATUS": ("SUFFICIENT" if sufficient
                   else "CONCENTRATION_CAP_EXCEEDED" if concentrated
                   else "INSUFFICIENT_ROLE_SUPPORT"),
    }
    digest = atomic_write_json(args.out, report)

    print(f"composite support: {dict(counts)}")
    print(f"promoted to ROBUST: {len(promoted)} {sorted(promoted)}")
    print(f"still sparse:       {len(still_sparse)} {sorted(still_sparse)}")
    print(f"observed support:   {len(observed)}")
    print(f"\n{'word':12s} {'events':>7s} {'denom':>6s} {'rate':>8s} {'fonts':>6s} {'strata':>7s}")
    for word in robust:
        entry = composite[word]
        print(f"{word:12s} {entry['clean_hallucination']:7d} "
              f"{entry['clean_bare_denominator']:6d} "
              f"{(entry['support_rate'] or 0):8.4f} "
              f"{len(entry['fonts_with_events']):6d} "
              f"{len(entry['strata_with_events']):7d}")
    print("\nroles:")
    for role, words in assignment.items():
        entry = concentration[role]
        print(f"  {role:32s} {len(words)}/{ROLE_MINIMUM} {words}")
        print(f"    events {entry['projected_events']}, "
              f"largest share {entry['largest_word_share']:.1%} "
              f"(cap {MAX_SHARE_OF_ROLE:.0%}) -> "
              f"{'ok' if entry['within_50pct_cap'] else 'EXCEEDS'}")
    print(f"\nSTATUS {report['STATUS']}")
    print(f"report sha256 {digest}")
    return 0 if sufficient else 1


if __name__ == "__main__":
    raise SystemExit(main())
