"""Classify word support and assign v3 splits by stable hash.

Two things must not happen here. A word must not be promoted because someone
looked at the numbers and liked it -- assignment is a hash of the spelling,
stratified so each split gets a comparable mix. And "no event observed at this
exposure" must not be written down as "the rate is zero"; 400 renderings can
easily miss a rate of one in a thousand.

The context analysis reports factors separately from word identity. An earlier
reading of partial data suggested the character before the target explained
everything, but supplement's reglage has a neutral predecessor and the highest
rate of any word measured, so that story was wrong. Factor rates are reported
with their denominators and left as association, not cause.
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

# Sealed split targets. Holdout words are reserved without being screened, so
# the final evidence set is never chosen by looking at baseline behaviour.
SPLIT_TARGETS = {
    "train_augment_v3": 12,
    "calibration_v3": 8,
    "preflight_v3": 8,
}
HOLDOUT_RESERVE_FRACTION = 0.25


def upper_bound_zero(trials: int, confidence: float = 0.95) -> float:
    return 1.0 - (1.0 - confidence) ** (1.0 / trials) if trials else 1.0


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    spread = z * math.sqrt(p * (1 - p) / trials
                           + z * z / (4 * trials * trials)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def classify(events: int, fonts: int, strata: int) -> str:
    if events >= ROBUST_MIN_EVENTS and (fonts >= ROBUST_MIN_BREADTH
                                        or strata >= ROBUST_MIN_BREADTH):
        return "ROBUST_SUPPORT"
    if events >= 1:
        return "SPARSE_SUPPORT"
    return "NOT_OBSERVED"


def stable_bucket(word: str, salt: str, buckets: int) -> int:
    digest = hashlib.sha256(f"{salt}|{word}".encode()).hexdigest()
    return int(digest[:16], 16) % buckets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line
            in (args.discovery / "checkpoint.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
    recipe = json.loads((args.discovery / "recipe.json").read_text(encoding="utf-8"))

    per_word: dict[str, dict] = defaultdict(
        lambda: {"renderings": 0, "clean_bare": 0, "events": 0,
                 "preservation": 0, "fonts": set(), "strata": set()})
    for row in rows:
        entry = per_word[row["word"]]
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

    context_by_word = {}
    for row in rows:
        context_by_word.setdefault(row["word"], {
            "group": row["word_group"],
            "preceding_class": row["preceding_class"],
            "following_class": row["following_class"],
            "word_length": row["word_length"],
            "normalized_position": row["normalized_position"],
            "local_context": row["local_context"],
            "e_count": row["e_count"],
        })

    registry = {}
    for word, entry in sorted(per_word.items()):
        support = classify(entry["events"], len(entry["fonts"]),
                           len(entry["strata"]))
        rate = (entry["events"] / entry["clean_bare"]
                if entry["clean_bare"] else None)
        registry[word] = {
            **context_by_word[word],
            "renderings": entry["renderings"],
            "clean_bare_denominator": entry["clean_bare"],
            "clean_hallucination": entry["events"],
            "clean_preservation": entry["preservation"],
            "fonts_with_events": sorted(entry["fonts"]),
            "strata_with_events": sorted(entry["strata"]),
            "support_rate": rate,
            "support_rate_ci95": wilson(entry["events"], entry["clean_bare"]),
            "zero_event_upper_bound_95": (
                upper_bound_zero(entry["clean_bare"])
                if entry["events"] == 0 and entry["clean_bare"] else None),
            "support": support,
        }

    by_support = Counter(v["support"] for v in registry.values())
    robust = sorted(w for w, v in registry.items()
                    if v["support"] == "ROBUST_SUPPORT")

    # ---- context factors, reported as association only --------------------
    def factor_table(key):
        table = defaultdict(lambda: [0, 0])
        for row in rows:
            if row["visual_target"] != "e" or row["terminal_reason"] not in CLEAN_BARE:
                continue
            table[row[key]][0] += 1
            table[row[key]][1] += int(row["clean_hallucination"])
        return {str(k): {"clean_bare": v[0], "events": v[1],
                         "rate": v[1] / v[0] if v[0] else None,
                         "ci95": wilson(v[1], v[0])}
                for k, v in sorted(table.items())}

    factors = {name: factor_table(name) for name in
               ("preceding_class", "following_class", "word_group",
                "word_length", "e_count", "font", "measured_stratum")}

    # Is any factor's effect present within a single word, or does it only
    # track which words happen to have events? Words vary far more than any
    # factor level does, which is what "not confirmed" means here.
    word_rates = [v["support_rate"] for v in registry.values()
                  if v["support_rate"] is not None]
    word_spread = (max(word_rates) / min(r for r in word_rates if r > 0)
                   if any(r > 0 for r in word_rates) else None)
    preceding_rates = [v["rate"] for v in factors["preceding_class"].values()
                       if v["rate"]]
    factor_spread = (max(preceding_rates) / min(preceding_rates)
                     if len(preceding_rates) > 1 else None)

    # ---- deterministic split assignment ----------------------------------
    # Stratify first so each split gets a comparable mix of rate band, e
    # position, neighbour class and length, then order within each stratum by
    # a stable hash. No human choice enters.
    def band(word: str) -> str:
        rate = registry[word]["support_rate"] or 0.0
        if rate >= 0.05:
            return "high"
        return "mid" if rate >= 0.02 else "low"

    strata_key = {w: (band(w), registry[w]["preceding_class"],
                      "short" if registry[w]["word_length"] <= 6 else "long")
                  for w in robust}
    grouped: dict[tuple, list[str]] = defaultdict(list)
    for word in robust:
        grouped[strata_key[word]].append(word)
    for words in grouped.values():
        words.sort(key=lambda w: stable_bucket(w, "order", 2 ** 32))

    reserve_count = max(1, int(round(len(robust) * HOLDOUT_RESERVE_FRACTION)))
    assignment: dict[str, list[str]] = {name: [] for name in SPLIT_TARGETS}
    assignment["final_holdout_reserved_v3"] = []

    # Round-robin across strata so no split monopolises one context.
    order = [name for name in SPLIT_TARGETS] + ["final_holdout_reserved_v3"]
    quota = dict(SPLIT_TARGETS)
    quota["final_holdout_reserved_v3"] = reserve_count
    pool = [w for key in sorted(grouped) for w in grouped[key]]
    pool.sort(key=lambda w: (stable_bucket(w, "assign", 2 ** 32),))
    cursor = 0
    for word in pool:
        for _ in range(len(order)):
            name = order[cursor % len(order)]
            cursor += 1
            if len(assignment[name]) < quota[name]:
                assignment[name].append(word)
                break

    met = {name: len(assignment[name]) >= quota[name] for name in quota}
    sufficient = all(met.values())
    overlaps = {}
    names = list(assignment)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            shared = sorted(set(assignment[first]) & set(assignment[second]))
            overlaps[f"{first} vs {second}"] = shared

    report = {
        "analysis": "word_support_registry_v1",
        "discovery_recipe_sha256": hashlib.sha256(
            (args.discovery / "recipe.json").read_bytes()).hexdigest(),
        "discovery_renderings": len(rows),
        "candidate_words": len(registry),
        "support_rules": recipe["support_rules"],
        "support_counts": dict(by_support),
        "registry": registry,
        "context_factors": factors,
        "factor_vs_word_variation": {
            "word_rate_spread": word_spread,
            "preceding_class_rate_spread": factor_spread,
            "reading": (
                "context factors are reported as association. An earlier read "
                "of partial data suggested the preceding character explained "
                "the effect, but supplement's reglage has a neutral "
                "predecessor and the highest measured rate of any word, so a "
                "single-factor account does not hold."),
        },
        "split_assignment": assignment,
        "split_targets": quota,
        "split_targets_met": met,
        "split_overlaps": overlaps,
        "assignment_method": (
            "stratified by rate band, preceding class and length, ordered "
            "within stratum by sha256 of the word, then round-robin. No word "
            "is chosen by inspection."),
        "holdout_note": (
            "final_holdout_reserved_v3 words are reserved unscreened; their "
            "baseline behaviour is deliberately not examined here"),
        "ROBUST_SUPPORT_WORDS": len(robust),
        "SPARSE_SUPPORT_WORDS": by_support.get("SPARSE_SUPPORT", 0),
        "NOT_OBSERVED_WORDS": by_support.get("NOT_OBSERVED", 0),
        "STATUS": "SUFFICIENT" if sufficient else "INSUFFICIENT_WORD_SUPPORT",
    }
    digest = atomic_write_json(args.out, report)

    print(f"discovery {len(rows):,} renderings, {len(registry)} words")
    print(f"support: {dict(by_support)}")
    print(f"\n{'word':14s} {'events':>7s} {'denom':>6s} {'rate':>8s}  context")
    for word in robust:
        entry = registry[word]
        print(f"{word:14s} {entry['clean_hallucination']:7d} "
              f"{entry['clean_bare_denominator']:6d} "
              f"{entry['support_rate']:8.4f}  "
              f"prev={entry['preceding_class']} next={entry['following_class']}")
    print(f"\nassignment:")
    for name, words in assignment.items():
        print(f"  {name:28s} {len(words):2d}/{quota[name]:2d} {words}")
    print(f"\noverlaps: {sum(len(v) for v in overlaps.values())}")
    print(f"STATUS {report['STATUS']}")
    print(f"report sha256 {digest}")
    return 0 if sufficient else 1


if __name__ == "__main__":
    raise SystemExit(main())
