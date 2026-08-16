"""Analyse the matched pixel-geometry diagnostic and rule on the domain.

The question is what actually governs the e -> é hallucination. Nominal point
size is the obvious candidate and the wrong one to trust: it is fixed before
padding, before any upscale, before the detector crops, and before the
recognizer resizes every line to a constant height. Two lines drawn at "size
12" can arrive at the recognizer at quite different scales, so this report
leads with measured geometry and shows nominal size beside it rather than in
place of it.

Rates are only computed against denominators the diagnostic actually recorded.
Cells with no events are kept and given a one-sided upper bound, because
deleting them would turn "we saw nothing here" into "there is nothing here".
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATOR_ROOT))

from ocr_roi_validator.diagnostic_runner import atomic_write_json  # noqa: E402
from ocr_roi_validator.glyph_geometry import (  # noqa: E402
    GLYPH_HEIGHT_BINS, INK_HEIGHT_BINS, OCCUPANCY_BINS, RESIZE_BINS, bin_value,
)

FUNNEL = ("N_RENDERED_ELIGIBLE", "N_DETECTOR_FOUND", "N_TARGET_LINE_MATCHED",
          "N_RECOGNIZER_DECODED", "N_SEQUENCE_ALIGNED", "N_CLEAN_ELIGIBLE",
          "N_CLEAN_HALLUCINATION", "N_CLEAN_PRESERVATION")


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    spread = z * math.sqrt(p * (1 - p) / trials
                           + z * z / (4 * trials * trials)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def zero_upper_bound(trials: int, confidence: float = 0.95) -> float:
    return 1.0 - (1.0 - confidence) ** (1.0 / trials) if trials else 1.0


def summarise(rows: list[dict]) -> dict:
    """Counts and rates for one group of eligible occurrences."""
    eligible = len(rows)
    clean_eligible = sum(1 for r in rows if r["clean_eligible"])
    hallucination = sum(1 for r in rows if r["clean_hallucination"])
    preservation = sum(1 for r in rows if r["clean_preservation"])
    visual_bare = [r for r in rows if r["visual_target"] == "e"]
    visual_accent = [r for r in rows if r["visual_target"] == "é"]
    bare_clean = sum(1 for r in visual_bare if r["clean_eligible"])
    accent_clean = sum(1 for r in visual_accent if r["clean_eligible"])
    return {
        "eligible": eligible,
        "detector_found": sum(1 for r in rows if r["detector_found"]),
        "target_line_matched": sum(1 for r in rows if r["target_line_matched"]),
        "recognizer_decoded": sum(1 for r in rows if r["recognizer_decoded"]),
        "sequence_aligned": sum(1 for r in rows if r["sequence_aligned"]),
        "clean_eligible": clean_eligible,
        "visual_bare_e": len(visual_bare),
        "visual_accent": len(visual_accent),
        "clean_eligible_bare_e": bare_clean,
        "clean_eligible_accent": accent_clean,
        "clean_hallucination": hallucination,
        "clean_preservation": preservation,
        "detector_yield": sum(1 for r in rows if r["detector_found"]) / eligible
        if eligible else 0.0,
        "alignment_yield": sum(1 for r in rows if r["sequence_aligned"]) / eligible
        if eligible else 0.0,
        "exact_decode_rate": clean_eligible / eligible if eligible else 0.0,
        # The honest denominator for a hallucination is the set of bare-e
        # occurrences that actually reached a clean comparison.
        "hallucination_rate": hallucination / bare_clean if bare_clean else None,
        "hallucination_ci95": wilson(hallucination, bare_clean),
        "hallucination_zero_upper_bound_95": (zero_upper_bound(bare_clean)
                                              if hallucination == 0 and bare_clean
                                              else None),
        "preservation_rate": preservation / accent_clean if accent_clean else None,
        "preservation_ci95": wilson(preservation, accent_clean),
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
    }


def group_by(rows: list[dict], key) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        value = key(row)
        if value is not None:
            grouped[str(value)].append(row)
    return {k: summarise(v) for k, v in sorted(grouped.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.dir / "checkpoint.jsonl"
    recipe = json.loads((args.dir / "recipe.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line
            in checkpoint.read_text(encoding="utf-8").splitlines() if line.strip()]

    expected = recipe["budget"]["total_renderings"]
    digests = {r["row_digest"] for r in rows}
    conditions = {r["base_condition"] for r in rows}
    per_condition = Counter(r["base_condition"] for r in rows)
    completeness = {
        "expected_rows": expected,
        "actual_rows": len(rows),
        "complete": len(rows) == expected,
        "unique_digests": len(digests),
        "duplicate_rows": len(rows) - len(digests),
        "distinct_base_conditions": len(conditions),
        "expected_base_conditions": recipe["budget"]["base_conditions"],
        "conditions_missing_a_font": sorted(
            c for c, n in per_condition.items() if n != recipe["budget"]["fonts"]),
        "eligible_rows": sum(1 for r in rows if r["eligible"]),
    }

    funnel = {
        "N_RENDERED_ELIGIBLE": sum(1 for r in rows if r["eligible"]),
        "N_DETECTOR_FOUND": sum(1 for r in rows if r["detector_found"]),
        "N_TARGET_LINE_MATCHED": sum(1 for r in rows if r["target_line_matched"]),
        "N_RECOGNIZER_DECODED": sum(1 for r in rows if r["recognizer_decoded"]),
        "N_SEQUENCE_ALIGNED": sum(1 for r in rows if r["sequence_aligned"]),
        "N_CLEAN_ELIGIBLE": sum(1 for r in rows if r["clean_eligible"]),
        "N_CLEAN_HALLUCINATION": sum(1 for r in rows if r["clean_hallucination"]),
        "N_CLEAN_PRESERVATION": sum(1 for r in rows if r["clean_preservation"]),
    }
    losses = {
        "detector_miss": sum(1 for r in rows if r["outcome"] == "DETECTOR_MISS"),
        "wrong_line_selected": sum(1 for r in rows
                                   if r["outcome"] == "WRONG_LINE_SELECTED"),
        "recognizer_failure": sum(1 for r in rows
                                  if r["outcome"] == "RECOGNIZER_FAILURE"),
        "alignment_ambiguity": sum(1 for r in rows
                                   if r["outcome"] == "ALIGNMENT_AMBIGUITY"),
        "insertion": sum(1 for r in rows if r["outcome"] == "INSERTION"),
        "deletion": sum(1 for r in rows if r["outcome"] == "DELETION"),
        "change_elsewhere": sum(1 for r in rows if r["outcome"] == "CHANGE_ELSEWHERE"),
        "multiple_changes": sum(1 for r in rows if r["outcome"] == "MULTIPLE_CHANGES"),
        "accent_lost": sum(1 for r in rows if r["outcome"] == "ACCENT_LOST"),
        "other_substitution": sum(1 for r in rows
                                  if r["outcome"] == "OTHER_SUBSTITUTION"),
        "render_or_detector_error": sum(
            1 for r in rows if r["outcome"] in {"RENDER_ERROR", "DETECTOR_ERROR"}),
    }

    breakdowns = {
        "by_font": group_by(rows, lambda r: r["font"]),
        "by_font_cohort": group_by(rows, lambda r: r["font_cohort"]),
        "by_nominal_size": group_by(rows, lambda r: r["nominal_size"]),
        "by_runtime_ink_height_bin": group_by(
            rows, lambda r: bin_value(r["runtime_ink_height"], INK_HEIGHT_BINS)
            if r["runtime_ink_height"] is not None else None),
        "by_rendered_glyph_height_bin": group_by(
            rows, lambda r: bin_value(r["rendered_glyph_height"], GLYPH_HEIGHT_BINS)
            if r["rendered_glyph_height"] is not None else None),
        "by_occupancy_bin": group_by(
            rows, lambda r: bin_value(r["rendered_occupancy"], OCCUPANCY_BINS)
            if r["rendered_occupancy"] is not None else None),
        "by_resize_scale_bin": group_by(
            rows, lambda r: bin_value(r["recognizer_resize_scale"], RESIZE_BINS)
            if r["recognizer_resize_scale"] is not None else None),
        "by_padding_bucket": group_by(rows, lambda r: r["padding_bucket"]),
        "by_upscale_bucket": group_by(rows, lambda r: r["upscale_bucket"]),
        "by_polarity": group_by(rows, lambda r: r["polarity"]),
        "by_contrast_bucket": group_by(rows, lambda r: r["contrast_bucket"]),
        "by_blur_bucket": group_by(rows, lambda r: r["blur_bucket"]),
    }
    interactions = {
        "font_x_ink_height": group_by(
            rows, lambda r: f"{r['font']}|{bin_value(r['runtime_ink_height'], INK_HEIGHT_BINS)}"
            if r["runtime_ink_height"] is not None else None),
        "ink_height_x_upscale": group_by(
            rows, lambda r: f"{bin_value(r['runtime_ink_height'], INK_HEIGHT_BINS)}|{r['upscale_bucket']}"
            if r["runtime_ink_height"] is not None else None),
        "padding_x_polarity": group_by(
            rows, lambda r: f"{r['padding_bucket']}|{r['polarity']}"),
        "nominal_size_x_upscale": group_by(
            rows, lambda r: f"{r['nominal_size']}|{r['upscale_bucket']}"),
    }

    hallucinations = [r for r in rows if r["clean_hallucination"]]
    fonts_hit = {r["font"] for r in hallucinations}
    templates_hit = {r["template_index"] for r in hallucinations}
    words_hit = {r["word_index"] for r in hallucinations}
    conditions_hit = {r["base_condition"] for r in hallucinations}

    # Which geometry bins carry trigger support, and is the effect monotone?
    ink_support = {k: v for k, v in breakdowns["by_runtime_ink_height_bin"].items()
                   if v["clean_hallucination"] > 0}
    size_support = {k: v for k, v in breakdowns["by_nominal_size"].items()
                    if v["clean_hallucination"] > 0}
    optical_support = {
        axis: {k: v["clean_hallucination"] for k, v in breakdowns[key].items()}
        for axis, key in (("padding", "by_padding_bucket"),
                          ("upscale", "by_upscale_bucket"),
                          ("polarity", "by_polarity"),
                          ("contrast", "by_contrast_bucket"),
                          ("blur", "by_blur_bucket"))
    }

    def spread(counts: dict) -> float:
        """Ratio of the largest to the smallest rate.

        A zero-count bin used to yield ``inf`` here, which then satisfied the
        ">= 3.0" test and declared a dependence that the data did not support.
        Zero is an absence of observations, not an infinitely small rate, so
        such bins are excluded from the ratio and reported separately with
        their upper bound.
        """
        values = [v for v in counts.values() if v is not None and v > 0]
        return (max(values) / min(values)) if len(values) >= 2 else 1.0

    def any_separated(entries: dict) -> bool:
        """True only if some pair of cells has non-overlapping 95% intervals.

        Comparing point estimates alone would let a handful of events look
        like a trend; with thirteen hallucinations spread over six bins, every
        interval overlaps and no ordering is real.
        """
        intervals = [v["hallucination_ci95"] for v in entries.values()
                     if v["clean_eligible_bare_e"] > 0]
        return any(a[1] < b[0] or b[1] < a[0]
                   for i, a in enumerate(intervals) for b in intervals[i + 1:])

    geometry_rates = {k: v["hallucination_rate"]
                      for k, v in breakdowns["by_runtime_ink_height_bin"].items()
                      if v["hallucination_rate"] is not None}
    optical_rates = {
        axis: {k: breakdowns[key][k]["hallucination_rate"]
               for k in breakdowns[key]
               if breakdowns[key][k]["hallucination_rate"] is not None}
        for axis, key in (("padding", "by_padding_bucket"),
                          ("upscale", "by_upscale_bucket"),
                          ("polarity", "by_polarity"))
    }
    geometry_spread = spread(geometry_rates)
    optical_spread = max((spread(v) for v in optical_rates.values() if v),
                         default=1.0)

    reproduced_across_fonts = len(fonts_hit) >= 3
    not_concentrated = len(templates_hit) >= 2 and len(words_hit) >= 2 \
        and len(conditions_hit) >= 3
    geometry_support = len(ink_support) >= 2

    # A verdict also requires that the funnel not have decided the answer for
    # us. When attrition is heavy and runs in opposite directions across the
    # axis under test, the surviving sample is selected by that axis, and any
    # apparent effect may be differential attrition rather than hallucination.
    attrition = 1.0 - (funnel["N_CLEAN_ELIGIBLE"] / funnel["N_RENDERED_ELIGIBLE"]
                       if funnel["N_RENDERED_ELIGIBLE"] else 1.0)
    geometry_separated = any_separated(breakdowns["by_runtime_ink_height_bin"])
    optical_separated = any(any_separated(breakdowns[key]) for key in
                            ("by_padding_bucket", "by_upscale_bucket",
                             "by_polarity"))
    funnel_usable = attrition <= 0.60

    if not hallucinations:
        verdict = "NOT_CONFIRMED"
    elif not funnel_usable:
        verdict = "NOT_CONFIRMED"
    elif not (reproduced_across_fonts and not_concentrated and geometry_support):
        verdict = "NOT_CONFIRMED"
    elif geometry_separated and not optical_separated:
        verdict = "PIXEL_GEOMETRY_DEPENDENT"
    elif optical_separated and not geometry_separated:
        verdict = "OPTICAL_CONDITION_DEPENDENT"
    elif geometry_separated and optical_separated:
        verdict = "INTERACTION_DEPENDENT"
    else:
        verdict = "NOT_CONFIRMED"

    report = {
        "diagnostic": "matched_pixel_geometry_v1",
        "recipe_budget": recipe["budget"],
        "exposure_completeness": completeness,
        "funnel": funnel,
        "funnel_losses": losses,
        "breakdowns": breakdowns,
        "interactions": interactions,
        "verdict_inputs": {
            "total_clean_hallucination": len(hallucinations),
            "fonts_with_hallucination": sorted(fonts_hit),
            "distinct_templates": len(templates_hit),
            "distinct_words": len(words_hit),
            "distinct_base_conditions": len(conditions_hit),
            "ink_height_bins_with_support": sorted(ink_support),
            "nominal_sizes_with_support": sorted(size_support),
            "geometry_rate_spread": geometry_spread,
            "optical_rate_spread": optical_spread,
            "optical_counts": optical_support,
            "reproduced_across_3_fonts": reproduced_across_fonts,
            "not_concentrated": not_concentrated,
            "geometry_support": geometry_support,
            "geometry_bins_statistically_separated": geometry_separated,
            "optical_cells_statistically_separated": optical_separated,
            "clean_eligible_attrition": round(attrition, 6),
            "funnel_usable": funnel_usable,
            "funnel_usable_threshold": 0.60,
        },
        "HALLUCINATION_DOMAIN": verdict,
        "nominal_size_caveat": (
            "nominal size and measured geometry are reported separately; an "
            "effect explained by measured geometry is not additionally claimed "
            "as a nominal-size effect"),
    }
    digest = atomic_write_json(args.out, report)

    print(f"rows {len(rows)}/{expected} complete={completeness['complete']} "
          f"duplicates={completeness['duplicate_rows']}")
    print("\nfunnel")
    for stage in FUNNEL:
        print(f"  {stage:26s} {funnel[stage]:7d}")
    print("\nlosses")
    for name, count in losses.items():
        if count:
            print(f"  {name:26s} {count:7d}")
    print(f"\n{'ink height bin':18s} {'bare-e clean':>12s} {'halluc':>7s} "
          f"{'rate':>9s} {'95% CI':>20s}")
    for name, entry in breakdowns["by_runtime_ink_height_bin"].items():
        rate = entry["hallucination_rate"]
        low, high = entry["hallucination_ci95"]
        shown = f"{rate:.4f}" if rate is not None else "n/a"
        print(f"  {name:16s} {entry['clean_eligible_bare_e']:12d} "
              f"{entry['clean_hallucination']:7d} {shown:>9s} "
              f"[{low:.4f},{high:.4f}]")
    print(f"\n{'font':14s} {'bare-e clean':>12s} {'halluc':>7s} {'rate':>9s}")
    for name, entry in breakdowns["by_font"].items():
        rate = entry["hallucination_rate"]
        shown = f"{rate:.4f}" if rate is not None else "n/a"
        print(f"  {name:12s} {entry['clean_eligible_bare_e']:12d} "
              f"{entry['clean_hallucination']:7d} {shown:>9s}")
    print(f"\ngeometry rate spread {geometry_spread:.2f}, "
          f"optical rate spread {optical_spread:.2f}")
    print(f"HALLUCINATION_DOMAIN : {verdict}")
    print(f"report sha256        : {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
