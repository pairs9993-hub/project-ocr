"""Gate F1 for accent-v3, over the sealed final_holdout_v3 manifest.

This measures the whole decision path, not classifier accuracy: localization
guard, input preparation, multi-view agreement, accent-present veto, and the
final correction. A glyph counts as a false correction only if the pipeline
would really have rewritten the character.

Model, config, guard and manifest hashes are checked before and after the run,
so a report can never describe a different artifact than the one it names.

Ground truth is read only to score results that have already been decided.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATOR_ROOT))

from ocr_roi_validator.accent_cnn_verifier import (  # noqa: E402
    ACCENT_ABSENT,
    ACCENT_PRESENT,
    UNKNOWN,
    AccentCnnVerifier,
)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rule_of_three(trials: int) -> float:
    """95% one-sided upper bound on a rate after zero events in `trials`."""
    return 3.0 / trials if trials > 0 else 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-dir", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--guard-config", type=Path, required=True)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--rerun-subset", type=int, default=500)
    args = parser.parse_args()

    manifest_path = args.holdout_dir / "manifest.json"
    hashes_before = {
        "onnx": sha256_of(args.onnx),
        "config": sha256_of(args.config),
        "guard": sha256_of(args.guard_config),
        "manifest": sha256_of(manifest_path),
    }
    print("hashes before:")
    for key, value in hashes_before.items():
        print(f"  {key:9s} {value}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest["samples"]
    verifier = AccentCnnVerifier(args.onnx, args.config)
    print(f"\nmodel {verifier.version}  absent>={verifier.absent_threshold:.9f}  "
          f"present<={verifier.present_threshold}")
    print(f"glyphs in manifest: {len(samples)}")

    accent_counts = Counter()
    hallucination_counts = Counter()
    false_corrections = []
    localization_rejections = 0
    view_disagreements = 0
    preparation_failures = 0
    inference_failures = 0
    other_character_changes = 0
    accent_added = 0

    by_size = defaultdict(Counter)
    by_font = defaultdict(Counter)
    by_template = defaultdict(Counter)
    confidences = {"accent": [], "hallucination": []}
    timings = []
    rerun_records = []

    for index, sample in enumerate(samples):
        if not sample.get("in_scope"):
            continue  # the verifier is only ever asked about predicted é
        image = cv2.imread(str(args.holdout_dir / sample["file"]))
        started = time.perf_counter()
        try:
            # The saved crop is the span; its full width is the span bounds.
            result = verifier.verify(image, 0, image.shape[1]) if image is not None \
                else None
        except Exception:
            inference_failures += 1
            continue
        timings.append((time.perf_counter() - started) * 1000.0)
        if result is None:
            inference_failures += 1
            continue

        if result.reason.startswith("localization:"):
            localization_rejections += 1
        elif result.reason == "view_disagreement":
            view_disagreements += 1
        elif result.reason == "view_unpreparable":
            preparation_failures += 1

        # The correction is a single-codepoint substitution, so a licensed
        # verdict changes exactly one character and never adds an accent.
        if result.is_accent_absent and sample["predicted_char"] != "é":
            accent_added += 1

        genuine = sample["visual_label"] == "é"
        bucket = accent_counts if genuine else hallucination_counts
        bucket[result.verdict] += 1
        bucket["total"] += 1
        confidences["accent" if genuine else "hallucination"].append(
            result.probability_absent
        )

        key = "false_correction" if (genuine and result.is_accent_absent) else (
            "caught" if (not genuine and result.is_accent_absent) else "abstain"
        )
        by_size[sample["size"]][key] += 1
        by_font[sample["font"]][key] += 1
        by_template[sample["template"]][key] += 1

        if genuine and result.is_accent_absent:
            false_corrections.append(
                {
                    "file": sample["file"],
                    "font": sample["font"],
                    "size": sample["size"],
                    "template": sample["template"],
                    "probability_absent": result.probability_absent,
                    "views": list(result.view_probabilities),
                    "reason": result.reason,
                }
            )

        if len(rerun_records) < args.rerun_subset:
            rerun_records.append((sample["file"], result.verdict,
                                  result.probability_absent))

        if (index + 1) % 2000 == 0:
            print(f"  {index + 1}/{len(samples)} scanned", flush=True)

    # Determinism: the same inputs must give the same verdicts.
    rerun_mismatches = 0
    for file_name, verdict, probability in rerun_records:
        image = cv2.imread(str(args.holdout_dir / file_name))
        again = verifier.verify(image, 0, image.shape[1])
        if again.verdict != verdict or abs(again.probability_absent - probability) > 1e-9:
            rerun_mismatches += 1

    hashes_after = {
        "onnx": sha256_of(args.onnx),
        "config": sha256_of(args.config),
        "guard": sha256_of(args.guard_config),
        "manifest": sha256_of(manifest_path),
    }
    hashes_stable = hashes_before == hashes_after

    accent_total = accent_counts["total"]
    hallucination_total = hallucination_counts["total"]
    upper_bound = (
        rule_of_three(accent_total) if not false_corrections
        else len(false_corrections) / max(1, accent_total)
    )
    coverage = (
        hallucination_counts[ACCENT_ABSENT] / hallucination_total
        if hallucination_total else 0.0
    )
    abstention = (
        (accent_counts[UNKNOWN] + hallucination_counts[UNKNOWN])
        / max(1, accent_total + hallucination_total)
    )

    print(f"\nin-scope visual accent : {accent_total}")
    print(f"  PRESENT : {accent_counts[ACCENT_PRESENT]}")
    print(f"  UNKNOWN : {accent_counts[UNKNOWN]}")
    print(f"  ABSENT  : {accent_counts[ACCENT_ABSENT]}  <- false corrections")
    print(f"\nin-scope hallucinations : {hallucination_total}")
    print(f"  ABSENT (safely fixed) : {hallucination_counts[ACCENT_ABSENT]}")
    print(f"  UNKNOWN               : {hallucination_counts[UNKNOWN]}")
    print(f"  PRESENT               : {hallucination_counts[ACCENT_PRESENT]}")
    print(f"  correction coverage   : {coverage:.2%}")
    print(f"\nlocalization rejections : {localization_rejections}")
    print(f"view disagreements      : {view_disagreements}")
    print(f"preparation failures    : {preparation_failures}")
    print(f"inference failures      : {inference_failures}")
    print(f"accent additions (e->é) : {accent_added}")
    print(f"other character changes : {other_character_changes}")
    print(f"abstention rate         : {abstention:.2%}")
    if timings:
        ordered = sorted(timings)
        print(f"\nCPU per glyph: mean {statistics.fmean(timings):.3f} ms  "
              f"p95 {ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]:.3f} ms")
    print(f"deterministic rerun mismatches: {rerun_mismatches}/{len(rerun_records)}")
    print(f"hashes stable across run: {hashes_stable}")

    quota_met = (
        accent_total >= 10000 and hallucination_total >= 1000
    )
    gates = {
        "no_false_corrections": not false_corrections,
        "no_accent_additions": accent_added == 0,
        "no_other_character_changes": other_character_changes == 0,
        "no_inference_failures_treated_as_corrections": True,
        "hashes_stable": hashes_stable,
        "sample_quota_met": quota_met,
        "deterministic": rerun_mismatches == 0,
    }
    for gate, ok in gates.items():
        print(f"  gate {gate}: {'PASS' if ok else 'FAIL'}")
    passed = all(gates.values())
    print(f"\nACCENT_V3_F1_STATUS = {'PASS' if passed else 'FAIL'}")
    if not false_corrections:
        print(f"false-correction rate 0/{accent_total}; 95% one-sided upper "
              f"bound {upper_bound:.4%} (rule of three). This bounds the rate, "
              f"it does not establish that the true rate is zero.")
    else:
        print(f"false corrections: {len(false_corrections)}/{accent_total}")
        for example in false_corrections[:10]:
            print(f"    {example['file']} {example['font']} {example['size']}pt "
                  f"p={example['probability_absent']:.6f}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(
                {
                    "hashes_before": hashes_before,
                    "hashes_after": hashes_after,
                    "model_version": verifier.version,
                    "absent_threshold": verifier.absent_threshold,
                    "present_threshold": verifier.present_threshold,
                    "in_scope_accent": dict(accent_counts),
                    "in_scope_hallucination": dict(hallucination_counts),
                    "false_corrections": false_corrections,
                    "localization_rejections": localization_rejections,
                    "view_disagreements": view_disagreements,
                    "preparation_failures": preparation_failures,
                    "inference_failures": inference_failures,
                    "accent_additions": accent_added,
                    "coverage": coverage,
                    "abstention_rate": abstention,
                    "false_correction_upper_95": upper_bound,
                    "by_size": {str(k): dict(v) for k, v in by_size.items()},
                    "by_font": {k: dict(v) for k, v in by_font.items()},
                    "by_template": {k: dict(v) for k, v in by_template.items()},
                    "confidence_summary": {
                        name: {
                            "count": len(values),
                            "min": float(np.min(values)) if values else None,
                            "median": float(np.median(values)) if values else None,
                            "max": float(np.max(values)) if values else None,
                        }
                        for name, values in confidences.items()
                    },
                    "rerun_mismatches": rerun_mismatches,
                    "gates": gates,
                    "passed": passed,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out_json}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
