"""Evaluate a frozen accent verifier on a synthetic split (Gate F1).

The gate that decides everything is the false-correction count: a real accent
judged ``e`` would let OCR erase a legitimate diacritic, which is exactly the
failure this project must never produce. One such case fails the gate.

Coverage -- how many hallucinated accents the verifier actually catches -- is
reported alongside, because a model that abstains on everything is safe but
useless.

A zero count is not proof of a zero rate, so the 95% upper confidence bound on
the false-correction rate is reported too.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATOR_ROOT))

from ocr_roi_validator.accent_verifier import (  # noqa: E402
    ACCENT_ABSENT,
    ACCENT_PRESENT,
    UNKNOWN,
    load_model,
    verify_accent_glyph,
)


def rule_of_three_upper_bound(trials: int) -> float:
    """95% upper bound on a rate after observing zero events in `trials`."""
    return 3.0 / trials if trials > 0 else 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    payload = args.model.read_text(encoding="utf-8")
    model = load_model(args.model)
    if model is None:
        print(f"no model at {args.model}", file=sys.stderr)
        return 1
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    manifest = json.loads(
        (args.split_dir / "manifest.json").read_text(encoding="utf-8")
    )
    samples = manifest["samples"]

    # Counters are split by what was *drawn*, never by what OCR said.
    accent_total = accent_correct = accent_unknown = accent_false = 0
    bare_total = bare_correct = bare_unknown = bare_wrong = 0
    unmeasurable = 0
    false_correction_examples = []

    for sample in samples:
        image = cv2.imread(str(args.split_dir / sample["file"]))
        result = verify_accent_glyph(image, model)
        if result.reason == "glyph_unmeasurable":
            unmeasurable += 1

        if sample["visual_label"] == "é":
            accent_total += 1
            if result.verdict == ACCENT_ABSENT:
                accent_false += 1
                if len(false_correction_examples) < 10:
                    false_correction_examples.append(
                        {
                            "file": sample["file"],
                            "font": sample["font"],
                            "size": sample["size"],
                            "probability_absent": result.probability_absent,
                        }
                    )
            elif result.verdict == ACCENT_PRESENT:
                accent_correct += 1
            else:
                accent_unknown += 1
        else:
            bare_total += 1
            if result.verdict == ACCENT_ABSENT:
                bare_correct += 1
            elif result.verdict == UNKNOWN:
                bare_unknown += 1
            else:
                bare_wrong += 1

    # Correction coverage counts only glyphs the recognizer actually misread as
    # an accent: those are the ones a verifier could repair.
    hallucinated = [s for s in samples
                    if s["visual_label"] == "e" and s["predicted_char"] == "é"]
    caught = 0
    for sample in hallucinated:
        image = cv2.imread(str(args.split_dir / sample["file"]))
        if verify_accent_glyph(image, model).is_accent_absent:
            caught += 1

    upper_bound = (
        rule_of_three_upper_bound(accent_total)
        if accent_false == 0
        else accent_false / accent_total
    )

    print(f"split      : {args.split_dir}")
    print(f"model      : {model.version}  sha256 {digest[:16]}")
    print(f"thresholds : absent>={model.absent_threshold:.4f} "
          f"present<={model.present_threshold:.4f}")
    print(f"glyphs     : {len(samples)} ({accent_total} accent, {bare_total} bare)")
    print(f"unmeasurable (abstained): {unmeasurable}")
    print()
    print("Drawn WITH an accent:")
    print(f"  judged accent (correct)      : {accent_correct}/{accent_total}")
    print(f"  judged unknown (safe)        : {accent_unknown}/{accent_total}")
    print(f"  judged e (FALSE CORRECTION)  : {accent_false}/{accent_total}")
    print()
    print("Drawn WITHOUT an accent:")
    print(f"  judged e (correct)           : {bare_correct}/{bare_total}")
    print(f"  judged unknown (safe)        : {bare_unknown}/{bare_total}")
    print(f"  judged accent (missed)       : {bare_wrong}/{bare_total}")
    print()
    print(f"recognizer hallucinations in split : {len(hallucinated)}")
    print(f"  of those, verifier would fix     : {caught}")
    print()
    if accent_false == 0:
        print(f"false-correction rate: 0/{accent_total}, "
              f"95% upper bound {upper_bound:.2%}")
    else:
        print(f"false-correction rate: {accent_false}/{accent_total} "
              f"= {upper_bound:.2%}")
        for example in false_correction_examples:
            print(f"    {example['file']} {example['font']} {example['size']}pt "
                  f"p={example['probability_absent']:.4f}")

    passed = accent_false == 0
    print(f"\nGATE (zero false corrections): {'PASS' if passed else 'FAIL'}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(
                {
                    "split_dir": str(args.split_dir),
                    "model_version": model.version,
                    "model_sha256": digest,
                    "absent_threshold": model.absent_threshold,
                    "present_threshold": model.present_threshold,
                    "accent_total": accent_total,
                    "accent_correct": accent_correct,
                    "accent_unknown": accent_unknown,
                    "accent_false_corrections": accent_false,
                    "bare_total": bare_total,
                    "bare_correct": bare_correct,
                    "bare_unknown": bare_unknown,
                    "bare_wrong": bare_wrong,
                    "hallucinations": len(hallucinated),
                    "hallucinations_caught": caught,
                    "false_correction_rate_upper_95": upper_bound,
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
