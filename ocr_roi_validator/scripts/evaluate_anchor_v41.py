"""Evaluate localization-v4.1 and decompose where v4's misassignments came from.

Two questions, on the same data:

* Which path produced v4's accepted misassignments -- the consensus itself, or
  the posterior fallback it used when consensus failed? If the fallback
  dominates, removing it is the fix rather than a tightening of tolerances.
* With the fallback removed, what does v4.1 accept, and is what it accepts
  correct?

The second question is only meaningful on data whose results have not been seen
before, so a development run reports the decomposition and a validation run
reports the gate.

Ground truth scores outcomes; it is never an input to a decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATOR_ROOT))
SCRIPTS = VALIDATOR_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_accent_glyph_dataset import build_engine, load_labels  # noqa: E402
from evaluate_anchor_localization_v4 import (  # noqa: E402
    anchor_candidates,
    collapse_ctc,
    ink_mask,
)
from ocr_roi_validator.anchor_localization_v41 import (  # noqa: E402
    AnchorConfig,
    Reason,
    locate_target,
)


def target_token(emitted, stride, gt_centre):
    """The emitted e/é token nearest the true centre.

    Ground truth picks *which* token to evaluate; it never enters the anchor or
    the guard.
    """
    best, best_distance = None, None
    for position, item in enumerate(emitted):
        if unicodedata.normalize("NFC", item["char"]) not in {"e", "é"}:
            continue
        centre = ((item["start"] + item["end"] + 1) / 2.0) * stride
        distance = abs(centre - gt_centre)
        if best_distance is None or distance < best_distance:
            best, best_distance = position, distance
    return best


def containment(patch, glyph_mask, accent_mask, line, crop_w):
    x0 = max(0, int(round(patch[0])))
    x1 = min(crop_w, int(round(patch[1])))
    columns = slice(x0, x1)
    glyph_total = max(1, int(glyph_mask.sum()))
    accent_total = int(accent_mask.sum())
    mask = ink_mask(line)
    patch_ink = int(mask[:, columns].sum())
    target_ink = int((glyph_mask | accent_mask)[:, columns].sum())
    return {
        "glyph": int(glyph_mask[:, columns].sum()) / glyph_total,
        "accent": (int(accent_mask[:, columns].sum()) / accent_total)
        if accent_total else None,
        "intrusion": max(0, patch_ink - target_ink) / max(1, patch_ink),
        "intrusion_pixels": max(0, patch_ink - target_ink),
    }


def neighbour_intrusion(patch, glyph_mask, accent_mask, line, crop_w,
                        anchor, central_half):
    """Split intruding ink by where it sits relative to the target."""
    x0 = max(0, int(round(patch[0])))
    x1 = min(crop_w, int(round(patch[1])))
    mask = ink_mask(line)
    target = glyph_mask | accent_mask
    foreign = mask & ~target

    rows = np.where(target.any(axis=1))[0]
    if rows.size:
        target_top, target_bottom = int(rows[0]), int(rows[-1])
    else:
        target_top, target_bottom = 0, line.shape[0] - 1
    x_height_top = target_top + int(0.35 * max(1, target_bottom - target_top))

    columns = slice(x0, x1)
    window = foreign[:, columns]
    body = int(window[x_height_top:, :].sum())
    ascender = int(window[:x_height_top, :].sum())

    central_x0 = max(x0, int(round(anchor - central_half)))
    central_x1 = min(x1, int(round(anchor + central_half)))
    central = int(foreign[:, central_x0:central_x1].sum()) \
        if central_x1 > central_x0 else 0

    return {"neighbour_body": body, "neighbour_ascender": ascender,
            "neighbour_in_central_zone": central}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=("decompose", "gate"), default="gate")
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    config = AnchorConfig()
    if args.config and args.config.is_file():
        stored = json.loads(args.config.read_text(encoding="utf-8"))
        config = AnchorConfig(**{
            **AnchorConfig().as_dict(),
            **{k: (tuple(v) if isinstance(v, list) else v)
               for k, v in stored.get("anchor", {}).items()
               if k in AnchorConfig().as_dict()},
        })
        print(f"config sha256: "
              f"{hashlib.sha256(args.config.read_bytes()).hexdigest()}")

    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    engine, package = build_engine(args.package, args.language)
    labels = load_labels(package.dictionary)
    recognizer = engine.text_rec
    channels, height, width = recognizer.rec_image_shape

    reasons = Counter()
    triggers = Counter()
    accepted_rows = []
    v4_decomposition = Counter()
    accepted_by_bucket = Counter()
    total_by_bucket = Counter()

    for sample in manifest["samples"]:
        line = cv2.imread(str(args.dataset / sample["line_file"]))
        if line is None:
            reasons["missing_image"] += 1
            continue
        ground = np.load(args.dataset / sample["gt_mask_file"])
        glyph_mask, accent_mask = ground["glyph_mask"], ground["accent_mask"]
        crop_h, crop_w = line.shape[:2]
        bucket = sample["bucket"]
        total_by_bucket[bucket] += 1

        max_wh_ratio = max(width / height, crop_w / crop_h)
        tensor = recognizer.resize_norm_img(line, max_wh_ratio)[np.newaxis, :]
        logits = np.asarray(recognizer.session(tensor.astype(np.float32))[0])
        probabilities = logits[0]
        decoded = recognizer.postprocess_op(
            logits, False, wh_ratio_list=[crop_w / float(crop_h)],
            max_wh_ratio=max_wh_ratio,
        )[0][0]
        emitted = collapse_ctc(probabilities.argmax(axis=-1).tolist(), labels)

        padded_w = int(height * max_wh_ratio)
        resized_w = min(padded_w, int(math.ceil(height * (crop_w / crop_h))))
        stride = (padded_w / probabilities.shape[0]) * (crop_w / resized_w)

        gt_x0, gt_x1 = sample["gt_glyph_x"]
        gt_centre = (gt_x0 + gt_x1) / 2.0
        gt_width = max(1e-6, gt_x1 - gt_x0)

        position = target_token(emitted, stride, gt_centre)
        if position is None:
            reasons[Reason.NO_TARGET_TOKEN] += 1
            triggers[f"visual_{sample['visual_label']}_no_token"] += 1
            continue

        predicted = unicodedata.normalize("NFC", emitted[position]["char"])
        visual = sample["visual_label"]
        if visual == "e" and predicted == "é":
            triggers["hallucination"] += 1
        elif visual == "é" and predicted == "é":
            triggers["preservation"] += 1
        else:
            triggers[f"visual_{visual}_predicted_e"] += 1

        centres = [((i["start"] + i["end"] + 1) / 2.0) * stride for i in emitted]

        # v4 decomposition: which path would v4 have taken, and was it right?
        if args.mode == "decompose":
            gaps = [b - a for a, b in zip(centres, centres[1:]) if b > a]
            pitch = float(statistics.median(gaps)) if gaps else 0.0
            v4_anchors = anchor_candidates(emitted, position, probabilities,
                                           stride, pitch)
            used_fallback = v4_anchors["consensus"] is None
            value = v4_anchors["consensus"] or v4_anchors["posterior_center"]
            neighbours = [centres[position + o] for o in (-1, 1)
                          if 0 <= position + o < len(centres)]
            misassigned = any(abs(value - n) < abs(value - gt_centre)
                              for n in neighbours)
            if misassigned:
                key = ("fallback_misassigned" if used_fallback
                       else "consensus_misassigned")
                v4_decomposition[key] += 1
                v4_decomposition[f"{key}:{bucket}"] += 1
                v4_decomposition[f"{key}:font={sample['font']}"] += 1
                repeated = any(
                    0 <= position + o < len(emitted)
                    and unicodedata.normalize("NFC", emitted[position + o]["char"])
                    == predicted
                    for o in (-1, 1)
                )
                if repeated:
                    v4_decomposition[f"{key}:repeated_character"] += 1
            else:
                v4_decomposition["fallback_ok" if used_fallback
                                 else "consensus_ok"] += 1

        result = locate_target(line, emitted, decoded, position, probabilities,
                               stride, config)
        reasons[result.reason] += 1
        if not result.accepted:
            continue

        accepted_by_bucket[bucket] += 1
        contain = containment(result.patch, glyph_mask, accent_mask, line, crop_w)
        neighbour = neighbour_intrusion(
            result.patch, glyph_mask, accent_mask, line, crop_w,
            result.anchor_x, config.central_zone_ratio * result.pitch,
        )
        neighbours = [centres[position + o] for o in (-1, 1)
                      if 0 <= position + o < len(centres)]
        accepted_rows.append({
            "bucket": bucket, "font": sample["font"],
            "visual_label": visual, "predicted": predicted,
            "center_error_norm": abs(result.anchor_x - gt_centre) / gt_width,
            "misassigned": any(abs(result.anchor_x - n)
                               < abs(result.anchor_x - gt_centre)
                               for n in neighbours),
            "clipped": result.patch[0] < 0 or result.patch[1] > crop_w,
            **contain, **neighbour,
        })

    scored = sum(reasons.values())
    accepted = len(accepted_rows)
    print(f"\nsamples: {scored}   accepted: {accepted} "
          f"({100 * accepted / max(1, scored):.1f}%)")
    print(f"triggers: {dict(triggers)}")
    print("\nUNKNOWN reasons:")
    for reason, count in reasons.most_common():
        if reason != Reason.ACCEPTED:
            print(f"  {reason:34s} {count:>5d}")

    if args.mode == "decompose":
        print("\nv4 accepted-misassignment decomposition:")
        for key in sorted(v4_decomposition):
            print(f"  {key:44s} {v4_decomposition[key]:>4d}")

    summary = {}
    if accepted_rows:
        glyph = np.array([r["glyph"] for r in accepted_rows])
        accent = np.array([r["accent"] for r in accepted_rows
                           if r["accent"] is not None])
        intrusion = np.array([r["intrusion"] for r in accepted_rows])
        summary = {
            "accepted": accepted,
            "misassigned": int(sum(r["misassigned"] for r in accepted_rows)),
            "clipped": int(sum(r["clipped"] for r in accepted_rows)),
            "glyph_below_95": int((glyph < 0.95).sum()),
            "accent_below_95": int((accent < 0.95).sum()) if accent.size else 0,
            "glyph_median": float(np.median(glyph)),
            "accent_median": float(np.median(accent)) if accent.size else None,
            "intrusion_positive": int((intrusion > 0).sum()),
            "intrusion_p95": float(np.percentile(intrusion, 95)),
            "intrusion_max": float(intrusion.max()),
            "neighbour_body_positive": int(sum(
                r["neighbour_body"] > 0 for r in accepted_rows)),
            "neighbour_ascender_positive": int(sum(
                r["neighbour_ascender"] > 0 for r in accepted_rows)),
            "neighbour_in_central_zone": int(sum(
                r["neighbour_in_central_zone"] > 0 for r in accepted_rows)),
        }
        print("\naccepted-sample quality:")
        for key, value in summary.items():
            print(f"  {key:32s} {value}")

    print("\naccepted coverage by ink-height bucket:")
    coverage = {}
    for bucket in sorted(total_by_bucket):
        total = total_by_bucket[bucket]
        got = accepted_by_bucket[bucket]
        coverage[bucket] = got / total if total else 0.0
        print(f"  {bucket:>7s}: {got:>5d}/{total:<5d} ({100 * coverage[bucket]:>5.1f}%)")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({
            "dataset": str(args.dataset), "mode": args.mode,
            "config": config.as_dict(), "scored": scored,
            "reasons": dict(reasons), "triggers": dict(triggers),
            "v4_decomposition": dict(v4_decomposition),
            "accepted_summary": summary,
            "coverage_by_bucket": coverage,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
