"""Compare x-anchor and context-patch strategies against rendered ground truth.

Stage 3D-0 established that a CTC token span is not a glyph bounding box: one
timestep covers ``8 * line_height / 48`` crop pixels, so a span's width tracks
line height rather than glyph width, and the fixed 4px pad dominates at small
sizes. This drops the span-as-box idea entirely.

Instead the CTC output is used only for *where* the character is -- an x-anchor
-- and the crop around it is sized from measurable page geometry: the line's
ink height, or the robust spacing between neighbouring character anchors.

Anchor candidates are scored on how close they land to the true character
centre and on whether they land nearer the target than its neighbours. Patch
candidates are scored on how much of the target's body and accent they contain,
and on how much neighbouring ink they drag in. Intrusion is reported as a
distribution, not a median: a median of zero hides that a quarter of crops
contain a neighbour.

Ground truth is only ever used to score a proposal, never inside one. No
expected text, no real UI image, and no CNN output is involved.
"""

from __future__ import annotations

import argparse
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

ANCHORS = (
    "argmax_center",
    "posterior_center",
    "viterbi_center",
    "blank_valley_center",
    "pitch_center",
    "consensus",
)
PATCHES = (
    "fixed_pad_4px",        # what accent-v3 used, kept only as a baseline
    "ink_height_scaled",
    "pitch_scaled",
    "bounded_combined",
    "multi_scale",
)

# Anchors must agree within this fraction of the local character pitch for the
# consensus anchor to fire. Chosen from geometry, not fitted to any sample.
CONSENSUS_TOLERANCE = 0.35


def collapse_ctc(argmax, labels):
    emitted, previous = [], 0
    for timestep, index in enumerate(argmax):
        if index != 0 and index != previous:
            emitted.append({"char": labels[index] if index < len(labels) else "?",
                            "start": timestep, "end": timestep, "label": index})
        elif emitted and index == previous and index != 0:
            emitted[-1]["end"] = timestep
        previous = index
    return emitted


def ink_mask(image: np.ndarray) -> np.ndarray:
    gray = image[..., :3].mean(axis=2) if image.ndim == 3 else image.astype(float)
    low, high = float(gray.min()), float(gray.max())
    if high - low < 12.0:
        return np.zeros(gray.shape, dtype=bool)
    bright = gray >= (low + high) / 2.0
    return bright if bright.mean() < 0.5 else ~bright


def anchor_candidates(
    emitted: list[dict], position: int, probabilities: np.ndarray,
    scale: float, pitch: float,
) -> dict[str, float | None]:
    """x positions, in crop pixels, for the target character's centre."""
    item = emitted[position]
    out: dict[str, float | None] = {}

    # 1. Centre of the collapsed argmax token.
    out["argmax_center"] = ((item["start"] + item["end"] + 1) / 2.0) * scale

    # 2. Posterior-weighted centre: timesteps weighted by this label's
    #    probability, which respects a soft, spread-out emission.
    window = slice(max(0, item["start"] - 2),
                   min(probabilities.shape[0], item["end"] + 3))
    weights = probabilities[window, item["label"]]
    if weights.sum() > 1e-9:
        indices = np.arange(window.start, window.stop) + 0.5
        out["posterior_center"] = float((indices * weights).sum() / weights.sum()) * scale
    else:
        out["posterior_center"] = out["argmax_center"]

    # 3. Peak-probability timestep, the single most confident emission point.
    peak = int(np.argmax(probabilities[window, item["label"]])) + window.start
    out["viterbi_center"] = (peak + 0.5) * scale

    # 4. Midpoint between the blank valleys either side of the emission.
    blank = probabilities[:, 0]
    left = item["start"]
    while left > 0 and blank[left - 1] >= blank[left]:
        left -= 1
    right = item["end"]
    while right + 1 < probabilities.shape[0] and blank[right + 1] >= blank[right]:
        right += 1
    out["blank_valley_center"] = ((left + right + 1) / 2.0) * scale

    # 5. Position implied by the neighbouring anchors and the local pitch,
    #    which is robust when one token's own emission is smeared.
    if position > 0 and position + 1 < len(emitted):
        previous = ((emitted[position - 1]["start"]
                     + emitted[position - 1]["end"] + 1) / 2.0) * scale
        following = ((emitted[position + 1]["start"]
                      + emitted[position + 1]["end"] + 1) / 2.0) * scale
        out["pitch_center"] = (previous + following) / 2.0
    else:
        out["pitch_center"] = out["argmax_center"]

    # 6. Consensus: only when the independent estimates agree within a
    #    fraction of the local pitch. Otherwise abstain.
    values = [out[name] for name in
              ("argmax_center", "posterior_center", "viterbi_center",
               "blank_valley_center")
              if out[name] is not None]
    if values and pitch > 0 and (max(values) - min(values)) <= CONSENSUS_TOLERANCE * pitch:
        out["consensus"] = float(statistics.median(values))
    else:
        out["consensus"] = None
    return out


def patch_candidates(
    anchor: float, ink_height: float, pitch: float, crop_w: int,
) -> dict[str, tuple[float, float] | list[tuple[float, float]] | None]:
    """Half-widths around the anchor, from geometry rather than token width."""
    out: dict = {}
    out["fixed_pad_4px"] = (anchor - 4.0, anchor + 4.0)

    # A lowercase letter is roughly 0.55 of the line's ink height wide; take a
    # little more so the accent and both sidebearings are inside.
    half_ink = 0.48 * ink_height
    out["ink_height_scaled"] = (anchor - half_ink, anchor + half_ink)

    half_pitch = 0.62 * pitch if pitch > 0 else half_ink
    out["pitch_scaled"] = (anchor - half_pitch, anchor + half_pitch)

    # Combine the two, bounded so neither can run away on its own.
    combined = min(max(min(half_ink, half_pitch), 0.30 * ink_height),
                   0.90 * ink_height)
    out["bounded_combined"] = (anchor - combined, anchor + combined)

    out["multi_scale"] = [
        (anchor - h, anchor + h)
        for h in (0.38 * ink_height, 0.48 * ink_height, 0.62 * ink_height)
    ]
    return out


def score_patch(
    span: tuple[float, float], glyph_mask: np.ndarray, accent_mask: np.ndarray,
    line: np.ndarray, crop_w: int,
) -> dict:
    x0 = max(0, int(round(span[0])))
    x1 = min(crop_w, int(round(span[1])))
    if x1 - x0 < 2:
        return {"abstained": True, "reason": "degenerate"}

    columns = slice(x0, x1)
    glyph_total = max(1, int(glyph_mask.sum()))
    glyph_inside = int(glyph_mask[:, columns].sum())
    accent_total = int(accent_mask.sum())
    accent_inside = int(accent_mask[:, columns].sum())

    mask = ink_mask(line)
    patch_ink = int(mask[:, columns].sum())
    # Ink in the patch that belongs to neither the target body nor its accent.
    target_ink = int((glyph_mask | accent_mask)[:, columns].sum())
    intrusion_pixels = max(0, patch_ink - target_ink)

    sub = mask[:, columns]
    touches_left = bool(sub[:, 0].any()) if sub.shape[1] else False
    touches_right = bool(sub[:, -1].any()) if sub.shape[1] else False

    return {
        "abstained": False,
        "glyph_containment": glyph_inside / glyph_total,
        "accent_containment": (accent_inside / accent_total)
        if accent_total else None,
        "intrusion": intrusion_pixels / max(1, patch_ink),
        "intrusion_pixels": intrusion_pixels,
        "touches_both_edges": touches_left and touches_right,
        "clipped_at_image_edge": span[0] < 0 or span[1] > crop_w,
        "width": x1 - x0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    engine, package = build_engine(args.package, args.language)
    labels = load_labels(package.dictionary)
    recognizer = engine.text_rec
    channels, height, width = recognizer.rec_image_shape

    anchor_rows = {name: [] for name in ANCHORS}
    patch_rows = {name: [] for name in PATCHES}
    trigger = Counter()
    skipped = Counter()
    scored = 0

    for sample in manifest["samples"]:
        line = cv2.imread(str(args.dataset / sample["line_file"]))
        if line is None:
            skipped["missing_image"] += 1
            continue
        ground = np.load(args.dataset / sample["gt_mask_file"])
        glyph_mask = ground["glyph_mask"]
        accent_mask = ground["accent_mask"]
        crop_h, crop_w = line.shape[:2]

        max_wh_ratio = max(width / height, crop_w / crop_h)
        tensor = recognizer.resize_norm_img(line, max_wh_ratio)[np.newaxis, :]
        logits = np.asarray(recognizer.session(tensor.astype(np.float32))[0])
        probabilities = logits[0]
        argmax = probabilities.argmax(axis=-1).tolist()
        decoded = recognizer.postprocess_op(
            logits, False, wh_ratio_list=[crop_w / float(crop_h)],
            max_wh_ratio=max_wh_ratio,
        )[0][0]
        emitted = collapse_ctc(argmax, labels)
        if "".join(i["char"] for i in emitted) != decoded:
            skipped["ctc_sequence_mismatch"] += 1
            continue

        padded_w = int(height * max_wh_ratio)
        resized_w = min(padded_w, int(math.ceil(height * (crop_w / crop_h))))
        scale = (padded_w / probabilities.shape[0]) * (crop_w / resized_w)

        gt_x0, gt_x1 = sample["gt_glyph_x"]
        gt_centre = (gt_x0 + gt_x1) / 2.0
        visual = sample["visual_label"]

        # Find the emitted e/é token nearest the true centre. Ground truth is
        # used to identify which token to score, never inside a candidate.
        best, best_distance = None, None
        for position, item in enumerate(emitted):
            if unicodedata.normalize("NFC", item["char"]) not in {"e", "é"}:
                continue
            centre = ((item["start"] + item["end"] + 1) / 2.0) * scale
            distance = abs(centre - gt_centre)
            if best_distance is None or distance < best_distance:
                best, best_distance = position, distance
        if best is None:
            skipped["no_matching_token"] += 1
            trigger[f"visual_{visual}_no_token"] += 1
            continue

        predicted = unicodedata.normalize("NFC", emitted[best]["char"])
        if visual == "e" and predicted == "é":
            trigger["hallucination_visual_e_predicted_accent"] += 1
        elif visual == "é" and predicted == "é":
            trigger["preservation_visual_accent_predicted_accent"] += 1
        elif predicted == "e":
            trigger[f"visual_{visual}_predicted_e"] += 1

        # Local character pitch from neighbouring token centres, which is
        # measurable at runtime without knowing the text.
        centres = [((e["start"] + e["end"] + 1) / 2.0) * scale for e in emitted]
        gaps = [b - a for a, b in zip(centres, centres[1:]) if b > a]
        pitch = float(statistics.median(gaps)) if gaps else 0.0

        ink = ink_mask(line)
        rows_with_ink = np.where(ink.any(axis=1))[0]
        ink_height = float(rows_with_ink[-1] - rows_with_ink[0] + 1) \
            if rows_with_ink.size else float(crop_h)

        scored += 1
        anchors = anchor_candidates(emitted, best, probabilities, scale, pitch)
        gt_width = max(1e-6, gt_x1 - gt_x0)

        # Neighbour centres, to detect an anchor landing on the wrong glyph.
        neighbour_centres = []
        for offset in (-1, 1):
            index = best + offset
            if 0 <= index < len(emitted):
                neighbour_centres.append(centres[index])

        for name, value in anchors.items():
            if value is None:
                anchor_rows[name].append({"abstained": True, **_meta(sample)})
                continue
            misassigned = any(
                abs(value - other) < abs(value - gt_centre)
                for other in neighbour_centres
            )
            anchor_rows[name].append({
                "abstained": False,
                "center_error_norm": (value - gt_centre) / gt_width,
                "misassigned": misassigned,
                **_meta(sample),
            })

        # Patches are built on the consensus anchor when it fires, else the
        # posterior centre; both are runtime-computable.
        anchor_value = anchors["consensus"] or anchors["posterior_center"]
        proposals = patch_candidates(anchor_value, ink_height, pitch, crop_w)
        for name, proposal in proposals.items():
            if name == "multi_scale":
                scores = [score_patch(s, glyph_mask, accent_mask, line, crop_w)
                          for s in proposal]
                usable = [s for s in scores if not s["abstained"]]
                if not usable:
                    patch_rows[name].append({"abstained": True, **_meta(sample)})
                    continue
                # Multi-scale accepts only when every scale contains the target,
                # which is the fail-closed reading of "the scales agree".
                entry = min(usable, key=lambda s: s["glyph_containment"])
                entry = dict(entry)
                entry["scales_agree"] = all(
                    s["glyph_containment"] >= 0.95 for s in usable)
                patch_rows[name].append({**entry, **_meta(sample)})
            else:
                entry = score_patch(proposal, glyph_mask, accent_mask, line, crop_w)
                patch_rows[name].append({**entry, **_meta(sample)})

    print(f"scored: {scored}")
    print(f"skipped: {dict(skipped)}")
    print(f"trigger classes: {dict(trigger)}\n")

    print(f"{'anchor':22s} {'n':>5s} {'abst':>5s} {'p50':>7s} {'p90':>7s} "
          f"{'p95':>7s} {'p99':>7s} {'max':>7s} {'misassigned':>12s}")
    anchor_summary = {}
    for name in ANCHORS:
        rows = anchor_rows[name]
        usable = [r for r in rows if not r["abstained"]]
        if not usable:
            print(f"{name:22s} {0:>5d} {len(rows):>5d}   (all abstained)")
            anchor_summary[name] = {"n": 0, "abstained": len(rows)}
            continue
        errors = np.abs([r["center_error_norm"] for r in usable])
        misassigned = sum(r["misassigned"] for r in usable)
        anchor_summary[name] = {
            "n": len(usable), "abstained": len(rows) - len(usable),
            "p50": float(np.percentile(errors, 50)),
            "p90": float(np.percentile(errors, 90)),
            "p95": float(np.percentile(errors, 95)),
            "p99": float(np.percentile(errors, 99)),
            "max": float(errors.max()),
            "misassigned": misassigned,
        }
        s = anchor_summary[name]
        print(f"{name:22s} {s['n']:>5d} {s['abstained']:>5d} {s['p50']:>7.3f} "
              f"{s['p90']:>7.3f} {s['p95']:>7.3f} {s['p99']:>7.3f} "
              f"{s['max']:>7.3f} {misassigned:>12d}")

    print(f"\n{'patch':20s} {'n':>5s} {'abst':>5s} {'glyph':>7s} {'gly<95':>7s} "
          f"{'accent':>7s} {'acc<95':>7s} {'intr>0':>7s} {'iP95':>6s} "
          f"{'iMax':>6s} {'edge':>6s}")
    patch_summary = {}
    for name in PATCHES:
        rows = patch_rows[name]
        usable = [r for r in rows if not r["abstained"]]
        if not usable:
            print(f"{name:20s} {0:>5d} {len(rows):>5d}   (all abstained)")
            patch_summary[name] = {"n": 0, "abstained": len(rows)}
            continue
        glyph = np.array([r["glyph_containment"] for r in usable])
        accent = np.array([r["accent_containment"] for r in usable
                           if r["accent_containment"] is not None])
        intrusion = np.array([r["intrusion"] for r in usable])
        patch_summary[name] = {
            "n": len(usable), "abstained": len(rows) - len(usable),
            "glyph_median": float(np.median(glyph)),
            "glyph_below_95": int((glyph < 0.95).sum()),
            "accent_median": float(np.median(accent)) if accent.size else None,
            "accent_below_95": int((accent < 0.95).sum()) if accent.size else 0,
            "intrusion_positive": int((intrusion > 0).sum()),
            "intrusion_p95": float(np.percentile(intrusion, 95)),
            "intrusion_p99": float(np.percentile(intrusion, 99)),
            "intrusion_max": float(intrusion.max()),
            "touches_both_edges": int(sum(r["touches_both_edges"] for r in usable)),
            "clipped_at_edge": int(sum(r["clipped_at_image_edge"] for r in usable)),
        }
        s = patch_summary[name]
        accent_median = s["accent_median"]
        print(f"{name:20s} {s['n']:>5d} {s['abstained']:>5d} "
              f"{s['glyph_median']:>7.3f} {s['glyph_below_95']:>7d} "
              f"{(accent_median if accent_median is not None else float('nan')):>7.3f} "
              f"{s['accent_below_95']:>7d} {s['intrusion_positive']:>7d} "
              f"{s['intrusion_p95']:>6.3f} {s['intrusion_max']:>6.3f} "
              f"{s['touches_both_edges']:>6d}")

    # Per-bucket view for the leading patch candidates.
    print("\nglyph containment by ink-height bucket:")
    buckets = sorted({r["bucket"] for r in patch_rows["fixed_pad_4px"]})
    header = "  ".join(f"{b:>9s}" for b in buckets)
    print(f"{'patch':20s} {header}")
    by_bucket = {}
    for name in PATCHES:
        usable = [r for r in patch_rows[name] if not r["abstained"]]
        cells, store = [], {}
        for bucket in buckets:
            values = [r["glyph_containment"] for r in usable if r["bucket"] == bucket]
            store[bucket] = float(np.median(values)) if values else None
            cells.append(f"{store[bucket]:>9.3f}" if values else f"{'-':>9s}")
        by_bucket[name] = store
        print(f"{name:20s} {'  '.join(cells)}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({
            "dataset": str(args.dataset), "scored": scored,
            "skipped": dict(skipped), "trigger_classes": dict(trigger),
            "anchor_summary": anchor_summary, "patch_summary": patch_summary,
            "patch_by_bucket": by_bucket,
            "consensus_tolerance": CONSENSUS_TOLERANCE,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out_json}")
    return 0


def _meta(sample: dict) -> dict:
    return {"font": sample["font"], "bucket": sample["bucket"],
            "ink_height": sample["ink_height"],
            "visual_label": sample["visual_label"]}


if __name__ == "__main__":
    raise SystemExit(main())
