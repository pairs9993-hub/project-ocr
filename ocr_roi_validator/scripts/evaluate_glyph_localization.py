"""Compare glyph localization strategies against rendered ground truth.

The accent-v3 F0 failure was that the localization guard rejected every target
glyph before the classifier ran. Whether the CTC span is genuinely too wide, or
whether the timestep-to-pixel mapping is wrong, or whether a token span simply
is not a bounding box, cannot be settled by looking at the target -- it needs
ground truth. This measures each candidate against boxes the renderer knows.

Candidates:

``argmax_span``      what accent-v3 uses: the collapsed argmax token span,
                     mapped through the padded tensor width.
``token_midpoint``   boundaries placed halfway between neighbouring token
                     centres, which does not assume a token covers a box.
``blank_valley``     boundaries pushed out to the surrounding blank-probability
                     peaks, i.e. where the model is confident nothing is.
``center_projection`` CTC centre as an anchor, with edges found from the
                     vertical-projection valleys of the actual ink.
``consensus``        the intersection of the above when they agree closely,
                     abstaining when they do not.

Metrics are normalized by the ground-truth box so they carry across fonts and
sizes. No expected text, no real UI image, and no CNN output is involved.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import unicodedata
from collections import defaultdict
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

CANDIDATES = (
    "argmax_span",
    "token_midpoint",
    "blank_valley",
    "center_projection",
    "consensus",
)


def collapse_ctc(argmax, labels):
    emitted, previous = [], 0
    for timestep, index in enumerate(argmax):
        if index != 0 and index != previous:
            emitted.append({"char": labels[index] if index < len(labels) else "?",
                            "start": timestep, "end": timestep})
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


def propose(
    emitted: list[dict],
    position: int,
    probabilities: np.ndarray,
    scale: float,
    crop: np.ndarray,
    crop_w: int,
) -> dict[str, tuple[float, float] | None]:
    """Every candidate's x-range for one emitted token, in crop pixels."""
    item = emitted[position]
    spans: dict[str, tuple[float, float] | None] = {}

    # 1. The current approach: token span, plus the fixed 4px pad v3 applies.
    start_x = (item["start"] + 0.5) * scale
    end_x = (item["end"] + 1 + 0.5) * scale
    spans["argmax_span"] = (max(0.0, start_x - 4), min(float(crop_w), end_x + 4))

    # 2. Midpoints between neighbouring token centres. A token marks *where* a
    #    character was emitted, not how wide it is, so the boundary to its
    #    neighbour is a better estimate of the character's extent.
    def centre(index: int) -> float:
        entry = emitted[index]
        return ((entry["start"] + entry["end"] + 1) / 2.0) * scale

    here = centre(position)
    left = (here + centre(position - 1)) / 2.0 if position > 0 else 0.0
    right = (
        (here + centre(position + 1)) / 2.0
        if position + 1 < len(emitted) else float(crop_w)
    )
    spans["token_midpoint"] = (max(0.0, left), min(float(crop_w), right))

    # 3. Blank valleys: walk outward to where blank probability peaks, which is
    #    where the model is most confident no character is present.
    blank = probabilities[:, 0]
    timesteps = probabilities.shape[0]
    left_t = item["start"]
    while left_t > 0 and blank[left_t - 1] >= blank[left_t]:
        left_t -= 1
    right_t = item["end"]
    while right_t + 1 < timesteps and blank[right_t + 1] >= blank[right_t]:
        right_t += 1
    spans["blank_valley"] = (
        max(0.0, (left_t + 0.5) * scale),
        min(float(crop_w), (right_t + 1 + 0.5) * scale),
    )

    # 4. CTC centre as an anchor, edges from ink valleys in the projection.
    mask = ink_mask(crop)
    column_ink = mask.sum(axis=0).astype(float)
    anchor = int(round(min(max(here, 0.0), crop_w - 1.0)))
    if column_ink.any():
        left_edge = anchor
        while left_edge > 0 and column_ink[left_edge - 1] > 0:
            left_edge -= 1
        right_edge = anchor
        while right_edge + 1 < crop_w and column_ink[right_edge + 1] > 0:
            right_edge += 1
        spans["center_projection"] = (float(left_edge), float(right_edge + 1))
    else:
        spans["center_projection"] = None

    # 5. Consensus: only when the independent proposals agree closely, take
    #    their intersection; otherwise abstain.
    usable = [
        spans[name] for name in ("token_midpoint", "blank_valley", "center_projection")
        if spans[name] is not None
    ]
    if len(usable) >= 2:
        lefts = [s[0] for s in usable]
        rights = [s[1] for s in usable]
        widths = [r - l for l, r in usable]
        median_width = statistics.median(widths)
        spread = (max(rights) - min(lefts))
        if median_width > 0 and spread <= 2.2 * median_width:
            spans["consensus"] = (max(lefts), min(rights))
        else:
            spans["consensus"] = None
    else:
        spans["consensus"] = None
    return spans


def score(
    span: tuple[float, float] | None,
    gt_box: list[float],
    gt_mask: np.ndarray,
    accent_box: list[float] | None,
    crop: np.ndarray,
    crop_w: int,
) -> dict:
    """Normalized quality of one proposed span against the ground truth."""
    if span is None:
        return {"abstained": True}
    x0, x1 = span
    if x1 - x0 < 2:
        return {"abstained": True, "reason": "degenerate"}

    gt_x0, gt_y0, gt_x1, gt_y1 = gt_box
    gt_width = max(1e-6, gt_x1 - gt_x0)
    gt_centre = (gt_x0 + gt_x1) / 2.0
    centre = (x0 + x1) / 2.0

    columns = slice(max(0, int(round(x0))), min(crop_w, int(round(x1))))
    inside = gt_mask[:, columns].sum()
    total = max(1, gt_mask.sum())

    accent_containment = None
    if accent_box is not None:
        ax0, _, ax1, _ = accent_box
        overlap = max(0.0, min(x1, ax1) - max(x0, ax0))
        accent_containment = overlap / max(1e-6, ax1 - ax0)

    # Ink inside the span that is not the target glyph: a neighbour intruding.
    mask = ink_mask(crop)
    span_ink = mask[:, columns].sum()
    intrusion = max(0, int(span_ink) - int(inside)) / max(1, int(span_ink))

    sub = mask[:, columns]
    touches_left = bool(sub[:, 0].any()) if sub.shape[1] else False
    touches_right = bool(sub[:, -1].any()) if sub.shape[1] else False

    return {
        "abstained": False,
        "centre_error_norm": (centre - gt_centre) / gt_width,
        "width_ratio": (x1 - x0) / gt_width,
        "glyph_containment": float(inside) / float(total),
        "accent_containment": accent_containment,
        "intrusion": intrusion,
        "touches_both_edges": touches_left and touches_right,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    manifest = json.loads((args.gt_dir / "manifest.json").read_text(encoding="utf-8"))
    engine, package = build_engine(args.package, args.language)
    labels = load_labels(package.dictionary)
    recognizer = engine.text_rec
    channels, height, width = recognizer.rec_image_shape

    results = {name: [] for name in CANDIDATES}
    failures = defaultdict(int)
    scanned = 0

    for sample in manifest["samples"]:
        crop = cv2.imread(str(args.gt_dir / sample["line_file"]))
        if crop is None:
            failures["missing_image"] += 1
            continue
        gt_mask = np.load(args.gt_dir / sample["gt_mask_file"])["glyph_mask"]
        crop_h, crop_w = crop.shape[:2]

        # A single line, so the batch ratio is its own.
        max_wh_ratio = max(width / height, crop_w / crop_h)
        tensor = recognizer.resize_norm_img(crop, max_wh_ratio)[np.newaxis, :]
        logits = np.asarray(recognizer.session(tensor.astype(np.float32))[0])
        probabilities = logits[0]
        argmax = probabilities.argmax(axis=-1).tolist()
        decoded = recognizer.postprocess_op(
            logits, False, wh_ratio_list=[crop_w / float(crop_h)],
            max_wh_ratio=max_wh_ratio,
        )[0][0]
        emitted = collapse_ctc(argmax, labels)
        if "".join(i["char"] for i in emitted) != decoded:
            failures["ctc_path_mismatch"] += 1
            continue

        # Find the emitted token nearest the ground-truth centre, among tokens
        # of the right character class. Ground truth is used to *evaluate*
        # localization, never inside a candidate.
        gt_box = sample["gt_glyph_box_crop"]
        gt_centre = (gt_box[0] + gt_box[2]) / 2.0
        padded_w = int(height * max_wh_ratio)
        resized_w = min(padded_w, int(math.ceil(height * (crop_w / crop_h))))
        scale = (padded_w / probabilities.shape[0]) * (crop_w / resized_w)

        best, best_distance = None, None
        for position, item in enumerate(emitted):
            if unicodedata.normalize("NFC", item["char"]) not in {"e", "é"}:
                continue
            centre = ((item["start"] + item["end"] + 1) / 2.0) * scale
            distance = abs(centre - gt_centre)
            if best_distance is None or distance < best_distance:
                best, best_distance = position, distance
        if best is None:
            failures["no_matching_token"] += 1
            continue

        scanned += 1
        spans = propose(emitted, best, probabilities, scale, crop, crop_w)
        for name, span in spans.items():
            entry = score(span, gt_box, gt_mask,
                          sample.get("gt_accent_box_crop"), crop, crop_w)
            entry.update({"font": sample["font"], "size": sample["size"],
                          "visual_label": sample["visual_label"]})
            results[name].append(entry)

    print(f"samples scored: {scanned}")
    print(f"skipped: {dict(failures)}\n")

    def summarize(entries: list[dict]) -> dict:
        scored = [e for e in entries if not e["abstained"]]
        if not scored:
            return {"n": 0, "abstained": len(entries)}
        accent = [e["accent_containment"] for e in scored
                  if e["accent_containment"] is not None]
        return {
            "n": len(scored),
            "abstained": len(entries) - len(scored),
            "abstention_rate": (len(entries) - len(scored)) / max(1, len(entries)),
            "centre_error_median": statistics.median(
                abs(e["centre_error_norm"]) for e in scored),
            "width_ratio_median": statistics.median(e["width_ratio"] for e in scored),
            "width_ratio_p90": float(np.percentile(
                [e["width_ratio"] for e in scored], 90)),
            "glyph_containment_median": statistics.median(
                e["glyph_containment"] for e in scored),
            "glyph_containment_below_95": sum(
                e["glyph_containment"] < 0.95 for e in scored),
            "accent_containment_median": statistics.median(accent) if accent else None,
            "accent_containment_below_95": sum(a < 0.95 for a in accent),
            "intrusion_median": statistics.median(e["intrusion"] for e in scored),
            "touches_both_edges": sum(e["touches_both_edges"] for e in scored),
        }

    summary = {name: summarize(results[name]) for name in CANDIDATES}
    print(f"{'candidate':19s} {'n':>5s} {'abst':>5s} {'ctrErr':>7s} {'wRatio':>7s} "
          f"{'wR-p90':>7s} {'glyph':>7s} {'accent':>7s} {'intr':>6s} {'bothEdge':>8s}")
    for name in CANDIDATES:
        s = summary[name]
        if not s["n"]:
            print(f"{name:19s} {0:>5d} {s['abstained']:>5d}   (all abstained)")
            continue
        accent = s["accent_containment_median"]
        print(f"{name:19s} {s['n']:>5d} {s['abstained']:>5d} "
              f"{s['centre_error_median']:>7.3f} {s['width_ratio_median']:>7.3f} "
              f"{s['width_ratio_p90']:>7.3f} {s['glyph_containment_median']:>7.3f} "
              f"{(accent if accent is not None else float('nan')):>7.3f} "
              f"{s['intrusion_median']:>6.3f} {s['touches_both_edges']:>8d}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps({"scanned": scanned, "skipped": dict(failures),
                        "summary": summary,
                        "per_candidate": {k: v for k, v in results.items()}},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
