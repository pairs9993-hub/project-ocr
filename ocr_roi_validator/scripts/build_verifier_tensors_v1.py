"""Materialise verifier input tensors from the sealed recipes.

No images were stored during generation -- only the settings needed to redraw
them. That is deliberate: every rendering is a deterministic function of its
recipe and index, so the pixels are reproducible without keeping 15,000 files
around, and the parity checks throughout this work rely on that property.

Here those renderings are replayed, pushed through the frozen detector and
recognizer, and converted into the Stage 3E-0 input contract: three planes and
a two-number query. The decoded string shapes the CTC position map's geometry
and is then discarded; it never reaches the tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
for extra in (VALIDATOR_ROOT, VALIDATOR_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from generate_counterfactual_v1 import Rendering as CFRendering  # noqa: E402
from generate_counterfactual_v1 import render as cf_render  # noqa: E402
from mine_line_triggers_v1 import build_engine, collapse_ctc, load_labels  # noqa: E402
from ocr_roi_validator.counterfactual_recipes import COUNTERFACTUAL_RECIPES  # noqa: E402
from ocr_roi_validator.diagnostic_runner import atomic_write_json, log_line  # noqa: E402
from ocr_roi_validator.line_verifier_input import (  # noqa: E402
    LineVerifierInputConfig, assert_no_text_leakage, build_line_input,
)

CLASS_INDEX = {"ACCENT_PRESENT": 0, "BARE_E": 1, "UNKNOWN": 2}


def target_centre_x(rendering, font_dir):
    """Centre of the target glyph in page pixels, from the renderer's layout.

    Used to pick the same detector box the generator chose, and to supervise
    attention. It is a training signal and a selection rule, never an input.
    """
    font = ImageFont.truetype(str(font_dir / rendering.font), rendering.size)
    draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    text = unicodedata.normalize("NFC", rendering.text)
    position = rendering.target_position()
    origin_x = rendering.pad_x + rendering.jitter_x
    before = draw.textlength(text[:position], font=font)
    width = draw.textlength(text[position], font=font)
    return (origin_x + before + width / 2.0) * rendering.upscale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--split", choices=tuple(COUNTERFACTUAL_RECIPES),
                        required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()

    recipe = COUNTERFACTUAL_RECIPES[args.split]
    rows = [json.loads(line) for line
            in (args.source / "checkpoint.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
    usable = [r for r in rows if r["usable"]]
    log_line("%s: %d usable of %d" % (args.split, len(usable), len(rows)))

    engine, package = build_engine(args.package, args.language)
    labels = load_labels(package.dictionary)
    recognizer = engine.text_rec
    _, rec_height, rec_width = recognizer.rec_image_shape
    config = LineVerifierInputConfig()

    planes, queries, targets, centres = [], [], [], []
    kept, dropped = [], Counter()
    started = time.time()

    for position, row in enumerate(usable):
        rendering = CFRendering(row["index"], recipe)
        page = cf_render(rendering, args.font_dir)
        bgr = np.asarray(page)[:, :, ::-1].copy()
        boxes, _ = engine.auto_text_det(bgr)
        if boxes is None or len(boxes) == 0:
            dropped["DETECTOR_MISS_ON_REPLAY"] += 1
            continue
        crops = engine.get_crop_img_list(bgr, boxes)

        centre_x = target_centre_x(rendering, args.font_dir)
        chosen = None
        for order, box in enumerate(boxes):
            xs = [point[0] for point in box]
            if min(xs) <= centre_x <= max(xs):
                chosen = order
                break
        if chosen is None:
            dropped["TARGET_NOT_IN_ANY_BOX"] += 1
            continue

        crop = crops[chosen]
        crop_height, crop_width = crop.shape[:2]
        if [int(crop_width), int(crop_height)] != [row["crop_width"],
                                                   row["crop_height"]]:
            dropped["CROP_SIZE_MISMATCH"] += 1
            continue

        max_wh_ratio = max(rec_width / rec_height, crop_width / float(crop_height))
        tensor = recognizer.resize_norm_img(crop, max_wh_ratio)[np.newaxis]
        logits = np.asarray(recognizer.session(tensor.astype(np.float32))[0])
        probabilities = logits[0]
        emitted = collapse_ctc(probabilities.argmax(axis=-1).tolist(), labels)

        ordinal = row["query_ordinal"]
        token_count = row["query_token_count"]
        if 0 <= ordinal < len(emitted):
            token = emitted[ordinal]
            token_label = token["label"]
            token_start, token_end = token["start"], token["end"]
        else:
            # An out-of-range query has no token. The map is placed at the edge
            # so the input stays well formed and the model must answer UNKNOWN
            # from the query itself rather than from a malformed plane.
            token_label = 0
            token_start = token_end = max(0, len(probabilities) - 1)

        prepared = build_line_input(
            crop, probabilities, token_label=token_label,
            token_start=token_start, token_end=token_end,
            target_ordinal=ordinal, decoded_length=max(1, token_count),
            config=config)
        if prepared is None:
            dropped["INPUT_BUILD_FAILED"] += 1
            continue
        assert_no_text_leakage(prepared)

        xs = [point[0] for point in boxes[chosen]]
        low, high = min(xs), max(xs)
        span = max(1.0, high - low)

        planes.append(prepared.planes)
        queries.append(prepared.query)
        targets.append(CLASS_INDEX[row["label"]])
        centres.append(float(np.clip((centre_x - low) / span, 0.0, 1.0)))
        kept.append({"index": row["index"], "row_digest": row["row_digest"],
                     "label": row["label"], "pair_id": row["pair_id"],
                     "context_index": row["context_index"],
                     "word_bare": row["word_bare"], "kind": row["kind"],
                     "unknown_kind": row["unknown_kind"], "font": row["font"]})

        if (position + 1) % args.progress_every == 0:
            log_line("  %d/%d kept=%d %.0fs"
                     % (position + 1, len(usable), len(kept),
                        time.time() - started))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stacked = np.stack(planes).astype(np.float32)
    np.savez_compressed(
        args.out, planes=stacked,
        query=np.stack(queries).astype(np.float32),
        label=np.array(targets, dtype=np.int64),
        target_centre=np.array(centres, dtype=np.float32))

    manifest = {
        "dataset": args.split + "_tensors",
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_checkpoint_sha256": hashlib.sha256(
            (args.source / "checkpoint.jsonl").read_bytes()).hexdigest(),
        "usable_rows": len(usable), "tensors": len(kept),
        "dropped": dict(dropped),
        "class_counts": dict(Counter(r["label"] for r in kept)),
        "npz_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "plane_shape": list(stacked.shape[1:]),
        "channel_order": ["line_image", "ctc_position_map", "valid_width_mask"],
        "query_contract": "ordinal and token count only; no character, no text",
        "rows": kept,
    }
    digest = atomic_write_json(args.out.with_suffix(".manifest.json"), manifest)
    log_line("  tensors %d  classes %s" % (len(kept), manifest["class_counts"]))
    log_line("  dropped %s" % dict(dropped))
    log_line("  npz sha256 %s" % manifest["npz_sha256"][:32])
    log_line("  manifest sha256 %s" % digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
