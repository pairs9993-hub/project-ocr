"""Build tensors from the calibration-v2 legitimate-accent preservation rows.

These are the safety anchor for the threshold. Unlike the counterfactual pairs
they are not synthetic constructions: a real accent was drawn, the frozen
baseline read it correctly, and a verifier that "corrects" one of them would
damage text that was already right. That is the failure the threshold exists to
prevent, so it is measured on real baseline output rather than on pairs built
for training.

calibration-v2 failed its hallucination quota and is barred from training and
from coverage evidence. Its preservation rows carry no such problem -- their
role was recorded as preservation_anchor_candidate when the split was retired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
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

from generate_v2_split import Rendering, render  # noqa: E402
from mine_line_triggers_v1 import build_engine, collapse_ctc, load_labels  # noqa: E402
from ocr_roi_validator.diagnostic_runner import atomic_write_json, log_line  # noqa: E402
from ocr_roi_validator.line_verifier_input import (  # noqa: E402
    LineVerifierInputConfig, assert_no_text_leakage, build_line_input,
)
from ocr_roi_validator.v2_recipes import V2_RECIPES  # noqa: E402

# Mirrors line_verifier_model.CLASS_INDEX. Duplicated deliberately: this
# script runs in the product venv, which has no torch, and importing the
# model module only to read three integers would make tensor building
# depend on the training environment.
CLASS_INDEX = {"ACCENT_PRESENT": 0, "BARE_E": 1, "UNKNOWN": 2}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()

    recipe = V2_RECIPES["line_calibration_v2"]
    rows = [json.loads(line) for line
            in (args.source / "checkpoint.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
    preservation = [r for r in rows if r.get("clean_preservation")]
    if args.limit:
        preservation = preservation[:args.limit]
    log_line("preservation anchor rows: %d" % len(preservation))

    engine, package = build_engine(args.package, args.language)
    labels = load_labels(package.dictionary)
    recognizer = engine.text_rec
    _, rec_height, rec_width = recognizer.rec_image_shape
    config = LineVerifierInputConfig()

    planes, queries, targets, centres = [], [], [], []
    kept, dropped = [], Counter()
    started = time.time()

    for position, row in enumerate(preservation):
        rendering = Rendering(row["index"], recipe)
        page = render(rendering, args.font_dir)
        bgr = np.asarray(page)[:, :, ::-1].copy()
        boxes, _ = engine.auto_text_det(bgr)
        if boxes is None or len(boxes) == 0:
            dropped["DETECTOR_MISS_ON_REPLAY"] += 1
            continue
        crops = engine.get_crop_img_list(bgr, boxes)

        drawn = unicodedata.normalize("NFC", rendering.text)
        target_position = rendering.target_position()
        font = ImageFont.truetype(str(args.font_dir / rendering.font),
                                  rendering.size)
        draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        origin_x = rendering.pad_x + rendering.jitter_x
        before = draw.textlength(drawn[:target_position], font=font)
        glyph = draw.textlength(drawn[target_position], font=font)
        centre_x = (origin_x + before + glyph / 2.0) * rendering.upscale

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

        # Locate the target's ordinal within the chosen box, the same way the
        # generator did, from the renderer's layout rather than the decode.
        xs = [point[0] for point in boxes[chosen]]
        low, high = min(xs), max(xs)
        ordinal, seen = -1, 0
        for order, character in enumerate(drawn):
            offset = draw.textlength(drawn[:order], font=font)
            width = draw.textlength(character, font=font)
            centre = (origin_x + offset + width / 2.0) * rendering.upscale
            if low <= centre <= high:
                if order == target_position:
                    ordinal = seen
                seen += 1
        if ordinal < 0 or ordinal >= len(emitted):
            dropped["TARGET_ORDINAL_UNUSABLE"] += 1
            continue

        token = emitted[ordinal]
        prepared = build_line_input(
            crop, probabilities, token_label=token["label"],
            token_start=token["start"], token_end=token["end"],
            target_ordinal=ordinal, decoded_length=len(emitted), config=config)
        if prepared is None:
            dropped["INPUT_BUILD_FAILED"] += 1
            continue
        assert_no_text_leakage(prepared)

        span = max(1.0, high - low)
        planes.append(prepared.planes)
        queries.append(prepared.query)
        targets.append(CLASS_INDEX["ACCENT_PRESENT"])
        centres.append(float(np.clip((centre_x - low) / span, 0.0, 1.0)))
        kept.append({"index": row["index"], "row_digest": row["row_digest"],
                     "font": row["font"], "measured_stratum": row.get("measured_stratum")})

        if (position + 1) % args.progress_every == 0:
            log_line("  %d/%d kept=%d %.0fs" % (position + 1, len(preservation),
                                                len(kept), time.time() - started))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stacked = np.stack(planes).astype(np.float32)
    np.savez_compressed(
        args.out, planes=stacked,
        query=np.stack(queries).astype(np.float32),
        label=np.array(targets, dtype=np.int64),
        target_centre=np.array(centres, dtype=np.float32))

    manifest = {
        "dataset": "calibration_v2_preservation_anchor_tensors",
        "role": "threshold safety anchor only; never training, never coverage",
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_checkpoint_sha256": hashlib.sha256(
            (args.source / "checkpoint.jsonl").read_bytes()).hexdigest(),
        "preservation_rows": len(preservation), "tensors": len(kept),
        "dropped": dict(dropped),
        "npz_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "fonts": dict(Counter(r["font"] for r in kept)),
        "strata": dict(Counter(r["measured_stratum"] for r in kept)),
        "rows": kept,
    }
    digest = atomic_write_json(args.out.with_suffix(".manifest.json"), manifest)
    log_line("  tensors %d  dropped %s" % (len(kept), dict(dropped)))
    log_line("  fonts %s" % manifest["fonts"])
    log_line("  npz sha256 %s" % manifest["npz_sha256"][:32])
    log_line("  manifest sha256 %s" % digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
