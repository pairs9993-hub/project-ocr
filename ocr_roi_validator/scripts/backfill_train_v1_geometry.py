"""Recover measured pixel geometry for line_train_v1 without touching it.

That dataset recorded nominal point size and the detector crop's dimensions,
neither of which is the ink height the domain analysis turned on. It did record
``render_seed`` alongside the font, size and drawn text, and the generator draws
every perturbation from a ``random.Random(render_seed)``, so the original page
is reproducible bit for bit.

That makes backfill a measurement rather than an estimate, but only if the
reproduction is verified. Each row is re-rendered, re-detected and re-decoded,
and the geometry is written only when the recomputed crop size and decoded text
match what was stored. A row that fails parity gets no stratum at all.

Explicitly not done here: using the detector box height as a stand-in for ink
height, inferring a stratum from point size, interpolating anything, or
rewriting the original manifest. The output is a new sidecar file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
for extra in (VALIDATOR_ROOT, VALIDATOR_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from erratum_interaction_v1 import macro_stratum  # noqa: E402
from mine_line_triggers_v1 import build_engine, render_phrase  # noqa: E402
from ocr_roi_validator.diagnostic_runner import (  # noqa: E402
    CheckpointWriter, atomic_write_json, load_checkpoint, log_line,
)
from ocr_roi_validator.glyph_geometry import (  # noqa: E402
    measure_line_geometry, measure_target_glyph,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only-hallucination", action="store_true")
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()

    source = args.source / "checkpoint.jsonl"
    rows = [json.loads(line) for line
            in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.only_hallucination:
        rows = [r for r in rows if r["classification"] == "CLEAN_HALLUCINATION"]
    if args.limit:
        rows = rows[:args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    state = load_checkpoint(args.out, unit_field="render_index",
                            digest_field="row_digest")
    done = state.digests
    log_line(f"{len(rows)} source rows, {len(done)} already backfilled")

    engine, package = build_engine(args.package, args.language)
    recognizer = engine.text_rec
    _, rec_height, rec_width = recognizer.rec_image_shape

    outcomes = Counter()
    strata = Counter()
    # Grouping by render_index means one page is rendered once even when it
    # produced several rows.
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["render_index"], []).append(row)

    with CheckpointWriter(args.out, done, flush_every=50) as writer:
        for position, (render_index, group) in enumerate(sorted(grouped.items())):
            if all(r["row_digest"] in done for r in group):
                continue
            reference = group[0]
            font_path = args.font_dir / reference["font"]
            try:
                page = render_phrase(
                    reference["drawn_text"], str(font_path), reference["size"],
                    random.Random(reference["render_seed"]))
                bgr = np.asarray(page)[:, :, ::-1].copy()
                boxes, _ = engine.auto_text_det(bgr)
            except Exception as error:               # pragma: no cover
                for row in group:
                    outcomes["RENDER_ERROR"] += 1
                    writer.append({"row_digest": row["row_digest"],
                                   "render_index": render_index,
                                   "parity": False, "reason": type(error).__name__,
                                   "macro_stratum": None})
                continue
            if boxes is None or len(boxes) == 0:
                for row in group:
                    outcomes["DETECTOR_MISS_ON_REPLAY"] += 1
                    writer.append({"row_digest": row["row_digest"],
                                   "render_index": render_index, "parity": False,
                                   "reason": "detector found nothing on replay",
                                   "macro_stratum": None})
                continue

            crops = engine.get_crop_img_list(bgr, boxes)
            ratios = [c.shape[1] / float(c.shape[0]) for c in crops]
            max_wh_ratio = max([rec_width / rec_height] + ratios)
            page_digest = hashlib.sha256(np.asarray(page).tobytes()).hexdigest()

            for row in group:
                if row["row_digest"] in done:
                    continue
                line_index = row["line_index"]
                if line_index >= len(crops):
                    outcomes["LINE_INDEX_OUT_OF_RANGE"] += 1
                    writer.append({"row_digest": row["row_digest"],
                                   "render_index": render_index, "parity": False,
                                   "reason": f"line {line_index} of {len(crops)}",
                                   "macro_stratum": None})
                    continue
                crop = crops[line_index]
                crop_height, crop_width = crop.shape[:2]

                # Parity gate one: the crop must come out the same size.
                if [int(crop_width), int(crop_height)] != list(row["crop_size"]):
                    outcomes["CROP_SIZE_MISMATCH"] += 1
                    writer.append({"row_digest": row["row_digest"],
                                   "render_index": render_index, "parity": False,
                                   "reason": "crop size differs from stored",
                                   "macro_stratum": None})
                    continue

                tensor = recognizer.resize_norm_img(crop, max_wh_ratio)[np.newaxis]
                logits = np.asarray(recognizer.session(tensor.astype(np.float32))[0])
                decoded = recognizer.postprocess_op(
                    logits, False, wh_ratio_list=[crop_width / float(crop_height)],
                    max_wh_ratio=max_wh_ratio)[0][0]

                # Parity gate two: the recognizer must say the same thing.
                if decoded != row["decoded_text"]:
                    outcomes["DECODE_MISMATCH"] += 1
                    writer.append({"row_digest": row["row_digest"],
                                   "render_index": render_index, "parity": False,
                                   "reason": "decoded text differs from stored",
                                   "macro_stratum": None})
                    continue

                geometry = measure_line_geometry(crop, rec_height)
                if geometry is None:
                    outcomes["NO_INK_MEASURED"] += 1
                    writer.append({"row_digest": row["row_digest"],
                                   "render_index": render_index, "parity": True,
                                   "reason": "crop carries no measurable ink",
                                   "macro_stratum": None})
                    continue
                target = "é" if row["classification"] == "CLEAN_PRESERVATION" else "e"
                glyph = measure_target_glyph(target, str(font_path),
                                             row["size"], geometry)
                stratum = macro_stratum(geometry.ink_height)
                outcomes["BACKFILLED"] += 1
                strata[stratum] += 1
                writer.append({
                    "row_digest": row["row_digest"],
                    "render_index": render_index,
                    "line_index": line_index,
                    "parity": True,
                    "reason": None,
                    "rendered_page_sha256": page_digest,
                    "crop_width": int(crop_width),
                    "crop_height": int(crop_height),
                    "runtime_ink_height": geometry.ink_height,
                    "runtime_ink_width": geometry.ink_width,
                    "recognizer_resize_scale": round(
                        geometry.recognizer_resize_scale, 6),
                    "horizontal_padding_ratio": round(
                        geometry.horizontal_padding_ratio, 6),
                    "clipped": bool(geometry.ink_top <= 0
                                    or geometry.ink_bottom >= crop_height - 1),
                    "target_glyph_width": glyph.glyph_width if glyph else None,
                    "target_glyph_height": glyph.glyph_height if glyph else None,
                    "target_glyph_occupancy": (round(glyph.glyph_occupancy, 8)
                                               if glyph else None),
                    "macro_stratum": stratum,
                })
            if (position + 1) % args.progress_every == 0:
                writer.sync()
                log_line(f"  page {position + 1}/{len(grouped)} "
                         f"backfilled={outcomes['BACKFILLED']} "
                         f"parity_failed={sum(v for k, v in outcomes.items() if k != 'BACKFILLED')}")

    total = sum(outcomes.values())
    parity_failures = total - outcomes["BACKFILLED"]
    summary = {
        "analysis": "line_train_v1_geometry_backfill",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_manifest_unmodified": True,
        "sidecar": str(args.out),
        "rows_considered": total,
        "backfilled": outcomes["BACKFILLED"],
        "parity_failures": parity_failures,
        "parity_rate": outcomes["BACKFILLED"] / total if total else 0.0,
        "outcomes": dict(outcomes),
        "macro_stratum_counts": dict(strata),
        "method": ("re-render from render_seed, re-detect, re-decode; geometry "
                   "written only when crop size and decoded text both match "
                   "the stored values"),
        "prohibited_and_not_used": [
            "detector box height as a proxy for ink height",
            "nominal point size to infer a stratum",
            "interpolation of missing geometry",
            "rewriting the original manifest",
        ],
        "GEOMETRY_BACKFILL": ("POSSIBLE" if outcomes["BACKFILLED"] else "IMPOSSIBLE"),
    }
    digest = atomic_write_json(
        args.out.with_name(args.out.stem + "_summary.json"), summary)

    print(f"\nrows considered {total}")
    for name, count in outcomes.most_common():
        print(f"  {name:28s} {count:6d}")
    print(f"macro strata: {dict(strata)}")
    print(f"GEOMETRY_BACKFILL = {summary['GEOMETRY_BACKFILL']}")
    print(f"summary sha256    = {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
