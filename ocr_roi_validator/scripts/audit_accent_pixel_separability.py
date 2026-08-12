"""Measure whether e and é are distinguishable at each pipeline stage.

Before training another classifier it is worth asking whether the pixels it
would see actually carry the answer. This renders paired phrases that differ in
exactly one character -- ``e`` versus ``é``, everything else held fixed
including the RNG seed -- and measures how far apart the two images are at each
stage the verifier could tap into:

1. ``native_line``    the rectified line crop, at detector resolution
2. ``recognizer_in``  that line resized to the recognizer's input height
3. ``normalized``     the tensor after mean/scale normalization
4. ``glyph_crop``     the accent-v2 input: a glyph crop of the resized line

A pair that is byte-identical at some stage cannot be separated by any
classifier reading that stage, however good. Those are counted explicitly,
because that is the number which decides whether this approach can work at all.

Upsampling is not counted as an improvement: interpolation invents no
information. Only stages that preserve distinct source pixels can gain.

No expected text and no real UI imagery is involved.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATOR_ROOT))

from ocr_roi_validator.model_package import load_model_package  # noqa: E402

STAGES = ("native_line", "recognizer_in", "normalized", "glyph_crop")

# Held fixed across a pair; only the word changes by one accent.
AUDIT_FONTS = [
    "arial.ttf", "calibri.ttf", "segoeui.ttf", "tahoma.ttf",
    "verdana.ttf", "GARA.TTF", "pala.ttf", "constan.ttf",
]
AUDIT_SIZES = [13, 14, 15, 16, 18, 21, 24, 28]
AUDIT_TEMPLATES = ["{}", "Bloc {} A1", "Vous {} localis", "Signal du {}, 2,5"]
AUDIT_WORDS = [("rélévé", "releve"), ("étété", "etete"), ("décalé", "decale")]


def build_engine(package_dir: Path, language: str):
    from rapidocr_onnxruntime import RapidOCR

    package = load_model_package(package_dir)
    p = package.preprocess
    engine = RapidOCR(
        det_model_path=str(package.detector_model),
        rec_model_path=str(package.recognizers[language]),
        rec_keys_path=str(package.dictionary),
        det_limit_type=str(p["det_limit_type"]),
        det_limit_side_len=int(p["det_limit_side_len"]),
        det_box_thresh=float(p["det_box_thresh"]),
        det_unclip_ratio=float(p["det_unclip_ratio"]),
        det_donot_use_dilation=bool(p["det_donot_use_dilation"]),
        use_cls=bool(p["use_cls"]),
    )
    return engine, package


def load_labels(dictionary: Path) -> list[str]:
    characters = dictionary.read_text(encoding="utf-8").split("\n")
    if characters and characters[-1] == "":
        characters = characters[:-1]
    return ["<blank>"] + characters + [" "]


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


def render(text: str, font_path: str, size: int, seed: int) -> Image.Image:
    """Deterministic render: the same seed gives pixel-identical styling."""
    rng = random.Random(seed)
    font = ImageFont.truetype(font_path, size)
    probe = Image.new("RGB", (8, 8))
    box = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
    pad_x, pad_y = rng.randint(14, 26), rng.randint(10, 20)
    width = max(box[2] - box[0] + pad_x * 2, 90)
    height = max(box[3] - box[1] + pad_y * 2, 44)
    background = (rng.randint(8, 34),) * 3
    foreground = (rng.randint(220, 252),) * 3
    image = Image.new("RGB", (width, height), background)
    offset = (pad_x + rng.uniform(-0.5, 0.5), pad_y + rng.uniform(-0.5, 0.5))
    ImageDraw.Draw(image).text(offset, text, font=font, fill=foreground)
    blur = rng.uniform(0.15, 0.55) if rng.random() < 0.4 else 0.0
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    contrast = rng.uniform(0.82, 1.20) if rng.random() < 0.4 else 1.0
    if contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(contrast)
    return image


def stage_images(engine, image: Image.Image, labels: list[str]) -> dict | None:
    """Return the target glyph's pixels at each pipeline stage."""
    bgr = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
    boxes, _ = engine.auto_text_det(bgr)
    if boxes is None or len(boxes) == 0:
        return None
    crops = engine.get_crop_img_list(bgr, boxes)
    recognizer = engine.text_rec
    channels, height, width = recognizer.rec_image_shape
    ratios = [c.shape[1] / float(c.shape[0]) for c in crops]
    max_wh_ratio = max([width / height] + ratios)

    for line_index, crop in enumerate(crops):
        crop_h, crop_w = crop.shape[:2]
        tensor = recognizer.resize_norm_img(crop, max_wh_ratio)
        logits = np.asarray(
            recognizer.session(tensor[np.newaxis, :].astype(np.float32))[0]
        )
        probabilities = logits[0]
        argmax = probabilities.argmax(axis=-1).tolist()
        decoded = recognizer.postprocess_op(
            logits, False, wh_ratio_list=[crop_w / float(crop_h)],
            max_wh_ratio=max_wh_ratio,
        )[0][0]
        emitted = collapse_ctc(argmax, labels)
        if "".join(i["char"] for i in emitted) != decoded:
            continue

        padded_w = int(height * max_wh_ratio)
        resized_w = min(padded_w, int(math.ceil(height * (crop_w / crop_h))))
        timesteps = probabilities.shape[0]
        scale = (padded_w / timesteps) * (crop_w / resized_w)

        for position, item in enumerate(emitted):
            character = unicodedata.normalize("NFC", item["char"])
            if character not in {"e", "é"}:
                continue
            # x-span in the ORIGINAL crop coordinates.
            x0 = max(0, int(math.floor((item["start"] + 0.5) * scale)) - 4)
            x1 = min(crop_w, int(math.ceil((item["end"] + 1 + 0.5) * scale)) + 4)
            if x1 - x0 < 4:
                continue

            # Stage 1: native rectified line, cropped at the same relative span.
            native = crop[:, x0:x1]

            # Stage 2: the line resized to recognizer height, same relative span.
            resized_line = cv2.resize(crop, (resized_w, height))
            rx0 = int(round(x0 * resized_w / crop_w))
            rx1 = max(rx0 + 1, int(round(x1 * resized_w / crop_w)))
            recognizer_in = resized_line[:, rx0:rx1]

            # Stage 3: normalized tensor over the same columns.
            normalized = tensor[:, :, rx0:rx1]

            # Stage 4: what accent-v2 actually used.
            glyph_crop = recognizer_in

            return {
                "predicted_char": character,
                "decoded": decoded,
                "native_line": native,
                "recognizer_in": recognizer_in,
                "normalized": normalized,
                "glyph_crop": glyph_crop,
                "native_size": [int(native.shape[1]), int(native.shape[0])],
                "recognizer_size": [
                    int(recognizer_in.shape[1]), int(recognizer_in.shape[0])
                ],
            }
    return None


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    """Distance metrics between two stage images of a matched pair."""
    if a is None or b is None:
        return {"comparable": False}
    # Compare over the overlapping region; sizes can differ by a pixel.
    height = min(a.shape[-2] if a.ndim == 3 and a.shape[0] <= 4 else a.shape[0],
                 b.shape[-2] if b.ndim == 3 and b.shape[0] <= 4 else b.shape[0])
    if a.ndim == 3 and a.shape[0] <= 4:      # CHW tensor
        left = a[:, :height, :]
        right = b[:, :height, :]
        width = min(left.shape[2], right.shape[2])
        left, right = left[:, :, :width], right[:, :, :width]
        left_gray = left.mean(axis=0)
        right_gray = right.mean(axis=0)
    else:                                     # HWC image
        width = min(a.shape[1], b.shape[1])
        left = a[:height, :width]
        right = b[:height, :width]
        left_gray = left[..., :3].mean(axis=2) if left.ndim == 3 else left
        right_gray = right[..., :3].mean(axis=2) if right.ndim == 3 else right

    difference = np.abs(left_gray.astype(np.float64) - right_gray.astype(np.float64))
    scale = 255.0 if difference.max() > 2.0 else 1.0
    normalized_difference = difference / scale

    # The accent lives in the top third of the glyph.
    third = max(1, height // 3)
    accent_band = normalized_difference[:third, :]

    return {
        "comparable": True,
        "identical": bool(np.array_equal(left_gray, right_gray)),
        "l1": float(normalized_difference.sum()),
        "l2": float(np.sqrt((normalized_difference ** 2).sum())),
        "changed_pixels": int((normalized_difference > (2.0 / 255.0)).sum()),
        "accent_band_max": float(accent_band.max()) if accent_band.size else 0.0,
        "accent_band_sum": float(accent_band.sum()) if accent_band.size else 0.0,
        "compared_shape": [int(height), int(width)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--pairs", type=int, default=240)
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    engine, package = build_engine(args.package, args.language)
    labels = load_labels(package.dictionary)

    fonts = [str(args.font_dir / f) for f in AUDIT_FONTS
             if (args.font_dir / f).is_file()]
    rng = random.Random(77)

    rows = []
    attempted = 0
    while len(rows) < args.pairs and attempted < args.pairs * 6:
        attempted += 1
        font = rng.choice(fonts)
        size = rng.choice(AUDIT_SIZES)
        template = rng.choice(AUDIT_TEMPLATES)
        accented, plain = rng.choice(AUDIT_WORDS)
        seed = rng.randrange(10 ** 6)

        accented_image = render(template.format(accented), font, size, seed)
        plain_image = render(template.format(plain), font, size, seed)
        try:
            accented_stages = stage_images(engine, accented_image, labels)
            plain_stages = stage_images(engine, plain_image, labels)
        except Exception:
            continue
        if accented_stages is None or plain_stages is None:
            continue

        row = {
            "font": Path(font).name,
            "size": size,
            "template": template,
            "word": accented,
            "accent_predicted": accented_stages["predicted_char"],
            "plain_predicted": plain_stages["predicted_char"],
            "in_scope": accented_stages["predicted_char"] == "é",
            "native_size": accented_stages["native_size"],
            "recognizer_size": accented_stages["recognizer_size"],
            "stages": {
                stage: compare(accented_stages[stage], plain_stages[stage])
                for stage in STAGES
            },
        }
        rows.append(row)
        if len(rows) % 40 == 0:
            print(f"  {len(rows)}/{args.pairs} pairs", flush=True)

    print(f"\npairs measured: {len(rows)} (attempted {attempted})")
    print(f"{'stage':16s} {'identical':>10s} {'medL1':>9s} {'medChgPx':>9s} "
          f"{'medAccMax':>10s} {'accBand=0':>10s}")
    summary = {}
    for stage in STAGES:
        measurements = [r["stages"][stage] for r in rows
                        if r["stages"][stage].get("comparable")]
        if not measurements:
            continue
        identical = sum(m["identical"] for m in measurements)
        accent_zero = sum(m["accent_band_max"] <= (2.0 / 255.0) for m in measurements)
        summary[stage] = {
            "pairs": len(measurements),
            "identical": identical,
            "median_l1": float(np.median([m["l1"] for m in measurements])),
            "median_changed_pixels": float(
                np.median([m["changed_pixels"] for m in measurements])
            ),
            "median_accent_band_max": float(
                np.median([m["accent_band_max"] for m in measurements])
            ),
            "accent_band_indistinguishable": accent_zero,
        }
        s = summary[stage]
        print(f"{stage:16s} {identical:>10d} {s['median_l1']:>9.2f} "
              f"{s['median_changed_pixels']:>9.0f} "
              f"{s['median_accent_band_max']:>10.4f} {accent_zero:>10d}")

    # Small sizes are where the accent is only a pixel or two.
    print("\nby rendered size (in-scope pairs only):")
    print(f"{'size':>5s} {'pairs':>6s} {'nativeIdent':>12s} {'glyphIdent':>11s} "
          f"{'nativeAccMax':>13s} {'glyphAccMax':>12s}")
    by_size = {}
    for size in sorted(AUDIT_SIZES):
        subset = [r for r in rows if r["size"] == size and r["in_scope"]]
        if not subset:
            continue
        native = [r["stages"]["native_line"] for r in subset]
        glyph = [r["stages"]["glyph_crop"] for r in subset]
        entry = {
            "pairs": len(subset),
            "native_identical": sum(m["identical"] for m in native),
            "glyph_identical": sum(m["identical"] for m in glyph),
            "native_accent_max": float(
                np.median([m["accent_band_max"] for m in native])
            ),
            "glyph_accent_max": float(
                np.median([m["accent_band_max"] for m in glyph])
            ),
        }
        by_size[size] = entry
        print(f"{size:>5d} {entry['pairs']:>6d} {entry['native_identical']:>12d} "
              f"{entry['glyph_identical']:>11d} {entry['native_accent_max']:>13.4f} "
              f"{entry['glyph_accent_max']:>12.4f}")

    collisions = {
        stage: summary.get(stage, {}).get("identical", 0) for stage in STAGES
    }
    total = len(rows)
    if total == 0:
        separability = "FAIL"
    elif collisions["native_line"] == 0 and collisions["glyph_crop"] == 0:
        separability = "PASS"
    elif collisions["native_line"] < total:
        separability = "PARTIAL"
    else:
        separability = "FAIL"

    # Choosing the stage is not a matter of picking the largest distance.
    # `recognizer_in` is `native_line` upsampled to the recognizer's height, and
    # `normalized` is `recognizer_in` rescaled into [-1, 1]; both therefore show
    # bigger raw distances without carrying any information the native crop
    # lacks. Interpolation and unit changes are not evidence.
    #
    # So: any stage with zero collisions is equally separable in principle, and
    # among those the native crop is preferred as the earliest, unresampled
    # source. A later stage is only worth choosing if the native crop collides
    # where it does not.
    zero_collision = [s for s in STAGES if summary.get(s, {}).get("identical") == 0]
    if "native_line" in zero_collision:
        best = "native_line"
        rationale = (
            "earliest stage with zero collisions; later stages are upsampled or "
            "rescaled versions of it and add no information"
        )
    elif zero_collision:
        best = zero_collision[0]
        rationale = "native crop collides; earliest non-colliding stage chosen"
    else:
        best = "native_line"
        rationale = "no stage separates the pairs; see collisions"
    print(f"\nPIXEL_SEPARABILITY = {separability}")
    print(f"SELECTED_VERIFIER_INPUT = {best}")
    print(f"  rationale: {rationale}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(
                {
                    "pairs": total,
                    "stage_summary": summary,
                    "by_size": by_size,
                    "collisions": collisions,
                    "pixel_separability": separability,
                    "selected_input": best,
                    "selection_rationale": rationale,
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
