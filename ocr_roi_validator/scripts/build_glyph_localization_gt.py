"""Render glyphs with known positions, to measure how well CTC localizes them.

Everything here is ground truth the renderer supplies: the target character's
box comes from the font's own layout metrics, and its foreground mask comes
from rendering the phrase twice -- once whole, once with the target character
omitted -- and taking the pixels that differ. The accent mask is obtained the
same way, from an ``e``/``é`` pair that is otherwise byte-identical.

No CTC output and no CNN prediction is ever used as a label. The real target
ROI plays no part in designing or parameterising this generator.

Boxes are recorded in the coordinate system of the rectified line crop, since
that is where a localizer has to place them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
SCRIPTS = VALIDATOR_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_accent_glyph_dataset import build_engine, load_labels  # noqa: E402

# Cohorts disjoint from every accent-v1/v2/v3 split.
FONT_SPLITS = {
    "locdev_v4": ["comic.ttf", "Inkfree.ttf", "ARLRDBD.TTF", "malgun.ttf"],
    # l_10646.ttf belongs to final_holdout_v3 and is deliberately excluded.
    "locval_v4": ["SitkaVF.ttf", "YuGothR.ttc"],
}
TEMPLATE_SPLITS = {
    "locdev_v4": ["{}", "Zj {} qv", "{} 7,4 kb", "Fx'{} pw"],
    "locval_v4": ["{} 9,2", "Nq {} zt"],
}
WORD_SPLITS = {
    "locdev_v4": [
        ("mémé", "meme"), ("tétu", "tetu"), ("bévué", "bevue"), ("cédé", "cede"),
    ],
    "locval_v4": [("fébrilé", "febrile"), ("zébré", "zebre")],
}
SIZE_SPLITS = {
    # Sizes not used by any accent-v1/v2/v3 split, so a size cohort cannot leak
    # between localization development and the earlier accent work.
    "locdev_v4": [10, 31, 33, 36],
    "locval_v4": [32, 34, 38],
}
SEEDS = {"locdev_v4": 700000, "locval_v4": 800000}


def render_phrase(
    text: str, font_path: str, size: int, style: dict
) -> Image.Image:
    """Render with a fully specified style, so a pair can be made identical."""
    font = ImageFont.truetype(font_path, size)
    image = Image.new("RGB", (style["width"], style["height"]), style["background"])
    ImageDraw.Draw(image).text(
        style["offset"], text, font=font, fill=style["foreground"]
    )
    if style["blur"]:
        image = image.filter(ImageFilter.GaussianBlur(style["blur"]))
    if style["contrast"] != 1.0:
        image = ImageEnhance.Contrast(image).enhance(style["contrast"])
    return image


def make_style(text: str, font_path: str, size: int, rng: random.Random) -> dict:
    font = ImageFont.truetype(font_path, size)
    probe = Image.new("RGB", (8, 8))
    box = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
    pad_x, pad_y = rng.randint(16, 26), rng.randint(12, 20)
    return {
        "width": max(box[2] - box[0] + pad_x * 2, 96),
        "height": max(box[3] - box[1] + pad_y * 2, 48),
        "background": (rng.randint(8, 30),) * 3,
        "foreground": (rng.randint(225, 250),) * 3,
        "offset": (pad_x + rng.uniform(-0.5, 0.5), pad_y + rng.uniform(-0.5, 0.5)),
        "blur": rng.uniform(0.15, 0.5) if rng.random() < 0.4 else 0.0,
        "contrast": rng.uniform(0.85, 1.18) if rng.random() < 0.4 else 1.0,
        "pad_x": pad_x,
    }


def target_box_from_metrics(
    text: str, index: int, font_path: str, size: int, style: dict
) -> tuple[float, float]:
    """x-range of ``text[index]`` from the font's own advance widths."""
    font = ImageFont.truetype(font_path, size)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    before = probe.textlength(text[:index], font=font)
    through = probe.textlength(text[: index + 1], font=font)
    origin = style["offset"][0]
    return origin + before, origin + through


def difference_mask(a: Image.Image, b: Image.Image) -> np.ndarray:
    """Pixels where two renderings differ, as a boolean mask."""
    left = np.asarray(a.convert("L")).astype(np.int16)
    right = np.asarray(b.convert("L")).astype(np.int16)
    return np.abs(left - right) > 8


def mask_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if not mask.any():
        return None
    rows = np.where(mask.any(axis=1))[0]
    columns = np.where(mask.any(axis=0))[0]
    return int(columns[0]), int(rows[0]), int(columns[-1]), int(rows[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--split", choices=tuple(FONT_SPLITS), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    args = parser.parse_args()

    # The label invariant: one e-form per rendering, so the target is unambiguous.
    for split, pairs in WORD_SPLITS.items():
        for accented, plain in pairs:
            for spelling, want in ((accented, "é"), (plain, "e")):
                forms = {c for c in unicodedata.normalize("NFC", spelling)
                         if c in {"e", "é"}}
                if forms != {want}:
                    print(f"word {spelling!r} has e-forms {sorted(forms)}",
                          file=sys.stderr)
                    return 1
    for split, templates in TEMPLATE_SPLITS.items():
        for template in templates:
            if {c for c in template.replace("{}", "") if c in {"e", "é"}}:
                print(f"template {template!r} contains an e-form", file=sys.stderr)
                return 1

    rng = random.Random(SEEDS[args.split])
    engine, package = build_engine(args.package, args.language)
    labels = load_labels(package.dictionary)
    fonts = [str(args.font_dir / f) for f in FONT_SPLITS[args.split]
             if (args.font_dir / f).is_file()]
    if not fonts:
        print("no fonts available", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "lines").mkdir(exist_ok=True)

    rows = []
    attempted = 0
    while len(rows) < args.samples and attempted < args.samples * 8:
        attempted += 1
        font = rng.choice(fonts)
        size = rng.choice(SIZE_SPLITS[args.split])
        template = rng.choice(TEMPLATE_SPLITS[args.split])
        accented_word, plain_word = rng.choice(WORD_SPLITS[args.split])
        use_accent = rng.random() < 0.5
        word = accented_word if use_accent else plain_word
        text = template.format(word)

        # Pick one target occurrence of the e-form in this phrase.
        positions = [i for i, c in enumerate(unicodedata.normalize("NFC", text))
                     if c in {"e", "é"}]
        if not positions:
            continue
        target_index = rng.choice(positions)
        visual = unicodedata.normalize("NFC", text)[target_index]

        style = make_style(text, font, size, rng)
        page = render_phrase(text, font, size, style)

        # Ground truth 1: the same phrase with the target character removed.
        # The pixels that differ are exactly that character's ink.
        without = text[:target_index] + text[target_index + 1:]
        page_without = render_phrase(without, font, size, style)

        # Ground truth 2: the e/é counterpart, identical in every other way.
        counterpart_char = "e" if visual == "é" else "é"
        counterpart = text[:target_index] + counterpart_char + text[target_index + 1:]
        page_counterpart = render_phrase(counterpart, font, size, style)

        page_bgr = np.asarray(page)[:, :, ::-1].copy()
        try:
            boxes, _ = engine.auto_text_det(page_bgr)
        except Exception:
            continue
        if boxes is None or len(boxes) == 0:
            continue
        crops = engine.get_crop_img_list(page_bgr, boxes)
        if len(crops) != 1:
            continue          # keep the geometry simple: one line per rendering

        # Masks in page coordinates. Removing a character reflows the text to
        # its right, so restrict the glyph mask to the target's own x-range
        # taken from font metrics.
        glyph_mask = difference_mask(page, page_without)
        x_start, x_end = target_box_from_metrics(text, target_index, font, size, style)
        column_window = np.zeros(glyph_mask.shape[1], dtype=bool)
        column_window[max(0, int(np.floor(x_start)) - 1):
                      min(glyph_mask.shape[1], int(np.ceil(x_end)) + 2)] = True
        glyph_mask = glyph_mask & column_window[None, :]
        accent_mask = difference_mask(page, page_counterpart)

        glyph_box = mask_box(glyph_mask)
        accent_box = mask_box(accent_mask)
        if glyph_box is None:
            continue

        # Map page coordinates into the rectified crop. The detector may rotate
        # or shift, so use the polygon's own origin and scale.
        polygon = np.asarray(boxes[0], dtype=np.float64)
        poly_x0, poly_y0 = polygon[:, 0].min(), polygon[:, 1].min()
        poly_x1, poly_y1 = polygon[:, 0].max(), polygon[:, 1].max()
        crop = crops[0]
        crop_h, crop_w = crop.shape[:2]
        scale_x = crop_w / max(1e-6, (poly_x1 - poly_x0))
        scale_y = crop_h / max(1e-6, (poly_y1 - poly_y0))

        def to_crop(x: float, y: float) -> tuple[float, float]:
            return (x - poly_x0) * scale_x, (y - poly_y0) * scale_y

        gx0, gy0 = to_crop(glyph_box[0], glyph_box[1])
        gx1, gy1 = to_crop(glyph_box[2], glyph_box[3])
        # The detector polygon often sits slightly inside the rendered text, so
        # a target at the very start or end maps a pixel or two beyond the crop.
        # Clamp those rather than discarding them; only reject a target that is
        # genuinely outside the detected line. The stricter earlier form threw
        # away roughly a third of otherwise usable renderings.
        if gx1 <= 0 or gx0 >= crop_w or (gx1 - gx0) < 1.0:
            continue
        gx0 = max(0.0, gx0)
        gx1 = min(float(crop_w), gx1)

        record = {
            "index": len(rows),
            "font": Path(font).name,
            "size": size,
            "template": template,
            "word": word,
            "text": text,
            "target_index": target_index,
            "visual_label": visual,
            "crop_size": [int(crop_w), int(crop_h)],
            "gt_glyph_box_crop": [float(gx0), float(gy0), float(gx1), float(gy1)],
            "gt_glyph_box_page": list(glyph_box),
        }
        if accent_box is not None:
            ax0, ay0 = to_crop(accent_box[0], accent_box[1])
            ax1, ay1 = to_crop(accent_box[2], accent_box[3])
            record["gt_accent_box_crop"] = [float(ax0), float(ay0),
                                            float(ax1), float(ay1)]

        name = f"{args.split}_{len(rows):05d}.png"
        cv2.imwrite(str(args.out_dir / "lines" / name), crop)
        # Store the ground-truth masks resampled into crop space, so a
        # containment metric can be computed without re-rendering.
        glyph_in_crop = cv2.warpAffine(
            glyph_mask.astype(np.uint8) * 255,
            np.array([[scale_x, 0, -poly_x0 * scale_x],
                      [0, scale_y, -poly_y0 * scale_y]], dtype=np.float64),
            (crop_w, crop_h), flags=cv2.INTER_NEAREST,
        )
        np.savez_compressed(
            args.out_dir / "lines" / f"{args.split}_{len(rows):05d}_gt.npz",
            glyph_mask=(glyph_in_crop > 127),
        )
        record["line_file"] = f"lines/{name}"
        record["gt_mask_file"] = f"lines/{args.split}_{len(rows):05d}_gt.npz"
        rows.append(record)

        if len(rows) % 100 == 0:
            print(f"  {len(rows)}/{args.samples}", flush=True)

    manifest = {
        "split": args.split,
        "seed": SEEDS[args.split],
        "fonts": FONT_SPLITS[args.split],
        "templates": TEMPLATE_SPLITS[args.split],
        "words": WORD_SPLITS[args.split],
        "sizes": SIZE_SPLITS[args.split],
        "attempted_renderings": attempted,
        "samples": rows,
        "ground_truth_source": (
            "renderer: character box from font advance metrics, glyph ink from "
            "the difference against the same phrase with the character removed, "
            "accent ink from the e/é counterpart rendering. No CTC or CNN output "
            "is used as a label."
        ),
    }
    text_payload = json.dumps(manifest, ensure_ascii=False, indent=2)
    (args.out_dir / "manifest.json").write_text(text_payload, encoding="utf-8")
    print(f"\nsplit {args.split}: {len(rows)} samples from {attempted} renderings")
    print(f"manifest sha256: {hashlib.sha256(text_payload.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
