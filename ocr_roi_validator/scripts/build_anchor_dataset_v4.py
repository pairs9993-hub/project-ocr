"""Build anchor/localization datasets that do not depend on the detector.

The previous validation attempt collapsed because the detector failed to find a
line in most large-font renderings, which destroyed the size balance of the
evaluation set. The localizer never sees a page anyway -- it sees a recognizer
line crop -- so the primary path skips detection entirely and synthesises the
line crop directly, exactly as the recognizer would receive it.

A secondary path keeps the detector in the loop, purely to check that the
primary path's geometry is plausible at runtime. Nothing in the primary
measurement is allowed to depend on it.

Ground truth comes from the renderer: character x-ranges from the font's own
advance metrics, glyph ink from the difference against the same line with that
character removed, accent ink from an e/é counterpart identical in every other
respect. No CTC output and no CNN prediction is ever a label.

Balancing is by rasterized ink height in pixels, not nominal point size,
because that is what determines how many CTC timesteps a glyph occupies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import unicodedata
from collections import Counter
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

from build_accent_glyph_dataset import build_engine  # noqa: E402

# Cohorts disjoint from every accent split and from locdev/locval_v4.
FONT_SPLITS = {
    "anchor_dev_v4": [
        "times.ttf", "georgia.ttf", "trebuc.ttf", "corbel.ttf",
        "Candara.ttf", "framd.ttf",
    ],
    "anchor_val_v4": ["BOOKOS.TTF", "CENTURY.TTF", "constan.ttf"],
}
TEMPLATE_SPLITS = {
    "anchor_dev_v4": ["{}", "Hb {} vk", "{} 3,8 mt", "Kp'{} nz", "{} qw"],
    "anchor_val_v4": ["{}", "Wd {} bj", "{} 5,1 xr"],
}
WORD_SPLITS = {
    "anchor_dev_v4": [
        ("réglé", "regle"), ("mémoiré", "memoire"), ("pédalé", "pedale"),
        ("téméré", "temere"), ("bébé", "bebe"),
    ],
    "anchor_val_v4": [
        ("céréalé", "cereale"), ("végété", "vegete"), ("réséqué", "reseque"),
    ],
}
SEEDS = {"anchor_dev_v4": 1300000, "anchor_val_v4": 1400000}

# Buckets are on rasterized ink height, the quantity that decides how many CTC
# timesteps a glyph gets. Nominal point size does not, because it interacts
# with the font's own metrics.
BUCKETS = ((8, 12), (13, 17), (18, 24), (25, 32), (33, 44))
# Point sizes are swept widely; each rendering is assigned to whichever bucket
# its measured ink height falls in.
POINT_SIZES = list(range(9, 61))


def bucket_of(ink_height: int) -> str | None:
    for low, high in BUCKETS:
        if low <= ink_height <= high:
            return f"{low}-{high}"
    return None


def make_style(text: str, font_path: str, size: int, rng: random.Random) -> dict:
    font = ImageFont.truetype(font_path, size)
    probe = Image.new("RGB", (8, 8))
    box = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
    pad_x, pad_y = rng.randint(6, 14), rng.randint(4, 10)
    return {
        "width": max(box[2] - box[0] + pad_x * 2, 48),
        "height": max(box[3] - box[1] + pad_y * 2, 20),
        "background": (rng.randint(8, 30),) * 3,
        "foreground": (rng.randint(225, 250),) * 3,
        "offset": (pad_x + rng.uniform(-0.5, 0.5), pad_y + rng.uniform(-0.5, 0.5)),
        "blur": rng.uniform(0.15, 0.5) if rng.random() < 0.4 else 0.0,
        "contrast": rng.uniform(0.85, 1.18) if rng.random() < 0.4 else 1.0,
    }


def render(text: str, font_path: str, size: int, style: dict) -> Image.Image:
    """Deterministic render: identical style gives identical pixels."""
    font = ImageFont.truetype(font_path, size)
    image = Image.new("RGB", (style["width"], style["height"]), style["background"])
    ImageDraw.Draw(image).text(style["offset"], text, font=font,
                               fill=style["foreground"])
    if style["blur"]:
        image = image.filter(ImageFilter.GaussianBlur(style["blur"]))
    if style["contrast"] != 1.0:
        image = ImageEnhance.Contrast(image).enhance(style["contrast"])
    return image


def char_x_range(text: str, index: int, font_path: str, size: int,
                 style: dict) -> tuple[float, float]:
    font = ImageFont.truetype(font_path, size)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    before = probe.textlength(text[:index], font=font)
    through = probe.textlength(text[: index + 1], font=font)
    return style["offset"][0] + before, style["offset"][0] + through


def difference_mask(a: Image.Image, b: Image.Image) -> np.ndarray:
    left = np.asarray(a.convert("L")).astype(np.int16)
    right = np.asarray(b.convert("L")).astype(np.int16)
    return np.abs(left - right) > 8


def ink_rows(image: Image.Image) -> tuple[int, int] | None:
    gray = np.asarray(image.convert("L")).astype(float)
    low, high = gray.min(), gray.max()
    if high - low < 12:
        return None
    mask = gray >= (low + high) / 2.0
    if mask.mean() > 0.5:
        mask = ~mask
    rows = np.where(mask.any(axis=1))[0]
    return (int(rows[0]), int(rows[-1])) if rows.size else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--split", choices=tuple(FONT_SPLITS), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-bucket", type=int, default=400)
    parser.add_argument("--max-attempts", type=int, default=200000)
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    parser.add_argument("--with-detector-check", action="store_true",
                        help="also record whether the detector would find this line")
    args = parser.parse_args()

    # Label invariant: exactly one e-form per rendering.
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
    fonts = [str(args.font_dir / f) for f in FONT_SPLITS[args.split]
             if (args.font_dir / f).is_file()]
    missing = [f for f in FONT_SPLITS[args.split]
               if not (args.font_dir / f).is_file()]
    if missing:
        print(f"missing fonts: {missing}", file=sys.stderr)
        return 1

    engine = package = None
    if args.with_detector_check:
        engine, package = build_engine(args.package, args.language)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "lines").mkdir(exist_ok=True)

    target_per_bucket = args.per_bucket
    counts = Counter()
    rows = []
    attempts = 0
    rejected = Counter()

    while attempts < args.max_attempts and any(
        counts[f"{lo}-{hi}"] < target_per_bucket for lo, hi in BUCKETS
    ):
        attempts += 1
        font = rng.choice(fonts)
        size = rng.choice(POINT_SIZES)
        template = rng.choice(TEMPLATE_SPLITS[args.split])
        accented_word, plain_word = rng.choice(WORD_SPLITS[args.split])
        word = accented_word if rng.random() < 0.5 else plain_word
        text = template.format(word)

        normalized = unicodedata.normalize("NFC", text)
        positions = [i for i, c in enumerate(normalized) if c in {"e", "é"}]
        if not positions:
            continue
        target_index = rng.choice(positions)
        visual = normalized[target_index]

        style = make_style(text, font, size, rng)
        page = render(text, font, size, style)

        extent = ink_rows(page)
        if extent is None:
            rejected["no_ink"] += 1
            continue
        ink_height = extent[1] - extent[0] + 1
        bucket = bucket_of(ink_height)
        if bucket is None or counts[bucket] >= target_per_bucket:
            rejected["bucket_full_or_out_of_range"] += 1
            continue

        # The primary path: crop the line directly, the way the recognizer
        # would receive it, with no detector involved.
        margin = max(2, ink_height // 6)
        top = max(0, extent[0] - margin)
        bottom = min(page.height, extent[1] + margin + 1)
        line = np.asarray(page)[top:bottom, :, ::-1].copy()
        if line.shape[0] < 6 or line.shape[1] < 12:
            rejected["line_too_small"] += 1
            continue

        # Ground truth 1: this character's own ink.
        without = text[:target_index] + text[target_index + 1:]
        page_without = render(without, font, size, style)
        glyph_mask_page = difference_mask(page, page_without)
        x_start, x_end = char_x_range(text, target_index, font, size, style)
        window = np.zeros(glyph_mask_page.shape[1], dtype=bool)
        window[max(0, int(np.floor(x_start)) - 1):
               min(glyph_mask_page.shape[1], int(np.ceil(x_end)) + 2)] = True
        glyph_mask_page = glyph_mask_page & window[None, :]

        # Ground truth 2: the accent ink, from the e/é counterpart.
        counterpart_char = "e" if visual == "é" else "é"
        counterpart = (text[:target_index] + counterpart_char
                       + text[target_index + 1:])
        accent_mask_page = difference_mask(page, render(counterpart, font, size, style))

        if not glyph_mask_page.any():
            rejected["empty_glyph_mask"] += 1
            continue

        glyph_mask = glyph_mask_page[top:bottom, :]
        accent_mask = accent_mask_page[top:bottom, :]
        columns = np.where(glyph_mask.any(axis=0))[0]
        if columns.size == 0:
            rejected["glyph_outside_line"] += 1
            continue
        gx0, gx1 = int(columns[0]), int(columns[-1]) + 1

        # Neighbour boxes, so intrusion can be attributed rather than guessed.
        neighbours = []
        for offset in (-1, 1):
            index = target_index + offset
            if 0 <= index < len(text) and text[index] != " ":
                nx0, nx1 = char_x_range(text, index, font, size, style)
                neighbours.append([float(nx0), float(nx1)])

        detector_found = None
        if engine is not None:
            page_bgr = np.asarray(page)[:, :, ::-1].copy()
            try:
                boxes, _ = engine.auto_text_det(page_bgr)
                detector_found = bool(boxes is not None and len(boxes) > 0)
            except Exception:
                detector_found = False

        index_in_split = len(rows)
        name = f"{args.split}_{index_in_split:05d}"
        cv2.imwrite(str(args.out_dir / "lines" / f"{name}.png"), line)
        np.savez_compressed(
            args.out_dir / "lines" / f"{name}_gt.npz",
            glyph_mask=glyph_mask, accent_mask=accent_mask,
        )
        counts[bucket] += 1
        rows.append({
            "index": index_in_split,
            "line_file": f"lines/{name}.png",
            "gt_mask_file": f"lines/{name}_gt.npz",
            "font": Path(font).name,
            "point_size": size,
            "ink_height": ink_height,
            "bucket": bucket,
            "template": template,
            "word": word,
            "text": text,
            "target_index": target_index,
            "visual_label": visual,
            "line_size": [int(line.shape[1]), int(line.shape[0])],
            "gt_glyph_x": [gx0, gx1],
            "neighbour_x_page": neighbours,
            "detector_would_find_line": detector_found,
        })

        if len(rows) % 200 == 0:
            print(f"  {len(rows)} samples; buckets "
                  f"{ {k: counts[k] for k in (f'{a}-{b}' for a, b in BUCKETS)} }",
                  flush=True)

    bucket_counts = {f"{lo}-{hi}": counts[f"{lo}-{hi}"] for lo, hi in BUCKETS}
    met = all(v >= target_per_bucket for v in bucket_counts.values())

    manifest = {
        "split": args.split,
        "seed": SEEDS[args.split],
        "fonts": FONT_SPLITS[args.split],
        "templates": TEMPLATE_SPLITS[args.split],
        "words": WORD_SPLITS[args.split],
        "point_sizes": [POINT_SIZES[0], POINT_SIZES[-1]],
        "buckets": bucket_counts,
        "target_per_bucket": target_per_bucket,
        "all_buckets_met": met,
        "attempts": attempts,
        "rejected": dict(rejected),
        "path": "primary_detector_independent",
        "ground_truth_source": (
            "renderer: character x-range from font advance metrics, glyph ink "
            "from the difference against the same line with the character "
            "removed, accent ink from the e/é counterpart. No CTC or CNN "
            "output is used as a label."
        ),
        "samples": rows,
    }
    payload = json.dumps(manifest, ensure_ascii=False, indent=2)
    (args.out_dir / "manifest.json").write_text(payload, encoding="utf-8")

    print(f"\nsplit {args.split}: {len(rows)} samples from {attempts} attempts")
    for key, value in bucket_counts.items():
        flag = "ok" if value >= target_per_bucket else "SHORT"
        print(f"  bucket {key:>7s}px : {value:>5d}  {flag}")
    print(f"rejected: {dict(rejected)}")
    if engine is not None:
        found = sum(1 for r in rows if r["detector_would_find_line"])
        print(f"detector would find the line in {found}/{len(rows)} "
              f"(diagnostic only)")
    print(f"all buckets met: {met}")
    print(f"manifest sha256: {hashlib.sha256(payload.encode()).hexdigest()}")
    return 0 if met else 1


if __name__ == "__main__":
    raise SystemExit(main())
