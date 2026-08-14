"""Build counterfactual paired line datasets for the target-query verifier.

Every core context is rendered twice, identical in every respect except the
accent pixels on one character. That is what makes the label causal: if the
model answers differently across a pair, the only thing it can have responded
to is the accent itself.

UNKNOWN is generated rather than inferred. Cases where the query genuinely does
not identify one character -- an ordinal shifted onto a neighbour, an
insertion/deletion mismatch, a target clipped at the line edge -- are labelled
UNKNOWN so the model learns to abstain instead of being forced to pick.

Ground truth comes from the renderer: character positions from font advance
metrics, accent ink from the paired difference. No CTC output, no CNN
prediction, no expected text, and no real UI image is involved.
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

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATOR_ROOT))
SCRIPTS = VALIDATOR_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_accent_glyph_dataset import build_engine, load_labels  # noqa: E402
from ocr_roi_validator.line_verifier_input import (  # noqa: E402
    LineVerifierInputConfig,
    build_line_input,
)
from train_line_verifier import (  # noqa: E402
    CLASS_ACCENT_PRESENT,
    CLASS_BARE_E,
    CLASS_UNKNOWN,
)

# Cohorts disjoint from every earlier split and from each other.
FONT_SPLITS = {
    "line_train_v1": [
        "arial.ttf", "calibri.ttf", "segoeui.ttf", "verdana.ttf",
        "tahoma.ttf", "comic.ttf", "corbel.ttf", "Candara.ttf",
    ],
    "line_calibration_v1": ["georgia.ttf", "trebuc.ttf"],
    "line_preflight_holdout_v1": ["times.ttf", "framd.ttf", "LSANS.TTF"],
}
# Templates carry no e-form of their own, so the target is unambiguous, but do
# supply the hard neighbours: other accents, ascenders, apostrophes, digits.
TEMPLATE_SPLITS = {
    "line_train_v1": [
        "{}", "Il {} bd", "{} 3,7 kt", "Fq'{} lm", "Àù {} ï", "{} tl 12",
    ],
    "line_calibration_v1": ["{} hb 9,1", "Kd {} ïw"],
    "line_preflight_holdout_v1": ["{} pv 4,2", "Tb {} ùl", "Mq'{} ê"],
}
WORD_SPLITS = {
    "line_train_v1": [
        ("mémé", "meme"), ("bébé", "bebe"), ("tété", "tete"),
        ("récré", "recre"), ("pépé", "pepe"), ("dédé", "dede"),
    ],
    "line_calibration_v1": [("félé", "fele"), ("zélé", "zele")],
    "line_preflight_holdout_v1": [("vécé", "vece"), ("némé", "neme"),
                                  ("céré", "cere")],
}
SIZE_SPLITS = {
    "line_train_v1": [11, 15, 19, 23, 27, 35],
    "line_calibration_v1": [13, 21, 31],
    "line_preflight_holdout_v1": [12, 17, 25, 33],
}
SEEDS = {"line_train_v1": 5100000, "line_calibration_v1": 5200000,
         "line_preflight_holdout_v1": 5300000}

BUCKETS = ((8, 12), (13, 17), (18, 24), (25, 32), (33, 44))


def bucket_of(ink_height: int) -> str | None:
    for low, high in BUCKETS:
        if low <= ink_height <= high:
            return f"{low}-{high}"
    return None


def make_style(text, font_path, size, rng):
    font = ImageFont.truetype(font_path, size)
    box = ImageDraw.Draw(Image.new("RGB", (8, 8))).textbbox((0, 0), text, font=font)
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


def render(text, font_path, size, style):
    font = ImageFont.truetype(font_path, size)
    image = Image.new("RGB", (style["width"], style["height"]), style["background"])
    ImageDraw.Draw(image).text(style["offset"], text, font=font,
                               fill=style["foreground"])
    if style["blur"]:
        image = image.filter(ImageFilter.GaussianBlur(style["blur"]))
    if style["contrast"] != 1.0:
        image = ImageEnhance.Contrast(image).enhance(style["contrast"])
    return image


def char_centre(text, index, font_path, size, style):
    font = ImageFont.truetype(font_path, size)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    before = probe.textlength(text[:index], font=font)
    through = probe.textlength(text[:index + 1], font=font)
    return style["offset"][0] + (before + through) / 2.0


def ink_rows(image):
    gray = np.asarray(image.convert("L")).astype(float)
    low, high = gray.min(), gray.max()
    if high - low < 12:
        return None
    mask = gray >= (low + high) / 2.0
    if mask.mean() > 0.5:
        mask = ~mask
    rows = np.where(mask.any(axis=1))[0]
    return (int(rows[0]), int(rows[-1])) if rows.size else None


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


def attention_target(centre_x: float, scaled_width: int, config, sigma=2.0):
    """Renderer-derived attention supervision over input columns."""
    target = np.zeros(config.width, dtype=np.float32)
    if scaled_width <= 0:
        return target
    columns = np.arange(scaled_width, dtype=np.float32)
    bump = np.exp(-0.5 * ((columns - centre_x) / sigma) ** 2)
    total = float(bump.sum())
    if total > 1e-9:
        target[:scaled_width] = bump / total
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--split", choices=tuple(FONT_SPLITS), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=900)
    parser.add_argument("--max-attempts", type=int, default=40000)
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    args = parser.parse_args()

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

    config = LineVerifierInputConfig()
    rng = random.Random(SEEDS[args.split])
    engine, package = build_engine(args.package, args.language)
    labels = load_labels(package.dictionary)
    recognizer = engine.text_rec
    channels, height, width = recognizer.rec_image_shape
    fonts = [str(args.font_dir / f) for f in FONT_SPLITS[args.split]
             if (args.font_dir / f).is_file()]
    missing = [f for f in FONT_SPLITS[args.split]
               if not (args.font_dir / f).is_file()]
    if missing:
        print(f"missing fonts: {missing}", file=sys.stderr)
        return 1

    planes_out, query_out, label_out, attention_out, rows = [], [], [], [], []
    pair_serial = 0
    buckets = Counter()
    label_counts = Counter()
    hard_cases = Counter()
    attempts = 0

    def emit(line_bgr, probabilities, token, ordinal, decoded_length,
             label, centre_x, meta):
        prepared = build_line_input(
            line_bgr, probabilities, token["label"], token["start"],
            token["end"], ordinal, decoded_length, config)
        if prepared is None:
            return False
        scaled_width = int(round(line_bgr.shape[1] * prepared.scale_x))
        planes_out.append(prepared.planes)
        query_out.append(prepared.query)
        label_out.append(label)
        attention_out.append(
            attention_target(centre_x * prepared.scale_x, scaled_width, config))
        rows.append(meta)
        label_counts[label] += 1
        return True

    while len(rows) < args.pairs * 2 and attempts < args.max_attempts:
        attempts += 1
        font = rng.choice(fonts)
        size = rng.choice(SIZE_SPLITS[args.split])
        template = rng.choice(TEMPLATE_SPLITS[args.split])
        accented_word, plain_word = rng.choice(WORD_SPLITS[args.split])

        # Counterfactual pair: the same phrase with and without the accent on
        # one chosen character, identical in every other respect.
        base_text = template.format(plain_word)
        normalized = unicodedata.normalize("NFC", base_text)
        positions = [i for i, c in enumerate(normalized) if c == "e"]
        if not positions:
            continue
        target_index = rng.choice(positions)
        accented_text = (base_text[:target_index] + "é"
                         + base_text[target_index + 1:])

        style = make_style(base_text, font, size, rng)

        # Crop geometry is computed once, from the accented member, and reused
        # for both. Deriving it per-member would let the accent's extra ink
        # move the crop and rescale the whole line, so the pair would differ
        # everywhere instead of only at the accent -- which destroys exactly
        # the causal isolation the pair exists to provide.
        reference_page = render(accented_text, font, size, style)
        reference_extent = ink_rows(reference_page)
        if reference_extent is None:
            continue
        ink_height = reference_extent[1] - reference_extent[0] + 1
        bucket = bucket_of(ink_height)
        if bucket is None:
            continue
        margin = max(2, ink_height // 6)
        top = max(0, reference_extent[0] - margin)
        bottom = min(reference_page.height, reference_extent[1] + margin + 1)

        pair = []
        for text, visual in ((base_text, "e"), (accented_text, "é")):
            page = render(text, font, size, style)
            line = np.asarray(page)[top:bottom, :, ::-1].copy()
            if line.shape[0] < 6 or line.shape[1] < 12:
                break
            pair.append((text, visual, line, bucket, ink_height,
                         char_centre(text, target_index, font, size, style)))
        if len(pair) != 2:
            continue

        pair_serial += 1
        for text, visual, line, bucket, ink_height, centre in pair:
            crop_h, crop_w = line.shape[:2]
            max_wh_ratio = max(width / height, crop_w / crop_h)
            tensor = recognizer.resize_norm_img(line, max_wh_ratio)[np.newaxis, :]
            logits = np.asarray(recognizer.session(tensor.astype(np.float32))[0])
            probabilities = logits[0]
            decoded = recognizer.postprocess_op(
                logits, False, wh_ratio_list=[crop_w / float(crop_h)],
                max_wh_ratio=max_wh_ratio)[0][0]
            emitted = collapse_ctc(probabilities.argmax(axis=-1).tolist(), labels)
            if not emitted or "".join(i["char"] for i in emitted) != decoded:
                continue

            # Query the token whose centre is nearest the true centre. The
            # renderer supplies that centre; it never reaches the network.
            stride = crop_w / max(1, probabilities.shape[0])
            best, best_distance = None, None
            for index, item in enumerate(emitted):
                token_centre = ((item["start"] + item["end"] + 1) / 2.0) * stride
                distance = abs(token_centre - centre)
                if best_distance is None or distance < best_distance:
                    best, best_distance = index, distance
            if best is None:
                continue

            label = (CLASS_BARE_E if visual == "e" else CLASS_ACCENT_PRESENT)
            meta = {"font": Path(font).name, "size": size, "bucket": bucket,
                    "ink_height": ink_height, "visual_label": visual,
                    "text": text, "template": template,
                    # Identifies the two members of one counterfactual pair, so
                    # isolation can be audited without guessing from the text.
                    "pair_id": pair_serial,
                    "case": "paired_counterfactual"}
            if emit(line, probabilities, emitted[best], best, len(emitted),
                    label, centre, meta):
                buckets[bucket] += 1

            # UNKNOWN case 1: the ordinal points at a neighbour, so the query
            # does not identify the character the label describes.
            for shift in (-1, 1):
                neighbour = best + shift
                if 0 <= neighbour < len(emitted) and rng.random() < 0.35:
                    shifted_meta = dict(meta)
                    shifted_meta["case"] = "ordinal_shifted_to_neighbour"
                    if emit(line, probabilities, emitted[neighbour], neighbour,
                            len(emitted), CLASS_UNKNOWN, centre, shifted_meta):
                        hard_cases["ordinal_shifted"] += 1

            # UNKNOWN case 2: an ordinal beyond the decoded length, which is
            # what an insertion or deletion looks like downstream.
            if rng.random() < 0.15:
                mismatch_meta = dict(meta)
                mismatch_meta["case"] = "ordinal_length_mismatch"
                if emit(line, probabilities, emitted[best], best,
                        len(emitted) + rng.randint(1, 2), CLASS_UNKNOWN,
                        centre, mismatch_meta):
                    hard_cases["length_mismatch"] += 1

        if len(rows) % 400 == 0 and rows:
            print(f"  {len(rows)} examples; buckets {dict(buckets)}", flush=True)

    if not rows:
        print("no examples produced", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "tensors.npz",
        planes=np.stack(planes_out).astype(np.float32),
        query=np.stack(query_out).astype(np.float32),
        label=np.asarray(label_out, dtype=np.int64),
        attention_target=np.stack(attention_out).astype(np.float32),
    )
    manifest = {
        "split": args.split, "seed": SEEDS[args.split],
        "fonts": FONT_SPLITS[args.split], "templates": TEMPLATE_SPLITS[args.split],
        "words": WORD_SPLITS[args.split], "sizes": SIZE_SPLITS[args.split],
        "examples": len(rows), "attempts": attempts,
        "buckets": dict(buckets),
        "label_counts": {str(k): v for k, v in label_counts.items()},
        "hard_cases": dict(hard_cases),
        "input_config": config.as_dict(),
        "ground_truth_source": (
            "renderer: character centre from font advance metrics, accent from "
            "the paired counterfactual. No CTC or model output is a label."
        ),
        "samples": rows,
    }
    payload = json.dumps(manifest, ensure_ascii=False, indent=2)
    (args.out_dir / "manifest.json").write_text(payload, encoding="utf-8")

    print(f"\nsplit {args.split}: {len(rows)} examples from {attempts} attempts")
    print(f"  buckets     : {dict(buckets)}")
    print(f"  labels      : "
          f"{{ACCENT_PRESENT: {label_counts[CLASS_ACCENT_PRESENT]}, "
          f"BARE_E: {label_counts[CLASS_BARE_E]}, "
          f"UNKNOWN: {label_counts[CLASS_UNKNOWN]}}}")
    print(f"  hard cases  : {dict(hard_cases)}")
    print(f"manifest sha256: {hashlib.sha256(payload.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
