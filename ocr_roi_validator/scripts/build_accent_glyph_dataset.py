"""Generate a synthetic e / é glyph dataset for the accent verifier.

Every sample is produced by rendering a French phrase, running the real
detector and recognizer over it, and taking the glyph crops the CTC alignment
points at. That matters: training on tidy isolated glyphs would not reflect the
crops the verifier actually receives at inference, which carry alignment error
and slivers of neighbouring characters.

The visual label is known from the rendering, not from OCR, so a sample is
labelled ``e`` or ``é`` by what was drawn even when the recognizer misreads it.
That is what makes hallucination cases usable as training data.

Splitting is by font family, phrase template and seed simultaneously, so no
holdout glyph shares a typeface *or* a phrase with anything seen in training.
Real UI images are never used here.
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

from ocr_roi_validator.model_package import load_model_package  # noqa: E402

# Font families, split so a holdout typeface is never seen in training.
FONT_SPLITS = {
    "train": [
        "arial.ttf", "calibri.ttf", "segoeui.ttf", "times.ttf",
        "georgia.ttf", "trebuc.ttf", "corbel.ttf", "Candara.ttf",
    ],
    "validation": ["tahoma.ttf", "ebrima.ttf"],
    # `holdout` was consumed as diagnostic data once a failure was inspected in
    # it. `holdout_v2` is its untouched replacement and shares no typeface with
    # any earlier split.
    "holdout": ["verdana.ttf", "gadugi.ttf", "micross.ttf", "seguisb.ttf"],
    "holdout_v2": [
        "constan.ttf", "framd.ttf", "LSANS.TTF", "pala.ttf",
        "BOOKOS.TTF", "GARA.TTF", "CENTURY.TTF", "ANTQUAB.TTF",
    ],
    # accent-v3 splits. train_v3 pools the typefaces already burned as
    # development data (train/validation/diagnostic_v1/v2), which is sound
    # because those are the fonts whose failure modes are understood.
    # final_holdout_v3 is sealed and reserves fonts none of them ever saw.
    "train_v3": [
        "arial.ttf", "calibri.ttf", "segoeui.ttf", "times.ttf",
        "georgia.ttf", "trebuc.ttf", "corbel.ttf", "Candara.ttf",
        "verdana.ttf", "gadugi.ttf", "micross.ttf", "seguisb.ttf",
    ],
    "validation_v3": ["tahoma.ttf", "ebrima.ttf", "GARA.TTF", "pala.ttf"],
    "final_holdout_v3": [
        "constan.ttf", "framd.ttf", "LSANS.TTF", "BOOKOS.TTF",
        "CENTURY.TTF", "ANTQUAB.TTF", "BELL.TTF", "l_10646.ttf",
    ],
}

# Phrase templates, also split. Templates deliberately contain no `e` or `é` of
# their own, so every e-form in a rendering comes from the inserted word and the
# glyph label is unambiguous. They still supply realistic neighbours:
# capitals, ascenders, apostrophes, digits and punctuation.
TEMPLATE_SPLITS = {
    "train": [
        "{}",
        "Vous {} localis",
        "Absorption {} habitul",
        "Cannaux {} tuyaux",
        "{} du bac 1,5",
        "L'{} par la nuit",
    ],
    "validation": [
        "Application {} tot",
        "{}: tirs oblats",
    ],
    "holdout": [
        "Voir {} du haut",
        "{} puis validu",
        "Statut du {} 30 min",
    ],
    "holdout_v2": [
        "Bloc {} A1",
        "{} avant tri",
        "Signal du {}, 2,5",
        "I'{} manual",
        "{}",
    ],
    # v3 templates deliberately supply the neighbour contexts that broke
    # earlier versions: capitals and ascenders on the same line, apostrophes
    # and tall glyphs directly beside the target, digits and punctuation.
    "train_v3": [
        "{}",
        "Vous {} localis",
        "Absorption {} habitul",
        "L'{} du bac 1,5",
        "Tk {} biffl",
        "{}: tud 30",
        "Il {} lla",
        "{} kd/hb 2,5",
    ],
    "validation_v3": [
        "Application {} tot",
        "H'{} ktb 1,5",
        "{} du top",
    ],
    "final_holdout_v3": [
        "Manual du {} A1",
        "{}, points 4,5",
        "Db'{} hkli",
        "Bloc {} 30",
        "{}",
    ],
}

# Word pairs: (accented spelling, unaccented spelling). Each word uses exactly
# one e-form throughout, so the drawn form alone determines the visual label.
WORD_SPLITS = {
    "train": [
        ("réglagé", "reglage"), ("élémént", "element"), ("décalé", "decale"),
        ("vérifiér", "verifier"), ("sécurité", "securite"), ("détécté", "detecte"),
        ("résérvé", "reserve"), ("général", "general"),
    ],
    "validation": [("répété", "repete"), ("préféré", "prefere")],
    "holdout": [
        ("réparé", "repare"), ("mélangé", "melange"),
        ("dégagé", "degage"), ("sévéré", "severe"),
    ],
    "holdout_v2": [
        ("créé", "cree"), ("prévénu", "prevenu"),
        ("étété", "etete"), ("rélévé", "releve"),
        ("dénéigé", "deneige"), ("téléphoné", "telephone"),
    ],
    "train_v3": [
        ("réglagé", "reglage"), ("élémént", "element"), ("décalé", "decale"),
        ("vérifiér", "verifier"), ("sécurité", "securite"), ("détécté", "detecte"),
        ("résérvé", "reserve"), ("général", "general"),
        ("réparé", "repare"), ("mélangé", "melange"), ("dégagé", "degage"),
        ("sévéré", "severe"), ("dénéigé", "deneige"),
    ],
    "validation_v3": [
        ("répété", "repete"), ("préféré", "prefere"),
        ("rélévé", "releve"), ("étété", "etete"),
    ],
    "final_holdout_v3": [
        ("bébé", "bebe"), ("céréalé", "cereale"), ("pénétré", "penetre"),
        ("réséqué", "reseque"), ("téméré", "temere"), ("végété", "vegete"),
    ],
}

SIZE_SPLITS = {
    "train": [15, 17, 19, 21, 24],
    "validation": [16, 20],
    "holdout": [18, 22, 26],
    "holdout_v2": [14, 23, 25, 28],
    # v3 covers small sizes deliberately: that is where the accent is only a
    # pixel or two and where accent-v2 failed.
    "train_v3": [12, 13, 15, 17, 19, 21, 24, 27],
    "validation_v3": [14, 16, 20, 26],
    "final_holdout_v3": [11, 18, 22, 23, 25, 30],
}


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


def collapse_ctc(argmax: list[int], labels: list[str]) -> list[dict]:
    emitted = []
    previous = 0
    for timestep, index in enumerate(argmax):
        if index != 0 and index != previous:
            emitted.append(
                {
                    "char": labels[index] if index < len(labels) else "?",
                    "start": timestep,
                    "end": timestep,
                }
            )
        elif emitted and index == previous and index != 0:
            emitted[-1]["end"] = timestep
        previous = index
    return emitted


def render_phrase(text: str, font_path: str, size: int, rng: random.Random) -> Image.Image:
    """Render one phrase with UI-like styling and mild capture noise."""
    font = ImageFont.truetype(font_path, size)
    probe = Image.new("RGB", (8, 8))
    box = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
    pad_x, pad_y = rng.randint(14, 26), rng.randint(10, 20)
    width = box[2] - box[0] + pad_x * 2
    height = box[3] - box[1] + pad_y * 2

    dark = rng.random() < 0.75
    background = (
        (rng.randint(8, 34),) * 3 if dark else (rng.randint(226, 250),) * 3
    )
    foreground = (
        (rng.randint(220, 252),) * 3 if dark else (rng.randint(8, 40),) * 3
    )

    image = Image.new("RGB", (max(width, 90), max(height, 44)), background)
    # Sub-pixel jitter, so glyphs do not always land on integer positions.
    offset = (pad_x + rng.uniform(-0.5, 0.5), pad_y + rng.uniform(-0.5, 0.5))
    ImageDraw.Draw(image).text(offset, text, font=font, fill=foreground)

    if rng.random() < 0.4:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.15, 0.55)))
    if rng.random() < 0.4:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.82, 1.20))
    if rng.random() < 0.3:
        scale = rng.uniform(1.15, 1.8)
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            rng.choice([Image.Resampling.BICUBIC, Image.Resampling.LANCZOS]),
        )
    return image


def extract_glyphs(
    engine, image: Image.Image, labels: list[str], rng: random.Random
) -> list[dict]:
    """Return crops for every predicted `é` and `e`, with CTC-derived spans."""
    bgr = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
    boxes, _ = engine.auto_text_det(bgr)
    if boxes is None or len(boxes) == 0:
        return []
    crops = engine.get_crop_img_list(bgr, boxes)
    recognizer = engine.text_rec
    channels, height, width = recognizer.rec_image_shape

    # ACCENT_PREPROCESS_BASELINE: the product's batched normalization, frozen.
    ratios = [c.shape[1] / float(c.shape[0]) for c in crops]
    max_wh_ratio = max([width / height] + ratios)

    found = []
    for line_index, crop in enumerate(crops):
        crop_h, crop_w = crop.shape[:2]
        tensor = recognizer.resize_norm_img(crop, max_wh_ratio)[np.newaxis, :]
        logits = np.asarray(recognizer.session(tensor.astype(np.float32))[0])
        probabilities = logits[0]
        argmax = probabilities.argmax(axis=-1).tolist()
        confidence = probabilities.max(axis=-1)
        decoded = recognizer.postprocess_op(
            logits, False,
            wh_ratio_list=[crop_w / float(crop_h)], max_wh_ratio=max_wh_ratio,
        )[0][0]

        emitted = collapse_ctc(argmax, labels)
        if "".join(item["char"] for item in emitted) != decoded:
            continue  # alignment not trustworthy for this line

        # Timesteps span the *padded* tensor width, not just the content, so
        # convert timestep -> padded x -> crop x. When a line is padded well
        # beyond its content these differ substantially.
        padded_w = int(height * max_wh_ratio)
        resized_w = min(padded_w, int(np.ceil(height * (crop_w / crop_h))))
        timesteps = probabilities.shape[0]
        for position, item in enumerate(emitted):
            character = unicodedata.normalize("NFC", item["char"])
            if character not in {"e", "é"}:
                continue
            span_confidence = float(
                np.mean(confidence[item["start"] : item["end"] + 1])
            )
            scale = (padded_w / timesteps) * (crop_w / resized_w)
            x_start = (item["start"] + 0.5) * scale
            x_end = (item["end"] + 1 + 0.5) * scale
            # Jitter the pad so training sees the same span error inference does.
            pad = rng.randint(2, 6)
            x0 = max(0, int(np.floor(x_start)) - pad)
            x1 = min(crop_w, int(np.ceil(x_end)) + pad)
            if x1 - x0 < 4:
                continue
            found.append(
                {
                    "line_index": line_index,
                    "position": position,
                    "predicted_char": character,
                    "decoded": decoded,
                    "x0": x0,
                    "x1": x1,
                    "span_confidence": span_confidence,
                    "line_crop": crop,
                }
            )
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--split",
                        choices=tuple(FONT_SPLITS),
                        required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=400,
                        help="number of phrase renderings to attempt")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    args = parser.parse_args()

    # Seeds are split too, so a holdout rendering can never repeat a train one.
    seed = args.seed
    if seed is None:
        seed = {
            "train": 1000, "validation": 5000, "holdout": 9000,
            "holdout_v2": 24000,
            # v3 seed ranges do not overlap any earlier split.
            "train_v3": 100000, "validation_v3": 200000,
            "final_holdout_v3": 300000,
        }[args.split]
    rng = random.Random(seed)

    # The labelling rule assumes each rendering contains exactly one e-form.
    # Verify that here rather than discovering mislabelled glyphs later.
    for split_name, pairs in WORD_SPLITS.items():
        for accented, plain in pairs:
            for spelling, expected in ((accented, "é"), (plain, "e")):
                forms = {
                    c for c in unicodedata.normalize("NFC", spelling)
                    if c in {"e", "é"}
                }
                if forms != {expected}:
                    print(
                        f"word {spelling!r} in split {split_name} has e-forms "
                        f"{sorted(forms)}, expected only {expected!r}",
                        file=sys.stderr,
                    )
                    return 1
    for split_name, templates_ in TEMPLATE_SPLITS.items():
        for template in templates_:
            forms = {
                c for c in unicodedata.normalize("NFC", template.replace("{}", ""))
                if c in {"e", "é"}
            }
            if forms:
                print(
                    f"template {template!r} in split {split_name} contains "
                    f"e-forms {sorted(forms)}; templates must contain none",
                    file=sys.stderr,
                )
                return 1

    engine, package = build_engine(args.package, args.language)
    labels = load_labels(package.dictionary)

    fonts = [str(args.font_dir / name) for name in FONT_SPLITS[args.split]]
    fonts = [f for f in fonts if Path(f).is_file()]
    if not fonts:
        print("no fonts available for this split", file=sys.stderr)
        return 1
    templates = TEMPLATE_SPLITS[args.split]
    words = WORD_SPLITS[args.split]
    sizes = SIZE_SPLITS[args.split]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "e").mkdir(exist_ok=True)
    (args.out_dir / "accent").mkdir(exist_ok=True)

    manifest = []
    counts = {"e": 0, "é": 0}
    hallucinations = 0
    misses = 0

    for index in range(args.samples):
        font = rng.choice(fonts)
        size = rng.choice(sizes)
        template = rng.choice(templates)
        accented_word, plain_word = rng.choice(words)
        use_accent = rng.random() < 0.5
        word = accented_word if use_accent else plain_word
        text = template.format(word)

        image = render_phrase(text, font, size, rng)
        try:
            glyphs = extract_glyphs(engine, image, labels, rng)
        except Exception:
            continue

        # The visual label comes from what was drawn, never from OCR -- that is
        # what makes a hallucinated accent usable as a training sample.
        #
        # A phrase mixes e and é, so a glyph can only be labelled when every
        # e-form in the drawn text agrees. Templates contribute plain `e`s, so
        # an accented word yields a usable label only when the whole phrase is
        # unambiguous; otherwise the glyph is skipped rather than guessed.
        drawn_e_forms = {
            character
            for character in unicodedata.normalize("NFC", text)
            if character in {"e", "é"}
        }
        if len(drawn_e_forms) != 1:
            continue
        visual = drawn_e_forms.pop()

        for glyph in glyphs:

            if glyph["predicted_char"] == "é" and visual == "e":
                hallucinations += 1
            if glyph["predicted_char"] == "e" and visual == "é":
                misses += 1

            crop = glyph["line_crop"][:, glyph["x0"] : glyph["x1"]]
            if crop.size == 0:
                continue
            digest = hashlib.sha256(
                np.ascontiguousarray(crop).tobytes()
            ).hexdigest()[:16]
            folder = "accent" if visual == "é" else "e"
            name = f"{args.split}_{index:05d}_{glyph['position']:03d}_{digest}.png"
            cv2.imwrite(str(args.out_dir / folder / name), crop)
            counts[visual] += 1
            manifest.append(
                {
                    "file": f"{folder}/{name}",
                    "visual_label": visual,
                    "predicted_char": glyph["predicted_char"],
                    "font": Path(font).name,
                    "size": size,
                    "template": template,
                    "word": word,
                    "text": text,
                    "span_confidence": glyph["span_confidence"],
                    "crop_size": [int(crop.shape[1]), int(crop.shape[0])],
                }
            )

        if (index + 1) % 50 == 0:
            print(f"  {index + 1}/{args.samples} renderings, "
                  f"{counts['e']} e / {counts['é']} accent", flush=True)

    (args.out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "seed": seed,
                "fonts": FONT_SPLITS[args.split],
                "templates": templates,
                "words": words,
                "sizes": sizes,
                "counts": counts,
                "recognizer_hallucinated_accent": hallucinations,
                "recognizer_missed_accent": misses,
                "samples": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nsplit={args.split} seed={seed}")
    print(f"  visual e      : {counts['e']}")
    print(f"  visual accent : {counts['é']}")
    print(f"  recognizer hallucinated an accent : {hallucinations}")
    print(f"  recognizer missed a real accent   : {misses}")
    print(f"wrote {args.out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
