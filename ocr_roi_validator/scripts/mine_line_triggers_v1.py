"""Mine clean e/é triggers through the detector-backed runtime path.

Three views exist in this stage and only one of them is evidence:

``detector_runtime_view``
    synthetic page -> frozen detector -> line crop -> frozen French baseline.
    This mirrors what run.bat does, so it is the only path whose trigger rates
    mean anything. Calibration and the sealed preflight use it alone.

``direct_visual_view``
    the detector-free crop, kept for auxiliary visual training only. Its
    statistics are never mixed with the runtime path's.

``causal_audit_view``
    label-neutral geometry, for pair auditing only. Never model input.

Two corrections are baked in here. accent-v3 counted a hallucination per glyph
whenever it predicted ``é`` on a plain word, without requiring the rest of the
line to decode correctly; its 209 are not 209 clean triggers, and the two
figures are not comparable. And the detector is not the cleaner path -- direct
crops actually decode more often -- so the detector is used because it is the
product's path, not because it reads better.

The query never comes from the renderer. Every ``é`` the baseline emits is
enumerated, and each occurrence's own token index and token count become the
ordinal query. The rendered text is consulted afterwards, purely to label what
was mined.

Inference is CPU-only by construction: GPU logits differ slightly and would
change which triggers are considered clean, while the product runs on CPU.
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

from ocr_roi_validator.model_package import load_model_package  # noqa: E402

# Cohorts disjoint across the three splits and from every earlier dataset.
FONT_SPLITS = {
    "line_train_v1": [
        "arial.ttf", "calibri.ttf", "segoeui.ttf", "verdana.ttf",
        "corbel.ttf", "Candara.ttf",
    ],
    "line_calibration_v1": ["trebuc.ttf", "georgia.ttf"],
    "line_preflight_holdout_v1": ["times.ttf", "framd.ttf", "tahoma.ttf"],
}
TEMPLATE_SPLITS = {
    "line_train_v1": [
        "{}", "Vous {} localis", "Il {} lla", "{} kd/hb 2,5", "L'{} du bac 1,5",
    ],
    "line_calibration_v1": ["Application {} tot", "H'{} ktb 1,5"],
    "line_preflight_holdout_v1": ["{} du top", "Tk {} biffl", "{}: tud 30"],
}
WORD_SPLITS = {
    "line_train_v1": [
        ("reglage", "réglagé"), ("element", "élémént"), ("decale", "décalé"),
        ("reserve", "résérvé"), ("general", "général"),
    ],
    "line_calibration_v1": [("repare", "réparé"), ("melange", "mélangé")],
    "line_preflight_holdout_v1": [
        ("degage", "dégagé"), ("severe", "sévéré"), ("deneige", "dénéigé"),
    ],
}
SIZE_SPLITS = {
    "line_train_v1": [12, 15, 17, 21, 24],
    "line_calibration_v1": [13, 19, 27],
    "line_preflight_holdout_v1": [14, 16, 22, 26],
}
SEEDS = {"line_train_v1": 7100000, "line_calibration_v1": 7200000,
         "line_preflight_holdout_v1": 7300000}

# Perturbation grid, fixed before generation. Matches the accent-v3 renderer,
# which is the recipe the dry run measured; it is not re-tuned afterwards.
PERTURBATION = {
    "pad_x": [14, 26], "pad_y": [10, 20],
    "dark_background_probability": 0.75,
    "dark_background": [8, 34], "dark_foreground": [220, 252],
    "light_background": [226, 250], "light_foreground": [8, 40],
    "subpixel_jitter": [-0.5, 0.5],
    "blur_probability": 0.4, "blur_radius": [0.15, 0.55],
    "contrast_probability": 0.4, "contrast": [0.82, 1.20],
    "rescale_probability": 0.3, "rescale": [1.15, 1.8],
}

QUOTAS = {
    "line_train_v1": {"clean_hallucination": 200, "clean_preservation": 1000},
    "line_calibration_v1": {"clean_hallucination": 100, "clean_preservation": 500},
    "line_preflight_holdout_v1": {"clean_hallucination": 100,
                                  "clean_preservation": 1000},
}
MAX_RENDERINGS = {
    "line_train_v1": 60000,
    "line_calibration_v1": 32000,
    "line_preflight_holdout_v1": 60000,
}


def render_phrase(text: str, font_path: str, size: int,
                  rng: random.Random) -> Image.Image:
    """The accent-v3 renderer, reproduced exactly."""
    font = ImageFont.truetype(font_path, size)
    box = ImageDraw.Draw(Image.new("RGB", (8, 8))).textbbox((0, 0), text, font=font)
    pad_x = rng.randint(*PERTURBATION["pad_x"])
    pad_y = rng.randint(*PERTURBATION["pad_y"])
    width = box[2] - box[0] + pad_x * 2
    height = box[3] - box[1] + pad_y * 2

    dark = rng.random() < PERTURBATION["dark_background_probability"]
    background = ((rng.randint(*PERTURBATION["dark_background"]),) * 3 if dark
                  else (rng.randint(*PERTURBATION["light_background"]),) * 3)
    foreground = ((rng.randint(*PERTURBATION["dark_foreground"]),) * 3 if dark
                  else (rng.randint(*PERTURBATION["light_foreground"]),) * 3)

    image = Image.new("RGB", (max(width, 90), max(height, 44)), background)
    jitter = PERTURBATION["subpixel_jitter"]
    offset = (pad_x + rng.uniform(*jitter), pad_y + rng.uniform(*jitter))
    ImageDraw.Draw(image).text(offset, text, font=font, fill=foreground)

    if rng.random() < PERTURBATION["blur_probability"]:
        image = image.filter(
            ImageFilter.GaussianBlur(rng.uniform(*PERTURBATION["blur_radius"])))
    if rng.random() < PERTURBATION["contrast_probability"]:
        image = ImageEnhance.Contrast(image).enhance(
            rng.uniform(*PERTURBATION["contrast"]))
    if rng.random() < PERTURBATION["rescale_probability"]:
        scale = rng.uniform(*PERTURBATION["rescale"])
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            rng.choice([Image.Resampling.BICUBIC, Image.Resampling.LANCZOS]))
    return image


def build_engine(package_dir: Path, language: str):
    """Frozen detector and baseline recognizer, CPU provider only.

    GPU logits differ slightly from CPU, which would change which triggers
    count as clean; the product runs on CPU, so mining must too.
    """
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
                            "start": timestep, "end": timestep, "label": index})
        elif emitted and index == previous and index != 0:
            emitted[-1]["end"] = timestep
        previous = index
    return emitted


def classify_occurrence(drawn: str, decoded: str, position: int) -> str:
    """Label one predicted-é occurrence by comparing the two strings.

    Runs after the query has been formed, so it cannot influence the input.
    """
    drawn = unicodedata.normalize("NFC", drawn)
    decoded = unicodedata.normalize("NFC", decoded)
    if len(drawn) != len(decoded):
        return "UNKNOWN_LENGTH_MISMATCH"
    differing = [i for i, (a, b) in enumerate(zip(drawn, decoded)) if a != b]
    if not differing:
        return ("CLEAN_PRESERVATION" if position < len(drawn)
                and drawn[position] == "é" else "UNKNOWN_NO_DIFFERENCE")
    if len(differing) > 1:
        return "UNKNOWN_MULTIPLE_CHANGES"
    index = differing[0]
    if index != position:
        return "UNKNOWN_CHANGE_ELSEWHERE"
    if drawn[index] == "e" and decoded[index] == "é":
        return "CLEAN_HALLUCINATION"
    return "UNKNOWN_OTHER_SUBSTITUTION"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--split", choices=tuple(FONT_SPLITS), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--max-renderings", type=int)
    args = parser.parse_args()

    split = args.split
    quota = QUOTAS[split]
    max_renderings = args.max_renderings or MAX_RENDERINGS[split]

    for name, pairs in WORD_SPLITS.items():
        for plain, accented in pairs:
            for spelling, want in ((plain, "e"), (accented, "é")):
                forms = {c for c in unicodedata.normalize("NFC", spelling)
                         if c in {"e", "é"}}
                if forms != {want}:
                    print(f"word {spelling!r} has e-forms {sorted(forms)}",
                          file=sys.stderr)
                    return 1
    for name, templates in TEMPLATE_SPLITS.items():
        for template in templates:
            if {c for c in template.replace("{}", "") if c in {"e", "é"}}:
                print(f"template {template!r} contains an e-form", file=sys.stderr)
                return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_dir / "checkpoint.jsonl"

    rows: list[dict] = []
    start_index = 0
    resume_count = 0
    if checkpoint_path.is_file():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if rows:
            start_index = max(r["render_index"] for r in rows) + 1
            resume_count = int(rows[-1].get("resume_count", 0)) + 1
            print(f"resuming from rendering {start_index} with {len(rows)} rows "
                  f"(resume #{resume_count})")

    engine, package = build_engine(args.package, args.language)
    labels = load_labels(package.dictionary)
    recognizer = engine.text_rec
    channels, height, width = recognizer.rec_image_shape

    fonts = [str(args.font_dir / f) for f in FONT_SPLITS[split]
             if (args.font_dir / f).is_file()]
    missing = [f for f in FONT_SPLITS[split]
               if not (args.font_dir / f).is_file()]
    if missing:
        print(f"missing fonts: {missing}", file=sys.stderr)
        return 1

    counts = Counter(r["classification"] for r in rows)
    seen = {r["row_digest"] for r in rows}
    detector_found = sum(1 for r in rows if r.get("detector_lines", 0) > 0)
    renderings_done = start_index
    lines_decoded = sum(1 for r in rows)

    rng = random.Random(SEEDS[split])
    handle = checkpoint_path.open("a", encoding="utf-8")
    try:
        for index in range(max_renderings):
            # Draw for every index so a resume reproduces the same stream.
            font = rng.choice(fonts)
            size = rng.choice(SIZE_SPLITS[split])
            template = rng.choice(TEMPLATE_SPLITS[split])
            plain, accented = rng.choice(WORD_SPLITS[split])
            use_accent = rng.random() < 0.5
            text = template.format(accented if use_accent else plain)
            render_seed = rng.randrange(10 ** 9)

            if index < start_index:
                continue
            if (counts["CLEAN_HALLUCINATION"] >= quota["clean_hallucination"]
                    and counts["CLEAN_PRESERVATION"] >= quota["clean_preservation"]):
                print(f"\nquota satisfied at rendering {index}")
                break

            renderings_done = index + 1
            try:
                page = render_phrase(text, font, size, random.Random(render_seed))
                bgr = np.asarray(page)[:, :, ::-1].copy()
                boxes, _ = engine.auto_text_det(bgr)
            except Exception:
                continue
            if boxes is None or len(boxes) == 0:
                continue
            detector_found += 1

            crops = engine.get_crop_img_list(bgr, boxes)
            ratios = [c.shape[1] / float(c.shape[0]) for c in crops]
            max_wh_ratio = max([width / height] + ratios)

            for line_index, crop in enumerate(crops):
                crop_h, crop_w = crop.shape[:2]
                tensor = recognizer.resize_norm_img(
                    crop, max_wh_ratio)[np.newaxis, :]
                logits = np.asarray(
                    recognizer.session(tensor.astype(np.float32))[0])
                probabilities = logits[0]
                decoded = recognizer.postprocess_op(
                    logits, False, wh_ratio_list=[crop_w / float(crop_h)],
                    max_wh_ratio=max_wh_ratio)[0][0]
                emitted = collapse_ctc(
                    probabilities.argmax(axis=-1).tolist(), labels)
                lines_decoded += 1
                if "".join(i["char"] for i in emitted) != decoded:
                    counts["UNKNOWN_CTC_SEQUENCE_MISMATCH"] += 1
                    continue

                # Enumerate every predicted é. The query is this occurrence's
                # own token index and the token count -- both runtime values.
                for position, item in enumerate(emitted):
                    if unicodedata.normalize("NFC", item["char"]) != "é":
                        continue
                    classification = classify_occurrence(text, decoded, position)
                    digest = hashlib.sha256(
                        f"{index}|{line_index}|{position}|{decoded}".encode()
                    ).hexdigest()
                    if digest in seen:
                        continue
                    seen.add(digest)
                    counts[classification] += 1
                    row = {
                        "render_index": index, "line_index": line_index,
                        "row_digest": digest,
                        "ordinal_query": position,
                        "decoded_length": len(emitted),
                        "token_start": item["start"], "token_end": item["end"],
                        "token_label": item["label"],
                        "classification": classification,
                        "drawn_text": text, "decoded_text": decoded,
                        "font": Path(font).name, "size": size,
                        "template": template, "render_seed": render_seed,
                        "crop_size": [int(crop_w), int(crop_h)],
                        "detector_lines": len(crops),
                        "view": "detector_runtime_view",
                        "resume_count": resume_count,
                    }
                    rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            if (index + 1) % args.progress_every == 0:
                handle.flush()
                print(f"  rendering {index + 1}: lines {lines_decoded} "
                      f"halluc {counts['CLEAN_HALLUCINATION']}/"
                      f"{quota['clean_hallucination']} "
                      f"preserv {counts['CLEAN_PRESERVATION']}/"
                      f"{quota['clean_preservation']}", flush=True)
    finally:
        handle.close()

    met = (counts["CLEAN_HALLUCINATION"] >= quota["clean_hallucination"]
           and counts["CLEAN_PRESERVATION"] >= quota["clean_preservation"])
    manifest = {
        "split": split, "seed": SEEDS[split], "view": "detector_runtime_view",
        "fonts": FONT_SPLITS[split], "templates": TEMPLATE_SPLITS[split],
        "words": WORD_SPLITS[split], "sizes": SIZE_SPLITS[split],
        "perturbation": PERTURBATION, "quota": quota,
        "max_renderings": max_renderings,
        "renderings_attempted": renderings_done,
        "detector_found_lines": detector_found,
        "lines_decoded": lines_decoded,
        "classification_counts": dict(counts),
        "rows": len(rows), "resume_count": resume_count,
        "quota_met": met,
        "query_provenance": (
            "ordinal index and token count come from the baseline's own decoded "
            "tokens; the rendered text is used only to label after the fact"
        ),
        "samples": rows,
    }
    payload = json.dumps(manifest, ensure_ascii=False, indent=2)
    (args.out_dir / "manifest.json").write_text(payload, encoding="utf-8")

    print(f"\nsplit {split}")
    print(f"  renderings      : {renderings_done}")
    print(f"  detector found  : {detector_found}")
    print(f"  lines decoded   : {lines_decoded}")
    for key in sorted(counts):
        print(f"  {key:34s} {counts[key]}")
    print(f"  quota met       : {met}")
    print(f"  manifest sha256 : {hashlib.sha256(payload.encode()).hexdigest()}")
    return 0 if met else 1


if __name__ == "__main__":
    raise SystemExit(main())
