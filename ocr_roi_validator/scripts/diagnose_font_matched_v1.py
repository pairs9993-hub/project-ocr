"""Matched factorial diagnostic: does the e -> é hallucination depend on font?

v1 mining produced 200 clean hallucinations across the six train fonts and
exactly zero across the five calibration/preflight fonts, over 92,000
renderings. That is suggestive but it is not evidence, because those splits
also differed in word, template, size and seed. Any of those could be the real
cause, with font merely along for the ride.

This diagnostic removes the confound. A *base condition* fixes everything --
text, template, target position, nominal size, padding, upscale, background,
contrast, blur, seed, canvas -- and then the same base condition is rendered
once per font. Every font therefore sees an identical optical setting, and the
only thing that varies within a matched group is the typeface.

No model is involved. Only the frozen detector and the frozen French baseline
recognizer, on CPU, exactly as the product runs them.

The verdict must come from the matched numbers. A font's name, or whether it
looks like a serif, is not evidence and is not consulted anywhere below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
for extra in (VALIDATOR_ROOT, VALIDATOR_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from mine_line_triggers_v1 import (  # noqa: E402
    build_engine, classify_occurrence, collapse_ctc, load_labels,
)

# Every font under test, drawn from all three v1 cohorts. Grouped only so the
# report can say where each came from; the grouping carries no assumption.
FONTS = {
    "arial.ttf": "train_v1", "calibri.ttf": "train_v1", "segoeui.ttf": "train_v1",
    "verdana.ttf": "train_v1", "corbel.ttf": "train_v1", "Candara.ttf": "train_v1",
    "trebuc.ttf": "calibration_v1", "georgia.ttf": "calibration_v1",
    "times.ttf": "preflight_v1", "framd.ttf": "preflight_v1",
    "tahoma.ttf": "preflight_v1",
}

# Base-condition axes. Words and templates are pooled from all three v1 cohorts
# so no font is judged on material only its own split ever saw.
WORDS = [
    ("reglage", "réglagé"), ("element", "élémént"), ("decale", "décalé"),
    ("reserve", "résérvé"), ("general", "général"), ("repare", "réparé"),
    ("melange", "mélangé"), ("degage", "dégagé"), ("severe", "sévéré"),
    ("deneige", "dénéigé"),
]
TEMPLATES = [
    "{}", "Vous {} localis", "Il {} lla", "{} kd/hb 2,5", "L'{} du bac 1,5",
    "Application {} tot", "H'{} ktb 1,5", "{} du top", "Tk {} biffl", "{}: tud 30",
]
SIZES = [12, 14, 16, 19, 22, 26]
DIAGNOSTIC_SEED = 8100000


class BaseCondition:
    """One optical setting, shared byte-for-byte across every font."""

    __slots__ = ("index", "word_index", "accented", "template", "size", "pad_x",
                 "pad_y", "dark", "background", "foreground", "jitter_x",
                 "jitter_y", "blur", "contrast", "upscale", "resample")

    def __init__(self, index: int, rng: random.Random) -> None:
        self.index = index
        self.word_index = rng.randrange(len(WORDS))
        self.accented = rng.random() < 0.5
        self.template = rng.choice(TEMPLATES)
        self.size = rng.choice(SIZES)
        self.pad_x = rng.randint(14, 26)
        self.pad_y = rng.randint(10, 20)
        self.dark = rng.random() < 0.75
        self.background = (rng.randint(8, 34) if self.dark
                           else rng.randint(226, 250))
        self.foreground = (rng.randint(220, 252) if self.dark
                           else rng.randint(8, 40))
        self.jitter_x = rng.uniform(-0.5, 0.5)
        self.jitter_y = rng.uniform(-0.5, 0.5)
        self.blur = rng.uniform(0.15, 0.55) if rng.random() < 0.4 else 0.0
        self.contrast = rng.uniform(0.82, 1.20) if rng.random() < 0.4 else 1.0
        self.upscale = rng.uniform(1.15, 1.8) if rng.random() < 0.3 else 1.0
        self.resample = rng.choice(["bicubic", "lanczos"])

    @property
    def text(self) -> str:
        plain, accented = WORDS[self.word_index]
        return self.template.format(accented if self.accented else plain)

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__slots__}

    def optical_key(self) -> dict:
        """The axes an interaction breakdown is reported against."""
        return {
            "size": self.size,
            "pad_x_band": "pad<=20" if self.pad_x <= 20 else "pad>20",
            "upscale_band": "none" if self.upscale == 1.0 else "upscaled",
            "background_band": "dark" if self.dark else "light",
        }


def render(condition: BaseCondition, font_path: Path) -> Image.Image:
    """Render one base condition in one font. Font is the only free variable."""
    font = ImageFont.truetype(str(font_path), condition.size)
    text = condition.text
    box = ImageDraw.Draw(Image.new("RGB", (8, 8))).textbbox((0, 0), text, font=font)
    width = box[2] - box[0] + condition.pad_x * 2
    height = box[3] - box[1] + condition.pad_y * 2
    image = Image.new("RGB", (max(width, 90), max(height, 44)),
                      (condition.background,) * 3)
    ImageDraw.Draw(image).text(
        (condition.pad_x + condition.jitter_x, condition.pad_y + condition.jitter_y),
        text, font=font, fill=(condition.foreground,) * 3)
    if condition.blur:
        image = image.filter(ImageFilter.GaussianBlur(condition.blur))
    if condition.contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(condition.contrast)
    if condition.upscale != 1.0:
        resample = (Image.Resampling.BICUBIC if condition.resample == "bicubic"
                    else Image.Resampling.LANCZOS)
        image = image.resize((int(image.width * condition.upscale),
                              int(image.height * condition.upscale)), resample)
    return image


def glyph_hash(font_path: Path, char: str, size: int = 20) -> str:
    font = ImageFont.truetype(str(font_path), size)
    canvas = Image.new("L", (60, 44), 0)
    ImageDraw.Draw(canvas).text((6, 6), char, font=font, fill=255)
    return hashlib.sha256(canvas.tobytes()).hexdigest()


def upper_bound_zero(trials: int, confidence: float = 0.95) -> float:
    """One-sided upper bound on a rate after observing zero events."""
    return 1.0 - (1.0 - confidence) ** (1.0 / trials) if trials else 1.0


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    spread = (z * math.sqrt(p * (1 - p) / trials
                            + z * z / (4 * trials * trials)) / denominator)
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    parser.add_argument("--lines-per-font", type=int, default=2000)
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()

    fonts: dict[str, dict] = {}
    for name, origin in FONTS.items():
        path = (args.font_dir / name).resolve()
        if not path.is_file():
            print(f"missing font {name}", file=sys.stderr)
            return 1
        bare, accented = glyph_hash(path, "e"), glyph_hash(path, "é")
        if bare == accented:
            print(f"{name} renders e and é identically -- broken font",
                  file=sys.stderr)
            return 1
        fonts[name] = {
            "path": str(path), "origin_cohort": origin,
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "family": ImageFont.truetype(str(path), 20).getname()[0],
            "glyph_hash_e": bare, "glyph_hash_eacute": accented,
        }
    shared = defaultdict(list)
    for name, info in fonts.items():
        shared[info["glyph_hash_eacute"]].append(name)
    collisions = {k: v for k, v in shared.items() if len(v) > 1}
    if collisions:
        print(f"font fallback suspected, shared rasters: {collisions}",
              file=sys.stderr)
        return 1

    engine, package = build_engine(args.package, args.language)
    labels = load_labels(package.dictionary)
    recognizer = engine.text_rec
    _, rec_height, rec_width = recognizer.rec_image_shape

    rng = random.Random(DIAGNOSTIC_SEED)
    conditions = [BaseCondition(i, rng) for i in range(args.lines_per_font)]

    stats = {name: Counter() for name in fonts}
    interactions = {name: defaultdict(Counter) for name in fonts}
    hallucination_rows: list[dict] = []
    matched_groups = 0

    for condition in conditions:
        matched_groups += 1
        for name, info in fonts.items():
            counter = stats[name]
            counter["attempted"] += 1
            try:
                page = render(condition, Path(info["path"]))
                bgr = np.asarray(page)[:, :, ::-1].copy()
                boxes, _ = engine.auto_text_det(bgr)
            except Exception:
                counter["render_or_detect_error"] += 1
                continue
            if boxes is None or len(boxes) == 0:
                counter["detector_empty"] += 1
                continue
            counter["detector_found"] += 1

            crops = engine.get_crop_img_list(bgr, boxes)
            ratios = [c.shape[1] / float(c.shape[0]) for c in crops]
            max_wh_ratio = max([rec_width / rec_height] + ratios)
            counter["lines"] += len(crops)

            drawn = unicodedata.normalize("NFC", condition.text)
            for crop in crops:
                crop_h, crop_w = crop.shape[:2]
                tensor = recognizer.resize_norm_img(crop, max_wh_ratio)[np.newaxis]
                logits = np.asarray(recognizer.session(tensor.astype(np.float32))[0])
                decoded = recognizer.postprocess_op(
                    logits, False, wh_ratio_list=[crop_w / float(crop_h)],
                    max_wh_ratio=max_wh_ratio)[0][0]
                emitted = collapse_ctc(logits[0].argmax(axis=-1).tolist(), labels)
                normalized = unicodedata.normalize("NFC", decoded)

                if "".join(i["char"] for i in emitted) == decoded:
                    counter["token_alignment_ok"] += 1
                else:
                    counter["token_alignment_failed"] += 1
                    continue
                if normalized == drawn:
                    counter["exact_decode"] += 1
                if len(normalized) != len(drawn):
                    counter["length_mismatch"] += 1
                    if len(normalized) > len(drawn):
                        counter["insertion"] += 1
                    else:
                        counter["deletion"] += 1
                # Glyph-level count: any é the recognizer emitted where the
                # rendering had a bare e, regardless of the rest of the line.
                # This is accent-v3's definition and is NOT a clean trigger.
                for position, item in enumerate(emitted):
                    if unicodedata.normalize("NFC", item["char"]) != "é":
                        continue
                    if position < len(drawn) and drawn[position] == "e":
                        counter["glyph_hallucination"] += 1
                    classification = classify_occurrence(drawn, decoded, position)
                    counter[classification] += 1
                    if classification == "CLEAN_HALLUCINATION":
                        for axis, value in condition.optical_key().items():
                            interactions[name][axis][value] += 1
                        hallucination_rows.append({
                            "font": name, "base_condition": condition.index,
                            "text": condition.text, "template": condition.template,
                            "size": condition.size, "position": position,
                            "decoded": decoded,
                            **condition.optical_key(),
                        })
        if matched_groups % args.progress_every == 0:
            done = {n: stats[n]["CLEAN_HALLUCINATION"] for n in fonts}
            print(f"  base condition {matched_groups}/{len(conditions)} "
                  f"clean-halluc {done}", flush=True)

    report = {
        "diagnostic": "matched_font_factorial_v1",
        "seed": DIAGNOSTIC_SEED,
        "matched_base_conditions": matched_groups,
        "fonts": fonts,
        "words": WORDS, "templates": TEMPLATES, "sizes": SIZES,
        "model_used": "none -- frozen detector and French baseline only, CPU",
        "per_font": {}, "interactions": {},
        "hallucination_rows": hallucination_rows,
    }
    for name in fonts:
        counter = stats[name]
        lines = counter["lines"] or 1
        aligned = counter["token_alignment_ok"]
        clean = counter["CLEAN_HALLUCINATION"]
        report["per_font"][name] = {
            "origin_cohort": fonts[name]["origin_cohort"],
            "attempted": counter["attempted"],
            "detector_found": counter["detector_found"],
            "detector_yield": counter["detector_found"] / (counter["attempted"] or 1),
            "lines": counter["lines"],
            "exact_decode": counter["exact_decode"],
            "exact_decode_rate": counter["exact_decode"] / lines,
            "token_alignment_ok": aligned,
            "token_alignment_rate": aligned / lines,
            "clean_hallucination": clean,
            "clean_hallucination_rate": clean / lines,
            "clean_hallucination_ci95": wilson(clean, counter["lines"]),
            "zero_upper_bound_95": (upper_bound_zero(counter["lines"])
                                    if clean == 0 else None),
            "clean_preservation": counter["CLEAN_PRESERVATION"],
            "clean_preservation_rate": counter["CLEAN_PRESERVATION"] / lines,
            "glyph_hallucination": counter["glyph_hallucination"],
            "glyph_hallucination_rate": counter["glyph_hallucination"] / lines,
            "insertion": counter["insertion"], "deletion": counter["deletion"],
            "length_mismatch": counter["length_mismatch"],
            "change_elsewhere": counter["UNKNOWN_CHANGE_ELSEWHERE"],
            "multiple_changes": counter["UNKNOWN_MULTIPLE_CHANGES"],
            "other_substitution": counter["UNKNOWN_OTHER_SUBSTITUTION"],
            "detector_empty": counter["detector_empty"],
            "render_or_detect_error": counter["render_or_detect_error"],
        }
        report["interactions"][name] = {
            axis: dict(values) for axis, values in interactions[name].items()}

    positive = [n for n in fonts if stats[n]["CLEAN_HALLUCINATION"] > 0]
    total = sum(stats[n]["CLEAN_HALLUCINATION"] for n in fonts)
    templates_hit = {r["template"] for r in hallucination_rows}
    conditions_hit = {r["base_condition"] for r in hallucination_rows}
    concentrated = (max((stats[n]["CLEAN_HALLUCINATION"] for n in fonts),
                        default=0) / total if total else 0.0)

    # A matched design means any surviving difference is attributable to font.
    # Interaction is claimed only when the effect is confined to part of the
    # optical grid; otherwise the plain font effect is the honest description.
    if total == 0:
        verdict = "NOT_CONFIRMED"
    elif len(positive) < 2:
        verdict = "NOT_CONFIRMED"
    elif len(templates_hit) < 2 or len(conditions_hit) < 2:
        verdict = "NOT_CONFIRMED"
    else:
        spread = {axis: len(report["interactions"][positive[0]].get(axis, {}))
                  for axis in ("size", "upscale_band", "background_band")}
        verdict = ("INTERACTION_DEPENDENT" if min(spread.values(), default=0) <= 1
                   else "FONT_DEPENDENT")

    report["verdict_inputs"] = {
        "fonts_with_clean_hallucination": positive,
        "total_clean_hallucination": total,
        "distinct_templates_hit": sorted(templates_hit),
        "distinct_base_conditions_hit": len(conditions_hit),
        "largest_font_share": concentrated,
    }
    report["HALLUCINATION_DOMAIN"] = verdict

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload, encoding="utf-8")

    print(f"\n{'font':16s} {'cohort':16s} {'lines':>6s} {'exact':>7s} "
          f"{'clean':>6s} {'glyph':>6s} {'preserv':>8s}")
    for name in fonts:
        entry = report["per_font"][name]
        print(f"{name:16s} {entry['origin_cohort']:16s} {entry['lines']:6d} "
              f"{entry['exact_decode_rate']:7.3f} {entry['clean_hallucination']:6d} "
              f"{entry['glyph_hallucination']:6d} {entry['clean_preservation']:8d}")
    print(f"\nfonts with clean hallucination : {positive}")
    print(f"largest single-font share      : {concentrated:.1%}")
    print(f"HALLUCINATION_DOMAIN           : {verdict}")
    print(f"report sha256                  : "
          f"{hashlib.sha256(payload.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
