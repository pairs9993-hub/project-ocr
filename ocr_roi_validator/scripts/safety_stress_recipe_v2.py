"""Seal the geometry safety-stress recipe.

The other three datasets are sized by how often the baseline hallucinates,
which makes them useless for the failure that actually matters at runtime: a
verifier "correcting" something it should have left alone. That risk lives in
conditions the quota-driven splits under-sample -- glyphs at the extremes of the
size range, the [18,24) band where hallucination is rarest, crops clipped at
their edge, lines carrying several e-forms, apostrophes and digits next to the
target.

So this set is built to fixed case counts per condition rather than mined for
events. It has no hallucination quota and must never be added to one: its
prevalence is an artifact of deliberate over-sampling, and pooling it with the
natural splits would corrupt any rate computed from them.

Every case is a condition the verifier must not act on, or must act on
correctly. Nothing here uses Expected text or product screenshots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import ImageFont

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
for extra in (VALIDATOR_ROOT, VALIDATOR_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from ocr_roi_validator.diagnostic_runner import atomic_write_json  # noqa: E402

STRESS_SEED = 20260818

# Fonts spanning all three cohorts: safety must hold everywhere, and unlike the
# hallucination quotas this does not depend on a font's trigger rate.
STRESS_FONTS = [
    "arial.ttf", "calibri.ttf", "segoeui.ttf", "verdana.ttf", "corbel.ttf",
    "Candara.ttf", "trebuc.ttf", "georgia.ttf", "palab.ttf", "times.ttf",
    "framd.ttf", "tahoma.ttf", "consola.ttf",
]

# Words and templates disjoint from every other cohort, including the pilot.
STRESS_WORDS = [
    ("etendre", "éténdré"), ("recolte", "récolté"), ("pelerin", "pélérin"),
    ("fenetre", "fénétré"), ("cepage", "cépagé"),
]
STRESS_TEMPLATES = [
    "{} qz", "Nb {} vw", "{} 7,9 xk", "Pq {} jd", "{}",
]

# Each condition gets a fixed count. These are evaluation cases, so the number
# is chosen up front and not adjusted by what the baseline happens to produce.
CONDITIONS = {
    "SMALL_GLYPH": {
        "cases": 600, "sizes": (8, 9, 10), "upscale": (1.0, 1.0),
        "why": "target near the lower limit of legibility",
    },
    "LARGE_GLYPH": {
        "cases": 600, "sizes": (26, 30, 34), "upscale": (1.0, 1.0),
        "why": "target well above the sizes the quota splits emphasise",
    },
    "LOW_RATE_BAND_18_24": {
        "cases": 800, "sizes": (16, 17, 18), "upscale": (1.35, 1.55),
        "why": "ink height 18-24, where hallucination is rarest and a "
               "verifier has least reason to act",
    },
    "CROP_EDGE": {
        "cases": 600, "sizes": (13, 16, 20), "upscale": (1.0, 1.0),
        "pad_x": (1, 3), "pad_y": (1, 3),
        "why": "target ink touching the crop boundary",
    },
    "HIGH_PADDING": {
        "cases": 600, "sizes": (12, 15, 18), "upscale": (1.0, 1.0),
        "pad_x": (80, 140), "pad_y": (20, 20),
        "why": "target occupies a small fraction of a wide crop",
    },
    "NEIGHBOURING_ACCENT": {
        "cases": 800, "sizes": (12, 15, 19), "upscale": (1.0, 1.4),
        "templates": ("Àé {} bl", "{} êt dâ", "Nô {} ï"),
        "why": "accents and ascenders adjacent to the target",
    },
    "MULTIPLE_E_FORMS": {
        "cases": 800, "sizes": (12, 15, 19), "upscale": (1.0, 1.4),
        "templates": ("{} et le", "Le {} de", "{} ee ée"),
        "why": "several e/é on one line, so the ordinal query must pick "
               "the right one",
    },
    "APOSTROPHE_DIGIT_PUNCT": {
        "cases": 800, "sizes": (12, 15, 19), "upscale": (1.0, 1.4),
        "templates": ("L'{} 3,5", "{}: 12/7", "D'{} (9)"),
        "why": "apostrophes, digits and punctuation beside the target",
    },
    "INSERTION_DELETION": {
        "cases": 600, "sizes": (10, 11, 24), "upscale": (1.0, 1.9),
        "blur": (0.55, 0.85),
        "why": "conditions that provoke length changes, where the ordinal "
               "and the rendered position disagree",
    },
    "ORDINAL_CTC_DISAGREEMENT": {
        "cases": 600, "sizes": (11, 13, 22), "upscale": (1.6, 2.1),
        "why": "wide crops where CTC token order and glyph order diverge",
    },
    "LEGITIMATE_ACCENT_PRESERVED": {
        "cases": 1200, "sizes": (11, 14, 18, 24), "upscale": (1.0, 1.5),
        "force_accent": True,
        "why": "a real é the baseline read correctly -- the verifier must "
               "leave it alone; this is the false-correction case",
    },
    "BARE_E_MUST_NOT_CHANGE": {
        "cases": 1200, "sizes": (11, 14, 18, 24), "upscale": (1.0, 1.5),
        "force_bare": True,
        "why": "a bare e the baseline read correctly -- the verifier must "
               "not invent an accent",
    },
}

TOTAL_CASES = sum(entry["cases"] for entry in CONDITIONS.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    args = parser.parse_args()

    fonts = {}
    for name in STRESS_FONTS:
        path = (args.font_dir / name).resolve()
        if not path.is_file():
            print(f"missing font {name}", file=sys.stderr)
            return 1
        fonts[name] = {
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "family": ImageFont.truetype(str(path), 20).getname()[0],
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    recipe = {
        "dataset": "line_geometry_safety_stress_v2",
        "role": "false_correction_safety_evaluation_only",
        "seed": STRESS_SEED,
        "fonts": fonts,
        "words": STRESS_WORDS,
        "templates": STRESS_TEMPLATES,
        "conditions": CONDITIONS,
        "total_cases": TOTAL_CASES,
        "fixed_case_counts": True,
        "hallucination_quota": None,
        "prohibited": [
            "counting toward any natural hallucination quota",
            "pooling with supplement/calibration/preflight for rate estimation",
            "use of Expected text or product screenshots",
            "i/l confusion, which is out of scope for this verifier",
        ],
        "why_separate": (
            "prevalence here is an artifact of deliberate over-sampling of "
            "rare conditions; mixing it into the natural splits would corrupt "
            "any rate computed from them"),
        "stop_rule": ("every condition renders its fixed case count exactly; "
                      "there is no event-driven stopping"),
    }
    digest = atomic_write_json(args.out_dir / "recipe.json", recipe)
    file_digest = hashlib.sha256(
        (args.out_dir / "recipe.json").read_bytes()).hexdigest()

    print(f"{'condition':30s} {'cases':>6s}  why")
    for name, entry in CONDITIONS.items():
        print(f"{name:30s} {entry['cases']:6d}  {entry['why'][:60]}")
    print(f"{'TOTAL':30s} {TOTAL_CASES:6d}")
    print(f"\nfonts {len(fonts)}, recipe file sha256 {file_digest}")
    print(f"payload sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
