"""Re-run the paired e/é audit on the final CNN input tensor.

The earlier audit showed the native crop separates the two classes. That is
necessary but not sufficient: the transformation into a fixed-size tensor
(downscaling, padding, normalization) can destroy the difference it preserved.
Any collision introduced *here* is a condition the network cannot possibly
judge, so it must be excluded from correction and answered ``UNKNOWN``.

Ground truth is the rendering, never OCR. No real UI imagery is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATOR_ROOT))
SCRIPTS = VALIDATOR_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_accent_pixel_separability import (  # noqa: E402
    AUDIT_FONTS,
    AUDIT_SIZES,
    AUDIT_TEMPLATES,
    AUDIT_WORDS,
    build_engine,
    load_labels,
    render,
    stage_images,
)
from ocr_roi_validator.accent_cnn_input import (  # noqa: E402
    AccentInputConfig,
    prepare_cnn_input,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--pairs", type=int, default=240)
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    import random

    config = AccentInputConfig()
    engine, package = build_engine(args.package, args.language)
    labels = load_labels(package.dictionary)
    rng = random.Random(77)  # same stream as the stage audit

    rows = []
    attempted = 0
    while len(rows) < args.pairs and attempted < args.pairs * 6:
        attempted += 1
        font = rng.choice(
            [str(args.font_dir / f) for f in AUDIT_FONTS
             if (args.font_dir / f).is_file()]
        )
        size = rng.choice(AUDIT_SIZES)
        template = rng.choice(AUDIT_TEMPLATES)
        accented, plain = rng.choice(AUDIT_WORDS)
        seed = rng.randrange(10 ** 6)

        try:
            accented_stages = stage_images(
                engine, render(template.format(accented), font, size, seed), labels
            )
            plain_stages = stage_images(
                engine, render(template.format(plain), font, size, seed), labels
            )
        except Exception:
            continue
        if accented_stages is None or plain_stages is None:
            continue

        accented_tensor = prepare_cnn_input(accented_stages["native_line"], config)
        plain_tensor = prepare_cnn_input(plain_stages["native_line"], config)
        if accented_tensor is None or plain_tensor is None:
            rows.append(
                {
                    "font": Path(font).name, "size": size,
                    "in_scope": accented_stages["predicted_char"] == "é",
                    "unpreparable": True,
                }
            )
            continue

        difference = np.abs(accented_tensor - plain_tensor)
        full_difference = difference[0, 0]
        accent_difference = difference[0, 1]
        rows.append(
            {
                "font": Path(font).name,
                "size": size,
                "in_scope": accented_stages["predicted_char"] == "é",
                "unpreparable": False,
                "identical": bool(np.array_equal(accented_tensor, plain_tensor)),
                "max_abs": float(difference.max()),
                "l1": float(difference.sum()),
                "changed_elements": int((difference > 1e-6).sum()),
                "full_view_max": float(full_difference.max()),
                "accent_view_max": float(accent_difference.max()),
            }
        )
        if len(rows) % 40 == 0:
            print(f"  {len(rows)}/{args.pairs} pairs", flush=True)

    usable = [r for r in rows if not r.get("unpreparable")]
    in_scope = [r for r in usable if r["in_scope"]]
    collisions = [r for r in usable if r["identical"]]
    unpreparable = [r for r in rows if r.get("unpreparable")]

    maxima = np.array([r["max_abs"] for r in usable]) if usable else np.array([0.0])
    print(f"\npairs: {len(rows)} (usable {len(usable)}, in-scope {len(in_scope)}, "
          f"unpreparable {len(unpreparable)})")
    print(f"byte-identical collisions after CNN transform : {len(collisions)}")
    print(f"max|diff| min      : {maxima.min():.6f}")
    print(f"max|diff| 1st pct  : {np.percentile(maxima, 1):.6f}")
    print(f"max|diff| median   : {np.median(maxima):.6f}")
    print()
    print(f"{'size':>5s} {'pairs':>6s} {'collide':>8s} {'minMax':>9s} "
          f"{'medAccentView':>14s}")
    by_size = {}
    for size in sorted({r["size"] for r in usable}):
        subset = [r for r in usable if r["size"] == size]
        entry = {
            "pairs": len(subset),
            "collisions": sum(r["identical"] for r in subset),
            "min_max_abs": float(min(r["max_abs"] for r in subset)),
            "median_accent_view_max": float(
                np.median([r["accent_view_max"] for r in subset])
            ),
        }
        by_size[size] = entry
        print(f"{size:>5d} {entry['pairs']:>6d} {entry['collisions']:>8d} "
              f"{entry['min_max_abs']:>9.6f} "
              f"{entry['median_accent_view_max']:>14.6f}")

    small = [r for r in usable if r["size"] in (13, 14)]
    if small:
        print(f"\n13/14pt: {len(small)} pairs, "
              f"{sum(r['identical'] for r in small)} collisions, "
              f"min max|diff| {min(r['max_abs'] for r in small):.6f}")

    status = "PASS" if not collisions else ("PARTIAL" if len(collisions) < len(usable) else "FAIL")
    print(f"\nCNN_INPUT_SEPARABILITY = {status}")
    if collisions:
        print("  collisions must be excluded from correction and answered UNKNOWN")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(
                {
                    "input_config": config.as_dict(),
                    "pairs": len(rows),
                    "usable": len(usable),
                    "in_scope": len(in_scope),
                    "unpreparable": len(unpreparable),
                    "collisions": len(collisions),
                    "min_max_abs": float(maxima.min()),
                    "p1_max_abs": float(np.percentile(maxima, 1)),
                    "by_size": by_size,
                    "status": status,
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
