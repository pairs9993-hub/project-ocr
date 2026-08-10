"""Build a normal/defect evaluation pair matching the real failure ROI.

The real captured ROI supplies the normal case (screen text is genuinely
``Veuillez``). There is no captured screen whose text is genuinely
``Véuillez``, so the defect counterpart is rendered synthetically to match the
captured ROI's font, size, colours and crop geometry as closely as possible.

Real screen images are used for evaluation only; nothing here is training data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

LINES_NORMAL = (
    "Veuillez allumer l\u2019eau.",
    "V\u00e9rifiez la pression de l\u2019eau",
    "et les tuyaux d\u2019alimentation.",
)
# Only the first glyph differs: e -> é on "Veuillez".
LINES_DEFECT = (
    "V\u00e9uillez allumer l\u2019eau.",
    "V\u00e9rifiez la pression de l\u2019eau",
    "et les tuyaux d\u2019alimentation.",
)


def sample_colors(image: Image.Image) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Estimate (background, foreground) from the darkest/brightest pixels."""
    arr = np.asarray(image.convert("RGB")).reshape(-1, 3)
    luma = arr @ np.array([0.299, 0.587, 0.114])
    background = tuple(int(v) for v in arr[luma <= np.percentile(luma, 20)].mean(axis=0))
    foreground = tuple(int(v) for v in arr[luma >= np.percentile(luma, 95)].mean(axis=0))
    return background, foreground


def render(
    lines: tuple[str, ...],
    size: tuple[int, int],
    font_path: str,
    font_size: int,
    background: tuple[int, int, int],
    foreground: tuple[int, int, int],
    line_spacing: int,
    top_offset: int,
) -> Image.Image:
    image = Image.new("RGB", size, background)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)
    y = top_offset
    for line in lines:
        width = draw.textlength(line, font=font)
        draw.text(((size[0] - width) / 2, y), line, font=font, fill=foreground)
        y += line_spacing
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-roi", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--font", default="C:/Windows/Fonts/segoeui.ttf")
    parser.add_argument("--font-size", type=int, default=21)
    parser.add_argument("--line-spacing", type=int, default=30)
    parser.add_argument("--top-offset", type=int, default=12)
    args = parser.parse_args()

    with Image.open(args.reference_roi) as handle:
        reference = handle.convert("RGB")
    size = reference.size
    background, foreground = sample_colors(reference)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, lines, visible in (
        ("normal", LINES_NORMAL, LINES_NORMAL),
        ("defect", LINES_DEFECT, LINES_DEFECT),
    ):
        image = render(
            lines,
            size,
            args.font,
            args.font_size,
            background,
            foreground,
            args.line_spacing,
            args.top_offset,
        )
        path = args.out_dir / f"veuillez_{name}.png"
        image.save(path)
        manifest.append(
            {
                "name": name,
                "path": str(path),
                "visible_text": "\n".join(visible),
                # Expected is always the correct product string; for the defect
                # image the visible text deliberately differs from it.
                "expected": "\n".join(LINES_NORMAL),
                "size": list(size),
                "font": args.font,
                "font_size": args.font_size,
                "background": list(background),
                "foreground": list(foreground),
            }
        )
        print(f"wrote {path}  visible={'|'.join(visible)}")

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
