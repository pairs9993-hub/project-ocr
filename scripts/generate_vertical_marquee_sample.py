"""Generate a vertically scrolling multi-line OCR validation sample."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from generate_marquee_ocr_sample import DEFAULT_FONT, draw_marquee_text, load_font, save_gif, save_mp4


DEFAULT_ROWS = [
    "Panne d'électricité",
    "Appuyez sur ▶Ⅱ",
    "pour redémarrer le cycle.",
    "Erreur de communication",
    "L'appareil ne fonctionne",
    "pas correctement.",
    "Si l'erreur persiste,",
    "appelez un technicien.",
    "Anomalie de tension",
    "Débranchez l'appareil.",
    "Si l'erreur persiste,",
    "appelez un technicien.",
    "Erreur de moteur",
    "Retirez des articles et",
    "redémarrez.",
    "Si l'erreur persiste,",
    "appelez un technicien.",
    "Erreur de gel",
    "Décongelez les parties gelées",
    "avant d'utiliser le produit.",
]


def render_frame(
    rows: list[str],
    font: ImageFont.FreeTypeFont,
    canvas_size: tuple[int, int],
    visible_rows: int,
    row_height: int,
    offset: int,
    horizontal_padding: int,
) -> Image.Image:
    width, height = canvas_size
    image = Image.new("RGB", canvas_size, (0, 0, 0))
    viewport_height = visible_rows * row_height
    viewport_top = (height - viewport_height) // 2
    viewport = Image.new("RGB", (width, viewport_height), (0, 0, 0))
    draw = ImageDraw.Draw(viewport)
    cycle_height = len(rows) * row_height

    for repeat in range(2):
        for row_index, text in enumerate(rows):
            y = repeat * cycle_height + row_index * row_height - offset
            bbox = draw.textbbox((0, 0), text, font=font)
            text_height = bbox[3] - bbox[1]
            baseline_y = y + (row_height - text_height) // 2 - bbox[1]
            draw_marquee_text(draw, text, horizontal_padding, baseline_y, font)

    image.paste(viewport, (0, viewport_top))
    return image


def build_frames(
    rows: list[str],
    font: ImageFont.FreeTypeFont,
    canvas_size: tuple[int, int],
    visible_rows: int,
    row_height: int,
    fps: int,
    hold_seconds: float,
    seconds_per_row: float,
    horizontal_padding: int,
) -> list[Image.Image]:
    hold_frames = max(1, round(hold_seconds * fps))
    scroll_frames = max(2, math.ceil(len(rows) * seconds_per_row * fps))
    cycle_height = len(rows) * row_height
    offsets = [0] * hold_frames
    offsets.extend(
        round(cycle_height * frame_index / scroll_frames)
        for frame_index in range(1, scroll_frames + 1)
    )
    return [
        render_frame(rows, font, canvas_size, visible_rows, row_height, offset, horizontal_padding)
        for offset in offsets
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", default=DEFAULT_ROWS)
    parser.add_argument(
        "--visible-rows",
        type=int,
        default=0,
        help="Rows shown at once; 0 fills the full canvas height.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/marquee_ocr_sample"))
    parser.add_argument("--output-name", default="vertical_french_alerts")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    parser.add_argument("--font-size", type=int, default=20)
    parser.add_argument("--row-height", type=int, default=24)
    parser.add_argument("--horizontal-padding", type=int, default=10)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument("--seconds-per-row", type=float, default=1.0)
    args = parser.parse_args()

    visible_rows = args.visible_rows or math.ceil(args.height / args.row_height)
    if len(args.rows) <= visible_rows:
        raise ValueError("The number of --rows must be greater than --visible-rows.")
    if visible_rows * args.row_height > args.height:
        raise ValueError("Visible rows do not fit inside the canvas height.")
    if args.horizontal_padding < 0 or args.horizontal_padding >= args.width:
        raise ValueError("Horizontal padding must be inside the canvas.")
    if args.fps <= 0 or args.hold_seconds < 0 or args.seconds_per_row <= 0:
        raise ValueError("FPS and row speed must be positive; hold duration cannot be negative.")

    font = load_font(args.font, args.font_size)
    frames = build_frames(
        args.rows,
        font,
        (args.width, args.height),
        visible_rows,
        args.row_height,
        args.fps,
        args.hold_seconds,
        args.seconds_per_row,
        args.horizontal_padding,
    )
    gif_path = args.output_dir / f"{args.output_name}.gif"
    mp4_path = args.output_dir / f"{args.output_name}.mp4"
    expected_path = args.output_dir / f"{args.output_name}_expected.txt"
    save_gif(frames, gif_path, args.fps)
    mp4_written = save_mp4(frames, mp4_path, args.fps)
    expected_path.write_text("\n".join(args.rows) + "\n", encoding="utf-8")

    print(f"GIF: {gif_path} ({len(frames)} frames, {len(frames) / args.fps:.2f}s)")
    print(f"MP4: {mp4_path}" if mp4_written else "MP4 skipped: no supported video encoder was found.")
    print(f"Expected: {expected_path}")


if __name__ == "__main__":
    main()
