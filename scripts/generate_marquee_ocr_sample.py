"""Generate a one-line UI marquee sample for OCR validation."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_TEXT = "Drainage error. Check for a blocked drain filter or bent drain hose."
DEFAULT_FONT = Path("C:/Windows/Fonts/segoeui.ttf")
START_KEY = "▶Ⅱ"


def smoothstep(progress: float) -> float:
    return progress * progress * (3.0 - 2.0 * progress)


def load_font(font_path: Path, font_size: int) -> ImageFont.FreeTypeFont:
    if not font_path.exists():
        raise FileNotFoundError(f"Font not found: {font_path}")
    return ImageFont.truetype(str(font_path), font_size)


def measure_marquee_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    width = 0
    height = 0
    parts = text.split(START_KEY)
    for index, part in enumerate(parts):
        if part:
            bbox = draw.textbbox((0, 0), part, font=font)
            width += bbox[2] - bbox[0]
            height = max(height, bbox[3] - bbox[1])
        if index < len(parts) - 1:
            width += max(18, int(font.size * 1.25))
            height = max(height, font.size)
    return width, height


def draw_marquee_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    baseline_y: int,
    font: ImageFont.FreeTypeFont,
) -> None:
    cursor = x
    parts = text.split(START_KEY)
    for index, part in enumerate(parts):
        if part:
            draw.text((cursor, baseline_y), part, fill=(255, 255, 255), font=font)
            bbox = draw.textbbox((0, 0), part, font=font)
            cursor += bbox[2] - bbox[0]
        if index >= len(parts) - 1:
            continue

        symbol_width = max(18, int(font.size * 1.25))
        top = baseline_y + max(1, int(font.size * 0.18))
        bottom = baseline_y + max(10, int(font.size * 0.82))
        middle = (top + bottom) // 2
        triangle_width = max(7, int(font.size * 0.35))
        draw.polygon(
            [(cursor + 1, top), (cursor + 1, bottom), (cursor + triangle_width, middle)],
            fill=(255, 255, 255),
        )
        bar_x = cursor + triangle_width + max(4, int(font.size * 0.12))
        bar_width = max(2, int(font.size * 0.08))
        draw.rectangle([bar_x, top, bar_x + bar_width, bottom], fill=(255, 255, 255))
        draw.rectangle(
            [bar_x + bar_width + 3, top, bar_x + bar_width * 2 + 3, bottom],
            fill=(255, 255, 255),
        )
        cursor += symbol_width


def render_frame(
    text: str,
    font: ImageFont.FreeTypeFont,
    canvas_size: tuple[int, int],
    viewport_margin: int,
    offset: int,
    cycle_length: int | None = None,
) -> Image.Image:
    width, height = canvas_size
    image = Image.new("RGB", canvas_size, (0, 0, 0))
    draw = ImageDraw.Draw(image)
    viewport_width = width - 2 * viewport_margin
    text_bbox = draw.textbbox((0, 0), "Ag", font=font)
    _, text_height = measure_marquee_text(draw, text, font)
    baseline_y = (height - text_height) // 2 - text_bbox[1]

    text_layer = Image.new("RGB", (viewport_width, height), (0, 0, 0))
    text_draw = ImageDraw.Draw(text_layer)
    draw_marquee_text(text_draw, text, -offset, baseline_y, font)
    if cycle_length is not None:
        draw_marquee_text(text_draw, text, cycle_length - offset, baseline_y, font)
    image.paste(text_layer, (viewport_margin, 0))
    return image


def build_frames(
    text: str,
    font: ImageFont.FreeTypeFont,
    canvas_size: tuple[int, int],
    viewport_margin: int,
    fps: int,
    hold_seconds: float,
    scroll_seconds: float,
    end_hold_seconds: float,
    loop_gap: int,
) -> list[Image.Image]:
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    text_width, _ = measure_marquee_text(probe, text, font)
    viewport_width = canvas_size[0] - 2 * viewport_margin
    max_offset = max(0, text_width - viewport_width)

    hold_frames = max(1, round(hold_seconds * fps))
    scroll_frames = max(2, round(scroll_seconds * fps)) if max_offset else 0
    end_hold_frames = max(1, round(end_hold_seconds * fps))
    offsets = [0] * hold_frames
    offsets.extend(
        round(max_offset * smoothstep(index / (scroll_frames - 1)))
        for index in range(scroll_frames)
    )
    offsets.extend([max_offset] * end_hold_frames)
    cycle_length = text_width + loop_gap
    if max_offset:
        average_step = max_offset / max(1, scroll_frames - 1)
        wrap_distance = viewport_width + loop_gap
        wrap_frames = max(2, math.ceil(wrap_distance / max(1.0, average_step)))
        offsets.extend(
            round(max_offset + wrap_distance * index / wrap_frames)
            for index in range(1, wrap_frames + 1)
        )
    return [
        render_frame(text, font, canvas_size, viewport_margin, offset, cycle_length)
        for offset in offsets
    ]


def save_gif(frames: list[Image.Image], output_path: Path, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / fps),
        loop=0,
        disposal=2,
    )


def save_mp4(frames: list[Image.Image], output_path: Path, fps: int) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if ffmpeg is not None:
        with tempfile.TemporaryDirectory(prefix="marquee_frames_") as temp_dir:
            frame_dir = Path(temp_dir)
            for index, frame in enumerate(frames):
                frame.save(frame_dir / f"frame_{index:05d}.png")
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-framerate",
                    str(fps),
                    "-i",
                    str(frame_dir / "frame_%05d.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ],
                check=True,
            )
        return True

    try:
        import cv2
        import numpy as np
    except ImportError:
        return False

    width, height = frames[0].size
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        return False
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/marquee_ocr_sample"))
    parser.add_argument("--output-name", default="drainage_marquee")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    parser.add_argument("--font-size", type=int, default=24)
    parser.add_argument("--margin", type=int, default=16)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument("--scroll-seconds", type=float, default=6.0)
    parser.add_argument("--end-hold-seconds", type=float, default=0.75)
    parser.add_argument("--loop-gap", type=int, default=32)
    args = parser.parse_args()

    if args.width <= 2 * args.margin or args.height <= 0 or args.loop_gap < 0:
        raise ValueError("Canvas dimensions must leave a positive text viewport.")
    if args.fps <= 0 or min(args.hold_seconds, args.scroll_seconds, args.end_hold_seconds) < 0:
        raise ValueError("FPS must be positive and durations must be non-negative.")

    font = load_font(args.font, args.font_size)
    frames = build_frames(
        args.text,
        font,
        (args.width, args.height),
        args.margin,
        args.fps,
        args.hold_seconds,
        args.scroll_seconds,
        args.end_hold_seconds,
        args.loop_gap,
    )
    gif_path = args.output_dir / f"{args.output_name}.gif"
    mp4_path = args.output_dir / f"{args.output_name}.mp4"
    save_gif(frames, gif_path, args.fps)
    mp4_written = save_mp4(frames, mp4_path, args.fps)

    duration = len(frames) / args.fps
    print(f"GIF: {gif_path} ({len(frames)} frames, {duration:.2f}s)")
    print(f"MP4: {mp4_path}" if mp4_written else "MP4 skipped: no supported video encoder was found.")


if __name__ == "__main__":
    main()