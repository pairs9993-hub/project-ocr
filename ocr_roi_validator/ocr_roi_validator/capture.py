from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep
from typing import Iterator, Optional, Tuple

import mss
from PIL import Image


Rect = Tuple[int, int, int, int]


@dataclass
class CaptureFrame:
    index: int
    timestamp: datetime
    image: Image.Image


def grab_screen_rect(rect: Rect) -> Image.Image:
    with mss.mss() as sct:
        return grab_screen_rect_with(sct, rect)


def grab_screen_rect_with(sct, rect: Rect) -> Image.Image:
    left, top, right, bottom = rect
    width = max(1, right - left)
    height = max(1, bottom - top)

    monitor = {
        "left": left,
        "top": top,
        "width": width,
        "height": height,
    }
    shot = sct.grab(monitor)
    return Image.frombytes("RGB", shot.size, shot.rgb)


def timed_capture(
    rect: Rect,
    duration_sec: float,
    fps: float = 2.0,
    save_dir: Optional[Path] = None,
    save_prefix: str = "capture",
) -> Iterator[CaptureFrame]:
    if fps <= 0:
        raise ValueError("fps must be > 0")

    interval = 1.0 / fps
    total_frames = max(1, int(duration_sec * fps))
    start = perf_counter()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    with mss.mss() as sct:
        for index in range(total_frames):
            frame_start = perf_counter()
            image = grab_screen_rect_with(sct, rect)
            now = datetime.now()

            if save_dir is not None:
                image.save(save_dir / f"{save_prefix}_{index:04d}.png")

            yield CaptureFrame(index=index, timestamp=now, image=image)

            elapsed = perf_counter() - frame_start
            remaining = interval - elapsed
            if remaining > 0:
                sleep(remaining)

    _ = perf_counter() - start
