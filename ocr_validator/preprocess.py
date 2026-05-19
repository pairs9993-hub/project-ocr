"""
Module 1: Preprocessor.

Composable pipeline for normalizing UI screenshots before OCR. Each stage is
on/off via PreprocessConfig; intermediate results are exposed via
`run_pipeline_traced` for visualization and debugging.

Stages:
  - to_gray         : BGR -> single-channel intensity (V channel of HSV by default
                       so colored text stays bright; optional plain luminance)
  - tophat          : white top-hat (gray - opening) to suppress slowly-varying
                       backgrounds (gradients, vignettes, toast bars)
  - clahe           : contrast-limited adaptive histogram equalization
  - invert          : invert if text is darker than background
  - pad_white       : white border around the image (helps detection at edges)

All stages return uint8 single-channel arrays except the final stage (which
returns BGR by stacking the gray channel, suitable for downstream RGB-input
detectors).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np


# ---------- config ----------


@dataclass
class PreprocessConfig:
    use_v_channel: bool = True          # HSV V vs plain BGR2GRAY
    apply_tophat: bool = True
    tophat_kernel_ratio: float = 1.0 / 3.0
    tophat_kernel_min: int = 9          # floor (odd) so kernel isn't degenerate
    apply_clahe: bool = True
    clahe_clip_limit: float = 3.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    apply_invert: bool = False
    pad_white: int = 8                  # px white border (0 = off)


# ---------- primitive stages ----------


def to_gray(bgr: np.ndarray, use_v_channel: bool) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr
    if use_v_channel:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _odd(n: int) -> int:
    n = max(1, int(n))
    return n if n % 2 == 1 else n + 1


def tophat(gray: np.ndarray, kernel_ratio: float, kernel_min: int) -> np.ndarray:
    h = gray.shape[0]
    k = max(kernel_min, _odd(int(round(h * kernel_ratio))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)


def apply_clahe(gray: np.ndarray, clip_limit: float, tile_grid: Tuple[int, int]) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)


def invert(gray: np.ndarray) -> np.ndarray:
    return 255 - gray


def pad_white(gray: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return gray
    return cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)


# ---------- pipeline runner ----------


Trace = List[Tuple[str, np.ndarray]]


def run_pipeline_traced(bgr: np.ndarray, cfg: PreprocessConfig) -> Trace:
    """Run the pipeline and return [(stage_name, image), ...] including the input."""
    trace: Trace = [("input", bgr.copy())]

    gray = to_gray(bgr, cfg.use_v_channel)
    trace.append(("gray_v" if cfg.use_v_channel else "gray", gray))

    if cfg.apply_tophat:
        gray = tophat(gray, cfg.tophat_kernel_ratio, cfg.tophat_kernel_min)
        trace.append(("tophat", gray))

    if cfg.apply_clahe:
        gray = apply_clahe(gray, cfg.clahe_clip_limit, cfg.clahe_tile_grid)
        trace.append(("clahe", gray))

    if cfg.apply_invert:
        gray = invert(gray)
        trace.append(("invert", gray))

    if cfg.pad_white > 0:
        gray = pad_white(gray, cfg.pad_white)
        trace.append(("pad", gray))

    return trace


def preprocess(bgr: np.ndarray, cfg: PreprocessConfig | None = None) -> np.ndarray:
    """Run the pipeline and return the final image (single-channel uint8)."""
    cfg = cfg or PreprocessConfig()
    trace = run_pipeline_traced(bgr, cfg)
    return trace[-1][1]


# ---------- visualization ----------


def trace_to_grid(trace: Trace, label_height: int = 22, scale: int = 2) -> np.ndarray:
    """Lay out the trace horizontally with stage names above each panel."""
    panels = []
    target_h = max(img.shape[0] for _, img in trace)
    for name, img in trace:
        if img.ndim == 2:
            disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            disp = img
        h, w = disp.shape[:2]
        if h != target_h:
            disp = cv2.copyMakeBorder(
                disp, 0, target_h - h, 0, 0,
                cv2.BORDER_CONSTANT, value=(40, 40, 40),
            )
        if scale != 1:
            disp = cv2.resize(disp, (disp.shape[1] * scale, disp.shape[0] * scale),
                              interpolation=cv2.INTER_NEAREST)
        # add label strip
        strip = np.full((label_height, disp.shape[1], 3), 30, dtype=np.uint8)
        cv2.putText(strip, name, (6, label_height - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        panel = np.vstack([strip, disp])
        panels.append(panel)
    # pad panels to equal height (label strip is constant; image already padded)
    max_h = max(p.shape[0] for p in panels)
    panels = [
        cv2.copyMakeBorder(p, 0, max_h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        for p in panels
    ]
    # 8px gutter between panels
    gutter = np.full((panels[0].shape[0], 8, 3), 0, dtype=np.uint8)
    out = panels[0]
    for p in panels[1:]:
        out = np.hstack([out, gutter, p])
    return out


def visualize(image_path: str | Path, output_path: str | Path,
              cfg: PreprocessConfig | None = None) -> None:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    cfg = cfg or PreprocessConfig()
    trace = run_pipeline_traced(bgr, cfg)
    grid = trace_to_grid(trace)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid)
