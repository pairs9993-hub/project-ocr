"""Thin OCR runner used by the validation app.

Wraps `rapidocr_onnxruntime.RapidOCR` with a tiny line-merger so callers
get a single multi-line string per image. Cached engine instance per
(model_path, keys_path) so Streamlit reruns don't re-load weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class OCRResult:
    text: str
    n_boxes: int
    elapsed_ms: float
    mean_score: float


@lru_cache(maxsize=4)
def _get_engine(
    rec_model_path: Optional[str],
    rec_keys_path: Optional[str],
    det_model_path: Optional[str] = None,
    det_unclip_ratio: float = 3.5,
):
    from rapidocr_onnxruntime import RapidOCR

    kwargs = {}
    if rec_model_path:
        kwargs["rec_model_path"] = rec_model_path
    if rec_keys_path:
        kwargs["rec_keys_path"] = rec_keys_path
    kwargs["use_cls"] = False
    if det_model_path:
        kwargs["det_model_path"] = det_model_path
        kwargs["det_limit_side_len"] = 640
        kwargs["det_limit_type"] = "min"
        kwargs["det_mean"] = [0.485, 0.456, 0.406]
        kwargs["det_std"] = [0.229, 0.224, 0.225]
        kwargs["det_box_thresh"] = 0.5
        kwargs["det_unclip_ratio"] = det_unclip_ratio
        kwargs["det_donot_use_dilation"] = True
    return RapidOCR(**kwargs)


def _mean_result_score(result) -> float:
    if not result:
        return 0.0
    return sum(float(item[2]) for item in result) / len(result)


def _select_result(primary, fallback):
    if not primary:
        return fallback or []
    if fallback and len(fallback) > len(primary) and _mean_result_score(fallback) >= _mean_result_score(primary):
        return fallback
    return primary


def _boxes_to_text(result) -> Tuple[str, float]:
    """Order boxes top-to-bottom, left-to-right; return (text, mean_score)."""
    if not result:
        return "", 0.0
    items = []
    for b in result:
        bbox, text, score = b[0], b[1], b[2]
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        items.append((sum(ys) / 4.0, sum(xs) / 4.0, text, float(score)))
    items.sort(key=lambda r: (r[0], r[1]))

    lines: List[List[Tuple[float, str]]] = []
    cur: List[Tuple[float, str]] = []
    cur_y: Optional[float] = None
    tol = 12.0
    for y, x, t, _ in items:
        if cur_y is None or abs(y - cur_y) <= tol:
            cur.append((x, t))
            cur_y = y if cur_y is None else (cur_y + y) / 2
        else:
            lines.append(cur)
            cur = [(x, t)]
            cur_y = y
    if cur:
        lines.append(cur)

    out = []
    for line in lines:
        line.sort()
        out.append(" ".join(t for _, t in line))

    mean_score = sum(s for *_, s in items) / len(items)
    return "\n".join(out), mean_score


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)  # RGB
    return arr[:, :, ::-1].copy()  # BGR


def ocr_image(
    img: Image.Image,
    rec_model_path: Optional[Path] = None,
    rec_keys_path: Optional[Path] = None,
    det_model_path: Optional[Path] = None,
) -> OCRResult:
    engine = _get_engine(
        str(rec_model_path) if rec_model_path else None,
        str(rec_keys_path) if rec_keys_path else None,
        str(det_model_path) if det_model_path else None,
    )
    bgr = pil_to_bgr(img)
    t0 = perf_counter()
    result, _ = engine(bgr)
    if det_model_path:
        fallback_engine = _get_engine(
            str(rec_model_path) if rec_model_path else None,
            str(rec_keys_path) if rec_keys_path else None,
            str(det_model_path),
            3.0,
        )
        fallback_result, _ = fallback_engine(bgr)
        result = _select_result(result or [], fallback_result or [])
    elapsed_ms = (perf_counter() - t0) * 1000.0
    text, mean_score = _boxes_to_text(result or [])
    return OCRResult(
        text=text,
        n_boxes=len(result or []),
        elapsed_ms=elapsed_ms,
        mean_score=mean_score,
    )
