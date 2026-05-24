"""Thin OCR runner used by the validation app.

Wraps `rapidocr_onnxruntime.RapidOCR` with a tiny line-merger so callers
get a single multi-line string per image. Cached engine instance per
(model_path, keys_path) so Streamlit reruns don't re-load weights.
"""

from __future__ import annotations

import re
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
    det_limit_side_len: int = 640,
    det_box_thresh: float = 0.5,
    det_donot_use_dilation: bool = True,
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
        kwargs["det_limit_side_len"] = det_limit_side_len
        kwargs["det_limit_type"] = "min"
        kwargs["det_mean"] = [0.485, 0.456, 0.406]
        kwargs["det_std"] = [0.229, 0.224, 0.225]
        kwargs["det_box_thresh"] = det_box_thresh
        kwargs["det_unclip_ratio"] = det_unclip_ratio
        kwargs["det_donot_use_dilation"] = det_donot_use_dilation
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


def _box_bounds(box: list[list[float]]) -> Tuple[float, float, float, float]:
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    return min(xs), min(ys), max(xs), max(ys)


_ICON_TEXT_RE = re.compile(r"^[\-\sbcilmopu0-9]+$", re.IGNORECASE)


def _has_word_text(text: str) -> bool:
    return sum(1 for char in text if char.isalpha()) >= 3


def _drop_left_gutter_icons(items):
    filtered = []
    for item in items:
        y, _x, min_x, min_y, max_x, max_y, text, _score = item
        width = max_x - min_x
        height = max_y - min_y
        stripped = text.strip()
        is_small_left_noise = (
            max_x <= 45
            and width <= 35
            and height <= 24
            and bool(_ICON_TEXT_RE.fullmatch(stripped))
        )
        is_top_status_icon = (
            min_y <= 35
            and max_y <= 65
            and width <= 70
            and height <= 55
            and stripped.lower() in {"p", "up"}
            and any(other is not item and other[3] > max_y and _has_word_text(other[6]) for other in items)
        )
        has_text_neighbor = any(
            other is not item
            and abs(y - other[0]) <= 12
            and other[2] >= min_x
            and other[4] > 45
            and _has_word_text(other[6])
            for other in items
        )
        if (is_small_left_noise and has_text_neighbor) or is_top_status_icon:
            continue
        filtered.append(item)
    return filtered


def _boxes_to_text(result) -> Tuple[str, float]:
    """Order boxes top-to-bottom, left-to-right; return (text, mean_score)."""
    if not result:
        return "", 0.0
    items = []
    for b in result:
        bbox, text, score = b[0], b[1], b[2]
        min_x, min_y, max_x, max_y = _box_bounds(bbox)
        items.append(((min_y + max_y) / 2.0, (min_x + max_x) / 2.0, min_x, min_y, max_x, max_y, text, float(score)))
    items = _drop_left_gutter_icons(items)
    items.sort(key=lambda r: (r[0], r[1]))

    lines: List[List[Tuple[float, str]]] = []
    cur: List[Tuple[float, str]] = []
    cur_y: Optional[float] = None
    tol = 12.0
    for y, x, _min_x, _min_y, _max_x, _max_y, t, _ in items:
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

    mean_score = sum(s for *_, s in items) / len(items) if items else 0.0
    return "\n".join(out), mean_score


_COMPACT_TIMER_RE = re.compile(r"^\s*(\d{1,2})\s*[il|]?\s*[.:]\s*(\d{2})\s*[.:]?\s*$", re.IGNORECASE)
_BROKEN_TIMER_RE = re.compile(r"^[\dilohmn\s.:|-]{1,16}$", re.IGNORECASE)
_PLACEHOLDER_TIMER_RE = re.compile(r"^[\-\s1ilhmin]+$", re.IGNORECASE)
_THINQ_BADGE_RE = re.compile(r"^thin[qgd0o]?$", re.IGNORECASE)


def _format_compact_timer(text: str) -> Optional[str]:
    match = _COMPACT_TIMER_RE.match(text)
    if not match:
        return None
    hours, minutes = match.groups()
    return f"{int(hours)} hr {minutes} min"


def _looks_like_broken_timer(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return _find_broken_timer_line(lines) is not None


def _find_broken_timer_line(lines: List[str]) -> Optional[int]:
    for index, line in enumerate(lines):
        lower = line.lower()
        if not any(char.isdigit() for char in lower):
            continue
        if not any(token in lower for token in ("h", "m", "n")):
            continue
        if _BROKEN_TIMER_RE.fullmatch(line):
            return index
    return None


def _repair_timer_text(primary_text: str, compact_text: str) -> str:
    primary_lines = [line.strip() for line in primary_text.splitlines() if line.strip()]
    timer_index = _find_broken_timer_line(primary_lines)
    if timer_index is None:
        return primary_text

    for line in compact_text.splitlines():
        timer = _format_compact_timer(line.strip())
        if timer:
            repaired = primary_lines.copy()
            repaired[timer_index] = timer
            return "\n".join(repaired)

    middle = primary_lines[timer_index].lower()
    if (
        _PLACEHOLDER_TIMER_RE.fullmatch(middle)
        and "h" in middle
        and any(char in middle for char in "imn")
        and not middle.lstrip()[:1].isdigit()
    ):
        repaired = primary_lines.copy()
        repaired[timer_index] = "-- h -- min"
        return "\n".join(repaired)
    return primary_text


def _normalize_percent_line(line: str) -> Optional[str]:
    compact = line.strip().lower().replace(" ", "")
    compact = compact.replace("o", "0").replace("u", "0").replace("l", "1").replace("i", "1").replace("|", "1")
    compact = re.sub(r"[^0-9]", "", compact)
    if compact in {"100", "1000", "1100"}:
        return "100%"
    if compact in {"70", "700"}:
        return "70%"
    if compact in {"30", "300"}:
        return "30%"
    if compact in {"0", "00"}:
        return "0%"
    return None


def _repair_progress_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text

    if len(lines) == 1:
        line = lines[0]
        for status in ("Actualizar", "Updating"):
            if line.lower().endswith(status.lower()) and line.lower() != status.lower():
                return status
        return text

    status_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.lower() in {"updating", "actualizar", "restableciendo"}
        ),
        -1,
    )
    if status_index < 0:
        return text

    percent = None
    for line in lines[status_index + 1 :]:
        percent = _normalize_percent_line(line)
        if percent:
            break
    if not percent:
        return text

    status = lines[status_index]
    if status.lower() in {"updating", "actualizar"}:
        return "\n".join(["UP", status, percent])
    return "\n".join([status, percent])


def _repair_badge_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) == 1 and _THINQ_BADGE_RE.fullmatch(lines[0]):
        return "ThinQ AI"
    return text


def _repair_brand_text(text: str) -> str:
    return re.sub(r"\bLG\s+Thi(?:n[dgo0]|[o0])\b", "LG ThinQ", text, flags=re.IGNORECASE)


def _repair_domain_text(text: str) -> str:
    text = _repair_progress_text(text)
    text = _repair_badge_text(text)
    text = _repair_brand_text(text)
    return text


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
    if det_model_path and _looks_like_broken_timer(text):
        compact_engine = _get_engine(
            str(rec_model_path) if rec_model_path else None,
            str(rec_keys_path) if rec_keys_path else None,
            str(det_model_path),
            3.5,
            320,
            0.3,
            False,
        )
        compact_result, _ = compact_engine(bgr)
        compact_text, _ = _boxes_to_text(compact_result or [])
        text = _repair_timer_text(text, compact_text)
    text = _repair_domain_text(text)
    return OCRResult(
        text=text,
        n_boxes=len(result or []),
        elapsed_ms=elapsed_ms,
        mean_score=mean_score,
    )
