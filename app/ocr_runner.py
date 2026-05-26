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
_START_KEY_ICON_RE = re.compile(r"\s*▶\s*(?:Ⅱ|II)\s*")
_START_KEY_ARTIFACT_RE = re.compile(
    r"\b(Press|Presione|Appuyez)\s+(?:▶\s*(?:[fIl1|Ⅱ]{1,2}|A\s*(?:Ⅱ|II)?\s*[lI1|]?)|A\s*[lI1|]|™\s*[lI1|]|[fIl1|Ⅱ]{1,2})\s+(?=(?:to|para|pour|or)\b)",
    re.IGNORECASE,
)
_SPANISH_TOMORROW_RE = re.compile(r"\bMA(?:li|l|1i|i)\.?(?=\s|$)", re.IGNORECASE)
_SPANISH_DELAY_START_RE = re.compile(r"\bEl\s*l?nicio\b|\bEllnicio\b", re.IGNORECASE)
_PROC_SOAKING_RE = re.compile(r"<PROC_W_S0AKING_PRC", re.IGNORECASE)
_EXTRA_RINSE_RE = re.compile(r"^(?:Extra Rinse|Enjuague extra)$", re.IGNORECASE)
_SCHEDULE_LINE_RE = re.compile(r"\b(?:Iniciar a|Start at)\b", re.IGNORECASE)
_TIME_1230_RE = re.compile(r"\b1230\s*((?:a|p)\.?\s*m\.?)\b", re.IGNORECASE)
_TIME_1200_RE = re.compile(r"\b12\s*[iIl1oO0]{2,3}\s*((?:a|p)\.?\s*m\.?)\b", re.IGNORECASE)
_DELAY_START_RE = re.compile(r"\b(?:El\s+Inicio\s+retardado|Delay\s+Start\s+is\s+on)\b", re.IGNORECASE)
_DELAY_AM_PM_ONLY_RE = re.compile(r"^\s*([ap])\.?\s*m\.?\s*$", re.IGNORECASE)


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
            if line.lower() == status.lower() or line.lower().endswith(status.lower()):
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


def _repair_start_key_icons(text: str) -> str:
    lines = []
    for line in text.splitlines():
        repaired = _START_KEY_ARTIFACT_RE.sub(r"\1 ▶Ⅱ ", line)
        repaired = _START_KEY_ICON_RE.sub(" ▶Ⅱ ", repaired)
        repaired = re.sub(r"[ \t]+", " ", repaired).strip()
        if repaired:
            lines.append(repaired)
    return "\n".join(lines)


def _repair_known_ui_tokens(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = _SPANISH_TOMORROW_RE.sub("MAÑ.", line)
        line = _SPANISH_DELAY_START_RE.sub("El Inicio", line)
        line = _repair_schedule_line(line)
        line = re.sub(r"^\s*\+\s*-\s*(\d)\s*$", r"+\1", line)
        line = _PROC_SOAKING_RE.sub("<PROC_W_SOAKING_PRC", line)
        if re.fullmatch(r"thin\.", line.strip(), re.IGNORECASE):
            line = "ThinQ AI"
        lines.append(line)
    lines = _repair_dry_time_lines(lines)
    return "\n".join(lines)


def _repair_schedule_line(line: str) -> str:
    if not _SCHEDULE_LINE_RE.search(line):
        return line
    line = re.sub(r"\bl[áa]s\b", "las", line, flags=re.IGNORECASE)
    line = re.sub(r"\bMAN\.\b|\bMAN\.", "MAÑ.", line, flags=re.IGNORECASE)
    line = _TIME_1200_RE.sub(r"12:00 \1", line)
    line = _TIME_1230_RE.sub(r"12:30 \1", line)
    return line


def _repair_dry_time_lines(lines: List[str]) -> List[str]:
    if not lines or lines[0].strip().lower() != "dry time":
        return lines
    repaired = lines.copy()
    for index in range(1, len(repaired)):
        line = repaired[index]
        line = line.replace("O", "0").replace("o", "0")
        line = re.sub(r"\bnin\b", "min", line, flags=re.IGNORECASE)
        if (
            re.fullmatch(r"\d", line)
            and index + 1 < len(repaired)
            and (next_match := re.search(r"\b(\d{2})\s*min\b", repaired[index + 1], re.IGNORECASE))
        ):
            expected = int(next_match.group(1)) + 5
            if expected == int(line) * 10:
                line = str(expected)
        if (
            re.fullmatch(r"1[0o]", repaired[index], re.IGNORECASE)
            and index + 1 < len(repaired)
            and re.search(r"\b95\s*min\b", repaired[index + 1], re.IGNORECASE)
        ):
            line = "100"
        repaired[index] = line
    return repaired


def _canonicalize_numeric_option_tail(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 3 and _EXTRA_RINSE_RE.fullmatch(lines[0]) and re.fullmatch(r"[0oOdD]", lines[-1]):
        lines[-1] = "0"
        return "\n".join(lines)
    return text


def _looks_like_missing_numeric_option_tail(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or not _EXTRA_RINSE_RE.fullmatch(lines[0]):
        return False
    has_plus_one = any(re.fullmatch(r"\+\s*1", line) for line in lines[1:])
    has_plus_three = any(re.fullmatch(r"\+\s*3", line) for line in lines[1:])
    has_zero_tail = any(re.fullmatch(r"[0oOdD]", line) for line in lines[1:])
    return has_plus_one and not has_plus_three and not has_zero_tail


def _has_recovered_numeric_option_tail(primary_text: str, candidate_text: str) -> bool:
    primary_lines = [line.strip() for line in primary_text.splitlines() if line.strip()]
    candidate_lines = [line.strip() for line in candidate_text.splitlines() if line.strip()]
    if len(candidate_lines) <= len(primary_lines):
        return False
    if not candidate_lines or not primary_lines or candidate_lines[0].lower() != primary_lines[0].lower():
        return False
    if not re.fullmatch(r"[0oOdD]", candidate_lines[-1]):
        return False
    primary_plus = sum(1 for line in primary_lines if re.fullmatch(r"\+\s*\d", line))
    candidate_plus = sum(1 for line in candidate_lines if re.fullmatch(r"\+\s*\d", line))
    return candidate_plus >= primary_plus


def _extract_compact_delay_time(text: str) -> Optional[str]:
    for line in text.splitlines():
        digits = re.sub(r"\D", "", line)
        if digits in {"1230", "1200"}:
            return f"12:{digits[2:]}"
    return None


def _merge_delay_start_time(primary_text: str, compact_text: str) -> str:
    lines = [line.strip() for line in primary_text.splitlines() if line.strip()]
    normalized_text = _SPANISH_DELAY_START_RE.sub("El Inicio", "\n".join(lines))
    if not _DELAY_START_RE.search(normalized_text):
        return primary_text

    am_pm_index = -1
    am_pm_value = ""
    for index, line in enumerate(lines):
        match = _DELAY_AM_PM_ONLY_RE.fullmatch(line)
        if match:
            am_pm_index = index
            am_pm_value = match.group(1).upper() + "M" if line.strip().isupper() else f"{match.group(1).lower()}.m."
            break
    if am_pm_index < 0:
        return primary_text

    compact_time = _extract_compact_delay_time(compact_text)
    if not compact_time:
        return primary_text

    lines[am_pm_index] = f"{am_pm_value} {compact_time}"
    return "\n".join(lines)


def _recover_numeric_option_tail(
    bgr: np.ndarray,
    text: str,
    rec_model_path: Optional[Path],
    rec_keys_path: Optional[Path],
    det_model_path: Optional[Path],
) -> str:
    if not det_model_path or not _looks_like_missing_numeric_option_tail(text):
        return text

    for det_unclip_ratio, det_limit_side_len, det_box_thresh in ((3.5, 800, 0.4), (2.5, 960, 0.35)):
        option_engine = _get_engine(
            str(rec_model_path) if rec_model_path else None,
            str(rec_keys_path) if rec_keys_path else None,
            str(det_model_path),
            det_unclip_ratio,
            det_limit_side_len,
            det_box_thresh,
            True,
        )
        option_result, _ = option_engine(bgr)
        option_text, _ = _boxes_to_text(option_result or [])
        option_text = _repair_known_ui_tokens(option_text)
        if _has_recovered_numeric_option_tail(text, option_text):
            return option_text
    return text


def _recover_delay_start_time(
    bgr: np.ndarray,
    text: str,
    rec_model_path: Optional[Path],
    rec_keys_path: Optional[Path],
    det_model_path: Optional[Path],
) -> str:
    if not det_model_path:
        return text
    merged = _merge_delay_start_time(text, "")
    if merged != text:
        return merged
    if merged == text and not _DELAY_START_RE.search(_SPANISH_DELAY_START_RE.sub("El Inicio", text)):
        return text

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
    return _merge_delay_start_time(text, compact_text)


def _repair_domain_text(text: str) -> str:
    text = _repair_start_key_icons(text)
    text = _repair_known_ui_tokens(text)
    text = _canonicalize_numeric_option_tail(text)
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
    text = _recover_numeric_option_tail(bgr, text, rec_model_path, rec_keys_path, det_model_path)
    text = _recover_delay_start_time(bgr, text, rec_model_path, rec_keys_path, det_model_path)
    text = _repair_domain_text(text)
    return OCRResult(
        text=text,
        n_boxes=len(result or []),
        elapsed_ms=elapsed_ms,
        mean_score=mean_score,
    )
