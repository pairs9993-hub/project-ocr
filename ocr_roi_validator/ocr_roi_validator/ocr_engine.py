from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import List, Tuple

import numpy as np
from PIL import Image

from .model_package import OCRModelPackage
from .paddle_package import PaddleModelPackage


@dataclass
class OCRRunResult:
    text: str
    mean_score: float
    n_boxes: int
    boxes: List["OCRBox"]


@dataclass
class OCRBox:
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    center_x: float
    center_y: float
    text: str
    score: float


def _box_center(box: List[List[float]]) -> Tuple[float, float]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return sum(xs) / len(xs), sum(ys) / len(ys)


_ICON_TEXT_RE = re.compile(r"^[\-\sbcilmopu0-9]+$", re.IGNORECASE)


def _box_bounds(box: list[list[float]]) -> Tuple[float, float, float, float]:
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    return min(xs), min(ys), max(xs), max(ys)


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


def _has_word_text(text: str) -> bool:
    return sum(1 for char in text if char.isalpha()) >= 3


def _drop_left_gutter_icons(items, preserve_small_left_noise: bool = False):
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
        if (is_small_left_noise and has_text_neighbor and not preserve_small_left_noise) or is_top_status_icon:
            continue
        filtered.append(item)
    return filtered


def _merge_items_to_text(items: List[Tuple[float, float, float]]) -> Tuple[str, float]:
    if not items:
        return "", 0.0

    sorted_items = sorted(items, key=lambda t: (t[0], t[1]))
    lines: List[List[Tuple[float, str, float]]] = []
    current: List[Tuple[float, str, float]] = []
    current_y = None
    y_tol = 12.0

    for y, x, text, score in sorted_items:
        if current_y is None or abs(y - current_y) <= y_tol:
            current.append((x, text, score))
            current_y = y if current_y is None else (current_y + y) / 2
        else:
            lines.append(current)
            current = [(x, text, score)]
            current_y = y

    if current:
        lines.append(current)

    out_lines: List[str] = []
    scores: List[float] = []
    for line in lines:
        line.sort(key=lambda item: item[0])
        out_lines.append(" ".join(seg for _, seg, _ in line))
        for _, _, s in line:
            scores.append(s)

    mean_score = sum(scores) / len(scores) if scores else 0.0
    return "\n".join(out_lines), mean_score


@lru_cache(maxsize=8)
def _build_engine(
    det_model_path: str,
    rec_model_path: str,
    rec_keys_path: str,
    det_limit_type: str,
    det_limit_side_len: int,
    det_box_thresh: float,
    det_unclip_ratio: float,
    det_donot_use_dilation: bool,
    use_cls: bool,
    det_mean: tuple,
    det_std: tuple,
):
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR(
        det_model_path=det_model_path,
        rec_model_path=rec_model_path,
        rec_keys_path=rec_keys_path,
        det_limit_type=det_limit_type,
        det_limit_side_len=det_limit_side_len,
        det_box_thresh=det_box_thresh,
        det_unclip_ratio=det_unclip_ratio,
        det_donot_use_dilation=det_donot_use_dilation,
        det_mean=list(det_mean),
        det_std=list(det_std),
        use_cls=use_cls,
    )


@lru_cache(maxsize=1)
def _build_default_engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


@lru_cache(maxsize=3)
def _build_paddle_default_engine(lang: str):
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR backend requested, but paddleocr is not installed. "
            "Install paddleocr and paddlepaddle in a supported Python environment (recommended Python 3.10)."
        ) from exc

    lang_map = {
        "en_es": "en",
        "fr": "fr",
        "zh": "ch",
    }
    paddle_lang = lang_map.get(lang, "en")

    return PaddleOCR(
        use_angle_cls=False,
        lang=paddle_lang,
        use_gpu=False,
        show_log=False,
        det_limit_side_len=640,
        det_limit_type="min",
    )


@lru_cache(maxsize=8)
def _build_paddle_custom_engine(
    det_model_dir: str,
    rec_model_dir: str,
    rec_char_dict_path: str,
    lang: str,
):
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR backend requested, but paddleocr is not installed. "
            "Install paddleocr and paddlepaddle in a supported Python environment (recommended Python 3.10)."
        ) from exc

    lang_map = {
        "en_es": "en",
        "fr": "fr",
        "zh": "ch",
    }
    paddle_lang = lang_map.get(lang, "en")

    return PaddleOCR(
        use_angle_cls=False,
        lang=paddle_lang,
        use_gpu=False,
        show_log=False,
        det_model_dir=det_model_dir,
        rec_model_dir=rec_model_dir,
        rec_char_dict_path=rec_char_dict_path,
        det_limit_side_len=640,
        det_limit_type="min",
    )


class OCREngine:
    def __init__(
        self,
        package: OCRModelPackage | None = None,
        paddle_package: PaddleModelPackage | None = None,
        use_rapid_default: bool = False,
        backend: str = "rapid",
        use_detection_fallback: bool = False,
    ):
        if backend not in {"rapid", "paddle"}:
            raise ValueError("backend must be 'rapid' or 'paddle'")
        if backend == "rapid" and package is None and not use_rapid_default:
            raise ValueError("Rapid backend requires package or use_rapid_default=True")
        self.package = package
        self.paddle_package = paddle_package
        self.use_rapid_default = use_rapid_default
        self.backend = backend
        self.use_detection_fallback = use_detection_fallback

    def _engine_for_language(self, lang: str):
        if self.backend == "paddle":
            if self.paddle_package is not None:
                rec_dir = self.paddle_package.recognizer_model_dirs.get(lang)
                if rec_dir is not None:
                    return _build_paddle_custom_engine(
                        str(self.paddle_package.detector_model_dir),
                        str(rec_dir),
                        str(self.paddle_package.dictionary),
                        lang,
                    )
            return _build_paddle_default_engine(lang)

        if self.use_rapid_default:
            return _build_default_engine()

        if self.package is None:
            raise ValueError("Model package is not configured")

        rec_path = self.package.recognizers[lang]
        p = self.package.preprocess
        return _build_engine(
            str(self.package.detector_model),
            str(rec_path),
            str(self.package.dictionary),
            str(p["det_limit_type"]),
            int(p["det_limit_side_len"]),
            float(p["det_box_thresh"]),
            float(p["det_unclip_ratio"]),
            bool(p["det_donot_use_dilation"]),
            bool(p["use_cls"]),
            tuple(p["det_mean"]),
            tuple(p["det_std"]),
        )

    def run(
        self,
        image: Image.Image,
        language: str,
        preserve_small_left_noise: bool = False,
    ) -> OCRRunResult:
        if self.backend == "rapid" and not self.use_rapid_default:
            if self.package is None:
                raise ValueError("Model package is not configured")
            if language not in self.package.recognizers:
                raise ValueError(f"Unsupported language: {language}")

        rgb = image.convert("RGB")
        arr = np.array(rgb)
        bgr = arr[:, :, ::-1].copy()
        engine = self._engine_for_language(language)

        if self.backend == "paddle":
            result = self._run_paddle(engine, bgr)
        else:
            result, _ = engine(bgr)
            should_retry_detection = self.use_detection_fallback or not result
            if should_retry_detection and self.package is not None and self.package.detector_model:
                p = self.package.preprocess
                fallback_engine = _build_engine(
                    str(self.package.detector_model),
                    str(self.package.recognizers[language]),
                    str(self.package.dictionary),
                    str(p["det_limit_type"]),
                    int(p["det_limit_side_len"]),
                    float(p["det_box_thresh"]),
                    3.0,
                    bool(p["det_donot_use_dilation"]),
                    bool(p["use_cls"]),
                    tuple(p["det_mean"]),
                    tuple(p["det_std"]),
                )
                fallback_result, _ = fallback_engine(bgr)
                result = _select_result(result or [], fallback_result or [])

        if not result:
            return OCRRunResult(text="", mean_score=0.0, n_boxes=0, boxes=[])

        items = []
        for box, text, score in result:
            min_x, min_y, max_x, max_y = _box_bounds(box)
            items.append(((min_y + max_y) / 2.0, (min_x + max_x) / 2.0, min_x, min_y, max_x, max_y, str(text), float(score)))

        items = _drop_left_gutter_icons(items, preserve_small_left_noise=preserve_small_left_noise)

        boxes: List[OCRBox] = []
        merge_items: List[Tuple[float, float, str, float]] = []
        for y, x, min_x, min_y, max_x, max_y, text, score in items:
            boxes.append(
                OCRBox(
                    min_x=min_x,
                    min_y=min_y,
                    max_x=max_x,
                    max_y=max_y,
                    center_x=x,
                    center_y=y,
                    text=text,
                    score=score,
                )
            )
            merge_items.append((y, x, text, score))

        text, mean_score = _merge_items_to_text(merge_items)
        return OCRRunResult(text=text, mean_score=mean_score, n_boxes=len(boxes), boxes=boxes)

    @staticmethod
    def text_from_boxes(boxes: List[OCRBox]) -> Tuple[str, float]:
        merge_items = [(box.center_y, box.center_x, box.text, box.score) for box in boxes]
        return _merge_items_to_text(merge_items)

    @staticmethod
    def _run_paddle(engine, arr: np.ndarray):
        ocr_raw = engine.ocr(arr, cls=False)
        if not ocr_raw:
            return []

        lines = ocr_raw[0] if isinstance(ocr_raw, list) else []
        parsed = []
        for row in lines:
            if not row or len(row) < 2:
                continue
            box = row[0]
            text_score = row[1]
            if not isinstance(text_score, (list, tuple)) or len(text_score) < 2:
                continue
            text = str(text_score[0])
            score = float(text_score[1])
            parsed.append((box, text, score))
        return parsed
