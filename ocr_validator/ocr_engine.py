"""Module 2: OCR engine wrapper.

Thin wrapper around RapidOCR (ONNX runtime) so the rest of the pipeline
doesn't depend on engine-specific call signatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from rapidocr_onnxruntime import RapidOCR


@dataclass
class OCRBox:
    bbox: List[List[float]]   # 4 corners (TL, TR, BR, BL)
    text: str
    score: float


class OCREngine:
    def __init__(self, rec_model_path: str | Path | None = None,
                 rec_keys_path: str | Path | None = None):
        kwargs = {}
        if rec_model_path is not None:
            kwargs["rec_model_path"] = str(rec_model_path)
        if rec_keys_path is not None:
            kwargs["rec_keys_path"] = str(rec_keys_path)
        self._engine = RapidOCR(**kwargs)

    def recognize(self, image: str | np.ndarray) -> List[OCRBox]:
        result, _ = self._engine(image)
        if not result:
            return []
        return [OCRBox(bbox=r[0], text=r[1], score=r[2]) for r in result]

    @staticmethod
    def boxes_to_text(boxes: List[OCRBox]) -> str:
        """Order boxes top-to-bottom, left-to-right; join with newlines."""
        if not boxes:
            return ""
        # group by line: cluster by y-center within a tolerance
        items = []
        for b in boxes:
            ys = [p[1] for p in b.bbox]
            xs = [p[0] for p in b.bbox]
            items.append((sum(ys) / 4.0, sum(xs) / 4.0, b.text))
        items.sort()  # by y first
        lines: List[List[tuple]] = []
        cur_line: List[tuple] = []
        cur_y = None
        tol = 12  # px
        for y, x, t in items:
            if cur_y is None or abs(y - cur_y) <= tol:
                cur_line.append((x, t))
                cur_y = y if cur_y is None else (cur_y + y) / 2
            else:
                lines.append(cur_line)
                cur_line = [(x, t)]
                cur_y = y
        if cur_line:
            lines.append(cur_line)
        # within each line, sort by x
        out_lines = []
        for line in lines:
            line.sort()
            out_lines.append(" ".join(t for _, t in line))
        return "\n".join(out_lines)
