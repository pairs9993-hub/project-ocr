"""Confirm a duplicated box boundary using independent line OCR, never truth."""
import math
import re
import time

from .ocr_engine import OCRBox, OCRRunResult


def apply_overlap_retry(image, result, recognize):
    result.overlap_evidence = None
    boxes = result.boxes
    if not isinstance(boxes, list) or len(boxes) != 2 or not all(isinstance(b, OCRBox) for b in boxes):
        return
    left, right = sorted(boxes, key=lambda b: b.min_x)
    if not all(math.isfinite(v) for box in (left, right)
               for v in (box.min_x, box.min_y, box.max_x, box.max_y, box.score)):
        return
    lh, rh = left.max_y-left.min_y, right.max_y-right.min_y
    overlap = left.max_x-right.min_x
    vertical = min(left.max_y, right.max_y)-max(left.min_y, right.min_y)
    if (min(lh, rh) <= 0 or not .65 <= rh/lh <= 1.5
            or vertical < .65*max(lh, rh) or min(left.score, right.score) < .9
            or not 0 < overlap <= .3*min(left.max_x-left.min_x, right.max_x-right.min_x)
            or not left.min_x < right.min_x < left.max_x < right.max_x):
        return
    a, b = left.text.strip(), right.text.strip()
    normalize = lambda text: re.sub(r'\s+', ' ', text).strip()
    candidates = {normalize(a+' '+b[n:].lstrip())
                  for n in range(1, min(8, len(a), len(b))+1)
                  if a[-n:] == b[:n] and b[:n].isalnum()
                  and len(b) > n and b[n].isspace()}
    if not candidates or normalize(result.text) != normalize(a+' '+b):
        return
    rect = (max(0, math.floor(left.min_x)-2),
            max(0, math.floor(min(left.min_y, right.min_y))-2),
            min(image.width, math.ceil(right.max_x)+2),
            min(image.height, math.ceil(max(left.max_y, right.max_y))+2))
    if rect[2] <= rect[0] or rect[3] <= rect[1]:
        return
    started = time.monotonic()
    retry = recognize(image.crop(rect))
    evidence = dict(status='UNCERTAIN', method='overlapping_boxes_line_retry',
                    raw_text=result.text, source_line_rect=rect, overlap_pixels=overlap,
                    retry_calls=1, retry_seconds=round(time.monotonic()-started, 4))
    result.overlap_evidence = evidence
    if not isinstance(retry, OCRRunResult):
        evidence['reason'] = 'LINE_RECOGNITION_UNAVAILABLE'
        return
    evidence.update(retry_text=retry.text, retry_score=retry.mean_score)
    if not math.isfinite(retry.mean_score) or retry.mean_score < .9 or normalize(retry.text) not in candidates:
        evidence['reason'] = 'LINE_RECOGNITION_NOT_CONFIRMED'
        return
    if not result.raw_text:
        result.raw_text = result.text
    result.text = normalize(retry.text)
    result.mean_score = min(result.mean_score, retry.mean_score)
    evidence.update(status='OVERLAP_DUPLICATION_CONFIRMED', text=result.text)
