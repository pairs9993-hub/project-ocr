"""Conservative, image-only spacing evidence for an isolated Latin text line."""
import re
import unicodedata
import numpy as np
from PIL import Image


def _runs(values):
    changes = np.diff(np.r_[False, values, False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def _otsu(gray):
    histogram = np.bincount(gray.ravel(), minlength=256).astype(float)
    weights = np.cumsum(histogram)
    sums = np.cumsum(histogram * np.arange(256))
    denominator = weights * (weights[-1] - weights)
    variance = np.zeros(256)
    valid = denominator > 0
    variance[valid] = (sums[-1]*weights[valid] - sums[valid]*weights[-1])**2 / denominator[valid]
    return int(np.argmax(variance))


def infer_spacing(image: Image.Image, raw_text: str):
    """Never take expected text as input; abstain when glyph mapping is unclear."""
    evidence = {"status": "UNCERTAIN", "raw_text": raw_text, "method": "stable_column_gaps_v1"}
    compact = re.sub(r"\s+", "", unicodedata.normalize('NFC', raw_text))
    if '\n' in raw_text or len(compact) < 4 or not all(c.isalpha() and 'LATIN' in unicodedata.name(c, '') for c in compact):
        return raw_text, {**evidence, "reason": "UNSUPPORTED_TEXT"}
    gray = np.asarray(image.convert('L'))
    if min(gray.shape) < 6 or int(gray.max())-int(gray.min()) < 50:
        return raw_text, {**evidence, "reason": "LOW_CONTRAST_OR_SMALL"}
    threshold = _otsu(gray)
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    background = float(np.median(border))
    dark = background > threshold
    candidates = []
    for cutoff in (threshold, (threshold+background)/2):
        ink = gray <= cutoff if dark else gray > cutoff
        if not .02 <= ink.mean() <= .55:
            return raw_text, {**evidence, "reason": "COMPLEX_BACKGROUND"}
        rows = _runs(ink.any(axis=1))
        if not rows:
            return raw_text, {**evidence, "reason": "NO_INK"}
        height = rows[-1][1]-rows[0][0]
        # A large horizontal blank band suggests two lines rather than accents.
        if any(b[0]-a[1] > height*.2 for a, b in zip(rows, rows[1:])):
            return raw_text, {**evidence, "reason": "MULTILINE_OR_FRAGMENTED"}
        glyphs = _runs(ink.any(axis=0))
        if len(glyphs) != len(compact):
            return raw_text, {**evidence, "reason": "GLYPH_COUNT_MISMATCH", "glyph_count": len(glyphs)}
        widths = [b-a for a, b in glyphs]
        if max(widths) > height*1.5:
            return raw_text, {**evidence, "reason": "AMBIGUOUS_GLYPHS"}
        gaps = [int(b[0]-a[1]) for a, b in zip(glyphs, glyphs[1:])]
        baseline = float(np.median(gaps))
        large = max(baseline*2.2, baseline+max(2, height*.08), height*.12)
        boundaries = [i+1 for i, gap in enumerate(gaps) if gap >= large]
        # Marginal gaps are not sufficient evidence for inserting a space.
        if any(large*.8 <= gap < large for gap in gaps):
            return raw_text, {**evidence, "reason": "AMBIGUOUS_GAP"}
        candidates.append((boundaries, gaps, baseline))
    if candidates[0][0] != candidates[1][0]:
        return raw_text, {**evidence, "reason": "UNSTABLE_THRESHOLD"}
    boundaries, gaps, baseline = candidates[0]
    if not boundaries:
        return raw_text, {**evidence, "status": "NO_WORD_GAP", "reason": "NO_CLEAR_WORD_GAP", "gaps_px": gaps}
    # Insert only; never erase an OCR-reported space using this heuristic.
    raw_boundaries, index = set(), 0
    for char in raw_text.strip():
        if char.isspace():
            raw_boundaries.add(index)
        else:
            index += 1
    if raw_boundaries - set(boundaries):
        return raw_text, {**evidence, "reason": "OCR_IMAGE_DISAGREEMENT"}
    candidate = ''.join((' ' if i in boundaries else '')+c for i, c in enumerate(compact))
    return candidate, {**evidence, "status": "IMAGE_GAP_CONFIRMED", "text": candidate,
                       "boundaries": boundaries, "gaps_px": gaps, "normal_gap_px": baseline}


def apply_image_spacing(image, result, expected):
    """Gate to complete single-box text with matching letters, then inspect pixels."""
    raw = result.text
    result.raw_text = raw
    result.spacing_evidence = None
    compact = lambda value: re.sub(r'\s+', '', unicodedata.normalize('NFC', value))
    if not expected or raw == expected or compact(raw) != compact(expected):
        return
    if len(result.boxes) != 1:
        result.spacing_evidence = {"status": "UNCERTAIN", "reason": "REQUIRES_SINGLE_TEXT_BOX", "raw_text": raw}
        return
    box = result.boxes[0]
    if not all(isinstance(getattr(box, key, None), (int, float)) for key in ('min_x', 'min_y', 'max_x', 'max_y')):
        return
    import math
    rect = (max(0, math.floor(box.min_x)-2), max(0, math.floor(box.min_y)-2),
            min(image.width, math.ceil(box.max_x)+2), min(image.height, math.ceil(box.max_y)+2))
    if rect[2] <= rect[0] or rect[3] <= rect[1]:
        return
    text, evidence = infer_spacing(image.crop(rect), raw)
    result.spacing_evidence = evidence
    result.text = text
