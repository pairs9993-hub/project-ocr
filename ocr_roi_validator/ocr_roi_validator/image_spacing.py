"""Conservative, image-only spacing evidence for an isolated Latin text line."""
import re
import unicodedata
import numpy as np
from PIL import Image, ImageOps


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
    if '\n' in raw_text or len(compact) < 4 or not all((c.isalpha() and 'LATIN' in unicodedata.name(c, '')) or c in ".,:;!?'-" for c in compact):
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


def word_regions(image):
    """Find stable large blank gaps without requiring one component per letter."""
    gray = np.asarray(image.convert('L'))
    if min(gray.shape) < 6 or int(gray.max())-int(gray.min()) < 50:
        return []
    threshold = _otsu(gray)
    background = float(np.median(np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))))
    dark = background > threshold
    candidates = []
    for cutoff in (threshold, (threshold+background)/2):
        ink = gray <= cutoff if dark else gray > cutoff
        if not .02 <= ink.mean() <= .55:
            return []
        rows = _runs(ink.any(axis=1))
        if not rows:
            return []
        height = rows[-1][1]-rows[0][0]
        if any(b[0]-a[1] > height*.2 for a, b in zip(rows, rows[1:])):
            return []
        runs = _runs(ink.any(axis=0))
        if len(runs) < 5:
            return []
        gaps = [b[0]-a[1] for a, b in zip(runs, runs[1:])]
        baseline = float(np.median(gaps))
        minimum = max(baseline*2.5, baseline+max(2, height*.1), height*.15)
        wide = [(int(a[1]), int(b[0])) for a, b in zip(runs, runs[1:]) if b[0]-a[1] >= minimum]
        if not 1 <= len(wide) <= 3:
            return []
        candidates.append(wide)
    if len(candidates[0]) != len(candidates[1]):
        return []
    cuts = []
    for first, second in zip(*candidates):
        left, right = max(first[0], second[0]), min(first[1], second[1])
        if right <= left:
            return []
        cuts.append((left+right)//2)
    edges = [0]+cuts+[image.width]
    return [(a, 0, b, image.height) for a, b in zip(edges, edges[1:])]


def recognize_separated_words(image, raw_text, recognize):
    regions = word_regions(image)
    if not regions:
        return raw_text, None
    words = []
    for rect in regions:
        piece = image.crop(rect)
        background = tuple(int(v) for v in np.median(np.asarray(piece.convert('RGB'))[0], axis=0))
        piece = ImageOps.expand(piece, border=max(4, piece.height//4), fill=background)
        result = recognize(piece)
        text = unicodedata.normalize('NFC', result.text).strip()
        if result.mean_score < .5 or len(text) < 2 or any(c.isspace() for c in text):
            return raw_text, None
        words.append(text)
    if ''.join(words) != re.sub(r'\s+', '', unicodedata.normalize('NFC', raw_text)):
        return raw_text, None
    # Existing spaces must also be supported by the independent word OCR.
    candidate = ' '.join(words)
    offset, original_spaces = 0, set()
    for char in raw_text.strip():
        if char.isspace():
            original_spaces.add(offset)
        else:
            offset += 1
    boundaries = set(np.cumsum([len(word) for word in words[:-1]]).tolist())
    if original_spaces - boundaries:
        return raw_text, None
    return candidate, {"status": "IMAGE_GAP_CONFIRMED", "raw_text": raw_text,
                       "text": candidate, "method": "stable_gaps_independent_word_ocr",
                       "word_regions": regions, "word_ocr": words}


def apply_image_spacing(image, result, expected, recognize=None):
    """Gate to complete single-box text with matching letters, then inspect pixels."""
    raw = result.text
    result.raw_text = raw
    result.spacing_evidence = None
    compact = lambda value: re.sub(r'\s+', '', unicodedata.normalize('NFC', value))
    if not expected or raw == expected:
        return
    if compact(raw) != compact(expected):
        base = lambda value: ''.join(c for c in unicodedata.normalize('NFD', compact(value))
                                     if not unicodedata.combining(c))
        if base(raw) == base(expected):
            result.spacing_evidence = {"status": "UNCERTAIN", "reason": "DIACRITIC_MISMATCH", "raw_text": raw}
            if recognize is not None and raw.strip() and len(result.boxes) == 1:
                retry_diacritics(image, result, recognize)
                if result.text == raw:
                    retry_words(image, result, expected, recognize)
        elif recognize is not None and len(result.boxes) == 1:
            retry_words(image, result, expected, recognize)
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
    line = image.crop(rect)
    text, evidence = infer_spacing(line, raw)
    if recognize is not None and evidence.get("reason") == "GLYPH_COUNT_MISMATCH":
        candidate, retry_evidence = recognize_separated_words(line, raw, recognize)
        if retry_evidence is not None:
            text, evidence = candidate, retry_evidence
    evidence["source_line_rect"] = rect
    result.spacing_evidence = evidence
    result.text = text


def retry_diacritics(image, result, recognize):
    """Accept only agreeing image OCR retries; expected text is never a candidate."""
    from PIL import Image
    raw = result.text
    base = lambda value: ''.join(c for c in unicodedata.normalize('NFD', value)
                                 if not unicodedata.combining(c))
    evidence = result.spacing_evidence
    evidence.update(method="diacritic_scaled_retry", retry_outputs=[])
    # Keep the complete input so accents above detection boxes are not cropped off.
    if image.width * image.height > 1_000_000:
        evidence['retry_skipped'] = 'INPUT_TOO_LARGE'
        return
    candidates = []
    for scale in (2, 3):
        enlarged = image.resize((image.width*scale, image.height*scale), Image.Resampling.LANCZOS)
        background = tuple(int(v) for v in np.median(np.asarray(enlarged.convert('RGB'))[0], axis=0))
        enlarged = ImageOps.expand(enlarged.convert('RGB'), border=12, fill=background)
        retry = recognize(enlarged)
        text = unicodedata.normalize('NFC', retry.text).strip()
        evidence['retry_outputs'].append({'scale': scale, 'text': text, 'score': retry.mean_score})
        if retry.mean_score < .5 or base(text) != base(unicodedata.normalize('NFC', raw).strip()):
            return
        candidates.append(text)
    if candidates[0] == candidates[1] and candidates[0] != unicodedata.normalize('NFC', raw).strip():
        result.text = candidates[0]
        result.mean_score = min(item['score'] for item in evidence['retry_outputs'])
        evidence.update(status="DIACRITIC_RETRY_CONFIRMED", text=result.text)
        evidence.pop('reason', None)


def retry_words(image, result, expected, recognize):
    """Read independently segmented words twice; never substitute expected text."""
    from difflib import SequenceMatcher
    from PIL import Image
    raw = result.text
    if image.width*image.height > 1_000_000 or '\n' in raw:
        return
    # Only near-complete observations justify the additional OCR cost.
    if SequenceMatcher(None, raw, expected).ratio() < .85:
        return
    regions = word_regions(image)
    if not regions:
        return
    outputs, words, scores = [], [], []
    evidence = {"status": "UNCERTAIN", "reason": "WORD_RETRY_DISAGREEMENT",
                "method": "independent_word_retry", "raw_text": raw,
                "word_regions": regions, "retry_outputs": outputs}
    result.spacing_evidence = evidence
    for rect in regions:
        piece = image.crop(rect).convert('RGB')
        background = tuple(int(v) for v in np.median(np.asarray(piece)[0], axis=0))
        alternatives = []
        for scale in (1, 2):
            variant = piece.resize((piece.width*scale, piece.height*scale), Image.Resampling.LANCZOS)
            variant = ImageOps.expand(variant, border=max(8, variant.height//4), fill=background)
            retry = recognize(variant)
            text = unicodedata.normalize('NFC', retry.text).strip()
            outputs.append({'rect': rect, 'scale': scale, 'text': text, 'score': retry.mean_score})
            if retry.mean_score < .5 or not text or any(c.isspace() for c in text):
                return
            alternatives.append(text)
            scores.append(retry.mean_score)
        if alternatives[0] != alternatives[1]:
            return
        words.append(alternatives[0])
    candidate = ' '.join(words)
    if SequenceMatcher(None, raw, candidate).ratio() < .75:
        return
    result.text = candidate
    result.mean_score = min(scores)
    evidence.update(status="WORD_RETRY_CONFIRMED", text=candidate)
    evidence.pop('reason', None)
