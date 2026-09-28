"""Infer separate superscript regions from geometry, preserving recognized text."""
from collections import Counter
from dataclasses import replace
import unicodedata
import math
import numpy as np
from .ocr_engine import OCRBox, OCREngine


def _super_characters():
    mapping = {}
    for start, end in ((0x00B2, 0x00BB), (0x1D2C, 0x1DC0), (0x2070, 0x2080), (0xA7F2, 0xA7F5)):
        for code in range(start, end):
            char = chr(code)
            decomposition = unicodedata.decomposition(char).split()
            if len(decomposition) == 2 and decomposition[0] == '<super>':
                mapping.setdefault(chr(int(decomposition[1], 16)), char)
    return mapping


_SUPER_CHARACTERS = _super_characters()


def superscript_text(text):
    """Render confirmed geometry, never infer it from unconfirmed OCR text."""
    if text == 'TM':
        return '\u2122'
    if text == 'MC':
        return '\U0001f16a'
    if text in ('\u2122', '\u00ae', '\U0001f16a'):
        return text
    converted = []
    for char in text:
        if char in _SUPER_CHARACTERS:
            converted.append(_SUPER_CHARACTERS[char])
        elif unicodedata.decomposition(char).startswith('<super>'):
            converted.append(char)
        else:
            # Plain-text UI fallback; Excel uses real superscript formatting.
            return '^{'+text+'}'
    return ''.join(converted)


def _pixel_separation(image, body, mark):
    """Resolve detector padding only when two thresholds show a real ink gap."""
    from .image_spacing import _otsu, _runs
    width = mark.max_x-mark.min_x
    overlap = body.max_x-mark.min_x
    if image is None or width <= 0 or not 0 < overlap <= .5*width or mark.max_x <= body.max_x:
        return None
    rect = (max(0, math.floor(body.min_x)), max(0, math.floor(min(body.min_y, mark.min_y))),
            min(image.width, math.ceil(mark.max_x)), min(image.height, math.ceil(max(body.max_y, mark.max_y))))
    if rect[2] <= rect[0] or rect[3] <= rect[1] or (rect[2]-rect[0])*(rect[3]-rect[1]) > 1_000_000:
        return None
    gray = np.asarray(image.crop(rect).convert('L'))
    if int(gray.max())-int(gray.min()) < 50:
        return None
    threshold = _otsu(gray)
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    background = float(np.median(border))
    foreground_pixels = gray[gray <= threshold] if background > threshold else gray[gray > threshold]
    if not foreground_pixels.size:
        return None
    foreground = float(np.median(foreground_pixels))
    confirmations = []
    # Use foreground cores; low-contrast resampling halos are not glyph edges.
    for cutoff in (threshold, (threshold+foreground)/2):
        ink = gray <= cutoff if background > threshold else gray > cutoff
        if not .02 <= ink.mean() <= .55:
            return None
        candidates = []
        for start, end in _runs(~ink.any(axis=0)):
            low, high = max(start, math.ceil(mark.min_x)-rect[0]), min(end, math.ceil(body.max_x)-rect[0])
            if low >= high:
                continue
            split = int((low+high)//2)
            bounds = []
            for offset, mask in ((0, ink[:, :split]), (split, ink[:, split:])):
                ys, xs = np.nonzero(mask)
                if not len(xs):
                    break
                bounds.append((int(xs.min())+offset, int(ys.min()), int(xs.max())+offset+1, int(ys.max())+1))
            if len(bounds) != 2:
                continue
            b, m = bounds
            height = b[3]-b[1]
            if (height > 0 and 0 < (m[3]-m[1])/height <= .7
                    and .65*width <= m[2]-m[0] <= height
                    and 0 < (m[0]-b[2])/height <= .35
                    and -.5 <= (m[1]-b[1])/height <= .25
                    and (b[3]-m[3])/height >= .2):
                candidates.append({'body_ink_rect': [b[0]+rect[0], b[1]+rect[1], b[2]+rect[0], b[3]+rect[1]],
                                   'script_ink_rect': [m[0]+rect[0], m[1]+rect[1], m[2]+rect[0], m[3]+rect[1]],
                                   'gap_pixels': m[0]-b[2]})
        if len(candidates) != 1:
            return None
        confirmations.append(candidates[0])
    return {'method': 'stable_ink_separation', 'thresholds': confirmations}


def apply_superscripts(result, image=None):
    boxes = result.boxes
    evidence = {"status": "NO_SEPARATE_REGION", "coordinate_space": "ocr_input_pixels",
                "boxes": [], "relations": [], "attachments": []}
    result.superscript_evidence = evidence
    if not isinstance(boxes, list) or not all(isinstance(b, OCRBox) for b in boxes):
        return
    evidence["boxes"] = [{"text": b.text, "rect": [b.min_x, b.min_y, b.max_x, b.max_y],
                          "score": b.score} for b in boxes]
    if len(boxes) < 2:
        return
    evidence["status"] = "GEOMETRY_NOT_CONFIRMED"
    proposals = []
    for mark_index, mark in enumerate(boxes):
        text = mark.text.strip()
        # Bound the annotation length, not a list of product-specific strings.
        if not 1 <= len(text) <= 4 or any(c.isspace() for c in text):
            continue
        if not any(c.isalnum() for c in text) and text not in ('™', '®', '\U0001f16a', '*', '+', '−', '-'):
            continue
        candidates = []
        for body_index, body in enumerate(boxes):
            if body_index == mark_index or not any(c.isalnum() for c in body.text):
                continue
            height = body.max_y-body.min_y
            if height <= 0:
                continue
            metrics = dict(height_ratio=(mark.max_y-mark.min_y)/height,
                           width_ratio=(mark.max_x-mark.min_x)/height,
                           gap_ratio=(mark.min_x-body.max_x)/height,
                           top_offset_ratio=(mark.min_y-body.min_y)/height,
                           bottom_raise_ratio=(body.max_y-mark.max_y)/height)
            overlap = max(0, body.max_x-mark.min_x)
            # Detection boxes can include padding around actual glyphs.
            # Permit only a small overlap, bounded by both body and mark size.
            metrics['overlap_pixels'] = overlap
            metrics['overlap_limit_pixels'] = min(.15*height, .25*max(0, mark.max_x-mark.min_x))
            # A lone letter near the crop edge may be the visible part of a
            # word/annotation (O from OK, M from MC). Do not promote it without
            # enough right-hand context. Multi-letter TM/MC keep their policy.
            single_letter = len(text) == 1 and text.isalpha()
            right_context = image.width-mark.max_x if image is not None else None
            if single_letter:
                metrics['right_context_pixels'] = right_context
            checks = dict(confidence=body.score >= .5 and mark.score >= .5,
                          single_letter_confidence=not single_letter or mark.score >= .9,
                          single_letter_context=(not single_letter or right_context is None or right_context >= height),
                          smaller=0 < metrics['height_ratio'] <= .7,
                          compact=0 < metrics['width_ratio'] <= 1,
                          adjacent=(metrics['gap_ratio'] <= .35
                                    and overlap <= metrics['overlap_limit_pixels']
                                    and mark.max_x > body.max_x),
                          upper_position=-.5 <= metrics['top_offset_ratio'] <= .25,
                          raised_baseline=metrics['bottom_raise_ratio'] >= .2)
            if not checks['adjacent'] and all(value for key, value in checks.items() if key != 'adjacent'):
                pixels = _pixel_separation(image, body, mark)
                if pixels is not None:
                    checks['adjacent'] = True
                    metrics['pixel_separation'] = pixels
            accepted = all(checks.values())
            evidence['relations'].append(dict(body_box=body_index, script_box=mark_index,
                text=text, relation='superscript', geometry_confirmed=accepted,
                rejected_checks=[key for key, passed in checks.items() if not passed], **metrics))
            if accepted:
                candidates.append(body_index)
        if len(candidates) == 1:
            proposals.append((candidates[0], mark_index))
        elif len(candidates) > 1:
            evidence['status'] = 'AMBIGUOUS_RELATION'
    counts = Counter(body for body, mark in proposals)
    parents = {body for body, mark in proposals}
    children = {mark for body, mark in proposals}
    # Abstain on competing annotations or nested chains instead of flattening them.
    attachments = {body: mark for body, mark in proposals
                   if counts[body] == 1 and body not in children and mark not in parents}
    if proposals and len(attachments) != len(proposals):
        evidence['status'] = 'AMBIGUOUS_RELATION'
    if not attachments:
        return
    evidence['attachments'] = [dict(body_box=body, script_box=mark,
                                    text=boxes[mark].text.strip(), rendered_text=superscript_text(boxes[mark].text.strip()),
                                    relation='superscript')
                               for body, mark in attachments.items()]
    original, _ = OCREngine.text_from_boxes(boxes)
    if result.text != original:
        evidence['status'] = 'TEXT_ALREADY_ADJUSTED'
        return
    removed = set(attachments.values())
    adjusted = [replace(b, text=b.text.rstrip()+superscript_text(boxes[attachments[i]].text.strip()))
                if i in attachments else b for i, b in enumerate(boxes) if i not in removed]
    if not result.raw_text:
        result.raw_text = result.text
    result.text, _ = OCREngine.text_from_boxes(adjusted)
    evidence.update(status='SUPERSCRIPT_ATTACHED', text=result.text)
