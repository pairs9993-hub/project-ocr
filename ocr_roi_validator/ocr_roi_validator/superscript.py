"""Infer separate superscript regions from geometry, preserving recognized text."""
from collections import Counter
from dataclasses import replace
from .ocr_engine import OCRBox, OCREngine


def apply_superscripts(result):
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
        if not any(c.isalnum() for c in text) and text not in ('™', '®', '*', '+', '−', '-'):
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
            checks = dict(confidence=body.score >= .5 and mark.score >= .5,
                          smaller=0 < metrics['height_ratio'] <= .7,
                          compact=0 < metrics['width_ratio'] <= 1,
                          adjacent=(metrics['gap_ratio'] <= .35
                                    and overlap <= metrics['overlap_limit_pixels']
                                    and mark.max_x > body.max_x),
                          upper_position=-.5 <= metrics['top_offset_ratio'] <= .25,
                          raised_baseline=metrics['bottom_raise_ratio'] >= .2)
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
                                    text=boxes[mark].text.strip(), relation='superscript')
                               for body, mark in attachments.items()]
    original, _ = OCREngine.text_from_boxes(boxes)
    if result.text != original:
        evidence['status'] = 'TEXT_ALREADY_ADJUSTED'
        return
    removed = set(attachments.values())
    adjusted = [replace(b, text=b.text.rstrip()+boxes[attachments[i]].text.strip())
                if i in attachments else b for i, b in enumerate(boxes) if i not in removed]
    if not result.raw_text:
        result.raw_text = result.text
    result.text, _ = OCREngine.text_from_boxes(adjusted)
    evidence.update(status='SUPERSCRIPT_ATTACHED', text=result.text)
