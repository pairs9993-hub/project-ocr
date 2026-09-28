"""Attach separately detected TM only when box geometry supports superscript."""
from dataclasses import replace
from .ocr_engine import OCRBox, OCREngine


def apply_superscript_tm(result):
    boxes = result.boxes
    result.superscript_evidence = {"status": "NO_SEPARATE_TM_BOX"}
    if not isinstance(boxes, list) or not all(isinstance(box, OCRBox) for box in boxes):
        return
    marks = [i for i, box in enumerate(boxes) if box.text.strip() == "TM"]
    if not marks:
        return
    evidence = {"status": "GEOMETRY_NOT_CONFIRMED", "attachments": [],
                "boxes": [{"text": b.text, "rect": [b.min_x, b.min_y, b.max_x, b.max_y],
                           "score": b.score} for b in boxes]}
    result.superscript_evidence = evidence
    original, _ = OCREngine.text_from_boxes(boxes)
    if result.text != original:
        evidence["status"] = "TEXT_ALREADY_ADJUSTED"
        return
    used = set()
    attachments = {}
    for mark_index in marks:
        mark = boxes[mark_index]
        candidates = []
        for index, body in enumerate(boxes):
            height = body.max_y-body.min_y
            mh = mark.max_y-mark.min_y
            gap = mark.min_x-body.max_x
            if index in marks or not body.text.strip() or not body.text.rstrip()[-1].isalnum():
                continue
            if (body.score >= .5 and mark.score >= .5 and height > 0
                    and 0 < mh <= .7*height
                    and 0 < mark.max_x-mark.min_x <= height
                    and 0 <= gap <= .35*height
                    and body.min_y-.5*height <= mark.min_y <= body.min_y+.25*height
                    and mark.max_y <= body.max_y-.2*height):
                candidates.append(index)
        if len(candidates) != 1 or candidates[0] in used:
            continue
        index = candidates[0]
        used.add(index)
        attachments[index] = mark_index
        evidence["attachments"].append({"body_box": index, "tm_box": mark_index})
    if not attachments:
        return
    removed = set(attachments.values())
    adjusted = [replace(box, text=box.text.rstrip()+"TM") if i in attachments else box
                for i, box in enumerate(boxes) if i not in removed]
    if not result.raw_text:
        result.raw_text = result.text
    result.text, _ = OCREngine.text_from_boxes(adjusted)
    evidence.update(status="SUPERSCRIPT_TM_ATTACHED", text=result.text)
