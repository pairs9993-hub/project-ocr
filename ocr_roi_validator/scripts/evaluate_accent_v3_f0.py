"""Gate F0 for accent-v3, on the exact recorded target ROI.

Runs the full product path -- detector, French baseline recognizer, CTC
alignment, native-line crop, localization guard, accent-v3 multi-view -- over
the recorded OCR input and six 1px perturbations of it.

The specialist recognizer is not involved. Every ``é`` the baseline emits is
examined by the same rule; nothing keys on the first word or on any position.
Expected text is read only after all OCR decisions are final, to score them.

The saved image is already what ``engine.run()`` received, so no margin,
padding or upscaling is re-applied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATOR_ROOT))

from ocr_roi_validator.accent_cnn_verifier import (  # noqa: E402
    ACCENT_ABSENT,
    ACCENT_PRESENT,
    UNKNOWN,
    AccentCnnVerifier,
)
from ocr_roi_validator.ctc_geometry import v3_crop_bounds  # noqa: E402
from ocr_roi_validator.model_package import load_model_package  # noqa: E402

# The seven inputs fixed by Gate A. No others, and none removed.
PERTURBATIONS = (
    "none",
    "crop_left_1px",
    "crop_right_1px",
    "crop_top_1px",
    "crop_bottom_1px",
    "pad_border_1px",
    "crop_all_1px",
)


def perturb(image: Image.Image, kind: str) -> Image.Image:
    width, height = image.size
    if kind == "none":
        return image
    if kind == "crop_left_1px":
        return image.crop((1, 0, width, height))
    if kind == "crop_right_1px":
        return image.crop((0, 0, width - 1, height))
    if kind == "crop_top_1px":
        return image.crop((0, 1, width, height))
    if kind == "crop_bottom_1px":
        return image.crop((0, 0, width, height - 1))
    if kind == "pad_border_1px":
        padded = Image.new(image.mode, (width + 2, height + 2), image.getpixel((0, 0)))
        padded.paste(image, (1, 1))
        return padded
    if kind == "crop_all_1px":
        return image.crop((1, 1, width - 1, height - 1))
    raise ValueError(kind)


def build_engine(package_dir: Path, language: str):
    from rapidocr_onnxruntime import RapidOCR

    package = load_model_package(package_dir)
    p = package.preprocess
    engine = RapidOCR(
        det_model_path=str(package.detector_model),
        rec_model_path=str(package.recognizers[language]),
        rec_keys_path=str(package.dictionary),
        det_limit_type=str(p["det_limit_type"]),
        det_limit_side_len=int(p["det_limit_side_len"]),
        det_box_thresh=float(p["det_box_thresh"]),
        det_unclip_ratio=float(p["det_unclip_ratio"]),
        det_donot_use_dilation=bool(p["det_donot_use_dilation"]),
        use_cls=bool(p["use_cls"]),
    )
    return engine, package


def load_labels(dictionary: Path) -> list[str]:
    characters = dictionary.read_text(encoding="utf-8").split("\n")
    if characters and characters[-1] == "":
        characters = characters[:-1]
    return ["<blank>"] + characters + [" "]


def collapse_ctc(argmax, labels):
    emitted, previous = [], 0
    for timestep, index in enumerate(argmax):
        if index != 0 and index != previous:
            emitted.append({"char": labels[index] if index < len(labels) else "?",
                            "start": timestep, "end": timestep})
        elif emitted and index == previous and index != 0:
            emitted[-1]["end"] = timestep
        previous = index
    return emitted


def first_word(text: str) -> str:
    line = (text or "").split("\n", 1)[0].strip()
    return line.split(" ", 1)[0] if line else ""


def run_once(engine, package, labels, verifier, bgr: np.ndarray) -> dict:
    """Full path for one image: baseline text, per-glyph verdicts, final text."""
    boxes, _ = engine.auto_text_det(bgr)
    if boxes is None or len(boxes) == 0:
        return {"baseline_text": "", "final_text": "", "lines": 0, "glyphs": []}

    crops = engine.get_crop_img_list(bgr, boxes)
    recognizer = engine.text_rec
    channels, height, width = recognizer.rec_image_shape
    ratios = [c.shape[1] / float(c.shape[0]) for c in crops]
    max_wh_ratio = max([width / height] + ratios)

    baseline_lines, final_lines, glyph_records = [], [], []
    for line_index, crop in enumerate(crops):
        crop_h, crop_w = crop.shape[:2]
        tensor = recognizer.resize_norm_img(crop, max_wh_ratio)[np.newaxis, :]
        logits = np.asarray(recognizer.session(tensor.astype(np.float32))[0])
        probabilities = logits[0]
        argmax = probabilities.argmax(axis=-1).tolist()
        decoded = recognizer.postprocess_op(
            logits, False, wh_ratio_list=[crop_w / float(crop_h)],
            max_wh_ratio=max_wh_ratio,
        )[0][0]
        baseline_lines.append(decoded)

        emitted = collapse_ctc(argmax, labels)
        if "".join(item["char"] for item in emitted) != decoded:
            final_lines.append(decoded)      # untrustworthy alignment: keep as-is
            continue

        padded_w = int(height * max_wh_ratio)
        resized_w = min(padded_w, int(math.ceil(height * (crop_w / crop_h))))
        timesteps = probabilities.shape[0]
        scale = (padded_w / timesteps) * (crop_w / resized_w)

        # Median span width on this line, the guard's scale reference.
        spans = [(item["end"] + 1 - item["start"]) * scale for item in emitted]
        median_span = float(np.median(spans)) if spans else None

        characters = list(decoded)
        for position, item in enumerate(emitted):
            # Every predicted é is examined by the same rule.
            if unicodedata.normalize("NFC", item["char"]) != "é":
                continue
            x0, x1 = v3_crop_bounds(item["start"], item["end"], scale, crop_w)
            result = verifier.verify(crop, x0, x1, median_span)
            applied = result.is_accent_absent and position < len(characters)
            if applied:
                characters[position] = "e"   # single codepoint, length preserved
            glyph_records.append(
                {
                    "line_index": line_index,
                    "position": position,
                    "x0": x0, "x1": x1,
                    "verdict": result.verdict,
                    "probability_absent": result.probability_absent,
                    "view_probabilities": list(result.view_probabilities),
                    "reason": result.reason,
                    "veto": result.reason == "accent_present_veto",
                    "applied": applied,
                }
            )
        final_lines.append("".join(characters))

    return {
        "baseline_text": "\n".join(baseline_lines),
        "final_text": "\n".join(final_lines),
        "lines": len(crops),
        "line_texts": baseline_lines,
        "glyphs": glyph_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if metadata.get("ocr_input_fidelity") != "exact_recorded_ocr_input":
        print("refusing: not an exact recorded OCR input", file=sys.stderr)
        return 1
    expected = metadata["expected"]           # scoring only, after all decisions
    expected_first_word = first_word(expected)

    engine, package = build_engine(args.package, args.language)
    labels = load_labels(package.dictionary)
    verifier = AccentCnnVerifier(args.onnx, args.config)
    print(f"model {verifier.version}  absent>={verifier.absent_threshold:.9f}")

    with Image.open(args.roi) as handle:
        base = handle.convert("RGB")

    rows = []
    for kind in PERTURBATIONS:
        image = perturb(base, kind)
        raw = np.asarray(image)[:, :, ::-1].copy()
        digest = hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()
        outcome = run_once(engine, package, labels, verifier, raw)

        baseline_text = outcome["baseline_text"]
        final_text = outcome["final_text"]
        baseline_first = first_word(baseline_text)
        final_first = first_word(final_text)

        # Which characters changed, and were any of them not an accent?
        changes = []
        if len(baseline_text) == len(final_text):
            for index, (before, after) in enumerate(zip(baseline_text, final_text)):
                if before != after:
                    changes.append({"index": index, "from": before, "to": after})
        else:
            changes.append({"index": -1, "from": "LENGTH", "to": "CHANGED"})
        non_accent_changes = [
            c for c in changes if not (c["from"] == "é" and c["to"] == "e")
        ]
        accent_additions = [c for c in changes if c["to"] == "é"]

        if baseline_first == expected_first_word:
            classification = "BASELINE_ALREADY_CORRECT"
        elif final_first == expected_first_word:
            classification = "CORRECTED_BY_V3"
        elif "é" in baseline_first and final_first == baseline_first:
            classification = "ABSTAINED"
        else:
            classification = "OTHER_OCR_FAILURE"

        rows.append(
            {
                "perturbation": kind,
                "image_size": [image.size[0], image.size[1]],
                "image_sha256": digest,
                "lines": outcome["lines"],
                "line_texts": outcome.get("line_texts", []),
                "baseline_text": baseline_text,
                "baseline_first_word": baseline_first,
                "final_text": final_text,
                "final_first_word": final_first,
                "glyphs": outcome["glyphs"],
                "changes": changes,
                "non_accent_changes": non_accent_changes,
                "accent_additions": accent_additions,
                "first_word_correct": final_first == expected_first_word,
                "full_roi_exact": final_text == expected,
                "classification": classification,
            }
        )

        print(f"\n--- {kind}  {image.size[0]}x{image.size[1]}  sha {digest[:12]}")
        print(f"  baseline first word : {baseline_first!r}")
        print(f"  final    first word : {final_first!r}   [{classification}]")
        for glyph in outcome["glyphs"]:
            print(f"    line {glyph['line_index']} pos {glyph['position']:>3d} "
                  f"x[{glyph['x0']},{glyph['x1']}] {glyph['verdict']:>8s} "
                  f"p={glyph['probability_absent']:.6f} applied={glyph['applied']} "
                  f"({glyph['reason'][:34]})")
        if changes:
            print(f"  changes: {changes}")

    # Scoring, once every OCR decision is already made.
    first_word_ok = sum(r["first_word_correct"] for r in rows)
    corrected = sum(r["classification"] == "CORRECTED_BY_V3" for r in rows)
    already = sum(r["classification"] == "BASELINE_ALREADY_CORRECT" for r in rows)
    abstained = sum(r["classification"] == "ABSTAINED" for r in rows)
    other = sum(r["classification"] == "OTHER_OCR_FAILURE" for r in rows)
    non_accent = sum(len(r["non_accent_changes"]) for r in rows)
    additions = sum(len(r["accent_additions"]) for r in rows)
    full_exact = sum(r["full_roi_exact"] for r in rows)

    # Legitimate accents: every é the baseline emitted that survived to final.
    legit_total = legit_preserved = 0
    for row in rows:
        for glyph in row["glyphs"]:
            if glyph["line_index"] == 0 and glyph["position"] <= 1:
                continue           # the target glyph, scored separately
            legit_total += 1
            if not glyph["applied"]:
                legit_preserved += 1

    print(f"\nfirst word correct : {first_word_ok}/7")
    print(f"  CORRECTED_BY_V3          : {corrected}")
    print(f"  BASELINE_ALREADY_CORRECT : {already}")
    print(f"  ABSTAINED                : {abstained}")
    print(f"  OTHER_OCR_FAILURE        : {other}")
    print(f"legitimate accents preserved : {legit_preserved}/{legit_total}")
    print(f"non-accent character changes : {non_accent}")
    print(f"accent additions (e->é)      : {additions}")
    print(f"full ROI exact               : {full_exact}/7  "
          f"(the dropped l in l'eau is a separate defect)")

    gates = {
        "first_word_correct_7_of_7": first_word_ok == 7,
        "no_legitimate_accent_removed": legit_preserved == legit_total,
        "no_non_accent_changes": non_accent == 0,
        "no_accent_additions": additions == 0,
    }
    for gate, ok in gates.items():
        print(f"  gate {gate}: {'PASS' if ok else 'FAIL'}")
    passed = all(gates.values())
    print(f"\nACCENT_V3_F0_STATUS = {'PASS' if passed else 'FAIL'}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(
                {
                    "roi": str(args.roi),
                    "model_version": verifier.version,
                    "absent_threshold": verifier.absent_threshold,
                    "expected": expected,
                    "rows": rows,
                    "first_word_correct": first_word_ok,
                    "classifications": {
                        "CORRECTED_BY_V3": corrected,
                        "BASELINE_ALREADY_CORRECT": already,
                        "ABSTAINED": abstained,
                        "OTHER_OCR_FAILURE": other,
                    },
                    "legitimate_accents_preserved": legit_preserved,
                    "legitimate_accents_total": legit_total,
                    "non_accent_changes": non_accent,
                    "accent_additions": additions,
                    "full_roi_exact": full_exact,
                    "gates": gates,
                    "passed": passed,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out_json}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
