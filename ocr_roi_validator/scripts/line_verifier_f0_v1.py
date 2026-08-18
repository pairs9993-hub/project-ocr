"""One-shot Gate F0 for the frozen line verifier, on the recorded target ROI.

Runs the product path end to end -- detector, French baseline, CTC alignment,
line-verifier input contract, pre-model gate, ONNX inference in the product
runtime, the frozen threshold and its uncertainty band -- over the exact
recorded OCR input and the same six 1px perturbations Gate A fixed.

Every é the baseline emits is examined by the same rule. Nothing keys on the
first word, on a position, or on the expected string; Expected is read only
after every decision is final, and only to score.

The product ONNX Runtime is the official result. PyTorch is not consulted.

This runs once. The instruction forbids adjusting anything after seeing it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
for extra in (VALIDATOR_ROOT, VALIDATOR_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from mine_line_triggers_v1 import collapse_ctc, load_labels  # noqa: E402
from ocr_roi_validator.line_verifier_input import (  # noqa: E402
    LineVerifierInputConfig, assert_no_text_leakage, build_line_input,
)
from ocr_roi_validator.model_package import load_model_package  # noqa: E402
from ocr_roi_validator.premodel_gate import check_premodel  # noqa: E402
from ocr_roi_validator.runtime_action import (  # noqa: E402
    APPLY_E_CORRECTION, KEEP_BASELINE, apply_action,
)
from ocr_roi_validator.runtime_guard import (  # noqa: E402
    BARE_E, RUNTIME_EPSILON, UNKNOWN, guarded_verdict,
)

PERTURBATIONS = ("none", "crop_left_1px", "crop_right_1px", "crop_top_1px",
                 "crop_bottom_1px", "pad_border_1px", "crop_all_1px")
VERDICT_NAMES = {0: "ACCENT_PRESENT", 1: "BARE_E", 2: "UNKNOWN"}


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
        padded = Image.new("RGB", (width + 2, height + 2), (0, 0, 0))
        padded.paste(image, (1, 1))
        return padded
    if kind == "crop_all_1px":
        return image.crop((1, 1, width - 1, height - 1))
    raise ValueError(kind)


def softmax(values):
    values = np.asarray(values, dtype=np.float32)
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return (exponent / exponent.sum(axis=1, keepdims=True)).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--roi", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--threshold-config", type=Path, required=True)
    parser.add_argument("--expected-from", type=Path, required=True,
                        help="prior F0 record; read only for scoring")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from rapidocr_onnxruntime import RapidOCR

    threshold = np.float32(json.loads(
        args.threshold_config.read_text(encoding="utf-8"))["threshold"])
    package = load_model_package(args.package)
    preprocess = package.preprocess
    engine = RapidOCR(
        det_model_path=str(package.detector_model),
        rec_model_path=str(package.recognizers[args.language]),
        rec_keys_path=str(package.dictionary),
        det_limit_type=str(preprocess["det_limit_type"]),
        det_limit_side_len=int(preprocess["det_limit_side_len"]),
        det_box_thresh=float(preprocess["det_box_thresh"]),
        det_unclip_ratio=float(preprocess["det_unclip_ratio"]),
        det_donot_use_dilation=bool(preprocess["det_donot_use_dilation"]),
        use_cls=bool(preprocess["use_cls"]))
    labels = load_labels(package.dictionary)
    recognizer = engine.text_rec
    _, rec_height, rec_width = recognizer.rec_image_shape
    config = LineVerifierInputConfig()

    session = ort.InferenceSession(str(args.onnx),
                                   providers=["CPUExecutionProvider"])
    input_names = [i.name for i in session.get_inputs()]

    original = Image.open(args.roi).convert("RGB")
    rows = []
    for kind in PERTURBATIONS:
        image = perturb(original, kind)
        bgr = np.asarray(image)[:, :, ::-1].copy()
        digest = hashlib.sha256(np.asarray(image).tobytes()).hexdigest()

        boxes, _ = engine.auto_text_det(bgr)
        if boxes is None or len(boxes) == 0:
            rows.append({"perturbation": kind, "image_sha256": digest,
                         "detector": "MISS", "lines": []})
            continue
        crops = engine.get_crop_img_list(bgr, boxes)
        ratios = [c.shape[1] / float(c.shape[0]) for c in crops]
        max_wh_ratio = max([rec_width / rec_height] + ratios)

        lines = []
        for line_index, crop in enumerate(crops):
            crop_height, crop_width = crop.shape[:2]
            tensor = recognizer.resize_norm_img(crop, max_wh_ratio)[np.newaxis]
            logits = np.asarray(recognizer.session(tensor.astype(np.float32))[0])
            probabilities = logits[0]
            decoded = recognizer.postprocess_op(
                logits, False, wh_ratio_list=[crop_width / float(crop_height)],
                max_wh_ratio=max_wh_ratio)[0][0]
            emitted = collapse_ctc(probabilities.argmax(axis=-1).tolist(), labels)
            aligned = "".join(i["char"] for i in emitted) == decoded

            baseline = decoded
            final = decoded
            decisions = []
            # Examine every é the baseline emitted, by the same rule.
            for ordinal, item in enumerate(emitted):
                if unicodedata.normalize("NFC", item["char"]) != "é":
                    continue
                gate = check_premodel(ordinal, len(emitted), input_built=aligned)
                entry = {"ordinal": ordinal,
                         "premodel_verdict": gate.verdict,
                         "premodel_reason": gate.reason,
                         "network_invoked": False,
                         "p_bare": None, "internal_verdict": None,
                         "action": KEEP_BASELINE}
                if gate.rejected:
                    decisions.append(entry)
                    continue
                prepared = build_line_input(
                    crop, probabilities, token_label=item["label"],
                    token_start=item["start"], token_end=item["end"],
                    target_ordinal=ordinal, decoded_length=len(emitted),
                    config=config)
                if prepared is None:
                    entry["premodel_verdict"] = "UNKNOWN"
                    entry["premodel_reason"] = "INPUT_BUILD_FAILED"
                    decisions.append(entry)
                    continue
                assert_no_text_leakage(prepared)
                outputs = session.run(None, {
                    input_names[0]: prepared.planes[None].astype(np.float32),
                    input_names[1]: prepared.query[None].astype(np.float32)})[0]
                probs = softmax(outputs)
                verdict = int(guarded_verdict(probs, threshold, RUNTIME_EPSILON)[0])
                entry["network_invoked"] = True
                entry["p_bare"] = float(probs[0, BARE_E])
                entry["p_accent"] = float(probs[0, 0])
                entry["internal_verdict"] = VERDICT_NAMES[verdict]
                entry["action"] = (APPLY_E_CORRECTION if verdict == BARE_E
                                   else KEEP_BASELINE)
                if entry["action"] == APPLY_E_CORRECTION:
                    # Positions refer to the current string; corrections are
                    # applied left to right and only ever swap one character.
                    final = apply_action(final, APPLY_E_CORRECTION, ordinal)
                decisions.append(entry)

            lines.append({
                "line_index": line_index, "baseline_text": baseline,
                "final_text": final, "sequence_aligned": aligned,
                "accent_decisions": decisions,
                "baseline_first_word": baseline.split()[0] if baseline.split() else "",
                "final_first_word": final.split()[0] if final.split() else "",
                "changed_characters": [
                    {"index": i, "from": a, "to": b}
                    for i, (a, b) in enumerate(zip(baseline, final)) if a != b],
            })
        rows.append({"perturbation": kind, "image_sha256": digest,
                     "detector": "OK", "lines": lines})

    # --- scoring; Expected is read only now -------------------------------
    prior = json.loads(args.expected_from.read_text(encoding="utf-8"))
    expected = prior["expected"]
    expected_lines = expected.split("\n")
    expected_first = expected_lines[0].split()[0]          # "Veuillez"

    scored = []
    for row in rows:
        lines = row.get("lines", [])
        first_line = lines[0] if lines else None
        first_word_ok = bool(first_line
                             and first_line["final_first_word"] == expected_first)
        # A legitimate accent is one the baseline read correctly and that must
        # survive; every é outside line 0's first word counts.
        accent_removed = 0
        non_accent_changes = 0
        accent_additions = 0
        for line in lines:
            for change in line["changed_characters"]:
                if change["from"] == "é" and change["to"] == "e":
                    if not (line is first_line and change["index"] <= 1):
                        accent_removed += 1
                elif change["from"] == "e" and change["to"] == "é":
                    accent_additions += 1
                else:
                    non_accent_changes += 1
        blocked = bool(first_line
                       and any(d["action"] == KEEP_BASELINE
                               and d["ordinal"] <= 1
                               for d in first_line["accent_decisions"]))
        full_roi_exact = ("\n".join(l["final_text"] for l in lines) == expected)
        scored.append({
            "perturbation": row["perturbation"],
            "image_sha256": row["image_sha256"],
            "detector": row["detector"],
            "baseline_first_word": first_line["baseline_first_word"] if first_line else None,
            "final_first_word": first_line["final_first_word"] if first_line else None,
            "first_word_correct": first_word_ok,
            "target_blocked_by_keep_baseline": blocked,
            "legitimate_accent_removed": accent_removed,
            "accent_additions": accent_additions,
            "non_accent_changes": non_accent_changes,
            "full_roi_exact": full_roi_exact,
            "lines": lines,
        })

    first_word_pass = sum(1 for s in scored if s["first_word_correct"])
    blocked_total = sum(1 for s in scored if s["target_blocked_by_keep_baseline"])
    gates = {
        "first_word_corrected_7_of_7": first_word_pass == len(PERTURBATIONS),
        "legitimate_accents_preserved": all(
            s["legitimate_accent_removed"] == 0 for s in scored),
        "no_accent_additions": all(s["accent_additions"] == 0 for s in scored),
        "no_non_accent_changes": all(s["non_accent_changes"] == 0 for s in scored),
        "target_never_blocked": blocked_total == 0,
    }
    passed = all(gates.values())

    report = {
        "gate": "line_verifier_f0_v1",
        "run_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "one_shot": True,
        "official_runtime": "product .venv ONNX Runtime %s" % ort.__version__,
        "roi": str(args.roi),
        "roi_sha256": hashlib.sha256(args.roi.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(args.onnx.read_bytes()).hexdigest(),
        "threshold": float(threshold),
        "runtime_epsilon": float(RUNTIME_EPSILON),
        "perturbations": list(PERTURBATIONS),
        "rows": scored,
        "summary": {
            "first_word_corrected": "%d/%d" % (first_word_pass, len(PERTURBATIONS)),
            "target_blocked": blocked_total,
            "legitimate_accents_removed": sum(
                s["legitimate_accent_removed"] for s in scored),
            "accent_additions": sum(s["accent_additions"] for s in scored),
            "non_accent_changes": sum(s["non_accent_changes"] for s in scored),
            "full_roi_exact": sum(1 for s in scored if s["full_roi_exact"]),
        },
        "gates": gates,
        "ACCENT_TARGET_FEASIBILITY": "PASS" if passed else "FAIL",
        "l_eau_note": ("the separate l'eau -> 'eau omission is out of scope for "
                       "this accent gate and is reported only under "
                       "full_roi_exact"),
    }
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    args.out.write_text(payload, encoding="utf-8")

    print("%-16s %-14s %-14s %-9s %-8s" % ("perturbation", "baseline 1st",
                                           "final 1st", "corrected", "blocked"))
    for entry in scored:
        print("%-16s %-14s %-14s %-9s %-8s"
              % (entry["perturbation"], entry["baseline_first_word"],
                 entry["final_first_word"], entry["first_word_correct"],
                 entry["target_blocked_by_keep_baseline"]))
    print("\nsummary %s" % report["summary"])
    print("gates   %s" % gates)
    print("ACCENT_TARGET_FEASIBILITY %s" % report["ACCENT_TARGET_FEASIBILITY"])
    print("report sha256 %s" % hashlib.sha256(payload.encode()).hexdigest())
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
