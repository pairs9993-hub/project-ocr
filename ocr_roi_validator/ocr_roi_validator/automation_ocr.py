"""Headless automation adapter for the GUI's existing OCR and scroll algorithms."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
import time

from PIL import Image
import mss

from .automation import Cancelled, check_cancel, reference_path
from .compare import compare_text
from .scroll_merge import VerticalListAccumulator
from .verification_policy import Confirmation, CycleTracker


@dataclass(frozen=True)
class FrozenValue:
    value: object

    def get(self):
        return self.value


def snapshot_ocr(gui):
    """Must run on the Tk thread. Worker uses only OCR methods, never widgets."""
    processor = copy.copy(gui)
    for name, value in vars(gui).items():
        if name.endswith("_var"):
            setattr(processor, name, FrozenValue(value.get()))
    processor.rois = {}
    processor._ocr_inputs = {}
    processor._last_ocr_input = None
    return processor


def validate_expected_ids(case, preset):
    prefix = f"{case.sheet}!{case.row} ({case.preset})"
    saved_ids = {r["id"] for r in preset["rois"]}
    if set(case.roi_modes) - saved_ids:
        raise ValueError(f"{prefix}: Verification_mode override references unknown ROI IDs.")
    unknown = set(case.expected) - saved_ids
    if unknown:
        raise ValueError(f"{prefix}: Expected columns reference unknown ROI IDs: {sorted(unknown)}")
    if case.expected_from_blocks:
        missing = saved_ids - set(case.expected)
        if missing:
            raise ValueError(f"{prefix}: Expected_Text missing ROI IDs: {sorted(missing)}. Include every saved ROI; preset/Image fallback is disabled for block format.")
        if any(not text.strip() for text in case.expected.values()):
            raise ValueError(f"{prefix}: Expected_Text has empty expected text.")


def prepare_rois(processor, case, preset, references: Path, stop=None):
    from .gui import ROIItem
    validate_expected_ids(case, preset)
    processor._preset_direction = preset["direction"]
    processor.vertical_list_mode_var = FrozenValue(preset["direction"] == "vertical")
    processor.scroll_mode_var = FrozenValue(preset["scrolling"])
    processor.rois = {r["id"]: ROIItem(r["id"], tuple(r["rect"]),
                       case.expected.get(r["id"], r.get("expected", ""))) for r in preset["rois"]}
    missing = [r for r in processor.rois.values() if not r.expected.strip()]
    if missing and case.image:
        path = reference_path(case, references)
        with Image.open(path) as image:
            reference = image.convert("RGB")
        if reference.size != tuple(preset["image_size"]):
            raise ValueError(f"Reference image must contain only the preset GUI screen, size {preset['image_size']}.")
        for roi in missing:
            # No expectation from the current Vesta frame is ever used as truth.
            service = getattr(processor, "_isolated_ocr", None)
            if service is not None:
                roi.expected = service.run(reference, roi.rect, "", stop, time.monotonic()+30).text
            else:
                roi.expected = processor._run_roi_ocr(reference, roi.rect, "").text
    if any(not r.expected.strip() for r in processor.rois.values()):
        raise ValueError("Missing expected text. Fill Expected_Text blocks or Expected_<ROI ID> columns/preset text, or provide a readable Image reference.")
    processor._ocr_inputs = {}
    return processor.rois


def validate_cases(cases, library, references: Path, require_anchors=True):
    from .window_anchor import validate_anchor
    from .roi_presets import automation_preset
    for case in cases:
        prefix = f"{case.sheet}!{case.row} ({case.preset})"
        preset = library["presets"].get(case.preset)
        if preset is None:
            raise ValueError(f"{prefix}: unknown ROI preset.")
        preset = automation_preset(library, case.preset)
        validate_expected_ids(case, preset)
        if require_anchors:
            if "capture_anchor" not in preset:
                raise ValueError(f"{prefix}: register the Vesta GUI screen reference once before running.")
            validate_anchor(preset["capture_anchor"])
        missing = any(not case.expected.get(r["id"], r.get("expected", "")).strip() for r in preset["rois"])
        if missing:
            if not case.image:
                raise ValueError(f"{prefix}: expected text is empty. Add Expected_Text blocks, Expected_<ID> columns or Image.")
            path = reference_path(case, references)
            with Image.open(path) as reference:
                if reference.size != tuple(preset["image_size"]):
                    raise ValueError(f"{prefix}: reference image size {reference.size} does not match {preset['image_size']}.")


def verify_case(processor, session, case, preset, references, duration, fps, stop, output):
    from .gui import _save_failed_roi_diagnostic
    rois = prepare_rois(processor, case, preset, references)
    scrolling = processor._scrolling_enabled()
    accumulators = {i: processor._new_live_accumulator(r) for i, r in rois.items()} if scrolling else {}
    confirmations = {i: Confirmation() for i in rois}
    trackers = {i: CycleTracker(r.expected) for i, r in rois.items()}
    records, text_map, scores, completed = {}, {}, {}, {}
    observed = {i: [] for i in rois}
    spacing_evidence = {}
    started_at = time.monotonic()
    deadline = started_at + min(30.0, case.max_observation, duration)
    reason = "TIMEOUT"
    frame = None
    captured_count = 0
    timings = {i: [] for i in rois}
    try:
        with mss.mss() as capture:
            while time.monotonic() < deadline:
                check_cancel(stop)
                started = time.monotonic()
                frame = session.frame(preset, capture)
                captured_at = time.monotonic()
                captured_count += 1
                for roi_id, roi in rois.items():
                    check_cancel(stop)
                    if roi_id in completed:
                        continue
                    if time.monotonic() >= deadline:
                        break
                    timing = {"frame": captured_count, "assembly_reason": "NOT_EVALUATED",
                              "captured_sec": round(captured_at-started_at, 4),
                              "ocr_started_sec": round(time.monotonic()-started_at, 4)}
                    timings[roi_id].append(timing)
                    try:
                        service = getattr(processor, "_isolated_ocr", None)
                        if service is not None:
                            ocr = service.run(frame, roi.rect, roi.expected, stop, deadline)
                        else:
                            ocr = processor._run_roi_ocr(frame, roi.rect, roi.expected)
                        raw_output = getattr(ocr, "raw_text", None)
                        timing["raw_text"] = raw_output if isinstance(raw_output, str) and raw_output else ocr.text
                        timing["evaluated_text"] = ocr.text
                        timing["status"] = "RETURNED"
                        tm = getattr(ocr, "superscript_evidence", None)
                        if isinstance(tm, dict):
                            timing["superscript"] = tm
                    except Exception as exc:
                        timing["status"] = type(exc).__name__
                        raise
                    finally:
                        timing["ocr_finished_sec"] = round(time.monotonic()-started_at, 4)
                    if time.monotonic() > deadline:
                        timing["assembly_reason"] = "DEADLINE_BEFORE_EVALUATION"
                        raise TimeoutError("OCR observation deadline reached")
                    if processor._last_ocr_input is not None:
                        records[roi_id] = processor._last_ocr_input
                    text_map[roi_id] = ocr.text
                    raw = getattr(ocr, "raw_text", None)
                    raw = raw if isinstance(raw, str) and raw else ocr.text
                    if not observed[roi_id] or observed[roi_id][-1] != raw:
                        observed[roi_id].append(raw)
                    spacing = getattr(ocr, "spacing_evidence", None)
                    if isinstance(spacing, dict):
                        spacing_evidence[roi_id] = spacing
                    trackers[roi_id].add(ocr.text)
                    comparison = compare_text(roi.expected, ocr.text, mode=processor.compare_mode_var.get(),
                                              similarity_threshold=float(processor.similarity_threshold_var.get()))
                    eligible, score = comparison.passed, comparison.score
                    if scrolling:
                        acc = accumulators[roi_id]
                        if isinstance(acc, VerticalListAccumulator):
                            acc.add(ocr.text, ocr.mean_score)
                            eligible, score = acc.passed, acc.coverage
                        else:
                            acc.add(ocr.text, ocr.mean_score, observed_at=time.perf_counter())
                            eligible, score = acc.cycle_complete, acc.coverage
                    evidence = getattr(accumulators.get(roi_id), "evidence", None)
                    if evidence is not None:
                        timing.update(assembly_accepted=evidence.last_aligned, assembly_reason=evidence.last_reason,
                                      aligned_start=evidence.last_start, substitutions=list(evidence.last_substitutions),
                                      character_coverage=evidence.coverage,
                                      spacing_status=evidence.spacing_status, order_confirmed=evidence.order_valid)
                    else:
                        timing["assembly_reason"] = "VERTICAL_EVALUATION" if scrolling else "STATIC_COMPARISON"
                    timing["comparison_passed"] = comparison.passed
                    timing["eligible"] = eligible
                    mode = case.roi_modes.get(roi_id, case.verification_mode)
                    if mode == "cycles":
                        eligible = eligible and trackers[roi_id].cycles >= case.required_cycles
                    scores[roi_id] = score
                    # A newly observed low-confidence/unrelated frame cannot confirm a cached success.
                    related = bool(ocr.text.strip()) and ocr.mean_score >= 0.25
                    if scrolling and getattr(accumulators[roi_id], "evidence", None) is not None:
                        related = related and accumulators[roi_id].evidence.last_aligned
                        related = related and not accumulators[roi_id].evidence.last_substitutions
                    elif scrolling and not comparison.passed:
                        from .verification_policy import normalized
                        related = related and normalized(ocr.text) in normalized(roi.expected)
                    if confirmations[roi_id].add(eligible and related, time.monotonic()):
                        completed[roi_id] = time.monotonic()-started_at
                if len(completed) == len(rois):
                    reason = "ALL_ROIS_PASSED"
                    break
                stop.wait(max(0, min(deadline-time.monotonic(), 1/fps-(time.monotonic()-started))))
    except Cancelled:
        reason = "CANCELLED"
    except TimeoutError:
        reason = "TIMEOUT"
    except Exception as exc:
        reason = f"ERROR: {exc}"
    artifact_error = None
    try:
        output.mkdir(parents=True, exist_ok=True)
        if frame is not None:
            frame.save(output / "screen.png")
    except OSError as exc:
        artifact_error = str(exc)
    results = []
    for roi_id, roi in rois.items():
        details = {"completed_cycles": trackers[roi_id].cycles, "cycle_evidence": trackers[roi_id].reason,
                   "ambiguous_observations": trackers[roi_id].ambiguous,
                   "mode": case.roi_modes.get(roi_id, case.verification_mode),
                   "required_cycles": case.required_cycles, "completed_at_sec": completed.get(roi_id)}
        details["raw_observations"] = observed[roi_id]
        details["capture"] = {"capture_policy": "synchronous_de22028", "captured_frames": captured_count,
                              "dropped_frames": 0, "pending_frames": 0, "superseded_frames": 0}
        details["ocr_calls"] = len(timings[roi_id])
        details["ocr_seconds"] = round(sum(t["ocr_finished_sec"]-t["ocr_started_sec"] for t in timings[roi_id]), 4)
        details["frame_timings"] = timings[roi_id]
        if roi_id in spacing_evidence:
            details["image_spacing"] = spacing_evidence[roi_id]
        if scrolling:
            acc = accumulators[roi_id]
            if isinstance(acc, VerticalListAccumulator):
                acc.finalize()
                details.update(order_valid=acc.order_valid, missing_indices=list(acc.missing_indices))
            details["accumulated_text"] = acc.final_text
        # Never report expected-derived accumulator reconstructions as raw detected text.
        roi.actual = "\n--- frame ---\n".join(observed[roi_id]) if scrolling else text_map.get(roi_id, "")
        if scrolling:
            details["raw_observations"] = observed[roi_id]
            evidence = getattr(accumulators[roi_id], "evidence", None)
            if evidence is not None:
                details.update(evidence.details())
                from .horizontal_scroll import observed_display
                roi.actual = observed_display(evidence, observed[roi_id])
                details["display_source"] = "assembled" if evidence.passed else "raw_observations"
        roi.passed = roi_id in completed
        status = "PASS" if roi.passed else "CANCELLED" if reason == "CANCELLED" else "ERROR" if reason.startswith("ERROR") else "FAIL_TIMEOUT"
        result = {**asdict(roi), "score": scores.get(roi_id, 0), "details": details,
                  "status": status, "reason": "VERIFIED" if roi.passed else reason}
        if not roi.passed and frame is not None:
            try:
                diagnostic = _save_failed_roi_diagnostic(frame, roi, output_root=output / "failures",
                    preprocess=processor._roi_preprocess_config(),
                    ocr_input=None if scrolling else records.get(roi_id))
                result["diagnostic"] = str(diagnostic)
            except OSError as exc:
                result["diagnostic_error"] = str(exc)
        results.append(result)
    status = "PASS" if reason == "ALL_ROIS_PASSED" else "CANCELLED" if reason == "CANCELLED" else "ERROR" if reason.startswith("ERROR") else "FAIL_TIMEOUT"
    return {"status": status, "reason": reason, "rois": results,
            "artifact_error": artifact_error, "elapsed_sec": time.monotonic()-started_at}, frame