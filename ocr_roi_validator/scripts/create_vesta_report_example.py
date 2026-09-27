"""Create a clearly labelled sample Excel report (not actual execution results)."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr_roi_validator.automation import TestCase
from ocr_roi_validator.automation_report import initial_result, write_excel_report


def build_example(path):
    preset = {"rois": [{"id": i, "expected": f"ROI {i} 정답 문구"} for i in (1, 2, 3)]}
    entries = []
    for number, status in enumerate(("PASS", "FAIL_TIMEOUT", "CANCELLED", "NOT_RUN"), 1):
        case = TestCase("tc_template", number+1, "양식 예시 — 실제 시험 결과 아님", "roi_courseop",
                        ["예시 명령 1", "예시 명령 2"], image=f"answer_{number}.png", tc_number=f"TC-{number:03d}")
        entry = initial_result(case, preset)
        entry.update(status=status, reason="EXAMPLE_ONLY")
        for roi in entry["rois"]:
            state = "PASS" if status in ("FAIL_TIMEOUT", "CANCELLED") and roi["roi_id"] < 3 else status
            roi.update(status=state, passed=state == "PASS", reason="EXAMPLE_ONLY",
                       actual="" if state == "NOT_RUN" else f"ROI {roi['roi_id']} 검출 문구 예시")
        entries.append(entry)
    write_excel_report(path, {"state": "EXAMPLE_ONLY — 실제 시험 결과 아님", "results": entries})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "examples" / "vesta_result_report_example.xlsx")
    parser.add_argument("--force", action="store_true", help="Replace a previously generated example workbook")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite: {args.output}; pass --force to regenerate")
    build_example(args.output)
    print(args.output)