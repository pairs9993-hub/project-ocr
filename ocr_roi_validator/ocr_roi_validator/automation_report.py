"""Automatic, formula-safe Excel summaries and ROI-level evidence."""
from pathlib import Path
import re

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


def append_safe(sheet, values):
    cleaned = []
    for value in values:
        if isinstance(value, str):
            value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", value)
            if len(value) > 32767:
                value = value[:32710]+"\n[TRUNCATED: see JSON]"
        cleaned.append(value)
    sheet.append(cleaned)
    for cell in sheet[sheet.max_row]:
        if isinstance(cell.value, str):
            cell.data_type = "s"


def initial_result(case, preset):
    return {"tc_number": case.tc_number or str(case.row), "sheet": case.sheet, "row": case.row,
            "title": case.title, "image": case.image, "commands": list(case.commands), "preset": case.preset,
            "status": "NOT_RUN", "reason": "NOT_STARTED", "rois": [
                {"roi_id": r["id"], "expected": case.expected.get(r["id"], r.get("expected", "")),
                 "actual": "", "status": "NOT_RUN", "passed": False, "score": 0, "reason": "NOT_STARTED"}
                for r in preset["rois"]]}


def verdict(status):
    return "PASS" if status == "PASS" else "FAIL" if status in ("FAIL", "FAIL_TIMEOUT", "ERROR") else "N/A"


def write_excel_report(path: Path, report):
    book = Workbook()
    summary = book.active
    summary.title = "TC Summary"
    summary.append(["TC 번호", "시트", "Excel 행", "TC 제목", "정답지 이름", "사용 명령어", "ROI preset", "상태", "사유", "관찰 시간(초)", "합부(PASS/FAIL)"])
    detail = book.create_sheet("ROI Results")
    detail.append(["TC 번호", "시트", "Excel 행", "TC 제목", "정답지 이름", "사용 명령어", "ROI preset",
                   "ROI ID", "정답 텍스트", "검출 텍스트", "상태", "사유", "점수", "완료 주기", "검증 모드", "합부(PASS/FAIL)", "문자 관측률", "공백 판정", "프레임별 OCR 원문", "이미지 공백 판정", "이미지 공백 상세", "스크롤 조합 텍스트", "캡처 수", "버린 프레임 수", "미처리 프레임 수", "ROI OCR 처리 수", "ROI OCR 시간(초)", "미관측 문자(위치:정답)", "최신 화면으로 교체한 프레임 수"])
    trace = book.create_sheet("OCR Timing")
    trace.append(["TC number", "Sheet", "Row", "Preset", "ROI", "Frame", "Captured sec",
                  "OCR started sec", "OCR finished sec", "Status", "Raw text", "Evaluated text"])
    for tc in report["results"]:
        common = [tc.get("tc_number", str(tc["row"])), tc["sheet"], tc["row"], tc["title"], tc.get("image", ""),
                  "\n".join(tc.get("commands", [])), tc["preset"]]
        append_safe(summary, common + [tc["status"], tc.get("reason", tc.get("error", "")), tc.get("elapsed_sec"), verdict(tc["status"])])
        for roi in tc.get("rois", []):
            evidence = roi.get("details", {})
            for timing in evidence.get("frame_timings", []):
                append_safe(trace, [tc.get("tc_number", str(tc["row"])), tc["sheet"], tc["row"], tc["preset"],
                    roi["roi_id"], timing["frame"], timing["captured_sec"], timing["ocr_started_sec"],
                    timing["ocr_finished_sec"], timing["status"], timing.get("raw_text", ""), timing.get("evaluated_text", "")])
            append_safe(detail, common + [roi["roi_id"], roi.get("expected", ""), roi.get("actual", ""),
                roi.get("status", "PASS" if roi.get("passed") else "FAIL"), roi.get("reason", ""), roi.get("score"),
                evidence.get("completed_cycles"), evidence.get("mode", ""), verdict(roi.get("status", "PASS" if roi.get("passed") else "FAIL")),
                evidence.get("character_coverage"), evidence.get("spacing_status", ""),
                "\n--- frame ---\n".join(evidence.get("raw_observations", [])),
                evidence.get("image_spacing", {}).get("status", ""),
                evidence.get("image_spacing", {}).get("reason", evidence.get("image_spacing", {}).get("method", "")),
                evidence.get("assembled_text", evidence.get("accumulated_text", "")),
                evidence.get("capture", {}).get("captured_frames"),
                evidence.get("capture", {}).get("dropped_frames"),
                evidence.get("capture", {}).get("pending_frames"),
                evidence.get("ocr_calls"), evidence.get("ocr_seconds"),
                " / ".join(f"{item['position']}:{item['expected']}" for item in evidence.get("missing_characters", [])),
                evidence.get("capture", {}).get("superseded_frames")])
    info = book.create_sheet("Run Info")
    info.append(["항목", "값"])
    for key in ("state", "started_at", "ended_at", "tc", "zip", "error"):
        append_safe(info, [key, str(report.get(key, ""))])
    info.append(["결과 구분", "PASS=검증 완료; FAIL_TIMEOUT=시간 내 미완료; ERROR=실행 오류; CANCELLED=사용자 중단; NOT_RUN=미실행"])
    info.append(["검출 텍스트", "가로 스크롤 검증이 미완료이면 읽은 OCR 원문을 표시합니다. 서로 다른 프레임은 구분선으로 분리합니다. 검증된 조합은 완료 시 표시하며 미완료 조합의 …는 스크롤 조합 텍스트 열에서만 확인합니다."])
    info.append(["길이 제한", "Excel 셀은 최대 32767자. 초과 시 잘림 표시; 전체 값은 같은 이름의 JSON 참조."])
    for sheet in book:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        sheet.sheet_view.showGridLines = False
        for row in sheet:
            for cell in row:
                if isinstance(cell.value, str):
                    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", cell.value)
                    if len(value) > 32767:
                        value = value[:32710]+"\n[TRUNCATED: see JSON]"
                    cell.value = value
                    cell.data_type = "s"  # Never evaluate user commands/OCR as Excel formulas.
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        for cell in sheet[1]:
            cell.fill = PatternFill("solid", fgColor="17365D")
            cell.font = Font(bold=True, color="FFFFFF")
        for col in range(1, sheet.max_column+1):
            sheet.column_dimensions[get_column_letter(col)].width = 24
        sheet.column_dimensions["F"].width = 45
        for row in range(2, sheet.max_row+1):
            sheet.row_dimensions[row].height = 65
    detail.column_dimensions["I"].width = 55
    detail.column_dimensions["J"].width = 65
    detail.column_dimensions["S"].width = 65
    info.column_dimensions["B"].width = 110
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem+".tmp.xlsx")
    try:
        book.save(temporary)
        temporary.replace(path)
    finally:
        book.close()
        if temporary.exists():
            temporary.unlink()