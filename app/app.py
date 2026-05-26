"""Streamlit GUI for OCR validation.

Run:
    streamlit run app/app.py

Two modes:
    1. 일반 테스트  (Single image OCR)
    2. 알집 테스트  (Compare test ZIP vs reference ZIP, export Excel)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

# allow running as `streamlit run app/app.py` from project root
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from PIL import Image

from excel_report import ReportRow, build_workbook, workbook_to_bytes
from metrics import cer, normalize_text, verdict, wer
from ocr_runner import ocr_image
from zip_compare import ZipImage, load_zip_images, pair_by_stem


# ----------------------------- page setup ----------------------------- #

st.set_page_config(
    page_title="OCR 검증 도구",
    page_icon="OCR",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main .block-container { max-width: 1400px; padding-top: 1.2rem; }
    h1, h2, h3 { letter-spacing: -0.01em; }
    .metric-tile {
        background: #f6f8fb;
        border: 1px solid #e3e8ef;
        border-radius: 8px;
        padding: 14px 16px;
        text-align: center;
    }
    .metric-tile .label { font-size: 13px; color: #5b6470; }
    .metric-tile .value { font-size: 26px; font-weight: 700; color: #1f2933; }
    .verdict-PASS { color: #056b2c; font-weight: 700; }
    .verdict-WARN { color: #a36900; font-weight: 700; }
    .verdict-FAIL { color: #b00020; font-weight: 700; }
    .ocr-text {
        background: #fafbfc;
        border: 1px solid #e3e8ef;
        border-radius: 6px;
        padding: 10px 12px;
        font-family: Consolas, "Courier New", monospace;
        font-size: 14px;
        line-height: 1.55;
        white-space: pre-wrap;
        min-height: 60px;
    }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-PASS { background: #d9f5e3; color: #056b2c; }
    .badge-WARN { background: #fff1cf; color: #a36900; }
    .badge-FAIL { background: #ffd7dd; color: #b00020; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------- sidebar ----------------------------- #

st.sidebar.title("OCR 검증 도구")
st.sidebar.caption("가전 UI 화면 OCR 정확도 검증")

st.sidebar.markdown("### 모델 설정")

# Auto-detect bundled custom models
DEFAULT_MODEL = ROOT / "models" / "rec_v0.onnx"
DEFAULT_FR_MODEL = ROOT / "models" / "rec_fr_v1.onnx"
DEFAULT_KEYS = ROOT / "models" / "ppocr_keys.txt"
DEFAULT_DET = ROOT / "models" / "det_v0.onnx"
MODEL_PRESETS = {
    "공용 production recognizer": DEFAULT_MODEL,
    "프랑스어 recognizer v1": DEFAULT_FR_MODEL,
}
available_model_presets = {
    name: path for name, path in MODEL_PRESETS.items() if path.exists() and DEFAULT_KEYS.exists()
}
has_bundled = bool(available_model_presets)
has_bundled_det = DEFAULT_DET.exists()

if has_bundled:
    use_custom = st.sidebar.toggle(
        "자체 학습 rec 모델 사용 (권장)",
        value=True,
        help="끄면 RapidOCR 사전학습 rec 모델로 비교 가능합니다.",
    )
    preset_names = list(available_model_presets)
    default_preset = "프랑스어 recognizer v1" if "프랑스어 recognizer v1" in available_model_presets else preset_names[0]
    selected_model_preset = st.sidebar.selectbox(
        "rec 모델 선택",
        preset_names,
        index=preset_names.index(default_preset),
    )
else:
    st.sidebar.info(
        "자체 학습 rec ONNX 모델이 `app/models/` 에 없어 RapidOCR 기본 모델을 사용합니다."
    )
    use_custom = False
    selected_model_preset = ""

if has_bundled_det:
    use_custom_det = st.sidebar.toggle(
        "자체 학습 det 모델 사용 (권장)",
        value=True,
        help="끄면 RapidOCR 사전학습 det 모델로 비교 가능합니다.",
    )
else:
    use_custom_det = False

with st.sidebar.expander("고급: 다른 ONNX 경로 지정", expanded=False):
    custom_model = st.text_input(
        "rec ONNX 경로", value="", placeholder="예: models/rec_v0.onnx"
    )
    custom_keys = st.text_input(
        "키(dict) 경로", value="", placeholder="예: models/ppocr_keys.txt"
    )
    custom_det = st.text_input(
        "det ONNX 경로", value="", placeholder="예: models/det_v0.onnx"
    )

if custom_model.strip():
    rec_model_path = Path(custom_model).expanduser()
elif use_custom:
    rec_model_path = available_model_presets[selected_model_preset]
else:
    rec_model_path = None

if custom_keys.strip():
    rec_keys_path = Path(custom_keys).expanduser()
elif use_custom:
    rec_keys_path = DEFAULT_KEYS
else:
    rec_keys_path = None

if custom_det.strip():
    det_model_path = Path(custom_det).expanduser()
elif use_custom_det:
    det_model_path = DEFAULT_DET
else:
    det_model_path = None

if rec_model_path and not rec_model_path.exists():
    st.sidebar.error(f"모델 파일을 찾을 수 없습니다: {rec_model_path}")
    rec_model_path = None
if rec_keys_path and not rec_keys_path.exists():
    st.sidebar.error(f"키 파일을 찾을 수 없습니다: {rec_keys_path}")
    rec_keys_path = None
if det_model_path and not det_model_path.exists():
    st.sidebar.error(f"det 모델 파일을 찾을 수 없습니다: {det_model_path}")
    det_model_path = None

if rec_model_path and rec_keys_path:
    st.sidebar.success(f"활성 rec 모델: **{rec_model_path.name}**")
else:
    st.sidebar.warning("활성 rec 모델: **RapidOCR 사전학습 (기본)**")

if det_model_path:
    st.sidebar.success(f"활성 det 모델: **{det_model_path.name}**")
else:
    st.sidebar.warning("활성 det 모델: **RapidOCR 사전학습 (기본)**")

st.sidebar.markdown("---")
st.sidebar.markdown("### 판정 기준")
st.sidebar.markdown(
    """
    <div style='font-size:13px;'>
      <span class='badge badge-PASS'>PASS</span> &nbsp; CER ≤ 5%<br><br>
      <span class='badge badge-WARN'>WARN</span> &nbsp; 5% &lt; CER ≤ 20%<br><br>
      <span class='badge badge-FAIL'>FAIL</span> &nbsp; CER &gt; 20%
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------- title ----------------------------- #

st.title("OCR 검증 도구")
st.caption(
    "테스트 이미지와 정답(reference) 이미지의 OCR 결과를 비교하여 "
    "문자 오류율(CER) 기반으로 판정합니다."
)

mode_tab_single, mode_tab_zip = st.tabs(["일반 테스트", "알집 테스트 (ZIP)"])


# =====================================================================
# 일반 테스트
# =====================================================================

with mode_tab_single:
    st.subheader("일반 테스트 — 단일 이미지 OCR")
    st.caption(
        "이미지 한 장을 업로드하여 OCR 결과를 즉시 확인합니다. "
        "정답 텍스트를 입력하면 CER/WER도 함께 계산합니다."
    )

    col_up, col_ref = st.columns([1, 1])
    with col_up:
        single_file = st.file_uploader(
            "이미지 파일",
            type=["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"],
            key="single_upload",
        )
    with col_ref:
        ref_text = st.text_area(
            "정답 텍스트 (선택)",
            value="",
            height=120,
            help="이 칸에 정답을 입력하면 CER/WER이 계산됩니다.",
            key="single_ref",
        )

    run_single = st.button("OCR 실행", type="primary", key="run_single",
                           disabled=single_file is None)

    if run_single and single_file is not None:
        try:
            img = Image.open(single_file)
            img.load()
        except Exception as e:
            st.error(f"이미지를 열 수 없습니다: {e}")
        else:
            with st.spinner("OCR 처리 중..."):
                res = ocr_image(img, rec_model_path, rec_keys_path, det_model_path)

            st.markdown("### 결과")
            c_img, c_txt = st.columns([1, 1])
            with c_img:
                st.image(img, caption=single_file.name, use_container_width=True)
            with c_txt:
                st.markdown("**OCR 인식 결과**")
                st.markdown(
                    f"<div class='ocr-text'>{(res.text or '<i>(인식된 텍스트 없음)</i>')}</div>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"박스 {res.n_boxes}개 · 평균 신뢰도 {res.mean_score:.3f} · "
                    f"{res.elapsed_ms:.0f} ms"
                )

                if ref_text.strip():
                    c = cer(ref_text, res.text)
                    w = wer(ref_text, res.text)
                    v = verdict(c)
                    st.markdown(
                        f"<div style='margin-top:14px'>"
                        f"<span class='badge badge-{v}'>{v}</span> &nbsp;"
                        f"<b>CER</b> {c:.4f} &nbsp; <b>WER</b> {w:.4f}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )


# =====================================================================
# 알집 테스트 (ZIP)
# =====================================================================

def _metric_tile(label: str, value: str) -> str:
    return (
        f"<div class='metric-tile'><div class='label'>{label}</div>"
        f"<div class='value'>{value}</div></div>"
    )


with mode_tab_zip:
    st.subheader("알집 테스트 — 테스트 ZIP vs 정답 ZIP 비교")
    st.caption(
        "두 알집의 동일한 파일명(확장자 제외)끼리 자동 매칭하여 "
        "양쪽 모두 OCR한 뒤 결과를 비교합니다."
    )

    c_test, c_ref = st.columns(2)
    with c_test:
        test_zip = st.file_uploader(
            "테스트 이미지 ZIP",
            type=["zip"],
            key="test_zip",
            help="현장에서 촬영/캡처한 검증 대상 이미지 묶음",
        )
    with c_ref:
        ref_zip = st.file_uploader(
            "정답 이미지 ZIP",
            type=["zip"],
            key="ref_zip",
            help="원본/기준이 되는 이미지 묶음",
        )

    run_zip = st.button(
        "검증 실행",
        type="primary",
        key="run_zip",
        disabled=(test_zip is None or ref_zip is None),
    )

    if run_zip and test_zip is not None and ref_zip is not None:
        # ------- load zips -------
        with st.spinner("알집 분석 중..."):
            try:
                test_imgs = load_zip_images(test_zip.getvalue())
                ref_imgs = load_zip_images(ref_zip.getvalue())
            except Exception as e:
                st.error(f"알집 처리 실패: {e}")
                st.stop()

        pair = pair_by_stem(test_imgs, ref_imgs)
        n_matched = len(pair.matched)

        st.markdown("### 매칭 결과")
        tile_cols = st.columns(4)
        tile_cols[0].markdown(
            _metric_tile("테스트 이미지", str(len(test_imgs))),
            unsafe_allow_html=True,
        )
        tile_cols[1].markdown(
            _metric_tile("정답 이미지", str(len(ref_imgs))),
            unsafe_allow_html=True,
        )
        tile_cols[2].markdown(
            _metric_tile("매칭된 쌍", str(n_matched)),
            unsafe_allow_html=True,
        )
        unpaired = len(pair.only_test) + len(pair.only_ref)
        tile_cols[3].markdown(
            _metric_tile("미매칭", str(unpaired)),
            unsafe_allow_html=True,
        )

        if unpaired:
            with st.expander(f"미매칭 파일 보기 ({unpaired})"):
                if pair.only_test:
                    st.markdown(f"**테스트 ZIP에만 있음 ({len(pair.only_test)})**")
                    st.code("\n".join(z.display_name for z in pair.only_test))
                if pair.only_ref:
                    st.markdown(f"**정답 ZIP에만 있음 ({len(pair.only_ref)})**")
                    st.code("\n".join(z.display_name for z in pair.only_ref))

        if n_matched == 0:
            st.warning(
                "매칭된 파일이 없습니다. 두 알집의 파일 이름(확장자 제외)이 "
                "동일해야 합니다."
            )
            st.stop()

        # ------- run OCR on every pair -------
        st.markdown("### OCR 실행 중...")
        progress = st.progress(0.0, text=f"0 / {n_matched}")
        live_metric = st.empty()

        rows: List[ReportRow] = []
        t_start = time.perf_counter()
        for i, (t_zi, r_zi) in enumerate(pair.matched, start=1):
            try:
                t_res = ocr_image(t_zi.image, rec_model_path, rec_keys_path, det_model_path)
                r_res = ocr_image(r_zi.image, rec_model_path, rec_keys_path, det_model_path)
            except Exception as e:
                st.error(f"{t_zi.display_name}: OCR 실패 — {e}")
                continue

            c_v = cer(r_res.text, t_res.text)
            w_v = wer(r_res.text, t_res.text)
            v = verdict(c_v)
            rows.append(
                ReportRow(
                    filename=t_zi.display_name,
                    test_image=t_zi.image,
                    ref_image=r_zi.image,
                    test_text=t_res.text,
                    ref_text=r_res.text,
                    cer_v=c_v,
                    wer_v=w_v,
                    verdict=v,
                )
            )
            progress.progress(i / n_matched, text=f"{i} / {n_matched}")
            elapsed = time.perf_counter() - t_start
            live_metric.caption(
                f"진행 {i}/{n_matched} · 경과 {elapsed:.1f}s · "
                f"평균 {elapsed * 1000 / i:.0f} ms/쌍"
            )
        progress.empty()

        if not rows:
            st.error("처리된 결과가 없습니다.")
            st.stop()

        # save into session so the user can re-filter and re-download without
        # re-running OCR.
        st.session_state["zip_rows"] = rows
        st.session_state["zip_only_test"] = [z.display_name for z in pair.only_test]
        st.session_state["zip_only_ref"] = [z.display_name for z in pair.only_ref]

    # ----------------- render results (from session) ----------------- #
    rows: List[ReportRow] = st.session_state.get("zip_rows", [])
    only_test_names: List[str] = st.session_state.get("zip_only_test", [])
    only_ref_names: List[str] = st.session_state.get("zip_only_ref", [])

    if rows:
        n = len(rows)
        n_pass = sum(1 for r in rows if r.verdict == "PASS")
        n_warn = sum(1 for r in rows if r.verdict == "WARN")
        n_fail = sum(1 for r in rows if r.verdict == "FAIL")
        mean_cer = sum(r.cer_v for r in rows) / n
        mean_wer = sum(r.wer_v for r in rows) / n

        st.markdown("---")
        st.markdown("### 종합 결과")
        cols = st.columns(6)
        cols[0].markdown(_metric_tile("검증 쌍", str(n)), unsafe_allow_html=True)
        cols[1].markdown(
            _metric_tile("PASS", f"{n_pass} ({n_pass/n*100:.1f}%)"),
            unsafe_allow_html=True,
        )
        cols[2].markdown(
            _metric_tile("WARN", f"{n_warn} ({n_warn/n*100:.1f}%)"),
            unsafe_allow_html=True,
        )
        cols[3].markdown(
            _metric_tile("FAIL", f"{n_fail} ({n_fail/n*100:.1f}%)"),
            unsafe_allow_html=True,
        )
        cols[4].markdown(
            _metric_tile("평균 CER", f"{mean_cer:.4f}"),
            unsafe_allow_html=True,
        )
        cols[5].markdown(
            _metric_tile("평균 WER", f"{mean_wer:.4f}"),
            unsafe_allow_html=True,
        )

        # ------- excel download -------
        with st.spinner("엑셀 보고서 생성 중..."):
            wb = build_workbook(rows, only_test_names, only_ref_names)
            xlsx_bytes = workbook_to_bytes(wb)
        ts = time.strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="엑셀 보고서 다운로드 (.xlsx)",
            data=xlsx_bytes,
            file_name=f"ocr_validation_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )

        # ------- filter + per-row detail -------
        st.markdown("### 상세 결과")
        c_filter, c_sort = st.columns([1, 1])
        with c_filter:
            f_choice = st.multiselect(
                "판정 필터", ["PASS", "WARN", "FAIL"],
                default=["WARN", "FAIL"],
                help="기본은 문제 있는 항목(WARN/FAIL)만 표시합니다.",
            )
        with c_sort:
            sort_choice = st.selectbox(
                "정렬",
                ["CER 높은 순", "CER 낮은 순", "파일명 순"],
                index=0,
            )

        filtered = [r for r in rows if r.verdict in f_choice] if f_choice else rows
        if sort_choice == "CER 높은 순":
            filtered.sort(key=lambda r: -r.cer_v)
        elif sort_choice == "CER 낮은 순":
            filtered.sort(key=lambda r: r.cer_v)
        else:
            filtered.sort(key=lambda r: r.filename.lower())

        st.caption(f"{len(filtered)} / {n} 표시")

        for idx, r in enumerate(filtered, start=1):
            with st.expander(
                f"[{r.verdict}] {r.filename}   ·   CER {r.cer_v:.3f}   ·   WER {r.wer_v:.3f}",
                expanded=(idx <= 3 and r.verdict != "PASS"),
            ):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**테스트 이미지**")
                    st.image(r.test_image, use_container_width=True)
                    st.markdown("**테스트 OCR**")
                    st.markdown(
                        f"<div class='ocr-text'>{r.test_text or '<i>(없음)</i>'}</div>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown("**정답 이미지**")
                    st.image(r.ref_image, use_container_width=True)
                    st.markdown("**정답 OCR**")
                    st.markdown(
                        f"<div class='ocr-text'>{r.ref_text or '<i>(없음)</i>'}</div>",
                        unsafe_allow_html=True,
                    )
