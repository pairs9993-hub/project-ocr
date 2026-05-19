# OCR Validation App

가전 UI 화면 OCR 결과를 검증하는 데스크톱 도구. 브라우저 기반 GUI(Streamlit).

## 기능

- **일반 테스트** — 이미지 한 장을 OCR하고 (선택적으로) 정답 텍스트와 비교
- **알집 테스트 (ZIP)** — 테스트 이미지 ZIP과 정답 이미지 ZIP을 업로드하면
  파일명(확장자 제외)으로 자동 매칭 → 양쪽 모두 OCR → CER/WER 계산 →
  **엑셀 보고서 다운로드** (테스트/정답 이미지 썸네일 + 추출 텍스트 + 판정 색상 포함)

판정 기준 — `PASS` CER ≤ 5%, `WARN` 5–20%, `FAIL` > 20%

## 처음 한 번 설치

1. Python 3.10 또는 3.11 (64-bit)을 [python.org](https://www.python.org/downloads/) 에서 설치.
   설치 화면에서 **Add Python to PATH** 체크.
2. 이 폴더(`app/`) 전체를 회사 PC로 복사.
3. `run.bat`을 더블클릭. 처음 실행 시 자동으로 `.venv_app` 가상환경을 만들고
   필요한 패키지를 설치합니다(최초 5~10분 소요, 인터넷 연결 필요).
4. 설치가 끝나면 브라우저가 자동으로 열리고 앱이 표시됩니다
   (http://localhost:8501).

## 일반 사용

이후에는 `run.bat`만 더블클릭하면 바로 앱이 뜹니다.

종료: 콘솔 창을 닫거나 `Ctrl+C`.

## 사용자 정의 모델 (선택)

`app/models/` 폴더에 다음 두 파일이 함께 있으면 **자동으로 사용**됩니다:

- `rec_v0.onnx` — 자체 학습한 PaddleOCR rec 모델 (ONNX)
- `ppocr_keys.txt` — 문자 사전

사이드바의 **"자체 학습 모델 사용"** 토글로 켜고 끌 수 있습니다 (기본 ON).
끄면 RapidOCR 사전학습 모델로 baseline 비교가 가능합니다.

별도 경로를 지정하고 싶다면 사이드바 → "고급: 다른 ONNX 경로 지정"에서 입력.

## 파일 명명 규칙 (ZIP 모드)

테스트 ZIP 안의 `screen_001.png` 는 정답 ZIP 안의 `screen_001.png`
또는 `screen_001.jpg` 등과 매칭됩니다 (대소문자 무시, 확장자 무시,
폴더 구조 무시 — leaf 파일 이름만 사용).

지원 확장자: `.png .jpg .jpeg .bmp .webp .tif .tiff`

## 트러블슈팅

- **`py -3` not found** — Python을 설치할 때 "Add Python to PATH" 옵션을
  체크하지 않았기 때문입니다. Python을 재설치하거나, `run.bat` 의
  `py` 부분을 설치된 Python 절대경로로 수정.
- **패키지 설치 실패** — 사내 방화벽 환경이면 `pip` 의 프록시 설정이
  필요할 수 있습니다. 시스템 관리자에게 문의.
- **OCR 결과가 비어있음** — 이미지 해상도가 너무 낮거나(짧은 변 < 32px)
  텍스트 영역이 너무 작을 수 있습니다.
