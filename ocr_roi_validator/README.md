# Vesta Excel 자동화 검증

기존 `run.bat` 실행 후 **자동화 검증** 버튼에서 Vesta ZIP과 Excel TC를 첨부할 수 있습니다.
Excel `ROI` 열의 이름으로 프리셋을 선택하고 기존 Horizontal/Vertical OCR 검증을 수행합니다.
기존 프리셋은 최초 한 번 **화면 기준 등록**으로 Vesta 창 상대 좌표를 저장해야 합니다.
Excel 양식: [최신 TC 템플릿](examples/vesta_automation_tc_template_v2.xlsx)
및 [결과서 예시](examples/vesta_result_report_example.xlsx).
ROI별 조기 판정/최대 30초 제한/자동 Excel 내보내기와 좌표 이식 방법은 [사용 안내](docs/VESTA_AUTOMATION.md)를 참고하세요.

# ROI OCR Validator (Standalone)

학습된 OCR 엔진(공용 detector + 언어별 recognizer)을 사용해 ROI 영역 OCR 결과를 기대 문자열과 비교하는 독립 실행 프로그램입니다.

기본 실행 백엔드는 PaddleOCR이며, 필요 시 RapidOCR로 전환할 수 있습니다.

## 핵심 기능

- Windows CPU only 동작
- 입력: PNG/JPG
- ROI 지정: 마우스 드래그
  - 단일 ROI
  - 다중 ROI
- 인식 결과 비교
  - Exact Match (기본)
  - Ignore Case
  - Ignore Space
  - Similarity Match
  - Regex Match
  - Expected 이미지 토큰: `{0:img_start}` → `▶Ⅱ`, `{0:img_check}` → `✓`
  - 프랑스어/스페인어 악센트 문자는 Unicode NFC로 정규화하되 서로 다른 문자는 구분
- 반복 캡처 OCR
  - 기본 2 FPS
  - 캡처 지속 시간 지정
  - 프레임 저장 여부 선택
- 모델 패키지 단일 관리
  - detector
  - recognizers(en_es, fr, zh)
  - dictionary
  - preprocess/config

## 프로젝트 구조

- main.py
- ocr_roi_validator/
  - app.py
  - gui.py
  - ocr_engine.py
  - model_package.py
  - compare.py
  - capture.py
- scripts/build_model_package.py
- models_package_example/manifest.json

## 빠른 시작

1) 의존성 설치 및 실행

- run.bat 실행

또는

- py -3 -m venv .venv
- .venv\\Scripts\\activate
- pip install -r requirements.txt
- python main.py --backend paddle

모델 패키지가 준비되지 않은 경우에도 실행 시 자동으로 RapidOCR 기본 모델로 fallback됩니다.

참고: PaddleOCR는 Windows에서 Python 3.10 환경이 가장 안정적입니다.

학습 모델을 Paddle 런타임에서 직접 사용하려면 Paddle inference 디렉토리(`inference.pdmodel`, `inference.pdiparams`)를 패키징해 `--paddle-package`로 지정하세요.

2) 실제 모델 패키지 생성

예시:

python scripts/build_model_package.py ^
  --output artifacts/my_ocr_package ^
  --detector ..\\artifacts\\models\\real_ui_company_pseudo_rec\\det.onnx ^
  --rec-en-es ..\\artifacts\\models\\real_ui_company_pseudo_rec\\rec.onnx ^
  --rec-fr ..\\artifacts\\models\\real_ui_fr_rec_v2_hard\\rec.onnx ^
  --rec-zh ..\\artifacts\\models\\real_ui_zh_2m_rec\\rec.onnx ^
  --dict ..\\artifacts\\models\\real_ui_company_pseudo_rec\\ppocr_keys.txt

3) 패키지 지정 실행

python main.py --backend rapid --model-package artifacts/my_ocr_package

4) RapidOCR 기본 모델로 실행

python main.py --backend rapid --rapid-default

5) Paddle 학습 모델 패키지 생성 및 실행

패키지 생성 예시:

python scripts/build_paddle_model_package.py ^
  --output artifacts/my_paddle_package ^
  --det-infer ..\\PaddleOCR\\output\\det_real_ui_1m_infer ^
  --rec-en-es-infer ..\\PaddleOCR\\output\\real_ui_company_pseudo_rec\\inference ^
  --rec-fr-infer ..\\PaddleOCR\\output\\real_ui_fr_rec_v2_hard_infer ^
  --dict ..\\artifacts\\models\\real_ui_company_pseudo_rec\\ppocr_keys.txt

실행:

python main.py --backend paddle --paddle-package artifacts/my_paddle_package

참고:
- `--paddle-package`를 지정하지 않으면 PaddleOCR 기본 사전학습 모델을 사용합니다.
- 패키지에 없는 언어 선택 시 해당 언어는 Paddle 기본 모델로 fallback됩니다.

## 실제 제품 화면 비교

- OCR이 띄어쓰기를 불안정하게 인식하면 Compare를 `ignore_space`로 선택합니다. 공백만 무시하며 `é/e`, `ñ/n` 같은 악센트 차이는 그대로 실패 처리합니다.
- 화면에 고정된 여러 줄 안내문은 Expected Text에 줄바꿈해서 입력하고 `Scrolling / Loop`를 끕니다. 이 경우 `static-multiline`으로 판정하며 반복 루프를 요구하지 않습니다.
- 가로로 흐르는 문자열은 `Scrolling / Loop`만 켭니다. 여러 프레임을 누적하고 한 바퀴 순환을 검증합니다.
- 위로 흐르는 행 목록은 `Vertical Rows`를 켭니다. `Scrolling / Loop`가 자동으로 켜지며 각 행의 내용, 순서, 한 바퀴 순환을 함께 검증합니다.
- Expected의 모든 행과 문자가 일치하면 `CONTENT=PASS`가 즉시 확정되고 OCR 검증이 자동으로 멈춥니다. 별도 Auto Stop 설정은 없습니다. Live 화면 캡처는 유지되므로 새 검증은 `Start OCR`로 다시 시작할 수 있습니다. 스크롤을 켠 경우 한 바퀴 전에는 `LOOP=PENDING`, 순환이 확인되면 `LOOP=PASS`로 별도 표시됩니다.
- Expected Text에서는 시작키를 `{0:img_start}`, 체크 표시를 `{0:img_check}`로 입력할 수 있습니다. 중괄호 앞의 숫자는 어떤 값이어도 됩니다.
- 조합형과 분해형 Unicode 악센트는 같은 글자로 처리합니다. 실제로 악센트를 누락하거나 다른 악센트로 읽은 경우는 OCR 오류로 유지됩니다.
- 작은 문자와 악센트가 잘 안 읽히면 `Auto Upscale`을 켭니다. 주변 문맥까지 detector가 필요할 때만 `Context Detect`를 켭니다.

## CPU 성능

- Live FPS 기본값은 2입니다. 회사 노트북에서는 1~2 FPS를 권장합니다.
- `Fast Long ROI`는 기본 ON입니다. 가로 또는 세로로 매우 긴 ROI를 패딩해 detector 입력이 과도하게 커지는 것을 막습니다. 긴 스크롤 영역에서 인식 결과가 달라지는 경우에만 끄세요.
- `Context Detect`는 기본 OFF입니다. 켜면 ROI보다 큰 영역을 OCR하므로 CPU 사용량이 증가합니다.
- 커스텀 Rapid 모델은 기본적으로 detector를 한 번만 실행합니다. 검출 누락이 심한 경우에만 `python main.py --backend rapid --model-package artifacts/my_rapid_package --thorough-detection`으로 두 번째 detector pass를 활성화합니다.
- 빠른 모드에서도 direct ROI 검출이 비면 detector fallback과 넓은 context ROI를 자동으로 한 번씩 시도합니다. 정상 검출 프레임에는 추가 비용이 없습니다.
- `run.bat`은 가상환경을 처음 만들 때만 패키지를 설치합니다. 의존성을 갱신해야 할 때는 직접 `pip install -r requirements.txt`를 실행합니다.

Live 로그가 계속 `…`만 표시되던 경우에는 이제 `RAW=<text> [score=..., boxes=...]`를 출력합니다. `RAW=<no text>`이면 ROI 또는 detector 문제이고, 텍스트가 있지만 coverage가 0%이면 Expected Text, 언어 선택, 또는 OCR 문자 인식 문제입니다.

## 사용 흐름

1. Load Image 또는 Capture Screen Area
2. 캔버스에서 마우스 드래그로 ROI 추가
3. ROI 목록에서 각 ROI 선택 후 Expected Text 입력
4. Language(en_es/fr/zh), Compare 모드 선택
5. Run Once 또는 Run Timed Capture 실행

## 모델 패키지 규격

manifest.json 필수 필드:

- detector_model
- dictionary
- recognizers.en_es
- recognizers.fr
- recognizers.zh
- preprocess
  - det_limit_type
  - det_limit_side_len
  - det_mean
  - det_std
  - det_box_thresh
  - det_unclip_ratio
  - det_donot_use_dilation
  - use_cls

이 규격을 통해 다른 시스템으로 이식 시 동일한 전처리/추론 조건으로 재현 가능합니다.

## 별도 Git 프로젝트로 분리

현재 폴더를 별도 저장소로 사용하면 됩니다.

- cd ocr_roi_validator
- git init
- git add .
- git commit -m "Initial standalone ROI OCR validator"

필요 시 이 폴더만 압축/배포하여 독립적으로 설치 가능합니다.

## 이름으로 ROI 묶음 저장하고 Excel에서 적용하기

1. `Capture Screen Area` 또는 `Load Image`로 기준 화면을 준비하고 ROI 1, 2, 3을 그립니다.
2. 각 ROI에 기대 문구를 입력합니다. `Vertical Rows`는 세로 목록 검증,
   체크 해제는 horizontal입니다. 가로 스크롤은 `Scrolling / Loop`도 체크합니다.
3. `ROI preset` 입력란에 `roi_courseop`를 입력하고 `Save Preset`을 누릅니다.
   ROI 좌표, 기대 문구, 기준 이미지 크기, horizontal/vertical 방향과 스크롤 설정이
   프로젝트의 `roi_presets.json`에 저장됩니다. 앱을 다시 열면 자동으로 불러옵니다.
   같은 이름을 저장할 때는 덮어쓰기를 확인합니다.
4. Excel의 첫 행에 열 이름을 작성하고, 사용할 행에 프리셋 이름을 넣습니다.

   | case_id | roi |
   | --- | --- |
   | case_001 | roi_courseop |
   | case_002 | roi_courseop |
   | case_003 | roi_other_screen |

5. `Load Excel`에서 `.xlsx` 또는 `.xlsm` 파일을 엽니다. 시트와 ROI 이름 열을
   선택합니다. `roi` 헤더는 기본 선택되며, 다른 이름의 열도 선택할 수 있습니다.
   첫 행은 헤더로 사용하며 원본 Excel 파일은 변경하지 않습니다.
6. Excel 목록에서 행을 선택하면 그 이름의 ROI 묶음과 방향이 즉시 적용됩니다.
   실제 대상 화면을 준비한 뒤 `Apply + Verify Selected Row`를 누릅니다.
   일반 화면은 최신 캡처로 `Run Once`, 스크롤/세로 목록은 설정한 Duration 동안
   `Run Timed Capture`를 실행합니다. 스크롤 대상은 사용자가 직접 움직입니다.
   이미지 파일만 로드한 상태에서는 스크롤 검증을 실행할 수 없습니다.
   여러 행을 자동 순회하거나 대상 앱의 화면을 자동 전환하지는 않습니다.

### 다른 사람에게 공유하기

`Export JSON`으로 모든 프리셋을 하나의 JSON 파일로 내보냅니다.
받는 사람은 `Import JSON`으로 기존 프리셋에 추가한 뒤 같은 Excel을 열면 됩니다.
동일한 이름이 있으면 덮어쓰기를 확인하고, 취소하면 가져오기 전체를 취소합니다.
JSON에는 캡처 이미지나 PC별 절대 화면 위치를 넣지 않습니다.

좌표는 **캡처 영역 내부의 픽셀 좌표**입니다. 받는 사람은 동일한 크기와 내용 배치의
영역을 캡처해야 합니다. 화면 위치가 달라도 사용할 수 있지만, 배율이나 해상도가 달라
기준 크기가 달라지면 적용을 막습니다. 크기가 같아도 UI 배치가 다르면 ROI를 다시 설정해야 합니다.
알 수 없는 이름·빈 셀은 이전 ROI로 검증하지 않고 오류를 표시합니다.
검증 중에는 ROI 전환을 막으므로 Live를 중지하거나 Timed Capture가 끝난 뒤 행을 전환합니다.

JSON 구조 예시는 [examples/roi_presets.json](examples/roi_presets.json)에 있습니다.
프리셋 이름은 대소문자를 구분하며 Excel 셀의 앞뒤 공백은 제거합니다.
수식 셀은 Excel이 마지막으로 저장한 계산 결과를 사용하므로, 수식으로 이름을 만든다면
Excel에서 계산 후 저장하세요. 기대 문구는 프리셋에 저장된 값을 사용합니다.
언어, 비교 방식, FPS, Duration 등의 나머지 검증 옵션은 현재 앱 설정을 사용합니다.

기존 가상환경으로 직접 실행할 때는 `python -m pip install -r requirements.txt`로
`openpyxl`을 설치하세요. `run.bat`은 Excel 지원 패키지가 없으면 설치합니다.
