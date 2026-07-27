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
