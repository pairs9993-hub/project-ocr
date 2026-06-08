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
