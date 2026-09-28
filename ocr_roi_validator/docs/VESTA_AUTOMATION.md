# Vesta + Excel 자동화 검증

## 시작

기존 `run.bat`으로 실행한 뒤 **자동화 검증** 버튼을 누릅니다.

1. 신뢰할 수 있는 **Vesta ZIP**과 **TC Excel (.xlsx/.xlsm)**을 선택합니다.
   알집으로 압축한 ZIP도 가능합니다. `.alz`, `.egg`, 비밀번호 ZIP은 지원하지 않습니다.
2. Product는 ZIP의 `sim_config/<Product>.cfg` 또는 `.json` 파일명입니다. 하나면 자동 선택되고,
   여러 개면 목록에서 선택합니다. LITE 경로 기본값은 옆의 `lite` 폴더입니다.
3. 기존 프리셋은 아래의 **화면 기준 등록**을 한 번 수행합니다.
4. **자동화 검증 시작**을 누릅니다. 기준이 이미 등록되었으면 Vesta 실행 버튼을 따로
   누르지 않아도 ZIP 해제 → Vesta 실행 → 엑셀 행별 명령 → 대기 → ROI 검증을 실행합니다.
5. 완료/오류/중지 시 이번 세션이 시작한 Vesta를 종료합니다. 수동 실행한 다른 Vesta는
   종료하지 않습니다. 같은 디버그 포트를 사용하는 기존 Vesta가 있으면 시작을 거부합니다.

## 기존 roi_courseop의 최초 1회 화면 기준 등록

기존 프리셋의 `image_size=[320,239]`와 ROI 좌표는 **과거에 드래그한 화면 내부**의 좌표입니다.
그 화면이 Vesta 창에서 어디였는지, 어떤 창이었는지는 과거 JSON에 없으므로 자동 복원할 수 없습니다.

1. **① Vesta 실행**을 누릅니다.
2. 기준 등록 프리셋에서 `roi_courseop`를 선택하고 **화면 기준 등록**을 누릅니다.
3. 별도로 열린 **Vesta 화면 미리보기 창**에서 과거 Capture Screen Area로 선택했던
   **GUI 화면 전체를 같은 경계로** 드래그하고 **선택 확정**을 누릅니다. 다중 모니터 전체를 덮는 화면 선택창은 사용하지 않습니다.
   창을 최대화하거나 `100% / 200% / 400%`, 마우스 휠로 확대할 수 있습니다.
   오른쪽 드래그 또는 스크롤바로 이동하며, 화면 X/Y·너비/높이를 직접 입력해 1px 단위로 보정할 수 있습니다.
   드래그를 놓아도 창이 닫히지 않으므로 확인 후 Enter 또는 선택 확정으로 완료하세요. Esc는 취소입니다.
   개별 ROI나 바탕화면/타이틀바를 선택하는 것이 아닙니다.
4. 메인 창의 미리보기에서 기존 ROI들이 정확한 위치에 있는지 확인하고 저장합니다.
5. 다른 캡처 경계를 사용하는 프리셋은 각각 등록합니다. 엑셀 TC를 실행하거나 ROI를
   다시 그릴 필요는 없습니다.

JSON에 `capture_anchor`가 추가됩니다. HWND/PID/바탕화면 절대좌표는 저장하지 않습니다.
실행한 Vesta PID의 창 클래스와 제목으로 대상 클라이언트를 찾고, 클라이언트 기준
비율 좌표로 캡처 영역을 계산합니다. 캡처마다 현재 위치를 다시 조회하며, 캡처 이미지를
프리셋의 원본 크기로 정규화한 후 기존 ROI를 그대로 적용합니다.

- 창 이동, 음수 좌표 모니터, 같은 레이아웃의 비례 DPI 확대에 대응합니다.
- 다른 PC에는 **Export JSON → Import JSON**으로 프리셋과 기준을 함께 전달합니다.
- 해상도/창 크기를 바꿨을 때 **비례 확대가 아니라 내부 GUI 배치가 변경되는 경우**,
  도구 모음/패널 배치가 달라지는 경우, 서로 다른 제품/스킨은 같은 좌표를 보장할 수 없습니다.
  원래 레이아웃으로 복원하거나 기준을 다시 등록하세요. 종횡비가 2% 이상 달라지면 오류로 중단합니다.
- Vesta는 최소화하지 말고 화면 안에 유지하세요. 화면 캡처 방식이므로 다른 창으로 가리면 안 됩니다.
  캡처의 9개 지점에서 다른 프로세스에 가려졌는지 검사합니다. 아주 작은/투명한 오버레이까지
  완전히 검출하는 방식은 아니므로 Vesta 화면 전체가 보여야 합니다.
- 창 후보가 여러 개라 정확하게 식별할 수 없으면 임의의 창을 골라 검증하지 않고 오류로 중단합니다.
- 정답 텍스트 편집 후 프리셋을 덮어쓸 때 기존 화면 기준을 유지할지 확인합니다.

## Excel 형식

바로 편집할 최신 양식: [vesta_automation_tc_template_v2.xlsx](../examples/vesta_automation_tc_template_v2.xlsx).
`tc_template`에는 3개 ROI 정답과 4개 ROI/여러 줄 정답의 예시가 있고, `사용안내` 시트가 포함됩니다.
예시 행은 모두 `Execution=N`입니다. 실제 정답/명령/프리셋으로 바꾸고 준비된 행만 `Y`로 변경하세요.

### 권장: Expected_Text 한 셀에 모든 ROI 정답

`ROI` 열에는 프리셋 이름을, `Expected_Text` 열의 한 셀에는 다음과 같이 입력합니다.
Excel 셀 내부 줄바꿈은 **Alt+Enter**입니다.

```text
[ROI1]
첫 번째 영역의 정답

[ROI2]
두 번째 영역의 첫째 줄
두 번째 영역의 둘째 줄

[ROI3]
세 번째 영역의 정답
```

- `[ROI번호]`는 **저장된 ROI ID**이며 독립된 줄로 입력합니다. 블록 순서는 자유입니다.
- 다음 마커 전까지 해당 ROI 정답입니다. 내부 줄바꿈/공백은 보존하고 블록 앞뒤 빈 줄만 제거합니다.
- 4개 이상도 `[ROI4]`, `[ROI5]` 등을 추가하면 됩니다. 없는 ID/중복 ID/빈 정답/빠진 ROI는 사전 오류입니다.
- 이 형식은 프리셋의 **모든 ROI 정답을 Excel로 제공**해야 합니다. 누락된 ROI를 프리셋/Image로 자동 보충하지 않습니다.
- 동일 행에서 `Expected_Text`와 기존 `Expected_1` 등의 개별 열을 동시에 채우면 오류입니다.
  `Expected_Text`가 빈 행에서는 기존 정답 우선순위를 그대로 사용합니다.
- `[ROI`로 시작하는 줄은 구분자용으로 예약되어 있습니다. `roi1:정답`이나 `[ROI1]:정답` 형식은 지원하지 않습니다.
- 여러 줄을 입력해도 Horizontal/Vertical 방향은 바뀌지 않으며 프리셋 설정을 따릅니다.

### 기존 개별 정답 열 방식 (계속 지원)

`tc_`로 시작하는 시트를 모두 순서대로 처리합니다. 없다면 Commands/ROI 헤더가 있는 일반 시트를
사용합니다. 제목/공백 행 다음에 헤더가 있어도 됩니다. ROI 열 이름은 대화상자에서 지정 가능합니다.

| No. | Title | Execution | Commands | Capture_waiting_time | ROI | Expected_1 | Expected_2 | Expected_3 | Image |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 코스 옵션 | Y | 실제 LITE 명령 | 1 | roi_courseop | 첫 번째 정답 | 두 번째 정답 | 세 번째 정답 | |

- Commands 셀의 줄바꿈(Alt+Enter)으로 명령 여러 개를 입력하면 위에서 아래로 순서대로 실행합니다. 비어 있으면 현재 화면을 검증합니다.
- 연속된 명령 사이에는 앞 명령 처리 완료 후 기본 1초를 기다립니다. 예: A → 1초 → B → 1초 → C.
  `:API System.sleep { time: 2 }`를 사이에 넣으면 해당 구간은 지정한 2초만 기다립니다.
  마지막 명령 뒤에는 기본 1초를 추가하지 않고 `Capture_waiting_time`을 적용합니다.
- Execution이 N/NO/FALSE/0이면 건너뜁니다. 나머지는 실행합니다.
- `Capture_waiting_time`: 모든 명령 실행 후 화면이 안정될 때까지 대기할 초. 기본 0.
- `ROI`: 프리셋 이름을 정확히 입력합니다. 빈 이름/없는 프리셋은 사전 검사 오류입니다.
- `Expected_1`의 숫자는 ROI 순번이 아닌 **저장된 ROI ID**입니다. `Expected_ROI_1`도 가능합니다.
  셀 내부 줄바꿈은 Vertical 검증의 정답 행으로 보존됩니다.
- 정답 우선순위: **Excel Expected_<ID> → 프리셋 expected → Image의 ROI OCR**.
- `Image`: 정답 이미지 폴더 기준 상대 경로 또는 유일한 파일명입니다. 폴더를 지정하지 않으면
  Excel이 있는 폴더에서 찾습니다. 같은 이름이 여럿이면 상대 경로를 입력합니다.
  이미지는 프리셋과 동일 크기의 GUI 화면이어야 합니다. 전체 데스크톱/다른 크기를 임의로 맞추지 않습니다.
- 모든 ROI에 정답이 있어야 합니다. 현재 저장된 `roi_courseop`에는 정답 3개가 모두 비어 있으므로
  Excel 정답 열을 채우거나 프리셋에 정답을 저장하거나 Image를 제공해야 합니다.
  빈 정답을 N/A 또는 PASS로 처리하지 않습니다.
- 이미지에서 얻는 정답은 OCR 결과이므로 인식 오류가 있을 수 있습니다. 특히 전체 문구가 한 프레임에
  나오지 않는 스크롤/Vertical 목록은 **전체 정답 텍스트를 Excel에 직접 입력**하는 것을 권장합니다.

## LITE 재사용 범위

LITE의 실제 `src/sdk/ldb.py`를 별도 Python 프로세스에서 import하여 통신합니다.
LITE GUI, pytest/tox 실행기, 전역 config.ini, 임시 폴더 삭제/서버 업로드 기능은 가져오지 않습니다.
Vesta 실행은 동일한 `vesta.exe -t <Product>` 방식이며 공백/한글 경로를 지원합니다.

지원하는 Commands:
- 일반 LDB shell 명령
- `:TBL key` → `tbl_*` 시트의 Data/Type/Label/Packet 4열 조회
- `:TBL {"Label":"UART1","Packet":"@..."}` → `sim_uart write`
- Label 없는 JSON Packet → `sim_irrc write`
- `:API System.sleep { time: 1 }`

LMOS/LCP 외부 시뮬레이터 API, reboot 생명주기, LDB_INSTALL/PUSH/PULL 등은 이 화면에서는
지원하지 않습니다. 이런 명령은 **실행 전에 오류**로 알려 주며 건너뛰거나 성공으로 처리하지 않습니다.
LDB 전송은 명령당 최대 30초로 제한합니다. LDB 프로토콜 성공이 shell 명령의 의미상 성공을
보장하지는 않으므로 최종 화면 검증 결과를 확인하세요. TC는 서로 독립적이어야 하며 각 행의 명령이
자기 시작 상태를 만들어야 합니다. 행 오류는 기록하고 다음 행으로 진행합니다.

## 검증과 결과

- OCR 언어, 비교 모드, ROI 전처리 설정은 시작 시 메인 창 값으로 고정합니다.
- Horizontal/Vertical과 Scrolling 여부는 행마다 해당 프리셋에서 복원합니다.
- content 모드에서는 기존 ScrollTextAccumulator/VerticalListAccumulator의 순서·범위·행 배치 판정을 재사용합니다.
  Horizontal은 모든 문자와 인접 문자 연결 관측·공백 판정, Vertical은 행 배치·내용 판정으로 PASS를 결정합니다.
  전체 내용이 한 프레임에 보이는 경우도 최소 2회, 0.5초 이상의 확인 관측 후 PASS입니다.
  정상 순환을 반드시 관찰하려면 아래 cycles 모드를 사용하세요.
- 이번 기능은 **ROI OCR 텍스트 검증**이며 LITE의 SSIM/PHASH 픽셀 유사도 판정을 병행하지 않습니다.
- 원본 Excel은 수정하지 않습니다.
- `captures/automation/<timestamp>/`에 Vesta 로그, 명령 로그, 결과 JSON, 행별 캡처,
  실패 ROI 진단을 저장합니다. 결과 JSON에는 시트/실제 Excel 행 번호/프리셋/정답/실제값/판정이 들어갑니다.
- 정적 OCR 실패는 실제 OCR 입력 이미지를 저장하고, 다중 프레임 누적 실패는 재구성 진단임을 구분합니다.
- 중지/창 닫기는 취소를 요청하고 별도 OCR 프로세스를 종료한 뒤 Excel을 자동 저장합니다.
  완료된 행과 완료된 ROI, 미완료 ROI의 관측 텍스트를 보존합니다.
- 검증 중 메인 창 ROI/수동 OCR/Live 작업은 잠급니다. 마지막 검증 화면과 ROI 결과를 완료 후 보여 줍니다.

## TC 전환 정책 (구현됨)

명령 실행 → `Capture_waiting_time` → ROI별 독립 관찰 → 모두 PASS면 조기 종료,
아니면 최대 시간 종료 → 결과 저장 → 다음 독립 TC.

| Excel 열 | 기본값 | 의미 |
|---|---|---|
| Max_observation_sec | 30 | 0보다 크고 30 이하. GUI 최대 관찰 시간과 이 값 중 작은 값을 사용 |
| Verification_mode | content | content 또는 cycles |
| Required_cycles | 1 | cycles 모드에서 필요한 정상 순환 수(양의 정수) |
| Verification_mode_1 | 빈 값 | ROI ID 1만 content/cycles로 덮어쓰기. 다른 ID도 같은 형식 지원 |

- **content:** 내용/순서 조건이 성립하고 관련된 텍스트를 0.5초 이상 간격에 걸쳐 최소 2회 확인하면 완료.
- **cycles:** 내용 조건에 더해 정상 순환 Required_cycles회가 필수. 단 한 번의 전체 정답 일치로 우회하지 않음.
- 완료된 ROI는 PASS를 고정하고 이후 OCR 대상에서 제외합니다. 모두 완료되면 즉시 TC를 종료합니다.
- 일부 ROI만 완료된 상태에서 최대 시간 종료: 완료 ROI=PASS, 나머지=FAIL_TIMEOUT, TC 합부=FAIL.
- 사용자 중단: 완료 ROI=PASS, 현재 미완료 ROI=CANCELLED, 남은 TC/ROI=NOT_RUN.
- FAIL_TIMEOUT/ERROR 후에도 다음 독립 TC를 진행합니다. 사용자 중지나 Vesta 초기 실행 실패는 전체 종료.
- **30초는 ROI 관찰 예산**입니다. 명령 실행·화면 전환 대기·별도 정답 이미지 OCR·최종 파일 저장은 별도입니다.
  OCR 엔진은 별도 프로세스에서 실행하며 관찰 마감/사용자 중지 시 종료합니다. 모델 최초 로딩도 관찰 예산에 포함됩니다.
  프로세스 정리/디스크 기록에 걸리는 시간만큼 다음 TC 진입은 30초보다 조금 늦어질 수 있습니다.
- `Animation` 열은 사용하지 않습니다. 각 TC마다 누적 문자열·주기·확인 상태는 새로 만듭니다.

### 주기를 보수적으로 판정하는 방법

같은 단어 재검출은 주기로 세지 않습니다. 정답 전체에서 **유일하게 위치를 식별할 수 있는 정확한 문자열 구간**을
추적하고, 구간이 정방향으로 겹쳐 이동하며 전체 문자를 관측한 뒤 시작 위치로 돌아온 경우만 한 주기로 셉니다.
문자열 공백/줄바꿈은 위치 추적용으로 정규화하지만 정답/검출 원문은 보고서에 보존합니다.

- 중간에서 시작한 부분 주기는 세지 않습니다. 두 주기를 요구하면 각 주기의 전체 내용 이동을 각각 확인합니다.
- 같은 프레임 또는 전체 정답을 반복 관측해도 이동/복귀가 없으면 cycles=0입니다.
- 같은 단어가 여러 위치에 있어 구별 불가, OCR 오독, 샘플 누락으로 구간 사이가 크게 건너뛴 경우
  진행 중 주기는 초기화합니다. 정상 완료했던 주기 수는 보존합니다.
- 완전히 같은 패턴의 반복이나 시작점을 샘플링하지 못하는 상황에서는 주기를 입증할 수 없어 시간초과가 날 수 있습니다.
  이는 잘못된 PASS 대신 보수적인 실패를 택한 구현입니다. 정적인 ROI는 content로 지정하세요.
- Vesta 창 위치/가림은 검사하지만 **화면 의미가 바뀌었는지 판별하는 별도 화면 식별 ROI 기능은 아직 없습니다.**
  따라서 한 TC의 관찰 중 화면이 다른 화면으로 전환되지 않도록 독립 TC 명령/대기를 구성해야 합니다.
  이 정책은 각 ROI가 한 번 검증 완료되면 고정하는 방식이며 동시 일치 보장은 아닙니다.

## 자동 Excel 결과서

결과 양식 예시: [vesta_result_report_example.xlsx](../examples/vesta_result_report_example.xlsx)
(예시임을 표시한 가상 결과이며 실제 시험 기록이 아닙니다.)

시작된 시험은 전체 완료, 사용자 중지, 실행 오류 종료 시 `.json` 옆에 `.xlsx`를 **자동 저장**합니다.
로그에 절대 경로를 표시합니다. 자동화 창/메인 창 닫기도 중지·저장을 마친 뒤 닫힙니다.
실행 전 Excel 사전 검사 오류는 아직 시험을 시작하지 않은 상태이므로 보고서를 생성하지 않습니다.

- **TC Summary:** 원본 No.(TC 번호), 시트/Excel 행, 제목, Image(정답지 이름), 명령어, 프리셋, 상태, 사유, 시간, PASS/FAIL 합부.
- **ROI Results:** 위 TC 정보 + ROI ID, ROI별 정답 텍스트, 실제 검출 텍스트, 상세 상태, 사유, 점수, 완료 주기, 모드, 합부.
- **Run Info:** 전체 실행 상태/시간, 원본 Excel/ZIP 경로, 상태 구분 설명.
- CANCELLED/NOT_RUN은 검증 미완료이므로 합부는 N/A입니다. PASS/FAIL_TIMEOUT/ERROR는 각각 PASS/FAIL/FAIL입니다.
- Image를 실제 정답 추출에 쓰지 않아도 값이 있으면 그대로 기록합니다. 빈 값도 허용합니다.
- 가로 스크롤 검출 텍스트는 **실제로 관측한 문자**를 정답 내의 유일한 위치에 정렬한 결과입니다.
  미확인 문자/연결은 `…`로 표시하며 정답 문자로 채우지 않습니다.
  `문자 관측률`, `공백 판정`, `프레임별 OCR 원문` 열을 함께 저장합니다. 전체 관측 원문과 정렬 정보는 JSON details에도 보존합니다.
  세로 스크롤 검출 텍스트는 기존과 같이 프레임별 OCR 원문을 기록합니다.
- Excel 셀 길이 32767자 초과 시 잘림 표시를 남기고 전체 내용은 JSON에 보존합니다.
  명령/정답/검출 텍스트는 수식으로 실행되지 않도록 문자열 셀로 저장합니다.
- 활성화된 TC의 미실행 행도 포함합니다. `Execution=N`으로 제외된 TC는 시험 대상에 포함하지 않습니다.
- 저장할 파일이 Excel에서 열려 있어 잠겼다면 다른 `_final_시간` 이름으로 저장합니다.
  디스크 쓰기 자체가 실패하면 오류를 표시합니다. OS 강제 종료/전원 차단까지 자동 Excel 저장을 보장하지는 않습니다.

## 개발 검증

테스트는 `tests/test_automation.py`와 기존 ROI/OCR/스크롤 테스트를 함께 실행합니다.
Windows 콘솔 기본 인코딩이 CP949인 환경에서는 기존 프랑스어 진단 테스트를 위해
`PYTHONUTF8=1`을 설정합니다.
## 가로 스크롤의 순서와 공백

정답의 중간부터 캡처를 시작해도 공백을 제외한 관측 글자들이 정답에서 유일하게
일치하는 위치를 찾습니다. 반복 단어처럼 여러 위치가 가능한 조각과 오인식 조각은
위치를 임의로 정하지 않고 보류합니다. 단어들을 따로 봤다는 사실만으로 순서를
확정하지 않으며, 인접 문자들이 같은 프레임에서 함께 관측되어야 합니다.
전체 정답을 계속 잘못 인식하면 이 방식으로도 복원할 수 없습니다.

단어 간 공백은 조각의 잘린 가장자리보다 안쪽 관측을 사용해 집계합니다.
공백이 있거나 있어야 하는 경계는 최소 2회의 관측 및 과반수로 판단합니다.
`SuciedadPesado`만 반복 관측했다면 문자 관측률은 100%여도 공백 판정은
`MISMATCH`이며 exact 모드에서 PASS가 되지 않습니다. 판정이 부족하면 `PENDING`입니다.
올바른 공백이 반복 관측되면 회복할 수 있지만, 화면에 실제로 공백이 있는지는
OCR 관측만으로 확정할 수 없습니다.

메인 창 Compare를 명시적으로 `ignore_space`로 설정한 경우에만 가로 스크롤의
공백 차이를 합부에서 제외합니다. 검출 문장과 공백 판정은 그대로 보존합니다.
`cycles`는 기존처럼 실제 순환 이동 증거를 별도로 요구합니다.

## 완전한 한 줄 문구의 이미지 공백 확인

OCR 결과가 `SuciedadPesado`, 정답이 `Suciedad Pesado`처럼 **공백을 제외한 글자가
정확하게 일치**할 때 자동으로 이미지 간격을 확인합니다. 하나의 OCR 박스에 담긴
완전한 라틴 문자 한 줄에 적용하며, 정적 문구가 주 대상입니다. 스크롤 중에도 전체 문구가
한 프레임에 보이는 경우에만 같은 조건으로 적용됩니다.

배경과 글자를 분리한 뒤 글자 세로 구간 수와 OCR 문자 수가 일치하는지 확인합니다.
주변의 보통 글자 간격보다 충분히 넓은 구간이 두 가지 밝기 기준에서 같은 위치에
검출될 때만 공백을 삽입합니다. 공백 위치는 정답의 위치가 아니라 이미지에서 결정합니다.
원래 OCR 공백을 삭제하거나 다른 글자를 정답으로 바꾸지 않습니다.

`ROI Results`의 **이미지 공백 판정** 열과 JSON `details.image_spacing`에 근거를 남깁니다.
- `IMAGE_GAP_CONFIRMED`: 이미지 간격으로 공백 위치를 확인해 비교에 사용함.
- `NO_WORD_GAP`: 뚜렷한 단어 간격이 없어 원문을 유지함.
- `UNCERTAIN`: 글자 구간 대응/간격이 불확실하여 원문을 유지함.

OCR 원문은 **프레임별 OCR 원문** 및 실패 진단의 `ocr_raw_output`에 보존합니다.
수동 Run Once는 Live Log에도 공백 확인 결과를 표시합니다. 작은 글씨, 붙은 글자,
복잡한 배경, 기울어진 글씨, 여러 줄 또는 여러 OCR 박스는 보정이 제한됩니다.
이는 보수적인 간격 추정이며 모든 글꼴에 대한 공백 인식을 보장하는 방식은 아닙니다.

글자끼리 붙어 세로 구간 수가 맞지 않는 경우에는, 두 밝기 기준에서 동일하게 확인된
넓은 단어 간격으로 이미지를 나누고 단어별 OCR을 추가 실행합니다(최대 4개 단어).
각 단어의 인식 신뢰도와, 다시 읽은 문자 전체가 원래 OCR 문자와 동일한지 확인한 뒤에만
공백을 삽입합니다. 이 경로의 실패 진단은 여러 입력을 사용했으므로 단일 exact 입력으로
표시하지 않습니다. 추가 OCR로 처리 시간이 늘 수 있으며 자동화의 기존 관찰 제한을 따릅니다.

`이미지 공백 상세` 열에 보정 경로 또는 보류 사유가 표시됩니다.
`GLYPH_COUNT_MISMATCH`는 글자 구간 대응 실패, `DIACRITIC_MISMATCH`는 공백 이외에
악센트 문자도 다르다는 의미입니다. `i`를 정답만 보고 `í`로 바꾸지 않습니다.
마침표 등 일반 문장부호가 포함된 라틴 문구도 공백 분석 대상으로 허용합니다.


### Static Start OCR and diacritic retries

With Scrolling / Loop and Vertical Rows both OFF, Start OCR now compares complete
stationary frames and displays the actual OCR text, including mismatches. It does
not reconstruct a scrolling sentence. Save the preset again to use these settings
in automation. Automatic live completion requires two matching observations at
least 0.5 seconds apart with confidence >= 0.25.

When a single detected text box differs from the expected text only in diacritics,
the shared OCR path retries the complete input at 2x and 3x with Lanczos scaling
and padding. Both OCR outputs must agree, have confidence >= 0.5, and retain the
original base letters, spacing, and punctuation. The expected string is never
substituted into the output. Disagreement preserves the original OCR result.
Inputs above one million pixels skip retries. Scaling cannot restore image detail
that was absent in the capture and may not fix a model's accent recognition.

Diagnostics retain raw_text and retry_outputs (text, score, scale), under
image_spacing with method diacritic_scaled_retry. Such captures are representative
of multiple OCR inputs, not a claim of one exact inference input. Retries can add
two OCR calls per applicable frame and share automation's observation deadline.

Manual check: compare Run Once and Start OCR with both scrolling toggles OFF,
expected `Temp. Agua fría`, and exact comparison. If OCR still reads `fria`, it
must remain visible and FAIL. Test `Centrifugado Baja` and `Suciedad Pesado` too.
Then enable horizontal scrolling and verify scrolling text still accumulates.


### Common screen reference for multiple ROI presets

In automation, select one representative preset, launch Vesta, then click
`공통 화면 기준 등록`. Select the complete GUI screen boundaries used when drawing
its ROIs and confirm the preview. All presets with the same `image_size`, including
ones saved later, use this shared reference. Their ROI coordinates and expected
text remain independent. They must have been drawn relative to the same screen
boundaries; equal dimensions alone do not establish that the layouts match.

The JSON library stores one `shared_capture` object. It takes precedence over old
individual anchors for matching sizes; individual anchors remain saved. Presets
with different sizes still use their own anchors and otherwise require registration.
Export JSON and Import JSON carry the common reference. Importing an older library
without a common reference preserves the current one; replacing a different shared
reference requires confirmation. Use the updated program on the receiving PC.


### Show observed text when horizontal verification is incomplete

The detected-text field now shows actual OCR observations when horizontal
verification has not passed, including observations rejected for alignment or
confidence. Distinct frames are separated by `--- frame ---`; they are not claimed
to form one sentence. Start OCR keeps the most recent 100 observations for this
display; automation retains observations in its report. Successful reconstruction
still displays the verified sentence. Excel adds a separate `스크롤 조합 텍스트`
column for the reconstruction, where ellipses indicate unconfirmed positions.
Original frame OCR remains in its existing column. This display fix does not
supply missing accents or superscript letters, or relax verification thresholds.


### Restore synchronous capture baseline (de22028)

The capture and recognition path is restored to de22028: capture one screen,
process its ROIs sequentially, then capture the next screen at the existing FPS
pacing. Background buffering, latest-frame replacement, the extra 30-second drain,
and the newly added independent word retry are removed. Original image-spacing
and diacritic retry behavior is retained exactly as in that baseline. Shared screen
references and raw observed-text display remain available.

Diagnostics add counts and relative monotonic timestamps without using them to
change sampling, comparison, or confirmation. JSON frame_timings and Excel's
OCR Timing sheet record every OCR attempt, including repeated identical outputs,
with frame number, capture completion time, OCR start/end time, raw/evaluated text,
and returned/error status. Captured/OCR counts include attempted work even if the
deadline expires. Zero dropped/pending/superseded frames reflect the absence of a
queue, not proof that every moment of a scrolling animation was sampled.

Compare the same TC, ROI presets, FPS, and observation duration before changing
recognition settings. This restores implementation behavior, not a guarantee of
recognition accuracy on a particular live screen.


### Per-frame alignment diagnosis and superscript TM

OCR Timing now includes Assembly accepted/reason, one-based Aligned start (in the
whitespace-free expected string), Character coverage, Spacing, Order confirmed,
and TM adjustment. ALIGNED means placement was accepted, not that the whole ROI
passed. NO_EXACT_ALIGNMENT means observed characters did not match a unique expected
window; AMBIGUOUS_POSITION means multiple placements; LOW_CONFIDENCE and
EMPTY_OR_TOO_SHORT are pre-alignment filters. DEADLINE_BEFORE_EVALUATION means OCR
returned after its processing budget. Static/vertical evaluation is labeled
separately. These diagnostics do not alter scroll matching or sampling.

A separate uppercase TM box is attached to the unique adjacent body box only when
its score and the body's score are >=0.5, its height is <=70% of the body height,
its width is <= one body height, its horizontal gap is <=35% of body height,
and its top/bottom place it above the baseline. Ambiguous, distant, normal-height,
low-confidence or single-box `ezDispense TM` observations are not changed. This is
a conservative layout heuristic, not verification of the exact trademark glyph
or typeface. It never converts TM into the Unicode trademark sign.

Raw OCR and original boxes remain unchanged; evaluated text can become
`ezDispenseTM`. JSON and failure metadata preserve geometry evidence.
SUPERSCRIPT_TM_ATTACHED identifies an adjustment; NO_SEPARATE_TM_BOX means OCR did
not provide a separate mark box; GEOMETRY_NOT_CONFIRMED means the layout criteria
were not met. No additional OCR calls or capture scheduling changes are introduced.


### Preserve substituted characters in horizontal assembly

If no exact placement exists, observations of at least eight characters may be
placed with at most one substitution per eight characters (maximum two). Placement
must have an exact run of at least six characters and beat every other circular
position by at least two errors. Ambiguous exact matches are never disambiguated
by this fallback. Observed letters are stored unchanged and voted as before; no
expected letter is inserted. Spacing, order, and final comparison still apply.

OCR Timing labels these frames ALIGNED_WITH_SUBSTITUTIONS and records each actual
letter and expected letter in Observed substitutions. For example `Codigo` can
appear in a complete assembly and still fail against `Código` in exact mode.
Such a frame cannot confirm a cached automation PASS. Insertions, deletions, short
fragments, and extensive errors are outside this fallback's scope. Capture timing,
OCR calls, and the separate raw-text display are unchanged.
