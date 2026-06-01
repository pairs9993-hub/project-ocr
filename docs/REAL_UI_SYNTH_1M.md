# Real UI Synthetic 1M Workflow

This workflow generates LG-style washer/dryer OCR training screens using only
English, French, and Spanish. It writes the same `labels.jsonl` schema as the
existing synthetic data, including per-line text boxes for detection and
recognition crop generation.

## Smoke Test

Run a small sample first:

```powershell
.venv\Scripts\python.exe scripts\generate_real_ui_synth.py `
  --output-dir artifacts\real_ui_synth_smoke `
  --count 30 `
  --languages en fr es

.venv\Scripts\python.exe scripts\build_manifest_from_labels.py `
  --source artifacts\real_ui_synth_smoke\labels.jsonl `
  --output artifacts\real_ui_synth_smoke_manifest.jsonl

.venv\Scripts\python.exe scripts\build_char_dict.py `
  --no-defaults `
  --label-file artifacts\real_ui_synth_smoke_manifest.jsonl `
  --output-dict artifacts\real_ui_synth_smoke_ppocr_keys.txt `
  --output-coverage artifacts\real_ui_synth_smoke_charset.json

.venv\Scripts\python.exe scripts\prepare_rec_data.py `
  --train-source artifacts\real_ui_synth_smoke_manifest.jsonl `
  --output-dir artifacts\real_ui_synth_smoke_rec `
  --char-dict artifacts\real_ui_synth_smoke_ppocr_keys.txt `
  --train-cap 30
```

Expected result: 30 generated screens, in-bounds text boxes, and zero OOV skips
when using the freshly built char dictionary.

## Generate 1M Images

Use chunked generation so the job can resume cleanly:

```powershell
powershell.exe -ExecutionPolicy Bypass -NoProfile `
  -File scripts\generate_real_ui_1m_chunks.ps1 `
  -TotalCount 1000000 `
  -ChunkSize 125000 `
  -OutputRoot generated_1000000_real_ui_en_fr_es `
  -BuildManifest
```

This creates:

- `generated_1000000_real_ui_en_fr_es/chunks/*/images/*.png`
- `generated_1000000_real_ui_en_fr_es/chunks/*/labels.jsonl`
- `dataset/train_manifest_real_ui_1m.jsonl`
- `data/dict/ppocr_keys_real_ui_en_fr_es.txt`
- `artifacts/charset/coverage_real_ui_en_fr_es.json`

The script skips chunks whose `labels.jsonl` already has the expected number of
rows. If a chunk is incomplete, it regenerates that chunk.

## Generate A Matching Validation Set

Use a separate seed and keep validation out of the train manifest:

```powershell
.venv\Scripts\python.exe scripts\generate_real_ui_synth.py `
  --output-dir artifacts\real_ui_synth_val_5000 `
  --count 5000 `
  --languages en fr es `
  --seed 20260523

.venv\Scripts\python.exe scripts\build_manifest_from_labels.py `
  --source artifacts\real_ui_synth_val_5000\labels.jsonl `
  --output artifacts\real_ui_synth_val_5000_manifest.jsonl

.venv\Scripts\python.exe scripts\build_char_dict.py `
  --no-defaults `
  --label-file dataset\train_manifest_real_ui_1m.jsonl `
  --label-file artifacts\real_ui_synth_val_5000_manifest.jsonl `
  --label-file artifacts\company_real_screens\llm_labels\gpt_all_labels.jsonl `
  --output-dict data\dict\ppocr_keys_real_ui_en_fr_es.txt `
  --output-coverage artifacts\charset\coverage_real_ui_en_fr_es.json
```

## Prepare Recognition Crops

After generation:

```powershell
.venv\Scripts\python.exe scripts\prepare_rec_data.py `
  --train-source dataset\train_manifest_real_ui_1m.jsonl `
  --val-source artifacts\real_ui_synth_val_5000_manifest.jsonl `
  --output-dir data\rec_dataset_real_ui_1m `
  --char-dict data\dict\ppocr_keys_real_ui_en_fr_es.txt `
  --project-root E:\OCR_Project
```

For a quick sandbox, add `--train-cap 5000`.

## Prepare Detection Labels

```powershell
.venv\Scripts\python.exe scripts\prepare_det_data.py `
  --train-source dataset\train_manifest_real_ui_1m.jsonl `
  --val-source artifacts\real_ui_synth_val_5000_manifest.jsonl `
  --val-prefix . `
  --out-dir data\det_dataset_real_ui_1m `
  --project-root E:\OCR_Project
```

`--val-prefix .` means the validation manifest already contains
project-root-relative image paths.

## Full-Image Evaluation Guardrail

Recognition crops are only one half of the pipeline. Before trusting a trained
model, evaluate the exported detector and recognizer together on full screen
images:

```powershell
.venv\Scripts\python.exe scripts\evaluate_app_ocr_against_labels.py `
  --labels artifacts\real_ui_synth_val_5000_manifest.jsonl `
  --images E:\OCR_Project `
  --det-model app\models\det_v0.onnx `
  --rec-model app\models\rec_v0.onnx `
  --rec-keys app\models\ppocr_keys.txt `
  --limit 200 `
  --out artifacts\app_eval\real_ui_val_ocr_vs_labels.jsonl
```

Use this as the promotion check after every detector/recognizer export. If crop
accuracy is high but full-image CER is bad, mine the failed rows from this JSONL
and add more full-screen detection data for those patterns.

## Practical Notes

- 1M PNG screenshots can take tens of GB depending on compression and text
  density. Keep the generated folder ignored by Git.
- The generator intentionally randomizes cycle names, settings, status text,
  delays, detergent/softener values, error messages, and washer/dryer phrases.
- GPT-reviewed real labels are added to the char-dict step so symbols like
  `▶Ⅱ`, `™`, accented French/Spanish letters, and internal-code-like strings are
  covered.