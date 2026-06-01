# Copilot Handoff Prompt — OCR Validator Project

Copy everything in this file into a Copilot chat (or use it as a `.copilot-instructions` reference). Append your specific task at the bottom.

---

You are picking up a multi-phase OCR validation project at **`e:\OCR_Project`** on Windows 11. Read this brief in full before touching code. Do not redo work already finished — extend it.

## Project goal

Build a CPU-deployable, fine-tuned OCR pipeline for **LG washer UI screenshots** that hits **CER ≤ 5%** on a 1,500-image validation set across **22 languages** (English, Korean, Chinese simplified/traditional, French, German, Dutch, Greek, Thai, Arabic, Bulgarian, Russian, Ukrainian, Czech, plus more — see `data/dict/ppocr_keys.txt`).

The pipeline is modular:

```
Module 1: Preprocessor  →  Module 2: Text OCR  →  Module 4: Merger  →  Module 5: Postprocess
                       ↘  Module 3: Icon Detect ↗
```

Inputs: 320×240 (small) and 1280×480 (large) PNG screen captures, mostly white text on black/dark backgrounds with gradients, vignettes, toast bars, inline icons. Output: per-image JSON with text + bounding boxes + icon labels (see spec in the original brief — full schema at the head of this repo's history).

## Constraints (non-negotiable)

- **Free / open source only.** No commercial OCR APIs.
- **Inference must run on CPU.** Training may use GPU (RTX 4070 Laptop 8GB available locally, or Colab).
- **Language extensibility.** Adding a new language must mean: regenerate synth, rebuild char dict, retrain. No hard-coded language assumptions.
- **Modularity.** Each module is independent and testable.
- **Reproducibility.** Fixed seeds (currently 42 for splits).
- **OS:** Windows 10/11, Python 3.10+. Dev on Linux/WSL is fine.

## Tech stack

| Concern | Library |
|---|---|
| OCR runtime | `rapidocr-onnxruntime` (ONNX, CPU) |
| Training | `paddlepaddle-gpu` 2.6.1 + `paddleocr` 2.7.3, base = PP-OCRv4 mobile_rec multilingual |
| ONNX export | `paddle2onnx` |
| Image | `opencv-python`, `pillow` |
| Synthetic data | `synth_generator.py` (custom, 1,143 lines, in repo root) |
| Metrics | `rapidfuzz` for Levenshtein-based CER/WER |
| Python | 3.12 venv at `.venv/`; separate `.venv_train/` recommended for Paddle |

## Directory layout (current)

```
e:\OCR_Project\
├── .venv\                                 # CPU venv (rapidocr, opencv, pillow, rapidfuzz)
├── ocr_validator\                         # main package
│   ├── __init__.py
│   ├── preprocess.py                      # Module 1: tophat / CLAHE / V-channel / pad
│   ├── ocr_engine.py                      # Module 2: RapidOCR wrapper + box-to-text
│   └── evaluate.py                        # CER/WER + script_of() mapping
├── scripts\
│   ├── split_dataset.py                   # 9-demo / 1500-test / rest-train split
│   ├── build_train_manifest.py            # unify 18k seed + 980k chunks → train_manifest.jsonl
│   ├── build_char_dict.py                 # scan all labels → 417-char dict
│   ├── prepare_rec_data.py                # crop text regions → PaddleOCR rec format
│   ├── preprocess_demo.py                 # visualize preprocess stages on 9 demos
│   ├── preprocess_bg_stress.py            # 4 background types stress test
│   ├── baseline_eval.py                   # run OCR + compute CER/WER on test set
│   ├── baseline_report.py                 # markdown + collage reports
│   └── baseline_chart.py                  # PIL-rendered bar chart + Unicode example panel
├── synth_generator.py                     # main synthetic image generator (do not break)
├── notebooks\finetune_colab.md            # Colab fine-tuning walkthrough
├── docs\LOCAL_GPU_TRAINING.md             # local RTX 4070 fine-tuning walkthrough
├── data\
│   ├── dict\ppocr_keys.txt                # 417-char dictionary across 22 langs
│   ├── rec_dataset_v0\                    # PaddleOCR rec training data (325MB)
│   │   ├── train_crops\                   # 63,606 crops
│   │   ├── val_crops\                     # 5,236 crops
│   │   ├── rec_train.txt
│   │   ├── rec_val.txt
│   │   └── ppocr_keys.txt
│   └── (icons\, fonts\ — to be added in Phase 5)
├── dataset\
│   ├── demo\        (9 images)            # for visual validation
│   ├── test\        (1,500 images)        # held-out — proxy for real screens
│   ├── train\       (18,491 images)       # seed train (copy of 20k batch minus demo/test)
│   └── train_manifest.jsonl               # 998,491 entries, unified across seed + 8 chunks
├── generated_20000_balanced\              # original 20k batch (sources for dataset splits)
├── generated_980000_balanced_vi\          # 980k batch in 8 × 122,500 chunks
│   ├── chunks\chunk_<start>_<end>\
│   └── logs\
├── artifacts\
│   ├── preprocess_demo\                   # Phase 2 visual proofs
│   ├── baseline\                          # Phase baseline results + report
│   │   ├── REPORT.md
│   │   ├── chart_cer_by_script.png        # the key visual
│   │   ├── examples_unicode.png
│   │   ├── raw_results.jsonl              # 1500 per-sample rows
│   │   └── preprocessed_results.jsonl
│   └── charset\coverage.json
├── rec_dataset_v0.zip                     # 165MB, ready for Colab upload
└── COPILOT_PROMPT.md                      # this file
```

## Phase status (as of 2026-05-11)

| Phase | Description | Status |
|---|---|---|
| 1 | Scaffolding + dataset split | ✅ done |
| 2 | Preprocessor (top-hat / CLAHE / V-channel / pad) | ✅ done, visually verified on 4 background types |
| 3 | Synthetic data generation (~1M images, 22 languages) | ✅ done — 998,491 train manifest entries |
| 4A | Build char dict + crop dataset + Colab/local-GPU training plan | ✅ done |
| **4B** | **Run fine-tuning on PP-OCRv4 mobile_rec** | ⏳ next |
| 4C | Plug ONNX into RapidOCR + re-evaluate on 1500 test | ⏳ pending 4B |
| 5 | Icon detection (Module 3) — template matching + color-channel detect | ⏳ |
| 6 | Merger (Module 4) + Postprocessor (Module 5) | ⏳ |
| 7 | End-to-end evaluation with CER breakdown by category | ⏳ |

## Baseline numbers (pretrained RapidOCR, no fine-tuning)

Established on 2026-05-07, 1,500 test images:

| Mode | mean CER | mean WER | inference |
|---|---|---|---|
| raw | **39.4%** | 69.2% | 199 ms/img |
| preprocessed | 39.9% | 71.1% | 194 ms/img |

By script (raw):
- Thai: 92.5%, Arabic: 90.6%, Cyrillic: 86.6%, Greek: 85.8% — catastrophic; pretrained dict doesn't cover these
- Chinese traditional: 24.6%, Latin: 14.4%, Chinese simplified: 13.6% — passable; lose mostly spaces and accents

Preprocessing did **not** help on synthetic data (digital-clean inputs); expected to help once real camera captures arrive. Don't delete it.

KPI = CER ≤ 5%. Fine-tuning needs to close an 8× gap globally and a near-100× gap on non-Latin scripts.

## Conventions to follow

1. **Don't reformat / mass-edit existing files.** Other people are also editing `synth_generator.py` (Copilot has been generating data there).
2. **Korean for prose, English for code/identifiers.** The user reads Korean comments fine but expects all code in English.
3. **No emojis in code or filenames.** Allowed only in human-readable reports if asked.
4. **No commented-out code, no narrative docstrings longer than 5 lines.** Keep modules tight.
5. **Use absolute paths in scripts** — relative paths break depending on how the user invokes them. `Path(__file__).resolve().parent.parent` is the project root pattern used throughout.
6. **All scripts go in `scripts/`**, all package code in `ocr_validator/`, all data in `data/` or `dataset/`. Don't break this.
7. **Reproducibility**: every random split uses `random.Random(42)` or `numpy.random.default_rng(42)`. If you add a new split, follow that.
8. **For long-running operations (>30s)** add periodic progress prints every N items, with mean throughput and ETA. The user runs these in background.
9. **Background data work** like 980k chunk extraction is the user's responsibility (Copilot in a separate workspace). Don't try to scale up training data yourself unless explicitly asked.

## Phase 4B next steps (what you should actually do)

The user is about to run fine-tuning either via:
- (a) Colab T4, following `notebooks/finetune_colab.md` after uploading `rec_dataset_v0.zip`, OR
- (b) Local RTX 4070, following `docs/LOCAL_GPU_TRAINING.md`

When training finishes you'll get `models/v0/rec.onnx` + `models/v0/ppocr_keys.txt`. Then:

1. Extend `ocr_validator/ocr_engine.py` so `OCREngine` accepts optional `rec_model_path` and `rec_keys_path`, passing them through to `RapidOCR(...)`.
2. Add a CLI flag `--model-dir <dir>` to `scripts/baseline_eval.py` that loads from there.
3. Run `python scripts/baseline_eval.py --mode raw --model-dir models/v0` and regenerate the report.
4. Add a "fine-tuned vs baseline" comparison table to `scripts/baseline_report.py`.
5. If CER ≤ 5%: declare Phase 4 done, move to Phase 5. If not: analyze which scripts/patterns still fail, decide whether to (i) increase epochs, (ii) scale crops to the full 998k manifest, or (iii) add targeted augmentation.

## Things NOT to do

- Don't invent a new data layout — keep `dataset/<split>/` and the manifest convention.
- Don't replace RapidOCR with PaddleOCR for inference (we want CPU-light deployment).
- Don't bake language assumptions into preprocess.py — it must stay script-agnostic.
- Don't drop the preprocessing module just because it didn't help on synthetic data; it will help on real captures.
- Don't commit `data/rec_dataset_v0/` or `rec_dataset_v0.zip` or the generated 980k chunks to git — they're large and regeneratable.

## Read-before-coding files

In order of importance for Phase 4B and beyond:

1. `ocr_validator/ocr_engine.py` — you'll extend this
2. `scripts/baseline_eval.py` and `scripts/baseline_report.py` — your evaluation harness
3. `ocr_validator/evaluate.py` — CER/WER + `LANG_SCRIPT` mapping
4. `notebooks/finetune_colab.md` — the training recipe (config is the source of truth for char dict size, image_shape, etc.)
5. `synth_generator.py` (only if generating new data; otherwise leave alone)

---

**Append your specific task below this line:**

(e.g. "Run the local-GPU training following docs/LOCAL_GPU_TRAINING.md, then implement the `--model-dir` flag and re-run baseline_eval. Report the new CER table.")
