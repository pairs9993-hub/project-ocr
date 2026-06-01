# OCR Multilingual Hardening Plan

Date: 2026-05-25

## Current Diagnosis

Latest company-screen evaluation:

- Result file: `artifacts/app_eval/company_real_gpt_eval_domain_repair_v4_full.jsonl`
- 1045 images: PASS 981 / WARN 64 / FAIL 0
- Average CER: 0.0107
- Previous timer-repair model: PASS 882 / WARN 157 / FAIL 6, average CER 0.0213
- Regression check: 110 rows improved, 0 rows regressed

The current app model is good enough to avoid FAIL on the 1045-image English/Spanish set, but it is not robust enough for broad multilingual rollout.

The most important blocker is character coverage. The bundled app recognizer uses `app/models/ppocr_keys.txt` with only 92 characters. It has no Cyrillic, no Greek, no Thai, and no Latin Extended block. That means Bulgarian, Russian, Greek, Thai, and parts of Czech/Polish/Latvian/etc. are not merely weak; many target characters are impossible for the current model to output.

There is already a broader project dictionary at `data/dict/ppocr_keys.txt` with 417 characters covering Latin Extended, Cyrillic, Greek, Thai, and other planned scripts. A multilingual production recognizer must be trained with this broader dictionary or a newer superset. The current 92-character recognizer cannot be fixed for Bulgarian/Czech by fine-tuning alone.

## Remaining English/Spanish Weak Patterns

From the 64 WARN rows in the latest evaluation:

| Pattern | Count | Why it fails | Fix type |
| --- | ---: | --- | --- |
| Scheduled start/status screens | 17 | `MAÑ.`, `a.m./p.m. 12:30`, `Delay Start`, and icon text are small and tightly spaced | Detector + recognizer hard examples |
| Duration/timer numeric confusion | 13 | `40 min`, `15 min`, `59 min`, `1 hr 9 min` collapse to `1 h` / `1 hi` | Recognizer hard examples + timer-specific synthetic templates |
| Dense small explanatory text | 3 high-impact rows | 4-5 small lines like “Washes clothes with...” are near the readability limit | Higher-resolution detector crops + recognizer hard examples |
| Small numeric option rows | 4 | final `0` in `+2/+1/0` is sometimes not detected | Detector recall examples |
| List rows / icon residue | 5 | residual icons such as `.- -`, `D 3`, `S` leak into row text | Detector negative/icon examples + geometric filter expansion |
| Misc recognition | 22 | mostly case, accent, punctuation, or one-character digit/letter confusions | Lexicon and recognizer hard examples |

These are not solved by simply running more epochs on the same dataset. The model needs targeted hard-pattern data.

## Strategy

Use a two-track plan:

1. Harden English/Spanish/French first so basic Latin languages are reliable.
2. Build a true multilingual recognizer with a unified character dictionary before adding Bulgarian, Czech, and other languages.

## Phase 1: Hard-Mine The Current Real Failures

Goal: convert the current WARN rows into training data instead of only post-processing them.

Actions:

1. Create a hard-case manifest from all non-PASS rows plus near-threshold PASS rows, e.g. CER >= 0.035.
2. For each hard screen, keep the full-screen label for detector training and generate detector-produced recognition crops for recognizer training.
3. Oversample these cases during fine-tuning. A practical starting ratio is 1 hard sample batch for every 2-3 synthetic batches.
4. Keep the existing 1045 real images as a regression test; do not train directly on all of them without a separate holdout once more real data arrives.

Target hard templates to synthesize:

- Scheduled start: `Today/TMRW./MAÑ.`, `AM/PM/a.m./p.m. 12:00/12:30`, delay-start lines.
- Short durations: `9 min`, `15 min`, `40 min`, `59 min`, `1 h 9 min`, `1 hr 9 min`, `1 h 30 min`.
- Numeric option rows: `+2/+1/0`, `70/65 min/60`, `110/105 min/100`, `1.0/0.9 oz/0.8`.
- Dense descriptions: 4-5 line paragraph screens using 10-14 px fonts.
- Icon negatives: checkmarks, arrows, warning icons, app/status glyphs near text but not part of text.
- Brand/control tokens: `Wi-Fi`, `ThinQ`, `TurboWash™`, `ColdWash™`, `▶Ⅱ`, internal codes like `<PROC_W_SOAKING_PRC`.

Promotion gate for Phase 1:

- 1045 real images: PASS >= 1015, WARN <= 30, FAIL = 0
- Average CER <= 0.006
- No regression rows versus `company_real_gpt_eval_domain_repair_v4_full.jsonl`

## Phase 2: Build The Multilingual OCR Foundation

Goal: make Bulgarian/Czech/etc. representable and trainable.

Actions:

1. Use a unified dictionary, starting from `data/dict/ppocr_keys.txt` or a superset built from all target UI strings.
2. Rebuild recognition datasets with the same dictionary. The 92-character `app/models/ppocr_keys.txt` must not be used for multilingual training.
3. Train a new recognizer with the broad dictionary. Because the CTC output head shape changes, treat this as a new multilingual recognizer. Backbone/neck can be warm-started from an existing checkpoint if PaddleOCR skips the incompatible CTC head, but the head must be trained fresh.
4. Keep the detector model, but fine-tune it with multilingual full-screen labels and hard small-text examples. Detection is script-agnostic in principle, but real layouts with Cyrillic/Latin-Extended text widths need coverage.

Recommended first multilingual language set:

- Latin baseline: `en`, `es`, `fr`
- Latin Extended: `cs`, `pl`, `de`, `it`, `pt`, `nl`, `lv`, `lt`, `no`
- Cyrillic: `bg`, `ru`, `uk`
- Greek: `el`
- Thai: `th`
- Vietnamese: `vi`
- Chinese: `zh_cn`, `zh_tw`
- Arabic should be handled as a separate milestone because shaping and RTL ordering need extra validation.

Promotion gates:

- OOV rate: 0 for every target language validation set.
- Synthetic per-language validation: average CER <= 0.01 for easy/medium screens, <= 0.03 for dense small-text screens.
- Real company regression: current 1045 set must stay FAIL = 0 and should not drop below PASS 970.
- New real multilingual smoke set: at least 50-100 real screenshots per language before declaring production quality.

## Phase 3: Data Generation Plan

Generate data in three layers, not one huge generic dataset.

### Layer A: Domain Synthetic Screens

Use LG washer/dryer UI layouts with actual UI phrases per language.

Suggested volume for first multilingual run:

- 1.5M to 2M full-screen synthetic images.
- Balanced by language, with hard-pattern oversampling.
- At least 50k images per language for the first 20-22 languages.
- Extra 100k-200k images for hard patterns shared across all languages.

### Layer B: Recognition Crop Stress Set

Create crop-level samples specifically for hard strings:

- Tiny numeric tokens and short timers.
- Accented Latin/Czech strings: `č`, `ř`, `ě`, `š`, `ů`, `ý`, `ž`.
- Bulgarian Cyrillic UI terms and short status words.
- Dense paragraph crops at low contrast.
- Similar-looking confusions: `0/O/o`, `1/I/l/Ⅱ`, `rn/m`, `cl/d`, `Wi-Fi/Wi-F`.

This should be used for recognizer fine-tuning and balanced so rare characters are not drowned out by English text.

### Layer C: Real Hard Cases

Use detector-generated crops from real company screenshots with known labels.

Minimum collection target:

- 200-300 real screens for English/Spanish/French hard cases.
- 50-100 real screens per new language for smoke testing.
- Eventually 300-500 screens per high-priority language for robust validation.

## Phase 4: Training Sequence

Recommended order:

1. Train multilingual recognizer on synthetic crop dataset with broad dictionary.
2. Fine-tune recognizer on real hard crops and pseudo-rec crops.
3. Fine-tune detector on full-screen multilingual synthetic labels plus real hard screens.
4. Export ONNX models.
5. Evaluate full-screen E2E, not only crop validation.
6. Promote only if current 1045 regression and per-language validation both pass.

Do not promote based only on recognition crop accuracy. The current project already showed that crop validation can look good while whole-screen E2E fails due to detector crop quality, icon leakage, and line grouping.

## Immediate Next Implementation Tasks

1. Add a hard-case mining script:
   - input: latest evaluation JSONL
   - output: manifests for non-PASS and near-threshold rows
   - groups rows by pattern and copies image references for training/evaluation
2. Extend `scripts/generate_real_ui_synth.py` with hard-pattern templates:
   - schedule screens
   - short duration screens
   - numeric option rows
   - dense paragraph cycle descriptions
   - icon-negative layouts
3. Add multilingual phrase tables for `bg` and `cs` first.
4. Build `ppocr_keys_real_ui_multilang.txt` from all target languages and verify OOV = 0.
5. Run a small 20k-image multilingual smoke generation before attempting 1M+.
6. Train a small multilingual recognizer smoke model and test Bulgarian/Czech output capability.

## Decision

There is a path. The right approach is not “more epochs on the current model.” The current model should be treated as the English/Spanish/French app baseline. For Bulgarian/Czech and broader rollout, build a new multilingual recognizer with a broad character dictionary, add targeted hard-pattern synthetic data, and keep full-screen E2E evaluation as the gate.