# French accent recovery — final technical report

**Status: NOT_SOLVED. The production French baseline is unchanged.**

```
ACCENT_TARGET_FEASIBILITY     = FAIL
LINE_VERIFIER_APPROACH_STATUS = REJECTED
ACCENT_TARGET_STATUS          = NOT_SOLVED
END_TO_END_ROI_STATUS         = NOT_SOLVED
RUNTIME_ENFORCE_ALLOWED       = NO
PRODUCTION_DECISION           = KEEP_EXISTING_FRENCH_BASELINE
```

Nothing in this report should be read as a demonstration that any approach is
safe. Several gates were passed on finite samples; none of them establishes an
error rate, and the one end-to-end test of the actual objective failed.

---

## 1. The problem and the constraints

On a correctly rendered screen the French baseline reads

> `Veuillez allumer l'eau.`

as

> `Véuillez allumer l'eau.`

An accent is invented where none was drawn. The requirement was to remove that
accent **without ever silently correcting a real on-screen typo**: if a user's
screen genuinely reads `Véuillez`, that must survive and the ROI must fail.

Constraints held throughout:

- Expected text never used for recognizer selection, correction, or routing —
  only for post-hoc scoring.
- The target ROI, F0 results and diagnostic failure images never used to choose
  thresholds, parameters, crops or guards.
- No training on real UI images.
- The baseline French ONNX never modified or replaced.
- The full recognizer never retrained.
- Scope limited to `e`/`é`. The `i`/`l` confusion and the separate `l` omission
  were excluded.
- No runtime integration, no `run.bat` or model-package changes, no promotion.

---

## 2. Provenance

The exact bytes `engine.run()` received were captured, not reconstructed. A
capture qualifies only when tagged `exact_recorded_ocr_input`; earlier attempts
were relabelled `approximate_saved_roi` when it emerged they were
reconstructions.

| Artifact | SHA-256 |
|---|---|
| detector | `21af37f36ce3940ba2fd201c6035571ae5807cf0333f1734d6d5b95c62135b7c` |
| dictionary | `7ff72cdde593c6f80ebd573dddb67b1a103a1607a444c11c4b2b7db57ae1d627` |
| recognizer `fr` | `d6a439c2b59b46051ea3e07a9d7df69cb76589489b4e487b3d365a773b903b0d` |
| recognizer `en_es` | `7f47f2d0a1c871656574ae2c5e8c5430fbfb0ca89ea7eb52f4d0babd1e571d0f` |
| recognizer `zh` | `ea841418e0a09bb17ff4ca0116ba67eca8856ddecd30b5d5ea6fe6274c25e944` |
| recorded ROI (`none`) | `8b29ed2b308c31ef01c34ef25d30db87322b8a7ca185adae237c7ae4b2573aa2` |

All five model hashes are unchanged from before this work.

---

## 3. What was tried, in order

### 3.1 Global recognizer recovery — 29 candidates, all rejected

Twenty-nine ONNX recognizers were funnelled through Gate A. None passed. The
gate was not relaxed to admit any of them.

### 3.2 Pure text router

A router permitting a single `é`→`e` substitution passed offline evaluation on
stored JSONL. Its limits are structural: it cannot distinguish a hallucinated
accent from a real typo, because both produce the same string. It was never a
candidate for the safety requirement, only a scaffold.

Two defects found here shaped everything after. I had compared stored baseline
metrics against recomputed routed metrics, producing a fabricated 328→343
"improvement"; and the router returned NFC-normalised text, corrupting
codepoints it was supposed to leave alone. Both were reported by the user, not
caught by me.

### 3.3 accent-v3 CNN — F1 PASS, F0 FAIL

A glyph-level classifier reached **0 false corrections in 15,702** holdout
cases (F1 PASS) and then scored **0/7** on the target (F0 FAIL): its guard
rejected every target glyph. Passing a large negative test and failing the one
positive test is the pattern that recurs throughout this work.

### 3.4 CTC localization — rejected

The hypothesis that a CTC token span approximates a glyph box was tested and
disproved: span width tracks line height, and the fixed 4px pad is 62–72% of
the crop. Two localization designs (v4, v4.1) accepted 79 and 56 misassignments
respectively; 82% of those originated inside "consensus", not in the fallback I
had blamed.

### 3.5 Line verifier — the main effort

Rather than crop a glyph, give the model the whole recognizer line plus an
ordinal query and let it locate the target itself.

**Data.** Mining natural hallucinations failed three times. Word-level
diagnosis over 150 candidates and 60,000 renderings explains why: 4 words carry
robust support, 16 sparse, and 130 produced no event in 400 renderings each.
Calibration-v2 produced **0 hallucinations in 39,189 renderings** across three
words — not a budget shortfall but an absence, which no budget increase can
fix. Training moved to counterfactual pairs: the same word, font, size and
optical settings rendered once bare and once accented, 12,000 for training and
3,000 for calibration.

**Training.** Three seeds under one sealed config. Seed 37 was degenerate — it
never predicted BARE_E at all. Seed 23 was selected on the sealed criteria,
losing to seed 11 on coverage but winning on UNKNOWN violations, 3 against 4.

**Threshold.** Frozen at `0.6443382502` on a held-out partition plus 1,089
legitimate-accent preservation rows.

| Measure | Result | One-sided 95% upper bound |
|---|---|---|
| legitimate accent false-corrected | 0 / 1,688 | 0.00177 |
| `e`→`é` wrong direction | 0 / 599 | — |
| non-accent change | 0 / 200 | — |
| bare-e coverage | 572 / 599 (95.5%) | — |

Zero observed errors on a finite sample is **not** a zero error rate. The
bounds above are what the data supports.

### 3.6 Lexical shortcut preflight — FAIL

| Probe | Rate | Tolerance | Result |
|---|---|---|---|
| MASKING_INVARIANCE | 0.7057 | 0.10 | FAIL |
| CONTEXT_INVARIANCE | 0.1250 (n=24) | 0.05 | FAIL |
| TARGET_SWAP_RESPONDS | 0.0050 | 0.05 | pass |
| ORDINAL_SHIFT_TO_UNKNOWN | 0.1971 | 0.20 | pass |
| INPUT_CONTRACT | 0.0 | 0.0 | pass |
| PREMODEL_FAIL_CLOSED | 0.0 | 0.0 | pass |

Blanking the columns outside a window around the queried glyph moves 820 of
1,162 confident verdicts, and **337 real accents become BARE**.

*Result vs inference.* The result is that the model is brittle to context
removal. The inference that this proves a lexical shortcut is **not** supported:
removing neighbours changes what the ordinal refers to and moves the input
off-distribution simultaneously, so the probe cannot separate memorisation from
destroyed localisation. Recorded as `MASKING_AS_LEXICAL_CAUSAL_PROOF = INVALID`.
The FAIL stands either way.

### 3.7 ONNX parity

| Layer | Result |
|---|---|
| class parity (15,051 rows) | 0 mismatches, all three runtimes |
| training ORT ↔ product ORT | identical, 0.0 error |
| **raw verdict parity** | **FAIL** — 1 row |
| **internal guarded verdict parity** | **FAIL** — 1 row |
| correction action parity | PASS — 0 mismatches |
| final text parity | PASS — 0 mismatches |
| unsafe action mismatch | 0 |

The surviving disagreement is one row where `p_accent` is 0.64433819 in PyTorch
and 0.64433837 in ONNX against a threshold of 0.64433825 — one runtime returns
UNKNOWN, the other ACCENT_PRESENT. Both map to `KEEP_BASELINE`, so no output
text differs. The internal parity status remains FAIL and was not reinterpreted.

A fail-closed uncertainty band (`epsilon = 1e-4`, ~17.6× the largest observed
5.692e-06 disagreement) was added as a successor decision rule. It is monotone
by construction — verified over 15,051 rows to only ever turn BARE_E into
UNKNOWN, never to create a correction. It did not cover this row because the
band gates `p_bare` and the disagreement is at the ACCENT boundary.

### 3.8 One-shot F0 — 2/7

Official runtime: product `.venv` ONNX Runtime.

| Perturbation | `p_bare` | Verdict | Action | Final first word |
|---|---|---|---|---|
| none | 0.0356 | UNKNOWN | KEEP_BASELINE | `Véuillez` |
| crop_left_1px | 0.0356 | UNKNOWN | KEEP_BASELINE | `Véuillez` |
| crop_top_1px | 0.0356 | UNKNOWN | KEEP_BASELINE | `Véuillez` |
| crop_bottom_1px | 0.5904 | UNKNOWN | KEEP_BASELINE | `Véuillez` |
| pad_border_1px | 0.5904 | UNKNOWN | KEEP_BASELINE | `Véuillez` |
| crop_right_1px | 0.9913 | BARE_E | APPLY_E_CORRECTION | `Veuillez` |
| crop_all_1px | 0.9913 | BARE_E | APPLY_E_CORRECTION | `Veuillez` |

| Gate | Result |
|---|---|
| first word corrected 7/7 | **2/7** |
| legitimate accents preserved | 7/7 |
| `e`→`é` additions | 0 |
| non-accent changes | 0 |
| target blocked by KEEP_BASELINE | 5/7 |

**Result:** the target probability takes three distinct values — 0.0356, 0.5904,
0.9913 — across seven inputs differing by a single pixel of cropping.

**Inference:** the model is not reading a stable property of the drawn glyph.
This is consistent with the masking finding, but F0 alone does not isolate the
cause.

The safety side held: no legitimate accent was removed, no `e` became `é`, and
no other character changed, in any of the seven. The failure is that the
correction does not fire, not that it fires wrongly. It improves on accent-v3's
0/7 and does not approach the 7/7 gate.

---

## 4. The separate `l` omission

`l'eau` is read as `'eau` in the same ROI. Root cause identified as
`RIGHT_PADDING_CONTEXT_EFFECT`: with byte-identical content, varying only the
zero-padding width flips the decode (padW 458 vs 459 give the same timestep
count and different text). A per-line fix was rejected — it repaired 3 cases and
broke 4. This is out of scope for the accent gate and remains unaddressed; it is
why `full_roi_exact` is 1 of 7 rather than 2.

---

## 5. Production state

| Check | Result |
|---|---|
| French baseline ONNX hash | unchanged |
| detector / dictionary / other recognizers | unchanged |
| `run.bat` behaviour | unchanged |
| runtime / model package | unchanged |
| verifier integrated into runtime | no |
| running training or evaluation processes | 0 |
| weights, ONNX, images, tensors pushed to GitHub | none |

The product behaves exactly as it did before this work began.

---

## 6. Reusable vs experimental

### Production-independent utility

Generally useful, no dependency on the accent experiment:

- `diagnostic_runner.py` — append-only checkpoints, fsync, atomic reports, RNG
  replay verified byte-identical across interruption
- `terminal_reason.py` — mutually exclusive terminal reasons that reconcile with
  the row count, separated from multi-label diagnostic flags
- `glyph_geometry.py` — measured pixel geometry, polarity-independent
- `premodel_gate.py` — fail-closed rejection of malformed queries before
  inference
- exact-OCR-input failure capture and the `run.bat` staleness guard
- fail-closed evaluator with alignment and duplicate detection
- model provenance and hash verification
- `interaction_stats.py`, `cluster_inference.py` — small-sample statistics with
  rank and starved-column guards
- the unit tests for all of the above

### Experimental only — do not merge to production

- French specialist router
- accent verifier v1 / v2 / v3
- line verifier (model, training, threshold, guard, ONNX)
- all synthetic datasets and target funnels
- experimental ONNX and configs

### Suggested cherry-pick candidates (not performed)

| Commit | Content |
|---|---|
| `ab2e7bd` | `diagnostic_runner.py`, `glyph_geometry.py` + tests |
| `c1e651e` | `terminal_reason.py` + invariant tests |
| `e764947` | attrition gating that separates pipeline loss from recognizer error |
| `54508a5` | `cluster_inference.py` with rank and starved-column guards |

Each mixes utility with experimental code and would need splitting before any
merge. No cherry-pick was performed.

---

## 7. Conditions for reopening

This is closed unless one of the following is newly true:

- an independent recognizer backbone different from the current one
- a new synthetic training recipe frozen without looking at the target
- an independent regression set broader than the current 430
- independent evaluation material from the real product renderer, covering both
  correct and mistyped text
- a change of policy on using real UI images for training

**Retraining or threshold adjustment based on this model's F0 result is not a
reopening condition.** The 2/7 outcome must not be used to steer a next attempt;
doing so would tune against the target.

---

## 8. What I got wrong

Recorded because the errata are part of the result:

- compared stored against recomputed metrics, fabricating an improvement
- returned NFC-normalised text from a router meant to preserve codepoints
- claimed `intrusion=0` from a median while p95 was 1.000
- asserted the detector path was cleaner; direct crops decode more often
- claimed font dependence from cohorts that also differed in size; matched
  testing showed all 11 fonts hallucinate
- sized a canvas with `box[3] - box[1]` where `textbbox` returns absolute
  coordinates, clipping glyphs and collapsing detector yield from 97% to 16%
- derived `pad_y` from `pad_x`, producing 3–5px canvases the detector upscaled
  20–30× and destroyed
- divided by a zero-count bin, producing `inf`, which satisfied my own ">= 3.0"
  test and declared a dependence from 13 events whose intervals all overlapped
- summed pipeline attrition with recognizer error, nearly failing a sound funnel
- described a stratified role assignment that the code discarded by re-sorting
- ranked an abstaining model best, because zero false corrections earned by
  declining to act is not safety
- introduced an UNKNOWN class the sealed input contract cannot represent
- reported a payload-text hash as a file hash

---

*Every gate that failed is recorded as failed. No tolerance, threshold or quota
was relaxed to obtain a pass.*
