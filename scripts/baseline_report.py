"""Generate a human-readable baseline report from JSONL results.

Reads:
  artifacts/baseline/raw_results.jsonl
  artifacts/baseline/preprocessed_results.jsonl

Writes:
  artifacts/baseline/REPORT.md
  artifacts/baseline/collage_worst_<mode>.png
  artifacts/baseline/collage_best_<mode>.png
  artifacts/baseline/per_script_examples_<mode>.png
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ocr_validator.evaluate import script_of


PROJECT = Path(__file__).resolve().parent.parent
ART = PROJECT / "artifacts/baseline"
TEST = PROJECT / "dataset/test"


def load(path: Path) -> List[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def fmt_pct(v: float) -> str:
    return f"{v*100:.1f}%"


def md_table(rows: List[Tuple], headers: List[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        cells = []
        for x in r:
            if isinstance(x, float):
                cells.append(fmt_pct(x))
            else:
                cells.append(str(x))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def aggregate(items, key_fn) -> List[Tuple[str, int, float, float]]:
    buckets: Dict[str, List[dict]] = {}
    for it in items:
        k = key_fn(it)
        buckets.setdefault(k, []).append(it)
    rows = []
    for k, vs in buckets.items():
        n = len(vs)
        rows.append((
            k, n,
            sum(v["cer"] for v in vs) / n,
            sum(v["wer"] for v in vs) / n,
        ))
    rows.sort(key=lambda r: -r[2])  # by CER desc
    return rows


def overall_stats(items):
    n = len(items)
    return (
        sum(i["cer"] for i in items) / n,
        sum(i["wer"] for i in items) / n,
        sum(i["elapsed_ms"] for i in items) / n,
    )


def comparison_table(raw: List[dict], tuned_runs: List[Tuple[str, str, List[dict]]]) -> str:
    raw_cer, raw_wer, raw_ms = overall_stats(raw)
    rows = [("pretrained/raw", len(raw), raw_cer, raw_wer, f"{raw_ms:.1f}", "baseline")]
    for tag, mode, items in tuned_runs:
        cer_, wer_, ms = overall_stats(items)
        rows.append((f"{tag}/{mode}", len(items), cer_, wer_, f"{ms:.1f}", f"{(cer_ - raw_cer) * 100:+.2f} pp"))
    return md_table(rows, ["run", "n", "mean CER", "mean WER", "ms/img", "CER delta"])


# ---------- visual collages ----------


def render_caption(text: str, width: int, height: int = 18,
                   font_scale: float = 0.4) -> np.ndarray:
    strip = np.full((height, width, 3), 30, dtype=np.uint8)
    cv2.putText(strip, text[:200], (4, height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (220, 220, 220), 1, cv2.LINE_AA)
    return strip


def render_text_block(text: str, width: int, line_height: int = 16,
                      max_lines: int = 4, color=(220, 220, 220)) -> np.ndarray:
    lines = []
    for raw_line in text.split("\n"):
        # wrap roughly
        chunks = [raw_line[i:i+50] for i in range(0, max(len(raw_line), 1), 50)] or [""]
        lines.extend(chunks)
    lines = lines[:max_lines]
    h = line_height * max(1, len(lines)) + 6
    img = np.full((h, width, 3), 22, dtype=np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (6, line_height * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img


def render_sample(item: dict, target_w: int = 480) -> np.ndarray:
    img_path = TEST / item["image_path"]
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        bgr = np.zeros((100, target_w, 3), dtype=np.uint8)
    # scale to target_w preserving aspect
    h, w = bgr.shape[:2]
    scale = target_w / w
    bgr = cv2.resize(bgr, (target_w, int(h * scale)))

    header = render_caption(
        f"{Path(item['image_path']).name}  | lang={item['language']} bg={item['background']} CER={fmt_pct(item['cer'])}",
        target_w, height=20, font_scale=0.42,
    )
    ref_block = render_text_block("REF: " + item["ref"].replace("\n", " | "), target_w,
                                  color=(180, 220, 180))
    hyp_block = render_text_block("HYP: " + item["hyp"].replace("\n", " | "), target_w,
                                  color=(180, 200, 240))
    sep = np.full((2, target_w, 3), 0, dtype=np.uint8)
    return np.vstack([header, bgr, ref_block, hyp_block, sep])


def make_collage(items: List[dict], cols: int = 2, target_w: int = 520) -> np.ndarray:
    panels = [render_sample(it, target_w=target_w) for it in items]
    if not panels:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    max_h = max(p.shape[0] for p in panels)
    panels = [
        cv2.copyMakeBorder(p, 0, max_h - p.shape[0], 0, 0,
                           cv2.BORDER_CONSTANT, value=(0, 0, 0))
        for p in panels
    ]
    rows = []
    gutter_v = np.full((max_h, 8, 3), 0, dtype=np.uint8)
    for i in range(0, len(panels), cols):
        row_panels = panels[i : i + cols]
        row = row_panels[0]
        for p in row_panels[1:]:
            row = np.hstack([row, gutter_v, p])
        rows.append(row)
    max_w = max(r.shape[1] for r in rows)
    rows = [
        cv2.copyMakeBorder(r, 0, 0, 0, max_w - r.shape[1],
                           cv2.BORDER_CONSTANT, value=(0, 0, 0))
        for r in rows
    ]
    gutter_h = np.full((10, max_w, 3), 0, dtype=np.uint8)
    out = rows[0]
    for r in rows[1:]:
        out = np.vstack([out, gutter_h, r])
    return out


# ---------- main ----------


def write_mode_section(name: str, items: List[dict]) -> str:
    cer_, wer_, ms = overall_stats(items)
    s = [f"### {name}\n",
         f"- samples: **{len(items)}**",
         f"- mean CER: **{fmt_pct(cer_)}**, mean WER: **{fmt_pct(wer_)}**",
         f"- mean inference time: {ms:.1f} ms/img\n",
         "**By script** (sorted by CER desc):\n",
         md_table(aggregate(items, lambda i: script_of(i["language"])),
                  ["script", "n", "CER", "WER"]),
         "\n**By language** (top 10 by CER):\n",
         md_table(aggregate(items, lambda i: i["language"])[:10],
                  ["language", "n", "CER", "WER"]),
         "\n**By background**:\n",
         md_table(aggregate(items, lambda i: i["background"]),
                  ["background", "n", "CER", "WER"]),
         "\n**By pattern**:\n",
         md_table(aggregate(items, lambda i: i["pattern"]),
                  ["pattern", "n", "CER", "WER"]),
         ""]
    return "\n".join(s)


def main():
    raw_path = ART / "raw_results.jsonl"
    pre_path = ART / "preprocessed_results.jsonl"

    if not raw_path.exists():
        sys.exit(f"missing {raw_path}")
    raw = load(raw_path)
    pre = load(pre_path) if pre_path.exists() else None
    tuned_runs = []
    for path in sorted(ART.glob("*_raw_results.jsonl")):
        if path.name == "raw_results.jsonl":
            continue
        tag = path.name[: -len("_raw_results.jsonl")]
        tuned_runs.append((tag, "raw", load(path)))
    for path in sorted(ART.glob("*_preprocessed_results.jsonl")):
        if path.name == "preprocessed_results.jsonl":
            continue
        tag = path.name[: -len("_preprocessed_results.jsonl")]
        tuned_runs.append((tag, "preprocessed", load(path)))

    md = ["# Baseline OCR Report (RapidOCR pretrained, no fine-tuning)\n",
          f"Test set: **{len(raw)} samples** from `dataset/test`. Pretrained model trained mainly "
          "on English + Simplified Chinese; this run is meant to expose where the model fails so "
          "the fine-tuning step can target those gaps.\n"]

    raw_cer, raw_wer, raw_ms = overall_stats(raw)
    if pre:
        pre_cer, pre_wer, pre_ms = overall_stats(pre)
        delta = (pre_cer - raw_cer) * 100
        md.append("## Overall comparison\n")
        md.append(md_table([
            ("raw", len(raw), raw_cer, raw_wer, f"{raw_ms:.1f}"),
            ("preprocessed", len(pre), pre_cer, pre_wer, f"{pre_ms:.1f}"),
        ], ["mode", "n", "mean CER", "mean WER", "ms/img"]))
        md.append(f"\nPreprocessing delta on CER: **{delta:+.2f} pp**\n")

    if tuned_runs:
        md.append("## Fine-tuned comparison\n")
        md.append(comparison_table(raw, tuned_runs))
        md.append("")

    md.append("\n## Per-mode breakdown\n")
    md.append(write_mode_section("Raw input", raw))
    if pre:
        md.append(write_mode_section("Preprocessed input", pre))
    for tag, mode, items in tuned_runs:
        md.append(write_mode_section(f"Fine-tuned {tag} {mode} input", items))

    # collages
    def save_collages(name: str, items: List[dict]):
        items_sorted = sorted(items, key=lambda i: i["cer"])
        best = items_sorted[:6]
        worst = items_sorted[-6:]
        cv2.imwrite(str(ART / f"collage_best_{name}.png"), make_collage(best))
        cv2.imwrite(str(ART / f"collage_worst_{name}.png"), make_collage(worst))
        # one example per script
        seen = set()
        per_script = []
        for it in items_sorted:
            sc = script_of(it["language"])
            if sc not in seen:
                seen.add(sc)
                per_script.append(it)
        cv2.imwrite(str(ART / f"per_script_examples_{name}.png"),
                    make_collage(per_script[:8], cols=2))

    save_collages("raw", raw)
    if pre:
        save_collages("preprocessed", pre)
    for tag, mode, items in tuned_runs:
        save_collages(f"{tag}_{mode}", items)

    md.append("\n## Visual examples\n")
    md.append("- `collage_best_raw.png` / `collage_worst_raw.png` — best/worst 6 raw cases")
    if pre:
        md.append("- `collage_best_preprocessed.png` / `collage_worst_preprocessed.png` — preprocessed")
    for tag, mode, _ in tuned_runs:
        md.append(f"- `collage_best_{tag}_{mode}.png` / `collage_worst_{tag}_{mode}.png` — fine-tuned {tag} {mode}")
    md.append("- `per_script_examples_<mode>.png` — one example per script family\n")

    md.append("\n## Interpretation\n")
    md.append(
        "**Why these numbers look like they do:**\n\n"
        "- The pretrained RapidOCR model is built on a Latin + Simplified Chinese dictionary. "
        "Languages outside that scope (Cyrillic, Arabic, Greek, Thai) get encoded as "
        "near-random Latin characters or empty strings, so their CER is close to 1.0 (100% errors).\n"
        "- Latin scripts work passably but lose **spaces and accented characters** (`é`, `ä`, `ü`, ...) "
        "because the recognizer's text formatting and dictionary aren't aligned with our domain.\n"
        "- Backgrounds with gradients, vignettes, or toast bars depress accuracy further because "
        "the detector treats them as noise. Preprocessing (top-hat) normalizes them.\n\n"
        "**What fine-tuning is expected to fix:**\n\n"
        "1. Coverage of the 22 languages we care about (Cyrillic / Arabic / Greek / Thai will become "
        "first-class instead of catastrophic).\n"
        "2. Restoring spaces and accents in Latin scripts.\n"
        "3. Domain-specific terms and units (washer-cycle vocabulary).\n"
    )

    out_md = ART / "REPORT.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
