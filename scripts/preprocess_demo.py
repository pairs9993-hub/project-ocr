"""Apply preprocessing to all demo images and save trace grids.

Output:
  artifacts/preprocess_demo/<stem>_grid.png   # per-image stage trace
  artifacts/preprocess_demo/summary.png        # 3x3 stack of inputs vs finals
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ocr_validator.preprocess import (
    PreprocessConfig,
    run_pipeline_traced,
    trace_to_grid,
)


def main():
    project = Path(__file__).resolve().parent.parent
    demo_dir = project / "dataset/demo/images"
    out_dir = project / "artifacts/preprocess_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = PreprocessConfig()

    images = sorted(demo_dir.glob("*.png"))
    if not images:
        sys.exit(f"no demo images in {demo_dir}")

    finals = []
    inputs = []
    for img_path in images:
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"skip (unreadable): {img_path}")
            continue
        trace = run_pipeline_traced(bgr, cfg)
        grid = trace_to_grid(trace)
        out = out_dir / f"{img_path.stem}_grid.png"
        cv2.imwrite(str(out), grid)
        print(f"{img_path.name}: {len(trace)} stages -> {out.name}")
        inputs.append(bgr)
        # final stage may be padded gray; convert to BGR for the summary
        final = trace[-1][1]
        if final.ndim == 2:
            final = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
        finals.append((img_path.name, bgr, final))

    # 3x3 summary: input | final pairs, 3 columns of pairs per row
    pair_imgs = []
    target_h = 240
    for name, bgr, final in finals:
        ih, iw = bgr.shape[:2]
        scale = target_h / ih
        bgr_s = cv2.resize(bgr, (int(iw * scale), target_h))
        fh, fw = final.shape[:2]
        scale_f = target_h / fh
        final_s = cv2.resize(final, (int(fw * scale_f), target_h))
        gutter = np.full((target_h, 4, 3), 80, dtype=np.uint8)
        pair = np.hstack([bgr_s, gutter, final_s])
        # caption strip
        strip = np.full((20, pair.shape[1], 3), 30, dtype=np.uint8)
        cv2.putText(strip, name, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (220, 220, 220), 1, cv2.LINE_AA)
        pair_imgs.append(np.vstack([strip, pair]))

    if pair_imgs:
        # arrange 3 per row
        rows = []
        max_w = max(p.shape[1] for p in pair_imgs)
        padded = [
            cv2.copyMakeBorder(p, 0, 0, 0, max_w - p.shape[1],
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))
            for p in pair_imgs
        ]
        gap = np.full((padded[0].shape[0], 12, 3), 0, dtype=np.uint8)
        for i in range(0, len(padded), 3):
            row_imgs = padded[i : i + 3]
            row = row_imgs[0]
            for ri in row_imgs[1:]:
                row = np.hstack([row, gap, ri])
            rows.append(row)
        max_w_row = max(r.shape[1] for r in rows)
        rows = [
            cv2.copyMakeBorder(r, 0, 0, 0, max_w_row - r.shape[1],
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))
            for r in rows
        ]
        vgap = np.full((12, max_w_row, 3), 0, dtype=np.uint8)
        summary = rows[0]
        for r in rows[1:]:
            summary = np.vstack([summary, vgap, r])
        cv2.imwrite(str(out_dir / "summary.png"), summary)
        print(f"summary -> {out_dir / 'summary.png'}")


if __name__ == "__main__":
    main()
