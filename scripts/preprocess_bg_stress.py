"""Stress-test preprocessor on one image per background type from the test set."""

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ocr_validator.preprocess import (
    PreprocessConfig,
    run_pipeline_traced,
    trace_to_grid,
)


def main():
    project = Path(__file__).resolve().parent.parent
    test_dir = project / "dataset/test"
    out_dir = project / "artifacts/preprocess_demo/bg_stress"
    out_dir.mkdir(parents=True, exist_ok=True)

    # pick first occurrence of each background
    by_bg = {}
    with (test_dir / "labels.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            bg = d["background"]
            if bg not in by_bg:
                by_bg[bg] = d
            if len(by_bg) >= 4:
                break

    cfg = PreprocessConfig()
    for bg, d in by_bg.items():
        img_path = test_dir / d["image_path"]
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"skip: {img_path}")
            continue
        trace = run_pipeline_traced(bgr, cfg)
        grid = trace_to_grid(trace)
        out = out_dir / f"bg_{bg}_{img_path.stem}.png"
        cv2.imwrite(str(out), grid)
        print(f"bg={bg:15s}  {img_path.name:45s} -> {out.name}")


if __name__ == "__main__":
    main()
