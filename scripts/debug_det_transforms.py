"""Debug: feed a few real samples through the transform pipeline and count
how many polygons survive MakeBorderMap / MakeShrinkMap."""
import sys, json, os
sys.path.insert(0, r"E:\OCR_Project\PaddleOCR")
os.chdir(r"E:\OCR_Project")

from ppocr.data.imaug import transform, create_operators

ops = create_operators([
    {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
    {"DetLabelEncode": None},
    {"EastRandomCropData": {"size": [256, 256], "max_tries": 50, "keep_ratio": True}},
    {"MakeBorderMap": {"shrink_ratio": 0.4, "thresh_min": 0.3, "thresh_max": 0.7}},
    {"MakeShrinkMap": {"shrink_ratio": 0.4, "min_text_size": 3}},
], None)

import numpy as np
n_samples = 0
n_with_pos_threshold_mask = 0
n_with_pos_shrink_map = 0
sum_threshold_mask_pos = 0
sum_shrink_map_pos = 0

with open(r"E:\OCR_Project\data\det_dataset_full\det_train_bench.txt", encoding="utf-8") as f:
    lines = f.readlines()

import random
random.seed(0)
random.shuffle(lines)

for line in lines[:200]:
    line = line.rstrip("\n")
    parts = line.split("\t")
    if len(parts) != 2:
        continue
    img_rel, label_json = parts
    img_path = os.path.join(r"E:\OCR_Project", img_rel)
    with open(img_path, "rb") as g:
        img_bytes = g.read()
    data = {"image": img_bytes, "label": label_json}
    try:
        out = transform(data, ops)
    except Exception as e:
        print("transform error:", e)
        continue
    if out is None:
        continue
    n_samples += 1
    tm = out["threshold_mask"]
    sm = out["shrink_map"]
    n_thr_pos = (tm > 0).sum()
    n_shr_pos = (sm > 0).sum()
    if n_thr_pos > 0:
        n_with_pos_threshold_mask += 1
    if n_shr_pos > 0:
        n_with_pos_shrink_map += 1
    sum_threshold_mask_pos += n_thr_pos
    sum_shrink_map_pos += n_shr_pos

print(f"n_samples={n_samples}")
print(f"n_with_pos_threshold_mask={n_with_pos_threshold_mask}")
print(f"n_with_pos_shrink_map={n_with_pos_shrink_map}")
print(f"avg threshold_mask positives = {sum_threshold_mask_pos / max(1,n_samples):.1f}")
print(f"avg shrink_map     positives = {sum_shrink_map_pos / max(1,n_samples):.1f}")
