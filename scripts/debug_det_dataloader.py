"""Replicate the actual SimpleDataSet path used by training."""
import sys, os
sys.path.insert(0, r"E:\OCR_Project\PaddleOCR")
os.chdir(r"E:\OCR_Project\PaddleOCR")

import yaml
import numpy as np
from ppocr.data import build_dataloader
from ppocr.utils.logging import get_logger

with open("configs/det/finetune_det_v0.yml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# small bench file
config["Train"]["dataset"]["label_file_list"] = [
    "E:/OCR_Project/data/det_dataset_full/det_train_bench.txt"
]
config["Train"]["loader"]["num_workers"] = 0
config["Train"]["loader"]["batch_size_per_card"] = 4

logger = get_logger()
loader = build_dataloader(config, "Train", "gpu", logger)

n_batches = 0
n_pos_shrink = 0
n_total = 0
for batch in loader:
    # batch is list of tensors: image, threshold_map, threshold_mask, shrink_map, shrink_mask
    print(f"batch types: {[type(x).__name__ for x in batch]}")
    print(f"batch shapes: {[tuple(x.shape) for x in batch]}")
    image, thr_map, thr_mask, shr_map, shr_mask = batch
    shr_np = shr_map.numpy()
    shr_mask_np = shr_mask.numpy()
    thr_map_np = thr_map.numpy()
    thr_mask_np = thr_mask.numpy()
    for i in range(shr_np.shape[0]):
        n_total += 1
        pos = (shr_np[i] * shr_mask_np[i]).sum()
        thr_pos = (thr_mask_np[i] > 0).sum()
        if pos > 0:
            n_pos_shrink += 1
        print(f"  sample {i}: shrink positives={int(shr_np[i].sum())}, "
              f"masked positives={int(pos)}, thr_mask pos={int(thr_pos)}, "
              f"image min={image.numpy()[i].min():.3f} max={image.numpy()[i].max():.3f}")
    n_batches += 1
    if n_batches >= 5:
        break

print(f"\nTotal samples seen: {n_total}, with positive shrink: {n_pos_shrink}")
