"""Compare baseline RapidOCR vs our fine-tuned rec ONNX on a few test images."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from PIL import Image
from ocr_runner import ocr_image
from metrics import cer

custom_model = ROOT / "app" / "models" / "rec_v0.onnx"
custom_keys = ROOT / "app" / "models" / "ppocr_keys.txt"

test_dir = ROOT / "dataset" / "test" / "images"
labels_path = ROOT / "dataset" / "test" / "labels.jsonl"

import json
labels = {}
with labels_path.open(encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        labels[Path(d["image_path"]).name] = d["raw_text"]

samples = sorted(test_dir.glob("*.png"))[:8]
print(f"{'file':<50} {'CER(base)':>10} {'CER(ours)':>10}")
print("-" * 75)
total_b = total_o = 0.0
for p in samples:
    name = p.name
    ref = labels.get(name, "")
    img = Image.open(p)
    base = ocr_image(img)
    ours = ocr_image(img, custom_model, custom_keys)
    c_b = cer(ref, base.text)
    c_o = cer(ref, ours.text)
    total_b += c_b
    total_o += c_o
    print(f"{name:<50} {c_b:>10.3f} {c_o:>10.3f}")

n = len(samples)
print("-" * 75)
print(f"{'MEAN':<50} {total_b/n:>10.3f} {total_o/n:>10.3f}")

# show details for first 3 cases
print()
for p in samples[:3]:
    name = p.name
    ref = labels.get(name, "")
    img = Image.open(p)
    base = ocr_image(img)
    ours = ocr_image(img, custom_model, custom_keys)
    print(f"\n=== {name} ===")
    print(f"REF : {ref!r}")
    print(f"BASE: {base.text!r}")
    print(f"OURS: {ours.text!r}")
