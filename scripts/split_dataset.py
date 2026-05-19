"""
Split a synthetic dataset into demo / test / train subsets.

Reads <source>/labels.jsonl + <source>/images/, produces:
    <output>/demo/{images, labels.jsonl}
    <output>/test/{images, labels.jsonl}
    <output>/train/{images, labels.jsonl}

Reproducible via fixed seed. Disjoint sets. Files are copied (source preserved).
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path


def load_labels(src: Path):
    with src.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_split(items, dst: Path, src_dir: Path):
    img_dir = dst / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    out_labels = []
    for item in items:
        rel = Path(item["image_path"])
        src_img = src_dir / rel
        if not src_img.exists():
            raise FileNotFoundError(f"missing image: {src_img}")
        dst_img = img_dir / rel.name
        shutil.copy2(src_img, dst_img)
        new_item = dict(item)
        new_item["image_path"] = f"images/{rel.name}"
        out_labels.append(new_item)
    with (dst / "labels.jsonl").open("w", encoding="utf-8") as f:
        for it in out_labels:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="dir containing labels.jsonl + images/")
    p.add_argument("--output", required=True, help="output root for demo/test/train")
    p.add_argument("--demo-count", type=int, default=9)
    p.add_argument("--test-count", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src_dir = Path(args.source)
    out_dir = Path(args.output)
    labels_path = src_dir / "labels.jsonl"
    if not labels_path.exists():
        sys.exit(f"labels.jsonl not found in {src_dir}")

    labels = load_labels(labels_path)
    total = len(labels)
    if args.demo_count + args.test_count >= total:
        sys.exit(f"demo+test ({args.demo_count + args.test_count}) >= total ({total})")

    rng = random.Random(args.seed)
    indices = list(range(total))
    rng.shuffle(indices)

    demo_idx = set(indices[: args.demo_count])
    test_idx = set(indices[args.demo_count : args.demo_count + args.test_count])
    train_idx = set(indices[args.demo_count + args.test_count :])

    demo = [labels[i] for i in indices[: args.demo_count]]
    test = [labels[i] for i in indices[args.demo_count : args.demo_count + args.test_count]]
    train = [labels[i] for i in indices[args.demo_count + args.test_count :]]

    print(f"source: {src_dir} ({total} items)")
    print(f"  demo  : {len(demo)}")
    print(f"  test  : {len(test)}")
    print(f"  train : {len(train)}")

    write_split(demo, out_dir / "demo", src_dir)
    write_split(test, out_dir / "test", src_dir)
    write_split(train, out_dir / "train", src_dir)

    assert demo_idx.isdisjoint(test_idx)
    assert demo_idx.isdisjoint(train_idx)
    assert test_idx.isdisjoint(train_idx)

    print(f"\nWritten to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
