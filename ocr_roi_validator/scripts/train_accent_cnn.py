"""Train the conservative accent CNN and export it to ONNX.

Runs in a separate PyTorch environment; the product venv is never touched and
inference uses ONNX Runtime only. The network is deliberately small: the
decision is a local texture question, and a big model would be slower on CPU
without being more trustworthy.

The objective is not balanced accuracy. Calling a real accent "absent" would
let OCR erase a diacritic that is genuinely on screen, so that error is
weighted far more heavily than the reverse, and the operating threshold is then
chosen on validation to drive it to zero outright. Coverage is whatever
survives that.

Labels come from what was rendered, never from OCR. No real UI image is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATOR_ROOT))

from ocr_roi_validator.accent_cnn_input import (  # noqa: E402
    AccentInputConfig,
    prepare_cnn_input,
)


def build_network(torch):
    """A two-channel CNN over the glyph and its accent band.

    Channel 0 is the whole glyph, channel 1 the upper band where a diacritic
    would sit. Keeping them as separate channels lets the first convolution
    compare them directly instead of having to locate the band itself.
    """
    nn = torch.nn

    class AccentNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(2, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Dropout(0.25), nn.Linear(48, 32), nn.ReLU(),
                nn.Linear(32, 2),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return AccentNet()


def load_split(directory: Path, config: AccentInputConfig):
    """Return (X, y, kept) where y=1 means the glyph was drawn WITHOUT an accent."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    tensors, labels, kept = [], [], []
    for sample in manifest["samples"]:
        image = cv2.imread(str(directory / sample["file"]))
        prepared = prepare_cnn_input(image, config)
        if prepared is None:
            continue
        tensors.append(prepared[0])
        labels.append(1.0 if sample["visual_label"] == "e" else 0.0)
        kept.append(sample)
    if not tensors:
        return np.zeros((0, 2, config.height, config.width), np.float32), \
               np.zeros((0,), np.float32), []
    return np.stack(tensors).astype(np.float32), np.asarray(labels, np.float32), kept


def choose_threshold(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, dict]:
    """Pick the absent-threshold that yields zero false corrections.

    It must sit strictly above every probability the model assigns to a glyph
    that really has an accent, so no genuine accent can be converted. A small
    margin keeps the boundary off the worst validation sample.
    """
    accent_probabilities = probabilities[labels == 0.0]
    bare_probabilities = probabilities[labels == 1.0]
    highest_accent = float(accent_probabilities.max()) if accent_probabilities.size else 0.0
    threshold = min(0.9995, highest_accent + 0.005)
    coverage = (
        float((bare_probabilities >= threshold).mean()) if bare_probabilities.size else 0.0
    )
    return threshold, {
        "validation_highest_accent_probability": highest_accent,
        "validation_bare_coverage_at_threshold": coverage,
        "validation_false_corrections": int(
            (accent_probabilities >= threshold).sum()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--out-onnx", type=Path, required=True)
    parser.add_argument("--out-config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument(
        "--accent-loss-weight", type=float, default=6.0,
        help="cost multiplier for calling a real accent absent",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--version", default="accent-v3")
    parser.add_argument("--opset", type=int, default=11)
    args = parser.parse_args()

    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(4)

    config = AccentInputConfig()
    train_x, train_y, _ = load_split(args.train_dir, config)
    validation_x, validation_y, _ = load_split(args.validation_dir, config)
    print(f"train      : {len(train_y)} glyphs "
          f"({int(train_y.sum())} bare, {int((1 - train_y).sum())} accent)")
    print(f"validation : {len(validation_y)} glyphs "
          f"({int(validation_y.sum())} bare, {int((1 - validation_y).sum())} accent)")
    if len(train_y) == 0 or len(validation_y) == 0:
        print("empty split", file=sys.stderr)
        return 1

    network = build_network(torch)
    parameters = sum(p.numel() for p in network.parameters())
    print(f"parameters : {parameters}")

    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)
    # Asymmetric cost: index 0 is "has accent", and mislabelling it is the
    # error that would erase a real diacritic.
    weights = torch.tensor([args.accent_loss_weight, 1.0], dtype=torch.float32)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    train_tensor = torch.from_numpy(train_x)
    train_labels = torch.from_numpy(train_y).long()
    order = np.arange(len(train_y))

    for epoch in range(args.epochs):
        network.train()
        np.random.shuffle(order)
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            optimizer.zero_grad()
            logits = network(train_tensor[batch])
            loss = criterion(logits, train_labels[batch])
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch)
        if (epoch + 1) % 4 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:>3d}  loss {total / len(order):.5f}")

    network.eval()
    with torch.no_grad():
        validation_logits = network(torch.from_numpy(validation_x))
        validation_probabilities = torch.softmax(validation_logits, dim=1)[:, 1].numpy()
        train_logits = network(train_tensor)
        train_probabilities = torch.softmax(train_logits, dim=1)[:, 1].numpy()

    train_accuracy = float(((train_probabilities >= 0.5) == train_y).mean())
    validation_accuracy = float(
        ((validation_probabilities >= 0.5) == validation_y).mean()
    )
    print(f"train accuracy @0.5      : {train_accuracy:.4f}")
    print(f"validation accuracy @0.5 : {validation_accuracy:.4f}")

    # Confusion at the plain 0.5 boundary, before the safety threshold.
    predicted = validation_probabilities >= 0.5
    confusion = {
        "accent_as_accent": int(((validation_y == 0) & (~predicted)).sum()),
        "accent_as_bare": int(((validation_y == 0) & predicted).sum()),
        "bare_as_bare": int(((validation_y == 1) & predicted).sum()),
        "bare_as_accent": int(((validation_y == 1) & (~predicted)).sum()),
    }
    print(f"validation confusion @0.5 : {confusion}")

    threshold, stats = choose_threshold(validation_probabilities, validation_y)
    print(f"absent_threshold : {threshold:.6f}")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    args.out_onnx.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 2, config.height, config.width)
    torch.onnx.export(
        network, dummy, str(args.out_onnx),
        input_names=["glyph"], output_names=["logits"],
        dynamic_axes={"glyph": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )
    onnx_bytes = args.out_onnx.read_bytes()
    onnx_digest = hashlib.sha256(onnx_bytes).hexdigest()

    payload = {
        "version": args.version,
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "onnx_file": args.out_onnx.name,
        "onnx_sha256": onnx_digest,
        "onnx_bytes": len(onnx_bytes),
        "parameters": parameters,
        "opset": args.opset,
        "input_config": config.as_dict(),
        "absent_threshold": threshold,
        # Anything below this is treated as a positive accent veto.
        "present_threshold": 0.05,
        "training": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "seed": args.seed,
            "deterministic": True,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "accent_loss_weight": args.accent_loss_weight,
            "train_dir": str(args.train_dir),
            "validation_dir": str(args.validation_dir),
            "train_glyphs": int(len(train_y)),
            "validation_glyphs": int(len(validation_y)),
        },
        "validation": {
            "accuracy_at_0p5": validation_accuracy,
            "confusion_at_0p5": confusion,
            **stats,
        },
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    args.out_config.write_text(text, encoding="utf-8")
    print(f"\nonnx sha256   : {onnx_digest}")
    print(f"config sha256 : {hashlib.sha256(text.encode()).hexdigest()}")
    print(f"wrote {args.out_onnx}")
    print(f"wrote {args.out_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
