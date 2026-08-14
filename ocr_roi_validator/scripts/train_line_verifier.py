"""Train the target-query line verifier and export it to ONNX.

Runs in a separate PyTorch environment; the product venv is untouched and
inference is ONNX Runtime only.

The model sees a whole recognizer line and a query naming which character
position is being asked about. An attention head over the line's column
features locates that character; an accent head reads the attended features and
answers BARE_E, ACCENT_PRESENT or UNKNOWN.

Why attention rather than a crop: every previous approach cut a box from CTC
timesteps, and all its estimates shared the same correlated error, so
"consensus" still pointed at a neighbour 56 times. Letting the model attend
over the line means the position decision is made from pixels and can be
supervised directly against the renderer's own character centres.

The ordinal query carries position only. No character, word or decoded string
ever reaches the network, so it cannot answer from spelling.

UNKNOWN is a trained class, not a threshold artefact: cases where the query is
genuinely ambiguous are labelled UNKNOWN so the model learns to abstain rather
than being forced to guess.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATOR_ROOT))

from ocr_roi_validator.line_verifier_input import (  # noqa: E402
    LineVerifierInputConfig,
)

# Class order is part of the frozen contract.
CLASS_ACCENT_PRESENT = 0
CLASS_BARE_E = 1
CLASS_UNKNOWN = 2
CLASS_NAMES = ("ACCENT_PRESENT", "BARE_E", "UNKNOWN")


def build_network(torch, config: LineVerifierInputConfig):
    """Small CPU model: line encoder, query-conditioned attention, two heads."""
    nn = torch.nn

    class LineVerifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Encoder keeps horizontal resolution while collapsing height, so
            # each output column corresponds to a slice of the line.
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.MaxPool2d((2, 1)),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d((2, 1)),
                nn.Conv2d(32, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
                nn.MaxPool2d((2, 1)),
                nn.Conv2d(48, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, None)),
            )
            self.query_encoder = nn.Sequential(
                nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 64),
            )
            # Attention logits per column, conditioned on the query.
            self.attention = nn.Sequential(
                nn.Conv1d(128, 64, 1), nn.ReLU(), nn.Conv1d(64, 1, 1),
            )
            self.accent_head = nn.Sequential(
                nn.Linear(64, 48), nn.ReLU(), nn.Dropout(0.2), nn.Linear(48, 3),
            )

        def forward(self, planes, query):
            features = self.encoder(planes).squeeze(2)          # (N, 64, W')
            columns = features.shape[2]
            query_vector = self.query_encoder(query)            # (N, 64)
            broadcast = query_vector.unsqueeze(2).expand(-1, -1, columns)
            joint = torch.cat([features, broadcast], dim=1)     # (N, 128, W')
            attention_logits = self.attention(joint).squeeze(1)  # (N, W')
            weights = torch.softmax(attention_logits, dim=1)
            attended = (features * weights.unsqueeze(1)).sum(dim=2)  # (N, 64)
            return self.accent_head(attended), attention_logits

    return LineVerifier()


def load_split(directory: Path):
    """Load a prepared split written by build_line_dataset_v1.py."""
    payload = np.load(directory / "tensors.npz")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    return (
        payload["planes"].astype(np.float32),
        payload["query"].astype(np.float32),
        payload["label"].astype(np.int64),
        payload["attention_target"].astype(np.float32),
        manifest,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--out-onnx", type=Path, required=True)
    parser.add_argument("--out-config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.5e-3)
    parser.add_argument("--attention-weight", type=float, default=2.0)
    parser.add_argument(
        "--accent-loss-weight", type=float, default=6.0,
        help="cost multiplier for calling a real accent bare",
    )
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--version", default="line-verifier-v1")
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument("--reference-logits", type=Path)
    args = parser.parse_args()

    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(4)

    config = LineVerifierInputConfig()
    train_planes, train_query, train_label, train_attention, _ = load_split(
        args.train_dir)
    cal_planes, cal_query, cal_label, cal_attention, _ = load_split(
        args.calibration_dir)
    print(f"train       : {len(train_label)} examples")
    print(f"calibration : {len(cal_label)} examples")
    for name, labels in (("train", train_label), ("calibration", cal_label)):
        counts = {CLASS_NAMES[i]: int((labels == i).sum()) for i in range(3)}
        print(f"  {name} classes: {counts}")

    network = build_network(torch, config)
    parameters = sum(p.numel() for p in network.parameters())
    print(f"parameters  : {parameters}")

    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)
    weights = torch.tensor(
        [args.accent_loss_weight, 1.0, 1.0], dtype=torch.float32)
    accent_loss = torch.nn.CrossEntropyLoss(weight=weights)

    planes_tensor = torch.from_numpy(train_planes)
    query_tensor = torch.from_numpy(train_query)
    label_tensor = torch.from_numpy(train_label)
    attention_tensor = torch.from_numpy(train_attention)
    order = np.arange(len(train_label))

    for epoch in range(args.epochs):
        network.train()
        np.random.shuffle(order)
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            optimizer.zero_grad()
            logits, attention_logits = network(
                planes_tensor[batch], query_tensor[batch])
            loss = accent_loss(logits, label_tensor[batch])
            # Supervise attention against the renderer's own character centre,
            # so localization is learned from pixels rather than inherited from
            # the CTC map.
            target = attention_tensor[batch]
            columns = attention_logits.shape[1]
            if target.shape[1] != columns:
                index = torch.linspace(0, target.shape[1] - 1, columns).long()
                target = target[:, index]
            target = target / target.sum(dim=1, keepdim=True).clamp(min=1e-6)
            attention_log = torch.log_softmax(attention_logits, dim=1)
            loss = loss + args.attention_weight * (
                -(target * attention_log).sum(dim=1).mean())
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch)
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:>3d}  loss {total / len(order):.5f}")

    network.eval()
    with torch.no_grad():
        cal_logits, _ = network(torch.from_numpy(cal_planes),
                                torch.from_numpy(cal_query))
        cal_probabilities = torch.softmax(cal_logits, dim=1).numpy()

    # The operating threshold must sit above every probability the model gives
    # BARE_E on a glyph that really carries an accent. No cap: if the classes
    # are not separable the threshold goes past 1.0 and coverage falls to zero,
    # which is the safe outcome.
    accent_mask = cal_label == CLASS_ACCENT_PRESENT
    bare_probabilities = cal_probabilities[:, CLASS_BARE_E]
    worst_accent = float(bare_probabilities[accent_mask].max()) \
        if accent_mask.any() else 0.0
    threshold = float(np.nextafter(np.float32(worst_accent), np.float32(1.0)))
    if not threshold > worst_accent:
        threshold = float("inf")
    bare_mask = cal_label == CLASS_BARE_E
    coverage = float((bare_probabilities[bare_mask] >= threshold).mean()) \
        if bare_mask.any() else 0.0
    false_corrections = int((bare_probabilities[accent_mask] >= threshold).sum())
    print(f"\nbare_threshold : {threshold:.9f}")
    print(f"  worst accent probability   : {worst_accent:.9f}")
    print(f"  calibration coverage       : {coverage:.4f}")
    print(f"  calibration false corrections: {false_corrections}")
    if false_corrections != 0:
        print("refusing to export: calibration false corrections are non-zero",
              file=sys.stderr)
        return 1

    args.out_onnx.parent.mkdir(parents=True, exist_ok=True)
    if args.reference_logits:
        reference_planes = cal_planes[:400]
        reference_query = cal_query[:400]
        with torch.no_grad():
            reference_logits, reference_attention = network(
                torch.from_numpy(reference_planes),
                torch.from_numpy(reference_query))
        np.savez_compressed(
            args.reference_logits, planes=reference_planes,
            query=reference_query, logits=reference_logits.numpy(),
            attention=reference_attention.numpy())
        print(f"wrote {args.reference_logits}")

    dummy_planes = torch.zeros(1, 3, config.height, config.width)
    dummy_query = torch.zeros(1, 2)
    torch.onnx.export(
        network, (dummy_planes, dummy_query), str(args.out_onnx),
        input_names=["planes", "query"],
        output_names=["logits", "attention"],
        dynamic_axes={"planes": {0: "batch"}, "query": {0: "batch"},
                      "logits": {0: "batch"}, "attention": {0: "batch"}},
        opset_version=args.opset,
    )
    onnx_bytes = args.out_onnx.read_bytes()

    payload = {
        "version": args.version,
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "onnx_sha256": hashlib.sha256(onnx_bytes).hexdigest(),
        "onnx_bytes": len(onnx_bytes),
        "parameters": parameters,
        "opset": args.opset,
        "class_names": list(CLASS_NAMES),
        "input_config": config.as_dict(),
        "bare_threshold": threshold,
        "calibration": {
            "worst_accent_bare_probability": worst_accent,
            "coverage": coverage,
            "false_corrections": false_corrections,
        },
        "training": {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "cuda": torch.cuda.is_available(), "seed": args.seed,
            "deterministic": True, "epochs": args.epochs,
            "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "attention_weight": args.attention_weight,
            "accent_loss_weight": args.accent_loss_weight,
            "train_examples": int(len(train_label)),
            "calibration_examples": int(len(cal_label)),
        },
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    args.out_config.write_text(text, encoding="utf-8")
    print(f"\nonnx  sha256 : {payload['onnx_sha256']}")
    print(f"config sha256: {hashlib.sha256(text.encode()).hexdigest()}")
    print(f"onnx bytes   : {len(onnx_bytes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
