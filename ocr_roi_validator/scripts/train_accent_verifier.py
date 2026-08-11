"""Fit the accent verifier on synthetic glyphs and freeze it.

The model is fitted on the training split only. Thresholds are then chosen on
the validation split, under the rule that matters most: never call a real
accent an ``e``. The threshold search maximises how many hallucinated accents
are caught *subject to* zero false corrections on validation, and abstains
everywhere else.

The holdout split is not touched here. It exists to measure the frozen model
once, afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATOR_ROOT))

from ocr_roi_validator.accent_verifier import (  # noqa: E402
    FEATURE_NAMES,
    AccentModel,
    extract_features,
)


def load_split(directory: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Return (features, labels, kept samples). Label 1 means visually `e`."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    features = []
    labels = []
    kept = []
    for sample in manifest["samples"]:
        image = cv2.imread(str(directory / sample["file"]))
        vector = extract_features(image)
        if vector is None:
            continue
        features.append(vector)
        labels.append(1.0 if sample["visual_label"] == "e" else 0.0)
        kept.append(sample)
    return np.asarray(features), np.asarray(labels), kept


def standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (features - mean) / scale, mean, scale


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    iterations: int = 4000,
    learning_rate: float = 0.35,
    l2: float = 1e-3,
) -> tuple[np.ndarray, float]:
    """Plain gradient descent; the problem is tiny and this keeps deps at numpy."""
    samples, dimensions = features.shape
    weights = np.zeros(dimensions)
    bias = 0.0
    for _ in range(iterations):
        z = features @ weights + bias
        predictions = 1.0 / (1.0 + np.exp(-z))
        error = predictions - labels
        weights -= learning_rate * (features.T @ error / samples + l2 * weights)
        bias -= learning_rate * float(error.mean())
    return weights, bias


def choose_thresholds(
    probabilities: np.ndarray, labels: np.ndarray
) -> tuple[float, float, dict]:
    """Pick thresholds that make zero false corrections on validation.

    ``absent_threshold`` must sit strictly above every probability assigned to
    a real accent, so no genuine accent can be converted. A margin is added so
    the boundary is not flush against the worst validation sample.
    """
    accent_probabilities = probabilities[labels == 0.0]
    bare_probabilities = probabilities[labels == 1.0]

    highest_accent = float(accent_probabilities.max()) if accent_probabilities.size else 0.0
    absent_threshold = min(0.999, highest_accent + 0.02)
    # Above this we would claim "accent present"; keep it conservative too.
    lowest_bare = float(bare_probabilities.min()) if bare_probabilities.size else 1.0
    present_threshold = max(0.001, min(0.5, lowest_bare - 0.02))

    coverage = float((bare_probabilities >= absent_threshold).mean()) if bare_probabilities.size else 0.0
    false_corrections = int((accent_probabilities >= absent_threshold).sum())
    return absent_threshold, present_threshold, {
        "validation_highest_accent_probability": highest_accent,
        "validation_bare_coverage_at_threshold": coverage,
        "validation_false_corrections": false_corrections,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--out-model", type=Path, required=True)
    parser.add_argument("--version", default="accent-v1")
    args = parser.parse_args()

    train_features, train_labels, _ = load_split(args.train_dir)
    validation_features, validation_labels, _ = load_split(args.validation_dir)
    print(f"train      : {len(train_labels)} glyphs "
          f"({int(train_labels.sum())} e, {int((1 - train_labels).sum())} accent)")
    print(f"validation : {len(validation_labels)} glyphs "
          f"({int(validation_labels.sum())} e, "
          f"{int((1 - validation_labels).sum())} accent)")

    scaled, mean, scale = standardize(train_features)
    weights, bias = fit_logistic(scaled, train_labels)

    # Fold standardization into the weights so inference needs no extra state.
    folded_weights = weights / scale
    folded_bias = float(bias - np.dot(weights, mean / scale))

    model = AccentModel(
        weights=tuple(folded_weights),
        bias=folded_bias,
        absent_threshold=0.5,
        present_threshold=0.5,
        version=args.version,
    )

    train_probabilities = np.array(
        [model.probability_absent(f) for f in train_features]
    )
    validation_probabilities = np.array(
        [model.probability_absent(f) for f in validation_features]
    )
    train_accuracy = float(((train_probabilities >= 0.5) == train_labels).mean())
    print(f"train accuracy at 0.5      : {train_accuracy:.4f}")

    absent_threshold, present_threshold, stats = choose_thresholds(
        validation_probabilities, validation_labels
    )
    model = AccentModel(
        weights=model.weights,
        bias=model.bias,
        absent_threshold=absent_threshold,
        present_threshold=present_threshold,
        version=args.version,
    )
    print(f"absent_threshold  : {absent_threshold:.4f}")
    print(f"present_threshold : {present_threshold:.4f}")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    payload = json.dumps(model.to_dict(), ensure_ascii=False, indent=2)
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    args.out_model.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    print(f"\nfeature order : {list(FEATURE_NAMES)}")
    print(f"model sha256  : {digest}")
    print(f"wrote {args.out_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
