"""Image-based e / é / unknown verification for a single glyph.

The text router cannot tell a hallucinated accent from a real one: both look
like the same string difference. This module answers the question from pixels.

Why not measure ink height directly
-----------------------------------
An earlier prototype compared the glyph's top edge against the *line's* ink
extent. That failed badly: any capital or ascender elsewhere on the line raises
the reference band and buries a genuine accent, so it erased about half of all
real accents on a font holdout.

The fix is to make the measurement local. The features below are computed from
the glyph crop alone and are scale-free, so neither the line's other characters
nor the font size can shift them.

Model
-----
A small logistic classifier over those features, fitted on synthetic glyphs only
and shipped as frozen coefficients. Inference needs nothing but numpy, so the
runtime has no new dependency and the decision is reproducible.

Scope and safety
----------------
* Only ``é`` -> ``e`` is ever proposed. This module is asked exclusively about
  glyphs the recognizer already read as ``é``; it never converts ``e`` to ``é``.
* A verdict of ``é`` or ``unknown`` leaves the baseline untouched. The abstain
  band is deliberately wide: refusing to answer is safe, guessing is not.
* i/l, digits, timers and punctuation are out of scope.
* No expected text, reference string or dictionary is an input.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .accent_localization import LocalizationConfig, assess_localization

__all__ = [
    "ACCENT_PRESENT",
    "ACCENT_ABSENT",
    "UNKNOWN",
    "AccentVerdict",
    "AccentModel",
    "extract_features",
    "FEATURE_NAMES",
    "load_model",
    "verify_accent_glyph",
    "verify_accent_in_line",
]

ACCENT_PRESENT = "é"
ACCENT_ABSENT = "e"
UNKNOWN = "unknown"

FEATURE_NAMES = (
    "aspect_ratio",          # ink height / ink width
    "upper_ink_fraction",    # ink in the top third, over total ink
    "upper_gap_fraction",    # tallest ink-free run in the upper half
    "top_row_density",       # ink density of the topmost ink row
    "upper_width_fraction",  # width of upper-third ink, over glyph ink width
    "vertical_centroid",     # centre of ink mass, 0 = top
)

MIN_GLYPH_WIDTH = 4
MIN_INK_HEIGHT = 7
MIN_INK_PIXELS = 12

DEFAULT_MODEL_PATH = Path(__file__).with_name("accent_model.json")


@dataclass(frozen=True)
class AccentVerdict:
    """Outcome of inspecting one glyph."""

    verdict: str
    probability_absent: float
    reason: str

    @property
    def is_accent_absent(self) -> bool:
        """True only for a confident `e` -- the sole case that may change text."""
        return self.verdict == ACCENT_ABSENT


@dataclass(frozen=True)
class AccentModel:
    """Frozen logistic model over :data:`FEATURE_NAMES`."""

    weights: tuple[float, ...]
    bias: float
    absent_threshold: float
    present_threshold: float
    version: str = "unversioned"

    def probability_absent(self, features: np.ndarray) -> float:
        z = float(np.dot(np.asarray(self.weights), features) + self.bias)
        return 1.0 / (1.0 + np.exp(-z))

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "feature_names": list(FEATURE_NAMES),
            "weights": list(self.weights),
            "bias": self.bias,
            "absent_threshold": self.absent_threshold,
            "present_threshold": self.present_threshold,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "AccentModel":
        names = tuple(payload.get("feature_names", FEATURE_NAMES))
        if names != FEATURE_NAMES:
            raise ValueError(
                f"model features {names} do not match {FEATURE_NAMES}"
            )
        return cls(
            weights=tuple(float(w) for w in payload["weights"]),
            bias=float(payload["bias"]),
            absent_threshold=float(payload["absent_threshold"]),
            present_threshold=float(payload["present_threshold"]),
            version=str(payload.get("version", "unversioned")),
        )


def load_model(path: Path | None = None) -> AccentModel | None:
    path = path or DEFAULT_MODEL_PATH
    if not Path(path).is_file():
        return None
    return AccentModel.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return image[..., :3].mean(axis=2)
    return image.astype(np.float64)


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Boolean ink mask, handling light-on-dark and dark-on-light alike."""
    low, high = float(gray.min()), float(gray.max())
    if high - low < 12.0:
        return np.zeros(gray.shape, dtype=bool)
    midpoint = (low + high) / 2.0
    bright = gray >= midpoint
    # Ink is the sparser polarity: text covers less area than background.
    return bright if bright.mean() < 0.5 else ~bright


def extract_features(glyph: np.ndarray) -> np.ndarray | None:
    """Scale-free shape features for one glyph crop, or None if unusable.

    Everything is measured inside the glyph's own ink bounding box and divided
    by that box's size, so the values do not depend on font size, line height
    or how generous the crop was.
    """
    if glyph is None or glyph.size == 0:
        return None
    height, width = glyph.shape[:2]
    if width < MIN_GLYPH_WIDTH:
        return None

    ink = _ink_mask(_to_grayscale(glyph))
    if ink.sum() < MIN_INK_PIXELS:
        return None

    rows = np.where(ink.any(axis=1))[0]
    columns = np.where(ink.any(axis=0))[0]
    top, bottom = int(rows[0]), int(rows[-1])
    left, right = int(columns[0]), int(columns[-1])
    ink_height = bottom - top + 1
    ink_width = right - left + 1
    if ink_height < MIN_INK_HEIGHT or ink_width < MIN_GLYPH_WIDTH:
        return None

    box = ink[top : bottom + 1, left : right + 1]
    total_ink = float(box.sum())
    if total_ink <= 0:
        return None

    third = max(1, int(round(ink_height / 3.0)))
    upper = box[:third, :]
    upper_ink = float(upper.sum())

    # Tallest ink-free horizontal run in the upper half: an accent is separated
    # from the letter body by a gap, at least at larger sizes.
    half = max(1, ink_height // 2)
    row_has_ink = box[:half, :].any(axis=1)
    longest_gap = current = 0
    for has_ink in row_has_ink:
        current = 0 if has_ink else current + 1
        longest_gap = max(longest_gap, current)

    upper_columns = np.where(upper.any(axis=0))[0]
    upper_width = len(upper_columns)

    row_indices = np.arange(box.shape[0])
    centroid = float((box.sum(axis=1) * row_indices).sum() / total_ink)

    return np.array(
        [
            ink_height / float(ink_width),
            upper_ink / total_ink,
            longest_gap / float(ink_height),
            float(box[0, :].sum()) / float(ink_width),
            upper_width / float(ink_width),
            centroid / float(ink_height),
        ],
        dtype=np.float64,
    )


def verify_accent_glyph(
    glyph: np.ndarray,
    model: AccentModel | None = None,
) -> AccentVerdict:
    """Decide whether ``glyph`` carries an acute accent.

    ``glyph`` is a crop around a single character, as located by CTC alignment.
    There is deliberately no parameter for the expected character.

    Without a fitted model the answer is always :data:`UNKNOWN`, so a missing
    model file degrades to "change nothing" rather than to guessing.

    This judges the crop as given. Callers that have the surrounding line
    should prefer :func:`verify_accent_in_line`, which first checks that the
    crop really isolates one glyph.
    """
    if model is None:
        model = load_model()
    if model is None:
        return AccentVerdict(UNKNOWN, 0.0, "no_model_available")

    features = extract_features(glyph)
    if features is None:
        return AccentVerdict(UNKNOWN, 0.0, "glyph_unmeasurable")

    probability = model.probability_absent(features)
    if probability >= model.absent_threshold:
        return AccentVerdict(ACCENT_ABSENT, probability, "confident_no_accent")
    if probability <= model.present_threshold:
        return AccentVerdict(ACCENT_PRESENT, probability, "confident_accent")
    return AccentVerdict(UNKNOWN, probability, "below_confidence_margin")


def verify_accent_in_line(
    line_image: np.ndarray,
    x0: int,
    x1: int,
    model: AccentModel | None = None,
    median_span_width: float | None = None,
    localization: LocalizationConfig | None = None,
) -> AccentVerdict:
    """Verify a glyph, refusing whenever the crop cannot be trusted.

    Three things must agree before a correction is licensed:

    1. the span looks like one isolated character (localization gate),
    2. the classifier is confident there is no accent, and
    3. that verdict survives small changes to the crop bounds.

    The third check matters because CTC spans are approximate: a verdict that
    flips when the crop moves by a pixel or two was resting on a boundary
    artefact rather than on the glyph. Any disagreement yields
    :data:`UNKNOWN`, which leaves the baseline untouched.
    """
    if model is None:
        model = load_model()
    if model is None:
        return AccentVerdict(UNKNOWN, 0.0, "no_model_available")

    config = localization or LocalizationConfig()
    report = assess_localization(line_image, x0, x1, median_span_width, config)
    if not report.usable:
        return AccentVerdict(UNKNOWN, 0.0, f"localization:{','.join(report.reasons)}")

    width = line_image.shape[1]
    primary = verify_accent_glyph(line_image[:, x0:x1], model)
    if primary.verdict != ACCENT_ABSENT:
        # Only an "absent" verdict can change text, so only it needs the
        # extra scrutiny below.
        return primary

    jitter = max(1, int(config.jitter_pixels))
    for left_shift, right_shift in (
        (-jitter, 0), (jitter, 0), (0, -jitter), (0, jitter), (-jitter, jitter),
    ):
        shifted_x0 = max(0, x0 + left_shift)
        shifted_x1 = min(width, x1 + right_shift)
        if shifted_x1 - shifted_x0 < 4:
            return AccentVerdict(
                UNKNOWN, primary.probability_absent, "jitter_span_degenerate"
            )
        variant = verify_accent_glyph(line_image[:, shifted_x0:shifted_x1], model)
        if variant.verdict != ACCENT_ABSENT:
            return AccentVerdict(
                UNKNOWN,
                primary.probability_absent,
                f"jitter_disagreement:{variant.verdict}",
            )

    return AccentVerdict(
        ACCENT_ABSENT, primary.probability_absent, "confident_no_accent_localized"
    )
