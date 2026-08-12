"""accent-v3: the full decision path from a glyph crop to a verdict.

This is the component the F1 gate measures. It is not a classifier wrapper: a
correction is licensed only when several independent checks agree, and any one
of them abstaining is enough to keep the baseline.

Order of checks, each able to stop the decision:

1. the localization guard must accept the span as one isolated glyph,
2. the crop must survive input preparation,
3. the CNN must be confident the accent is absent,
4. that verdict must hold on tight, expanded and shifted views of the crop,
5. no view may return a positive accent verdict as a veto.

Only ``é`` -> ``e`` can ever be proposed. The reverse is impossible by
construction: this module is asked exclusively about glyphs the recognizer read
as ``é``, and it has no branch that emits an accent where the baseline had
none.

The model is executed through ONNX Runtime, so the product needs no new
dependency. Expected text appears in no signature here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .accent_cnn_input import AccentInputConfig, prepare_cnn_input
from .accent_localization import LocalizationConfig, assess_localization

__all__ = [
    "ACCENT_ABSENT",
    "ACCENT_PRESENT",
    "UNKNOWN",
    "AccentCnnVerdict",
    "AccentCnnVerifier",
]

ACCENT_ABSENT = "e"
ACCENT_PRESENT = "é"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class AccentCnnVerdict:
    """Outcome of one glyph decision, with the reason it came out that way."""

    verdict: str
    probability_absent: float
    reason: str
    view_probabilities: tuple[float, ...] = ()

    @property
    def is_accent_absent(self) -> bool:
        """True only for a licensed correction -- the one case that edits text."""
        return self.verdict == ACCENT_ABSENT


class AccentCnnVerifier:
    """Runs the accent-v3 decision path with a frozen model and thresholds."""

    def __init__(
        self,
        onnx_path: Path,
        config_path: Path,
        localization: LocalizationConfig | None = None,
    ) -> None:
        settings = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.version = settings.get("version", "unknown")
        self.absent_threshold = float(settings["absent_threshold"])
        self.present_threshold = float(settings["present_threshold"])
        self.input_config = AccentInputConfig(**settings["input_config"])
        self.localization_config = localization or LocalizationConfig()
        self._onnx_path = Path(onnx_path)
        self._session = None

    def _run(self, batch: np.ndarray) -> np.ndarray:
        """Return P(no accent) for each row of a prepared batch."""
        if self._session is None:
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                str(self._onnx_path), providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
        logits = self._session.run(None, {self._input_name: batch})[0]
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        return (exponentiated / exponentiated.sum(axis=1, keepdims=True))[:, 1]

    def _views(self, line_image: np.ndarray, x0: int, x1: int) -> list[tuple[int, int]]:
        """Tight, expanded and shifted spans around the same glyph."""
        jitter = max(1, int(self.input_config.jitter_pixels))
        width = line_image.shape[1]
        candidates = [
            (x0, x1),
            (x0 + jitter, x1 - jitter),      # tight
            (x0 - jitter, x1 + jitter),      # expanded
            (x0 - jitter, x1),               # shifted left
            (x0, x1 + jitter),               # shifted right
        ]
        spans = []
        for left, right in candidates:
            left = max(0, left)
            right = min(width, right)
            if right - left >= 4:
                spans.append((left, right))
        return spans

    def verify(
        self,
        line_image: np.ndarray,
        x0: int,
        x1: int,
        median_span_width: float | None = None,
    ) -> AccentCnnVerdict:
        """Decide whether the glyph at ``[x0:x1]`` may have its accent removed.

        There is deliberately no parameter for the expected character.
        """
        if line_image is None or line_image.size == 0:
            return AccentCnnVerdict(UNKNOWN, 0.0, "empty_line_image")

        report = assess_localization(
            line_image, x0, x1, median_span_width, self.localization_config
        )
        if not report.usable:
            return AccentCnnVerdict(
                UNKNOWN, 0.0, f"localization:{','.join(report.reasons)}"
            )

        spans = self._views(line_image, x0, x1)
        if not spans:
            return AccentCnnVerdict(UNKNOWN, 0.0, "no_usable_view")

        tensors = []
        for left, right in spans:
            prepared = prepare_cnn_input(
                line_image[:, left:right], self.input_config
            )
            if prepared is None:
                # A view that cannot even be prepared is missing information,
                # so the decision is not trustworthy.
                return AccentCnnVerdict(UNKNOWN, 0.0, "view_unpreparable")
            tensors.append(prepared[0])

        probabilities = self._run(np.stack(tensors).astype(np.float32))
        primary = float(probabilities[0])
        views = tuple(float(p) for p in probabilities)

        # A positive accent reading on any view vetoes the correction outright.
        if float(probabilities.min()) <= self.present_threshold:
            return AccentCnnVerdict(
                ACCENT_PRESENT if primary <= self.present_threshold else UNKNOWN,
                primary,
                "accent_present_veto",
                views,
            )

        # Every view must independently clear the absent threshold.
        if float(probabilities.min()) >= self.absent_threshold:
            return AccentCnnVerdict(
                ACCENT_ABSENT, primary, "all_views_confident_absent", views
            )

        if primary >= self.absent_threshold:
            return AccentCnnVerdict(
                UNKNOWN, primary, "view_disagreement", views
            )
        return AccentCnnVerdict(
            UNKNOWN, primary, "below_confidence_margin", views
        )
