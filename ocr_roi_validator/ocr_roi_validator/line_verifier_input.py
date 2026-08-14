"""Input contract for the target-query line verifier.

Hard glyph cropping was rejected: every anchor estimate derived from the same
CTC output, so their errors were correlated and "consensus" still accepted 56
misassignments. Rather than keep trying to cut an exact box, the verifier now
receives the whole recognizer line and a *query* saying which character is
being asked about, and locates it itself.

What the network may see
------------------------
1. the native recognizer line image,
2. a soft CTC position map for the queried token,
3. a valid-width mask, and
4. an ordinal query: which character index, out of how many.

What it may never see
---------------------
The decoded text, any word or character identity, the expected text, file
names, dictionary lookups, synthetic label metadata, or the ground-truth
position. The ordinal query carries *where* the baseline emitted a character,
never *what* it read -- so the model cannot learn that a particular word
usually carries an accent.

That distinction is the whole point: a model that memorised "Veuillez" would
score well and be worthless, because it would answer from spelling rather than
from pixels. :func:`assert_no_text_leakage` and the audits in Stage 3E-0 exist
to keep that honest.

The CTC map is normalized to unit peak on purpose. Its amplitude correlates
with recognizer confidence, which correlates with the answer, so passing raw
confidence would leak the label through a side channel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

__all__ = [
    "LineVerifierInputConfig",
    "LineVerifierInput",
    "build_line_input",
    "assert_no_text_leakage",
    "CHANNEL_NAMES",
]

# Channel order is part of the frozen contract; a model trained on one order
# cannot read another.
CHANNEL_NAMES = ("line_image", "ctc_position_map", "valid_width_mask")


@dataclass(frozen=True)
class LineVerifierInputConfig:
    """Fixed input geometry, frozen alongside the model."""

    height: int = 32
    width: int = 320
    # Rendered width of one CTC timestep in the resized line, used to give the
    # position map a sensible extent rather than a single spike.
    position_sigma_timesteps: float = 1.0
    # Ordinal query is encoded as two scalars, both bounded.
    max_decoded_length: int = 128
    interpolation: str = "area"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LineVerifierInput:
    """One prepared example: image planes plus the ordinal query."""

    planes: np.ndarray          # (3, H, W) float32
    query: np.ndarray           # (2,) float32: normalized index, normalized length
    target_ordinal: int
    decoded_length: int
    scale_x: float              # crop px -> input px, for interpreting attention


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return image[..., :3].mean(axis=2)
    return image.astype(np.float64)


def _normalize(gray: np.ndarray) -> np.ndarray:
    """Scale to [-1, 1] with ink positive, whatever the polarity."""
    low, high = float(gray.min()), float(gray.max())
    if high - low < 1e-6:
        return np.zeros(gray.shape, dtype=np.float32)
    scaled = (gray - low) / (high - low)
    if (scaled >= 0.5).mean() > 0.5:
        scaled = 1.0 - scaled
    return (scaled * 2.0 - 1.0).astype(np.float32)


def build_line_input(
    line_image: np.ndarray,
    ctc_probabilities: np.ndarray,
    token_label: int,
    token_start: int,
    token_end: int,
    target_ordinal: int,
    decoded_length: int,
    config: LineVerifierInputConfig | None = None,
) -> LineVerifierInput | None:
    """Prepare one example, or None when it cannot be represented.

    ``ctc_probabilities`` is the (timesteps, labels) posterior for this line and
    ``token_label`` the label index of the queried token. Only the token's
    *position* is used; its identity never reaches the planes.
    """
    config = config or LineVerifierInputConfig()
    if line_image is None or line_image.size == 0:
        return None
    if decoded_length <= 0 or not (0 <= target_ordinal < decoded_length):
        return None

    crop_h, crop_w = line_image.shape[:2]
    if crop_h < 4 or crop_w < 8:
        return None

    # Preserve aspect ratio: scale to the target height, then place the line at
    # the left of a fixed-width canvas and mark the remainder invalid.
    scale = config.height / float(crop_h)
    scaled_w = max(1, min(config.width, int(round(crop_w * scale))))
    interpolation = (cv2.INTER_AREA if scaled_w <= crop_w else cv2.INTER_LINEAR)
    resized = cv2.resize(_to_grayscale(line_image).astype(np.float32),
                         (scaled_w, config.height), interpolation=interpolation)

    image_plane = np.zeros((config.height, config.width), dtype=np.float32)
    image_plane[:, :scaled_w] = _normalize(resized)

    valid_plane = np.zeros((config.height, config.width), dtype=np.float32)
    valid_plane[:, :scaled_w] = 1.0

    # CTC position map: a soft bump over the queried token's timesteps, mapped
    # into input columns. Normalized to unit peak so recognizer confidence
    # cannot leak through amplitude.
    position_plane = np.zeros((config.height, config.width), dtype=np.float32)
    timesteps = ctc_probabilities.shape[0]
    if timesteps > 0:
        centre_timestep = (token_start + token_end + 1) / 2.0
        centre_column = (centre_timestep / timesteps) * scaled_w
        sigma = max(1.0, config.position_sigma_timesteps * (scaled_w / timesteps))
        columns = np.arange(scaled_w, dtype=np.float32)
        bump = np.exp(-0.5 * ((columns - centre_column) / sigma) ** 2)
        peak = float(bump.max())
        if peak > 1e-9:
            bump = bump / peak
        position_plane[:, :scaled_w] = bump[None, :]

    query = np.array([
        target_ordinal / float(max(1, decoded_length - 1))
        if decoded_length > 1 else 0.0,
        min(decoded_length, config.max_decoded_length)
        / float(config.max_decoded_length),
    ], dtype=np.float32)

    planes = np.stack([image_plane, position_plane, valid_plane]).astype(np.float32)
    return LineVerifierInput(
        planes=planes, query=query, target_ordinal=target_ordinal,
        decoded_length=decoded_length, scale_x=scaled_w / float(crop_w),
    )


def assert_no_text_leakage(payload: object) -> None:
    """Raise if anything text-like is about to reach the network.

    Cheap, and it makes an accidental regression loud rather than silent.
    """
    if isinstance(payload, LineVerifierInput):
        payload = {"planes": payload.planes, "query": payload.query}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, str):
                raise ValueError(
                    f"string field {key!r} would reach the verifier; only image "
                    "planes and the numeric ordinal query are permitted"
                )
            if isinstance(value, (list, tuple)) and any(
                isinstance(item, str) for item in value
            ):
                raise ValueError(f"string sequence {key!r} would reach the verifier")
