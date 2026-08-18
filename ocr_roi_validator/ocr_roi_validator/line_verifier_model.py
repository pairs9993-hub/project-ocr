"""The line verifier network, as specified in Stage 3E-0.

The model receives the whole recognizer line plus a query naming which
character position is being asked about, and must answer from the pixels at
that position. It never sees the decoded string, so it cannot answer from
spelling -- which is the entire point, since the string that matters in
production is one no training set contains.

Three planes go in (line image, unit-normalised CTC position map, valid-width
mask) together with two query scalars. Attention over the horizontal axis is
supervised toward the renderer's target centre during training; at inference it
is the model's own.

Kept deliberately small. The decision is local and the training set is
fifteen thousand lines, so capacity beyond this buys memorisation rather than
accuracy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LineVerifier", "CLASS_NAMES", "CLASS_INDEX"]

CLASS_NAMES = ("ACCENT_PRESENT", "BARE_E", "UNKNOWN")
CLASS_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


class LineVerifier(nn.Module):
    """Query-conditioned line classifier with supervised attention."""

    def __init__(self, channels: int = 3, width: int = 320, height: int = 32,
                 hidden: int = 96, classes: int = 3) -> None:
        super().__init__()
        self.width = width
        # Pooling is vertical only. Horizontal resolution is what locates the
        # queried character, so collapsing it would defeat the query.
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.BatchNorm2d(hidden),
            nn.ReLU(),
        )
        # The query is two numbers -- an ordinal and a count -- never a
        # character. It is projected and added to every column so the encoder
        # can condition on position without being told what is there.
        self.query_encoder = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, hidden),
        )
        self.attention = nn.Conv1d(hidden, 1, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, classes),
        )

    def forward(self, planes: torch.Tensor, query: torch.Tensor):
        # Collapse the remaining height by averaging. This was an
        # AdaptiveAvgPool2d((1, None)), which opset 11 cannot export because the
        # None makes output_size non-constant. Three vertical poolings leave a
        # fixed height of 4, so a plain mean is the same computation -- verified
        # identical on the trained weights, not merely assumed.
        features = self.encoder(planes).mean(dim=2)          # (B, hidden, W)
        conditioned = features + self.query_encoder(query).unsqueeze(-1)
        scores = self.attention(conditioned).squeeze(1)      # (B, W)
        weights = F.softmax(scores, dim=-1)
        pooled = torch.bmm(conditioned, weights.unsqueeze(-1)).squeeze(-1)
        return self.head(pooled), weights
