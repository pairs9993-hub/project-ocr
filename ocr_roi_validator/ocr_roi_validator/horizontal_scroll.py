"""Expected-guided placement of observed horizontal OCR, without filling from truth."""
from collections import Counter
import re

from .compare import compare_text, normalize_ui_text


def compact_observation(text):
    text = re.sub(r"\s+", " ", normalize_ui_text(text)).strip()
    chars, gaps = [], {}
    space = False
    for char in text:
        if char == " ":
            space = True
            continue
        if chars:
            gaps[len(chars)] = space
        chars.append(char)
        space = False
    return chars, gaps


class HorizontalScrollEvidence:
    def __init__(self, expected, compare_mode="exact", threshold=0.9):
        self.expected = re.sub(r"\s+", " ", normalize_ui_text(expected)).strip()
        self.chars, self.expected_gaps = compact_observation(self.expected)
        self.tokens = tuple(c.casefold() for c in self.chars)
        self.compare_mode, self.threshold = compare_mode, threshold
        self.letters = [Counter() for _ in self.chars]
        self.gaps = [Counter() for _ in self.chars]
        self.edges = set()
        self.last_aligned = False
        self.aligned_frames = self.ambiguous_frames = self.unmatched_frames = 0

    def add(self, text):
        chars, gaps = compact_observation(text)
        self.last_aligned = False
        n, length = len(self.chars), len(chars)
        if not n or length < min(3, n) or length > n:
            self.unmatched_frames += 1
            return
        observed = tuple(c.casefold() for c in chars)
        doubled = self.tokens + self.tokens
        starts = [i for i in range(n) if doubled[i:i+length] == observed]
        whole_text = re.sub(r"\s+", " ", normalize_ui_text(text)).strip()
        if length == n and whole_text.casefold() == self.expected.casefold():
            starts = [0]  # A complete, directly observed sentence needs no phase guess.
        if len(starts) != 1:
            if starts:
                self.ambiguous_frames += 1
            else:
                self.unmatched_frames += 1
            return
        start = starts[0]
        self.last_aligned = True
        self.aligned_frames += 1
        for offset, char in enumerate(chars):
            position = (start + offset) % n
            self.letters[position][char] += 1
            if offset and position:
                self.edges.add(position)
                # Crop edges cannot establish a space reliably. Require context
                # on both sides, except when the entire sentence is visible.
                if length == n or (offset >= 2 and length-offset >= 2):
                    self.gaps[position][gaps[offset]] += 1

    @property
    def coverage(self):
        return sum(bool(c) for c in self.letters) / len(self.letters) if self.letters else 0.0

    @staticmethod
    def winner(counts):
        ranked = counts.most_common()
        if not ranked or (len(ranked) > 1 and ranked[0][1] == ranked[1][1]):
            return None
        return ranked[0][0]

    @property
    def spacing_status(self):
        pending = False
        for position in range(1, len(self.chars)):
            expected_space = self.expected_gaps[position]
            votes = self.gaps[position]
            # Every expected or ever-observed space must be established by
            # at least two observations and a strict majority.
            if not expected_space and not votes.get(True):
                continue
            winner = self.winner(votes)
            if winner is None or votes[winner] < 2:
                pending = True
            elif winner != expected_space:
                return "MISMATCH"
        return "PENDING" if pending else "MATCH"

    @property
    def assembled_text(self):
        parts = []
        for position, counts in enumerate(self.letters):
            if position:
                if position not in self.edges:
                    parts.append("□")
                elif self.winner(self.gaps[position]) is True:
                    parts.append(" ")
            char = self.winner(counts)
            parts.append(char if char is not None else "□")
        return re.sub("□+", "…", "".join(parts))

    @property
    def order_valid(self):
        # Adjacent letters must actually co-occur; disjoint words alone do not
        # prove their order, regardless of their arrival order or expected text.
        return len(self.edges) == max(0, len(self.chars)-1)

    @property
    def passed(self):
        if self.coverage != 1.0 or not self.order_valid:
            return False
        if self.compare_mode != "ignore_space" and self.spacing_status != "MATCH":
            return False
        return compare_text(self.expected, self.assembled_text, self.compare_mode, self.threshold).passed

    def details(self):
        return {"assembled_text": self.assembled_text, "character_coverage": self.coverage,
                "spacing_status": self.spacing_status, "order_confirmed": self.order_valid,
                "aligned_frames": self.aligned_frames, "ambiguous_frames": self.ambiguous_frames,
                "unmatched_frames": self.unmatched_frames,
                "assembly_method": "unique_expected_alignment_of_observed_characters"}
