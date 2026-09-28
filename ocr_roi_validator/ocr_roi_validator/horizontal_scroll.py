"""Expected-guided placement of observed horizontal OCR, without filling from truth."""
from collections import Counter
import re

from .compare import compare_text, normalize_ui_text


def compact_observation(text, group_trademark=False, group_mc=False):
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
    if group_trademark or group_mc:
        grouped, grouped_gaps = [], {}
        index = 0
        while index < len(chars):
            if grouped:
                grouped_gaps[len(grouped)] = gaps[index]
            pair = ''.join(chars[index:index+2])
            if (index+1 < len(chars) and not gaps[index+1]
                    and ((group_trademark and pair == 'TM') or (group_mc and pair.casefold() == 'mc'))):
                grouped.append(pair)  # Preserve actual letters/case; grouping is placement only.
                index += 2
            else:
                grouped.append(chars[index])
                index += 1
        return grouped, grouped_gaps
    return chars, gaps


def substitution_position(tokens, observed):
    """Locate a long observation with few substitutions and an unambiguous margin.

    No insertions/deletions: every observed character retains one actual position.
    """
    n, length = len(tokens), len(observed)
    if length < 8 or length > n:
        return None
    maximum = min(2, length // 8)
    doubled = tokens + tokens
    ranked = []
    for start in range(n):
        differences = [i for i, char in enumerate(observed) if char != doubled[start+i]]
        ranked.append((len(differences), start, differences))
    ranked.sort()
    errors, start, differences = ranked[0]
    if not 1 <= errors <= maximum:
        return None
    if len(ranked) > 1 and ranked[1][0] < errors+2:
        return None
    # Require a substantial exactly matching run to anchor the placement.
    boundaries = [-1] + differences + [length]
    if max(b-a-1 for a,b in zip(boundaries, boundaries[1:])) < 6:
        return None
    return start, differences


class HorizontalScrollEvidence:
    def __init__(self, expected, compare_mode="exact", threshold=0.9):
        self.expected = re.sub(r"\s+", " ", normalize_ui_text(expected)).strip()
        self.group_trademark = "\u2122" in self.expected
        self.group_mc = "\U0001f16a" in self.expected
        self.chars, self.expected_gaps = compact_observation(self.expected, self.group_trademark, self.group_mc)
        self.tokens = tuple(self._placement_token(c) for c in self.chars)
        self.compare_mode, self.threshold = compare_mode, threshold
        self.letters = [Counter() for _ in self.chars]
        self.gaps = [Counter() for _ in self.chars]
        self.edges = set()
        self.last_aligned = False
        self.last_reason = "NOT_EVALUATED"
        self.last_start = None
        self.last_substitutions = []
        self.aligned_frames = self.ambiguous_frames = self.unmatched_frames = 0

    def _placement_token(self, char):
        # Equivalence is for positioning only. Final comparison keeps actual TM/™.
        if self.group_trademark and char in ("TM", "\u2122"):
            return "\u2122"
        if self.group_mc and (char.casefold() == 'mc' or char == '\U0001f16a'):
            return '\U0001f16a'
        return char.casefold()

    def add(self, text):
        chars, gaps = compact_observation(text, self.group_trademark, self.group_mc)
        self.last_aligned = False
        self.last_reason = "NOT_EVALUATED"
        self.last_start = None
        self.last_substitutions = []
        n, length = len(self.chars), len(chars)
        if not n or length < min(3, n) or length > n:
            self.last_reason = "EMPTY_OR_LENGTH_OUT_OF_RANGE"
            self.unmatched_frames += 1
            return
        observed = tuple(self._placement_token(c) for c in chars)
        doubled = self.tokens + self.tokens
        starts = [i for i in range(n) if doubled[i:i+length] == observed]
        whole_text = re.sub(r"\s+", " ", normalize_ui_text(text)).strip()
        if length == n and whole_text.casefold() == self.expected.casefold():
            starts = [0]  # A complete, directly observed sentence needs no phase guess.
        tolerant = substitution_position(self.tokens, observed) if not starts else None
        if tolerant is not None:
            start, offsets = tolerant
            starts = [start]
            self.last_substitutions = [{"position": (start+i) % n+1,
                                        "expected": self.chars[(start+i) % n], "observed": chars[i]}
                                       for i in offsets]
        if len(starts) != 1:
            if starts:
                self.last_reason = "AMBIGUOUS_POSITION"
                self.ambiguous_frames += 1
            else:
                self.last_reason = "NO_EXACT_ALIGNMENT"
                self.unmatched_frames += 1
            return
        start = starts[0]
        self.last_aligned = True
        self.last_reason = "ALIGNED_WITH_SUBSTITUTIONS" if self.last_substitutions else "ALIGNED"
        self.last_start = start+1
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
    def assembly_complete(self):
        """A readable reconstruction is distinct from a matching verdict."""
        if self.coverage != 1.0 or not self.order_valid:
            return False
        if any(self.winner(counts) is None for counts in self.letters):
            return False
        for position in range(1, len(self.chars)):
            votes = self.gaps[position]
            if self.expected_gaps[position] or votes.get(True):
                winner = self.winner(votes)
                if winner is None or votes[winner] < 2:
                    return False
        return True

    @property
    def passed(self):
        if self.coverage != 1.0 or not self.order_valid:
            return False
        if self.compare_mode != "ignore_space" and self.spacing_status != "MATCH":
            return False
        return compare_text(self.expected, self.assembled_text, self.compare_mode, self.threshold).passed

    def details(self):
        return {"missing_characters": [{"position": i+1, "expected": self.chars[i]}
                                       for i, counts in enumerate(self.letters) if not counts],
                "assembled_text": self.assembled_text, "assembly_complete": self.assembly_complete,
                "unresolved_characters": [
                    {"position": i+1, "observed_votes": dict(counts)}
                    for i, counts in enumerate(self.letters) if counts and self.winner(counts) is None],
                "character_coverage": self.coverage,
                "spacing_status": self.spacing_status, "order_confirmed": self.order_valid,
                "aligned_frames": self.aligned_frames, "ambiguous_frames": self.ambiguous_frames,
                "unmatched_frames": self.unmatched_frames,
                "assembly_method": "unique_alignment_preserving_observed_substitutions"}


def observed_display(evidence, observations):
    """Keep incomplete/mismatched OCR visible without treating it as validated."""
    if evidence.assembly_complete:
        return evidence.assembled_text
    unique = list(dict.fromkeys(text for text in observations if text.strip()))
    return "\n--- frame ---\n".join(unique)
