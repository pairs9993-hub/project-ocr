"""Conservative cycle evidence. Repeated words/identical frames are not wraps."""
import re
import unicodedata


def normalized(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


class CycleTracker:
    """Require unique exact windows and overlapping forward motion for each cycle.

    An ambiguous/reversed/skipped observation discards the pending cycle, never
    manufactures a count. Each counted cycle starts at position zero, covers
    every character in order and returns to zero. OCR errors may cause timeout.
    """
    def __init__(self, expected):
        self.expected = normalized(expected)
        self.previous = None
        self.previous_end = None
        self.covered_end = None
        self.cycles = 0
        self.ambiguous = 0
        self.reason = "WAITING_FOR_START"

    def add(self, text):
        observed = normalized(text)
        if len(observed) < 3 or not self.expected:
            return
        cycle = self.expected + " "
        wrapped = cycle + cycle
        starts = [i for i in range(len(self.expected)) if wrapped.startswith(observed, i)]
        if len(starts) != 1:
            self.ambiguous += 1
            self.previous = self.previous_end = self.covered_end = None
            self.reason = "AMBIGUOUS_OR_UNMATCHED"
            return
        start = starts[0]
        end = min(len(self.expected), start + len(observed))
        if self.previous is None:
            self.covered_end = end if start == 0 else None
        elif start == self.previous:
            if self.covered_end is not None:
                self.covered_end = max(self.covered_end, end)
        elif start > self.previous and start <= self.previous_end:
            if self.covered_end is not None:
                self.covered_end = max(self.covered_end, end)
        elif start == 0 and self.previous > 0 and self.covered_end == len(self.expected):
            self.cycles += 1
            self.covered_end = end
            self.reason = "CYCLE_CONFIRMED"
        else:
            self.covered_end = end if start == 0 else None
            self.reason = "ORDER_OR_GAP_RESET"
        self.previous, self.previous_end = start, end


class Confirmation:
    def __init__(self):
        self.since = None
        self.samples = 0

    def add(self, passed, now):
        if not passed:
            self.since, self.samples = None, 0
            return False
        if self.since is None:
            self.since = now
        self.samples += 1
        return self.samples >= 2 and now - self.since >= 0.5