"""Live verification of complete stationary text, without scroll reconstruction."""
from .compare import compare_text


class StaticTextAccumulator:
    def __init__(self, expected, mode="exact", threshold=.9):
        self.expected_text = expected
        self.mode, self.threshold = mode, threshold
        self.final_text = ""
        self.passed = False
        self.coverage = 0.0
        self._first_match = None
        self._matches = 0

    def add(self, text, score, observed_at):
        self.final_text = text
        comparison = compare_text(self.expected_text, text, self.mode, self.threshold)
        self.coverage = comparison.score
        self.passed = bool(self.expected_text.strip() and text.strip() and score >= .25 and comparison.passed)
        if self.passed:
            if self._first_match is None:
                self._first_match = observed_at
            self._matches += 1
        else:
            self._first_match = None
            self._matches = 0

    def ready_to_stop(self, now):
        return self.passed and self._matches >= 2 and now-self._first_match >= .5
