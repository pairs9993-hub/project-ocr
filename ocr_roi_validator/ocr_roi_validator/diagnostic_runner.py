"""Crash-safe scaffolding for long diagnostics.

The previous font diagnostic ran for fifteen hours and produced nothing usable.
It buffered its progress, held every result in memory, and wrote its report only
at the very end -- so when the machine was shut down at base condition 1000 of
2000, the entire run was lost. A partial log survived only by accident.

This module removes each of those failure modes:

* progress is written line-buffered and flushed, so the log reflects reality;
* every unit of work is appended to a JSONL checkpoint and fsynced, so an
  interrupted run keeps everything it had already finished;
* resuming replays the RNG from the beginning and skips completed units, so the
  stream of conditions is identical whether or not the run was interrupted;
* the final report is written to a temporary file and atomically renamed, so a
  half-written report can never be mistaken for a finished one.

The parity requirement is what makes resume trustworthy: an interrupted run
resumed partway must produce byte-identical checkpoint rows to an uninterrupted
one. :func:`verify_resume_parity` checks exactly that, and callers are expected
to refuse to start a real diagnostic until it passes.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

__all__ = [
    "CheckpointWriter",
    "ResumeState",
    "atomic_write_json",
    "log_line",
    "load_checkpoint",
    "resumable_units",
    "verify_resume_parity",
]


def log_line(message: str) -> None:
    """Write one progress line and flush it immediately.

    Long runs are only observable if their output survives the moment they are
    killed, so nothing here is left sitting in a buffer.
    """
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


@dataclass(frozen=True)
class ResumeState:
    """What a previous run left behind."""

    rows: list[dict]
    completed_units: set[int]
    digests: set[str]
    resume_count: int

    @property
    def next_unit(self) -> int:
        return max(self.completed_units) + 1 if self.completed_units else 0


def load_checkpoint(path: Path, unit_field: str = "unit",
                    digest_field: str = "row_digest") -> ResumeState:
    """Read a checkpoint, tolerating a truncated final line.

    A run killed mid-write can leave a partial JSON object. That line is
    dropped rather than repaired: a half-written row is not a result, and
    guessing at its content would fabricate data.
    """
    if not path.is_file():
        return ResumeState([], set(), set(), 0)
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue          # truncated tail from an interrupted write
    completed = {int(r[unit_field]) for r in rows if unit_field in r}
    digests = {r[digest_field] for r in rows if digest_field in r}
    # Resume count is run provenance, so it lives in a sidecar rather than in
    # the rows -- rows must stay byte-identical across a resume. Reading is
    # side-effect free; CheckpointWriter records the resume when it opens the
    # file, so counting does not depend on how often this function is called.
    return ResumeState(rows, completed, digests, _read_resume_count(path))


def _resume_marker(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".resumes")


def _read_resume_count(path: Path) -> int:
    marker = _resume_marker(path)
    if not marker.is_file():
        return 0
    try:
        return int(marker.read_text(encoding="utf-8").strip() or 0)
    except ValueError:
        return 0


# Fields describing *this invocation* rather than the measurement. Recording
# any of them in a row would make resumed output differ from uninterrupted
# output for reasons that have nothing to do with the result, which is exactly
# what the parity gate exists to catch. They belong in the manifest.
RUN_LOCAL_FIELDS = frozenset({
    "resume_count", "started_at", "finished_at", "elapsed_seconds", "pid",
    "hostname", "worker_id",
})


class CheckpointWriter:
    """Append-only JSONL writer that fsyncs on a schedule.

    Duplicate digests are rejected rather than silently overwritten, so a
    resume that replays work cannot quietly double-count it. Run-local fields
    are refused outright -- see :data:`RUN_LOCAL_FIELDS`.
    """

    def __init__(self, path: Path, known_digests: set[str] | None = None,
                 flush_every: int = 25) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Opening an existing checkpoint for append *is* the resume event, so
        # it is counted here rather than on every read.
        if path.is_file() and path.stat().st_size > 0:
            _resume_marker(path).write_text(
                str(_read_resume_count(path) + 1), encoding="utf-8")
        self._path = path
        self._handle = path.open("a", encoding="utf-8")
        self._digests = set(known_digests or ())
        self._flush_every = max(1, flush_every)
        self._since_flush = 0
        self.written = 0
        self.duplicates_rejected = 0

    def append(self, row: dict, digest_field: str = "row_digest") -> bool:
        contaminated = RUN_LOCAL_FIELDS.intersection(row)
        if contaminated:
            raise ValueError(
                f"checkpoint rows must not carry run-local fields "
                f"{sorted(contaminated)}; they break resume parity and belong "
                "in the manifest"
            )
        digest = row.get(digest_field)
        if digest is not None:
            if digest in self._digests:
                self.duplicates_rejected += 1
                return False
            self._digests.add(digest)
        self._handle.write(json.dumps(row, ensure_ascii=False,
                                      sort_keys=True) + "\n")
        self.written += 1
        self._since_flush += 1
        if self._since_flush >= self._flush_every:
            self.sync()
        return True

    def sync(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._since_flush = 0

    def close(self) -> None:
        try:
            self.sync()
        finally:
            self._handle.close()

    def __enter__(self) -> "CheckpointWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def resumable_units(total: int, build: Callable[[int], object],
                    completed: set[int]) -> Iterator[tuple[int, object]]:
    """Yield unfinished units, having built every earlier one first.

    ``build`` is called for *all* indices in order, including completed ones,
    because it consumes a shared RNG. Skipping the completed indices outright
    would shift the random stream and make a resumed run diverge from an
    uninterrupted one.
    """
    for index in range(total):
        unit = build(index)
        if index in completed:
            continue
        yield index, unit


def atomic_write_json(path: Path, payload: dict) -> str:
    """Write a report so that it either exists complete or not at all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_resume_parity(
    work: Callable[[Path, int | None], None], directory: Path, total: int,
    interrupt_at: int,
) -> dict:
    """Run a workload straight through, then again in two interrupted pieces.

    Both checkpoints must match byte for byte. If they do not, resume changes
    the result, and no diagnostic built on it can be trusted.
    """
    directory.mkdir(parents=True, exist_ok=True)
    uninterrupted = directory / "parity_uninterrupted.jsonl"
    resumed = directory / "parity_resumed.jsonl"
    for path in (uninterrupted, resumed):
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".resumes").unlink(missing_ok=True)

    work(uninterrupted, None)               # one clean pass
    work(resumed, interrupt_at)             # stop early ...
    work(resumed, None)                     # ... then resume to the end

    first, second = uninterrupted.read_bytes(), resumed.read_bytes()
    first_digest = hashlib.sha256(first).hexdigest()
    second_digest = hashlib.sha256(second).hexdigest()
    first_rows = first.decode("utf-8").splitlines()
    second_rows = second.decode("utf-8").splitlines()
    mismatch = next((i for i, (a, b) in enumerate(zip(first_rows, second_rows))
                     if a != b), None)
    return {
        "total_units": total,
        "interrupt_at": interrupt_at,
        "uninterrupted_rows": len(first_rows),
        "resumed_rows": len(second_rows),
        "uninterrupted_sha256": first_digest,
        "resumed_sha256": second_digest,
        "bytes_identical": first == second,
        "first_mismatch_row": mismatch,
        "parity": first == second and len(first_rows) == len(second_rows),
    }
