"""Report whether a usable exact target capture exists.

Scans the failure diagnostics and prints a plain verdict, so confirming a
capture never requires reading metadata.json by hand.

A capture qualifies when the recognizer input was recorded at the OCR call site
(Run Once, Context Detect off, direct ROI path), the language is French, and
the expected/actual pair shows the Veuillez accent failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_HASHES = {
    "detector": "21af37f36ce3940ba2fd201c6035571ae5807cf0333f1734d6d5b95c62135b7c",
    "fr": "d6a439c2b59b46051ea3e07a9d7df69cb76589489b4e487b3d365a773b903b0d",
    "dictionary": "7ff72cdde593c6f80ebd573dddb67b1a103a1607a444c11c4b2b7db57ae1d627",
}


def load(directory: Path) -> dict | None:
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def disqualifications(directory: Path, metadata: dict) -> list[str]:
    reasons = []
    if metadata.get("ocr_input_fidelity") != "exact_recorded_ocr_input":
        reasons.append(
            f"fidelity={metadata.get('ocr_input_fidelity') or 'missing'} "
            "(need exact_recorded_ocr_input)"
        )
    if metadata.get("ocr_path") != "direct":
        reasons.append(
            f"ocr_path={metadata.get('ocr_path') or 'missing'} (need direct)"
        )
    if str(metadata.get("language", "")).lower() not in {"fr", "french"}:
        reasons.append(f"language={metadata.get('language') or 'missing'} (need fr)")
    if not (directory / "roi_ocr_input.png").is_file():
        reasons.append("roi_ocr_input.png missing")
    expected = metadata.get("expected") or ""
    actual = metadata.get("actual") or ""
    if "Veuillez allumer" not in expected:
        reasons.append("expected does not contain the target phrase")
    if "Véuillez allumer" not in actual:
        if not actual:
            reasons.append("actual is empty (no text detected)")
        else:
            reasons.append("actual does not show the Veuillez accent failure")
    return reasons


def check_hashes(metadata: dict) -> list[str]:
    models = metadata.get("models") or {}
    problems = []
    detector = (models.get("detector") or {}).get("sha256")
    dictionary = (models.get("dictionary") or {}).get("sha256")
    recognizer = ((models.get("recognizers") or {}).get("fr") or {}).get("sha256")
    for name, actual, want in (
        ("detector", detector, EXPECTED_HASHES["detector"]),
        ("dictionary", dictionary, EXPECTED_HASHES["dictionary"]),
        ("fr recognizer", recognizer, EXPECTED_HASHES["fr"]),
    ):
        if actual is None:
            problems.append(f"{name} hash missing")
        elif actual != want:
            problems.append(f"{name} hash {actual[:12]} != expected {want[:12]}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--failures-dir",
        type=Path,
        default=VALIDATOR_ROOT / "captures" / "failures",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    directories = sorted(
        (p for p in args.failures_dir.glob("failure_*") if p.is_dir()),
        key=lambda p: p.name,
    )
    if not directories:
        print(f"No failure captures found under {args.failures_dir}")
        print("Run run.bat, select French, turn Context Detect OFF, press Run Once.")
        return 1

    qualifying: list[tuple[Path, dict, str]] = []
    rejected: list[tuple[Path, list[str]]] = []
    for directory in directories:
        metadata = load(directory)
        if metadata is None:
            rejected.append((directory, ["metadata.json missing or invalid"]))
            continue
        reasons = disqualifications(directory, metadata)
        if reasons:
            rejected.append((directory, reasons))
            continue
        digest = hashlib.sha256((directory / "roi_ocr_input.png").read_bytes()).hexdigest()
        qualifying.append((directory, metadata, digest))

    print(f"Scanned {len(directories)} capture(s) in {args.failures_dir}")
    print(f"  qualifying : {len(qualifying)}")
    print(f"  rejected   : {len(rejected)}")

    if args.verbose or not qualifying:
        print("\nRejected captures:")
        for directory, reasons in rejected[-10:]:
            print(f"  {directory.name}")
            for reason in reasons:
                print(f"      - {reason}")
        if len(rejected) > 10:
            print(f"  ... and {len(rejected) - 10} more")

    if not qualifying:
        print("\nRESULT: NO USABLE CAPTURE")
        print("To produce one:")
        print("  1. close the app completely")
        print("  2. run ocr_roi_validator\\run.bat")
        print("  3. Language = French, Context Detect = OFF")
        print("  4. select the ROI around the Veuillez text, fill in Expected")
        print("  5. press Run Once (not Start / Timed Capture)")
        print("  6. the log should print CAPTURE=USABLE")
        return 1

    digests = {digest for _, _, digest in qualifying}
    latest, metadata, digest = qualifying[-1]

    try:
        shown_path = latest.relative_to(VALIDATOR_ROOT)
    except ValueError:
        # Captures can live outside the repo (e.g. a temporary directory).
        shown_path = latest

    print("\nRESULT: USABLE CAPTURE FOUND")
    print(f"  directory      : {shown_path}")
    print(f"  image sha256   : {digest}")
    print(f"  image size     : {metadata.get('roi_ocr_input_size')}")
    print(f"  expected       : {metadata.get('expected')!r}")
    print(f"  actual         : {metadata.get('actual')!r}")
    print(f"  raw ocr output : {metadata.get('ocr_raw_output')!r}")
    if len(qualifying) > 1:
        if len(digests) == 1:
            print(f"  note           : {len(qualifying)} captures, all the same image")
        else:
            print(
                f"  WARNING        : {len(qualifying)} captures with "
                f"{len(digests)} different images; review before evaluating"
            )
            for directory, _, other in qualifying:
                print(f"      {directory.name}  {other[:16]}")

    hash_problems = check_hashes(metadata)
    if hash_problems:
        print("\n  MODEL HASH MISMATCH:")
        for problem in hash_problems:
            print(f"      - {problem}")
        return 1
    print("  model hashes   : detector, dictionary and fr recognizer all match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
