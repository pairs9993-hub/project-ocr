"""Locate every baseline-predicted `é` from CTC alignment alone.

A glyph verifier can only be safe if it inspects *every* accent the recognizer
claims to have seen, chosen without reference to the answer. This tool derives
those positions from the CTC timestep alignment of the recognizer's own output,
so the same rule finds both a hallucinated accent and a genuine one.

Ground truth is never read: positions come from the decoded text and the logits,
nothing else.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATOR_ROOT))

# Characters this tool is scoped to. i/l, digits, timers and punctuation are
# deliberately out of scope.
ACCENT_CHARS = frozenset("é")

# An `é` span is unusable when the alignment is ambiguous.
MIN_CONFIDENCE = 0.30


def load_dictionary(path: Path) -> list[str]:
    """CTC label order: blank at index 0, then dictionary, then space."""
    characters = path.read_text(encoding="utf-8").split("\n")
    if characters and characters[-1] == "":
        characters = characters[:-1]
    return ["<blank>"] + characters + [" "]


def collapse_ctc(argmax: list[int], labels: list[str]) -> list[dict]:
    """Collapse a CTC argmax path into emitted characters with timestep spans."""
    emitted = []
    previous = 0
    for timestep, index in enumerate(argmax):
        if index != 0 and index != previous:
            character = labels[index] if index < len(labels) else "?"
            emitted.append(
                {
                    "char": character,
                    "label_index": int(index),
                    "start_timestep": timestep,
                    "end_timestep": timestep,
                }
            )
        elif emitted and index == previous and index != 0:
            emitted[-1]["end_timestep"] = timestep
        previous = index
    return emitted


def timestep_to_x(
    timestep: int, total_timesteps: int, resized_width: int, crop_width: int
) -> float:
    """Map a CTC timestep back to an x coordinate in the original line crop.

    The recognizer downsamples width by a fixed factor, and the line was resized
    into the padded tensor before that. Both are linear, so the mapping is a
    simple proportion through the resized width.
    """
    if total_timesteps <= 0 or resized_width <= 0:
        return 0.0
    x_in_resized = (timestep + 0.5) * (resized_width / total_timesteps)
    return x_in_resized * (crop_width / resized_width)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line-dump", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--crops-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path)
    parser.add_argument("--pad-x", type=int, default=4)
    parser.add_argument("--accent-headroom", type=float, default=0.55,
                        help="fraction of glyph height kept above the x-height")
    args = parser.parse_args()

    dump = json.loads(args.line_dump.read_text(encoding="utf-8"))
    labels = load_dictionary(args.dictionary)
    channels, rec_height, _ = dump["rec_image_shape"]

    findings = []
    for line in dump["lines"]:
        crop_path = args.crops_dir / f"line{line['line_index']}_crop.png"
        crop = cv2.imread(str(crop_path)) if crop_path.is_file() else None
        crop_w, crop_h = line["crop_size"]

        argmax = line["ctc_argmax"]
        confidence = line["ctc_argmax_confidence"]
        total_timesteps = line["ctc_timesteps"]
        emitted = collapse_ctc(argmax, labels)

        decoded = "".join(item["char"] for item in emitted)
        # The collapsed path should reconstruct the decoded text; if it does
        # not, the alignment is not trustworthy for this line.
        alignment_ok = decoded == line["decoded_text"]

        # Width the line actually occupied inside the padded tensor.
        batch_ratio = line["batch_max_wh_ratio"]
        padded_width = int(rec_height * batch_ratio)
        resized_width = min(
            padded_width, int(np.ceil(rec_height * line["wh_ratio"]))
        )

        for position, item in enumerate(emitted):
            character = unicodedata.normalize("NFC", item["char"])
            if character not in ACCENT_CHARS:
                continue

            span_confidence = float(
                np.mean(
                    confidence[item["start_timestep"] : item["end_timestep"] + 1]
                )
            )
            x_start = timestep_to_x(
                item["start_timestep"], total_timesteps, resized_width, crop_w
            )
            x_end = timestep_to_x(
                item["end_timestep"] + 1, total_timesteps, resized_width, crop_w
            )
            x0 = max(0, int(np.floor(x_start)) - args.pad_x)
            x1 = min(crop_w, int(np.ceil(x_end)) + args.pad_x)

            reasons = []
            if not alignment_ok:
                reasons.append("ctc_path_does_not_reconstruct_text")
            if span_confidence < MIN_CONFIDENCE:
                reasons.append(f"low_confidence_{span_confidence:.2f}")
            if x1 - x0 < 3:
                reasons.append("span_too_narrow")
            if crop is None:
                reasons.append("crop_image_unavailable")

            finding = {
                "line_index": line["line_index"],
                "char_position": position,
                "decoded_text": line["decoded_text"],
                "char": character,
                "start_timestep": item["start_timestep"],
                "end_timestep": item["end_timestep"],
                "span_confidence": span_confidence,
                "x0": x0,
                "x1": x1,
                "crop_size": [crop_w, crop_h],
                # Accent ink sits above the x-height; keep the full glyph plus
                # headroom so the diacritic cannot be cropped away.
                "y0": 0,
                "y1": crop_h,
                "usable": not reasons,
                "unknown_reasons": reasons,
            }
            findings.append(finding)

            if args.overlay_dir and crop is not None:
                args.overlay_dir.mkdir(parents=True, exist_ok=True)
                overlay = crop.copy()
                cv2.rectangle(overlay, (x0, 0), (x1, crop_h - 1), (0, 0, 255), 1)
                cv2.imwrite(
                    str(
                        args.overlay_dir
                        / f"line{line['line_index']}_pos{position}_overlay.png"
                    ),
                    overlay,
                )
                glyph = crop[:, x0:x1]
                if glyph.size:
                    cv2.imwrite(
                        str(
                            args.overlay_dir
                            / f"line{line['line_index']}_pos{position}_glyph.png"
                        ),
                        glyph,
                    )

    print(f"lines: {len(dump['lines'])}")
    print(f"baseline-predicted 'é' glyphs found: {len(findings)}")
    for finding in findings:
        status = "usable" if finding["usable"] else f"UNKNOWN {finding['unknown_reasons']}"
        print(
            f"  line {finding['line_index']} pos {finding['char_position']:>3d} "
            f"x[{finding['x0']:>4d},{finding['x1']:>4d}] "
            f"conf {finding['span_confidence']:.3f}  {status}"
        )
        print(f"      in: {finding['decoded_text']!r}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps({"findings": findings}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
