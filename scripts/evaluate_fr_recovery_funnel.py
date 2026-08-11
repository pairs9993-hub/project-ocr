"""Resumable Gate A funnel over the French recovery ONNX candidates.

Runs the cheap gate first -- the exact recorded target input and a defect
counterpart, over a fixed set of runtime-plausible perturbations -- so that
expensive stress and full-screen evaluation only ever run on survivors.

Results are keyed by candidate SHA-256 and appended to a JSONL as each
candidate finishes, so an interrupted run resumes where it stopped and a
candidate that fails to load does not abort the sweep.

Expected text is read only after inference, to score the result. It is never
passed to a recognizer or to the router.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_ROOT = ROOT / "ocr_roi_validator"
for path in (str(VALIDATOR_ROOT), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from ocr_roi_validator.fr_specialist_router import route_specialist_text  # noqa: E402
from ocr_roi_validator.model_package import load_model_package  # noqa: E402
from ocr_roi_validator.ocr_engine import OCREngine  # noqa: E402

# The exact recorded OCR input already went through margin, padding and
# upscaling, so perturbations are applied to it directly and nothing is
# re-applied afterwards.
PERTURBATIONS = (
    "none",
    "crop_left_1px",
    "crop_right_1px",
    "crop_top_1px",
    "crop_bottom_1px",
    "pad_all_1px",
    "crop_all_1px",
)


def perturb(image: Image.Image, kind: str) -> Image.Image:
    width, height = image.size
    if kind == "none":
        return image
    if kind == "crop_left_1px":
        return image.crop((1, 0, width, height))
    if kind == "crop_right_1px":
        return image.crop((0, 0, width - 1, height))
    if kind == "crop_top_1px":
        return image.crop((0, 1, width, height))
    if kind == "crop_bottom_1px":
        return image.crop((0, 0, width, height - 1))
    if kind == "pad_all_1px":
        padded = Image.new(image.mode, (width + 2, height + 2), image.getpixel((0, 0)))
        padded.paste(image, (1, 1))
        return padded
    if kind == "crop_all_1px":
        return image.crop((1, 1, width - 1, height - 1))
    raise ValueError(f"unknown perturbation: {kind}")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output_classes(model: Path) -> int | None:
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        shape = session.get_outputs()[0].shape
        return int(shape[-1])
    except Exception:
        return None


def build_engine(package_dir: Path, language: str, recognizer: Path) -> OCREngine:
    package = load_model_package(package_dir)
    recognizers = dict(package.recognizers)
    recognizers[language] = recognizer.resolve()
    return OCREngine(package=replace(package, recognizers=recognizers), backend="rapid")


def discover_candidates(search_root: Path) -> list[Path]:
    return sorted(
        p for p in search_root.rglob("*.onnx") if "veuillez_recovery" in p.as_posix()
    )


def load_done(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    done = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("candidate_sha256"):
            done[row["candidate_sha256"]] = row
    return done


def evaluate_candidate(
    baseline_engine: OCREngine,
    specialist_engine: OCREngine,
    language: str,
    target_image: Image.Image,
    defect_image: Image.Image | None,
    target_expected: str,
) -> dict:
    """Run Gate A for one candidate. Expected is used only to score, after OCR."""
    target_rows = []
    for kind in PERTURBATIONS:
        image = perturb(target_image, kind)
        baseline_text = baseline_engine.run(image, language).text
        specialist_text = specialist_engine.run(image, language).text
        decision = route_specialist_text(baseline_text, specialist_text)
        target_rows.append(
            {
                "perturbation": kind,
                "baseline": baseline_text,
                "specialist": specialist_text,
                "final": decision.final_text,
                "route": decision.route,
                "specialist_applied": decision.specialist_applied,
                # Scoring only, after every OCR decision is made.
                "final_matches_expected": decision.final_text == target_expected,
                "target_word_fixed": (
                    "Véuillez" in baseline_text and "Véuillez" not in decision.final_text
                ),
            }
        )

    defect_rows = []
    if defect_image is not None:
        for kind in PERTURBATIONS:
            image = perturb(defect_image, kind)
            baseline_text = baseline_engine.run(image, language).text
            specialist_text = specialist_engine.run(image, language).text
            decision = route_specialist_text(baseline_text, specialist_text)
            defect_rows.append(
                {
                    "perturbation": kind,
                    "baseline": baseline_text,
                    "specialist": specialist_text,
                    "final": decision.final_text,
                    "route": decision.route,
                    "specialist_applied": decision.specialist_applied,
                    # A false pass is the defect reading back as the clean string.
                    "false_pass": decision.final_text == target_expected,
                }
            )

    target_full_exact = sum(1 for r in target_rows if r["final_matches_expected"])
    target_word_fixed = sum(1 for r in target_rows if r["target_word_fixed"])
    defect_false_pass = sum(1 for r in defect_rows if r["false_pass"])
    return {
        "target_rows": target_rows,
        "defect_rows": defect_rows,
        "target_full_exact": target_full_exact,
        "target_word_fixed": target_word_fixed,
        "defect_false_pass": defect_false_pass,
        "gate_a_pass": (
            target_full_exact == len(PERTURBATIONS) and defect_false_pass == 0
        ),
        "gate_a_word_level_pass": (
            target_word_fixed == len(PERTURBATIONS) and defect_false_pass == 0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-input", type=Path, required=True)
    parser.add_argument("--target-metadata", type=Path, required=True)
    parser.add_argument("--defect-image", type=Path)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, default=ROOT / "PaddleOCR" / "output")
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    metadata = json.loads(args.target_metadata.read_text(encoding="utf-8"))
    if metadata.get("ocr_input_fidelity") != "exact_recorded_ocr_input":
        print(
            "refusing to run: target metadata is not exact_recorded_ocr_input",
            file=sys.stderr,
        )
        return 1
    target_expected = metadata["expected"]

    with Image.open(args.target_input) as handle:
        target_image = handle.convert("RGB")
    defect_image = None
    if args.defect_image and args.defect_image.is_file():
        with Image.open(args.defect_image) as handle:
            defect_image = handle.convert("RGB")

    baseline_engine = build_engine(args.package, args.language, args.baseline_model)
    baseline_probe = baseline_engine.run(target_image, args.language).text
    if "Véuillez" not in baseline_probe:
        print(
            "refusing to run: baseline does not reproduce the target failure on this "
            f"input (got {baseline_probe!r})",
            file=sys.stderr,
        )
        return 1
    print(f"baseline reproduces target failure: {baseline_probe!r}\n")

    candidates = discover_candidates(args.search_root)
    done = load_done(args.out_jsonl) if args.resume else {}
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    seen_sha: dict[str, Path] = {}
    queue: list[tuple[Path, str]] = []
    for path in candidates:
        digest = sha256_of(path)
        if digest in seen_sha:
            continue
        seen_sha[digest] = path
        queue.append((path, digest))
    if args.limit:
        queue = queue[: args.limit]

    print(f"{len(candidates)} file(s), {len(queue)} unique candidate(s)")
    if done:
        print(f"resuming: {len(done)} already evaluated")

    with args.out_jsonl.open("a", encoding="utf-8") as handle:
        for index, (path, digest) in enumerate(queue, 1):
            name = path.relative_to(args.search_root).as_posix()
            if digest in done:
                print(f"[{index}/{len(queue)}] skip (done) {name}")
                continue

            row: dict = {
                "candidate_sha256": digest,
                "candidate_path": name,
                "candidate_bytes": path.stat().st_size,
            }
            try:
                classes = output_classes(path)
                row["output_classes"] = classes
                if classes is not None and classes != 94:
                    row["status"] = "incompatible"
                    row["error"] = f"output classes {classes} != 94"
                    print(f"[{index}/{len(queue)}] INCOMPATIBLE {name} ({classes})")
                else:
                    engine = build_engine(args.package, args.language, path)
                    row.update(
                        evaluate_candidate(
                            baseline_engine,
                            engine,
                            args.language,
                            target_image,
                            defect_image,
                            target_expected,
                        )
                    )
                    row["status"] = "evaluated"
                    print(
                        f"[{index}/{len(queue)}] {name}: "
                        f"target_full {row['target_full_exact']}/{len(PERTURBATIONS)} "
                        f"target_word {row['target_word_fixed']}/{len(PERTURBATIONS)} "
                        f"defect_fp {row['defect_false_pass']}/{len(defect_image and PERTURBATIONS or ())} "
                        f"{'PASS' if row['gate_a_pass'] else 'fail'}"
                    )
            except Exception as exc:  # one bad candidate must not stop the sweep
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["traceback"] = traceback.format_exc(limit=3)
                print(f"[{index}/{len(queue)}] ERROR {name}: {exc}")

            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

    print(f"\nwrote {args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
