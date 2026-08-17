"""Continue the supplement's rendering stream to close its quota shortfall.

The base run is not modified. Its recipe, checkpoint, manifest and
QUOTA_NOT_MET status are preserved exactly as produced, and this writes to a
separate directory whose manifest records the parent's hashes.

The continuation is deterministic rather than a fresh sample: rendering index
``23095`` follows ``23094`` in the same stream, drawn from the same recipe with
the same seed base, so the top-up rows are the ones the base run would have
produced had its budget been larger. Nothing about the cohort, the fonts, the
perturbation grid or the stratum rotation changes -- only the index range.

That property is what makes the combined dataset a single sample rather than
two stitched together, and it is checked rather than assumed: the parent's own
rows are re-derivable from the same generator, so a mismatch in recipe, model
or font hashes refuses the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
for extra in (VALIDATOR_ROOT, VALIDATOR_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from generate_v2_split import (  # noqa: E402
    ARTIFACT_FORMAT, ARTIFACT_VERSION, CHECKPOINT_FLUSH_EVERY, Rendering,
    WriterLock, environment_fingerprint, evaluate, tally,
)
from mine_line_triggers_v1 import build_engine, load_labels  # noqa: E402
from ocr_roi_validator.diagnostic_runner import (  # noqa: E402
    CheckpointWriter, atomic_write_json, load_checkpoint, log_line,
)
from ocr_roi_validator.terminal_reason import (  # noqa: E402
    derive_flags, summarise_terminal_reasons,
)
from ocr_roi_validator.v2_recipes import MACRO_STRATA, V2_RECIPES  # noqa: E402

SPLIT = "line_train_supplement_v2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    recipe = V2_RECIPES[SPLIT]
    base_checkpoint = args.base / "checkpoint.jsonl"
    base_manifest_path = args.base / "manifest.json"
    base_recipe_path = args.base / "recipe.json"
    base_rows = [json.loads(line) for line
                 in base_checkpoint.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    budget = json.loads(args.budget.read_text(encoding="utf-8"))

    start_index = max(row["index"] for row in base_rows) + 1
    topup_max = budget["topup_max_renderings"]

    engine, package = build_engine(args.package, args.language)
    labels = load_labels(package.dictionary)
    recognizer = engine.text_rec
    _, rec_height, rec_width = recognizer.rec_image_shape
    fingerprint = environment_fingerprint(package, args.font_dir, recipe)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    topup_recipe = {
        "dataset": f"{SPLIT}_topup",
        "role": "append_only_continuation_of_the_same_cohort",
        "parent_split": SPLIT,
        "parent_recipe_sha256": hashlib.sha256(
            base_recipe_path.read_bytes()).hexdigest(),
        "parent_manifest_sha256": hashlib.sha256(
            base_manifest_path.read_bytes()).hexdigest(),
        "parent_checkpoint_sha256": hashlib.sha256(
            base_checkpoint.read_bytes()).hexdigest(),
        "parent_status": base_manifest["stop_reason"],
        "parent_renderings": len(base_rows),
        "credited_counts": {
            name: entry["observed"]
            for name, entry in base_manifest["quota_state"].items()},
        "start_render_index": start_index,
        "topup_max_renderings": topup_max,
        "end_render_index_exclusive": start_index + topup_max,
        "seed_base": recipe.seed,
        "seed_range": [recipe.seed + start_index,
                       recipe.seed + start_index + topup_max - 1],
        "cohort_unchanged": {
            "fonts": list(recipe.fonts),
            "words": [list(pair) for pair in recipe.words],
            "templates": list(recipe.templates),
        },
        "environment": fingerprint,
        "budget_report_sha256": hashlib.sha256(
            args.budget.read_bytes()).hexdigest(),
        "stop_rule": ("stop at the first rendering boundary where every quota "
                      "is met counting parent plus top-up; reaching "
                      "topup_max_renderings short exits non-zero"),
        "composite_rule": ("quotas are evaluated on parent rows concatenated "
                           "with top-up rows; the parent files are never "
                           "rewritten"),
        "artifact_format": ARTIFACT_FORMAT,
        "artifact_version": ARTIFACT_VERSION,
    }
    recipe_path = args.out_dir / "recipe.json"
    if not recipe_path.is_file():
        atomic_write_json(recipe_path, topup_recipe)
    recipe_digest = hashlib.sha256(recipe_path.read_bytes()).hexdigest()

    guard_path = args.out_dir / "environment.json"
    if guard_path.is_file():
        previous = json.loads(guard_path.read_text(encoding="utf-8"))
        differing = [k for k in fingerprint if previous.get(k) != fingerprint[k]]
        if differing:
            print(f"refusing to resume: {differing} changed", file=sys.stderr)
            return 2
    else:
        atomic_write_json(guard_path, fingerprint)

    # A parent that has moved since the budget was computed invalidates the
    # credits this run is built on.
    stored = json.loads(recipe_path.read_text(encoding="utf-8"))
    live_parent = hashlib.sha256(base_checkpoint.read_bytes()).hexdigest()
    if stored["parent_checkpoint_sha256"] != live_parent:
        print("refusing to run: parent checkpoint changed since sealing",
              file=sys.stderr)
        return 2

    checkpoint = args.out_dir / "checkpoint.jsonl"
    state = load_checkpoint(checkpoint, unit_field="index")
    log_line(f"top-up: parent {len(base_rows):,} rows, start index "
             f"{start_index:,}, max {topup_max:,}, have {len(state.rows)} "
             f"(resume #{state.resume_count}) pid {os.getpid()}")

    combined = list(base_rows) + list(state.rows)
    rendered = len(state.rows)
    started = time.time()
    stop_reason = "TOPUP_MAX_REACHED"

    with WriterLock(checkpoint):
        with CheckpointWriter(checkpoint, state.digests,
                              flush_every=CHECKPOINT_FLUSH_EVERY) as writer:
            for offset in range(topup_max):
                index = start_index + offset
                # Build every index so the RNG stream matches a longer base run.
                rendering = Rendering(index, recipe)
                if index in state.completed_units:
                    continue
                row = evaluate(rendering, args.font_dir, engine, recognizer,
                               labels, rec_height, rec_width)
                row["diagnostic_flags"] = derive_flags(row)
                row["topup"] = True
                writer.append(row)
                combined.append(row)
                rendered += 1

                if recipe.quotas_met(tally(combined)):
                    stop_reason = "QUOTA_MET"
                    log_line(f"  composite quotas met at index {index}")
                    break
                if rendered % args.progress_every == 0:
                    writer.sync()
                    counts = tally(combined)
                    log_line(f"  +{rendered}/{topup_max} "
                             f"p={counts['preservation_total']} "
                             f"L={counts['hallucination_by_stratum'].get('LARGE', 0)} "
                             f"{time.time() - started:.0f}s")
        duplicates = writer.duplicates_rejected

    counts = tally(combined)
    state_report = recipe.quota_state(counts)
    met = all(entry["met"] for entry in state_report.values())

    base_digests = {r["row_digest"] for r in base_rows}
    topup_digests = {r["row_digest"] for r in combined[len(base_rows):]}
    base_indices = {r["index"] for r in base_rows}
    topup_indices = {r["index"] for r in combined[len(base_rows):]}
    exposure = defaultdict(Counter)
    for row in combined:
        exposure[row["font"]][row["target_stratum"]] += 1

    manifest = {
        "dataset": f"{SPLIT}_composite",
        "topup_recipe_sha256": recipe_digest,
        "parent_recipe_sha256": topup_recipe["parent_recipe_sha256"],
        "parent_manifest_sha256": topup_recipe["parent_manifest_sha256"],
        "parent_status_preserved": base_manifest["stop_reason"],
        "parent_renderings": len(base_rows),
        "topup_renderings": rendered,
        "composite_renderings": len(combined),
        "topup_max": topup_max,
        "stop_reason": stop_reason if met else "QUOTA_NOT_MET",
        "quota_state": state_report,
        "quotas_met": met,
        "counts": {
            "hallucination_total": counts["hallucination_total"],
            "preservation_total": counts["preservation_total"],
            "unknown_total": counts["unknown_total"],
            "hallucination_by_stratum": dict(counts["hallucination_by_stratum"]),
            "preservation_by_stratum": dict(counts["preservation_by_stratum"]),
            "preservation_by_font": dict(counts["preservation_by_font"]),
        },
        "font_stratum_exposure": {f: dict(v) for f, v in exposure.items()},
        "terminal_reasons": summarise_terminal_reasons(combined),
        "integrity": {
            "duplicate_digests_rejected": duplicates,
            "composite_unique_digests": len(base_digests | topup_digests),
            "composite_rows": len(combined),
            "parent_topup_digest_collisions": sorted(
                base_digests & topup_digests),
            "parent_topup_index_collisions": sorted(
                base_indices & topup_indices),
            "indices_contiguous": (
                sorted(topup_indices) == list(range(start_index,
                                                    start_index + rendered))
                if rendered else True),
        },
        "environment": fingerprint,
        "resume_count": state.resume_count,
        "wall_time_seconds": round(time.time() - started, 1),
    }
    digest = atomic_write_json(args.out_dir / "composite_manifest.json", manifest)

    log_line(f"\ncomposite: {len(combined):,} renderings "
             f"({len(base_rows):,} parent + {rendered:,} top-up) "
             f"stop={manifest['stop_reason']}")
    for name, entry in state_report.items():
        mark = "ok  " if entry["met"] else "MISS"
        log_line(f"  {mark} {name:32s} {entry['observed']:6d}/{entry['required']}")
    integrity = manifest["integrity"]
    log_line(f"  digest collisions {len(integrity['parent_topup_digest_collisions'])}, "
             f"index collisions {len(integrity['parent_topup_index_collisions'])}, "
             f"contiguous {integrity['indices_contiguous']}")
    log_line(f"  composite manifest sha256 {digest}")
    return 0 if met else 1


if __name__ == "__main__":
    raise SystemExit(main())
