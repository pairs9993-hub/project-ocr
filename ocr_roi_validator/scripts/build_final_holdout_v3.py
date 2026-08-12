"""Generate the sealed final_holdout_v3 set, without any CNN involvement.

Sample inclusion and the stopping decision use only the synthetic visual ground
truth, the baseline recognizer's output, whether alignment and input
preparation succeeded, and SHA-256 deduplication. The accent CNN is never
loaded here, so no prediction can influence which glyphs end up in the set.

Generation is resumable: every rendering's glyphs are appended to a JSONL
checkpoint as they are produced, and a restart replays the deterministic RNG
stream from the beginning and skips rendering indices already recorded. That
keeps the seed sequence identical across resumes.

The last batch is not trimmed to hit the target exactly. The stop rule is
"finish the rendering that satisfies both quotas", which is deterministic and
independent of what the extra samples happen to contain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATOR_ROOT))
SCRIPTS = VALIDATOR_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_accent_glyph_dataset import (  # noqa: E402
    FONT_SPLITS,
    SIZE_SPLITS,
    TEMPLATE_SPLITS,
    WORD_SPLITS,
    build_engine,
    extract_glyphs,
    load_labels,
    render_phrase,
)

SPLIT = "final_holdout_v3"
SEED = 300000


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_checkpoint(path: Path) -> tuple[list[dict], int, int]:
    """Return (rows, next rendering index, resume count)."""
    if not path.is_file():
        return [], 0, 0
    rows = []
    highest = -1
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(row)
        highest = max(highest, int(row.get("render_index", -1)))
    resumes = int(rows[-1].get("resume_count", 0)) if rows else 0
    return rows, highest + 1, resumes


def existing_crop_digests(*directories: Path) -> set[str]:
    """Crop digests already used by other splits, to prove disjointness.

    Older manifests predate the ``crop_sha256`` field, so the digest is
    recomputed from the stored image when it is absent. The digest covers the
    decoded pixels, matching how new crops are hashed.
    """
    digests = set()
    for directory in directories:
        manifest = directory / "manifest.json"
        if not manifest.is_file():
            continue
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        for sample in payload.get("samples", []):
            digest = sample.get("crop_sha256")
            if not digest:
                image = cv2.imread(str(directory / sample["file"]))
                if image is None:
                    continue
                digest = sha256_of(np.ascontiguousarray(image).tobytes())
            digests.add(digest)
    return digests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    parser.add_argument("--max-renderings", type=int, default=200000)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--other-split", type=Path, action="append", default=[])
    args = parser.parse_args()

    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    target_accent = int(recipe["target_samples"]["in_scope_visual_accent"])
    target_hallucination = int(recipe["target_samples"]["visual_bare_hallucination"])
    print(f"recipe sha256 : {sha256_of(args.recipe.read_bytes())}")
    print(f"targets       : in-scope accent >= {target_accent}, "
          f"hallucination >= {target_hallucination}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "e").mkdir(exist_ok=True)
    (args.out_dir / "accent").mkdir(exist_ok=True)
    checkpoint_path = args.out_dir / "checkpoint.jsonl"

    rows, start_index, previous_resumes = load_checkpoint(checkpoint_path)
    resume_count = previous_resumes + (1 if rows else 0)
    if rows:
        print(f"resuming from rendering {start_index} with {len(rows)} glyphs "
              f"(resume #{resume_count})")

    foreign_digests = existing_crop_digests(*args.other_split)
    print(f"crop digests in other splits: {len(foreign_digests)}")

    engine, package = build_engine(args.package, args.language)
    labels = load_labels(package.dictionary)

    fonts = [str(args.font_dir / f) for f in FONT_SPLITS[SPLIT]
             if (args.font_dir / f).is_file()]
    templates = TEMPLATE_SPLITS[SPLIT]
    words = WORD_SPLITS[SPLIT]
    sizes = SIZE_SPLITS[SPLIT]
    if not fonts:
        print("no fonts available", file=sys.stderr)
        return 1

    seen_digests = {row["crop_sha256"] for row in rows}
    counts = Counter()
    for row in rows:
        counts["accent" if row["visual_label"] == "é" else "e"] += 1
        if row["predicted_char"] == "é":
            counts["in_scope"] += 1
            if row["visual_label"] == "é":
                counts["in_scope_accent"] += 1
            else:
                counts["hallucination"] += 1

    excluded = Counter()
    duplicates = 0
    overlaps = 0

    rng = random.Random(SEED)
    handle = checkpoint_path.open("a", encoding="utf-8")
    try:
        for index in range(args.max_renderings):
            # Draw from the RNG for every index so a resume reproduces the
            # same stream; only rendering is skipped.
            font = rng.choice(fonts)
            size = rng.choice(sizes)
            template = rng.choice(templates)
            accented_word, plain_word = rng.choice(words)
            use_accent = rng.random() < 0.5
            word = accented_word if use_accent else plain_word
            text = template.format(word)
            render_seed = rng.randrange(10 ** 9)

            if index < start_index:
                continue

            if (counts["in_scope_accent"] >= target_accent
                    and counts["hallucination"] >= target_hallucination):
                print(f"\nstop rule satisfied at rendering {index}")
                break

            visual = "é" if "é" in word else "e"
            try:
                image = render_phrase(text, font, size, random.Random(render_seed))
                glyphs = extract_glyphs(engine, image, labels, random.Random(render_seed))
            except Exception as exc:                     # keep the sweep alive
                excluded[f"render_or_ocr_error:{type(exc).__name__}"] += 1
                continue
            if not glyphs:
                excluded["no_alignable_glyph"] += 1
                continue

            for glyph in glyphs:
                crop = glyph["line_crop"][:, glyph["x0"] : glyph["x1"]]
                if crop.size == 0:
                    excluded["empty_crop"] += 1
                    continue
                digest = sha256_of(np.ascontiguousarray(crop).tobytes())
                if digest in seen_digests:
                    duplicates += 1
                    continue
                if digest in foreign_digests:
                    overlaps += 1
                    excluded["duplicate_of_other_split"] += 1
                    continue
                seen_digests.add(digest)

                folder = "accent" if visual == "é" else "e"
                name = f"fh3_{index:06d}_{glyph['position']:03d}_{digest[:16]}.png"
                cv2.imwrite(str(args.out_dir / folder / name), crop)

                row = {
                    "render_index": index,
                    "file": f"{folder}/{name}",
                    "crop_sha256": digest,
                    "visual_label": visual,
                    "predicted_char": glyph["predicted_char"],
                    "in_scope": glyph["predicted_char"] == "é",
                    "font": Path(font).name,
                    "size": size,
                    "template": template,
                    "word": word,
                    "text": text,
                    "render_seed": render_seed,
                    "span_confidence": glyph["span_confidence"],
                    "crop_size": [int(crop.shape[1]), int(crop.shape[0])],
                    "resume_count": resume_count,
                }
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

                counts["accent" if visual == "é" else "e"] += 1
                if row["in_scope"]:
                    counts["in_scope"] += 1
                    counts["in_scope_accent" if visual == "é" else "hallucination"] += 1

            if (index + 1) % args.progress_every == 0:
                handle.flush()
                print(f"  rendering {index + 1}: glyphs {len(rows)}  "
                      f"in-scope accent {counts['in_scope_accent']}/{target_accent}  "
                      f"hallucination {counts['hallucination']}/{target_hallucination}",
                      flush=True)
    finally:
        handle.close()

    met = (counts["in_scope_accent"] >= target_accent
           and counts["hallucination"] >= target_hallucination)

    def cohort(field: str) -> dict:
        return dict(Counter(str(row[field]) for row in rows).most_common())

    manifest = {
        "split": SPLIT,
        "role": "sealed final holdout for accent-v3 F1",
        "seed": SEED,
        "recipe_sha256": sha256_of(args.recipe.read_bytes()),
        "renderings_attempted": (rows[-1]["render_index"] + 1) if rows else 0,
        "total_glyphs": len(rows),
        "by_visual_label": {"accent": counts["accent"], "e": counts["e"]},
        "by_baseline_trigger": {
            "predicted_accent": counts["in_scope"],
            "predicted_e": len(rows) - counts["in_scope"],
        },
        "in_scope_visual_accent": counts["in_scope_accent"],
        "hallucinations": counts["hallucination"],
        "excluded": dict(excluded),
        "duplicates_removed": duplicates,
        "crop_overlap_with_other_splits": overlaps,
        "cohorts": {
            "font": cohort("font"),
            "size": cohort("size"),
            "template": cohort("template"),
            "word": cohort("word"),
        },
        "resume_count": resume_count,
        "checkpoint_rows": len(rows),
        "targets_met": met,
        "samples": rows,
    }
    text = json.dumps(manifest, ensure_ascii=False, indent=2)
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(text, encoding="utf-8")

    label_digest = sha256_of(
        "\n".join(f"{r['file']}\t{r['visual_label']}\t{r['predicted_char']}"
                  for r in rows).encode("utf-8")
    )

    print(f"\nrenderings      : {manifest['renderings_attempted']}")
    print(f"glyphs          : {len(rows)}")
    print(f"visual accent   : {counts['accent']}   visual e: {counts['e']}")
    print(f"in-scope accent : {counts['in_scope_accent']} (target {target_accent})")
    print(f"hallucinations  : {counts['hallucination']} (target {target_hallucination})")
    print(f"duplicates      : {duplicates}   overlap with other splits: {overlaps}")
    print(f"excluded        : {dict(excluded)}")
    print(f"resumes         : {resume_count}")
    print(f"targets met     : {met}")
    print(f"manifest sha256 : {sha256_of(text.encode('utf-8'))}")
    print(f"label   sha256  : {label_digest}")
    return 0 if met else 1


if __name__ == "__main__":
    raise SystemExit(main())
