"""Extend exposure on the sixteen words that already showed some support.

Discovery gave every candidate 400 renderings. Four cleared the ROBUST bar,
sixteen produced between one and four events, and 130 produced none. A word
with four events in 400 renderings is not the same as a word with zero: the
first has demonstrated the baseline can be made to hallucinate on it, the
second has only shown that 400 renderings did not find it.

So the sixteen get more exposure -- 1,600 renderings each, all sixteen, whether
or not they look promising. Extending only the better ones would tie exposure
to outcome and corrupt the recomputed rates. The 130 unobserved words get
nothing further, as instructed; screening a thousand new candidates is not the
route being taken.

The extension continues each word's own stream: rendering 400 of a word follows
399, from the same recipe and seed base, so the combined 2,000 are one sample
rather than two. Support is then reclassified on the composite under the
original sealed rules, unchanged.

Output stays development_word_support_diagnostic_only. No row may enter model
training or any quota.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
for extra in (VALIDATOR_ROOT, VALIDATOR_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from discover_word_support_v1 import (  # noqa: E402
    DISCOVERY_FONTS, DISCOVERY_SEED, DISCOVERY_TEMPLATES, DiscoveryCase,
    evaluate,
)
from generate_v2_split import WriterLock  # noqa: E402
from mine_line_triggers_v1 import build_engine, load_labels  # noqa: E402
from ocr_roi_validator.diagnostic_runner import (  # noqa: E402
    CheckpointWriter, atomic_write_json, load_checkpoint, log_line,
)
from ocr_roi_validator.terminal_reason import derive_flags  # noqa: E402
from ocr_roi_validator.word_candidates import (  # noqa: E402
    CANDIDATES, RENDERINGS_PER_WORD,
)

ADDITIONAL_PER_WORD = 1600


def sparse_words(registry_path: Path) -> list[str]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))["registry"]
    return sorted(w for w, v in registry.items()
                  if v["support"] == "SPARSE_SUPPORT")


class ExtensionCase(DiscoveryCase):
    """A discovery case at an index beyond the original 400 for its word."""

    def __init__(self, word_index: int, offset: int) -> None:
        # Index arithmetic mirrors DiscoveryCase: word_index * 400 + within.
        # Continuing past 400 would spill into the next word, so the stream is
        # extended by offsetting into a reserved block instead.
        super().__init__(word_index * RENDERINGS_PER_WORD)
        self._reinit(word_index, offset)

    def _reinit(self, word_index: int, offset: int) -> None:
        import random
        from ocr_roi_validator.v2_recipes import MACRO_STRATA, STRATUM_TARGETS

        within = RENDERINGS_PER_WORD + offset
        # Negative indices keep the extension's namespace disjoint from
        # discovery's 0..59999, so a composite of the two cannot collide.
        self.index = -(word_index * 100000 + within)
        self.candidate = CANDIDATES[word_index]
        self.font = DISCOVERY_FONTS[within % len(DISCOVERY_FONTS)]
        self.stratum = MACRO_STRATA[within % len(MACRO_STRATA)]
        self.template = DISCOVERY_TEMPLATES[within % len(DISCOVERY_TEMPLATES)]
        self.accented = (within % 2 == 1)

        target = STRATUM_TARGETS[self.stratum]
        rng = random.Random(DISCOVERY_SEED + word_index * 100000 + within)
        self.size = target["sizes"][within % len(target["sizes"])]
        self.upscale = round(rng.uniform(*target["upscale"]), 4)
        self.pad_x = rng.randint(12, 30)
        self.pad_y = 20
        dark = rng.random() < 0.75
        self.background = rng.randint(8, 34) if dark else rng.randint(226, 250)
        self.foreground = rng.randint(220, 252) if dark else rng.randint(8, 40)
        self.contrast = round(rng.uniform(0.85, 1.20), 4)
        self.blur = round(rng.uniform(0.0, 0.5), 4)
        self.jitter_x = round(rng.uniform(-0.5, 0.5), 4)
        self.jitter_y = round(rng.uniform(-0.5, 0.5), 4)
        self.resample = rng.choice(["bicubic", "lanczos"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seal-only", action="store_true")
    args = parser.parse_args()

    words = sparse_words(args.registry)
    index_of = {c.bare: i for i, c in enumerate(CANDIDATES)}
    missing = [w for w in words if w not in index_of]
    if missing:
        print(f"words not in the candidate set: {missing}", file=sys.stderr)
        return 2

    plan = [(word, index_of[word], offset)
            for word in words for offset in range(ADDITIONAL_PER_WORD)]
    total = min(args.limit or len(plan), len(plan))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    recipe = {
        "dataset": "sparse_word_extension_v1",
        "role": "development_word_support_diagnostic_only",
        "prohibited_uses": ["model training", "any split quota",
                            "threshold determination", "safety gate evidence"],
        "parent_registry_sha256": hashlib.sha256(
            args.registry.read_bytes()).hexdigest(),
        "sparse_words": words,
        "word_count": len(words),
        "additional_per_word": ADDITIONAL_PER_WORD,
        "total_additional_renderings": len(words) * ADDITIONAL_PER_WORD,
        "all_sparse_words_extended_equally": True,
        "not_observed_words_extended": False,
        "support_rules_unchanged": True,
        "composite_rule": ("support is reclassified on the original 400 plus "
                           "these 1,600, under the rules sealed before "
                           "discovery"),
        "stopping_rule": ("every word renders all 1,600; extending only the "
                          "promising ones would tie exposure to outcome"),
        "fonts": list(DISCOVERY_FONTS),
        "templates": list(DISCOVERY_TEMPLATES),
        "seed_base": DISCOVERY_SEED,
        "index_namespace": ("negative indices, disjoint from discovery's "
                            "0..59999"),
    }
    recipe_path = args.out_dir / "recipe.json"
    if not recipe_path.is_file():
        atomic_write_json(recipe_path, recipe)
    digest = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    log_line(f"extension recipe sha256 {digest}")
    log_line(f"{len(words)} sparse words x {ADDITIONAL_PER_WORD} = "
             f"{len(plan):,} renderings")
    if args.seal_only:
        return 0

    engine, package = build_engine(args.package, args.language)
    labels = load_labels(package.dictionary)
    recognizer = engine.text_rec
    _, rec_height, rec_width = recognizer.rec_image_shape

    checkpoint = args.out_dir / "checkpoint.jsonl"
    state = load_checkpoint(checkpoint, unit_field="index")
    done = state.digests
    log_line(f"resuming with {len(state.rows)} rows "
             f"(resume #{state.resume_count}) pid {os.getpid()}")

    counts = Counter()
    started = time.time()
    with WriterLock(checkpoint):
        with CheckpointWriter(checkpoint, done, flush_every=100) as writer:
            for position in range(total):
                word, word_index, offset = plan[position]
                case = ExtensionCase(word_index, offset)
                digest_key = hashlib.sha256(
                    f"ext|{word}|{offset}".encode()).hexdigest()
                if digest_key in done:
                    continue
                row = evaluate(case, args.font_dir, engine, recognizer, labels,
                               rec_height, rec_width)
                row["row_digest"] = digest_key
                row["extension"] = True
                row["diagnostic_flags"] = derive_flags(row)
                writer.append(row)
                counts[row["terminal_reason"]] += 1
                if (position + 1) % args.progress_every == 0:
                    writer.sync()
                    log_line(f"  {position + 1}/{total} "
                             f"h={counts['CLEAN_HALLUCINATION']} "
                             f"{time.time() - started:.0f}s")
        duplicates = writer.duplicates_rejected

    log_line(f"finished, {duplicates} duplicates, {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
