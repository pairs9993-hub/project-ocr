"""Train the line verifier under the sealed config.

Three seeds, one config, no tuning between them. The seed is chosen afterwards
on model_selection_v1 alone, by criteria fixed before any result was seen, so
this script must not look at the threshold partition or anything downstream.

The loss is deliberately lopsided. Mis-correcting a real accent is the failure
that reaches a user, so it carries eight times the weight of an ordinary
mistake; failing to correct a hallucination merely leaves the status quo.

Pair consistency applies only to complete pairs. An incomplete pair has no
counterpart to be consistent with, and pulling a lone member toward an absent
partner would be optimising against nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATOR_ROOT))

from ocr_roi_validator.line_verifier_model import CLASS_INDEX, LineVerifier

ACCENT, BARE, UNKNOWN = CLASS_INDEX["ACCENT_PRESENT"], CLASS_INDEX["BARE_E"], \
    CLASS_INDEX["UNKNOWN"]

# An epoch must cover at least this share of the available BARE cases before its
# safety numbers mean anything. Ranking on raw counts alone hands the win to a
# model that never predicts BARE_E: it scores zero false corrections by
# abstaining, which is not safety.
MINIMUM_BARE_COVERAGE = 0.30


class TensorRows(Dataset):
    def __init__(self, npz_paths, manifests, weights=None):
        planes, query, label, centre, source = [], [], [], [], []
        for order, (npz_path, manifest_path) in enumerate(zip(npz_paths, manifests)):
            data = np.load(npz_path)
            rows = json.loads(Path(manifest_path).read_text(encoding="utf-8"))["rows"]
            planes.append(data["planes"])
            query.append(data["query"])
            label.append(data["label"])
            centre.append(data["target_centre"])
            source.extend([(order, r) for r in rows])
        self.planes = np.concatenate(planes)
        self.query = np.concatenate(query)
        self.label = np.concatenate(label)
        self.centre = np.concatenate(centre)
        self.source = source
        self.weights = weights

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return (torch.from_numpy(self.planes[index]),
                torch.from_numpy(self.query[index]),
                int(self.label[index]),
                float(self.centre[index]),
                index)


def attention_target(centre, width, device, sigma=0.06):
    """A soft window at the renderer's target centre, for supervision only."""
    positions = torch.linspace(0, 1, width, device=device).unsqueeze(0)
    centres = centre.unsqueeze(1)
    target = torch.exp(-0.5 * ((positions - centres) / sigma) ** 2)
    return target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)


def evaluate(model, loader, device):
    """Counts that the selection criteria are defined on."""
    model.eval()
    counts = Counter()
    with torch.no_grad():
        for planes, query, label, _, _ in loader:
            planes, query = planes.to(device), query.to(device)
            logits, _ = model(planes, query)
            predicted = logits.argmax(dim=1).cpu()
            for want, got in zip(label.tolist(), predicted.tolist()):
                counts["total"] += 1
                if want == got:
                    counts["correct"] += 1
                # The failure that reaches a user: a real accent called bare.
                if want == ACCENT and got == BARE:
                    counts["accent_false_correction"] += 1
                # An unanswerable query answered confidently either way.
                if want == UNKNOWN and got != UNKNOWN:
                    counts["unknown_violation"] += 1
                if want == BARE:
                    counts["bare_total"] += 1
                    if got == BARE:
                        counts["bare_covered"] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-npz", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--anchor-npz", type=Path)
    parser.add_argument("--anchor-manifest", type=Path)
    parser.add_argument("--selection-npz", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    npz = [args.train_npz]
    manifests = [args.train_manifest]
    if args.anchor_npz and args.anchor_npz.is_file():
        npz.append(args.anchor_npz)
        manifests.append(args.anchor_manifest)
    train = TensorRows(npz, manifests)
    selection = TensorRows([args.selection_npz], [args.selection_manifest])

    # Complete pairs only: both members present in the training tensors.
    pair_members = {}
    for position, (_, row) in enumerate(train.source):
        if row.get("pair_id"):
            pair_members.setdefault(row["pair_id"], []).append(position)
    complete = {k: v for k, v in pair_members.items() if len(v) == 2}
    pair_index = {}
    for members in complete.values():
        pair_index[members[0]] = members[1]
        pair_index[members[1]] = members[0]

    loader = DataLoader(train, batch_size=config["schedule"]["batch_size"],
                        shuffle=True, drop_last=False)
    selection_loader = DataLoader(selection, batch_size=128, shuffle=False)

    model = LineVerifier().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["optimizer"]["learning_rate"],
        weight_decay=config["optimizer"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["schedule"]["max_epochs"])

    asymmetric = config["loss"]["asymmetric_penalty"]["weight"]
    attention_weight = config["loss"]["attention_supervision"]["weight"]
    pair_weight = config["loss"]["pair_consistency"]["weight"]
    dropout = config["loss"]["context_dropout"]["probability"]

    best = None
    history = []
    patience = config["schedule"]["patience"]
    stale = 0
    started = time.time()

    for epoch in range(config["schedule"]["max_epochs"]):
        model.train()
        running = 0.0
        for planes, query, label, centre, index in loader:
            planes = planes.to(device)
            query, label = query.to(device), label.to(device)
            centre = centre.to(device).float()

            if dropout > 0:
                # Blank random columns outside the queried region so the model
                # cannot lean on the rest of the word.
                mask = (torch.rand(planes.shape[0], 1, 1, planes.shape[3],
                                   device=device) > dropout).float()
                planes = planes * mask

            logits, attention = model(planes, query)
            per_sample = F.cross_entropy(logits, label, reduction="none")
            predicted = logits.argmax(dim=1)
            penalty = torch.where((label == ACCENT) & (predicted == BARE),
                                  torch.full_like(per_sample, asymmetric),
                                  torch.ones_like(per_sample))
            loss = (per_sample * penalty).mean()

            target = attention_target(centre, attention.shape[1], device)
            loss = loss + attention_weight * F.kl_div(
                attention.clamp_min(1e-8).log(), target, reduction="batchmean")

            partners = [(position, pair_index[int(i)])
                        for position, i in enumerate(index.tolist())
                        if int(i) in pair_index]
            if partners and pair_weight > 0:
                rows = torch.tensor([p for p, _ in partners], device=device)
                others = torch.stack([
                    torch.from_numpy(train.planes[q]) for _, q in partners
                ]).to(device)
                other_query = torch.stack([
                    torch.from_numpy(train.query[q]) for _, q in partners
                ]).to(device)
                with torch.no_grad():
                    other_logits, _ = model(others, other_query)
                mine = F.log_softmax(logits[rows], dim=1)
                theirs = F.softmax(other_logits, dim=1)
                # Members of a pair differ only in the target glyph, so their
                # UNKNOWN mass should agree even though their classes differ.
                loss = loss + pair_weight * F.kl_div(
                    mine[:, UNKNOWN:UNKNOWN + 1].clamp_min(-30),
                    theirs[:, UNKNOWN:UNKNOWN + 1], reduction="batchmean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss) * planes.shape[0]
        scheduler.step()

        counts = evaluate(model, selection_loader, device)
        # Early stopping watches the safety metrics, but only among epochs that
        # actually do the job. Ranking on raw counts alone hands the win to a
        # model that never predicts BARE_E: it scores zero false corrections by
        # abstaining, which is not safety. An epoch qualifies only once it
        # covers a minimum share of the available BARE cases.
        coverage = (counts["bare_covered"] / counts["bare_total"]
                    if counts["bare_total"] else 0.0)
        qualifies = coverage >= MINIMUM_BARE_COVERAGE
        key = (0 if qualifies else 1,
               counts["accent_false_correction"], counts["unknown_violation"],
               -counts["bare_covered"])
        history.append({"epoch": epoch, "loss": running / len(train),
                        **{k: counts[k] for k in
                           ("total", "correct", "accent_false_correction",
                            "unknown_violation", "bare_total", "bare_covered")}})
        if best is None or key < best[0]:
            best = (key, epoch, {k: v.clone() for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    args.out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.out_dir / ("seed_%d.pt" % args.seed)
    torch.save(best[2], weights_path)
    counts = history[best[1]]
    coverage = (counts["bare_covered"] / counts["bare_total"]
                if counts["bare_total"] else 0.0)
    degenerate = coverage < MINIMUM_BARE_COVERAGE
    report = {
        "degenerate": degenerate,
        "minimum_bare_coverage": MINIMUM_BARE_COVERAGE,
        "selected_coverage": round(coverage, 4),
        "seed": args.seed, "config_sha256": hashlib.sha256(
            args.config.read_bytes()).hexdigest(),
        "train_rows": len(train), "complete_pairs": len(complete),
        "selection_rows": len(selection),
        "best_epoch": best[1], "epochs_run": len(history),
        "selection_counts": counts,
        "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "wall_time_seconds": round(time.time() - started, 1),
        "history": history,
    }
    (args.out_dir / ("seed_%d.json" % args.seed)).write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    if degenerate:
        print("seed %d: DEGENERATE -- no epoch reached %.0f%% BARE coverage"
              % (args.seed, MINIMUM_BARE_COVERAGE * 100))
    print("seed %d: best epoch %d/%d  accent_fc=%d unknown_viol=%d "
          "bare %d/%d  acc %.4f  %.0fs"
          % (args.seed, best[1], len(history), counts["accent_false_correction"],
             counts["unknown_violation"], counts["bare_covered"],
             counts["bare_total"], counts["correct"] / counts["total"],
             report["wall_time_seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
