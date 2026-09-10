"""R0 — Adam reference for the stage-1 V3 gate on the session-wise LOSO
split (decision D1, 2026-09-06; spec docs/superpowers/specs/2026-09-10-d1-d5-protocol-change-design.md).

For every requested (config, fold) this trains the PyTorch DepthwiseCNN
(tools/stage1_reference.py) with the trial's own training config from
data/model_and_dataset/<config>/config.json — widths, kernel_size,
conv_dilation, p_drop, lr, batch_size, n_epochs — under Adam, cosine
annealing and weight_decay 0 (assumed Adam defaults until Florian confirms),
and appends one CSV row that tools/aggregate_stage1.py --reference reads.
Resumable: (config, fold) rows already in the CSV are skipped.

Usage:
    uv run tools/stage1_r0.py --configs trial-3650 --folds 0-14 --out runs/r0/reference.csv
    uv run tools/stage1_r0.py --configs trial-2353,trial-3408,trial-3650,trial-4223 --folds 0-14 --out runs/r0/reference.csv --device mps
Smoke: uv run tools/stage1_r0.py --configs trial-3650 --folds 0 --epochs 2 --out $SCRATCH/r0.csv
"""
import argparse
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_reference import train_run  # noqa: E402

COLUMNS = ["config_name", "fold_idx", "optim", "lr", "weight_decay", "epochs", "seed",
           "final_val_acc", "best_val_acc", "best_epoch", "final_test_acc", "n_train", "n_eval"]


def parse_folds(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def done_keys(csv_path: Path) -> set[tuple[str, int]]:
    if not csv_path.exists():
        return set()
    with csv_path.open() as f:
        return {(r["config_name"], int(r["fold_idx"])) for r in csv.DictReader(f)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--configs", required=True, help="comma-separated, e.g. trial-3650,trial-4223")
    p.add_argument("--folds", default="0-14")
    p.add_argument("--out", type=Path, default=Path("runs/r0/reference.csv"))
    p.add_argument("--model-root", type=Path, default=Path("data/model_and_dataset"))
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--epochs", type=int, default=None, help="override config n_epochs (smoke tests)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(args.out)
    write_header = not args.out.exists()
    for config in [c for c in args.configs.split(",") if c]:
        cfg = json.loads((args.model_root / config / "config.json").read_text())
        data_dir = args.data_root / f"smatable-{config}"
        epochs = args.epochs or int(cfg["n_epochs"])
        for fold in parse_folds(args.folds):
            if (config, fold) in done:
                print(f"skip {config} fold {fold}: already in {args.out}")
                continue
            run_args = Namespace(
                data_dir=str(data_dir), fold=fold, widths=",".join(str(w) for w in cfg["widths"]),
                kernel=int(cfg["kernel_size"]), dilation=int(cfg["conv_dilation"]),
                seed=args.seed, batch=int(cfg["batch_size"]), norm="groupnorm",
                p_drop=float(cfg["p_drop"]), optim="adam", lr=float(cfg["lr"]), momentum=0.0,
                weight_decay=args.weight_decay, epochs=epochs, device=args.device)
            print(f"== {config} fold {fold}: {run_args}")
            r = train_run(run_args)
            row = {"config_name": config, "fold_idx": fold, "optim": "adam", "lr": run_args.lr,
                   "weight_decay": args.weight_decay, "epochs": epochs, "seed": args.seed, **r}
            with args.out.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=COLUMNS)
                if write_header:
                    w.writeheader()
                    write_header = False
                w.writerow(row)
            done.add((config, fold))
            print(f"RESULT {config} fold {fold}: final={r['final_val_acc']:.4f} "
                  f"best={r['best_val_acc']:.4f}@{r['best_epoch']} test={r['final_test_acc']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
