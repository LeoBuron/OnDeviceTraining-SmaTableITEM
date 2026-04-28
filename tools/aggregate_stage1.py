"""Aggregate a stage-1 Optuna run: mean/std LOSO accuracy per hyperparameter
combo (over folds), compare against the Adam reference, select the winner.

Usage: uv run tools/aggregate_stage1.py --run-dir runs/optuna-local/stage1_smoke--<ts> \
           --config-name trial-3650 --expect-folds 2 --out stage1_selection.json
"""
import argparse
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REFERENCE = Path("data/model_and_dataset/summary.csv")
FOLD_KEY = "fold"
NON_COMBO = {"trial", "state", "value", FOLD_KEY, "n_params", "best_epoch", "wall_clock_s",
             "ckpt_dir"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--config-name", required=True, help="e.g. trial-4223 (reference lookup)")
    p.add_argument("--expect-folds", type=int, default=15)
    p.add_argument("--out", type=Path, default=Path("stage1_selection.json"))
    args = p.parse_args()

    rows = list(csv.DictReader((args.run_dir / "trials.csv").open()))
    combos = defaultdict(list)
    for r in rows:
        if r["state"] != "COMPLETE":
            continue
        key = tuple((k, r[k]) for k in sorted(r) if k not in NON_COMBO)
        combos[key].append(r)

    ref_accs = [float(r["best_val_acc"]) for r in csv.DictReader(REFERENCE.open())
                if r["config_name"] == args.config_name]
    ref_mean = st.mean(ref_accs) if ref_accs else float("nan")

    report = []
    for key, rs in sorted(combos.items()):
        accs = [float(r["value"]) for r in rs]
        complete = len(rs) == args.expect_folds
        entry = {
            "params": dict(key),
            "n_folds": len(rs),
            "complete": complete,
            "mean_acc": st.mean(accs),
            "std_acc": st.stdev(accs) if len(accs) > 1 else 0.0,
            "min_acc": min(accs),
            "folds": sorted(
                ({"fold": int(r[FOLD_KEY]), "accuracy": float(r["value"]),
                  "ckpt": str(args.run_dir / f"trial_{int(r['trial']):05d}" / "ckpt")}
                 for r in rs), key=lambda d: d["fold"]),
        }
        report.append(entry)
        flag = "" if complete else "  [INCOMPLETE]"
        print(f"{dict(key)}: mean={entry['mean_acc']:.4f} +/- {entry['std_acc']:.4f} "
              f"min={entry['min_acc']:.4f} ({len(rs)} folds){flag}")

    complete_combos = [e for e in report if e["complete"]]
    if not complete_combos:
        print("no complete combos — nothing to select", file=sys.stderr)
        return 1
    best = max(complete_combos, key=lambda e: e["mean_acc"])
    gap = best["mean_acc"] - ref_mean
    print(f"\nreference ({args.config_name}, Adam): mean={ref_mean:.4f}")
    print(f"best SGD combo: mean={best['mean_acc']:.4f} (gap {gap:+.4f}) "
          f"params={best['params']}")
    print(f"V3 ({'PASS' if gap >= -0.03 else 'FAIL'}): within 3pp of reference"
          if ref_accs else "V3: no reference rows found")

    args.out.write_text(json.dumps(
        {"config_name": args.config_name, "reference_mean": ref_mean,
         "run_dir": str(args.run_dir), "best": best, "all_combos": report}, indent=2))
    print(f"selection -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
