"""Aggregate a stage-1 Optuna run: mean/std LOSO accuracy per hyperparameter
combo (over folds), compare against the Adam reference, select the winner.

Usage: uv run tools/aggregate_stage1.py --run-dir runs/optuna-local/stage1_smoke--<ts> \
           --config-name trial-3650 --expect-folds 2 --out stage1_selection.json

--run-dir accepts multiple directories: a base study plus extension studies
over complementary sub-grids (e.g. stage1_trial3408 wd=0.0 + the
stage1_trial3408_ext_wd wd=0.0001 complement) merge into one selection.
All studies must share the same search-space KEY SET (values may differ);
`run_dir` in the output JSON is always a list.
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


def resolve_search_space(explicit: Path | None, run_dirs: list[Path]) -> Path:
    """Explicit --search-space wins; else fall back to the copy run_optuna.py
    drops into the FIRST run dir. Exit 2 (not an exception) if neither exists —
    this is a usage error, not a bug in the run."""
    if explicit is not None:
        if not explicit.exists():
            print(f"--search-space not found: {explicit}", file=sys.stderr)
            raise SystemExit(2)
        return explicit
    fallback = run_dirs[0] / "search_space.json"
    if fallback.exists():
        return fallback
    print(
        "no search-space file found — pass --search-space <path> explicitly, "
        f"or ensure it was copied to {fallback} by run_optuna.py",
        file=sys.stderr,
    )
    raise SystemExit(2)


def check_key_sets(space_keys: list[str], run_dirs: list[Path]) -> None:
    """Merging studies whose grids differ in KEYS (not values) would silently
    fragment combos — refuse. Extension studies vary values only."""
    expected = set(space_keys) | {FOLD_KEY}
    for rd in run_dirs:
        copy = rd / "search_space.json"
        if not copy.exists():
            continue
        got = set(json.loads(copy.read_text()))
        if got != expected:
            print(f"search-space key set of {copy} {sorted(got)} does not match "
                  f"the resolved key set {sorted(expected)} — refusing to merge",
                  file=sys.stderr)
            raise SystemExit(2)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True, nargs="+",
                   help="one or more Optuna run dirs; extras are extension "
                        "studies over complementary sub-grids, merged in")
    p.add_argument("--config-name", required=True, help="e.g. trial-4223 (reference lookup)")
    p.add_argument("--expect-folds", type=int, default=15)
    p.add_argument("--search-space", type=Path, default=None,
                   help="search-space JSON used for this run (defines the combo key). "
                        "Defaults to <run-dir>/search_space.json.")
    p.add_argument("--out", type=Path, default=Path("stage1_selection.json"))
    args = p.parse_args()

    search_space_path = resolve_search_space(args.search_space, args.run_dir)
    space_keys = [k for k in json.loads(search_space_path.read_text()) if k != FOLD_KEY]
    check_key_sets(space_keys, args.run_dir)

    rows = []
    for di, rd in enumerate(args.run_dir):
        for r in csv.DictReader((rd / "trials.csv").open()):
            r["_run_dir"] = rd
            r["_dir_index"] = di
            rows.append(r)
    # CLI order then trial number: dedup keeps the earliest study's row.
    rows.sort(key=lambda r: (r["_dir_index"], int(r["trial"])))
    combos = defaultdict(list)
    for r in rows:
        if r["state"] != "COMPLETE":
            continue
        key = tuple((k, r[k]) for k in sorted(space_keys))
        combos[key].append(r)

    ref_accs = [float(r["best_val_acc"]) for r in csv.DictReader(REFERENCE.open())
                if r["config_name"] == args.config_name]
    ref_mean = st.mean(ref_accs) if ref_accs else float("nan")

    report = []
    for key, rs in sorted(combos.items()):
        # GridSampler in distributed mode can schedule the same grid cell
        # twice; keep the first COMPLETE row (by trial number, already
        # sorted above) per fold and warn about the rest.
        by_fold: dict[str, dict] = {}
        for r in rs:
            fold = r[FOLD_KEY]
            if fold in by_fold:
                print(f"duplicate trial {r['trial']} ({r['_run_dir']}) for fold "
                      f"{fold} ignored (GridSampler distributed re-run or "
                      f"cross-study overlap)", file=sys.stderr)
                continue
            by_fold[fold] = r
        deduped = list(by_fold.values())

        accs = [float(r["value"]) for r in deduped]
        complete = len(deduped) == args.expect_folds
        entry = {
            "params": dict(key),
            "n_folds": len(deduped),
            "complete": complete,
            "mean_acc": st.mean(accs),
            "std_acc": st.stdev(accs) if len(accs) > 1 else 0.0,
            "min_acc": min(accs),
            "folds": sorted(
                ({"fold": int(r[FOLD_KEY]), "accuracy": float(r["value"]),
                  "ckpt": str(r["_run_dir"] / f"trial_{int(r['trial']):05d}" / "ckpt")}
                 for r in deduped), key=lambda d: d["fold"]),
        }
        report.append(entry)
        flag = "" if complete else "  [INCOMPLETE]"
        print(f"{dict(key)}: mean={entry['mean_acc']:.4f} +/- {entry['std_acc']:.4f} "
              f"min={entry['min_acc']:.4f} ({len(deduped)} folds){flag}")

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
         "run_dir": [str(d) for d in args.run_dir], "best": best,
         "all_combos": report}, indent=2))
    print(f"selection -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
