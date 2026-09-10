"""Aggregate a stage-1 Optuna run: mean/std LOSO accuracy per hyperparameter
combo (over folds), compare against the Adam reference, select the winner.

Decision D1: a trial's `value` is the FINAL-epoch accuracy (not best-epoch);
the winning combo is selected by mean final-epoch accuracy, and V3 compares
that mean against the reference's final_val_acc. Each combo also reports
mean_best_acc (mean of the trainer's best_val_acc user attr, when present),
mean_test_acc (mean of the trainer's test_acc user attr, when present — a
trials.csv column on runs at or after this pin; a run dir whose trials.csv
predates D1 has no test_acc column and gets a WARNING on stderr, since its
`value` is then best-epoch accuracy), and selection_bias_pp =
100*(mean_best_acc - mean_final) — the optimistic bias that best-epoch
selection would have introduced.

Reference handling: rows are filtered by --config-name, then (when the
resolved search-space JSON has a single-element `epochs` list, as every
stage-1 grid does) by that epoch count — rows at other epoch counts are
excluded with a WARNING naming how many and which epochs; with no `epochs`
key or several values the filter is skipped and the reference line says so.
The reference CSV's header is checked against the columns stage1_r0.py
writes (config_name, fold_idx, epochs, final_val_acc, best_val_acc); a file
missing any of them (e.g. the old data/model_and_dataset/summary.csv) exits
2 with a message naming the missing columns instead of a confusing
KeyError. The reference line and output JSON report `reference_n_folds`
(rows kept after filtering) and `reference_epochs` (the epoch count filtered
on, or null when the filter was skipped); when the reference is non-empty
but reference_n_folds != --expect-folds, a WARNING notes the V3 verdict
rests on a partial reference.

Usage: uv run tools/aggregate_stage1.py --run-dir runs/optuna-local/stage1_smoke--<ts> \
           --config-name trial-3650 --expect-folds 2 --reference runs/r0/reference.csv \
           --out stage1_selection.json

--run-dir accepts multiple directories: a base study plus extension studies
over complementary sub-grids (e.g. stage1_trial3408 wd=0.0 + the
stage1_trial3408_ext_wd wd=0.0001 complement) merge into one selection.
All studies must share the same search-space KEY SET (values may differ);
`run_dir` in the output JSON is always a list.

--reference points at the Adam reference CSV written by tools/stage1_r0.py
(columns config_name,fold_idx,optim,lr,weight_decay,epochs,seed,final_val_acc,
best_val_acc,best_epoch,final_test_acc,n_train,n_eval); rows are filtered by
--config-name and epochs (see above) and averaged over fold_idx for both
final_val_acc and best_val_acc.
"""
import argparse
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_REFERENCE = Path("runs/r0/reference.csv")
FOLD_KEY = "fold"
REFERENCE_REQUIRED_COLUMNS = {"config_name", "fold_idx", "epochs", "final_val_acc", "best_val_acc"}


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


def load_reference_rows(reference: Path, config_name: str) -> list[dict]:
    """Rows from `reference` matching config_name. Exits 2 with a friendly
    message if the file exists but isn't a tools/stage1_r0.py CSV (catches
    the old data/model_and_dataset/summary.csv, which has different columns
    and would otherwise fail deep inside with a KeyError)."""
    if not reference.exists():
        print(f"reference file {reference} not found — run tools/stage1_r0.py", file=sys.stderr)
        return []
    with reference.open() as f:
        reader = csv.DictReader(f)
        header = set(reader.fieldnames or [])
        missing = sorted(REFERENCE_REQUIRED_COLUMNS - header)
        if missing:
            print(f"reference CSV {reference} is not a stage1_r0.py output "
                  f"(missing columns: {', '.join(missing)}) — run tools/stage1_r0.py",
                  file=sys.stderr)
            raise SystemExit(2)
        return [r for r in reader if r["config_name"] == config_name]


def filter_reference_epochs(rows: list[dict], epochs_values) -> tuple[list[dict], int | None, str]:
    """Keep only reference rows whose `epochs` matches the sweep's single
    epochs value (every stage-1 grid has exactly one). Returns
    (filtered_rows, reference_epochs_or_None, note_for_reference_line)."""
    if not (isinstance(epochs_values, list) and len(epochs_values) == 1):
        return rows, None, " (epochs filter skipped: search space has no single epochs value)"
    reference_epochs = int(epochs_values[0])
    kept = [r for r in rows if int(r["epochs"]) == reference_epochs]
    excluded = [r for r in rows if int(r["epochs"]) != reference_epochs]
    if excluded:
        excluded_epochs = sorted(set(int(r["epochs"]) for r in excluded))
        print(f"WARNING: excluded {len(excluded)} reference row(s) with epochs "
              f"!= {reference_epochs} (found epochs={excluded_epochs})", file=sys.stderr)
    return kept, reference_epochs, ""


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
    p.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE,
                   help="R0 CSV from tools/stage1_r0.py (final_val_acc/best_val_acc per config, fold)")
    p.add_argument("--out", type=Path, default=Path("stage1_selection.json"))
    args = p.parse_args()

    search_space_path = resolve_search_space(args.search_space, args.run_dir)
    search_space = json.loads(search_space_path.read_text())
    space_keys = [k for k in search_space if k != FOLD_KEY]
    check_key_sets(space_keys, args.run_dir)

    rows = []
    for di, rd in enumerate(args.run_dir):
        with (rd / "trials.csv").open() as f:
            reader = csv.DictReader(f)
            if "test_acc" not in (reader.fieldnames or []):
                print(f"WARNING: {rd} predates D1 (trials.csv has no test_acc column; "
                      "its value is best-epoch accuracy)", file=sys.stderr)
            for r in reader:
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

    ref_rows = load_reference_rows(args.reference, args.config_name)
    ref_rows, reference_epochs, epochs_note = filter_reference_epochs(ref_rows, search_space.get("epochs"))
    ref_final = [float(r["final_val_acc"]) for r in ref_rows]
    ref_best = [float(r["best_val_acc"]) for r in ref_rows]
    n_ref = len(ref_final)
    ref_final_mean = st.mean(ref_final) if ref_final else float("nan")
    ref_best_mean = st.mean(ref_best) if ref_best else float("nan")
    if ref_final and n_ref != args.expect_folds:
        print(f"WARNING: reference has {n_ref} rows, expected {args.expect_folds} "
              "— V3 rests on a partial reference", file=sys.stderr)

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
        best_accs = [float(r["best_val_acc"]) for r in deduped if r.get("best_val_acc") not in (None, "")]
        test_accs = [float(r["test_acc"]) for r in deduped if r.get("test_acc") not in (None, "")]
        mean_final = st.mean(accs)
        mean_best = st.mean(best_accs) if len(best_accs) == len(accs) else float("nan")
        mean_test = st.mean(test_accs) if len(test_accs) == len(accs) else float("nan")
        complete = len(deduped) == args.expect_folds
        entry = {
            "params": dict(key),
            "n_folds": len(deduped),
            "complete": complete,
            "mean_acc": mean_final,
            "mean_best_acc": mean_best,
            "mean_test_acc": mean_test,
            "selection_bias_pp": 100.0 * (mean_best - mean_final),
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
              f"min={entry['min_acc']:.4f} best={entry['mean_best_acc']:.4f} "
              f"test={entry['mean_test_acc']:.4f} "
              f"bias={entry['selection_bias_pp']:+.1f}pp ({len(deduped)} folds){flag}")

    complete_combos = [e for e in report if e["complete"]]
    if not complete_combos:
        print("no complete combos — nothing to select", file=sys.stderr)
        return 1
    best = max(complete_combos, key=lambda e: e["mean_acc"])
    gap = best["mean_acc"] - ref_final_mean
    print(f"\nreference ({args.config_name}, Adam, final epoch, {n_ref} folds): "
          f"mean={ref_final_mean:.4f} (best-epoch mean {ref_best_mean:.4f}){epochs_note}")
    print(f"best SGD combo (final epoch): mean={best['mean_acc']:.4f} (gap {gap:+.4f}) "
          f"best-epoch mean={best['mean_best_acc']:.4f} bias={best['selection_bias_pp']:+.1f}pp "
          f"params={best['params']}")
    print(f"V3 ({'PASS' if gap >= -0.03 else 'FAIL'}): within 3pp of the final-epoch reference"
          if ref_final else "V3: no reference rows found")

    args.out.write_text(json.dumps(
        {"config_name": args.config_name, "reference_csv": str(args.reference),
         "reference_final_mean": ref_final_mean, "reference_best_mean": ref_best_mean,
         "reference_n_folds": n_ref, "reference_epochs": reference_epochs,
         "run_dir": [str(d) for d in args.run_dir], "best": best,
         "all_combos": report}, indent=2))
    print(f"selection -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
