"""Fixture-driven check of tools/aggregate_stage1.py under decision D1:
winner by mean FINAL accuracy, V3 against the reference's final_val_acc,
selection-bias column = 100*(mean best - mean final), reference via --reference.

Run: uv run tests/aggregate_stage1_check.py
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def write_run(run_dir: Path, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "search_space.json").write_text(json.dumps({"fold": [0, 1], "lr": [0.1, 0.01]}))
    cols = ["trial", "state", "value", "fold", "lr", "best_epoch", "best_val_acc"]
    with (run_dir / "trials.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # lr=0.1: final 0.70/0.80 (mean 0.75), best 0.80/0.90 (mean 0.85) -> bias +10 pp
        # lr=0.01: final 0.60/0.70 (mean 0.65) but best 0.95/0.95 -> must NOT win under D1
        write_run(td / "run", [
            {"trial": 0, "state": "COMPLETE", "value": 0.70, "fold": 0, "lr": 0.1, "best_epoch": 5, "best_val_acc": 0.80},
            {"trial": 1, "state": "COMPLETE", "value": 0.80, "fold": 1, "lr": 0.1, "best_epoch": 7, "best_val_acc": 0.90},
            {"trial": 2, "state": "COMPLETE", "value": 0.60, "fold": 0, "lr": 0.01, "best_epoch": 2, "best_val_acc": 0.95},
            {"trial": 3, "state": "COMPLETE", "value": 0.70, "fold": 1, "lr": 0.01, "best_epoch": 3, "best_val_acc": 0.95},
            {"trial": 4, "state": "PRUNED", "value": "", "fold": 1, "lr": 0.01, "best_epoch": "", "best_val_acc": ""},
        ])
        ref = td / "reference.csv"
        ref.write_text("config_name,fold_idx,optim,lr,weight_decay,epochs,seed,final_val_acc,best_val_acc,best_epoch,final_test_acc,n_train,n_eval\n"
                       "trial-x,0,adam,0.001,0,250,42,0.76,0.80,90,0.75,7560,600\n"
                       "trial-x,1,adam,0.001,0,250,42,0.78,0.82,80,0.77,7560,600\n"
                       "trial-y,0,adam,0.001,0,250,42,0.10,0.20,1,0.1,7560,600\n")
        out = td / "sel.json"
        cmd = ["uv", "run", "tools/aggregate_stage1.py", "--run-dir", str(td / "run"),
               "--config-name", "trial-x", "--expect-folds", "2", "--reference", str(ref),
               "--out", str(out)]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        print(proc.stdout)
        assert proc.returncode == 0, proc.stderr
        sel = json.loads(out.read_text())
        assert sel["best"]["params"] == {"lr": "0.1"}, sel["best"]["params"]
        assert abs(sel["best"]["mean_acc"] - 0.75) < 1e-9
        assert abs(sel["best"]["mean_best_acc"] - 0.85) < 1e-9
        assert abs(sel["best"]["selection_bias_pp"] - 10.0) < 1e-6
        assert abs(sel["reference_final_mean"] - 0.77) < 1e-9
        assert abs(sel["reference_best_mean"] - 0.81) < 1e-9
        assert "V3 (PASS)" in proc.stdout, proc.stdout  # 0.75 vs 0.77 -> gap -0.02 >= -0.03
        # missing reference file: still exits 0, reports no reference rows
        proc2 = subprocess.run(cmd[:-4] + ["--reference", str(td / "nope.csv"), "--out", str(out)],
                               cwd=ROOT, capture_output=True, text=True)
        assert proc2.returncode == 0, proc2.stderr
        assert "no reference rows" in proc2.stdout, proc2.stdout
    print("AGGREGATE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
