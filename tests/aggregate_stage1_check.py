"""Fixture-driven check of tools/aggregate_stage1.py under decision D1:
winner by mean FINAL accuracy, V3 against the reference's final_val_acc,
selection-bias column = 100*(mean best - mean final), reference via --reference.

Also covers the final-review hardening: mean_test_acc per combo, the
reference epochs filter (only rows at the sweep's epoch count count toward
V3), the partial-reference WARNING, the pre-D1 no-test_acc-column WARNING,
and the friendly reference-schema check.

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


def write_run(run_dir: Path, rows: list[dict], search_space: dict | None = None) -> None:
    run_dir.mkdir(parents=True)
    space = search_space if search_space is not None else {"fold": [0, 1], "lr": [0.1, 0.01], "epochs": [250]}
    (run_dir / "search_space.json").write_text(json.dumps(space))
    cols = ["trial", "state", "value", "fold", "lr", "epochs", "best_epoch", "best_val_acc", "test_acc"]
    with (run_dir / "trials.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # lr=0.1: final 0.70/0.80 (mean 0.75), best 0.80/0.90 (mean 0.85) -> bias +10 pp
        #         test 0.68/0.79 (mean 0.735)
        # lr=0.01: final 0.60/0.70 (mean 0.65) but best 0.95/0.95 -> must NOT win under D1
        write_run(td / "run", [
            {"trial": 0, "state": "COMPLETE", "value": 0.70, "fold": 0, "lr": 0.1, "epochs": 250, "best_epoch": 5, "best_val_acc": 0.80, "test_acc": 0.68},
            {"trial": 1, "state": "COMPLETE", "value": 0.80, "fold": 1, "lr": 0.1, "epochs": 250, "best_epoch": 7, "best_val_acc": 0.90, "test_acc": 0.79},
            {"trial": 2, "state": "COMPLETE", "value": 0.60, "fold": 0, "lr": 0.01, "epochs": 250, "best_epoch": 2, "best_val_acc": 0.95, "test_acc": 0.58},
            {"trial": 3, "state": "COMPLETE", "value": 0.70, "fold": 1, "lr": 0.01, "epochs": 250, "best_epoch": 3, "best_val_acc": 0.95, "test_acc": 0.69},
            {"trial": 4, "state": "PRUNED", "value": "", "fold": 1, "lr": 0.01, "epochs": "", "best_epoch": "", "best_val_acc": "", "test_acc": ""},
        ])
        ref = td / "reference.csv"
        ref.write_text("config_name,fold_idx,optim,lr,weight_decay,epochs,seed,final_val_acc,best_val_acc,best_epoch,final_test_acc,n_train,n_eval\n"
                       "trial-x,0,adam,0.001,0,250,42,0.76,0.80,90,0.75,7560,600\n"
                       "trial-x,1,adam,0.001,0,250,42,0.78,0.82,80,0.77,7560,600\n"
                       "trial-y,0,adam,0.001,0,250,42,0.10,0.20,1,0.1,7560,600\n"
                       # must be excluded by the epochs filter (sweep epochs=250), not by config_name:
                       "trial-x,0,adam,0.001,0,2,42,0.10,0.20,1,0.1,7560,600\n")
        out = td / "sel.json"
        cmd = ["uv", "run", "tools/aggregate_stage1.py", "--run-dir", str(td / "run"),
               "--config-name", "trial-x", "--expect-folds", "2", "--reference", str(ref),
               "--out", str(out)]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        print(proc.stdout)
        assert proc.returncode == 0, proc.stderr
        assert "excluded 1 reference row" in proc.stderr, proc.stderr
        sel = json.loads(out.read_text())
        assert sel["best"]["params"] == {"epochs": "250", "lr": "0.1"}, sel["best"]["params"]
        assert abs(sel["best"]["mean_acc"] - 0.75) < 1e-9
        assert abs(sel["best"]["mean_best_acc"] - 0.85) < 1e-9
        assert abs(sel["best"]["mean_test_acc"] - 0.735) < 1e-9
        assert abs(sel["best"]["selection_bias_pp"] - 10.0) < 1e-6
        assert abs(sel["reference_final_mean"] - 0.77) < 1e-9
        assert abs(sel["reference_best_mean"] - 0.81) < 1e-9
        assert sel["reference_epochs"] == 250, sel["reference_epochs"]
        assert sel["reference_n_folds"] == 2, sel["reference_n_folds"]
        assert "V3 (PASS)" in proc.stdout, proc.stdout  # 0.75 vs 0.77 -> gap -0.02 >= -0.03
        # missing reference file: still exits 0, reports no reference rows
        proc2 = subprocess.run(cmd[:-4] + ["--reference", str(td / "nope.csv"), "--out", str(out)],
                               cwd=ROOT, capture_output=True, text=True)
        assert proc2.returncode == 0, proc2.stderr
        assert "no reference rows" in proc2.stdout, proc2.stdout

        # Partial-reference WARNING: isolated scenario with 3 actual folds so
        # --expect-folds 3 still leaves the combo COMPLETE (this repo's combo
        # completeness gate is unrelated to reference completeness) while the
        # reference itself has only 2 rows for the matching config/epochs.
        write_run(td / "run3", [
            {"trial": 0, "state": "COMPLETE", "value": 0.70, "fold": 0, "lr": 0.1, "epochs": 250, "best_epoch": 5, "best_val_acc": 0.80, "test_acc": 0.68},
            {"trial": 1, "state": "COMPLETE", "value": 0.72, "fold": 1, "lr": 0.1, "epochs": 250, "best_epoch": 5, "best_val_acc": 0.80, "test_acc": 0.68},
            {"trial": 2, "state": "COMPLETE", "value": 0.74, "fold": 2, "lr": 0.1, "epochs": 250, "best_epoch": 5, "best_val_acc": 0.80, "test_acc": 0.68},
        ], search_space={"fold": [0, 1, 2], "lr": [0.1], "epochs": [250]})
        ref3 = td / "reference3.csv"
        ref3.write_text("config_name,fold_idx,optim,lr,weight_decay,epochs,seed,final_val_acc,best_val_acc,best_epoch,final_test_acc,n_train,n_eval\n"
                        "trial-z,0,adam,0.001,0,250,42,0.70,0.75,90,0.69,7560,600\n"
                        "trial-z,1,adam,0.001,0,250,42,0.72,0.77,80,0.71,7560,600\n")
        cmd3 = ["uv", "run", "tools/aggregate_stage1.py", "--run-dir", str(td / "run3"),
                "--config-name", "trial-z", "--expect-folds", "3", "--reference", str(ref3),
                "--out", str(td / "sel3.json")]
        proc3 = subprocess.run(cmd3, cwd=ROOT, capture_output=True, text=True)
        assert proc3.returncode == 0, proc3.stderr
        assert "partial reference" in proc3.stderr, proc3.stderr

        # Pre-D1 run dir: trials.csv with no test_acc column at all.
        write_run(td / "run_pre_d1", [
            {"trial": 0, "state": "COMPLETE", "value": 0.70, "fold": 0, "lr": 0.1, "epochs": 250, "best_epoch": 5, "best_val_acc": 0.80},
            {"trial": 1, "state": "COMPLETE", "value": 0.80, "fold": 1, "lr": 0.1, "epochs": 250, "best_epoch": 7, "best_val_acc": 0.90},
        ])
        # write_run always adds a test_acc column; strip it back out here so
        # this run dir genuinely has no test_acc column, as a pre-D1 run would.
        rows = list(csv.DictReader((td / "run_pre_d1" / "trials.csv").open()))
        cols_no_test_acc = ["trial", "state", "value", "fold", "lr", "epochs", "best_epoch", "best_val_acc"]
        with (td / "run_pre_d1" / "trials.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols_no_test_acc)
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in cols_no_test_acc})
        cmd_pre_d1 = ["uv", "run", "tools/aggregate_stage1.py", "--run-dir", str(td / "run_pre_d1"),
                      "--config-name", "trial-x", "--expect-folds", "2", "--reference", str(td / "nope.csv"),
                      "--out", str(td / "sel_pre_d1.json")]
        proc_pre_d1 = subprocess.run(cmd_pre_d1, cwd=ROOT, capture_output=True, text=True)
        assert proc_pre_d1.returncode == 0, proc_pre_d1.stderr
        assert "predates D1" in proc_pre_d1.stderr, proc_pre_d1.stderr

        # Old data/model_and_dataset/summary.csv header: friendly exit 2.
        old_ref = td / "old_summary.csv"
        old_ref.write_text("config_name,fold_idx,test_person,best_epoch,best_val_acc\n"
                            "trial-x,0,p1,10,0.5\n")
        cmd_old = ["uv", "run", "tools/aggregate_stage1.py", "--run-dir", str(td / "run"),
                   "--config-name", "trial-x", "--expect-folds", "2", "--reference", str(old_ref),
                   "--out", str(td / "sel_old.json")]
        proc_old = subprocess.run(cmd_old, cwd=ROOT, capture_output=True, text=True)
        assert proc_old.returncode == 2, proc_old.stdout + proc_old.stderr
        assert "not a stage1_r0.py output" in proc_old.stderr, proc_old.stderr
    print("AGGREGATE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
