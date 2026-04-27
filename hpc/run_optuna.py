"""Optuna driver — CPU-only, mirrors smatable-offline/src/optuna/gridsearch.py
minus the GPU bits.

Trial body = subprocess invocation of a per-RQ HOST binary (built once,
outside this script). The binary reads its hyperparams + dataset fold from
env vars and prints a single `RESULT ...` line to stdout (see the
per-RQ contract in docs/superpowers/specs/2026-04-27-hpc-experiment-harness-design.md).

Multi-process coordination is via Optuna's JournalFileBackend storage
(file-based, lock-free, identical pattern to the reference repo).

Usage (local smoke test):

    uv run hpc/run_optuna.py \\
        --rq rq0_toy_synthetic \\
        --host-bin build/HOST-Debug/HOST \\
        --data-dir data/smatable \\
        --search-space hpc/search_space/rq0_toy.json \\
        --log-dir runs/optuna-local \\
        --n-workers 2

Usage (Amplitude):  invoked from hpc/run_optuna_amplitude.sh inside the
container with /data, /logs bind-mounted.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import optuna
from optuna.samplers import GridSampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


RESULT_RE = re.compile(r"^RESULT\s+(.*)$", re.MULTILINE)
KV_RE     = re.compile(r"(\w+)=([^\s\"]+|\"[^\"]*\")")


def parse_result_line(stdout: str) -> dict:
    """Pull the LAST `RESULT key=value ...` line from stdout into a dict.
    Values are float/int/str depending on shape."""
    matches = RESULT_RE.findall(stdout)
    if not matches:
        raise ValueError("no RESULT line in stdout")
    out: dict = {}
    for key, val in KV_RE.findall(matches[-1]):
        if val.startswith('"') and val.endswith('"'):
            out[key] = val.strip('"')
            continue
        try:
            out[key] = int(val) if val.isdigit() or (val.startswith('-') and val[1:].isdigit()) else float(val)
        except ValueError:
            out[key] = val
    return out


def env_from_trial(trial: optuna.Trial, search_space: dict) -> dict[str, str]:
    """Map trial.params to ODT_*/SMATABLE_* env vars. Conventions:
    - 'fold' key                       -> SMATABLE_FOLD
    - 'fold_scheme' key                -> SMATABLE_FOLD_SCHEME
    - any other key                    -> ODT_<KEY upper> (lr, epochs, etc.)
    """
    env: dict[str, str] = {}
    for k, choices in search_space.items():
        v = trial.suggest_categorical(k, choices)
        if k == "fold":
            env["SMATABLE_FOLD"] = str(v)
        elif k == "fold_scheme":
            env["SMATABLE_FOLD_SCHEME"] = str(v)
        else:
            env[f"ODT_{k.upper()}"] = str(v)
    return env


def make_objective(*, host_bin: Path, data_dir: Path, fold_scheme: str,
                   search_space: dict, log_dir: Path, timeout_s: float):
    def objective(trial: optuna.Trial) -> float:
        env = os.environ.copy()
        env.update(env_from_trial(trial, search_space))
        env.setdefault("SMATABLE_FOLD_SCHEME", fold_scheme)
        env["SMATABLE_DATA_DIR"] = str(data_dir)

        trial_dir = log_dir / f"trial_{trial.number:05d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "env.json").write_text(json.dumps(
            {k: v for k, v in env.items() if k.startswith(("ODT_", "SMATABLE_"))},
            indent=2,
        ))

        try:
            proc = subprocess.run(
                [str(host_bin)],
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            (trial_dir / "stdout.log").write_text(e.stdout or "")
            (trial_dir / "stderr.log").write_text(e.stderr or "")
            raise optuna.TrialPruned(f"timeout after {timeout_s}s")

        (trial_dir / "stdout.log").write_text(proc.stdout)
        (trial_dir / "stderr.log").write_text(proc.stderr)

        if proc.returncode != 0:
            raise optuna.TrialPruned(
                f"exit {proc.returncode}: {proc.stderr[-200:].strip()}")

        try:
            result = parse_result_line(proc.stdout)
        except ValueError as e:
            raise optuna.TrialPruned(f"unparsed stdout: {e}") from e

        if "skipped" in result:
            raise optuna.TrialPruned(f"binary skipped: {result.get('reason', '')}")

        for k in ("n_params", "best_epoch", "wall_clock_s"):
            if k in result:
                trial.set_user_attr(k, result[k])
        return float(result["accuracy"])

    return objective


def worker(_idx, *, journal_path: Path, study_name: str, host_bin: Path,
           data_dir: Path, fold_scheme: str, search_space: dict, log_dir: Path,
           timeout_s: float):
    """One worker process. Re-opens the shared study (load_if_exists), then
    optimize() drains enqueued trials cooperatively with the other workers
    via the file-journal lock."""
    storage = JournalStorage(JournalFileBackend(file_path=str(journal_path)))
    study = optuna.load_study(study_name=study_name, storage=storage,
                              sampler=GridSampler(search_space=search_space, seed=42))
    obj = make_objective(host_bin=host_bin, data_dir=data_dir,
                         fold_scheme=fold_scheme, search_space=search_space,
                         log_dir=log_dir, timeout_s=timeout_s)
    study.optimize(obj)


def cartesian_count(space: dict) -> int:
    n = 1
    for vs in space.values():
        n *= len(vs)
    return n


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rq", required=True, help="RQ tag (e.g. rq0_toy_synthetic) — used for study name + log dir.")
    p.add_argument("--host-bin", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--search-space", type=Path, required=True)
    p.add_argument("--log-dir", type=Path, required=True)
    p.add_argument("--fold-scheme", default="LOSO",
                   help="default scheme if 'fold_scheme' is not in the search space")
    p.add_argument("--n-workers", type=int, default=int(os.getenv("N_WORKERS", "2")))
    p.add_argument("--timeout-s", type=float, default=float(os.getenv("TRIAL_TIMEOUT_S", "300")))
    args = p.parse_args()

    if not args.host_bin.exists():
        raise SystemExit(f"host binary not found: {args.host_bin}")
    if not args.data_dir.exists():
        raise SystemExit(f"dataset dir not found: {args.data_dir}")

    search_space = json.loads(args.search_space.read_text())
    n_grid = cartesian_count(search_space)
    print(f"search space: {n_grid} grid points across {list(search_space)}")

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    study_name = f"{args.rq}--{timestamp}"
    run_dir = args.log_dir / study_name
    run_dir.mkdir(parents=True, exist_ok=True)
    journal_path = run_dir / "journal.log"
    print(f"study: {study_name}")
    print(f"journal: {journal_path}")

    storage = JournalStorage(JournalFileBackend(file_path=str(journal_path)))
    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name=study_name,
        sampler=GridSampler(search_space=search_space, seed=42),
        load_if_exists=False,
    )

    mp.set_start_method("spawn", force=True)

    t0 = time.time()
    if args.n_workers <= 1:
        # in-process for ease of debugging
        worker(
            0,
            journal_path=journal_path, study_name=study_name,
            host_bin=args.host_bin.resolve(), data_dir=args.data_dir.resolve(),
            fold_scheme=args.fold_scheme,
            search_space=search_space, log_dir=run_dir, timeout_s=args.timeout_s,
        )
    else:
        with mp.Pool(processes=args.n_workers, maxtasksperchild=8) as pool:
            pool.map(
                partial(
                    worker,
                    journal_path=journal_path, study_name=study_name,
                    host_bin=args.host_bin.resolve(), data_dir=args.data_dir.resolve(),
                    fold_scheme=args.fold_scheme,
                    search_space=search_space, log_dir=run_dir,
                    timeout_s=args.timeout_s,
                ),
                range(args.n_workers),
            )
    wall = time.time() - t0
    print(f"all workers joined in {wall:.1f}s")

    # dump CSV of completed trials
    completed = study.get_trials(deepcopy=False)
    csv_path = run_dir / "trials.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial", "state", "value",
            *list(search_space),
            "n_params", "best_epoch", "wall_clock_s",
        ])
        for t in completed:
            row = [t.number, t.state.name, t.value]
            row.extend(t.params.get(k, "") for k in search_space)
            row.extend([t.user_attrs.get("n_params", ""),
                        t.user_attrs.get("best_epoch", ""),
                        t.user_attrs.get("wall_clock_s", "")])
            writer.writerow(row)
    print(f"trials.csv: {csv_path}  ({len(completed)} trials)")
    if study.best_trial is not None:
        bt = study.best_trial
        print(f"best: trial #{bt.number} value={bt.value:.4f} params={bt.params}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
