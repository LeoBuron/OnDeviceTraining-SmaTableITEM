"""Optuna driver — CPU-only, mirrors smatable-offline/src/optuna/gridsearch.py
minus the GPU bits.

Trial body = subprocess invocation of a per-RQ HOST binary (built once,
outside this script). The binary reads its hyperparams + dataset fold from
env vars and prints a single `RESULT ...` line to stdout (see the
per-RQ contract in docs/superpowers/specs/2026-04-27-hpc-experiment-harness-design.md).

Multi-process coordination is via Optuna's RDBStorage. For SQLite URLs we
flip the underlying file to WAL mode before any Optuna operation runs, which
gives multi-reader / single-writer concurrency suitable for N_WORKERS=64.
Place the .db on local tmpfs (/tmp on the compute node) — Lustre POSIX
advisory locks saturate the cluster lock manager and were the actual cause
of the "lock taking >10s" warnings on the previous JournalFileBackend
implementation (sbatch 685747).

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
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import optuna
from optuna.samplers import GridSampler
from optuna.storages import RDBStorage


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
        env["ODT_CKPT_DIR"] = str(trial_dir / "ckpt")
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

        for k, v in result.items():
            if k != "accuracy":
                trial.set_user_attr(k, v)
        return float(result["accuracy"])

    return objective


SQLITE_PREFIX = "sqlite:///"


def _sqlite_path(storage_url: str) -> Path | None:
    """Extract the on-disk path from a SQLAlchemy SQLite URL, or None."""
    if not storage_url.startswith(SQLITE_PREFIX):
        return None
    return Path(storage_url[len(SQLITE_PREFIX):])


def _prep_sqlite_wal(db_path: Path) -> None:
    """One-shot: ensure the SQLite file exists and is in WAL mode with a
    relaxed durability fsync setting. WAL is persisted in the DB header, so
    every later Optuna connection inherits it automatically."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")


def make_storage(storage_url: str) -> RDBStorage:
    """Build the RDBStorage. Long busy_timeout (60 s) replaces the noisy
    JournalFileBackend "lock taking >10s" warnings: SQLite blocks silently
    until the timeout, then raises — that's our actual saturation signal."""
    return RDBStorage(
        url=storage_url,
        engine_kwargs={"connect_args": {"timeout": 60.0}},
    )


def worker(_idx, *, storage_url: str, study_name: str, host_bin: Path,
           data_dir: Path, fold_scheme: str, search_space: dict, log_dir: Path,
           timeout_s: float):
    """One worker process. Re-opens the shared study (load_if_exists), then
    optimize() drains enqueued trials cooperatively with the other workers
    via the SQLite WAL writer queue."""
    storage = make_storage(storage_url)
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
    p.add_argument(
        "--storage-url",
        default=None,
        help="SQLAlchemy URL for Optuna RDBStorage. Default: sqlite:////<run_dir>/study.db. "
             "On Amplitude, run_optuna_amplitude.sh overrides this with a /tmp tmpfs path "
             "and copies the resulting .db back to LOG_DIR after study.optimize returns.",
    )
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

    storage_url = args.storage_url or f"sqlite:///{run_dir / 'study.db'}"
    sqlite_path = _sqlite_path(storage_url)
    if sqlite_path is not None:
        _prep_sqlite_wal(sqlite_path)
    print(f"study: {study_name}")
    print(f"storage: {storage_url}")

    storage = make_storage(storage_url)
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
            storage_url=storage_url, study_name=study_name,
            host_bin=args.host_bin.resolve(), data_dir=args.data_dir.resolve(),
            fold_scheme=args.fold_scheme,
            search_space=search_space, log_dir=run_dir, timeout_s=args.timeout_s,
        )
    else:
        with mp.Pool(processes=args.n_workers, maxtasksperchild=8) as pool:
            pool.map(
                partial(
                    worker,
                    storage_url=storage_url, study_name=study_name,
                    host_bin=args.host_bin.resolve(), data_dir=args.data_dir.resolve(),
                    fold_scheme=args.fold_scheme,
                    search_space=search_space, log_dir=run_dir,
                    timeout_s=args.timeout_s,
                ),
                range(args.n_workers),
            )
    wall = time.time() - t0
    print(f"all workers joined in {wall:.1f}s")

    # Fold the WAL back into the main DB before any rsync — otherwise the
    # post-job copy could leave a study.db whose recent commits live only in
    # study.db-wal and would be lost without the matching -wal/-shm files.
    if sqlite_path is not None and sqlite_path.exists():
        with sqlite3.connect(str(sqlite_path)) as con:
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    # dump CSV of completed trials
    completed = study.get_trials(deepcopy=False)
    csv_path = run_dir / "trials.csv"
    attr_keys = sorted({k for t in completed for k in t.user_attrs})
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "state", "value", *list(search_space), *attr_keys])
        for t in completed:
            row = [t.number, t.state.name, t.value]
            row.extend(t.params.get(k, "") for k in search_space)
            row.extend(t.user_attrs.get(k, "") for k in attr_keys)
            writer.writerow(row)
    print(f"trials.csv: {csv_path}  ({len(completed)} trials)")
    try:
        bt = study.best_trial
        print(f"best: trial #{bt.number} value={bt.value:.4f} params={bt.params}")
    except ValueError:
        # optuna raises (rather than returning None) when zero trials are
        # COMPLETE — e.g. every trial got PRUNED.
        print("no completed trials — best_trial unavailable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
