"""
Vergleichs-Harness zwischen ODT-Binary und einer PyTorch-Referenz.

Usage (aus einem per-example Skript):

    from compare import compare_with_pytorch, Config

    def build_pytorch_model():
        import torch.nn as nn
        return nn.Sequential(nn.Linear(784, 20), nn.ReLU(), nn.Linear(20, 10))

    def train_pytorch(model, seed): ...  # returns final metric (float)

    cfg = Config(
        binary_path="build/HOST-Debug/HOST",
        example_name="mlp_mnist_float32_host",
        stdout_metric_regex=r"accuracy=([\d.]+)%",
        higher_is_better=True,
        n_seeds=5,
        sigma_multiplier=2.0,
        pytorch_build=build_pytorch_model,
        pytorch_train=train_pytorch,
    )
    compare_with_pytorch(cfg)

Exit code 0 bei Pass (ODT innerhalb N σ von PyTorch-Mean), 1 bei Fail.
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import math
import os
import pathlib
import re
import statistics
import subprocess
import sys
from typing import Callable

_RUNS_BASE = pathlib.Path(__file__).resolve().parents[3] / "runs"
# Optional subdirectory to keep new comparison runs isolated from older baselines.
# Per-example wrappers should import RUNS_DIR (lazily resolved) instead of caching.
def _runs_dir() -> pathlib.Path:
    sub = os.environ.get("RUNS_SUBDIR", "").strip()
    return (_RUNS_BASE / sub) if sub else _RUNS_BASE

RUNS_DIR = _runs_dir()


def write_run_metadata(csv_path, meta):
    """Schreibt ein JSON-Sidecar neben die CSV (.csv → .json).

    meta muss ein Dict mit den Feldern framework, example_name, architecture,
    training, init, data (und optional seed) sein — gespiegelt zum ODT-Format.
    timestamp_utc wird automatisch ergänzt.
    """
    csv_path = pathlib.Path(csv_path)
    json_path = csv_path.with_suffix(".json")
    meta = dict(meta)
    meta.setdefault(
        "timestamp_utc",
        datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)


@dataclasses.dataclass
class Config:
    binary_path: str
    example_name: str
    stdout_metric_regex: str            # capture group 1 = float metric
    higher_is_better: bool
    n_seeds: int
    sigma_multiplier: float
    pytorch_build: Callable[[], "torch.nn.Module"]  # noqa: F821
    pytorch_train: Callable[["torch.nn.Module", int], float]  # noqa: F821
    odt_build_preset: str = "HOST-Debug"
    odt_metadata: dict | None = None    # JSON sidecar next to {example}_odt.csv


def run_odt_seeds(cfg: Config) -> list[float]:
    """Baut das ODT-Example einmal, dann N Läufe mit ODT_SEED=0..N-1.

    Schreibt pro Seed runs/{example}_odt_seed{NN}.csv + .json (JSON-Sidecar
    spiegelt PyTorch-Seite — gleiche Struktur, framework='odt', seed wird
    pro Lauf injected). Liefert eine Liste der End-Metriken.
    """
    runs_dir = _runs_dir()
    runs_dir.mkdir(parents=True, exist_ok=True)

    env_build = os.environ.copy()
    cmake_cfg = subprocess.run(
        ["cmake", "--preset", cfg.odt_build_preset,
         f"-DODT_EXAMPLE={cfg.example_name}"],
        capture_output=True, text=True, env=env_build,
    )
    if cmake_cfg.returncode != 0:
        raise RuntimeError(f"cmake configure failed:\n{cmake_cfg.stderr}")
    cmake_build = subprocess.run(
        ["cmake", "--build", "--preset", cfg.odt_build_preset],
        capture_output=True, text=True, env=env_build,
    )
    if cmake_build.returncode != 0:
        raise RuntimeError(f"cmake build failed:\n{cmake_build.stderr}")

    results: list[float] = []
    for seed in range(cfg.n_seeds):
        csv_path = runs_dir / f"{cfg.example_name}_odt_seed{seed:02d}.csv"
        # ODT's rngSetSeed maps 0→state=1, which collides with seed=1. Offset
        # by +1 so seeds 0..N-1 drive N distinct xorshift32 states. The `seed`
        # field in sidecar metadata keeps the bookkeeping index; `shuffle_seed`
        # records the actual ODT state used.
        odt_rng_seed = seed + 1
        env = os.environ.copy()
        env["ODT_CSV_PATH"] = str(csv_path)
        env["ODT_SEED"] = str(odt_rng_seed)
        run = subprocess.run(
            [cfg.binary_path], capture_output=True, text=True, env=env,
        )
        if run.returncode != 0:
            raise RuntimeError(
                f"ODT binary failed (seed={seed}):\n{run.stdout}\n{run.stderr}"
            )
        match = re.search(cfg.stdout_metric_regex, run.stdout)
        if not match:
            raise RuntimeError(
                f"Konnte Metrik nicht parsen (seed={seed}). "
                f"Regex: {cfg.stdout_metric_regex}\nStdout:\n{run.stdout}"
            )
        metric = float(match.group(1))
        results.append(metric)
        if cfg.odt_metadata is not None:
            meta = dict(cfg.odt_metadata)
            meta["seed"] = seed
            # ODT drives shuffle + weight init from one global RNG stream
            # (dataLoaderInit seeds it, shuffle consumes state, init draws
            # from the advanced state). Record the actual ODT_SEED passed.
            if isinstance(meta.get("data"), dict):
                meta["data"] = dict(meta["data"])
                meta["data"]["shuffle_seed"] = odt_rng_seed
            write_run_metadata(csv_path, meta)
        print(f"  [odt seed={seed}] metric={metric:.4f}")
    return results


def run_pytorch_seeds(cfg: Config) -> list[float]:
    import torch
    results = []
    for seed in range(cfg.n_seeds):
        torch.manual_seed(seed)
        model = cfg.pytorch_build()
        metric = cfg.pytorch_train(model, seed)
        results.append(metric)
        print(f"  [pytorch seed={seed}] metric={metric:.4f}")
    return results


def compare_with_pytorch(cfg: Config) -> int:
    print(f"=== compare: {cfg.example_name} ===")
    print(f"Running ODT reference (N={cfg.n_seeds} seeds)...")
    odt_metrics = run_odt_seeds(cfg)
    odt_mean = statistics.fmean(odt_metrics)
    odt_std = statistics.pstdev(odt_metrics) if len(odt_metrics) > 1 else 0.0
    print(f"ODT: mean={odt_mean:.4f} std={odt_std:.4f} "
          f"(raw: {[f'{m:.4f}' for m in odt_metrics]})")

    print(f"Running PyTorch reference (N={cfg.n_seeds} seeds)...")
    pt_metrics = run_pytorch_seeds(cfg)
    pt_mean = statistics.fmean(pt_metrics)
    pt_std = statistics.pstdev(pt_metrics) if len(pt_metrics) > 1 else 0.0
    print(f"PyTorch: mean={pt_mean:.4f} std={pt_std:.4f} "
          f"(raw: {[f'{m:.4f}' for m in pt_metrics]})")

    diff = odt_mean - pt_mean
    # 2σ-Regel gegen PyTorch-std als Baseline (ODT-Streuung wird informativ
    # gelistet, geht aber nicht in die Schwelle ein — bewusst konservativ
    # gegenüber der Referenz). Fallback 5% relativ bei pt_std≈0.
    if pt_std < 1e-9:
        threshold = 0.05 * abs(pt_mean) if abs(pt_mean) > 1e-9 else 1e-3
        rule = "fallback 5% relative"
    else:
        threshold = cfg.sigma_multiplier * pt_std
        rule = f"{cfg.sigma_multiplier}σ of pytorch"

    passed = abs(diff) <= threshold
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] odt_mean - pt_mean = {diff:+.4f} threshold=±{threshold:.4f} ({rule})")
    return 0 if passed else 1


def main_cli(cfg: Config):
    """Entry point für per-example scripts: sys.exit mit compare-Returncode."""
    sys.exit(compare_with_pytorch(cfg))
