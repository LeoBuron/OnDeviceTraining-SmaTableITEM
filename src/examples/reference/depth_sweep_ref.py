"""Layer-Depth-Sweep PyTorch-Referenz fuer Plan 2 (USERAPI-Audit Pass 1).

Betriebsarten:
  - Full-Training (default): trainiert N=20 Seeds, misst final accuracy.
    Wird gegen ODT-Sweeps bei DEPTH=0/1/4 verglichen.
  - State-Dump (--state-dump <dir>): fixiert Seed, dumpt pre-weights, post-
    activations, post-gradients, loss fuer den ersten Mini-Batch. Input fuer
    state_dump_compare.py (Task 2+).

Aufruf:
  uv run python src/examples/reference/depth_sweep_ref.py --hidden-layers 1
  uv run python src/examples/reference/depth_sweep_ref.py --hidden-layers 1 --state-dump runs/audit1/dump_pt_d1 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compare import Config, main_cli, write_run_metadata  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import os as _os

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RUNS_DIR = Path(__file__).resolve().parents[3] / "runs" / _os.environ.get("RUNS_SUBDIR", "")
EXAMPLE_NAME = "mlp_mnist_depth_sweep_host"

HIDDEN_DIM = 32
INPUT_DIM = 784
OUTPUT_DIM = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10


def load_mnist():
    x_train = np.load(DATA_DIR / "mnist_train_x.npy").astype(np.float32).reshape(-1, INPUT_DIM)
    y_train = np.load(DATA_DIR / "mnist_train_y.npy")
    x_test = np.load(DATA_DIR / "mnist_test_x.npy").astype(np.float32).reshape(-1, INPUT_DIM)
    y_test = np.load(DATA_DIR / "mnist_test_y.npy")
    if y_train.ndim == 2 and y_train.shape[1] == 10:
        y_train = y_train.argmax(axis=1)
        y_test = y_test.argmax(axis=1)
    return x_train, y_train.astype(np.int64), x_test, y_test.astype(np.int64)


def build_model(hidden_layers: int) -> nn.Sequential:
    """MLP mit uniformer Hidden-Dim=32. hidden_layers = 0/1/4."""
    if hidden_layers == 0:
        modules = [nn.Linear(INPUT_DIM, OUTPUT_DIM)]
    else:
        modules = [nn.Linear(INPUT_DIM, HIDDEN_DIM), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            modules += [nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU()]
        modules += [nn.Linear(HIDDEN_DIM, OUTPUT_DIM)]
    model = nn.Sequential(*modules)
    for m in model:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return model


def _eval(model, x_test_t, y_test_t):
    model.eval()
    with torch.no_grad():
        logits = model(x_test_t)
        probs = F.softmax(logits, dim=1)
        eval_loss = F.nll_loss(torch.log(probs + 1e-12), y_test_t).item()
        preds = logits.argmax(dim=1)
        acc = (preds == y_test_t).float().mean().item() * 100.0
    return eval_loss, acc


def train(model, seed: int, hidden_layers: int, shuffle: bool = True):
    """Full-training — spiegelt die ODT-Laufumgebung (fixed-perm, bs=32, lr=0.001,
    Softmax+CE, Xavier-Init, Bias=0). Schreibt auch eine per-Epoch-CSV damit
    Plot-Vergleiche wie bei mlp_mnist_float32_host moeglich sind.

    shuffle=True: einmalige numpy-Permutation der Trainingsdaten (spiegelt
    ODT DataLoader mit shuffle=true, das genau einmal beim Init shuffelt und
    dann die permutierte Reihenfolge ueber alle Epochen beibehaelt).
    shuffle=False: identische Reihenfolge wie ODT mit ODT_DISABLE_SHUFFLE=1
    (Indizes 0..N-1 in Original-NPY-Reihenfolge ueber alle Epochen). Verwendet
    fuer H3 (DataLoader-Shuffle-Isolation).
    """
    torch.manual_seed(seed)
    x_train, y_train, x_test, y_test = load_mnist()

    # ODT DataLoader shuffelt EINMAL (seed=odtSeed) und iteriert dann in fester
    # Reihenfolge. Die ODT-Seed-Konvention ist seed_index + 1 (rngSetSeed(0) ==
    # rngSetSeed(1)); compare.py injected ODT_SEED=seed+1. PyTorch-Seite nutzt
    # den Python-seed als numpy-Permutation-Seed und kommt auf eine eigene
    # Reihenfolge — das ist OK, weil der Vergleich statistisch (N-σ) ist, nicht
    # bitweise.
    if shuffle:
        rng = np.random.default_rng(seed + 1)
        perm = rng.permutation(len(x_train))
        x_train = x_train[perm]
        y_train = y_train[perm]

    x_test_t = torch.from_numpy(x_test)
    y_test_t = torch.from_numpy(y_test)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    optim = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    # Example-Name bekommt Depth-Suffix (+ optional "_noshuffle"), damit die
    # Sweeps nicht kollidieren.
    name_suffix = "_noshuffle" if not shuffle else ""
    example_name = f"{EXAMPLE_NAME}_d{hidden_layers}{name_suffix}"
    csv_path = RUNS_DIR / f"{example_name}_pytorch_seed{seed:02d}.csv"
    write_run_metadata(csv_path, {
        "framework": "pytorch",
        "framework_version": f"torch-{torch.__version__}",
        "example_name": example_name,
        "seed": seed,
        "architecture": {
            "type": "mlp",
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "hidden_layers": hidden_layers,
            "output_dim": OUTPUT_DIM,
            "dtype": "float32",
            "notes": "PyTorch nn.Sequential; F.softmax applied in loss",
        },
        "training": {
            "optimizer": "SGD",
            "learning_rate": LEARNING_RATE,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "loss": "nll_loss(log(softmax))",
        },
        "init": {"weights": "xavier_uniform_", "bias": "zeros_"},
        "data": {
            "dataset": "MNIST",
            "train_size": int(len(x_train)),
            "test_size": int(len(x_test)),
            "shuffle_seed": (seed + 1) if shuffle else None,
            "shuffle_semantics": ("once-at-init (matches ODT DataLoader)"
                                   if shuffle else "no shuffle (matches ODT_DISABLE_SHUFFLE=1)"),
        },
    })
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "train_loss", "eval_loss", "test_accuracy"])

        for epoch in range(NUM_EPOCHS):
            model.train()
            loss_sum = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                optim.zero_grad()
                probs = F.softmax(model(xb), dim=1)
                loss = F.nll_loss(torch.log(probs + 1e-12), yb)
                loss.backward()
                optim.step()
                loss_sum += loss.item()
                n_batches += 1
            train_loss = loss_sum / max(n_batches, 1)
            eval_loss, acc = _eval(model, x_test_t, y_test_t)
            w.writerow([epoch + 1, f"{train_loss:.6f}", f"{eval_loss:.6f}", f"{acc:.6f}"])
            fh.flush()

    return acc


def state_dump(hidden_layers: int, dump_dir: Path, seed: int,
               load_init_from: Path | None = None) -> None:
    """Single-Batch-Snapshot: pre-weights, pre-ReLU activations, softmax output,
    loss (sum + mean) und gradients (sum + mean reduction).

    Wenn load_init_from gesetzt ist, werden die Pre-Weights aus dem gegebenen
    Verzeichnis geladen (pre_w_k.npy / pre_b_k.npy, typischerweise ein ODT-
    Dump), damit die Initialisierung bitweise mit dem ODT-Run matcht. Das ist
    noetig fuer H4/H2/H6-Vergleiche, bei denen der PyTorch-Init sonst die
    Divergenz dominieren wuerde.
    """
    dump_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_train, y_train, _, _ = load_mnist()
    # Deterministische Batch: erste BATCH_SIZE Indizes — ODT's dataLoaderInit
    # mit shuffle=false liefert genau diese Reihenfolge (Index 0..BATCH_SIZE-1).
    xb = torch.from_numpy(x_train[:BATCH_SIZE])
    yb = torch.from_numpy(y_train[:BATCH_SIZE])

    # Basis-Model bauen + (optional) Pre-Weights aus externer Quelle laden +
    # dann dumpen.
    model = build_model(hidden_layers)
    if load_init_from is not None:
        load_init_from = Path(load_init_from)
        k = 0
        for m in model:
            if isinstance(m, nn.Linear):
                w = np.load(load_init_from / f"pre_w_{k}.npy")
                # Bias wurde als {1, out_features} gedumpt — squeeze auf 1D.
                b = np.load(load_init_from / f"pre_b_{k}.npy").squeeze()
                with torch.no_grad():
                    m.weight.copy_(torch.from_numpy(w))
                    m.bias.copy_(torch.from_numpy(b))
                k += 1
    k = 0
    linear_ref = []
    for m in model:
        if isinstance(m, nn.Linear):
            np.save(dump_dir / f"pre_w_{k}.npy", m.weight.detach().numpy())
            # Bias als [1, out_features] — ODT dumpt in dieser Shape.
            np.save(dump_dir / f"pre_b_{k}.npy", m.bias.detach().numpy()[None, :])
            linear_ref.append((k, m.weight.detach().clone(), m.bias.detach().clone()))
            k += 1

    # Forward mit Pre-ReLU-Activations-Capture.
    acts_x = xb
    k = 0
    for m in model:
        if isinstance(m, nn.Linear):
            acts_x = m(acts_x)
            np.save(dump_dir / f"pre_relu_{k}.npy", acts_x.detach().numpy())
            k += 1
        elif isinstance(m, nn.ReLU):
            acts_x = m(acts_x)
    probs = F.softmax(acts_x, dim=1)
    np.save(dump_dir / "softmax_out.npy", probs.detach().numpy())

    # Loss in beiden Reduktionen dumpen.
    # ODT's CrossEntropy summiert `y_onehot * -log(p)` ueber ALLE Elemente
    # (batch * classes), ohne Division — das ist "sum" ueber den Batch.
    y_onehot = F.one_hot(yb, num_classes=OUTPUT_DIM).float()
    loss_sum = -(y_onehot * torch.log(probs + 1e-7)).sum()
    loss_mean = F.nll_loss(torch.log(probs + 1e-12), yb)
    np.save(dump_dir / "loss_sum.npy",  loss_sum.detach().numpy().reshape(1))
    np.save(dump_dir / "loss_mean.npy", loss_mean.detach().numpy().reshape(1))

    # Backward mit SUM-Reduktion (matcht ODT).
    model.zero_grad()
    loss_sum.backward()
    k = 0
    for m in model:
        if isinstance(m, nn.Linear):
            np.save(dump_dir / f"post_grad_w_sum_{k}.npy",
                    m.weight.grad.detach().numpy())
            np.save(dump_dir / f"post_grad_b_sum_{k}.npy",
                    m.bias.grad.detach().numpy()[None, :])
            k += 1

    # Backward mit MEAN-Reduktion (PyTorch-Default): zweites Modell mit
    # identischen Initial-Weights, gleicher Batch, gleicher Pfad.
    model2 = build_model(hidden_layers)
    # Gewichte bitweise kopieren, damit die Initialisierung exakt matcht.
    j = 0
    lin_iter = iter(linear_ref)
    for m in model2:
        if isinstance(m, nn.Linear):
            _, w_ref, b_ref = next(lin_iter)
            with torch.no_grad():
                m.weight.copy_(w_ref)
                m.bias.copy_(b_ref)
            j += 1
    model2.zero_grad()
    probs2 = F.softmax(model2(xb), dim=1)
    loss_mean2 = F.nll_loss(torch.log(probs2 + 1e-12), yb)
    loss_mean2.backward()
    k = 0
    for m in model2:
        if isinstance(m, nn.Linear):
            np.save(dump_dir / f"post_grad_w_mean_{k}.npy",
                    m.weight.grad.detach().numpy())
            np.save(dump_dir / f"post_grad_b_mean_{k}.npy",
                    m.bias.grad.detach().numpy()[None, :])
            k += 1

    print(f"[state-dump] wrote to {dump_dir} (hidden_layers={hidden_layers}, seed={seed})")


def main_full(hidden_layers: int, n_seeds: int = 20, shuffle: bool = True) -> None:
    name_suffix = "_noshuffle" if not shuffle else ""
    example_name = f"{EXAMPLE_NAME}_d{hidden_layers}{name_suffix}"
    cfg = Config(
        binary_path="build/HOST-Debug/HOST",
        example_name=EXAMPLE_NAME,  # source file name (without depth suffix)
        stdout_metric_regex=r"accuracy=([\d.]+)%",
        higher_is_better=True,
        n_seeds=n_seeds,
        sigma_multiplier=2.0,
        pytorch_build=lambda: build_model(hidden_layers),
        pytorch_train=lambda m, s: train(m, s, hidden_layers, shuffle=shuffle),
        odt_build_preset="HOST-Debug",
        odt_metadata={
            "framework": "odt",
            "example_name": example_name,
            "architecture": {
                "type": "mlp",
                "input_dim": INPUT_DIM,
                "hidden_dim": HIDDEN_DIM,
                "hidden_layers": hidden_layers,
                "output_dim": OUTPUT_DIM,
                "dtype": "float32",
            },
            "training": {
                "optimizer": "SGD",
                "learning_rate": LEARNING_RATE,
                "momentum": 0.0,
                "weight_decay": 0.0,
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "loss": "CrossEntropy",
            },
            "init": {"weights": "XAVIER_UNIFORM", "bias": "ZEROS"},
            "data": {
                "dataset": "MNIST",
                "train_size": 60000,
                "test_size": 10000,
                "shuffle_semantics": "once-at-init (ODT DataLoader)",
            },
        },
    )
    main_cli(cfg)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hidden-layers", type=int, default=1, choices=[0, 1, 4])
    p.add_argument("--state-dump", type=Path, default=None,
                   help="If set, run single-batch state-dump to this directory.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=20,
                   help="Number of seeds for full-training sweep (default 20).")
    p.add_argument("--load-init-from", type=Path, default=None,
                   help="Directory with pre_w_k.npy / pre_b_k.npy to override "
                        "Xavier init (matches an ODT dump bitwise). Only valid "
                        "with --state-dump.")
    p.add_argument("--no-shuffle", action="store_true",
                   help="Disable the numpy one-shot permutation on PyTorch side "
                        "(matches ODT_DISABLE_SHUFFLE=1 on the ODT side). "
                        "Used for H3 (DataLoader-shuffle-semantics isolation).")
    args = p.parse_args()
    if args.state_dump:
        state_dump(args.hidden_layers, args.state_dump, args.seed,
                   load_init_from=args.load_init_from)
    else:
        main_full(args.hidden_layers, args.n_seeds, shuffle=not args.no_shuffle)
