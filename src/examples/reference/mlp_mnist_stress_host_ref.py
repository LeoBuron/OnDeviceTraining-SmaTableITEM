"""Vergleicht mlp_mnist_stress_host gegen PyTorch-Referenz (5-hidden-layer MLP).

Schreibt pro Seed eine CSV unter runs/mlp_mnist_stress_host_pytorch_seed{NN}.csv
mit Spalten epoch, train_loss, eval_loss, test_accuracy — spiegelt das Format
der ODT-Seite für Plot-Vergleiche.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compare import Config, main_cli, write_run_metadata

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import os as _os

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RUNS_DIR = Path(__file__).resolve().parents[3] / "runs" / _os.environ.get("RUNS_SUBDIR", "")
EXAMPLE_NAME = "mlp_mnist_stress_host"
NUM_EPOCHS = 5


def load_mnist():
    x_train = np.load(DATA_DIR / "mnist_train_x.npy").astype(np.float32).reshape(-1, 784)
    y_train = np.load(DATA_DIR / "mnist_train_y.npy").astype(np.int64)
    x_test  = np.load(DATA_DIR / "mnist_test_x.npy").astype(np.float32).reshape(-1, 784)
    y_test  = np.load(DATA_DIR / "mnist_test_y.npy").astype(np.int64)
    if y_train.ndim == 2:
        y_train = y_train.argmax(axis=1)
        y_test  = y_test.argmax(axis=1)
    return x_train, y_train, x_test, y_test


def build_model():
    model = nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 64),  nn.ReLU(),
        nn.Linear(64, 32),   nn.ReLU(),
        nn.Linear(32, 10),
    )
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


def train(model, seed):
    torch.manual_seed(seed)
    x_train, y_train, x_test, y_test = load_mnist()
    x_test_t = torch.from_numpy(x_test)
    y_test_t = torch.from_numpy(y_test)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    optim = torch.optim.SGD(model.parameters(), lr=0.001)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RUNS_DIR / f"{EXAMPLE_NAME}_pytorch_seed{seed:02d}.csv"
    write_run_metadata(csv_path, {
        "framework": "pytorch",
        "framework_version": f"torch-{torch.__version__}",
        "example_name": EXAMPLE_NAME,
        "seed": seed,
        "architecture": {
            "type": "mlp",
            "layers": [
                "Linear(784,256)", "ReLU",
                "Linear(256,128)", "ReLU",
                "Linear(128,64)", "ReLU",
                "Linear(64,32)", "ReLU",
                "Linear(32,10)",
            ],
            "dtype": "float32",
            "notes": "PyTorch nn.Sequential; no explicit Softmax — F.softmax applied in loss",
        },
        "training": {
            "optimizer": "SGD",
            "learning_rate": 0.001,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "batch_size": 32,
            "num_epochs": NUM_EPOCHS,
            "loss": "nll_loss(log(softmax))",
        },
        "init": {"weights": "xavier_uniform_", "bias": "zeros_"},
        "data": {
            "dataset": "MNIST",
            "train_size": int(len(x_train)),
            "test_size": int(len(x_test)),
            "shuffle_seed": seed,
            "shuffle_semantics": "per-epoch (PyTorch DataLoader default) — differs from ODT once-at-init",
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


if __name__ == "__main__":
    cfg = Config(
        binary_path="build/HOST-Debug/HOST",
        example_name=EXAMPLE_NAME,
        stdout_metric_regex=r"accuracy=([\d.]+)%",
        higher_is_better=True,
        n_seeds=5,
        sigma_multiplier=2.0,
        pytorch_build=build_model,
        pytorch_train=train,
        odt_metadata={
            "framework": "odt",
            "example_name": EXAMPLE_NAME,
            "seed": 42,
            "architecture": {
                "type": "mlp",
                "layers": [
                    "Linear(784,256)", "ReLU",
                    "Linear(256,128)", "ReLU",
                    "Linear(128,64)", "ReLU",
                    "Linear(64,32)", "ReLU",
                    "Linear(32,10)", "Softmax",
                ],
                "dtype": "float32",
                "notes": "ODT has explicit Softmax layer; loss = CrossEntropy over softmax probs",
            },
            "training": {
                "optimizer": "SGD",
                "learning_rate": 0.001,
                "momentum": 0.0,
                "weight_decay": 0.0,
                "batch_size": 32,
                "num_epochs": NUM_EPOCHS,
                "loss": "CrossEntropy",
            },
            "init": {"weights": "XAVIER_UNIFORM", "bias": "ZEROS"},
            "data": {
                "dataset": "MNIST",
                "train_size": 60000,
                "test_size": 10000,
                "shuffle_seed": 42,
                "shuffle_semantics": "once-at-init (ODT DataLoader)",
            },
        },
    )
    main_cli(cfg)
