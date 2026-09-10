"""V0 gate: the rebuilt PyTorch DepthwiseCNN must reproduce Florian's recorded
metrics EXACTLY (accuracy + confusion matrix) for every trial x person.

Locked reconstruction (functionally determined 2026-07-02): GroupNorm(1, C),
padding='same', per-config dilation, bias-free convs, block order
dw->pw->norm->relu->pool(->drop); head aap->flatten->fc16->relu(->drop)->fc6.

Usage:
    uv run tools/verify_reference.py            # all 4 trials x 15 persons
    uv run tools/verify_reference.py --quick    # P001 only per trial
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent / "data" / "model_and_dataset"
TRIALS = ["trial-2353", "trial-3408", "trial-3650", "trial-4223"]


class Block(nn.Module):
    def __init__(self, cin, cout, k, dilation, p_drop=0.0):
        super().__init__()
        self.depthwise = nn.Conv1d(cin, cin, k, padding="same", dilation=dilation,
                                   groups=cin, bias=False)
        self.pointwise = nn.Conv1d(cin, cout, 1, bias=False)
        self.norm = nn.GroupNorm(1, cout)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.drop = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.pool(self.act(self.norm(self.pointwise(self.depthwise(x))))))


class DepthwiseCNN(nn.Module):
    """Reference model. Dropout (after each block's pool and before the last
    Linear) is Identity when p_drop == 0 and inactive in eval mode, so V0
    (eval-only reproduction of Florian's confusion matrices) is unaffected;
    stage1_reference.py --train passes the trial's p_drop for R0."""

    def __init__(self, widths, k, dilation, in_ch=4, n_cls=6, p_drop=0.0):
        super().__init__()
        blocks, prev = [], in_ch
        for w in widths:
            blocks.append(Block(prev, w, k, dilation, p_drop))
            prev = w
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(prev, 16), nn.ReLU(),
                                  nn.Dropout(p_drop) if p_drop > 0 else nn.Identity(),
                                  nn.Linear(16, n_cls))

    def forward(self, x):
        return self.head(self.features(x))


def load_windows(dataset_dir: Path, person: str, label_to_id: dict):
    xs, ys = [], []
    for f in sorted((dataset_dir / f"sub-{person}").rglob("*.npz")):
        xs.append(np.load(f)["x"])
        ys.append(label_to_id[f.parent.name])
    return torch.from_numpy(np.stack(xs)), np.array(ys)


def check(trial: str, person: str) -> bool:
    mdir = ROOT / trial / "models" / f"person-{person}"
    ckpt = torch.load(mdir / "model.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    metrics = json.loads((mdir / "metrics.json").read_text())
    ref_cm = np.array(metrics["best"]["conf_mat"])

    model = DepthwiseCNN(cfg["widths"], cfg["kernel_size"], cfg["conv_dilation"])
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    X, y = load_windows(ROOT / trial / "dataset", person, ckpt["label_to_id"])
    with torch.no_grad():
        pred = model(X).argmax(1).numpy()
    cm = np.zeros((6, 6), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    ok = np.array_equal(cm, ref_cm)
    acc = (pred == y).mean()
    print(f"{trial} {person}: acc={acc:.6f} ref={metrics['best']['val_acc']:.6f} "
          f"conf_mat={'EXACT' if ok else 'MISMATCH'}")
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="P001 only per trial")
    args = p.parse_args()
    persons = ["P001"] if args.quick else [f"P{i:03d}" for i in range(1, 16)]
    failures = 0
    for trial in TRIALS:
        for person in persons:
            if not check(trial, person):
                failures += 1
    print(f"{'PASS' if failures == 0 else 'FAIL'} ({failures} mismatches)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
