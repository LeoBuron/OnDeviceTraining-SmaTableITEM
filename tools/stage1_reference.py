"""PyTorch golden reference for stage1_pretrain parity (V1 forward, V2 grads)
and SGD feasibility probing (--train).

The data comes from the PREPPED dir (smatable_x.npy + fold id files) so the
inputs are byte-identical to what the C binary reads.

Norm mode (--norm): the C binary is currently built with
STAGE1_USE_GROUPNORM=0 (see src/examples/stage1_pretrain.c) — a
plumbing-equivalent LayerNorm([C, L_pre_pool]) substitute for the paper's
GroupNorm(1, C), used only until the feat/groupnorm ODT branch lands. Default
here is "groupnorm" (the eventual paper configuration, matching
verify_reference.DepthwiseCNN); pass --norm layernorm to match the CURRENT
C build. Whichever mode is chosen, the exported state-dict / grad file names
are identical (features.<i>.depthwise.weight, features.<i>.pointwise.weight,
features.<i>.norm.weight/.bias, head.2.weight/.bias, head.5.weight/.bias) —
these match the C checkpoint registry (g_params[] in stage1_pretrain.c)
regardless of which norm the "norm" sublayer actually is.

Usage (V1+V2 fixture generation, LayerNorm mode — matches the current C build):
    uv run tools/stage1_reference.py --data-dir data/smatable-trial-4223 --fold 0 \
        --widths 8,12,8 --kernel 7 --dilation 3 --seed 42 --norm layernorm \
        --out runs/parity-4223/ref
Usage (SGD anchor / approach-C fallback):
    uv run tools/stage1_reference.py --data-dir data/smatable-trial-3650 --fold 0 \
        --widths 8,8 --kernel 5 --dilation 2 --train --lr 0.1 --epochs 250
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_reference import DepthwiseCNN  # noqa: E402

N_EVAL = 600  # whole left-out subject; must match the C eval-only dump length


class LNBlock(nn.Module):
    """Same dw->pw->norm->relu->pool block as verify_reference.Block, but with
    norm = nn.LayerNorm([cout, length]) instead of nn.GroupNorm(1, cout) —
    mirrors stage1_pretrain.c's STAGE1_USE_GROUPNORM=0 substitute. `length` is
    this block's PRE-pool sequence length (SAME-padded convs don't change L)."""

    def __init__(self, cin, cout, k, dilation, length):
        super().__init__()
        self.depthwise = nn.Conv1d(cin, cin, k, padding="same", dilation=dilation,
                                   groups=cin, bias=False)
        self.pointwise = nn.Conv1d(cin, cout, 1, bias=False)
        self.norm = nn.LayerNorm([cout, length], eps=1e-5)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        return self.pool(self.act(self.norm(self.pointwise(self.depthwise(x)))))


class LayerNormDepthwiseCNN(nn.Module):
    """LayerNorm-substitute twin of verify_reference.DepthwiseCNN. Same
    attribute names/structure (features Sequential of blocks with
    .depthwise/.pointwise/.norm; head Sequential with Linear at indices 2 and
    5) so the exported state dict loads into the C checkpoint registry
    unchanged — only the norm layer's forward/backward math differs.

    `T` (dataset window length) is REQUIRED: normalizedShape for block i's
    norm is [widths[i], L_i] where L_0 = T and L_i = L_{i-1} // 2 (integer
    division, matching stage1_pretrain.c's `L = L / 2` after each block's
    MaxPool1d(2))."""

    def __init__(self, widths, k, dilation, T, in_ch=4, n_cls=6):
        super().__init__()
        blocks, prev, length = [], in_ch, T
        for w in widths:
            blocks.append(LNBlock(prev, w, k, dilation, length))
            prev = w
            length = length // 2
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(prev, 16), nn.ReLU(), nn.Identity(),
                                  nn.Linear(16, n_cls))

    def forward(self, x):
        return self.head(self.features(x))


def load_fold(data_dir: Path, fold: int, scheme: str = "LOSO"):
    x = np.load(data_dir / "smatable_x.npy")
    y = np.load(data_dir / "smatable_y.npy")
    tr = np.load(data_dir / "folds" / scheme / f"fold_{fold:02d}_train.npy")
    te = np.load(data_dir / "folds" / scheme / f"fold_{fold:02d}_test.npy")
    return (torch.from_numpy(x[tr]), torch.from_numpy(y[tr].astype(np.int64)),
            torch.from_numpy(x[te]), torch.from_numpy(y[te].astype(np.int64)))


def build(args, T=None):
    """T (window length) is required for --norm layernorm; ignored for
    --norm groupnorm (DepthwiseCNN's GroupNorm(1,C) has no length dependence).
    Callers load data (for T) BEFORE calling build() — data loading draws no
    torch RNG, so this ordering doesn't disturb the "seed before model
    construction" contract below."""
    torch.manual_seed(args.seed)
    widths = [int(w) for w in args.widths.split(",")]
    if args.norm == "groupnorm":
        return DepthwiseCNN(widths, args.kernel, args.dilation)
    if T is None:
        raise ValueError("--norm layernorm requires T (window length) from the dataset")
    return LayerNormDepthwiseCNN(widths, args.kernel, args.dilation, T)


def export_fixtures(args):
    xtr, ytr, xte, _ = load_fold(Path(args.data_dir), args.fold)
    T = xtr.shape[-1]
    model = build(args, T)
    model.eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in model.state_dict().items():
        np.save(out / f"{name}.npy", t.numpy().astype(np.float32))

    with torch.no_grad():
        probs = F.softmax(model(xte[:N_EVAL]), dim=1)
    np.save(out / "ref_logits.npy", probs.numpy().astype(np.float32))
    np.save(out / "ref_preds.npy", probs.argmax(1).numpy().astype(np.float32))

    model.zero_grad()
    loss = F.cross_entropy(model(xtr[: args.batch]), ytr[: args.batch], reduction="mean")
    loss.backward()
    for name, p in model.named_parameters():
        np.save(out / f"ref_grad_{name}.npy", p.grad.numpy().astype(np.float32))
    print(f"fixtures -> {out} (loss={loss.item():.6f})")


def train_probe(args):
    xtr, ytr, xte, yte = load_fold(Path(args.data_dir), args.fold)
    T = xtr.shape[-1]
    model = build(args, T)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = 0.0
    for e in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr) - args.batch + 1, args.batch):
            idx = perm[i : i + args.batch]
            opt.zero_grad()
            F.cross_entropy(model(xtr[idx]), ytr[idx]).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            acc = (model(xte).argmax(1) == yte).float().mean().item()
        best = max(best, acc)
        if e % 10 == 0 or e == 1:
            print(f"epoch {e}: val_acc={acc:.4f} best={best:.4f}")
    print(f"RESULT accuracy={best:.6f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--widths", default="8,12,8")
    p.add_argument("--kernel", type=int, default=7)
    p.add_argument("--dilation", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--out", default="runs/parity/ref")
    p.add_argument("--norm", choices=["groupnorm", "layernorm"], default="groupnorm",
                   help="groupnorm = paper config (GroupNorm(1,C), matches "
                        "verify_reference.DepthwiseCNN); layernorm = matches the "
                        "CURRENT C build (STAGE1_USE_GROUPNORM=0)")
    p.add_argument("--train", action="store_true")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=250)
    args = p.parse_args()
    if args.train:
        train_probe(args)
    else:
        export_fixtures(args)


if __name__ == "__main__":
    main()
