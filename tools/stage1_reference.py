"""PyTorch golden reference for stage1_pretrain parity (V1 forward, V2 grads)
and Adam/SGD training (--train; see tools/stage1_r0.py for the R0 driver).

The data comes from the PREPPED dir (smatable_x.npy + fold id files) so the
inputs are byte-identical to what the C binary reads.

Norm mode (--norm): default here is "groupnorm" (GroupNorm(1, C), matching
verify_reference.DepthwiseCNN and the current C build); pass --norm layernorm
to use the LayerNorm([C, L_pre_pool]) substitute kept for historical parity
fixtures predating the GroupNorm(1,C) ODT branch. Whichever mode is chosen,
the exported state-dict / grad file names are identical (features.<i>.depthwise.weight,
features.<i>.pointwise.weight, features.<i>.norm.weight/.bias, head.2.weight/.bias,
head.5.weight/.bias) — these match the C checkpoint registry (g_params[] in
stage1_pretrain.c) regardless of which norm the "norm" sublayer actually is.

Usage (V1+V2 fixture generation, matches the current C build):
    uv run tools/stage1_reference.py --data-dir data/smatable-trial-4223 --fold 0 \
        --widths 8,12,8 --kernel 7 --dilation 3 --seed 42 \
        --out runs/parity-4223/ref
Usage (Adam training, R0 path):
    uv run tools/stage1_reference.py --data-dir data/smatable-trial-3650 --fold 0 \
        --widths 8,8 --kernel 5 --dilation 2 --train --optim adam --lr 0.001 --epochs 250
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


class LNBlock(nn.Module):
    """Same dw->pw->norm->relu->pool block as verify_reference.Block, but with
    norm = nn.LayerNorm([cout, length]) instead of nn.GroupNorm(1, cout) —
    kept for historical LayerNorm-mode parity fixtures. `length` is this
    block's PRE-pool sequence length (SAME-padded convs don't change L)."""

    def __init__(self, cin, cout, k, dilation, length, p_drop=0.0):
        super().__init__()
        self.depthwise = nn.Conv1d(cin, cin, k, padding="same", dilation=dilation,
                                   groups=cin, bias=False)
        self.pointwise = nn.Conv1d(cin, cout, 1, bias=False)
        self.norm = nn.LayerNorm([cout, length], eps=1e-5)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.drop = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.pool(self.act(self.norm(self.pointwise(self.depthwise(x))))))


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

    def __init__(self, widths, k, dilation, T, in_ch=4, n_cls=6, p_drop=0.0):
        super().__init__()
        blocks, prev, length = [], in_ch, T
        for w in widths:
            blocks.append(LNBlock(prev, w, k, dilation, length, p_drop))
            prev = w
            length = length // 2
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(prev, 16), nn.ReLU(),
                                  nn.Dropout(p_drop) if p_drop > 0 else nn.Identity(),
                                  nn.Linear(16, n_cls))

    def forward(self, x):
        return self.head(self.features(x))


def load_fold(data_dir: Path, fold: int, scheme: str = "LOSO"):
    """Session-wise LOSO (D5): train = fold_KK_train.npy; eval = calib rows
    followed by test rows — the same order stage1_pretrain.c materializes
    its eval set in, so V1 dumps line up element-wise. Returns
    (xtr, ytr, xev, yev, n_calib)."""
    x = np.load(data_dir / "smatable_x.npy")
    y = np.load(data_dir / "smatable_y.npy")
    fd = data_dir / "folds" / scheme
    calib_path = fd / f"fold_{fold:02d}_calib.npy"
    if not calib_path.exists():
        raise SystemExit(f"{calib_path} missing — re-prep with tools/prep_smatable.py "
                         "(session-wise LOSO, split_version 2)")
    tr = np.load(fd / f"fold_{fold:02d}_train.npy")
    ca = np.load(calib_path)
    te = np.load(fd / f"fold_{fold:02d}_test.npy")
    ev = np.concatenate([ca, te])
    return (torch.from_numpy(x[tr]), torch.from_numpy(y[tr].astype(np.int64)),
            torch.from_numpy(x[ev]), torch.from_numpy(y[ev].astype(np.int64)), int(ca.size))


def build(args, T=None):
    """T (window length) is required for --norm layernorm; ignored for
    --norm groupnorm (DepthwiseCNN's GroupNorm(1,C) has no length dependence).
    Callers load data (for T) BEFORE calling build() — data loading draws no
    torch RNG, so this ordering doesn't disturb the "seed before model
    construction" contract below."""
    torch.manual_seed(args.seed)
    widths = [int(w) for w in args.widths.split(",")]
    if args.norm == "groupnorm":
        return DepthwiseCNN(widths, args.kernel, args.dilation, p_drop=args.p_drop)
    if T is None:
        raise ValueError("--norm layernorm requires T (window length) from the dataset")
    return LayerNormDepthwiseCNN(widths, args.kernel, args.dilation, T, p_drop=args.p_drop)


def export_fixtures(args):
    xtr, ytr, xte, _, _ = load_fold(Path(args.data_dir), args.fold)
    T = xtr.shape[-1]
    model = build(args, T)
    model.eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in model.state_dict().items():
        np.save(out / f"{name}.npy", t.numpy().astype(np.float32))

    with torch.no_grad():
        probs = F.softmax(model(xte), dim=1)
    np.save(out / "ref_logits.npy", probs.numpy().astype(np.float32))
    np.save(out / "ref_preds.npy", probs.argmax(1).numpy().astype(np.float32))

    model.zero_grad()
    loss = F.cross_entropy(model(xtr[: args.batch]), ytr[: args.batch], reduction="mean")
    loss.backward()
    for name, p in model.named_parameters():
        np.save(out / f"ref_grad_{name}.npy", p.grad.numpy().astype(np.float32))
    print(f"fixtures -> {out} (loss={loss.item():.6f})")


def train_run(args) -> dict:
    """One training run; returns the metrics dict the R0 driver appends to
    its CSV. Eval accuracy is measured on calib+test every epoch (headline
    = final epoch, D1); best epoch is kept as the selection-bias
    diagnostic; final_test_acc is the test-only slice of the final epoch."""
    xtr, ytr, xev, yev, n_calib = load_fold(Path(args.data_dir), args.fold)
    T = xtr.shape[-1]
    model = build(args, T).to(args.device)
    xtr, ytr, xev, yev = (t.to(args.device) for t in (xtr, ytr, xev, yev))
    if args.optim == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    gen = torch.Generator().manual_seed(args.seed)
    best, best_epoch, final, final_test = 0.0, 0, 0.0, 0.0
    for e in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr), generator=gen).to(args.device)
        for i in range(0, len(xtr) - args.batch + 1, args.batch):
            idx = perm[i : i + args.batch]
            opt.zero_grad()
            F.cross_entropy(model(xtr[idx]), ytr[idx]).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            correct = model(xev).argmax(1) == yev
        acc = correct.float().mean().item()
        test_acc = correct[n_calib:].float().mean().item()
        final, final_test = acc, test_acc
        if acc > best:
            best, best_epoch = acc, e
        if e % 10 == 0 or e == 1:
            print(f"epoch {e}: val_acc={acc:.4f} test_acc={test_acc:.4f} best={best:.4f}@{best_epoch}")
    return {"final_val_acc": final, "best_val_acc": best, "best_epoch": best_epoch,
            "final_test_acc": final_test, "n_train": int(len(xtr)), "n_eval": int(len(xev))}


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
                        "verify_reference.DepthwiseCNN); layernorm = historical "
                        "parity-fixture mode (LayerNorm substitute)")
    p.add_argument("--train", action="store_true")
    p.add_argument("--optim", choices=["sgdm", "adam"], default="sgdm")
    p.add_argument("--p-drop", type=float, default=0.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=250)
    args = p.parse_args()
    if args.train:
        r = train_run(args)
        print(f"RESULT final_val_acc={r['final_val_acc']:.6f} best_val_acc={r['best_val_acc']:.6f} "
              f"best_epoch={r['best_epoch']} final_test_acc={r['final_test_acc']:.6f}")
    else:
        export_fixtures(args)


if __name__ == "__main__":
    main()
