"""Compare C parity dumps against PyTorch reference fixtures.

V1: preds exact + probs allclose. V2: every grad_<name> allclose.
Usage: uv run tools/compare_stage1_parity.py --ref runs/parity-4223/ref \
           --c-dir runs/parity-4223/c [--grads]
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def close(name, a, b, rtol, atol):
    ok = np.allclose(a, b, rtol=rtol, atol=atol)
    print(f"{name}: shape={a.shape} max|diff|={np.abs(a - b).max():.3e} "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ref", type=Path, required=True)
    p.add_argument("--c-dir", type=Path, required=True)
    p.add_argument("--grads", action="store_true")
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--atol", type=float, default=1e-6)
    args = p.parse_args()

    ok = True
    if args.grads:
        for ref_f in sorted(args.ref.glob("ref_grad_*.npy")):
            name = ref_f.stem.removeprefix("ref_")
            c = np.load(args.c_dir / f"{name}.npy")
            ok &= close(name, np.load(ref_f).reshape(-1), c.reshape(-1), args.rtol, args.atol)
    else:
        ref_pred = np.load(args.ref / "ref_preds.npy")
        c_pred = np.load(args.c_dir / "preds.npy")
        exact = np.array_equal(ref_pred, c_pred)
        print(f"preds: {int((ref_pred == c_pred).sum())}/{len(ref_pred)} equal "
              f"{'OK' if exact else 'FAIL'}")
        ok &= exact
        ok &= close("probs", np.load(args.ref / "ref_logits.npy"),
                    np.load(args.c_dir / "logits.npy"), args.rtol, args.atol)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
