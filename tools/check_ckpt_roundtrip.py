"""V4 gate: train briefly with ODT_CKPT_DIR, reload the checkpoint in a fresh
process (ODT_STATE_DICT_DIR + ODT_EVAL_ONLY), and demand that the reloaded
eval accuracy equals the manifest's final_val_acc exactly and the reloaded
test_acc equals the manifest's test_acc (same binary, same data,
deterministic eval -> bitwise-identical accuracies). Under decision D1 the
checkpoint IS the final-epoch model, so no restore step is involved.

Usage: uv run tools/check_ckpt_roundtrip.py --host-bin build/HOST-Debug/HOST --data-dir data/smatable-trial-3650
       uv run tools/check_ckpt_roundtrip.py --host-bin build/HOST-Debug/HOST --data-dir data/smatable-trial-2353 --widths 4 --kernel 7 --dilation 3 --p-drop 0.1
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def run(binary, data_dir, arch, extra):
    env = {"SMATABLE_DATA_DIR": str(data_dir), "SMATABLE_FOLD_SCHEME": "LOSO",
           "SMATABLE_FOLD": "0", **arch, **extra}
    proc = subprocess.run([str(binary)], env={**os.environ, **env},
                          capture_output=True, text=True, timeout=3600, check=True)
    m = re.findall(r"^RESULT\s+(.*)$", proc.stdout, re.MULTILINE)
    assert m, proc.stdout[-2000:]
    return dict(kv.split("=") for kv in m[-1].split())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host-bin", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--widths", default="8,8")
    p.add_argument("--kernel", default="5")
    p.add_argument("--dilation", default="2")
    p.add_argument("--p-drop", default="0")
    p.add_argument("--epochs", default="3")
    args = p.parse_args()
    arch = {"ODT_WIDTHS": args.widths, "ODT_KERNEL_SIZE": args.kernel,
            "ODT_DILATION": args.dilation, "ODT_P_DROP": args.p_drop}

    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "ckpt"
        trained = run(args.host_bin, args.data_dir, arch,
                      {"ODT_LR": "0.1", "ODT_EPOCHS": args.epochs, "ODT_CKPT_DIR": str(ckpt)})
        manifest = json.loads((ckpt / "manifest.json").read_text())
        reloaded = run(args.host_bin, args.data_dir, arch,
                       {"ODT_STATE_DICT_DIR": str(ckpt), "ODT_EVAL_ONLY": "1",
                        "ODT_DUMP_DIR": td})
        checks = [
            ("accuracy", float(reloaded["accuracy"]), float(manifest["final_val_acc"])),
            ("test_acc", float(reloaded["test_acc"]), float(manifest["test_acc"])),
            ("ckpt_val_acc==final_val_acc", float(manifest["ckpt_val_acc"]), float(manifest["final_val_acc"])),
            ("RESULT accuracy==manifest", float(trained["accuracy"]), float(manifest["final_val_acc"])),
        ]
        ok = True
        for name, got, want in checks:
            good = abs(got - want) < 1e-9
            ok &= good
            print(f"{name}: got={got:.6f} want={want:.6f} -> {'PASS' if good else 'FAIL'}")
        print(f"V4 {'PASS' if ok else 'FAIL'} (best_epoch={manifest['best_epoch']} "
              f"best_val_acc={manifest['best_val_acc']} final_epoch={manifest['final_epoch']})")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
