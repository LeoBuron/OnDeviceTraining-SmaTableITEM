"""V4 gate: train briefly with ODT_CKPT_DIR, reload the checkpoint in a fresh
process (ODT_STATE_DICT_DIR + ODT_EVAL_ONLY), and demand the reloaded eval
accuracy equals the manifest's restored_val_acc exactly (same binary, same
data, deterministic eval -> bitwise-identical accuracy).

Usage: uv run tools/check_ckpt_roundtrip.py --host-bin build/HOST-Debug/HOST \
           --data-dir data/smatable-trial-3650
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ARCH = {"ODT_WIDTHS": "8,8", "ODT_KERNEL_SIZE": "5", "ODT_DILATION": "2", "ODT_P_DROP": "0"}


def run(binary, data_dir, extra):
    env = {"SMATABLE_DATA_DIR": str(data_dir), "SMATABLE_FOLD_SCHEME": "LOSO",
           "SMATABLE_FOLD": "0", **ARCH, **extra}
    import os
    proc = subprocess.run([str(binary)], env={**os.environ, **env},
                          capture_output=True, text=True, timeout=1800, check=True)
    m = re.findall(r"^RESULT\s+(.*)$", proc.stdout, re.MULTILINE)
    assert m, proc.stdout[-2000:]
    return dict(kv.split("=") for kv in m[-1].split())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host-bin", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "ckpt"
        run(args.host_bin, args.data_dir,
            {"ODT_LR": "0.1", "ODT_EPOCHS": "3", "ODT_CKPT_DIR": str(ckpt)})
        manifest = json.loads((ckpt / "manifest.json").read_text())

        reloaded = run(args.host_bin, args.data_dir,
                       {"ODT_STATE_DICT_DIR": str(ckpt), "ODT_EVAL_ONLY": "1",
                        "ODT_DUMP_DIR": td})
        got = float(reloaded["accuracy"])
        want = float(manifest["restored_val_acc"])
        ok = abs(got - want) < 1e-9
        print(f"reload accuracy={got:.6f} manifest restored_val_acc={want:.6f} "
              f"-> {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
