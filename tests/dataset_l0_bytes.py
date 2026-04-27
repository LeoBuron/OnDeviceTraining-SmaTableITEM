"""L0 equivalence test (Python).

For every fold .h written by prep_smatable.py, parse the baked uint32_t bit
patterns and assert they reconstruct the exact same float32 bytes as the
corresponding rows of smatable_x.npy. Same for int32 labels.

Catches: prep-script bugs, header generator off-by-one, accidental float-to-text
rounding, endianness regressions.

Run:
    uv run tests/dataset_l0_bytes.py --dst data/smatable
    uv run tests/dataset_l0_bytes.py --dst /tmp/smatable-fixtures
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


HEX_RE = re.compile(r"0x([0-9a-fA-F]{8})")


def parse_uint32_array(text: str, var: str) -> np.ndarray:
    """Pulls the contents of `static const uint32_t <var>[...] = { ... };`."""
    m = re.search(rf"const\s+uint32_t\s+{re.escape(var)}\s*\[[^\]]+\]\s*=\s*\{{(.*?)\}};",
                  text, flags=re.DOTALL)
    if not m:
        raise SystemExit(f"could not find uint32_t {var}[] in header")
    nums = [int(h, 16) for h in HEX_RE.findall(m.group(1))]
    return np.array(nums, dtype="<u4")


def parse_int32_array(text: str, var: str) -> np.ndarray:
    """Pulls the contents of `static const int32_t <var>[...] = { ... };`."""
    m = re.search(rf"const\s+int32_t\s+{re.escape(var)}\s*\[[^\]]+\]\s*=\s*\{{(.*?)\}};",
                  text, flags=re.DOTALL)
    if not m:
        raise SystemExit(f"could not find int32_t {var}[] in header")
    nums = [int(s) for s in re.findall(r"-?\d+", m.group(1))]
    return np.array(nums, dtype=np.int32)


def parse_define_int(text: str, name: str) -> int:
    m = re.search(rf"#define\s+{re.escape(name)}\s+(\d+)", text)
    if not m:
        raise SystemExit(f"missing #define {name}")
    return int(m.group(1))


def check_fold(header_path: Path, ids_path: Path, x_global: np.ndarray, y_global: np.ndarray) -> None:
    text = header_path.read_text()
    n = parse_define_int(text, "SMATABLE_BAKED_N_SAMPLES")
    C = parse_define_int(text, "SMATABLE_BAKED_N_CHANNELS")
    T = parse_define_int(text, "SMATABLE_BAKED_WINDOW_SAMPLES")

    ids = np.load(ids_path)
    capped_ids = ids[:n]  # max-samples-per-fold cap applied to baked, not to .npy

    x_bits = parse_uint32_array(text, "smatable_baked_x_bits")
    if x_bits.size != n * C * T:
        raise SystemExit(f"{header_path}: x_bits size {x_bits.size} != {n}*{C}*{T}")
    baked_x = np.frombuffer(x_bits.tobytes(), dtype="<f4").reshape(n, C, T)

    y = parse_int32_array(text, "smatable_baked_y")
    if y.size != n:
        raise SystemExit(f"{header_path}: y size {y.size} != {n}")

    # The thing the L0 test actually proves: bytes match, exactly.
    expected_x = x_global[capped_ids].astype("<f4", copy=False)
    if expected_x.tobytes() != baked_x.tobytes():
        raise SystemExit(f"{header_path}: x bytes diverge from .npy slice")
    if not np.array_equal(y, y_global[capped_ids].astype(np.int32)):
        raise SystemExit(f"{header_path}: y diverges from .npy slice")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dst", type=Path, required=True)
    args = p.parse_args()

    meta = json.loads((args.dst / "smatable_meta.json").read_text())
    x = np.load(args.dst / "smatable_x.npy")
    y = np.load(args.dst / "smatable_y.npy")
    folds_root = args.dst / "folds"

    n_checked = 0
    for scheme_dir in sorted(folds_root.iterdir()):
        if not scheme_dir.is_dir():
            continue
        for header in sorted(scheme_dir.glob("fold_*_*.h")):
            ids = scheme_dir / (header.stem + ".npy")
            if not ids.exists():
                raise SystemExit(f"missing ids for {header}")
            check_fold(header, ids, x, y)
            n_checked += 1
            print(f"L0 OK: {header.relative_to(args.dst)}")

    print(f"L0 PASS — {n_checked} fold headers verified, x.sha={meta['sha256_x'][:12]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
