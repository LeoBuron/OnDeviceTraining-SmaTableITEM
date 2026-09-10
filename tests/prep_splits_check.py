"""Checks the session-wise LOSO split contract of tools/prep_smatable.py
(decision D5, 2026-09-09; spec docs/superpowers/specs/2026-09-10-d1-d5-protocol-change-design.md).

Per fold k (held-out subject = k-th subject in sorted order), with sessions
first = min and last = max:
    train  = other subjects, session <  last      (ascending canonical ids)
    retain = other subjects, session == last      (ascending)
    calib  = held-out subject, session == first   (event-major: sorted by (event, gesture))
    test   = held-out subject, session >  first   (ascending)
The four sets partition the dataset.

Run: uv run tests/prep_splits_check.py --dst data/smatable
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SPLITS = ("train", "retain", "calib", "test")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dst", type=Path, required=True)
    args = p.parse_args()

    info = json.loads((args.dst / "smatable_meta.json").read_text())
    assert info.get("split_version") == 2, f"split_version {info.get('split_version')!r} != 2 — old prep"
    meta = np.load(args.dst / "smatable_meta.npy")
    subjects = sorted(set(int(s) for s in meta["subject"]))
    sessions = sorted(set(int(s) for s in meta["session"]))
    first, last = sessions[0], sessions[-1]
    loso = args.dst / "folds" / "LOSO"

    sizes_by_split: dict[str, list[int]] = {s: [] for s in SPLITS}
    for k, sub in enumerate(subjects):
        ids = {s: np.load(loso / f"fold_{k:02d}_{s}.npy") for s in SPLITS}
        for s, arr in ids.items():
            assert arr.dtype == np.int32, f"fold {k} {s}: dtype {arr.dtype}"
            assert arr.size > 0, f"fold {k} {s}: split is empty"
            sizes_by_split[s].append(arr.size)
        cat = np.concatenate(list(ids.values()))
        assert len(set(cat.tolist())) == cat.size, f"fold {k}: splits overlap"
        assert cat.size == meta.size, f"fold {k}: splits cover {cat.size} != {meta.size} windows"
        m = {s: meta[ids[s]] for s in SPLITS}
        assert (m["train"]["subject"] != sub).all() and (m["train"]["session"] < last).all(), f"fold {k}: train"
        assert (m["retain"]["subject"] != sub).all() and (m["retain"]["session"] == last).all(), f"fold {k}: retain"
        assert (m["calib"]["subject"] == sub).all() and (m["calib"]["session"] == first).all(), f"fold {k}: calib"
        assert (m["test"]["subject"] == sub).all() and (m["test"]["session"] > first).all(), f"fold {k}: test"
        order = list(zip(m["calib"]["event"].tolist(), m["calib"]["gesture"].tolist()))
        assert order == sorted(order), f"fold {k}: calib is not event-major"
        for s in ("train", "retain", "test"):
            assert (np.diff(ids[s]) > 0).all(), f"fold {k}: {s} ids not strictly ascending"

    def fmt(s: str) -> str:
        lo, hi = min(sizes_by_split[s]), max(sizes_by_split[s])
        return f"{s}={lo}" if lo == hi else f"{s}={lo}..{hi}"

    sizes = " ".join(fmt(s) for s in SPLITS)
    print(f"PREP-SPLITS PASS — {len(subjects)} LOSO folds, sessions {first}..{last}; {sizes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
