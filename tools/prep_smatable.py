"""Prep step: build the canonical SmaTable artifacts the C harness consumes.

Two modes:

  --src <preprocessed-windows-dir>   read .npz event windows produced by
                                      Florian's denspp.offline pipeline
                                      (default ~/Nextcloud/.../preprocessed-windows).

  --synthetic                         bypass --src, emit fake [4, T] windows whose
                                      class signal is a per-class sinusoid in
                                      channel 0 plus zero-mean noise. Used by the
                                      toy harness and by L0/L1 equivalence tests.

Outputs deterministic artifacts under --dst (default data/smatable/):

    smatable_x.npy           float32 [N, 4, T]
    smatable_y.npy           int32   [N]            class id 0..NC-1
    smatable_meta.json       layout, channel ids, label_to_id, sha256 of x and y
    folds/<scheme>/fold_<k>_train.npy    int32 sample IDs into smatable_x
    folds/<scheme>/fold_<k>_test.npy
    folds/<scheme>/fold_<k>_train.h      uint32_t[] of float bit patterns
                                          + int32_t[] of labels (L0 bit-identity)
    folds/<scheme>/fold_<k>_test.h

Schemes mirror smatable-offline/src/model_training/dataset_splits.py:
    LOSO   leave-one-subject-out, 15 folds, mirrors RQ2/RQ4
    AOS    adapt-one-session, 15 folds (calibration session retained)
    80_20  session-based, 5 folds

Determinism: the prep is a pure function of (input contents, scheme, label_to_id).
SHA-256 of the global x/y arrays is recorded in smatable_meta.json so cross-
machine re-runs can be byte-verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LABEL_TO_ID = {
    "knock": 0,
    "tap": 1,
    "swipe-down": 2,
    "swipe-up": 3,
    "swipe-left": 4,
    "swipe-right": 5,
}

CHANNELS = [1, 4, 6, 8]  # zero-based; matches preprocessed-windows README

# Filename: sub-PXXX_ses-SXXX_GESTURE_event-NN.npz; capture parts loosely.
NAME_RE = re.compile(
    r"^sub-P(?P<sub>\d{3})_ses-S(?P<ses>\d{3})_(?P<gesture>[a-z\-]+)_event-(?P<evt>\d{2})$"
)


@dataclass
class Window:
    x: np.ndarray  # shape [4, T], float32
    subject: int
    session: int
    gesture_id: int
    event: int


# ---------- loading ----------------------------------------------------------


def discover_windows(src: Path) -> list[Window]:
    """Walk preprocessed-windows tree, return one Window per .npz."""
    out: list[Window] = []
    for npz in sorted(src.rglob("*.npz")):
        m = NAME_RE.match(npz.stem)
        if not m:
            print(f"skip (name mismatch): {npz}", file=sys.stderr)
            continue
        gesture = m["gesture"]
        if gesture not in LABEL_TO_ID:
            print(f"skip (unknown gesture {gesture}): {npz}", file=sys.stderr)
            continue
        with np.load(npz) as f:
            x = f["x"].astype(np.float32, copy=False)
        if x.ndim != 2 or x.shape[0] != len(CHANNELS):
            raise ValueError(f"{npz}: expected shape [{len(CHANNELS)}, T], got {x.shape}")
        out.append(
            Window(
                x=x,
                subject=int(m["sub"]),
                session=int(m["ses"]),
                gesture_id=LABEL_TO_ID[gesture],
                event=int(m["evt"]),
            )
        )
    if not out:
        raise SystemExit(f"no .npz windows found under {src}")
    return out


def make_synthetic(n_subjects: int, n_sessions: int, n_per_class: int, T: int,
                   noise_std: float, seed: int) -> list[Window]:
    """Per-class sinusoid in channel 0, noise in others. Linearly separable
    after flatten so a Linear-only toy model can hit > 0.9 accuracy."""
    rng = np.random.default_rng(seed)
    out: list[Window] = []
    for subject in range(1, n_subjects + 1):
        for session in range(1, n_sessions + 1):
            for gesture, gid in LABEL_TO_ID.items():
                freq = 5.0 + 3.0 * gid  # distinct freq per class
                t = np.arange(T, dtype=np.float32) / T
                signal = np.sin(2 * np.pi * freq * t).astype(np.float32)
                for evt in range(1, n_per_class + 1):
                    x = rng.normal(0.0, noise_std, size=(len(CHANNELS), T)).astype(np.float32)
                    x[0] += signal  # plant signal in channel 0
                    out.append(
                        Window(
                            x=x,
                            subject=subject,
                            session=session,
                            gesture_id=gid,
                            event=evt,
                        )
                    )
    return out


# ---------- canonical buffer + meta -----------------------------------------


def build_canonical(windows: list[Window]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Stack into deterministic order (sub, ses, gesture, evt) so the same input
    always produces the same byte layout."""
    windows = sorted(windows, key=lambda w: (w.subject, w.session, w.gesture_id, w.event))
    Ts = {w.x.shape[1] for w in windows}
    if len(Ts) != 1:
        raise SystemExit(f"inconsistent window length T across inputs: {sorted(Ts)}")
    T = Ts.pop()

    x = np.stack([w.x for w in windows], axis=0)  # [N, 4, T]
    y = np.array([w.gesture_id for w in windows], dtype=np.int32)
    meta = np.zeros(
        len(windows),
        dtype=np.dtype(
            [("subject", np.int8), ("session", np.int8),
             ("gesture", np.int8), ("event", np.int16)],
        ),
    )
    for i, w in enumerate(windows):
        meta[i] = (w.subject, w.session, w.gesture_id, w.event)

    return x, y, meta, T


def sha256_of(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


# ---------- fold construction -----------------------------------------------


def fold_indices_loso(meta: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """15 folds — fold k: subject (k+1) is test, rest is train."""
    folds = []
    subjects = sorted(set(int(s) for s in meta["subject"]))
    for k, sub in enumerate(subjects):
        test = np.where(meta["subject"] == sub)[0].astype(np.int32)
        train = np.where(meta["subject"] != sub)[0].astype(np.int32)
        folds.append((train, test))
    return folds


def fold_indices_aos(meta: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Adapt-one-session — train is all-other-subjects + session-1 of held-out
    subject; test is sessions 2..10 of held-out subject. Mirrors
    smatable-offline SmartTableAOSDataset semantics."""
    folds = []
    subjects = sorted(set(int(s) for s in meta["subject"]))
    for sub in subjects:
        is_held = meta["subject"] == sub
        train_mask = (~is_held) | (is_held & (meta["session"] == 1))
        test_mask = is_held & (meta["session"] != 1)
        folds.append((np.where(train_mask)[0].astype(np.int32),
                      np.where(test_mask)[0].astype(np.int32)))
    return folds


def fold_indices_80_20(meta: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """5 folds — fold k: sessions {2k+1, 2k+2} are test, rest is train."""
    folds = []
    sessions = sorted(set(int(s) for s in meta["session"]))
    pairs = [sessions[i:i + 2] for i in range(0, len(sessions), 2)]
    for pair in pairs:
        is_test = np.isin(meta["session"], pair)
        folds.append((np.where(~is_test)[0].astype(np.int32),
                      np.where(is_test)[0].astype(np.int32)))
    return folds


SCHEMES = {
    "LOSO": fold_indices_loso,
    "AOS": fold_indices_aos,
    "80_20": fold_indices_80_20,
}


# ---------- baked-header generation (L0 bit-identity) -----------------------


HEADER_TEMPLATE = """\
/* Generated by tools/prep_smatable.py — DO NOT EDIT.
 * Source: {scheme} fold {k} ({split})
 * Subjects: {subjects_repr}
 * Sessions: {sessions_repr}
 * Sample count (after --max-samples-per-fold cap if any): {n}
 * Layout: x is row-major float32, shape [{n}, {C}, {T}]; y is int32 [{n}].
 * L0 invariant: float bytes are identical to the corresponding slice of
 *   data/smatable/smatable_x.npy as written by the same prep run.
 */
#pragma once
#include <stdint.h>

#define SMATABLE_BAKED_FOLD            {k}
#define SMATABLE_BAKED_SCHEME          "{scheme}"
#define SMATABLE_BAKED_SPLIT           "{split}"
#define SMATABLE_BAKED_N_SAMPLES       {n}
#define SMATABLE_BAKED_N_CHANNELS      {C}
#define SMATABLE_BAKED_WINDOW_SAMPLES  {T}
#define SMATABLE_BAKED_N_CLASSES       {NC}
#define SMATABLE_BAKED_FLOATS_PER_SAMPLE ({C} * {T})

static const uint32_t smatable_baked_x_bits[{n} * {C} * {T}] = {{
{x_lines}
}};

static const int32_t smatable_baked_y[{n}] = {{
{y_lines}
}};
"""


def _format_uint32_block(arr: np.ndarray, per_row: int = 8) -> str:
    """Pack uint32 values as 0xXXXXXXXX, per_row per source line."""
    flat = arr.ravel()
    lines = []
    for i in range(0, flat.size, per_row):
        chunk = flat[i:i + per_row]
        lines.append("    " + ", ".join(f"0x{v:08x}" for v in chunk) + ",")
    return "\n".join(lines)


def _format_int32_block(arr: np.ndarray, per_row: int = 16) -> str:
    flat = arr.ravel()
    lines = []
    for i in range(0, flat.size, per_row):
        chunk = flat[i:i + per_row]
        lines.append("    " + ", ".join(f"{int(v)}" for v in chunk) + ",")
    return "\n".join(lines)


def write_baked_header(path: Path, *, scheme: str, k: int, split: str,
                       x_slice: np.ndarray, y_slice: np.ndarray,
                       meta_slice: np.ndarray) -> None:
    n, C, T = x_slice.shape
    NC = len(LABEL_TO_ID)
    # L0 bytes: read float32 bits as little-endian uint32. ndarray.view enforces
    # this exactly because numpy stores in machine-native byteorder for default
    # dtypes and we wrote the .npy as <f4 — so x_slice.tobytes() is little-endian.
    x_bits = np.frombuffer(x_slice.tobytes(order="C"), dtype="<u4")
    body = HEADER_TEMPLATE.format(
        scheme=scheme,
        k=k,
        split=split,
        subjects_repr=sorted(set(int(s) for s in meta_slice["subject"])),
        sessions_repr=sorted(set(int(s) for s in meta_slice["session"])),
        n=n,
        C=C,
        T=T,
        NC=NC,
        x_lines=_format_uint32_block(x_bits),
        y_lines=_format_int32_block(y_slice),
    )
    path.write_text(body)


# ---------- main ------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path,
                   default=Path.home() / "Nextcloud/Dokumente/UDE/Research/Datasets/smatable-data/preprocessed-windows")
    p.add_argument("--dst", type=Path, default=Path("data/smatable"))
    p.add_argument("--schemes", default="LOSO,AOS,80_20",
                   help=f"comma-separated subset of {','.join(SCHEMES)}")
    p.add_argument("--synthetic", action="store_true",
                   help="bypass --src, generate fake windows for the toy harness / tests")
    p.add_argument("--syn-subjects", type=int, default=4)
    p.add_argument("--syn-sessions", type=int, default=2)
    p.add_argument("--syn-per-class", type=int, default=3)
    p.add_argument("--syn-T", type=int, default=64,
                   help="window length T for synthetic mode (real mode reads T from .npz)")
    p.add_argument("--syn-noise-std", type=float, default=0.05)
    p.add_argument("--syn-seed", type=int, default=42)
    p.add_argument("--max-samples-per-fold", type=int, default=0,
                   help="cap baked-header rows per fold (0 = no cap). Useful to keep .h "
                        "files small for MCU flash and CI artifacts.")
    p.add_argument("--no-baked", action="store_true",
                   help="skip writing .h baked headers (faster, host-only runs)")
    args = p.parse_args()

    schemes = [s for s in args.schemes.split(",") if s]
    for s in schemes:
        if s not in SCHEMES:
            raise SystemExit(f"unknown scheme {s}; available: {','.join(SCHEMES)}")

    if args.synthetic:
        windows = make_synthetic(
            n_subjects=args.syn_subjects,
            n_sessions=args.syn_sessions,
            n_per_class=args.syn_per_class,
            T=args.syn_T,
            noise_std=args.syn_noise_std,
            seed=args.syn_seed,
        )
    else:
        windows = discover_windows(args.src)

    x, y, meta, T = build_canonical(windows)
    NC = len(LABEL_TO_ID)
    C = len(CHANNELS)

    args.dst.mkdir(parents=True, exist_ok=True)
    np.save(args.dst / "smatable_x.npy", x)
    np.save(args.dst / "smatable_y.npy", y)
    np.save(args.dst / "smatable_meta.npy", meta)

    sha_x = sha256_of(x)
    sha_y = sha256_of(y)

    info = {
        "label_to_id": LABEL_TO_ID,
        "channels": CHANNELS,
        "n_events": int(x.shape[0]),
        "n_channels": C,
        "window_samples": int(T),
        "n_classes": NC,
        "synthetic": bool(args.synthetic),
        "sha256_x": sha_x,
        "sha256_y": sha_y,
        "max_samples_per_fold": int(args.max_samples_per_fold),
        "schemes": schemes,
    }
    (args.dst / "smatable_meta.json").write_text(json.dumps(info, indent=2))

    print(f"global: x={x.shape} y={y.shape} sha_x={sha_x[:12]}… sha_y={sha_y[:12]}…")

    folds_root = args.dst / "folds"
    folds_root.mkdir(exist_ok=True)
    for scheme in schemes:
        sdir = folds_root / scheme
        sdir.mkdir(exist_ok=True)
        folds = SCHEMES[scheme](meta)
        for k, (train_ids, test_ids) in enumerate(folds):
            if args.max_samples_per_fold > 0:
                train_ids = train_ids[: args.max_samples_per_fold]
                test_ids = test_ids[: args.max_samples_per_fold]
            np.save(sdir / f"fold_{k:02d}_train.npy", train_ids)
            np.save(sdir / f"fold_{k:02d}_test.npy", test_ids)
            print(f"  {scheme} fold {k:02d}: train={train_ids.size} test={test_ids.size}")
            if args.no_baked:
                continue
            for split, ids in (("train", train_ids), ("test", test_ids)):
                write_baked_header(
                    sdir / f"fold_{k:02d}_{split}.h",
                    scheme=scheme, k=k, split=split,
                    x_slice=x[ids],
                    y_slice=y[ids],
                    meta_slice=meta[ids],
                )

    print(f"done. wrote artifacts under {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
