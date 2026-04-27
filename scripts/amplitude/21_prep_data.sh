#!/bin/bash
# WHERE: Amplitude login-node, inside the cloned repo (after 20_setup.sh).
# WHAT:  Build the canonical SmaTable artifacts — synthetic (small, fast,
#        used by the toy smoke test) AND real data (~9000 events, full
#        LOSO/AOS/80_20 fold matrix). Symlinks the active dataset to the
#        synthetic variant so the first smoke test runs against
#        deterministic data.
# WHY:   Split out from 20_setup.sh because the real-data prep takes
#        minutes (reads the full ~140 MB .npz tree) and you don't want it
#        re-running on every "redo the toolchain" pass.
# COST:  Synthetic ~5 s, real data ~1–3 min on Lustre.
#
# Override the source dir via env:  SMATABLE_INPUT_DIR=/path/to/preprocessed-windows

source "$(dirname "$0")/_lib.sh"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

say "Resolving paths"
# On amplitude: HPC_HOME=/lustre/hpc_home/<user> (fast Lustre, working area)
# and HOME=/homes/<user> (persistent NFS, where rsync uploads landed).
HPC_HOME="${HPC_HOME:-$HOME}"
DATASET_INPUT_DIR="${SMATABLE_INPUT_DIR:-${HOME}/data/smatable-preprocessed}"
echo "    HPC_HOME           = ${HPC_HOME}    (prep output / Lustre)"
echo "    HOME               = ${HOME}        (rsync target)"
echo "    DATASET_INPUT_DIR  = ${DATASET_INPUT_DIR}  (raw .npz tree)"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not on PATH — run 20_setup.sh first."
    exit 1
fi

say "Prep — synthetic (small, fast)"
uv run tools/prep_smatable.py --synthetic \
    --dst "${HPC_HOME}/data/smatable-syn" --schemes LOSO \
    --syn-T 32 --syn-subjects 4 --syn-sessions 3 --syn-per-class 5 2>&1 | tail -10
expect "global: x=(360, 4, 32) … plus 4 LOSO folds with train=270 test=90."

say "Prep — real data (ALL schemes, no cap)"
if [ -d "${DATASET_INPUT_DIR}" ]; then
    uv run tools/prep_smatable.py \
        --src "${DATASET_INPUT_DIR}" \
        --dst "${HPC_HOME}/data/smatable-real" \
        --schemes LOSO,AOS,80_20 2>&1 | tail -25
    expect "global: x=(~9000, 4, ~1000) … with 15 LOSO + 15 AOS + 5 80_20 folds."
else
    echo "    SKIP: ${DATASET_INPUT_DIR} not found (run 11_upload_dataset.sh first, or set SMATABLE_INPUT_DIR=...)"
fi

say "Symlink active dataset to synthetic for the smoke test"
ln -sfn "${HPC_HOME}/data/smatable-syn" "${HPC_HOME}/data/smatable"
ls -la "${HPC_HOME}/data/smatable"
echo "    (later, switch to real data via:  ln -sfn ${HPC_HOME}/data/smatable-real ${HPC_HOME}/data/smatable)"

pass
