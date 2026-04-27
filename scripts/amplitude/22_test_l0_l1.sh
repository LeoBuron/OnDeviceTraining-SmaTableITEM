#!/bin/bash
# WHERE: Amplitude login-node, inside the repo.
# WHAT:  L0 (Python byte-equivalence) + L1 (C backend equivalence) + build
#        the rq0 HOST binary + run it once standalone.
# WHY:   Proves the whole HOST-side path on Amplitude before involving Slurm
#        or Apptainer. If L1 fails here, the sbatch run will fail too.
# COST:  ~30 s.

source "$(dirname "$0")/_lib.sh"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

HPC_HOME="${HPC_HOME:-$HOME}"
DATA_DIR="${HPC_HOME}/data/smatable"

if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: ${DATA_DIR} missing — run 20_setup.sh first."
    exit 1
fi

say "L0 — header bytes ≡ .npy slice"
uv run tests/dataset_l0_bytes.py --dst "${DATA_DIR}" 2>&1 | tail -10
expect "'L0 PASS — N fold headers verified, x.sha=…'."

say "Build per-RQ HOST binary (rq0_toy_synthetic)"
hpc/build_all_rqs.sh rq0_toy_synthetic 2>&1 | tail -8
expect "'-> hpc/bin/rq0_toy_synthetic.host' written."
ls -la hpc/bin/rq0_toy_synthetic.host

say "Build the L1 equivalence test"
# Point the baked-backend include search at the active dataset's parent dir
# (the prep tree on Lustre, NOT the empty repo-local data/).
cmake --preset HOST-Debug \
    -DODT_EXAMPLE=rq0_toy_synthetic \
    -DODT_L1_TEST=ON \
    -DSMATABLE_DATA_INCLUDE_DIR="$(dirname "${DATA_DIR}")" 2>&1 | tail -3
cmake --build --preset HOST-Debug --target dataset_l1_equiv 2>&1 | tail -5

say "L1 — NPY backend ≡ baked backend (LOSO fold 0)"
SMATABLE_DATA_DIR="${DATA_DIR}" SMATABLE_FOLD_SCHEME=LOSO SMATABLE_FOLD=0 \
    build/HOST-Debug/tests/dataset_l1_equiv 2>&1 | tail -5
expect "'L1 PASS — HOST(npy) and MCU(baked) backends bytewise equal'."

say "Standalone rq0 run (LOSO fold 0, 3 epochs)"
SMATABLE_DATA_DIR="${DATA_DIR}" SMATABLE_FOLD_SCHEME=LOSO SMATABLE_FOLD=0 \
    ODT_EPOCHS=3 hpc/bin/rq0_toy_synthetic.host 2>&1 | tail -10
expect "BEGIN/EPOCH/RESULT lines, accuracy ≥ 0.9 on synthetic data."

pass
