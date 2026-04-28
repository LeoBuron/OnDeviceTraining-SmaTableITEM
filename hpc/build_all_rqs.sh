#!/bin/bash
# Login-node pre-job step: build per-RQ HOST binaries once, copy them into
# hpc/bin/<rq>.host. Slurm jobs then bind-mount these binaries into the
# container — no recompile per Optuna trial.
#
# Usage:  hpc/build_all_rqs.sh                   # build all known RQs
#         hpc/build_all_rqs.sh rq0_toy_synthetic # build a single RQ
#
# Idempotent. Run once after a clean checkout, then again whenever you
# change a per-RQ .c file or the dataset backend.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || dirname "${SCRIPT_DIR}")"
cd "${ROOT}"
BIN_DIR="${ROOT}/hpc/bin"
mkdir -p "${BIN_DIR}"

RQS=("$@")
if [ ${#RQS[@]} -eq 0 ]; then
    RQS=(rq0_toy_synthetic rq1_replay_buffer rq2_unseen_user rq3_new_gesture rq4_stability rq5_noise_augmentation stage1_pretrain)
fi

# pico-sdk + ODT must already be fetched. Run PREPARE if not.
if [ ! -d "${ROOT}/OnDeviceTraining/src" ] || [ ! -d "${ROOT}/pico-sdk/src" ]; then
    echo "PREPARE step (one-time fetch)..."
    cmake --preset PREPARE >/dev/null
fi

for rq in "${RQS[@]}"; do
    echo "==> building ${rq}"
    cmake --preset HOST-Debug -DODT_EXAMPLE="${rq}" >/dev/null
    cmake --build --preset HOST-Debug --target HOST >/dev/null
    cp -f "${ROOT}/build/HOST-Debug/HOST" "${BIN_DIR}/${rq}.host"
    echo "    -> ${BIN_DIR}/${rq}.host"
done

echo "done. binaries:"
ls -lh "${BIN_DIR}"
