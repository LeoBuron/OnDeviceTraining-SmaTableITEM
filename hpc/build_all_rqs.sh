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
    # HOST-Release (-O2, DEBUG_MODE_ERROR still on -- that define is set
    # unconditionally in host_post.cmake, independent of CMAKE_BUILD_TYPE):
    # these binaries run real Optuna sweeps, and -O0 (HOST-Debug) is slow
    # enough on math-heavy examples (e.g. rq1_replay_buffer's PPCA merge) to
    # risk blowing per-trial timeouts. ODT_MEM_PROFILE=ON: turns on
    # StorageApi's heap high-water counter (a relaxed atomic add per
    # reserveMemory call) so production sweep binaries carry
    # mem_heap_peak_b/mem_reconciliation_gap_b -- overhead is negligible next
    # to training time.
    cmake --preset HOST-Release -DODT_EXAMPLE="${rq}" -DODT_MEM_PROFILE=ON >/dev/null
    cmake --build --preset HOST-Release --target HOST >/dev/null
    cp -f "${ROOT}/build/HOST-Release/HOST" "${BIN_DIR}/${rq}.host"
    echo "    -> ${BIN_DIR}/${rq}.host"
done

echo "done. binaries:"
ls -lh "${BIN_DIR}"
