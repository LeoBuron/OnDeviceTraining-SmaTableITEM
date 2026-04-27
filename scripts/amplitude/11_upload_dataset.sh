#!/bin/bash
# WHERE: LOCAL Mac.
# WHAT:  rsync ~/Nextcloud/.../preprocessed-windows to Amplitude under
#        $HPC_HOME/data/smatable-preprocessed/.
# WHY:   The prep step on Amplitude needs the .npz event-window tree.
# COST:  ~1.3 GB — can be 1–10 minutes depending on uplink.
#
# Override host / source via env:
#   AMPLITUDE_HOST=login-ampl
#   DATASET_SRC=/Users/leo/Nextcloud/.../preprocessed-windows
#   AMPLITUDE_HPC_HOME=  (defaults to '$HOME' on the remote — script will
#                        resolve it via ssh)

source "$(dirname "$0")/_lib.sh"

TARGET_HOST="${AMPLITUDE_HOST:-gateway.amplitude.uni-due.de}"
DATASET_SRC="${DATASET_SRC:-$HOME/Nextcloud/Dokumente/UDE/Research/Datasets/smatable-data/preprocessed-windows}"

if [ ! -d "${DATASET_SRC}" ]; then
    echo "ERROR: dataset source not found at ${DATASET_SRC}"
    echo "Set DATASET_SRC=/path/to/preprocessed-windows and re-run."
    exit 1
fi

say "Resolving HPC_HOME on ${TARGET_HOST}"
REMOTE_HOME="${AMPLITUDE_HPC_HOME:-$(ssh "${TARGET_HOST}" 'echo "${HPC_HOME:-$HOME}"')}"
TARGET_DIR="${REMOTE_HOME}/data/smatable-preprocessed"
echo "    HPC_HOME on remote = ${REMOTE_HOME}"
echo "    target dir         = ${TARGET_DIR}"

say "Source: ${DATASET_SRC}"
du -sh "${DATASET_SRC}" 2>/dev/null

ssh "${TARGET_HOST}" "mkdir -p '${TARGET_DIR}'"

rsync -av --human-readable --partial \
    "${DATASET_SRC}/" \
    "${TARGET_HOST}:${TARGET_DIR}/"

expect "rsync ends with 'sent X bytes received Y bytes ... total size is ~1.3G'."

pass
