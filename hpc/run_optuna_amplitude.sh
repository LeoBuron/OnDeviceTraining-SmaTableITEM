#!/bin/bash
# sbatch script for one Optuna campaign on Amplitude (UDE).
#
# Pattern adapted from /Users/leo/work/smatable-offline/run_gridsearch_amplitude.sh:
#   - CPU partition (no --gres=gpu).
#   - ws_allocate / ws_find for Lustre scratch (the workspace persists 30d).
#   - Dataset rsync'd to scratch on first run, reused on subsequent runs.
#   - Apptainer container bind-mounts /data, /logs, /search_space.json.
#   - Pre-built per-RQ HOST binary in hpc/bin/<rq>.host (built by build_all_rqs.sh).
#
# Submit:  sbatch hpc/run_optuna_amplitude.sh
# Override:  sbatch --export=ALL,RQ=rq1_replay_buffer hpc/run_optuna_amplitude.sh
#
#SBATCH --job-name=smatable_optuna
#SBATCH --partition=STD-s-96h
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Partition naming convention on amplitUDE is s/m/l = small/medium/large
# *node-count*, NOT walltime. STD-l-12h has MinNodes=93 (≥93 nodes per job),
# STD-m-48h has MinNodes=46, STD-s-96h is the default for single-node jobs.
# We submit one node, so STD-s-96h. (Walltime cap there is 96 h, way more
# than our --time=12:00:00 needs.)

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

echo "JobID: ${SLURM_JOB_ID}"
echo "Node:  ${SLURM_NODELIST}"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"

# ---- configuration -------------------------------------------------------

RQ="${RQ:-rq0_toy_synthetic}"
# Decouple binary name from study tag: a single dataset-agnostic binary like
# rq0_toy_synthetic.host can run against multiple search-spaces / datasets
# (synthetic vs real LOSO). Override HOST_BIN_NAME to point at a different
# binary while RQ still picks the search-space + log-dir tag.
HOST_BIN_NAME="${HOST_BIN_NAME:-${RQ}}"
N_WORKERS="${N_WORKERS:-32}"
TRIAL_TIMEOUT_S="${TRIAL_TIMEOUT_S:-300}"
FOLD_SCHEME="${FOLD_SCHEME:-LOSO}"

IMAGE="${SLURM_SUBMIT_DIR}/hpc/run_container.sif"
HOST_BIN_REL="hpc/bin/${HOST_BIN_NAME}.host"
HOST_BIN="${SLURM_SUBMIT_DIR}/${HOST_BIN_REL}"
SEARCH_SPACE_REL="hpc/search_space/${RQ}.json"
SEARCH_SPACE="${SLURM_SUBMIT_DIR}/${SEARCH_SPACE_REL}"
# Default dataset is the canonical synthetic toy; override DATASET_DIR for
# real LOSO runs (e.g. DATASET_DIR=${HPC_HOME}/data/smatable-real).
DATASET_SRC="${DATASET_DIR:-${HPC_HOME}/data/smatable}"
FINAL_LOG_DIR="${HPC_HOME}/experiments"

WS_NAME="smatable-ws"
WS_DAYS=30

# Container-mode is opt-in. Default is "no container" — the compute node runs
# the same uv venv that the login node already populated on Lustre. Container
# build on amplitude is broken until the proxy whitelist + URL-format issues
# are sorted; until then this path is the smoke-test workflow.
USE_CONTAINER="${USE_CONTAINER:-0}"

if [ "${USE_CONTAINER}" = "1" ] && [ ! -f "${IMAGE}" ]; then
    echo "USE_CONTAINER=1 but missing image: ${IMAGE}"; exit 1
fi
if [ ! -f "${HOST_BIN}" ];     then echo "missing HOST binary: ${HOST_BIN} (run hpc/build_all_rqs.sh)"; exit 1; fi
if [ ! -f "${SEARCH_SPACE}" ]; then echo "missing search space: ${SEARCH_SPACE}"; exit 1; fi

# ---- workspace -----------------------------------------------------------

if ws_find "${WS_NAME}" >/dev/null 2>&1; then
    WS_PATH="$(ws_find "${WS_NAME}" | tail -n 1)"
    echo "Workspace exists: ${WS_NAME} -> ${WS_PATH}"
else
    echo "Allocating workspace: ws_allocate ${WS_NAME} ${WS_DAYS}"
    ws_allocate "${WS_NAME}" "${WS_DAYS}"
    WS_PATH="$(ws_find "${WS_NAME}" | tail -n 1)"
fi

JOB_SCRATCH="${WS_PATH}/jobs/${SLURM_JOB_ID}"
# One cache dir PER DATASET (keyed by the source dir's basename). A single
# shared cache silently reused the first campaign's dataset for every later
# submission with a different DATASET_DIR (stage 1 runs four trial datasets).
DATA_CACHE="${WS_PATH}/datasets/$(basename "${DATASET_SRC}")"
LOG_DIR="${JOB_SCRATCH}/logs"
mkdir -p "${JOB_SCRATCH}" "${LOG_DIR}" "${WS_PATH}/datasets"

# ---- dataset -------------------------------------------------------------

if [ ! -d "${DATA_CACHE}" ] || [ -z "$(ls -A "${DATA_CACHE}" 2>/dev/null || true)" ]; then
    echo "Dataset not on workspace -> rsync from ${DATASET_SRC}"
    mkdir -p "${DATA_CACHE}"
    rsync -a "${DATASET_SRC}/" "${DATA_CACHE}/"
else
    echo "Dataset already on workspace -> reusing ${DATA_CACHE}"
fi

# A pre-2026-09-10 cache has no calib split (session-wise D5 re-prep added it);
# reusing one silently makes every trial exit 1 and prune, "succeeding" with zero complete trials.
[ -f "${DATA_CACHE}/folds/LOSO/fold_00_calib.npy" ] || { echo "stale dataset cache: ${DATA_CACHE} has no calib split (pre-2026-09-10 prep) — delete it and resubmit"; exit 1; }

# ---- storage backend (SQLite on local tmpfs) ---------------------------
# Track-A fix: replaced the JournalFileBackend on Lustre — its POSIX advisory
# locks were going through the cluster lock manager (~100 ms+/acquire) and
# triggered "lock taking >10s" warnings even at N_WORKERS=4. SQLite WAL on
# /tmp tmpfs uses kernel-local locks (sub-µs) and the writer queue serializes
# at MMU speed instead of network speed.
DB_HOST_DIR="/tmp/optuna-${SLURM_JOB_ID}"
mkdir -p "${DB_HOST_DIR}"
echo "Storage: SQLite WAL on ${DB_HOST_DIR}/study.db (tmpfs, copied to LOG_DIR after run)"

# ---- run ---------------------------------------------------------------

echo "RQ=${RQ} N_WORKERS=${N_WORKERS} TIMEOUT=${TRIAL_TIMEOUT_S}s FOLD_SCHEME=${FOLD_SCHEME} USE_CONTAINER=${USE_CONTAINER}"

if [ "${USE_CONTAINER}" = "1" ]; then
    # The container bakes /app/pyproject.toml + /app/.venv but NOT the driver
    # script itself — keeping it bind-mounted means edits to run_optuna.py
    # don't require a rebuild + scp each iteration. cwd inside the
    # container is /app (set by the runscript), so we mount the script flat
    # into /app/run_optuna.py and reference it relatively.
    #
    # /optuna_db is the explicit bind for the SQLite DB. --writable-tmpfs
    # would shadow the container's /tmp with an ephemeral overlay, so we use
    # a dedicated mountpoint that survives the bind without being clobbered.
    srun --ntasks=1 apptainer run \
        --writable-tmpfs \
        --bind "${DATA_CACHE}:/data" \
        --bind "${LOG_DIR}:/logs" \
        --bind "${HOST_BIN}:/host_bin:ro" \
        --bind "${SEARCH_SPACE}:/search_space.json:ro" \
        --bind "${SLURM_SUBMIT_DIR}/hpc/run_optuna.py:/app/run_optuna.py:ro" \
        --bind "${DB_HOST_DIR}:/optuna_db" \
        --env N_WORKERS="${N_WORKERS}" \
        --env TRIAL_TIMEOUT_S="${TRIAL_TIMEOUT_S}" \
        "${IMAGE}" \
        run_optuna.py \
            --rq "${RQ}" \
            --host-bin /host_bin \
            --data-dir /data \
            --search-space /search_space.json \
            --log-dir /logs \
            --fold-scheme "${FOLD_SCHEME}" \
            --n-workers "${N_WORKERS}" \
            --timeout-s "${TRIAL_TIMEOUT_S}" \
            --storage-url "sqlite:////optuna_db/study.db"
else
    # No-container path: the compute node mounts the same Lustre as the login
    # node, so it sees the cloned repo, the populated .venv, and the dataset.
    # Modules + uv must be on PATH on the compute node — we re-do them here
    # since SLURM strips the user shell environment by default.
    srun --ntasks=1 bash -c "
        set -euo pipefail
        if command -v module >/dev/null 2>&1; then
            module load cmake/3.29.6 gcc/13.3.0 2>/dev/null || true
        fi
        export PATH=\"\$HOME/.local/bin:\$PATH\"
        cd '${SLURM_SUBMIT_DIR}'
        uv run hpc/run_optuna.py \
            --rq '${RQ}' \
            --host-bin '${HOST_BIN}' \
            --data-dir '${DATA_CACHE}' \
            --search-space '${SEARCH_SPACE}' \
            --log-dir '${LOG_DIR}' \
            --fold-scheme '${FOLD_SCHEME}' \
            --n-workers '${N_WORKERS}' \
            --timeout-s '${TRIAL_TIMEOUT_S}' \
            --storage-url 'sqlite:///${DB_HOST_DIR}/study.db'
    "
fi

# ---- collect ------------------------------------------------------------

# Copy the SQLite DB off tmpfs before the node tears down /tmp. The script
# checkpoints WAL into the main file before exit, so a plain cp is safe.
if [ -f "${DB_HOST_DIR}/study.db" ]; then
    cp -- "${DB_HOST_DIR}/study.db" "${LOG_DIR}/study.db"
    echo "study.db -> ${LOG_DIR}/study.db ($(stat -c%s "${DB_HOST_DIR}/study.db" 2>/dev/null || echo "?") bytes)"
fi
rm -rf "${DB_HOST_DIR}" || true

cp -- "${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "${LOG_DIR}/" || true
cp -- "${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "${LOG_DIR}/" || true
cp -- "${SEARCH_SPACE}"                       "${LOG_DIR}/" || true

mkdir -p "${FINAL_LOG_DIR}"
rsync -a "${LOG_DIR}/" "${FINAL_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}/"
echo "Logs copied to ${FINAL_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}/"
