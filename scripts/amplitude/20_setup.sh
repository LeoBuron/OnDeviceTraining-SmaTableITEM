#!/bin/bash
# WHERE: Amplitude login-node, inside the rsync'd repo (after 10_upload_repo.sh).
# WHAT:  uv sync, cmake PREPARE (fetches pico-sdk + OnDeviceTraining), and
#        prep_smatable.py both real-data + synthetic. Symlinks the synthetic
#        output as the default dataset for the first smoke test.
# WHY:   Bootstraps everything the rest of the test sequence depends on.
# COST:  PREPARE ~75 s (network), uv sync ~30 s, prep ~10–30 s (synthetic) /
#        ~1–2 min (real data depending on disk speed).

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

say "uv install (if needed) + uv sync"
# amplitude login nodes use a corporate proxy with a whitelist that does NOT
# include astral.sh but DOES include github.com — so we bypass the upstream
# installer and pull uv straight from its GitHub release tarball. PATH is
# extended in-process; persist by adding the export to ~/.bashrc if you want.
if ! command -v uv >/dev/null 2>&1; then
    UV_REL_URL="${UV_REL_URL:-https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz}"
    say "Downloading uv via proxy from ${UV_REL_URL}"
    mkdir -p "$HOME/.local/bin"
    tmp="$(mktemp -d)"
    curl -fsSL "${UV_REL_URL}" -o "${tmp}/uv.tar.gz"
    tar -xzf "${tmp}/uv.tar.gz" -C "${tmp}"
    cp "${tmp}"/uv-x86_64-unknown-linux-gnu/uv  "$HOME/.local/bin/uv"
    cp "${tmp}"/uv-x86_64-unknown-linux-gnu/uvx "$HOME/.local/bin/uvx"
    chmod +x "$HOME/.local/bin/uv" "$HOME/.local/bin/uvx"
    rm -rf "${tmp}"
fi
export PATH="$HOME/.local/bin:${PATH}"
uv --version
uv sync 2>&1 | tail -10

say "cmake --preset PREPARE (one-time fetch of pico-sdk + ODT)"
if [ -d OnDeviceTraining/src ] && [ -d pico-sdk/src ]; then
    echo "    pico-sdk + ODT already fetched — skipping"
else
    cmake --preset PREPARE 2>&1 | tail -5
fi
expect "Configuring done with two 'Klone nach …' lines (or 'already fetched')."

say "Prep — synthetic (small, fast — for smoke test)"
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
