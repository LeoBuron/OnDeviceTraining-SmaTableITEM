#!/bin/bash
# WHERE: LOCAL Mac (not Amplitude).
# WHAT:  rsync this repo to Amplitude under ~/projects/OnDeviceTraining-SmaTableITEM.
# WHY:   Pushes only source — excludes build artifacts, vendored deps,
#        pre-built binaries, ALL of data/ (datasets go via 11_upload_dataset.sh
#        or the explicit rsync in docs/runbook-stage1-amplitude.md), and .jj
#        internals.
# COST:  ~10–60 s depending on uplink.
#
# Override target host via env. Default is gateway.amplitude.uni-due.de —
# the documented dedicated data-transfer host for amplitUDE. It typically
# shares Lustre + $HOME with the cluster, and unlike the interactive
# `Host amplitude` alias (which uses RequestTTY+RemoteCommand to chain
# through a jump host) it has no RemoteCommand directive, so rsync works
# without any flag overrides.

source "$(dirname "$0")/_lib.sh"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TARGET_HOST="${AMPLITUDE_HOST:-gateway.amplitude.uni-due.de}"
TARGET_DIR="${AMPLITUDE_REPO_DIR:-projects/OnDeviceTraining-SmaTableITEM}"

say "Source: ${REPO_ROOT}"
say "Target: ${TARGET_HOST}:${TARGET_DIR}"

# rsync only creates the leaf dir; ensure parents exist (same pattern 11 uses).
ssh "${TARGET_HOST}" "mkdir -p '${TARGET_DIR}'"

rsync -av --human-readable \
    --exclude=build \
    --exclude=hpc/bin \
    --exclude=hpc/run_container.sif \
    --exclude=data \
    --exclude=paper/build \
    --exclude=pico-sdk \
    --exclude=OnDeviceTraining \
    --exclude=runs \
    --exclude=.jj \
    --exclude=.git \
    --exclude=__pycache__ \
    --exclude=.uv-cache \
    --exclude=.venv \
    --exclude='scripts/amplitude/logs' \
    "${REPO_ROOT}/" \
    "${TARGET_HOST}:${TARGET_DIR}/"

expect "rsync ends with 'sent ... received ... bytes' and no error."

pass
