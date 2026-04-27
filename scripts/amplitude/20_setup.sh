#!/bin/bash
# WHERE: Amplitude login-node, inside the cloned repo.
# WHAT:  Toolchain only — uv install, ninja install, module load (cmake +
#        gcc), uv sync, cmake PREPARE (fetches pico-sdk + OnDeviceTraining).
# WHY:   Bootstraps every binary subsequent scripts depend on. Idempotent —
#        re-runnable any time without side effects on already-installed
#        artifacts. Dataset prep is split into 21_prep_data.sh because it
#        takes minutes on the real .npz tree.
# COST:  uv install/sync ~10–60 s, ninja ~5 s, cmake PREPARE ~75 s (network).
#        Subsequent re-runs: ~5 s.

source "$(dirname "$0")/_lib.sh"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

say "Resolving paths"
HPC_HOME="${HPC_HOME:-$HOME}"
echo "    HPC_HOME = ${HPC_HOME}"
echo "    HOME     = ${HOME}"

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

say "ninja install (if needed) — same proxy-via-github trick as uv"
if ! command -v ninja >/dev/null 2>&1; then
    NINJA_REL_URL="${NINJA_REL_URL:-https://github.com/ninja-build/ninja/releases/latest/download/ninja-linux.zip}"
    say "Downloading ninja via proxy from ${NINJA_REL_URL}"
    tmp="$(mktemp -d)"
    curl -fsSL "${NINJA_REL_URL}" -o "${tmp}/ninja.zip"
    if command -v unzip >/dev/null 2>&1; then
        unzip -q "${tmp}/ninja.zip" -d "${tmp}"
    else
        # Fallback if unzip is missing — Python is here via uv venv.
        ( cd "${tmp}" && python3 -c "import zipfile; zipfile.ZipFile('ninja.zip').extractall()" )
    fi
    cp "${tmp}/ninja" "$HOME/.local/bin/ninja"
    chmod +x "$HOME/.local/bin/ninja"
    rm -rf "${tmp}"
fi
ninja --version

say "Toolchain check (HPC modules + ninja)"
which gcc g++ cmake ninja 2>&1 || true

say "cmake --preset PREPARE (one-time fetch of pico-sdk + ODT)"
if [ -d OnDeviceTraining/src ] && [ -d pico-sdk/src ]; then
    echo "    pico-sdk + ODT already fetched — skipping"
else
    cmake --preset PREPARE 2>&1 | tail -5
fi
expect "Configuring done with two 'Klone nach …' lines (or 'already fetched')."

echo
echo "Toolchain ready. Next:"
echo "  21_prep_data.sh   # dataset prep (synthetic + real, ~minutes)"
echo "  22_test_l0_l1.sh  # L0 + L1 + standalone rq0 (depends on 21)"

pass
