#!/bin/bash
# WHERE: Amplitude login-node (after `ssh login-ampl`).
# WHAT:  Verify the Amplitude environment matches the assumptions baked into
#        hpc/run_optuna_amplitude.sh — partition name, workspace tooling,
#        Apptainer presence, HPC_HOME existence.
# WHY:   Catches "wrong --partition", "ws_allocate not found", "apptainer not
#        loaded" before any sbatch attempt.
# COST:  ~5 seconds.

source "$(dirname "$0")/_lib.sh"

say "Slurm partitions"
sinfo -o "%P %D %c %m %t" 2>&1 | head -30 || echo "    (sinfo failed — Slurm not on PATH?)"
expect "a CPU-only partition listed (often 'standard'/'normal'/'cpu'). Note its name."

echo
say "Tools the sbatch script depends on"
for tool in apptainer ws_allocate ws_find rsync sbatch squeue; do
    if path=$(command -v "$tool" 2>/dev/null); then
        printf "    %-13s -> %s\n" "$tool" "$path"
    else
        printf "    %-13s -> MISSING\n" "$tool"
    fi
done
expect "all six tools present. If apptainer or ws_* is missing, try: module avail apptainer ; module avail workspace"

echo
say "Module system (if any)"
if command -v module >/dev/null 2>&1; then
    module avail apptainer 2>&1 | head -10 || true
    module avail workspace 2>&1 | head -10 || true
fi

echo
say "HPC_HOME and disk picture"
echo "    HPC_HOME = ${HPC_HOME:-(unset — falling back to \$HOME=$HOME)}"
df -h "${HPC_HOME:-$HOME}" 2>&1 | head -5

echo
say "Account / partition limits (best-effort)"
sacctmgr -n show user "$USER" format=Account,DefaultAccount 2>&1 | head -5 || echo "    (sacctmgr not available)"
scontrol show partition 2>&1 | head -20 || true

pass
