#!/bin/bash
# WHERE: Amplitude login-node (or your laptop after copying the experiment dir).
# WHAT:  Pretty-print the result of a previous Optuna campaign — trials.csv,
#        per-trial dirs, journal size, best-trial params.
# WHY:   Quick post-mortem after 40_smoke_sbatch.sh, or after a real run.
#        Re-runnable any time.
# COST:  ~1 s.
#
# Job ID is read from scripts/amplitude/logs/last_job_id (written by
# 40_smoke_sbatch.sh) unless passed as the first argument:
#   scripts/amplitude/50_inspect_result.sh 12345

source "$(dirname "$0")/_lib.sh"

JOBID="${1:-$(cat "$(dirname "$0")/logs/last_job_id" 2>/dev/null || echo "")}"
if [ -z "${JOBID}" ]; then
    echo "ERROR: no job id given and scripts/amplitude/logs/last_job_id is missing."
    echo "Usage: $0 <jobid>"
    exit 1
fi

HPC_HOME="${HPC_HOME:-$HOME}"
EXPDIR="${HPC_HOME}/experiments/smatable_optuna_${JOBID}"

say "Job ${JOBID} — accounting"
sacct -j "${JOBID}" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,ReqCPUs,Partition 2>&1 | head -10

if [ ! -d "${EXPDIR}" ]; then
    echo "ERROR: ${EXPDIR} not found."
    exit 1
fi

say "Experiment dir contents"
ls -la "${EXPDIR}"

say "trials.csv summary"
if [ -f "${EXPDIR}/trials.csv" ]; then
    n_trials=$(($(wc -l < "${EXPDIR}/trials.csv") - 1))
    echo "    trial count: ${n_trials}"
    echo "    --- header + first 5 trials ---"
    head -6 "${EXPDIR}/trials.csv" | column -t -s,
    echo "    --- best (highest 'value' column) ---"
    awk -F, 'NR==1 {for (i=1;i<=NF;i++) if ($i=="value") col=i; print; next}
             $col != "" {print $0 | "sort -t, -k"col"gr"}' "${EXPDIR}/trials.csv" | head -5 | column -t -s,
else
    echo "    MISSING trials.csv"
fi

say "Per-trial layout (first trial dir)"
first_trial=$(find "${EXPDIR}" -maxdepth 1 -type d -name "trial_*" | sort | head -1)
if [ -n "${first_trial}" ]; then
    ls -la "${first_trial}"
    echo "    --- env.json ---"
    cat "${first_trial}/env.json" 2>/dev/null | head -20
    echo "    --- stdout.log (last 15) ---"
    tail -15 "${first_trial}/stdout.log" 2>/dev/null
fi

say "Slurm logs (in repo dir)"
ls -la "smatable_optuna_${JOBID}".{out,err} 2>&1 | head -3

pass
