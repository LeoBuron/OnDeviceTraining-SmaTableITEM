#!/bin/bash
# WHERE: Amplitude login-node, inside the repo.
# WHAT:  sbatch the rq0 toy against the synthetic dataset; wait for the job
#        to finish; dump slurm out/err and the experiment dir into this
#        script's log.
# WHY:   The end-to-end smoke: Slurm + Apptainer + ws_allocate +
#        bind-mounts + Optuna + HOST binary + trials.csv all in one shot.
# COST:  ~2–5 min (job itself ~1 min + Slurm queue time).
#
# Override config via env:
#   RQ=rq0_toy_synthetic   N_WORKERS=4   FOLD_SCHEME=LOSO

source "$(dirname "$0")/_lib.sh"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

RQ="${RQ:-rq0_toy_synthetic}"
N_WORKERS="${N_WORKERS:-4}"
FOLD_SCHEME="${FOLD_SCHEME:-LOSO}"

say "Pre-flight checks"
for f in hpc/run_container.sif "hpc/bin/${RQ}.host" "hpc/search_space/${RQ%_*}_toy.json" "hpc/run_optuna_amplitude.sh"; do
    case "$f" in
        *_toy.json)
            f="hpc/search_space/rq0_toy.json"
            ;;
    esac
    if [ ! -f "$f" ]; then
        echo "    MISSING: $f"
        exit 1
    fi
    printf "    OK  %s\n" "$f"
done

say "sbatch submit"
JOBID=$(sbatch --parsable --export="ALL,RQ=${RQ},N_WORKERS=${N_WORKERS},FOLD_SCHEME=${FOLD_SCHEME}" \
        hpc/run_optuna_amplitude.sh)
echo "    JobID = ${JOBID}"
echo "${JOBID}" > "$(dirname "$0")/logs/last_job_id"

say "squeue (initial)"
squeue -j "${JOBID}" 2>&1 || true

say "Waiting for job to complete (polling every 10s)"
spin=0
while squeue -j "${JOBID}" -h -o "%T" 2>/dev/null | grep -qE "^(PENDING|RUNNING|COMPLETING|CONFIGURING|RESV_DEL_HOLD|REQUEUED)$"; do
    state=$(squeue -j "${JOBID}" -h -o "%T" 2>/dev/null || echo "?")
    printf "\r    [%3ds] state=%s  " "$((spin*10))" "${state}"
    spin=$((spin+1))
    if [ "${spin}" -gt 60 ]; then
        echo
        echo "    Job >10 min in queue/run — bailing out of polling. Check manually with: squeue -j ${JOBID}"
        echo "    (Slurm output may be in: smatable_optuna_${JOBID}.{out,err})"
        exit 1
    fi
    sleep 10
done
echo

say "Final accounting"
sacct -j "${JOBID}" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,ReqCPUs 2>&1 | head -10 || true

say "slurm stdout (smatable_optuna_${JOBID}.out)"
if [ -f "smatable_optuna_${JOBID}.out" ]; then
    cat "smatable_optuna_${JOBID}.out"
else
    echo "    (file missing — job may have failed before writing output)"
fi

say "slurm stderr (smatable_optuna_${JOBID}.err)"
if [ -f "smatable_optuna_${JOBID}.err" ] && [ -s "smatable_optuna_${JOBID}.err" ]; then
    cat "smatable_optuna_${JOBID}.err"
else
    echo "    (empty or missing — usually a good sign)"
fi

say "Final experiment dir"
HPC_HOME="${HPC_HOME:-$HOME}"
EXPDIR="${HPC_HOME}/experiments/smatable_optuna_${JOBID}"
if [ -d "${EXPDIR}" ]; then
    ls -la "${EXPDIR}"
    if [ -f "${EXPDIR}/trials.csv" ]; then
        echo
        echo "    --- trials.csv (head) ---"
        head -10 "${EXPDIR}/trials.csv"
    fi
else
    echo "    MISSING: ${EXPDIR}"
fi

pass
