# Runbook — stage-1 LOSO sweeps on amplitUDE

*Written 2026-09-05, re-baselined 2026-09-10 for the session-wise split (decision D5) and the final-epoch headline (D1): the four `data/smatable-trial-*` datasets were re-prepped with four index sets per fold (train / retain / calib / test) and must be re-uploaded (A5); the B6 anchor comes from the re-built Release binary. Runs the four stage-1 Optuna studies (one per DepthwiseCNN configuration) and brings the checkpoints back. Every command is one line; run the blocks in order. Where a job id appears as `123456`, substitute the id `sbatch` printed.*

Why these steps exist: the cluster was last synced 2026-04-27, its 30-day workspace has expired, the vendored ODT there is at the April checkout, and the four trial datasets have never left this Mac. The container image does not need a rebuild (`uv.lock` unchanged since 2026-04-28) and is not used here anyway (`USE_CONTAINER=0`, the path the April smoke ran on).

## A. Local — Mac, repo root

**A1. Describe the working copy** (docs refresh, F9, paper scaffold, this runbook, the wrapper fix) and start a fresh change:

```
jj describe -m "docs+paper: refresh after missed ITEM deadline; recover paper plan; record F9; LNCS scaffold; stage-1 Amplitude runbook; per-dataset workspace cache"
jj bookmark set paper0 -r @
jj new
```

**A2. Repository visibility.** The GitHub repo is public and now carries the paper plan and draft; the April plan said to keep the paper private until camera-ready. If you agree:

```
gh repo edit LeoBuron/OnDeviceTraining-SmaTableITEM --visibility private --accept-visibility-change-consequences
```

**A3. Push.** Origin's `paper0` tip (`f8582263`) was rewritten locally, so this is a non-fast-forward move. jj knows about the rewrite and will do it. If it refuses because the remote bookmark "moved unexpectedly", run the fetch line again and tell me before forcing anything.

```
jj git fetch
jj git push --bookmark paper0
```

**A4. Upload the repo** (source only; `data/` is excluded; ~1 min):

```
scripts/amplitude/10_upload_repo.sh
```

**A5. Upload the four prepped trial datasets** (~500 MB, lands in `~/data/` on the cluster):

```
rsync -av --partial data/smatable-trial-2353 data/smatable-trial-3408 data/smatable-trial-3650 data/smatable-trial-4223 gateway.amplitude.uni-due.de:data/
```

## B. Login node — `ssh amplitude`, then `cd ~/projects/OnDeviceTraining-SmaTableITEM`

**B1. Toolchain for this shell:**

```
module load cmake/3.29.6 gcc/13.3.0
export PATH=$HOME/.local/bin:$PATH
```

**B2. Move the vendored ODT (cloned there in April) to the pin:**

```
git -C OnDeviceTraining/src fetch origin
git -C OnDeviceTraining/src checkout --detach 7d7f1d5
git -C OnDeviceTraining/src log -1 --oneline
```

Expect the last line to start with `7d7f1d5 test(optimizer): bound SYM-decoded v`.

**B3. Python env** (idempotent; skips PREPARE because both vendored dirs exist):

```
scripts/amplitude/20_setup.sh
```

**B4. Fold coverage** — must print the dataset id followed by `15 15`, four times (15 `train` and 15 `calib` index files each; a missing `calib` means the pre-2026-09-10 prep):

```
for t in 2353 3408 3650 4223; do echo $t $(ls $HOME/data/smatable-trial-$t/folds/LOSO | grep -c train.npy) $(ls $HOME/data/smatable-trial-$t/folds/LOSO | grep -c calib.npy); done
```

**B5. Build the trainer** (`HOST-Release`, `ODT_MEM_PROFILE=ON`):

```
hpc/build_all_rqs.sh stage1_pretrain
ls -la hpc/bin/stage1_pretrain.host
```

**B6. Three-epoch smoke on the login node** (~1 min). Same configuration as the local anchor run (Release binary, 2026-09-10), which printed `dataset: train=7559 eval=600 (calib=60 test=540)` and `RESULT accuracy=0.531667 test_acc=0.527778 best_val_acc=0.531667 best_epoch=3 n_params=434` (`accuracy` is the final-epoch accuracy on calib+test; `test_acc` the test rows only). Mac-vs-x86 bit-identity was verified for the toy example in April but not yet for this trainer, so if the accuracy differs, send me the RESULT line — it is information, not a failure.

```
SMATABLE_DATA_DIR=$HOME/data/smatable-trial-3650 SMATABLE_FOLD_SCHEME=LOSO SMATABLE_FOLD=0 ODT_WIDTHS=8,8 ODT_KERNEL_SIZE=5 ODT_DILATION=2 ODT_P_DROP=0.0 ODT_LR=0.1 ODT_MOMENTUM=0.9 ODT_WEIGHT_DECAY=0.0 ODT_BATCH_SIZE=128 ODT_SEED=42 ODT_LR_SCHEDULE=cosine ODT_EPOCHS=3 hpc/bin/stage1_pretrain.host | tail -2
```

**B7. Submit the four studies.** Each takes one full node (64 cores); one trial is one single-threaded process, so 48 workers. `--time` on the command line overrides the 12 h default in the script.

```
sbatch --time=24:00:00 --export=ALL,RQ=stage1_trial3650,HOST_BIN_NAME=stage1_pretrain,DATASET_DIR=$HOME/data/smatable-trial-3650,N_WORKERS=48,TRIAL_TIMEOUT_S=14400,USE_CONTAINER=0 hpc/run_optuna_amplitude.sh
sbatch --time=48:00:00 --export=ALL,RQ=stage1_trial4223,HOST_BIN_NAME=stage1_pretrain,DATASET_DIR=$HOME/data/smatable-trial-4223,N_WORKERS=48,TRIAL_TIMEOUT_S=14400,USE_CONTAINER=0 hpc/run_optuna_amplitude.sh
sbatch --time=48:00:00 --export=ALL,RQ=stage1_trial2353,HOST_BIN_NAME=stage1_pretrain,DATASET_DIR=$HOME/data/smatable-trial-2353,N_WORKERS=48,TRIAL_TIMEOUT_S=14400,USE_CONTAINER=0 hpc/run_optuna_amplitude.sh
sbatch --time=96:00:00 --export=ALL,RQ=stage1_trial3408,HOST_BIN_NAME=stage1_pretrain,DATASET_DIR=$HOME/data/smatable-trial-3408,N_WORKERS=48,TRIAL_TIMEOUT_S=72000,USE_CONTAINER=0 hpc/run_optuna_amplitude.sh
```

Expected wall clock, from local per-trial timings: trial-3650 about 2.5 h (150 trials × ~46 min / 48 workers); trial-3408 about 30 h (75 × ~17.5 h / 48), inside the 96 h partition cap; trial-2353 and trial-4223 are unmeasured at 250 epochs, 48 h is a guess with margin. Because the fold is a grid dimension, `GridSampler` re-evaluates a few cells at 48 workers; the duplicates are pruned after the fact and are harmless. The wrapper now caches each dataset under its own name in the workspace (fixed 2026-09-05; before, every study silently reused the first dataset). The per-trial timings were measured under the old split; the train set is now 7559 instead of 8399 windows, so epochs run slightly faster.

**B8. Monitor:**

```
squeue -u $USER
tail -n 20 smatable_optuna_123456.out
scripts/amplitude/50_inspect_result.sh 123456
```

## C. Collect and aggregate — Mac

**C0. Start the Adam reference on the Mac** right after B7 (hours; trial-3408 is the slow one). It is the V3 gate's reference and C3 reads it from `runs/r0/reference.csv` by default; the script is resumable:

```
uv run tools/stage1_r0.py --configs trial-2353,trial-3408,trial-3650,trial-4223 --folds 0-14 --out runs/r0/reference.csv
```

**C1. Find the Lustre home once:**

```
H=$(ssh gateway.amplitude.uni-due.de 'echo "${HPC_HOME:-$HOME}"')
```

**C2. Pull every finished study:**

```
rsync -av gateway.amplitude.uni-due.de:$H/experiments/ runs/optuna-amplitude/
```

**C3. Aggregate**, one call per architecture (directory names come from C2's listing):

```
uv run tools/aggregate_stage1.py --run-dir runs/optuna-amplitude/smatable_optuna_123456 --config-name trial-3650 --out runs/stage1_selection_3650.json
```

For trial-3408 with the optional weight-decay extension study, pass both run directories (see "Trial-3408" in `hpc/README.md`). The four selection JSONs are the inputs for the RQ1 port and stage 2 — hand them to me.

## Deliberately not in this runbook

- No container (`USE_CONTAINER=0`): the login-node `uv` venv on Lustre is what the April smoke used; the `.sif` only matters if that venv breaks.
- No `11_upload_dataset.sh`: it uploads the raw Nextcloud window tree for `smatable-real`, which stage 1 does not use.
- No MCU work: blocked by F9 (`docs/odt-userapi-findings-misc.md`).
