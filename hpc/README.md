# HPC harness — Amplitude

Optuna-orchestrated CV sweep of per-RQ HOST binaries on the Amplitude (UDE)
Slurm cluster. Pattern adapted from
[`smatable-offline`](file:///Users/leo/work/smatable-offline) — same
Apptainer + `ws_allocate` + `JournalFileBackend` + `mp.Pool` shape, with
GPU stripped out (CPU-only FP32 ODT trials).

Spec: `docs/superpowers/specs/2026-04-27-hpc-experiment-harness-design.md`.

## Quickstart (local smoke test)

```bash
# 1. Generate synthetic dataset (or run without --synthetic against the real
#    preprocessed-windows tree).
uv run tools/prep_smatable.py --synthetic --dst data/smatable --schemes LOSO \
    --syn-T 32 --syn-subjects 4 --syn-sessions 3 --syn-per-class 5

# 2. L0 byte-equivalence check (Python).
uv run tests/dataset_l0_bytes.py --dst data/smatable

# 3. Build per-RQ HOST binaries.
hpc/build_all_rqs.sh                     # all 6 RQs
hpc/build_all_rqs.sh rq0_toy_synthetic   # one RQ

# 4. L1 backend-equivalence check (HOST only).
cmake --preset HOST-Debug -DODT_EXAMPLE=rq0_toy_synthetic -DODT_L1_TEST=ON
cmake --build --preset HOST-Debug --target dataset_l1_equiv
SMATABLE_DATA_DIR=$PWD/data/smatable SMATABLE_FOLD_SCHEME=LOSO SMATABLE_FOLD=0 \
    build/HOST-Debug/tests/dataset_l1_equiv

# 5. Run an Optuna campaign locally.
uv run hpc/run_optuna.py \
    --rq rq0_toy_synthetic \
    --host-bin hpc/bin/rq0_toy_synthetic.host \
    --data-dir data/smatable \
    --search-space hpc/search_space/rq0_toy.json \
    --log-dir runs/optuna-local \
    --fold-scheme LOSO \
    --n-workers 4
```

## Stage-1 datasets

The four stage-1 trial datasets are prepared from preprocessed gesture-windows stored under `data/model_and_dataset/trial-{2353,3408,3650,4223}/dataset/`. Each run generates 8999 canonical samples (4-channel, variable-length windows) with 15-fold LOSO and AOS fold splits:

```bash
uv run tools/prep_smatable.py --src data/model_and_dataset/trial-2353/dataset --dst data/smatable-trial-2353 --schemes LOSO,AOS --no-baked
uv run tools/prep_smatable.py --src data/model_and_dataset/trial-3408/dataset --dst data/smatable-trial-3408 --schemes LOSO,AOS --no-baked
uv run tools/prep_smatable.py --src data/model_and_dataset/trial-3650/dataset --dst data/smatable-trial-3650 --schemes LOSO,AOS --no-baked
uv run tools/prep_smatable.py --src data/model_and_dataset/trial-4223/dataset --dst data/smatable-trial-4223 --schemes LOSO,AOS --no-baked
```

| Trial | T (window samples) | n_events | n_classes |
|---|---|---|---|
| 2353 | 1250 | 8999 | 6 |
| 3408 | 1250 | 8999 | 6 |
| 3650 | 250 | 8999 | 6 |
| 4223 | 625 | 8999 | 6 |

All outputs land in gitignored `data/smatable-trial-<id>/` directories with bytewise-canonical `smatable_x.npy [8999,4,T]` and `smatable_y.npy` plus fold indices for LOSO (15 subjects, ~8399 train / ~600 test per fold) and AOS (15 sessions, ~8459 train / ~540 test per fold). The `--no-baked` flag ensures no MCU header files are generated (stage 2 integrates baked backends for RP2350 runs).

### Trial-3408: reduced grid + optional extension

Trial-3408 does not fit the RP2350 (measured 1953 KiB vs 520 KB SRAM) and costs ~17.5 h per 250-epoch trial, so its base study runs a **reduced grid**: `stage1_trial3408.json` fixes `weight_decay=0.0` (75 trials instead of 150). Its trials also need a larger per-trial timeout: `TRIAL_TIMEOUT_S=72000` (the 14400 s used for the other configs kills every 3408 trial).

If compute allows, extend to the full protocol by running the **complement grid** `stage1_trial3408_ext_wd.json` (`weight_decay=0.0001`, the exact 75 missing combos) as a second study — never widen the grid of an existing study: `GridSampler` identifies cells by index within the serialized search space, so a changed space re-runs everything. Aggregate both studies as one by passing multiple run dirs:

```bash
uv run tools/aggregate_stage1.py --run-dir runs/optuna-amplitude/stage1_trial3408--<ts> runs/optuna-amplitude/stage1_trial3408_ext_wd--<ts> --config-name trial-3408 --out stage1_selection_3408.json
```

The aggregator refuses to merge studies whose search-space *key sets* differ (values may differ — that is the point of an extension).

### RESULT keys

`stage1_pretrain`'s RESULT line carries 23 keys total: the 19 base keys (accuracy/params/wall-clock, the 7-term static `mem_*_b` budget — params+grads+optstate+act+gradbuf+io+masks — plus RSS/CPU milestones and `stack_peak_b`, measured via upstream `measurePeakStackBytes`), and 4 heap-counter keys (`mem_heap_peak_b`, `mem_dataset_heap_b`, `mem_model_heap_b`, `mem_reconciliation_gap_b`) gated by the `ODT_MEM_PROFILE` CMake flag — all four print 0 on a binary built without it. `hpc/build_all_rqs.sh` now passes `-DODT_MEM_PROFILE=ON` by default. `mem_mcu_total_b` (static budget) plus `stack_peak_b` (measured) against the RP2350's 520 KB SRAM is the on-device feasibility figure.

### RQ1: replay-buffer sweep

`rq1_replay_buffer` has its own build-and-timeout requirements, found during a
pre-flight audit before the first real Amplitude submission:

- **Build with `HOST-Release`, not `HOST-Debug`.** `hpc/build_all_rqs.sh`
  already does this by default. At real dataset scale (WF≈4000), the PPCA
  arm's per-class absorption (a Jacobi eigendecomposition, O(p^3) per merge
  call) is expensive enough under `-O0` that even the cheapest arm (`none`,
  no PPCA at all) came within a hair of a 300s timeout — `-O2` alone brought
  every arm comfortably under it. `DEBUG_MODE_ERROR` stays on regardless
  (set unconditionally in `host_post.cmake`, independent of build type), so
  `PRINT_ERROR` diagnostics aren't lost by switching.
- **`TRIAL_TIMEOUT_S=1200`** (20 min) for all three rq1 search spaces
  (`rq1_replay_buffer.json`, `rq1_ppca_rank_sweep.json`,
  `rq1_chunk_size_sweep.json`). Measured worst case with the `HOST-Release`
  build: 389s (`ppca` at `ODT_MAX_SESSION_SAMPLES=512`, the chunk-size
  sweep's most expensive point) — 1200s gives >3x margin uniformly across
  all three.
- **`DATASET_DIR` must point at a dataset with all 15 LOSO folds.** The
  search spaces sweep `fold: 0..14`; a dataset prepared with a smaller
  subject cohort (e.g. the synthetic smoke-test set) silently fails
  `smatableDatasetOpen` on the missing folds and Optuna prunes those trials
  rather than erroring loudly. Verify fold coverage before submitting:
  `ls $DATASET_DIR/folds/LOSO | grep -c train.npy` should print `15`.

## On Amplitude

```bash
# 1. One-time: rsync the preprocessed dataset to $HPC_HOME/data/smatable.
#    (1.3 GB — fits comfortably under the workspace 30-day allocation.)
rsync -av path/to/preprocessed-windows/ $HPC_HOME/data/smatable-preprocessed/
ssh amplitude  uv run tools/prep_smatable.py \
    --src $HPC_HOME/data/smatable-preprocessed \
    --dst $HPC_HOME/data/smatable

# 2. One-time: build the Apptainer image on a build node.
apptainer build hpc/run_container.sif hpc/run_container.def

# 3. Per campaign: build the per-RQ HOST binary on the login node, then submit.
hpc/build_all_rqs.sh rq0_toy_synthetic
sbatch --export=ALL,RQ=rq0_toy_synthetic,N_WORKERS=32 hpc/run_optuna_amplitude.sh
```

The sbatch script:
- claims/reuses the `smatable-ws` Lustre workspace (30-day allocation),
- rsyncs the dataset to scratch on first run, reuses on subsequent,
- bind-mounts `/data` (dataset), `/logs` (Optuna journal + per-trial logs),
  `/host_bin` (pre-built RQ binary), `/search_space.json`,
- launches `hpc/run_optuna.py` inside the container,
- copies `slurm-*.{out,err}` plus the search space JSON into the run dir,
- final-rsyncs everything into `$HPC_HOME/experiments/`.

## Per-RQ binary contract

Each binary in `hpc/bin/<rq>.host` is a self-contained executable that:

1. reads env: `SMATABLE_DATA_DIR`, `SMATABLE_FOLD_SCHEME`, `SMATABLE_FOLD`,
   `ODT_*` hyperparams (RQ-specific keys: `ODT_LR`, `ODT_EPOCHS`,
   `ODT_HIDDEN`, `ODT_SEED`, `ODT_BUFFER_SIZE`, `ODT_REPLAY_MODE`, …),
2. prints `BEGIN <rq> <iso8601>` to stdout,
3. prints `EPOCH <e> train_loss=<f> train_acc=<f> val_loss=<f> val_acc=<f>`
   per epoch,
4. prints `RESULT accuracy=<f> best_epoch=<i> n_params=<i> wall_clock_s=<f>`
   (or `RESULT skipped reason="<...>"`),
5. exits 0 on success, non-zero on failure (Optuna marks failed trials
   pruned and continues).

Conv1d/LayerNorm/GroupNorm all landed upstream at the current pin (7d7f1d5),
so this blocker is resolved. `rq1_replay_buffer` is implemented (three-arm
PPCA/exemplar replay comparison, see `experiments/rq1-replay-buffer/README.md`).
Stubs `rq2..rq5` still print `RESULT skipped` pending their own per-RQ
training-loop design — see the spec's "Step 4 — deferred work".

## HOST↔MCU equivalence layers

| Layer | What | Where | Cost |
|---|---|---|---|
| L0 | header bytes ≡ .npy slice | `tests/dataset_l0_bytes.py` | µs |
| L1 | NPY backend ≡ baked backend at runtime | `tests/dataset_backend_equiv.c` (CMake target `dataset_l1_equiv`) | µs |
| L2 | training step on HOST(NPY) ≡ HOST(baked) | TBD — reuses `state_dump_compare.py` from Plan-2 audit | seconds |
| L3 | training step on HOST(baked) ≡ RP2350(baked) | TBD — UART collector + same comparator | minutes |

L0 + L1 are wired and green for synthetic LOSO fold 0. L2 + L3 land alongside
the real Conv1d/LayerNorm RQ implementations.

## Known issue surfaced by the L1 test

Upstream `tensorInit` INT32 path destroys int32 inputs (treats `data*` as
`float*`, casts via `(int32_t)data[i]`). The NPY backend works around it by
reading int32 .npy files directly. Documented as **F7** in
`docs/odt-userapi-findings-misc.md`.
