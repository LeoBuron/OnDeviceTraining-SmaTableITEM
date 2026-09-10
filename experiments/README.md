# Experiments

One subdirectory per research question. The paper's spec (claim, contribution claims, RQ definitions, cut order) and its maintained current state live in `docs/paper-plan.md`.

Convention per RQ (set by RQ1): the binary is `src/examples/<rq>.c`, the Optuna grid is `hpc/search_space/<rq>*.json`, the design is the README in this directory, raw trial output goes under `runs/` (gitignored). Analysis notebooks and figures land in the RQ subdirectory once results exist.

| Dir | RQ | Status (2026-09-03) |
|---|---|---|
| `rq1-replay-buffer/` | RQ1 — replay sizing at iso-byte budgets: `ppca` vs `exemplar_random` vs `exemplar_herding` vs `none` | **Implemented, not swept.** Runs on a placeholder MLP; needs a port to the stage-1 DepthwiseCNN checkpoint before any number counts |
| `rq2-unseen-user/` | RQ2 — unseen-user fine-tuning sample-count sweep (LOOCV); this is "stage 2" of the stage-1 / stage-2 design | stub |
| `rq3-new-gesture/` | RQ3 — new-gesture sample-count × contributor-count sweep | stub; dataset-construction question to Florian open |
| `rq4-stability/` | RQ4 — seed variance; init vs data-order decomposition | stub; the trainer needs a second seed env var first |
| `rq5-noise-augmentation/` | RQ5 — Gaussian-noise ablation on RQ1 / RQ2 budgets | stub |

Stubs print `RESULT skipped reason="not implemented yet (...)"`; `hpc/run_optuna.py` prunes such trials, so the driver can still iterate them for plumbing tests.

Prerequisite for every RQ: stage-1 checkpoints from `src/examples/stage1_pretrain.c` (final-epoch npy state dict + `manifest.json` + `memory.json` per trial, reloaded via `modelLoadStateDict`), trained on the session-wise LOSO split (per fold: `train` = other users × sessions 1–9, `retain` = other users × session 10, `calib` = held-out user × session 1, `test` = held-out user × sessions 2–10; `tools/prep_smatable.py`). Stage 1 is verified locally on that split (gates V0, V1/V2, V4) but the Amplitude sweeps have not run yet — see "Current state" in `docs/paper-plan.md`.

Cross-cutting reminders:

- **Upstream-bug status at the current pin (`7d7f1d5`):** F1 (CE-gradient mean scaling), F2 (DataLoader `indices[]`) and F8 (inferenceStats heap overflow) are fixed upstream. F7 (`tensorInit` INT32 path) is still live and worked around in `src/dataset/smatable_dataset_npy.c`. Details: `docs/odt-userapi-findings-misc.md`.
- **Numerics parity:** the supercomputer Optuna trial must link the same C training code as the on-device firmware. Don't fork the example for the sweep — parameterise it via env vars (`ODT_*`, `SMATABLE_*`) or CMake cache variables.
- **Build sweep binaries with `HOST-Release`** (`hpc/build_all_rqs.sh` does); `-O0` pushes the PPCA arm past trial timeouts.
- **Grids are `GridSampler` studies:** never widen an existing study's grid (grid ids are index-based, a changed space re-runs everything). Add a complement JSON as a second study and merge with `tools/aggregate_stage1.py --run-dir A --run-dir B`.
- **Cut order** (from `docs/paper-plan.md`): drop RP2040 → drop RQ5 → drop RQ3 → short paper with RQ1 + RQ2 + RQ4 only.
- **Noise floor:** until RQ4 exists, RQ1's 3-value `seed` axis is the only run-to-run variance estimate.
