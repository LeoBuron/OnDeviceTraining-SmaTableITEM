# Experiments

One subdirectory per research question. Layout matches the planned tree in the top-level `README.md` ("Repository structure (planned)" section) and the sprint plan's W2–W3 windows.

| Dir | RQ | Sprint window | Status |
|---|---|---|---|
| `rq1-replay-buffer/` | RQ1 — replay-buffer sizing curve (random vs herding) for original-user retention | D10–D12 (01.–03.05) | pending |
| `rq2-unseen-user/` | RQ2 — unseen-user fine-tuning sample-count sweep (LOOCV) | D13–D14 (04.–05.05) | pending |
| `rq3-new-gesture/` | RQ3 — new-gesture sample-count × contributor-count sweep | D15 (06.05) | pending |
| `rq4-stability/` | RQ4 — on-device training stability across ≥ 3 seeds; init vs data-order variance decomposition | D8–D9 (29.–30.04) | pending |
| `rq5-noise-augmentation/` | RQ5 — additive Gaussian noise ablation on RQ1 / RQ2 sample budgets | D16 (07.05) | pending |

Each subdir owns its Optuna config, the analysis notebook(s), and the per-RQ figures. Raw measurement artifacts go under `runs/` (gitignored, pattern reused from the Plan 2 audit).

Cross-cutting reminders:

- **Upstream bugs F1 + F2 are still present.** Any pre-fix run is conditioned on `effective_lr = batch_size × user_lr` and on per-epoch unique-sample count `≈ dataset_size / batch_size`. See `docs/odt-userapi-findings-misc.md`.
- **Numerics parity:** the supercomputer Optuna trial must link the same C training code as the on-device firmware. Don't fork the example for the sweep — parameterise it via CMake cache variables (precedent: `DEPTH_SWEEP_HIDDEN_LAYERS` in `mlp_mnist_depth_sweep_host`).
- **Cut order if D+22 slips** (from `README.md`): drop RP2400 → drop RQ5 → drop RQ3 → retreat to a 6-page short paper with RQ1 + RQ2 + RQ4 only.
