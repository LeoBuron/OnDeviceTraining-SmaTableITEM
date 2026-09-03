# RQ5 — Sample efficiency via noise augmentation

**Question.** Does additive Gaussian noise on the SmaTable vibration samples reduce the replay-buffer size (RQ1) or the new-user sample count (RQ2) at matched accuracy?

**Status (2026-09-03).** Stub — `src/examples/rq5_noise_augmentation.c` prints `RESULT skipped`. Second in the cut order (after RP2040). Light-touch claim by design — not load-bearing for the paper. Needs the RQ1 / RQ2 budget points to exist first.

**Sweep axes.**
- noise σ (a small grid, noise-only — no amplitude-scaling, no PCA)
- aug on/off, evaluated against RQ1 and RQ2 sample-budget points

**Measurement.** Δ accuracy at matched buffer size / sample count between aug-on and aug-off.

**Layout.**
```
rq5-noise-augmentation/
├── README.md              (this file)
└── runs/                  gitignored

src/examples/rq5_noise_augmentation.c         the binary (stub today; likely a noise knob on the RQ1 / RQ2 binaries rather than its own loop)
hpc/search_space/rq5_noise_augmentation.json  Optuna grid (to be written)
```
