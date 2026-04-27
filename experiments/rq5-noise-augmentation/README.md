# RQ5 — Sample efficiency via noise augmentation

**Question.** Does additive Gaussian noise on the SmaTable vibration samples reduce the replay-buffer size (RQ1) or the new-user sample count (RQ2) at matched accuracy?

**Sweep axes.**
- noise σ (a small grid, noise-only — no amplitude-scaling, no PCA)
- aug on/off, evaluated against RQ1 and RQ2 sample-budget points

**Measurement.** Δ accuracy at matched buffer size / sample count between aug-on and aug-off.

**Sprint window.** D16 (07.05). **Second in the cut order** after RP2400 if the schedule slips. Light-touch claim by design — not load-bearing for the paper.

**Layout (planned).**
```
rq5-noise-augmentation/
├── README.md
├── sweep.yaml
├── run_trial.py
├── analysis.ipynb
└── runs/                  gitignored
```
