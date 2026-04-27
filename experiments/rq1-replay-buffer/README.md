# RQ1 — Replay-buffer sizing (existing-user retention)

**Question.** How many samples per class from the original pre-training set must the replay buffer retain to prevent catastrophic forgetting during on-device fine-tuning to a new user?

**Sweep axes.**
- replay-buffer size per class
- selection strategy: `random` (baseline) vs `herding`

**Measurement.** Accuracy on original-user test set after on-device fine-tuning, on RP2350. Optuna drives the sweep on the supercomputer; top-3 configs re-verified on real hardware.

**Sprint window.** D10–D12 (01.–05.05). Depends on RQ4 noise-floor (D8–D9) for error-bar interpretation.

**Layout (planned).**
```
rq1-replay-buffer/
├── README.md              (this file)
├── sweep.yaml             Optuna study config
├── run_trial.py           single-trial wrapper invoking the CMake build + on-host execution
├── analysis.ipynb         curve fits + figure generation
└── runs/                  gitignored — raw per-trial CSVs and state dumps
```
