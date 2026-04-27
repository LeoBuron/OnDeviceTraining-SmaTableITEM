# RQ2 — Fine-tuning sample efficiency (unseen user)

**Question.** How many labelled samples from a previously-unseen user are required before on-device fine-tuning recovers accuracy on that user under leave-one-user-out CV?

**Sweep axes.**
- new-user sample count K
- held-out user (LOOCV across the 15-user SmaTable cohort)

**Measurement.** Two metrics per fold:
- target-user accuracy gain (held-out user's test events)
- original-users retention (replay-buffer pulled from RQ1's chosen size)

**Sprint window.** D13–D14 (04.–05.05). Depends on RQ1 to fix the replay-buffer size used during fine-tuning.

**Layout (planned).**
```
rq2-unseen-user/
├── README.md
├── sweep.yaml             Optuna config (per-K, per-fold)
├── run_trial.py
├── analysis.ipynb
└── runs/                  gitignored
```
