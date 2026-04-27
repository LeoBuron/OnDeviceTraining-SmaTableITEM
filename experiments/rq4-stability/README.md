# RQ4 — On-device training stability

**Question.** Across ≥ 3 seeds, what is the run-to-run variance of final accuracy for the fine-tuning procedure on RP2350? Is the variance dominated by new-parameter initialisation, by on-device data order, or by both?

**Sweep axes.**
- seed (≥ 3, more if budget allows)
- variance-decomposition factor: {init only, data-order only, both}

**Measurement.** std-dev of final accuracy across seeds; variance shares attributed to init vs data order via a controlled-seed ANOVA.

**Sprint window.** D8–D9 (29.–30.04). **Runs first** — the noise floor it produces is the error-bar reference for RQ1 / RQ2 / RQ3 / RQ5.

**Layout (planned).**
```
rq4-stability/
├── README.md
├── seeds.yaml             explicit seed list + factor combinations
├── run_trial.py
├── analysis.ipynb         variance decomposition + noise-floor figure
└── runs/                  gitignored
```
