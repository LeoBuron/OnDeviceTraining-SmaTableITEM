# RQ4 — On-device training stability

**Question.** Across ≥ 3 seeds, what is the run-to-run variance of final accuracy for the fine-tuning procedure on RP2350? Is the variance dominated by new-parameter initialisation, by on-device data order, or by both?

**Status (2026-09-03).** Stub — `src/examples/rq4_stability.c` prints `RESULT skipped`. Meant to run first, as the noise-floor reference for the error bars of every other RQ. One code change stands in the way of making it a search-space-only study: `src/examples/stage1_pretrain.c` seeds both the weight init (`rngSetSeed(seed)` before model construction) and the per-epoch shuffle (loader seeded with `seed + epoch`) from the single `ODT_SEED`, so the init-vs-data-order decomposition needs a second env var (e.g. `ODT_DATA_SEED`) in whichever binary carries the fine-tune loop. Until then RQ1's in-grid `seed` axis is the interim noise floor. Survives every step of the cut order.

**Sweep axes.**
- seed (≥ 3, more if budget allows)
- variance-decomposition factor: {init only, data-order only, both}

**Measurement.** std-dev of final accuracy across seeds; variance shares attributed to init vs data order via a controlled-seed ANOVA.

**Layout.**
```
rq4-stability/
├── README.md              (this file)
└── runs/                  gitignored

src/examples/rq4_stability.c            the binary (stub today; may end up as a mode of the RQ2 fine-tune binary rather than its own file)
hpc/search_space/rq4_stability.json     Optuna grid (to be written: fold × init_seed × data_seed × factor)
```
