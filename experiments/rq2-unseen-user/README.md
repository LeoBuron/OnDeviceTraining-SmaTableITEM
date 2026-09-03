# RQ2 — Fine-tuning sample efficiency (unseen user)

**Question.** How many labelled samples from a previously-unseen user are required before on-device fine-tuning recovers accuracy on that user under leave-one-user-out CV?

**Status (2026-09-03).** Stub — `src/examples/rq2_unseen_user.c` prints `RESULT skipped`. This RQ is "stage 2" of the stage-1 / stage-2 design: stage 1 (`src/examples/stage1_pretrain.c`) trains the DepthwiseCNN LOSO and writes a reloadable npy state-dict checkpoint per fold; RQ2 reloads that checkpoint (`modelLoadStateDict`) and fine-tunes on K samples of the held-out user. Depends on the stage-1 Amplitude sweeps (checkpoints) and on RQ1 for the replay budget used during fine-tuning.

**Sweep axes.**
- new-user sample count K
- held-out user (LOOCV across the 15-user SmaTable cohort)

**Measurement.** Three metrics per fold:
- target-user accuracy gain (held-out user's test events)
- original-users retention (replay budget pulled from RQ1's chosen size)
- **FWT (forward transfer)**, GEM's definition: target-user accuracy at K=0 (zero-shot, right after pretraining on the original users, before any fine-tuning) minus a random-init reference baseline. This is the natural home for FWT — it's exactly the K=0 anchor point of this sweep, unlike RQ1 where it would be a fold-constant, sweep-invariant number (see that RQ's README for why it was deferred here).

**Design notes.**
- Baselines from `docs/paper-plan.md` map onto this sweep: B1 = pretrained, no fine-tune (the K=0 point); B2 = naïve full-model fine-tune without replay; B3 = last-layer-only; B4 = full-model with replay (primary).
- Stage-1 checkpoints support full-network fine-tuning (all backward passes exist), so head-only vs. full-model is a runtime knob, not a checkpoint property.

**Layout.**
```
rq2-unseen-user/
├── README.md              (this file)
└── runs/                  gitignored — raw per-trial output, later figures

src/examples/rq2_unseen_user.c          the HOST/MCU binary (stub today)
hpc/search_space/rq2_unseen_user.json   Optuna grid (to be written: fold × K × finetune_mode × seed)
```
