# Towards Continual Learning for SmaTables

**Replay-Buffer Sizing and On-Device Fine-Tuning for Edge-MCU Gesture Recognition**

> First on-RP2350 measurements of both inference and training of the SmaTable gesture classifier, plus quantitative sample-budget and stability envelopes for three continual-learning scenarios the SmaTable application actually requires.

> **Provenance.** This file is the paper's working spec. It was the repo README during the April 2026 sprint, was lost from the tracked tree in the upstream/develop merge, and was recovered on 2026-09-03 from orphaned commit `e571381` (`git show e571381:README.md`). The "Current state" section is maintained here; everything from "TL;DR" down is the paper design and changes only when the paper's scope changes.

---

## Status

| | |
|---|---|
| **Phase** | Active again since 2026-09-05; venue still open, the full five-RQ scope stays the plan of record |
| **Venue** | **TBD** (decision deferred). Original target was ITEM 2026 @ ECML PKDD (Naples, 11 Sep 2026); its submission deadline (2026-06-05) was missed |
| **Format** | Assumed LNCS-style full paper (≤ 12 pages) until the venue is fixed |
| **Deadline** | None set |
| **Authors** | Leo Buron (first), Florian Hettstedt, Andreas Erbslöh, Gregor Schiele |
| **Affiliation** | Intelligent Embedded Systems, University of Duisburg-Essen |
| **Original sprint** | 2026-04-21 → 2026-05-13 (historical; the plan is kept at the end of "Current state") |

---

## Current state

*Updated 2026-09-03. Last code change 2026-07-21 (`15cca8a`). Execution history of the stage-1 work is in the local, gitignored ledger `.superpowers/sdd/progress.md`.*

### Workstreams

| Workstream | State | Blocked on |
|---|---|---|
| Optuna + Slurm + Apptainer harness (`hpc/`) | Done. SQLite-WAL storage on node-local tmpfs; verified on Amplitude in April on the synthetic toy set | — |
| Stage 1: DepthwiseCNN LOSO pretrain in ODT (`src/examples/stage1_pretrain.c`) | Code done. GroupNorm(1,C) is upstream; gates V0–V4 green on all four trial architectures; local 15-fold pilot on trial-3650 passes V3 (mean 0.7205 vs. Adam reference 0.7045) | Amplitude sweeps (4 grids) never started: 15 commits on `paper0` unpushed, cluster last synced 2026-04-27, trial datasets never uploaded |
| RP2350 memory feasibility | Done, paper-ready. trial-3650 = 109 KiB, trial-2353 = 242 KiB, trial-4223 = 303 KiB fit in 520 KB SRAM; trial-3408 = 1953 KiB does not (`executeOp` VLA stack scales with width × T) | — |
| RQ1 replay buffer (`src/examples/rq1_replay_buffer.c`) | Implemented: four arms (`ppca` / `exemplar_random` / `exemplar_herding` / `none`) at iso-byte budgets, three search spaces (900 + 105 + 90 trials). Runs on a **placeholder MLP**; never swept | Stage-1 checkpoints → port to the DepthwiseCNN via `modelLoadStateDict` |
| RQ2 unseen user (= "stage 2") | Stub | Implementation; stage-1 checkpoints; RQ1's chosen replay budget |
| RQ3 new gesture | Stub | Dataset-construction decision with Florian (open since April) |
| RQ4 stability | Stub. Trainer seeds weight init and data order from one `ODT_SEED` | Second seed env var + a grid; small |
| RQ5 noise augmentation | Stub | RQ1 / RQ2 budget points |
| On-device RP2350 | **Nothing yet, and currently unbuildable.** Bench: RP2350 setup incl. energy measurement almost ready (2026-09-05); RP2040 companion setup ordered, expected ~2026-09-12. At pin `7d7f1d5` the MCU configure fails (reproduced on `PICO2_W`; PICO1/STM32 share the toolchain): upstream `MemProfile` hard-requires pthreads (finding **F9**, `docs/odt-userapi-findings-misc.md`, found 2026-09-03). Nothing flashed, L3 (HOST ↔ MCU single-step parity) never run. Baked-header dataset backend and the `PICO2_W` target are wired; L0/L1 pass | F9 fix upstream on `develop` (decided 2026-09-05), re-pin, then hardware bring-up |
| Paper draft | Does not exist | Results |

### Dependency chain

Each step needs the artifact of the previous one.

1. Push `paper0` (non-fast-forward: origin's tip `f8582263` was rewritten locally; jj knows), re-upload repo + the four trial datasets, re-allocate the Lustre workspace. The container needs no rebuild (`uv.lock` unchanged since 2026-04-28).
2. Stage-1 sweeps on Amplitude: `stage1_trial{2353,3650,4223}.json` (150 trials each), `stage1_trial3408.json` (75 trials, `TRIAL_TIMEOUT_S=72000`; optional `_ext_wd` complement +75). Aggregate with `tools/aggregate_stage1.py`, pick per-architecture winners.
3. RQ1: replace the placeholder MLP with the DepthwiseCNN reloaded from a stage-1 checkpoint; sweep the main grid plus the two PPCA side studies.
4. RQ2: implement the K-sample fine-tune on the held-out user (FWT at K = 0). RQ4: second seed + grid. RQ3 / RQ5 per cut order.
5. RP2350: fix F9 (MCU configure blocker), build with `PICO2_W` + baked fold headers, flash, run inference, run one training step, L3 comparator vs. HOST, then the top configs on device. Critical path: it cannot be parallelised on the cluster and it carries the headline claim.
6. Figures, draft (§1–§6), co-author review round.

Known per-trial wall clock at 250 epochs (local, 4 workers): trial-3650 ≈ 46 min; trial-3408 ≈ 17.5 h; trial-2353 / trial-4223 not yet measured. RQ1 worst case 389 s with `HOST-Release`.

### Cut order (scope-reduction ladder)

Still the agreed ladder. Cut in this order:

1. Drop RP2040 portability run.
2. Drop RQ5 noise-augmentation.
3. Drop RQ3 new-gesture (depends on Florian's dataset support).
4. Retreat to a short paper: RQ1 + RQ2 + RQ4 only.

### Open blockers

- [ ] **Venue + deadline** — sets the cut level and whether the COI process below applies.
- [ ] **F9 MCU-build blocker** — upstream `MemProfile` pthread dependency breaks every cross-compile at pin `7d7f1d5`. Decision 2026-09-05: fix upstream on `develop`, then re-pin. The HOST sweeps do not need it, so stage 1 runs at `7d7f1d5` meanwhile.
- [ ] **RP2350 bring-up** — no on-device run exists; the one-sentence claim depends on it.
- [ ] **Dataset availability statement** — usage of Florian's dataset for this paper is cleared (2026-09-05). Still needed for the paper: how it is cited (DOI / Hettstedt 2026 / institutional contact) and whether the preprocessed windows can be published or are "available on request".
- [ ] **RQ3 new-gesture construction** — confirm with Florian whether a held-out class is acceptable, or whether an unreleased class exists.
- [ ] **COI declaration** to Gregor — required if ITEM (any year) is the venue.
- [x] ~~Pre-trained base-model checkpoint from Florian~~ — resolved differently: stage 1 retrains Florian's four DepthwiseCNN configurations inside ODT from his trial datasets (`data/model_and_dataset/trial-*`, gitignored); the rebuilt PyTorch model reproduces his reference confusion matrices exactly (`tools/verify_reference.py`).
- [x] ~~Conv1d / LayerNorm / fine-tune hook parity~~ — Conv1d, pooling, Dropout, LayerNorm, GroupNorm and PPCA replay are all in upstream ODT at pin `7d7f1d5`; V1/V2 parity at float32 precision on all four architectures.

### Original 22-day sprint plan (historical, April 2026)

<details>
<summary>Sprint table as of 2026-04-27 (D+6) — kept for the record</summary>

| Window | Days | Calendar | Work | Status then |
|---|---|---|---|---|
| W1 | D1–D2 | 22.–23.04 | Scope-freeze with Florian; dataset spec + base-model checkpoint; architecture decision | 🟡 needs status check |
| W1 | D3–D5 | 24.–26.04 | ODT extensions: Conv1d, LayerNorm, fine-tune hook + PyTorch parity tests | 🟡 needs status check |
| W1–W2 | D6–D7 | 27.–28.04 | RP2350 bring-up; inference-only validation | 🔵 in progress |
| W2 | D8–D9 | 29.–30.04 | **RQ4 stability** — ≥ 3 seeds, one user, noise-floor reference | ⚪ pending |
| W2 | D10–D12 | 01.–03.05 | **RQ1 replay-buffer sweep** (Optuna on supercomputer; top-3 verified on RP2350) | ⚪ pending |
| W2 | D13–D14 | 04.–05.05 | **RQ2 unseen-user LOOCV** sample-count sweep | ⚪ pending |
| W3 | D15 | 06.05 | **RQ3 new-gesture** experiment | ⚪ pending |
| W3 | D16 | 07.05 | **RQ5 noise-augmentation** ablation | ⚪ pending |
| W3 | D17 | 08.05 | RP2040 companion run (1 seed, reduced matrix) | ⚪ pending |
| W3 | D18–D19 | 09.–10.05 | Draft v1 (§1 Intro, §2 Related Work, §3 System, §4 Results, §5 Discussion, §6 Forthcoming) | ⚪ pending |
| W3 | D20 | 11.05 | Send to Florian (fact-check) + Gregor (scientific review, COI declared) | ⚪ pending |
| W3 | D21 | 12.05 | Reviews → v2 draft; LNCS formatting; single-blind scrub; ≤ 12 pages | ⚪ pending |
| W3 | D22 | **13.05** | Final proofread; submit OR hand pre-approved PDF to co-author for upload before 2026-06-05 | ⚪ pending |

What actually happened: the harness (D6–D7) landed on schedule; the ODT extensions (D3–D5) arrived upstream in July instead of April; stage 1 (DepthwiseCNN retrain + gates) and the RQ1 binary were built in July; nothing after that ran.

</details>

---

## TL;DR

On a real edge-MCU (**RP2350** primary, **RP2040** companion) deployment of the SmaTable surface-vibration gesture **classifier** — the rest of the SmaTable pipeline (event-detection, analog filtering) stays off-device in this paper's scope — we quantify:

1. how many replay-buffer samples from initial training are needed to prevent catastrophic forgetting during on-device fine-tuning,
2. how many new-user samples are needed for effective personalisation,
3. how many samples from how many contributors are needed to add a new gesture class,
4. how stable the resulting on-device training is across seeds, and
5. whether simple additive-noise augmentation reduces the sample requirements of (1)–(3).

**Experimental strategy.** Optuna orchestrates a massively-parallel sweep on a supercomputer, where each trial builds, links, and executes the *same C training code that targets the MCU* — trials differ from the on-device runs only in wall-clock and parallelism, not in numerics. Selected configurations are re-verified on the actual RP2350 to confirm prediction parity with the server runs.

**No QPU, no HW-NAS, no distillation.** Framework work (Conv1d, LayerNorm, GroupNorm, PPCA replay — all in upstream ODT at the pinned commit) is reported as an enabler, not as a contribution.

---

## One-sentence claim

> We report the **first on-RP2350 measurements of both inference and training** of the SmaTable gesture classifier — all prior SmaTable work has kept both stages on a PC — and, on that deployment, quantify the sample budgets and stability envelopes needed for the three continual-learning scenarios the SmaTable application actually requires: user personalisation, new-gesture addition, and forgetting control via replay.

---

## Why this paper, why now

The SmaTable deployment exposes three continual-learning problems that cannot be answered from the literature alone:

- how much data does a user-adaptation need,
- how much does a new gesture need, and
- how big must the replay buffer be to keep the original users working.

The immediate prior work is Hettstedt et al. (2026), which establishes the data-acquisition chain (RP2350 + 9-channel piezo AFE, 4-of-9 channels used, 1 kHz sampling, 15 users × 10 sessions × 10 repetitions × 6 gestures = 9 000 events) and a 1D-SepCNN classifier at 8 722 parameters reaching 97.32 % 5-fold top-1 **offline on a PC**. Despite the paper's "end-to-end" title, **only the DAQ stage actually runs on the RP2350** — inference and training both stay on the PC.

This work therefore simultaneously delivers:

- the **first on-RP2350 inference measurement** for this system,
- the **first on-RP2350 training measurement** for this system, and
- sample-budget / stability answers to the CL questions Hettstedt 2026 does not address.

Earlier prior work on the SmaTable chain (Yoshida 2023a/b, NAIST) is purely PC-class. Yoshida 2023b reaches 0.90 accuracy under LOPOCV + one calibration session, but the calibration data has to leave the device for PC training, contradicting the privacy-by-hidden-sensor motivation of the SmaTable concept.

---

## Contribution claims

Five concrete claims, each testable and each able to survive the others being weakened:

1. **Replay sizing curve (RQ1).** For the SmaTable gesture classifier, accuracy-on-original-users as a function of per-class replay memory, under three replay mechanisms at matched bytes per class — random exemplars, herding exemplars (iCaRL-style), and PPCA generative replay (upstream #326) — plus a no-replay floor, measured on-device on RP2350. *(PPCA arm added 2026-07; the April claim named random + herding only.)*
2. **User-adaptation sample curve (RQ2).** For an unseen user, accuracy on that user as a function of the number of new-user samples used in on-device fine-tuning, under leave-one-user-out cross-validation. Reports both target-user gain and original-users retention.
3. **New-gesture sample curve (RQ3).** For one newly added gesture class, accuracy on the new class as a function of (samples per contributor × number of contributors), while bounding degradation on the pre-existing classes.
4. **On-device training stability (RQ4).** Run-to-run variance of final accuracy across ≥ 3 seeds, decomposed into contributions from new-parameter initialisation and on-device data order.
5. **Augmentation effect (RQ5, light-touch).** Whether additive Gaussian noise on the vibration samples reduces the sample budgets measured in RQ1 / RQ2 at matched accuracy.

### Non-contributions made explicit in the paper

- Framework additions (Conv1d, LayerNorm, GroupNorm, PPCA replay in upstream ODT) are engineering enablers, not claims.
- Model architecture is fixed: Florian's DepthwiseCNN configurations (trial-2353 / 3408 / 3650 / 4223), retrained in ODT. HW-NAS is future work.
- QPU is absent. A dedicated QPU paper follows once QPU integration lands in ODT.
- The analytical memory cost model stays a separate paper (DATE / ACM TECS track).
- The SmaTable data-acquisition pipeline is reused unchanged from Hettstedt 2026; only what happens *after* event-detection (the classifier and its on-device fine-tuning) is extended.

---

## Research questions

**RQ1 — Replay sizing (existing-user retention).**
How much per-class replay memory from the original pre-training set must be retained to prevent catastrophic forgetting during on-device fine-tuning to a new user? Random vs. herding exemplar selection vs. PPCA generative replay, at iso-byte memory per class.

**RQ2 — Fine-tuning sample efficiency (unseen user).**
How many labelled samples from a previously-unseen user are required before on-device fine-tuning recovers accuracy on that user under leave-one-user-out CV?

**RQ3 — Fine-tuning sample efficiency (new gesture).**
When one new gesture class is added, how many samples from how many contributors are required before the new class reaches a stated accuracy floor, while pre-existing classes do not drop below a stated tolerance?

**RQ4 — On-device training stability.**
Across ≥ 3 seeds, what is the run-to-run variance of final accuracy for the fine-tuning procedure on RP2350? Is the variance dominated by new-parameter initialisation, by on-device data order, or by both?

**RQ5 — Sample efficiency via noise augmentation.**
Does additive Gaussian noise on the SmaTable vibration samples reduce the replay-buffer size (RQ1) or the new-user sample count (RQ2) at matched accuracy?

---

## Scope

### In scope

- Fixed model architecture: Florian's DepthwiseCNN, the four configurations from his search.
- Replay sweep at iso-byte budgets (random / herding exemplars, PPCA generative replay, no-replay floor).
- Leave-one-user-out CV for unseen-user fine-tuning (RQ2).
- New-gesture experiment (one added class, RQ3).
- Noise-only augmentation ablation (RQ5).
- Training-stability measurement across ≥ 3 seeds (RQ4).
- Optuna-driven hyperparameter search on a supercomputer, executing the same MCU-targeted C code in parallel.
- Framework additions (GroupNorm; Conv1d / LayerNorm / PPCA from upstream) as enablers.
- Hardware: RP2350 (primary), RP2040 (companion / portability check).

### Out of scope (this paper)

- HW-NAS inside the Optuna loop. Architecture is fixed.
- **QPU** — not yet integrated; forthcoming paper.
- Amplitude-scaling and PCA-based augmentation (RQ5 is noise-only).
- Environmental-noise–induced drift / adaptation — standalone paper.
- Distillation, EWC, LwF, NCM — replay-only in v3.
- STM32H743 Cortex-M7 portability run.
- Analytical memory cost model — separate DATE / TECS paper.
- Multi-domain sequences: RQ1 measures exactly one domain transition (original users → one new user), so PPCA's constant-memory-across-domains property is not exercised (see `experiments/rq1-replay-buffer/README.md`).

### Forthcoming (explicit roadmap section in the paper body)

- Joint HW-NAS + CL search with a simple analytical memory / MAC cost model.
- QPU variant of the same experimental matrix, once QPU integration lands.
- Environmental-noise adaptation — standalone paper.
- Richer replay-selection strategies (gradient-, uncertainty-, loss-based).
- Additional CL methods (EWC-diag, LwF, NCM head).
- Sequential multi-user domain chain to exercise PPCA's memory advantage.

---

## System setup

**Framework.** [`es-ude/OnDeviceTraining`](https://github.com/es-ude/OnDeviceTraining) pinned at upstream `main` `7d7f1d5` (2026-07-15): Conv1d (grouped → depthwise), MaxPool1d / AdaptiveAvgPool1d, Dropout, LayerNorm, GroupNorm, Linear, ReLU, Softmax fused with CE, SGD+momentum, npy state-dict checkpoints (`modelLoadStateDict`), PPCA generative replay + exemplar buffer (#326). No local fork. Everything runs FLOAT32.

**Hardware.**

| Role | Board | SoC | Memory | Notes |
|---|---|---|---|---|
| Primary | RP2350 | Dual Cortex-M33 + dual Hazard3 RISC-V | 520 KB SRAM | All primary RQ measurements; FPU on M33 |
| Companion | RP2040 (Pico 1) | Dual Cortex-M0+, no FPU | 264 KB SRAM | Portability check within the RP family; soft-float; bench setup ordered (expected ~2026-09-12) |
| ~~Dropped~~ | ~~STM32H743~~ | ~~Cortex-M7~~ | — | Removed to buy budget for RQ3 + RQ5 |

Measured feasibility (stage-1 host-side memory probes, 2026-07): trial-3650 / 2353 / 4223 fit the RP2350 with headroom; trial-3408 — the most accurate configuration (0.904 reference) — needs ~1953 KiB (heap + `executeOp` VLA stack) and does not fit. This is itself a deployment result the paper can use.

**Optimiser.** SGD+momentum with cosine LR schedule and best-epoch snapshot, FP32. No Adam (two extra state buffers per parameter rule it out on the MCU), **no QPU.**

**Datasets.** Florian Hettstedt's SmaTable gesture dataset: 15 subjects, 6 gestures, 8 999 preprocessed windows of 4 channels (channels 1, 4, 6, 8); window length T = 1250 / 1250 / 250 / 625 samples for trial-2353 / 3408 / 3650 / 4223. Prepared per trial by `tools/prep_smatable.py` into LOSO (15 folds) and AOS splits under `data/smatable-trial-<id>/` (gitignored). Label map: knock 0, tap 1, swipe-down 2, swipe-up 3, swipe-left 4, swipe-right 5. No new data collection.

**Baselines.**

| ID | Description |
|---|---|
| B0 | PyTorch FP32 pre-trained, off-device. Upper bound. |
| B1 | Pre-trained, no on-device fine-tuning. Lower bound. |
| B2 | Naïve full-model on-device fine-tuning (no replay). |
| B3 | Last-layer-only on-device fine-tuning. |
| B4 | Full-model fine-tuning **with replay buffer** — primary v3 method for RQ1 / RQ2. |

B0 reference LOSO means (PyTorch, Adam, best epoch on the left-out subject): trial-3408 0.904 (3050 params), trial-4223 0.811 (694), trial-3650 0.705 (434), trial-2353 0.618 (234). The ODT stage-1 retrain uses SGD+momentum, so its LR is re-searched; the V3 gate is "within 3 pp of B0 on the 15-fold mean".

---

## Metrics

| Metric | Instrument | Units | Serves |
|---|---|---|---|
| Accuracy on original users vs. replay memory per class | On-device eval | % vs. bytes/class (and iso-exemplar count) | RQ1 |
| Accuracy on held-out user vs. new-user sample count | On-device eval (LOOCV) | % vs. K | RQ2 |
| Accuracy on new gesture vs. samples × contributors; accuracy retained on old classes | On-device eval | % | RQ3 |
| Run-to-run accuracy variance (≥ 3 seeds) | On-device eval | std-dev % | RQ4 |
| Variance decomposition: new-param init vs. data order | Controlled seeds | variance share | RQ4 |
| Accuracy at matched buffer size / sample count, noise-aug vs. no-aug | On-device eval | Δ % | RQ5 |
| Peak SRAM during fine-tuning | Host-side probes (`memory.json` per trial: `mem_mcu_total_b`, `stack_peak_b`) + on-device DWT / linker map | KB | context |
| Wall-clock per update step | DWT cycle counter | ms | context |
| Gradient-parity error (Conv1d, GroupNorm) on SmaTable tensors | PyTorch FP32 ref vs. ODT (`tools/compare_stage1_parity.py`) | max abs / rel | tooling correctness |

| Energy per inference / per fine-tune epoch | On-device power measurement bench (RP2350 setup nearly ready 2026-09-05) | mJ | context |

Energy was out of scope in the April plan; the measurement bench now exists, so it is reported if the numbers are in by the draft.

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ~~Florian's pre-trained base-model checkpoint not available~~ | — | — | Resolved: stage 1 retrains in ODT from his trial datasets |
| RP2350 bring-up takes longer than planned | Medium | **High** — carries the headline claim | Start it in parallel with the cluster sweeps; a HOST-only fallback weakens the claim to "MCU-bit-equivalent training" and must be stated as such |
| Replay curves come out non-monotonic / noisy | Medium | Medium | RQ4 establishes the noise floor (interim: RQ1's own seed axis); report curves with error bars; herding as the stable anchor |
| Optuna wall-clock blows up on the supercomputer | Low | High | Grids are small and measured (see "Current state"); trial-3408 is the outlier (17.5 h/trial) and has a reduced grid + extension complement |
| RQ3 dataset manipulation introduces artefacts | Medium | Medium | Construction declared explicitly; RQ3 is a secondary, non-load-bearing claim |
| Paper reads as "fine-tuning study" not "CL study" | Medium | Low | Title, intro, and RQ1 framing all foreground replay + forgetting control |
| COI at ITEM (if chosen) mishandled | Low | High | COI declared to Gregor before submission; submission-handler co-author designated |
| Schedule slips | Medium | Medium | Cut order is explicit; RQ1 + RQ2 + RQ4 are the irreducible core |

---

## Repository structure (actual, 2026-09)

```
.
├── docs/paper-plan.md          # this file: paper spec + maintained current state
├── docs/                       # audit findings, HPC cheatsheet, dated handoff notes
├── experiments/                # README + per-RQ design READMEs (analysis + figures land here later)
├── src/examples/               # one C file per binary: stage1_pretrain, rq0_toy_synthetic, rq1_replay_buffer, rq2..rq5 stubs
├── src/dataset/                # smatable_dataset_{npy,baked}.c behind src/include/smatable_dataset.h
├── hpc/                        # run_optuna.py, search_space/*.json, sbatch wrapper, container def, bin/ (built HOSTs, gitignored)
├── scripts/amplitude/          # numbered cluster scripts (env check, upload, setup, prep, L0/L1, container, smoke, inspect)
├── tools/                      # prep_smatable, verify_reference (V0), stage1_reference + compare_stage1_parity (V1/V2),
│                               # check_ckpt_roundtrip (V4), aggregate_stage1, stack_bound
├── tests/                      # L0 / L1 dataset-backend equivalence
├── data/                       # gitignored: model_and_dataset/trial-*, smatable-trial-*, smatable-real, smatable (synthetic)
└── runs/                       # gitignored: Optuna studies, per-trial checkpoints, selection JSONs
```

Not yet existing: `paper/` (LNCS sources), `analysis/` (figure notebooks), on-device flash / DWT-timing scripts.

---

## Conflict of interest

Applies if ITEM (any year) is the venue. Andreas Erbslöh and Gregor Schiele sit on the ITEM organising / advisory board. A UDE-authored paper at this workshop is not disqualifying — chairs routinely assign such papers to independent PC members — but the COI is declared in writing to Gregor before submission, acknowledged at submission via the venue's COI form, and reflected in review handling by an independent PC member.

---

## Citation

To be added on acceptance. Until then, please cite the working repository:

```bibtex
@misc{buron2026smatable_cl,
  author       = {Buron, Leo and Hettstedt, Florian and Erbsl{\"o}h, Andreas and Schiele, Gregor},
  title        = {{Towards Continual Learning for SmaTables: Replay-Buffer Sizing
                   and On-Device Fine-Tuning for Edge-MCU Gesture Recognition}},
  year         = {2026},
  howpublished = {Working repository, University of Duisburg-Essen, Intelligent Embedded Systems},
  note         = {In preparation; venue to be decided}
}
```

---

## License

To be decided jointly with co-authors before submission. Default intent: code under Apache-2.0; dataset access under institutional agreement.

---

## Contact

Leo Buron — Intelligent Embedded Systems, University of Duisburg-Essen.
