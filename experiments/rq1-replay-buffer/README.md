# RQ1 — Replay-buffer sizing (existing-user retention)

**Question.** How much replay does on-device fine-tuning to a new user need to avoid forgetting the original users, and does upstream's PPCA generative replay (#326) beat a raw exemplar buffer at the same memory budget?

**Design.** Two-domain, domain-incremental protocol over one LOSO fold (`src/examples/rq1_replay_buffer.c`): domain 0 = pooled original users (fold's `train` split), domain 1 = the held-out new user (fold's `test` split). Pretrain on domain-0-fit, snapshot domain-0-held accuracy (the BWT "before" baseline), build the replay source from domain-0-fit, fine-tune on domain-1-fit with replay injected per sample, then re-evaluate domain-0-held ("after" — the headline `accuracy` metric) and domain-1-held (`new_acc`, confirms fine-tuning worked). `bwt = accuracy_after − accuracy_before`, GEM's backward-transfer definition specialized to two domains.

**Sweep axes.**
- `replay_mode`: `ppca` (upstream #326 generative replay) / `exemplar_random` / `exemplar_herding` (iCaRL-style, greedy running-mean selection) / `none` (no-replay forgetting floor)
- `buffer_size`: one shared **iso-byte** replay budget per class across all arms. For the exemplar arms it's the buffer capacity directly; for `ppca` the binary derives a subspace rank `k = buffer_size − 1` from upstream's own per-class memory formula (`bytes_per_class ≈ (k+1)·d·b`) so all four arms spend roughly the same bytes/class at a given sweep point. The *actual* achieved parity (not the approximation) is logged every run via `ppcaReplayBytes`/`ppcaReplayIsoExemplarCount` (`bytes_per_class`, `iso_exemplar_count` in `trials.csv`) — verify iso-byte parity from that column, don't assume it from `buffer_size` alone.
- `fold`: all 15 LOSO folds (0–14)

**Measurement.** `accuracy` (domain-0-held, post-fine-tune — the Optuna objective) plus `new_acc`, `bwt`, `bytes_per_class`, `iso_exemplar_count` as logged trial attributes. Optuna drives the sweep on the supercomputer (`hpc/search_space/rq1_replay_buffer.json`, `GridSampler` over fold × replay_mode × buffer_size × seed = 900 trials); top-3 configs re-verified on real hardware. The `seed` axis (3 values) exists because RQ1 has no other noise-floor reference available yet — RQ4, which is meant to supply one, is still an unimplemented stub — so each (fold, replay_mode, buffer_size) cell gets 3 independent runs rather than 1, giving RQ1 its own in-situ estimate of run-to-run variance.

**PPCA rank sensitivity (secondary sweep).** `buffer_size` drives `ppca_rank` by the iso-byte formula above in the main grid, but `ODT_PPCA_RANK` independently overrides it (clamped to `[1, WF-1]`), so rank can be swept on its own — orthogonal to the buffer_size/replay_mode comparison, and answering a different question ("how sensitive is PPCA replay quality to subspace rank," not "does PPCA beat exemplars at matched memory"). Mixing it into the main grid would break iso-byte parity by design, so it's a separate study: `hpc/search_space/rq1_ppca_rank_sweep.json` (`fold × ppca_rank`, `replay_mode` fixed to `ppca`, 15×7 = 105 trials). `buffer_size` is left at the binary default (8) in this sweep and printed but not meaningful — read `ppca_rank`/`bytes_per_class`/`iso_exemplar_count` instead.

**PPCA absorption chunk-size sensitivity (secondary sweep).** `ODT_MAX_SESSION_SAMPLES` caps how many raw samples get absorbed into a class's PPCA generator per merge call (classes with more fit-samples are absorbed in multiple chunked calls); default is 64, down from an original 512 that turned out to be far larger than PPCA's incremental-merge design — or the actual RP2350 target's 520 KB SRAM — was ever meant to handle in one call. Like rank, it's orthogonal to the iso-byte comparison (doesn't change `bytes_per_class`, only how the generator is built), so it gets its own study: `hpc/search_space/rq1_chunk_size_sweep.json` (`fold × max_session_samples ∈ {16,32,64,128,256,512}`, `replay_mode` fixed to `ppca`, 15×6 = 90 trials). Answers two questions: the actual cost-vs-chunk-size curve, and whether chunking granularity measurably affects final replay quality (it shouldn't — the underlying Chan-Golub-LeVeque merge is exact regardless of chunk size — but this sweep verifies that rather than assuming it).

**Build and timeout requirements.** See `hpc/README.md`'s "RQ1: replay-buffer sweep" section — `HOST-Release` (not `HOST-Debug`) and `TRIAL_TIMEOUT_S=1200` are both required for real-dataset-scale runs; a pre-flight audit found the PPCA arm's absorption cost blows past a 300s timeout under an unoptimized build.

**FWT (forward transfer).** Not measured here. GEM's FWT is "zero-shot accuracy on the next domain before training on it, vs. a random-init reference" — in this protocol that's domain-1-held accuracy right after pretraining, before fine-tuning/replay ever touches the model. Structurally it can't depend on `replay_mode`/`buffer_size` (it's measured before either is exercised), so it would print as a per-fold constant across all 20 replay-mode×buffer-size trials — not useful as a comparison axis here. It belongs to RQ2 instead, where it's exactly the K=0 anchor point of the sample-count sweep; see that RQ's README.

**Status (2026-09-03).** Implemented and smoke-tested locally, never swept. Blocked on stage-1 checkpoints for the DepthwiseCNN port (see the model caveat below). RQ4 would supply an external noise floor for error-bar interpretation but is still a stub, hence the in-grid `seed` axis.

**Model caveat — provisional results.** `rq1_replay_buffer.c` uses the same placeholder Linear→ReLU→Linear→Softmax MLP as `rq0_toy_synthetic.c`, pending the per-RQ training-loop design sub-spec that would settle the real Conv1d/GroupNorm architecture (`stage1_pretrain.c`'s DepthwiseCNN). This sweep's buffer_size/replay_mode rankings are **provisional**, not a final result: a full re-run on the DepthwiseCNN architecture is required before any conclusion from this sweep is cited in the paper. Running it now on the placeholder MLP is still worthwhile — it validates the four-arm replay wiring end to end on real gesture data, and both replay mechanisms operate on raw sensor windows rather than architecture-internal features, so the *relative ranking* between arms is plausibly less architecture-sensitive than the absolute accuracy/bwt numbers — but the numbers themselves do not settle the final architecture question.

**Scope limitation — single domain transition.** This protocol measures forgetting across exactly one domain transition (original-users-pool → one new user), which cannot exercise PPCA's actual selling point: constant memory as the number of *absorbed* domains grows (see `OnDeviceTraining/src/docs/CONTINUAL_LEARNING.md`), a property that's invisible with only one prior domain. RQ1 answers a real but narrower question — "at one domain transition, does a k-rank PPCA reconstruction match `buffer_size` raw exemplars at iso-byte parity?" — not "does PPCA's memory advantage hold up across a long sequence of new users." None of RQ2/RQ3/RQ5 test a sequential multi-domain chain either; a small sequential-domain addendum is future work, not part of this sprint.

**Layout.**
```
rq1-replay-buffer/
├── README.md              (this file)
└── runs/                  gitignored — raw per-trial CSVs and state dumps

hpc/search_space/rq1_replay_buffer.json         full grid (900 trials, incl. seed axis)
hpc/search_space/rq1_replay_buffer_smoke.json   plumbing smoke test (1 fold, 2 buffer sizes)
hpc/search_space/rq1_ppca_rank_sweep.json       secondary: PPCA rank sensitivity (105 trials)
hpc/search_space/rq1_chunk_size_sweep.json      secondary: PPCA chunk-size sensitivity (90 trials)
src/examples/rq1_replay_buffer.c                the HOST binary itself
```

Run locally:
```bash
uv run hpc/run_optuna.py --rq rq1_replay_buffer \
    --host-bin hpc/bin/rq1_replay_buffer.host --data-dir data/smatable \
    --search-space hpc/search_space/rq1_replay_buffer_smoke.json \
    --log-dir runs/optuna-local --n-workers 4
```
