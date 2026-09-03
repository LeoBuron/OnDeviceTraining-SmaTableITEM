# ODT USERAPI Audit — Pass 1 Findings (Misc Bucket)

**Scope:** Bugs, inconsistencies, silent failures, and missing guardrails discovered while auditing ODT's USERAPI via the base-project repo (Plan 2 of the audit series). Primary abstraction critique (boilerplate-as-cost, missing factories, broadcasting) is reserved for Plan 2b / Plan 3.

**Version:** Stand Plan 2 Abschluss (Task 7). Commit hash: see `jj log` for this change — this doc is committed alongside the `phase5e-failures.md` appendix as the sole output of Task 7.
**Raw measurement artifacts:** `runs/audit1/` (gitignored). Only distilled findings appear below; numerical details are quoted inline with pointers back to the raw `h{1..5}_result.md` files.

---

## Summary

| # | Hypothesis / Finding | Verdict | Priority |
|---|---|---|---|
| F1 | CE-gradient lacks batch-size normalization (H1) | **CONFIRMED — bug** | High |
| F2 | DataLoader `indices[]` under-initialized (bonus, out of H1-H5 scope) | **CONFIRMED — bug** | High |
| F3 | Xavier-uniform formula (H2) | Refuted — negative result | — |
| F4 | FP32 MatMul accumulation drift (H4) | Refuted — negative result | — |
| F5 | DataLoader-shuffle semantics (H3) | Refuted — negative result | — |
| F6 | Loss-reduction unit mismatch (H5) | Refuted — plan hypothesis was wrong | Low (docs) |
| S1 | `inference` is implicitly per-sample | Abstraction gap | Medium (Plan 2b) |
| S2 | `DEBUG_MODE_ERROR` not propagated to ODT libs | UX / silent errors | Medium |
| S3 | `tensorInitFloat` retains `dims` pointer verbatim (dangling hazard) | Latent bug | Medium |
| S4 | `rngSetSeed(0) == rngSetSeed(1)` aliasing | Latent bug | Low |
| S5 | `freeTensor` aborts on user-owned buffers | Documented footgun | Low |
| S6 | `snprintf` between `init()` and `trainingRun()` → latent `exit(1)` | Latent memory bug | Medium |
| F7 | `tensorInit` INT32 path treats input as `float*` → destroys int32 inputs from NPYLoader | **CONFIRMED — bug** | High (silent) |
| F8 | `reserveInferenceStats` sizes the stats output from the label's rank → heap overflow (found 2026-07 via `ODT_MEM_PROFILE`) | **CONFIRMED — bug** | High (fatal under `ODT_MEM_PROFILE`) |
| F9 | `MemProfile` hard-requires pthreads (`find_package(Threads REQUIRED)` + unconditional `<pthread.h>`) → every MCU cross-build fails at configure (found 2026-09-03) | **CONFIRMED — build blocker** | High (blocks RP2350 bring-up) |

Two High-priority findings drive Plan 3 issue-filing: **F1 (CE-gradient 32× scaling)** — the primary divergence source behind the Phase-5e FAILs — and **F2 (DataLoader indices under-init)** — a surprise bonus bug surfaced during H3 that silently caps per-epoch unique samples at ~3% of the dataset on every ODT training run in every example in this repo.

**Status at the current vendored pin (upstream `main` `7d7f1d5`, 2026-07-15; table added 2026-09-03):**

| # | Status |
|---|---|
| F1 | **Fixed upstream** at `3e768c7`: the training loop applies `computeMeanScaleCE` (1/batch) via `scaleOptimizerGradients` when `backwardReduction == REDUCTION_MEAN`; the fused CE backward itself still emits raw `softmax − target`. |
| F2 | **Fixed upstream** at `3e768c7`: `indices[]` is allocated and filled to the full dataset size. |
| F7 | **Still open.** Workaround `readInt32NpyDirect` in `src/dataset/smatable_dataset_npy.c` stays until the INT32 `tensorInit` successor path is audited. |
| F8 | **Fixed upstream** at `7d7f1d5` (commit `61d9cb4`): the stats shape is now derived from the produced output. |
| F9 | **Open.** No MCU target configures at `7d7f1d5`; see the F9 section at the end of this document for the two-part cause and the fix options. |
| S1–S6 | Not re-audited after the factory-API migration; the MNIST examples they were found in no longer build at this pin. |

The Phase-5e FAILs in `docs/phase5e-failures.md` were measured under F1 + F2 and were not repeated after the fixes.

---

## Finding F7: tensorInit INT32 path destroys int32 inputs (High Priority, surfaced 2026-04-27)

**Kategorie:** Bug / Silent Numerical Semantics. Surfaced during HPC-harness L1
equivalence-test debugging (commit forthcoming).

**Was:** `OnDeviceTraining/src/src/userApi/tensor/TensorApi.c:57-71` —
`tensorInit(float *data, ...)` for `quantization->type == INT32` does:

```c
size_t size = 0;
for (size_t i = 0; i < numberOfDims; i++) size += dims[i];      /* sum, not product! */
int32_t *dataInt = *reserveMemory(size * sizeof(int32_t));
for (size_t i = 0; i < size; i++)
    dataInt[i] = (int32_t)data[i];                              /* float-cast int32 input! */
```

Two layered bugs in one function:

1. **Element count is `sum(dims)`, not `prod(dims)`.** For a `[N]` shape, sum
   == prod, so 1-D tensors happen to allocate correctly. For higher-rank
   tensors, the buffer is dramatically undersized.
2. **The input `data` is interpreted as `float*` and each element is
   `(int32_t)`-cast.** When `npyLoad` reads an `<i4` .npy and passes the raw
   int32 bytes through `tensorInit`, those bytes are reinterpreted as
   float32 first (small floats → 0 after int cast). For our int32 fold-ID
   files: every ID came back as 0, so every "train sample" resolved to global
   sample 0 — perfect silent corruption.

**Verifikation:** L1 backend-equivalence test
(`tests/dataset_backend_equiv.c`, 2026-04-27): with NPY backend going through
`npyLoad`-then-`tensorInit`, train[0] returned `x[0,0,0] = 0x1.f33f9cp-7`
instead of `x[90,0,0] = -0x1.06d25ap-11` (the value the baked backend
returned correctly from its compile-time integer slice).

Worked around in `src/dataset/smatable_dataset_npy.c` by reading int32 .npy
files directly via `openNPYFile + readHeader + fread` (function
`readInt32NpyDirect`), bypassing `tensorInit` entirely.

**Impact:**
- Any NPY-loaded int32 tensor in any ODT example is silently zeroed.
- Combined with F1 + F2, this completes the picture of why every existing
  example with int32 labels (none, in this repo — MNIST labels are float32
  one-hot which dodges the bug) would have produced meaningless training.
- The new HPC harness in this repo would have produced fake 100% accuracy
  results from constant-zero labels were it not for the L1 test catching it.

**Suggested remedy:**
- Fix the count: `size = calcNumberOfElementsByShape(...)`.
- Fix the cast: take `int32_t *` (or `void *`) and `memcpy` instead of
  per-element float-cast.
- Add a unit test that loads an int32 .npy and asserts ID values round-trip.

**Related:**
- Same surface as F2 (DataLoader silent under-init) — both produce silent
  fake training. F7 was caught by an L1-style cross-backend test that didn't
  exist before this repo's harness work.

---

## Finding F1: CE-gradient lacks batch-size normalization (High Priority)

> **Status 2026-09-03: fixed upstream at `3e768c7`** (see the status table in the Summary). The text below describes the pre-fix behaviour and the measurements that found it.

**Kategorie:** Bug / Silent Numerical Semantics.

**Was:** `crossEntropySoftmaxBackwardFloat` in `OnDeviceTraining/src/src/loss_functions/CrossEntropy.c:57-67` computes `lossGrad[i] = softmaxOutputFloat[i] - distributionFloat[i]` element-wise over all `batch × num_classes` entries with **no division by `batch_size`**. When the loss function is CE with SGD, the gradient magnitude passed to the optimizer is `batch_size` times PyTorch's `F.nll_loss(..., reduction='mean')`-driven gradient. Therefore the **effective learning rate at the optimizer is `batch_size × user_lr`**.

Note on the forward path: ODT's *live* training-loss path (`TrainingBatchDefault.c:18` → divide by `batch->size`; `TrainingEpochDefault.c:25` → divide by `numberOfBatches`) correctly produces a per-sample mean CE for reporting. So `train_loss` and `eval_loss` share units. The mismatch exists **only at the backward / gradient path**, not the forward-loss-reporting path. This is the mirror image of H5's original framing (see F6 below).

**Verifikation:** Task 2 — weight-aligned state-dump comparison at DEPTH=1 (784→32→10, BS=32). With bitwise-identical init weights and the same first 32 unshuffled MNIST samples driving both frameworks:

- ODT gradient vs PyTorch **sum**-reduction gradient: `max_abs_diff ≤ 1.00e-5` across all four tensors (`post_grad_w_0`, `post_grad_b_0`, `post_grad_w_1`, `post_grad_b_1`); `mean_abs_diff ≈ 6e-7` — numerically the same expression up to FP32 accumulation-order rounding.
- `(ODT × 1/32)` vs PyTorch **mean**-reduction gradient: `max_abs_diff ≤ 1.5e-8` — pure float32 rounding noise.

The factor is exactly `batch_size = 32`. Full numbers in `runs/audit1/h1_result.md`.

**Secondary validation:** Task 2 Step 5 ran PyTorch full-training at DEPTH=1 with `lr = 0.032 = 0.001 × 32`, testing the naive prediction "boost PyTorch LR by 32×, see accuracy drop toward ODT's ~89%". Actual result: PyTorch accuracy **rose** to ~96%. This **refuted the naive narrative** that effective-LR-mismatch alone explains the accuracy gap at DEPTH=1. The H1 *numerical* fingerprint (grad scale factor of exactly `batch_size`) holds rigorously; its *training-dynamics* consequence is depth-dependent. At DEPTH=1 the baseline `lr=0.001` is under-trained, so a 32× boost helps rather than hurts. At DEPTH=4+ (deeper nets) a 32×-too-large effective LR is plausibly destabilizing — consistent with the 3.6×-larger gap Phase-5e observed at 5 layers.

**Impact:**
- Users who tune LR by PyTorch convention (common in cross-framework ports) work with an effective LR that is `batch_size` times larger than they expect — 32× at the MNIST BS=32 setup.
- The Phase-5e depth effect (gap grows from ~3.3pp at 2 layers to ~11.8pp at 5 layers) is qualitatively consistent with backprop amplifying the gradient-scale error through additional layers.
- This finding is sufficient by itself to explain both Host-FAILs (`mlp_mnist_float32_host`, `mlp_mnist_stress_host`) and, separately, the MCU-FAIL divergence at BS=1 (where the factor degenerates to 1× and hides the bug).

**Suggested remedy:**
- **Option A (recommended):** divide by `batch_size` inside `crossEntropySoftmaxBackwardFloat`. Aligns user-supplied LR with PyTorch convention. Zero API surface change; one-line patch inside a private function.
- **Option B:** document the LR convention in `TrainingLoopApi.h` and leave the code. Less friendly for users porting from PyTorch; still a silent footgun.

**Related:**
- Existing ODT issues #60 and #61 are not directly related.
- F6 below (H5 refuted) is the mirror axis: the forward-loss path correctly divides, so `train_loss` is reported in per-sample units; only the backward-gradient path fails to divide.

---

## Finding F2: DataLoader `indices[]` under-initialized (High Priority, bonus)

> **Status 2026-09-03: fixed upstream at `3e768c7`** (see the status table in the Summary). The text below describes the pre-fix behaviour and the measurements that found it.

**Kategorie:** Bug / Silent Training Semantics. **Not part of the original H1-H5 hypothesis set**; surfaced during Task 5 (H3) close-reading of the DataLoader code path.

**Was:** Two layers of the DataLoader disagree about the size of the `indices[]` array, and the inner layer fills fewer entries than the outer layer allocates:

- `OnDeviceTraining/src/src/userApi/data_loader/DataLoaderApi.c:28-29` — allocates `indices` with `numberOfIndices = getDatasetSize()` entries (60000 for the full MNIST train set; `calloc`-zeroed).
- `OnDeviceTraining/src/src/data_loader/DataLoader.c:37` — fills only the first `numberOfIndices = sizeDataset / batchSize` = 1875 entries with `0..1874`.
- `DataLoaderApi.c:54-57` (`getSampleByIndex`) then reads `indices[index]` for `index ∈ 0..59999` via `getBatch` (`sampleIndex + i` for `sampleIndex = batch_idx × batchSize`).

Consequence: `indices[1875..59999]` stays `calloc`-zeroed → every sample index ≥ 1875 resolves to sample 0.

**Consequence in training:** In a full-epoch pass with BS=32 over 60000 samples, the first ~58 batches (indices 0..1874) use distinct samples. **Every subsequent batch in every epoch reads sample 0 repeatedly.** Per-epoch unique samples: ~1875 of 60000 = 3% of the dataset.

The bug is silent because:

- **No crash:** sample 0 is valid MNIST data.
- **No obvious accuracy catastrophe:** MNIST is easy; 1875 samples still suffice to train a 2-layer MLP to ~87%.
- **No error log:** ODT's error-printing macro is silently compiled out at default DLEVEL (see S2 below).
- **Gradient on the repeats eventually goes to near-zero** once the model memorizes sample 0, so it degenerates to a no-op rather than introducing visible instability.

**Verifikation:** Direct source inspection, cross-read of `DataLoader.c:36-40` (shuffle-time loop filling `indices[0..sizeDataset/batchSize - 1]`) against `getSampleByIndex` and `getBatch` access patterns. Corroborated by the H3 experiment's puzzling shuffle-invariance: forcing shuffle off on both sides did not close the gap (ODT−PT: −2.36pp with shuffle, −2.84pp without) — partially explained by this bug, since shuffle only reorders the 1875 usable indices, so its impact on the gradient trajectory is capped. See `runs/audit1/h3_result.md` side-finding section for the full source-inspection notes and shuffle-run numbers.

**Impact:**
- **Every ODT training run in every example in this repo** (and presumably in other ODT users' codebases) silently trains on ~3% of the intended dataset.
- Cross-framework accuracy comparisons (PyTorch sees full 60k, ODT effectively sees 1875) are apples-to-oranges until this is fixed.
- Fixed-dataset accuracy ceilings are much lower than users expect. MNIST survives this (~87% on 2-layer MLP); harder datasets would not.
- Combined with F1 (CE-grad scale), this also partially explains why H3 (shuffle) looked like a dead-end: shuffle semantics genuinely don't matter much when only 1875 samples per epoch are ever reached.

**Suggested remedy:**
- **One-line fix:** change `DataLoader.c:37` to `numberOfIndices = sizeDataset;` so the shuffle loop fills all dataset indices.
- Alternative reading: if the intent at `DataLoader.c:37` was "drop last partial batch", the correct form would be `(sizeDataset / batchSize) * batchSize` — but since the shuffling loop at lines 38-40 treats these as sample indices `0..N-1`, the 1:1 fix is appropriate.
- Regression test candidate: after training, assert that the multiset of requested sample indices over one epoch covers at least `0.9 × dataset_size` unique values.

**Related:**
- Co-located with S4 (RNG seed-0/1 alias) and S2 (silent `PRINT_ERROR`) as a "DataLoader + error-reporting surface is undercovered-by-tests" cluster.
- Partial explanation for F5 (H3 refuted): shuffle-invariance is an expected consequence of this bug.

---

## Finding F3: H2 Xavier-uniform — refuted (verified)

**Kategorie:** Negative Result.

**Was:** ODT's `xavierUniform` in `OnDeviceTraining/src/src/arithmetic/Distributions.c:42-45` computes `limit = gain * sqrtf(6.0f / (fanIn + fanOut))`, identical to PyTorch's `nn.init.xavier_uniform_` default. Sampling statistics from `runs/audit1/dump_odt_d1/pre_w_0.npy` (25088 samples, fan_in=784, fan_out=32) match the theoretical limit `sqrt(6/(816)) ≈ 0.085749` within < 0.25% relative error on `|min|`, `|max|`, and `std` — two orders of magnitude below the 5% decision threshold in the plan.

**Relevance:** Excludes an a-priori plausible divergence source. Confirms the init implementation is numerically correct. No issue needed. Full statistics in `runs/audit1/h2_result.md`.

---

## Finding F4: H4 FP32 MatMul accumulation drift — refuted (verified)

**Kategorie:** Negative Result.

**Was:** Per-Linear-layer pre-ReLU activations compared between ODT and PyTorch with bitwise-identical init weights loaded via `--load-init-from <odt_dump>`. At DEPTH=4 (5 Linear layers, widths 784→32→32→32→32→10):

| Layer | shape | `max_abs_diff` |
|---|---|---|
| `pre_relu_0` | (32, 32) | 2.98e-7 |
| `pre_relu_1` | (32, 32) | 2.38e-7 |
| `pre_relu_2` | (32, 32) | 1.79e-7 |
| `pre_relu_3` | (32, 32) | 1.64e-7 |
| `pre_relu_4` | (32, 10) | 1.19e-7 |

All layer activations agree within FP32 machine epsilon (~1.19e-7). Diffs **decrease** with layer depth (not grow), ruling out accumulating rounding drift as a secondary divergence source.

**Relevance:** Excludes FP32 rounding-order divergence as a contributor. H1 (F1) alone explains the numerical divergence. See `runs/audit1/h4_result.md` for the workaround needed to produce these dumps (per-sample inference loop; ODT lacks batch-forward, see S1).

---

## Finding F5: H3 DataLoader-shuffle — refuted (dominated by F2)

**Kategorie:** Negative Result.

**Was:** Forcing `shuffle=False` on both ODT and PyTorch at DEPTH=1, 10 epochs, lr=0.001, BS=32, produced:

| mode | PyTorch (N=5 seeds) | ODT (N=1, seed=42) | gap (ODT − PT) |
|---|---|---|---|
| shuffle | 89.69% ± 0.16 | 87.33% | −2.36 pp |
| no-shuffle | 89.67% ± 0.12 | 86.83% | −2.84 pp |

PyTorch is effectively shuffle-invariant (Δ = 0.018pp). ODT's own run is also nearly shuffle-invariant (Δ = 0.50pp, mostly seed noise). The gap **widened slightly** when shuffle was forced off on both sides — it did not shrink.

**Relevance:** Shuffle semantics are not a meaningful divergence source. Corroborated by F2 (the DataLoader `indices[]` bug), which makes ODT's shuffle largely a no-op anyway — shuffle only reorders the 1875 effectively-usable indices per epoch, so its effect on the gradient trajectory is capped by the bug. See `runs/audit1/h3_result.md`.

---

## Finding F6: H5 Loss-reduction units — refuted (plan hypothesis was wrong)

**Kategorie:** Negative Result + Plan Correction.

**Was:** The plan hypothesized that ODT's reported `train_loss` was batch-summed (per-batch CE summed over the batch, only divided by `numberOfBatches`), which would give `train_loss ≈ batch_size × eval_loss` — a clean 32× mismatch at BS=32. The plan derived this from `TrainingEpochDefault.c:25` alone (`totalLoss / numberOfBatches`).

Source inspection at the **live** training path showed two divisions:

- `OnDeviceTraining/src/src/userApi/training_loop/training_batch/TrainingBatchDefault.c:18` — `return totalLoss / (float)batch->size;` (per-sample mean within each batch).
- `OnDeviceTraining/src/src/userApi/training_loop/training_epoch/TrainingEpochDefault.c:25` — `return totalLoss / (float)numberOfBatches;` (mean over batches).

Net effect: the reported `train_loss` **IS a per-sample mean CE**, same units as `eval_loss` (which uses the structurally-identical `evaluationBatch` / `evaluationEpoch` pair). **No unit mismatch.**

The plan's hypothesis was fed by two things the plan author missed:

1. `TrainingBatchDefault.c` (the file with the second divisor) was not on the "read as reference" list in the plan's File Structure section.
2. The state-dump branch in `mlp_mnist_depth_sweep_host.c` exposes a raw batch-summed CE (`loss_sum` artifact — the code path was written for H1 analysis and bypasses `trainingBatchDefault`). The plan author confused this offline-dump code path with the live-training reporting path.

**Numerical refutation:** `runs/audit1/odt_d1_noshuffle.csv` shows `eval_loss / train_loss ≈ 37–57×` across epochs and **drifts upward** with training (epoch 1 ratio = 57, epoch 10 ratio = 40). A true fixed-unit mismatch would produce a constant 32×. The observed pattern is consistent with normal ML dynamics: `train_loss` is a during-epoch running average over 1875 mutating-weight batches (average picks up the late-batch ultra-low values as weights converge), while `eval_loss` is a single post-epoch full-pass measurement with a fixed (for that epoch) weight state. Generalization gap plus measurement-window difference — no bug.

**Relevance:** The plot disparity at `src/examples/plots/01_training_curves_*.png` (train ~0.01 vs eval ~0.3) is normal ML behavior, not a reporting bug. No code remedy needed. See `runs/audit1/h5_result.md` for the full source trace and CSV analysis.

**Docs-clarity suggestion (minor):** a one-line note in `TrainingLoopApi.h` documenting that `finalTrainLoss` is a "during-epoch mean-of-batch-means" (vs post-epoch full-pass) would prevent similar confusion in future audits.

---

## Side-findings (not primary H1-H5 scope, surfaced during audit)

### S1: `inference` is implicitly per-sample (missing batch-forward)

`OnDeviceTraining/src/src/arithmetic/Arithmetic.c` `doDimensionsMatch` (line 29) enforces strict shape equality — no broadcasting. A bias of shape `{1, 32}` cannot be `addFloat32TensorsInplace`'d to a full-batch output of shape `{32, 32}`. Therefore `inference(model, 2*k+1, {32, 784})` fails at the first Linear+bias step with PRINT_ERROR "Dimensions don't match". Task 4 implementer hit this while attempting batch-forward for pre-ReLU dumps and worked around it with a per-sample loop (5 × 32 = 160 single-sample `inference` calls per batch at DEPTH=4).

Additionally, `Softmax.c:32-48` normalizes globally over all tensor elements rather than per-row — so even if broadcasting were added, batch-softmax would need a separate fix for per-row normalization.

**Remedies:** (a) add `{1, M}`-broadcasting to `addFloat32TensorsInplace` and per-row semantics to `softmaxFloat`, OR (b) document the per-sample-only contract in `InferenceApi.h`. **Candidate for Plan 2b (abstractions audit).**

### S2: `DEBUG_MODE_ERROR` defined on HOST target, not propagated to ODT library

`cmake/host/host_post.cmake:12-14` sets `target_compile_definitions(HOST PRIVATE DEBUG_MODE_ERROR)` on the application target, but ODT's own `Common__hdrs` interface target does NOT inherit it. `PRINT_ERROR` inside ODT library code is therefore compiled to no-op at the default DLEVEL (0). **Runtime errors inside ODT code are silently compiled away.** Task 4 implementer had to explicitly reconfigure with `-DDEBUG_MODE_ERROR=ON` at the CMake-level to get any error output while diagnosing the broadcasting failure above.

**Remedy:** set the compile-definition on the relevant ODT-internal targets (or on `Common__hdrs` as a PUBLIC definition) so `PRINT_ERROR` fires for any configured ODT-host user regardless of their application-target settings. Or change the DLEVEL default. Either way, the current state (errors are silent unless the user already knows to add a magic CMake flag) is a diagnosis-blocker for every user triaging an ODT runtime bug.

### S3: `tensorInitFloat` / `setShape` retain `dims` pointer verbatim (dangling hazard)

The tensor-API functions in `OnDeviceTraining/src/src/userApi/tensor/TensorApi.c` (`tensorInitFloat` / `setShape`) store the user's `size_t *dims` pointer without copying. If the user passes a stack-local `size_t bd[] = {1, N}`, the pointer dangles after the defining block exits. Observed failure mode during Task 1 implementation: a bias tensor dumped with shape `(13615674994029168, 32)` after the calling scope exited — classic dangling-pointer tell.

**Safe pattern:** allocate `dims` via `*reserveMemory(2 * sizeof(size_t))` in a long-lived arena, or in `main`'s scope for the lifetime of the tensor.

**Remedy:** either (a) document the ownership contract in `TensorApi.h` with a warning against stack-local `dims`, or (b) `memcpy` the `dims` array into tensor-owned storage inside `setShape`. Option (b) is safer; option (a) is the minimum-effort mitigation. Candidate for API hardening.

### S4: `rngSetSeed(0) == rngSetSeed(1)` aliasing

`OnDeviceTraining/src/src/arithmetic/RNG.c:44` has `rng.state = seed ? seed : 1;` — silently aliases seed=0 to seed=1. Observed during Plan 1.5 N=20 seed sweeps as duplicate-run rows; the Python harness `compare.py` dodges it via a hardcoded `seed + 1` offset when spawning ODT.

**Remedy:** either (a) reject seed=0 with `PRINT_ERROR` (useless today because of S2, but correct once S2 is fixed), or (b) scramble via `splitmix32(seed)` so all 32-bit seed values are valid. Low-impact but silent.

### S5: `freeTensor` aborts on user-owned buffers (documented footgun)

`freeTensor(t)` eventually calls `free(t->data)`, which aborts with a glibc/macOS heap-checker diagnostic if `t->data` is a user-owned stack or static array wrapped by `tensorInitFloat`. No ownership flag on the tensor — the framework assumes all tensor data is framework-allocated.

**Safe pattern:** never call `freeTensor` on user-buffer tensors; only call it on framework-allocated tensors (e.g. the return of `inference`).

**Remedy:** add an `ownsData` bool to `tensor_t`, set by the allocator, checked inside `freeTensor`. Candidate `enhancement` label.

### S6: `snprintf` between `init()` and `trainingRun()` causes latent `exit(1)` on macOS

Observed during Plan 1.5 development: a `snprintf` call (or `gmtime_r`) placed between `init()` and `trainingRun()` triggered a silent `exit(1)` at the start of the ODT training loop. Root cause not fully isolated; working hypothesis is a memory-corruption bug in an ODT static-init path whose symptom gets masked when `snprintf`/`gmtime_r` do NOT disturb stack or dyld layout in the "wrong" way.

**Safe pattern:** move all string-formatting / time calls out-of-process (the Python harness writes JSON sidecars externally; the C code keeps its runtime path purely numerical).

**Remedy:** a memory-sanitizer (ASan / MSan / UBSan) run over ODT static-init paths would surface this; the fix is almost certainly a buffer-overrun or uninitialized-read one-liner inside ODT. Candidate `bug` label; flag for mem-sanitizer pass.

---

## Prioritization for Plan 3 (issue-filing)

| Finding | Kategorie | Priority | Issue candidate | Labels |
|---|---|---|---|---|
| **F1 (CE-gradient)** | Bug | **High** | Yes — 1 issue, reproducer via weight-aligned state-dump | `bug` |
| **F2 (DataLoader indices)** | Bug | **High** | Yes — 1 issue, reproducer: unique-sample-count assertion over full epoch | `bug` |
| F3 (Xavier) | Negative | — | No issue | — |
| F4 (FP32 MatMul) | Negative | — | No issue | — |
| F5 (Shuffle) | Negative | — | No issue (dominated by F2) | — |
| F6 (Loss-units / plan-miss) | Negative + docs | Low | Optional — 1 minor docs-clarity issue for `TrainingLoopApi.h` | `documentation` |
| S1 (inference per-sample) | Abstractions | Medium | Plan 2b scope | Plan 2b |
| S2 (`DEBUG_MODE_ERROR` wiring) | UX | Medium | 1 issue | `enhancement` |
| S3 (`dims` dangling) | Bug (latent) | Medium | 1 issue | `bug` |
| S4 (seed=0/1 alias) | Bug | Low | 1 issue | `bug` |
| S5 (`freeTensor` abort) | Bug (documented) | Low | 1 issue — add `ownsData` flag | `enhancement` |
| S6 (`snprintf` latent) | Bug (latent) | Medium | 1 issue — flag for mem-sanitizer pass | `bug` |

**Anmerkung:** The missing-abstractions inventory (LinearLayer factory, batch-forward, per-row softmax, ergonomic model-building) is cataloged in `odt-userapi-abstractions.md` (Plan 2b / Plan 3 scope). Pass 1 captured only the numerical divergences (F1-F6) + secondary bug catch (F2) + side-findings (S1-S6).

---

## References

- Raw measurement artifacts: `runs/audit1/h{1..5}_result.md`, dumps in `runs/audit1/dump_{odt,pt}_d{0,1,4}[_sameinit]/`, training CSVs `runs/audit1/odt_d1_{shuffle_baseline,noshuffle}.csv`.
- Plan: `docs/superpowers/plans/2026-04-22-userapi-audit-pass1.md` (gitignored local draft).
- Plan-1.5 failure context and gotchas: `docs/phase5e-failures.md` (Plan-2-outcome section at the end references this doc).
- Reproducer binary: `src/examples/mlp_mnist_depth_sweep_host.c` (parametrized by `-DDEPTH_SWEEP_HIDDEN_LAYERS={0,1,4}`; shuffle toggled by env `ODT_DISABLE_SHUFFLE`; single-batch dump via env `ODT_SINGLE_BATCH` + `ODT_STATE_DUMP_PATH`).
- Numerical compare harness: `src/examples/reference/state_dump_compare.py`, `src/examples/reference/depth_sweep_ref.py`.

## F8 — `reserveInferenceStats` sizes the output shape from the label's rank (heap overflow)

- **Found:** 2026-07-07, by the stage-1 memory instrumentation (the exact-size
  `ODT_MEM_PROFILE` allocator turns a silent overflow into a deterministic
  SIGTRAP before any output; ASan-confirmed).
- **Mechanism:** `userApi/InferenceApi.c` `reserveInferenceStats(label)`
  allocates `inferenceStats->output`'s shape arrays via
  `getShapeLike(label->shape)`; `inferenceWithLoss` later `copyShape()`s the
  model's actual output shape (rank >= 2 after Flatten+Linear) into it. A
  rank-1 label (`[NC]`) under-sizes `dimensions`/`orderOfDimensions` by one
  `size_t` each -> out-of-bounds write on every eval. Silent under plain
  calloc slack; fatal under `ODT_MEM_PROFILE`.
- **Repro:** any Flatten->Linear(->Softmax) model + rank-1 label +
  `inferenceWithLoss` (upstream's own kws_raw/har_classifier label shapes
  hit the silent variant).
- **Workaround (this repo):** rank-2 `[1, NC]` one-hot labels in
  `stage1_pretrain.c` `buildSplit` (matches `conventions/loss.md`
  `dimensions[0]=B`). Verified numerically transparent: bit-identical
  training anchor, V1/V2 PyTorch parity unchanged (4.47e-8 / <=1.26e-7).
- **Upstream fix suggestion:** size the copy destination from the SOURCE
  shape (`getShapeLike(output->shape)`) or fail fast on rank mismatch inside
  `copyShape`. Related doc divergence: `loss.md` calls B=1 outputs "[F]
  implicit" while the runtime always produces rank >= 2.

---

## F9 — `MemProfile` has no bare-metal guard: MCU configure fails under pico-sdk, and the source does not compile for newlib (surfaced 2026-09-03)

**Kategorie:** Build blocker / missing platform guard (upstream), masked by upstream CI.

**Was (two layers):**

1. *Configure.* `OnDeviceTraining/src/src/userApi/CMakeLists.txt:8-11`:

   ```cmake
   add_library(MemProfile MemProfile.c)
   find_package(Threads REQUIRED)
   target_link_libraries(MemProfile PRIVATE Threads::Threads)
   ```

   Under this repo's `PICO2_W` preset (pico-sdk toolchain, executable try-compiles) `FindThreads` links a probe, finds no `pthread_create`, and aborts the configure of *every* example:

   ```
   CMake Error at .../FindThreads.cmake:289 (find_package_handle_standard_args):
     Could NOT find Threads (missing: Threads_FOUND)
   Call Stack: OnDeviceTraining/src/src/userApi/CMakeLists.txt:10 (find_package)
   ```

   Reproduced with `cmake --preset PICO2_W -DODT_EXAMPLE=rq0_toy_synthetic` (an example that uses nothing from `MemProfile`). `PICO1` and the three STM32 presets share the toolchain and run into the same check.

2. *Compile.* `MemProfile.c` includes `<pthread.h>` unconditionally, calls `pthread_attr_init` / `pthread_attr_setstack` / `pthread_create` / `pthread_join` / `pthread_attr_destroy` (`MemProfile.c:34-55`) and reads `ru_maxrss` (`:84`). Under `arm-none-eabi` newlib none of these exist: five `-Werror=implicit-function-declaration` errors plus `'struct rusage' has no member named 'ru_maxrss'`. Reproduced with upstream's own `arm_cross` preset (`cmake -S OnDeviceTraining/src --preset arm_cross -B <dir>`, then `cmake --build <dir> --target MemProfile`).

**Why upstream CI does not catch it:** `cmake/arm-none-eabi.cmake` sets `CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY`, so `FindThreads`' "pthread_create in pthreads" probe compiles without linking and reports `Found Threads: TRUE` — the configure passes there. The `c-arm-cross-compile` job is marked non-required; whether it builds the `MemProfile` target at all was not checked here.

**Wirkung:** No MCU binary — including the L3 equivalence probe and every on-device measurement the paper claims — can be built at the current pin. This repo's top-level `CMakeLists.txt:95` also lists `MemProfile` in the shared `ODT_LIBS`, so even a passing configure would try to link it on MCU.

**Fix-Optionen:**
1. *Upstream (preferred — the pthread stack-paint is host-only by nature):* in `userApi/CMakeLists.txt`, `find_package(Threads)` without `REQUIRED` and build `MemProfile` only `if(Threads_FOUND AND NOT CMAKE_SYSTEM_NAME STREQUAL "Generic")` — or keep the target and give `MemProfile.c` an `#if defined(__unix__) || defined(__APPLE__)` host path plus a bare-metal path (`measurePeakStackBytes` → paint-and-scan of the current stack, or a no-op returning 0; `memProfileRssPeakKb` → 0). The heap counter (`StorageApi`, no pthread) stays available everywhere. Add `MemProfile` to the `arm_cross` build so CI actually exercises it.
2. *Repo-local (until the upstream fix is pinned):* drop `MemProfile` from `ODT_LIBS` on MCU platforms, mark the upstream target `EXCLUDE_FROM_ALL` there, ship a `FindThreads.cmake` shim on `CMAKE_MODULE_PATH` for the pico/stm32 presets that defines an empty `Threads::Threads` INTERFACE target, and `#ifdef`-guard the `measurePeakStackBytes` / `memProfileRssPeakKb` call sites in `stage1_pretrain.c`.

**Status:** open. Tracked in `docs/paper-plan.md` "Current state" as the first item of the RP2350 bring-up.
