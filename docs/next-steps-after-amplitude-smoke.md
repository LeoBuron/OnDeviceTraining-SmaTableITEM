# Next steps after the Amplitude smoke test

**Written:** 2026-04-27 (D+6), end of session.
**Reader:** future Claude session with cleared context, OR Leo himself a few days later.
**State at handoff:** End-to-end Amplitude pipeline green. paper0 bookmark on GitHub at `4f48726` (or whatever tip is current). All 8 amplitude scripts (00, 10, 11, 20, 21, 22, 40, 50) pass; sbatch job 685737 completed in 1m45s with 21 trials and best `accuracy=1.0000` on the synthetic toy. Numerics bit-identical between local Mac (Apple Silicon Clang) and Amplitude (x86_64 Sapphire Rapids GCC 13.3.0).

Cross-references:
- Spec: `docs/superpowers/specs/2026-04-27-hpc-experiment-harness-design.md` (gitignored, local only)
- HPC cheatsheet: `docs/amplitude-cheatsheet.md`
- HPC operator README: `hpc/README.md`
- Audit findings (F1, F2, F7): `docs/odt-userapi-findings-misc.md`
- Sprint plan: `README.md`, "Sprint progress (22-day plan)"

---

## Context recap (one paragraph)

The HPC harness — single-source ODT C code compiled HOST or MCU, dataset behind one C ABI with NPY (host) / baked-header (MCU) backends, Optuna driver in Python that subprocesses the HOST binary and parses a `RESULT` line, sbatch+ws_allocate+Lustre on Amplitude — is **fully operational** for the toy synthetic dataset. The real RQ implementations (rq1..rq5) are still stubs that print `RESULT skipped`; they wait on upstream-ODT W1 D3-D5 work (Conv1d + LayerNorm + fine-tune hook). The harness itself does not need to change for those.

---

## Track A — scale-up correctness ✅ DONE (2026-04-28 D+7 morning)

Picked the SQLite-on-tmpfs option. Trade-off rationale: workers all live on a single Slurm-allocated node (`mp.Pool` in `run_optuna.py`), so the actual problem the JournalFileBackend hit was Lustre's POSIX advisory locks (~100 ms per acquire via the cluster lock manager), not the storage protocol itself. Tmpfs locks are kernel-local (~µs), which subsumes most of what Redis would have bought us. Optuna's RDBStorage is the canonical default and `optuna-dashboard` reads it natively for live monitoring. Redis-fallback is available as a 5-line swap if a future RQ saturates SQLite's single-writer queue.

**Sweep results (USE_CONTAINER=1, rq0_toy_synthetic, sbatch 686949–686952):**

| N_WORKERS | walltime (`all workers joined`) | trials | lock warnings |
|-----------|---------------------------------|--------|---------------|
| 4         | 4.0 s                           | 20     | 0             |
| 16        | 2.7 s                           | 31     | 0             |
| 32        | 3.1 s                           | 63     | 0             |
| 64        | 4.7 s                           | 110    | 0             |

Baseline 685747 (JournalFileBackend, 32 W, 54 trials): 241 s with dozens of "lock taking >10s" warnings. Same N_WORKERS=32 on the new backend: **3.1 s — roughly 78× faster**. Best trial #0 value=1.0000 with `{fold: 0, lr: 0.01, epochs: 3, hidden: 16, seed: 42}` is **bit-identical** to the 685747 baseline — GridSampler reproducibility intact under the new backend.

**Implementation summary:**
- `hpc/run_optuna.py` — RDBStorage replaces JournalStorage; `_prep_sqlite_wal` flips `journal_mode=WAL` once at study creation; `--storage-url` CLI flag (default writes next to logs); WAL checkpoint(TRUNCATE) before exit so the post-run `cp` lands a consolidated DB.
- `hpc/run_optuna_amplitude.sh` — per-job `/tmp/optuna-${SLURM_JOB_ID}/` directory (kernel-local tmpfs) bind-mounted as `/optuna_db` inside the container. After `srun` returns, `cp` the consolidated `study.db` into `${LOG_DIR}` for archival.
- `--writable-tmpfs` would shadow the container's `/tmp`, so the SQLite mount goes to a dedicated bind path (`/optuna_db`) that survives the overlay.

**Bonus, bundled with the same container rebuild:** torch slim-down moved torch + nvidia-* CUDA wheels to `[dependency-groups.dev]` and made `run_container.def` use `uv sync --no-dev` + `uv run --no-dev` runscript → .sif dropped from 2.9 GB to 318 MB (~9.5× smaller, ~9× faster scp). [NOTE: this slim-down was reverted in a follow-up commit — image size was not actually a problem, and keeping torch in the runtime container preserves optionality for future audit / state-dump comparison runs that bring `src/examples/reference/*.py` workflows alongside the Optuna trials.]

**What's next on the harness front:** known-good-state. Real RQ sweeps (RQ1..RQ5 with 100+ grid points) are now coordination-bottleneck-free; they wait on upstream-ODT Conv1d + LayerNorm and Florian's pre-trained checkpoint, not on infra.

---

## Track B — Apptainer container build ✅ DONE (2026-04-27 D+6 evening)

End-to-end green: container-mode sbatch job 685747 ran 54 trials in 241s,
best trial #0 value=1.0000 — bitwise-equal to the no-container smoke 685737.
Lock-file contention warnings on the JournalFileBackend confirm Track A is
a real concern at N_WORKERS=32 (not theoretical).

**What ended up working** (different from the original plan below):

- **proxy URL fix** in `run_container.def` %post — yes, as planned.
- **uv installer** swapped to GitHub release tarball — yes, as planned.
- **apt mirror access**: turned out the proxy whitelist also blocks
  `archive.ubuntu.com` (403). Ubuntu mirror swap was NOT pursued — instead
  the container is now **built off-cluster on a Mac** via apptainer-in-docker
  (`docker run --privileged --platform linux/amd64 kaczmarj/apptainer
  build …`) and rsync'd via `gateway.amplitude.uni-due.de` (the dedicated
  transfer host accepts pubkey directly; the regular `amplitude` SSH alias
  has a `RemoteCommand` directive that conflicts with rsync).
- **bind-mount fix** in `hpc/run_optuna_amplitude.sh`: the .def's `%files`
  only stages `pyproject.toml` + `uv.lock`, not `run_optuna.py`. Mounting
  the script flat at `/app/run_optuna.py` keeps it editable without a 2.9 GB
  rebuild + transfer per Python edit.

**Future-Track-B housekeeping (optional, not blocking):**

- 2.9 GB .sif size is dominated by torch + nvidia-* CUDA wheels pulled by
  `uv sync`. The .def header explicitly says "CPU-only — drop torch", but
  torch sits as a runtime dep in `pyproject.toml`. Move it to
  `[dependency-groups.dev]` and switch the .def's `uv sync` to `uv sync
  --no-dev` → expect ~500 MB .sif (≈5× smaller, ≈5× faster scp).
- `hpc/README.md` should document the local-build-and-transfer pattern,
  since amplitude-side build is permanently blocked by the apt whitelist.

**Original plan (kept for reference; superseded above):**

`run_container.def` build fails on Amplitude login nodes with:
1. UDE proxy env vars miss the `http://` scheme prefix → apt-get rejects "Unsupported proxy".
2. Astral installer URL not on the proxy whitelist (same issue we hit for uv on the host).

**Action — both fixes in one edit to `hpc/run_container.def`:**

```
%post
    # (1) Normalize the proxy env to apt-friendly URL format.
    if [ -n "${http_proxy:-}" ] && [[ "$http_proxy" != http* ]]; then
        export http_proxy="http://${http_proxy}"
        export https_proxy="http://${https_proxy:-${http_proxy#http://}}"
        export HTTP_PROXY="$http_proxy"
        export HTTPS_PROXY="$https_proxy"
    fi

    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates curl git clang ninja-build cmake python3
    rm -rf /var/lib/apt/lists/*

    # (2) Replace astral.sh installer with the GitHub release tarball
    # (same trick as scripts/amplitude/20_setup.sh).
    UV_REL_URL="https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz"
    tmp="$(mktemp -d)"
    curl -fsSL "$UV_REL_URL" -o "${tmp}/uv.tar.gz"
    tar -xzf "${tmp}/uv.tar.gz" -C "${tmp}"
    cp "${tmp}/uv-x86_64-unknown-linux-gnu/uv"  /usr/local/bin/uv
    cp "${tmp}/uv-x86_64-unknown-linux-gnu/uvx" /usr/local/bin/uvx
    chmod +x /usr/local/bin/uv /usr/local/bin/uvx
    rm -rf "${tmp}"

    export UV_PYTHON_INSTALL_DIR=/opt/uv/python
    export UV_CACHE_DIR=/tmp/uv-cache
    export UV_PROJECT_ENVIRONMENT=/app/.venv
    cd /app
    uv python install
    uv sync
```

Then re-run `scripts/amplitude/30_build_container.sh`. If apt-get still complains: probably PyPI access is needed and PyPI isn't whitelisted either. Mitigation: copy `~/.local/lib/python.../site-packages` into the image as `%files`. Defer until/unless container is actually needed.

When fixed, set `USE_CONTAINER=1` in `40_smoke_sbatch.sh` calls and verify the same RESULT line shows up.

**Files:** only `hpc/run_container.def` + maybe `scripts/amplitude/30_build_container.sh` (already has --fakeroot fallback, should keep working).

**Estimate:** 30 min if PyPI is whitelisted, half a day otherwise. Not blocking — the no-container path works.

---

## Track C — real RQ implementations (gated by upstream ODT W1 D3-D5)

The 5 stubs in `src/examples/rq{1..5}_*.c` print `RESULT skipped reason="awaiting Conv1d/LayerNorm in upstream ODT (sprint W1 D3-D5)"`. Each needs to be replaced with a real training loop once upstream lands:

- **Conv1d**: `OnDeviceTraining/src/src/layer/Conv1d.c` is a 3-line skeleton. Sprint plan W1 D3-D5 fills it in.
- **LayerNorm**: doesn't exist in upstream at all. New layer + USERAPI wrapper + parity test.
- **Fine-tune hook**: API for freezing layers / training only the last layer (RQ2's last-layer-only finetune mode).

**Per-RQ implementation sketches** (from the spec, pending its own sub-spec when Conv1d lands):

| RQ | C file | Loop logic | Sweep axes |
|---|---|---|---|
| RQ1 | `rq1_replay_buffer.c` | pretrain on N-1 subjects → finetune to subject N with replay buffer of K samples per class from pretrain set | replay_buffer_size, selection_strategy (random vs herding) |
| RQ2 | `rq2_unseen_user.c` | pretrain on N-1 subjects → freeze backbone, train head only on K samples from held-out subject | n_new_user_samples (0..50), held-out subject (LOOCV) |
| RQ3 | `rq3_new_gesture.c` | pretrain on 5 gestures → grow head by 1 class → train on K_per_contributor × N_contributors samples of new class | K_per_contributor, N_contributors, replay buffer |
| RQ4 | `rq4_stability.c` | retrain N seeds, decompose variance into init vs data-order contributions | seed (≥3), variance_factor ∈ {init_only, data_order_only, both} |
| RQ5 | `rq5_noise_augmentation.c` | wrap RQ1 or RQ2 loop with on-the-fly Gaussian noise on inputs | noise_std grid, with vs without aug |

**The harness itself does not change.** Only the per-RQ `.c` body. Each RQ gets its own search-space JSON in `hpc/search_space/<rq>.json`.

**For each RQ, the workflow is:**
1. Implement `src/examples/rq<N>_*.c` (replace the stub).
2. Update `hpc/search_space/rq<N>_*.json` with the real grid.
3. `hpc/build_all_rqs.sh rq<N>_*` — produces `hpc/bin/rq<N>_*.host`.
4. `git push` paper0; on amplitude `git pull && hpc/build_all_rqs.sh rq<N>_*`.
5. `RQ=rq<N>_* sbatch hpc/run_optuna_amplitude.sh`.
6. After completion: `scripts/amplitude/50_inspect_result.sh <jobid>` — pull the trials.csv into the per-RQ analysis notebook under `experiments/rq<N>/`.

**Estimate:** ~1 day per RQ once Conv1d is in. RQ4 first (it's the noise-floor reference for RQ1/RQ2 error bars per the sprint plan).

---

## Track D — L2 + L3 verification (do alongside Track C)

L0 (header bytes ≡ .npy slice) and L1 (NPY backend ≡ baked backend bytewise) are wired and green. L2 and L3 from the equivalence-layers spec are still TBD:

- **L2** = single-step training on HOST(NPY) vs HOST(baked) byte-identical weights/grads/loss. Reuses the Plan-2 audit pattern (`ODT_SINGLE_BATCH=1` + `ODT_STATE_DUMP_PATH` + `state_dump_compare.py`). Trivial once an RQ has a real model — for the toy it would just compare zero-trained Linear weights, not interesting.
- **L3** = HOST(baked, fold k) vs RP2350(baked, fold k) single-step. UART collector + same comparator. Needs RP2350 hardware in the loop.

Both land naturally as part of Track C — the first real RQ that runs on RP2350 needs L3 anyway for the paper claim.

---

## Track E — operational hygiene

Small things that are nice-to-have:

1. **`scripts/amplitude/_lib.sh`** — currently re-runs `module load` + `PATH` extension on every script. The module load output is noisy in every log. Add `>/dev/null 2>&1` around the `module load` to silence it.
2. **`hpc/run_optuna_amplitude.sh`** — the no-container path uses `srun bash -c "..."` with a long here-doc. Consider extracting that into `hpc/inner_run.sh` for cleaner reads.
3. **`scripts/amplitude/40_smoke_sbatch.sh`** — the polling loop caps at 60 × 10s = 10 min. For real RQ runs (hours), bump to a much larger ceiling (or use `sbatch --wait` instead of polling).
4. **README quickstart** — `README.md` doesn't mention the harness or scripts/amplitude/ tree at all. Add a section or point readers at `hpc/README.md`.

These are all <1 hour each. Do as you encounter them, don't pre-plan.

---

## Open upstream-ODT issue list (for Plan 3 issue filing)

This session surfaced a third upstream bug to add to the existing Plan-2 list:

- **F1** — CE-grad missing batch-size normalization (existing). High priority.
- **F2** — DataLoader `indices[]` under-initialized (existing). High priority.
- **F7** — `tensorInit` INT32 path treats input as `float*`, sum-of-dims instead of prod-of-dims, destroys int32 inputs from NPYLoader (NEW, surfaced 2026-04-27 via L1 test). High priority. Documented in `docs/odt-userapi-findings-misc.md`.

When filing upstream issues, F7 should be one of the first — it's a one-function fix, has a working repro (the L1 test), and currently silently zeroes any int32 .npy reads.

---

## Sprint timing context

D+6 (today, 2026-04-27) is done. Sprint plan calls D7 (28.04.) for "RP2350 bring-up; inference-only validation" and D8–D9 (29.–30.04.) for "RQ4 stability — ≥ 3 seeds, one user, noise-floor reference". RQ4 is the first real RQ run, and it depends on Conv1d/LayerNorm being in upstream. If those slip, RQ4 also slips, and the cut-order kicks in (drop RQ5 → drop RQ3 → 6-page short paper with RQ1+RQ2+RQ4 only).

The harness is now off the critical path. The remaining critical-path items are upstream Conv1d/LayerNorm (out of our hands until Florian gets to it) and Florian's pre-trained checkpoint (also out of our hands).
