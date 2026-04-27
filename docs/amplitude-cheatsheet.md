# amplitUDE — operator cheatsheet

Quick reference distilled from the [official UDE-eScience HPC docs](https://escience-wissr.gitpages.uni-due.de/hpc-support/) (fetched 2026-04-27). Use this for the project; cross-reference upstream when in doubt because the docs change.

## Hosts

| Purpose | Hostname | Notes |
|---|---|---|
| **SSH login (jump host)** | `login.hpc.uni-due.de` | Load-balanced front-end. Accepts password + OTP-token (UDE 2FA). |
| **HPC cluster (Slurm + Lustre + ws_*)** | `amplitude` | Reachable only **from** the jump host. The doc says password+OTP works here too; *empirically* (2026-04-27) amplitude only listed `publickey,hostbased` — open question, see "Login issues" below. |
| **Data transfer (rsync/scp)** | `gateway.amplitude.uni-due.de` | Separate dedicated transfer server. Path on Lustre: `/lustre/scratch/<workspace>/`. |

## Documented login (matches our `~/.ssh/config`)

```ssh-config
Host login-ampl
    Hostname amplitude
    User <lowercase-UDE-ID>
    ProxyJump login.hpc.uni-due.de
    ForwardX11 no
```

Then: `ssh login-ampl` → password + OTP for the gateway, then (per docs) password + OTP again for amplitude. **If amplitude refuses password** (Permission denied, publickey,hostbased only), see Login Issues below.

Alternative one-liner (no config): `ssh -t <user>@login.hpc.uni-due.de ssh amplitude`. The inner `ssh amplitude` runs **on the gateway**, where hostbased auth between gateway and amplitude might bypass the publickey requirement.

## Account format

- Lowercase UDE ID (e.g. `soleburo`).
- Account types: `default` (researcher/PhD, 1 year), `test` (6 weeks), `student` (thesis duration), `external`. Apply via the amplitUDE web form on the CCSS portal.
- 2FA: standard UDE OTP-token (TAN list / app, separate setup). External UDE IT-Sec docs cover that — German only.

## Login issues — empirical findings 2026-04-27

- The docs say "password + OTP" works for amplitude. The actual server returned `Authentications that can continue: publickey,hostbased` and refused all password attempts.
- Likely causes: amplitude has `PasswordAuthentication no` for some/all accounts, OR the account requires a one-time activation step that hasn't happened.
- **Workarounds to try in order:**
  1. `ssh -t <user>@login.hpc.uni-due.de ssh amplitude` — uses gateway-side hostbased.
  2. Login to gateway first (`ssh <user>@login.hpc.uni-due.de`), then from inside that shell run `ssh amplitude` — same idea, more interactive.
  3. Email HPC support (`ccss@uni-due.de` per docs) with the public key, asking either for first-time activation or for direct authorized_keys install on amplitude. Mention "Permission denied (publickey,hostbased)" symptom.

## Slurm — partitions on amplitUDE

| Family | Partition | Walltime cap | MinNodes | Memory class |
|---|---|---|---|---|
| **CPU Standard 512 GB** | `STD-s-96h` (default `*`) | 96 h | 1 (small jobs) | 512 GB |
| | `STD-m-48h` | 48 h | **46** | 512 GB |
| | `STD-l-12h` | 12 h | **93** | 512 GB |
| **CPU High-Mem 1 TB** | `HIM-{s,m,l}-{96,48,12}h` | 96/48/12 h | s=1, m=?, l=? | 1 TB |
| **CPU Super-High-Mem 2 TB** | `SHM-{s,l}-{96,24}h` | 96 / 24 h | small | 2 TB |
| **GPU big** | `GPU-big` | — | — | nodes 001-008, 011-019 (4× H100 each) |
| **GPU small** | `GPU-small` | — | — | nodes 009-010 (2× H100 each) |

**Critical naming gotcha**: the `s`/`m`/`l` suffix is **node-count tier** (small / medium / large *per job*), NOT walltime. The walltime suffix (`-96h`/`-48h`/`-12h`) is real but inversely correlated. Picking the wrong tier means your job sits in the queue forever or gets rejected by the scheduler.

Our single-node Optuna jobs fit `STD-s-96h` (no MinNodes restriction, 96 h cap = 8× our needed 12 h). Set in `hpc/run_optuna_amplitude.sh`.

Standard sbatch flags: `-J`, `--nodes`, `--ntasks-per-node`, `--cpus-per-task`, `--time`, `--mem`, `--gres=gpu:N`. `sbatch` for batch, `salloc` + `srun` for interactive. Job-array syntax not documented on the wiki page (use `--array=0-N` per Slurm convention).

## Workspaces (`ws_*`) — amplitUDE only

| Limit | Value |
|---|---|
| Max lifetime | 100 days |
| Default lifetime | 100 days |
| Retention after expiration | 30 days |
| Max extensions | 4 |

Commands: `ws_allocate <name> <days>`, `ws_find <name>`, `ws_list`, `ws_extend <name> <days>`, `ws_release <name>`, `ws_restore --list` / `ws_restore <ws> <target>`.

Workspaces live on the parallel Lustre filesystem (`/lustre/scratch/...`). **Not backed up** — copy results back to `$HOME` or external storage. Optional `~/.ws_user.conf` with `mail:` + `reminder:` keys for expiration notifications. Our `hpc/run_optuna_amplitude.sh` already follows the `ws_find || ws_allocate` pattern.

## Apptainer — load via module

```bash
module load apptainer
apptainer --version
```

Modes:
- `apptainer shell image.sif` — interactive shell.
- `apptainer exec image.sif <cmd>` — single command.
- `apptainer run image.sif` — runscript / default entrypoint.

Common flags: `--bind <host_path>:<container_path>`, `--nv` (GPU), `--writable-tmpfs` (writeable scratch).

**Build images:** `sudo apptainer build` is needed for plain builds and **NOT available on login nodes**. Use `--fakeroot` instead:

```bash
apptainer build --fakeroot run_container.sif run_container.def
```

Pre-built images can be pulled: `apptainer pull docker://ubuntu:22.04`. Store large images in `$SCRATCH` (workspace) not `$HOME`.

Our `scripts/amplitude/30_build_container.sh` already tries plain → `--fakeroot` → fail-with-instructions in that order.

## Data transfer

- **`gateway.amplitude.uni-due.de`** for rsync/scp from your laptop.
- Path: `/lustre/scratch/<workspace>/`.
- Recommended: `rsync -av <local> <user>@gateway.amplitude.uni-due.de:/lustre/scratch/<workspace>/`.
- For inter-cluster (magnitUDE → amplitUDE) moves: initiate from the magnitUDE side.

Our `scripts/amplitude/11_upload_dataset.sh` rsyncs to `~/data/` (i.e. `$HOME`); to use Lustre directly, change `TARGET_DIR` in that script.

## Modules / software

`module avail`, `module load <name>`, `module list`, `module purge`. Software-specific docs cover Python, VASP, MATLAB, Apptainer.

## File system layout

- `$HOME` = `/homes/<user>` — NFS, quota-bound, backed up, **shared with `gateway.amplitude.uni-due.de`** (so rsync uploads land here and are visible from amplitude).
- `$HPC_HOME` = `/lustre/hpc_home/<user>` — Lustre per-user "home", high-throughput, **not** backed up. amplitude only. Set automatically in the user's environment on amplitude (not on the gateway).
- `/lustre/scratch/<workspace>` — Lustre workspaces, time-limited, allocated via `ws_allocate`.
- **Convention used by our scripts**: raw uploads (repo source, dataset .npz tree) go to `$HOME` via gateway; prep output and per-job working state go to `$HPC_HOME` (Lustre); sbatch's per-job scratch goes via `ws_allocate` to `/lustre/scratch/...`.

## Cross-references in this repo

- Operator README: `hpc/README.md`
- Test scripts: `scripts/amplitude/{00..50}_*.sh`
- Sbatch script: `hpc/run_optuna_amplitude.sh`
- Container recipe: `hpc/run_container.def`
