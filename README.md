# OnDeviceTraining Base Project

Beispielsammlung ("example zoo") für die [OnDeviceTraining](https://github.com/es-ude/OnDeviceTraining) C-Library. Jedes Example demonstriert einen Use-Case (Training, Inference, verschiedene Quantizations) und läuft — wenn als `host`-portabel markiert — auf HOST, PICO1/2_W und STM32-Targets aus dem gleichen Source.

> **Stand 2026-09-03 — Paper-Workspace.** Dieses Repo ist außerdem der Arbeitsbereich für das SmaTable-Continual-Learning-Paper (Buron, Hettstedt, Erbslöh, Schiele; Venue offen).
> - Paper-Spec + aktueller Stand + Abhängigkeitskette: `docs/paper-plan.md`
> - Experimente (RQ1–RQ5) und ihr Status: `experiments/README.md`
> - HPC-Harness (Optuna auf amplitUDE): `hpc/README.md`, Cluster-Skripte unter `scripts/amplitude/`
> - Upstream-Bug-Status (F1/F2/F8 behoben, F7 Workaround): `docs/odt-userapi-findings-misc.md`
>
> Am aktuellen ODT-Pin (`7d7f1d5`, Factory-API) bauen nur `rq0_toy_synthetic`, `stage1_pretrain`, `rq1_replay_buffer` und die `rq2..rq5`-Stubs. Die MNIST-/Linear-Regression-Examples unten nutzen die alte API (`tensorInitFloat`, 6-Arg-`linearLayerInit`) und bauen nicht mehr; sie bleiben als Audit-Beleg der Phase 5e.

## Quickstart

```bash
# 1. Setup (einmal):
cmake --preset PREPARE  # fetched pico-sdk + OnDeviceTraining

# 2. Synthetischen SmaTable-Datensatz erzeugen (einmal):
uv run tools/prep_smatable.py --synthetic --dst data/smatable --schemes LOSO --syn-T 32 --syn-subjects 4 --syn-sessions 3 --syn-per-class 5

# 3. Example wählen + builden (Host):
cmake --preset HOST-Debug -DODT_EXAMPLE=rq0_toy_synthetic
cmake --build --preset HOST-Debug
SMATABLE_DATA_DIR=$PWD/data/smatable SMATABLE_FOLD_SCHEME=LOSO SMATABLE_FOLD=0 ./build/HOST-Debug/HOST

# 4. Für MCU (z.B. Pico2 W; gebackene Fold-Header aus data/smatable/folds/) —
#    schlägt am aktuellen Pin bereits beim Configure fehl (F9, siehe Hardware-Test-Matrix):
cmake --preset PICO2_W -DODT_EXAMPLE=rq0_toy_synthetic
cmake --build --preset PICO2_W
# Flash UF2 aus build/PICO2_W/
```

Sweep-Binaries für den Cluster baut `hpc/build_all_rqs.sh` (Preset `HOST-Release`, Ausgabe unter `hpc/bin/`).

## Examples

| Name | Was | Host | MCU | Validierung | Baut am Pin `7d7f1d5` |
|---|---|---|---|---|---|
| `rq0_toy_synthetic` | Linear→ReLU→Linear auf synthetischem SmaTable-Set; Harness-Smoke | ✓ | ✗ (F9, s.u.) | L0/L1-Dataset-Gates | ✓ (Host) |
| `stage1_pretrain` | DepthwiseCNN LOSO-Pretrain (Depthwise-/Pointwise-Conv1d, GroupNorm(1,C), MaxPool, Dropout), Cosine-LR, npy-Checkpoints, Memory-Probes | ✓ | ✗ (F9) | V0–V4 gegen PyTorch (`tools/`) | ✓ (Host) |
| `rq1_replay_buffer` | Vier Replay-Arme (`ppca` / `exemplar_random` / `exemplar_herding` / `none`) bei gleichem Byte-Budget; Platzhalter-MLP | ✓ | – | provisorisch (siehe `experiments/rq1-replay-buffer/README.md`) | ✓ |
| `rq2_unseen_user`, `rq3_new_gesture`, `rq4_stability`, `rq5_noise_augmentation` | Stubs (`RESULT skipped`) | ✓ | – | – | ✓ |
| `linear_regression` | 1 Linear Layer, 3 Samples, MSE, 100 SGD | ✓ | ✓ | Built-in self-check (3% tolerance) | ✗ (alte API) |
| `mlp_mnist_float32_host` | 784→20→10 MLP, CE, full MNIST | ✓ | – | PyTorch ref (N=5, 2σ) | ✗ |
| `mlp_mnist_float32_mcu` | Gleiche MLP, 100-Sample Subset | ✓ | ✓ | PyTorch ref (N=5, 2σ) | ✗ |
| `mnist_inference` | Forward-only mit pretrained weights | ✓ | ✓ | Impliziert via float32_host | ✗ |
| `mlp_mnist_stress_host` | 5-Hidden-Layer MLP (Stresstest) | ✓ | – | PyTorch ref (N=5, 2σ) | ✗ |
| `mlp_mnist_depth_sweep_host` | Depth-Sweep + State-Dump für das Plan-2-Audit | ✓ | – | `state_dump_compare.py` | ✗ |
| `mlp_mnist_asym_host` | ASYM-Forward Template | ✗ | ✗ | Gated durch #error ([ODT #61](https://github.com/es-ude/OnDeviceTraining/issues/61)) | – |
| `mlp_mnist_asym_mcu` | ASYM-Forward Template (MCU) | ✗ | ✗ | Gated durch #error ([ODT #61](https://github.com/es-ude/OnDeviceTraining/issues/61)) | – |

Host/MCU-Spalten der alten Examples beschreiben den Stand vor dem Pin-Bump.

## Hardware-Test-Matrix

Cross-Kompilation der MCU-fähigen Examples war für alle Targets vor dem Pin-Bump verifiziert.
"Flash+Run" heißt: Binary wurde auf echter Hardware geflasht und erfolgreich
ausgeführt. **Stand 2026-09-03: am aktuellen Pin konfiguriert kein MCU-Target** (reproduziert für `PICO2_W`; PICO1/STM32 nutzen dieselbe Toolchain) — Upstream-
`MemProfile` verlangt pthreads (`find_package(Threads REQUIRED)`), was unter `arm-none-eabi`
fehlschlägt (Finding **F9** in `docs/odt-userapi-findings-misc.md`, reproduziert mit
`cmake --preset PICO2_W -DODT_EXAMPLE=rq0_toy_synthetic`). Nichts geflasht; der RP2350-Bring-up
ist der kritische Pfad des Papers (siehe `docs/paper-plan.md`).

| Target | Flash+Run verifiziert |
|---|---|
| HOST | Alle host-fähigen Examples |
| PICO1 | *(nachtragen wenn getestet)* |
| PICO2_W | *(nachtragen wenn getestet)* |
| STM32F756ZGT6 | *(nachtragen wenn getestet)* |
| STM32L476RG | *(nachtragen wenn getestet)* |
| STM32L4R5ZI | *(nachtragen wenn getestet)* |

## Validierung per PyTorch-Referenz

Die alten MNIST-Training-Examples wurden gegen PyTorch validiert mit N=5 Seeds und 2σ-Toleranz
(`src/examples/reference/README.md`, historisch). Der aktuelle SmaTable-Trainer `stage1_pretrain`
hat eigene Gates unter `tools/`: `verify_reference.py` (V0: Referenzmodell reproduziert Florians
Confusion-Matrizen exakt), `stage1_reference.py` + `compare_stage1_parity.py` (V1 Prediction-/V2
Gradient-Parität), `check_ckpt_roundtrip.py` (V4 Checkpoint-Roundtrip), `aggregate_stage1.py`
(V3: LOSO-Mittel innerhalb 3 pp der Referenz).

## ODT-Gotchas beim Schreiben eigener Examples

- **`freeTensor` crasht auf user-allokierten Buffers.** Wenn du einen eigenen `float[...]` in einen Tensor wrappst, **nicht `freeTensor` aufrufen** — ODTs Allocator kann keine stack/static Pointer freigeben.
- **Bias-Shape ist `{1, out_features}`, nicht `{out_features, 1}`.** Sonst Shape-Mismatch in `addFloat32TensorsInplace`.
- **`PRINT_ERROR` ist silent bei `DLEVEL=0` (Default).** Für Diagnose: `-DDEBUG_MODE_ERROR` in den Compile-Definitionen (HOST-Debug und HOST-Release haben das bereits).
- **ASYM-Quantization ist in `linearForward` nicht dispatched** (ODT Issue #61). Bis das gefixt ist: `SYM_INT32` oder `FLOAT32` nutzen.
- **`sgdMCreateOptim`-Signatur seit Pin `7d7f1d5`:** `(lr, momentum, weightDecay, model, sizeModel, quantization_t *momentumQuant, arithmetic_t updateMath)` — `updateMath` immer explizit übergeben (Vorbild: `rq0_toy_synthetic.c`, `stage1_pretrain.c`).
- **`tensorInit` INT32-Pfad (F7) liest int32-`.npy` als Nullen** — int32-Fold-IDs über `readInt32NpyDirect` in `src/dataset/smatable_dataset_npy.c` laden.
