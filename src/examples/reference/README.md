# PyTorch Reference Comparisons (Phase 5e)

> **Stand 2026-09-03.** Die hier gelisteten MNIST-/Linear-Regression-Examples bauen am aktuellen ODT-Pin (`7d7f1d5`) nicht mehr (Pre-Factory-API); die Skripte bleiben als Audit-Beleg der Phase 5e. Die aktuellen Paritäts-Gates für den SmaTable-Trainer `stage1_pretrain` liegen unter `tools/`: `verify_reference.py` (V0), `stage1_reference.py` + `compare_stage1_parity.py` (V1/V2), `check_ckpt_roundtrip.py` (V4).

Diese Skripte vergleichen ODT-Examples gegen PyTorch-Referenzen mit N=5 Seeds
und 2σ-Acceptance-Kriterium.

## Abgedeckte Examples

| Example | Reference Script | Methode |
|---|---|---|
| `linear_regression` | *(built-in)* | Deterministisch; self-check in `linear_regression.c` gegen fixe expected weights mit 3% Toleranz |
| `mlp_mnist_float32_host` | `mlp_mnist_float32_host_ref.py` | N=5 Seeds PyTorch, 2σ-Vergleich auf `accuracy` |
| `mlp_mnist_float32_mcu` | `mlp_mnist_float32_mcu_ref.py` | N=5 Seeds PyTorch, 2σ-Vergleich auf `subset_accuracy` |
| `mlp_mnist_stress_host` | `mlp_mnist_stress_host_ref.py` | N=5 Seeds PyTorch, 2σ-Vergleich auf `accuracy` |
| `mnist_inference` | *(keine)* | Nur Forward-Only; validiert implizit wenn `mlp_mnist_float32_host` validiert |
| `mlp_mnist_asym_*` | *(gesperrt)* | `#error`-Gate — ODT Issue #61 |

## Voraussetzungen

- MNIST .npy unter `src/examples/data/` (einmal `uv run src/examples/data/generate_subset.py`)
- `HOST-Debug` Preset gebaut
- PyTorch + numpy installiert (`uv sync`)

## Ausführen

```
uv run python src/examples/reference/<example>_ref.py
```

Exit 0 bei PASS, 1 bei FAIL. Bei persistentem FAIL: Hyperparameter-Mismatch
zwischen C-Code und PyTorch-Modell prüfen, oder als Finding ins USERAPI-Audit
(Plan 2) mitnehmen.
