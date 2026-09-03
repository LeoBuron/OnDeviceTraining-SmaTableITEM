# Phase 5e Comparison Failures

> **Stand 2026-09-03 (historisch).** Alle FAILs unten entstanden unter den Upstream-Bugs F1 (CE-Gradient ohne Batch-Normierung) und F2 (DataLoader `indices[]`), beide seit Pin `3e768c7` upstream behoben (Details: `docs/odt-userapi-findings-misc.md`). Die betroffenen MNIST-Examples bauen am aktuellen Pin `7d7f1d5` nicht mehr (Pre-Factory-API, siehe `README.md`); die Vergleiche wurden nach dem Fix nicht wiederholt. Dokument bleibt als Audit-Beleg.

Zeitbegrenzter Ablage-Ort für ODT-vs-PyTorch-Vergleiche, die das 2σ-Akzeptanzkriterium nicht erfüllen. Diese Findings wandern in Plan 2 (USERAPI-Audit) in die endgültige Issue-Dokumentation.

## 2026-04-19 — `mlp_mnist_float32_host`

**Setup:** 784→20→ReLU→10→Softmax + CrossEntropy + SGD lr=0.001 + BS=32 + 10 Epochen, Xavier-uniform / Zero-Bias Init, voller MNIST-Trainset.

**Ergebnis:**
- ODT accuracy: **86.00%**
- PyTorch mean (N=5 Seeds): **89.31% ± 0.31%**
- diff: **-3.31 Prozentpunkte** (ODT schlechter)
- 2σ-Toleranz: **±0.62 Prozentpunkte**
- Verdict: **FAIL** (Diff ≈ 10.7σ außerhalb)

**PyTorch-Rohwerte:** `[89.77, 89.03, 89.38, 89.46, 88.91]` — sehr geringe Streuung, klarer systematischer Unterschied.

**Hypothesen (unverifiziert):**
1. Unterschied in der Cross-Entropy-Gradientenberechnung gegenüber PyTorchs `nll_loss(log(softmax(.)))`.
2. Unterschied in der Xavier-Uniform-Init-Formel (fan_in vs. fan_in+fan_out Skalierung).
3. Unterschied in der Shuffle-Semantik des DataLoaders (evtl. unterschiedliche Batch-Zusammenstellungen über Epochen).
4. Akkumulations-/Rounding-Unterschiede im CE-Loss auf großen Batches.

**Nächste Schritte (Plan 2):** Isoliertes Repro-Micro-Example bauen, das nach 1 Epoche den State (Weights, Gradienten, Loss) exakt abgreift und gegen PyTorch numerisch vergleicht, um die Divergenz-Quelle einzugrenzen.

**Commit mit Script:** siehe `src/examples/reference/mlp_mnist_float32_host_ref.py` (wurde trotz FAIL committed — das Script selbst ist korrekt, FAIL ist ein echtes ODT-Finding).

### 2026-04-20 — Re-Run: FAIL persistiert, sogar schärfer (symmetrische 50 Epochen, N=20)

**Re-Run Setup:** Identisches Modell/Hyperparams, aber **beide Seiten 50 Epochen**, PyTorch **N=20 Seeds**.

**Ergebnis:**
- ODT accuracy: **88.88%** (vorher 86.00% bei 10 Epochen → +2.88pp durch mehr Training)
- PyTorch mean (N=20): **92.74% ± 0.21%** (vorher 89.31% ± 0.31% bei 10ep → +3.43pp, std tighter)
- diff: **-3.86 Prozentpunkte** (vorher -3.31)
- 2σ-Toleranz: **±0.42 Prozentpunkte**
- Verdict: **FAIL** (Diff ≈ 18σ außerhalb, vorher 10.7σ)

**PyTorch-Rohwerte:** 20 Seeds mit Range [92.45, 93.22]. **Kein einziger PyTorch-Seed unter 92.45%**, ODT bei 88.88% = 3.57pp unter PyTorch-Minimum.

**Interpretation:** Im Gegensatz zu MCU (wo der FAIL mit mehr Epochen zu einem PASS wurde) **bleibt der Host-Gap auf full-MNIST erhalten und wird statistisch schärfer**:
- Beide Seiten konvergieren weiter mit mehr Epochen, aber PyTorch gewinnt +0.55pp mehr als ODT.
- N=20 schrumpft PyTorch-std von 0.31 auf 0.21 → engere Statistik, FAIL geht von 10.7σ auf 18σ.
- Kein Seed-Artefakt, kein Init-Streuungs-Artefakt, kein Epochen-Artefakt: **echter systematischer Unterschied**.

**Konsequenz für Plan 2:** Priorität ist dieser Host-Fall, nicht MCU. Die Hypothesen aus Task 4 (CE-Gradient, Xavier-Skalierung, DataLoader-Shuffle, Akkumulation) bleiben unverändert gültig. Das Micro-Example sollte auf full-MNIST / Host-Skala basieren, nicht auf Subsets.

## 2026-04-19 — `mlp_mnist_float32_mcu`

**Setup:** 784→20→ReLU→10→Softmax + CrossEntropy + SGD lr=0.01 + BS=1 + 3 Epochen, Xavier-uniform / Zero-Bias Init, 100-Sample-Trainset / 20-Sample-Testset (MCU-Budget-Subset).

**Ergebnis:**
- ODT accuracy: **30.00%**
- PyTorch mean (N=5 Seeds): **63.00% ± 8.12%**
- diff: **-33.00 Prozentpunkte** (ODT schlechter)
- 2σ-Toleranz: **±16.25 Prozentpunkte**
- Verdict: **FAIL** (Diff ≈ 4.06σ außerhalb)

**PyTorch-Rohwerte:** `[65.00, 50.00, 65.00, 60.00, 75.00]` — hohe Streuung erwartbar bei 20 Test-Samples, aber ODT liegt deutlich unterhalb der gesamten PyTorch-Verteilung (min=50.00%).

**Zweites Signal für denselben Befund:** Task 4 (volles MNIST, BS=32, lr=0.001) zeigte -3.31 Prozentpunkte. Task 5 (100-Sample-Subset, BS=1, lr=0.01) zeigt -33.00 Prozentpunkte. Zwei unterschiedliche Konfigurationen, beide ODT schlechter — das spricht für einen systematischen Unterschied, nicht für Hyperparameter-Sensitivität.

**Hypothesen (unverifiziert, zusätzlich zu Task 4):**
1. BS=1 verstärkt Unterschiede in der CE-Gradientenberechnung (pro-Sample Gradient statt Batch-Mittel).
2. lr=0.01 × BS=1 mit nur 100 Samples ist numerisch empfindlich — Einzel-Step-Fehler akkumulieren.
3. Weights-Initialisierung könnte bei kleinen Modellen (784→20→10) kritischer sein als bei größeren.

**Nächste Schritte (Plan 2):** Gemeinsam mit Task-4-Finding in das USERAPI-Audit-Repro-Beispiel. Beide Szenarien sollten durch dasselbe Micro-Example (1 Epoche, State-Dump, Numerik-Vergleich) reproduzierbar sein.

**Commit mit Script:** siehe `src/examples/reference/mlp_mnist_float32_mcu_ref.py` (FAIL-Marker im Commit).

### 2026-04-20 — Resolution: PASS bei symmetrischem 50-Epochen-Setup

**Re-Run Setup:** Identisches Modell/Hyperparams, aber **beide Seiten 50 Epochen**, PyTorch **N=20 Seeds**.

**Ergebnis:**
- ODT accuracy: **55.00%** (vorher 30.00% bei 3 Epochen → +25pp durch mehr Training)
- PyTorch mean (N=20): **65.25% ± 5.58%** (vorher 63.00% ± 8.12% bei 3ep → nur +2.25pp, da auf 100-Sample-Subset fast sofort plateaued)
- diff: **-10.25 Prozentpunkte**
- 2σ-Toleranz: **±11.17 Prozentpunkte**
- Verdict: **PASS** (gerade so, -1.83σ)

**Interpretation:** Der ursprüngliche FAIL war überwiegend ein Epochen-Mismatch-Artefakt — ODT brauchte deutlich mehr Iterationen, um PyTorchs Plateau zu erreichen. Die Hypothesen 1-3 (BS=1-Sensitivität, Numerik, Init bei kleinen Modellen) sind damit für diesen Fall **schwächer motiviert**. ODT's Trajectory ist aber nach wie vor ~10pp unter PyTorch — nicht identisch, sondern "asymptotisch im selben Bereich". Kein harter Bug-Fingerprint mehr auf MCU-Skala.

## 2026-04-19 — `mlp_mnist_stress_host`

**Setup:** 784→256→128→64→32→10 + 4× ReLU + Softmax + CrossEntropy + SGD lr=0.001 + BS=32 + 5 Epochen, Xavier-uniform / Zero-Bias Init, voller MNIST-Trainset (~244K Parameter).

**Ergebnis:**
- ODT accuracy: **78.04%**
- PyTorch mean (N=5 Seeds): **89.87% ± 0.36%**
- diff: **-11.83 Prozentpunkte** (ODT schlechter)
- 2σ-Toleranz: **±0.72 Prozentpunkte**
- Verdict: **FAIL** (Diff ≈ 33σ außerhalb)

**PyTorch-Rohwerte:** `[90.51, 89.88, 89.51, 89.55, 89.92]` — sehr geringe Streuung, klarer systematischer Unterschied.

**Drittes Signal — Pattern verdichtet sich:**

| Example | Tiefe | Absolute Lücke | σ-Abstand |
|---|---|---|---|
| `mlp_mnist_float32_host` | 2-Layer (784→20→10) | -3.31pp | 10.7σ |
| `mlp_mnist_float32_mcu` | 2-Layer (Subset BS=1) | -33.00pp | 4.06σ |
| `mlp_mnist_stress_host` | 5-Layer (784→256→128→64→32→10) | -11.83pp | 33σ |

Die 5-Layer-Lücke ist 3.6× größer als die 2-Layer-Lücke bei identischen Hyperparametern (BS=32, lr=0.001, voller MNIST). Das spricht dafür, dass der ODT-PyTorch-Unterschied pro Layer akkumuliert — konsistent mit einer Gradientenberechnungs-Divergenz, die sich über Backprop-Ketten multipliziert.

**Hypothesen (verdichtet aus Task 4 + 5 + 6):**
1. **Gradientenakkumulation über Layer**: CE-Gradient weicht ab, Unterschied wächst pro Layer durch Backprop.
2. **Xavier-Uniform-Formel**: Möglicherweise unterschiedliche fan_in/fan_out-Skalierung, mit größerem Effekt bei tieferen Netzen.
3. **DataLoader-Shuffle-Semantik**: Nur bei unterschiedlichen Batch-Kompositionen über Epochen hinweg, nicht pro Layer — unwahrscheinlicher als Haupt-Ursache.
4. **Float32-Akkumulations-Rundung in MatMul**: Bei 256×128 MatMul größer als bei 20×10 — erklärt den Tiefen-Effekt.

**Nächste Schritte (Plan 2 USERAPI-Audit):** Das Repro-Micro-Example **muss** layer-depth-sweep enthalten (1, 2, 5 Layer, gleiche Hyperparams), um zu verifizieren dass die Lücke tatsächlich pro Layer wächst. Hypothese 1 oder 4 hat dann einen quantitativen Fingerabdruck.

---

## 2026-04-22+ — Plan 2 Divergence Investigation Outcome

Die drei Phase-5e-FAILs (`mlp_mnist_float32_host`, `mlp_mnist_float32_mcu`, `mlp_mnist_stress_host`) wurden in Plan 2 systematisch auf fünf Hypothesen geprüft. Ergebnisse:

| Hypothese | Status | Referenz |
|---|---|---|
| H1: CE-Gradient ohne Batch-Normalisierung | **CONFIRMED** | `odt-userapi-findings-misc.md` F1 |
| H2: Xavier-Uniform-Formel | REFUTED (negative result) | F3 |
| H3: DataLoader-Shuffle-Semantik | REFUTED (partially dominated by F2) | F5 |
| H4: FP32-MatMul-Akkumulation | REFUTED (negative result) | F4 |
| H5: Loss-Reduktions-Semantik | REFUTED (plan hypothesis was wrong) | F6 |

**Dominante Divergenzquelle:** Nur H1 überlebt als echte Divergenzquelle. Der CE-Gradient wird elementweise ohne Division durch `batch_size` geschrieben (`CrossEntropy.c:57-67`), was die effektive Lernrate am Optimizer auf `batch_size × user_lr` hebt. Plus ein **Bonus-Befund** (F2: ODT DataLoader `indices[]` unter-initialisiert — `DataLoaderApi.c:28` allokiert 60000 Einträge, `DataLoader.c:37` füllt nur 1875 davon, Rest bleibt calloc-zero → jedes Training sieht silently nur ~3% des Datasets pro Epoch), der H3's Shuffle-Unsensitivität teilweise erklärt und jede ODT-Trainingsläufe in diesem Repo betrifft.

Das volle Dossier mit Begründungen, Zahlen und Remediation-Vorschlägen steht in `docs/odt-userapi-findings-misc.md`. Plan 3 (Issue-Filing auf `es-ude/OnDeviceTraining`) baut auf dieser Finding-Liste auf — primär F1 + F2 als High-Priority-Bugs, S2/S3/S6 als Medium-Priority, F3-F5 explizit als ausgeschlossene Divergenzquellen dokumentiert (negative results reduzieren die Issue-Fläche).

**Commit mit Script:** siehe `src/examples/reference/mlp_mnist_stress_host_ref.py` (FAIL-Marker im Commit).
