/*
 * Example: mlp_mnist_depth_sweep_host  (HOST-only, Plan 2 Audit Pass 1)
 *
 * Parametrisierbares MLP zur Messung des Layer-Depth-Effekts auf die
 * ODT-vs-PyTorch-Divergenz. Tiefe wird am Build per Cache-Variable gewählt:
 *
 *   cmake --preset HOST-Debug -DODT_EXAMPLE=mlp_mnist_depth_sweep_host \
 *         -DDEPTH_SWEEP_HIDDEN_LAYERS=4
 *
 * 0 = 1-Layer  (784 -> 10 + Softmax)            ~ Log-Regression
 * 1 = 2-Layer  (784 -> 32 -> 10 + Softmax)      ~ analog mlp_mnist_float32_host
 * 4 = 5-Layer  (784 -> 32 -> 32 -> 32 -> 32 -> 10 + Softmax)
 *
 * Anders als mlp_mnist_stress_host ist das NICHT ein Boilerplate-Demonstrator —
 * Loops ueber die Layer sind hier erlaubt, damit die Tiefe der einzige
 * Freiheitsgrad ist.
 *
 * Zusaetzlich: "Single-Batch State-Dump"-Modus, aktiviert ueber
 *   ODT_SINGLE_BATCH=1  und  ODT_STATE_DUMP_PATH=<dir>
 * In diesem Modus:
 *   1. Laedt MNIST-Train.
 *   2. Zieht genau EINEN Batch der Groesse BATCH_SIZE, deterministische Reihenfolge
 *      (kein Shuffle, Indizes 0..BATCH_SIZE-1 — so gibt dataLoaderInit sie zurueck).
 *   3. Dumpt Initial-Weights/-Biases aller Linear-Layer in .npy-Dateien
 *      (pre_w_<k>.npy, pre_b_<k>.npy).
 *   4. Dumpt Post-forward Pre-ReLU-Activations pro Linear-Layer
 *      (pre_relu_<k>.npy, Shape {BATCH_SIZE, out_features}). Da ODT's Linear-
 *      Layer keine Broadcasting-Bias-Add unterstuetzt (Arithmetic.c erzwingt
 *      strikte Shape-Gleichheit), wird pro Sample inference(model, 2*k+1,
 *      sample[i]->item) aufgerufen, die {1, out}-Ergebnisse werden zeilenweise
 *      in einen Batch-Buffer geschrieben. Siehe Task-4-Kommentar unten.
 *   5. Laeuft BATCH_SIZE-mal calculateGradsSequential pro Sample und akkumuliert
 *      Gradients (= "sum"-Reduktion; das macht trainingBatchDefault genauso,
 *      zero_grad wird erst nach dem Optimizer-Step aufgerufen).
 *   6. Dumpt Post-Backward-Gradients (post_grad_w_<k>.npy, post_grad_b_<k>.npy)
 *      und loss_sum.npy (Summe der per-Sample-Verluste, siehe CrossEntropy.c).
 *   7. Beendet ohne SGD-Step.
 *
 * API-Deltas ggue. dem urspruenglichen Plan-Snippet (verifiziert gegen
 * OnDeviceTraining/src/src/...):
 *   - calculateGradsSequential(model, modelSize, lossType, input, label)
 *     nimmt input+label separat (NICHT einen batch_t *) und liefert
 *     trainingStats_t* (NICHT float); stats->loss ist das Per-Sample-Loss.
 *     Muss pro Sample aufgerufen + mit freeTrainingStats befreit werden.
 *   - Gradienten akkumulieren in linearCalcWeightGradsFloat32 per
 *     addFloat32TensorsInplace — trainingBatchDefault ruft zero_grad erst
 *     nach dem Optimizer-Step. Das Dump-Mode-Verhalten spiegelt das.
 *   - inference(model, numberOfLayers, input) → tensor_t* (3 Args, Task 4).
 *
 * ODT-Gotchas beachtet:
 *   - Keine freeTensor-Aufrufe auf user-owned static Buffers.
 *   - Bias-Shape { 1, out_features } (NICHT { out_features, 1 }).
 *   - Nur quantizationInitFloat (ASYM ist unwired).
 *   - snprintf wird AUSSCHLIESSLICH im singleBatch-Branch benutzt, der VOR
 *     trainingRun mit return 0 beendet → keine Latenz-Corruption auf dem
 *     Training-Pfad.
 *   - rngSetSeed(0) == rngSetSeed(1): der Harness in compare.py kompensiert das
 *     per +1-Offset; hier nehmen wir ODT_SEED direkt wie existierende Beispiele.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>  // for clock() on the training path

#include "hardware_init.h"

#include "Layer.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "QuantizationApi.h"
#include "LinearApi.h"
#include "ReluApi.h"
#include "SoftmaxApi.h"
#include "SgdApi.h"
#include "InferenceApi.h"
#include "TrainingLoopApi.h"
#include "CalculateGradsSequential.h"
#include "DataLoaderApi.h"
#include "DataLoader.h"
#include "NPYLoaderApi.h"
#include "Dataset.h"
#include "LossFunction.h"
#include "StorageApi.h"
#include "RNG.h"

#ifndef MNIST_DATA_DIR
#define MNIST_DATA_DIR "."
#endif
#define MNIST_TRAIN_X MNIST_DATA_DIR "/mnist_train_x.npy"
#define MNIST_TRAIN_Y MNIST_DATA_DIR "/mnist_train_y.npy"
#define MNIST_TEST_X  MNIST_DATA_DIR "/mnist_test_x.npy"
#define MNIST_TEST_Y  MNIST_DATA_DIR "/mnist_test_y.npy"

/* Tiefe: wieviele Hidden-Layer (Linear+ReLU). 0 ⇒ nur Output-Linear + Softmax. */
#ifndef DEPTH_SWEEP_HIDDEN_LAYERS
#define DEPTH_SWEEP_HIDDEN_LAYERS 1
#endif

#define INPUT_DIM     (28 * 28)
#define HIDDEN_DIM    32
#define OUTPUT_DIM    10
#define NUM_CLASSES   10
#define BATCH_SIZE    32
#define NUM_EPOCHS    10
#define LEARNING_RATE 0.001f

/* Kette: pro Hidden-Layer ein Linear + ReLU (2 Eintraege), plus Output-Linear
 * + Softmax (2 Eintraege). Also: 2*N + 2. */
#define MODEL_SIZE    (2 * DEPTH_SWEEP_HIDDEN_LAYERS + 2)

/* Max-5 Hidden-Layer erlaubt statische Buffer; aktuell brauchen wir 0/1/4. */
#define MAX_HIDDEN 5
#if DEPTH_SWEEP_HIDDEN_LAYERS > MAX_HIDDEN
#error "DEPTH_SWEEP_HIDDEN_LAYERS too large for static buffers"
#endif

#define DEFAULT_CSV_PATH "runs/mlp_mnist_depth_sweep_host_odt.csv"

static dataset_t trainDataset;
static dataset_t testDataset;
static layer_t *model[MODEL_SIZE];
static dataLoader_t *testDL;
static const char *csvPath = DEFAULT_CSV_PATH;

static sample_t *getTrainSample(size_t id) { return npyGetSample(&trainDataset, id); }
static sample_t *getTestSample (size_t id) { return npyGetSample(&testDataset,  id); }
static size_t    getTrainSize(void)        { return trainDataset.items->size; }
static size_t    getTestSize (void)        { return testDataset.items->size;  }

/* Items in a fresh .npy MNIST load have shape [1, 28, 28]. Linear expects
 * [1, 784], so rewrite the shape metadata in place (same helper as
 * mlp_mnist_float32_host.c). */
static void flattenItems(tensorArray_t *arr) {
    for (size_t i = 0; i < arr->size; i++) {
        shape_t *shape = arr->array[i]->shape;
        size_t *newDims  = reserveMemory(2 * sizeof(size_t));
        size_t *newOrder = reserveMemory(2 * sizeof(size_t));
        newDims[0] = shape->dimensions[0];
        newDims[1] = shape->dimensions[1] * shape->dimensions[2];
        newOrder[0] = 0;
        newOrder[1] = 1;
        freeReservedMemory(shape->dimensions);
        freeReservedMemory(shape->orderOfDimensions);
        shape->dimensions = newDims;
        shape->orderOfDimensions = newOrder;
        shape->numberOfDimensions = 2;
    }
}

/* Statische Backing-Stores. Reihenfolge der Einsprung-Weights:
 *   wInputHidden (Input -> Hidden)            — nur wenn N >= 1
 *   wHidden[k]   (Hidden -> Hidden), k=0..N-2 — N-1 Stueck
 *   wOutHidden   (Hidden -> Output)           — nur wenn N >= 1
 *   wOutDirect   (Input  -> Output)           — nur wenn N == 0
 */
static float wInputHidden[HIDDEN_DIM * INPUT_DIM];
static float bInputHidden[HIDDEN_DIM];
static float wHidden[MAX_HIDDEN][HIDDEN_DIM * HIDDEN_DIM];
static float bHidden[MAX_HIDDEN][HIDDEN_DIM];
static float wOutHidden[OUTPUT_DIM * HIDDEN_DIM];
static float wOutDirect[OUTPUT_DIM * INPUT_DIM];
static float bOut[OUTPUT_DIM];

static void csvInit(void) {
    const char *envPath = getenv("ODT_CSV_PATH");
    if (envPath && envPath[0] != '\0') csvPath = envPath;
    FILE *f = fopen(csvPath, "w");
    if (!f) {
        fprintf(stderr, "Could not open CSV log %s\n", csvPath);
        return;
    }
    fprintf(f, "epoch,train_loss,eval_loss,test_accuracy\n");
    fclose(f);
    printf("CSV log: %s\n", csvPath);
}

static void onEpochEnd(size_t epoch, float trainLoss, epochStats_t evalStats) {
    FILE *f = fopen(csvPath, "a");
    if (f) {
        fprintf(f, "%zu,%.6f,%.6f,%.6f\n",
                epoch + 1, (double)trainLoss, (double)evalStats.loss,
                (double)(evalStats.accuracy * 100.0));
        fclose(f);
    }
    printf("  epoch %zu: train_loss=%.4f eval_loss=%.4f test_acc=%.2f%%\n",
           epoch + 1, (double)trainLoss, (double)evalStats.loss,
           (double)(evalStats.accuracy * 100.0));
}

/* --- Minimaler NPY-Writer (nur little-endian float32). --------------------
 * Wird NUR im Dump-Branch benutzt (snprintf-Gotcha: dieser Branch endet mit
 * return 0 bevor trainingRun je angefasst wird). */
static int writeNpyFloat(const char *path, const float *data, const size_t *dims,
                         size_t numDims) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    /* NPY v1.0: magic + version */
    unsigned char magic[] = { 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0 };
    fwrite(magic, 1, sizeof(magic), f);
    /* Dictionary header */
    char header[256] = {0};
    int n = snprintf(header, sizeof(header),
                     "{'descr': '<f4', 'fortran_order': False, 'shape': (");
    for (size_t i = 0; i < numDims; i++) {
        n += snprintf(header + n, sizeof(header) - (size_t)n, "%zu%s",
                      dims[i], (i + 1 < numDims || numDims == 1) ? ", " : "");
    }
    n += snprintf(header + n, sizeof(header) - (size_t)n, "), }");
    /* Pad mit Spaces bis inkl. '\n' die Gesamtlaenge (10 + headerLen) %64 == 0.
     * Die numpy-Referenz verwendet %64; %16 reicht auch fuer die Rohversion.
     * Wir nutzen %64, damit np.load genau gleich parst wie bei np.save. */
    size_t totalBefore = 10 + (size_t)n + 1;
    size_t pad = (64 - totalBefore % 64) % 64;
    for (size_t i = 0; i < pad; i++) header[n + i] = ' ';
    header[n + pad] = '\n';
    unsigned short headerLen = (unsigned short)(n + pad + 1);
    fwrite(&headerLen, 2, 1, f);
    fwrite(header, 1, headerLen, f);
    size_t total = 1;
    for (size_t i = 0; i < numDims; i++) total *= dims[i];
    fwrite(data, sizeof(float), total, f);
    fclose(f);
    return 0;
}

static void dumpScalar(const char *dir, const char *name, float value) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.npy", dir, name);
    size_t dims[1] = {1};
    writeNpyFloat(path, &value, dims, 1);
}

static void dumpMatrix(const char *dir, const char *name, const float *data,
                       size_t rows, size_t cols) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.npy", dir, name);
    size_t dims[2] = {rows, cols};
    writeNpyFloat(path, data, dims, 2);
}

int main(void) {
    init();
    printf("mlp_mnist_depth_sweep_host: DEPTH=%d HIDDEN=%d MODEL_SIZE=%d\n",
           DEPTH_SWEEP_HIDDEN_LAYERS, HIDDEN_DIM, MODEL_SIZE);

    const char *dumpDir   = getenv("ODT_STATE_DUMP_PATH");
    const int   singleBatch = (getenv("ODT_SINGLE_BATCH") != NULL);

    uint32_t odtSeed = 42;
    const char *seedEnv = getenv("ODT_SEED");
    if (seedEnv && seedEnv[0] != '\0') odtSeed = (uint32_t)atoi(seedEnv);
    printf("ODT_SEED=%u singleBatch=%d\n", (unsigned)odtSeed, singleBatch);

    trainDataset.items  = npyLoad(MNIST_TRAIN_X);
    trainDataset.labels = npyLoad(MNIST_TRAIN_Y);
    testDataset.items   = npyLoad(MNIST_TEST_X);
    testDataset.labels  = npyLoad(MNIST_TEST_Y);
    if (!trainDataset.items || !trainDataset.labels ||
        !testDataset.items  || !testDataset.labels) {
        fprintf(stderr, "Could not load MNIST .npy files.\n");
        return 1;
    }
    flattenItems(trainDataset.items);
    flattenItems(testDataset.items);

    /* Im Dump-Modus: kein Shuffle, Batch-Indizes 0..BATCH_SIZE-1.
     * ODT_DISABLE_SHUFFLE=1: kein Shuffle im Full-Training-Pfad (fuer H3 —
     * DataLoader-Shuffle-Isolation). singleBatch-Dump-Modus impliziert das
     * bereits; der zusaetzliche Flag schaltet es unabhaengig davon. */
    const int disableShuffle = (getenv("ODT_DISABLE_SHUFFLE") != NULL);
    const int noShuffle = singleBatch || disableShuffle;
    if (disableShuffle) printf("ODT_DISABLE_SHUFFLE=1 (H3): shuffle=false\n");
    dataLoader_t *trainDL = dataLoaderInit(getTrainSample, getTrainSize, BATCH_SIZE,
                                            NULL, NULL, !noShuffle, odtSeed, true);
    testDL = dataLoaderInit(getTestSample, getTestSize, 1,
                             NULL, NULL, false, 0, true);

    quantization_t *q = quantizationInitFloat();

    /* Parameter-Zeiger in Arrays, damit Dump-Code einheitlich iterieren kann. */
    const int numLinear = DEPTH_SWEEP_HIDDEN_LAYERS + 1;
    tensor_t *wTensors[MAX_HIDDEN + 1] = {NULL};
    tensor_t *bTensors[MAX_HIDDEN + 1] = {NULL};
    tensor_t *wGrads  [MAX_HIDDEN + 1] = {NULL};
    tensor_t *bGrads  [MAX_HIDDEN + 1] = {NULL};

    if (DEPTH_SWEEP_HIDDEN_LAYERS > 0) {
        /* --- Erstes Linear-Layer: Input → Hidden --- */
        size_t *wd = reserveMemory(2 * sizeof(size_t));
        wd[0] = HIDDEN_DIM; wd[1] = INPUT_DIM;
        tensor_t *wp = tensorInitWithDistribution(XAVIER_UNIFORM, wInputHidden,
                                                   wd, 2, q, NULL, INPUT_DIM, HIDDEN_DIM);
        tensor_t *wg = gradInitFloat(wp, NULL);
        size_t *bd = reserveMemory(2 * sizeof(size_t));
        bd[0] = 1; bd[1] = HIDDEN_DIM;
        tensor_t *bp = tensorInitWithDistribution(ZEROS, bInputHidden,
                                                   bd, 2, q, NULL, 1, HIDDEN_DIM);
        tensor_t *bg = gradInitFloat(bp, NULL);
        wTensors[0] = wp; bTensors[0] = bp; wGrads[0] = wg; bGrads[0] = bg;
        model[0] = linearLayerInit(parameterInit(wp, wg), parameterInit(bp, bg),
                                    q, q, q, q);
        model[1] = reluLayerInit(q, q);

        /* --- Zwischen-Hidden-Layer (Hidden → Hidden), (N-1) Stueck --- */
        for (int k = 1; k < DEPTH_SWEEP_HIDDEN_LAYERS; k++) {
            size_t *wd2 = reserveMemory(2 * sizeof(size_t));
            wd2[0] = HIDDEN_DIM; wd2[1] = HIDDEN_DIM;
            tensor_t *wp2 = tensorInitWithDistribution(XAVIER_UNIFORM, wHidden[k - 1],
                                                        wd2, 2, q, NULL, HIDDEN_DIM, HIDDEN_DIM);
            tensor_t *wg2 = gradInitFloat(wp2, NULL);
            size_t *bd2 = reserveMemory(2 * sizeof(size_t));
            bd2[0] = 1; bd2[1] = HIDDEN_DIM;
            tensor_t *bp2 = tensorInitWithDistribution(ZEROS, bHidden[k - 1],
                                                        bd2, 2, q, NULL, 1, HIDDEN_DIM);
            tensor_t *bg2 = gradInitFloat(bp2, NULL);
            wTensors[k] = wp2; bTensors[k] = bp2; wGrads[k] = wg2; bGrads[k] = bg2;
            model[2 * k]     = linearLayerInit(parameterInit(wp2, wg2),
                                                 parameterInit(bp2, bg2), q, q, q, q);
            model[2 * k + 1] = reluLayerInit(q, q);
        }

        /* --- Output-Layer: Hidden → Output --- */
        int outIdx = DEPTH_SWEEP_HIDDEN_LAYERS;
        size_t *wdO = reserveMemory(2 * sizeof(size_t));
        wdO[0] = OUTPUT_DIM; wdO[1] = HIDDEN_DIM;
        tensor_t *wpO = tensorInitWithDistribution(XAVIER_UNIFORM, wOutHidden,
                                                    wdO, 2, q, NULL, HIDDEN_DIM, OUTPUT_DIM);
        tensor_t *wgO = gradInitFloat(wpO, NULL);
        size_t *bdO = reserveMemory(2 * sizeof(size_t));
        bdO[0] = 1; bdO[1] = OUTPUT_DIM;
        tensor_t *bpO = tensorInitWithDistribution(ZEROS, bOut,
                                                    bdO, 2, q, NULL, 1, OUTPUT_DIM);
        tensor_t *bgO = gradInitFloat(bpO, NULL);
        wTensors[outIdx] = wpO; bTensors[outIdx] = bpO;
        wGrads  [outIdx] = wgO; bGrads  [outIdx] = bgO;
        model[MODEL_SIZE - 2] = linearLayerInit(parameterInit(wpO, wgO),
                                                 parameterInit(bpO, bgO), q, q, q, q);
    } else {
        /* DEPTH=0 ⇒ direkter Input → Output */
        size_t *wd = reserveMemory(2 * sizeof(size_t));
        wd[0] = OUTPUT_DIM; wd[1] = INPUT_DIM;
        tensor_t *wp = tensorInitWithDistribution(XAVIER_UNIFORM, wOutDirect,
                                                   wd, 2, q, NULL, INPUT_DIM, OUTPUT_DIM);
        tensor_t *wg = gradInitFloat(wp, NULL);
        size_t *bd = reserveMemory(2 * sizeof(size_t));
        bd[0] = 1; bd[1] = OUTPUT_DIM;
        tensor_t *bp = tensorInitWithDistribution(ZEROS, bOut,
                                                   bd, 2, q, NULL, 1, OUTPUT_DIM);
        tensor_t *bg = gradInitFloat(bp, NULL);
        wTensors[0] = wp; bTensors[0] = bp; wGrads[0] = wg; bGrads[0] = bg;
        model[MODEL_SIZE - 2] = linearLayerInit(parameterInit(wp, wg),
                                                 parameterInit(bp, bg), q, q, q, q);
    }
    model[MODEL_SIZE - 1] = softmaxLayerInit(q, q);

    /* ========================================================================
     * DUMP MODUS: pre-init-Weights/-Biases DUMPEN, dann BATCH_SIZE Forward+Backward
     * Durchlaeufe akkumulieren Gradients (wie trainingBatchDefault es tut, nur
     * ohne Optimizer-Step), dann dumpen. Beenden vor trainingRun.
     * ======================================================================== */
    if (singleBatch && dumpDir) {
        /* 1. Pre-init Weights & Biases dumpen. */
        for (int k = 0; k < numLinear; k++) {
            shape_t *ws = wTensors[k]->shape;
            shape_t *bs = bTensors[k]->shape;
            char name[32];
            snprintf(name, sizeof(name), "pre_w_%d", k);
            dumpMatrix(dumpDir, name, (float *)wTensors[k]->data,
                       ws->dimensions[0], ws->dimensions[1]);
            snprintf(name, sizeof(name), "pre_b_%d", k);
            dumpMatrix(dumpDir, name, (float *)bTensors[k]->data,
                       bs->dimensions[0], bs->dimensions[1]);
        }

        /* 2. Batch 0 holen (deterministische Reihenfolge dank shuffle=false). */
        batch_t *batch = trainDL->getBatch(trainDL, 0);

        /* 2b. Post-forward Activations dumpen (Task 4 / H4).
         *
         *  Variante 2 + Per-Sample-Loop (gewaehlt): ODT's layer_t exponiert
         *  kein output-Member (siehe OnDeviceTraining/src/src/layer/include/
         *  Layer.h — layer_t ist { layerType_t type; layerConfig_t* config; },
         *  ohne Activations-Cache). Eine erste Version versuchte Full-Batch-
         *  Inference ({32,784}), aber ODT's linearForwardFloat ruft
         *  addFloat32TensorsInplace(output, bias), und doDimensionsMatch in
         *  Arithmetic.c:29 verlangt strikt gleiche Shapes OHNE Broadcasting.
         *  Output {32,32} + Bias {1,32} => PRINT_ERROR "Dimensions don't match".
         *
         *  Workaround: pro Sample inference(model, 2*k+1, sample[i]->item),
         *  das Ergebnis ({1, out}) zeilenweise in einen Batch-Buffer
         *  ({BATCH_SIZE, out}) einsortieren, einmal pro Linear-Layer k. Kosten:
         *  O(numLinear * batch_size) Teil-Inferences, macht fuer N=4, BS=32
         *  ca. 5*32=160 Aufrufe — immer noch unterhalb der Training-Zeit.
         *
         *  WICHTIG: NICHT bis zum Softmax ausfuehren — ODT's Softmax hat einen
         *  globalen Max/Sum ueber alle Elemente (siehe Softmax.c:32-48) und
         *  rechnet ergo bei Batch-Input FALSCH; fuer pre_relu reicht stop-
         *  before-softmax vollkommen.
         */

        /* Per-Linear-Layer Dump-Buffer. Grosze out_features max = max(HIDDEN_DIM,
         * OUTPUT_DIM) = 32 (HIDDEN_DIM). Reserve {BATCH_SIZE, HIDDEN_DIM}. */
        static float preReluBuf[BATCH_SIZE * HIDDEN_DIM];

        for (int k = 0; k < numLinear; k++) {
            int numLayersToRun = (k < DEPTH_SWEEP_HIDDEN_LAYERS) ? (2 * k + 1)
                                                                 : (MODEL_SIZE - 1);
            size_t outFeatures = 0;

            for (size_t i = 0; i < batch->size; i++) {
                tensor_t *out = inference(model, (size_t)numLayersToRun,
                                          batch->samples[i]->item);
                /* out->shape ist {1, out_features}. */
                if (i == 0) {
                    outFeatures = out->shape->dimensions[1];
                }
                memcpy(&preReluBuf[i * outFeatures], out->data,
                       outFeatures * sizeof(float));
                /* Output von inference() ist Framework-allocated (via
                 * reserveMemory in InferenceApi.c). freeTensor hier ist der
                 * gleiche Pfad den evaluationBatchAccuracy in
                 * TrainingLoopApi.c:65 benutzt. Gotcha #1 gilt nur fuer
                 * user-Buffers, nicht fuer inference-Outputs. */
                freeTensor(out);
            }

            char name[32];
            snprintf(name, sizeof(name), "pre_relu_%d", k);
            dumpMatrix(dumpDir, name, preReluBuf, (size_t)batch->size, outFeatures);
        }

        /* 3. Per-Sample calculateGradsSequential; Gradienten akkumulieren
         *    intern (linearCalcWeightGradsFloat32 macht addFloat32TensorsInplace).
         *    Loss-Sum wird separat aufgesammelt. */
        float lossSum = 0.f;
        for (size_t i = 0; i < batch->size; i++) {
            trainingStats_t *stats =
                calculateGradsSequential(model, MODEL_SIZE, CROSS_ENTROPY,
                                         batch->samples[i]->item,
                                         batch->samples[i]->label);
            lossSum += stats->loss;
            freeTrainingStats(stats);
        }

        /* 4. Dump loss_sum (= ODT's batch-sum CrossEntropy) und accumulated grads. */
        dumpScalar(dumpDir, "loss_sum", lossSum);
        /* Zusaetzlich die Batch-Mean-Variante (vergleichbar mit PyTorch
         * reduction='mean' ohne zusaetzliche Batch-Division, was trainingBatch
         * per `/batch->size` am Ende macht — hier replizieren wir das). */
        dumpScalar(dumpDir, "loss_mean", lossSum / (float)batch->size);

        for (int k = 0; k < numLinear; k++) {
            shape_t *ws = wGrads[k]->shape;
            shape_t *bs = bGrads[k]->shape;
            char name[32];
            snprintf(name, sizeof(name), "post_grad_w_%d", k);
            dumpMatrix(dumpDir, name, (float *)wGrads[k]->data,
                       ws->dimensions[0], ws->dimensions[1]);
            snprintf(name, sizeof(name), "post_grad_b_%d", k);
            dumpMatrix(dumpDir, name, (float *)bGrads[k]->data,
                       bs->dimensions[0], bs->dimensions[1]);
        }

        freeBatch(batch);
        printf("[state-dump] wrote pre_w/pre_b/pre_relu/post_grad_w/post_grad_b/"
               "loss_sum/loss_mean for %d linear layer(s) to %s\n",
               numLinear, dumpDir);
        /* Deliberately exit before trainingRun — snprintf above is OK because
         * we're about to return, no trainingRun will run afterwards. */
        return 0;
    }

    /* ========================================================================
     * NORMALER TRAINING-PFAD (wie mlp_mnist_float32_host).
     * ======================================================================== */
    optimizer_t *sgd = sgdMCreateOptim(LEARNING_RATE, 0.f, 0.f, model, MODEL_SIZE, FLOAT32);

    csvInit();

    lossConfig_t lossConfig = { .funcType = CROSS_ENTROPY, .reduction = REDUCTION_MEAN };
    clock_t t0 = clock();
    trainingRunResult_t res = trainingRun(
        model, MODEL_SIZE, lossConfig,
        trainDL, testDL, sgd, NUM_EPOCHS,
        calculateGradsSequential, inferenceWithLoss, onEpochEnd);
    clock_t t1 = clock();

    float accuracy = evaluationEpochWithMetrics(
        model, MODEL_SIZE, CROSS_ENTROPY, testDL, inferenceWithLoss).accuracy;
    printf("Training done in %.2fs. final_train_loss=%.4f final_eval_loss=%.4f accuracy=%.4f%%\n",
           (double)(t1 - t0) / CLOCKS_PER_SEC,
           (double)res.finalTrainLoss, (double)res.finalEvalStats.loss,
           (double)accuracy * 100.0);
    return 0;
}
