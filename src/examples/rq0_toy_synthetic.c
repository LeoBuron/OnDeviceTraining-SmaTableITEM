/*
 * Example: rq0_toy_synthetic  (HOST primary, MCU baked secondary)
 *
 * End-to-end smoke test for the experiment harness. Linear-only model
 * (flatten -> Linear -> ReLU -> Linear -> Softmax) trained on the synthetic
 * SmaTable dataset produced by `tools/prep_smatable.py --synthetic`. Reaches
 * > 0.9 accuracy in a few epochs because the per-class signal in channel 0
 * is linearly separable after flatten.
 *
 * Why Linear-only: the upstream architecture gap is closed as of 3e768c7
 * /main (Conv1d, LayerNorm, GroupNorm, pooling and Dropout all shipped), but
 * this toy stays Linear-only on purpose — it is the harness smoke example,
 * proving dataset prep -> NPY/baked backends -> per-trial env-driven config
 * -> Optuna parse -> result. The real Conv1d/GroupNorm architecture lives in
 * the per-RQ examples; the harness stays.
 *
 * Model construction uses the modern layerQuant_t + linearLayerInit /
 * reluLayerInit / softmaxLayerInit factory idiom (INIT_XAVIER_UNIFORM weight
 * init via linearInit_t.weightInit) — the pre-3e768c7 manual-tensor /
 * *Legacy-factory model block is gone; the Legacy shims (linearLayerInitLegacy
 * et al.) it depended on no longer exist upstream. Factory bias init is
 * uniform(+/-1/sqrt(fanIn)) per PyTorch parity (was zero under the old
 * manual-tensor block) — immaterial for this smoke test.
 *
 * Why a manual training loop and not trainingRun(): simplicity and
 * visibility — walking the training split sample-by-sample keeps the
 * stdout contract and per-epoch stats trivially auditable. Historical
 * note: this loop originally also worked around upstream bugs F1 (CE-grad
 * missing batch-size normalization) and F2 (DataLoader indices[]
 * under-initialized); both are FIXED as of 3e768c7/main (mean-scaling is
 * applied in the training loop via computeMeanScale*, and DataLoader fills
 * indices[] for the full dataset), so the loop is a style choice now, not
 * a bug workaround.
 *
 * Stdout contract (read by hpc/run_optuna.py):
 *   BEGIN  rq0_toy_synthetic <iso8601>
 *   EPOCH  <e> train_loss=<f> train_acc=<f> val_loss=<f> val_acc=<f>
 *   RESULT accuracy=<f> best_epoch=<i> n_params=<i> wall_clock_s=<f>
 *
 * Env (all optional; defaults make standalone runs reproducible):
 *   ODT_LR        learning rate                    default 0.01
 *   ODT_EPOCHS    epoch count                      default 5
 *   ODT_HIDDEN    hidden width                     default 16
 *   ODT_SEED      RNG seed for Xavier init         default 42
 *
 * Dataset env required by smatable_dataset_npy backend (HOST):
 *   SMATABLE_DATA_DIR
 *   SMATABLE_FOLD_SCHEME
 *   SMATABLE_FOLD
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "hardware_init.h"

#include "Tensor.h"
#include "TensorApi.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "ReluApi.h"
#include "SoftmaxApi.h"
#include "QuantizationApi.h"
#include "SgdApi.h"
#include "OptimizerApi.h"
#include "InferenceApi.h"
#include "CalculateGradsSequential.h"
#include "LossFunction.h"
#include "RNG.h"
#include "StorageApi.h"

#include "smatable_dataset.h"

#define MODEL_SIZE 4

static int env_int(const char *name, int dflt) {
    const char *v = getenv(name);
    return (v && *v) ? atoi(v) : dflt;
}
static float env_float(const char *name, float dflt) {
    const char *v = getenv(name);
    return (v && *v) ? (float)atof(v) : dflt;
}

static void iso8601_now(char *out, size_t n) {
    time_t t = time(NULL);
    struct tm *tm = gmtime(&t);
    strftime(out, n, "%Y-%m-%dT%H:%M:%SZ", tm);
}

/* Argmax over [1, NC] softmax output. */
static int32_t argmax(const tensor_t *t) {
    const float *p = (const float *)t->data;
    size_t nc = t->shape->dimensions[1];
    int32_t best = 0;
    for (size_t k = 1; k < nc; k++) if (p[k] > p[best]) best = (int32_t)k;
    return best;
}

/* Cross-entropy on one sample given softmax output and integer label.
 * Used purely for printing — the gradient path uses ODT's internal CE. */
static float ce_one(const tensor_t *softmax, int32_t label) {
    const float *p = (const float *)softmax->data;
    float py = p[(size_t)label];
    if (py < 1e-9f) py = 1e-9f;
    return -logf(py);
}

int main(void) {
    init();

    char ts[32]; iso8601_now(ts, sizeof(ts));
    printf("BEGIN rq0_toy_synthetic %s\n", ts);

    smatable_dataset_t *ds = smatableDatasetOpen();
    const size_t WF = smatableDatasetWindowFloats(ds);
    const size_t NC = smatableDatasetNClasses(ds);
    const size_t N_TRAIN = smatableDatasetTrainCount(ds);
    const size_t N_TEST  = smatableDatasetTestCount(ds);

    int hidden  = env_int("ODT_HIDDEN", 16);
    int n_epoch = env_int("ODT_EPOCHS", 5);
    float lr    = env_float("ODT_LR", 0.01f);
    uint32_t seed = (uint32_t)env_int("ODT_SEED", 42);
    rngSetSeed(seed);

    printf("  dataset: train=%zu test=%zu window_floats=%zu n_classes=%zu\n",
           N_TRAIN, N_TEST, WF, NC);
    printf("  hyperparams: hidden=%d epochs=%d lr=%.5f seed=%u\n",
           hidden, n_epoch, (double)lr, (unsigned)seed);

    quantization_t *q = quantizationInitFloat();

    /* Model: Linear(WF->hidden) -> ReLU -> Linear(hidden->NC) -> Softmax,
     * built via the modern layerQuant_t + linearLayerInit/reluLayerInit/
     * softmaxLayerInit factory idiom (the pre-3e768c7 manual-tensor /
     * *Legacy-factory block is gone — see file docstring). Factory bias
     * init is uniform(+/-1/sqrt(fanIn)) per PyTorch parity (was zero under
     * the old manual-tensor block) — immaterial for this smoke test. */
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *model[MODEL_SIZE];
    model[0] = linearLayerInit(
        &(linearInit_t){.inFeatures = WF,
                        .outFeatures = (size_t)hidden,
                        .bias = BIAS_TRUE,
                        .weightInit = {.scheme = INIT_XAVIER_UNIFORM, .gain = 1.0f}},
        &lq);
    model[1] = reluLayerInit(&lq);
    model[2] = linearLayerInit(
        &(linearInit_t){.inFeatures = (size_t)hidden,
                        .outFeatures = NC,
                        .bias = BIAS_TRUE,
                        .weightInit = {.scheme = INIT_XAVIER_UNIFORM, .gain = 1.0f}},
        &lq);
    model[3] = softmaxLayerInit(&lq);

    optimizer_t *sgd =
        sgdMCreateOptim(lr, 0.f, 0.f, model, MODEL_SIZE, FLOAT32, quantizationInitFloat());
    optimizerFunctions_t sgdFns = optimizerFunctions[SGD_M];

    size_t n_params = (size_t)hidden * WF + (size_t)hidden + (size_t)hidden * NC + NC;

    /* Reusable input + label tensors. Shape [1, WF] flattened input, [1, NC]
     * one-hot label. tensorInitFloat is gone upstream; build with
     * initTensor(shape, quantizationInitFloat(), NULL), which owns its own
     * zeroed buffer, and write per-sample data directly into x_t->data /
     * y_t->data (dims/order/shape allocation pattern mirrors
     * stage1_pretrain.c's buildSplit — the shape_t must outlive the tensor,
     * so it and its dims/order arrays are heap-allocated via reserveMemory,
     * not stack locals that would go out of scope). */
    size_t *xDims = reserveMemory(2 * sizeof(size_t));
    xDims[0] = 1;
    xDims[1] = WF;
    size_t *xOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, xOrder);
    shape_t *xShape = reserveMemory(sizeof(shape_t));
    setShape(xShape, xDims, 2, xOrder);
    tensor_t *x_t = initTensor(xShape, quantizationInitFloat(), NULL);

    size_t *yDims = reserveMemory(2 * sizeof(size_t));
    yDims[0] = 1;
    yDims[1] = NC;
    size_t *yOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, yOrder);
    shape_t *yShape = reserveMemory(sizeof(shape_t));
    setShape(yShape, yDims, 2, yOrder);
    tensor_t *y_t = initTensor(yShape, quantizationInitFloat(), NULL);

    /* Scratch buffer smatableDatasetGetTrain/GetTest write into; memcpy'd
     * into x_t->data per sample (x_t->data is ODT-owned, not a caller
     * buffer the dataset API can fill directly). */
    float *tmp = calloc(WF, sizeof(float));

    clock_t t0 = clock();
    float best_acc = 0.f;
    int best_epoch = 0;

    /* forwardReduction is a per-call parameter, not a config field. Loop
     * rationale: see top-of-file docstring. */
    lossConfig_t lossConfig = {
        .funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_MEAN, .classWeights = NULL};

    for (int e = 0; e < n_epoch; e++) {
        float epoch_loss = 0.f;
        size_t epoch_correct = 0;
        for (size_t i = 0; i < N_TRAIN; i++) {
            int32_t label;
            smatableDatasetGetTrain(ds, i, tmp, &label);
            memcpy(x_t->data, tmp, WF * sizeof(float));
            memset(y_t->data, 0, NC * sizeof(float));
            ((float *)y_t->data)[(size_t)label] = 1.f;

            trainingStats_t *stats = calculateGradsSequential(
                model, MODEL_SIZE, lossConfig, REDUCTION_MEAN, x_t, y_t);

            tensor_t *out = inference(model, MODEL_SIZE, x_t);
            int32_t pred = argmax(out);
            if (pred == label) epoch_correct++;
            epoch_loss += ce_one(out, label);

            sgdFns.step(sgd);
            sgdFns.zero(sgd);
            freeTrainingStats(stats);
        }
        float train_loss = epoch_loss / (float)N_TRAIN;
        float train_acc = (float)epoch_correct / (float)N_TRAIN;

        /* Eval pass — no gradients. */
        float val_loss = 0.f;
        size_t val_correct = 0;
        for (size_t i = 0; i < N_TEST; i++) {
            int32_t label;
            smatableDatasetGetTest(ds, i, tmp, &label);
            memcpy(x_t->data, tmp, WF * sizeof(float));
            tensor_t *out = inference(model, MODEL_SIZE, x_t);
            if (argmax(out) == label) val_correct++;
            val_loss += ce_one(out, label);
        }
        val_loss /= (float)N_TEST;
        float val_acc = (float)val_correct / (float)N_TEST;

        printf("EPOCH %d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f\n",
               e + 1, (double)train_loss, (double)train_acc,
               (double)val_loss, (double)val_acc);

        if (val_acc > best_acc) { best_acc = val_acc; best_epoch = e + 1; }
    }

    clock_t t1 = clock();
    float wall = (float)(t1 - t0) / (float)CLOCKS_PER_SEC;
    printf("RESULT accuracy=%.4f best_epoch=%d n_params=%zu wall_clock_s=%.3f\n",
           (double)best_acc, best_epoch, n_params, (double)wall);

    smatableDatasetClose(ds);
    return 0;
}
