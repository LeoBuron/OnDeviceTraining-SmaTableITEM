/*
 * Example: rq0_toy_synthetic  (HOST primary, MCU baked secondary)
 *
 * End-to-end smoke test for the experiment harness. Linear-only model
 * (flatten -> Linear -> ReLU -> Linear -> Softmax) trained on the synthetic
 * SmaTable dataset produced by `tools/prep_smatable.py --synthetic`. Reaches
 * > 0.9 accuracy in a few epochs because the per-class signal in channel 0
 * is linearly separable after flatten.
 *
 * Why Linear-only and not the real Conv1d/LayerNorm architecture: upstream
 * ODT does not yet ship Conv1d (the .a is empty) and has no LayerNorm at
 * all. Sprint plan W1 D3-D5 fills those gaps. Until then this example
 * proves the *harness* works — dataset prep -> NPY/baked backends ->
 * per-trial env-driven config -> Optuna parse -> result. The architecture
 * gets swapped in once Conv1d lands; the harness stays.
 *
 * Why a manual training loop and not trainingRun(): upstream bug F2
 * (DataLoader indices[] under-initialized) silently caps per-epoch unique
 * samples to ~3% of the dataset. Bypassed entirely here by walking the
 * training split sample-by-sample. Combined with batch size = 1, this also
 * neuters bug F1 (CE-grad missing batch-size normalization, since
 * batch_size=1 -> factor of 1).
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

    /* Layer 0: Linear (WF -> hidden). Weights via Xavier-uniform; bias zero. */
    float *w0_data = calloc((size_t)hidden * WF, sizeof(float));
    float *w0_grad = calloc((size_t)hidden * WF, sizeof(float));
    size_t w0_dims[] = { (size_t)hidden, WF };
    tensor_t *w0_p = tensorInitWithDistribution(XAVIER_UNIFORM, w0_data, w0_dims, 2, q, NULL, WF, (size_t)hidden);
    tensor_t *w0_g = tensorInitFloat(w0_grad, w0_dims, 2, NULL);
    parameter_t *w0_pm = parameterInit(w0_p, w0_g);

    float *b0_data = calloc((size_t)hidden, sizeof(float));
    float *b0_grad = calloc((size_t)hidden, sizeof(float));
    size_t b0_dims[] = { 1, (size_t)hidden };
    tensor_t *b0_p = tensorInitFloat(b0_data, b0_dims, 2, NULL);
    tensor_t *b0_g = tensorInitFloat(b0_grad, b0_dims, 2, NULL);
    parameter_t *b0_pm = parameterInit(b0_p, b0_g);

    /* Layer 2: Linear (hidden -> NC). */
    float *w1_data = calloc((size_t)hidden * NC, sizeof(float));
    float *w1_grad = calloc((size_t)hidden * NC, sizeof(float));
    size_t w1_dims[] = { NC, (size_t)hidden };
    tensor_t *w1_p = tensorInitWithDistribution(XAVIER_UNIFORM, w1_data, w1_dims, 2, q, NULL, (size_t)hidden, NC);
    tensor_t *w1_g = tensorInitFloat(w1_grad, w1_dims, 2, NULL);
    parameter_t *w1_pm = parameterInit(w1_p, w1_g);

    float *b1_data = calloc(NC, sizeof(float));
    float *b1_grad = calloc(NC, sizeof(float));
    size_t b1_dims[] = { 1, NC };
    tensor_t *b1_p = tensorInitFloat(b1_data, b1_dims, 2, NULL);
    tensor_t *b1_g = tensorInitFloat(b1_grad, b1_dims, 2, NULL);
    parameter_t *b1_pm = parameterInit(b1_p, b1_g);

    layer_t *model[MODEL_SIZE];
    model[0] = linearLayerInit(w0_pm, b0_pm, q, q, q, q);
    model[1] = reluLayerInit(q, q);
    model[2] = linearLayerInit(w1_pm, b1_pm, q, q, q, q);
    model[3] = softmaxLayerInit(q, q);

    optimizer_t *sgd = sgdMCreateOptim(lr, 0.f, 0.f, model, MODEL_SIZE, FLOAT32);
    optimizerFunctions_t sgdFns = optimizerFunctions[SGD_M];

    size_t n_params = (size_t)hidden * WF + (size_t)hidden + (size_t)hidden * NC + NC;

    /* Reusable input + label tensors. Shape [1, WF] flattened input,
     * [1, NC] one-hot label. */
    float *x_buf = calloc(WF, sizeof(float));
    size_t x_dims[] = { 1, WF };
    tensor_t *x_t = tensorInitFloat(x_buf, x_dims, 2, NULL);

    float *y_buf = calloc(NC, sizeof(float));
    size_t y_dims[] = { 1, NC };
    tensor_t *y_t = tensorInitFloat(y_buf, y_dims, 2, NULL);

    clock_t t0 = clock();
    float best_acc = 0.f;
    int best_epoch = 0;

    for (int e = 0; e < n_epoch; e++) {
        float epoch_loss = 0.f;
        size_t epoch_correct = 0;
        for (size_t i = 0; i < N_TRAIN; i++) {
            int32_t label;
            smatableDatasetGetTrain(ds, i, x_buf, &label);
            memset(y_buf, 0, NC * sizeof(float));
            y_buf[(size_t)label] = 1.f;

            trainingStats_t *stats = calculateGradsSequential(
                model, MODEL_SIZE, CROSS_ENTROPY, x_t, y_t);

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
            smatableDatasetGetTest(ds, i, x_buf, &label);
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
