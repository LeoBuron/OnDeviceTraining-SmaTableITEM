/*
 * stage1_pretrain — LOSO pretraining of the SmaTable DepthwiseCNN in ODT.
 *
 * Architecture per block: Conv1d(depthwise, k, dil, SAME, no bias) ->
 * Conv1d(pointwise 1x1, no bias) -> GroupNorm(1, C) -> ReLU -> MaxPool1d(2)
 * [-> Dropout(p)]; head: AdaptiveAvgPool1d(1) -> Flatten -> Linear(->16) ->
 * ReLU [-> Dropout] -> Linear(16->NC) -> Softmax (fused with CE loss).
 * Mirrors the verified PyTorch reference (tools/verify_reference.py).
 *
 * Protocol (design-review decisions D1 2026-09-06 + D5 2026-09-09, spec
 * docs/superpowers/specs/2026-09-10-d1-d5-protocol-change-design.md):
 *   - session-wise LOSO: train on the `train` split (other subjects, all
 *     sessions but the last); evaluate on the held-out subject's calib+test
 *     splits (all of its sessions, calib rows first). A dataset without a
 *     calib split is refused (old prep).
 *   - final-epoch model is the checkpoint and the headline; the best epoch
 *     is tracked only as a diagnostic (selection-bias column in the paper).
 *   - cosine LR annealing (CosineAnnealingLR, T_max = epochs, eta_min 0,
 *     stepped per epoch), SGD-M (stage-1 grid re-searches lr/wd).
 *
 * Stdout contract (hpc/run_optuna.py):
 *   BEGIN stage1_pretrain <iso8601>
 *   EPOCH <e> train_loss=<f> train_acc=-1 val_loss=<f> val_acc=<f>
 *   RESULT accuracy=<final-epoch acc, calib+test> test_acc=<final-epoch acc, test only>
 *          best_val_acc=<max val_acc over epochs> best_epoch=<i> n_params=<i> wall_clock_s=<f> …
 * (train_acc=-1: not computed — a second forward pass per epoch is not worth it.)
 *
 * Env contract: see the table in the implementation plan / hpc/README.md.
 */

#define SOURCE_FILE "stage1_pretrain"

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "hardware_init.h"

#include "AdaptivePool1dApi.h"
#include "CalculateGradsSequential.h"
#include "Conv1d.h"
#include "Conv1dApi.h"
#include "DataLoader.h"
#include "DataLoaderApi.h"
#include "DropoutApi.h"
#include "FlattenApi.h"
#include "GroupNormApi.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "NPYLoaderApi.h"
#include "Optimizer.h"
#include "OptimizerApi.h"
#include "Pool1dApi.h"
#include "QuantizationApi.h"
#include "ReluApi.h"
#include "RNG.h"
#include "Sgd.h"
#include "SgdApi.h"
#include "SoftmaxApi.h"
#include "StateDictApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TrainingEpochDefault.h"
#include "TrainingLoopApi.h"

#include "smatable_dataset.h"
#include "MemProfile.h"
#include "mem_probe.h"

#define MAX_BLOCKS 8
#define MAX_LAYERS 64
#define MAX_PARAM_LAYERS 32
#define PI_F 3.14159265358979323846f

/* ---------- env helpers ---------- */

static int env_int(const char *name, int dflt) {
    const char *v = getenv(name);
    return (v && *v) ? atoi(v) : dflt;
}
static float env_float(const char *name, float dflt) {
    const char *v = getenv(name);
    return (v && *v) ? (float)atof(v) : dflt;
}
static const char *env_str(const char *name, const char *dflt) {
    const char *v = getenv(name);
    return (v && *v) ? v : dflt;
}
static bool env_flag(const char *name) {
    const char *v = getenv(name);
    return v != NULL && v[0] != '\0' && strcmp(v, "0") != 0;
}

static size_t parseWidths(const char *csv, size_t *out, size_t maxN) {
    size_t n = 0;
    const char *p = csv;
    while (*p && n < maxN) {
        long v = strtol(p, (char **)&p, 10);
        if (v <= 0) {
            fprintf(stderr, "ODT_WIDTHS: bad entry in '%s'\n", csv);
            exit(1);
        }
        out[n++] = (size_t)v;
        if (*p == ',') {
            p++;
        }
    }
    return n;
}

static void iso8601_now(char *out, size_t n) {
    time_t t = time(NULL);
    struct tm *tm = gmtime(&t);
    strftime(out, n, "%Y-%m-%dT%H:%M:%SZ", tm);
}

static double mono_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int ensureDir(const char *p) {
    if (mkdir(p, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0 || errno == EEXIST) {
        return 0;
    }
    fprintf(stderr, "ERROR: cannot create %s: %s\n", p, strerror(errno));
    return 1;
}

/* ---------- minimal npy float32 writer (same as mlp_mnist_depth_sweep_host.c) ---------- */

static int writeNpyFloat(const char *path, const float *data, const size_t *dims,
                         size_t numDims) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        return -1;
    }
    unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    fwrite(magic, 1, sizeof(magic), f);
    char header[256] = {0};
    int n = snprintf(header, sizeof(header),
                     "{'descr': '<f4', 'fortran_order': False, 'shape': (");
    for (size_t i = 0; i < numDims; i++) {
        n += snprintf(header + n, sizeof(header) - (size_t)n, "%zu%s", dims[i],
                      (i + 1 < numDims || numDims == 1) ? ", " : "");
    }
    n += snprintf(header + n, sizeof(header) - (size_t)n, "), }");
    size_t totalBefore = 10 + (size_t)n + 1;
    size_t pad = (64 - totalBefore % 64) % 64;
    for (size_t i = 0; i < pad; i++) {
        header[n + i] = ' ';
    }
    header[n + pad] = '\n';
    unsigned short headerLen = (unsigned short)(n + pad + 1);
    fwrite(&headerLen, 2, 1, f);
    fwrite(header, 1, headerLen, f);
    size_t total = 1;
    for (size_t i = 0; i < numDims; i++) {
        total *= dims[i];
    }
    fwrite(data, sizeof(float), total, f);
    fclose(f);
    return 0;
}

/* Both return 0 on success; on failure they print the offending path to
 * stderr (same loud style as dumpParamTensor) and return -1 so parity
 * callers can abort with a non-zero exit instead of silently dropping
 * dump files. */
static int writeNpyFloat_1(const char *dir, const char *name, const float *d, size_t n) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.npy", dir, name);
    size_t dims[1] = {n};
    if (writeNpyFloat(path, d, dims, 1) != 0) {
        fprintf(stderr, "ERROR: cannot write %s\n", path);
        return -1;
    }
    return 0;
}
static int writeNpyFloat_2(const char *dir, const char *name, const float *d, size_t r,
                           size_t c) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.npy", dir, name);
    size_t dims[2] = {r, c};
    if (writeNpyFloat(path, d, dims, 2) != 0) {
        fprintf(stderr, "ERROR: cannot write %s\n", path);
        return -1;
    }
    return 0;
}

/* ---------- dataset: copy smatable split into a DataLoader-compatible dataset_t ---------- */

static dataset_t g_train;
static dataset_t g_eval; /* calib rows first, then test rows */
static size_t g_nCalib;
static size_t g_C, g_T, g_NC;

/* Materializes the concatenation of `splits` (in the given order) into one
 * DataLoader-compatible dataset_t. */
static void buildSplit(dataset_t *dst, const smatable_dataset_t *ds, const smatable_split_t *splits,
                       size_t nSplits) {
    size_t n = 0;
    for (size_t s = 0; s < nSplits; s++) {
        n += smatableDatasetSplitCount(ds, splits[s]);
    }
    size_t wf = smatableDatasetWindowFloats(ds);

    tensorArray_t *items = reserveMemory(sizeof(tensorArray_t));
    items->size = n;
    items->array = reserveMemory(n * sizeof(tensor_t *));
    tensorArray_t *labels = reserveMemory(sizeof(tensorArray_t));
    labels->size = n;
    labels->array = reserveMemory(n * sizeof(tensor_t *));

    float *tmp = reserveMemory(wf * sizeof(float));
    size_t i = 0;
    for (size_t s = 0; s < nSplits; s++) {
        size_t ns = smatableDatasetSplitCount(ds, splits[s]);
        for (size_t j = 0; j < ns; j++, i++) {
            int32_t lab;
            smatableDatasetGetSplit(ds, splits[s], j, tmp, &lab);

            size_t *xd = reserveMemory(3 * sizeof(size_t));
            xd[0] = 1;
            xd[1] = g_C;
            xd[2] = g_T;
            size_t *xo = reserveMemory(3 * sizeof(size_t));
            setOrderOfDimsForNewTensor(3, xo);
            shape_t *xs = reserveMemory(sizeof(shape_t));
            setShape(xs, xd, 3, xo);
            tensor_t *x = initTensor(xs, quantizationInitFloat(), NULL);
            memcpy(x->data, tmp, wf * sizeof(float));
            items->array[i] = x;

            /* rank-2 [1, NC] (not rank-1 [NC]): must match the model's real
             * softmax-output rank. reserveInferenceStats() (InferenceApi.c)
             * sizes inferenceStats->output via getShapeLike(label->shape), then
             * inferenceWithLoss() copyTensor()s the model's actual output shape
             * into it (copyShape(), Tensor.c) -- a rank-1 label makes that dest
             * one size_t short, so the copy overflows the allocation (fatal
             * under ODT_MEM_PROFILE's exact-size allocator; silently absorbed
             * into calloc() slack otherwise). Data layout is unaffected (still
             * NC contiguous floats) and no consumer reads label->shape rank as
             * a batch size (CE reads softmaxOutput's shape, not label's; see
             * upstream F8). */
            size_t *yd = reserveMemory(2 * sizeof(size_t));
            yd[0] = 1;
            yd[1] = g_NC;
            size_t *yo = reserveMemory(2 * sizeof(size_t));
            setOrderOfDimsForNewTensor(2, yo);
            shape_t *ys = reserveMemory(sizeof(shape_t));
            setShape(ys, yd, 2, yo);
            tensor_t *y = initTensor(ys, quantizationInitFloat(), NULL);
            ((float *)y->data)[(size_t)lab] = 1.0f;
            labels->array[i] = y;
        }
    }
    freeReservedMemory(tmp);
    dst->items = items;
    dst->labels = labels;
}

static sample_t *getTrainSample(size_t id) { return npyGetSample(&g_train, id); }
static size_t getTrainSize(void) { return g_train.items->size; }
static sample_t *getEvalSample(size_t id) { return npyGetSample(&g_eval, id); }
static size_t getEvalSize(void) { return g_eval.items->size; }
/* test-only view of g_eval (rows after the calib block) — no data copy */
static sample_t *getTestOnlySample(size_t id) { return npyGetSample(&g_eval, id + g_nCalib); }
static size_t getTestOnlySize(void) { return g_eval.items->size - g_nCalib; }

/* ---------- model + param registry ---------- */

typedef struct paramLayer {
    char base[48];    /* PyTorch state-dict base name, e.g. "features.0.depthwise" */
    parameter_t *w;   /* weight / gamma */
    parameter_t *b;   /* bias / beta; NULL for bias-free convs */
} paramLayer_t;

static paramLayer_t g_params[MAX_PARAM_LAYERS];
static size_t g_numParamLayers = 0;
static size_t g_maskBytes = 0; /* bit-packed BOOL dropout masks, accumulated in buildModel */

static void regParam(const char *base, parameter_t *w, parameter_t *b) {
    if (g_numParamLayers >= MAX_PARAM_LAYERS) {
        fprintf(stderr, "param registry overflow\n");
        exit(1);
    }
    snprintf(g_params[g_numParamLayers].base, sizeof(g_params[0].base), "%s", base);
    g_params[g_numParamLayers].w = w;
    g_params[g_numParamLayers].b = b;
    g_numParamLayers++;
}

/* Builds the model, fills the registry, returns layer count. Also allocates
 * dropout masks (caller never frees them; process-lifetime). */
static size_t buildModel(layer_t **model, layerQuant_t *lq, const size_t *widths, size_t nBlocks,
                         size_t k, size_t dil, float pDrop) {
    size_t m = 0;
    size_t prevC = g_C;
    size_t L = g_T;
    char base[48];

    for (size_t i = 0; i < nBlocks; i++) {
        size_t w = widths[i];
        model[m++] = conv1dLayerInit(&(conv1dInit_t){.inChannels = prevC,
                                                     .outChannels = prevC,
                                                     .kernelSize = k,
                                                     .padding = SAME,
                                                     .dilation = dil,
                                                     .groups = prevC,
                                                     .bias = BIAS_FALSE},
                                     lq);
        snprintf(base, sizeof(base), "features.%zu.depthwise", i);
        regParam(base, model[m - 1]->config->conv1d->weights, NULL);

        model[m++] = conv1dLayerInit(
            &(conv1dInit_t){
                .inChannels = prevC, .outChannels = w, .kernelSize = 1, .bias = BIAS_FALSE},
            lq);
        snprintf(base, sizeof(base), "features.%zu.pointwise", i);
        regParam(base, model[m - 1]->config->conv1d->weights, NULL);

        model[m++] = groupNormLayerInit(&(groupNormInit_t){.numGroups = 1, .numChannels = w}, lq);
        snprintf(base, sizeof(base), "features.%zu.norm", i);
        regParam(base, model[m - 1]->config->groupNorm->gamma,
                 model[m - 1]->config->groupNorm->beta);

        model[m++] = reluLayerInit(lq);
        model[m++] = maxPool1dLayerInit(
            &(maxPool1dInit_t){
                .kernelSize = 2, .stride = 2, .inputChannels = w, .inputLength = L},
            lq);
        L = L / 2;

        if (pDrop > 0.0f) {
            size_t nMask = w * L;
            size_t *md = reserveMemory(sizeof(size_t));
            md[0] = nMask;
            size_t *mo = reserveMemory(sizeof(size_t));
            setOrderOfDimsForNewTensor(1, mo);
            shape_t *ms = reserveMemory(sizeof(shape_t));
            setShape(ms, md, 1, mo);
            tensor_t *mask = initTensor(ms, quantizationInitBool(), NULL);
            g_maskBytes += (nMask + 7) / 8;
            model[m++] = dropoutLayerInit(pDrop, mask, lq->outputQ, lq->propLossQ);
        }
        prevC = w;
    }

    model[m++] = adaptiveAvgPool1dLayerInit(&(adaptiveAvgPool1dInit_t){.outputSize = 1}, lq);
    model[m++] = flattenLayerInit();
    model[m++] = linearLayerInit(&(linearInit_t){.inFeatures = prevC, .outFeatures = 16}, lq);
    regParam("head.2", model[m - 1]->config->linear->weights, model[m - 1]->config->linear->bias);
    model[m++] = reluLayerInit(lq);
    if (pDrop > 0.0f) {
        size_t *md = reserveMemory(sizeof(size_t));
        md[0] = 16;
        size_t *mo = reserveMemory(sizeof(size_t));
        setOrderOfDimsForNewTensor(1, mo);
        shape_t *ms = reserveMemory(sizeof(shape_t));
        setShape(ms, md, 1, mo);
        tensor_t *mask = initTensor(ms, quantizationInitBool(), NULL);
        g_maskBytes += (16 + 7) / 8;
        model[m++] = dropoutLayerInit(pDrop, mask, lq->outputQ, lq->propLossQ);
    }
    model[m++] = linearLayerInit(&(linearInit_t){.inFeatures = 16, .outFeatures = g_NC}, lq);
    regParam("head.5", model[m - 1]->config->linear->weights, model[m - 1]->config->linear->bias);
    model[m++] = softmaxLayerInit(lq);
    return m;
}

static size_t countParams(void) {
    size_t n = 0;
    for (size_t i = 0; i < g_numParamLayers; i++) {
        n += calcNumberOfElementsByTensor(g_params[i].w->param);
        if (g_params[i].b != NULL) {
            n += calcNumberOfElementsByTensor(g_params[i].b->param);
        }
    }
    return n;
}

/* ---- Layer A: exact model accounting (bytes; spec 2026-07-02-memory-time-probes) ----
 * Walks the live g_params registry and replays calcOutputShape, so it stays
 * correct without a separate accounting path per layer type. */
typedef struct memBudget {
    size_t params, grads, optstate, act, gradbuf, io, masks, mcuTotal, datasetHost;
} memBudget_t;

static size_t shapeBytesF32(const shape_t *s) {
    size_t n = 1;
    for (size_t d = 0; d < s->numberOfDimensions; d++) {
        n *= s->dimensions[d];
    }
    return n * sizeof(float);
}

static memBudget_t computeMemBudget(layer_t **model, size_t modelSize) {
    memBudget_t b = {0};
    for (size_t i = 0; i < g_numParamLayers; i++) {
        b.params += calcNumberOfElementsByTensor(g_params[i].w->param) * sizeof(float);
        b.grads += calcNumberOfElementsByTensor(g_params[i].w->grad) * sizeof(float);
        if (g_params[i].b != NULL) {
            b.params += calcNumberOfElementsByTensor(g_params[i].b->param) * sizeof(float);
            b.grads += calcNumberOfElementsByTensor(g_params[i].b->grad) * sizeof(float);
        }
    }
    b.optstate = b.params; /* SGD-M: one momentum buffer per parameter element */

    /* Activation chain: replay calcOutputShape from [1, C, T]. ODT keeps every
     * layer output alive across one sample's fwd+bwd, so act = sum of outputs;
     * the backward ping-pong grad buffers add max adjacent pair. */
    size_t dimsA[8] = {1, g_C, g_T};
    size_t ordA[8] = {0, 1, 2};
    size_t dimsB[8] = {0};
    size_t ordB[8] = {0};
    shape_t cur = {.numberOfDimensions = 3, .dimensions = dimsA, .orderOfDimensions = ordA};
    shape_t nxt = {.numberOfDimensions = 0, .dimensions = dimsB, .orderOfDimensions = ordB};

    size_t outBytes[MAX_LAYERS + 1];
    outBytes[0] = shapeBytesF32(&cur); /* layerOutputs[0] = the input */
    for (size_t i = 0; i < modelSize; i++) {
        layerFunctions[model[i]->type].calcOutputShape(model[i], &cur, &nxt);
        outBytes[i + 1] = shapeBytesF32(&nxt);
        b.act += outBytes[i + 1];
        memcpy(cur.dimensions, nxt.dimensions, nxt.numberOfDimensions * sizeof(size_t));
        memcpy(cur.orderOfDimensions, nxt.orderOfDimensions,
               nxt.numberOfDimensions * sizeof(size_t));
        cur.numberOfDimensions = nxt.numberOfDimensions;
    }
    for (size_t i = 0; i < modelSize; i++) {
        size_t pair = outBytes[i] + outBytes[i + 1];
        if (pair > b.gradbuf) {
            b.gradbuf = pair;
        }
    }
    b.io = outBytes[0] + g_NC * sizeof(float); /* one window in RAM + one-hot label */
    b.masks = g_maskBytes;
    b.mcuTotal = b.params + b.grads + b.optstate + b.act + b.gradbuf + b.io + b.masks;
    /* Host-only, flash-resident on MCU — excluded from mcuTotal by design: */
    b.datasetHost = (getTrainSize() + getEvalSize()) *
                    (g_C * g_T * sizeof(float) + g_NC * sizeof(float));
    return b;
}

/* ---------- checkpoint I/O ---------- */

static void dumpParamTensor(const char *dir, const char *base, const char *suffix,
                            tensor_t *t) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.%s.npy", dir, base, suffix);
    if (writeNpyFloat(path, (const float *)t->data, t->shape->dimensions,
                      t->shape->numberOfDimensions) != 0) {
        fprintf(stderr, "ERROR: cannot write %s\n", path);
        exit(1);
    }
}

static void ckptWrite(const char *dir) {
    for (size_t i = 0; i < g_numParamLayers; i++) {
        dumpParamTensor(dir, g_params[i].base, "weight", g_params[i].w->param);
        if (g_params[i].b) {
            dumpParamTensor(dir, g_params[i].base, "bias", g_params[i].b->param);
        }
    }
}

static void stateDictLoad(layer_t **model, size_t modelSize, const char *dir) {
    stateDictEntry_t entries[MAX_PARAM_LAYERS];
    tensor_t *loaded[MAX_PARAM_LAYERS][2] = {{0}};
    char path[512];
    for (size_t i = 0; i < g_numParamLayers; i++) {
        snprintf(path, sizeof(path), "%s/%s.weight.npy", dir, g_params[i].base);
        loaded[i][0] = npyLoadFlat(path);
        if (loaded[i][0] == NULL) {
            fprintf(stderr, "state dict: missing %s\n", path);
            exit(1);
        }
        entries[i].name = g_params[i].base;
        entries[i].weightData = (float *)loaded[i][0]->data;
        entries[i].biasData = NULL;
        if (g_params[i].b != NULL) {
            snprintf(path, sizeof(path), "%s/%s.bias.npy", dir, g_params[i].base);
            loaded[i][1] = npyLoadFlat(path);
            if (loaded[i][1] == NULL) {
                fprintf(stderr, "state dict: missing %s\n", path);
                exit(1);
            }
            entries[i].biasData = (float *)loaded[i][1]->data;
        }
    }
    modelLoadStateDict(model, modelSize, entries, g_numParamLayers);
    for (size_t i = 0; i < g_numParamLayers; i++) {
        freeTensor(loaded[i][0]);
        if (loaded[i][1]) {
            freeTensor(loaded[i][1]);
        }
    }
}

/* ---------- training entry point (stack-measured via upstream MemProfile.h) ---------- */

typedef struct trainCtx {
    layer_t **model;
    size_t modelSize;
    optimizer_t *sgd;
    dataLoader_t *evalLoader;
    dataLoader_t *testOnlyLoader;
    lossConfig_t lossCfg;
    /* config */
    int nEpochs, batchSize;
    uint32_t seed;
    float lr0;
    bool cosine;
    const char *ckptDir;
    /* outputs */
    float finalAcc, testAcc, bestAcc;
    int bestEpoch;
} trainCtx_t;

/* Everything the epoch loop + final-epoch checkpoint + ckpt/manifest write used
 * to do directly in main() now lives here so it can run under upstream's
 * measurePeakStackBytes (MemProfile.h), which measures it on a fresh,
 * paint-scannable pthread stack. ensureDir()/calloc() failures that used to
 * `return 1;` from main() now exit(1) directly — exit() terminates the whole
 * process regardless of which thread calls it, so the abort semantics are
 * unchanged. k/dil/pDrop/nParams are re-derived here (env vars are
 * process-global; countParams() re-reads the already-built g_params
 * registry) rather than threaded through trainCtx_t, to keep the context
 * struct limited to what the epoch loop itself needs. */
static void trainMain(void *argp) {
    trainCtx_t *ctx = (trainCtx_t *)argp;

    FILE *hist = NULL;
    if (ctx->ckptDir != NULL) {
        if (ensureDir(ctx->ckptDir) != 0) {
            exit(1);
        }
        char hp[512];
        snprintf(hp, sizeof(hp), "%s/history.csv", ctx->ckptDir);
        hist = fopen(hp, "w");
        if (hist) {
            fprintf(hist, "epoch,lr,train_loss,val_loss,val_acc,val_precision,val_recall\n");
        }
    }

    float bestAcc = -1.0f, finalAcc = -1.0f;
    int bestEpoch = 0;
    for (int e = 0; e < ctx->nEpochs; e++) {
        float lrE = ctx->cosine
                        ? 0.5f * ctx->lr0 * (1.0f + cosf(PI_F * (float)e / (float)ctx->nEpochs))
                        : ctx->lr0;
        ctx->sgd->impl->sgd->learningRate = lrE;

        /* fresh loader per epoch -> per-epoch reshuffle, deterministic in (seed, epoch) */
        dataLoader_t *trainLoader =
            dataLoaderInit(getTrainSample, getTrainSize, (uint16_t)ctx->batchSize, NULL, NULL,
                           true, (uint64_t)ctx->seed + (uint64_t)e, true);
        float trainLoss = trainingEpochDefault(ctx->model, ctx->modelSize, ctx->lossCfg,
                                               trainLoader, ctx->sgd, calculateGradsSequential,
                                               REDUCTION_MEAN);
        freeDataLoader(trainLoader);

        epochStats_t st = evaluationEpochWithMetrics(ctx->model, ctx->modelSize, CROSS_ENTROPY,
                                                     ctx->evalLoader, inferenceWithLoss,
                                                     REDUCTION_MEAN);
        printf("EPOCH %d train_loss=%.4f train_acc=-1 val_loss=%.4f val_acc=%.4f\n", e + 1,
               (double)trainLoss, (double)st.loss, (double)st.accuracy);
        fflush(stdout);
        if (hist) {
            fprintf(hist, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", e + 1, (double)lrE,
                    (double)trainLoss, (double)st.loss, (double)st.accuracy, (double)st.precision,
                    (double)st.recall);
        }
        finalAcc = st.accuracy;
        if (st.accuracy > bestAcc) {
            bestAcc = st.accuracy;
            bestEpoch = e + 1;
        }
    }
    if (hist) {
        fclose(hist);
    }

    epochStats_t stT = evaluationEpochWithMetrics(ctx->model, ctx->modelSize, CROSS_ENTROPY,
                                                  ctx->testOnlyLoader, inferenceWithLoss,
                                                  REDUCTION_MEAN);
    float testAcc = stT.accuracy;

    if (ctx->ckptDir != NULL) {
        ckptWrite(ctx->ckptDir); /* final-epoch weights (D1) */

        /* confusion matrix of the written weights, for the manifest
         * (heap-sized to g_NC*g_NC — evaluationEpochWithReport writes that
         * many entries, so a fixed buffer would overflow for NC > 6). Its
         * accuracy must equal the last epoch's — same weights, deterministic
         * eval. Anything else means the eval path is not deterministic (e.g.
         * dropout active), which would invalidate every number, so fail
         * loud. */
        size_t *cm = calloc(g_NC * g_NC, sizeof(size_t));
        if (cm == NULL) {
            fprintf(stderr, "ERROR: cannot allocate %zux%zu confusion matrix\n", g_NC, g_NC);
            exit(1);
        }
        classificationReport_t rep = evaluationEpochWithReport(
            ctx->model, ctx->modelSize, CROSS_ENTROPY, ctx->evalLoader, inferenceWithLoss, cm,
            g_NC, REDUCTION_MEAN);
        if (fabsf(rep.stats.accuracy - finalAcc) > 1e-6f) {
            fprintf(stderr, "ERROR: re-evaluation of the written checkpoint gives %.6f, last epoch "
                            "gave %.6f — eval is not deterministic\n",
                    (double)rep.stats.accuracy, (double)finalAcc);
            exit(1);
        }

        size_t nParams = countParams();
        size_t k = (size_t)env_int("ODT_KERNEL_SIZE", 7);
        size_t dil = (size_t)env_int("ODT_DILATION", 3);
        float pDrop = env_float("ODT_P_DROP", 0.0f);
        float momentum = ctx->sgd->impl->sgd->momentumFactor;
        float weightDecay = ctx->sgd->impl->sgd->weightDecay;

        char mp[512];
        snprintf(mp, sizeof(mp), "%s/manifest.json", ctx->ckptDir);
        FILE *mf = fopen(mp, "w");
        if (mf) {
            fprintf(mf, "{\n  \"example\": \"stage1_pretrain\",\n");
            fprintf(mf, "  \"data_dir\": \"%s\",\n  \"fold_scheme\": \"%s\",\n  \"fold\": %s,\n",
                    env_str("SMATABLE_DATA_DIR", "?"), env_str("SMATABLE_FOLD_SCHEME", "?"),
                    env_str("SMATABLE_FOLD", "?"));
            fprintf(mf, "  \"widths\": \"%s\", \"kernel_size\": %zu, \"dilation\": %zu, "
                        "\"p_drop\": %.4f,\n",
                    env_str("ODT_WIDTHS", "8,12,8"), k, dil, (double)pDrop);
            fprintf(mf, "  \"lr\": %.6f, \"momentum\": %.4f, \"weight_decay\": %.6f, "
                        "\"epochs\": %d, \"batch_size\": %d, \"seed\": %u, "
                        "\"lr_schedule\": \"%s\",\n",
                    (double)ctx->lr0, (double)momentum, (double)weightDecay, ctx->nEpochs,
                    ctx->batchSize, (unsigned)ctx->seed, ctx->cosine ? "cosine" : "constant");
            fprintf(mf, "  \"eval_set\": \"calib+test\", \"n_calib\": %zu, \"n_test\": %zu,\n",
                    g_nCalib, getTestOnlySize());
            fprintf(mf, "  \"final_epoch\": %d, \"final_val_acc\": %.6f, \"test_acc\": %.6f,\n",
                    ctx->nEpochs, (double)finalAcc, (double)testAcc);
            fprintf(mf, "  \"best_epoch\": %d, \"best_val_acc\": %.6f,\n", bestEpoch,
                    (double)bestAcc);
            fprintf(mf, "  \"ckpt_val_acc\": %.6f, \"val_precision\": %.6f, "
                        "\"val_recall\": %.6f,\n",
                    (double)rep.stats.accuracy, (double)rep.stats.precision,
                    (double)rep.stats.recall);
            fprintf(mf, "  \"n_params\": %zu,\n  \"conf_mat_pred_x_actual\": [", nParams);
            for (size_t i = 0; i < g_NC * g_NC; i++) {
                fprintf(mf, "%zu%s", cm[i], (i + 1 < g_NC * g_NC) ? ", " : "");
            }
            fprintf(mf, "]\n}\n");
            fclose(mf);
        }
        free(cm);
    }

    ctx->finalAcc = finalAcc;
    ctx->testAcc = testAcc;
    ctx->bestAcc = bestAcc;
    ctx->bestEpoch = bestEpoch;
}

/* ---------- main ---------- */

int main(void) {
    init();
    memProfileReset();

    char ts[32];
    iso8601_now(ts, sizeof(ts));
    printf("BEGIN stage1_pretrain %s\n", ts);
    double t0 = mono_s();

    /* hyperparams */
    size_t widths[MAX_BLOCKS];
    size_t nBlocks = parseWidths(env_str("ODT_WIDTHS", "8,12,8"), widths, MAX_BLOCKS);
    size_t k = (size_t)env_int("ODT_KERNEL_SIZE", 7);
    size_t dil = (size_t)env_int("ODT_DILATION", 3);
    float pDrop = env_float("ODT_P_DROP", 0.0f);
    float lr0 = env_float("ODT_LR", 0.1f);
    float momentum = env_float("ODT_MOMENTUM", 0.9f);
    float weightDecay = env_float("ODT_WEIGHT_DECAY", 0.0f);
    int nEpochs = env_int("ODT_EPOCHS", 250);
    int batchSize = env_int("ODT_BATCH_SIZE", 128);
    uint32_t seed = (uint32_t)env_int("ODT_SEED", 42);
    bool cosine = strcmp(env_str("ODT_LR_SCHEDULE", "cosine"), "cosine") == 0;
    const char *ckptDir = getenv("ODT_CKPT_DIR");
    const char *sdDir = getenv("ODT_STATE_DICT_DIR");
    const char *dumpDir = env_str("ODT_DUMP_DIR", ".");

    /* data */
    smatable_dataset_t *ds = smatableDatasetOpen();
    g_C = smatableDatasetNChannels(ds);
    g_T = smatableDatasetWindowSamples(ds);
    g_NC = smatableDatasetNClasses(ds);
    g_nCalib = smatableDatasetSplitCount(ds, SMATABLE_SPLIT_CALIB);
    if (g_nCalib == 0) {
        fprintf(stderr,
                "ERROR: dataset has no calib split — stage 1 needs the session-wise LOSO prep "
                "(tools/prep_smatable.py, split_version 2); re-prep SMATABLE_DATA_DIR\n");
        return 1;
    }
    buildSplit(&g_train, ds, (const smatable_split_t[]){SMATABLE_SPLIT_TRAIN}, 1);
    buildSplit(&g_eval, ds, (const smatable_split_t[]){SMATABLE_SPLIT_CALIB, SMATABLE_SPLIT_TEST},
               2);
    smatableDatasetClose(ds);
    /* Heap live-bytes right after dataset materialization + close (baseline
     * memProfileReset() at main() entry) — mirrored into RESULT/memory.json
     * as mem_dataset_heap_b. 0 when built without -DODT_MEM_PROFILE. */
    size_t heapDataset = memProfileMark();
#if MEM_PROBE_AVAILABLE
    long rssAfterData = memProbeRssNowKb();
#else
    long rssAfterData = -1;
#endif
    printf("  dataset: train=%zu eval=%zu (calib=%zu test=%zu) C=%zu T=%zu NC=%zu\n",
           getTrainSize(), getEvalSize(), g_nCalib, getTestOnlySize(), g_C, g_T, g_NC);

    /* model (seed BEFORE init so factory weight init is reproducible) */
    rngSetSeed(seed);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, quantizationInitFloat());
    layer_t *model[MAX_LAYERS];
    size_t heapBeforeModel = memProfileMark();
    size_t modelSize = buildModel(model, &lq, widths, nBlocks, k, dil, pDrop);
    size_t nParams = countParams();
    printf("  model: layers=%zu param_layers=%zu n_params=%zu\n", modelSize, g_numParamLayers,
           nParams);

    if (sdDir != NULL) {
        stateDictLoad(model, modelSize, sdDir);
        printf("  loaded state dict from %s\n", sdDir);
    }

    lossConfig_t lossCfg = {
        .funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_MEAN, .classWeights = NULL};

    dataLoader_t *evalLoader = dataLoaderInit(getEvalSample, getEvalSize, 1, NULL, NULL, false, 0,
                                              true);
    dataLoader_t *testOnlyLoader =
        dataLoaderInit(getTestOnlySample, getTestOnlySize, 1, NULL, NULL, false, 0, true);

    /* ---- V1: eval-only parity mode ---- */
    if (env_flag("ODT_EVAL_ONLY")) {
        if (ensureDir(dumpDir) != 0) {
            return 1;
        }
        size_t nEval = getEvalSize();
        float *logits = malloc(nEval * g_NC * sizeof(float));
        float *preds = malloc(nEval * sizeof(float));
        size_t correct = 0;
        size_t correctTest = 0;
        for (size_t i = 0; i < nEval; i++) {
            sample_t *s = getEvalSample(i);
            tensor_t *out = inference(model, modelSize, s->item);
            const float *p = (const float *)out->data;
            size_t am = 0;
            for (size_t c = 1; c < g_NC; c++) {
                if (p[c] > p[am]) {
                    am = c;
                }
            }
            memcpy(logits + i * g_NC, p, g_NC * sizeof(float));
            preds[i] = (float)am;
            const float *lab = (const float *)s->label->data;
            size_t truth = 0;
            for (size_t c = 1; c < g_NC; c++) {
                if (lab[c] > lab[truth]) {
                    truth = c;
                }
            }
            if (am == truth) {
                correct++;
                if (i >= g_nCalib) {
                    correctTest++;
                }
            }
            freeTensor(out);
            freeSample(s);
        }
        if (writeNpyFloat_2(dumpDir, "logits", logits, nEval, g_NC) != 0 ||
            writeNpyFloat_1(dumpDir, "preds", preds, nEval) != 0) {
            return 1;
        }
        printf("RESULT accuracy=%.6f test_acc=%.6f best_val_acc=0 best_epoch=0 n_params=%zu "
               "wall_clock_s=%.3f\n",
               (double)correct / (double)getEvalSize(),
               (double)correctTest / (double)getTestOnlySize(), nParams, mono_s() - t0);
        return 0;
    }

    optimizer_t *sgd = sgdMCreateOptim(
        lr0, momentum, weightDecay, model, modelSize, quantizationInitFloat(),
        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    size_t heapAfterOpt = memProfileMark();
    optimizerFunctions_t optimFns = optimizerFunctions[SGD_M];
#if MEM_PROBE_AVAILABLE
    long rssAfterModel = memProbeRssNowKb();
#else
    long rssAfterModel = -1;
#endif
    memBudget_t mb = computeMemBudget(model, modelSize);
    if (mb.params != nParams * sizeof(float) || mb.optstate != mb.params) {
        fprintf(stderr, "mem accounting inconsistent: params=%zu n_params*4=%zu\n", mb.params,
                nParams * sizeof(float));
        return 1;
    }

    /* ---- V2: single-batch grad-parity mode ---- */
    if (env_flag("ODT_SINGLE_BATCH")) {
        if (ensureDir(dumpDir) != 0) {
            return 1;
        }
        optimFns.zero(sgd);
        size_t B = (size_t)batchSize;
        for (size_t i = 0; i < B; i++) {
            sample_t *s = getTrainSample(i);
            trainingStats_t *st =
                calculateGradsSequential(model, modelSize, lossCfg, REDUCTION_MEAN, s->item,
                                         s->label);
            freeTrainingStats(st);
            freeSample(s);
        }
        if (B > 1) {
            scaleOptimizerGradients(sgd, 1.0f / (float)B);
        }
        char path[512];
        for (size_t i = 0; i < g_numParamLayers; i++) {
            snprintf(path, sizeof(path), "%s/grad_%s.weight.npy", dumpDir, g_params[i].base);
            if (writeNpyFloat(path, (const float *)g_params[i].w->grad->data,
                              g_params[i].w->grad->shape->dimensions,
                              g_params[i].w->grad->shape->numberOfDimensions) != 0) {
                fprintf(stderr, "ERROR: cannot write %s\n", path);
                return 1;
            }
            if (g_params[i].b) {
                snprintf(path, sizeof(path), "%s/grad_%s.bias.npy", dumpDir, g_params[i].base);
                if (writeNpyFloat(path, (const float *)g_params[i].b->grad->data,
                                  g_params[i].b->grad->shape->dimensions,
                                  g_params[i].b->grad->shape->numberOfDimensions) != 0) {
                    fprintf(stderr, "ERROR: cannot write %s\n", path);
                    return 1;
                }
            }
        }
        printf("RESULT accuracy=0 test_acc=0 best_val_acc=0 best_epoch=0 n_params=%zu "
               "wall_clock_s=%.3f\n",
               nParams, mono_s() - t0);
        return 0;
    }

    /* ---- training (runs on trainMain, driven via upstream
     * measurePeakStackBytes so its stack high-water mark can be measured) ---- */
    trainCtx_t ctx = {.model = model,
                      .modelSize = modelSize,
                      .sgd = sgd,
                      .evalLoader = evalLoader,
                      .testOnlyLoader = testOnlyLoader,
                      .lossCfg = lossCfg,
                      .nEpochs = nEpochs,
                      .batchSize = batchSize,
                      .seed = seed,
                      .lr0 = lr0,
                      .cosine = cosine,
                      .ckptDir = ckptDir,
                      .finalAcc = -1.0f,
                      .testAcc = -1.0f,
                      .bestAcc = -1.0f,
                      .bestEpoch = 0};
    /* Upstream MemProfile.h (ODT main 3e768c7) replaces the old in-header
     * painted-stack probe; it fails loud (exit 1) internally on any pthread
     * error rather than falling back, so there is no -1 inline-fallback path
     * to keep here. Probe size is env-tunable: trial-3408-class geometries
     * (wide widths x large T) may exceed the 1 MiB default because upstream
     * executeOp keeps VLA scratch ~ 3*maxElems*4B on the stack. */
    size_t stackProbeBytes = (size_t)env_int("ODT_STACK_PROBE_BYTES", 1 << 20);
    long stackPeak = (long)measurePeakStackBytes(trainMain, &ctx, stackProbeBytes);

    double cpuU, cpuS;
    long maxRssKb;
#if MEM_PROBE_AVAILABLE
    memProbeRusage(&cpuU, &cpuS, &maxRssKb);
#else
    cpuU = 0.0;
    cpuS = 0.0;
    maxRssKb = -1;
#endif

    /* Heap-counter reconciliation (StorageApi.h memProfile*; active only when
     * built with -DODT_MEM_PROFILE — the four mem_heap_peak_b/mem_dataset_
     * heap_b/mem_model_heap_b/mem_reconciliation_gap_b keys below all read 0
     * otherwise, never crash). mb.mcuTotal (computed above by
     * computeMemBudget) is a 7-term STATIC accounting — params + grads +
     * optstate + act + gradbuf + io + masks — a deliberately conservative
     * superset of upstream har_classifier's 5-term figure. heapPeak is the
     * MEASURED live-byte high-water mark across the whole process (dataset
     * copies, DataLoader scratch, per-op transients, everything mcuTotal
     * doesn't model). reconGap is recorded AS-IS, signed, and never tuned to
     * close it: positive = heap bytes unaccounted for by the static budget;
     * negative would mean the static budget over-counts something the heap
     * never actually touched (e.g. a freed scratch buffer). */
    size_t heapPeak = memProfilePeakBytes();
    long reconGap = (long)heapPeak - (long)mb.mcuTotal;

    printf("RESULT accuracy=%.6f test_acc=%.6f best_val_acc=%.6f best_epoch=%d n_params=%zu "
           "wall_clock_s=%.3f "
           "mem_params_b=%zu mem_grads_b=%zu mem_optstate_b=%zu mem_act_b=%zu "
           "mem_gradbuf_b=%zu mem_io_b=%zu mem_masks_b=%zu mem_mcu_total_b=%zu "
           "mem_dataset_host_b=%zu rss_peak_kb=%ld rss_after_data_kb=%ld "
           "rss_after_model_kb=%ld cpu_user_s=%.3f cpu_sys_s=%.3f stack_peak_b=%ld "
           "mem_heap_peak_b=%zu mem_dataset_heap_b=%zu mem_model_heap_b=%zu "
           "mem_reconciliation_gap_b=%ld\n",
           (double)ctx.finalAcc, (double)ctx.testAcc, (double)ctx.bestAcc, ctx.bestEpoch, nParams,
           mono_s() - t0, mb.params, mb.grads,
           mb.optstate, mb.act, mb.gradbuf, mb.io, mb.masks, mb.mcuTotal, mb.datasetHost,
           maxRssKb, rssAfterData, rssAfterModel, cpuU, cpuS, stackPeak, heapPeak, heapDataset,
           heapAfterOpt - heapBeforeModel, reconGap);

    /* Same probe keys as RESULT, mirrored to <ckpt>/memory.json; manifest.json
     * (written inside trainMain) is untouched. */
    if (ckptDir != NULL) {
        char memPath[512];
        snprintf(memPath, sizeof(memPath), "%s/memory.json", ckptDir);
        FILE *mjf = fopen(memPath, "w");
        if (mjf) {
            fprintf(mjf,
                    "{\n"
                    "  \"mem_params_b\": %zu,\n"
                    "  \"mem_grads_b\": %zu,\n"
                    "  \"mem_optstate_b\": %zu,\n"
                    "  \"mem_act_b\": %zu,\n"
                    "  \"mem_gradbuf_b\": %zu,\n"
                    "  \"mem_io_b\": %zu,\n"
                    "  \"mem_masks_b\": %zu,\n"
                    "  \"mem_mcu_total_b\": %zu,\n"
                    "  \"mem_dataset_host_b\": %zu,\n"
                    "  \"rss_peak_kb\": %ld,\n"
                    "  \"rss_after_data_kb\": %ld,\n"
                    "  \"rss_after_model_kb\": %ld,\n"
                    "  \"cpu_user_s\": %.6f,\n"
                    "  \"cpu_sys_s\": %.6f,\n"
                    "  \"stack_peak_b\": %ld,\n"
                    "  \"mem_heap_peak_b\": %zu,\n"
                    "  \"mem_dataset_heap_b\": %zu,\n"
                    "  \"mem_model_heap_b\": %zu,\n"
                    "  \"mem_reconciliation_gap_b\": %ld\n"
                    "}\n",
                    mb.params, mb.grads, mb.optstate, mb.act, mb.gradbuf, mb.io, mb.masks,
                    mb.mcuTotal, mb.datasetHost, maxRssKb, rssAfterData, rssAfterModel, cpuU,
                    cpuS, stackPeak, heapPeak, heapDataset, heapAfterOpt - heapBeforeModel,
                    reconGap);
            fclose(mjf);
        }
    }

    return 0;
}
