/*
 * Example: rq1_replay_buffer (HOST primary)
 *
 * RQ1: how much replay does on-device fine-tuning to a new user need to
 * avoid forgetting the original users? Two-domain, domain-incremental
 * protocol over one LOSO fold: domain 0 = pooled original users (fold's
 * `train` split), domain 1 = the held-out new user (fold's `test` split).
 * Three interchangeable replay-source arms, chosen via ODT_REPLAY_MODE:
 *
 *   ppca              per-class PPCA generative replay (upstream #326)
 *   exemplar_random   raw exemplar buffer, uniform-random selection
 *   exemplar_herding  raw exemplar buffer, iCaRL-style herding selection
 *   none              no replay -- the forgetting baseline
 *
 * All three non-"none" arms share ONE sweep axis, ODT_BUFFER_SIZE, an
 * iso-byte "replay budget" per class. For the exemplar arms it is the
 * buffer capacity directly; for ppca it is converted to a subspace rank
 * via rank = bufferSize - 1, which follows directly from upstream's own
 * per-class memory formula (docs/CONTINUAL_LEARNING.md: bytes_per_class ~=
 * (k+1)*d*b -- setting that equal to bufferSize*d*b, the dominant term for
 * an exemplar buffer, cancels d*b and leaves k = bufferSize - 1). The
 * *actual* achieved parity (not the approximation) is logged every run via
 * ppcaReplayBytes/ppcaReplayIsoExemplarCount, so analysis can verify it
 * rather than assume it.
 *
 * ODT_PPCA_RANK independently overrides the derived rank (ppca mode only),
 * clamped to [1, WF-1] regardless of source. This is a *separate* knob from
 * the iso-byte comparison above: sweeping it breaks parity with the
 * exemplar arms by design, so it belongs in its own PPCA-only study
 * (hpc/search_space/rq1_ppca_rank_sweep.json) answering "how sensitive is
 * PPCA replay quality to subspace rank," not in the main 4-arm grid.
 *
 * ODT_MAX_SESSION_SAMPLES (ppca mode only) caps how many raw samples get
 * absorbed into a class's PPCA generator per ppcaReplayUpdate call; classes
 * with more fit-samples than this are absorbed in multiple chunked calls.
 * Each call's cost is O(p^3) in p = rank + chunk + 1 (a Jacobi eigendecomp
 * of a p x p Gram matrix), so total absorption cost across all chunks for a
 * class scales as O(n * m^2) (n = fixed sample count, m = this knob) --
 * shrinking m is a large, non-approximate win (the Chan-Golub-LeVeque merge
 * is exact regardless of chunk size). It's also orthogonal to the iso-byte
 * comparison (doesn't change bytes_per_class or replay-time behavior, only
 * how the generator gets built), so like rank it gets its own PPCA-only
 * study rather than joining the main grid: hpc/search_space/
 * rq1_chunk_size_sweep.json, answering both "how expensive is this in
 * practice" and "does chunking granularity measurably affect replay
 * quality" (it shouldn't, per the exact-merge property above, but the
 * sweep verifies that rather than assuming it).
 *
 * Protocol (mirrors upstream docs/CONTINUAL_LEARNING.md's sequencing rule,
 * specialized to two domains):
 *   1. pretrain on domain-0-fit
 *   2. snapshot domain-0-held accuracy right after pretraining (the BWT
 *      "before" baseline -- GEM's rows[0][0])
 *   3. build the replay source from domain-0-fit (per ODT_REPLAY_MODE)
 *   4. fine-tune on domain-1-fit; after every real sample, inject
 *      ODT_R_PER_CLASS replay samples for every eligible class
 *   5. eval domain-0-held again ("after" -- GEM's rows[1][0], reported as
 *      `accuracy`, the RQ1 headline/Optuna-objective metric) and
 *      domain-1-held (`new_acc`, confirms fine-tuning actually worked)
 *
 * Each class's `fit` slice is only ever used for pretraining/replay-source
 * construction; `held` slices are never trained on -- held0 measures
 * retention, held1 measures new-user gain. This keeps RQ1 orthogonal to
 * RQ2 (unseen-user *sample-count* sweep): domain-1-fit size is fixed by
 * ODT_NEW_HELDOUT_FRAC, not swept here.
 *
 * Deliberately NOT using upstream's replayDataLoaderWrap/dataLoader_t
 * machinery: this repo's per-RQ convention (rq0_toy_synthetic.c,
 * stage1_pretrain.c) is a manual per-sample loop for auditability. Replay
 * samples are injected the same way -- one extra calculateGradsSequential +
 * step + zero per synthetic/exemplar sample, immediately after the real
 * sample it follows (B=1 throughout, consistent with the rest of this
 * repo's examples).
 *
 * Model: the same Linear->ReLU->Linear->Softmax MLP as rq0_toy_synthetic.c
 * -- a placeholder pending the per-RQ training-loop design sub-spec (the
 * real Conv1d/GroupNorm architecture is stage1_pretrain.c's DepthwiseCNN).
 * This binary's job is to exercise the three-arm replay wiring end to end
 * on real gesture data; it does not settle the final architecture.
 *
 * Stdout contract (read by hpc/run_optuna.py):
 *   BEGIN  rq1_replay_buffer <iso8601>
 *   EPOCH  <phase> <e> loss=<f> acc=<f>
 *   RESULT accuracy=<f> new_acc=<f> bwt=<f> replay_mode="<m>"
 *          buffer_size=<i> ppca_rank=<i> bytes_per_class=<i>
 *          iso_exemplar_count=<i> n_params=<i> wall_clock_s=<f>
 *
 * Env (all optional; defaults make standalone runs reproducible):
 *   ODT_REPLAY_MODE          ppca|exemplar_random|exemplar_herding|none  default ppca
 *   ODT_BUFFER_SIZE          iso-byte replay budget per class (see above)  default 8
 *   ODT_PPCA_RANK            override the derived rank, clamped [1,WF-1]  default bufferSize-1
 *                            (ppca mode only; see rq1_ppca_rank_sweep.json)
 *   ODT_R_PER_CLASS          replay samples injected per class per real sample  default 2
 *   ODT_MIN_COUNT            ppca eligibility gate (generator count >=)   default 2*rank
 *   ODT_MAX_SESSION_SAMPLES  ppca per-call absorption chunk size          default 64
 *                            (ppca mode only; see rq1_chunk_size_sweep.json --
 *                            cost scales ~quadratically with this value, see
 *                            file-header note above rank derivation)
 *   ODT_HIDDEN               model hidden width                          default 16
 *   ODT_LR                   learning rate (both phases)                 default 0.01
 *   ODT_SEED                 RNG seed                                    default 42
 *   ODT_PRETRAIN_EPOCHS      epochs over domain-0-fit                    default 10
 *   ODT_FT_EPOCHS            epochs over domain-1-fit                    default 5
 *   ODT_ORIG_HELDOUT_FRAC    domain-0 fraction held out for eval-only     default 0.2
 *   ODT_NEW_HELDOUT_FRAC     domain-1 fraction held out for eval-only     default 0.3
 *
 * Dataset env required by smatable_dataset_npy backend (HOST):
 *   SMATABLE_DATA_DIR, SMATABLE_FOLD_SCHEME, SMATABLE_FOLD
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "hardware_init.h"

#include "ArithmeticType.h"
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

#include "ExemplarBuffer.h"
#include "PpcaReplay.h"
#include "PpcaReplayApi.h"

#include "smatable_dataset.h"

#define MODEL_SIZE 4

typedef enum { REPLAY_NONE, REPLAY_PPCA, REPLAY_EXEMPLAR_RANDOM, REPLAY_EXEMPLAR_HERDING } rq1ReplayMode_t;

typedef struct {
    size_t *idx; /* split-local sample indices (train-local for domain 0, test-local for domain 1) */
    size_t n;
} classBucket_t;

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

static void iso8601_now(char *out, size_t n) {
    time_t t = time(NULL);
    struct tm *tm = gmtime(&t);
    strftime(out, n, "%Y-%m-%dT%H:%M:%SZ", tm);
}

static rq1ReplayMode_t parseReplayMode(const char *s) {
    if (strcmp(s, "ppca") == 0) return REPLAY_PPCA;
    if (strcmp(s, "exemplar_random") == 0) return REPLAY_EXEMPLAR_RANDOM;
    if (strcmp(s, "exemplar_herding") == 0) return REPLAY_EXEMPLAR_HERDING;
    if (strcmp(s, "none") == 0) return REPLAY_NONE;
    fprintf(stderr, "rq1_replay_buffer: unknown ODT_REPLAY_MODE '%s'\n", s);
    exit(1);
}
static const char *replayModeName(rq1ReplayMode_t m) {
    switch (m) {
        case REPLAY_PPCA: return "ppca";
        case REPLAY_EXEMPLAR_RANDOM: return "exemplar_random";
        case REPLAY_EXEMPLAR_HERDING: return "exemplar_herding";
        default: return "none";
    }
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
 * Used purely for printing -- the gradient path uses ODT's internal CE. */
static float ce_one(const tensor_t *softmax, int32_t label) {
    const float *p = (const float *)softmax->data;
    float py = p[(size_t)label];
    if (py < 1e-9f) py = 1e-9f;
    return -logf(py);
}

static tensor_t *buildTensor(size_t rows, size_t cols) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = rows;
    dims[1] = cols;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    return initTensor(shape, quantizationInitFloat(), NULL);
}

static void trainOneSample(layer_t **model, tensor_t *x_t, tensor_t *y_t, lossConfig_t lossConfig,
                           optimizer_t *sgd, optimizerFunctions_t sgdFns) {
    trainingStats_t *st =
        calculateGradsSequential(model, MODEL_SIZE, lossConfig, REDUCTION_MEAN, x_t, y_t);
    sgdFns.step(sgd);
    sgdFns.zero(sgd);
    freeTrainingStats(st);
}

/* Bucket every sample of a split (train=domain 0, test=domain 1) by label. */
static void bucketByClass(smatable_dataset_t *ds, int isTrain, size_t total, size_t NC, size_t WF,
                          float *tmp, classBucket_t *buckets) {
    size_t *counts = calloc(NC, sizeof(size_t));
    int32_t *labels = malloc(total * sizeof(int32_t));
    for (size_t i = 0; i < total; i++) {
        int32_t label;
        if (isTrain) smatableDatasetGetTrain(ds, i, tmp, &label);
        else smatableDatasetGetTest(ds, i, tmp, &label);
        labels[i] = label;
        counts[(size_t)label]++;
    }
    for (size_t c = 0; c < NC; c++) {
        buckets[c].idx = malloc((counts[c] ? counts[c] : 1) * sizeof(size_t));
        buckets[c].n = 0;
    }
    for (size_t i = 0; i < total; i++) {
        size_t c = (size_t)labels[i];
        buckets[c].idx[buckets[c].n++] = i;
    }
    free(labels);
    free(counts);
    (void)WF;
}

static size_t *flattenBuckets(classBucket_t *buckets, size_t NC, size_t *totalOut) {
    size_t total = 0;
    for (size_t c = 0; c < NC; c++) total += buckets[c].n;
    size_t *flat = malloc((total ? total : 1) * sizeof(size_t));
    size_t k = 0;
    for (size_t c = 0; c < NC; c++)
        for (size_t i = 0; i < buckets[c].n; i++) flat[k++] = buckets[c].idx[i];
    *totalOut = total;
    return flat;
}

static float evalAcc(smatable_dataset_t *ds, int isTrain, const size_t *idx, size_t n,
                     layer_t **model, tensor_t *x_t, float *tmp, size_t WF) {
    size_t correct = 0;
    for (size_t i = 0; i < n; i++) {
        int32_t label;
        if (isTrain) smatableDatasetGetTrain(ds, idx[i], tmp, &label);
        else smatableDatasetGetTest(ds, idx[i], tmp, &label);
        memcpy(x_t->data, tmp, WF * sizeof(float));
        tensor_t *out = inference(model, MODEL_SIZE, x_t);
        if (argmax(out) == label) correct++;
    }
    return n ? (float)correct / (float)n : 0.f;
}

/* iCaRL-style herding (Rebuffi et al. 2017): greedily grow the selected set
 * so its running mean tracks the class's true mean as closely as possible.
 * Operates on the same flattened WF-dim raw-window representation PPCA
 * uses, so the "features" both arms herd/generate over are the same
 * space. O(k * n * WF); fine for a HOST Optuna trial, not for MCU. */
static void herdingSelect(smatable_dataset_t *ds, const size_t *idx, size_t n, size_t WF, size_t k,
                          size_t *selectedOut, size_t *nSelectedOut) {
    if (n == 0) { *nSelectedOut = 0; return; }
    if (k > n) k = n;

    float *cache = malloc(n * WF * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        int32_t label;
        smatableDatasetGetTrain(ds, idx[i], cache + i * WF, &label);
    }
    float *classMean = calloc(WF, sizeof(float));
    for (size_t i = 0; i < n; i++)
        for (size_t d = 0; d < WF; d++) classMean[d] += cache[i * WF + d];
    for (size_t d = 0; d < WF; d++) classMean[d] /= (float)n;

    float *runningSum = calloc(WF, sizeof(float));
    uint8_t *taken = calloc(n, sizeof(uint8_t));
    for (size_t s = 0; s < k; s++) {
        size_t best = 0;
        float bestDist = INFINITY;
        for (size_t i = 0; i < n; i++) {
            if (taken[i]) continue;
            float dist = 0.f;
            for (size_t d = 0; d < WF; d++) {
                float cand = (runningSum[d] + cache[i * WF + d]) / (float)(s + 1);
                float diff = cand - classMean[d];
                dist += diff * diff;
            }
            if (dist < bestDist) { bestDist = dist; best = i; }
        }
        taken[best] = 1;
        for (size_t d = 0; d < WF; d++) runningSum[d] += cache[best * WF + d];
        selectedOut[s] = idx[best];
    }
    *nSelectedOut = k;
    free(taken);
    free(runningSum);
    free(classMean);
    free(cache);
}

static void buildExemplarBuffer(smatable_dataset_t *ds, size_t NC, size_t WF, classBucket_t *fit0,
                                int bufferSize, rq1ReplayMode_t mode, tensor_t *scratch,
                                exemplarBuffer_t *buf) {
    for (size_t c = 0; c < NC; c++) {
        size_t k = (size_t)bufferSize < fit0[c].n ? (size_t)bufferSize : fit0[c].n;
        size_t *chosen = NULL;
        size_t nChosen = 0;
        if (mode == REPLAY_EXEMPLAR_HERDING) {
            chosen = malloc((k ? k : 1) * sizeof(size_t));
            herdingSelect(ds, fit0[c].idx, fit0[c].n, WF, k, chosen, &nChosen);
        } else {
            /* fit0[c].idx was shuffled at the fit/held split, so its first k
             * entries already ARE a uniform random subset of the class. */
            chosen = fit0[c].idx;
            nChosen = k;
        }
        for (size_t i = 0; i < nChosen; i++) {
            int32_t label;
            smatableDatasetGetTrain(ds, chosen[i], (float *)scratch->data, &label);
            exemplarBufferAdd(buf, scratch, c);
        }
        if (mode == REPLAY_EXEMPLAR_HERDING) free(chosen);
    }
}

int main(void) {
    init();

    char ts[32];
    iso8601_now(ts, sizeof(ts));
    printf("BEGIN rq1_replay_buffer %s\n", ts);

    smatable_dataset_t *ds = smatableDatasetOpen();
    const size_t WF = smatableDatasetWindowFloats(ds);
    const size_t NC = smatableDatasetNClasses(ds);
    const size_t N_TRAIN = smatableDatasetTrainCount(ds); /* domain 0: original users */
    const size_t N_TEST = smatableDatasetTestCount(ds);   /* domain 1: new user */

    rq1ReplayMode_t replayMode = parseReplayMode(env_str("ODT_REPLAY_MODE", "ppca"));
    int bufferSize = env_int("ODT_BUFFER_SIZE", 8);
    int rankDefault = bufferSize - 1;
    if (rankDefault < 2) rankDefault = 2;
    int rank = env_int("ODT_PPCA_RANK", rankDefault);
    if (rank < 1) rank = 1;
    if ((size_t)rank >= WF) rank = (WF > 1) ? (int)WF - 1 : 1;
    int rPerClass = env_int("ODT_R_PER_CLASS", 2);
    int minCount = env_int("ODT_MIN_COUNT", 2 * rank);
    int maxSessionSamples = env_int("ODT_MAX_SESSION_SAMPLES", 64);

    int hidden = env_int("ODT_HIDDEN", 16);
    float lr = env_float("ODT_LR", 0.01f);
    uint32_t seed = (uint32_t)env_int("ODT_SEED", 42);
    int pretrainEpochs = env_int("ODT_PRETRAIN_EPOCHS", 10);
    int ftEpochs = env_int("ODT_FT_EPOCHS", 5);
    float origHeldoutFrac = env_float("ODT_ORIG_HELDOUT_FRAC", 0.2f);
    float newHeldoutFrac = env_float("ODT_NEW_HELDOUT_FRAC", 0.3f);

    rngSetSeed(seed);

    printf("  dataset: domain0(orig)=%zu domain1(new)=%zu window_floats=%zu n_classes=%zu\n",
           N_TRAIN, N_TEST, WF, NC);
    printf("  replay: mode=%s buffer_size=%d ppca_rank=%d r_per_class=%d min_count=%d\n",
           replayModeName(replayMode), bufferSize, rank, rPerClass, minCount);
    printf("  hyperparams: hidden=%d lr=%.5f seed=%u pretrain_epochs=%d ft_epochs=%d\n",
           hidden, (double)lr, (unsigned)seed, pretrainEpochs, ftEpochs);

    float *tmp = calloc(WF, sizeof(float));

    /* Bucket both domains by class, then split each class into a `fit`
     * slice (pretraining / replay-source construction) and a `held` slice
     * (eval-only, never trained on). Shuffling here means fit0[c].idx's
     * first k entries are already a uniform random subset -- reused
     * directly by the exemplar_random arm below. */
    classBucket_t buckets0[SMATABLE_DATASET_N_CLASSES], buckets1[SMATABLE_DATASET_N_CLASSES];
    bucketByClass(ds, 1, N_TRAIN, NC, WF, tmp, buckets0);
    bucketByClass(ds, 0, N_TEST, NC, WF, tmp, buckets1);

    classBucket_t fit0[SMATABLE_DATASET_N_CLASSES], held0[SMATABLE_DATASET_N_CLASSES];
    classBucket_t fit1[SMATABLE_DATASET_N_CLASSES], held1[SMATABLE_DATASET_N_CLASSES];
    for (size_t c = 0; c < NC; c++) {
        rngShuffleIndices(buckets0[c].idx, buckets0[c].n);
        size_t nHeld0 = (size_t)((float)buckets0[c].n * origHeldoutFrac);
        fit0[c].n = buckets0[c].n - nHeld0;
        fit0[c].idx = buckets0[c].idx;
        held0[c].n = nHeld0;
        held0[c].idx = buckets0[c].idx + fit0[c].n;

        rngShuffleIndices(buckets1[c].idx, buckets1[c].n);
        size_t nHeld1 = (size_t)((float)buckets1[c].n * newHeldoutFrac);
        fit1[c].n = buckets1[c].n - nHeld1;
        fit1[c].idx = buckets1[c].idx;
        held1[c].n = nHeld1;
        held1[c].idx = buckets1[c].idx + fit1[c].n;
    }

    size_t nFit0, nHeld0Total, nFit1, nHeld1Total;
    size_t *fit0Flat = flattenBuckets(fit0, NC, &nFit0);
    size_t *held0Flat = flattenBuckets(held0, NC, &nHeld0Total);
    size_t *fit1Flat = flattenBuckets(fit1, NC, &nFit1);
    size_t *held1Flat = flattenBuckets(held1, NC, &nHeld1Total);

    /* --- Model: Linear(WF->hidden) -> ReLU -> Linear(hidden->NC) -> Softmax --- */
    quantization_t *q = quantizationInitFloat();
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

    optimizer_t *sgd = sgdMCreateOptim(
        lr, 0.f, 0.f, model, MODEL_SIZE, quantizationInitFloat(),
        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    optimizerFunctions_t sgdFns = optimizerFunctions[SGD_M];
    size_t n_params = (size_t)hidden * WF + (size_t)hidden + (size_t)hidden * NC + NC;

    lossConfig_t lossConfig = {
        .funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_MEAN, .classWeights = NULL};

    tensor_t *x_t = buildTensor(1, WF);
    tensor_t *y_t = buildTensor(1, NC);
    tensor_t *syn_x_t = buildTensor(1, WF);
    tensor_t *syn_y_t = buildTensor(1, NC);

    clock_t t0 = clock();

    /* --- Phase 1: pretrain on domain-0-fit --- */
    for (int e = 0; e < pretrainEpochs; e++) {
        rngShuffleIndices(fit0Flat, nFit0);
        float epoch_loss = 0.f;
        size_t epoch_correct = 0;
        for (size_t i = 0; i < nFit0; i++) {
            int32_t label;
            smatableDatasetGetTrain(ds, fit0Flat[i], tmp, &label);
            memcpy(x_t->data, tmp, WF * sizeof(float));
            memset(y_t->data, 0, NC * sizeof(float));
            ((float *)y_t->data)[(size_t)label] = 1.f;

            trainOneSample(model, x_t, y_t, lossConfig, sgd, sgdFns);

            tensor_t *out = inference(model, MODEL_SIZE, x_t);
            if (argmax(out) == label) epoch_correct++;
            epoch_loss += ce_one(out, label);
        }
        printf("EPOCH pretrain %d loss=%.4f acc=%.4f\n", e + 1,
               (double)(epoch_loss / (float)nFit0), (double)((float)epoch_correct / (float)nFit0));
    }

    /* BWT baseline: domain-0 accuracy right after pretraining, before any
     * fine-tuning on domain 1 has touched the model. */
    float origAccBefore = evalAcc(ds, 1, held0Flat, nHeld0Total, model, x_t, tmp, WF);
    printf("  orig_acc_before_finetune=%.4f (n=%zu)\n", (double)origAccBefore, nHeld0Total);

    /* --- Phase 2: build the replay source from domain-0-fit --- */
    ppcaReplaySet_t *ppcaSet = NULL;
    exemplarBuffer_t *exemplars = NULL;
    rng32_t ppcaRng = {.state = seed * 2654435761u | 1u};
    size_t bytesPerClass = 0;
    size_t isoExemplarCount = 0;

    if (replayMode == REPLAY_PPCA) {
        ppcaReplayConfig_t cfg = {
            .dim = WF,
            .rank = (size_t)rank,
            .maxSessionSamples = (size_t)maxSessionSamples,
            .mergeMath = {.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
            .streamMath = {.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
            .sampleMath = {.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
            .meanQ = quantizationInitFloat(),
            .basisQ = quantizationInitFloat(),
            .eigvalsQ = quantizationInitFloat(),
            .sigma2Floor = 1e-6f,
            .shrinkageGamma = 0.0f,
        };
        ppcaSet = ppcaReplaySetCreate(NC, &cfg);

        for (size_t c = 0; c < NC; c++) {
            size_t done = 0;
            while (done < fit0[c].n) {
                size_t m = fit0[c].n - done;
                if (m > (size_t)maxSessionSamples) m = (size_t)maxSessionSamples;
                tensor_t *chunk = buildTensor(m, WF);
                for (size_t r = 0; r < m; r++) {
                    int32_t label;
                    smatableDatasetGetTrain(ds, fit0[c].idx[done + r], (float *)chunk->data + r * WF,
                                            &label);
                }
                ppcaReplayUpdate(ppcaSet->generators[c], chunk, ppcaSet->workspace);
                freeTensor(chunk);
                done += m;
            }
        }
        bytesPerClass = ppcaReplayBytes(ppcaSet->generators[0]);
        isoExemplarCount = ppcaReplayIsoExemplarCount(ppcaSet->generators[0], WF * sizeof(float));
    } else if (replayMode == REPLAY_EXEMPLAR_RANDOM || replayMode == REPLAY_EXEMPLAR_HERDING) {
        exemplars = exemplarBufferCreate(NC, (size_t)bufferSize);
        buildExemplarBuffer(ds, NC, WF, fit0, bufferSize, replayMode, syn_x_t, exemplars);
        bytesPerClass = (size_t)bufferSize * WF * sizeof(float);
        isoExemplarCount = (size_t)bufferSize;
    }

    /* --- Phase 3: fine-tune on domain-1-fit, replay injected per sample --- */
    for (int e = 0; e < ftEpochs; e++) {
        rngShuffleIndices(fit1Flat, nFit1);
        float epoch_loss = 0.f;
        size_t epoch_correct = 0;
        for (size_t i = 0; i < nFit1; i++) {
            int32_t label;
            smatableDatasetGetTest(ds, fit1Flat[i], tmp, &label);
            memcpy(x_t->data, tmp, WF * sizeof(float));
            memset(y_t->data, 0, NC * sizeof(float));
            ((float *)y_t->data)[(size_t)label] = 1.f;

            trainOneSample(model, x_t, y_t, lossConfig, sgd, sgdFns);

            tensor_t *out = inference(model, MODEL_SIZE, x_t);
            if (argmax(out) == label) epoch_correct++;
            epoch_loss += ce_one(out, label);

            if (replayMode == REPLAY_NONE) continue;
            for (size_t c = 0; c < NC; c++) {
                int eligible = (replayMode == REPLAY_PPCA)
                                  ? (ppcaSet->generators[c]->count >= (uint32_t)minCount)
                                  : (exemplars->counts[c] > 0);
                if (!eligible) continue;
                for (int r = 0; r < rPerClass; r++) {
                    if (replayMode == REPLAY_PPCA) {
                        ppcaReplaySample(ppcaSet->generators[c], &ppcaRng, syn_x_t);
                    } else {
                        uint32_t idx = (uint32_t)(rngNextFloat() * (float)exemplars->counts[c]);
                        if (idx >= exemplars->counts[c]) idx = exemplars->counts[c] - 1;
                        memcpy(syn_x_t->data, exemplars->items[c * (size_t)bufferSize + idx]->data,
                               WF * sizeof(float));
                    }
                    memset(syn_y_t->data, 0, NC * sizeof(float));
                    ((float *)syn_y_t->data)[c] = 1.f;
                    trainOneSample(model, syn_x_t, syn_y_t, lossConfig, sgd, sgdFns);
                }
            }
        }
        printf("EPOCH finetune %d loss=%.4f acc=%.4f\n", e + 1,
               (double)(epoch_loss / (float)nFit1), (double)((float)epoch_correct / (float)nFit1));
    }

    /* --- Phase 4: final eval --- */
    float origAccAfter = evalAcc(ds, 1, held0Flat, nHeld0Total, model, x_t, tmp, WF);
    float newAcc = evalAcc(ds, 0, held1Flat, nHeld1Total, model, x_t, tmp, WF);
    float bwt = origAccAfter - origAccBefore;

    clock_t t1 = clock();
    float wall = (float)(t1 - t0) / (float)CLOCKS_PER_SEC;

    printf("RESULT accuracy=%.4f new_acc=%.4f bwt=%+.4f replay_mode=\"%s\" buffer_size=%d "
           "ppca_rank=%d bytes_per_class=%zu iso_exemplar_count=%zu n_params=%zu wall_clock_s=%.3f\n",
           (double)origAccAfter, (double)newAcc, (double)bwt, replayModeName(replayMode), bufferSize,
           rank, bytesPerClass, isoExemplarCount, n_params, (double)wall);

    smatableDatasetClose(ds);
    return 0;
}
