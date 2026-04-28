/*
 * Example: mlp_mnist_float32_mcu
 *
 * Trains the same 784 -> 20 -> 10 MLP as mlp_mnist_float32_host but on an
 * embedded MNIST subset (compile-time byte arrays in data/mnist_*_subset.h).
 * batchSize is fixed to 1 because the sample callback reuses a single RAM
 * buffer for the float-normalised image — multi-sample batches would
 * overwrite each other.
 *
 * Also writes a per-epoch CSV log (epoch, train_loss, eval_loss, test_accuracy).
 * Path comes from ODT_CSV_PATH, default runs/mlp_mnist_float32_mcu_odt.csv.
 * On MCU, `runs/` must exist and be writable; on HOST builds compare.py
 * handles that.
 *
 * Runs on HOST too (useful for quickly validating the subset generator).
 * On MCU expect several minutes of training for the default NUM_EPOCHS — the
 * point is to demonstrate the flow, not to compete with the host run.
 *
 * Prerequisite: run generate_subset.py once so data/mnist_train_subset.h
 * and data/mnist_test_subset.h contain real arrays.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#include "hardware_init.h"
#include "debug_lib.h"

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
#include "Dataset.h"
#include "LossFunction.h"
#include "RNG.h"

#include "data/mnist_train_subset.h"
#include "data/mnist_test_subset.h"

#define INPUT_DIM     (28 * 28)
#define HIDDEN_DIM    20
#define OUTPUT_DIM    10
#define NUM_CLASSES   10
#define BATCH_SIZE    1
#define NUM_EPOCHS    50
#define LEARNING_RATE 0.01f
#define MODEL_SIZE    4

#define DEFAULT_CSV_PATH "runs/mlp_mnist_float32_mcu_odt.csv"

/* Reusable per-call buffers. Safe only while BATCH_SIZE == 1.
 * Tensor structs are created once and reused — only the underlying data
 * gets overwritten. DO NOT call freeTensor on these; the data points to
 * static BSS, not heap. */
static float itemBuf[INPUT_DIM];
static float labelBuf[OUTPUT_DIM];
static size_t itemDims[]  = {1, INPUT_DIM};
static size_t labelDims[] = {1, OUTPUT_DIM};
static tensor_t *curItem  = NULL;
static tensor_t *curLabel = NULL;
static sample_t  curSample;

static layer_t *model[MODEL_SIZE];
static dataLoader_t *testDL;
static const char *csvPath = DEFAULT_CSV_PATH;

static void initSampleTensors(void) {
    curItem  = tensorInitFloat(itemBuf,  itemDims,  2, NULL);
    curLabel = tensorInitFloat(labelBuf, labelDims, 2, NULL);
    curSample.item  = curItem;
    curSample.label = curLabel;
}

static sample_t *buildSample(const uint8_t *images, const uint8_t *labels, size_t id) {
    for (size_t i = 0; i < INPUT_DIM; i++) {
        itemBuf[i] = (float)images[id * INPUT_DIM + i] / 255.0f;
    }
    for (size_t i = 0; i < OUTPUT_DIM; i++) labelBuf[i] = 0.0f;
    labelBuf[labels[id]] = 1.0f;
    return &curSample;
}

static sample_t *getTrainSample(size_t id) {
    return buildSample(mnist_train_subset_images, mnist_train_subset_labels, id);
}

static sample_t *getTestSample(size_t id) {
    return buildSample(mnist_test_subset_images, mnist_test_subset_labels, id);
}

static size_t getTrainSize(void) { return MNIST_TRAIN_SUBSET_SIZE; }
static size_t getTestSize(void)  { return MNIST_TEST_SUBSET_SIZE;  }

static void buildModel(layer_t **m, quantization_t *q) {
    static float w0[HIDDEN_DIM * INPUT_DIM] = {0};
    static size_t w0Dims[] = {HIDDEN_DIM, INPUT_DIM};
    tensor_t *w0P = tensorInitWithDistribution(XAVIER_UNIFORM, w0, w0Dims, 2, q, NULL, INPUT_DIM, HIDDEN_DIM);
    tensor_t *w0G = gradInitFloat(w0P, NULL);
    parameter_t *w0Pm = parameterInit(w0P, w0G);

    static float b0[HIDDEN_DIM] = {0};
    static size_t b0Dims[] = {1, HIDDEN_DIM};
    tensor_t *b0P = tensorInitWithDistribution(ZEROS, b0, b0Dims, 2, q, NULL, 1, HIDDEN_DIM);
    tensor_t *b0G = gradInitFloat(b0P, NULL);
    parameter_t *b0Pm = parameterInit(b0P, b0G);

    m[0] = linearLayerInit(w0Pm, b0Pm, q, q, q, q);
    m[1] = reluLayerInit(q, q);

    static float w1[OUTPUT_DIM * HIDDEN_DIM] = {0};
    static size_t w1Dims[] = {OUTPUT_DIM, HIDDEN_DIM};
    tensor_t *w1P = tensorInitWithDistribution(XAVIER_UNIFORM, w1, w1Dims, 2, q, NULL, HIDDEN_DIM, OUTPUT_DIM);
    tensor_t *w1G = gradInitFloat(w1P, NULL);
    parameter_t *w1Pm = parameterInit(w1P, w1G);

    static float b1[OUTPUT_DIM] = {0};
    static size_t b1Dims[] = {1, OUTPUT_DIM};
    tensor_t *b1P = tensorInitWithDistribution(ZEROS, b1, b1Dims, 2, q, NULL, 1, OUTPUT_DIM);
    tensor_t *b1G = gradInitFloat(b1P, NULL);
    parameter_t *b1Pm = parameterInit(b1P, b1G);

    m[2] = linearLayerInit(w1Pm, b1Pm, q, q, q, q);
    m[3] = softmaxLayerInit(q, q);
}

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

int main(void) {
    init();
    debug_sleep(1000);
    printf("mlp_mnist_float32_mcu: %d train + %d test samples\n",
           MNIST_TRAIN_SUBSET_SIZE, MNIST_TEST_SUBSET_SIZE);

    /* ODT_SEED: same-seed cascade as host.c (see comment there). */
    uint32_t odtSeed = 42;
    const char *seedEnv = getenv("ODT_SEED");
    if (seedEnv && seedEnv[0] != '\0') odtSeed = (uint32_t)atoi(seedEnv);
    printf("ODT_SEED=%u\n", (unsigned)odtSeed);

    initSampleTensors();

    dataLoader_t *trainDL = dataLoaderInit(getTrainSample, getTrainSize, BATCH_SIZE,
                                            NULL, NULL, true, odtSeed, true);
    testDL = dataLoaderInit(getTestSample, getTestSize, BATCH_SIZE,
                             NULL, NULL, false, 0, true);

    quantization_t *q = quantizationInitFloat();
    buildModel(model, q);

    optimizer_t *sgd = sgdMCreateOptim(LEARNING_RATE, 0.f, 0.f, model, MODEL_SIZE, FLOAT32);

    csvInit();

    lossConfig_t lossConfig = { .funcType = CROSS_ENTROPY, .reduction = REDUCTION_MEAN };
    trainingRunResult_t res = trainingRun(
        model, MODEL_SIZE, lossConfig,
        trainDL, testDL, sgd, NUM_EPOCHS,
        calculateGradsSequential, inferenceWithLoss, onEpochEnd);

    float accuracy = evaluationEpochWithMetrics(
        model, MODEL_SIZE, CROSS_ENTROPY, testDL, inferenceWithLoss).accuracy;

    printf("Done. train_loss=%.4f eval_loss=%.4f subset_accuracy=%.2f%%\n",
           (double)res.finalTrainLoss, (double)res.finalEvalStats.loss,
           (double)accuracy * 100.0);
    return 0;
}
