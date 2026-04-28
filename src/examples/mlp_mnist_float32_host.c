/*
 * Example: mlp_mnist_float32_host  (HOST-only)
 *
 * Trains a 2-layer MLP (784 -> 20 -> 10, ReLU + Softmax + CrossEntropy) on the
 * full MNIST dataset loaded from .npy files, then emits the learned weights
 * as src/examples/data/mnist_pretrained_float32.h so the mnist_inference
 * example can flash the exact same network onto an MCU.
 *
 * Also writes a per-epoch CSV log (epoch, train_loss, eval_loss, test_accuracy)
 * for offline plotting against the PyTorch reference runs. Path comes from
 * the ODT_CSV_PATH env var and falls back to runs/mlp_mnist_float32_host_odt.csv.
 *
 * Data prerequisite: run `uv run src/examples/data/generate_subset.py` once
 * to create the .npy files under src/examples/data/.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>  // for clock() used to time training

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
#include "StorageApi.h"
#include "LossFunction.h"
#include "Linear.h"
#include "RNG.h"

#ifndef MNIST_DATA_DIR
#define MNIST_DATA_DIR "."
#endif
#define MNIST_TRAIN_X MNIST_DATA_DIR "/mnist_train_x.npy"
#define MNIST_TRAIN_Y MNIST_DATA_DIR "/mnist_train_y.npy"
#define MNIST_TEST_X  MNIST_DATA_DIR "/mnist_test_x.npy"
#define MNIST_TEST_Y  MNIST_DATA_DIR "/mnist_test_y.npy"

#define PRETRAINED_OUT_PATH "src/examples/data/mnist_pretrained_float32.h"
#define DEFAULT_CSV_PATH    "runs/mlp_mnist_float32_host_odt.csv"

#define INPUT_DIM    (28 * 28)
#define HIDDEN_DIM   20
#define OUTPUT_DIM   10
#define NUM_CLASSES  10
#define BATCH_SIZE   32
#define NUM_EPOCHS   50
#define LEARNING_RATE 0.001f
#define MODEL_SIZE   4

static dataset_t trainDataset;
static dataset_t testDataset;
static layer_t *model[MODEL_SIZE];
static dataLoader_t *testDL;
static const char *csvPath = DEFAULT_CSV_PATH;

static sample_t *getTrainSample(size_t id) { return npyGetSample(&trainDataset, id); }
static sample_t *getTestSample (size_t id) { return npyGetSample(&testDataset,  id); }
static size_t getTrainSize(void) { return trainDataset.items->size; }
static size_t getTestSize (void) { return testDataset.items->size;  }

/* Items in a fresh .npy MNIST load have shape [1, 28, 28]. The Linear layer
 * expects a 2D [1, 784] tensor, so rewrite the shape metadata in place. */
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

static void writeFloatArray(FILE *f, const char *name, const float *data, size_t n) {
    fprintf(f, "static const float %s[%zu] = {\n    ", name, n);
    for (size_t i = 0; i < n; i++) {
        fprintf(f, "%.9gf,", (double)data[i]);
        fprintf(f, ((i + 1) % 8 == 0 && i + 1 < n) ? "\n    " : " ");
    }
    fprintf(f, "\n};\n\n");
}

static void savePretrained(const char *path, layer_t *lin0, layer_t *lin1) {
    linearConfig_t *c0 = lin0->config->linear;
    linearConfig_t *c1 = lin1->config->linear;
    const float *w0 = (const float *)c0->weights->param->data;
    const float *b0 = (const float *)c0->bias->param->data;
    const float *w1 = (const float *)c1->weights->param->data;
    const float *b1 = (const float *)c1->bias->param->data;

    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "Failed to open %s for writing\n", path);
        return;
    }
    fprintf(f, "/* Generated by mlp_mnist_float32_host. Do not edit by hand. */\n");
    fprintf(f, "#ifndef MNIST_PRETRAINED_FLOAT32_H\n");
    fprintf(f, "#define MNIST_PRETRAINED_FLOAT32_H\n\n");
    fprintf(f, "#define MNIST_PRETRAINED_INPUT_DIM  %d\n", INPUT_DIM);
    fprintf(f, "#define MNIST_PRETRAINED_HIDDEN_DIM %d\n", HIDDEN_DIM);
    fprintf(f, "#define MNIST_PRETRAINED_OUTPUT_DIM %d\n\n", OUTPUT_DIM);
    writeFloatArray(f, "mnist_pretrained_w0", w0, HIDDEN_DIM * INPUT_DIM);
    writeFloatArray(f, "mnist_pretrained_b0", b0, HIDDEN_DIM);
    writeFloatArray(f, "mnist_pretrained_w1", w1, OUTPUT_DIM * HIDDEN_DIM);
    writeFloatArray(f, "mnist_pretrained_b1", b1, OUTPUT_DIM);
    fprintf(f, "#endif\n");
    fclose(f);
    printf("Wrote pretrained weights to %s\n", path);
}

static void csvInit(void) {
    const char *envPath = getenv("ODT_CSV_PATH");
    if (envPath && envPath[0] != '\0') csvPath = envPath;
    FILE *f = fopen(csvPath, "w");
    if (!f) {
        fprintf(stderr, "Could not open CSV log %s: %s\n", csvPath, "fopen failed");
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
    printf("mlp_mnist_float32_host: loading MNIST from %s\n", MNIST_DATA_DIR);

    /* ODT_SEED env var controls both DataLoader shuffle AND Xavier init —
     * dataLoaderInit calls rngSetSeed which then flows into tensorInitWithDistribution
     * via rngNextFloat. Default 42 matches the pre-seed-sweep baseline. */
    uint32_t odtSeed = 42;
    const char *seedEnv = getenv("ODT_SEED");
    if (seedEnv && seedEnv[0] != '\0') odtSeed = (uint32_t)atoi(seedEnv);
    printf("ODT_SEED=%u\n", (unsigned)odtSeed);

    trainDataset.items  = npyLoad(MNIST_TRAIN_X);
    trainDataset.labels = npyLoad(MNIST_TRAIN_Y);
    testDataset.items   = npyLoad(MNIST_TEST_X);
    testDataset.labels  = npyLoad(MNIST_TEST_Y);
    if (!trainDataset.items || !trainDataset.labels ||
        !testDataset.items  || !testDataset.labels) {
        fprintf(stderr, "Could not load MNIST .npy files — run generate_subset.py first.\n");
        return 1;
    }
    flattenItems(trainDataset.items);
    flattenItems(testDataset.items);

    dataLoader_t *trainDL = dataLoaderInit(getTrainSample, getTrainSize, BATCH_SIZE,
                                            NULL, NULL, true, odtSeed, true);
    testDL = dataLoaderInit(getTestSample, getTestSize, 1,
                             NULL, NULL, false, 0, true);

    quantization_t *q = quantizationInitFloat();
    buildModel(model, q);

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

    printf("Training done in %.2fs. final_train_loss=%.4f final_eval_loss=%.4f accuracy=%.2f%%\n",
           (double)(t1 - t0) / CLOCKS_PER_SEC,
           (double)res.finalTrainLoss, (double)res.finalEvalStats.loss,
           (double)accuracy * 100.0);

    /* Only write canonical pretrained.h when no seed override is in effect —
     * keeps the committed .h deterministic during seed-sweep comparisons. */
    if (!seedEnv) {
        savePretrained(PRETRAINED_OUT_PATH, model[0], model[2]);
    }
    return 0;
}
