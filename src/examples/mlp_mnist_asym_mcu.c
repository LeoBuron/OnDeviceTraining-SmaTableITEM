/*
 * Example: mlp_mnist_asym_mcu
 *
 * MCU-portable variant of mlp_mnist_asym_host: same asymmetric-forward pattern
 * (8-bit ASYM forwardQ, float gradients), but driven by the embedded MNIST
 * subset instead of .npy files. batchSize=1 as in mlp_mnist_float32_mcu.
 *
 * Runs on HOST too for a quick sanity check of the subset generator.
 */

#error "ASYM forward dispatch ist in ODT nicht implementiert — siehe https://github.com/es-ude/OnDeviceTraining/issues/61. Example bleibt als Vorlage stehen; sobald Issue #61 geschlossen ist, diesen #error entfernen."

#include <stdio.h>
#include <stddef.h>

#include "hardware_init.h"
#include "debug_lib.h"

#include "Layer.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "QuantizationApi.h"
#include "Quantization.h"
#include "Rounding.h"
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

#include "data/mnist_train_subset.h"
#include "data/mnist_test_subset.h"

#define INPUT_DIM     (28 * 28)
#define HIDDEN_DIM    20
#define OUTPUT_DIM    10
#define NUM_CLASSES   10
#define BATCH_SIZE    1
#define NUM_EPOCHS    3
#define LEARNING_RATE 0.01f
#define MODEL_SIZE    4
#define ASYM_QBITS    8

static float itemBuf[INPUT_DIM];
static float labelBuf[OUTPUT_DIM];
static size_t itemDims[]  = {1, INPUT_DIM};
static size_t labelDims[] = {1, OUTPUT_DIM};
static tensor_t *curItem  = NULL;
static tensor_t *curLabel = NULL;
static sample_t  curSample;

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

static void buildModel(layer_t **model, quantization_t *forwardQ, quantization_t *floatQ) {
    static float w0[HIDDEN_DIM * INPUT_DIM] = {0};
    static size_t w0Dims[] = {HIDDEN_DIM, INPUT_DIM};
    tensor_t *w0P = tensorInitWithDistribution(XAVIER_UNIFORM, w0, w0Dims, 2, floatQ, NULL, INPUT_DIM, HIDDEN_DIM);
    parameter_t *w0Pm = parameterInit(w0P, gradInitFloat(w0P, NULL));

    static float b0[HIDDEN_DIM] = {0};
    static size_t b0Dims[] = {1, HIDDEN_DIM};
    tensor_t *b0P = tensorInitWithDistribution(ZEROS, b0, b0Dims, 2, floatQ, NULL, 1, HIDDEN_DIM);
    parameter_t *b0Pm = parameterInit(b0P, gradInitFloat(b0P, NULL));

    model[0] = linearLayerInit(w0Pm, b0Pm, forwardQ, floatQ, floatQ, floatQ);
    model[1] = reluLayerInit(forwardQ, floatQ);

    static float w1[OUTPUT_DIM * HIDDEN_DIM] = {0};
    static size_t w1Dims[] = {OUTPUT_DIM, HIDDEN_DIM};
    tensor_t *w1P = tensorInitWithDistribution(XAVIER_UNIFORM, w1, w1Dims, 2, floatQ, NULL, HIDDEN_DIM, OUTPUT_DIM);
    parameter_t *w1Pm = parameterInit(w1P, gradInitFloat(w1P, NULL));

    static float b1[OUTPUT_DIM] = {0};
    static size_t b1Dims[] = {1, OUTPUT_DIM};
    tensor_t *b1P = tensorInitWithDistribution(ZEROS, b1, b1Dims, 2, floatQ, NULL, 1, OUTPUT_DIM);
    parameter_t *b1Pm = parameterInit(b1P, gradInitFloat(b1P, NULL));

    model[2] = linearLayerInit(w1Pm, b1Pm, forwardQ, floatQ, floatQ, floatQ);
    model[3] = softmaxLayerInit(floatQ, floatQ);
}

static void onEpochEnd(size_t epoch, float trainLoss, epochStats_t evalStats) {
    printf("  epoch %zu: train_loss=%.4f eval_loss=%.4f\n",
           epoch + 1, (double)trainLoss, (double)evalStats.loss);
}

int main(void) {
    init();
    debug_sleep(1000);
    printf("mlp_mnist_asym_mcu: %d-bit ASYM forward, %d train + %d test samples\n",
           ASYM_QBITS, MNIST_TRAIN_SUBSET_SIZE, MNIST_TEST_SUBSET_SIZE);

    initSampleTensors();

    dataLoader_t *trainDL = dataLoaderInit(getTrainSample, getTrainSize, BATCH_SIZE,
                                            NULL, NULL, true, 42, true);
    dataLoader_t *testDL  = dataLoaderInit(getTestSample,  getTestSize,  BATCH_SIZE,
                                            NULL, NULL, false, 0, true);

    quantization_t *asymQ  = quantizationInitAsym(ASYM_QBITS, HTE);
    quantization_t *floatQ = quantizationInitFloat();
    layer_t *model[MODEL_SIZE];
    buildModel(model, asymQ, floatQ);

    optimizer_t *sgd = sgdMCreateOptim(LEARNING_RATE, 0.f, 0.f, model, MODEL_SIZE, FLOAT32);

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
