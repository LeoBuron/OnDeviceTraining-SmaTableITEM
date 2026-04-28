/*
 * Example: mlp_mnist_asym_host  (HOST-only)
 *
 * Same 784 -> 20 -> 10 MLP as mlp_mnist_float32_host, but the Linear layers
 * use asymmetric 8-bit quantization for their *forward* pass (forwardQ=ASYM),
 * while gradients and backpropagated loss stay in FLOAT32 for numerical
 * stability. This is the "quantize-aware training forward" pattern: weights
 * and activations travel through an 8-bit bottleneck at inference time while
 * training remains in float.
 *
 * Trains on the full MNIST dataset from .npy files; no pretrained weight dump
 * (the asym-quantized numerics differ from the float variant, so deploying to
 * an 8-bit accelerator would want its own asym-specific export).
 */

#error "ASYM forward dispatch ist in ODT nicht implementiert — siehe https://github.com/es-ude/OnDeviceTraining/issues/61. Example bleibt als Vorlage stehen; sobald Issue #61 geschlossen ist, diesen #error entfernen."

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "hardware_init.h"

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
#include "NPYLoaderApi.h"
#include "Dataset.h"
#include "StorageApi.h"
#include "LossFunction.h"

#ifndef MNIST_DATA_DIR
#define MNIST_DATA_DIR "."
#endif
#define MNIST_TRAIN_X MNIST_DATA_DIR "/mnist_train_x.npy"
#define MNIST_TRAIN_Y MNIST_DATA_DIR "/mnist_train_y.npy"
#define MNIST_TEST_X  MNIST_DATA_DIR "/mnist_test_x.npy"
#define MNIST_TEST_Y  MNIST_DATA_DIR "/mnist_test_y.npy"

#define INPUT_DIM     (28 * 28)
#define HIDDEN_DIM    20
#define OUTPUT_DIM    10
#define NUM_CLASSES   10
#define BATCH_SIZE    32
#define NUM_EPOCHS    10
#define LEARNING_RATE 0.001f
#define MODEL_SIZE    4
#define ASYM_QBITS    8

static dataset_t trainDataset;
static dataset_t testDataset;

static sample_t *getTrainSample(size_t id) { return npyGetSample(&trainDataset, id); }
static sample_t *getTestSample (size_t id) { return npyGetSample(&testDataset,  id); }
static size_t getTrainSize(void) { return trainDataset.items->size; }
static size_t getTestSize (void) { return testDataset.items->size;  }

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

static void buildModel(layer_t **model, quantization_t *forwardQ, quantization_t *floatQ) {
    static float w0[HIDDEN_DIM * INPUT_DIM] = {0};
    static size_t w0Dims[] = {HIDDEN_DIM, INPUT_DIM};
    tensor_t *w0P = tensorInitWithDistribution(XAVIER_UNIFORM, w0, w0Dims, 2, floatQ, NULL, INPUT_DIM, HIDDEN_DIM);
    parameter_t *w0Pm = parameterInit(w0P, gradInitFloat(w0P, NULL));

    static float b0[HIDDEN_DIM] = {0};
    static size_t b0Dims[] = {1, HIDDEN_DIM};
    tensor_t *b0P = tensorInitWithDistribution(ZEROS, b0, b0Dims, 2, floatQ, NULL, 1, HIDDEN_DIM);
    parameter_t *b0Pm = parameterInit(b0P, gradInitFloat(b0P, NULL));

    /* forwardQ=ASYM, grads stay in float. The linear layer internally
     * quantizes its activation output into the forwardQ-typed buffer. */
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
    printf("mlp_mnist_asym_host: ASYM %d-bit forward + float grads\n", ASYM_QBITS);

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

    dataLoader_t *trainDL = dataLoaderInit(getTrainSample, getTrainSize, BATCH_SIZE,
                                            NULL, NULL, true, 42, true);
    dataLoader_t *testDL  = dataLoaderInit(getTestSample,  getTestSize,  1,
                                            NULL, NULL, false, 0, true);

    quantization_t *asymQ  = quantizationInitAsym(ASYM_QBITS, HTE);
    quantization_t *floatQ = quantizationInitFloat();

    layer_t *model[MODEL_SIZE];
    buildModel(model, asymQ, floatQ);

    optimizer_t *sgd = sgdMCreateOptim(LEARNING_RATE, 0.f, 0.f, model, MODEL_SIZE, FLOAT32);

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
    return 0;
}
