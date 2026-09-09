/* HOST backend: load the global SmaTable .npy + per-fold index .npy via
 * ODT's NPYLoader. Per fold there are up to four split index files
 * (fold_KK_train/retain/calib/test.npy); RETAIN and CALIB are optional and
 * report count 0 when their file is absent (schemes that don't define them,
 * e.g. AOS / 80_20). smatableDatasetGetSplit memcpys the addressed window
 * into the caller-supplied buffer; GetTrain/GetTest are thin wrappers.
 *
 * Env vars (all required, no defaults — fail loud):
 *   SMATABLE_DATA_DIR       directory containing smatable_x.npy + smatable_y.npy + folds/
 *   SMATABLE_FOLD_SCHEME    one of LOSO / AOS / 80_20
 *   SMATABLE_FOLD           integer fold index (0..n_folds-1 for the chosen scheme)
 *
 * Sample-cap parity with the baked backend: the .npy ID files are pre-capped
 * by the prep script (--max-samples-per-fold), so no env var is needed here.
 */

#define SOURCE_FILE "SMATABLE_DATASET_NPY"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "Common.h"
#include "Dataset.h"
#include "NPYLoader.h"
#include "NPYLoaderApi.h"
#include "Tensor.h"

#include "smatable_dataset.h"

/* Direct int32 .npy reader — bypasses ODT's tensorInit() because its INT32
 * code path treats the input buffer as float* and casts each element via
 * (int32_t)data[i], destroying int32 input. Confirmed by L1 test debugging
 * 2026-04-27 — see L1 fail trace where every fold-ID resolved to 0.
 *
 * Reads an .npy whose dtype is <i4 and shape is 1-D. Returns malloc'd int32
 * buffer (caller frees) and *outN = element count. */
static int32_t *readInt32NpyDirect(const char *path, size_t *outN) {
    FILE *f = openNPYFile((char *)path);
    checkMagic(f);
    uint32_t hdrSz = readHeaderSize(f);
    char *header = malloc(hdrSz + 1);
    if (!header) { fprintf(stderr, "[SMATABLE_DATASET_NPY] OOM header\n"); exit(1); }
    readHeader(header, hdrSz, f);

    dtype_t dt = getDTypeFromHeader(header);
    if (dt != INT_32) {
        fprintf(stderr, "[SMATABLE_DATASET_NPY] %s: expected <i4 dtype\n", path);
        exit(1);
    }
    size_t nd = getNumberOfDimsFromHeader(header);
    if (nd != 1) {
        fprintf(stderr, "[SMATABLE_DATASET_NPY] %s: expected 1-D shape, got %zu dims\n", path, nd);
        exit(1);
    }
    shape_t shape;
    size_t dims[1], order[1];
    getShapeFromHeader(&shape, dims, order, header, nd);
    size_t n = dims[0];

    int32_t *buf = malloc(n * sizeof(int32_t));
    if (!buf) { fprintf(stderr, "[SMATABLE_DATASET_NPY] OOM int32 buf\n"); exit(1); }
    size_t got = fread(buf, sizeof(int32_t), n, f);
    if (got != n) {
        fprintf(stderr, "[SMATABLE_DATASET_NPY] %s: short read %zu/%zu\n", path, got, n);
        exit(1);
    }
    fclose(f);
    free(header);
    *outN = n;
    return buf;
}

/* Symbol-suffix machinery so this backend can co-exist with the baked backend
 * inside the L1 equivalence test (tests/dataset_backend_equiv.c). Normal
 * single-backend builds leave SMATABLE_FN_SUFFIX undefined, so the public
 * symbols match the prototypes in smatable_dataset.h. */
#ifndef SMATABLE_FN_SUFFIX
# define SMATABLE_FN_SUFFIX
#endif
#define _SMA_CAT_INNER(a, b) a##b
#define _SMA_CAT(a, b) _SMA_CAT_INNER(a, b)
#define SMA(name) _SMA_CAT(name, SMATABLE_FN_SUFFIX)

struct smatable_dataset {
    tensorArray_t *x;       /* shape per row [C, T], float32 — npyLoad is correct on FLOAT32 */
    int32_t *y;             /* int32[N], read via readInt32NpyDirect */
    size_t nGlobal;
    int32_t *ids[SMATABLE_SPLIT_COUNT]; /* per split, NULL when absent */
    size_t n[SMATABLE_SPLIT_COUNT];     /* per split, 0 when absent */
    size_t nChannels;
    size_t windowSamples;
    size_t windowFloats;
};

static char *requireEnv(const char *name) {
    char *v = getenv(name);
    if (!v || !*v) {
        fprintf(stderr, "[SMATABLE_DATASET_NPY] missing env %s\n", name);
        exit(1);
    }
    return v;
}

static char *joinPath(const char *base, const char *suffix) {
    size_t n = strlen(base) + strlen(suffix) + 2;
    char *out = malloc(n);
    if (!out) {
        fprintf(stderr, "[SMATABLE_DATASET_NPY] OOM joining %s/%s\n", base, suffix);
        exit(1);
    }
    snprintf(out, n, "%s/%s", base, suffix);
    return out;
}

static char *foldFile(const char *dir, const char *scheme, int fold, const char *split) {
    char rel[128];
    snprintf(rel, sizeof(rel), "folds/%s/fold_%02d_%s.npy", scheme, fold, split);
    return joinPath(dir, rel);
}

smatable_dataset_t *SMA(smatableDatasetOpen)(void) {
    char *dir = requireEnv("SMATABLE_DATA_DIR");
    char *scheme = requireEnv("SMATABLE_FOLD_SCHEME");
    char *foldStr = requireEnv("SMATABLE_FOLD");

    char *endp = NULL;
    long fold = strtol(foldStr, &endp, 10);
    if (endp == foldStr || *endp || fold < 0) {
        fprintf(stderr, "[SMATABLE_DATASET_NPY] SMATABLE_FOLD must be a non-negative integer, got '%s'\n", foldStr);
        exit(1);
    }

    smatable_dataset_t *ds = calloc(1, sizeof(*ds));
    if (!ds) {
        fprintf(stderr, "[SMATABLE_DATASET_NPY] OOM allocating dataset\n");
        exit(1);
    }

    char *xPath = joinPath(dir, "smatable_x.npy");
    char *yPath = joinPath(dir, "smatable_y.npy");

    ds->x = npyLoad(xPath);
    ds->y = readInt32NpyDirect(yPath, &ds->nGlobal);

    free(xPath); free(yPath);

    for (int s = 0; s < SMATABLE_SPLIT_COUNT; s++) {
        char *path = foldFile(dir, scheme, (int)fold, smatableSplitName((smatable_split_t)s));
        bool required = (s == SMATABLE_SPLIT_TRAIN || s == SMATABLE_SPLIT_TEST);
        if (!required && access(path, F_OK) != 0) {
            ds->ids[s] = NULL;
            ds->n[s] = 0;
        } else {
            ds->ids[s] = readInt32NpyDirect(path, &ds->n[s]);
        }
        free(path);
    }

    if (ds->x->size == 0 || ds->nGlobal == 0) {
        fprintf(stderr, "[SMATABLE_DATASET_NPY] empty global x/y\n");
        exit(1);
    }
    if (ds->x->size != ds->nGlobal) {
        fprintf(stderr, "[SMATABLE_DATASET_NPY] x size %zu != y size %zu\n",
                ds->x->size, ds->nGlobal);
        exit(1);
    }

    /* Layout from the first row's shape: [C, T] */
    tensor_t *first = ds->x->array[0];
    if (!first || !first->shape || first->shape->numberOfDimensions != 2) {
        fprintf(stderr, "[SMATABLE_DATASET_NPY] expected per-row shape [C, T]\n");
        exit(1);
    }
    ds->nChannels     = first->shape->dimensions[0];
    ds->windowSamples = first->shape->dimensions[1];
    ds->windowFloats  = ds->nChannels * ds->windowSamples;

    return ds;
}

void SMA(smatableDatasetClose)(smatable_dataset_t *ds) {
    /* NPYLoader uses ODT's reserveMemory pool for the X tensors — we cannot
     * free per-tensor cleanly there. Our int32 buffers (y, per-split ids)
     * came from malloc and are freed here. The reserveMemory pool is
     * reclaimed at process exit. */
    free(ds->y);
    for (int s = 0; s < SMATABLE_SPLIT_COUNT; s++) {
        free(ds->ids[s]);
    }
    free(ds);
}

size_t SMA(smatableDatasetNChannels)    (const smatable_dataset_t *ds) { return ds->nChannels;    }
size_t SMA(smatableDatasetWindowSamples)(const smatable_dataset_t *ds) { return ds->windowSamples;}
size_t SMA(smatableDatasetWindowFloats) (const smatable_dataset_t *ds) { return ds->windowFloats; }
size_t SMA(smatableDatasetNClasses)     (const smatable_dataset_t *ds) { (void)ds; return SMATABLE_DATASET_N_CLASSES; }

static void getBy(const smatable_dataset_t *ds, const int32_t *ids, size_t n, size_t i,
                  float *outX, int32_t *outY) {
    if (i >= n) {
        fprintf(stderr, "[SMATABLE_DATASET_NPY] index %zu out of range (count=%zu)\n", i, n);
        exit(1);
    }
    int32_t globalId = ids[i];
    if (globalId < 0 || (size_t)globalId >= ds->x->size) {
        fprintf(stderr, "[SMATABLE_DATASET_NPY] fold id %d out of bounds (N=%zu)\n",
                globalId, ds->x->size);
        exit(1);
    }
    memcpy(outX, ds->x->array[globalId]->data, ds->windowFloats * sizeof(float));
    *outY = ds->y[globalId];
}

static void checkSplit(smatable_split_t split) {
    if ((int)split < 0 || split >= SMATABLE_SPLIT_COUNT) {
        fprintf(stderr, "[SMATABLE_DATASET_NPY] bad split id %d\n", (int)split);
        exit(1);
    }
}

size_t SMA(smatableDatasetSplitCount)(const smatable_dataset_t *ds, smatable_split_t split) {
    checkSplit(split);
    return ds->n[split];
}

void SMA(smatableDatasetGetSplit)(const smatable_dataset_t *ds, smatable_split_t split, size_t i,
                                  float *outX, int32_t *outY) {
    checkSplit(split);
    if (ds->ids[split] == NULL) {
        fprintf(stderr, "[SMATABLE_DATASET_NPY] split '%s' is absent from this dataset\n",
                smatableSplitName(split));
        exit(1);
    }
    getBy(ds, ds->ids[split], ds->n[split], i, outX, outY);
}

size_t SMA(smatableDatasetTrainCount)(const smatable_dataset_t *ds) { return ds->n[SMATABLE_SPLIT_TRAIN]; }
size_t SMA(smatableDatasetTestCount)(const smatable_dataset_t *ds) { return ds->n[SMATABLE_SPLIT_TEST]; }
void SMA(smatableDatasetGetTrain)(const smatable_dataset_t *ds, size_t i, float *outX, int32_t *outY) {
    SMA(smatableDatasetGetSplit)(ds, SMATABLE_SPLIT_TRAIN, i, outX, outY);
}
void SMA(smatableDatasetGetTest)(const smatable_dataset_t *ds, size_t i, float *outX, int32_t *outY) {
    SMA(smatableDatasetGetSplit)(ds, SMATABLE_SPLIT_TEST, i, outX, outY);
}
