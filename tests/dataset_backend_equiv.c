/* L1 equivalence test: HOST-NPY backend vs MCU-baked backend, bytewise.
 *
 * Both backends are compiled into THIS binary with disambiguating function
 * suffixes (-DSMATABLE_FN_SUFFIX=_npy / _baked) so we can call each through
 * its own entry point. Normal builds still link exactly one backend with the
 * unsuffixed names — the suffix scheme is only invoked here.
 *
 * What this proves:
 *   1. Every split the fixture carries (train, retain, calib, test) returns
 *      BYTEWISE equal float[] for every i, on the same fold; absent splits
 *      (retain/calib on schemes that don't define them) are absent in both
 *      backends.
 *   2. Per-split counts agree.
 *   3. Layout helpers (n_channels, window_samples, …) agree.
 *   4. The legacy TrainCount/TestCount/GetTrain/GetTest wrappers alias
 *      SMATABLE_SPLIT_TRAIN / SMATABLE_SPLIT_TEST on both backends.
 *
 * What this does NOT prove: training equivalence end-to-end (that is L2,
 * via state_dump_compare.py + the audit harness). A passing L1 means any
 * remaining HOST-vs-MCU divergence is *not* in the dataset path.
 *
 * Required env (NPY backend reads them — baked is compile-time only):
 *   SMATABLE_DATA_DIR
 *   SMATABLE_FOLD_SCHEME = LOSO   (must match the baked headers)
 *   SMATABLE_FOLD        = 0      (must match the baked headers)
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "smatable_dataset.h"

/* The two backends were compiled with SMATABLE_FN_SUFFIX=_npy / _baked, so
 * their public functions are exported with those suffixes. Forward-declare. */
smatable_dataset_t *smatableDatasetOpen_npy(void);
void                smatableDatasetClose_npy(smatable_dataset_t *);
size_t              smatableDatasetTrainCount_npy(const smatable_dataset_t *);
size_t              smatableDatasetTestCount_npy (const smatable_dataset_t *);
size_t              smatableDatasetNChannels_npy    (const smatable_dataset_t *);
size_t              smatableDatasetWindowSamples_npy(const smatable_dataset_t *);
size_t              smatableDatasetWindowFloats_npy (const smatable_dataset_t *);
size_t              smatableDatasetNClasses_npy     (const smatable_dataset_t *);
void                smatableDatasetGetTrain_npy(const smatable_dataset_t *, size_t, float *, int32_t *);
void                smatableDatasetGetTest_npy (const smatable_dataset_t *, size_t, float *, int32_t *);
size_t smatableDatasetSplitCount_npy  (const smatable_dataset_t *, smatable_split_t);
void   smatableDatasetGetSplit_npy  (const smatable_dataset_t *, smatable_split_t, size_t, float *, int32_t *);

smatable_dataset_t *smatableDatasetOpen_baked(void);
void                smatableDatasetClose_baked(smatable_dataset_t *);
size_t              smatableDatasetTrainCount_baked(const smatable_dataset_t *);
size_t              smatableDatasetTestCount_baked (const smatable_dataset_t *);
size_t              smatableDatasetNChannels_baked    (const smatable_dataset_t *);
size_t              smatableDatasetWindowSamples_baked(const smatable_dataset_t *);
size_t              smatableDatasetWindowFloats_baked (const smatable_dataset_t *);
size_t              smatableDatasetNClasses_baked     (const smatable_dataset_t *);
void                smatableDatasetGetTrain_baked(const smatable_dataset_t *, size_t, float *, int32_t *);
void                smatableDatasetGetTest_baked (const smatable_dataset_t *, size_t, float *, int32_t *);
size_t smatableDatasetSplitCount_baked(const smatable_dataset_t *, smatable_split_t);
void   smatableDatasetGetSplit_baked(const smatable_dataset_t *, smatable_split_t, size_t, float *, int32_t *);

#define DIE(msg) do { fprintf(stderr, "L1 FAIL: %s\n", msg); exit(1); } while (0)
#define EQ(label, a, b) do { \
    size_t _a = (a), _b = (b); \
    if (_a != _b) { fprintf(stderr, "L1 FAIL: %s: npy=%zu baked=%zu\n", label, _a, _b); exit(1); } \
} while (0)

static int compare_named_split(smatable_split_t split, smatable_dataset_t *npy,
                               smatable_dataset_t *baked, size_t window_floats) {
    const char *name = smatableSplitName(split);
    size_t n_npy = smatableDatasetSplitCount_npy(npy, split);
    size_t n_baked = smatableDatasetSplitCount_baked(baked, split);
    if (n_npy != n_baked) {
        fprintf(stderr, "L1 FAIL: %s count: npy=%zu baked=%zu\n", name, n_npy, n_baked);
        return 1;
    }
    if (n_npy == 0) {
        printf("  L1 OK: %s — absent in both backends\n", name);
        return 0;
    }
    float *x_npy = malloc(window_floats * sizeof(float));
    float *x_baked = malloc(window_floats * sizeof(float));
    if (!x_npy || !x_baked) DIE("OOM");
    for (size_t i = 0; i < n_npy; i++) {
        int32_t y_npy = -1, y_baked = -2;
        smatableDatasetGetSplit_npy(npy, split, i, x_npy, &y_npy);
        smatableDatasetGetSplit_baked(baked, split, i, x_baked, &y_baked);
        if (y_npy != y_baked) {
            fprintf(stderr, "L1 FAIL: %s[%zu] label: npy=%d baked=%d\n", name, i, y_npy, y_baked);
            return 1;
        }
        if (memcmp(x_npy, x_baked, window_floats * sizeof(float)) != 0) {
            for (size_t j = 0; j < window_floats; j++) {
                if (x_npy[j] != x_baked[j]) {
                    fprintf(stderr, "L1 FAIL: %s[%zu] x[%zu]: npy=%a baked=%a\n", name, i, j,
                            (double)x_npy[j], (double)x_baked[j]);
                    return 1;
                }
            }
            DIE("memcmp diverged but per-float scan did not — invariant broken");
        }
    }
    free(x_npy);
    free(x_baked);
    printf("  L1 OK: %s — %zu samples bytewise equal\n", name, n_npy);
    return 0;
}

int main(void) {
    smatable_dataset_t *npy   = smatableDatasetOpen_npy();
    smatable_dataset_t *baked = smatableDatasetOpen_baked();

    EQ("n_channels",    smatableDatasetNChannels_npy(npy),
                        smatableDatasetNChannels_baked(baked));
    EQ("window_samples", smatableDatasetWindowSamples_npy(npy),
                         smatableDatasetWindowSamples_baked(baked));
    EQ("window_floats",  smatableDatasetWindowFloats_npy(npy),
                         smatableDatasetWindowFloats_baked(baked));
    EQ("n_classes",      smatableDatasetNClasses_npy(npy),
                         smatableDatasetNClasses_baked(baked));

    size_t WF = smatableDatasetWindowFloats_npy(npy);

    int rc = 0;
    for (int s = 0; s < SMATABLE_SPLIT_COUNT; s++) {
        rc |= compare_named_split((smatable_split_t)s, npy, baked, WF);
    }
    /* legacy wrappers must alias TRAIN / TEST on both backends */
    EQ("wrapper train count (npy)",   smatableDatasetTrainCount_npy(npy),
                                      smatableDatasetSplitCount_npy(npy, SMATABLE_SPLIT_TRAIN));
    EQ("wrapper test count (npy)",    smatableDatasetTestCount_npy(npy),
                                      smatableDatasetSplitCount_npy(npy, SMATABLE_SPLIT_TEST));
    EQ("wrapper train count (baked)", smatableDatasetTrainCount_baked(baked),
                                      smatableDatasetSplitCount_baked(baked, SMATABLE_SPLIT_TRAIN));
    EQ("wrapper test count (baked)",  smatableDatasetTestCount_baked(baked),
                                      smatableDatasetSplitCount_baked(baked, SMATABLE_SPLIT_TEST));
    if (smatableDatasetSplitCount_npy(npy, SMATABLE_SPLIT_CALIB) == 0) DIE("fixture has no calib split — re-prep");

    smatableDatasetClose_npy(npy);
    smatableDatasetClose_baked(baked);

    if (rc) { fprintf(stderr, "L1 FAIL\n"); return 1; }
    printf("L1 PASS — HOST(npy) and MCU(baked) backends bytewise equal\n");
    return 0;
}
