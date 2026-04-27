/* L1 equivalence test: HOST-NPY backend vs MCU-baked backend, bytewise.
 *
 * Both backends are compiled into THIS binary with disambiguating function
 * suffixes (-DSMATABLE_FN_SUFFIX=_npy / _baked) so we can call each through
 * its own entry point. Normal builds still link exactly one backend with the
 * unsuffixed names — the suffix scheme is only invoked here.
 *
 * What this proves:
 *   1. dataset_get_train(i) and dataset_get_test(i) return BYTEWISE equal
 *      float[] for every i, on the same fold.
 *   2. train/test counts agree.
 *   3. Layout helpers (n_channels, window_samples, …) agree.
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

#define DIE(msg) do { fprintf(stderr, "L1 FAIL: %s\n", msg); exit(1); } while (0)
#define EQ(label, a, b) do { \
    size_t _a = (a), _b = (b); \
    if (_a != _b) { fprintf(stderr, "L1 FAIL: %s: npy=%zu baked=%zu\n", label, _a, _b); exit(1); } \
} while (0)

static int compare_split(
    const char *split,
    smatable_dataset_t *npy, smatable_dataset_t *baked,
    size_t (*count_npy)  (const smatable_dataset_t *),
    size_t (*count_baked)(const smatable_dataset_t *),
    void (*get_npy)  (const smatable_dataset_t *, size_t, float *, int32_t *),
    void (*get_baked)(const smatable_dataset_t *, size_t, float *, int32_t *),
    size_t window_floats)
{
    size_t n_npy = count_npy(npy);
    size_t n_baked = count_baked(baked);
    if (n_npy != n_baked) {
        fprintf(stderr, "L1 FAIL: %s count: npy=%zu baked=%zu\n", split, n_npy, n_baked);
        return 1;
    }

    float *x_npy   = malloc(window_floats * sizeof(float));
    float *x_baked = malloc(window_floats * sizeof(float));
    if (!x_npy || !x_baked) DIE("OOM");

    for (size_t i = 0; i < n_npy; i++) {
        int32_t y_npy = -1, y_baked = -2;
        get_npy  (npy,   i, x_npy,   &y_npy);
        get_baked(baked, i, x_baked, &y_baked);

        if (y_npy != y_baked) {
            fprintf(stderr, "L1 FAIL: %s[%zu] label: npy=%d baked=%d\n", split, i, y_npy, y_baked);
            return 1;
        }
        if (memcmp(x_npy, x_baked, window_floats * sizeof(float)) != 0) {
            /* Locate the first differing float for a useful error. */
            for (size_t j = 0; j < window_floats; j++) {
                if (x_npy[j] != x_baked[j]) {
                    fprintf(stderr, "L1 FAIL: %s[%zu] x[%zu]: npy=%a baked=%a\n",
                            split, i, j, (double)x_npy[j], (double)x_baked[j]);
                    return 1;
                }
            }
            DIE("memcmp diverged but per-float scan did not — invariant broken");
        }
    }

    free(x_npy);
    free(x_baked);
    printf("  L1 OK: %s — %zu samples bytewise equal\n", split, n_npy);
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
    rc |= compare_split("train", npy, baked,
                        smatableDatasetTrainCount_npy, smatableDatasetTrainCount_baked,
                        smatableDatasetGetTrain_npy,   smatableDatasetGetTrain_baked,
                        WF);
    rc |= compare_split("test", npy, baked,
                        smatableDatasetTestCount_npy, smatableDatasetTestCount_baked,
                        smatableDatasetGetTest_npy,   smatableDatasetGetTest_baked,
                        WF);

    smatableDatasetClose_npy(npy);
    smatableDatasetClose_baked(baked);

    if (rc) { fprintf(stderr, "L1 FAIL\n"); return 1; }
    printf("L1 PASS — HOST(npy) and MCU(baked) backends bytewise equal\n");
    return 0;
}
