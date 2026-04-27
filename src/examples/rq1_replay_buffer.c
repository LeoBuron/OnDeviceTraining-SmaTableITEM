/*
 * Example: rq1_replay_buffer  (HOST + MCU)
 *
 * STUB — full implementation deferred until upstream ODT ships Conv1d /
 * LayerNorm (sprint plan W1 D3-D5) and the per-RQ training-loop design
 * sub-spec is written. Today this binary just prints a parseable
 * "RESULT skipped" line so the Optuna harness can iterate it for plumbing
 * tests without producing fake numbers.
 *
 * RQ1 question: how many samples per class from the original training set
 * must the replay buffer retain to prevent catastrophic forgetting during
 * on-device fine-tuning to a new user?
 */

#include <stdio.h>
#include <time.h>

#include "hardware_init.h"
#include "smatable_dataset.h"

int main(void) {
    init();
    char ts[32];
    time_t t = time(NULL); struct tm *tm = gmtime(&t);
    strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", tm);
    printf("BEGIN rq1_replay_buffer %s\n", ts);
    printf("RESULT skipped reason=\"awaiting Conv1d/LayerNorm in upstream ODT (sprint W1 D3-D5)\"\n");
    return 0;
}
