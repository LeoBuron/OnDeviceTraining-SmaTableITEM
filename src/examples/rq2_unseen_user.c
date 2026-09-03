/*
 * Example: rq2_unseen_user  (HOST + MCU)  — STUB.
 * RQ2: how many labelled samples from a previously-unseen user are required
 * before on-device fine-tuning recovers accuracy on that user under LOOCV?
 * Not implemented yet; design + status in experiments/rq2-unseen-user/README.md.
 */
#include <stdio.h>
#include <time.h>
#include "hardware_init.h"
#include "smatable_dataset.h"
int main(void) {
    init();
    char ts[32]; time_t t = time(NULL); struct tm *tm = gmtime(&t);
    strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", tm);
    printf("BEGIN rq2_unseen_user %s\n", ts);
    printf("RESULT skipped reason=\"not implemented yet (see experiments/rq2-unseen-user/README.md)\"\n");
    return 0;
}
