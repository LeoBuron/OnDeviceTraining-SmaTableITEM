/* Host-side CPU/RSS-milestone probes for the stage-1 binaries.
 * Layer B (process truth): getrusage + /proc/self/status milestones.
 * Stack high-water measurement (formerly Layer C's painted-stack probe here)
 * now lives upstream in MemProfile.h's measurePeakStackBytes (ODT main
 * 3e768c7) — see stage1_pretrain.c. This header owns only the CPU-time and
 * RSS-milestone helpers upstream doesn't provide.
 * MCU builds: this header is host-only; example code guards its use with
 * the same #if so MCU targets compile the probes away entirely. */
#ifndef STAGE1_MEM_PROBE_H
#define STAGE1_MEM_PROBE_H

#if defined(__unix__) || defined(__APPLE__)
#define MEM_PROBE_AVAILABLE 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

static inline void memProbeRusage(double *cpuUserS, double *cpuSysS, long *maxRssKb) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    *cpuUserS = (double)ru.ru_utime.tv_sec + (double)ru.ru_utime.tv_usec * 1e-6;
    *cpuSysS = (double)ru.ru_stime.tv_sec + (double)ru.ru_stime.tv_usec * 1e-6;
#ifdef __APPLE__
    *maxRssKb = ru.ru_maxrss / 1024; /* macOS reports bytes */
#else
    *maxRssKb = ru.ru_maxrss; /* Linux reports KiB */
#endif
}

/* Current VmRSS in KiB (Linux); -1 where /proc is unavailable. */
static inline long memProbeRssNowKb(void) {
#ifdef __linux__
    FILE *f = fopen("/proc/self/status", "r");
    if (f == NULL) {
        return -1;
    }
    char line[256];
    long kb = -1;
    while (fgets(line, sizeof(line), f) != NULL) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            kb = atol(line + 6);
            break;
        }
    }
    fclose(f);
    return kb;
#else
    return -1;
#endif
}

#endif /* __unix__ || __APPLE__ */
#endif /* STAGE1_MEM_PROBE_H */
