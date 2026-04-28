/* Host-side time/memory probes for the stage-1 binaries.
 * Layer B (process truth): getrusage + /proc/self/status milestones.
 * Layer C (observed stack): run a function on a dedicated painted stack.
 * MCU builds: this header is host-only; example code guards its use with
 * the same #if so MCU targets compile the probes away entirely. */
#ifndef STAGE1_MEM_PROBE_H
#define STAGE1_MEM_PROBE_H

#if defined(__unix__) || defined(__APPLE__)
#define MEM_PROBE_AVAILABLE 1

#include <pthread.h>
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

/* ---- Layer C: observed stack high-water ----
 * Runs fn(arg) on a dedicated 1 MiB stack pre-painted with 0xA5, then scans
 * from the LOW end for the first non-paint byte (stacks grow down).
 * Returns observed peak stack bytes, or -1 on any pthread failure (caller
 * must then run fn(arg) inline and report stack_peak_b=-1).
 * Caveats: no guard page below the buffer — a peak within 64 KiB of the
 * 1 MiB ceiling is reported but should be treated as an overflow-risk
 * warning (any training needing ~1 MiB stack is an automatic MCU fail). */
#define MEM_PROBE_STACK_SIZE ((size_t)1 << 20)
#define MEM_PROBE_PAINT 0xA5

typedef void *(*memProbeThreadFn)(void *);

static inline long memProbeRunOnPaintedStack(memProbeThreadFn fn, void *arg) {
    unsigned char *stack = NULL;
    if (posix_memalign((void **)&stack, 4096, MEM_PROBE_STACK_SIZE) != 0) {
        return -1;
    }
    memset(stack, MEM_PROBE_PAINT, MEM_PROBE_STACK_SIZE);
    pthread_attr_t attr;
    pthread_t tid;
    if (pthread_attr_init(&attr) != 0) {
        free(stack);
        return -1;
    }
    if (pthread_attr_setstack(&attr, stack, MEM_PROBE_STACK_SIZE) != 0 ||
        pthread_create(&tid, &attr, fn, arg) != 0) {
        pthread_attr_destroy(&attr);
        free(stack);
        return -1;
    }
    pthread_join(tid, NULL);
    pthread_attr_destroy(&attr);
    size_t i = 0;
    while (i < MEM_PROBE_STACK_SIZE && stack[i] == MEM_PROBE_PAINT) {
        i++;
    }
    long peak = (long)(MEM_PROBE_STACK_SIZE - i);
    free(stack);
    return peak;
}

#endif /* __unix__ || __APPLE__ */
#endif /* STAGE1_MEM_PROBE_H */
