#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

/* ============================================================
   Forward declarations from hashtable.c (no header file)
   Opaque type in this file.
   ============================================================ */
typedef struct hashtable hashtable_t;

hashtable_t* ht_create(size_t nbuckets);
void ht_free(hashtable_t* ht);

void ht_insert_coarse(hashtable_t* ht, int key, int value);
int  ht_find_coarse(hashtable_t* ht, int key, int* out);
int  ht_erase_coarse(hashtable_t* ht, int key);

void ht_insert_fine(hashtable_t* ht, int key, int value);
int  ht_find_fine(hashtable_t* ht, int key, int* out);
int  ht_erase_fine(hashtable_t* ht, int key);

/* ============================================================
   Timing
   ============================================================ */
static inline uint64_t now_ns() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

/* ============================================================
   Avoid POSIX mode_t naming conflict
   ============================================================ */
typedef enum { WORK_LOOKUP, WORK_INSERT, WORK_MIXED } workload_t;
typedef enum { IMPL_COARSE, IMPL_FINE } impl_t;

typedef struct {
    hashtable_t* ht;
    impl_t impl;
    workload_t workload;
    int ops;
    int keyspace;
} thread_arg_t;

static void* worker(void* arg) {
    thread_arg_t* a = (thread_arg_t*)arg;
    unsigned int seed = (unsigned int)(uintptr_t)pthread_self();
    int tmp;

    for (int i = 0; i < a->ops; i++) {
        int key = (int)(rand_r(&seed) % (unsigned int)a->keyspace);
        int r   = (int)(rand_r(&seed) % 10);

        if (a->workload == WORK_LOOKUP || (a->workload == WORK_MIXED && r < 7)) {
            if (a->impl == IMPL_COARSE) (void)ht_find_coarse(a->ht, key, &tmp);
            else                        (void)ht_find_fine(a->ht, key, &tmp);
        } else {
            if (a->impl == IMPL_COARSE) ht_insert_coarse(a->ht, key, key);
            else                        ht_insert_fine(a->ht, key, key);
        }
    }
    return NULL;
}

static workload_t parse_workload(const char* s) {
    if (!strcmp(s, "lookup")) return WORK_LOOKUP;
    if (!strcmp(s, "insert")) return WORK_INSERT;
    if (!strcmp(s, "mixed"))  return WORK_MIXED;
    fprintf(stderr, "Unknown --mode %s (use lookup|insert|mixed)\n", s);
    exit(1);
}

static impl_t parse_impl(const char* s) {
    if (!strcmp(s, "coarse")) return IMPL_COARSE;
    if (!strcmp(s, "fine"))   return IMPL_FINE;
    fprintf(stderr, "Unknown --impl %s (use coarse|fine)\n", s);
    exit(1);
}

static void usage(const char* prog) {
    fprintf(stderr,
      "Usage:\n"
      "  %s --impl coarse|fine --mode lookup|insert|mixed --threads N --keys K --ops OPS [--prefill P] [--runid ID]\n"
      "Output CSV:\n"
      "  run_id,impl,mode,keys,threads,ops,prefill,seconds,throughput\n", prog);
    exit(1);
}

int main(int argc, char** argv) {
    impl_t impl = IMPL_COARSE;
    workload_t workload = WORK_LOOKUP;
    int threads = 1;
    int keys = 100000;
    int ops  = 1000000;
    int prefill = -1;
    long run_id = -1;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--impl") && i + 1 < argc) {
            impl = parse_impl(argv[++i]);
        } else if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
            workload = parse_workload(argv[++i]);
        } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--keys") && i + 1 < argc) {
            keys = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--ops") && i + 1 < argc) {
            ops = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--prefill") && i + 1 < argc) {
            prefill = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--runid") && i + 1 < argc) {
            run_id = atol(argv[++i]);
        } else {
            usage(argv[0]);
        }
    }

    if (threads < 1 || keys < 1 || ops < 1) usage(argv[0]);

    if (prefill < 0) prefill = (workload == WORK_INSERT) ? 0 : keys;
    if (prefill > keys) prefill = keys;

    hashtable_t* ht = ht_create((size_t)keys);
    if (!ht) { fprintf(stderr, "ht_create failed\n"); return 1; }

    /* Prefill (single-threaded, not timed) */
    for (int k = 0; k < prefill; k++) ht_insert_coarse(ht, k, k);

    pthread_t* tids = (pthread_t*)calloc((size_t)threads, sizeof(pthread_t));
    thread_arg_t* args = (thread_arg_t*)calloc((size_t)threads, sizeof(thread_arg_t));
    if (!tids || !args) { fprintf(stderr, "alloc failed\n"); return 1; }

    int per = ops / threads;
    int rem = ops % threads;

    uint64_t t0 = now_ns();
    for (int i = 0; i < threads; i++) {
        args[i].ht = ht;
        args[i].impl = impl;
        args[i].workload = workload;
        args[i].ops = per + (i < rem ? 1 : 0);
        args[i].keyspace = keys;
        pthread_create(&tids[i], NULL, worker, &args[i]);
    }
    for (int i = 0; i < threads; i++) pthread_join(tids[i], NULL);
    uint64_t t1 = now_ns();

    double secs = (double)(t1 - t0) / 1e9;
    double thr  = (double)ops / secs;

    const char* impl_s = (impl == IMPL_COARSE) ? "coarse" : "fine";
    const char* mode_s = (workload == WORK_LOOKUP) ? "lookup" :
                         (workload == WORK_INSERT) ? "insert" : "mixed";

    /* CSV line: run_id first so we can join with perf.csv */
    printf("%ld,%s,%s,%d,%d,%d,%d,%.6f,%.2f\n",
           run_id, impl_s, mode_s, keys, threads, ops, prefill, secs, thr);

    free(tids);
    free(args);
    ht_free(ht);
    return 0;
}

