/*
 * main.c - Algebraic Connectivity Benchmark Driver (Parallel)
 *
 * Usage: ./baselines [--w N] [--algo er,fv,sw] [--ring] [--seed S] [--db] [N1 N2 ...]
 *
 * Runs ER, FV, and SW baselines in parallel using fork()+exec().
 * ER/FV run BASELINE_NUM_SEEDS seeds each, reporting mean ± std.
 * SW runs SW_NUM_SEEDS seeds internally per (n, rho).
 *
 * --w N       Number of parallel workers (default: all CPUs)
 * --algo X    Comma-separated: er, fv, sw (default: all)
 * --ring      Use ring initialization instead of random spanning tree
 * --seed S    Set random seed for reproducibility
 * --db        Load results into HuggingFace after completion
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "common.h"
#include "er.h"
#include "fv.h"
#include "sw.h"

/* ============================================================================
 * CONFIGURATION
 * ============================================================================ */

#define MAX_N_VALUES 256
#define DEFAULT_N_VALUES {32, 64, 128, 256}
#define DEFAULT_NUM_N 4
#define DATA_DIR "data"
#define BASELINE_NUM_SEEDS 10

/* SW rho values */
static const double SW_RHOS[] = {0.25, 0.50, 0.75};
static const char *SW_RHO_NAMES[] = {"r25", "r50", "r75"};

/* Global config */
static InitType g_init = INIT_TREE;
static uint64_t g_seed = 0;
static const char *g_exe = "./baselines";

/* Algo selection */
#define ALGO_ER  (1 << 0)
#define ALGO_FV  (1 << 1)
#define ALGO_SW  (1 << 2)
#define ALGO_ALL (ALGO_ER | ALGO_FV | ALGO_SW)
static int g_algo_mask = ALGO_ALL;

static int parse_algo(const char *s) {
    int mask = 0;
    while (*s) {
        if (strncmp(s, "er", 2) == 0)      { mask |= ALGO_ER; s += 2; }
        else if (strncmp(s, "fv", 2) == 0)  { mask |= ALGO_FV; s += 2; }
        else if (strncmp(s, "sw", 2) == 0)  { mask |= ALGO_SW; s += 2; }
        else s++;
        if (*s == ',') s++;
    }
    return mask ? mask : ALGO_ALL;
}

/* Task: (algo, n, seed_id).  seed_id = -1 for SW (handles seeds internally) */
typedef struct {
    int n;
    int algo;       /* 0=ER, 1=FV, 3-5=SW rho 0.25/0.50/0.75 */
    int seed_id;    /* 0..BASELINE_NUM_SEEDS-1 for ER/FV, -1 for SW */
    double cost;
} Task;

/* ============================================================================
 * UTILITIES
 * ============================================================================ */

static int file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}

static void make_path(char *buf, size_t size, const char *algo, int n) {
    snprintf(buf, size, "%s/%s_%d.csv", DATA_DIR, algo, n);
}

static int detect_cpus(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? (int)n : 8;
}

static int task_cmp_desc(const void *a, const void *b) {
    double ca = ((const Task *)a)->cost;
    double cb = ((const Task *)b)->cost;
    if (cb > ca) return 1;
    if (cb < ca) return -1;
    return 0;
}

static const char *algo_label(int algo, int seed_id) {
    static char buf[32];
    const char *name;
    switch (algo) {
        case 0: name = "ER"; break;
        case 1: name = "FV"; break;
        case 3: name = "SW_r25"; break;
        case 4: name = "SW_r50"; break;
        case 5: name = "SW_r75"; break;
        default: name = "???"; break;
    }
    if (seed_id >= 0)
        snprintf(buf, sizeof(buf), "%s[s%d]", name, seed_id);
    else
        snprintf(buf, sizeof(buf), "%s", name);
    return buf;
}

/* ============================================================================
 * INDIVIDUAL ALGORITHM RUNNERS (called by child processes)
 * ============================================================================ */

/* ER/FV: write per-seed temp CSV (m,score) */
static void run_er_seed(int n, int seed_id) {
    char path[256];
    snprintf(path, sizeof(path), "%s/.tmp_ER_%d_s%d.csv", DATA_DIR, n, seed_id);

    /* Seed RNG deterministically per (n, algo, seed_id) */
    uint64_t base = g_seed ? g_seed : 12345ULL;
    rng_seed(base ^ ((uint64_t)seed_id * 999983ULL) ^
             ((uint64_t)n * 1000003ULL));

    int max_m = n * (n - 1) / 2;
    ERResult *result = er_result_create(max_m);
    er_run(n, result, g_init);

    FILE *fp = fopen(path, "w");
    if (!fp) { fprintf(stderr, "Error: Cannot write %s\n", path); er_result_free(result); return; }
    fprintf(fp, "m,score\n");
    for (int i = 0; i < result->count; i++)
        fprintf(fp, "%d,%.10f\n", result->m_values[i], result->scores[i]);
    fclose(fp);

    er_result_free(result);
    printf("  [ER s%d] Done n=%d -> %s\n", seed_id, n, path);
}

static void run_fv_seed(int n, int seed_id) {
    char path[256];
    snprintf(path, sizeof(path), "%s/.tmp_FV_%d_s%d.csv", DATA_DIR, n, seed_id);

    uint64_t base = g_seed ? g_seed : 12345ULL;
    rng_seed(base ^ ((uint64_t)seed_id * 999983ULL) ^
             ((uint64_t)n * 1000003ULL) ^ 777ULL);

    int max_m = n * (n - 1) / 2;
    FVResult *result = fv_result_create(max_m);
    fv_run(n, result, g_init);

    FILE *fp = fopen(path, "w");
    if (!fp) { fprintf(stderr, "Error: Cannot write %s\n", path); fv_result_free(result); return; }
    fprintf(fp, "m,score\n");
    for (int i = 0; i < result->count; i++)
        fprintf(fp, "%d,%.10f\n", result->m_values[i], result->scores[i]);
    fclose(fp);

    fv_result_free(result);
    printf("  [FV s%d] Done n=%d -> %s\n", seed_id, n, path);
}

static void run_sw(int n, int rho_idx) {
    char path[256];
    snprintf(path, sizeof(path), "%s/SW_%s_%d.csv",
             DATA_DIR, SW_RHO_NAMES[rho_idx], n);

    if (file_exists(path)) {
        printf("  [SKIP] SW_%s_%d.csv exists\n", SW_RHO_NAMES[rho_idx], n);
        return;
    }

    printf("  [SW rho=%.2f] Starting n=%d...\n", SW_RHOS[rho_idx], n);
    int max_m = n * (n - 1) / 2;

    SWResult *result = sw_result_create(max_m, SW_RHOS[rho_idx]);
    sw_run(n, SW_RHOS[rho_idx], result, 1);

    FILE *fp = fopen(path, "w");
    if (!fp) { fprintf(stderr, "Error: Cannot write %s\n", path); sw_result_free(result); return; }
    fprintf(fp, "m,score,std\n");
    for (int i = 0; i < result->count; i++)
        fprintf(fp, "%d,%.10f,%.10f\n",
                result->m_values[i], result->scores[i], result->stds[i]);
    fclose(fp);
    printf("  [SW rho=%.2f] Saved: %s\n", SW_RHOS[rho_idx], path);

    sw_result_free(result);
}

/* ============================================================================
 * TASK EXECUTION (child process)
 * ============================================================================ */

static void run_task(Task *task) {
    setenv("OMP_NUM_THREADS", "1", 1);
    setenv("OPENBLAS_NUM_THREADS", "1", 1);
    setenv("MKL_NUM_THREADS", "1", 1);
    setenv("VECLIB_MAXIMUM_THREADS", "1", 1);

    switch (task->algo) {
        case 0: run_er_seed(task->n, task->seed_id); break;
        case 1: run_fv_seed(task->n, task->seed_id); break;
        default:
            /* SW: seed the RNG for internal seed-loop determinism */
            if (g_seed)
                rng_seed(g_seed ^ (uint64_t)task->n ^ (uint64_t)task->algo);
            else
                rng_seed((uint64_t)time(NULL) ^ (uint64_t)getpid());
            run_sw(task->n, task->algo - 3);
            break;
    }
}

/* ============================================================================
 * COST ESTIMATION
 * ============================================================================ */

static double estimate_cost(int n, int algo) {
    double dn = (double)n;
    double M = dn * (dn - 1.0) / 2.0;
    double N3 = dn * dn * dn;

    switch (algo) {
        case 0: return M * N3 * 2.0;                    /* ER: 1 seed */
        case 1: return M * N3;                           /* FV: 1 seed */
        default: return M * (double)SW_NUM_SEEDS * N3;   /* SW: all seeds */
    }
}

/* ============================================================================
 * WORKER POOL (dynamic, LPT-ordered)
 * ============================================================================ */

static void run_worker_pool(Task *tasks, int num_tasks, int pool_size) {
    pid_t *workers = calloc((size_t)pool_size, sizeof(pid_t));
    int *worker_task = calloc((size_t)pool_size, sizeof(int));
    int active = 0, next_task = 0, completed = 0;

    printf("  [POOL] %d tasks, %d workers\n", num_tasks, pool_size);
    printf("  [POOL] Heaviest: %s n=%d (cost=%.2e)\n",
           algo_label(tasks[0].algo, tasks[0].seed_id), tasks[0].n, tasks[0].cost);
    printf("  [POOL] Lightest: %s n=%d (cost=%.2e)\n",
           algo_label(tasks[num_tasks-1].algo, tasks[num_tasks-1].seed_id),
           tasks[num_tasks-1].n, tasks[num_tasks-1].cost);
    fflush(stdout);

    while (next_task < num_tasks || active > 0) {
        while (active < pool_size && next_task < num_tasks) {
            pid_t pid = fork();
            if (pid == 0) {
                /* Build argv for child */
                char algo_str[8], n_str[16], sid_str[8], seed_str[32];
                snprintf(algo_str, sizeof(algo_str), "%d", tasks[next_task].algo);
                snprintf(n_str, sizeof(n_str), "%d", tasks[next_task].n);
                snprintf(sid_str, sizeof(sid_str), "%d", tasks[next_task].seed_id);
                snprintf(seed_str, sizeof(seed_str), "%llu", (unsigned long long)g_seed);

                char *args[16];
                int ai = 0;
                args[ai++] = (char *)g_exe;
                args[ai++] = "--single";
                args[ai++] = algo_str;
                args[ai++] = n_str;
                args[ai++] = "--sid";
                args[ai++] = sid_str;
                if (g_init == INIT_RING) args[ai++] = "--ring";
                if (g_seed) { args[ai++] = "--seed"; args[ai++] = seed_str; }
                args[ai] = NULL;
                execv(g_exe, args);
                _exit(1);
            }
            workers[active] = pid;
            worker_task[active] = next_task;
            active++;
            next_task++;
        }

        if (active > 0) {
            int status;
            pid_t done = wait(&status);
            for (int i = 0; i < active; i++) {
                if (workers[i] == done) {
                    completed++;
                    if (completed % 20 == 0 || completed == num_tasks)
                        printf("  [POOL] Progress: %d/%d tasks done\n",
                               completed, num_tasks);
                    fflush(stdout);
                    workers[i] = workers[active - 1];
                    worker_task[i] = worker_task[active - 1];
                    active--;
                    break;
                }
            }
        }
    }

    free(workers);
    free(worker_task);
}

/* ============================================================================
 * MERGE: aggregate per-seed temp CSVs into final CSV with mean ± std
 * ============================================================================ */

static void merge_seed_csvs(const char *algo, int n, int num_seeds) {
    /* Read seed 0 to get row count */
    char path[256];
    snprintf(path, sizeof(path), "%s/.tmp_%s_%d_s0.csv", DATA_DIR, algo, n);
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "  [MERGE] Missing %s\n", path); return; }

    char line[256];
    fgets(line, sizeof(line), fp); /* skip header */
    int count = 0;
    while (fgets(line, sizeof(line), fp)) count++;
    fclose(fp);
    if (count == 0) return;

    /* Allocate: m_values[count], all_scores[num_seeds][count] */
    int *m_values = malloc((size_t)count * sizeof(int));
    double *flat = malloc((size_t)num_seeds * count * sizeof(double));

    /* Read all seed files */
    for (int s = 0; s < num_seeds; s++) {
        snprintf(path, sizeof(path), "%s/.tmp_%s_%d_s%d.csv", DATA_DIR, algo, n, s);
        fp = fopen(path, "r");
        if (!fp) { fprintf(stderr, "  [MERGE] Missing %s\n", path); free(m_values); free(flat); return; }
        fgets(line, sizeof(line), fp); /* skip header */
        for (int i = 0; i < count; i++) {
            if (fscanf(fp, "%d,%lf\n", &m_values[i], &flat[s * count + i]) != 2) {
                fprintf(stderr, "  [MERGE] Parse error in %s line %d\n", path, i + 2);
            }
        }
        fclose(fp);
        remove(path); /* delete temp file */
    }

    /* Write final CSV with mean, std */
    char final_path[256];
    make_path(final_path, sizeof(final_path), algo, n);
    fp = fopen(final_path, "w");
    if (!fp) { fprintf(stderr, "Error: Cannot write %s\n", final_path); free(m_values); free(flat); return; }

    fprintf(fp, "m,score,std\n");
    for (int i = 0; i < count; i++) {
        double sum = 0.0;
        for (int s = 0; s < num_seeds; s++) sum += flat[s * count + i];
        double mean = sum / num_seeds;

        double sum_sq = 0.0;
        for (int s = 0; s < num_seeds; s++) {
            double d = flat[s * count + i] - mean;
            sum_sq += d * d;
        }
        double std = sqrt(sum_sq / num_seeds);

        fprintf(fp, "%d,%.10f,%.10f\n", m_values[i], mean, std);
    }
    fclose(fp);

    free(m_values);
    free(flat);
    printf("  [%s] Merged %d seeds -> %s (%d m-values)\n",
           algo, num_seeds, final_path, count);
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(int argc, char *argv[]) {
    /* --single mode: called by worker pool via fork()+exec() */
    if (argc >= 4 && strcmp(argv[1], "--single") == 0) {
        setenv("OMP_NUM_THREADS", "1", 1);
        setenv("OPENBLAS_NUM_THREADS", "1", 1);
        setenv("MKL_NUM_THREADS", "1", 1);
        setenv("VECLIB_MAXIMUM_THREADS", "1", 1);
        setenv("FLEXIBLAS_NUM_THREADS", "1", 1);

        int algo = atoi(argv[2]);
        int n = atoi(argv[3]);
        int sid = -1;

        for (int i = 4; i < argc; i++) {
            if (strcmp(argv[i], "--ring") == 0) g_init = INIT_RING;
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
                g_seed = (uint64_t)atoll(argv[++i]);
            else if (strcmp(argv[i], "--sid") == 0 && i + 1 < argc)
                sid = atoi(argv[++i]);
        }

        Task t = {.n = n, .algo = algo, .seed_id = sid, .cost = 0};
        run_task(&t);
        return 0;
    }

    g_exe = argv[0];

    /* Parse flags and N values */
    int n_values[MAX_N_VALUES] = DEFAULT_N_VALUES;
    int num_n = DEFAULT_NUM_N;
    int load_db = 0;
    int num_jobs = 0;

    int n_args = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--db") == 0) {
            load_db = 1;
        } else if (strcmp(argv[i], "--algo") == 0 && i + 1 < argc) {
            g_algo_mask = parse_algo(argv[++i]);
        } else if (strcmp(argv[i], "--ring") == 0) {
            g_init = INIT_RING;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            g_seed = (uint64_t)atoll(argv[++i]);
        } else if ((strcmp(argv[i], "--w") == 0 ||
                    strcmp(argv[i], "--jobs") == 0 ||
                    strcmp(argv[i], "-j") == 0) && i + 1 < argc) {
            num_jobs = atoi(argv[++i]);
        } else {
            if (n_args == 0) num_n = 0;
            if (num_n < MAX_N_VALUES)
                n_values[num_n++] = atoi(argv[i]);
            n_args++;
        }
    }

    if (num_jobs <= 0) num_jobs = detect_cpus();

    printf("Algebraic Connectivity Benchmark\n");
    printf("Workers: %d", num_jobs);
    if (num_jobs == detect_cpus()) printf(" (auto)");
    printf("\nN values (%d): ", num_n);
    for (int i = 0; i < num_n; i++) printf("%d ", n_values[i]);
    printf("\nInit: %s\n", g_init == INIT_RING ? "ring" : "random spanning tree");
    printf("Seeds: %d (ER/FV), %d (SW)\n", BASELINE_NUM_SEEDS, SW_NUM_SEEDS);
    if (g_seed) printf("Seed: %llu\n", (unsigned long long)g_seed);
    if (load_db) printf("HuggingFace load: enabled\n");

    /*
     * Build task list:
     *   ER/FV: BASELINE_NUM_SEEDS tasks per (n, algo) — each is one seed
     *   SW:    1 task per (n, rho) — handles seeds internally
     */
    int max_tasks = num_n * (2 * BASELINE_NUM_SEEDS + 3); /* upper bound */
    Task *tasks = malloc((size_t)max_tasks * sizeof(Task));
    int t = 0;

    for (int i = 0; i < num_n; i++) {
        int n = n_values[i];

        if (g_algo_mask & ALGO_ER) {
            char path[256]; make_path(path, sizeof(path), "ER", n);
            if (!file_exists(path)) {
                for (int s = 0; s < BASELINE_NUM_SEEDS; s++) {
                    tasks[t].n = n;
                    tasks[t].algo = 0;
                    tasks[t].seed_id = s;
                    tasks[t].cost = estimate_cost(n, 0);
                    t++;
                }
            } else {
                printf("  [SKIP] ER_%d.csv exists\n", n);
            }
        }

        if (g_algo_mask & ALGO_FV) {
            char path[256]; make_path(path, sizeof(path), "FV", n);
            if (!file_exists(path)) {
                for (int s = 0; s < BASELINE_NUM_SEEDS; s++) {
                    tasks[t].n = n;
                    tasks[t].algo = 1;
                    tasks[t].seed_id = s;
                    tasks[t].cost = estimate_cost(n, 1);
                    t++;
                }
            } else {
                printf("  [SKIP] FV_%d.csv exists\n", n);
            }
        }

        if (g_algo_mask & ALGO_SW) {
            for (int r = 0; r < 3; r++) {
                char path[256];
                snprintf(path, sizeof(path), "%s/SW_%s_%d.csv",
                         DATA_DIR, SW_RHO_NAMES[r], n);
                if (!file_exists(path)) {
                    tasks[t].n = n;
                    tasks[t].algo = 3 + r;
                    tasks[t].seed_id = -1;
                    tasks[t].cost = estimate_cost(n, 3 + r);
                    t++;
                } else {
                    printf("  [SKIP] SW_%s_%d.csv exists\n", SW_RHO_NAMES[r], n);
                }
            }
        }
    }

    int num_tasks = t;
    if (num_tasks == 0) {
        printf("Nothing to run (all CSVs exist).\n");
        free(tasks);
        return 0;
    }

    qsort(tasks, (size_t)num_tasks, sizeof(Task), task_cmp_desc);

    printf("Running %d tasks with %d workers (LPT scheduled)\n",
           num_tasks, num_jobs);
    fflush(stdout);

    run_worker_pool(tasks, num_tasks, num_jobs);
    free(tasks);

    /* Merge ER/FV per-seed temp CSVs into final CSVs */
    for (int i = 0; i < num_n; i++) {
        int n = n_values[i];
        if (g_algo_mask & ALGO_ER) {
            char path[256]; make_path(path, sizeof(path), "ER", n);
            if (!file_exists(path)) merge_seed_csvs("ER", n, BASELINE_NUM_SEEDS);
        }
        if (g_algo_mask & ALGO_FV) {
            char path[256]; make_path(path, sizeof(path), "FV", n);
            if (!file_exists(path)) merge_seed_csvs("FV", n, BASELINE_NUM_SEEDS);
        }
    }

    printf("\n========================================\n");
    printf("All benchmarks complete!\n");
    printf("Results saved to: %s/\n", DATA_DIR);
    printf("========================================\n");

    if (load_db) {
        printf("\nLoading results into HuggingFace...\n");
        fflush(stdout);
        int ret = system("python3 database/db_baselines.py load");
        if (ret != 0)
            fprintf(stderr, "Warning: HuggingFace load failed (exit code %d)\n", ret);
    }

    return 0;
}
