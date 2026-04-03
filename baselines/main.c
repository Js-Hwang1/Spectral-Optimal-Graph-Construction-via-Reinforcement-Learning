/*
 * main.c - Algebraic Connectivity Benchmark Driver (Parallel)
 *
 * Usage: ./benchmark [--jobs N] [--db] [--ring] [--seed S] [N1 N2 N3 ...]
 *
 * Runs ER, FV, OURS, and SW in parallel using fork().
 * Saves separate CSV files per algorithm in data/:
 *   ER_{N}.csv, FV_{N}.csv, OURS_{N}.csv
 *   SW_r25_{N}.csv, SW_r50_{N}.csv, SW_r75_{N}.csv
 *
 * Skips algorithms whose CSV already exists.
 *
 * --jobs N   Number of parallel workers (default: all CPUs)
 * --db       Load results into HuggingFace after completion
 * --ring     Use ring initialization instead of random spanning tree
 * --seed S   Set random seed for reproducibility (default: time-based)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

/* SW rho values */
static const double SW_RHOS[] = {0.25, 0.50, 0.75};
static const char *SW_RHO_NAMES[] = {"r25", "r50", "r75"};

/* Number of algorithm types per N: ER + FV + 3 SW = 5 */
#define NUM_ALGO_TYPES 5

/* Global config passed to child processes via --single */
static InitType g_init = INIT_TREE;
static uint64_t g_seed = 0;  /* 0 = time-based */
static const char *g_exe = "./baselines";

/* Task definition with estimated cost for scheduling */
typedef struct {
    int n;
    int algo;       /* 0=ER, 1=FV, 3-5=SW rho 0.25/0.50/0.75 */
    double cost;    /* estimated relative cost for LPT scheduling */
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

static const char *algo_name(int algo) {
    switch (algo) {
        case 0: return "ER";
        case 1: return "FV";
        case 3: return "SW_r25";
        case 4: return "SW_r50";
        case 5: return "SW_r75";
        default: return "???";
    }
}

/* ============================================================================
 * CSV WRITERS
 * ============================================================================ */

static void save_er_csv(int n, ERResult *result) {
    char path[256];
    make_path(path, sizeof(path), "ER", n);

    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot write to %s\n", path);
        return;
    }

    fprintf(fp, "m,score\n");
    for (int i = 0; i < result->count; i++) {
        fprintf(fp, "%d,%.10f\n", result->m_values[i], result->scores[i]);
    }
    fclose(fp);
    printf("  [ER] Saved: %s\n", path);
}

static void save_fv_csv(int n, FVResult *result) {
    char path[256];
    make_path(path, sizeof(path), "FV", n);

    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot write to %s\n", path);
        return;
    }

    fprintf(fp, "m,score\n");
    for (int i = 0; i < result->count; i++) {
        fprintf(fp, "%d,%.10f\n", result->m_values[i], result->scores[i]);
    }
    fclose(fp);
    printf("  [FV] Saved: %s\n", path);
}

static void save_sw_csv(int n, int rho_idx, SWResult *result) {
    char path[256];
    snprintf(path, sizeof(path), "%s/SW_%s_%d.csv",
             DATA_DIR, SW_RHO_NAMES[rho_idx], n);

    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot write to %s\n", path);
        return;
    }

    fprintf(fp, "m,score\n");
    for (int i = 0; i < result->count; i++) {
        fprintf(fp, "%d,%.10f\n", result->m_values[i], result->scores[i]);
    }
    fclose(fp);
    printf("  [SW rho=%.2f] Saved: %s\n", SW_RHOS[rho_idx], path);
}


/* ============================================================================
 * INDIVIDUAL ALGORITHM RUNNERS (for child processes)
 * ============================================================================ */

static void run_er(int n) {
    char path[256];
    make_path(path, sizeof(path), "ER", n);

    if (file_exists(path)) {
        printf("  [SKIP] ER_%d.csv already exists\n", n);
        return;
    }

    printf("  [ER] Starting for N=%d...\n", n);
    int max_m = n * (n - 1) / 2;
    ERResult *result = er_result_create(max_m);
    er_run(n, result, g_init);
    save_er_csv(n, result);
    er_result_free(result);
}

static void run_fv(int n) {
    char path[256];
    make_path(path, sizeof(path), "FV", n);

    if (file_exists(path)) {
        printf("  [SKIP] FV_%d.csv already exists\n", n);
        return;
    }

    printf("  [FV] Starting for N=%d...\n", n);
    int max_m = n * (n - 1) / 2;
    FVResult *result = fv_result_create(max_m);
    fv_run(n, result, g_init);
    save_fv_csv(n, result);
    fv_result_free(result);
}

static void run_sw(int n, int rho_idx) {
    char path[256];
    snprintf(path, sizeof(path), "%s/SW_%s_%d.csv",
             DATA_DIR, SW_RHO_NAMES[rho_idx], n);

    if (file_exists(path)) {
        printf("  [SKIP] SW_%s_%d.csv already exists\n", SW_RHO_NAMES[rho_idx], n);
        return;
    }

    printf("  [SW rho=%.2f] Starting for N=%d...\n", SW_RHOS[rho_idx], n);
    int max_m = n * (n - 1) / 2;
    int step = 1;

    SWResult *result = sw_result_create(max_m, SW_RHOS[rho_idx]);
    sw_run(n, SW_RHOS[rho_idx], result, step);
    save_sw_csv(n, rho_idx, result);
    sw_result_free(result);
}


/* ============================================================================
 * TASK EXECUTION
 * ============================================================================ */

static void run_task(Task *task) {
    /* Pin each worker to single-threaded LAPACK to prevent oversubscription */
    setenv("OMP_NUM_THREADS", "1", 1);
    setenv("OPENBLAS_NUM_THREADS", "1", 1);
    setenv("MKL_NUM_THREADS", "1", 1);
    setenv("VECLIB_MAXIMUM_THREADS", "1", 1);

    if (g_seed)
        rng_seed(g_seed ^ (uint64_t)task->n ^ (uint64_t)task->algo);
    else
        rng_seed((uint64_t)time(NULL) ^ (uint64_t)getpid());

    switch (task->algo) {
        case 0: run_er(task->n); break;
        case 1: run_fv(task->n); break;
        default:
            run_sw(task->n, task->algo - 3);
            break;
    }
}

/* ============================================================================
 * COST ESTIMATION (for LPT scheduling)
 *
 * ER/FV: M × N³ per task  (greedy loop: eigendecomp per edge added)
 *   ER has ~2× constant vs FV (pseudoinverse vs eigenvector)
 * SW:    M × seeds × N³    (eigendecomp per (m, seed) pair)
 *
 * M = N(N-1)/2 - (N-1) ≈ N²/2
 * ============================================================================ */

static double estimate_cost(int n, int algo) {
    double dn = (double)n;
    double M = dn * (dn - 1.0) / 2.0;
    double N3 = dn * dn * dn;

    switch (algo) {
        case 0: return M * N3 * 2.0;                   /* ER: pinv is ~2× eigvec */
        case 1: return M * N3;                          /* FV */
        default: return M * (double)SW_NUM_SEEDS * N3;  /* SW: 5 seeds × eigendecomp */
    }
}

/* ============================================================================
 * WORKER POOL (dynamic size, LPT-ordered)
 * ============================================================================ */

static void run_worker_pool(Task *tasks, int num_tasks, int pool_size) {
    pid_t *workers = calloc((size_t)pool_size, sizeof(pid_t));
    int *worker_task = calloc((size_t)pool_size, sizeof(int)); /* which task each worker runs */
    int active = 0;
    int next_task = 0;
    int completed = 0;

    printf("  [POOL] %d tasks, %d workers\n", num_tasks, pool_size);
    printf("  [POOL] Heaviest task: %s n=%d (cost=%.2e)\n",
           algo_name(tasks[0].algo), tasks[0].n, tasks[0].cost);
    printf("  [POOL] Lightest task: %s n=%d (cost=%.2e)\n",
           algo_name(tasks[num_tasks-1].algo), tasks[num_tasks-1].n,
           tasks[num_tasks-1].cost);
    fflush(stdout);

    while (next_task < num_tasks || active > 0) {
        /* Spawn workers up to pool_size */
        while (active < pool_size && next_task < num_tasks) {
            pid_t pid = fork();
            if (pid == 0) {
                /* Child: exec fresh process to avoid heap corruption */
                char algo_str[8], n_str[16];
                snprintf(algo_str, sizeof(algo_str), "%d", tasks[next_task].algo);
                snprintf(n_str, sizeof(n_str), "%d", tasks[next_task].n);
                char seed_str[32];
                snprintf(seed_str, sizeof(seed_str), "%llu", (unsigned long long)g_seed);
                if (g_init == INIT_RING && g_seed)
                    execl(g_exe, g_exe, "--single", algo_str, n_str,
                          "--ring", "--seed", seed_str, NULL);
                else if (g_init == INIT_RING)
                    execl(g_exe, g_exe, "--single", algo_str, n_str,
                          "--ring", NULL);
                else if (g_seed)
                    execl(g_exe, g_exe, "--single", algo_str, n_str,
                          "--seed", seed_str, NULL);
                else
                    execl(g_exe, g_exe, "--single", algo_str, n_str, NULL);
                _exit(1); /* exec failed */
            }
            workers[active] = pid;
            worker_task[active] = next_task;
            active++;
            next_task++;
        }

        /* Wait for any child to finish */
        if (active > 0) {
            int status;
            pid_t done = wait(&status);

            /* Remove from active list */
            for (int i = 0; i < active; i++) {
                if (workers[i] == done) {
                    completed++;
                    if (completed % 20 == 0 || completed == num_tasks) {
                        printf("  [POOL] Progress: %d/%d tasks done\n",
                               completed, num_tasks);
                        fflush(stdout);
                    }
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
 * MAIN
 * ============================================================================ */

int main(int argc, char *argv[]) {
    /* Single-task mode: ./benchmark --single <algo> <n>
     * Used by worker pool via fork()+exec() to get a clean address space,
     * avoiding heap corruption from FlexiBLAS/OpenBLAS internal state. */
    if (argc >= 4 && strcmp(argv[1], "--single") == 0) {
        setenv("OMP_NUM_THREADS", "1", 1);
        setenv("OPENBLAS_NUM_THREADS", "1", 1);
        setenv("MKL_NUM_THREADS", "1", 1);
        setenv("VECLIB_MAXIMUM_THREADS", "1", 1);
        setenv("FLEXIBLAS_NUM_THREADS", "1", 1);
        /* Parse extra flags after --single <algo> <n> */
        for (int i = 4; i < argc; i++) {
            if (strcmp(argv[i], "--ring") == 0) g_init = INIT_RING;
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
                g_seed = (uint64_t)atoll(argv[++i]);
        }
        Task t = {.n = atoi(argv[3]), .algo = atoi(argv[2]), .cost = 0};
        run_task(&t);
        return 0;
    }

    g_exe = argv[0];

    /* Parse flags and N values */
    int n_values[MAX_N_VALUES] = DEFAULT_N_VALUES;
    int num_n = DEFAULT_NUM_N;
    int load_db = 0;
    int num_jobs = 0;  /* 0 = auto-detect */

    int n_args = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--db") == 0) {
            load_db = 1;
        } else if (strcmp(argv[i], "--ring") == 0) {
            g_init = INIT_RING;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            g_seed = (uint64_t)atoll(argv[++i]);
        } else if ((strcmp(argv[i], "--jobs") == 0 || strcmp(argv[i], "-j") == 0)
                   && i + 1 < argc) {
            num_jobs = atoi(argv[++i]);
        } else {
            if (n_args == 0) num_n = 0;
            if (num_n < MAX_N_VALUES) {
                n_values[num_n++] = atoi(argv[i]);
            }
            n_args++;
        }
    }

    /* Auto-detect CPU count if not specified */
    if (num_jobs <= 0) {
        num_jobs = detect_cpus();
    }

    printf("Algebraic Connectivity Benchmark (Worker Pool)\n");
    printf("Workers: %d", num_jobs);
    if (num_jobs == detect_cpus()) printf(" (auto-detected)");
    printf("\n");
    printf("N values (%d): ", num_n);
    for (int i = 0; i < num_n; i++) {
        printf("%d ", n_values[i]);
    }
    printf("\n");
    printf("Init: %s\n", g_init == INIT_RING ? "ring" : "random spanning tree");
    if (g_seed) printf("Seed: %llu\n", (unsigned long long)g_seed);
    if (load_db) printf("HuggingFace load: enabled (--db)\n");

    /* Build task list with cost estimates */
    static const int ALGOS[] = {0, 1, 3, 4, 5}; /* ER, FV, SW×3 */
    int num_tasks = num_n * NUM_ALGO_TYPES;
    Task *tasks = malloc((size_t)num_tasks * sizeof(Task));

    int t = 0;
    for (int i = 0; i < num_n; i++) {
        for (int a = 0; a < NUM_ALGO_TYPES; a++) {
            int algo = ALGOS[a];
            tasks[t].n = n_values[i];
            tasks[t].algo = algo;
            tasks[t].cost = estimate_cost(n_values[i], algo);
            t++;
        }
    }

    /* Sort tasks by cost descending (Longest Processing Time first) */
    qsort(tasks, (size_t)num_tasks, sizeof(Task), task_cmp_desc);

    printf("Running %d total tasks with %d workers (LPT scheduled)\n",
           num_tasks, num_jobs);
    fflush(stdout);

    /* Run all tasks via worker pool */
    run_worker_pool(tasks, num_tasks, num_jobs);

    free(tasks);

    printf("\n========================================\n");
    printf("All benchmarks complete!\n");
    printf("Results saved to: %s/\n", DATA_DIR);
    printf("========================================\n");

    /* Load results into HuggingFace if --db flag was given */
    if (load_db) {
        printf("\nLoading results into HuggingFace...\n");
        fflush(stdout);
        int ret = system("python3 database/db_baselines.py load");
        if (ret != 0) {
            fprintf(stderr, "Warning: HuggingFace load failed (exit code %d)\n", ret);
        }
    }

    return 0;
}
