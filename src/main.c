/*
 * main.c - Algebraic Connectivity Benchmark Driver (Parallel)
 *
 * Usage: ./benchmark [N1 N2 N3 ...]
 *
 * Runs ER, FV, OURS, and SW (all rhos) in parallel using fork().
 * Saves separate CSV files per algorithm in data/:
 *   ER_{N}.csv, FV_{N}.csv, OURS_{N}.csv
 *   SW_r0_{N}.csv, SW_r25_{N}.csv, SW_r50_{N}.csv, SW_r75_{N}.csv, SW_r100_{N}.csv
 *
 * Skips algorithms whose CSV already exists.
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
#include "ours.h"

/* ============================================================================
 * CONFIGURATION
 * ============================================================================ */

#define MAX_N_VALUES 16
#define DEFAULT_N_VALUES {32, 64, 128, 256}
#define DEFAULT_NUM_N 4
#define DATA_DIR "data"

/* SW rho values (0, 0.25, 0.50, 0.75, 1.0) */
static const double SW_RHOS[] = {0.0, 0.25, 0.50, 0.75, 1.0};
static const int SW_NUM_RHOS = 5;
static const char *SW_RHO_NAMES[] = {"r0", "r25", "r50", "r75", "r100"};

/* Number of algorithm types per N: ER + FV + OURS + 5 SW = 8 */
#define NUM_ALGO_TYPES 8

/* Worker pool size (number of concurrent processes) */
#define POOL_SIZE 8

/* Task definition */
typedef struct {
    int n;
    int algo;  /* 0=ER, 1=FV, 2=OURS, 3-7=SW rho 0-4 */
} Task;

/* ============================================================================
 * FILE UTILITIES
 * ============================================================================ */

static int file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}

static void make_path(char *buf, size_t size, const char *algo, int n) {
    snprintf(buf, size, "%s/%s_%d.csv", DATA_DIR, algo, n);
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

static void save_ours_csv(int n, int *m_values, double *scores, int count) {
    char path[256];
    make_path(path, sizeof(path), "OURS", n);

    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot write to %s\n", path);
        return;
    }

    fprintf(fp, "m,score\n");
    for (int i = 0; i < count; i++) {
        fprintf(fp, "%d,%.10f\n", m_values[i], scores[i]);
    }
    fclose(fp);
    printf("  [OURS] Saved: %s\n", path);
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
    er_run(n, result);
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
    fv_run(n, result);
    save_fv_csv(n, result);
    fv_result_free(result);
}

static void run_ours(int n) {
    char path[256];
    make_path(path, sizeof(path), "OURS", n);

    if (file_exists(path)) {
        printf("  [SKIP] OURS_%d.csv already exists\n", n);
        return;
    }

    printf("  [OURS] Starting for N=%d...\n", n);
    int max_m = n * (n - 1) / 2;
    int step = n / 16;
    if (step < 2) step = 2;

    int num_points = 0;
    for (int m = n; m < max_m; m += step) num_points++;

    int *m_values = malloc((size_t)num_points * sizeof(int));
    double *scores = malloc((size_t)num_points * sizeof(double));

    int idx = 0;
    for (int m = n; m < max_m; m += step) {
        m_values[idx] = m;
        scores[idx] = ours_score(n, m);
        idx++;
    }
    printf("  [OURS] Done for N=%d\n", n);

    save_ours_csv(n, m_values, scores, num_points);
    free(m_values);
    free(scores);
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
    int step = n / 16;
    if (step < 2) step = 2;

    SWResult *result = sw_result_create(max_m, SW_RHOS[rho_idx]);
    sw_run(n, SW_RHOS[rho_idx], result, step);
    save_sw_csv(n, rho_idx, result);
    sw_result_free(result);
}


/* ============================================================================
 * TASK EXECUTION
 * ============================================================================ */

static void run_task(Task *task) {
    rng_seed((uint64_t)time(NULL) ^ (uint64_t)getpid());

    switch (task->algo) {
        case 0: run_er(task->n); break;
        case 1: run_fv(task->n); break;
        case 2: run_ours(task->n); break;
        default:
            /* SW rho index = algo - 3 */
            run_sw(task->n, task->algo - 3);
            break;
    }
}

/* ============================================================================
 * WORKER POOL
 * ============================================================================ */

static void run_worker_pool(Task *tasks, int num_tasks) {
    pid_t workers[POOL_SIZE];
    int active = 0;
    int next_task = 0;

    printf("  [POOL] %d tasks, %d workers\n", num_tasks, POOL_SIZE);
    fflush(stdout);

    while (next_task < num_tasks || active > 0) {
        /* Spawn workers up to POOL_SIZE */
        while (active < POOL_SIZE && next_task < num_tasks) {
            pid_t pid = fork();
            if (pid == 0) {
                /* Child: run one task and exit */
                run_task(&tasks[next_task]);
                exit(0);
            }
            workers[active] = pid;
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
                    workers[i] = workers[active - 1];
                    active--;
                    break;
                }
            }
        }
    }
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(int argc, char *argv[]) {
    rng_seed((uint64_t)time(NULL));

    /* Parse N values */
    int n_values[MAX_N_VALUES] = DEFAULT_N_VALUES;
    int num_n = DEFAULT_NUM_N;

    if (argc > 1) {
        num_n = argc - 1;
        if (num_n > MAX_N_VALUES) num_n = MAX_N_VALUES;
        for (int i = 0; i < num_n; i++) {
            n_values[i] = atoi(argv[i + 1]);
        }
    }

    printf("Algebraic Connectivity Benchmark (Worker Pool)\n");
    printf("N values: ");
    for (int i = 0; i < num_n; i++) {
        printf("%d ", n_values[i]);
    }
    printf("\n");

    /* Build task list: all (N, algo) combinations */
    int num_tasks = num_n * NUM_ALGO_TYPES;
    Task *tasks = malloc((size_t)num_tasks * sizeof(Task));

    int t = 0;
    for (int i = 0; i < num_n; i++) {
        for (int algo = 0; algo < NUM_ALGO_TYPES; algo++) {
            tasks[t].n = n_values[i];
            tasks[t].algo = algo;
            t++;
        }
    }

    printf("Running %d total tasks with %d workers\n", num_tasks, POOL_SIZE);
    fflush(stdout);

    /* Run all tasks via worker pool */
    run_worker_pool(tasks, num_tasks);

    free(tasks);

    printf("\n========================================\n");
    printf("All benchmarks complete!\n");
    printf("Results saved to: %s/\n", DATA_DIR);
    printf("========================================\n");

    return 0;
}
