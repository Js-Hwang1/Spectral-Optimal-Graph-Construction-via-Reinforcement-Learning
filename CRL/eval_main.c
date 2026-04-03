/*
 * eval_main.c — Fast C-based evaluation for CRL checkpoints.
 *
 * Runs REWIRE episodes with argmax action selection (no stochastic sampling).
 * Supports any graph size n (dynamic allocation, not limited by N_MAX).
 * Skips per-swap eigensolves — only computes final λ₂.
 *
 * Usage:
 *   ./crl_eval checkpoint.bin --n 16,24,36,64 --trials 5 --k-steps 20 --swap-frac 0.05
 *   ./crl_eval checkpoint.bin --n 64 --trials 5 --baselines baselines.csv
 */

#include "crl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include <sys/time.h>

/* ========================================================================== */
/* Timing                                                                      */
/* ========================================================================== */

static double wall_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* ========================================================================== */
/* Argmax over masked logits                                                   */
/* ========================================================================== */

static int argmax_masked(const float *logits, const uint8_t *mask, int total) {
    int best = -1;
    float best_val = -FLT_MAX;
    for (int i = 0; i < total; i++) {
        if (mask[i] && logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

/* Global flag: 0=argmax, 1=stochastic softmax sampling */
static int use_stochastic = 0;

static int select_action(const float *logits, const uint8_t *mask, int total, RNG *rng) {
    if (use_stochastic) {
        double lp;
        return softmax_sample(logits, mask, total, rng, &lp);
    }
    return argmax_masked(logits, mask, total);
}

/* ========================================================================== */
/* Batched rescore: two nodes at once for one MLP (BLAS-accelerated)            */
/* ========================================================================== */

/*
 * Rescore rows/cols for two nodes (a, b) using batched MLP forward.
 * Builds 4n features → one mlp_forward_batch call → scatter results.
 * work_feat: pre-allocated (4n * EDGE_FEAT_DIM) float buffer
 * work_logit: pre-allocated (4n) float buffer
 */
static void rescore_two_nodes(const MLP *mlp, const double *V_rr,
                               const int *degrees, int n, int step,
                               int k_steps, int node_a, int node_b,
                               float *logits,
                               float *work_feat, float *work_logit) {
    double nm1 = fmax(n - 1, 1);
    float step_frac = (float)step / fmax(k_steps, 1);

    /* node_a: row (node_a, j) */
    double v2a = V_rr[node_a * RR_K + 0];
    float dega = (float)(degrees[node_a] / nm1);
    for (int j = 0; j < n; j++) {
        float *f = work_feat + j * EDGE_FEAT_DIM;
        double v2j = V_rr[j * RR_K + 0];
        double gap = v2a - v2j;
        f[0] = (float)(n * gap * gap);
        f[1] = dega;
        f[2] = (float)(degrees[j] / nm1);
        f[3] = step_frac;
    }
    /* node_a: col (j, node_a) */
    for (int j = 0; j < n; j++) {
        float *f = work_feat + (n + j) * EDGE_FEAT_DIM;
        double v2j = V_rr[j * RR_K + 0];
        double gap = v2j - v2a;
        f[0] = (float)(n * gap * gap);
        f[1] = (float)(degrees[j] / nm1);
        f[2] = dega;
        f[3] = step_frac;
    }
    /* node_b: row (node_b, j) */
    double v2b = V_rr[node_b * RR_K + 0];
    float degb = (float)(degrees[node_b] / nm1);
    for (int j = 0; j < n; j++) {
        float *f = work_feat + (2 * n + j) * EDGE_FEAT_DIM;
        double v2j = V_rr[j * RR_K + 0];
        double gap = v2b - v2j;
        f[0] = (float)(n * gap * gap);
        f[1] = degb;
        f[2] = (float)(degrees[j] / nm1);
        f[3] = step_frac;
    }
    /* node_b: col (j, node_b) */
    for (int j = 0; j < n; j++) {
        float *f = work_feat + (3 * n + j) * EDGE_FEAT_DIM;
        double v2j = V_rr[j * RR_K + 0];
        double gap = v2j - v2b;
        f[0] = (float)(n * gap * gap);
        f[1] = (float)(degrees[j] / nm1);
        f[2] = degb;
        f[3] = step_frac;
    }

    mlp_forward_batch(mlp, work_feat, 4 * n, work_logit);

    /* Scatter results */
    for (int j = 0; j < n; j++) logits[node_a * n + j] = work_logit[j];
    for (int j = 0; j < n; j++) logits[j * n + node_a] = work_logit[n + j];
    for (int j = 0; j < n; j++) logits[node_b * n + j] = work_logit[2 * n + j];
    for (int j = 0; j < n; j++) logits[j * n + node_b] = work_logit[3 * n + j];
}

/* ========================================================================== */
/* Single eval episode (argmax, no reward computation, any n)                  */
/* ========================================================================== */

static double eval_episode(const PolicyWeights *pw, int n, int m,
                            int k_steps, double swap_frac, uint64_t seed) {
    int nn = n * n;
    int num_swaps = (int)(m * swap_frac);
    if (num_swaps < 1) num_swaps = 1;

    /* Dynamic allocation */
    uint8_t *adj = (uint8_t *)calloc((size_t)nn, 1);
    int *degrees = (int *)calloc((size_t)n, sizeof(int));
    double *V_rr = (double *)malloc((size_t)n * RR_K * sizeof(double));
    double *lams_rr = (double *)malloc((size_t)RR_K * sizeof(double));
    double *v_warm = (double *)malloc((size_t)n * sizeof(double));
    float *add_logits = (float *)malloc((size_t)nn * sizeof(float));
    float *rem_logits = (float *)malloc((size_t)nn * sizeof(float));
    uint8_t *bridge_mask = (uint8_t *)malloc((size_t)nn);

    /* BLAS batch workspace (reused across steps) */
    float *feat_batch = (float *)malloc((size_t)nn * EDGE_FEAT_DIM * sizeof(float));
    float *work_feat  = (float *)malloc((size_t)(4 * n) * EDGE_FEAT_DIM * sizeof(float));
    float *work_logit = (float *)malloc((size_t)(4 * n) * sizeof(float));

    RNG rng;
    rng_init(&rng, seed);
    build_ring_random(adj, degrees, n, m, &rng);

    int has_warm = 0;

    for (int kk = 0; kk < k_steps; kk++) {
        /* Lanczos -> RR subspace */
        lanczos_ext_k(adj, n, has_warm ? v_warm : NULL,
                      LANCZOS_K, RR_K, V_rr, lams_rr);
        for (int i = 0; i < n; i++) v_warm[i] = V_rr[i * RR_K + 0];
        has_warm = 1;

        /* Build all n² features contiguously, then batch-score */
        double nm1 = fmax(n - 1, 1);
        float step_frac = (float)kk / fmax(k_steps, 1);
        for (int i = 0; i < n; i++) {
            double v2i = V_rr[i * RR_K + 0];
            float degi = (float)(degrees[i] / nm1);
            for (int j = 0; j < n; j++) {
                float *f = feat_batch + (i * n + j) * EDGE_FEAT_DIM;
                double v2j = V_rr[j * RR_K + 0];
                double gap = v2i - v2j;
                f[0] = (float)(n * gap * gap);
                f[1] = degi;
                f[2] = (float)(degrees[j] / nm1);
                f[3] = step_frac;
            }
        }
        mlp_forward_batch(&pw->add_mlp, feat_batch, nn, add_logits);
        mlp_forward_batch(&pw->rem_mlp, feat_batch, nn, rem_logits);

        for (int s = 0; s < num_swaps; s++) {
            /* === ADD phase === */
            uint8_t *add_mask = (uint8_t *)calloc((size_t)nn, 1);
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                    if (!adj[i * n + j])
                        add_mask[i * n + j] = 1;

            int add_flat = select_action(add_logits, add_mask, nn, &rng);
            free(add_mask);
            if (add_flat < 0) break;

            int ai = add_flat / n, aj = add_flat % n;

            /* Add edge */
            adj[ai * n + aj] = 1;
            adj[aj * n + ai] = 1;
            degrees[ai]++;
            degrees[aj]++;

            /* RR update + batched rescore (2 nodes × 2 MLPs = 2 batch calls) */
            rr_update(V_rr, lams_rr, n, RR_K, ai, aj, +1.0);
            rescore_two_nodes(&pw->add_mlp, V_rr, degrees, n, kk, k_steps,
                              ai, aj, add_logits, work_feat, work_logit);
            rescore_two_nodes(&pw->rem_mlp, V_rr, degrees, n, kk, k_steps,
                              ai, aj, rem_logits, work_feat, work_logit);

            /* === REMOVE phase === */
            find_bridges(adj, n, bridge_mask);

            uint8_t *rem_mask = (uint8_t *)calloc((size_t)nn, 1);
            int has_valid = 0;
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                    if (adj[i * n + j] && !bridge_mask[i * n + j]) {
                        int a2 = ai < aj ? ai : aj;
                        int b2 = ai < aj ? aj : ai;
                        if (i == a2 && j == b2) continue;
                        rem_mask[i * n + j] = 1;
                        has_valid = 1;
                    }

            if (!has_valid) {
                /* Undo add */
                adj[ai * n + aj] = 0;
                adj[aj * n + ai] = 0;
                degrees[ai]--;
                degrees[aj]--;
                rr_update(V_rr, lams_rr, n, RR_K, ai, aj, -1.0);
                rescore_two_nodes(&pw->add_mlp, V_rr, degrees, n, kk, k_steps,
                                  ai, aj, add_logits, work_feat, work_logit);
                rescore_two_nodes(&pw->rem_mlp, V_rr, degrees, n, kk, k_steps,
                                  ai, aj, rem_logits, work_feat, work_logit);
                free(rem_mask);
                break;
            }

            int rem_flat = select_action(rem_logits, rem_mask, nn, &rng);
            free(rem_mask);

            if (rem_flat < 0) {
                /* Undo add */
                adj[ai * n + aj] = 0;
                adj[aj * n + ai] = 0;
                degrees[ai]--;
                degrees[aj]--;
                rr_update(V_rr, lams_rr, n, RR_K, ai, aj, -1.0);
                break;
            }

            int ri = rem_flat / n, rj = rem_flat % n;

            /* Remove edge */
            adj[ri * n + rj] = 0;
            adj[rj * n + ri] = 0;
            degrees[ri]--;
            degrees[rj]--;

            /* RR update + batched rescore */
            rr_update(V_rr, lams_rr, n, RR_K, ri, rj, -1.0);
            rescore_two_nodes(&pw->add_mlp, V_rr, degrees, n, kk, k_steps,
                              ri, rj, add_logits, work_feat, work_logit);
            rescore_two_nodes(&pw->rem_mlp, V_rr, degrees, n, kk, k_steps,
                              ri, rj, rem_logits, work_feat, work_logit);
        }
    }

    double final_l2 = exact_lambda2(adj, n);

    free(adj); free(degrees); free(V_rr); free(lams_rr);
    free(v_warm); free(add_logits); free(rem_logits); free(bridge_mask);
    free(feat_batch); free(work_feat); free(work_logit);

    return final_l2;
}

/* ========================================================================== */
/* Baseline CSV loading                                                        */
/* ========================================================================== */

/* Simple baseline entry: n, m, best_l2 */
typedef struct {
    int n, m;
    double fv, er, sw025, sw050, sw075;
    double best;
} BaselineEntry;

static BaselineEntry *baselines = NULL;
static int num_baselines = 0;

static void load_baselines(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Warning: cannot open baselines file %s\n", path);
        return;
    }
    char line[1024];
    /* Skip header */
    if (fgets(line, sizeof(line), f) == NULL) { fclose(f); return; }

    int capacity = 4096;
    baselines = (BaselineEntry *)malloc((size_t)capacity * sizeof(BaselineEntry));

    while (fgets(line, sizeof(line), f)) {
        BaselineEntry e = {0};
        if (sscanf(line, "%d,%d,%lf,%lf,%lf,%lf,%lf",
                   &e.n, &e.m, &e.fv, &e.er, &e.sw025, &e.sw050, &e.sw075) >= 2) {
            e.best = e.fv;
            if (e.er > e.best) e.best = e.er;
            if (e.sw025 > e.best) e.best = e.sw025;
            if (e.sw050 > e.best) e.best = e.sw050;
            if (e.sw075 > e.best) e.best = e.sw075;

            if (num_baselines >= capacity) {
                capacity *= 2;
                baselines = (BaselineEntry *)realloc(
                    baselines, (size_t)capacity * sizeof(BaselineEntry));
            }
            baselines[num_baselines++] = e;
        }
    }
    fclose(f);
    printf("Loaded %d baseline entries from %s\n", num_baselines, path);
}

static BaselineEntry *find_baseline(int n, int m) {
    for (int i = 0; i < num_baselines; i++)
        if (baselines[i].n == n && baselines[i].m == m)
            return &baselines[i];
    return NULL;
}

/* ========================================================================== */
/* Main                                                                        */
/* ========================================================================== */

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s checkpoint.bin --n 16,24,36 [options]\n", argv[0]);
        printf("  --n N1,N2,...       Graph sizes to evaluate\n");
        printf("  --trials T          Trials per (n,m) config (default: 5)\n");
        printf("  --k-steps K         Lanczos snapshots per episode (default: 20)\n");
        printf("  --swap-frac F       Fraction of edges to swap (default: 0.05)\n");
        printf("  --baselines FILE    CSV file with baselines (n,m,fv,er,sw025,sw050,sw075)\n");
        printf("  --out FILE          Output CSV path\n");
        printf("  --m-step S          Evaluate every S-th m value (default: 1)\n");
        printf("  --stochastic        Use softmax sampling instead of argmax\n");
        return 1;
    }

    char *checkpoint_path = argv[1];
    int n_values[64], num_n = 0;
    int num_trials = 5;
    int k_steps = 20;
    double swap_frac = 0.05;
    char *baselines_path = NULL;
    char *out_path = NULL;
    int m_step = 1;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            char *tok = strtok(argv[++i], ",");
            while (tok && num_n < 64) {
                n_values[num_n++] = atoi(tok);
                tok = strtok(NULL, ",");
            }
        } else if (strcmp(argv[i], "--trials") == 0 && i + 1 < argc)
            num_trials = atoi(argv[++i]);
        else if (strcmp(argv[i], "--k-steps") == 0 && i + 1 < argc)
            k_steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--swap-frac") == 0 && i + 1 < argc)
            swap_frac = atof(argv[++i]);
        else if (strcmp(argv[i], "--baselines") == 0 && i + 1 < argc)
            baselines_path = argv[++i];
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc)
            out_path = argv[++i];
        else if (strcmp(argv[i], "--m-step") == 0 && i + 1 < argc)
            m_step = atoi(argv[++i]);
        else if (strcmp(argv[i], "--stochastic") == 0)
            use_stochastic = 1;
    }

    if (num_n == 0) {
        fprintf(stderr, "Error: --n required\n");
        return 1;
    }

    /* Load checkpoint */
    PolicyWeights pw;
    uint32_t episode;
    float best_imp;
    if (load_checkpoint(checkpoint_path, &pw, &episode, &best_imp) != 0) {
        fprintf(stderr, "Failed to load checkpoint: %s\n", checkpoint_path);
        return 1;
    }
    printf("Loaded checkpoint: %s (episode %u)\n", checkpoint_path, episode);

    /* Load baselines if provided */
    if (baselines_path) load_baselines(baselines_path);

    /* Open output CSV */
    FILE *csv = NULL;
    if (out_path) {
        csv = fopen(out_path, "w");
        if (csv) fprintf(csv, "n,m,rl_mean,rl_std,best_baseline,win,tie\n");
    }

    printf("\n");
    printf("================================================================\n");
    printf("CRL Fast Eval (C) — K=%d, swap_frac=%.2f, trials=%d%s\n",
           k_steps, swap_frac, num_trials,
           use_stochastic ? " [STOCHASTIC]" : " [ARGMAX]");
    printf("OpenMP threads: %d\n", omp_get_max_threads());
    printf("================================================================\n");

    int total_wins = 0, total_ties = 0, total_configs = 0;

    for (int ni = 0; ni < num_n; ni++) {
        int n = n_values[ni];
        int max_m = n * (n - 1) / 2;
        int min_m = n - 1;
        int n_configs = 0;
        for (int m = min_m; m <= max_m; m += m_step) n_configs++;

        printf("\nn=%d: %d configs (m=%d..%d, step=%d)\n",
               n, n_configs, min_m, max_m, m_step);

        /* Build list of m values */
        int *m_list = (int *)malloc((size_t)n_configs * sizeof(int));
        {
            int idx = 0;
            for (int m = min_m; m <= max_m; m += m_step)
                m_list[idx++] = m;
        }

        /* Allocate results: all (config, trial) pairs */
        int total_jobs = n_configs * num_trials;
        double *all_l2 = (double *)malloc((size_t)total_jobs * sizeof(double));

        double t0 = wall_time();

        /* Parallel over ALL (config, trial) pairs */
        int progress_done = 0;
        #pragma omp parallel for schedule(dynamic, 1)
        for (int job = 0; job < total_jobs; job++) {
            int ci = job / num_trials;
            int ti = job % num_trials;
            int m = m_list[ci];
            uint64_t seed = (uint64_t)(n * 100000 + m * 1000 + ti);
            all_l2[job] = eval_episode(&pw, n, m, k_steps, swap_frac, seed);

            /* Progress (atomic) */
            int done;
            #pragma omp atomic capture
            done = ++progress_done;
            if (done % (total_jobs / 20 + 1) == 0 || done == total_jobs) {
                double elapsed = wall_time() - t0;
                double rate = done / fmax(elapsed, 0.001);
                printf("  %d/%d jobs (%.0f%%) | %.1f job/s | ETA %.0fs\r",
                       done, total_jobs, 100.0 * done / total_jobs,
                       rate, (total_jobs - done) / fmax(rate, 0.001));
                fflush(stdout);
            }
        }
        printf("\n");

        /* Aggregate results per config */
        int n_wins = 0, n_ties = 0;
        double sum_rl = 0, sum_best = 0;
        int n_with_baseline = 0;

        for (int ci = 0; ci < n_configs; ci++) {
            int m = m_list[ci];
            double *trials = &all_l2[ci * num_trials];

            double mean = 0, std_val = 0;
            for (int t = 0; t < num_trials; t++) mean += trials[t];
            mean /= num_trials;
            for (int t = 0; t < num_trials; t++)
                std_val += (trials[t] - mean) * (trials[t] - mean);
            std_val = sqrt(std_val / fmax(num_trials - 1, 1));

            sum_rl += mean;

            int win = 0, tie = 0;
            double best_bl = 0;
            BaselineEntry *bl = find_baseline(n, m);
            if (bl && bl->best > 0) {
                best_bl = bl->best;
                sum_best += best_bl;
                n_with_baseline++;
                if (mean > best_bl + 1e-6) {
                    win = 1; n_wins++; total_wins++;
                } else if (mean >= 0.95 * best_bl) {
                    tie = 1; n_ties++; total_ties++;
                }
            }
            total_configs++;

            if (csv)
                fprintf(csv, "%d,%d,%.6f,%.6f,%.6f,%d,%d\n",
                        n, m, mean, std_val, best_bl, win, tie);
        }

        double elapsed = wall_time() - t0;
        free(m_list);
        free(all_l2);
        double avg_rl = n_configs > 0 ? sum_rl / n_configs : 0;
        double avg_best = n_with_baseline > 0 ? sum_best / n_with_baseline : 0;

        printf("\n  n=%d: %d wins, %d ties / %d configs | "
               "avg RL=%.3f, avg best=%.3f (%.1f%%) | %.1fs\n",
               n, n_wins, n_ties, n_configs,
               avg_rl, avg_best,
               avg_best > 0 ? 100.0 * avg_rl / avg_best : 0,
               elapsed);
    }

    printf("\n================================================================\n");
    printf("TOTAL: %d wins, %d ties / %d configs\n",
           total_wins, total_ties, total_configs);
    printf("================================================================\n");

    if (csv) {
        fclose(csv);
        printf("CSV saved: %s\n", out_path);
    }

    free(baselines);
    return 0;
}
