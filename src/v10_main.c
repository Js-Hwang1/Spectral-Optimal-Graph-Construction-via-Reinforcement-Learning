/*
 * v10_main.c — Node-Centric O(N³) Graph Optimization
 *
 * Iterate over nodes with fresh warm Lanczos each iteration.
 * Per node: ADD best non-neighbor, REM worst non-bridge neighbor.
 * Fresh v₂ every iteration — zero staleness.
 *
 * Complexity: passes × N iterations × O(N²) Lanczos = O(N³).
 * Init: random spanning tree + random edges.
 *
 * Usage:
 *   ./crl_v10 --n 16,24 --seeds 5 --baselines baselines.csv
 *   ./crl_v10 --n 16 --passes 5 --proposal-tau 0.01
 */

#include "crl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <pthread.h>
#include <stdatomic.h>

/* ========================================================================== */
/* Timing                                                                      */
/* ========================================================================== */

static double wall_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* ========================================================================== */
/* Random spanning tree + random edges                                         */
/* ========================================================================== */

static void build_rst_random(uint8_t *adj, int *degrees, int n, int m, RNG *rng) {
    memset(adj, 0, (size_t)n * n);
    memset(degrees, 0, (size_t)n * sizeof(int));

    /* Random permutation */
    int *perm = (int *)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) perm[i] = i;
    for (int i = n - 1; i > 0; i--) {
        int j = rng_int(rng, i + 1);
        int tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }

    /* Random spanning tree: each node connects to a random earlier node */
    for (int i = 1; i < n; i++) {
        int u = perm[i];
        int v = perm[rng_int(rng, i)];
        adj[u * n + v] = adj[v * n + u] = 1;
        degrees[u]++; degrees[v]++;
    }
    free(perm);

    /* Fill remaining edges randomly */
    int edges = n - 1;
    while (edges < m) {
        int u = rng_int(rng, n);
        int v = rng_int(rng, n);
        if (u == v || adj[u * n + v]) continue;
        adj[u * n + v] = adj[v * n + u] = 1;
        degrees[u]++; degrees[v]++;
        edges++;
    }
}

/* ========================================================================== */
/* Softmax sampling with temperature                                           */
/* ========================================================================== */

static int softmax_sample_scores(const double *scores, int count,
                                  double tau, RNG *rng) {
    if (count <= 0) return -1;
    if (count == 1) return 0;

    double max_s = scores[0] / tau;
    for (int i = 1; i < count; i++) {
        double s = scores[i] / tau;
        if (s > max_s) max_s = s;
    }

    double sum = 0.0;
    double *probs = (double *)malloc((size_t)count * sizeof(double));
    for (int i = 0; i < count; i++) {
        probs[i] = exp(scores[i] / tau - max_s);
        sum += probs[i];
    }

    double r = rng_double(rng) * sum;
    double cumsum = 0.0;
    int chosen = count - 1;
    for (int i = 0; i < count; i++) {
        cumsum += probs[i];
        if (r <= cumsum) { chosen = i; break; }
    }
    free(probs);
    return chosen;
}

static int argmax_scores(const double *scores, int count) {
    if (count <= 0) return -1;
    int best = 0;
    for (int i = 1; i < count; i++)
        if (scores[i] > scores[best]) best = i;
    return best;
}

/* ========================================================================== */
/* v10 Node-Centric Optimization                                               */
/* ========================================================================== */

typedef struct {
    double best_lam2;
    double final_lam2;
    int    total_swaps;
} V10Result;

static V10Result v10_refine(int n, int m, int num_passes,
                             double proposal_tau, int greedy,
                             uint64_t seed) {
    V10Result result = {0};

    RNG rng;
    rng_init(&rng, seed);

    /* Allocate */
    uint8_t *adj       = (uint8_t *)calloc((size_t)n * n, 1);
    uint8_t *best_adj  = (uint8_t *)malloc((size_t)n * n);
    int     *degrees   = (int *)calloc((size_t)n, sizeof(int));
    double  *V_rr      = (double *)malloc((size_t)n * RR_K * sizeof(double));
    double  *lams_rr   = (double *)malloc((size_t)RR_K * sizeof(double));
    uint8_t *bridge_mask = (uint8_t *)calloc((size_t)n * n, 1);
    double  *v_warm    = (double *)malloc((size_t)n * sizeof(double));
    int     *perm      = (int *)malloc((size_t)n * sizeof(int));
    /* N for ADD candidates, N² for global REM candidates */
    size_t cand_cap = (size_t)n * n;
    int     *cand_idx  = (int *)malloc(cand_cap * sizeof(int));
    double  *cand_scr  = (double *)malloc(cand_cap * sizeof(double));

    /* Init graph: random spanning tree + random edges */
    build_rst_random(adj, degrees, n, m, &rng);
    memcpy(best_adj, adj, (size_t)n * n);

    /* Initial Lanczos (cold start) */
    lanczos_ext_k(adj, n, NULL, LANCZOS_K, RR_K, V_rr, lams_rr);
    double best_lam2_rr = lams_rr[0];
    int total_swaps = 0;

    for (int pass = 0; pass < num_passes; pass++) {

        /* Random permutation for node visit order */
        for (int i = 0; i < n; i++) perm[i] = i;
        for (int i = n - 1; i > 0; i--) {
            int j = rng_int(&rng, i + 1);
            int tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
        }

        for (int ni = 0; ni < n; ni++) {
            int node = perm[ni];

            /* Fresh warm Lanczos → exact v₂ */
            for (int i = 0; i < n; i++) v_warm[i] = V_rr[i * RR_K + 0];
            lanczos_ext_k(adj, n, v_warm, LANCZOS_K, RR_K, V_rr, lams_rr);

            double v2_node = V_rr[node * RR_K + 0];

            /* === ADD: score non-neighbors of node === */
            int num_add = 0;
            for (int j = 0; j < n; j++) {
                if (j == node || adj[node * n + j]) continue;
                double gap = v2_node - V_rr[j * RR_K + 0];
                cand_idx[num_add] = j;
                cand_scr[num_add] = gap * gap;
                num_add++;
            }
            if (num_add == 0) continue;  /* fully connected node */

            int pick = greedy ? argmax_scores(cand_scr, num_add)
                       : softmax_sample_scores(cand_scr, num_add,
                                               proposal_tau, &rng);
            int j_add = cand_idx[pick];

            /* Execute ADD */
            int ai = node < j_add ? node : j_add;
            int aj = node < j_add ? j_add : node;
            adj[ai * n + aj] = adj[aj * n + ai] = 1;
            degrees[ai]++; degrees[aj]++;
            rr_update(V_rr, lams_rr, n, RR_K, ai, aj, +1.0);

            /* Bridge detection on post-ADD graph — O(N+M) */
            find_bridges(adj, n, bridge_mask);

            /* === REM: score neighbors of node (non-bridge) === */
            /* Use RR-updated v₂ (one perturbation from Lanczos — very close) */
            v2_node = V_rr[node * RR_K + 0];
            int num_rem = 0;
            for (int j = 0; j < n; j++) {
                if (j == node || !adj[node * n + j]) continue;
                if (j == j_add) continue;  /* don't undo the add */
                if (bridge_mask[node * n + j]) continue;
                double gap = v2_node - V_rr[j * RR_K + 0];
                cand_idx[num_rem] = j;
                cand_scr[num_rem] = -(gap * gap);  /* prefer removing low-gap */
                num_rem++;
            }

            if (num_rem == 0) {
                /* Undo ADD — no valid removal target */
                adj[ai * n + aj] = adj[aj * n + ai] = 0;
                degrees[ai]--; degrees[aj]--;
                rr_update(V_rr, lams_rr, n, RR_K, ai, aj, -1.0);
                continue;
            }

            pick = greedy ? argmax_scores(cand_scr, num_rem)
                       : softmax_sample_scores(cand_scr, num_rem,
                                               proposal_tau, &rng);
            int j_rem = cand_idx[pick];

            /* Execute REM */
            int ri = node < j_rem ? node : j_rem;
            int rj = node < j_rem ? j_rem : node;
            adj[ri * n + rj] = adj[rj * n + ri] = 0;
            degrees[ri]--; degrees[rj]--;
            rr_update(V_rr, lams_rr, n, RR_K, ri, rj, -1.0);

            total_swaps++;

            /* Track best via RR-estimated λ₂ */
            if (lams_rr[0] > best_lam2_rr) {
                best_lam2_rr = lams_rr[0];
                memcpy(best_adj, adj, (size_t)n * n);
            }
        }
    }

    /* Exact evaluation on best graph */
    result.best_lam2 = exact_lambda2(best_adj, n);
    result.final_lam2 = exact_lambda2(adj, n);
    result.total_swaps = total_swaps;

    free(adj); free(best_adj); free(degrees);
    free(V_rr); free(lams_rr); free(bridge_mask);
    free(v_warm); free(perm);
    free(cand_idx); free(cand_scr);

    return result;
}

/* ========================================================================== */
/* Baseline CSV loading                                                        */
/* ========================================================================== */

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
        fprintf(stderr, "Error: cannot open baselines file %s\n", path);
        return;
    }
    char line[1024];
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
/* Parallel config runner                                                      */
/* ========================================================================== */

typedef struct {
    double v10_best;
    double fv_bl;
    double best_bl;
} ConfigResult;

typedef struct {
    int n;
    int *m_vals;
    int num_m;
    int num_passes;
    int num_seeds;
    double proposal_tau;
    int greedy;
    atomic_int *work_idx;
    atomic_int *done_count;
    ConfigResult *results;
    double t0;
} ThreadArgs;

static void *worker_fn(void *arg) {
    ThreadArgs *a = (ThreadArgs *)arg;
    while (1) {
        int mi = atomic_fetch_add(a->work_idx, 1);
        if (mi >= a->num_m) break;

        int m = a->m_vals[mi];
        double best_v10 = -1e30;
        for (int s = 0; s < a->num_seeds; s++) {
            uint64_t seed = (uint64_t)(a->n * 10000 + m * 100 + s + 1);
            V10Result r = v10_refine(a->n, m, a->num_passes,
                                      a->proposal_tau, a->greedy, seed);
            if (r.best_lam2 > best_v10) best_v10 = r.best_lam2;
        }

        BaselineEntry *bl = find_baseline(a->n, m);
        a->results[mi].v10_best = best_v10;
        a->results[mi].fv_bl = bl ? bl->fv : 0;
        a->results[mi].best_bl = bl ? bl->best : 0;

        int d = atomic_fetch_add(a->done_count, 1) + 1;
        if (d % 50 == 0 || d == a->num_m) {
            double elapsed = wall_time() - a->t0;
            printf("  %d/%d configs (%.1fs)\n", d, a->num_m, elapsed);
            fflush(stdout);
        }
    }
    return NULL;
}

/* ========================================================================== */
/* Main                                                                        */
/* ========================================================================== */

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s --n 16,24 [options]\n", argv[0]);
        printf("  --n N1,N2,...       Graph sizes to evaluate\n");
        printf("  --seeds S           Seeds per (n,m) config (default: 5)\n");
        printf("  --baselines FILE    CSV baselines file\n");
        printf("  --passes P          Passes over all nodes (default: 5)\n");
        printf("  --proposal-tau F    FV proposal temperature (default: 0.01)\n");
        printf("  --out FILE          Output CSV path\n");
        return 1;
    }

    int n_values[64], num_n = 0;
    int num_seeds = 5;
    int num_passes = 5;
    double proposal_tau = 0.01;
    int greedy = 0;
    int num_threads = 1;
    char *baselines_path = NULL;
    char *out_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            char *tok = strtok(argv[++i], ",");
            while (tok && num_n < 64) {
                n_values[num_n++] = atoi(tok);
                tok = strtok(NULL, ",");
            }
        }
        else if (strcmp(argv[i], "--seeds") == 0 && i + 1 < argc)
            num_seeds = atoi(argv[++i]);
        else if (strcmp(argv[i], "--baselines") == 0 && i + 1 < argc)
            baselines_path = argv[++i];
        else if (strcmp(argv[i], "--passes") == 0 && i + 1 < argc)
            num_passes = atoi(argv[++i]);
        else if (strcmp(argv[i], "--proposal-tau") == 0 && i + 1 < argc)
            proposal_tau = atof(argv[++i]);
        else if (strcmp(argv[i], "--greedy") == 0)
            greedy = 1;
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            num_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc)
            out_path = argv[++i];
    }

    if (num_n == 0) {
        fprintf(stderr, "Error: --n required\n");
        return 1;
    }

    if (baselines_path) load_baselines(baselines_path);

    printf("v10 Node-Centric (C) | passes=%d | %s | seeds=%d\n",
           num_passes, greedy ? "GREEDY" : "softmax", num_seeds);
    printf("=============================================\n\n");

    FILE *csv = NULL;
    if (out_path) {
        csv = fopen(out_path, "w");
        if (csv) fprintf(csv, "n,m,v10_best,fv_bl,best_bl,pct_fv,pct_best\n");
    }

    for (int ni = 0; ni < num_n; ni++) {
        int n = n_values[ni];
        int max_m = n * (n - 1) / 2;

        /* Collect all valid m values from baselines */
        int *m_vals = (int *)malloc((size_t)(max_m + 1) * sizeof(int));
        int num_m = 0;

        if (num_baselines > 0) {
            for (int m = n + 1; m <= max_m; m++) {
                BaselineEntry *bl = find_baseline(n, m);
                if (bl && bl->best > 0)
                    m_vals[num_m++] = m;
            }
        } else {
            for (int m = n + 1; m <= max_m; m += (max_m - n) / 10 + 1)
                m_vals[num_m++] = m;
        }

        if (num_m == 0) {
            printf("n=%d: no configs found, skipping\n", n);
            free(m_vals);
            continue;
        }

        printf("n=%d (%d configs, %d threads)\n", n, num_m, num_threads);
        double t0 = wall_time();

        ConfigResult *results = (ConfigResult *)malloc(
            (size_t)num_m * sizeof(ConfigResult));

        atomic_int work_idx = 0;
        atomic_int done_count = 0;

        ThreadArgs targs = {
            .n = n, .m_vals = m_vals, .num_m = num_m,
            .num_passes = num_passes, .num_seeds = num_seeds,
            .proposal_tau = proposal_tau, .greedy = greedy,
            .work_idx = &work_idx, .done_count = &done_count,
            .results = results, .t0 = t0
        };

        /* Launch threads */
        int nt = num_threads < num_m ? num_threads : num_m;
        pthread_t *threads = (pthread_t *)malloc((size_t)nt * sizeof(pthread_t));
        for (int t = 0; t < nt; t++)
            pthread_create(&threads[t], NULL, worker_fn, &targs);
        for (int t = 0; t < nt; t++)
            pthread_join(threads[t], NULL);
        free(threads);

        /* Reduce results */
        double sum_v10 = 0, sum_fv = 0, sum_best = 0;
        int wins_fv = 0, wins_best = 0, ties_fv = 0;
        for (int mi = 0; mi < num_m; mi++) {
            double v10 = results[mi].v10_best;
            double fv = results[mi].fv_bl;
            double best = results[mi].best_bl;
            sum_v10 += v10; sum_fv += fv; sum_best += best;
            if (v10 > fv + 1e-6) wins_fv++;
            else if (fabs(v10 - fv) < 1e-6) ties_fv++;
            if (v10 > best + 1e-6) wins_best++;

            if (csv && fv > 0) {
                fprintf(csv, "%d,%d,%.6f,%.6f,%.6f,%.1f,%.1f\n",
                        n, m_vals[mi], v10, fv, best,
                        v10 / fv * 100.0, best > 0 ? v10 / best * 100.0 : 0.0);
            }
        }

        int total_configs = num_m;
        double avg_v10 = sum_v10 / total_configs;
        double avg_fv = sum_fv / total_configs;
        double avg_best = sum_best / total_configs;
        double pct_fv = avg_fv > 0 ? avg_v10 / avg_fv * 100.0 : 0;
        double pct_best = avg_best > 0 ? avg_v10 / avg_best * 100.0 : 0;
        double elapsed = wall_time() - t0;

        printf("\n  n=%d SUMMARY (%d configs, %.1fs):\n", n, total_configs, elapsed);
        printf("    v10 avg:    %.4f\n", avg_v10);
        printf("    FV avg:     %.4f\n", avg_fv);
        printf("    Best avg:   %.4f\n", avg_best);
        printf("    vs FV:      %.1f%% (%dW %dT)\n", pct_fv, wins_fv, ties_fv);
        printf("    vs Best:    %.1f%% (%dW)\n", pct_best, wins_best);
        printf("\n");

        free(results);

        free(m_vals);
    }

    if (csv) {
        fclose(csv);
        printf("Results written to %s\n", out_path);
    }
    free(baselines);

    return 0;
}
