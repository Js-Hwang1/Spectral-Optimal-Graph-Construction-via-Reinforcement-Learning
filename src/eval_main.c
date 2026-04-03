/*
 * eval_main.c — Dual-Phase Spectral Refine evaluation.
 *
 * Runs dual-phase episodes with MLP-augmented scoring.
 * Supports any graph size n (dynamic allocation).
 *
 * Usage:
 *   ./crl_eval checkpoint.bin --n 16,24,36,64 --trials 5 --baselines baselines.csv
 *   ./crl_eval checkpoint.bin --n 64 --trials 5 --cycle-mult 4.0 --batch-frac 0.1
 */

#include "crl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include <sys/time.h>

static double wall_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* ========================================================================== */
/* Single eval episode (any n, dynamic alloc)                                  */
/* ========================================================================== */

/* Global flags */
static int use_metropolis = 0;       /* eigensolve Metropolis (expensive, validation) */
static int use_lanczos_metro = 0;    /* Lanczos-based Metropolis (cheap, O(N²)) */
static int use_tabu = 0;            /* tabu search (no eigensolve, O(1) per edge) */
static double metro_T_start = 1.0, metro_T_end = 0.001;
static double tabu_tenure_frac = 0.1; /* tabu tenure = frac * effective_K */

static int use_ref_augmented = 0;    /* two-pass with reference graph */
static int use_ga = 0;              /* GA: parallel refine + crossover */
static int ga_pop = 2;              /* GA population size */

/* Forward declaration */
static double eval_episode(const PolicyWeights *pw, int n, int m,
                            int num_cycles, double swap_frac,
                            double tau_start, double tau_end,
                            uint64_t seed,
                            const uint8_t *ref_adj,
                            uint8_t *out_best_adj);

/* ========================================================================== */
/* FV-guided crossover: union of two parents → prune to m edges                */
/* ========================================================================== */

static void fv_crossover(const uint8_t *parent_a, const uint8_t *parent_b,
                          int n, int m, uint8_t *child, int *child_degrees) {
    int nn = n * n;

    /* Union */
    uint8_t *uadj = (uint8_t *)calloc((size_t)nn, 1);
    for (int i = 0; i < nn; i++)
        uadj[i] = parent_a[i] | parent_b[i];

    int union_m = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (uadj[i * n + j]) union_m++;

    if (union_m <= m) {
        /* Union already has ≤ m edges, just use it */
        memcpy(child, uadj, (size_t)nn);
        memset(child_degrees, 0, (size_t)n * sizeof(int));
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (child[i * n + j]) child_degrees[i]++;
        free(uadj);
        return;
    }

    /* Lanczos on union → Fiedler vector */
    double *v2 = (double *)malloc((size_t)n * sizeof(double));
    double lam[1];
    lanczos_ext_k(uadj, n, NULL, LANCZOS_K, 1, v2, lam);

    /* Score all union edges by FV gap (ascending = expendable first) */
    int to_remove = union_m - m;
    typedef struct { double gap; int i, j; } scored_edge;
    scored_edge *edges = (scored_edge *)malloc((size_t)union_m * sizeof(scored_edge));
    int ne = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (uadj[i * n + j]) {
                double g = v2[i] - v2[j];
                edges[ne].gap = g * g;
                edges[ne].i = i;
                edges[ne].j = j;
                ne++;
            }

    /* Sort by gap ascending (lowest gap = most expendable) */
    for (int i = 0; i < ne - 1; i++)
        for (int j = i + 1; j < ne; j++)
            if (edges[j].gap < edges[i].gap) {
                scored_edge tmp = edges[i];
                edges[i] = edges[j];
                edges[j] = tmp;
            }

    /* Remove lowest-gap edges, skip if removal disconnects */
    int removed = 0;
    for (int k = 0; k < ne && removed < to_remove; k++) {
        int ei = edges[k].i, ej = edges[k].j;
        uadj[ei * n + ej] = uadj[ej * n + ei] = 0;

        /* Quick BFS connectivity check */
        int *visited = (int *)calloc((size_t)n, sizeof(int));
        int *queue = (int *)malloc((size_t)n * sizeof(int));
        visited[0] = 1;
        queue[0] = 0;
        int head = 0, tail = 1, count = 1;
        while (head < tail) {
            int u = queue[head++];
            for (int v = 0; v < n; v++)
                if (uadj[u * n + v] && !visited[v]) {
                    visited[v] = 1;
                    queue[tail++] = v;
                    count++;
                }
        }
        free(visited);
        free(queue);

        if (count < n) {
            /* Disconnected — undo */
            uadj[ei * n + ej] = uadj[ej * n + ei] = 1;
        } else {
            removed++;
        }
    }

    memcpy(child, uadj, (size_t)nn);
    memset(child_degrees, 0, (size_t)n * sizeof(int));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (child[i * n + j]) child_degrees[i]++;

    free(uadj);
    free(v2);
    free(edges);
}

/* ========================================================================== */
/* GA eval episode                                                             */
/* ========================================================================== */

static double ga_eval_episode(const PolicyWeights *pw, int n, int m,
                               int num_cycles, double swap_frac,
                               double tau_start, double tau_end,
                               uint64_t seed) {
    int nn = n * n;
    int P = ga_pop;

    /* Budget: split 75% for initial refine, 25% for post-crossover refine */
    int K_phase1 = (int)(0.75 * num_cycles);
    int K_phase3 = num_cycles - K_phase1;
    if (K_phase1 < 1) K_phase1 = 1;
    if (K_phase3 < 1) K_phase3 = 1;

    /* Phase 1: independent refine for each individual */
    uint8_t **pop_adj = (uint8_t **)malloc((size_t)P * sizeof(uint8_t *));
    double *pop_l2 = (double *)malloc((size_t)P * sizeof(double));

    for (int p = 0; p < P; p++) {
        pop_adj[p] = (uint8_t *)malloc((size_t)nn);
        uint64_t p_seed = seed + (uint64_t)p * 999983;

        /* Run refine with K_phase1/P cycles per individual */
        int k_per_ind = K_phase1 / P;
        if (k_per_ind < 1) k_per_ind = 1;

        pop_l2[p] = eval_episode(pw, n, m, k_per_ind, swap_frac,
                                  tau_start, tau_end, p_seed,
                                  NULL, pop_adj[p]);
    }

    /* Phase 2: crossover best pair */
    /* Find two best */
    int best1 = 0, best2 = 1;
    if (P > 1 && pop_l2[1] > pop_l2[0]) { best1 = 1; best2 = 0; }
    for (int p = 2; p < P; p++) {
        if (pop_l2[p] > pop_l2[best1]) {
            best2 = best1;
            best1 = p;
        } else if (pop_l2[p] > pop_l2[best2]) {
            best2 = p;
        }
    }

    uint8_t *child_adj = (uint8_t *)calloc((size_t)nn, 1);
    int *child_deg = (int *)calloc((size_t)n, sizeof(int));
    fv_crossover(pop_adj[best1], pop_adj[best2], n, m, child_adj, child_deg);

    /* Phase 3: refine the child */
    /* Set up child as initial graph for refine by using it as starting point.
     * We can't directly pass an initial adj to eval_episode, so we use the
     * crossover result's λ₂ as the baseline and run a fresh refine seeded
     * to produce the child. Instead, let's just compute child's λ₂ and do
     * a simpler approach: run a refine starting from the child graph. */

    /* For simplicity: evaluate child directly + run one more refine pass
     * using child as reference (ref-augmented) */
    double child_l2 = exact_lambda2(child_adj, n);

    /* Run a final refine pass with child as reference */
    uint64_t final_seed = seed + 7777777;
    double refine_l2 = eval_episode(pw, n, m, K_phase3, swap_frac,
                                     tau_start, tau_end, final_seed,
                                     child_adj, NULL);

    /* Best of all */
    double result = child_l2;
    for (int p = 0; p < P; p++)
        if (pop_l2[p] > result) result = pop_l2[p];
    if (refine_l2 > result) result = refine_l2;

    /* Cleanup */
    for (int p = 0; p < P; p++) free(pop_adj[p]);
    free(pop_adj);
    free(pop_l2);
    free(child_adj);
    free(child_deg);

    return result;
}

static double eval_episode(const PolicyWeights *pw, int n, int m,
                            int num_cycles, double swap_frac,
                            double tau_start, double tau_end,
                            uint64_t seed,
                            const uint8_t *ref_adj,
                            uint8_t *out_best_adj) {
    int nn = n * n;
    int max_pairs = n * (n - 1) / 2;

    uint8_t *adj      = (uint8_t *)calloc((size_t)nn, 1);
    int     *degrees  = (int *)calloc((size_t)n, sizeof(int));
    double  *v2       = (double *)malloc((size_t)n * sizeof(double));
    double  *v_warm   = (double *)malloc((size_t)n * sizeof(double));
    double   lam_out[1];
    uint8_t *bridge_mask = (uint8_t *)malloc((size_t)nn);
    uint8_t *added_mask  = (uint8_t *)calloc((size_t)nn, 1);
    uint8_t *best_adj    = (uint8_t *)malloc((size_t)nn);

    int    *ci = (int *)malloc((size_t)max_pairs * sizeof(int));
    int    *cj = (int *)malloc((size_t)max_pairs * sizeof(int));
    float  *feat = (float *)malloc((size_t)max_pairs * EDGE_FEAT_DIM * sizeof(float));
    double *fv_gaps = (double *)malloc((size_t)max_pairs * sizeof(double));
    double *scores = (double *)malloc((size_t)max_pairs * sizeof(double));
    int    *sel = (int *)malloc((size_t)(n + 1) * sizeof(int));
    double *lp = (double *)malloc((size_t)(n + 1) * sizeof(double));
    int    *add_ci_buf = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int    *add_cj_buf = (int *)malloc((size_t)(n + 1) * sizeof(int));

    RNG rng;
    rng_init(&rng, seed);
    build_ring_random(adj, degrees, n, m, &rng);

    /* Compute reference Fiedler if provided */
    double *ref_v2 = NULL;
    if (ref_adj) {
        ref_v2 = (double *)malloc((size_t)n * sizeof(double));
        double ref_lam[1];
        lanczos_ext_k(ref_adj, n, NULL, LANCZOS_K, 1, ref_v2, ref_lam);
    }

    int has_warm = 0;
    double best_l2 = -1e30;
    double cur_l2 = exact_lambda2(adj, n);
    best_l2 = cur_l2;
    memcpy(best_adj, adj, (size_t)nn);

    /* Metropolis state (needed for either mode) */
    int need_metro = use_metropolis || use_lanczos_metro;
    uint8_t *saved_adj = NULL;
    int *saved_degrees = NULL;
    if (need_metro) {
        saved_adj = (uint8_t *)malloc((size_t)nn);
        saved_degrees = (int *)malloc((size_t)n * sizeof(int));
    }

    /* Tabu lists: tabu_add[i*n+j] = epoch until which we can't ADD (i,j)
     *             tabu_rem[i*n+j] = epoch until which we can't REMOVE (i,j) */
    int *tabu_add_list = NULL, *tabu_rem_list = NULL;
    int tabu_tenure = 0;
    if (use_tabu) {
        tabu_add_list = (int *)calloc((size_t)nn, sizeof(int));
        tabu_rem_list = (int *)calloc((size_t)nn, sizeof(int));
    }

    int total_swaps = (int)(m * swap_frac);
    if (total_swaps < 1) total_swaps = 1;
    int effective_K = num_cycles < total_swaps ? num_cycles : total_swaps;
    int spe = (total_swaps + effective_K - 1) / effective_K;
    if (spe < 1) spe = 1;

    if (use_tabu) {
        tabu_tenure = (int)(tabu_tenure_frac * effective_K);
        if (tabu_tenure < 1) tabu_tenure = 1;
    }

    int swap_count = 0;

    for (int epoch = 0; epoch < effective_K; epoch++) {
        if (swap_count >= total_swaps) break;

        double progress = (effective_K > 1)
            ? (double)epoch / (effective_K - 1) : 0.0;
        double tau = tau_start * pow(tau_end / fmax(tau_start, 1e-15), progress);

        int B = spe;
        int remaining = total_swaps - swap_count;
        if (B > remaining) B = remaining;

        /* Save state for potential Metropolis revert */
        double saved_l2 = cur_l2;
        double lam2_before_epoch = lam_out[0]; /* will be set by Lanczos #1 below */
        if (need_metro) {
            memcpy(saved_adj, adj, (size_t)nn);
            memcpy(saved_degrees, degrees, (size_t)n * sizeof(int));
        }

        /* === ADD PHASE === */
        lanczos_ext_k(adj, n, has_warm ? v_warm : NULL,
                      LANCZOS_K, 1, v2, lam_out);
        memcpy(v_warm, v2, (size_t)n * sizeof(double));
        has_warm = 1;
        lam2_before_epoch = lam_out[0];

        int num_add_cand = 0;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (!adj[i * n + j]) {
                    /* skip if tabu (recently removed, can't re-add yet) */
                    if (use_tabu && tabu_add_list[i * n + j] > epoch)
                        continue;
                    ci[num_add_cand] = i;
                    cj[num_add_cand] = j;
                    num_add_cand++;
                }
        if (num_add_cand == 0) continue;

        build_candidate_features(v2, degrees, n, progress,
                                 ci, cj, num_add_cand,
                                 ref_adj, ref_v2, feat, fv_gaps);
        score_candidates(&pw->add_mlp, feat, fv_gaps,
                         num_add_cand, 0, scores);

        int actual_B = B < num_add_cand ? B : num_add_cand;
        int n_added = softmax_sample_batch(scores, num_add_cand, actual_B,
                                           tau, &rng, sel, lp);

        memset(added_mask, 0, (size_t)nn);
        for (int s = 0; s < n_added; s++) {
            int ai = ci[sel[s]], aj = cj[sel[s]];
            adj[ai * n + aj] = adj[aj * n + ai] = 1;
            degrees[ai]++;
            degrees[aj]++;
            added_mask[ai * n + aj] = added_mask[aj * n + ai] = 1;
            add_ci_buf[s] = ai;
            add_cj_buf[s] = aj;
            /* tabu: can't remove this edge for tenure epochs */
            if (use_tabu) {
                tabu_rem_list[ai * n + aj] = epoch + tabu_tenure;
                tabu_rem_list[aj * n + ai] = epoch + tabu_tenure;
            }
        }

        /* === LANCZOS REFRESH === */
        lanczos_ext_k(adj, n, v_warm, LANCZOS_K, 1, v2, lam_out);
        memcpy(v_warm, v2, (size_t)n * sizeof(double));

        /* === REMOVE PHASE === */
        find_bridges(adj, n, bridge_mask);

        int num_rem_cand = 0;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (adj[i * n + j] && !added_mask[i * n + j] &&
                    !bridge_mask[i * n + j]) {
                    /* skip if tabu (recently added, can't remove yet) */
                    if (use_tabu && tabu_rem_list[i * n + j] > epoch)
                        continue;
                    ci[num_rem_cand] = i;
                    cj[num_rem_cand] = j;
                    num_rem_cand++;
                }

        if (num_rem_cand == 0) {
            for (int s = 0; s < n_added; s++) {
                adj[add_ci_buf[s] * n + add_cj_buf[s]] = 0;
                adj[add_cj_buf[s] * n + add_ci_buf[s]] = 0;
                degrees[add_ci_buf[s]]--;
                degrees[add_cj_buf[s]]--;
            }
            continue;
        }

        build_candidate_features(v2, degrees, n, progress,
                                 ci, cj, num_rem_cand,
                                 ref_adj, ref_v2, feat, fv_gaps);
        score_candidates(&pw->rem_mlp, feat, fv_gaps,
                         num_rem_cand, 1, scores);

        int to_remove = n_added < num_rem_cand ? n_added : num_rem_cand;
        int n_removed = softmax_sample_batch(scores, num_rem_cand, to_remove,
                                             tau, &rng, sel, lp);

        for (int s = 0; s < n_removed; s++) {
            int ri = ci[sel[s]], rj = cj[sel[s]];
            adj[ri * n + rj] = adj[rj * n + ri] = 0;
            degrees[ri]--;
            degrees[rj]--;
            /* tabu: can't re-add this edge for tenure epochs */
            if (use_tabu) {
                tabu_add_list[ri * n + rj] = epoch + tabu_tenure;
                tabu_add_list[rj * n + ri] = epoch + tabu_tenure;
            }
        }

        if (n_removed < n_added) {
            int excess = n_added - n_removed;
            for (int s = n_added - 1; s >= n_added - excess; s--) {
                int ai = add_ci_buf[s], aj = add_cj_buf[s];
                if (adj[ai * n + aj]) {
                    adj[ai * n + aj] = adj[aj * n + ai] = 0;
                    degrees[ai]--;
                    degrees[aj]--;
                }
            }
        }

        swap_count += n_added;

        if (use_metropolis) {
            /* Eigensolve Metropolis (expensive, for validation) */
            double T = metro_T_start * pow(metro_T_end / fmax(metro_T_start, 1e-15), progress);
            double new_l2 = exact_lambda2(adj, n);
            double delta = new_l2 - saved_l2;
            int accept;
            if (delta > 0) accept = 1;
            else if (T > 1e-15) accept = rng_double(&rng) < exp(delta / T);
            else accept = 0;

            if (accept) {
                cur_l2 = new_l2;
                if (cur_l2 > best_l2) { best_l2 = cur_l2; memcpy(best_adj, adj, (size_t)nn); }
            } else {
                memcpy(adj, saved_adj, (size_t)nn);
                memcpy(degrees, saved_degrees, (size_t)n * sizeof(int));
                cur_l2 = saved_l2;
            }

        } else if (use_lanczos_metro) {
            /* Lanczos Metropolis (cheap, O(N²)) */
            double T = metro_T_start * pow(metro_T_end / fmax(metro_T_start, 1e-15), progress);

            /* 3rd Lanczos: estimate λ₂ after removes */
            lanczos_ext_k(adj, n, v_warm, LANCZOS_K, 1, v2, lam_out);
            memcpy(v_warm, v2, (size_t)n * sizeof(double));
            double lam2_after_epoch = lam_out[0];
            double delta = lam2_after_epoch - lam2_before_epoch;

            int accept;
            if (delta > 0) accept = 1;
            else if (T > 1e-15) accept = rng_double(&rng) < exp(delta / T);
            else accept = 0;

            if (accept) {
                /* Periodic exact eigensolve for best-tracking */
                int check_freq = effective_K / 10;
                if (check_freq < 1) check_freq = 1;
                if ((epoch + 1) % check_freq == 0 || epoch == effective_K - 1) {
                    double l2_now = exact_lambda2(adj, n);
                    cur_l2 = l2_now;
                    if (l2_now > best_l2) { best_l2 = l2_now; memcpy(best_adj, adj, (size_t)nn); }
                }
            } else {
                memcpy(adj, saved_adj, (size_t)nn);
                memcpy(degrees, saved_degrees, (size_t)n * sizeof(int));
                cur_l2 = saved_l2;
            }

        } else {
            /* No Metropolis: track best via periodic eigensolve */
            int check_freq = effective_K / 10;
            if (check_freq < 1) check_freq = 1;
            if ((epoch + 1) % check_freq == 0 || epoch == effective_K - 1) {
                double l2_now = exact_lambda2(adj, n);
                cur_l2 = l2_now;
                if (l2_now > best_l2) { best_l2 = l2_now; memcpy(best_adj, adj, (size_t)nn); }
            }
        }
    }

    double final_l2 = exact_lambda2(best_adj, n);
    double end_l2 = exact_lambda2(adj, n);
    if (end_l2 > final_l2) {
        final_l2 = end_l2;
        memcpy(best_adj, adj, (size_t)nn);
    }

    if (out_best_adj) memcpy(out_best_adj, best_adj, (size_t)nn);

    free(adj); free(degrees); free(v2); free(v_warm);
    free(bridge_mask); free(added_mask); free(best_adj);
    free(ci); free(cj); free(feat); free(fv_gaps);
    free(scores); free(sel); free(lp);
    free(add_ci_buf); free(add_cj_buf);
    if (saved_adj) free(saved_adj);
    if (saved_degrees) free(saved_degrees);
    if (ref_v2) free(ref_v2);
    if (tabu_add_list) free(tabu_add_list);
    if (tabu_rem_list) free(tabu_rem_list);

    return final_l2;
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
        fprintf(stderr, "Warning: cannot open baselines file %s\n", path);
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
/* Main                                                                        */
/* ========================================================================== */

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s checkpoint.bin --n 16,24,36 [options]\n", argv[0]);
        printf("  --n N1,N2,...       Graph sizes to evaluate\n");
        printf("  --trials T          Trials per (n,m) config (default: 5)\n");
        printf("  --cycle-mult F      K = F*n Lanczos epochs (default: 20.0)\n");
        printf("  --swap-frac F       Fraction of edges to swap (default: 0.05)\n");
        printf("  --batch-frac F      Batch = F*n edges per phase (default: 0.1)\n");
        printf("  --tau-start F       Start temperature (default: 0.2)\n");
        printf("  --tau-end F         End temperature (default: 0.002)\n");
        printf("  --baselines FILE    CSV baselines file\n");
        printf("  --out FILE          Output CSV path\n");
        printf("  --m-step S          Evaluate every S-th m value (default: 1)\n");
        return 1;
    }

    char *checkpoint_path = argv[1];
    int n_values[64], num_n = 0;
    int num_trials = 5;
    double cycle_mult = 20.0;
    double swap_frac = 0.05;
    double tau_start = 0.2;
    double tau_end = 0.002;
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
        else if (strcmp(argv[i], "--cycle-mult") == 0 && i + 1 < argc)
            cycle_mult = atof(argv[++i]);
        else if (strcmp(argv[i], "--swap-frac") == 0 && i + 1 < argc)
            swap_frac = atof(argv[++i]);
        else if (strcmp(argv[i], "--tau-start") == 0 && i + 1 < argc)
            tau_start = atof(argv[++i]);
        else if (strcmp(argv[i], "--tau-end") == 0 && i + 1 < argc)
            tau_end = atof(argv[++i]);
        else if (strcmp(argv[i], "--baselines") == 0 && i + 1 < argc)
            baselines_path = argv[++i];
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc)
            out_path = argv[++i];
        else if (strcmp(argv[i], "--m-step") == 0 && i + 1 < argc)
            m_step = atoi(argv[++i]);
        else if (strcmp(argv[i], "--metropolis") == 0)
            use_metropolis = 1;
        else if (strcmp(argv[i], "--lanczos-metro") == 0)
            use_lanczos_metro = 1;
        else if (strcmp(argv[i], "--tabu") == 0)
            use_tabu = 1;
        else if (strcmp(argv[i], "--tabu-tenure") == 0 && i + 1 < argc)
            tabu_tenure_frac = atof(argv[++i]);
        else if (strcmp(argv[i], "--ref-augmented") == 0)
            use_ref_augmented = 1;
        else if (strcmp(argv[i], "--ga") == 0)
            use_ga = 1;
        else if (strcmp(argv[i], "--ga-pop") == 0 && i + 1 < argc)
            ga_pop = atoi(argv[++i]);
    }

    if (num_n == 0) {
        fprintf(stderr, "Error: --n required\n");
        return 1;
    }

    PolicyWeights pw;
    uint32_t episode;
    float best_imp;
    if (load_checkpoint(checkpoint_path, &pw, &episode, &best_imp) != 0) {
        fprintf(stderr, "Failed to load checkpoint: %s\n", checkpoint_path);
        return 1;
    }
    printf("Loaded checkpoint: %s (episode %u)\n", checkpoint_path, episode);

    if (baselines_path) load_baselines(baselines_path);

    FILE *csv = NULL;
    if (out_path) {
        csv = fopen(out_path, "w");
        if (csv) fprintf(csv, "n,m,rl_mean,rl_std,best_baseline,win,tie\n");
    }

    printf("\n");
    printf("================================================================\n");
    printf("CRL Eval — K=%.0f*N, swap_frac=%.2f, tau=%.3f->%.4f%s\n",
           cycle_mult, swap_frac, tau_start, tau_end,
           use_metropolis ? " [METROPOLIS]" :
           use_lanczos_metro ? " [LANCZOS-METRO]" :
           use_tabu ? " [TABU]" :
           use_ref_augmented ? " [REF-AUGMENTED]" :
           use_ga ? " [GA]" : "");
    printf("trials=%d, OpenMP threads: %d\n", num_trials, omp_get_max_threads());
    printf("================================================================\n");

    int total_wins = 0, total_ties = 0, total_configs = 0;

    for (int ni = 0; ni < num_n; ni++) {
        int n = n_values[ni];
        int max_m = n * (n - 1) / 2;
        int min_m = n - 1;
        int n_configs = 0;
        for (int mv = min_m; mv <= max_m; mv += m_step) n_configs++;

        int num_cycles = (int)(cycle_mult * n);
        if (num_cycles < 1) num_cycles = 1;

        printf("\nn=%d: %d configs, K=%d, swap_frac=%.2f\n",
               n, n_configs, num_cycles, swap_frac);

        int *m_list = (int *)malloc((size_t)n_configs * sizeof(int));
        { int idx = 0; for (int mv = min_m; mv <= max_m; mv += m_step) m_list[idx++] = mv; }

        int total_jobs = n_configs * num_trials;
        double *all_l2 = (double *)malloc((size_t)total_jobs * sizeof(double));

        double t0 = wall_time();
        int progress_done = 0;

        #pragma omp parallel for schedule(dynamic, 1)
        for (int job = 0; job < total_jobs; job++) {
            int ci2 = job / num_trials;
            int ti = job % num_trials;
            int mv = m_list[ci2];
            uint64_t s = (uint64_t)(n * 100000 + mv * 1000 + ti);
            if (use_ga) {
                all_l2[job] = ga_eval_episode(&pw, n, mv, num_cycles,
                                               swap_frac, tau_start, tau_end, s);
            } else if (use_ref_augmented) {
                /* Two-pass: first without ref, second with ref=first's best */
                int nn2 = n * n;
                uint8_t *ref = (uint8_t *)malloc((size_t)nn2);
                double l2_1 = eval_episode(&pw, n, mv, num_cycles, swap_frac,
                                            tau_start, tau_end, s,
                                            NULL, ref);
                double l2_2 = eval_episode(&pw, n, mv, num_cycles, swap_frac,
                                            tau_start, tau_end, s + 500000,
                                            ref, NULL);
                free(ref);
                all_l2[job] = fmax(l2_1, l2_2);
            } else {
                all_l2[job] = eval_episode(&pw, n, mv, num_cycles, swap_frac,
                                            tau_start, tau_end, s,
                                            NULL, NULL);
            }

            int done;
            #pragma omp atomic capture
            done = ++progress_done;
            if (done % (total_jobs / 20 + 1) == 0 || done == total_jobs) {
                double elapsed = wall_time() - t0;
                double rate = done / fmax(elapsed, 0.001);
                printf("  %d/%d (%.0f%%) %.1f job/s ETA %.0fs\r",
                       done, total_jobs, 100.0 * done / total_jobs,
                       rate, (total_jobs - done) / fmax(rate, 0.001));
                fflush(stdout);
            }
        }
        printf("\n");

        int n_wins = 0, n_ties = 0;
        double sum_rl = 0, sum_best = 0;
        int n_with_baseline = 0;

        for (int ci2 = 0; ci2 < n_configs; ci2++) {
            int mv = m_list[ci2];
            double *trials = &all_l2[ci2 * num_trials];

            double mean = 0, std_val = 0;
            for (int t = 0; t < num_trials; t++) mean += trials[t];
            mean /= num_trials;
            for (int t = 0; t < num_trials; t++)
                std_val += (trials[t] - mean) * (trials[t] - mean);
            std_val = sqrt(std_val / fmax(num_trials - 1, 1));

            sum_rl += mean;

            int win = 0, tie = 0;
            double best_bl = 0;
            BaselineEntry *bl = find_baseline(n, mv);
            if (bl && bl->best > 0) {
                best_bl = bl->best;
                sum_best += best_bl;
                n_with_baseline++;
                if (mean > best_bl + 1e-6) { win = 1; n_wins++; total_wins++; }
                else if (mean >= 0.95 * best_bl) { tie = 1; n_ties++; total_ties++; }
            }
            total_configs++;

            if (csv)
                fprintf(csv, "%d,%d,%.6f,%.6f,%.6f,%d,%d\n",
                        n, mv, mean, std_val, best_bl, win, tie);
        }

        double elapsed = wall_time() - t0;
        free(m_list); free(all_l2);
        double avg_rl = n_configs > 0 ? sum_rl / n_configs : 0;
        double avg_best = n_with_baseline > 0 ? sum_best / n_with_baseline : 0;

        printf("  n=%d: %d wins, %d ties / %d configs | "
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

    if (csv) { fclose(csv); printf("CSV saved: %s\n", out_path); }
    free(baselines);
    return 0;
}
