/*
 * sa_main.c — Simulated Annealing REFINE for G(n,m) algebraic connectivity.
 *
 * FV gradient proposes swaps, Metropolis criterion accepts/rejects.
 * No learning — pure SA with FV-biased proposals.
 *
 * Usage:
 *   ./crl_sa --n 16,24 --seeds 5 --baselines baselines.csv
 *   ./crl_sa --n 16 --T-start 1.0 --T-end 0.001 --iter-mult 2
 */

#include "crl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
extern void dsyev_(char *jobz, char *uplo, int *n, double *a, int *lda,
                   double *w, double *work, int *lwork, int *info);
#endif

/* ========================================================================== */
/* Timing                                                                      */
/* ========================================================================== */

static double wall_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* ========================================================================== */
/* Eigensolve returning λ₂ and v₂ (Fiedler vector)                            */
/* ========================================================================== */

/*
 * Compute eigenvalues + eigenvectors of Laplacian.
 * Returns λ₂. v2_out[i] = Fiedler vector component for node i.
 * L_work: pre-allocated (n*n) double buffer (destroyed by dsyev).
 */
static double eigensolve_fiedler(const uint8_t *adj, int n,
                                  double *v2_out, double *L_work) {
    /* Build Laplacian */
    for (int i = 0; i < n; i++) {
        double deg = 0.0;
        for (int j = 0; j < n; j++) {
            if (adj[i * n + j]) {
                deg += 1.0;
                L_work[i * n + j] = -1.0;
            } else {
                L_work[i * n + j] = 0.0;
            }
        }
        L_work[i * n + i] = deg;
    }

    double *eigvals = (double *)malloc((size_t)n * sizeof(double));
    char jobz = 'V', uplo = 'U';
    int info, lwork = -1;
    double work_query;
    int n_copy = n;

    dsyev_(&jobz, &uplo, &n_copy, L_work, &n_copy, eigvals,
           &work_query, &lwork, &info);
    lwork = (int)work_query + 256;
    double *work = (double *)malloc((size_t)lwork * sizeof(double));

    dsyev_(&jobz, &uplo, &n_copy, L_work, &n_copy, eigvals,
           work, &lwork, &info);

    double lam2 = (info == 0 && n > 1) ? eigvals[1] : 0.0;

    /* dsyev stores eigenvectors column-major in L_work.
     * Column 1 (index 1) is the Fiedler vector. */
    if (v2_out) {
        for (int i = 0; i < n; i++)
            v2_out[i] = L_work[1 * n + i];  /* column 1 */
    }

    free(eigvals);
    free(work);
    return lam2;
}

/* ========================================================================== */
/* Softmax sampling with temperature                                           */
/* ========================================================================== */

/*
 * Sample from scores[0..count-1] via softmax with temperature tau.
 * Returns index into scores array.
 */
static int softmax_sample_scores(const double *scores, int count,
                                  double tau, RNG *rng) {
    if (count <= 0) return -1;
    if (count == 1) return 0;

    /* Find max for numerical stability */
    double max_s = scores[0] / tau;
    for (int i = 1; i < count; i++) {
        double s = scores[i] / tau;
        if (s > max_s) max_s = s;
    }

    /* Compute exp and sum */
    double sum = 0.0;
    double *probs = (double *)malloc((size_t)count * sizeof(double));
    for (int i = 0; i < count; i++) {
        probs[i] = exp(scores[i] / tau - max_s);
        sum += probs[i];
    }

    /* Sample */
    double r = rng_double(rng) * sum;
    double cumsum = 0.0;
    int chosen = count - 1;
    for (int i = 0; i < count; i++) {
        cumsum += probs[i];
        if (r <= cumsum) {
            chosen = i;
            break;
        }
    }
    free(probs);
    return chosen;
}

/* ========================================================================== */
/* SA REFINE core                                                              */
/* ========================================================================== */

typedef struct {
    double best_lam2;
    double final_lam2;
    double accept_rate;
    int    worsen_accepts;
    int    total_iters;
} SAResult;

static SAResult sa_refine(int n, int m, int total_iters,
                           double T_start, double T_end,
                           double proposal_tau, uint64_t seed) {
    SAResult result = {0};
    result.total_iters = total_iters;

    RNG rng;
    rng_init(&rng, seed);

    /* Allocate */
    uint8_t *adj  = (uint8_t *)calloc((size_t)n * n, 1);
    uint8_t *best_adj = (uint8_t *)malloc((size_t)n * n);
    int *degrees  = (int *)calloc((size_t)n, sizeof(int));
    double *L_work = (double *)malloc((size_t)n * n * sizeof(double));
    double *v2     = (double *)malloc((size_t)n * sizeof(double));
    double *fv_scores = (double *)malloc((size_t)n * n * sizeof(double));
    uint8_t *bridge_mask = (uint8_t *)malloc((size_t)n * n);

    /* Buffers for candidate indices */
    int max_pairs = n * (n - 1) / 2;
    int *add_idx_i = (int *)malloc((size_t)max_pairs * sizeof(int));
    int *add_idx_j = (int *)malloc((size_t)max_pairs * sizeof(int));
    double *add_scores = (double *)malloc((size_t)max_pairs * sizeof(double));
    int *rem_idx_i = (int *)malloc((size_t)max_pairs * sizeof(int));
    int *rem_idx_j = (int *)malloc((size_t)max_pairs * sizeof(int));
    double *rem_scores = (double *)malloc((size_t)max_pairs * sizeof(double));

    /* Build initial graph: ring + random */
    build_ring_random(adj, degrees, n, m, &rng);
    memcpy(best_adj, adj, (size_t)n * n);

    /* Initial eigensolve */
    double lam2 = eigensolve_fiedler(adj, n, v2, L_work);
    double best_lam2 = lam2;

    /* FV scores: (v2[i] - v2[j])^2 */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double gap = v2[i] - v2[j];
            fv_scores[i * n + j] = gap * gap;
        }

    int accepts = 0;
    int worsen_accepts = 0;
    int total_proposals = 0;

    for (int it = 0; it < total_iters; it++) {
        /* Geometric cooling */
        double progress = (double)it / fmax(total_iters - 1, 1);
        double T = T_start * pow(T_end / fmax(T_start, 1e-15), progress);

        /* --- Collect ADD candidates (non-edges, upper triangle) --- */
        int num_add = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (!adj[i * n + j]) {
                    add_idx_i[num_add] = i;
                    add_idx_j[num_add] = j;
                    add_scores[num_add] = fv_scores[i * n + j];
                    num_add++;
                }
            }
        }
        if (num_add == 0) continue;

        /* --- Collect REMOVE candidates (edges, upper tri, no bridges) --- */
        find_bridges(adj, n, bridge_mask);
        int num_rem = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (adj[i * n + j] && !bridge_mask[i * n + j]) {
                    rem_idx_i[num_rem] = i;
                    rem_idx_j[num_rem] = j;
                    rem_scores[num_rem] = -fv_scores[i * n + j]; /* negate: prefer low FV gap */
                    num_rem++;
                }
            }
        }
        if (num_rem == 0) continue;

        /* --- PROPOSE: sample add + remove --- */
        int add_pick = softmax_sample_scores(add_scores, num_add, proposal_tau, &rng);
        int rem_pick = softmax_sample_scores(rem_scores, num_rem, proposal_tau, &rng);
        int ai = add_idx_i[add_pick], aj = add_idx_j[add_pick];
        int ri = rem_idx_i[rem_pick], rj = rem_idx_j[rem_pick];

        total_proposals++;

        /* --- APPLY SWAP --- */
        adj[ai * n + aj] = adj[aj * n + ai] = 1;
        adj[ri * n + rj] = adj[rj * n + ri] = 0;

        /* --- EVALUATE --- */
        double lam2_new = eigensolve_fiedler(adj, n, NULL, L_work);
        double delta = lam2_new - lam2;

        /* --- METROPOLIS ACCEPTANCE --- */
        int accept;
        if (delta > 0) {
            accept = 1;
        } else if (T > 1e-15) {
            accept = rng_double(&rng) < exp(delta / T);
        } else {
            accept = 0;
        }

        if (accept) {
            accepts++;
            if (delta < -1e-8) worsen_accepts++;

            lam2 = lam2_new;

            /* Re-compute Fiedler vector for new FV scores */
            eigensolve_fiedler(adj, n, v2, L_work);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                    double gap = v2[i] - v2[j];
                    fv_scores[i * n + j] = gap * gap;
                }

            /* Update degrees */
            degrees[ai]++; degrees[aj]++;
            degrees[ri]--; degrees[rj]--;

            if (lam2 > best_lam2) {
                best_lam2 = lam2;
                memcpy(best_adj, adj, (size_t)n * n);
            }
        } else {
            /* Undo swap */
            adj[ai * n + aj] = adj[aj * n + ai] = 0;
            adj[ri * n + rj] = adj[rj * n + ri] = 1;
        }
    }

    result.best_lam2 = best_lam2;
    result.final_lam2 = lam2;
    result.accept_rate = (double)accepts / fmax(total_proposals, 1);
    result.worsen_accepts = worsen_accepts;

    free(adj); free(best_adj); free(degrees); free(L_work); free(v2);
    free(fv_scores); free(bridge_mask);
    free(add_idx_i); free(add_idx_j); free(add_scores);
    free(rem_idx_i); free(rem_idx_j); free(rem_scores);

    return result;
}

/* ========================================================================== */
/* Baseline CSV loading (same format as eval_main.c)                           */
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
/* Main                                                                        */
/* ========================================================================== */

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s --n 16,24 [options]\n", argv[0]);
        printf("  --n N1,N2,...       Graph sizes to evaluate\n");
        printf("  --seeds S           Seeds per (n,m) config (default: 5)\n");
        printf("  --baselines FILE    CSV baselines file\n");
        printf("  --T-start F         SA start temperature (default: 1.0)\n");
        printf("  --T-end F           SA end temperature (default: 0.001)\n");
        printf("  --iter-mult F       Iterations = mult * m (default: 2.0)\n");
        printf("  --proposal-tau F    FV proposal temperature (default: 0.01)\n");
        printf("  --out FILE          Output CSV path\n");
        return 1;
    }

    int n_values[64], num_n = 0;
    int num_seeds = 5;
    double T_start = 1.0, T_end = 0.001;
    double iter_mult = 2.0;
    double proposal_tau = 0.01;
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
        else if (strcmp(argv[i], "--T-start") == 0 && i + 1 < argc)
            T_start = atof(argv[++i]);
        else if (strcmp(argv[i], "--T-end") == 0 && i + 1 < argc)
            T_end = atof(argv[++i]);
        else if (strcmp(argv[i], "--iter-mult") == 0 && i + 1 < argc)
            iter_mult = atof(argv[++i]);
        else if (strcmp(argv[i], "--proposal-tau") == 0 && i + 1 < argc)
            proposal_tau = atof(argv[++i]);
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc)
            out_path = argv[++i];
    }

    if (num_n == 0) {
        fprintf(stderr, "Error: --n required\n");
        return 1;
    }

    if (baselines_path) load_baselines(baselines_path);

    printf("SA-REFINE (C) | T=%.3f->%.4f | iter=%g*m | tau=%.3f | seeds=%d\n",
           T_start, T_end, iter_mult, proposal_tau, num_seeds);
    printf("=============================================\n\n");

    FILE *csv = NULL;
    if (out_path) {
        csv = fopen(out_path, "w");
        if (csv) fprintf(csv, "n,m,sa_best,fv_bl,best_bl,pct_fv,pct_best\n");
    }

    for (int ni = 0; ni < num_n; ni++) {
        int n = n_values[ni];
        int max_m = n * (n - 1) / 2;

        /* Collect all valid m values */
        int *m_vals = (int *)malloc((size_t)(max_m + 1) * sizeof(int));
        int num_m = 0;
        for (int m = n + 1; m <= max_m; m++) {
            BaselineEntry *bl = find_baseline(n, m);
            if (bl && bl->best > 0) {
                m_vals[num_m++] = m;
            }
        }

        if (num_m == 0) {
            printf("n=%d: no baseline entries found, skipping\n", n);
            free(m_vals);
            continue;
        }

        printf("n=%d (%d configs)\n", n, num_m);
        double t0 = wall_time();

        /* Accumulators for summary */
        double sum_sa = 0, sum_fv = 0, sum_best = 0;
        int wins_fv = 0, wins_best = 0, ties_fv = 0;
        int total_configs = 0;
        double sum_acc = 0, sum_worsen = 0;

        for (int mi = 0; mi < num_m; mi++) {
            int m = m_vals[mi];
            int total_iters = (int)(iter_mult * m);
            BaselineEntry *bl = find_baseline(n, m);

            /* Run SA across seeds, take best */
            double best_sa = -1e30;
            double sum_acc_m = 0, sum_worsen_m = 0;

            for (int s = 0; s < num_seeds; s++) {
                uint64_t seed = (uint64_t)(n * 10000 + m * 100 + s + 1);
                SAResult r = sa_refine(n, m, total_iters,
                                        T_start, T_end, proposal_tau, seed);
                if (r.best_lam2 > best_sa) best_sa = r.best_lam2;
                sum_acc_m += r.accept_rate;
                sum_worsen_m += r.worsen_accepts;
            }

            double fv_bl = bl->fv;
            double best_bl = bl->best;

            sum_sa += best_sa;
            sum_fv += fv_bl;
            sum_best += best_bl;
            sum_acc += sum_acc_m / num_seeds;
            sum_worsen += sum_worsen_m / num_seeds;

            if (best_sa > fv_bl + 1e-6) wins_fv++;
            else if (fabs(best_sa - fv_bl) < 1e-6) ties_fv++;
            if (best_sa > best_bl + 1e-6) wins_best++;
            total_configs++;

            if (csv) {
                double pct_fv = fv_bl > 0 ? best_sa / fv_bl * 100.0 : 0;
                double pct_best = best_bl > 0 ? best_sa / best_bl * 100.0 : 0;
                fprintf(csv, "%d,%d,%.6f,%.6f,%.6f,%.1f,%.1f\n",
                        n, m, best_sa, fv_bl, best_bl, pct_fv, pct_best);
            }

            /* Progress */
            if ((mi + 1) % 20 == 0 || mi + 1 == num_m) {
                double elapsed = wall_time() - t0;
                printf("  %d/%d configs (%.1fs)\n", mi + 1, num_m, elapsed);
                fflush(stdout);
            }
        }

        /* Summary */
        double avg_sa = sum_sa / total_configs;
        double avg_fv = sum_fv / total_configs;
        double avg_best = sum_best / total_configs;
        double pct_fv = avg_fv > 0 ? avg_sa / avg_fv * 100.0 : 0;
        double pct_best = avg_best > 0 ? avg_sa / avg_best * 100.0 : 0;
        double avg_acc = sum_acc / total_configs * 100.0;
        double avg_worsen = sum_worsen / total_configs;
        double elapsed = wall_time() - t0;

        printf("\n  n=%d SUMMARY (%d configs, %.1fs):\n", n, total_configs, elapsed);
        printf("    SA avg:     %.4f\n", avg_sa);
        printf("    FV avg:     %.4f\n", avg_fv);
        printf("    Best avg:   %.4f\n", avg_best);
        printf("    vs FV:      %.1f%% (%dW %dT)\n", pct_fv, wins_fv, ties_fv);
        printf("    vs Best:    %.1f%% (%dW)\n", pct_best, wins_best);
        printf("    Accept:     %.1f%%  worsen: %.1f/run\n", avg_acc, avg_worsen);
        printf("\n");

        free(m_vals);
    }

    if (csv) {
        fclose(csv);
        printf("Results written to %s\n", out_path);
    }
    free(baselines);

    return 0;
}
