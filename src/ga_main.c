/*
 * ga_main.c — Genetic Algorithm for G(n,m) algebraic connectivity.
 *
 * Population of graphs, FV-guided crossover + RL-learned mutation.
 * Uses Lanczos for Fiedler (O(N²)) + bridge-accelerated crossover.
 *
 * Modes:
 *   Eval:  ./crl_ga --n 16,24 --seeds 3 --baselines baselines.csv [--checkpoint model.bin]
 *   Train: ./crl_ga --n 8,16 --seeds 1 --train --episodes 5000 --save-dir logs/ga_rl
 */

#include "crl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
extern void dsyev_(char*, char*, int*, double*, int*, double*, double*, int*, int*);
#endif
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <omp.h>

static double wall_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* ========================================================================== */
/* EdgeScore for qsort-based sorting                                          */
/* ========================================================================== */

typedef struct { int i, j; double val; } EdgeScore;

static int edge_score_cmp_asc(const void *a, const void *b) {
    double d = ((const EdgeScore *)a)->val - ((const EdgeScore *)b)->val;
    return d < 0 ? -1 : d > 0 ? 1 : 0;
}

/* ========================================================================== */
/* Incremental Bridge Maintenance (local copy from test.c)                     */
/* ========================================================================== */

typedef struct {
    int *parent;      /* parent[v] in spanning tree, -1 for root */
    int *depth;       /* depth[v] in spanning tree */
    uint8_t *is_tree; /* is_tree[i*n+j] = 1 if (i,j) is a tree edge */
    int *cover;       /* cover[v] = # non-tree edges spanning tree edge (parent[v],v) */
    uint8_t *bridge;  /* bridge[i*n+j] = 1 if (i,j) is a bridge */
    int n;
} BridgeState;

static void bridge_alloc(int n, BridgeState *bs) {
    bs->n = n;
    bs->parent = (int *)malloc((size_t)n * sizeof(int));
    bs->depth = (int *)malloc((size_t)n * sizeof(int));
    bs->is_tree = (uint8_t *)calloc((size_t)n * n, 1);
    bs->cover = (int *)calloc((size_t)n, sizeof(int));
    bs->bridge = (uint8_t *)calloc((size_t)n * n, 1);
}

static void bridge_free(BridgeState *bs) {
    free(bs->parent); free(bs->depth); free(bs->is_tree);
    free(bs->cover); free(bs->bridge);
}

/* O(N^2) full init: BFS tree, compute covers, set bridge flags */
static void bridge_init(const uint8_t *adj, int n, BridgeState *bs) {
    memset(bs->is_tree, 0, (size_t)n * n);
    memset(bs->cover, 0, (size_t)n * sizeof(int));
    memset(bs->bridge, 0, (size_t)n * n);
    for (int i = 0; i < n; i++) { bs->parent[i] = -1; bs->depth[i] = -1; }

    int *queue = (int *)malloc((size_t)n * sizeof(int));
    int qh = 0, qt = 0;
    bs->depth[0] = 0; bs->parent[0] = -1;
    queue[qt++] = 0;
    while (qh < qt) {
        int u = queue[qh++];
        for (int v = 0; v < n; v++) {
            if (adj[u * n + v] && bs->depth[v] < 0) {
                bs->parent[v] = u;
                bs->depth[v] = bs->depth[u] + 1;
                bs->is_tree[u * n + v] = bs->is_tree[v * n + u] = 1;
                queue[qt++] = v;
            }
        }
    }
    free(queue);

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (!adj[i * n + j] || bs->is_tree[i * n + j]) continue;
            int u = i, v = j;
            while (bs->depth[u] > bs->depth[v]) { bs->cover[u]++; u = bs->parent[u]; }
            while (bs->depth[v] > bs->depth[u]) { bs->cover[v]++; v = bs->parent[v]; }
            while (u != v) { bs->cover[u]++; bs->cover[v]++; u = bs->parent[u]; v = bs->parent[v]; }
        }
    }

    for (int v = 1; v < n; v++) {
        int p = bs->parent[v];
        if (bs->cover[v] == 0)
            bs->bridge[p * n + v] = bs->bridge[v * n + p] = 1;
    }
}

/* O(N) -- called AFTER adding (u,v) to adj. New edge is always non-tree. */
static void bridge_add_edge(int n, int u, int v, BridgeState *bs) {
    (void)n;
    int a = u, b = v;
    while (bs->depth[a] > bs->depth[b]) {
        bs->cover[a]++;
        if (bs->cover[a] == 1) {
            int p = bs->parent[a];
            bs->bridge[p * n + a] = bs->bridge[a * n + p] = 0;
        }
        a = bs->parent[a];
    }
    while (bs->depth[b] > bs->depth[a]) {
        bs->cover[b]++;
        if (bs->cover[b] == 1) {
            int p = bs->parent[b];
            bs->bridge[p * n + b] = bs->bridge[b * n + p] = 0;
        }
        b = bs->parent[b];
    }
    while (a != b) {
        bs->cover[a]++;
        if (bs->cover[a] == 1) {
            int p = bs->parent[a];
            bs->bridge[p * n + a] = bs->bridge[a * n + p] = 0;
        }
        bs->cover[b]++;
        if (bs->cover[b] == 1) {
            int p = bs->parent[b];
            bs->bridge[p * n + b] = bs->bridge[b * n + p] = 0;
        }
        a = bs->parent[a]; b = bs->parent[b];
    }
}

/* O(N) amortized -- called BEFORE removing (u,v) from adj. */
static void bridge_rem_edge(const uint8_t *adj, int n, int u, int v, BridgeState *bs) {
    if (!bs->is_tree[u * n + v]) {
        int a = u, b = v;
        while (bs->depth[a] > bs->depth[b]) {
            bs->cover[a]--;
            if (bs->cover[a] == 0) {
                int p = bs->parent[a];
                bs->bridge[p * n + a] = bs->bridge[a * n + p] = 1;
            }
            a = bs->parent[a];
        }
        while (bs->depth[b] > bs->depth[a]) {
            bs->cover[b]--;
            if (bs->cover[b] == 0) {
                int p = bs->parent[b];
                bs->bridge[p * n + b] = bs->bridge[b * n + p] = 1;
            }
            b = bs->parent[b];
        }
        while (a != b) {
            bs->cover[a]--;
            if (bs->cover[a] == 0) {
                int p = bs->parent[a];
                bs->bridge[p * n + a] = bs->bridge[a * n + p] = 1;
            }
            bs->cover[b]--;
            if (bs->cover[b] == 0) {
                int p = bs->parent[b];
                bs->bridge[p * n + b] = bs->bridge[b * n + p] = 1;
            }
            a = bs->parent[a]; b = bs->parent[b];
        }
    } else {
        /* Tree edge removal: full recompute */
        uint8_t *madj = (uint8_t *)adj;
        madj[u * n + v] = madj[v * n + u] = 0;
        bridge_init(madj, n, bs);
        madj[u * n + v] = madj[v * n + u] = 1;
    }
}

/* ========================================================================== */
/* Lanczos helpers                                                             */
/* ========================================================================== */

static double lanczos_lambda2(const uint8_t *adj, int n) {
    double *v2 = (double *)malloc((size_t)n * sizeof(double));
    double lam[1];
    lanczos_ext_k(adj, n, NULL, LANCZOS_K, 1, v2, lam);
    free(v2);
    return lam[0];
}

static void lanczos_fiedler(const uint8_t *adj, int n,
                             double *v2_out, double *lam2_out) {
    double lam[1];
    lanczos_ext_k(adj, n, NULL, LANCZOS_K, 1, v2_out, lam);
    *lam2_out = lam[0];
}

/* ========================================================================== */
/* BFS connectivity check                                                      */
/* ========================================================================== */

static int is_connected(const uint8_t *adj, int n) {
    int *visited = (int *)calloc((size_t)n, sizeof(int));
    int *queue = (int *)malloc((size_t)n * sizeof(int));
    visited[0] = 1;
    queue[0] = 0;
    int head = 0, tail = 1, count = 1;
    while (head < tail) {
        int u = queue[head++];
        for (int v = 0; v < n; v++)
            if (adj[u * n + v] && !visited[v]) {
                visited[v] = 1;
                queue[tail++] = v;
                count++;
            }
    }
    free(visited);
    free(queue);
    return count == n;
}

/* ========================================================================== */
/* Count edges                                                                 */
/* ========================================================================== */

static int count_edges(const uint8_t *adj, int n) {
    int m = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (adj[i * n + j]) m++;
    return m;
}

/* ========================================================================== */
/* FV-guided crossover: union → Lanczos Fiedler → bridge-prune to m            */
/* ========================================================================== */

static void crossover(const uint8_t *pa, const uint8_t *pb,
                       int n, int m, uint8_t *child, RNG *rng) {
    (void)rng;
    int nn = n * n;

    for (int i = 0; i < nn; i++)
        child[i] = pa[i] | pb[i];

    int union_m = count_edges(child, n);
    if (union_m <= m) return;

    double *v2 = (double *)malloc((size_t)n * sizeof(double));
    double lam;
    lanczos_fiedler(child, n, v2, &lam);

    int max_e = n * (n - 1) / 2;
    EdgeScore *edges = (EdgeScore *)malloc((size_t)max_e * sizeof(EdgeScore));
    int ne = 0;

    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (child[i * n + j]) {
                double g = v2[i] - v2[j];
                edges[ne].i = i;
                edges[ne].j = j;
                edges[ne].val = g * g;
                ne++;
            }

    qsort(edges, (size_t)ne, sizeof(EdgeScore), edge_score_cmp_asc);

    int to_remove = union_m - m;
    int removed = 0;

    BridgeState bs;
    bridge_alloc(n, &bs);
    bridge_init(child, n, &bs);

    for (int k = 0; k < ne && removed < to_remove; k++) {
        int a = edges[k].i, b = edges[k].j;
        if (!child[a * n + b]) continue;
        if (bs.bridge[a * n + b]) continue;
        bridge_rem_edge(child, n, a, b, &bs);
        child[a * n + b] = child[b * n + a] = 0;
        removed++;
    }

    bridge_free(&bs);
    free(v2);
    free(edges);
}

/* Effective-resistance-guided crossover: union → eigendecomp → prune by R_eff */
static void crossover_reff(const uint8_t *pa, const uint8_t *pb,
                            int n, int m, uint8_t *child, RNG *rng) {
    (void)rng;
    int nn = n * n;

    for (int i = 0; i < nn; i++)
        child[i] = pa[i] | pb[i];

    int union_m = count_edges(child, n);
    if (union_m <= m) return;

    /* Build Laplacian */
    int *degrees = (int *)calloc((size_t)n, sizeof(int));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (child[i * n + j]) degrees[i]++;

    double *L = (double *)malloc((size_t)nn * sizeof(double));
    double *evals = (double *)malloc((size_t)n * sizeof(double));
    double *evecs = (double *)malloc((size_t)nn * sizeof(double));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            L[i * n + j] = -(double)child[i * n + j];
        L[i * n + i] = (double)degrees[i];
    }
    memcpy(evecs, L, (size_t)nn * sizeof(double));
    char jobz = 'V', uplo = 'U';
    int info, lwork = -1; double work_query;
    int n_int = n;
    dsyev_(&jobz, &uplo, &n_int, evecs, &n_int, evals, &work_query, &lwork, &info);
    lwork = (int)work_query;
    double *work = (double *)malloc((size_t)lwork * sizeof(double));
    memcpy(evecs, L, (size_t)nn * sizeof(double));
    dsyev_(&jobz, &uplo, &n_int, evecs, &n_int, evals, work, &lwork, &info);
    free(work);

    /* Build L+ (pseudoinverse) from eigendecomposition */
    double *lpinv = (double *)calloc((size_t)nn, sizeof(double));
    for (int k = 0; k < n; k++) {
        if (evals[k] < 1e-10) continue;
        double inv_lam = 1.0 / evals[k];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                lpinv[i * n + j] += evecs[k * n + i] * evecs[k * n + j] * inv_lam;
    }

    /* Collect edges with R_eff via O(1) lookup from L+ */
    int max_e = n * (n - 1) / 2;
    EdgeScore *edges = (EdgeScore *)malloc((size_t)max_e * sizeof(EdgeScore));
    int ne = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (child[i * n + j]) {
                edges[ne].i = i;
                edges[ne].j = j;
                edges[ne].val = lpinv[i * n + i] + lpinv[j * n + j] - 2.0 * lpinv[i * n + j];
                ne++;
            }

    qsort(edges, (size_t)ne, sizeof(EdgeScore), edge_score_cmp_asc);
    free(lpinv);

    int to_remove = union_m - m;
    int removed = 0;

    BridgeState bs;
    bridge_alloc(n, &bs);
    bridge_init(child, n, &bs);

    for (int k = 0; k < ne && removed < to_remove; k++) {
        int a = edges[k].i, b = edges[k].j;
        if (!child[a * n + b]) continue;
        if (bs.bridge[a * n + b]) continue;
        bridge_rem_edge(child, n, a, b, &bs);
        child[a * n + b] = child[b * n + a] = 0;
        removed++;
    }

    bridge_free(&bs);
    free(degrees); free(L); free(evals); free(evecs);
    free(edges);
}

static int use_reff_crossover = 0;
static int no_crossover = 0;

/* ========================================================================== */
/* Mutation: FV-guided or MLP-augmented                                        */
/* ========================================================================== */

/*
 * Per-swap mutation. If pw is NULL, use pure FV scoring.
 * If pw provided, use FV + MLP_SCALE * mlp(features).
 * Returns delta_lambda2 for this swap (for RL training).
 */
static int use_random_mut = 0; /* set from main */

static double mutate_one_swap(uint8_t *adj, int *degrees, int n,
                               double gen_progress,
                               const PolicyWeights *pw, RNG *rng,
                               BridgeState *bs) {
    double *v2 = NULL;
    double lam = 0;
    double lam_before = 0;

    if (!use_random_mut) {
        v2 = (double *)malloc((size_t)n * sizeof(double));
        lanczos_fiedler(adj, n, v2, &lam);
        lam_before = lam;
    }

    int max_pairs = n * (n - 1) / 2;
    int *ci = (int *)malloc((size_t)max_pairs * sizeof(int));
    int *cj = (int *)malloc((size_t)max_pairs * sizeof(int));
    double *scores = (double *)malloc((size_t)max_pairs * sizeof(double));

    /* Collect non-edges */
    int num_ne = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (!adj[i * n + j]) {
                ci[num_ne] = i;
                cj[num_ne] = j;
                num_ne++;
            }

    if (num_ne == 0) {
        free(v2); free(ci); free(cj); free(scores);
        return 0;
    }

    /* Score add candidates */
    double tau = 0.1;
    if (use_random_mut) {
        for (int i = 0; i < num_ne; i++) scores[i] = 1.0;
        tau = 1.0;
    } else if (pw) {
        float *feat = (float *)malloc((size_t)num_ne * EDGE_FEAT_DIM * sizeof(float));
        double *fv_gaps = (double *)malloc((size_t)num_ne * sizeof(double));
        build_candidate_features(v2, degrees, n, gen_progress,
                                 ci, cj, num_ne, NULL, NULL,
                                 feat, fv_gaps);
        score_candidates(&pw->add_mlp, feat, fv_gaps, num_ne, 0, scores);
        free(feat);
        free(fv_gaps);
    } else {
        for (int i = 0; i < num_ne; i++) {
            double g = v2[ci[i]] - v2[cj[i]];
            scores[i] = g * g;
        }
    }

    /* Softmax sample add */
    double max_s = -1e30;
    for (int i = 0; i < num_ne; i++) {
        double s = scores[i] / tau;
        if (s > max_s) max_s = s;
    }
    double sum = 0;
    double *probs = (double *)malloc((size_t)num_ne * sizeof(double));
    for (int i = 0; i < num_ne; i++) {
        probs[i] = exp(scores[i] / tau - max_s);
        sum += probs[i];
    }
    double r = rng_double(rng) * sum;
    double csum = 0;
    int add_idx = num_ne - 1;
    for (int i = 0; i < num_ne; i++) {
        csum += probs[i];
        if (r <= csum) { add_idx = i; break; }
    }
    free(probs);

    int ai = ci[add_idx], aj = cj[add_idx];

    /* Collect removable edges (non-bridge) using BridgeState */
    int num_e = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (adj[i * n + j] && !bs->bridge[i * n + j]) {
                ci[num_e] = i;
                cj[num_e] = j;
                num_e++;
            }

    if (num_e == 0) {
        if (v2) free(v2); free(ci); free(cj); free(scores);
        return 0;
    }

    /* Score remove candidates */
    if (use_random_mut) {
        for (int i = 0; i < num_e; i++) scores[i] = 1.0;
    } else if (pw) {
        float *feat = (float *)malloc((size_t)num_e * EDGE_FEAT_DIM * sizeof(float));
        double *fv_gaps = (double *)malloc((size_t)num_e * sizeof(double));
        build_candidate_features(v2, degrees, n, gen_progress,
                                 ci, cj, num_e, NULL, NULL,
                                 feat, fv_gaps);
        score_candidates(&pw->rem_mlp, feat, fv_gaps, num_e, 1, scores);
        free(feat);
        free(fv_gaps);
    } else {
        double max_gap = 0;
        for (int i = 0; i < num_e; i++) {
            double g = v2[ci[i]] - v2[cj[i]];
            scores[i] = g * g;
            if (scores[i] > max_gap) max_gap = scores[i];
        }
        for (int i = 0; i < num_e; i++)
            scores[i] = max_gap - scores[i];
    }

    /* Softmax sample remove */
    max_s = -1e30;
    for (int i = 0; i < num_e; i++) {
        double s = scores[i] / tau;
        if (s > max_s) max_s = s;
    }
    sum = 0;
    probs = (double *)malloc((size_t)num_e * sizeof(double));
    for (int i = 0; i < num_e; i++) {
        probs[i] = exp(scores[i] / tau - max_s);
        sum += probs[i];
    }
    r = rng_double(rng) * sum;
    csum = 0;
    int rem_idx = num_e - 1;
    for (int i = 0; i < num_e; i++) {
        csum += probs[i];
        if (r <= csum) { rem_idx = i; break; }
    }
    free(probs);

    int ri = ci[rem_idx], rj = cj[rem_idx];

    /* Apply swap: add first, then remove. Update bridge state incrementally. */
    adj[ai * n + aj] = adj[aj * n + ai] = 1;
    degrees[ai]++;
    degrees[aj]++;
    bridge_add_edge(n, ai, aj, bs);

    bridge_rem_edge(adj, n, ri, rj, bs);
    adj[ri * n + rj] = adj[rj * n + ri] = 0;
    degrees[ri]--;
    degrees[rj]--;

    /* Compute delta for RL reward */
    double lam_after = lanczos_lambda2(adj, n);

    if (v2) free(v2); free(ci); free(cj); free(scores);
    return lam_after - lam_before;
}

/*
 * Batched mutation: ONE Lanczos, then K swaps from same Fiedler.
 * Cost: O(N²) for Lanczos + K × O(N) for incremental bridge updates per swap.
 */
static void mutate(uint8_t *adj, int n, int num_swaps,
                   double gen_progress, const PolicyWeights *pw, RNG *rng) {
    if (num_swaps <= 0) return;

    int max_pairs = n * (n - 1) / 2;
    int *degrees = (int *)calloc((size_t)n, sizeof(int));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (adj[i * n + j]) degrees[i]++;

    double *v2 = (double *)malloc((size_t)n * sizeof(double));
    double lam;
    int *ci = (int *)malloc((size_t)max_pairs * sizeof(int));
    int *cj = (int *)malloc((size_t)max_pairs * sizeof(int));
    double *scores = (double *)malloc((size_t)max_pairs * sizeof(double));

    BridgeState bs;
    bridge_alloc(n, &bs);
    bridge_init(adj, n, &bs);

    /* Refresh Lanczos every refresh_k swaps to prevent stale signal */
    int refresh_k = n / 4;
    if (refresh_k < 2) refresh_k = 2;

    for (int s = 0; s < num_swaps; s++) {
        /* Refresh Lanczos periodically (bridge state is maintained incrementally) */
        if (s % refresh_k == 0) {
            if (!use_random_mut)
                lanczos_fiedler(adj, n, v2, &lam);
        }

        /* Collect non-edges */
        int num_ne = 0;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (!adj[i * n + j]) {
                    ci[num_ne] = i;
                    cj[num_ne] = j;
                    num_ne++;
                }
        if (num_ne == 0) break;

        /* Score add candidates */
        double tau = 0.1;
        if (use_random_mut) {
            for (int i = 0; i < num_ne; i++) scores[i] = 1.0;
            tau = 1.0;
        } else if (pw) {
            float *feat = (float *)malloc((size_t)num_ne * EDGE_FEAT_DIM * sizeof(float));
            double *fv_gaps = (double *)malloc((size_t)num_ne * sizeof(double));
            build_candidate_features(v2, degrees, n, gen_progress,
                                     ci, cj, num_ne, NULL, NULL, feat, fv_gaps);
            score_candidates(&pw->add_mlp, feat, fv_gaps, num_ne, 0, scores);
            free(feat); free(fv_gaps);
        } else {
            for (int i = 0; i < num_ne; i++) {
                double g = v2[ci[i]] - v2[cj[i]];
                scores[i] = g * g;
            }
        }

        /* Softmax sample add */
        double max_s = -1e30;
        for (int i = 0; i < num_ne; i++) {
            double sc = scores[i] / tau;
            if (sc > max_s) max_s = sc;
        }
        double sum = 0;
        double *probs = (double *)malloc((size_t)num_ne * sizeof(double));
        for (int i = 0; i < num_ne; i++) {
            probs[i] = exp(scores[i] / tau - max_s);
            sum += probs[i];
        }
        double r = rng_double(rng) * sum;
        double csum = 0;
        int add_idx = num_ne - 1;
        for (int i = 0; i < num_ne; i++) {
            csum += probs[i];
            if (r <= csum) { add_idx = i; break; }
        }
        free(probs);
        int ai = ci[add_idx], aj = cj[add_idx];

        /* Collect removable edges (non-bridge) using incremental BridgeState */
        int num_e = 0;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (adj[i * n + j] && !bs.bridge[i * n + j]) {
                    ci[num_e] = i;
                    cj[num_e] = j;
                    num_e++;
                }
        if (num_e == 0) break;

        /* Score remove candidates */
        if (use_random_mut) {
            for (int i = 0; i < num_e; i++) scores[i] = 1.0;
        } else if (pw) {
            float *feat = (float *)malloc((size_t)num_e * EDGE_FEAT_DIM * sizeof(float));
            double *fv_gaps = (double *)malloc((size_t)num_e * sizeof(double));
            build_candidate_features(v2, degrees, n, gen_progress,
                                     ci, cj, num_e, NULL, NULL, feat, fv_gaps);
            score_candidates(&pw->rem_mlp, feat, fv_gaps, num_e, 1, scores);
            free(feat); free(fv_gaps);
        } else {
            double max_gap = 0;
            for (int i = 0; i < num_e; i++) {
                double g = v2[ci[i]] - v2[cj[i]];
                scores[i] = g * g;
                if (scores[i] > max_gap) max_gap = scores[i];
            }
            for (int i = 0; i < num_e; i++)
                scores[i] = max_gap - scores[i];
        }

        /* Softmax sample remove */
        max_s = -1e30;
        for (int i = 0; i < num_e; i++) {
            double sc = scores[i] / tau;
            if (sc > max_s) max_s = sc;
        }
        sum = 0;
        probs = (double *)malloc((size_t)num_e * sizeof(double));
        for (int i = 0; i < num_e; i++) {
            probs[i] = exp(scores[i] / tau - max_s);
            sum += probs[i];
        }
        r = rng_double(rng) * sum;
        csum = 0;
        int rem_idx = num_e - 1;
        for (int i = 0; i < num_e; i++) {
            csum += probs[i];
            if (r <= csum) { rem_idx = i; break; }
        }
        free(probs);
        int ri = ci[rem_idx], rj = cj[rem_idx];

        /* Apply swap with incremental bridge updates */
        adj[ai * n + aj] = adj[aj * n + ai] = 1;
        degrees[ai]++; degrees[aj]++;
        bridge_add_edge(n, ai, aj, &bs);

        bridge_rem_edge(adj, n, ri, rj, &bs);
        adj[ri * n + rj] = adj[rj * n + ri] = 0;
        degrees[ri]--; degrees[rj]--;
    }

    bridge_free(&bs);
    free(degrees); free(v2); free(ci); free(cj); free(scores);
}

/* ========================================================================== */
/* GA core                                                                     */
/* ========================================================================== */

typedef struct {
    double best_lam2;
} GAResult;

static GAResult ga_optimize(int n, int m, int pop_size, int generations,
                             double mutation_rate, int num_mutations,
                             double elite_frac, const PolicyWeights *pw,
                             uint64_t seed) {
    GAResult result = {0};
    RNG rng;
    rng_init(&rng, seed);

    int nn = n * n;

    uint8_t **pop = (uint8_t **)malloc((size_t)pop_size * sizeof(uint8_t *));
    double *fitness = (double *)malloc((size_t)pop_size * sizeof(double));

    for (int p = 0; p < pop_size; p++) {
        pop[p] = (uint8_t *)calloc((size_t)nn, 1);
        int *deg = (int *)calloc((size_t)n, sizeof(int));
        RNG prng;
        rng_init(&prng, seed + (uint64_t)p * 12345);
        build_ring_random(pop[p], deg, n, m, &prng);
        free(deg);
    }

    double best_ever = -1e30;

    for (int gen = 0; gen < generations; gen++) {
        double gen_progress = (generations > 1)
            ? (double)gen / (generations - 1) : 0.0;

        /* Parallel fitness evaluation */
        #pragma omp parallel for schedule(static)
        for (int p = 0; p < pop_size; p++)
            fitness[p] = lanczos_lambda2(pop[p], n);

        for (int p = 0; p < pop_size; p++)
            if (fitness[p] > best_ever) best_ever = fitness[p];

        int n_elite = (int)(elite_frac * pop_size);
        if (n_elite < 1) n_elite = 1;

        int *sorted = (int *)malloc((size_t)pop_size * sizeof(int));
        for (int i = 0; i < pop_size; i++) sorted[i] = i;
        for (int i = 0; i < n_elite; i++) {
            int best_idx = i;
            for (int j = i + 1; j < pop_size; j++)
                if (fitness[sorted[j]] > fitness[sorted[best_idx]])
                    best_idx = j;
            int tmp = sorted[i]; sorted[i] = sorted[best_idx]; sorted[best_idx] = tmp;
        }

        uint8_t **new_pop = (uint8_t **)malloc((size_t)pop_size * sizeof(uint8_t *));
        for (int i = 0; i < n_elite; i++) {
            new_pop[i] = (uint8_t *)malloc((size_t)nn);
            memcpy(new_pop[i], pop[sorted[i]], (size_t)nn);
        }

        /* Pre-compute parent selections + mutation flags (sequential, uses shared RNG) */
        int n_children = pop_size - n_elite;
        int *parent1 = (int *)malloc((size_t)n_children * sizeof(int));
        int *parent2 = (int *)malloc((size_t)n_children * sizeof(int));
        int *do_mutate = (int *)malloc((size_t)n_children * sizeof(int));
        uint64_t *child_seeds = (uint64_t *)malloc((size_t)n_children * sizeof(uint64_t));

        for (int i = 0; i < n_children; i++) {
            int t_size = 3;
            if (t_size > pop_size) t_size = pop_size;

            parent1[i] = rng_int(&rng, pop_size);
            for (int t = 1; t < t_size; t++) {
                int c = rng_int(&rng, pop_size);
                if (fitness[c] > fitness[parent1[i]]) parent1[i] = c;
            }
            parent2[i] = rng_int(&rng, pop_size);
            for (int t = 1; t < t_size; t++) {
                int c = rng_int(&rng, pop_size);
                if (fitness[c] > fitness[parent2[i]]) parent2[i] = c;
            }
            do_mutate[i] = rng_double(&rng) < mutation_rate;
            child_seeds[i] = rng_next(&rng);
        }

        /* Parallel crossover + mutation (each child independent) */
        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < n_children; i++) {
            RNG child_rng;
            rng_init(&child_rng, child_seeds[i]);

            new_pop[n_elite + i] = (uint8_t *)calloc((size_t)nn, 1);
            if (no_crossover) {
                memcpy(new_pop[n_elite + i], pop[parent1[i]], (size_t)nn);
            } else if (use_reff_crossover)
                crossover_reff(pop[parent1[i]], pop[parent2[i]], n, m,
                               new_pop[n_elite + i], &child_rng);
            else
                crossover(pop[parent1[i]], pop[parent2[i]], n, m,
                          new_pop[n_elite + i], &child_rng);

            if (do_mutate[i])
                mutate(new_pop[n_elite + i], n, num_mutations,
                       gen_progress, pw, &child_rng);
        }

        free(parent1); free(parent2); free(do_mutate); free(child_seeds);

        for (int p = 0; p < pop_size; p++) free(pop[p]);
        free(pop);
        free(sorted);
        pop = new_pop;
    }

    for (int p = 0; p < pop_size; p++) {
        double f = exact_lambda2(pop[p], n);
        if (f > best_ever) best_ever = f;
    }

    result.best_lam2 = best_ever;

    for (int p = 0; p < pop_size; p++) free(pop[p]);
    free(pop);
    free(fitness);

    return result;
}

/* ========================================================================== */
/* Training: run GA episodes, collect per-mutation rewards, PPO update          */
/* ========================================================================== */

static void ga_train(int min_n, int max_n, int num_episodes,
                      int pop_size, int generations, int num_mutations,
                      double mutation_rate, double elite_frac,
                      double lr, int ppo_epochs, double clip_eps,
                      double ent_coef, double grad_clip, double weight_decay,
                      uint64_t seed, const char *save_dir) {
    RNG train_rng;
    rng_init(&train_rng, seed);

    PolicyWeights pw;
    RNG init_rng;
    rng_init(&init_rng, seed);
    policy_init_orthogonal(&pw, &init_rng, 0.5f);

    AdamState adam;
    adam_init(&adam);

    mkdir(save_dir, 0755);

    /* Per-mutation transaction storage */
    int max_muts_per_ep = pop_size * num_mutations * generations;
    int max_txns = max_muts_per_ep < 4096 ? max_muts_per_ep : 4096;
    PhaseTxn *txns = (PhaseTxn *)malloc((size_t)max_txns * sizeof(PhaseTxn));
    double *all_values = (double *)malloc((size_t)max_txns * sizeof(double));
    double *all_rewards = (double *)malloc((size_t)max_txns * sizeof(double));
    double *all_advantages = (double *)malloc((size_t)max_txns * sizeof(double));
    double *all_returns = (double *)malloc((size_t)max_txns * sizeof(double));
    double *all_old_lps = (double *)malloc((size_t)max_txns * sizeof(double));

    double best_avg = -1e30;
    double t_start = wall_time();

    printf("GA+RL Training | n=[%d,%d] | pop=%d gen=%d mut=%d | lr=%.1e | episodes=%d\n",
           min_n, max_n, pop_size, generations, num_mutations, lr, num_episodes);
    printf("================================================================\n\n");

    for (int ep = 0; ep < num_episodes; ep++) {
        int n = min_n + rng_int(&train_rng, max_n - min_n + 1);
        int max_m = n * (n - 1) / 2;
        int m = n + rng_int(&train_rng, max_m - n + 1);
        uint64_t ep_seed = seed + (uint64_t)ep * 99991;

        /* Run one GA episode, collecting per-mutation PhaseTxns */
        RNG rng;
        rng_init(&rng, ep_seed);
        int nn = n * n;

        uint8_t **pop = (uint8_t **)malloc((size_t)pop_size * sizeof(uint8_t *));
        double *fitness = (double *)malloc((size_t)pop_size * sizeof(double));

        for (int p = 0; p < pop_size; p++) {
            pop[p] = (uint8_t *)calloc((size_t)nn, 1);
            int *deg = (int *)calloc((size_t)n, sizeof(int));
            RNG prng;
            rng_init(&prng, ep_seed + (uint64_t)p * 12345);
            build_ring_random(pop[p], deg, n, m, &prng);
            free(deg);
        }

        int num_txns = 0;
        double ep_best = -1e30;

        for (int gen = 0; gen < generations; gen++) {
            double gen_progress = (generations > 1)
                ? (double)gen / (generations - 1) : 0.0;

            for (int p = 0; p < pop_size; p++)
                fitness[p] = lanczos_lambda2(pop[p], n);

            for (int p = 0; p < pop_size; p++)
                if (fitness[p] > ep_best) ep_best = fitness[p];

            int n_elite = (int)(elite_frac * pop_size);
            if (n_elite < 1) n_elite = 1;

            int *sorted_idx = (int *)malloc((size_t)pop_size * sizeof(int));
            for (int i = 0; i < pop_size; i++) sorted_idx[i] = i;
            for (int i = 0; i < n_elite; i++) {
                int bi = i;
                for (int j = i + 1; j < pop_size; j++)
                    if (fitness[sorted_idx[j]] > fitness[sorted_idx[bi]]) bi = j;
                int tmp = sorted_idx[i]; sorted_idx[i] = sorted_idx[bi]; sorted_idx[bi] = tmp;
            }

            uint8_t **new_pop = (uint8_t **)malloc((size_t)pop_size * sizeof(uint8_t *));
            for (int i = 0; i < n_elite; i++) {
                new_pop[i] = (uint8_t *)malloc((size_t)nn);
                memcpy(new_pop[i], pop[sorted_idx[i]], (size_t)nn);
            }

            for (int i = n_elite; i < pop_size; i++) {
                int t_size = 3 < pop_size ? 3 : pop_size;
                int p1 = rng_int(&rng, pop_size);
                for (int t = 1; t < t_size; t++) {
                    int c = rng_int(&rng, pop_size);
                    if (fitness[c] > fitness[p1]) p1 = c;
                }
                int p2 = rng_int(&rng, pop_size);
                for (int t = 1; t < t_size; t++) {
                    int c = rng_int(&rng, pop_size);
                    if (fitness[c] > fitness[p2]) p2 = c;
                }

                new_pop[i] = (uint8_t *)calloc((size_t)nn, 1);
                if (no_crossover) {
                    memcpy(new_pop[i], pop[p1], (size_t)nn);
                } else if (use_reff_crossover)
                    crossover_reff(pop[p1], pop[p2], n, m, new_pop[i], &rng);
                else
                    crossover(pop[p1], pop[p2], n, m, new_pop[i], &rng);

                /* RL-guided mutation with per-swap PhaseTxn collection */
                if (rng_double(&rng) < mutation_rate) {
                    int *degrees = (int *)calloc((size_t)n, sizeof(int));
                    for (int ii = 0; ii < n; ii++)
                        for (int jj = 0; jj < n; jj++)
                            if (new_pop[i][ii * n + jj]) degrees[ii]++;

                    for (int s = 0; s < num_mutations && num_txns < max_txns; s++) {
                        /* Lanczos → features */
                        double *v2 = (double *)malloc((size_t)n * sizeof(double));
                        double lam_val;
                        lanczos_fiedler(new_pop[i], n, v2, &lam_val);
                        double lam_before = lam_val;

                        int max_p = n * (n - 1) / 2;

                        /* === ADD === */
                        int *add_ci = (int *)malloc((size_t)max_p * sizeof(int));
                        int *add_cj = (int *)malloc((size_t)max_p * sizeof(int));
                        int num_add = 0;
                        for (int ii = 0; ii < n; ii++)
                            for (int jj = ii + 1; jj < n; jj++)
                                if (!new_pop[i][ii * n + jj]) {
                                    add_ci[num_add] = ii;
                                    add_cj[num_add] = jj;
                                    num_add++;
                                }

                        if (num_add == 0 || num_add > MAX_CAND) {
                            free(v2); free(add_ci); free(add_cj);
                            continue;
                        }

                        float *add_feat = (float *)malloc((size_t)num_add * EDGE_FEAT_DIM * sizeof(float));
                        double *add_fv = (double *)malloc((size_t)num_add * sizeof(double));
                        double *add_scores = (double *)malloc((size_t)num_add * sizeof(double));

                        build_candidate_features(v2, degrees, n, gen_progress,
                                                 add_ci, add_cj, num_add,
                                                 NULL, NULL, add_feat, add_fv);
                        score_candidates(&pw.add_mlp, add_feat, add_fv,
                                         num_add, 0, add_scores);

                        /* Sample one add */
                        int sel_add[1];
                        double lp_add[1];
                        softmax_sample_batch(add_scores, num_add, 1, 0.1,
                                             &rng, sel_add, lp_add);

                        int ai = add_ci[sel_add[0]], aj = add_cj[sel_add[0]];

                        /* Store add PhaseTxn */
                        PhaseTxn *txn = &txns[num_txns];
                        memset(txn, 0, sizeof(PhaseTxn));
                        txn->n = n;
                        txn->phase = 0;
                        txn->num_cand = num_add;
                        memcpy(txn->features, add_feat,
                               (size_t)num_add * EDGE_FEAT_DIM * sizeof(float));
                        memcpy(txn->fv_gaps, add_fv,
                               (size_t)num_add * sizeof(double));
                        memcpy(txn->cand_i, add_ci, (size_t)num_add * sizeof(int));
                        memcpy(txn->cand_j, add_cj, (size_t)num_add * sizeof(int));
                        txn->selected[0] = sel_add[0];
                        txn->log_probs[0] = lp_add[0];
                        txn->num_selected = 1;

                        float gfeat[GRAPH_FEAT_DIM];
                        build_graph_features_dp(lam_val, degrees, n,
                                                gen_progress, gfeat);
                        memcpy(txn->gfeat, gfeat, sizeof(gfeat));
                        txn->value = (double)mlp_forward_single(&pw.val_mlp, gfeat);

                        /* Apply add */
                        new_pop[i][ai * n + aj] = new_pop[i][aj * n + ai] = 1;
                        degrees[ai]++;
                        degrees[aj]++;

                        free(add_feat); free(add_fv); free(add_scores);
                        free(add_ci); free(add_cj);

                        /* === REMOVE === */
                        uint8_t *bmask = (uint8_t *)malloc((size_t)nn);
                        find_bridges(new_pop[i], n, bmask);

                        int *rem_ci = (int *)malloc((size_t)max_p * sizeof(int));
                        int *rem_cj = (int *)malloc((size_t)max_p * sizeof(int));
                        int num_rem = 0;
                        for (int ii = 0; ii < n; ii++)
                            for (int jj = ii + 1; jj < n; jj++)
                                if (new_pop[i][ii * n + jj] && !bmask[ii * n + jj] &&
                                    !(ii == ai && jj == aj) && !(ii == aj && jj == ai)) {
                                    rem_ci[num_rem] = ii;
                                    rem_cj[num_rem] = jj;
                                    num_rem++;
                                }

                        if (num_rem == 0 || num_rem > MAX_CAND) {
                            /* undo add */
                            new_pop[i][ai * n + aj] = new_pop[i][aj * n + ai] = 0;
                            degrees[ai]--;
                            degrees[aj]--;
                            free(bmask); free(rem_ci); free(rem_cj); free(v2);
                            continue;
                        }

                        float *rem_feat = (float *)malloc((size_t)num_rem * EDGE_FEAT_DIM * sizeof(float));
                        double *rem_fv = (double *)malloc((size_t)num_rem * sizeof(double));
                        double *rem_scores = (double *)malloc((size_t)num_rem * sizeof(double));

                        build_candidate_features(v2, degrees, n, gen_progress,
                                                 rem_ci, rem_cj, num_rem,
                                                 NULL, NULL, rem_feat, rem_fv);
                        score_candidates(&pw.rem_mlp, rem_feat, rem_fv,
                                         num_rem, 1, rem_scores);

                        int sel_rem[1];
                        double lp_rem[1];
                        softmax_sample_batch(rem_scores, num_rem, 1, 0.1,
                                             &rng, sel_rem, lp_rem);

                        int ri = rem_ci[sel_rem[0]], rj = rem_cj[sel_rem[0]];

                        /* Apply remove */
                        new_pop[i][ri * n + rj] = new_pop[i][rj * n + ri] = 0;
                        degrees[ri]--;
                        degrees[rj]--;

                        /* Reward: delta lambda2 for this swap */
                        double lam_after = lanczos_lambda2(new_pop[i], n);
                        txn->reward = lam_after - lam_before;

                        num_txns++;

                        free(rem_feat); free(rem_fv); free(rem_scores);
                        free(rem_ci); free(rem_cj); free(bmask); free(v2);
                    }
                    free(degrees);
                }
            }

            for (int p = 0; p < pop_size; p++) free(pop[p]);
            free(pop);
            free(sorted_idx);
            pop = new_pop;
        }

        /* Final fitness */
        for (int p = 0; p < pop_size; p++) {
            double f = exact_lambda2(pop[p], n);
            if (f > ep_best) ep_best = f;
        }

        for (int p = 0; p < pop_size; p++) free(pop[p]);
        free(pop);
        free(fitness);

        /* PPO update if we have enough txns */
        if (num_txns >= 8) {
            for (int t = 0; t < num_txns; t++) {
                all_values[t] = txns[t].value;
                all_rewards[t] = txns[t].reward;
                double olp = 0;
                for (int s2 = 0; s2 < txns[t].num_selected; s2++)
                    olp += txns[t].log_probs[s2];
                all_old_lps[t] = olp;
            }

            compute_gae(all_values, all_rewards, num_txns,
                        0.99, 0.95, all_advantages, all_returns);
            normalize_advantages(all_advantages, num_txns);

            for (int ppo_ep = 0; ppo_ep < ppo_epochs; ppo_ep++) {
                PolicyGrad pg;
                policy_grad_zero(&pg);

                for (int t = 0; t < num_txns; t++) {
                    ppo_backward_phase(&pw, &txns[t],
                                       all_advantages[t], all_returns[t],
                                       all_old_lps[t],
                                       ent_coef, clip_eps, &pg);
                }

                policy_grad_scale(&pg, 1.0f / (float)num_txns);
                policy_grad_clip(&pg, (float)grad_clip);
                adam_step(&pw, &pg, &adam, lr, 0.9, 0.999, 1e-8, weight_decay);
            }
        }

        if (ep_best > best_avg) {
            best_avg = ep_best;
            char path[512];
            snprintf(path, sizeof(path), "%s/best.bin", save_dir);
            save_checkpoint(path, &pw, (uint32_t)(ep + 1), (float)best_avg);
        }

        if ((ep + 1) % 10 == 0 || ep == num_episodes - 1) {
            double elapsed = wall_time() - t_start;
            printf("ep %5d/%d | best %.3f | txns %4d | %.1f ep/s | %.1fm\n",
                   ep + 1, num_episodes, ep_best, num_txns,
                   (ep + 1) / fmax(elapsed, 0.001), elapsed / 60.0);
        }
    }

    /* Final checkpoint */
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/final.bin", save_dir);
        save_checkpoint(path, &pw, (uint32_t)num_episodes, (float)best_avg);
    }

    printf("\nTraining complete. Best: %.4f. Saved to %s\n", best_avg, save_dir);

    free(txns);
    free(all_values); free(all_rewards);
    free(all_advantages); free(all_returns);
    free(all_old_lps);
}

/* ========================================================================== */
/* Baseline CSV loading                                                        */
/* ========================================================================== */

typedef struct {
    int n, m;
    double fv, er, sw025, sw050, sw075, best;
} BaselineEntry;

static BaselineEntry *baselines = NULL;
static int num_baselines = 0;

static void load_baselines(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Warning: cannot open %s\n", path); return; }
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
                baselines = (BaselineEntry *)realloc(baselines, (size_t)capacity * sizeof(BaselineEntry));
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
    int n_values[64], num_n = 0;
    int num_seeds = 3;
    int pop_size = 0;         /* 0 = use pop_mult * n */
    double pop_mult = 1.0;    /* population = pop_mult * n */
    int generations = 0;      /* 0 = use gen_mult * n */
    double gen_mult = 3.0;    /* generations = gen_mult * n */
    int num_mutations = 0;     /* 0 = use mut_mult * n */
    double mut_mult = 0.5;    /* mutations = mut_mult * n */
    double mutation_rate = 0.8;
    double elite_frac = 0.1;
    char *baselines_path = NULL;
    char *checkpoint_path = NULL;
    int do_train = 0;
    int num_episodes = 1000;
    char save_dir[256] = "logs/ga_rl";
    double lr = 3e-4;
    uint64_t seed = 42;
    int random_mutation = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            char *tok = strtok(argv[++i], ",");
            while (tok && num_n < 64) { n_values[num_n++] = atoi(tok); tok = strtok(NULL, ","); }
        }
        else if (strcmp(argv[i], "--seeds") == 0 && i+1 < argc) num_seeds = atoi(argv[++i]);
        else if (strcmp(argv[i], "--pop") == 0 && i+1 < argc) pop_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--pop-mult") == 0 && i+1 < argc) pop_mult = atof(argv[++i]);
        else if (strcmp(argv[i], "--gen") == 0 && i+1 < argc) generations = atoi(argv[++i]);
        else if (strcmp(argv[i], "--gen-mult") == 0 && i+1 < argc) gen_mult = atof(argv[++i]);
        else if (strcmp(argv[i], "--mutations") == 0 && i+1 < argc) num_mutations = atoi(argv[++i]);
        else if (strcmp(argv[i], "--mut-mult") == 0 && i+1 < argc) mut_mult = atof(argv[++i]);
        else if (strcmp(argv[i], "--mut-rate") == 0 && i+1 < argc) mutation_rate = atof(argv[++i]);
        else if (strcmp(argv[i], "--elite") == 0 && i+1 < argc) elite_frac = atof(argv[++i]);
        else if (strcmp(argv[i], "--reff") == 0) use_reff_crossover = 1;
        else if (strcmp(argv[i], "--no-crossover") == 0) no_crossover = 1;
        else if (strcmp(argv[i], "--baselines") == 0 && i+1 < argc) baselines_path = argv[++i];
        else if (strcmp(argv[i], "--checkpoint") == 0 && i+1 < argc) checkpoint_path = argv[++i];
        else if (strcmp(argv[i], "--train") == 0) do_train = 1;
        else if (strcmp(argv[i], "--episodes") == 0 && i+1 < argc) num_episodes = atoi(argv[++i]);
        else if (strcmp(argv[i], "--save-dir") == 0 && i+1 < argc) snprintf(save_dir, sizeof(save_dir), "%s", argv[++i]);
        else if (strcmp(argv[i], "--lr") == 0 && i+1 < argc) lr = atof(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc) seed = (uint64_t)atol(argv[++i]);
        else if (strcmp(argv[i], "--random-mutation") == 0) random_mutation = 1;
    }

    if (num_n == 0) {
        printf("Usage: %s --n 8,16,24 [options]\n", argv[0]);
        printf("  --pop N           Population size (default: 20)\n");
        printf("  --gen N           Generations (default: 50)\n");
        printf("  --mutations N     Mutations per child (default: 2)\n");
        printf("  --checkpoint F    Load MLP weights for RL mutation\n");
        printf("  --train           Train RL mutation operator\n");
        printf("  --episodes N      Training episodes (default: 1000)\n");
        printf("  --save-dir PATH   Training output dir\n");
        printf("  --lr F            Learning rate (default: 3e-4)\n");
        printf("  --seeds N         Seeds per config (default: 3)\n");
        printf("  --baselines F     Baselines CSV\n");
        return 1;
    }

    /* Training mode */
    if (do_train) {
        ga_train(n_values[0], num_n > 1 ? n_values[1] : n_values[0],
                 num_episodes, pop_size, generations, num_mutations,
                 mutation_rate, elite_frac, lr, 4, 0.2, 0.01, 1.0, 1e-4,
                 seed, save_dir);
        return 0;
    }

    /* Eval mode */
    use_random_mut = random_mutation;
    PolicyWeights pw;
    PolicyWeights *pw_ptr = NULL;
    if (checkpoint_path) {
        uint32_t ep;
        float best_imp;
        if (load_checkpoint(checkpoint_path, &pw, &ep, &best_imp) != 0) {
            fprintf(stderr, "Failed to load checkpoint: %s\n", checkpoint_path);
            return 1;
        }
        printf("Loaded RL mutation model: %s (ep %u)\n", checkpoint_path, ep);
        pw_ptr = &pw;
    }

    if (baselines_path) load_baselines(baselines_path);

    printf("GA (C) | pop=%d gen=%d mut=%d rate=%.1f elite=%.1f seeds=%d%s\n",
           pop_size, generations, num_mutations, mutation_rate, elite_frac,
           num_seeds, pw_ptr ? " [RL MUTATION]" : "");
    printf("================================================================\n\n");

    for (int ni = 0; ni < num_n; ni++) {
        int n = n_values[ni];
        int max_m = n * (n - 1) / 2;

        int *m_vals = (int *)malloc((size_t)(max_m + 1) * sizeof(int));
        int num_m = 0;
        for (int m = n + 1; m <= max_m; m++) {
            BaselineEntry *bl = find_baseline(n, m);
            if (bl && bl->best > 0) m_vals[num_m++] = m;
        }

        if (num_m == 0) { free(m_vals); continue; }

        /* Compute pop, generations, mutations: scale with n if not explicitly set */
        int pop_for_n = pop_size > 0 ? pop_size : (int)(pop_mult * n);
        if (pop_for_n < 4) pop_for_n = 4;
        int gen_for_n = generations > 0 ? generations : (int)(gen_mult * n);
        if (gen_for_n < 1) gen_for_n = 1;
        int mut_for_n = num_mutations > 0 ? num_mutations : (int)(mut_mult * n);
        if (mut_for_n < 1) mut_for_n = 1;

        printf("n=%d (%d configs, pop=%d, gen=%d, mut=%d, threads=%d)\n",
               n, num_m, pop_for_n, gen_for_n, mut_for_n, omp_get_max_threads());
        double t0 = wall_time();

        /* Parallel over all (m, seed) jobs */
        int total_jobs = num_m * num_seeds;
        double *job_l2 = (double *)malloc((size_t)total_jobs * sizeof(double));

        int progress_done = 0;
        #pragma omp parallel for schedule(dynamic, 1)
        for (int job = 0; job < total_jobs; job++) {
            int mi = job / num_seeds;
            int si = job % num_seeds;
            int m = m_vals[mi];
            uint64_t s_seed = (uint64_t)(n * 10000 + m * 100 + si + 1);
            job_l2[job] = ga_optimize(n, m, pop_for_n, gen_for_n,
                                       mutation_rate, mut_for_n,
                                       elite_frac, pw_ptr, s_seed).best_lam2;

            int done;
            #pragma omp atomic capture
            done = ++progress_done;
            if (done % (total_jobs / 10 + 1) == 0 || done == total_jobs) {
                double elapsed = wall_time() - t0;
                printf("  %d/%d jobs (%.0f%%) %.1f job/s\r",
                       done, total_jobs, 100.0 * done / total_jobs,
                       done / fmax(elapsed, 0.001));
                fflush(stdout);
            }
        }
        printf("\n");

        /* Aggregate: best-of-seeds per config */
        double sum_ga = 0, sum_fv = 0, sum_best = 0;
        int wins_fv = 0, wins_best = 0, total = 0;

        for (int mi = 0; mi < num_m; mi++) {
            int m = m_vals[mi];

            double best_ga = -1e30;
            for (int s = 0; s < num_seeds; s++) {
                double l2 = job_l2[mi * num_seeds + s];
                if (l2 > best_ga) best_ga = l2;
            }

            BaselineEntry *bl = find_baseline(n, m);
            if (bl && bl->best > 0) {
                sum_ga += best_ga;
                sum_fv += bl->fv;
                sum_best += bl->best;
                if (best_ga > bl->fv + 1e-6) wins_fv++;
                if (best_ga > bl->best + 1e-6) wins_best++;
                total++;
            }
        }

        free(job_l2);

        double elapsed = wall_time() - t0;
        double avg_ga = total > 0 ? sum_ga / total : 0;
        double avg_fv = total > 0 ? sum_fv / total : 0;
        double avg_best = total > 0 ? sum_best / total : 0;

        printf("\n  n=%d SUMMARY (%d configs, %.1fs):\n", n, total, elapsed);
        printf("    GA avg:     %.4f\n", avg_ga);
        printf("    FV avg:     %.4f\n", avg_fv);
        printf("    Best avg:   %.4f\n", avg_best);
        printf("    vs FV:      %.1f%% (%dW)\n",
               avg_fv > 0 ? avg_ga/avg_fv*100 : 0, wins_fv);
        printf("    vs Best:    %.1f%% (%dW)\n",
               avg_best > 0 ? avg_ga/avg_best*100 : 0, wins_best);
        printf("\n");

        free(m_vals);
    }

    free(baselines);
    return 0;
}
