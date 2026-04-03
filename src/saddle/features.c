/*
 * features.c -- Graph state management for saddle-aware RL.
 *
 * Maintains adjacency, degrees, triangle counts, sum-of-neighbor-degrees,
 * common neighbors, bridge detection, Laplacian pseudoinverse (L+), and
 * warm Lanczos eigenvector state. All structures support O(N) or O(N^2)
 * incremental updates per edge add/remove.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "saddle.h"

extern void dsyev_(const char *, const char *, const int *, double *,
                   const int *, double *, double *, const int *, int *);

/* ========================================================================== */
/* Triangle counts + sum-of-neighbor-degrees                                   */
/* ========================================================================== */

static void init_tri_snd(const uint8_t *adj, const int *deg, int n,
                          int *tri, int *snd) {
    memset(tri, 0, (size_t)n * sizeof(int));
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            if (!adj[i * n + j]) continue;
            for (int k = j + 1; k < n; k++)
                if (adj[i * n + k] && adj[j * n + k])
                    { tri[i]++; tri[j]++; tri[k]++; }
        }
    for (int i = 0; i < n; i++) {
        snd[i] = 0;
        for (int j = 0; j < n; j++)
            if (adj[i * n + j]) snd[i] += deg[j];
    }
}

/* Call AFTER adding (u,v) to adj and incrementing deg. */
static void update_tri_snd_add(const uint8_t *adj, const int *deg, int n,
                                int u, int v, int *tri, int *snd) {
    for (int k = 0; k < n; k++)
        if (k != u && k != v && adj[u * n + k] && adj[v * n + k])
            { tri[u]++; tri[v]++; tri[k]++; }
    snd[u] += deg[v]; snd[v] += deg[u];
    for (int k = 0; k < n; k++) {
        if (k == u || k == v) continue;
        if (adj[u * n + k]) snd[k]++;
        if (adj[v * n + k]) snd[k]++;
    }
}

/* Call BEFORE removing (u,v) from adj and decrementing deg. */
static void update_tri_snd_rem(const uint8_t *adj, const int *deg, int n,
                                int u, int v, int *tri, int *snd) {
    for (int k = 0; k < n; k++)
        if (k != u && k != v && adj[u * n + k] && adj[v * n + k])
            { tri[u]--; tri[v]--; tri[k]--; }
    snd[u] -= deg[v]; snd[v] -= deg[u];
    for (int k = 0; k < n; k++) {
        if (k == u || k == v) continue;
        if (adj[u * n + k]) snd[k]--;
        if (adj[v * n + k]) snd[k]--;
    }
}

/* ========================================================================== */
/* Common neighbors                                                            */
/* ========================================================================== */

static void init_cn(const uint8_t *adj, int n, int *cn) {
    memset(cn, 0, (size_t)n * n * sizeof(int));
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            int c = 0;
            for (int k = 0; k < n; k++)
                if (adj[i * n + k] && adj[j * n + k]) c++;
            cn[i * n + j] = cn[j * n + i] = c;
        }
}

/* Call AFTER adding (u,v) to adj. */
static void update_cn_add(const uint8_t *adj, int n, int u, int v, int *cn) {
    for (int k = 0; k < n; k++) {
        if (k == u || k == v) continue;
        if (adj[v * n + k]) { cn[u * n + k]++; cn[k * n + u]++; }
        if (adj[u * n + k]) { cn[v * n + k]++; cn[k * n + v]++; }
    }
}

/* Call BEFORE removing (u,v) from adj. */
static void update_cn_rem(const uint8_t *adj, int n, int u, int v, int *cn) {
    for (int k = 0; k < n; k++) {
        if (k == u || k == v) continue;
        if (adj[v * n + k]) { cn[u * n + k]--; cn[k * n + u]--; }
        if (adj[u * n + k]) { cn[v * n + k]--; cn[k * n + v]--; }
    }
}

/* ========================================================================== */
/* Bridge detection (incremental via spanning tree covers)                      */
/* ========================================================================== */

static void sd_bridge_alloc(int n, SdBridgeState *bs) {
    bs->n = n;
    bs->parent = (int *)malloc((size_t)n * sizeof(int));
    bs->depth = (int *)malloc((size_t)n * sizeof(int));
    bs->is_tree = (uint8_t *)calloc((size_t)n * n, 1);
    bs->cover = (int *)calloc((size_t)n, sizeof(int));
    bs->bridge = (uint8_t *)calloc((size_t)n * n, 1);
}

static void sd_bridge_free(SdBridgeState *bs) {
    free(bs->parent); free(bs->depth); free(bs->is_tree);
    free(bs->cover); free(bs->bridge);
}

/* O(N^2): BFS spanning tree, compute covers, set bridge flags. */
static void sd_bridge_init(const uint8_t *adj, int n, SdBridgeState *bs) {
    memset(bs->is_tree, 0, (size_t)n * n);
    memset(bs->cover, 0, (size_t)n * sizeof(int));
    memset(bs->bridge, 0, (size_t)n * n);
    for (int i = 0; i < n; i++) { bs->parent[i] = -1; bs->depth[i] = -1; }

    /* BFS from node 0 */
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

    /* Non-tree edges: walk both endpoints to LCA, incrementing covers */
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            if (!adj[i * n + j] || bs->is_tree[i * n + j]) continue;
            int u = i, v = j;
            while (bs->depth[u] > bs->depth[v]) { bs->cover[u]++; u = bs->parent[u]; }
            while (bs->depth[v] > bs->depth[u]) { bs->cover[v]++; v = bs->parent[v]; }
            while (u != v) {
                bs->cover[u]++; bs->cover[v]++;
                u = bs->parent[u]; v = bs->parent[v];
            }
        }

    /* Tree edge (parent[v], v) is a bridge iff cover[v] == 0 */
    for (int v = 1; v < n; v++) {
        int p = bs->parent[v];
        if (bs->cover[v] == 0)
            bs->bridge[p * n + v] = bs->bridge[v * n + p] = 1;
    }
}

/* O(N): new non-tree edge covers path from u,v to LCA. */
static void sd_bridge_add_edge(int n, int u, int v, SdBridgeState *bs) {
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

/* O(N) amortized. Called BEFORE removing (u,v) from adj. */
static void sd_bridge_rem_edge(const uint8_t *adj, int n, int u, int v,
                                SdBridgeState *bs) {
    if (!bs->is_tree[u * n + v]) {
        /* Non-tree edge: walk to LCA, decrement covers */
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
        /* Tree edge removal: full recompute. Temporarily remove from adj. */
        uint8_t *madj = (uint8_t *)adj;
        madj[u * n + v] = madj[v * n + u] = 0;
        sd_bridge_init(madj, n, bs);
        madj[u * n + v] = madj[v * n + u] = 1;
    }
}

/* ========================================================================== */
/* Laplacian pseudoinverse (L+) via eigendecomposition                         */
/* ========================================================================== */

static void compute_lpinv_from_adj(const uint8_t *adj, int n, double *lpinv) {
    int N = n;
    double *L = (double *)malloc((size_t)N * N * sizeof(double));

    /* Build Laplacian in column-major for dsyev_ */
    for (int i = 0; i < n; i++) {
        double d = 0;
        for (int j = 0; j < n; j++) d += adj[i * n + j];
        for (int j = 0; j < n; j++)
            L[j * N + i] = (i == j) ? d : -(double)adj[i * n + j];
    }

    double *evals = (double *)malloc((size_t)n * sizeof(double));
    int lwork = 3 * N + 1;
    double *work = (double *)malloc((size_t)lwork * sizeof(double));
    int info;
    dsyev_("V", "U", &N, L, &N, evals, work, &lwork, &info);

    /* L+ = sum_{k: lam_k > 0} (1/lam_k) * v_k * v_k^T */
    memset(lpinv, 0, (size_t)n * n * sizeof(double));
    for (int k = 0; k < n; k++) {
        if (evals[k] < 1e-10) continue;
        double inv_lam = 1.0 / evals[k];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                lpinv[i * n + j] += L[i + k * N] * L[j + k * N] * inv_lam;
    }

    free(L); free(evals); free(work);
}

/* Sherman-Morrison rank-1 update for edge addition.
 * Uses current L+ (which reflects graph BEFORE the add).
 * L+_new = L+_old - w*w^T / sigma, where w = L+[:,u] - L+[:,v],
 * sigma = 1 + R_eff(u,v). */
static void lpinv_add(double *lpinv, int n, int u, int v) {
    double *w = (double *)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++)
        w[i] = lpinv[i * n + u] - lpinv[i * n + v];
    double sigma = 1.0 + lpinv[u * n + u] + lpinv[v * n + v] - 2.0 * lpinv[u * n + v];
    double inv_s = 1.0 / sigma;
    for (int i = 0; i < n; i++)
        for (int j = i; j < n; j++) {
            double d = w[i] * w[j] * inv_s;
            lpinv[i * n + j] -= d;
            if (i != j) lpinv[j * n + i] -= d;
        }
    free(w);
}

/* Sherman-Morrison rank-1 update for edge removal.
 * Uses current L+ (which reflects graph WITH the edge present).
 * L+_new = L+_old + w*w^T / sigma, where w = L+[:,u] - L+[:,v],
 * sigma = 1 - R_eff(u,v). */
static void lpinv_rem(double *lpinv, int n, int u, int v) {
    double *w = (double *)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++)
        w[i] = lpinv[i * n + u] - lpinv[i * n + v];
    double reff = lpinv[u * n + u] + lpinv[v * n + v] - 2.0 * lpinv[u * n + v];
    double sigma = 1.0 - reff;
    double inv_s = 1.0 / sigma;
    for (int i = 0; i < n; i++)
        for (int j = i; j < n; j++) {
            double d = w[i] * w[j] * inv_s;
            lpinv[i * n + j] += d;
            if (i != j) lpinv[j * n + i] += d;
        }
    free(w);
}

/* ========================================================================== */
/* Warm Lanczos: extract lambda2, lambda3, and Fiedler vector                  */
/* ========================================================================== */

static void warm_lanczos(GraphState *gs) {
    int n = gs->n;
    /* Pack V_out as n*2 to get both v2 and v3 eigenvalues */
    double *V_buf = (double *)malloc((size_t)n * 2 * sizeof(double));
    double lams[2];

    lanczos_ext_k(gs->adj, n, gs->v2_warm, SD_LANCZOS_K, 2, V_buf, lams);

    /* Extract Fiedler vector (eigenvector 0 = lambda2) */
    for (int i = 0; i < n; i++) {
        gs->v2[i] = V_buf[i * 2 + 0];
        gs->v2_warm[i] = V_buf[i * 2 + 0];
    }
    gs->v2_lam = lams[0];
    gs->v3_lam = lams[1];

    free(V_buf);
}

/* ========================================================================== */
/* Public API                                                                  */
/* ========================================================================== */

GraphState *graph_state_alloc(int n) {
    GraphState *gs = (GraphState *)calloc(1, sizeof(GraphState));
    gs->n = n;
    size_t nn = (size_t)n * n;

    gs->adj    = (uint8_t *)calloc(nn, 1);
    gs->deg    = (int *)calloc((size_t)n, sizeof(int));
    gs->tri    = (int *)calloc((size_t)n, sizeof(int));
    gs->snd    = (int *)calloc((size_t)n, sizeof(int));
    gs->cn     = (int *)calloc(nn, sizeof(int));
    gs->lpinv  = (double *)calloc(nn, sizeof(double));
    gs->v2     = (double *)calloc((size_t)n, sizeof(double));
    gs->v2_warm = (double *)calloc((size_t)n, sizeof(double));

    sd_bridge_alloc(n, &gs->bs);
    return gs;
}

void graph_state_free(GraphState *gs) {
    if (!gs) return;
    free(gs->adj); free(gs->deg); free(gs->tri); free(gs->snd);
    free(gs->cn); free(gs->lpinv); free(gs->v2); free(gs->v2_warm);
    sd_bridge_free(&gs->bs);
    free(gs);
}

void graph_state_init(GraphState *gs, const uint8_t *adj, int n, int m) {
    gs->n = n;
    gs->m = m;
    size_t nn = (size_t)n * n;

    memcpy(gs->adj, adj, nn);

    /* Compute degrees from adj */
    for (int i = 0; i < n; i++) {
        int d = 0;
        for (int j = 0; j < n; j++) d += adj[i * n + j];
        gs->deg[i] = d;
    }

    init_tri_snd(adj, gs->deg, n, gs->tri, gs->snd);
    init_cn(adj, n, gs->cn);
    sd_bridge_init(adj, n, &gs->bs);
    compute_lpinv_from_adj(adj, n, gs->lpinv);

    /* Cold Lanczos start (no warm vector) */
    memset(gs->v2_warm, 0, (size_t)n * sizeof(double));
    double *V_buf = (double *)malloc((size_t)n * 2 * sizeof(double));
    double lams[2];
    lanczos_ext_k(adj, n, NULL, SD_LANCZOS_K, 2, V_buf, lams);
    for (int i = 0; i < n; i++) {
        gs->v2[i] = V_buf[i * 2 + 0];
        gs->v2_warm[i] = V_buf[i * 2 + 0];
    }
    gs->v2_lam = lams[0];
    gs->v3_lam = lams[1];
    free(V_buf);
}

GraphState *graph_state_clone(const GraphState *gs) {
    int n = gs->n;
    GraphState *c = graph_state_alloc(n);
    size_t nn = (size_t)n * n;

    memcpy(c->adj, gs->adj, nn);
    memcpy(c->deg, gs->deg, (size_t)n * sizeof(int));
    memcpy(c->tri, gs->tri, (size_t)n * sizeof(int));
    memcpy(c->snd, gs->snd, (size_t)n * sizeof(int));
    memcpy(c->cn, gs->cn, nn * sizeof(int));
    memcpy(c->lpinv, gs->lpinv, nn * sizeof(double));
    memcpy(c->v2, gs->v2, (size_t)n * sizeof(double));
    memcpy(c->v2_warm, gs->v2_warm, (size_t)n * sizeof(double));
    c->v2_lam = gs->v2_lam;
    c->v3_lam = gs->v3_lam;
    c->m = gs->m;

    /* Deep copy bridge state */
    memcpy(c->bs.parent, gs->bs.parent, (size_t)n * sizeof(int));
    memcpy(c->bs.depth, gs->bs.depth, (size_t)n * sizeof(int));
    memcpy(c->bs.is_tree, gs->bs.is_tree, nn);
    memcpy(c->bs.cover, gs->bs.cover, (size_t)n * sizeof(int));
    memcpy(c->bs.bridge, gs->bs.bridge, nn);

    return c;
}

void graph_state_restore(GraphState *dst, const GraphState *src) {
    int n = src->n;
    size_t nn = (size_t)n * n;

    dst->n = n;
    dst->m = src->m;

    memcpy(dst->adj, src->adj, nn);
    memcpy(dst->deg, src->deg, (size_t)n * sizeof(int));
    memcpy(dst->tri, src->tri, (size_t)n * sizeof(int));
    memcpy(dst->snd, src->snd, (size_t)n * sizeof(int));
    memcpy(dst->cn, src->cn, nn * sizeof(int));
    memcpy(dst->lpinv, src->lpinv, nn * sizeof(double));
    memcpy(dst->v2, src->v2, (size_t)n * sizeof(double));
    memcpy(dst->v2_warm, src->v2_warm, (size_t)n * sizeof(double));
    dst->v2_lam = src->v2_lam;
    dst->v3_lam = src->v3_lam;

    memcpy(dst->bs.parent, src->bs.parent, (size_t)n * sizeof(int));
    memcpy(dst->bs.depth, src->bs.depth, (size_t)n * sizeof(int));
    memcpy(dst->bs.is_tree, src->bs.is_tree, nn);
    memcpy(dst->bs.cover, src->bs.cover, (size_t)n * sizeof(int));
    memcpy(dst->bs.bridge, src->bs.bridge, nn);
}

/*
 * Incremental edge addition. Update order:
 *   1. adj + deg  (foundation for all other updates)
 *   2. tri + snd  (need updated adj/deg)
 *   3. cn         (need updated adj)
 *   4. bridges    (need updated adj)
 *   5. L+         (Sherman-Morrison uses pre-add L+ still in lpinv)
 *   6. Lanczos    (needs final graph state)
 */
void graph_state_add_edge(GraphState *gs, int u, int v) {
    int n = gs->n;

    /* 1. adj + deg */
    gs->adj[u * n + v] = gs->adj[v * n + u] = 1;
    gs->deg[u]++; gs->deg[v]++;
    gs->m++;

    /* 2. tri + snd */
    update_tri_snd_add(gs->adj, gs->deg, n, u, v, gs->tri, gs->snd);

    /* 3. cn */
    update_cn_add(gs->adj, n, u, v, gs->cn);

    /* 4. bridges */
    sd_bridge_add_edge(n, u, v, &gs->bs);

    /* 5. L+ (Sherman-Morrison uses old L+ values, adj already updated) */
    lpinv_add(gs->lpinv, n, u, v);

    /* 6. Lanczos */
    warm_lanczos(gs);
}

/*
 * Incremental edge removal. Update order is reversed:
 *   1. L+         (needs edge still in graph for Sherman-Morrison)
 *   2. bridges    (needs edge still in adj)
 *   3. cn         (needs edge still in adj)
 *   4. tri + snd  (need current deg before decrement)
 *   5. adj + deg  (final removal)
 *   6. Lanczos    (needs final graph state)
 */
void graph_state_rem_edge(GraphState *gs, int u, int v) {
    int n = gs->n;

    /* 1. L+ (Sherman-Morrison uses current L+ with edge present) */
    lpinv_rem(gs->lpinv, n, u, v);

    /* 2. bridges */
    sd_bridge_rem_edge(gs->adj, n, u, v, &gs->bs);

    /* 3. cn */
    update_cn_rem(gs->adj, n, u, v, gs->cn);

    /* 4. tri + snd (before deg decrement) */
    update_tri_snd_rem(gs->adj, gs->deg, n, u, v, gs->tri, gs->snd);

    /* 5. adj + deg */
    gs->adj[u * n + v] = gs->adj[v * n + u] = 0;
    gs->deg[u]--; gs->deg[v]--;
    gs->m--;

    /* 6. Lanczos */
    warm_lanczos(gs);
}

void lpinv_refresh(GraphState *gs) {
    compute_lpinv_from_adj(gs->adj, gs->n, gs->lpinv);
}
