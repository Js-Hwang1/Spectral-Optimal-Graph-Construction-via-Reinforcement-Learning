/*
 * test.c — AlphaZero-style RL for G(n,m) algebraic connectivity.
 *
 * 4 MLPs: {add_node, add_target, rem_node, rem_target}.
 * Training: 3-step exhaustive sweep -> value distribution per node -> KL divergence.
 * Agent plays, sweep evaluates from agent's states.
 * Features: degree/(n-1), step/3, tri_coeff, snd_norm.
 * Target scorer input: node_embed (H-dim) || cn_normalized (1-dim) = (H+1)-dim.
 * Epochs: C * (max_m - m), NO CAP.
 */

#include "crl.h"
#ifdef USE_GPU_SWEEP
#include "sweep_gpu.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include <sys/time.h>
#include <sys/stat.h>

#define H 64
#ifdef NO_FV
  #define NE_IN 5       /* degree/(n-1), step/3, tri_coeff, snd_norm, target_density */
  #define TS_IN (H + 1) /* node_embed (H) || cn_normalized (1) */
  #define MLP_MAGIC 0x4D4C5045
#else
  #define NE_IN 6       /* + v2_sq */
  #define TS_IN (H + 2) /* + fv_gap */
  #define MLP_MAGIC 0x4D4C5044
#endif
#define LANCZOS_WARM_K 10

static double wall_time(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static inline float silu_f(float x) { return x / (1.0f + expf(-x)); }
static inline float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float silu_d(float x) { float s = sigmoid_f(x); return s * (1.0f + x * (1.0f - s)); }

/* ========================================================================== */
/* Weights: shared embed + 4 scorer heads                                      */
/* ========================================================================== */

typedef struct {
    float ne_w0[H * NE_IN]; float ne_b0[H];    /* node embed: NE_IN -> H */
    float ne_w1[H * H]; float ne_b1[H];         /* node embed: H -> H */
    float an_w0[H * H]; float an_b0[H]; float an_w1[H]; float an_b1[1];
    float at_w0[H * TS_IN]; float at_b0[H]; float at_w1[H]; float at_b1[1]; /* TS_IN = H+1 */
    float rn_w0[H * H]; float rn_b0[H]; float rn_w1[H]; float rn_b1[1];
    float rt_w0[H * TS_IN]; float rt_b0[H]; float rt_w1[H]; float rt_b1[1]; /* TS_IN = H+1 */
} MLPWeights;

typedef struct {
    float ne_w0[H * NE_IN]; float ne_b0[H];
    float ne_w1[H * H]; float ne_b1[H];
    float an_w0[H * H]; float an_b0[H]; float an_w1[H]; float an_b1[1];
    float at_w0[H * TS_IN]; float at_b0[H]; float at_w1[H]; float at_b1[1];
    float rn_w0[H * H]; float rn_b0[H]; float rn_w1[H]; float rn_b1[1];
    float rt_w0[H * TS_IN]; float rt_b0[H]; float rt_w1[H]; float rt_b1[1];
} PolicyGrads;

/* ========================================================================== */
/* Graph state: triangle counts + SND                                          */
/* ========================================================================== */

static void init_tri_snd(const uint8_t *adj, const int *deg, int n,
                          int *tri, int *snd) {
    memset(tri, 0, (size_t)n * sizeof(int));
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            if (!adj[i * n + j]) continue;
            for (int k = j + 1; k < n; k++)
                if (adj[i * n + k] && adj[j * n + k]) { tri[i]++; tri[j]++; tri[k]++; }
        }
    for (int i = 0; i < n; i++) {
        snd[i] = 0;
        for (int j = 0; j < n; j++) if (adj[i * n + j]) snd[i] += deg[j];
    }
}

/* Call AFTER adding edge (u,v) to adj and incrementing deg[u], deg[v]. */
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

/* Call BEFORE removing edge (u,v) from adj and decrementing deg[u], deg[v]. */
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
/* Common Neighbors: incrementally maintained                                  */
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

/* Call AFTER adding edge (u,v) to adj. */
static void update_cn_add(const uint8_t *adj, int n, int u, int v, int *cn) {
    /* u gained neighbor v: pairs (u,k) gain common neighbor v if k is neighbor of v */
    for (int k = 0; k < n; k++) {
        if (k == u || k == v) continue;
        if (adj[v * n + k]) { cn[u * n + k]++; cn[k * n + u]++; }
        if (adj[u * n + k]) { cn[v * n + k]++; cn[k * n + v]++; }
    }
}

/* Call BEFORE removing edge (u,v) from adj. */
static void update_cn_rem(const uint8_t *adj, int n, int u, int v, int *cn) {
    for (int k = 0; k < n; k++) {
        if (k == u || k == v) continue;
        if (adj[v * n + k]) { cn[u * n + k]--; cn[k * n + u]--; }
        if (adj[u * n + k]) { cn[v * n + k]--; cn[k * n + v]--; }
    }
}

/* ========================================================================== */
/* Incremental Bridge Maintenance                                              */
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

    /* BFS from node 0 to build spanning tree */
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

    /* For each non-tree edge, walk to LCA and increment covers */
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (!adj[i * n + j] || bs->is_tree[i * n + j]) continue;
            /* non-tree edge (i,j): walk both up to LCA */
            int u = i, v = j;
            while (bs->depth[u] > bs->depth[v]) { bs->cover[u]++; u = bs->parent[u]; }
            while (bs->depth[v] > bs->depth[u]) { bs->cover[v]++; v = bs->parent[v]; }
            while (u != v) { bs->cover[u]++; bs->cover[v]++; u = bs->parent[u]; v = bs->parent[v]; }
        }
    }

    /* Set bridge flags: tree edge (parent[v], v) is a bridge iff cover[v] == 0 */
    for (int v = 1; v < n; v++) {
        int p = bs->parent[v];
        if (bs->cover[v] == 0) {
            bs->bridge[p * n + v] = bs->bridge[v * n + p] = 1;
        }
    }
}

/* O(N) — called AFTER adding (u,v) to adj. New edge is always non-tree. */
static void bridge_add_edge(int n, int u, int v, BridgeState *bs) {
    (void)n;
    /* Walk from u and v up to their LCA, incrementing covers */
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

/* O(N) amortized — called BEFORE removing (u,v) from adj. */
static void bridge_rem_edge(const uint8_t *adj, int n, int u, int v, BridgeState *bs) {
    if (!bs->is_tree[u * n + v]) {
        /* Case A: non-tree edge. Walk to LCA, decrement covers. */
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
        /* Case B: tree edge. Rare (prob (n-1)/m). Full recompute for simplicity. */
        /* We need adj with (u,v) already removed for reinit, but caller hasn't
         * removed it yet. Temporarily remove, reinit, then restore. */
        uint8_t *madj = (uint8_t *)adj; /* cast away const for temp mutation */
        madj[u * n + v] = madj[v * n + u] = 0;
        bridge_init(madj, n, bs);
        madj[u * n + v] = madj[v * n + u] = 1;
    }
}

/* ========================================================================== */
/* Forward                                                                     */
/* ========================================================================== */

static void node_embed_all(const MLPWeights *w, const int *deg, const int *tri,
                            const int *snd, const double *v2,
                            int n, float nm1, float step_f,
                            float target_density, float *hh) {
    float nm1sq = nm1 * nm1;
    for (int i = 0; i < n; i++) {
        float d = (float)deg[i] / nm1;
        float max_tri = (float)(deg[i] * (deg[i] - 1)) / 2.0f;
        float tri_coeff = max_tri > 0.5f ? (float)tri[i] / max_tri : 0.0f;
        float snd_norm = nm1sq > 0.5f ? (float)snd[i] / nm1sq : 0.0f;
#ifndef NO_FV
        float v2_sq = (float)(n * v2[i] * v2[i]);
        float feats[NE_IN] = {d, step_f, tri_coeff, snd_norm, target_density, v2_sq};
#else
        (void)v2;
        float feats[NE_IN] = {d, step_f, tri_coeff, snd_norm, target_density};
#endif
        float h0[H];
        for (int k = 0; k < H; k++) {
            float s = w->ne_b0[k];
            for (int f = 0; f < NE_IN; f++) s += w->ne_w0[k * NE_IN + f] * feats[f];
            h0[k] = silu_f(s);
        }
        for (int k = 0; k < H; k++) {
            float s = w->ne_b1[k];
            for (int j = 0; j < H; j++) s += w->ne_w1[k * H + j] * h0[j];
            hh[i * H + k] = silu_f(s);
        }
    }
}

/* Node scorer: H -> H -> 1 */
static float scorer_fwd(const float *h, const float *W0, const float *b0,
                         const float *W1, const float *b1) {
    float hid[H];
    for (int i = 0; i < H; i++) {
        float s = b0[i]; for (int j = 0; j < H; j++) s += W0[i * H + j] * h[j];
        hid[i] = silu_f(s);
    }
    float out = b1[0]; for (int j = 0; j < H; j++) out += W1[j] * hid[j];
    return out;
}


/* Target scorer: TS_IN -> H -> 1 (node_embed || cn_normalized || fv_gap) */
static float target_scorer_fwd(const float *h, float cn_norm, float fv_gap,
                                const float *W0, const float *b0,
                                const float *W1, const float *b1) {
    float hid[H];
    for (int i = 0; i < H; i++) {
        float s = b0[i];
        for (int j = 0; j < H; j++) s += W0[i * TS_IN + j] * h[j];
        s += W0[i * TS_IN + H] * cn_norm;
#ifndef NO_FV
        s += W0[i * TS_IN + H + 1] * fv_gap;
#else
        (void)fv_gap;
#endif
        hid[i] = silu_f(s);
    }
    float out = b1[0]; for (int j = 0; j < H; j++) out += W1[j] * hid[j];
    return out;
}

typedef struct {
    float pre[H]; float hid[H]; float input[TS_IN];
} TSCache;

static float target_scorer_fwd_cached(const float *h, float cn_norm, float fv_gap,
                                       const float *W0, const float *b0,
                                       const float *W1, const float *b1,
                                       TSCache *c) {
    memcpy(c->input, h, (size_t)H * sizeof(float));
    c->input[H] = cn_norm;
#ifndef NO_FV
    c->input[H + 1] = fv_gap;
#else
    (void)fv_gap;
#endif
    for (int i = 0; i < H; i++) {
        float s = b0[i];
        for (int j = 0; j < TS_IN; j++) s += W0[i * TS_IN + j] * c->input[j];
        c->pre[i] = s; c->hid[i] = silu_f(s);
    }
    float out = b1[0]; for (int j = 0; j < H; j++) out += W1[j] * c->hid[j];
    return out;
}

static void target_scorer_bwd(const TSCache *c, const float *W0, const float *W1,
                               float d_out, float *dW0, float *db0, float *dW1, float *db1,
                               float *d_h) {
    db1[0] += d_out;
    float dh[H];
    for (int j = 0; j < H; j++) { dW1[j] += d_out * c->hid[j]; dh[j] = d_out * W1[j]; }
    for (int i = 0; i < H; i++) {
        float dp = dh[i] * silu_d(c->pre[i]); db0[i] += dp;
        for (int j = 0; j < TS_IN; j++) dW0[i * TS_IN + j] += dp * c->input[j];
        /* Accumulate gradient into d_h for the first H dims (node embed part) */
        for (int j = 0; j < H; j++) d_h[j] += dp * W0[i * TS_IN + j];
    }
}

/* ========================================================================== */
/* Checkpoint                                                                  */
/* ========================================================================== */

static void save_weights(const char *path, const MLPWeights *w) {
    FILE *f = fopen(path, "wb"); if (!f) return;
    uint32_t magic = MLP_MAGIC; fwrite(&magic, 4, 1, f);
    fwrite(w, sizeof(MLPWeights), 1, f); fclose(f);
}

static int load_weights(const char *path, MLPWeights *w) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    uint32_t magic; fread(&magic, 4, 1, f);
    if (magic != MLP_MAGIC) { fclose(f); return -2; }
    fread(w, sizeof(MLPWeights), 1, f); fclose(f); return 0;
}

static void init_head(float *w0, float *b0, float *w1, float *b1, int in, int hid, RNG *rng) {
    int np = hid * in;
    for (int i = 0; i < np; i += 2) {
        double u1 = rng_double(rng) * 0.998 + 0.001, u2 = rng_double(rng);
        double r = sqrt(-2 * log(u1));
        w0[i] = (float)(r * cos(2 * M_PI * u2));
        if (i + 1 < np) w0[i + 1] = (float)(r * sin(2 * M_PI * u2));
    }
    float sc = sqrtf(2.0f / (float)(in + hid));
    for (int i = 0; i < np; i++) w0[i] *= sc;
    memset(b0, 0, (size_t)hid * sizeof(float));
    for (int i = 0; i < hid; i += 2) {
        double u1 = rng_double(rng) * 0.998 + 0.001, u2 = rng_double(rng);
        double r = sqrt(-2 * log(u1));
        w1[i] = (float)(r * cos(2 * M_PI * u2));
        if (i + 1 < hid) w1[i + 1] = (float)(r * sin(2 * M_PI * u2));
    }
    sc = sqrtf(2.0f / (float)(hid + 1));
    for (int i = 0; i < hid; i++) w1[i] *= sc;
    b1[0] = 0;
}

static void init_weights(MLPWeights *w, uint64_t seed) {
    RNG rng; rng_init(&rng, seed);
    init_head(w->ne_w0, w->ne_b0, w->ne_w1, w->ne_b1, NE_IN, H, &rng);
    /* Fix ne layer 1: H->H */
    int np = H * H;
    for (int i = 0; i < np; i += 2) {
        double u1 = rng_double(&rng) * 0.998 + 0.001, u2 = rng_double(&rng);
        double r = sqrt(-2 * log(u1));
        w->ne_w1[i] = (float)(r * cos(2 * M_PI * u2));
        if (i + 1 < np) w->ne_w1[i + 1] = (float)(r * sin(2 * M_PI * u2));
    }
    float sc = sqrtf(2.0f / (float)(H + H));
    for (int i = 0; i < np; i++) w->ne_w1[i] *= sc;
    memset(w->ne_b1, 0, sizeof(w->ne_b1));
    init_head(w->an_w0, w->an_b0, w->an_w1, w->an_b1, H, H, &rng);
    init_head(w->at_w0, w->at_b0, w->at_w1, w->at_b1, TS_IN, H, &rng);
    init_head(w->rn_w0, w->rn_b0, w->rn_w1, w->rn_b1, H, H, &rng);
    init_head(w->rt_w0, w->rt_b0, w->rt_w1, w->rt_b1, TS_IN, H, &rng);
}

/* ========================================================================== */
/* Sweep: exhaustive 3-step search for best (source, target) sequence          */
/* ========================================================================== */

typedef struct { int src[3]; int tgt[3]; double l2; } Seq3;

static void sweep_add(uint8_t *adj, int *deg, int n, int depth, int max_depth,
                       int *csrc, int *ctgt, Seq3 *best) {
    if (depth == max_depth) {
        double l2 = exact_lambda2(adj, n);
        if (l2 > best->l2) {
            best->l2 = l2;
            memcpy(best->src, csrc, (size_t)max_depth * sizeof(int));
            memcpy(best->tgt, ctgt, (size_t)max_depth * sizeof(int));
        }
        return;
    }
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (adj[i * n + j]) continue;
            csrc[depth] = i; ctgt[depth] = j;
            adj[i * n + j] = adj[j * n + i] = 1; deg[i]++; deg[j]++;
            sweep_add(adj, deg, n, depth + 1, max_depth, csrc, ctgt, best);
            adj[i * n + j] = adj[j * n + i] = 0; deg[i]--; deg[j]--;
        }
    }
}

static void sweep_rem(uint8_t *adj, int *deg, int n, int depth, int max_depth,
                       int *csrc, int *ctgt, uint8_t *bmask, Seq3 *best) {
    if (depth == max_depth) {
        double l2 = exact_lambda2(adj, n);
        if (l2 > best->l2) {
            best->l2 = l2;
            memcpy(best->src, csrc, (size_t)max_depth * sizeof(int));
            memcpy(best->tgt, ctgt, (size_t)max_depth * sizeof(int));
        }
        return;
    }
    find_bridges(adj, n, bmask);
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (!adj[i * n + j] || bmask[i * n + j]) continue;
            csrc[depth] = i; ctgt[depth] = j;
            adj[i * n + j] = adj[j * n + i] = 0; deg[i]--; deg[j]--;
            sweep_rem(adj, deg, n, depth + 1, max_depth, csrc, ctgt, bmask, best);
            adj[i * n + j] = adj[j * n + i] = 1; deg[i]++; deg[j]++;
        }
    }
}

static void sweep_add_values(uint8_t *adj, int *deg, int n, int remaining_depth,
                              int *action_src, int *action_tgt, double *action_val,
                              int *num_actions) {
    int na = 0;
    int csrc[3], ctgt[3];
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (adj[i * n + j]) continue;
            adj[i * n + j] = adj[j * n + i] = 1; deg[i]++; deg[j]++;
            if (remaining_depth <= 1) {
                action_val[na] = exact_lambda2(adj, n);
            } else {
                Seq3 best = { .l2 = -1e30 };
                sweep_add(adj, deg, n, 0, remaining_depth - 1, csrc, ctgt, &best);
                action_val[na] = best.l2;
            }
            action_src[na] = i;
            action_tgt[na] = j;
            na++;
            adj[i * n + j] = adj[j * n + i] = 0; deg[i]--; deg[j]--;
        }
    }
    *num_actions = na;
}

static void sweep_rem_values(uint8_t *adj, int *deg, int n, int remaining_depth,
                              uint8_t *bmask,
                              int *action_src, int *action_tgt, double *action_val,
                              int *num_actions) {
    int na = 0;
    int csrc[3], ctgt[3];
    find_bridges(adj, n, bmask);
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (!adj[i * n + j] || bmask[i * n + j]) continue;
            adj[i * n + j] = adj[j * n + i] = 0; deg[i]--; deg[j]--;
            if (remaining_depth <= 1) {
                action_val[na] = exact_lambda2(adj, n);
            } else {
                Seq3 best = { .l2 = -1e30 };
                sweep_rem(adj, deg, n, 0, remaining_depth - 1, csrc, ctgt, bmask, &best);
                action_val[na] = best.l2;
            }
            action_src[na] = i;
            action_tgt[na] = j;
            na++;
            adj[i * n + j] = adj[j * n + i] = 1; deg[i]++; deg[j]++;
        }
    }
    *num_actions = na;
}

/* ========================================================================== */
/* Training: AlphaZero style                                                   */
/* ========================================================================== */

typedef struct {
    float pre0[H]; float h0[H]; float pre1[H]; float h1[H]; float in_val[NE_IN];
} NECache;

typedef struct {
    float pre[H]; float hid[H]; float input[H];
} SCCache;

/* Node scorers use SCCache (H-dim). Target scorers use TSCache (TS_IN-dim). */

static void node_embed_all_cached(const MLPWeights *w, const int *deg, const int *tri,
                                    const int *snd, const double *v2,
                                    int n, float nm1, float step_f,
                                    float target_density,
                                    float *hh, NECache *nec) {
    float nm1sq = nm1 * nm1;
    for (int i = 0; i < n; i++) {
        float d = (float)deg[i] / nm1;
        float max_tri = (float)(deg[i] * (deg[i] - 1)) / 2.0f;
        float tri_coeff = max_tri > 0.5f ? (float)tri[i] / max_tri : 0.0f;
        float snd_norm = nm1sq > 0.5f ? (float)snd[i] / nm1sq : 0.0f;
#ifndef NO_FV
        float v2_sq = (float)(n * v2[i] * v2[i]);
        float feats[NE_IN] = {d, step_f, tri_coeff, snd_norm, target_density, v2_sq};
#else
        (void)v2;
        float feats[NE_IN] = {d, step_f, tri_coeff, snd_norm, target_density};
#endif
        for (int f = 0; f < NE_IN; f++) nec[i].in_val[f] = feats[f];
        for (int k = 0; k < H; k++) {
            float s = w->ne_b0[k];
            for (int f = 0; f < NE_IN; f++) s += w->ne_w0[k * NE_IN + f] * feats[f];
            nec[i].pre0[k] = s; nec[i].h0[k] = silu_f(s);
        }
        for (int k = 0; k < H; k++) {
            float s = w->ne_b1[k];
            for (int j = 0; j < H; j++) s += w->ne_w1[k * H + j] * nec[i].h0[j];
            nec[i].pre1[k] = s; nec[i].h1[k] = silu_f(s);
            hh[i * H + k] = nec[i].h1[k];
        }
    }
}

static float scorer_fwd_cached(const float *h, const float *W0, const float *b0,
                                const float *W1, const float *b1, SCCache *c) {
    memcpy(c->input, h, (size_t)H * sizeof(float));
    for (int i = 0; i < H; i++) {
        float s = b0[i]; for (int j = 0; j < H; j++) s += W0[i * H + j] * h[j];
        c->pre[i] = s; c->hid[i] = silu_f(s);
    }
    float out = b1[0]; for (int j = 0; j < H; j++) out += W1[j] * c->hid[j];
    return out;
}


static void scorer_bwd(const SCCache *c, const float *W0, const float *W1,
                        float d_out, float *dW0, float *db0, float *dW1, float *db1,
                        float *d_h) {
    db1[0] += d_out;
    float dh[H];
    for (int j = 0; j < H; j++) { dW1[j] += d_out * c->hid[j]; dh[j] = d_out * W1[j]; }
    for (int i = 0; i < H; i++) {
        float dp = dh[i] * silu_d(c->pre[i]); db0[i] += dp;
        for (int j = 0; j < H; j++) { dW0[i * H + j] += dp * c->input[j]; d_h[j] += dp * W0[i * H + j]; }
    }
}


static void node_embed_bwd(const MLPWeights *w, const NECache *c, const float *d_h, PolicyGrads *pg) {
    float dh1[H];
    for (int k = 0; k < H; k++) dh1[k] = d_h[k] * silu_d(c->pre1[k]);
    for (int k = 0; k < H; k++) {
        pg->ne_b1[k] += dh1[k];
        for (int j = 0; j < H; j++) pg->ne_w1[k * H + j] += dh1[k] * c->h0[j];
    }
    float dh0[H]; memset(dh0, 0, sizeof(dh0));
    for (int k = 0; k < H; k++) for (int j = 0; j < H; j++) dh0[j] += dh1[k] * w->ne_w1[k * H + j];
    for (int k = 0; k < H; k++) {
        float dp = dh0[k] * silu_d(c->pre0[k]);
        pg->ne_b0[k] += dp;
        for (int f = 0; f < NE_IN; f++) pg->ne_w0[k * NE_IN + f] += dp * c->in_val[f];
    }
}

/* Train one step: given sweep values for each (src,tgt) action,
 * backprop cross-entropy through node_scorer and target_scorer.
 * Node scorer uses H-dim embed; target scorer uses (H+1)-dim [embed || cn_norm]. */
static void az_train_step(const MLPWeights *w, PolicyGrads *pg,
                           const float *W0_ns, const float *b0_ns, const float *W1_ns, const float *b1_ns,
                           float *dW0_ns, float *db0_ns, float *dW1_ns, float *db1_ns,
                           const float *W0_ts, const float *b0_ts, const float *W1_ts, const float *b1_ts,
                           float *dW0_ts, float *db0_ts, float *dW1_ts, float *db1_ts,
                           const uint8_t *adj, const int *deg, const int *tri, const int *snd,
                           const int *cn, int n, float nm1, float step_f, float tgt_dens, float *hh,
                           const double *v2,
                           int *a_src, int *a_tgt, double *a_val, int na,
                           int agent_src, int agent_tgt) {
    (void)agent_tgt; (void)adj;
    if (na == 0) return;

    NECache nec[256]; SCCache nsc[256]; TSCache tsc[256];
    node_embed_all_cached(w, deg, tri, snd, v2, n, nm1, step_f, tgt_dens, hh, nec);

    /* Score all nodes with node_scorer */
    float ns[256];
    for (int i = 0; i < n; i++)
        ns[i] = scorer_fwd_cached(hh + i * H, W0_ns, b0_ns, W1_ns, b1_ns, &nsc[i]);

    /* Compute node values: for each SOURCE node, its value = max sweep value over its targets */
    double node_val[256];
    for (int i = 0; i < n; i++) node_val[i] = -1e30;
    for (int a = 0; a < na; a++) {
        int s = a_src[a] < a_tgt[a] ? a_src[a] : a_tgt[a];
        int t = a_src[a] < a_tgt[a] ? a_tgt[a] : a_src[a];
        if (a_val[a] > node_val[s]) node_val[s] = a_val[a];
        if (a_val[a] > node_val[t]) node_val[t] = a_val[a];
    }

    /* Build valid node set + softmax target from advantage values */
    int vn[256]; int nv = 0; double vmax = -1e30;
    for (int i = 0; i < n; i++) {
        if (node_val[i] > -1e20) { vn[nv++] = i; if (node_val[i] > vmax) vmax = node_val[i]; }
    }
    if (nv == 0) return;

    double tau = 0.1;
    double z_node = 0;
    for (int v = 0; v < nv; v++) z_node += exp((node_val[vn[v]] - vmax) / tau);

    /* Node scorer cross-entropy gradient */
    float nsm = -FLT_MAX;
    for (int v = 0; v < nv; v++) if (ns[vn[v]] > nsm) nsm = ns[vn[v]];
    double z_policy = 0;
    for (int v = 0; v < nv; v++) z_policy += exp((double)(ns[vn[v]] - nsm));

    for (int v = 0; v < nv; v++) {
        int idx = vn[v];
        double p_policy = exp((double)(ns[idx] - nsm)) / z_policy;
        double p_target = exp((node_val[idx] - vmax) / tau) / z_node;
        float d_score = (float)(p_policy - p_target);
        float d_h[H]; memset(d_h, 0, sizeof(d_h));
        scorer_bwd(&nsc[idx], W0_ns, W1_ns, d_score, dW0_ns, db0_ns, dW1_ns, db1_ns, d_h);
        node_embed_bwd(w, &nec[idx], d_h, pg);
    }

    /* Target scorer: train on the agent's chosen source node's target distribution */
    int t_idx[256]; double t_val[256]; int nt = 0;
    for (int a = 0; a < na; a++) {
        if (a_src[a] == agent_src || a_tgt[a] == agent_src) {
            int target = (a_src[a] == agent_src) ? a_tgt[a] : a_src[a];
            t_idx[nt] = target; t_val[nt] = a_val[a]; nt++;
        }
    }
    if (nt == 0) return;

    /* Score targets with target_scorer (H+2 dim input: embed || cn_norm || fv_gap) */
    float ts[256];
    float inv_n = 1.0f / (float)n;
    for (int t = 0; t < nt; t++) {
        float cn_norm = (float)cn[agent_src * n + t_idx[t]] * inv_n;
#ifndef NO_FV
        float fv_gap = (float)(n * (v2[agent_src] - v2[t_idx[t]]) * (v2[agent_src] - v2[t_idx[t]]));
#else
        float fv_gap = 0.0f;
#endif
        ts[t] = target_scorer_fwd_cached(hh + t_idx[t] * H, cn_norm, fv_gap,
                                          W0_ts, b0_ts, W1_ts, b1_ts, &tsc[t]);
    }

    /* Target distribution (softmax of sweep values) */
    double tvmax = -1e30;
    for (int t = 0; t < nt; t++) if (t_val[t] > tvmax) tvmax = t_val[t];
    double z_tgt = 0;
    for (int t = 0; t < nt; t++) z_tgt += exp((t_val[t] - tvmax) / tau);

    float tsm = -FLT_MAX;
    for (int t = 0; t < nt; t++) if (ts[t] > tsm) tsm = ts[t];
    double z_tp = 0;
    for (int t = 0; t < nt; t++) z_tp += exp((double)(ts[t] - tsm));

    for (int t = 0; t < nt; t++) {
        double p_pol = exp((double)(ts[t] - tsm)) / z_tp;
        double p_tgt = exp((t_val[t] - tvmax) / tau) / z_tgt;
        float d_score = (float)(p_pol - p_tgt);
        float d_h[H]; memset(d_h, 0, sizeof(d_h));
        target_scorer_bwd(&tsc[t], W0_ts, W1_ts, d_score, dW0_ts, db0_ts, dW1_ts, db1_ts, d_h);
        node_embed_bwd(w, &nec[t_idx[t]], d_h, pg);
    }
}

static void mlp_train(int min_n, int max_n, int num_episodes, int swaps,
                       double epoch_C, double lr, uint64_t seed, const char *save_dir) {
    MLPWeights w; init_weights(&w, seed);
    int np = (int)(sizeof(MLPWeights) / sizeof(float));
    float *adam_m = (float *)calloc((size_t)np, sizeof(float));
    float *adam_v = (float *)calloc((size_t)np, sizeof(float));
    int adam_t = 0;
    RNG trng; rng_init(&trng, seed);
    mkdir(save_dir, 0755);
    double t0 = wall_time();

    int N = max_n, NN = N * N;
    float *hh = (float *)malloc((size_t)N * H * sizeof(float));
    float *scores = (float *)malloc((size_t)N * sizeof(float));
    uint8_t *adj = (uint8_t *)calloc((size_t)NN, 1);
    int *deg = (int *)calloc((size_t)N, sizeof(int));
    int *tri = (int *)calloc((size_t)N, sizeof(int));
    int *snd = (int *)calloc((size_t)N, sizeof(int));
    int *cn = (int *)calloc((size_t)NN, sizeof(int));
    uint8_t *bmask = (uint8_t *)malloc((size_t)NN);  /* still needed by sweep functions */
    BridgeState bs; bridge_alloc(N, &bs);
    int max_actions = N * (N - 1) / 2;
    int *a_src = (int *)malloc((size_t)max_actions * sizeof(int));
    int *a_tgt = (int *)malloc((size_t)max_actions * sizeof(int));
    double *a_val = (double *)malloc((size_t)max_actions * sizeof(double));

#ifdef USE_GPU_SWEEP
    gpu_init();
#endif
    printf("AZ+Degree | n=[%d,%d] | ep=%d | swaps=%d | C=%.1f | lr=%.1e\n",
           min_n, max_n, num_episodes, swaps, epoch_C, lr);
    printf("================================================================\n");

    for (int ep = 0; ep < num_episodes; ep++) {
        int n = min_n + rng_int(&trng, max_n - min_n + 1);
        int max_m = n * (n - 1) / 2, m = n + rng_int(&trng, max_m - n + 1);
        int nn = n * n;
        float nm1 = (float)(n - 1); if (nm1 < 1.0f) nm1 = 1.0f;
        float tgt_dens = (float)(m - (n - 1)) / (float)(max_m - (n - 1));
        float spe_inv = 1.0f / (float)swaps;
        float inv_n = 1.0f / (float)n;

        int num_epochs_ep = (int)(epoch_C * n);
        if (num_epochs_ep < 4) num_epochs_ep = 4;

        memset(adj, 0, (size_t)nn); memset(deg, 0, (size_t)n * sizeof(int));
        RNG graph_rng; rng_init(&graph_rng, rng_next(&trng));
        build_ring_random(adj, deg, n, m, &graph_rng);
        init_tri_snd(adj, deg, n, tri, snd);
        init_cn(adj, n, cn);
        bridge_init(adj, n, &bs);
        double *v2 = (double *)malloc((size_t)n * sizeof(double));
        double v2_lam;
        double *v2_warm = (double *)malloc((size_t)n * sizeof(double));

        PolicyGrads grad; memset(&grad, 0, sizeof(grad));
        int total_steps = 0;
        RNG fwd_rng; rng_init(&fwd_rng, rng_next(&trng));

        for (int epoch = 0; epoch < num_epochs_ep; epoch++) {
            lanczos_ext_k(adj, n, epoch > 0 ? v2_warm : NULL, LANCZOS_WARM_K, 1, v2_warm, &v2_lam);
            memcpy(v2, v2_warm, (size_t)n * sizeof(double));
            /* === ADD PHASE: swaps steps === */
            for (int s = 0; s < swaps; s++) {
                float step_f = (float)s * spe_inv;
                int remaining = swaps - s;

                int na;
#ifdef USE_GPU_SWEEP
                na = gpu_sweep_add(adj, deg, n, remaining, a_src, a_tgt, a_val);
#else
                sweep_add_values(adj, deg, n, remaining, a_src, a_tgt, a_val, &na);
#endif
                if (na == 0) break;

                node_embed_all(&w, deg, tri, snd, v2, n, nm1, step_f, tgt_dens, hh);

                for (int i = 0; i < n; i++)
                    scores[i] = scorer_fwd(hh + i * H, w.an_w0, w.an_b0, w.an_w1, w.an_b1);
                int vn[256]; int nv = 0; float nsm = -FLT_MAX;
                for (int i = 0; i < n; i++) {
                    int has = (deg[i] < n - 1);
                    if (has) { vn[nv++] = i; if (scores[i] > nsm) nsm = scores[i]; }
                }
                if (nv == 0) break;
                double z = 0; for (int v = 0; v < nv; v++) z += exp((double)(scores[vn[v]] - nsm));
                double r = rng_double(&fwd_rng) * z, c = 0; int si = nv - 1;
                for (int v = 0; v < nv; v++) { c += exp((double)(scores[vn[v]] - nsm)); if (r <= c) { si = v; break; } }
                int src = vn[si];

                /* Target scorer: softmax sample among src's non-neighbors */
                int tn[256]; int ntgt = 0; float tsm = -FLT_MAX;
                float tscores[256];
                for (int j = 0; j < n; j++) {
                    if (j == src || adj[src * n + j]) continue;
                    float cn_norm = (float)cn[src * n + j] * inv_n;
#ifndef NO_FV
                    float fv_gap = (float)(n * (v2[src] - v2[j]) * (v2[src] - v2[j]));
#else
                    float fv_gap = 0.0f;
#endif
                    tscores[ntgt] = target_scorer_fwd(hh + j * H, cn_norm, fv_gap,
                                                       w.at_w0, w.at_b0, w.at_w1, w.at_b1);
                    tn[ntgt] = j; if (tscores[ntgt] > tsm) tsm = tscores[ntgt]; ntgt++;
                }
                if (ntgt == 0) break;
                z = 0; for (int t = 0; t < ntgt; t++) z += exp((double)(tscores[t] - tsm));
                r = rng_double(&fwd_rng) * z; c = 0; int ti = ntgt - 1;
                for (int t = 0; t < ntgt; t++) { c += exp((double)(tscores[t] - tsm)); if (r <= c) { ti = t; break; } }
                int tgt = tn[ti];

                /* Train on sweep values */
                az_train_step(&w, &grad,
                              w.an_w0, w.an_b0, w.an_w1, w.an_b1,
                              grad.an_w0, grad.an_b0, grad.an_w1, grad.an_b1,
                              w.at_w0, w.at_b0, w.at_w1, w.at_b1,
                              grad.at_w0, grad.at_b0, grad.at_w1, grad.at_b1,
                              adj, deg, tri, snd, cn,
                              n, nm1, step_f, tgt_dens, hh, v2,
                              a_src, a_tgt, a_val, na, src, tgt);

                /* Apply agent's action: add edge, update tri/snd/cn/bridges */
                int u = src < tgt ? src : tgt, v = src < tgt ? tgt : src;
                adj[u * n + v] = adj[v * n + u] = 1; deg[u]++; deg[v]++;
                update_tri_snd_add(adj, deg, n, u, v, tri, snd);
                update_cn_add(adj, n, u, v, cn);
                bridge_add_edge(n, u, v, &bs);
                total_steps++;
            }

            /* === REM PHASE: swaps steps === */
            for (int s = 0; s < swaps; s++) {
                float step_f = (float)s * spe_inv;
                int remaining = swaps - s;

                int na;
#ifdef USE_GPU_SWEEP
                na = gpu_sweep_rem(adj, deg, n, remaining, a_src, a_tgt, a_val);
#else
                sweep_rem_values(adj, deg, n, remaining, bmask, a_src, a_tgt, a_val, &na);
#endif
                if (na == 0) break;

                node_embed_all(&w, deg, tri, snd, v2, n, nm1, step_f, tgt_dens, hh);

                for (int i = 0; i < n; i++)
                    scores[i] = scorer_fwd(hh + i * H, w.rn_w0, w.rn_b0, w.rn_w1, w.rn_b1);

                int vn[256]; int nv = 0; float nsm = -FLT_MAX;
                for (int i = 0; i < n; i++) {
                    int has = (deg[i] > 0);
                    if (has) { vn[nv++] = i; if (scores[i] > nsm) nsm = scores[i]; }
                }
                if (nv == 0) break;
                double z = 0; for (int v = 0; v < nv; v++) z += exp((double)(scores[vn[v]] - nsm));
                double r = rng_double(&fwd_rng) * z, c = 0; int si = nv - 1;
                for (int v = 0; v < nv; v++) { c += exp((double)(scores[vn[v]] - nsm)); if (r <= c) { si = v; break; } }
                int src = vn[si];

                int tn[256]; int ntgt = 0; float tsm2 = -FLT_MAX;
                float tscores[256];
                for (int j = 0; j < n; j++) {
                    if (j == src || !adj[src * n + j]) continue;
                    if (bs.bridge[src * n + j]) continue;
                    float cn_norm = (float)cn[src * n + j] * inv_n;
#ifndef NO_FV
                    float fv_gap = (float)(n * (v2[src] - v2[j]) * (v2[src] - v2[j]));
#else
                    float fv_gap = 0.0f;
#endif
                    tscores[ntgt] = target_scorer_fwd(hh + j * H, cn_norm, fv_gap,
                                                       w.rt_w0, w.rt_b0, w.rt_w1, w.rt_b1);
                    tn[ntgt] = j; if (tscores[ntgt] > tsm2) tsm2 = tscores[ntgt]; ntgt++;
                }
                if (ntgt == 0) break;
                z = 0; for (int t = 0; t < ntgt; t++) z += exp((double)(tscores[t] - tsm2));
                r = rng_double(&fwd_rng) * z; c = 0; int ti = ntgt - 1;
                for (int t = 0; t < ntgt; t++) { c += exp((double)(tscores[t] - tsm2)); if (r <= c) { ti = t; break; } }
                int tgt = tn[ti];

                az_train_step(&w, &grad,
                              w.rn_w0, w.rn_b0, w.rn_w1, w.rn_b1,
                              grad.rn_w0, grad.rn_b0, grad.rn_w1, grad.rn_b1,
                              w.rt_w0, w.rt_b0, w.rt_w1, w.rt_b1,
                              grad.rt_w0, grad.rt_b0, grad.rt_w1, grad.rt_b1,
                              adj, deg, tri, snd, cn,
                              n, nm1, step_f, tgt_dens, hh, v2,
                              a_src, a_tgt, a_val, na, src, tgt);

                /* Apply agent's action: remove edge, update bridges/cn/tri/snd BEFORE modifying adj */
                int u = src < tgt ? src : tgt, v = src < tgt ? tgt : src;
                bridge_rem_edge(adj, n, u, v, &bs);
                update_cn_rem(adj, n, u, v, cn);
                update_tri_snd_rem(adj, deg, n, u, v, tri, snd);
                adj[u * n + v] = adj[v * n + u] = 0; deg[u]--; deg[v]--;
                total_steps++;
            }
        }

        if (total_steps == 0) { free(v2); free(v2_warm); continue; }

        /* Grad scale + clip + Adam */
        float *gp = (float *)&grad; float gnorm = 0;
        for (int i = 0; i < np; i++) gnorm += (double)gp[i] * gp[i]; gnorm = sqrtf(gnorm);
        float scale = 1.0f / (float)total_steps;
        if (gnorm * scale > 1.0f) scale = 1.0f / gnorm;
        for (int i = 0; i < np; i++) gp[i] *= scale;

        adam_t++; float *wp = (float *)&w;
        double bc1 = 1.0 - pow(0.9, adam_t), bc2 = 1.0 - pow(0.999, adam_t);
        for (int i = 0; i < np; i++) {
            adam_m[i] = (float)(0.9 * adam_m[i] + 0.1 * gp[i]);
            adam_v[i] = (float)(0.999 * adam_v[i] + 0.001 * gp[i] * gp[i]);
            float mh = (float)(adam_m[i] / bc1), vh = (float)(adam_v[i] / bc2);
            wp[i] -= (float)(lr * (mh / (sqrtf(vh) + 1e-8f) + 1e-4 * wp[i]));
        }

        if ((ep + 1) % 50 == 0) {
            double elapsed = wall_time() - t0;
            printf("ep %5d/%d | l2=%.3f | n=%d m=%d steps=%d | %.2f ep/s | %.1fm\n",
                   ep + 1, num_episodes, exact_lambda2(adj, n), n, m, total_steps,
                   (ep + 1) / fmax(elapsed, 0.001), elapsed / 60.0);
        }
        if ((ep + 1) % 500 == 0) {
            char p[512]; snprintf(p, sizeof(p), "%s/ckpt_%d.bin", save_dir, ep + 1);
            save_weights(p, &w);
        }
        free(v2); free(v2_warm);
    }
    char p[512]; snprintf(p, sizeof(p), "%s/final.bin", save_dir); save_weights(p, &w);
    printf("Saved to %s\n", save_dir);
    free(adam_m); free(adam_v); free(hh); free(scores);
    free(adj); free(deg); free(tri); free(snd); free(cn); free(bmask);
    bridge_free(&bs);
    free(a_src); free(a_tgt); free(a_val);
}

/* ========================================================================== */
/* L+ pseudoinverse for ER heuristic                                           */
/* ========================================================================== */

extern void dsyev_(const char *, const char *, const int *, double *, const int *,
                   double *, double *, const int *, int *);

static void compute_lpinv(const uint8_t *adj, int n, double *lpinv) {
    int N = n;
    double *L = (double *)malloc((size_t)N * N * sizeof(double));
    for (int i = 0; i < n; i++) {
        double d = 0;
        for (int j = 0; j < n; j++) d += adj[i * n + j];
        for (int j = 0; j < n; j++) L[j * N + i] = (i == j) ? d : -(double)adj[i * n + j];
    }
    double *evals = (double *)malloc((size_t)n * sizeof(double));
    int lwork = 3 * N + 1; double *work = (double *)malloc((size_t)lwork * sizeof(double));
    int info;
    dsyev_("V", "U", &N, L, &N, evals, work, &lwork, &info);
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

static void lpinv_add_edge(double *lpinv, int n, int u, int v) {
    double *w = (double *)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) w[i] = lpinv[i * n + u] - lpinv[i * n + v];
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

static void lpinv_rem_edge(double *lpinv, int n, int u, int v) {
    double *w = (double *)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) w[i] = lpinv[i * n + u] - lpinv[i * n + v];
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

/* ER heuristic episode: random ADD + ER-greedy REM, Sherman-Morrison per swap */
static double er_episode(int n, int m, int swaps, double epoch_C, uint64_t seed) {
    int nn = n * n;
    uint8_t *adj = (uint8_t *)calloc((size_t)nn, 1);
    int *deg = (int *)calloc((size_t)n, sizeof(int));
    RNG rng; rng_init(&rng, seed);
    build_ring_random(adj, deg, n, m, &rng);
    BridgeState bs; bridge_alloc(n, &bs); bridge_init(adj, n, &bs);

    double *lpinv = (double *)malloc((size_t)nn * sizeof(double));
    compute_lpinv(adj, n, lpinv);
    int *tabu = (int *)calloc((size_t)nn, sizeof(int));
    int tabu_len = 0; /* disabled */

    int num_epochs = (int)(epoch_C * n);
    if (num_epochs < 4) num_epochs = 4;
    uint8_t *best_adj = (uint8_t *)malloc((size_t)nn);
    double best_l2 = -1;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        int added[3][2]; int n_added = 0;
        /* ADD: random edges (respect tabu) */
        for (int s = 0; s < swaps; s++) {
            int vn[256]; int nv = 0;
            for (int i = 0; i < n; i++) if (deg[i] < n - 1) vn[nv++] = i;
            if (nv == 0) break;
            int sel = vn[(int)(rng_double(&rng) * nv)];
            int tn[256]; int nt = 0;
            for (int j = 0; j < n; j++) {
                if (j != sel && !adj[sel * n + j] && tabu[sel * n + j] <= epoch) tn[nt++] = j;
            }
            if (nt == 0) break;
            int tgt = tn[(int)(rng_double(&rng) * nt)];
            int u = sel < tgt ? sel : tgt, v = sel < tgt ? tgt : sel;
            adj[u * n + v] = adj[v * n + u] = 1; deg[u]++; deg[v]++;
            bridge_add_edge(n, u, v, &bs);
            lpinv_add_edge(lpinv, n, u, v);
            if (n_added < 3) { added[n_added][0] = u; added[n_added][1] = v; n_added++; }
        }
        /* REM: ER-greedy. Node = lowest L+[i,i] (most connected). Target = lowest R_eff neighbor. */
        for (int s = 0; s < swaps; s++) {
            /* Node selection: pick node with lowest ER centrality (L+[i,i]) among those with removable edges */
            int sel = -1; double best_diag = 1e30;
            for (int i = 0; i < n; i++) {
                if (deg[i] <= 0) continue;
                /* check if node has at least one non-bridge edge */
                int has_rem = 0;
                for (int j = 0; j < n; j++) {
                    if (adj[i * n + j] && !bs.bridge[i * n + j]) { has_rem = 1; break; }
                }
                if (!has_rem) continue;
                if (lpinv[i * n + i] < best_diag) { best_diag = lpinv[i * n + i]; sel = i; }
            }
            if (sel < 0) break;
            /* Target: lowest R_eff neighbor (most redundant edge) */
            int best_j = -1; double best_reff = 1e30;
            for (int j = 0; j < n; j++) {
                if (!adj[sel * n + j] || bs.bridge[sel * n + j]) continue;
                double reff = lpinv[sel * n + sel] + lpinv[j * n + j] - 2.0 * lpinv[sel * n + j];
                if (reff < best_reff) { best_reff = reff; best_j = j; }
            }
            if (best_j < 0) break;
            int u = sel < best_j ? sel : best_j, v = sel < best_j ? best_j : sel;
            lpinv_rem_edge(lpinv, n, u, v);
            bridge_rem_edge(adj, n, u, v, &bs);
            adj[u * n + v] = adj[v * n + u] = 0; deg[u]--; deg[v]--;
        }
        /* Tabu: ban added edges that got removed */
        for (int k = 0; k < n_added; k++) {
            int u = added[k][0], v = added[k][1];
            if (!adj[u * n + v]) tabu[u * n + v] = tabu[v * n + u] = epoch + tabu_len;
        }
        /* Track best */
        double cur_l2 = exact_lambda2(adj, n);
        if (cur_l2 > best_l2) { best_l2 = cur_l2; memcpy(best_adj, adj, (size_t)nn); }
    }
    double final_l2 = best_l2 > 0 ? best_l2 : exact_lambda2(adj, n);
    free(adj); free(deg); free(lpinv); free(tabu); free(best_adj); bridge_free(&bs);
    return final_l2;
}

/* ========================================================================== */
/* Inference                                                                   */
/* ========================================================================== */

static double mlp_episode(const MLPWeights *w, int n, int m,
                           int swaps, double epoch_C, int tabu_mult, double epsilon, int trace, uint64_t seed) {
    int nn = n * n;
    uint8_t *adj = (uint8_t *)calloc((size_t)nn, 1);
    int *degrees = (int *)calloc((size_t)n, sizeof(int));
    int *tri = (int *)calloc((size_t)n, sizeof(int));
    int *snd = (int *)calloc((size_t)n, sizeof(int));
    int *cn = (int *)calloc((size_t)nn, sizeof(int));
    RNG rng; rng_init(&rng, seed);
    build_ring_random(adj, degrees, n, m, &rng);
    init_tri_snd(adj, degrees, n, tri, snd);
    init_cn(adj, n, cn);
    float nm1 = (float)(n - 1); if (nm1 < 1.0f) nm1 = 1.0f;
    float spe_inv = 1.0f / (float)swaps;
    float inv_n = 1.0f / (float)n;

    int max_m = n * (n - 1) / 2;
    float tgt_dens = (float)(m - (n - 1)) / (float)(max_m - (n - 1));
    int num_epochs = (int)(epoch_C * n);
    if (num_epochs < 4) num_epochs = 4;

    float *hh = (float *)malloc((size_t)n * H * sizeof(float));
    float *scores = (float *)malloc((size_t)n * sizeof(float));
    int *tabu = (int *)calloc((size_t)nn, sizeof(int));
    int tabu_len = tabu_mult > 0 ? tabu_mult * n : 0;
    BridgeState bs; bridge_alloc(n, &bs); bridge_init(adj, n, &bs);
    double *v2 = (double *)malloc((size_t)n * sizeof(double));
    double v2_lam;
    double *v2_warm = (double *)malloc((size_t)n * sizeof(double));

    float prev_add_std = -1.0f; int degen_count = 0; int active_swaps = swaps;

    /* Metropolis state: save graph before each epoch for revert */
    uint8_t *save_adj = (uint8_t *)malloc((size_t)nn);
    int *save_deg = (int *)malloc((size_t)n * sizeof(int));
    int *save_tri = (int *)malloc((size_t)n * sizeof(int));
    int *save_snd = (int *)malloc((size_t)n * sizeof(int));
    int *save_cn = (int *)malloc((size_t)nn * sizeof(int));
    double *save_v2_warm = (double *)malloc((size_t)n * sizeof(double));
    /* SA temperature: geometric cooling */
    double T_start = 16.0 / n, T_end = 0.032 / n;  /* scale with 1/n to match Δλ₂ magnitude */
    double T_ratio = pow(T_end / T_start, 1.0 / fmax(num_epochs - 1, 1));

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        double T = T_start * pow(T_ratio, epoch);

        /* Lanczos at epoch start */
        lanczos_ext_k(adj, n, epoch > 0 ? v2_warm : NULL, LANCZOS_WARM_K, 1, v2_warm, &v2_lam);
        memcpy(v2, v2_warm, (size_t)n * sizeof(double));
        double l2_before = v2_lam;

        /* Save state before epoch */
        memcpy(save_adj, adj, (size_t)nn);
        memcpy(save_deg, degrees, (size_t)n * sizeof(int));
        memcpy(save_tri, tri, (size_t)n * sizeof(int));
        memcpy(save_snd, snd, (size_t)n * sizeof(int));
        memcpy(save_cn, cn, (size_t)nn * sizeof(int));
        memcpy(save_v2_warm, v2_warm, (size_t)n * sizeof(double));

        /* Adaptive swaps: demote on degeneracy at current level */
        int cur_swaps = active_swaps;

        int added_edges[3][2]; int n_added = 0;
        /* ADD */
        for (int s = 0; s < cur_swaps; s++) {
            float step_f = (float)s * spe_inv;
            node_embed_all(w, degrees, tri, snd, v2, n, nm1, step_f, tgt_dens, hh);
            for (int i = 0; i < n; i++)
                scores[i] = scorer_fwd(hh + i * H, w->an_w0, w->an_b0, w->an_w1, w->an_b1);
            int vn[256]; int nv = 0;
            int sel = -1; float bv = -FLT_MAX;
            for (int i = 0; i < n; i++) {
                if (degrees[i] < n - 1) { vn[nv++] = i; if (scores[i] > bv) { bv = scores[i]; sel = i; } }
            }
            if (nv == 0) break;
            /* Track degeneracy at step 0 */
            if (s == 0 && nv > 1) {
                float sum = 0, sum2 = 0;
                for (int v = 0; v < nv; v++) { float sc = scores[vn[v]]; sum += sc; sum2 += sc * sc; }
                float mean_sc = sum / nv; float add_std = sqrtf(sum2 / nv - mean_sc * mean_sc);
                if (prev_add_std >= 0 && fabsf(add_std - prev_add_std) < 0.001f) {
                    degen_count++;
                    /* promotion disabled -- testing random ADD only */
                } else {
                    degen_count = 0;
                    /* don't reset -- stay at promoted level to see if it degenerates there */
                }
                if (trace) {
                    double l2_now = exact_lambda2(adj, n);
                    printf("%d,%.6f,%.4f,%d,%d\n", epoch, l2_now, add_std, cur_swaps, degen_count);
                }
                prev_add_std = add_std;
            }
            /* Random ADD when degenerate (no promotion) */
            if (degen_count > 0) sel = vn[(int)(rng_double(&rng) * nv)];
            float tscores[256]; int tn[256]; int nt = 0; float tsm = -FLT_MAX;
            for (int j = 0; j < n; j++) {
                if (j == sel || adj[sel * n + j]) continue;
                if (tabu[sel * n + j] > epoch) continue;
                float cn_norm = (float)cn[sel * n + j] * inv_n;
#ifndef NO_FV
                float fv_gap = (float)(n * (v2[sel] - v2[j]) * (v2[sel] - v2[j]));
#else
                float fv_gap = 0.0f;
#endif
                tscores[nt] = target_scorer_fwd(hh + j * H, cn_norm, fv_gap,
                                                 w->at_w0, w->at_b0, w->at_w1, w->at_b1);
                if (tscores[nt] > tsm) tsm = tscores[nt];
                tn[nt] = j; nt++;
            }
            if (nt == 0) break;
            int best_t = 0;
            if (degen_count > 0) {
                best_t = (int)(rng_double(&rng) * nt);
            } else {
                for (int t = 1; t < nt; t++) if (tscores[t] > tscores[best_t]) best_t = t;
            }
            int tgt = tn[best_t];
            int u = sel < tgt ? sel : tgt, v = sel < tgt ? tgt : sel;
            adj[u * n + v] = adj[v * n + u] = 1; degrees[u]++; degrees[v]++;
            update_tri_snd_add(adj, degrees, n, u, v, tri, snd);
            update_cn_add(adj, n, u, v, cn);
            bridge_add_edge(n, u, v, &bs);
            lanczos_ext_k(adj, n, v2_warm, LANCZOS_WARM_K, 1, v2_warm, &v2_lam);
            memcpy(v2, v2_warm, (size_t)n * sizeof(double));
            if (n_added < 3) { added_edges[n_added][0] = u; added_edges[n_added][1] = v; n_added++; }
        }
        /* REM */
        for (int s = 0; s < cur_swaps; s++) {
            float step_f = (float)s * spe_inv;
            node_embed_all(w, degrees, tri, snd, v2, n, nm1, step_f, tgt_dens, hh);
            for (int i = 0; i < n; i++)
                scores[i] = scorer_fwd(hh + i * H, w->rn_w0, w->rn_b0, w->rn_w1, w->rn_b1);
            int sel = -1; float bv = -FLT_MAX;
            for (int i = 0; i < n; i++) {
                if (degrees[i] > 0 && scores[i] > bv) { bv = scores[i]; sel = i; }
            }
            if (sel < 0) break;
            float tscores[256]; int tn[256]; int nt = 0;
            for (int j = 0; j < n; j++) {
                if (j == sel || !adj[sel * n + j]) continue;
                if (bs.bridge[sel * n + j]) continue;
                float cn_norm = (float)cn[sel * n + j] * inv_n;
#ifndef NO_FV
                float fv_gap = (float)(n * (v2[sel] - v2[j]) * (v2[sel] - v2[j]));
#else
                float fv_gap = 0.0f;
#endif
                tscores[nt] = target_scorer_fwd(hh + j * H, cn_norm, fv_gap,
                                                 w->rt_w0, w->rt_b0, w->rt_w1, w->rt_b1);
                tn[nt] = j; nt++;
            }
            if (nt == 0) break;
            int best_t = 0; for (int t = 1; t < nt; t++) if (tscores[t] > tscores[best_t]) best_t = t;
            int tgt = tn[best_t];
            int u = sel < tgt ? sel : tgt, v = sel < tgt ? tgt : sel;
            bridge_rem_edge(adj, n, u, v, &bs);
            update_cn_rem(adj, n, u, v, cn);
            update_tri_snd_rem(adj, degrees, n, u, v, tri, snd);
            adj[u * n + v] = adj[v * n + u] = 0; degrees[u]--; degrees[v]--;
            lanczos_ext_k(adj, n, v2_warm, LANCZOS_WARM_K, 1, v2_warm, &v2_lam);
            memcpy(v2, v2_warm, (size_t)n * sizeof(double));
        }
        /* Metropolis at max level: reject worsening swaps to lock in gains */
        if (active_swaps >= 5) {
            double l2_after_lam;
            lanczos_ext_k(adj, n, v2_warm, LANCZOS_WARM_K, 1, v2_warm, &l2_after_lam);
            if (l2_after_lam < l2_before) {
                memcpy(adj, save_adj, (size_t)nn);
                memcpy(degrees, save_deg, (size_t)n * sizeof(int));
                memcpy(tri, save_tri, (size_t)n * sizeof(int));
                memcpy(snd, save_snd, (size_t)n * sizeof(int));
                memcpy(cn, save_cn, (size_t)nn * sizeof(int));
                memcpy(v2_warm, save_v2_warm, (size_t)n * sizeof(double));
                memcpy(v2, v2_warm, (size_t)n * sizeof(double));
                bridge_init(adj, n, &bs);
            }
        }
        if (trace) {
            double l2_now = exact_lambda2(adj, n);
            printf("%d,%.6f,%.4f,%d,%d\n", epoch, l2_now, prev_add_std, cur_swaps, degen_count);
        }
    }
    double final_l2 = exact_lambda2(adj, n);
    free(adj); free(degrees); free(tri); free(snd); free(cn);
    free(hh); free(scores); free(tabu); bridge_free(&bs);
    free(v2); free(v2_warm); free(save_adj); free(save_deg); free(save_tri);
    free(save_snd); free(save_cn); free(save_v2_warm);
    return final_l2;
}

/* ========================================================================== */
/* Main                                                                        */
/* ========================================================================== */

typedef struct { int n, m; double fv, er, sw025, sw050, sw075, best; } BL;
static BL *bls = NULL; static int nbl = 0;

static void load_bl(const char *path) {
    FILE *f = fopen(path, "r"); if (!f) return;
    char line[1024]; fgets(line, sizeof(line), f);
    int cap = 8192; bls = (BL *)malloc((size_t)cap * sizeof(BL));
    while (fgets(line, sizeof(line), f)) {
        BL e = {0};
        sscanf(line, "%d,%d,%lf,%lf,%lf,%lf,%lf", &e.n, &e.m, &e.fv, &e.er, &e.sw025, &e.sw050, &e.sw075);
        e.best = e.fv; if (e.er > e.best) e.best = e.er;
        if (e.sw025 > e.best) e.best = e.sw025;
        if (e.sw050 > e.best) e.best = e.sw050;
        if (e.sw075 > e.best) e.best = e.sw075;
        if (nbl >= cap) { cap *= 2; bls = realloc(bls, (size_t)cap * sizeof(BL)); }
        bls[nbl++] = e;
    }
    fclose(f); printf("Loaded %d baselines\n", nbl);
}

static BL *find_bl(int n, int m) {
    for (int i = 0; i < nbl; i++) if (bls[i].n == n && bls[i].m == m) return &bls[i];
    return NULL;
}

int main(int argc, char **argv) {
    int n_values[64], num_n = 0; char *ckpt = NULL; char *bl_path = NULL; char *csv_path = NULL;
    int do_train = 0, train_ep = 5000;
    int min_n = 8, max_n = 10; double train_lr = 1e-3;
    int swaps = 3; double epoch_C = 1.0;
    int tabu_mult = 1;  /* tabu_len = tabu_mult * n. 0 = no tabu */
    double epsilon = 0.0; /* epsilon-greedy exploration. 0 = pure greedy */
    int m_filter = 0;   /* 0=all, 1=odd m only, 2=even m only */
    int er_heuristic = 0;
    char save_dir[256] = "logs/test_az"; uint64_t seed = 42;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            char *tok = strtok(argv[++i], ",");
            while (tok && num_n < 64) { n_values[num_n++] = atoi(tok); tok = strtok(NULL, ","); }
        }
        else if (strcmp(argv[i], "--checkpoint") == 0 && i + 1 < argc) ckpt = argv[++i];
        else if (strcmp(argv[i], "--baselines") == 0 && i + 1 < argc) bl_path = argv[++i];
        else if (strcmp(argv[i], "--train") == 0) do_train = 1;
        else if (strcmp(argv[i], "--episodes") == 0 && i + 1 < argc) train_ep = atoi(argv[++i]);
        else if (strcmp(argv[i], "--min-n") == 0 && i + 1 < argc) min_n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-n") == 0 && i + 1 < argc) max_n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) train_lr = atof(argv[++i]);
        else if (strcmp(argv[i], "--swaps") == 0 && i + 1 < argc) swaps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--epoch-C") == 0 && i + 1 < argc) epoch_C = atof(argv[++i]);
        else if (strcmp(argv[i], "--save-dir") == 0 && i + 1 < argc) snprintf(save_dir, sizeof(save_dir), "%s", argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint64_t)atol(argv[++i]);
        else if (strcmp(argv[i], "--tabu") == 0 && i + 1 < argc) tabu_mult = atoi(argv[++i]);
        else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) csv_path = argv[++i];
        else if (strcmp(argv[i], "--m-filter") == 0 && i + 1 < argc) m_filter = atoi(argv[++i]);
        else if (strcmp(argv[i], "--epsilon") == 0 && i + 1 < argc) epsilon = atof(argv[++i]);
        else if (strcmp(argv[i], "--er-heuristic") == 0) er_heuristic = 1;
        else if (strcmp(argv[i], "--trace") == 0 && i + 1 < argc) {
            /* Trace single (n, m): print per-epoch l2 and add_std */
            int tn = n_values[0], tm = atoi(argv[++i]);
            MLPWeights tw; load_weights(ckpt, &tw);
            printf("epoch,l2,add_std,swaps,degen\n");
            mlp_episode(&tw, tn, tm, swaps, epoch_C, tabu_mult, epsilon, 1, (uint64_t)(tn * 100 + tm));
            return 0;
        }
    }

    if (do_train) { mlp_train(min_n, max_n, train_ep, swaps, epoch_C, train_lr, seed, save_dir); return 0; }

    if (!ckpt || num_n == 0) {
        printf("Usage: %s --train --min-n 8 --max-n 10 --episodes 5000\n", argv[0]);
        printf("   or: %s --checkpoint model.bin --n 8,16,24 --baselines bl.csv\n", argv[0]);
        return 1;
    }

    MLPWeights w;
    if (load_weights(ckpt, &w) != 0) { fprintf(stderr, "Failed to load: %s\n", ckpt); return 1; }
    printf("Loaded: %s\n", ckpt);
    if (bl_path) load_bl(bl_path);

    FILE *csv_f = NULL;
    if (csv_path) {
        csv_f = fopen(csv_path, "w");
        if (csv_f) fprintf(csv_f, "m,score\n");
    }
    printf("AZ Eval | swaps=%d, C=%.1f, threads=%d\n", swaps, epoch_C, omp_get_max_threads());
    printf("================================================================\n");

    for (int ni = 0; ni < num_n; ni++) {
        int n = n_values[ni]; int mx = n * (n - 1) / 2;
        int *m_list = (int *)malloc((size_t)(mx + 1) * sizeof(int)); int nm = 0;
        for (int mv = n - 1; mv <= mx; mv++) {
            if (m_filter == 1 && mv % 2 == 0) continue;  /* odd only */
            if (m_filter == 2 && mv % 2 == 1) continue;  /* even only */
            BL *b = find_bl(n, mv);
            if (b && b->best > 0) m_list[nm++] = mv;
            else if (nbl == 0) m_list[nm++] = mv;  /* no baselines: include all m */
        }
        if (nm == 0) { free(m_list); continue; }
        printf("\nn=%d (%d configs)\n", n, nm);
        double t0e = wall_time();
        double *jl = (double *)malloc((size_t)nm * sizeof(double));
        #pragma omp parallel for schedule(dynamic, 1)
        for (int mi = 0; mi < nm; mi++) {
            uint64_t s = (uint64_t)(n * 100 + m_list[mi]);
            jl[mi] = er_heuristic
                ? er_episode(n, m_list[mi], swaps, epoch_C, s)
                : mlp_episode(&w, n, m_list[mi], swaps, epoch_C, tabu_mult, epsilon, 0, s);
        }
        double sr = 0, sf = 0, sb = 0; int wf = 0, wb = 0, tot = 0;
        double bin_rl[10] = {0}, bin_fv[10] = {0}, bin_best[10] = {0};
        int bin_w[10] = {0}, bin_cnt[10] = {0};
        for (int mi = 0; mi < nm; mi++) {
            BL *b = find_bl(n, m_list[mi]);
            double fv_val = b ? b->fv : 0, best_val = b ? b->best : 0;
            sr += jl[mi]; sf += fv_val; sb += best_val;
            if (b && jl[mi] > fv_val + 1e-6) wf++;
            if (b && jl[mi] > best_val + 1e-6) wb++;
            tot++;
            double density = (double)(m_list[mi] - (n-1)) / (double)(mx - (n-1));
            int bi = (int)(density * 10); if (bi >= 10) bi = 9; if (bi < 0) bi = 0;
            bin_rl[bi] += jl[mi]; bin_best[bi] += best_val;
            if (b && jl[mi] > best_val + 1e-6) bin_w[bi]++;
            bin_cnt[bi]++;
            if (csv_f) fprintf(csv_f, "%d,%.10f\n", m_list[mi], jl[mi]);
        }
        double elapsed = wall_time() - t0e;
        if (sf > 0)
            printf("  vs FV: %.1f%% (%dW/%d) | vs Best: %.1f%% (%dW/%d) | %.1fs\n",
                   sr / sf * 100, wf, tot, sr / sb * 100, wb, tot, elapsed);
        else
            printf("  %d configs | avg λ₂=%.4f | %.1fs\n", tot, sr / tot, elapsed);
        printf("  Density: ");
        for (int bi = 0; bi < 10; bi++) {
            if (bin_cnt[bi] == 0) continue;
            double pct = bin_rl[bi] / bin_best[bi] * 100;
            printf("%d0-%d0%%:%.1f%%(%dW/%d) ", bi, bi+1, pct, bin_w[bi], bin_cnt[bi]);
        }
        printf("\n");
        free(m_list); free(jl);
    }
    if (csv_f) { fclose(csv_f); printf("Results saved to %s\n", csv_path); }
    free(bls); return 0;
}
