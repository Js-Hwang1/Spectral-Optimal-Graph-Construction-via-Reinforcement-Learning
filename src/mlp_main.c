/*
 * mlp_main.c — MLP-only RL for G(n,m) algebraic connectivity.
 *
 * No GNN. Node embedding from degree only. Hierarchical action: node→edge.
 * O(N³) inference: K=4N epochs × 2N removals × O(N) per step.
 *
 * Training: REINFORCE with per-step Δλ₂ reward, mean-return baseline.
 * Converges in ~1000 episodes (~7 seconds).
 *
 * Usage:
 *   ./crl_mlp --train --min-n 8 --max-n 32 --episodes 5000
 *   ./crl_mlp --checkpoint model.bin --n 8,16,24,32 --baselines baselines.csv
 */

#include "crl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include <sys/time.h>
#include <sys/stat.h>

#define H 64
#define MLP_MAGIC 0x4D4C5031

static double wall_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static inline float silu_f(float x) { return x / (1.0f + expf(-x)); }
static inline float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float silu_d(float x) { float s = sigmoid_f(x); return s * (1.0f + x * (1.0f - s)); }

/* ========================================================================== */
/* Weights                                                                     */
/* ========================================================================== */

typedef struct {
    float ne_w0[H]; float ne_b0[H];          /* node_embed layer 0: (1)→H */
    float ne_w1[H * H]; float ne_b1[H];      /* node_embed layer 1: H→H */
    float ns_w0[H * H]; float ns_b0[H];      /* node_scorer layer 0: H→H */
    float ns_w1[H]; float ns_b1[1];           /* node_scorer layer 1: H→1 */
    float es_w0[H * 2 * H]; float es_b0[H];  /* edge_scorer layer 0: 2H→H */
    float es_w1[H]; float es_b1[1];           /* edge_scorer layer 1: H→1 */
} MLPWeights;

typedef struct {
    float ne_w0[H]; float ne_b0[H];
    float ne_w1[H * H]; float ne_b1[H];
    float ns_w0[H * H]; float ns_b0[H];
    float ns_w1[H]; float ns_b1[1];
    float es_w0[H * 2 * H]; float es_b0[H];
    float es_w1[H]; float es_b1[1];
} PolicyGrads;

/* ========================================================================== */
/* Forward helpers                                                             */
/* ========================================================================== */

static void node_embed(const MLPWeights *w, float deg, float *out) {
    float h0[H];
    for (int i = 0; i < H; i++)
        h0[i] = silu_f(w->ne_w0[i] * deg + w->ne_b0[i]);
    for (int i = 0; i < H; i++) {
        float s = w->ne_b1[i];
        for (int j = 0; j < H; j++) s += w->ne_w1[i * H + j] * h0[j];
        out[i] = silu_f(s);
    }
}

static float mlp_fwd(const float *input, int in_dim,
                      const float *W0, const float *b0,
                      const float *W1, const float *b1) {
    float hid[H];
    for (int i = 0; i < H; i++) {
        float s = b0[i];
        for (int j = 0; j < in_dim; j++) s += W0[i * in_dim + j] * input[j];
        hid[i] = silu_f(s);
    }
    float out = b1[0];
    for (int j = 0; j < H; j++) out += W1[j] * hid[j];
    return out;
}

/* ========================================================================== */
/* Checkpoint I/O                                                              */
/* ========================================================================== */

static void save_weights(const char *path, const MLPWeights *w) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    uint32_t magic = MLP_MAGIC;
    fwrite(&magic, 4, 1, f);
    fwrite(w, sizeof(MLPWeights), 1, f);
    fclose(f);
}

static int load_weights(const char *path, MLPWeights *w) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t magic;
    fread(&magic, 4, 1, f);
    if (magic != MLP_MAGIC) { fclose(f); return -2; }
    fread(w, sizeof(MLPWeights), 1, f);
    fclose(f);
    return 0;
}

static void init_weights(MLPWeights *w, uint64_t seed) {
    RNG rng; rng_init(&rng, seed);
    float *p = (float *)w;
    int np = (int)(sizeof(MLPWeights) / sizeof(float));
    for (int i = 0; i < np; i += 2) {
        double u1 = rng_double(&rng) * 0.998 + 0.001, u2 = rng_double(&rng);
        double r = sqrt(-2.0 * log(u1));
        p[i] = (float)(r * cos(2.0 * M_PI * u2));
        if (i + 1 < np) p[i + 1] = (float)(r * sin(2.0 * M_PI * u2));
    }
    float s;
    s = sqrtf(2.0f / (1 + H)); for (int i = 0; i < H; i++) w->ne_w0[i] *= s;
    memset(w->ne_b0, 0, sizeof(w->ne_b0));
    s = sqrtf(2.0f / (H + H)); for (int i = 0; i < H * H; i++) w->ne_w1[i] *= s;
    memset(w->ne_b1, 0, sizeof(w->ne_b1));
    s = sqrtf(2.0f / (H + H)); for (int i = 0; i < H * H; i++) w->ns_w0[i] *= s;
    memset(w->ns_b0, 0, sizeof(w->ns_b0));
    s = sqrtf(2.0f / (H + 1)); for (int i = 0; i < H; i++) w->ns_w1[i] *= s;
    memset(w->ns_b1, 0, sizeof(w->ns_b1));
    s = sqrtf(2.0f / (2 * H + H)); for (int i = 0; i < H * 2 * H; i++) w->es_w0[i] *= s;
    memset(w->es_b0, 0, sizeof(w->es_b0));
    s = sqrtf(2.0f / (H + 1)); for (int i = 0; i < H; i++) w->es_w1[i] *= s;
    memset(w->es_b1, 0, sizeof(w->es_b1));
}

/* ========================================================================== */
/* Inference episode                                                           */
/* ========================================================================== */

static double mlp_episode(const MLPWeights *w, int n, int m,
                           int num_epochs, int num_adds, uint64_t seed) {
    int nn = n * n;
    uint8_t *adj = (uint8_t *)calloc((size_t)nn, 1);
    int *degrees = (int *)calloc((size_t)n, sizeof(int));
    RNG rng; rng_init(&rng, seed);
    build_ring_random(adj, degrees, n, m, &rng);
    double best_l2 = exact_lambda2(adj, n);
    float nm1 = (float)(n - 1); if (nm1 < 1.0f) nm1 = 1.0f;

    float *h = (float *)malloc((size_t)n * H * sizeof(float));
    float *ns = (float *)malloc((size_t)n * sizeof(float));
    uint8_t *bridge_mask = (uint8_t *)malloc((size_t)nn);
    int max_pairs = n * (n - 1) / 2;
    int *ne_i = (int *)malloc((size_t)max_pairs * sizeof(int));
    int *ne_j = (int *)malloc((size_t)max_pairs * sizeof(int));
    int *tabu = (int *)calloc((size_t)nn, sizeof(int));
    int tt = n / 2;
    uint8_t *added = (uint8_t *)malloc((size_t)nn);

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        memset(added, 0, (size_t)nn);

        /* Add random non-edges (skip tabu) */
        int ne_c = 0;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (!adj[i * n + j] && tabu[i * n + j] <= epoch) {
                    ne_i[ne_c] = i; ne_j[ne_c] = j; ne_c++;
                }
        int actual = num_adds < ne_c ? num_adds : ne_c;
        for (int i = 0; i < actual; i++) {
            int k = i + rng_int(&rng, ne_c - i);
            int ti = ne_i[i]; ne_i[i] = ne_i[k]; ne_i[k] = ti;
            int tj = ne_j[i]; ne_j[i] = ne_j[k]; ne_j[k] = tj;
        }
        for (int i = 0; i < actual; i++) {
            int ai = ne_i[i], aj = ne_j[i];
            adj[ai * n + aj] = adj[aj * n + ai] = 1;
            degrees[ai]++; degrees[aj]++;
            added[ai * n + aj] = added[aj * n + ai] = 1;
        }

        int cur_m = 0;
        for (int i = 0; i < n; i++) cur_m += degrees[i];
        cur_m /= 2;
        int to_remove = cur_m - m;

        /* Greedy autoregressive removal */
        for (int step = 0; step < to_remove; step++) {
            for (int i = 0; i < n; i++)
                node_embed(w, (float)degrees[i] / nm1, h + i * H);
            for (int i = 0; i < n; i++)
                ns[i] = mlp_fwd(h + i * H, H, w->ns_w0, w->ns_b0, w->ns_w1, w->ns_b1);

            cur_m = 0;
            for (int i = 0; i < n; i++) cur_m += degrees[i];
            cur_m /= 2;
            if (cur_m <= 2 * n) find_bridges(adj, n, bridge_mask);
            else memset(bridge_mask, 0, (size_t)nn);

            int best_node = -1;
            float best_ns_val = -FLT_MAX;
            for (int i = 0; i < n; i++) {
                int has = 0;
                for (int j = 0; j < n; j++)
                    if (i != j && adj[i * n + j] && !bridge_mask[(i < j ? i : j) * n + (i < j ? j : i)]) { has = 1; break; }
                if (has && ns[i] > best_ns_val) { best_ns_val = ns[i]; best_node = i; }
            }
            if (best_node < 0) break;

            float ef[2 * H];
            memcpy(ef, h + best_node * H, (size_t)H * sizeof(float));
            int best_nbr = -1;
            float best_es = -FLT_MAX;
            for (int j = 0; j < n; j++) {
                if (j == best_node || !adj[best_node * n + j]) continue;
                int a = best_node < j ? best_node : j, b = best_node < j ? j : best_node;
                if (bridge_mask[a * n + b]) continue;
                memcpy(ef + H, h + j * H, (size_t)H * sizeof(float));
                float es = mlp_fwd(ef, 2 * H, w->es_w0, w->es_b0, w->es_w1, w->es_b1);
                if (es > best_es) { best_es = es; best_nbr = j; }
            }
            if (best_nbr < 0) break;

            adj[best_node * n + best_nbr] = adj[best_nbr * n + best_node] = 0;
            degrees[best_node]--; degrees[best_nbr]--;
        }

        /* Tabu: ban added-then-removed edges */
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (added[i * n + j] && !adj[i * n + j])
                    tabu[i * n + j] = tabu[j * n + i] = epoch + tt;

    }

    double final_l2 = exact_lambda2(adj, n);
    if (final_l2 > best_l2) best_l2 = final_l2;

    free(adj); free(degrees); free(h); free(ns);
    free(bridge_mask); free(ne_i); free(ne_j);
    free(tabu); free(added);
    return best_l2;
}

/* ========================================================================== */
/* Training: REINFORCE with per-step Δλ₂ reward                               */
/* ========================================================================== */

typedef struct { float pre0[H]; float h0[H]; float pre1[H]; float h1[H]; float in_val; } NECache;
typedef struct { float pre[H]; float hid[H]; float input[2 * H]; int in_dim; } M2Cache;

static void node_embed_cached(const MLPWeights *w, float deg, float *out, NECache *c) {
    c->in_val = deg;
    for (int i = 0; i < H; i++) { c->pre0[i] = w->ne_w0[i] * deg + w->ne_b0[i]; c->h0[i] = silu_f(c->pre0[i]); }
    for (int i = 0; i < H; i++) {
        float s = w->ne_b1[i]; for (int j = 0; j < H; j++) s += w->ne_w1[i * H + j] * c->h0[j];
        c->pre1[i] = s; c->h1[i] = silu_f(s); out[i] = c->h1[i];
    }
}

static float mlp_fwd_cached(const float *input, int in_dim,
                              const float *W0, const float *b0,
                              const float *W1, const float *b1, M2Cache *c) {
    c->in_dim = in_dim; memcpy(c->input, input, (size_t)in_dim * sizeof(float));
    for (int i = 0; i < H; i++) {
        float s = b0[i]; for (int j = 0; j < in_dim; j++) s += W0[i * in_dim + j] * input[j];
        c->pre[i] = s; c->hid[i] = silu_f(s);
    }
    float out = b1[0]; for (int j = 0; j < H; j++) out += W1[j] * c->hid[j];
    return out;
}

static void mlp_bwd(const M2Cache *c, const float *W0, const float *W1, int h_dim,
                     float d_out, float *dW0, float *db0, float *dW1, float *db1, float *d_in) {
    int in_dim = c->in_dim;
    db1[0] += d_out;
    float dh[H];
    for (int j = 0; j < h_dim; j++) { dW1[j] += d_out * c->hid[j]; dh[j] = d_out * W1[j]; }
    for (int i = 0; i < h_dim; i++) {
        float dp = dh[i] * silu_d(c->pre[i]); db0[i] += dp;
        for (int j = 0; j < in_dim; j++) { dW0[i * in_dim + j] += dp * c->input[j]; if (d_in) d_in[j] += dp * W0[i * in_dim + j]; }
    }
}

static void mlp_train(int min_n, int max_n, int num_episodes, int num_adds_mult,
                       double lr, uint64_t seed, const char *save_dir) {
    MLPWeights w; init_weights(&w, seed);
    int np = (int)(sizeof(MLPWeights) / sizeof(float));
    float *adam_m = (float *)calloc((size_t)np, sizeof(float));
    float *adam_v = (float *)calloc((size_t)np, sizeof(float));
    int adam_t = 0;
    RNG trng; rng_init(&trng, seed);
    mkdir(save_dir, 0755);
    double t0 = wall_time();
    int max_n_alloc = max_n, max_steps = max_n * num_adds_mult + max_n;
    float *h = (float *)malloc((size_t)max_n_alloc * H * sizeof(float));
    float *ns = (float *)malloc((size_t)max_n_alloc * sizeof(float));
    uint8_t *adj = (uint8_t *)calloc((size_t)max_n_alloc * max_n_alloc, 1);
    uint8_t *adj_saved = (uint8_t *)calloc((size_t)max_n_alloc * max_n_alloc, 1);
    int *deg = (int *)calloc((size_t)max_n_alloc, sizeof(int));
    int *deg_saved = (int *)calloc((size_t)max_n_alloc, sizeof(int));
    uint8_t *bmask = (uint8_t *)malloc((size_t)max_n_alloc * max_n_alloc);
    double *rewards = (double *)malloc((size_t)max_steps * sizeof(double));
    int *s_node = (int *)malloc((size_t)max_steps * sizeof(int));
    int *s_nbr = (int *)malloc((size_t)max_steps * sizeof(int));

    printf("MLP Train | n=[%d,%d] | ep=%d | adds=%d*N | lr=%.1e\n", min_n, max_n, num_episodes, num_adds_mult, lr);
    printf("================================================================\n");

    for (int ep = 0; ep < num_episodes; ep++) {
        int n = min_n + rng_int(&trng, max_n - min_n + 1);
        int max_m = n * (n - 1) / 2, m = n + rng_int(&trng, max_m - n + 1);
        int nn = n * n;
        float nm1 = (float)(n - 1); if (nm1 < 1.0f) nm1 = 1.0f;

        memset(adj, 0, (size_t)nn); memset(deg, 0, (size_t)n * sizeof(int));
        build_ring_random(adj, deg, n, m, &trng);

        int num_adds = num_adds_mult * n;
        int ne_i[1024], ne_j[1024]; int ne_c = 0;
        for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++)
            if (!adj[i * n + j] && ne_c < 1024) { ne_i[ne_c] = i; ne_j[ne_c] = j; ne_c++; }
        int actual = num_adds < ne_c ? num_adds : ne_c;
        for (int i = 0; i < actual; i++) { int k = i + rng_int(&trng, ne_c - i);
            int ti = ne_i[i]; ne_i[i] = ne_i[k]; ne_i[k] = ti;
            int tj = ne_j[i]; ne_j[i] = ne_j[k]; ne_j[k] = tj; }
        for (int i = 0; i < actual; i++) { adj[ne_i[i]*n+ne_j[i]] = adj[ne_j[i]*n+ne_i[i]] = 1; deg[ne_i[i]]++; deg[ne_j[i]]++; }

        memcpy(adj_saved, adj, (size_t)nn); memcpy(deg_saved, deg, (size_t)n * sizeof(int));
        int cur_m = 0; for (int i = 0; i < n; i++) cur_m += deg[i]; cur_m /= 2;
        int to_rem = cur_m - m;

        /* Forward: sample actions, collect rewards */
        int num_steps = 0;
        for (int step = 0; step < to_rem; step++) {
            for (int i = 0; i < n; i++) node_embed(&w, (float)deg[i] / nm1, h + i * H);
            for (int i = 0; i < n; i++) ns[i] = mlp_fwd(h + i * H, H, w.ns_w0, w.ns_b0, w.ns_w1, w.ns_b1);
            cur_m = 0; for (int i = 0; i < n; i++) cur_m += deg[i]; cur_m /= 2;
            if (cur_m <= 2 * n) find_bridges(adj, n, bmask); else memset(bmask, 0, (size_t)nn);

            int vn[64]; int nv = 0; float nsm = -FLT_MAX;
            for (int i = 0; i < n; i++) { int has = 0;
                for (int j = 0; j < n; j++) if (i != j && adj[i*n+j] && !bmask[(i<j?i:j)*n+(i<j?j:i)]) { has = 1; break; }
                if (has) { vn[nv++] = i; if (ns[i] > nsm) nsm = ns[i]; } }
            if (nv == 0) break;

            double nsum = 0; for (int v = 0; v < nv; v++) nsum += exp((double)(ns[vn[v]] - nsm));
            double nr = rng_double(&trng) * nsum, ncsum = 0; int svi = nv - 1;
            for (int v = 0; v < nv; v++) { ncsum += exp((double)(ns[vn[v]] - nsm)); if (nr <= ncsum) { svi = v; break; } }
            int ni = vn[svi];

            int nb[64]; int nnb = 0; float ev[64], ef[2*H], esm = -FLT_MAX;
            memcpy(ef, h + ni * H, (size_t)H * sizeof(float));
            for (int j = 0; j < n; j++) { if (j == ni || !adj[ni*n+j]) continue;
                int a = ni<j?ni:j, b = ni<j?j:ni; if (bmask[a*n+b]) continue;
                memcpy(ef + H, h + j * H, (size_t)H * sizeof(float));
                ev[nnb] = mlp_fwd(ef, 2*H, w.es_w0, w.es_b0, w.es_w1, w.es_b1);
                nb[nnb] = j; if (ev[nnb] > esm) esm = ev[nnb]; nnb++; }
            if (nnb == 0) break;

            double esum = 0; for (int e = 0; e < nnb; e++) esum += exp((double)(ev[e] - esm));
            double er = rng_double(&trng) * esum, ecsum = 0; int sei = nnb - 1;
            for (int e = 0; e < nnb; e++) { ecsum += exp((double)(ev[e] - esm)); if (er <= ecsum) { sei = e; break; } }
            int ej = nb[sei];

            s_node[num_steps] = ni; s_nbr[num_steps] = ej;
            double l2b = exact_lambda2(adj, n);
            adj[ni*n+ej] = adj[ej*n+ni] = 0; deg[ni]--; deg[ej]--;
            rewards[num_steps] = exact_lambda2(adj, n) - l2b;
            num_steps++;
        }
        if (num_steps == 0) continue;

        /* Returns + mean baseline */
        double *rets = (double *)malloc((size_t)num_steps * sizeof(double));
        double G = 0; for (int t = num_steps - 1; t >= 0; t--) { G = rewards[t] + 0.99 * G; rets[t] = G; }
        double mr = 0; for (int t = 0; t < num_steps; t++) mr += rets[t]; mr /= num_steps;

        /* Backward: replay from saved state */
        PolicyGrads grad; memset(&grad, 0, sizeof(grad));
        memcpy(adj, adj_saved, (size_t)nn); memcpy(deg, deg_saved, (size_t)n * sizeof(int));

        for (int step = 0; step < num_steps; step++) {
            double adv = rets[step] - mr;
            int ni2 = s_node[step], ej2 = s_nbr[step];

            NECache nec[64]; M2Cache nsc[64];
            for (int i = 0; i < n; i++) node_embed_cached(&w, (float)deg[i] / nm1, h + i * H, &nec[i]);
            for (int i = 0; i < n; i++) ns[i] = mlp_fwd_cached(h + i * H, H, w.ns_w0, w.ns_b0, w.ns_w1, w.ns_b1, &nsc[i]);

            cur_m = 0; for (int i = 0; i < n; i++) cur_m += deg[i]; cur_m /= 2;
            if (cur_m <= 2 * n) find_bridges(adj, n, bmask); else memset(bmask, 0, (size_t)nn);

            int vn2[64]; int nv2 = 0; float nsm2 = -FLT_MAX;
            for (int i = 0; i < n; i++) { int has = 0;
                for (int j = 0; j < n; j++) if (i != j && adj[i*n+j] && !bmask[(i<j?i:j)*n+(i<j?j:i)]) { has = 1; break; }
                if (has) { vn2[nv2++] = i; if (ns[i] > nsm2) nsm2 = ns[i]; } }

            double nsum2 = 0; for (int v = 0; v < nv2; v++) nsum2 += exp((double)(ns[vn2[v]] - nsm2));
            float d_h[64 * H]; memset(d_h, 0, (size_t)n * H * sizeof(float));

            for (int v = 0; v < nv2; v++) {
                int idx = vn2[v]; double sm = exp((double)(ns[idx] - nsm2)) / nsum2;
                double d_score = -adv * ((idx == ni2 ? 1.0 : 0.0) - sm);
                float d_in[H]; memset(d_in, 0, sizeof(d_in));
                mlp_bwd(&nsc[idx], w.ns_w0, w.ns_w1, H, (float)d_score, grad.ns_w0, grad.ns_b0, grad.ns_w1, grad.ns_b1, d_in);
                for (int k = 0; k < H; k++) d_h[idx * H + k] += d_in[k];
            }

            int nb2[64]; int nnb2 = 0; M2Cache esc[64]; float ev2[64];
            float ef2[2*H]; memcpy(ef2, h + ni2 * H, (size_t)H * sizeof(float)); float esm2 = -FLT_MAX;
            for (int j = 0; j < n; j++) { if (j == ni2 || !adj[ni2*n+j]) continue;
                int a = ni2<j?ni2:j, b = ni2<j?j:ni2; if (bmask[a*n+b]) continue;
                memcpy(ef2 + H, h + j * H, (size_t)H * sizeof(float));
                ev2[nnb2] = mlp_fwd_cached(ef2, 2*H, w.es_w0, w.es_b0, w.es_w1, w.es_b1, &esc[nnb2]);
                nb2[nnb2] = j; if (ev2[nnb2] > esm2) esm2 = ev2[nnb2]; nnb2++; }

            if (nnb2 > 0) {
                double esum2 = 0; for (int e = 0; e < nnb2; e++) esum2 += exp((double)(ev2[e] - esm2));
                for (int e = 0; e < nnb2; e++) {
                    double sm = exp((double)(ev2[e] - esm2)) / esum2;
                    double d_score = -adv * ((nb2[e] == ej2 ? 1.0 : 0.0) - sm);
                    float d_ef[2*H]; memset(d_ef, 0, sizeof(d_ef));
                    mlp_bwd(&esc[e], w.es_w0, w.es_w1, H, (float)d_score, grad.es_w0, grad.es_b0, grad.es_w1, grad.es_b1, d_ef);
                    for (int k = 0; k < H; k++) { d_h[ni2 * H + k] += d_ef[k]; d_h[nb2[e] * H + k] += d_ef[H + k]; }
                }
            }

            for (int i = 0; i < n; i++) {
                float dh1[H]; for (int k = 0; k < H; k++) dh1[k] = d_h[i*H+k] * silu_d(nec[i].pre1[k]);
                for (int k = 0; k < H; k++) { grad.ne_b1[k] += dh1[k]; for (int j = 0; j < H; j++) grad.ne_w1[k*H+j] += dh1[k] * nec[i].h0[j]; }
                float dh0[H]; memset(dh0, 0, sizeof(dh0));
                for (int k = 0; k < H; k++) for (int j = 0; j < H; j++) dh0[j] += dh1[k] * w.ne_w1[k*H+j];
                for (int k = 0; k < H; k++) { float dp = dh0[k] * silu_d(nec[i].pre0[k]); grad.ne_b0[k] += dp; grad.ne_w0[k] += dp * nec[i].in_val; }
            }

            adj[ni2*n+ej2] = adj[ej2*n+ni2] = 0; deg[ni2]--; deg[ej2]--;
        }
        free(rets);

        /* Grad scale + clip + Adam */
        float *gp = (float *)&grad; float gnorm = 0;
        for (int i = 0; i < np; i++) gnorm += (double)gp[i] * gp[i]; gnorm = sqrtf(gnorm);
        float scale = 1.0f / (float)num_steps;
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

        if ((ep + 1) % 100 == 0) {
            double elapsed = wall_time() - t0;
            printf("ep %5d/%d | l2=%.3f | n=%d m=%d steps=%d | %.1f ep/s | %.1fm\n",
                   ep + 1, num_episodes, exact_lambda2(adj, n), n, m, num_steps,
                   (ep + 1) / fmax(elapsed, 0.001), elapsed / 60.0);
        }
        if ((ep + 1) % 1000 == 0) { char p[512]; snprintf(p, sizeof(p), "%s/ckpt_%d.bin", save_dir, ep + 1); save_weights(p, &w); }
    }
    char p[512]; snprintf(p, sizeof(p), "%s/final.bin", save_dir); save_weights(p, &w);
    printf("Saved to %s\n", save_dir);
    free(adam_m); free(adam_v); free(h); free(ns); free(adj); free(adj_saved);
    free(deg); free(deg_saved); free(bmask); free(rewards); free(s_node); free(s_nbr);
}

/* ========================================================================== */
/* Baseline CSV                                                                */
/* ========================================================================== */

typedef struct { int n, m; double fv, er, sw025, sw050, sw075, best; } BL;
static BL *bls = NULL; static int nbl = 0;
static void load_bl(const char *path) {
    FILE *f = fopen(path, "r"); if (!f) return; char line[1024]; fgets(line, sizeof(line), f);
    int cap = 8192; bls = (BL *)malloc((size_t)cap * sizeof(BL));
    while (fgets(line, sizeof(line), f)) { BL e = {0};
        sscanf(line, "%d,%d,%lf,%lf,%lf,%lf,%lf", &e.n, &e.m, &e.fv, &e.er, &e.sw025, &e.sw050, &e.sw075);
        e.best = e.fv; if (e.er > e.best) e.best = e.er; if (e.sw025 > e.best) e.best = e.sw025;
        if (e.sw050 > e.best) e.best = e.sw050; if (e.sw075 > e.best) e.best = e.sw075;
        if (nbl >= cap) { cap *= 2; bls = realloc(bls, (size_t)cap * sizeof(BL)); } bls[nbl++] = e; }
    fclose(f); printf("Loaded %d baselines\n", nbl); }
static BL *find_bl(int n, int m) { for (int i = 0; i < nbl; i++) if (bls[i].n == n && bls[i].m == m) return &bls[i]; return NULL; }

/* ========================================================================== */
/* Main                                                                        */
/* ========================================================================== */

int main(int argc, char **argv) {
    int n_values[64], num_n = 0; char *ckpt = NULL; char *bl_path = NULL;
    double epoch_mult = 4.0; int add_mult = 2; int do_train = 0; int train_ep = 5000;
    int min_n = 8, max_n = 32; double train_lr = 1e-3;
    char save_dir[256] = "logs/mlp_train"; uint64_t seed = 42;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) { char *tok = strtok(argv[++i], ",");
            while (tok && num_n < 64) { n_values[num_n++] = atoi(tok); tok = strtok(NULL, ","); } }
        else if (strcmp(argv[i], "--checkpoint") == 0 && i + 1 < argc) ckpt = argv[++i];
        else if (strcmp(argv[i], "--baselines") == 0 && i + 1 < argc) bl_path = argv[++i];
        else if (strcmp(argv[i], "--epoch-mult") == 0 && i + 1 < argc) epoch_mult = atof(argv[++i]);
        else if (strcmp(argv[i], "--add-mult") == 0 && i + 1 < argc) add_mult = atoi(argv[++i]);
        else if (strcmp(argv[i], "--train") == 0) do_train = 1;
        else if (strcmp(argv[i], "--episodes") == 0 && i + 1 < argc) train_ep = atoi(argv[++i]);
        else if (strcmp(argv[i], "--min-n") == 0 && i + 1 < argc) min_n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-n") == 0 && i + 1 < argc) max_n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) train_lr = atof(argv[++i]);
        else if (strcmp(argv[i], "--save-dir") == 0 && i + 1 < argc) snprintf(save_dir, sizeof(save_dir), "%s", argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint64_t)atol(argv[++i]);
    }

    if (do_train) { mlp_train(min_n, max_n, train_ep, add_mult, train_lr, seed, save_dir); return 0; }

    if (!ckpt || num_n == 0) {
        printf("Usage: %s --train --min-n 8 --max-n 32 --episodes 5000\n", argv[0]);
        printf("   or: %s --checkpoint model.bin --n 8,16,24,32 --baselines bl.csv\n", argv[0]);
        return 1;
    }

    MLPWeights w;
    if (load_weights(ckpt, &w) != 0) { fprintf(stderr, "Failed to load: %s\n", ckpt); return 1; }
    printf("Loaded: %s\n", ckpt);
    if (bl_path) load_bl(bl_path);

    printf("MLP Eval | K=%.0f*N, adds=%d*N, tabu=N/2, threads=%d\n", epoch_mult, add_mult, omp_get_max_threads());
    printf("================================================================\n");

    for (int ni = 0; ni < num_n; ni++) {
        int n = n_values[ni]; int mx = n * (n - 1) / 2;
        int K = (int)(epoch_mult * n); int na = add_mult * n;
        int *m_list = (int *)malloc((size_t)(mx + 1) * sizeof(int)); int nm = 0;
        for (int mv = n + 1; mv <= mx; mv++) { BL *b = find_bl(n, mv); if (b && b->best > 0) m_list[nm++] = mv; }
        if (nm == 0) { free(m_list); continue; }

        printf("\nn=%d (%d configs, K=%d, adds=%d)\n", n, nm, K, na);
        double t0_eval = wall_time();
        double *jl = (double *)malloc((size_t)nm * sizeof(double));

        #pragma omp parallel for schedule(dynamic, 1)
        for (int mi = 0; mi < nm; mi++) {
            uint64_t s = (uint64_t)(n * 100 + m_list[mi]);
            jl[mi] = mlp_episode(&w, n, m_list[mi], K, na, s);
        }

        double sr = 0, sf = 0, sb = 0; int wf = 0, wb = 0, tot = 0;
        double bin_rl[10] = {0}, bin_fv[10] = {0}; int bin_cnt[10] = {0};
        for (int mi = 0; mi < nm; mi++) { BL *b = find_bl(n, m_list[mi]); if (!b) continue;
            sr += jl[mi]; sf += b->fv; sb += b->best;
            if (jl[mi] > b->fv + 1e-6) wf++; if (jl[mi] > b->best + 1e-6) wb++; tot++;
            double density = (double)m_list[mi] / (double)mx;
            int bin = (int)(density * 10); if (bin >= 10) bin = 9;
            bin_rl[bin] += jl[mi]; bin_fv[bin] += b->fv; bin_cnt[bin]++; }

        double elapsed = wall_time() - t0_eval;
        printf("  vs FV: %.1f%% (%dW/%d) | vs Best: %.1f%% (%dW/%d) | %.1fs\n",
               sr / sf * 100, wf, tot, sr / sb * 100, wb, tot, elapsed);
        printf("  Density: ");
        for (int b = 0; b < 10; b++) { if (bin_cnt[b] == 0) continue;
            printf("%.0f%%:%.1f%% ", ((double)b + 0.5) * 10, bin_rl[b] / bin_fv[b] * 100); }
        printf("\n");
        free(m_list); free(jl);
    }
    free(bls); return 0;
}
