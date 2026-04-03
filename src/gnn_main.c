/*
 * gnn_main.c — GNN RL inference for G(n,m) algebraic connectivity.
 *
 * Multi-epoch: K=c*N epochs, each adds c2*N non-edges then
 * autoregressively removes with hierarchical GNN (node→edge).
 *
 * GNN: 2-layer GCN + node scorer + edge scorer. Degree-only features.
 * O(N³) inference: K=O(N) epochs × O(N) adds/removes × O(N) per step.
 *
 * Usage:
 *   ./crl_gnn --n 8,16,24 --checkpoint logs/gnn_policy.bin --baselines baselines.csv
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

#define GNN_HIDDEN 64
#define GNN_MAGIC 0x474E4E31

static double wall_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* ========================================================================== */
/* GNN weights                                                                 */
/* ========================================================================== */

typedef struct {
    /* GCN layer 1: (1) → (64) */
    float conv1_w[GNN_HIDDEN * 1];    /* (64, 1) */
    float conv1_b[GNN_HIDDEN];

    /* GCN layer 2: (64) → (64) */
    float conv2_w[GNN_HIDDEN * GNN_HIDDEN]; /* (64, 64) */
    float conv2_b[GNN_HIDDEN];

    /* Node scorer: (64) → (64) → (1) */
    float ns_w0[GNN_HIDDEN * GNN_HIDDEN];
    float ns_b0[GNN_HIDDEN];
    float ns_w1[1 * GNN_HIDDEN];
    float ns_b1[1];

    /* Edge scorer: (128) → (64) → (1) */
    float es_w0[GNN_HIDDEN * (2 * GNN_HIDDEN)];
    float es_b0[GNN_HIDDEN];
    float es_w1[1 * GNN_HIDDEN];
    float es_b1[1];

    /* Value head: (64) → (64) → (1) */
    float vh_w0[GNN_HIDDEN * GNN_HIDDEN];
    float vh_b0[GNN_HIDDEN];
    float vh_w1[1 * GNN_HIDDEN];
    float vh_b1[1];
} GNNWeights;

static int load_gnn_weights(const char *path, GNNWeights *gw) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    uint32_t magic, num_tensors;
    fread(&magic, 4, 1, f);
    if (magic != GNN_MAGIC) { fclose(f); return -2; }
    fread(&num_tensors, 4, 1, f);

    /* Read tensors by name */
    for (uint32_t t = 0; t < num_tensors; t++) {
        uint32_t name_len;
        fread(&name_len, 4, 1, f);
        char name[256];
        fread(name, 1, name_len, f);
        name[name_len] = '\0';

        uint32_t ndim;
        fread(&ndim, 4, 1, f);
        uint32_t total = 1;
        for (uint32_t d = 0; d < ndim; d++) {
            uint32_t dim;
            fread(&dim, 4, 1, f);
            total *= dim;
        }

        float *dst = NULL;
        if (strcmp(name, "conv1.lin.weight") == 0) dst = gw->conv1_w;
        else if (strcmp(name, "conv1.bias") == 0) dst = gw->conv1_b;
        else if (strcmp(name, "conv2.lin.weight") == 0) dst = gw->conv2_w;
        else if (strcmp(name, "conv2.bias") == 0) dst = gw->conv2_b;
        else if (strcmp(name, "node_scorer.0.weight") == 0) dst = gw->ns_w0;
        else if (strcmp(name, "node_scorer.0.bias") == 0) dst = gw->ns_b0;
        else if (strcmp(name, "node_scorer.2.weight") == 0) dst = gw->ns_w1;
        else if (strcmp(name, "node_scorer.2.bias") == 0) dst = gw->ns_b1;
        else if (strcmp(name, "edge_scorer.0.weight") == 0) dst = gw->es_w0;
        else if (strcmp(name, "edge_scorer.0.bias") == 0) dst = gw->es_b0;
        else if (strcmp(name, "edge_scorer.2.weight") == 0) dst = gw->es_w1;
        else if (strcmp(name, "edge_scorer.2.bias") == 0) dst = gw->es_b1;
        else if (strcmp(name, "value_head.0.weight") == 0) dst = gw->vh_w0;
        else if (strcmp(name, "value_head.0.bias") == 0) dst = gw->vh_b0;
        else if (strcmp(name, "value_head.2.weight") == 0) dst = gw->vh_w1;
        else if (strcmp(name, "value_head.2.bias") == 0) dst = gw->vh_b1;

        if (dst) {
            fread(dst, sizeof(float), total, f);
        } else {
            fseek(f, (long)(total * sizeof(float)), SEEK_CUR);
        }
    }

    fclose(f);
    return 0;
}

/* ========================================================================== */
/* GNN forward                                                                 */
/* ========================================================================== */

static inline float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

static inline float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float silu_deriv_f(float x) {
    float s = sigmoid_f(x);
    return s * (1.0f + x * (1.0f - s));
}

/* ========================================================================== */
/* GNN gradients                                                               */
/* ========================================================================== */

typedef struct {
    float conv1_w[GNN_HIDDEN * 1];
    float conv1_b[GNN_HIDDEN];
    float conv2_w[GNN_HIDDEN * GNN_HIDDEN];
    float conv2_b[GNN_HIDDEN];
    float ns_w0[GNN_HIDDEN * GNN_HIDDEN];
    float ns_b0[GNN_HIDDEN];
    float ns_w1[1 * GNN_HIDDEN];
    float ns_b1[1];
    float es_w0[GNN_HIDDEN * (2 * GNN_HIDDEN)];
    float es_b0[GNN_HIDDEN];
    float es_w1[1 * GNN_HIDDEN];
    float es_b1[1];
    float vh_w0[GNN_HIDDEN * GNN_HIDDEN];
    float vh_b0[GNN_HIDDEN];
    float vh_w1[1 * GNN_HIDDEN];
    float vh_b1[1];
} GNNGrad;

/* Adam state for GNN */
#define GNN_TOTAL_PARAMS (GNN_HIDDEN*1 + GNN_HIDDEN + GNN_HIDDEN*GNN_HIDDEN + GNN_HIDDEN + \
    GNN_HIDDEN*GNN_HIDDEN + GNN_HIDDEN + GNN_HIDDEN + 1 + \
    GNN_HIDDEN*2*GNN_HIDDEN + GNN_HIDDEN + GNN_HIDDEN + 1)

typedef struct {
    float m[GNN_TOTAL_PARAMS];
    float v[GNN_TOTAL_PARAMS];
    int t;
} GNNAdam;

static void gnn_grad_zero(GNNGrad *g) { memset(g, 0, sizeof(GNNGrad)); }

static void gnn_grad_add(GNNGrad *dst, const GNNGrad *src) {
    float *d = (float *)dst; const float *s = (const float *)src;
    for (int i = 0; i < (int)(sizeof(GNNGrad)/sizeof(float)); i++) d[i] += s[i];
}

static void gnn_grad_scale(GNNGrad *g, float s) {
    float *p = (float *)g;
    for (int i = 0; i < (int)(sizeof(GNNGrad)/sizeof(float)); i++) p[i] *= s;
}

static float gnn_grad_norm(const GNNGrad *g) {
    const float *p = (const float *)g;
    double sum = 0;
    for (int i = 0; i < (int)(sizeof(GNNGrad)/sizeof(float)); i++) sum += (double)p[i]*p[i];
    return (float)sqrt(sum);
}

static void gnn_grad_clip(GNNGrad *g, float max_norm) {
    float norm = gnn_grad_norm(g);
    if (norm > max_norm) gnn_grad_scale(g, max_norm / norm);
}

static void gnn_adam_init(GNNAdam *a) { memset(a, 0, sizeof(GNNAdam)); }

static void gnn_adam_step(GNNWeights *w, const GNNGrad *g, GNNAdam *a,
                           double lr, double wd) {
    a->t++;
    float *wp = (float *)w;
    const float *gp = (const float *)g;
    double bc1 = 1.0 - pow(0.9, a->t);
    double bc2 = 1.0 - pow(0.999, a->t);
    int np = (int)(sizeof(GNNGrad)/sizeof(float));
    for (int i = 0; i < np; i++) {
        a->m[i] = (float)(0.9*a->m[i] + 0.1*gp[i]);
        a->v[i] = (float)(0.999*a->v[i] + 0.001*gp[i]*gp[i]);
        float mh = (float)(a->m[i]/bc1);
        float vh = (float)(a->v[i]/bc2);
        wp[i] -= (float)(lr * (mh/(sqrtf(vh)+1e-8f) + wd*wp[i]));
    }
}

/* ========================================================================== */
/* GCN with cached activations (for backward)                                  */
/* ========================================================================== */

typedef struct {
    float *d_hat;   /* (n,) degree+1 */
    float *agg;     /* (n, in_dim) aggregated features */
    float *pre_act; /* (n, out_dim) W@agg+b before SiLU */
    int n, in_dim, out_dim;
} GCNCache;

static void gcn_layer_cached(const uint8_t *adj, int n,
                              const float *h_in, int in_dim,
                              const float *W, const float *b, int out_dim,
                              float *h_out, GCNCache *cache) {
    cache->n = n; cache->in_dim = in_dim; cache->out_dim = out_dim;
    cache->d_hat = (float *)malloc((size_t)n * sizeof(float));
    cache->agg = (float *)calloc((size_t)n * in_dim, sizeof(float));
    cache->pre_act = (float *)malloc((size_t)n * out_dim * sizeof(float));

    for (int i = 0; i < n; i++) {
        float deg = 1.0f;
        for (int j = 0; j < n; j++) deg += adj[i*n+j];
        cache->d_hat[i] = deg;
    }

    for (int i = 0; i < n; i++) {
        float di = cache->d_hat[i];
        float norm = 1.0f / sqrtf(di*di);
        for (int k = 0; k < in_dim; k++)
            cache->agg[i*in_dim+k] += norm * h_in[i*in_dim+k];
        for (int j = 0; j < n; j++) {
            if (adj[i*n+j]) {
                norm = 1.0f / sqrtf(di * cache->d_hat[j]);
                for (int k = 0; k < in_dim; k++)
                    cache->agg[i*in_dim+k] += norm * h_in[j*in_dim+k];
            }
        }
    }

    for (int i = 0; i < n; i++) {
        for (int o = 0; o < out_dim; o++) {
            float sum = b[o];
            for (int k = 0; k < in_dim; k++)
                sum += W[o*in_dim+k] * cache->agg[i*in_dim+k];
            cache->pre_act[i*out_dim+o] = sum;
            h_out[i*out_dim+o] = silu_f(sum);
        }
    }
}

/*
 * GCN backward: given d_h_out (n, out_dim), compute dW, db, and d_h_in.
 */
static void gcn_layer_backward(const uint8_t *adj, const float *h_in,
                                const GCNCache *cache,
                                const float *W, const float *d_h_out,
                                float *dW, float *db, float *d_h_in) {
    int n = cache->n, in_dim = cache->in_dim, out_dim = cache->out_dim;

    /* d_pre_act = d_h_out * silu_deriv(pre_act) */
    float *d_pre = (float *)malloc((size_t)n * out_dim * sizeof(float));
    for (int i = 0; i < n * out_dim; i++)
        d_pre[i] = d_h_out[i] * silu_deriv_f(cache->pre_act[i]);

    /* dW += d_pre[i] ⊗ agg[i], db += d_pre[i] */
    for (int i = 0; i < n; i++) {
        for (int o = 0; o < out_dim; o++) {
            db[o] += d_pre[i*out_dim+o];
            for (int k = 0; k < in_dim; k++)
                dW[o*in_dim+k] += d_pre[i*out_dim+o] * cache->agg[i*in_dim+k];
        }
    }

    /* d_agg[i] = W^T @ d_pre[i] */
    float *d_agg = (float *)calloc((size_t)n * in_dim, sizeof(float));
    for (int i = 0; i < n; i++)
        for (int k = 0; k < in_dim; k++)
            for (int o = 0; o < out_dim; o++)
                d_agg[i*in_dim+k] += W[o*in_dim+k] * d_pre[i*out_dim+o];

    /* d_h_in: scatter d_agg back through aggregation */
    if (d_h_in) {
        memset(d_h_in, 0, (size_t)n * in_dim * sizeof(float));
        for (int i = 0; i < n; i++) {
            float di = cache->d_hat[i];
            /* Self-loop contribution */
            float norm = 1.0f / sqrtf(di*di);
            for (int k = 0; k < in_dim; k++)
                d_h_in[i*in_dim+k] += norm * d_agg[i*in_dim+k];
            /* Neighbor contributions: d_agg[i] flows to h_in[j] */
            for (int j = 0; j < n; j++) {
                if (adj[i*n+j]) {
                    norm = 1.0f / sqrtf(di * cache->d_hat[j]);
                    for (int k = 0; k < in_dim; k++)
                        d_h_in[j*in_dim+k] += norm * d_agg[i*in_dim+k];
                }
            }
        }
    }

    free(d_pre);
    free(d_agg);
}

static void gcn_cache_free(GCNCache *c) {
    free(c->d_hat); free(c->agg); free(c->pre_act);
}

/* MLP2 forward with cache */
typedef struct {
    float hidden[GNN_HIDDEN];
    float pre_act[GNN_HIDDEN]; /* before SiLU */
    float input[2 * GNN_HIDDEN];
    float out;
    int in_dim;
} MLP2Cache;

static float mlp2_forward_cached(const float *input, int in_dim,
                                  const float *W0, const float *b0, int h_dim,
                                  const float *W1, const float *b1,
                                  MLP2Cache *cache) {
    cache->in_dim = in_dim;
    memcpy(cache->input, input, (size_t)in_dim * sizeof(float));
    for (int i = 0; i < h_dim; i++) {
        float sum = b0[i];
        for (int j = 0; j < in_dim; j++) sum += W0[i*in_dim+j]*input[j];
        cache->pre_act[i] = sum;
        cache->hidden[i] = silu_f(sum);
    }
    float out = b1[0];
    for (int j = 0; j < h_dim; j++) out += W1[j]*cache->hidden[j];
    cache->out = out;
    return out;
}

/* MLP2 backward: d_out → dW0, db0, dW1, db1, d_input */
static void mlp2_backward(const MLP2Cache *cache,
                           const float *W0, const float *W1, int h_dim,
                           float d_out,
                           float *dW0, float *db0, float *dW1, float *db1,
                           float *d_input) {
    int in_dim = cache->in_dim;
    /* Layer 1 grad */
    db1[0] += d_out;
    float dh[GNN_HIDDEN];
    for (int j = 0; j < h_dim; j++) {
        dW1[j] += d_out * cache->hidden[j];
        dh[j] = d_out * W1[j];
    }
    /* Layer 0 grad */
    for (int i = 0; i < h_dim; i++) {
        float dp = dh[i] * silu_deriv_f(cache->pre_act[i]);
        db0[i] += dp;
        for (int j = 0; j < in_dim; j++) {
            dW0[i*in_dim+j] += dp * cache->input[j];
            if (d_input) d_input[j] += dp * W0[i*in_dim+j];
        }
    }
}

/* ========================================================================== */
/* Save GNN weights (same format as Python export)                             */
/* ========================================================================== */

static void save_gnn_weights(const char *path, const GNNWeights *gw) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    uint32_t magic = GNN_MAGIC;
    fwrite(&magic, 4, 1, f);

    /* Write each tensor with name */
    struct { const char *name; const float *data; int dims[2]; int ndim; } tensors[] = {
        {"conv1.bias", gw->conv1_b, {GNN_HIDDEN, 0}, 1},
        {"conv1.lin.weight", gw->conv1_w, {GNN_HIDDEN, 1}, 2},
        {"conv2.bias", gw->conv2_b, {GNN_HIDDEN, 0}, 1},
        {"conv2.lin.weight", gw->conv2_w, {GNN_HIDDEN, GNN_HIDDEN}, 2},
        {"node_scorer.0.weight", gw->ns_w0, {GNN_HIDDEN, GNN_HIDDEN}, 2},
        {"node_scorer.0.bias", gw->ns_b0, {GNN_HIDDEN, 0}, 1},
        {"node_scorer.2.weight", gw->ns_w1, {1, GNN_HIDDEN}, 2},
        {"node_scorer.2.bias", gw->ns_b1, {1, 0}, 1},
        {"edge_scorer.0.weight", gw->es_w0, {GNN_HIDDEN, 2*GNN_HIDDEN}, 2},
        {"edge_scorer.0.bias", gw->es_b0, {GNN_HIDDEN, 0}, 1},
        {"edge_scorer.2.weight", gw->es_w1, {1, GNN_HIDDEN}, 2},
        {"edge_scorer.2.bias", gw->es_b1, {1, 0}, 1},
        {"value_head.0.weight", gw->vh_w0, {GNN_HIDDEN, GNN_HIDDEN}, 2},
        {"value_head.0.bias", gw->vh_b0, {GNN_HIDDEN, 0}, 1},
        {"value_head.2.weight", gw->vh_w1, {1, GNN_HIDDEN}, 2},
        {"value_head.2.bias", gw->vh_b1, {1, 0}, 1},
    };
    uint32_t nt = 16;
    fwrite(&nt, 4, 1, f);

    for (int t = 0; t < 12; t++) {
        uint32_t nlen = (uint32_t)strlen(tensors[t].name);
        fwrite(&nlen, 4, 1, f);
        fwrite(tensors[t].name, 1, nlen, f);
        uint32_t ndim = (uint32_t)tensors[t].ndim;
        fwrite(&ndim, 4, 1, f);
        uint32_t total = 1;
        for (int d = 0; d < tensors[t].ndim; d++) {
            uint32_t dim = (uint32_t)tensors[t].dims[d];
            fwrite(&dim, 4, 1, f);
            total *= dim;
        }
        fwrite(tensors[t].data, sizeof(float), total, f);
    }
    fclose(f);
}

/* ========================================================================== */
/* Random init for GNN weights                                                 */
/* ========================================================================== */

static void gnn_init_random(GNNWeights *gw, uint64_t seed) {
    RNG rng; rng_init(&rng, seed);
    /* Xavier/Glorot init: scale = sqrt(2 / (fan_in + fan_out)) */
    float *p = (float *)gw;
    int np = (int)(sizeof(GNNWeights)/sizeof(float));
    /* Fill with normal(0, 0.1) as baseline */
    for (int i = 0; i < np; i += 2) {
        double u1 = rng_double(&rng) * 0.998 + 0.001;
        double u2 = rng_double(&rng);
        double r = sqrt(-2.0 * log(u1));
        p[i] = (float)(r * cos(2.0 * M_PI * u2));
        if (i + 1 < np) p[i+1] = (float)(r * sin(2.0 * M_PI * u2));
    }
    /* Scale each weight matrix by Xavier factor */
    /* conv1: (64, 1) → scale = sqrt(2/(1+64)) */
    float s = sqrtf(2.0f / (1 + GNN_HIDDEN));
    for (int i = 0; i < GNN_HIDDEN*1; i++) gw->conv1_w[i] *= s;
    memset(gw->conv1_b, 0, sizeof(gw->conv1_b));
    /* conv2: (64, 64) → scale = sqrt(2/(64+64)) */
    s = sqrtf(2.0f / (GNN_HIDDEN + GNN_HIDDEN));
    for (int i = 0; i < GNN_HIDDEN*GNN_HIDDEN; i++) gw->conv2_w[i] *= s;
    memset(gw->conv2_b, 0, sizeof(gw->conv2_b));
    /* node_scorer: (64,64)→(1,64) */
    s = sqrtf(2.0f / (GNN_HIDDEN + GNN_HIDDEN));
    for (int i = 0; i < GNN_HIDDEN*GNN_HIDDEN; i++) gw->ns_w0[i] *= s;
    memset(gw->ns_b0, 0, sizeof(gw->ns_b0));
    s = sqrtf(2.0f / (GNN_HIDDEN + 1));
    for (int i = 0; i < GNN_HIDDEN; i++) gw->ns_w1[i] *= s;
    memset(gw->ns_b1, 0, sizeof(gw->ns_b1));
    /* edge_scorer: (64,128)→(1,64) */
    s = sqrtf(2.0f / (2*GNN_HIDDEN + GNN_HIDDEN));
    for (int i = 0; i < GNN_HIDDEN*2*GNN_HIDDEN; i++) gw->es_w0[i] *= s;
    memset(gw->es_b0, 0, sizeof(gw->es_b0));
    s = sqrtf(2.0f / (GNN_HIDDEN + 1));
    for (int i = 0; i < GNN_HIDDEN; i++) gw->es_w1[i] *= s;
    memset(gw->es_b1, 0, sizeof(gw->es_b1));
    /* value_head: (64,64)→(1,64) */
    s = sqrtf(2.0f / (GNN_HIDDEN + GNN_HIDDEN));
    for (int i = 0; i < GNN_HIDDEN*GNN_HIDDEN; i++) gw->vh_w0[i] *= s;
    memset(gw->vh_b0, 0, sizeof(gw->vh_b0));
    s = sqrtf(2.0f / (GNN_HIDDEN + 1));
    for (int i = 0; i < GNN_HIDDEN; i++) gw->vh_w1[i] *= s;
    memset(gw->vh_b1, 0, sizeof(gw->vh_b1));
}

/*
 * GCN message passing: h_out[i] = SiLU(W @ agg[i] + b)
 * where agg[i] = sum_{j in N(i) + self} (1/sqrt((d_i+1)(d_j+1))) * h_in[j]
 *
 * adj: (n, n) uint8. h_in: (n, in_dim). h_out: (n, out_dim).
 * W: (out_dim, in_dim). b: (out_dim).
 */
static void gcn_layer(const uint8_t *adj, int n,
                       const float *h_in, int in_dim,
                       const float *W, const float *b, int out_dim,
                       float *h_out) {
    /* Compute degrees + 1 (self-loop) */
    float *d_hat = (float *)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) {
        float deg = 1.0f; /* self-loop */
        for (int j = 0; j < n; j++) deg += adj[i * n + j];
        d_hat[i] = deg;
    }

    /* Aggregation: agg[i] = sum_j norm * h_in[j] */
    float *agg = (float *)calloc((size_t)n * in_dim, sizeof(float));
    for (int i = 0; i < n; i++) {
        float di = d_hat[i];
        /* Self-loop */
        float norm = 1.0f / sqrtf(di * di);
        for (int k = 0; k < in_dim; k++)
            agg[i * in_dim + k] += norm * h_in[i * in_dim + k];
        /* Neighbors */
        for (int j = 0; j < n; j++) {
            if (adj[i * n + j]) {
                norm = 1.0f / sqrtf(di * d_hat[j]);
                for (int k = 0; k < in_dim; k++)
                    agg[i * in_dim + k] += norm * h_in[j * in_dim + k];
            }
        }
    }

    /* Linear + SiLU: h_out[i] = SiLU(W @ agg[i] + b) */
    for (int i = 0; i < n; i++) {
        for (int o = 0; o < out_dim; o++) {
            float sum = b[o];
            for (int k = 0; k < in_dim; k++)
                sum += W[o * in_dim + k] * agg[i * in_dim + k];
            h_out[i * out_dim + o] = silu_f(sum);
        }
    }

    free(d_hat);
    free(agg);
}

/* MLP: 2-layer (in→hidden→1) with SiLU */
static float mlp2_forward(const float *input, int in_dim,
                           const float *W0, const float *b0, int h_dim,
                           const float *W1, const float *b1) {
    float hidden[GNN_HIDDEN];
    for (int i = 0; i < h_dim; i++) {
        float sum = b0[i];
        for (int j = 0; j < in_dim; j++)
            sum += W0[i * in_dim + j] * input[j];
        hidden[i] = silu_f(sum);
    }
    float out = b1[0];
    for (int j = 0; j < h_dim; j++)
        out += W1[j] * hidden[j];
    return out;
}

/* ========================================================================== */
/* GNN episode                                                                 */
/* ========================================================================== */

static double gnn_episode(const GNNWeights *gw, int n, int m,
                           int num_epochs, int num_adds, uint64_t seed) {
    int nn = n * n;
    uint8_t *adj = (uint8_t *)calloc((size_t)nn, 1);
    int *degrees = (int *)calloc((size_t)n, sizeof(int));
    RNG rng;
    rng_init(&rng, seed);
    build_ring_random(adj, degrees, n, m, &rng);

    double best_l2 = exact_lambda2(adj, n);

    /* GNN buffers */
    float *h0 = (float *)malloc((size_t)n * sizeof(float));       /* input: degree/(n-1) */
    float *h1 = (float *)malloc((size_t)n * GNN_HIDDEN * sizeof(float)); /* after conv1 */
    float *h2 = (float *)malloc((size_t)n * GNN_HIDDEN * sizeof(float)); /* after conv2 */
    float *node_scores = (float *)malloc((size_t)n * sizeof(float));

    int max_pairs = n * (n - 1) / 2;
    int *ne_i = (int *)malloc((size_t)max_pairs * sizeof(int));
    int *ne_j = (int *)malloc((size_t)max_pairs * sizeof(int));
    uint8_t *bridge_mask = (uint8_t *)malloc((size_t)nn);

    /* Tabu matrix: epoch until which each edge is banned from adding */
    int *tabu = (int *)calloc((size_t)nn, sizeof(int));
    int tabu_tenure = n;  /* ban for N epochs after pruning */
    /* Track which edges were added this epoch */
    uint8_t *added_this_epoch = (uint8_t *)malloc((size_t)nn);

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        memset(added_this_epoch, 0, (size_t)nn);

        /* Add num_adds random non-edges, skipping tabu */
        int ne_count = 0;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (!adj[i * n + j] && tabu[i * n + j] <= epoch) {
                    ne_i[ne_count] = i;
                    ne_j[ne_count] = j;
                    ne_count++;
                }

        int actual_adds = num_adds < ne_count ? num_adds : ne_count;
        /* Fisher-Yates partial shuffle */
        for (int i = 0; i < actual_adds && i < ne_count; i++) {
            int k = i + rng_int(&rng, ne_count - i);
            int ti = ne_i[i]; ne_i[i] = ne_i[k]; ne_i[k] = ti;
            int tj = ne_j[i]; ne_j[i] = ne_j[k]; ne_j[k] = tj;
        }
        for (int i = 0; i < actual_adds; i++) {
            int ai = ne_i[i], aj = ne_j[i];
            adj[ai * n + aj] = adj[aj * n + ai] = 1;
            degrees[ai]++;
            degrees[aj]++;
            added_this_epoch[ai * n + aj] = added_this_epoch[aj * n + ai] = 1;
        }

        int cur_m = 0;
        for (int i = 0; i < n; i++) cur_m += degrees[i];
        cur_m /= 2;
        int to_remove = cur_m - m;

        /* Autoregressive removal with hierarchical GNN */
        for (int step = 0; step < to_remove; step++) {
            /* Node features: degree/(n-1) */
            float nm1 = (float)(n - 1);
            if (nm1 < 1.0f) nm1 = 1.0f;
            for (int i = 0; i < n; i++)
                h0[i] = (float)degrees[i] / nm1;

            /* GCN layer 1: (n,1) → (n,64) */
            gcn_layer(adj, n, h0, 1, gw->conv1_w, gw->conv1_b, GNN_HIDDEN, h1);

            /* GCN layer 2: (n,64) → (n,64) */
            gcn_layer(adj, n, h1, GNN_HIDDEN, gw->conv2_w, gw->conv2_b, GNN_HIDDEN, h2);

            /* Score nodes */
            for (int i = 0; i < n; i++)
                node_scores[i] = mlp2_forward(h2 + i * GNN_HIDDEN, GNN_HIDDEN,
                                               gw->ns_w0, gw->ns_b0, GNN_HIDDEN,
                                               gw->ns_w1, gw->ns_b1);

            /* Lazy bridges */
            cur_m = 0;
            for (int i = 0; i < n; i++) cur_m += degrees[i];
            cur_m /= 2;
            if (cur_m <= 2 * n)
                find_bridges(adj, n, bridge_mask);
            else
                memset(bridge_mask, 0, (size_t)nn);

            /* Find best node (has removable edges, highest score) */
            int best_node = -1;
            float best_ns = -FLT_MAX;
            for (int i = 0; i < n; i++) {
                int has_removable = 0;
                for (int j = 0; j < n; j++) {
                    if (i != j && adj[i * n + j]) {
                        int a = i < j ? i : j, b = i < j ? j : i;
                        if (!bridge_mask[a * n + b]) {
                            has_removable = 1;
                            break;
                        }
                    }
                }
                if (has_removable && node_scores[i] > best_ns) {
                    best_ns = node_scores[i];
                    best_node = i;
                }
            }

            if (best_node < 0) break;

            /* Score edges of best_node */
            int best_nbr = -1;
            float best_es = -FLT_MAX;
            float edge_feat[2 * GNN_HIDDEN];

            /* Copy h2[best_node] to first half */
            memcpy(edge_feat, h2 + best_node * GNN_HIDDEN,
                   (size_t)GNN_HIDDEN * sizeof(float));

            for (int j = 0; j < n; j++) {
                if (j == best_node || !adj[best_node * n + j]) continue;
                int a = best_node < j ? best_node : j;
                int b = best_node < j ? j : best_node;
                if (bridge_mask[a * n + b]) continue;

                /* Second half: h2[j] */
                memcpy(edge_feat + GNN_HIDDEN, h2 + j * GNN_HIDDEN,
                       (size_t)GNN_HIDDEN * sizeof(float));

                float es = mlp2_forward(edge_feat, 2 * GNN_HIDDEN,
                                         gw->es_w0, gw->es_b0, GNN_HIDDEN,
                                         gw->es_w1, gw->es_b1);
                if (es > best_es) {
                    best_es = es;
                    best_nbr = j;
                }
            }

            if (best_nbr < 0) break;

            /* Remove edge */
            adj[best_node * n + best_nbr] = adj[best_nbr * n + best_node] = 0;
            degrees[best_node]--;
            degrees[best_nbr]--;
        }

        /* Update tabu: edges added this epoch that got pruned — ban for tenure */
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (added_this_epoch[i * n + j] && !adj[i * n + j]) {
                    tabu[i * n + j] = tabu[j * n + i] = epoch + tabu_tenure;
                }

        double l2 = exact_lambda2(adj, n);
        if (l2 > best_l2) best_l2 = l2;
    }

    free(adj); free(degrees);
    free(h0); free(h1); free(h2); free(node_scores);
    free(ne_i); free(ne_j); free(bridge_mask);
    free(tabu); free(added_this_epoch);

    return best_l2;
}

/* ========================================================================== */
/* Baseline CSV loading                                                        */
/* ========================================================================== */

typedef struct { int n, m; double fv, er, sw025, sw050, sw075, best; } BL;
static BL *baselines = NULL;
static int num_bl = 0;

static void load_bl(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return;
    char line[1024];
    fgets(line, sizeof(line), f);
    int cap = 4096;
    baselines = (BL *)malloc((size_t)cap * sizeof(BL));
    while (fgets(line, sizeof(line), f)) {
        BL e = {0};
        sscanf(line, "%d,%d,%lf,%lf,%lf,%lf,%lf",
               &e.n, &e.m, &e.fv, &e.er, &e.sw025, &e.sw050, &e.sw075);
        e.best = e.fv;
        if (e.er > e.best) e.best = e.er;
        if (e.sw025 > e.best) e.best = e.sw025;
        if (e.sw050 > e.best) e.best = e.sw050;
        if (e.sw075 > e.best) e.best = e.sw075;
        if (num_bl >= cap) { cap *= 2; baselines = realloc(baselines, (size_t)cap * sizeof(BL)); }
        baselines[num_bl++] = e;
    }
    fclose(f);
    printf("Loaded %d baselines\n", num_bl);
}

static BL *find_bl(int n, int m) {
    for (int i = 0; i < num_bl; i++)
        if (baselines[i].n == n && baselines[i].m == m) return &baselines[i];
    return NULL;
}

/* ========================================================================== */
/* Training: REINFORCE with single-epoch episodes                              */
/* ========================================================================== */

static void gnn_train(int min_n, int max_n, int num_episodes, int num_adds_mult,
                       double lr, uint64_t seed, const char *save_dir) {
    GNNWeights gw;
    gnn_init_random(&gw, seed);
    GNNAdam adam;
    gnn_adam_init(&adam);

    RNG train_rng;
    rng_init(&train_rng, seed);

    int max_n_sq = max_n * max_n;
    int max_removals = max_n * num_adds_mult + max_n;

    float *h0 = (float *)malloc((size_t)max_n * sizeof(float));
    float *h1 = (float *)malloc((size_t)max_n * GNN_HIDDEN * sizeof(float));
    float *h2 = (float *)malloc((size_t)max_n * GNN_HIDDEN * sizeof(float));
    float *node_scores = (float *)malloc((size_t)max_n * sizeof(float));
    float *d_h2 = (float *)calloc((size_t)max_n * GNN_HIDDEN, sizeof(float));
    float *d_h1 = (float *)calloc((size_t)max_n * GNN_HIDDEN, sizeof(float));
    uint8_t *adj = (uint8_t *)calloc((size_t)max_n_sq, 1);
    uint8_t *adj_saved = (uint8_t *)calloc((size_t)max_n_sq, 1); /* save post-add state */
    int *degrees = (int *)calloc((size_t)max_n, sizeof(int));
    int *degrees_saved = (int *)calloc((size_t)max_n, sizeof(int));
    uint8_t *bridge_mask = (uint8_t *)malloc((size_t)max_n_sq);

    double *rewards = (double *)malloc((size_t)max_removals * sizeof(double));
    int *step_node = (int *)malloc((size_t)max_removals * sizeof(int));
    int *step_nbr = (int *)malloc((size_t)max_removals * sizeof(int));

    mkdir(save_dir, 0755);
    double t_start = wall_time();

    printf("GNN Train | n=[%d,%d] | episodes=%d | adds=%d*N | lr=%.1e\n",
           min_n, max_n, num_episodes, num_adds_mult, lr);
    printf("================================================================\n");

    for (int ep = 0; ep < num_episodes; ep++) {
        int n = min_n + rng_int(&train_rng, max_n - min_n + 1);
        int max_m = n * (n - 1) / 2;
        int m = n + rng_int(&train_rng, max_m - n + 1);
        int nn = n * n;

        memset(adj, 0, (size_t)nn);
        memset(degrees, 0, (size_t)n * sizeof(int));
        build_ring_random(adj, degrees, n, m, &train_rng);

        int num_adds = num_adds_mult * n;
        int ne_count = 0;
        int ne_i[1024], ne_j[1024];
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (!adj[i*n+j] && ne_count < 1024) {
                    ne_i[ne_count] = i; ne_j[ne_count] = j; ne_count++;
                }
        int actual_adds = num_adds < ne_count ? num_adds : ne_count;
        for (int i = 0; i < actual_adds; i++) {
            int k = i + rng_int(&train_rng, ne_count - i);
            int ti = ne_i[i]; ne_i[i] = ne_i[k]; ne_i[k] = ti;
            int tj = ne_j[i]; ne_j[i] = ne_j[k]; ne_j[k] = tj;
        }
        for (int i = 0; i < actual_adds; i++) {
            adj[ne_i[i]*n+ne_j[i]] = adj[ne_j[i]*n+ne_i[i]] = 1;
            degrees[ne_i[i]]++; degrees[ne_j[i]]++;
        }

        /* SAVE the post-add state for backward replay */
        memcpy(adj_saved, adj, (size_t)nn);
        memcpy(degrees_saved, degrees, (size_t)n * sizeof(int));

        int cur_m = 0;
        for (int i = 0; i < n; i++) cur_m += degrees[i];
        cur_m /= 2;
        int to_remove = cur_m - m;

        /* Forward pass: collect actions + rewards */
        int num_steps = 0;
        GCNCache c1, c2;
        MLP2Cache ns_cache, es_cache;

        for (int step = 0; step < to_remove; step++) {
            float nm1 = (float)(n - 1); if (nm1 < 1.0f) nm1 = 1.0f;
            for (int i = 0; i < n; i++) h0[i] = (float)degrees[i] / nm1;

            gcn_layer_cached(adj, n, h0, 1, gw.conv1_w, gw.conv1_b, GNN_HIDDEN, h1, &c1);
            gcn_layer_cached(adj, n, h1, GNN_HIDDEN, gw.conv2_w, gw.conv2_b, GNN_HIDDEN, h2, &c2);

            cur_m = 0; for (int i = 0; i < n; i++) cur_m += degrees[i]; cur_m /= 2;
            if (cur_m <= 2*n) find_bridges(adj, n, bridge_mask);
            else memset(bridge_mask, 0, (size_t)nn);

            float ns_max = -FLT_MAX;
            int valid_nodes[64]; int num_valid = 0;
            for (int i = 0; i < n; i++) {
                int has = 0;
                for (int j = 0; j < n; j++)
                    if (i!=j && adj[i*n+j] && !bridge_mask[(i<j?i:j)*n+(i<j?j:i)]) { has=1; break; }
                if (has) {
                    node_scores[i] = mlp2_forward_cached(h2+i*GNN_HIDDEN, GNN_HIDDEN,
                        gw.ns_w0, gw.ns_b0, GNN_HIDDEN, gw.ns_w1, gw.ns_b1, &ns_cache);
                    valid_nodes[num_valid++] = i;
                    if (node_scores[i] > ns_max) ns_max = node_scores[i];
                } else node_scores[i] = -1e9f;
            }
            if (num_valid == 0) { gcn_cache_free(&c1); gcn_cache_free(&c2); break; }

            double node_sum = 0;
            for (int v = 0; v < num_valid; v++)
                node_sum += exp((double)(node_scores[valid_nodes[v]] - ns_max));
            double nr = rng_double(&train_rng) * node_sum;
            double ncsum = 0;
            int sel_vi = num_valid - 1;
            for (int v = 0; v < num_valid; v++) {
                ncsum += exp((double)(node_scores[valid_nodes[v]] - ns_max));
                if (nr <= ncsum) { sel_vi = v; break; }
            }
            int ni = valid_nodes[sel_vi];

            int nbrs[64]; int num_nbrs = 0;
            float es_vals[64], edge_feat[2*GNN_HIDDEN];
            memcpy(edge_feat, h2+ni*GNN_HIDDEN, (size_t)GNN_HIDDEN*sizeof(float));
            float es_max = -FLT_MAX;
            for (int j = 0; j < n; j++) {
                if (j==ni || !adj[ni*n+j]) continue;
                int a=ni<j?ni:j, b=ni<j?j:ni;
                if (bridge_mask[a*n+b]) continue;
                memcpy(edge_feat+GNN_HIDDEN, h2+j*GNN_HIDDEN, (size_t)GNN_HIDDEN*sizeof(float));
                es_vals[num_nbrs] = mlp2_forward_cached(edge_feat, 2*GNN_HIDDEN,
                    gw.es_w0, gw.es_b0, GNN_HIDDEN, gw.es_w1, gw.es_b1, &es_cache);
                nbrs[num_nbrs] = j;
                if (es_vals[num_nbrs] > es_max) es_max = es_vals[num_nbrs];
                num_nbrs++;
            }
            if (num_nbrs == 0) { gcn_cache_free(&c1); gcn_cache_free(&c2); break; }

            double edge_sum = 0;
            for (int e = 0; e < num_nbrs; e++) edge_sum += exp((double)(es_vals[e]-es_max));
            double er = rng_double(&train_rng) * edge_sum;
            double ecsum = 0;
            int sel_e = num_nbrs - 1;
            for (int e = 0; e < num_nbrs; e++) {
                ecsum += exp((double)(es_vals[e]-es_max));
                if (er <= ecsum) { sel_e = e; break; }
            }
            int ej = nbrs[sel_e];

            step_node[num_steps] = ni;
            step_nbr[num_steps] = ej;

            double l2_before = exact_lambda2(adj, n);
            adj[ni*n+ej] = adj[ej*n+ni] = 0;
            degrees[ni]--; degrees[ej]--;
            rewards[num_steps] = exact_lambda2(adj, n) - l2_before;

            gcn_cache_free(&c1);
            gcn_cache_free(&c2);
            num_steps++;
        }

        if (num_steps == 0) continue;

        /* Compute returns + baseline */
        double *returns = (double *)malloc((size_t)num_steps * sizeof(double));
        double G = 0;
        for (int t = num_steps-1; t >= 0; t--) {
            G = rewards[t] + 0.99 * G;
            returns[t] = G;
        }
        /* Backward pass: restore saved state, use mean-return baseline */
        double mean_ret = 0;
        for (int t = 0; t < num_steps; t++) mean_ret += returns[t];
        mean_ret /= num_steps;

        GNNGrad grad;
        gnn_grad_zero(&grad);

        memcpy(adj, adj_saved, (size_t)nn);
        memcpy(degrees, degrees_saved, (size_t)n * sizeof(int));

        for (int step = 0; step < num_steps; step++) {
            double advantage = returns[step] - mean_ret;
            float nm1 = (float)(n-1); if (nm1<1.0f) nm1=1.0f;
            for (int i = 0; i < n; i++) h0[i] = (float)degrees[i]/nm1;

            gcn_layer_cached(adj, n, h0, 1, gw.conv1_w, gw.conv1_b, GNN_HIDDEN, h1, &c1);
            gcn_layer_cached(adj, n, h1, GNN_HIDDEN, gw.conv2_w, gw.conv2_b, GNN_HIDDEN, h2, &c2);

            int ni2 = step_node[step], ej2 = step_nbr[step];

            /* Node scores + softmax for gradient */
            cur_m = 0; for (int i=0;i<n;i++) cur_m+=degrees[i]; cur_m/=2;
            if (cur_m<=2*n) find_bridges(adj,n,bridge_mask); else memset(bridge_mask,0,(size_t)nn);

            int vn[64]; int nv=0;
            MLP2Cache ns_caches[64];
            for (int i=0;i<n;i++) {
                int has=0;
                for (int j=0;j<n;j++)
                    if (i!=j&&adj[i*n+j]&&!bridge_mask[(i<j?i:j)*n+(i<j?j:i)]){has=1;break;}
                if (has) {
                    node_scores[i]=mlp2_forward_cached(h2+i*GNN_HIDDEN,GNN_HIDDEN,
                        gw.ns_w0,gw.ns_b0,GNN_HIDDEN,gw.ns_w1,gw.ns_b1,&ns_caches[nv]);
                    vn[nv++]=i;
                } else node_scores[i]=-1e9f;
            }

            /* Node softmax probs */
            float ns_max2=-FLT_MAX;
            for (int v=0;v<nv;v++) if (node_scores[vn[v]]>ns_max2) ns_max2=node_scores[vn[v]];
            double nsum2=0;
            for (int v=0;v<nv;v++) nsum2+=exp((double)(node_scores[vn[v]]-ns_max2));

            /* d(log_prob_node)/d(score_i) = (i==chosen ? 1 : 0) - softmax(i) */
            /* Scale by -advantage for REINFORCE */
            memset(d_h2, 0, (size_t)n*GNN_HIDDEN*sizeof(float));
            for (int v=0;v<nv;v++) {
                int idx=vn[v];
                double sm = exp((double)(node_scores[idx]-ns_max2)) / nsum2;
                double d_score = -advantage * ((idx==ni2 ? 1.0 : 0.0) - sm);
                /* Backprop through node_scorer MLP */
                float d_input[GNN_HIDDEN];
                memset(d_input, 0, sizeof(d_input));
                mlp2_backward(&ns_caches[v], gw.ns_w0, gw.ns_w1, GNN_HIDDEN,
                              (float)d_score, grad.ns_w0, grad.ns_b0,
                              grad.ns_w1, grad.ns_b1, d_input);
                /* Accumulate d_h2[idx] */
                for (int k=0;k<GNN_HIDDEN;k++) d_h2[idx*GNN_HIDDEN+k]+=d_input[k];
            }

            /* Edge scorer gradient (for selected node's edges) */
            int nb[64]; int nnb=0;
            MLP2Cache es_caches[64];
            float esv[64];
            float ef[2*GNN_HIDDEN];
            memcpy(ef, h2+ni2*GNN_HIDDEN, (size_t)GNN_HIDDEN*sizeof(float));
            float esm=-FLT_MAX;
            for (int j=0;j<n;j++) {
                if (j==ni2||!adj[ni2*n+j]) continue;
                int a=ni2<j?ni2:j,b=ni2<j?j:ni2;
                if (bridge_mask[a*n+b]) continue;
                memcpy(ef+GNN_HIDDEN, h2+j*GNN_HIDDEN, (size_t)GNN_HIDDEN*sizeof(float));
                esv[nnb]=mlp2_forward_cached(ef,2*GNN_HIDDEN,
                    gw.es_w0,gw.es_b0,GNN_HIDDEN,gw.es_w1,gw.es_b1,&es_caches[nnb]);
                nb[nnb]=j;
                if (esv[nnb]>esm) esm=esv[nnb];
                nnb++;
            }

            if (nnb > 0) {
                double esum2=0;
                for (int e=0;e<nnb;e++) esum2+=exp((double)(esv[e]-esm));

                for (int e=0;e<nnb;e++) {
                    double sm=exp((double)(esv[e]-esm))/esum2;
                    double d_score=-advantage*((nb[e]==ej2?1.0:0.0)-sm);
                    float d_ef[2*GNN_HIDDEN];
                    memset(d_ef,0,sizeof(d_ef));
                    mlp2_backward(&es_caches[e],gw.es_w0,gw.es_w1,GNN_HIDDEN,
                                  (float)d_score,grad.es_w0,grad.es_b0,
                                  grad.es_w1,grad.es_b1,d_ef);
                    /* d_ef flows to h2[ni2] (first half) and h2[nb[e]] (second half) */
                    for (int k=0;k<GNN_HIDDEN;k++) {
                        d_h2[ni2*GNN_HIDDEN+k]+=d_ef[k];
                        d_h2[nb[e]*GNN_HIDDEN+k]+=d_ef[GNN_HIDDEN+k];
                    }
                }
            }

            /* Backprop through GCN layers */
            gcn_layer_backward(adj, h1, &c2, gw.conv2_w, d_h2,
                               grad.conv2_w, grad.conv2_b, d_h1);
            gcn_layer_backward(adj, h0, &c1, gw.conv1_w, d_h1,
                               grad.conv1_w, grad.conv1_b, NULL);

            gcn_cache_free(&c1);
            gcn_cache_free(&c2);
            memset(d_h2, 0, (size_t)n*GNN_HIDDEN*sizeof(float));
            memset(d_h1, 0, (size_t)n*GNN_HIDDEN*sizeof(float));

            /* Apply the removal (replay) */
            adj[ni2*n+ej2]=adj[ej2*n+ni2]=0;
            degrees[ni2]--; degrees[ej2]--;
        }

        free(returns);

        /* Average gradients and update */
        gnn_grad_scale(&grad, 1.0f / (float)num_steps);
        gnn_grad_clip(&grad, 1.0f);
        gnn_adam_step(&gw, &grad, &adam, lr, 1e-4);

        /* Logging */
        if ((ep+1) % 100 == 0) {
            double elapsed = wall_time() - t_start;
            double final_l2 = exact_lambda2(adj, n);
            printf("ep %5d/%d | l2=%.3f | n=%d m=%d steps=%d | %.1f ep/s | %.1fm\n",
                   ep+1, num_episodes, final_l2, n, m, num_steps,
                   (ep+1)/fmax(elapsed,0.001), elapsed/60.0);
        }

        /* Save best */
        if ((ep+1) % 500 == 0) {
            char path[512];
            snprintf(path, sizeof(path), "%s/ckpt_%d.bin", save_dir, ep+1);
            save_gnn_weights(path, &gw);
        }
    }

    /* Final save */
    char path[512];
    snprintf(path, sizeof(path), "%s/final.bin", save_dir);
    save_gnn_weights(path, &gw);
    printf("Saved to %s\n", save_dir);

    free(h0);free(h1);free(h2);free(node_scores);free(d_h2);free(d_h1);
    free(adj);free(adj_saved);free(degrees);free(degrees_saved);free(bridge_mask);
    free(rewards);free(step_node);free(step_nbr);
}

/* ========================================================================== */
/* Main                                                                        */
/* ========================================================================== */

int main(int argc, char **argv) {
    int n_values[64], num_n = 0;
    char *ckpt_path = NULL;
    char *bl_path = NULL;
    double epoch_mult = 4.0;
    int add_mult = 2;
    int do_train = 0;
    int train_episodes = 5000;
    int train_min_n = 8, train_max_n = 32;
    double train_lr = 1e-3;
    char save_dir[256] = "logs/gnn_train";
    uint64_t seed = 42;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--n") == 0 && i+1 < argc) {
            char *tok = strtok(argv[++i], ",");
            while (tok && num_n < 64) { n_values[num_n++] = atoi(tok); tok = strtok(NULL, ","); }
        }
        else if (strcmp(argv[i], "--checkpoint") == 0 && i+1 < argc) ckpt_path = argv[++i];
        else if (strcmp(argv[i], "--baselines") == 0 && i+1 < argc) bl_path = argv[++i];
        else if (strcmp(argv[i], "--epoch-mult") == 0 && i+1 < argc) epoch_mult = atof(argv[++i]);
        else if (strcmp(argv[i], "--add-mult") == 0 && i+1 < argc) add_mult = atoi(argv[++i]);
        else if (strcmp(argv[i], "--train") == 0) do_train = 1;
        else if (strcmp(argv[i], "--episodes") == 0 && i+1 < argc) train_episodes = atoi(argv[++i]);
        else if (strcmp(argv[i], "--min-n") == 0 && i+1 < argc) train_min_n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-n") == 0 && i+1 < argc) train_max_n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--lr") == 0 && i+1 < argc) train_lr = atof(argv[++i]);
        else if (strcmp(argv[i], "--save-dir") == 0 && i+1 < argc) snprintf(save_dir,sizeof(save_dir),"%s",argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc) seed = (uint64_t)atol(argv[++i]);
    }

    if (do_train) {
        gnn_train(train_min_n, train_max_n, train_episodes, add_mult,
                  train_lr, seed, save_dir);
        return 0;
    }

    if (!ckpt_path || num_n == 0) {
        printf("Usage: %s --checkpoint gnn.bin --n 8,16,24 --baselines bl.csv\n", argv[0]);
        printf("  --epoch-mult F   K = F*n epochs (default: 4.0)\n");
        printf("  --add-mult N     adds = N*n non-edges per epoch (default: 2)\n");
        printf("  --train          Train mode\n");
        printf("  --episodes N     Training episodes (default: 5000)\n");
        printf("  --min-n N        Min training n (default: 8)\n");
        printf("  --max-n N        Max training n (default: 32)\n");
        printf("  --lr F           Learning rate (default: 1e-3)\n");
        printf("  --save-dir PATH  Save directory (default: logs/gnn_train)\n");
        return 1;
    }

    GNNWeights gw;
    if (load_gnn_weights(ckpt_path, &gw) != 0) {
        fprintf(stderr, "Failed to load GNN weights: %s\n", ckpt_path);
        return 1;
    }
    printf("Loaded GNN weights: %s\n", ckpt_path);

    if (bl_path) load_bl(bl_path);

    printf("GNN Eval | epoch_mult=%.1f, add_mult=%d, threads=%d\n",
           epoch_mult, add_mult, omp_get_max_threads());
    printf("================================================================\n");

    for (int ni = 0; ni < num_n; ni++) {
        int n = n_values[ni];
        int max_m = n * (n - 1) / 2;
        int num_epochs = (int)(epoch_mult * n);
        int num_adds = add_mult * n;

        int *m_list = (int *)malloc((size_t)(max_m + 1) * sizeof(int));
        int num_m = 0;
        for (int mv = n + 1; mv <= max_m; mv++) {
            BL *b = find_bl(n, mv);
            if (b && b->best > 0) m_list[num_m++] = mv;
        }
        if (num_m == 0) { free(m_list); continue; }

        printf("\nn=%d (%d configs, K=%d, adds=%d)\n", n, num_m, num_epochs, num_adds);
        double t0 = wall_time();

        int total_jobs = num_m;
        double *job_l2 = (double *)malloc((size_t)total_jobs * sizeof(double));

        #pragma omp parallel for schedule(dynamic, 1)
        for (int mi = 0; mi < total_jobs; mi++) {
            int mv = m_list[mi];
            uint64_t s = (uint64_t)(n * 100 + mv);
            job_l2[mi] = gnn_episode(&gw, n, mv, num_epochs, num_adds, s);
        }

        double sum_rl = 0, sum_fv = 0, sum_best = 0;
        int wins_fv = 0, wins_best = 0, total = 0;
        for (int mi = 0; mi < num_m; mi++) {
            BL *b = find_bl(n, m_list[mi]);
            if (!b) continue;
            sum_rl += job_l2[mi]; sum_fv += b->fv; sum_best += b->best;
            if (job_l2[mi] > b->fv + 1e-6) wins_fv++;
            if (job_l2[mi] > b->best + 1e-6) wins_best++;
            total++;
        }

        double elapsed = wall_time() - t0;
        printf("  vs FV: %.1f%% (%dW/%d) | vs Best: %.1f%% (%dW/%d) | %.1fs\n",
               sum_rl/sum_fv*100, wins_fv, total,
               sum_rl/sum_best*100, wins_best, total, elapsed);

        free(m_list); free(job_l2);
    }

    free(baselines);
    return 0;
}
