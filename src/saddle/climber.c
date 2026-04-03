/*
 * climber.c -- Edge-level scoring for ADD and REM decisions.
 *
 * Architecture:
 *   Shared node embedder: NE_IN(5) -> H -> H (SiLU)
 *   ADD edge scorer:  [embed_i || embed_j || cn || fv_gap || reff] -> H -> 1
 *   REM edge scorer:  same input structure -> H -> 1
 *
 * Scores ALL candidate edges directly. No node-first decomposition.
 */

#include "saddle.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define H   SD_H
#define NI  SD_NE_IN
#define EI  SD_ES_IN

/* ========================================================================== */
/* Weight initialization                                                       */
/* ========================================================================== */

static void init_layer(float *w, float *b, int in, int out, RNG *rng) {
    float sc = sqrtf(2.0f / (float)(in + out));
    int np = out * in;
    for (int i = 0; i < np; i += 2) {
        double u1 = rng_double(rng) * 0.998 + 0.001;
        double u2 = rng_double(rng);
        double r = sqrt(-2.0 * log(u1));
        w[i] = (float)(r * cos(2.0 * M_PI * u2)) * sc;
        if (i + 1 < np)
            w[i + 1] = (float)(r * sin(2.0 * M_PI * u2)) * sc;
    }
    memset(b, 0, (size_t)out * sizeof(float));
}

void climber_init_weights(ClimberWeights *w, uint64_t seed) {
    RNG rng; rng_init(&rng, seed);
    init_layer(w->ne_w0, w->ne_b0, NI, H, &rng);
    init_layer(w->ne_w1, w->ne_b1, H, H, &rng);
    init_layer(w->ae_w0, w->ae_b0, EI, H, &rng);
    memset(w->ae_w1, 0, sizeof(w->ae_w1)); w->ae_b1[0] = 0;
    init_layer(w->re_w0, w->re_b0, EI, H, &rng);
    memset(w->re_w1, 0, sizeof(w->re_w1)); w->re_b1[0] = 0;
}

/* ========================================================================== */
/* Node embedding                                                              */
/* ========================================================================== */

void climber_embed_all(const ClimberWeights *w, const GraphState *gs, float *hh) {
    int n = gs->n;
    float nm1 = (float)(n > 1 ? n - 1 : 1);
    int max_m = n * (n - 1) / 2;
    float tgt_dens = (float)(gs->m - (n - 1)) / (float)(max_m - (n - 1) > 0 ? max_m - (n - 1) : 1);

    for (int i = 0; i < n; i++) {
        float d = (float)gs->deg[i] / nm1;
        float mt = (float)(gs->deg[i] * (gs->deg[i] - 1)) / 2.0f;
        float tc = mt > 0.5f ? (float)gs->tri[i] / mt : 0.0f;
        float v2sq = (float)(n * gs->v2[i] * gs->v2[i]);
        float er = (float)(n * gs->lpinv[i * n + i]);
        float feats[NI] = {d, tc, tgt_dens, v2sq, er};

        float h0[H];
        for (int k = 0; k < H; k++) {
            float s = w->ne_b0[k];
            for (int f = 0; f < NI; f++) s += w->ne_w0[k * NI + f] * feats[f];
            h0[k] = sd_silu(s);
        }
        for (int k = 0; k < H; k++) {
            float s = w->ne_b1[k];
            for (int j = 0; j < H; j++) s += w->ne_w1[k * H + j] * h0[j];
            hh[i * H + k] = sd_silu(s);
        }
    }
}

void climber_embed_all_cached(const ClimberWeights *w, const GraphState *gs,
                               float *hh, NECache *nec) {
    int n = gs->n;
    float nm1 = (float)(n > 1 ? n - 1 : 1);
    int max_m = n * (n - 1) / 2;
    float tgt_dens = (float)(gs->m - (n - 1)) / (float)(max_m - (n - 1) > 0 ? max_m - (n - 1) : 1);

    for (int i = 0; i < n; i++) {
        float d = (float)gs->deg[i] / nm1;
        float mt = (float)(gs->deg[i] * (gs->deg[i] - 1)) / 2.0f;
        float tc = mt > 0.5f ? (float)gs->tri[i] / mt : 0.0f;
        float v2sq = (float)(n * gs->v2[i] * gs->v2[i]);
        float er = (float)(n * gs->lpinv[i * n + i]);
        float feats[NI] = {d, tc, tgt_dens, v2sq, er};
        for (int f = 0; f < NI; f++) nec[i].in_val[f] = feats[f];

        for (int k = 0; k < H; k++) {
            float s = w->ne_b0[k];
            for (int f = 0; f < NI; f++) s += w->ne_w0[k * NI + f] * feats[f];
            nec[i].pre0[k] = s; nec[i].h0[k] = sd_silu(s);
        }
        for (int k = 0; k < H; k++) {
            float s = w->ne_b1[k];
            for (int j = 0; j < H; j++) s += w->ne_w1[k * H + j] * nec[i].h0[j];
            nec[i].pre1[k] = s;
            hh[i * H + k] = sd_silu(s);
        }
    }
}

/* ========================================================================== */
/* Edge scoring                                                                */
/* ========================================================================== */

static float edge_score(const float *hi, const float *hj, float cn, float fv, float re,
                         const float *W0, const float *b0, const float *W1, const float *b1) {
    float input[EI];
    memcpy(input, hi, (size_t)H * sizeof(float));
    memcpy(input + H, hj, (size_t)H * sizeof(float));
    input[2*H] = cn; input[2*H+1] = fv; input[2*H+2] = re;

    float hid[H];
    for (int k = 0; k < H; k++) {
        float s = b0[k];
        for (int f = 0; f < EI; f++) s += W0[k * EI + f] * input[f];
        hid[k] = sd_silu(s);
    }
    float out = b1[0];
    for (int k = 0; k < H; k++) out += W1[k] * hid[k];
    return out;
}

int climber_score_add_edges(const ClimberWeights *w, const float *hh,
                             const GraphState *gs, float *scores, int edges[][2]) {
    int n = gs->n, cnt = 0;
    float inv_n = 1.0f / (float)n;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            if (gs->adj[i * n + j]) continue;
            float cn = (float)gs->cn[i * n + j] * inv_n;
            float fv = (float)(n * (gs->v2[i] - gs->v2[j]) * (gs->v2[i] - gs->v2[j]));
            double rv = gs->lpinv[i*n+i] + gs->lpinv[j*n+j] - 2.0*gs->lpinv[i*n+j];
            scores[cnt] = edge_score(hh + i*H, hh + j*H, cn, fv, (float)(n*rv),
                                      w->ae_w0, w->ae_b0, w->ae_w1, w->ae_b1);
            edges[cnt][0] = i; edges[cnt][1] = j; cnt++;
        }
    return cnt;
}

int climber_score_rem_edges(const ClimberWeights *w, const float *hh,
                             const GraphState *gs, float *scores, int edges[][2]) {
    int n = gs->n, cnt = 0;
    float inv_n = 1.0f / (float)n;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            if (!gs->adj[i * n + j] || gs->bs.bridge[i * n + j]) continue;
            float cn = (float)gs->cn[i * n + j] * inv_n;
            float fv = (float)(n * (gs->v2[i] - gs->v2[j]) * (gs->v2[i] - gs->v2[j]));
            double rv = gs->lpinv[i*n+i] + gs->lpinv[j*n+j] - 2.0*gs->lpinv[i*n+j];
            scores[cnt] = edge_score(hh + i*H, hh + j*H, cn, fv, (float)(n*rv),
                                      w->re_w0, w->re_b0, w->re_w1, w->re_b1);
            edges[cnt][0] = i; edges[cnt][1] = j; cnt++;
        }
    return cnt;
}

int climber_greedy_pick(const float *scores, int count) {
    int best = 0;
    for (int i = 1; i < count; i++)
        if (scores[i] > scores[best]) best = i;
    return best;
}

/* ========================================================================== */
/* Cached forward + backward                                                   */
/* ========================================================================== */

float climber_edge_fwd_cached(const float *hi, const float *hj,
                               float cn, float fv_gap, float reff,
                               const float *W0, const float *b0,
                               const float *W1, const float *b1, ESCache *ec) {
    memcpy(ec->input, hi, (size_t)H * sizeof(float));
    memcpy(ec->input + H, hj, (size_t)H * sizeof(float));
    ec->input[2*H] = cn; ec->input[2*H+1] = fv_gap; ec->input[2*H+2] = reff;

    for (int k = 0; k < H; k++) {
        float s = b0[k];
        for (int f = 0; f < EI; f++) s += W0[k * EI + f] * ec->input[f];
        ec->pre[k] = s; ec->hid[k] = sd_silu(s);
    }
    float out = b1[0];
    for (int k = 0; k < H; k++) out += W1[k] * ec->hid[k];
    return out;
}

void climber_edge_bwd(const ESCache *c, const float *W0, const float *W1,
                       float d_out, float *dW0, float *db0, float *dW1, float *db1,
                       float *d_hi, float *d_hj) {
    db1[0] += d_out;
    float dh[H];
    for (int k = 0; k < H; k++) {
        dW1[k] += d_out * c->hid[k];
        dh[k] = d_out * W1[k];
    }
    float d_in[EI]; memset(d_in, 0, sizeof(d_in));
    for (int k = 0; k < H; k++) {
        float dp = dh[k] * sd_silu_d(c->pre[k]);
        db0[k] += dp;
        for (int f = 0; f < EI; f++) {
            dW0[k * EI + f] += dp * c->input[f];
            d_in[f] += dp * W0[k * EI + f];
        }
    }
    for (int k = 0; k < H; k++) d_hi[k] = d_in[k];
    for (int k = 0; k < H; k++) d_hj[k] = d_in[H + k];
}

void climber_embed_bwd(const ClimberWeights *w, const NECache *c,
                        const float *d_h, ClimberGrads *pg) {
    float dh1[H];
    for (int k = 0; k < H; k++) dh1[k] = d_h[k] * sd_silu_d(c->pre1[k]);
    for (int k = 0; k < H; k++) {
        pg->ne_b1[k] += dh1[k];
        for (int j = 0; j < H; j++) pg->ne_w1[k * H + j] += dh1[k] * c->h0[j];
    }
    float dh0[H]; memset(dh0, 0, sizeof(dh0));
    for (int k = 0; k < H; k++)
        for (int j = 0; j < H; j++) dh0[j] += dh1[k] * w->ne_w1[k * H + j];
    for (int k = 0; k < H; k++) {
        float dp = dh0[k] * sd_silu_d(c->pre0[k]);
        pg->ne_b0[k] += dp;
        for (int f = 0; f < NI; f++) pg->ne_w0[k * NI + f] += dp * c->in_val[f];
    }
}
