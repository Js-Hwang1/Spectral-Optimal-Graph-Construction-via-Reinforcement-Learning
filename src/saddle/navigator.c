/*
 * navigator.c -- Navigator agent for saddle detection + DFS backtracking.
 *
 * Small MLP (16 -> 32 -> 32 -> 1) scores candidates at potential saddle
 * points. DFS stack saves/restores graph state for backtracking.
 */

#include "saddle.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define NH    SD_NAV_NH
#define N_IN  SD_NAV_IN

/* ========================================================================== */
/* Weight initialization                                                       */
/* ========================================================================== */

void navigator_init_weights(NavigatorWeights *w, uint64_t seed) {
    RNG rng;
    rng_init(&rng, seed);

    /* Layer 0: NAV_IN -> NH */
    int np0 = NH * N_IN;
    float sc0 = sqrtf(2.0f / (float)(N_IN + NH));
    for (int i = 0; i < np0; i += 2) {
        double u1 = rng_double(&rng) * 0.998 + 0.001;
        double u2 = rng_double(&rng);
        double r = sqrt(-2.0 * log(u1));
        w->w0[i] = (float)(r * cos(2.0 * M_PI * u2)) * sc0;
        if (i + 1 < np0)
            w->w0[i + 1] = (float)(r * sin(2.0 * M_PI * u2)) * sc0;
    }
    memset(w->b0, 0, sizeof(w->b0));

    /* Layer 1: NH -> NH */
    int np1 = NH * NH;
    float sc1 = sqrtf(2.0f / (float)(NH + NH));
    for (int i = 0; i < np1; i += 2) {
        double u1 = rng_double(&rng) * 0.998 + 0.001;
        double u2 = rng_double(&rng);
        double r = sqrt(-2.0 * log(u1));
        w->w1[i] = (float)(r * cos(2.0 * M_PI * u2)) * sc1;
        if (i + 1 < np1)
            w->w1[i + 1] = (float)(r * sin(2.0 * M_PI * u2)) * sc1;
    }
    memset(w->b1, 0, sizeof(w->b1));

    /* Output layer: NH -> 1, zero-init weights, bias = -2.0 */
    memset(w->w2, 0, sizeof(w->w2));
    w->b2[0] = 0.0f;  /* neutral init: let training decide the boundary */
}

/* ========================================================================== */
/* Feature computation                                                         */
/* ========================================================================== */

/* qsort comparator: descending floats */
static int cmp_float_desc(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    if (fa > fb) return -1;
    if (fa < fb) return 1;
    return 0;
}

void navigator_compute_features(const float *add_scores, int n_add,
                                const float *rem_scores, int n_rem,
                                const GraphState *gs, const NavFeatures *prev,
                                NavFeatures *out) {
    (void)rem_scores; (void)n_rem;
    out->valid = 1;

    if (n_add < 2) {
        memset(out, 0, sizeof(*out));
        return;
    }

    /* Z-score the climber's raw scores to calibrate across n.
     * Then compute entropy on softmax(z-scored) / log(n_add). */
    double s_mean = 0;
    for (int i = 0; i < n_add; i++) s_mean += add_scores[i];
    s_mean /= n_add;
    double s_var = 0;
    for (int i = 0; i < n_add; i++) {
        double d = add_scores[i] - s_mean;
        s_var += d * d;
    }
    s_var /= n_add;
    double s_std = (s_var > 1e-12) ? sqrt(s_var) : 1.0;

    /* Softmax on z-scored values: softmax((score - mean) / std) */
    double z_max = (add_scores[0] - s_mean) / s_std;
    for (int i = 1; i < n_add; i++) {
        double zi = (add_scores[i] - s_mean) / s_std;
        if (zi > z_max) z_max = zi;
    }
    double z = 0;
    for (int i = 0; i < n_add; i++)
        z += exp((add_scores[i] - s_mean) / s_std - z_max);
    double log_z = log(z);
    double ent = 0;
    for (int i = 0; i < n_add; i++) {
        double lp = (add_scores[i] - s_mean) / s_std - z_max - log_z;
        double p = exp(lp);
        if (p > 1e-12) ent -= p * lp;
    }
    double log_n = log((double)n_add);
    out->entropy = (log_n > 1e-12) ? (float)(ent / log_n) : 0.0f;

    /* l2/n */
    int n = gs->n;
    out->l2_n = (float)(gs->v2_lam / n);

    /* rho = (m - m_min) / (m_max - m_min) */
    int m_min = n - 1;
    int m_max = n * (n - 1) / 2;
    double denom = (double)(m_max - m_min);
    out->mean_deg_n = (denom > 0) ? (float)((gs->m - m_min) / denom) : 0.0f;

    /* spectral gap: (lambda3 - lambda2) / n */
    out->spec_gap = (float)((gs->v3_lam - gs->v2_lam) / n);

    /* entropy derivative: transition from confident to unsure */
    out->d_entropy = (prev && prev->valid) ? out->entropy - prev->entropy : 0.0f;
}

/* ========================================================================== */
/* Score a single candidate                                                    */
/* ========================================================================== */

float navigator_score_candidate(const NavigatorWeights *w, const NavFeatures *nf,
                                const EdgeCandidate *cand, int rank, int n_cands) {
    (void)cand; (void)rank; (void)n_cands;
    float x[N_IN];
    x[0] = nf->spec_gap;
    x[1] = nf->mean_deg_n;

    /* Forward: x -> h0 (SiLU) -> h1 (SiLU) -> scalar */
    float h0[NH];
    for (int i = 0; i < NH; i++) {
        float s = w->b0[i];
        for (int j = 0; j < N_IN; j++)
            s += w->w0[i * N_IN + j] * x[j];
        h0[i] = sd_silu(s);
    }

    float h1[NH];
    for (int i = 0; i < NH; i++) {
        float s = w->b1[i];
        for (int j = 0; j < NH; j++)
            s += w->w1[i * NH + j] * h0[j];
        h1[i] = sd_silu(s);
    }

    float out = w->b2[0];
    for (int j = 0; j < NH; j++)
        out += w->w2[j] * h1[j];

    return out;
}

/* ========================================================================== */
/* Saddle detection                                                            */
/* ========================================================================== */

int navigator_detect_saddle(const float *saddle_scores, int n_cands, float threshold) {
    float best = -FLT_MAX;
    for (int i = 0; i < n_cands; i++) {
        if (saddle_scores[i] > best) best = saddle_scores[i];
    }
    return best > threshold;
}

/* ========================================================================== */
/* DFS stack operations                                                        */
/* ========================================================================== */

void stack_init(SaddleStack *st) {
    st->sp = 0;
}

void stack_push(SaddleStack *st, const GraphState *gs,
                const EdgeCandidate *cands, int n_cands) {
    if (st->sp >= SD_MAX_STACK) return;

    SaddleFrame *f = &st->frames[st->sp];

    /* Deep clone the graph state */
    f->saved_gs = graph_state_clone(gs);

    /* Copy candidates (capped at SD_MAX_CAND) */
    f->n_cands = (n_cands < SD_MAX_CAND) ? n_cands : SD_MAX_CAND;
    memcpy(f->cands, cands, (size_t)f->n_cands * sizeof(EdgeCandidate));

    /* First candidate (rank 0) is the one the climber already took,
     * so the next untried branch starts at index 1. */
    f->tried = 1;

    st->sp++;
}

int stack_backtrack(SaddleStack *st, GraphState *gs, EdgeCandidate *next_cand) {
    while (st->sp > 0) {
        SaddleFrame *f = &st->frames[st->sp - 1];

        if (f->tried < f->n_cands) {
            /* Restore graph to the saved state */
            graph_state_restore(gs, f->saved_gs);

            /* Return the next untried candidate */
            *next_cand = f->cands[f->tried];
            f->tried++;
            return 1;
        }

        /* All branches exhausted at this depth -- pop and try parent */
        graph_state_free(f->saved_gs);
        f->saved_gs = NULL;
        st->sp--;
    }

    return 0;
}

void stack_free(SaddleStack *st) {
    for (int i = 0; i < st->sp; i++) {
        if (st->frames[i].saved_gs) {
            graph_state_free(st->frames[i].saved_gs);
            st->frames[i].saved_gs = NULL;
        }
    }
    st->sp = 0;
}
