/*
 * train_navigator.c -- Train the Navigator as a DETECTOR.
 *
 * Binary classification: does branching at this step cause divergence?
 * Input: [entropy, l2_n, mean_deg_n] -> sigmoid -> P(saddle)
 * Label: 1 if top-6 branches reach different leaves (max - min > eps), 0 otherwise
 * Loss: BCE, every step is a training example (dense signal)
 *
 * Detection and ranking are SEPARATE concerns. This trains detection only.
 */

#include "saddle.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <sys/stat.h>

#define H SD_H

static double wall_time(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* Greedy rollout to stagnation using the climber. Returns leaf lambda2. */
static double greedy_to_leaf(const ClimberWeights *cw, GraphState *gs, int max_steps) {
    int n = gs->n;
    int max_edges = n * (n - 1) / 2;
    float *hh = (float *)malloc((size_t)(n * H) * sizeof(float));
    float *sc = (float *)malloc((size_t)max_edges * sizeof(float));
    int (*ed)[2] = (int (*)[2])malloc((size_t)max_edges * 2 * sizeof(int));
    double prev = -1; int stag = 0;

    for (int s = 0; s < max_steps; s++) {
        climber_embed_all(cw, gs, hh);
        int na = climber_score_add_edges(cw, hh, gs, sc, ed);
        if (na == 0) break;
        int p = climber_greedy_pick(sc, na);
        graph_state_add_edge(gs, ed[p][0], ed[p][1]);

        climber_embed_all(cw, gs, hh);
        int nr = climber_score_rem_edges(cw, hh, gs, sc, ed);
        if (nr > 0) { p = climber_greedy_pick(sc, nr); graph_state_rem_edge(gs, ed[p][0], ed[p][1]); }

        double cur = exact_lambda2(gs->adj, n);
        if (fabs(cur - prev) < 1e-6) stag++; else stag = 0;
        prev = cur;
        if (stag >= 3) break;
    }
    double leaf = exact_lambda2(gs->adj, n);
    free(hh); free(sc); free(ed);
    return leaf;
}

/* Navigator forward: 3 -> NH -> NH -> 1 */
static float nav_fwd(const NavigatorWeights *w, const float *x,
                      float *h0, float *p0, float *h1, float *p1) {
    for (int i = 0; i < SD_NAV_NH; i++) {
        float s = w->b0[i];
        for (int j = 0; j < SD_NAV_IN; j++) s += w->w0[i * SD_NAV_IN + j] * x[j];
        p0[i] = s; h0[i] = sd_silu(s);
    }
    for (int i = 0; i < SD_NAV_NH; i++) {
        float s = w->b1[i];
        for (int j = 0; j < SD_NAV_NH; j++) s += w->w1[i * SD_NAV_NH + j] * h0[j];
        p1[i] = s; h1[i] = sd_silu(s);
    }
    float out = w->b2[0];
    for (int j = 0; j < SD_NAV_NH; j++) out += w->w2[j] * h1[j];
    return out;
}

/* Navigator backward */
static void nav_bwd(const NavigatorWeights *w, NavigatorGrads *g,
                     const float *x, const float *h0, const float *p0,
                     const float *h1, const float *p1, float d_out) {
    g->b2[0] += d_out;
    float dh1[SD_NAV_NH];
    for (int j = 0; j < SD_NAV_NH; j++) { g->w2[j] += d_out * h1[j]; dh1[j] = d_out * w->w2[j]; }
    float dh0[SD_NAV_NH]; memset(dh0, 0, sizeof(dh0));
    for (int i = 0; i < SD_NAV_NH; i++) {
        float dp = dh1[i] * sd_silu_d(p1[i]);
        g->b1[i] += dp;
        for (int j = 0; j < SD_NAV_NH; j++) { g->w1[i*SD_NAV_NH+j] += dp*h0[j]; dh0[j] += dp*w->w1[i*SD_NAV_NH+j]; }
    }
    for (int i = 0; i < SD_NAV_NH; i++) {
        float dp = dh0[i] * sd_silu_d(p0[i]);
        g->b0[i] += dp;
        for (int j = 0; j < SD_NAV_IN; j++) g->w0[i*SD_NAV_IN+j] += dp*x[j];
    }
}

void train_navigator(const ClimberWeights *frozen_cw, SaddleTrainConfig *cfg) {
    NavigatorWeights w;
    navigator_init_weights(&w, cfg->seed);

    int np = (int)(sizeof(NavigatorWeights) / sizeof(float));
    float *adam_m = (float *)calloc((size_t)np, sizeof(float));
    float *adam_v = (float *)calloc((size_t)np, sizeof(float));
    int adam_t = 0;

    char cmd[512]; snprintf(cmd, sizeof(cmd), "mkdir -p %s", cfg->save_dir);
    system(cmd);

    RNG rng; rng_init(&rng, cfg->seed + 12345);
    double t0 = wall_time();
    int total_pos = 0, total_neg = 0, total_correct = 0, total_examples = 0;
    double sum_loss = 0;
    /* Windowed stats (last log_freq episodes) */
    int win_correct = 0, win_examples = 0;
    double win_loss = 0;
    /* Feature stats for positive vs negative */
    double feat_sum_pos[2] = {0}, feat_sum_neg[2] = {0};
    double feat_ssq_pos[2] = {0}, feat_ssq_neg[2] = {0};
    int n_pos_feat = 0, n_neg_feat = 0;

    for (int ep = 0; ep < cfg->num_episodes; ep++) {
        int n = cfg->min_n + rng_int(&rng, cfg->max_n - cfg->min_n + 1);
        int max_m = n * (n - 1) / 2;
        int max_edges = max_m;
        /* Clip m to mid-density regime: rho in [0.1, 0.8] */
        int m_lo = (n - 1) + (int)(0.1 * (max_m - (n - 1)));
        int m_hi = (n - 1) + (int)(0.8 * (max_m - (n - 1)));
        if (m_lo < n) m_lo = n;
        int m = m_lo + rng_int(&rng, m_hi - m_lo + 1);

        uint8_t *adj = (uint8_t *)calloc((size_t)(n * n), 1);
        int *deg = (int *)calloc((size_t)n, sizeof(int));
        build_ring_random(adj, deg, n, m, &rng);
        GraphState *gs = graph_state_alloc(n);
        graph_state_init(gs, adj, n, m);
        free(adj); free(deg);

        float *hh = (float *)malloc((size_t)(n * H) * sizeof(float));
        float *add_scores = (float *)malloc((size_t)max_edges * sizeof(float));
        int (*add_edges)[2] = (int (*)[2])malloc((size_t)max_edges * 2 * sizeof(int));
        float *rem_scores = (float *)malloc((size_t)max_edges * sizeof(float));
        int (*rem_edges)[2] = (int (*)[2])malloc((size_t)max_edges * 2 * sizeof(int));

        NavigatorGrads grad; memset(&grad, 0, sizeof(grad));
        NavFeatures nf = {0}, prev_nf = {0};
        int ep_examples = 0;
        double prev_l2 = -1; int stagnant = 0;

        int num_steps = cfg->budget_mult * n;
        for (int step = 0; step < num_steps; step++) {
            /* Score ADD edges */
            climber_embed_all(frozen_cw, gs, hh);
            int na = climber_score_add_edges(frozen_cw, hh, gs, add_scores, add_edges);
            if (na < 2) break;

            /* Compute detector features (with prev for derivative) */
            navigator_compute_features(add_scores, na, NULL, 0, gs,
                                       prev_nf.valid ? &prev_nf : NULL, &nf);

            /* Find top-6 candidates */
            int K = SD_MAX_CAND < na ? SD_MAX_CAND : na;
            int top[SD_MAX_CAND];
            uint8_t *used = (uint8_t *)calloc((size_t)na, 1);
            for (int k = 0; k < K; k++) {
                int best = -1;
                for (int e = 0; e < na; e++) {
                    if (used[e]) continue;
                    if (best < 0 || add_scores[e] > add_scores[best]) best = e;
                }
                top[k] = best; used[best] = 1;
            }
            free(used);

            /* Evaluate each branch: clone, apply, rollout to leaf */
            double branch_l2[SD_MAX_CAND];
            for (int k = 0; k < K; k++) {
                GraphState *clone = graph_state_clone(gs);
                int ei = add_edges[top[k]][0], ej = add_edges[top[k]][1];
                graph_state_add_edge(clone, ei < ej ? ei : ej, ei < ej ? ej : ei);

                /* Quick REM */
                climber_embed_all(frozen_cw, clone, hh);
                int nr = climber_score_rem_edges(frozen_cw, hh, clone, rem_scores, rem_edges);
                if (nr > 0) {
                    int rp = climber_greedy_pick(rem_scores, nr);
                    graph_state_rem_edge(clone, rem_edges[rp][0], rem_edges[rp][1]);
                }
                branch_l2[k] = greedy_to_leaf(frozen_cw, clone, 4 * n);
                graph_state_free(clone);
            }

            /* Label: does GOD-best beat greedy? Train on GOD trajectories
             * so training distribution matches inference distribution. */
            int god_best = 0;
            for (int k = 1; k < K; k++) {
                if (branch_l2[k] > branch_l2[god_best]) god_best = k;
            }
            double gain = branch_l2[god_best] - branch_l2[0];
            float label = (gain > 0.01) ? 1.0f : 0.0f;

            /* Navigator forward */
            float x[SD_NAV_IN] = {nf.spec_gap, nf.mean_deg_n};
            float h0[SD_NAV_NH], p0[SD_NAV_NH], h1[SD_NAV_NH], p1[SD_NAV_NH];
            float logit = nav_fwd(&w, x, h0, p0, h1, p1);
            float pred = sd_sigmoid(logit);

            /* BCE loss gradient: d = pred - label */
            float d_bce = pred - label;
            nav_bwd(&w, &grad, x, h0, p0, h1, p1, d_bce);

            /* Stats */
            float bce = -(label * logf(pred + 1e-7f) + (1-label) * logf(1-pred + 1e-7f));
            sum_loss += bce; win_loss += bce;
            if (label > 0.5f) {
                total_pos++;
                for (int fi = 0; fi < 2; fi++) { feat_sum_pos[fi] += x[fi]; feat_ssq_pos[fi] += x[fi]*x[fi]; }
                n_pos_feat++;
            } else {
                total_neg++;
                for (int fi = 0; fi < 2; fi++) { feat_sum_neg[fi] += x[fi]; feat_ssq_neg[fi] += x[fi]*x[fi]; }
                n_neg_feat++;
            }
            int correct = ((pred > 0.5f) == (label > 0.5f));
            total_correct += correct; win_correct += correct;
            total_examples++; win_examples++;
            ep_examples++;

            /* Apply GOD-best to continue trajectory (matches inference) */
            int pick = top[god_best];
            int ei = add_edges[pick][0], ej = add_edges[pick][1];
            graph_state_add_edge(gs, ei < ej ? ei : ej, ei < ej ? ej : ei);

            climber_embed_all(frozen_cw, gs, hh);
            int nr = climber_score_rem_edges(frozen_cw, hh, gs, rem_scores, rem_edges);
            if (nr > 0) {
                int rp = climber_greedy_pick(rem_scores, nr);
                graph_state_rem_edge(gs, rem_edges[rp][0], rem_edges[rp][1]);
            }

            /* Save prev features for derivative */
            prev_nf = nf;

            /* Stagnation check */
            double cur = exact_lambda2(gs->adj, n);
            if (fabs(cur - prev_l2) < 1e-6) stagnant++; else stagnant = 0;
            prev_l2 = cur;
            if (stagnant >= 3) break;
        }

        /* Adam step */
        if (ep_examples > 0) {
            float *gp = (float *)&grad;
            float gnorm = 0;
            for (int i = 0; i < np; i++) gnorm += gp[i] * gp[i];
            gnorm = sqrtf(gnorm);
            float scale = 1.0f / (float)ep_examples;
            if (gnorm * scale > (float)cfg->grad_clip) scale = (float)cfg->grad_clip / gnorm;
            for (int i = 0; i < np; i++) gp[i] *= scale;

            adam_t++;
            float *wp = (float *)&w;
            double bc1 = 1.0 - pow(0.9, adam_t), bc2 = 1.0 - pow(0.999, adam_t);
            for (int i = 0; i < np; i++) {
                adam_m[i] = (float)(0.9 * adam_m[i] + 0.1 * gp[i]);
                adam_v[i] = (float)(0.999 * adam_v[i] + 0.001 * gp[i] * gp[i]);
                float mh = (float)(adam_m[i] / bc1), vh = (float)(adam_v[i] / bc2);
                wp[i] -= (float)(cfg->lr * (mh / (sqrtf(vh) + 1e-8f) + cfg->weight_decay * wp[i]));
            }
        }

        if ((ep + 1) % cfg->log_freq == 0) {
            double acc = total_examples > 0 ? (double)total_correct / total_examples * 100 : 0;
            double wacc = win_examples > 0 ? (double)win_correct / win_examples * 100 : 0;
            double wloss = win_examples > 0 ? win_loss / win_examples : 0;
            double elapsed = wall_time() - t0;
            printf("ep %5d/%d | ex=%d | acc=%.1f%% win=%.1f%% loss=%.4f | %.1f ep/s | %.1fm\n",
                   ep + 1, cfg->num_episodes, total_examples, acc, wacc, wloss,
                   (ep + 1) / fmax(elapsed, 0.001), elapsed / 60.0);
            win_correct = 0; win_examples = 0; win_loss = 0;
        }
        if ((ep + 1) % cfg->ckpt_freq == 0) {
            char p[512]; snprintf(p, sizeof(p), "%s/navigator_%d.bin", cfg->save_dir, ep + 1);
            save_navigator(p, &w, (uint32_t)(ep + 1));
        }

        graph_state_free(gs);
        free(hh); free(add_scores); free(add_edges); free(rem_scores); free(rem_edges);
    }

    char p[512]; snprintf(p, sizeof(p), "%s/navigator_final.bin", cfg->save_dir);
    save_navigator(p, &w, (uint32_t)cfg->num_episodes);
    double final_acc = total_examples > 0 ? (double)total_correct / total_examples * 100 : 0;
    double avg_loss = total_examples > 0 ? sum_loss / total_examples : 0;
    printf("Final: %d examples (pos=%d neg=%d), acc=%.1f%%, loss=%.4f\n",
           total_examples, total_pos, total_neg, final_acc, avg_loss);
    /* Feature means: pos vs neg */
    if (n_pos_feat > 0 && n_neg_feat > 0) {
        const char *fname[] = {"spec_gap", "rho"};
        printf("Feature means (pos vs neg):\n");
        for (int fi = 0; fi < 2; fi++) {
            double mp = feat_sum_pos[fi] / n_pos_feat;
            double mn = feat_sum_neg[fi] / n_neg_feat;
            double vp = feat_ssq_pos[fi] / n_pos_feat - mp*mp;
            double vn = feat_ssq_neg[fi] / n_neg_feat - mn*mn;
            double sp = vp > 0 ? sqrt(vp) : 0, sn = vn > 0 ? sqrt(vn) : 0;
            double pooled = sqrt((vp + vn) / 2);
            double t_stat = pooled > 1e-12 ? (mp - mn) / (pooled * sqrt(1.0/n_pos_feat + 1.0/n_neg_feat)) : 0;
            printf("  %12s: pos=%.4f(%.4f) neg=%.4f(%.4f) t=%.1f\n",
                   fname[fi], mp, sp, mn, sn, t_stat);
        }
    }
    printf("Saved to %s\n", cfg->save_dir);
    free(adam_m); free(adam_v);
}
