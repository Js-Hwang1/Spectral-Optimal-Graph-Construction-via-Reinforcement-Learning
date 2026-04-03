/*
 * train_climber.c -- Train the Climber agent via KL divergence against
 * exact lambda2 targets. Evaluates each candidate edge by eigensolve.
 *
 * Uses edge-level scoring: all candidate edges scored at once, with
 * KL divergence backward through climber_edge_fwd_cached/climber_edge_bwd.
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

/* Train one step: score all candidate edges, compute KL loss, backprop.
 * is_add: 1 for ADD phase (non-edges), 0 for REM phase (removable edges) */
static void train_step(const ClimberWeights *w, ClimberGrads *grad,
                       GraphState *gs, float *hh, NECache *nec,
                       int is_add, double tau, RNG *rng) {
    int n = gs->n;
    int max_edges = n * (n - 1) / 2;
    float inv_n = 1.0f / (float)n;

    /* Embed all nodes (cached for backward) */
    climber_embed_all_cached(w, gs, hh, nec);

    /* Select the correct edge scorer head weights */
    const float *W0, *b0, *W1, *b1;
    float *dW0, *db0, *dW1, *db1;
    if (is_add) {
        W0 = w->ae_w0; b0 = w->ae_b0; W1 = w->ae_w1; b1 = w->ae_b1;
        dW0 = grad->ae_w0; db0 = grad->ae_b0; dW1 = grad->ae_w1; db1 = grad->ae_b1;
    } else {
        W0 = w->re_w0; b0 = w->re_b0; W1 = w->re_w1; b1 = w->re_b1;
        dW0 = grad->re_w0; db0 = grad->re_b0; dW1 = grad->re_w1; db1 = grad->re_b1;
    }

    /* Enumerate candidate edges and score each with cached forward */
    int *edge_i = (int *)malloc((size_t)max_edges * sizeof(int));
    int *edge_j = (int *)malloc((size_t)max_edges * sizeof(int));
    float *scores = (float *)malloc((size_t)max_edges * sizeof(float));
    double *edge_val = (double *)malloc((size_t)max_edges * sizeof(double));
    ESCache *esc = (ESCache *)malloc((size_t)max_edges * sizeof(ESCache));
    int ne = 0;

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            int ok;
            if (is_add)
                ok = !gs->adj[i * n + j];
            else
                ok = gs->adj[i * n + j] && !gs->bs.bridge[i * n + j];
            if (!ok) continue;

            float cn = (float)gs->cn[i * n + j] * inv_n;
            float fv_gap = (float)(n * (gs->v2[i] - gs->v2[j]) * (gs->v2[i] - gs->v2[j]));
            double rv = gs->lpinv[i*n+i] + gs->lpinv[j*n+j] - 2.0*gs->lpinv[i*n+j];
            float reff = (float)(n * rv);

            scores[ne] = climber_edge_fwd_cached(
                hh + i * H, hh + j * H, cn, fv_gap, reff,
                W0, b0, W1, b1, &esc[ne]);

            /* Exact lambda2 for this edge (temporarily apply + eigensolve) */
            if (is_add) {
                gs->adj[i*n+j] = gs->adj[j*n+i] = 1;
                gs->deg[i]++; gs->deg[j]++;
            } else {
                gs->adj[i*n+j] = gs->adj[j*n+i] = 0;
                gs->deg[i]--; gs->deg[j]--;
            }
            edge_val[ne] = exact_lambda2(gs->adj, n);
            if (is_add) {
                gs->adj[i*n+j] = gs->adj[j*n+i] = 0;
                gs->deg[i]--; gs->deg[j]--;
            } else {
                gs->adj[i*n+j] = gs->adj[j*n+i] = 1;
                gs->deg[i]++; gs->deg[j]++;
            }

            edge_i[ne] = i;
            edge_j[ne] = j;
            ne++;
        }
    }

    if (ne == 0) goto cleanup;

    /* KL divergence: target = softmax(edge_val/tau), policy = softmax(scores) */
    {
        /* Target distribution */
        double vmax = -1e30;
        for (int k = 0; k < ne; k++) if (edge_val[k] > vmax) vmax = edge_val[k];
        double z_target = 0;
        for (int k = 0; k < ne; k++) z_target += exp((edge_val[k] - vmax) / tau);

        /* Policy distribution */
        float smax = -1e30f;
        for (int k = 0; k < ne; k++) if (scores[k] > smax) smax = scores[k];
        double z_policy = 0;
        for (int k = 0; k < ne; k++) z_policy += exp((double)(scores[k] - smax));

        /* Backward through each edge */
        for (int k = 0; k < ne; k++) {
            double p_policy = exp((double)(scores[k] - smax)) / z_policy;
            double p_target = exp((edge_val[k] - vmax) / tau) / z_target;
            float d_score = (float)(p_policy - p_target);

            float d_hi[H], d_hj[H];
            memset(d_hi, 0, sizeof(d_hi));
            memset(d_hj, 0, sizeof(d_hj));
            climber_edge_bwd(&esc[k], W0, W1, d_score, dW0, db0, dW1, db1, d_hi, d_hj);
            climber_embed_bwd(w, &nec[edge_i[k]], d_hi, grad);
            climber_embed_bwd(w, &nec[edge_j[k]], d_hj, grad);
        }
    }

    /* Apply greedy action to graph state for next step */
    {
        int best = 0;
        for (int k = 1; k < ne; k++) if (scores[k] > scores[best]) best = k;
        int bi = edge_i[best], bj = edge_j[best];
        if (is_add) graph_state_add_edge(gs, bi, bj);
        else graph_state_rem_edge(gs, bi, bj);
    }

cleanup:
    free(edge_i);
    free(edge_j);
    free(scores);
    free(edge_val);
    free(esc);
}

void train_climber(SaddleTrainConfig *cfg) {
    ClimberWeights w;
    climber_init_weights(&w, cfg->seed);

    int np = (int)(sizeof(ClimberWeights) / sizeof(float));
    float *adam_m = (float *)calloc((size_t)np, sizeof(float));
    float *adam_v = (float *)calloc((size_t)np, sizeof(float));
    int adam_t = 0;

    /* Ensure save directory exists */
    char cmd[512]; snprintf(cmd, sizeof(cmd), "mkdir -p %s", cfg->save_dir);
    system(cmd);

    RNG rng; rng_init(&rng, cfg->seed);
    double t0 = wall_time();

    for (int ep = 0; ep < cfg->num_episodes; ep++) {
        int n = cfg->min_n + rng_int(&rng, cfg->max_n - cfg->min_n + 1);
        int max_m = n * (n - 1) / 2;
        int m = (n - 1) + rng_int(&rng, max_m - (n - 1) + 1);

        /* Build random graph */
        uint8_t *adj = (uint8_t *)calloc((size_t)(n * n), 1);
        int *deg = (int *)calloc((size_t)n, sizeof(int));
        build_ring_random(adj, deg, n, m, &rng);

        GraphState *gs = graph_state_alloc(n);
        graph_state_init(gs, adj, n, m);
        free(adj); free(deg);

        /* Allocate per-episode buffers */
        float *hh = (float *)malloc((size_t)(n * H) * sizeof(float));
        NECache *nec = (NECache *)malloc((size_t)n * sizeof(NECache));

        ClimberGrads grad;
        memset(&grad, 0, sizeof(grad));
        int total_steps = 0;

        int num_epochs = cfg->budget_mult * n;
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            /* ADD phase */
            train_step(&w, &grad, gs, hh, nec, 1, cfg->tau_kl, &rng);
            total_steps++;

            /* REM phase */
            train_step(&w, &grad, gs, hh, nec, 0, cfg->tau_kl, &rng);
            total_steps++;
        }

        if (total_steps == 0) { graph_state_free(gs); free(hh); free(nec); continue; }

        /* Gradient normalize + clip + Adam */
        float *gp = (float *)&grad;
        float gnorm = 0;
        for (int i = 0; i < np; i++) gnorm += gp[i] * gp[i];
        gnorm = sqrtf(gnorm);
        float scale = 1.0f / (float)total_steps;
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

        /* Logging */
        if ((ep + 1) % cfg->log_freq == 0) {
            double l2 = exact_lambda2(gs->adj, gs->n);
            double elapsed = wall_time() - t0;
            printf("ep %5d/%d | l2=%.3f | n=%d m=%d | %.1f ep/s | %.1fm\n",
                   ep + 1, cfg->num_episodes, l2, n, m,
                   (ep + 1) / fmax(elapsed, 0.001), elapsed / 60.0);
        }

        /* Checkpoint */
        if ((ep + 1) % cfg->ckpt_freq == 0) {
            char p[512]; snprintf(p, sizeof(p), "%s/climber_%d.bin", cfg->save_dir, ep + 1);
            save_climber(p, &w, (uint32_t)(ep + 1));
        }

        graph_state_free(gs);
        free(hh); free(nec);
    }

    /* Final save */
    char p[512]; snprintf(p, sizeof(p), "%s/climber_final.bin", cfg->save_dir);
    save_climber(p, &w, (uint32_t)cfg->num_episodes);
    printf("Saved to %s\n", cfg->save_dir);

    free(adam_m); free(adam_v);
}
