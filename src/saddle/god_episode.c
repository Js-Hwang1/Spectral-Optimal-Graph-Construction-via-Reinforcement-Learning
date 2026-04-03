/*
 * god_episode.c -- Exhaustive branch evaluation ("God navigator").
 *
 * Uses the trained climber for scoring but replaces the navigator with
 * exhaustive evaluation: at each step, try top-k alternative ADD edges,
 * run each to stagnation, pick the one reaching the highest lambda2.
 *
 * This measures the CEILING of the climber -- the best it can achieve
 * with perfect navigation.
 */

#include "saddle.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#define H SD_H

/* Run the climber greedily from current state until lambda2 stagnates.
 * Returns the leaf lambda2. Does NOT modify the input gs (works on a clone). */
static double greedy_to_leaf(const ClimberWeights *cw, GraphState *gs, int max_steps) {
    int n = gs->n;
    int max_edges = n * (n - 1) / 2;
    float *hh = (float *)malloc((size_t)(n * H) * sizeof(float));
    float *scores = (float *)malloc((size_t)max_edges * sizeof(float));
    int (*edges)[2] = (int (*)[2])malloc((size_t)max_edges * 2 * sizeof(int));

    double prev_l2 = -1; int stagnant = 0;

    for (int s = 0; s < max_steps; s++) {
        /* ADD */
        climber_embed_all(cw, gs, hh);
        int na = climber_score_add_edges(cw, hh, gs, scores, edges);
        if (na == 0) break;
        int pick = climber_greedy_pick(scores, na);
        graph_state_add_edge(gs, edges[pick][0], edges[pick][1]);

        /* REM */
        climber_embed_all(cw, gs, hh);
        int nr = climber_score_rem_edges(cw, hh, gs, scores, edges);
        if (nr > 0) {
            pick = climber_greedy_pick(scores, nr);
            graph_state_rem_edge(gs, edges[pick][0], edges[pick][1]);
        }

        /* Stagnation */
        double cur = exact_lambda2(gs->adj, n);
        if (fabs(cur - prev_l2) < 1e-6) stagnant++;
        else stagnant = 0;
        prev_l2 = cur;
        if (stagnant >= 3) break;
    }

    double leaf = exact_lambda2(gs->adj, n);
    free(hh); free(scores); free(edges);
    return leaf;
}

double god_episode(const ClimberWeights *cw, int n, int m,
                   int budget_mult, int n_branches, uint64_t seed) {
    RNG rng; rng_init(&rng, seed);
    size_t nn = (size_t)n * n;
    int max_edges = n * (n - 1) / 2;

    uint8_t *init_adj = (uint8_t *)calloc(nn, 1);
    int *init_deg = (int *)calloc((size_t)n, sizeof(int));
    build_ring_random(init_adj, init_deg, n, m, &rng);

    GraphState *gs = graph_state_alloc(n);
    graph_state_init(gs, init_adj, n, m);
    free(init_adj); free(init_deg);

    double best_l2 = exact_lambda2(gs->adj, n);

    float *hh = (float *)malloc((size_t)(n * H) * sizeof(float));
    float *add_scores = (float *)malloc((size_t)max_edges * sizeof(float));
    int (*add_edges)[2] = (int (*)[2])malloc((size_t)max_edges * 2 * sizeof(int));
    float *rem_scores = (float *)malloc((size_t)max_edges * sizeof(float));
    int (*rem_edges)[2] = (int (*)[2])malloc((size_t)max_edges * 2 * sizeof(int));

    double prev_l2 = best_l2; int stagnant = 0;
    int total_steps = budget_mult * 4 * n;
    NavFeatures prev_nf; memset(&prev_nf, 0, sizeof(prev_nf));
    float prev_gap12 = 0;

    for (int step = 0; step < total_steps; step++) {
        climber_embed_all(cw, gs, hh);
        int na = climber_score_add_edges(cw, hh, gs, add_scores, add_edges);
        if (na == 0) break;

        /* Find top-k ADD edges by score */
        int top_k = n_branches < na ? n_branches : na;
        int top_idx[16]; /* max branches */
        uint8_t *used = (uint8_t *)calloc((size_t)na, 1);
        for (int k = 0; k < top_k; k++) {
            int best = -1;
            for (int e = 0; e < na; e++) {
                if (used[e]) continue;
                if (best < 0 || add_scores[e] > add_scores[best]) best = e;
            }
            top_idx[k] = best; used[best] = 1;
        }
        free(used);

        /* Try each top-k ADD edge: clone, apply, run to leaf */
        int best_branch = 0;
        double best_branch_l2 = -1e30;
        double branch_l2[16];
        for (int k = 0; k < top_k; k++) {
            GraphState *clone = graph_state_clone(gs);
            int ei = add_edges[top_idx[k]][0], ej = add_edges[top_idx[k]][1];
            graph_state_add_edge(clone, ei, ej);

            /* Do one greedy REM from cloned state */
            climber_embed_all(cw, clone, hh);
            int nr = climber_score_rem_edges(cw, hh, clone, rem_scores, rem_edges);
            if (nr > 0) {
                int rp = climber_greedy_pick(rem_scores, nr);
                graph_state_rem_edge(clone, rem_edges[rp][0], rem_edges[rp][1]);
            }

            double leaf = greedy_to_leaf(cw, clone, 4 * n);
            branch_l2[k] = leaf;
            if (leaf > best_branch_l2) {
                best_branch_l2 = leaf;
                best_branch = k;
            }
            graph_state_free(clone);
        }

        /* Accumulate GOD step statistics with candidate features */
        {
            double greedy_l2 = branch_l2[0];
            double gain = best_branch_l2 - greedy_l2;
            int is_saddle = (gain > 0.01);

            /* Compute candidate features */
            NavFeatures nf; memset(&nf, 0, sizeof(nf));
            navigator_compute_features(add_scores, na, NULL, 0, gs, NULL, &nf);

            /* F0: entropy (existing) */
            /* F1: l2_n (existing) */
            /* F2: rho (existing) */
            /* Snapshot features */
            float f3 = (top_k >= 2) ? add_scores[top_idx[0]] - add_scores[top_idx[1]] : 0;
            float f4 = (top_k >= 2) ? add_scores[top_idx[0]] - add_scores[top_idx[top_k-1]] : 0;
            float f5 = (float)step / (float)total_steps;
            int gi = add_edges[top_idx[0]][0], gj = add_edges[top_idx[0]][1];
            double dv = gs->v2[gi] - gs->v2[gj];
            float f6 = (float)(n * dv * dv);
            float f8 = (float)((gs->v3_lam - gs->v2_lam) / n);

            /* Top-K entropy: softmax over top-6 only, normalized by log(K) */
            float topK_ent = 0;
            {
                int K6 = top_k < 6 ? top_k : 6;
                if (K6 >= 2) {
                    float mx = add_scores[top_idx[0]];
                    double zz = 0;
                    for (int kk = 0; kk < K6; kk++) zz += exp((double)(add_scores[top_idx[kk]] - mx));
                    double lzz = log(zz);
                    double ee = 0;
                    for (int kk = 0; kk < K6; kk++) {
                        double lp = (double)(add_scores[top_idx[kk]] - mx) - lzz;
                        double pp = exp(lp);
                        if (pp > 1e-12) ee -= pp * lp;
                    }
                    double lK = log((double)K6);
                    topK_ent = (lK > 1e-12) ? (float)(ee / lK) : 0;
                }
            }
            float d_topK = (prev_nf.valid) ? topK_ent - prev_nf.d_entropy : 0; /* reuse field for prev topK */

            /* Derivative features */
            float d_entropy = (prev_nf.valid) ? nf.entropy - prev_nf.entropy : 0;

            /* Candidate invariant features */
            double mean_deg = (gs->m > 0) ? 2.0 * gs->m / n : 1.0;
            float l2_deg = (float)(gs->v2_lam / mean_deg);  /* lambda2/mean_degree */
            float spec_rel = (gs->v2_lam > 1e-6) ?
                (float)((gs->v3_lam - gs->v2_lam) / gs->v2_lam) : 0; /* relative spectral gap */
            float l2_prog = (float)(gs->v2_lam / (nf.mean_deg_n * n + 1e-6)); /* l2/(rho*n) */

            #define NF 12
            float feats[NF] = {nf.entropy, nf.l2_n, nf.mean_deg_n, f5, f8,
                               d_entropy, l2_deg, spec_rel, f3, f4, f6, l2_prog};

            /* Bin by rho: [0.1,0.3), [0.3,0.5), [0.5,0.8) */
            float cur_rho = nf.mean_deg_n;
            int rbin = (cur_rho < 0.3f) ? 0 : (cur_rho < 0.5f) ? 1 : 2;

            /* Features to track per rho bin */
            float gap12 = (top_k >= 2) ? add_scores[top_idx[0]] - add_scores[top_idx[1]] : 0;
            float gap1K = (top_k >= 2) ? add_scores[top_idx[0]] - add_scores[top_idx[top_k-1]] : 0;
            /* raw score std of top-6 */
            float topK_mean = 0;
            int K6 = top_k < 6 ? top_k : 6;
            for (int kk = 0; kk < K6; kk++) topK_mean += add_scores[top_idx[kk]];
            topK_mean /= K6;
            float topK_std = 0;
            for (int kk = 0; kk < K6; kk++) {
                float d = add_scores[top_idx[kk]] - topK_mean;
                topK_std += d * d;
            }
            topK_std = sqrtf(topK_std / K6);
            float d_gap12_val = (prev_nf.valid) ? gap12 - prev_gap12 : 0;

            #define NBINS 3
            #define NFEAT 7
            static int g_total[NBINS] = {0}, g_ng[NBINS] = {0};
            static double ng_s[NBINS][NFEAT] = {{0}}, ng_s2[NBINS][NFEAT] = {{0}};
            static double gr_s[NBINS][NFEAT] = {{0}}, gr_s2[NBINS][NFEAT] = {{0}};
            static int g_print = 0;

            /* z-scored gap12: gap12 / std(all_scores) */
            double gs_mean = 0;
            for (int ii = 0; ii < na; ii++) gs_mean += add_scores[ii];
            gs_mean /= na;
            double gs_var = 0;
            for (int ii = 0; ii < na; ii++) { double dd = add_scores[ii] - gs_mean; gs_var += dd*dd; }
            gs_var /= na;
            double gs_std = (gs_var > 1e-12) ? sqrt(gs_var) : 1.0;
            float z_gap12 = (float)(gap12 / gs_std);
            float z_gap1K = (float)(gap1K / gs_std);

            float fvals[NFEAT] = {nf.entropy, z_gap12, z_gap1K, d_entropy, f8, gap12, topK_std};

            #pragma omp critical
            {
                g_total[rbin]++;
                if (is_saddle) {
                    g_ng[rbin]++;
                    for (int ff = 0; ff < NFEAT; ff++) { ng_s[rbin][ff] += fvals[ff]; ng_s2[rbin][ff] += fvals[ff]*fvals[ff]; }
                } else {
                    for (int ff = 0; ff < NFEAT; ff++) { gr_s[rbin][ff] += fvals[ff]; gr_s2[rbin][ff] += fvals[ff]*fvals[ff]; }
                }
                g_print++;

                if (g_print % 200 == 0) {
                    const char *bname[NBINS] = {"rho<0.3", "0.3-0.5", "rho>0.5"};
                    const char *fname[NFEAT] = {"z_ent", "z_gap12", "z_gap1K", "d_z_ent", "spec_gap", "raw_gap12", "raw_std"};
                    printf("  [GOD by rho] n=%d\n", n);
                    for (int b = 0; b < NBINS; b++) {
                        if (g_ng[b] < 3 || g_total[b] - g_ng[b] < 3) continue;
                        int gok = g_total[b] - g_ng[b];
                        printf("    %s: %d steps (%d sad, %.0f%%)\n", bname[b], g_total[b], g_ng[b], 100.0*g_ng[b]/g_total[b]);
                        for (int ff = 0; ff < NFEAT; ff++) {
                            double ms = ng_s[b][ff]/g_ng[b], mo = gr_s[b][ff]/gok;
                            double vs = ng_s2[b][ff]/g_ng[b] - ms*ms;
                            double vo = gr_s2[b][ff]/gok - mo*mo;
                            double p = sqrt((vs+vo)/2);
                            double t = p>1e-12 ? (ms-mo)/(p*sqrt(1.0/g_ng[b]+1.0/gok)) : 0;
                            printf("      %-10s sad=%.4f ok=%.4f t=%5.1f\n", fname[ff], ms, mo, t);
                        }
                    }
                }
            }
            #undef NBINS
            #undef NFEAT
            #undef NBINS
            #undef NF
            /* Save for next step's derivatives */
            prev_nf = nf;
            prev_nf.spec_gap = f8;
            prev_nf.d_entropy = topK_ent; /* stash topK for next step's d_topK */
            prev_gap12 = f3;
        }

        if (best_branch_l2 > best_l2) best_l2 = best_branch_l2;

        /* Apply the best branch's ADD */
        int ei = add_edges[top_idx[best_branch]][0];
        int ej = add_edges[top_idx[best_branch]][1];
        graph_state_add_edge(gs, ei, ej);

        /* Greedy REM */
        climber_embed_all(cw, gs, hh);
        int nr = climber_score_rem_edges(cw, hh, gs, rem_scores, rem_edges);
        if (nr > 0) {
            int rp = climber_greedy_pick(rem_scores, nr);
            graph_state_rem_edge(gs, rem_edges[rp][0], rem_edges[rp][1]);
        }

        /* Track best + stagnation */
        double cur = exact_lambda2(gs->adj, n);
        if (cur > best_l2) best_l2 = cur;

        if (fabs(cur - prev_l2) < 1e-6) stagnant++;
        else stagnant = 0;
        prev_l2 = cur;
        if (stagnant >= 3) break;
    }

    graph_state_free(gs);
    free(hh); free(add_scores); free(add_edges);
    free(rem_scores); free(rem_edges);
    return best_l2;
}
