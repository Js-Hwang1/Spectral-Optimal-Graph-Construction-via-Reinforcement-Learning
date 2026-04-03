/*
 * episode.c -- Main inference loop for saddle-aware graph construction.
 *
 * Alternates ADD/REM phases using the climber for greedy scoring and
 * the navigator for saddle detection + DFS backtracking.
 */

#include "saddle.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

#define H  SD_H

double saddle_episode(const ClimberWeights *cw, const NavigatorWeights *nw,
                      int n, int m, int budget_mult, float nav_threshold,
                      uint64_t seed) {
    RNG rng;
    rng_init(&rng, seed);

    /* Build initial graph: ring + random edges */
    size_t nn = (size_t)n * n;
    uint8_t *init_adj = (uint8_t *)calloc(nn, 1);
    int *init_deg = (int *)calloc((size_t)n, sizeof(int));
    build_ring_random(init_adj, init_deg, n, m, &rng);

    /* Initialize full graph state (L+, Lanczos, tri/snd/cn/bridges) */
    GraphState *gs = graph_state_alloc(n);
    graph_state_init(gs, init_adj, n, m);

    /* Track best graph */
    double best_l2 = exact_lambda2(gs->adj, n);
    uint8_t *best_adj = (uint8_t *)malloc(nn);
    memcpy(best_adj, gs->adj, nn);

    /* Cycle detection: lambda2 stagnation */
    double prev_l2 = best_l2;
    int stagnant_count = 0;
    #define STAGNANT_THRESH 3  /* steps without lambda2 change = leaf */

    /* Navigator feature state */
    NavFeatures prev_nf, cur_nf;
    memset(&prev_nf, 0, sizeof(prev_nf));
    memset(&cur_nf, 0, sizeof(cur_nf));

    /* Working buffers -- heap-allocate edge arrays since n*(n-1)/2 can be large */
    int max_edges = n * (n - 1) / 2;
    float *hh = (float *)malloc((size_t)n * H * sizeof(float));
    float *add_scores = (float *)malloc((size_t)max_edges * sizeof(float));
    float *rem_scores = (float *)malloc((size_t)max_edges * sizeof(float));
    int (*add_edges)[2] = (int (*)[2])malloc((size_t)max_edges * 2 * sizeof(int));
    int (*rem_edges)[2] = (int (*)[2])malloc((size_t)max_edges * 2 * sizeof(int));
    float saddle_scores[SD_MAX_CAND];

    /* L+ refresh counter */
    int lpinv_refresh_counter = 0;

    /* DFS backtracking stack */
    SaddleStack stack;
    stack_init(&stack);

    int total_steps = budget_mult * 4 * n;
    int backtracked = 0;
    int bt_add_i = -1, bt_add_j = -1;
    int n_saddle_fires = 0, n_greedy_steps = 0;

    for (int step = 0; step < total_steps; step++) {
        int add_src, add_tgt;

        if (backtracked) {
            /* After backtracking: skip ADD scoring, use the candidate's edge */
            add_src = bt_add_i;
            add_tgt = bt_add_j;
            backtracked = 0;

            /* Still need embeddings for the REM phase that follows */
            climber_embed_all(cw, gs, hh);
        } else {
            /* ----- ADD phase: score all non-edges and pick ----- */
            climber_embed_all(cw, gs, hh);

            int n_add = climber_score_add_edges(cw, hh, gs, add_scores, add_edges);
            if (n_add == 0) break;
            int add_pick = climber_greedy_pick(add_scores, n_add);
            add_src = add_edges[add_pick][0];
            add_tgt = add_edges[add_pick][1];

            /* ----- Saddle detection: entropy-based ----- */
            navigator_compute_features(add_scores, n_add, NULL, 0,
                                       gs, prev_nf.valid ? &prev_nf : NULL,
                                       &cur_nf);

            /* Detect saddle via navigator (entropy + density -> threshold) */
            float det_score = navigator_score_candidate(nw, &cur_nf, NULL, 0, 0);
            n_greedy_steps++;
            if (det_score > nav_threshold && n_add >= 2) {
                n_saddle_fires++;
                /* SADDLE: build top-k candidates and GOD-rank by rollout */
                EdgeCandidate nav_cands[SD_MAX_CAND];
                int n_nav_cands = 0;
                uint8_t *used = (uint8_t *)calloc((size_t)n_add, 1);
                for (int k = 0; k < SD_MAX_CAND && k < n_add; k++) {
                    int best = -1;
                    for (int e = 0; e < n_add; e++) {
                        if (used[e]) continue;
                        if (best < 0 || add_scores[e] > add_scores[best]) best = e;
                    }
                    if (best < 0) break;
                    used[best] = 1;
                    nav_cands[n_nav_cands].add_i = add_edges[best][0];
                    nav_cands[n_nav_cands].add_j = add_edges[best][1];
                    nav_cands[n_nav_cands].score = add_scores[best];
                    n_nav_cands++;
                }
                free(used);

                /* GOD-style exhaustive ranking: try each branch to leaf */
                int god_best = 0; double god_best_l2 = -1e30;
                for (int k = 0; k < n_nav_cands; k++) {
                    GraphState *clone = graph_state_clone(gs);
                    int ei = nav_cands[k].add_i, ej = nav_cands[k].add_j;
                    graph_state_add_edge(clone, ei < ej ? ei : ej, ei < ej ? ej : ei);

                    /* Greedy REM on clone */
                    climber_embed_all(cw, clone, hh);
                    int nr = climber_score_rem_edges(cw, hh, clone, rem_scores, rem_edges);
                    if (nr > 0) {
                        int rp = climber_greedy_pick(rem_scores, nr);
                        graph_state_rem_edge(clone, rem_edges[rp][0], rem_edges[rp][1]);
                    }

                    /* Greedy rollout to stagnation */
                    double prev_rl2 = -1; int rstag = 0;
                    for (int rs = 0; rs < 4 * n; rs++) {
                        climber_embed_all(cw, clone, hh);
                        int rna = climber_score_add_edges(cw, hh, clone, add_scores, add_edges);
                        if (rna == 0) break;
                        int rp = climber_greedy_pick(add_scores, rna);
                        graph_state_add_edge(clone, add_edges[rp][0], add_edges[rp][1]);
                        climber_embed_all(cw, clone, hh);
                        int rnr = climber_score_rem_edges(cw, hh, clone, rem_scores, rem_edges);
                        if (rnr > 0) { rp = climber_greedy_pick(rem_scores, rnr); graph_state_rem_edge(clone, rem_edges[rp][0], rem_edges[rp][1]); }
                        double rl2 = exact_lambda2(clone->adj, n);
                        if (fabs(rl2 - prev_rl2) < 1e-6) rstag++; else rstag = 0;
                        prev_rl2 = rl2;
                        if (rstag >= 3) break;
                    }

                    double leaf = exact_lambda2(clone->adj, n);
                    if (leaf > god_best_l2) { god_best_l2 = leaf; god_best = k; }
                    graph_state_free(clone);
                }

                /* Save state and pick GOD's best */
                stack_push(&stack, gs, nav_cands, n_nav_cands);
                add_src = nav_cands[god_best].add_i;
                add_tgt = nav_cands[god_best].add_j;
            }
        }

        /* Apply ADD edge (canonical order) */
        int u = add_src < add_tgt ? add_src : add_tgt;
        int v = add_src < add_tgt ? add_tgt : add_src;
        graph_state_add_edge(gs, u, v);

        /* ----- REM phase: re-embed, score all removable edges, pick ----- */
        climber_embed_all(cw, gs, hh);

        int rem_src = -1, rem_tgt = -1;
        int n_rem = climber_score_rem_edges(cw, hh, gs, rem_scores, rem_edges);
        if (n_rem == 0) goto next_step;
        int rem_pick = climber_greedy_pick(rem_scores, n_rem);
        rem_src = rem_edges[rem_pick][0];
        rem_tgt = rem_edges[rem_pick][1];

        /* Apply REM edge (canonical order) */
        u = rem_src < rem_tgt ? rem_src : rem_tgt;
        v = rem_src < rem_tgt ? rem_tgt : rem_src;
        graph_state_rem_edge(gs, u, v);

    next_step:
        /* Track best lambda2 every step (Lanczos estimate is cheap) */
        if (gs->v2_lam > best_l2) {
            double check = exact_lambda2(gs->adj, n);
            if (check > best_l2) {
                best_l2 = check;
                memcpy(best_adj, gs->adj, (size_t)nn);
            }
        }

        /* ----- Leaf detection: lambda2 stagnation ----- */
        {
            double cur_l2 = exact_lambda2(gs->adj, n);
            if (fabs(cur_l2 - prev_l2) < 1e-6)
                stagnant_count++;
            else
                stagnant_count = 0;
            prev_l2 = cur_l2;

            if (stagnant_count >= STAGNANT_THRESH) {
                /* Leaf: local optimum (lambda2 hasn't changed) */
                if (cur_l2 > best_l2) {
                    best_l2 = cur_l2;
                    memcpy(best_adj, gs->adj, nn);
                }

                /* Backtrack to nearest saddle with untried branches */
                EdgeCandidate next_cand;
                if (stack_backtrack(&stack, gs, &next_cand)) {
                    backtracked = 1;
                    bt_add_i = next_cand.add_i;
                    bt_add_j = next_cand.add_j;
                    stagnant_count = 0;
                    prev_l2 = -1;  /* force re-evaluate after restore */
                    lpinv_refresh_counter = 0;
                } else {
                    break;  /* stack empty */
                }
                goto skip_update;
            }
        }

        /* Save current nav features as prev for derivative computation */
        prev_nf = cur_nf;

    skip_update:
        /* Periodic L+ refresh to prevent numerical drift */
        lpinv_refresh_counter++;
        if (lpinv_refresh_counter >= n) {
            lpinv_refresh(gs);
            lpinv_refresh_counter = 0;
        }
    }

    /* Final check: current graph might be the best */
    double final_l2 = exact_lambda2(gs->adj, n);
    if (final_l2 > best_l2) {
        best_l2 = final_l2;
    }

    /* Diagnostic: saddle fire rate (thread-safe via atomic would be better,
     * but for diagnostics this is fine) */
    static int g_fires = 0, g_steps = 0, g_calls = 0;
    #pragma omp critical
    {
        g_fires += n_saddle_fires;
        g_steps += n_greedy_steps;
        g_calls++;
        if (g_calls % 50 == 0) {
            printf("  [det] %d/%d steps = %.1f%% fire rate (%d episodes)\n",
                   g_fires, g_steps, g_steps > 0 ? 100.0 * g_fires / g_steps : 0, g_calls);
        }
    }

    /* Cleanup */
    stack_free(&stack);
    graph_state_free(gs);
    free(init_adj);
    free(init_deg);
    free(best_adj);
    free(hh);
    free(add_scores);
    free(rem_scores);
    free(add_edges);
    free(rem_edges);

    return best_l2;
}
