/*
 * saddle_main.c -- Saddle tree DFS for lambda2 maximization.
 *
 * Uses FV signals to detect saddles (score std) and navigate (FV gap).
 * Saves saddle states on a stack, backtracks on leaf (local optimum).
 *
 * Per-step: O(N^2) warm Lanczos + O(N) FV scoring.
 * No exhaustive pair evaluation. No MLP. No training.
 */

#include "crl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include <sys/time.h>

static double wall_time(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

#define LANCZOS_K 15
#define MAX_STACK 64

/* Stack depth limit */

/* ========================================================================== */
/* FV-scored swap: add highest-gap non-edge, remove lowest-gap edge            */
/* (apply_edge_candidate defined after EdgeCandidate below) */

/* ========================================================================== */
/* Compute node scores and detect saddle                                       */
/* Returns: 0=climb, 1=saddle, 2=leaf                                          */
/* ========================================================================== */

/*
 * Edge-level saddle detection.
 * Score = FV gap of non-edge minus FV gap of best removable edge for that node.
 * This approximates the net lambda2 change of a rewire without eigensolve.
 *
 * Candidate = (add_node, add_tgt, rem_u, rem_v, net_fv_gap)
 * Saddle = top candidates have similar net_fv_gap.
 */

typedef struct { int add_i, add_j, rem_i, rem_j; double score; } EdgeCandidate;

static void apply_edge_candidate(uint8_t *adj, int *deg, int n,
                                  const EdgeCandidate *c) {
    adj[c->add_i * n + c->add_j] = adj[c->add_j * n + c->add_i] = 1;
    deg[c->add_i]++; deg[c->add_j]++;
    adj[c->rem_i * n + c->rem_j] = adj[c->rem_j * n + c->rem_i] = 0;
    deg[c->rem_i]--; deg[c->rem_j]--;
}

static int cmp_edge_cand_desc(const void *a, const void *b) {
    double d = ((const EdgeCandidate *)b)->score - ((const EdgeCandidate *)a)->score;
    return d < 0 ? -1 : d > 0 ? 1 : 0;
}

static int detect_and_score(uint8_t *adj, int *deg, int n, const double *v2,
                             const uint8_t *bmask, double tau_rel,
                             EdgeCandidate *out_cands, int *out_n, int max_cands) {
    /* Build top-k add edges (highest FV gap non-edges) */
    typedef struct { int i, j; double gap; } EG;
    int max_e = n * (n-1) / 2;
    EG *adds = (EG *)malloc((size_t)max_e * sizeof(EG));
    EG *rems = (EG *)malloc((size_t)max_e * sizeof(EG));
    int na = 0, nr = 0;

    for (int i = 0; i < n; i++)
        for (int j = i+1; j < n; j++) {
            double g = (v2[i] - v2[j]) * (v2[i] - v2[j]);
            if (!adj[i*n+j]) { adds[na].i=i; adds[na].j=j; adds[na].gap=g; na++; }
            else if (!bmask[i*n+j]) { rems[nr].i=i; rems[nr].j=j; rems[nr].gap=g; nr++; }
        }

    if (na == 0 || nr == 0) { *out_n = 0; free(adds); free(rems); return 2; }

    /* Sort adds desc, rems asc */
    for (int i = 0; i < na-1 && i < max_cands; i++) {
        int b = i;
        for (int j = i+1; j < na; j++) if (adds[j].gap > adds[b].gap) b = j;
        if (b != i) { EG t = adds[i]; adds[i] = adds[b]; adds[b] = t; }
    }
    for (int i = 0; i < nr-1 && i < max_cands; i++) {
        int b = i;
        for (int j = i+1; j < nr; j++) if (rems[j].gap < rems[b].gap) b = j;
        if (b != i) { EG t = rems[i]; rems[i] = rems[b]; rems[b] = t; }
    }

    /* Build edge-level candidates: top_add x top_rem */
    int ka = max_cands < na ? max_cands : na;
    int kr = max_cands < nr ? max_cands : nr;
    int nc = 0;

    for (int ai = 0; ai < ka; ai++) {
        for (int ri = 0; ri < kr; ri++) {
            if (adds[ai].i == rems[ri].i && adds[ai].j == rems[ri].j) continue;
            out_cands[nc].add_i = adds[ai].i;
            out_cands[nc].add_j = adds[ai].j;
            out_cands[nc].rem_i = rems[ri].i;
            out_cands[nc].rem_j = rems[ri].j;
            out_cands[nc].score = adds[ai].gap - rems[ri].gap; /* net FV benefit */
            nc++;
        }
    }

    qsort(out_cands, (size_t)nc, sizeof(EdgeCandidate), cmp_edge_cand_desc);
    *out_n = nc;

    free(adds); free(rems);

    if (nc < 2) return (nc == 0) ? 2 : 0;
    if (out_cands[0].score < 1e-10) return 2; /* no positive candidate */

    double rel_gap = (out_cands[0].score - out_cands[1].score) /
                      (out_cands[0].score > 1e-8 ? out_cands[0].score : 1.0);
    if (rel_gap < tau_rel) return 1; /* saddle */
    return 0; /* climb */
}

/* ========================================================================== */
/* Saddle episode: DFS with backtracking                                       */
/* ========================================================================== */

static double saddle_episode(int n, int m, int max_rounds, double tau_rel,
                              int swaps, uint64_t seed) {
    int nn = n * n;
    uint8_t *adj = (uint8_t *)calloc((size_t)nn, 1);
    int *deg = (int *)calloc((size_t)n, sizeof(int));
    RNG rng; rng_init(&rng, seed);
    build_ring_random(adj, deg, n, m, &rng);

    double *v2 = (double *)malloc((size_t)n * sizeof(double));
    double lam;
    lanczos_ext_k(adj, n, NULL, LANCZOS_K, 1, v2, &lam);

    uint8_t *best_adj = (uint8_t *)malloc((size_t)nn);
    double best_l2 = 0;
    int total_steps = 0;
    int max_steps = max_rounds * 4 * n;
    int top_k = 6; /* candidates per side for edge scoring */
    int max_ec = top_k * top_k;
    EdgeCandidate *cands = (EdgeCandidate *)malloc((size_t)max_ec * sizeof(EdgeCandidate));
    uint8_t *bmask = (uint8_t *)malloc((size_t)nn);

    /* Saddle stack: store adj/deg/v2 + candidate list at each fork */
    /* Simple approach: store up to MAX_STACK saddle states */
    typedef struct {
        uint8_t *s_adj; int *s_deg; double *s_v2;
        EdgeCandidate *s_cands; int s_nc; int s_tried;
    } SFrame;
    SFrame stack[MAX_STACK]; int sp = 0;

    while (total_steps < max_steps) {
        lanczos_ext_k(adj, n, v2, LANCZOS_K, 1, v2, &lam);
        find_bridges(adj, n, bmask);

        int nc;
        int state = detect_and_score(adj, deg, n, v2, bmask, tau_rel,
                                      cands, &nc, top_k);

        if (state == 0 && nc > 0) {
            /* CLIMB: apply best candidate */
            apply_edge_candidate(adj, deg, n, &cands[0]);
            lanczos_ext_k(adj, n, v2, LANCZOS_K, 1, v2, &lam);
            total_steps++;
        }
        else if (state == 1 && nc >= 2 && sp < MAX_STACK) {
            /* SADDLE: count tied, save state, take lowest-score branch (inductive bias) */
            int n_tied = 1;
            for (int i = 1; i < nc; i++) {
                double rg = (cands[0].score - cands[i].score) /
                            (cands[0].score > 1e-8 ? cands[0].score : 1.0);
                if (rg < tau_rel) n_tied++; else break;
            }
            if (n_tied > 6) n_tied = 6;

            /* Push frame */
            SFrame *f = &stack[sp];
            f->s_adj = (uint8_t *)malloc((size_t)nn);
            f->s_deg = (int *)malloc((size_t)n * sizeof(int));
            f->s_v2 = (double *)malloc((size_t)n * sizeof(double));
            f->s_cands = (EdgeCandidate *)malloc((size_t)n_tied * sizeof(EdgeCandidate));
            memcpy(f->s_adj, adj, (size_t)nn);
            memcpy(f->s_deg, deg, (size_t)n * sizeof(int));
            memcpy(f->s_v2, v2, (size_t)n * sizeof(double));
            memcpy(f->s_cands, cands, (size_t)n_tied * sizeof(EdgeCandidate));
            /* Candidates are already sorted by net_score descending from detect_and_score */
            f->s_nc = n_tied;
            f->s_tried = 1; /* we'll take first branch (lowest add_gap) now */
            sp++;

            /* Take last tied = lowest net_score (performed best empirically) */
            apply_edge_candidate(adj, deg, n, &f->s_cands[n_tied - 1]);
            lanczos_ext_k(adj, n, v2, LANCZOS_K, 1, v2, &lam);
            total_steps++;
        }
        else {
            /* LEAF or no candidates: record and backtrack */
            double leaf_l2 = exact_lambda2(adj, n);
            if (leaf_l2 > best_l2) {
                best_l2 = leaf_l2;
                memcpy(best_adj, adj, (size_t)nn);
            }

            /* Backtrack */
            int backtracked = 0;
            while (sp > 0) {
                SFrame *f = &stack[sp - 1];
                if (f->s_tried < f->s_nc) {
                    /* Restore state */
                    memcpy(adj, f->s_adj, (size_t)nn);
                    memcpy(deg, f->s_deg, (size_t)n * sizeof(int));
                    memcpy(v2, f->s_v2, (size_t)n * sizeof(double));
                    /* Take next branch from the end (ascending net_score) */
                    int idx = f->s_nc - 1 - f->s_tried;
                    apply_edge_candidate(adj, deg, n, &f->s_cands[idx]);
                    lanczos_ext_k(adj, n, v2, LANCZOS_K, 1, v2, &lam);
                    f->s_tried++;
                    total_steps++;
                    backtracked = 1;
                    break;
                }
                /* Pop exhausted frame */
                free(f->s_adj); free(f->s_deg); free(f->s_v2); free(f->s_cands);
                sp--;
            }
            if (!backtracked) {
                /* Stack empty: all uphill branches exhausted.
                 * Bidirectional: try worsening moves from best leaf,
                 * then re-ascend to find a different basin. */
                memcpy(adj, best_adj, (size_t)nn);
                memset(deg, 0, (size_t)n * sizeof(int));
                for (int ii = 0; ii < n; ii++)
                    for (int jj = 0; jj < n; jj++) deg[ii] += adj[ii * n + jj];
                lanczos_ext_k(adj, n, v2, LANCZOS_K, 1, v2, &lam);

                find_bridges(adj, n, bmask);
                int nc2;
                detect_and_score(adj, deg, n, v2, bmask, tau_rel, cands, &nc2, top_k);

                /* Try each candidate as a worsening step */
                int found_exit = 0;
                for (int ci = 0; ci < nc2; ci++) {
                    uint8_t *t_adj = (uint8_t *)malloc((size_t)nn);
                    int *t_deg = (int *)malloc((size_t)n * sizeof(int));
                    double t_v2[256];
                    memcpy(t_adj, adj, (size_t)nn);
                    memcpy(t_deg, deg, (size_t)n * sizeof(int));
                    memcpy(t_v2, v2, (size_t)n * sizeof(double));

                    apply_edge_candidate(t_adj, t_deg, n, &cands[ci]);
                    lanczos_ext_k(t_adj, n, t_v2, LANCZOS_K, 1, t_v2, &lam);

                    /* Greedy ascend from here: follow FV for a few steps */
                    for (int gs = 0; gs < 4 * n; gs++) {
                        find_bridges(t_adj, n, bmask);
                        int gnc;
                        detect_and_score(t_adj, t_deg, n, t_v2, bmask, 999.0,
                                          cands, &gnc, top_k);
                        if (gnc == 0 || cands[0].score < 1e-10) break;
                        apply_edge_candidate(t_adj, t_deg, n, &cands[0]);
                        lanczos_ext_k(t_adj, n, t_v2, LANCZOS_K, 1, t_v2, &lam);
                    }

                    double new_l2 = exact_lambda2(t_adj, n);
                    if (new_l2 > best_l2 + 1e-6) {
                        best_l2 = new_l2;
                        memcpy(best_adj, t_adj, (size_t)nn);
                        memcpy(adj, t_adj, (size_t)nn);
                        memcpy(deg, t_deg, (size_t)n * sizeof(int));
                        memcpy(v2, t_v2, (size_t)n * sizeof(double));
                        found_exit = 1;
                        free(t_adj); free(t_deg);
                        break;
                    }
                    free(t_adj); free(t_deg);
                }
                /* 2-step descent if 1-step failed */
                if (!found_exit) {
                    memcpy(adj, best_adj, (size_t)nn);
                    memset(deg, 0, (size_t)n * sizeof(int));
                    for (int ii = 0; ii < n; ii++)
                        for (int jj = 0; jj < n; jj++) deg[ii] += adj[ii * n + jj];
                    lanczos_ext_k(adj, n, v2, LANCZOS_K, 1, v2, &lam);
                    find_bridges(adj, n, bmask);
                    int nc3;
                    EdgeCandidate *cands2 = (EdgeCandidate *)malloc((size_t)max_ec * sizeof(EdgeCandidate));
                    detect_and_score(adj, deg, n, v2, bmask, tau_rel, cands, &nc3, top_k);

                    for (int c1 = 0; c1 < nc3 && c1 < 15 && !found_exit; c1++) {
                        uint8_t *t1_adj = (uint8_t *)malloc((size_t)nn);
                        int *t1_deg = (int *)malloc((size_t)n * sizeof(int));
                        double t1_v2[256];
                        memcpy(t1_adj, adj, (size_t)nn);
                        memcpy(t1_deg, deg, (size_t)n * sizeof(int));
                        memcpy(t1_v2, v2, (size_t)n * sizeof(double));
                        apply_edge_candidate(t1_adj, t1_deg, n, &cands[c1]);
                        lanczos_ext_k(t1_adj, n, t1_v2, LANCZOS_K, 1, t1_v2, &lam);
                        find_bridges(t1_adj, n, bmask);
                        int nc4;
                        detect_and_score(t1_adj, t1_deg, n, t1_v2, bmask, tau_rel, cands2, &nc4, top_k);

                        for (int c2 = 0; c2 < nc4 && c2 < 15; c2++) {
                            uint8_t *t2_adj = (uint8_t *)malloc((size_t)nn);
                            int *t2_deg = (int *)malloc((size_t)n * sizeof(int));
                            double t2_v2[256];
                            memcpy(t2_adj, t1_adj, (size_t)nn);
                            memcpy(t2_deg, t1_deg, (size_t)n * sizeof(int));
                            memcpy(t2_v2, t1_v2, (size_t)n * sizeof(double));
                            apply_edge_candidate(t2_adj, t2_deg, n, &cands2[c2]);
                            lanczos_ext_k(t2_adj, n, t2_v2, LANCZOS_K, 1, t2_v2, &lam);

                            /* Greedy ascend */
                            for (int gs = 0; gs < 4 * n; gs++) {
                                find_bridges(t2_adj, n, bmask);
                                int gnc;
                                detect_and_score(t2_adj, t2_deg, n, t2_v2, bmask, 999.0,
                                                  cands2, &gnc, top_k);
                                if (gnc == 0 || cands2[0].score < 1e-10) break;
                                apply_edge_candidate(t2_adj, t2_deg, n, &cands2[0]);
                                lanczos_ext_k(t2_adj, n, t2_v2, LANCZOS_K, 1, t2_v2, &lam);
                            }
                            double new_l2 = exact_lambda2(t2_adj, n);
                            if (new_l2 > best_l2 + 1e-6) {
                                best_l2 = new_l2;
                                memcpy(best_adj, t2_adj, (size_t)nn);
                                memcpy(adj, t2_adj, (size_t)nn);
                                memcpy(deg, t2_deg, (size_t)n * sizeof(int));
                                memcpy(v2, t2_v2, (size_t)n * sizeof(double));
                                found_exit = 1;
                            }
                            free(t2_adj); free(t2_deg);
                            if (found_exit) break;
                        }
                        free(t1_adj); free(t1_deg);
                    }
                    free(cands2);
                }
                if (!found_exit) break; /* truly stuck */
                /* If found exit, continue the outer loop from new state */
            }
        }
    }

    double final_l2 = exact_lambda2(adj, n);
    if (final_l2 > best_l2) best_l2 = final_l2;

    /* Cleanup stack */
    for (int i = 0; i < sp; i++) {
        free(stack[i].s_adj); free(stack[i].s_deg);
        free(stack[i].s_v2); free(stack[i].s_cands);
    }
    free(cands); free(bmask);

    free(adj); free(deg); free(v2); free(best_adj);
    return best_l2;
}

/* ========================================================================== */
/* Baselines + Main                                                            */
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
    int n_values[64], num_n = 0;
    char *bl_path = NULL; char *csv_path = NULL;
    int max_rounds = 10; double tau_rel = 0.07;
    int swaps = 3;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            char *tok = strtok(argv[++i], ",");
            while (tok && num_n < 64) { n_values[num_n++] = atoi(tok); tok = strtok(NULL, ","); }
        }
        else if (strcmp(argv[i], "--baselines") == 0 && i + 1 < argc) bl_path = argv[++i];
        else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) csv_path = argv[++i];
        else if (strcmp(argv[i], "--rounds") == 0 && i + 1 < argc) max_rounds = atoi(argv[++i]);
        else if (strcmp(argv[i], "--tau") == 0 && i + 1 < argc) tau_rel = atof(argv[++i]);
        else if (strcmp(argv[i], "--swaps") == 0 && i + 1 < argc) swaps = atoi(argv[++i]);
    }

    if (num_n == 0) {
        printf("Usage: %s --n 8,16 --baselines bl.csv [--rounds 10] [--tau 0.07] [--swaps 3]\n", argv[0]);
        return 1;
    }

    if (bl_path) load_bl(bl_path);

    FILE *csv_f = NULL;
    if (csv_path) { csv_f = fopen(csv_path, "w"); if (csv_f) fprintf(csv_f, "m,score\n"); }

    printf("Saddle DFS | rounds=%d tau=%.3f swaps=%d threads=%d\n",
           max_rounds, tau_rel, swaps, omp_get_max_threads());

    for (int ni = 0; ni < num_n; ni++) {
        int n = n_values[ni]; int mx = n * (n - 1) / 2;
        int *m_list = (int *)malloc((size_t)(mx + 1) * sizeof(int)); int nm = 0;
        for (int mv = n - 1; mv <= mx; mv++) {
            BL *b = find_bl(n, mv);
            if (b && b->best > 0) m_list[nm++] = mv;
            else if (nbl == 0) m_list[nm++] = mv;
        }
        if (nm == 0) { free(m_list); continue; }
        printf("\nn=%d (%d configs)\n", n, nm);

        double t0 = wall_time();
        double *results = (double *)malloc((size_t)nm * sizeof(double));

        #pragma omp parallel for schedule(dynamic, 1)
        for (int mi = 0; mi < nm; mi++) {
            uint64_t s = (uint64_t)(n * 100 + m_list[mi]);
            results[mi] = saddle_episode(n, m_list[mi], max_rounds, tau_rel, swaps, s);
        }

        double sr = 0, sf = 0, sb = 0; int wf = 0, wb = 0, tot = 0;
        for (int mi = 0; mi < nm; mi++) {
            BL *b = find_bl(n, m_list[mi]);
            double fv_val = b ? b->fv : 0, best_val = b ? b->best : 0;
            sr += results[mi]; sf += fv_val; sb += best_val;
            if (results[mi] > fv_val * 1.001) wf++;
            if (results[mi] > best_val * 1.001) wb++;
            tot++;
            if (csv_f) fprintf(csv_f, "%d,%.10f\n", m_list[mi], results[mi]);
        }

        double elapsed = wall_time() - t0;
        printf("  vs FV: %.1f%% (%dW/%d) | vs Best: %.1f%% (%dW/%d) | %.1fs\n",
               sf > 0 ? sr / sf * 100 : 0, wf, tot,
               sb > 0 ? sr / sb * 100 : 0, wb, tot, elapsed);

        free(results); free(m_list);
    }

    if (csv_f) fclose(csv_f);
    free(bls);
    return 0;
}
