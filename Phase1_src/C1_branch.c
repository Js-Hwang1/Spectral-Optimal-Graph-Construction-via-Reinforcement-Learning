/* C1_branch.c — Circulant Construction for Odd n, Even k, k < (n+1)/2
 *
 * DESCRIPTION:
 *   Circulant-based construction for odd n graphs with even k and k < ceil(n/2).
 *   Uses ring-based approach with systematic step additions.
 *
 * STRATEGY:
 *   1. Start with a ring (cycle) giving each vertex degree 2
 *   2. Add circulant steps systematically to reach target even degree k
 *   3. For k=4: Use steps {1, 2} (ring + second-nearest neighbors)
 *   4. For k=6: Use steps {1, 2, 3} (ring + second + third neighbors)
 *   5. For k=8: Use steps {1, 2, 3, 4} and so on
 *
 * ALGORITHM:
 *   For odd n and even k with 4 ≤ k < (n+1)/2:
 *   1. Initialize empty adjacency matrix
 *   2. Add ring connections (step = 1)
 *   3. Add additional steps: 2, 3, ..., k/2 to reach degree k
 *   4. Verify k-regularity and connectivity
 *
 * USAGE:
 *   Called when: n % 2 == 1 && k % 2 == 0 && 4 <= k < (n+1)/2
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "Special_Builder.h"
#include "eigenvalue.h"
#include "Algorithm1.h"

// Forward declarations
static void build_circulant_ring_plus_steps(int n, int k, int **adj_matrix);
static void add_circulant_step(int n, int step, int **adj_matrix);
static int count_vertex_degree(int n, int vertex, int **adj_matrix);
static void print_degree_distribution(int n, int **adj_matrix);
static void build_spectral_order_circulant(int n, int k, int **adj_matrix);
typedef struct { double val; int idx; } pair_di;
typedef struct { int step; double score; } StepScore;
typedef struct { int u, v; double s; } PairScore;
typedef struct { double ang; int idx; } PairAng;
static int cmp_pair_di(const void *a, const void *b);
static int cmp_steps_desc(const void *a, const void *b);
static int cmp_ang(const void *a, const void *b);
static void spectral_refine_by_2switch(int n, int k, int **adj_matrix);
static int pick_least_neighbor(int u, int avoid, int n, int **adj, double *rsrow);
static int cmp_pairs_desc(const void *a, const void *b);
static int gcd_int(int a, int b);

/**
 * Build k-regular graph using circulant construction.
 * Strategy: Ring (step=1) + additional steps {2, 3, ..., k/2} for even k.
 * 
 * @param n Number of vertices (odd)
 * @param k Target degree (even, k < (n+1)/2)
 * @param adj_matrix Output adjacency matrix
 */
static void build_circulant_ring_plus_steps(int n, int k, int **adj_matrix) {
    // Initialize empty matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            adj_matrix[i][j] = 0;
        }
    }
    
    // Step 1: Add ring (cycle) connections - gives degree 2
    add_circulant_step(n, 1, adj_matrix);
    
    // Step 2: Add additional steps to reach target degree k
    // For k=4: add step 2
    // For k=6: add steps 2, 3
    // For k=8: add steps 2, 3, 4
    // etc.
    
    int num_additional_steps = (k - 2) / 2;  // k=4 needs 1, k=6 needs 2, k=8 needs 3
    
    for (int step = 2; step <= num_additional_steps + 1; step++) {
        add_circulant_step(n, step, adj_matrix);
    }
}

/**
 * Add circulant step connections to the graph.
 * Each vertex i connects to vertices (i±step) mod n.
 * 
 * @param n Number of vertices
 * @param step Circulant step size
 * @param adj_matrix Adjacency matrix to modify
 */
static void add_circulant_step(int n, int step, int **adj_matrix) {
    for (int i = 0; i < n; i++) {
        int j_forward = (i + step) % n;
        int j_backward = (i - step + n) % n;
        
        // Add forward connection if not self-loop and not already connected
        if (j_forward != i && adj_matrix[i][j_forward] == 0) {
            adj_matrix[i][j_forward] = 1;
            adj_matrix[j_forward][i] = 1;
        }
        
        // Add backward connection if different from forward and not self-loop
        if (j_backward != i && j_backward != j_forward && adj_matrix[i][j_backward] == 0) {
            adj_matrix[i][j_backward] = 1;
            adj_matrix[j_backward][i] = 1;
        }
    }
}

/**
 * Count degree of a specific vertex.
 * 
 * @param n Number of vertices
 * @param vertex Vertex index
 * @param adj_matrix Adjacency matrix
 * @return Degree of the vertex
 */
static int count_vertex_degree(int n, int vertex, int **adj_matrix) {
    int degree = 0;
    for (int j = 0; j < n; j++) {
        degree += adj_matrix[vertex][j];
    }
    return degree;
}

/**
 * Print degree distribution for debugging.
 * 
 * @param n Number of vertices
 * @param adj_matrix Adjacency matrix
 */
static void print_degree_distribution(int n, int **adj_matrix) {
    fprintf(stderr, "C1-Circulant: Degree distribution - ");
    for (int i = 0; i < n; i++) {
        int deg = count_vertex_degree(n, i, adj_matrix);
        fprintf(stderr, "v%d:%d ", i, deg);
    }
    fprintf(stderr, "\n");
}

/**
 * Main C1 branch function.
 * Handles odd n with even k and k < (n+1)/2.
 */
void c1_branch_main(int n, int k, int **adj_matrix) {
    // Validate input parameters
    if (n % 2 == 0) {
        printf("Error: C1 branch requires odd n. Got n=%d\n", n);
        return;
    }
    
    if (k % 2 != 0) {
        printf("Error: C1 branch requires even k. Got k=%d\n", k);
        return;
    }
    
    int ceil_half = (n + 1) / 2;
    if (k < 4 || k >= ceil_half) {
        printf("Error: C1 branch requires 4 ≤ k < (n+1)/2. Got n=%d, k=%d, ceil(n/2)=%d\n", n, k, ceil_half);
        return;
    }
    
    if ((n * k) % 2 != 0) {
        printf("Error: Invalid (n,k) pair for simple graphs. n*k must be even. Got n=%d, k=%d\n", n, k);
        return;
    }
    
    // Prefer spectral-guided construction; fallback to simple circulant if it fails
    build_spectral_order_circulant(n, k, adj_matrix);
    // Spectral 2-switch refinement for very sparse k only (keeps runtime modest)
    int refine_threshold = (n / 6) + 2; if (refine_threshold < 4) refine_threshold = 4;
    if (k <= refine_threshold && n <= 45) {
        spectral_refine_by_2switch(n, k, adj_matrix);
    }
    
    // Verify k-regularity; if not, fallback to baseline circulant construction
    for (int i = 0; i < n; i++) {
        int degree = count_vertex_degree(n, i, adj_matrix);
        if (degree != k) {
            fprintf(stderr, "C1 Warning: Spectral-guided build degree mismatch at v%d (deg=%d, k=%d), falling back to ring+steps.\n", i, degree, k);
            build_circulant_ring_plus_steps(n, k, adj_matrix);
            break;
        }
    }
    
    // Verify k-regularity
    for (int i = 0; i < n; i++) {
        int degree = count_vertex_degree(n, i, adj_matrix);
        if (degree != k) {
            fprintf(stderr, "C1 Error: Vertex %d has degree %d, expected %d\n", i, degree, k);
            return;
        }
    }
}

/* ========================================================================
 * SPECTRAL-GUIDED CIRCULANT ON FIEDLER ORDER
 * ======================================================================== */

static int cmp_pair_di(const void *a, const void *b) {
    const pair_di *pa = (const pair_di*)a;
    const pair_di *pb = (const pair_di*)b;
    if (pa->val < pb->val) return -1;
    if (pa->val > pb->val) return 1;
    return (pa->idx - pb->idx);
}

static int cmp_steps_desc(const void *a, const void *b) {
    const StepScore *sa = (const StepScore*)a;
    const StepScore *sb = (const StepScore*)b;
    if (sa->score < sb->score) return 1;
    if (sa->score > sb->score) return -1;
    return (sa->step - sb->step);
}

static int cmp_ang(const void *a, const void *b) {
    const PairAng *xa = (const PairAng*)a;
    const PairAng *xb = (const PairAng*)b;
    if (xa->ang < xb->ang) return -1;
    if (xa->ang > xb->ang) return 1;
    return (xa->idx - xb->idx);
}

static int gcd_int(int a, int b) {
    if (a < 0) a = -a; if (b < 0) b = -b;
    while (b != 0) { int t = b; b = a % b; a = t; }
    return a;
}

static void build_spectral_order_circulant(int n, int k, int **adj_matrix) {
    // Clear output matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) adj_matrix[i][j] = 0;
    }

    // Build a high-λ2 dense base to extract a meaningful Fiedler ordering
    // For odd n, use the C2 special bipartite case base (near-optimal in mid-dense)
    int **base = (int**)calloc(n, sizeof(int*));
    for (int i = 0; i < n; i++) base[i] = (int*)calloc(n, sizeof(int));

    int ceil_half = (n + 1) / 2;
    int k_base = (ceil_half % 2 == 0) ? ceil_half : (ceil_half + 1);
    build_complete_bipartite_special_case(n, k_base, base);

    // Compute several lowest positive eigenpairs of Laplacian of the base
    int r = 6;
    double *evals = (double*)malloc(r * sizeof(double));
    double *evecs = (double*)malloc(r * n * sizeof(double));
    int got = compute_low_k_eigenpairs(n, base, r, evals, evecs);
    int ok = (got >= 1);

    // Free base matrix
    for (int i = 0; i < n; i++) free(base[i]);
    free(base);

    if (!ok) {
        // If Fiedler fails, fall back to deterministic long-step circulant on natural order
        int m = k / 2;
        double spacing = (double)n / (2.0 * (m + 1));
        int *steps = (int*)malloc(m * sizeof(int));
        int used_count = 0;
        for (int t = 1; t <= m; t++) {
            int s = (int)llround(t * spacing);
            if (s <= 0) s = t; // safety
            if (s >= n/2) s = n/2 - 1;
            // Ensure uniqueness
            int dup = 0;
            for (int u = 0; u < used_count; u++) if (steps[u] == s) { dup = 1; break; }
            while (dup && s < n/2) {
                s++; dup = 0; for (int u = 0; u < used_count; u++) if (steps[u] == s) { dup = 1; break; }
            }
            if (s >= n/2) s = (t % (n/2)); if (s == 0) s = 1;
            steps[used_count++] = s;
        }
        for (int idx = 0; idx < used_count; idx++) {
            int step = steps[idx];
            for (int i = 0; i < n; i++) {
                int j = (i + step) % n;
                if (i != j && adj_matrix[i][j] == 0) {
                    adj_matrix[i][j] = 1;
                    adj_matrix[j][i] = 1;
                }
            }
        }
        free(steps);
        return;
    }

    // Derive ordering:
    // If we have >=2 eigenvectors, use angle ordering in 2D (y1,y2); else use Fiedler sort
    int *order = (int*)malloc(n * sizeof(int));
    if (got >= 2) {
        PairAng *pa = (PairAng*)malloc(n * sizeof(PairAng));
        // Compute angles
        for (int i = 0; i < n; i++) {
            double y1 = evecs[0 * n + i];
            double y2 = evecs[1 * n + i];
            pa[i].ang = atan2(y2, y1);
            pa[i].idx = i;
        }
        qsort(pa, n, sizeof(PairAng), cmp_ang);
        for (int i = 0; i < n; i++) order[i] = pa[i].idx;
        free(pa);
    } else {
        pair_di *arr = (pair_di*)malloc(n * sizeof(pair_di));
        for (int i = 0; i < n; i++) { arr[i].val = evecs[0 * n + i]; arr[i].idx = i; }
        qsort(arr, n, sizeof(pair_di), cmp_pair_di);
        for (int i = 0; i < n; i++) order[i] = arr[i].idx;
        free(arr);
    }

    // Score each circulant step via multi-eigenpair effective-resistance proxy
    int smax = (n - 1) / 2; // for odd n
    StepScore *scores = (StepScore*)malloc(smax * sizeof(StepScore));
    for (int s = 1; s <= smax; s++) {
        double score = 0.0;
        for (int i = 0; i < n; i++) {
            int u = order[i];
            int v = order[(i + s) % n];
            for (int p = 0; p < got; p++) {
                double lam = evals[p]; if (lam <= 1e-12) continue;
                double du = evecs[p*n + u];
                double dv = evecs[p*n + v];
                double d = du - dv;
                score += (d*d) / lam;
            }
        }
        scores[s - 1].step = s;
        scores[s - 1].score = score;
    }
    qsort(scores, smax, sizeof(StepScore), cmp_steps_desc);

    // For very small m, brute-force best step set among top T candidates using exact λ₂
    int m = k / 2;
    int use_bruteforce = (m <= 4 && n <= 35);
    if (use_bruteforce) {
        int T = smax < 12 ? smax : 12;
        int *cands = (int*)malloc(T * sizeof(int));
        for (int i = 0; i < T; i++) cands[i] = scores[i].step;

        // temporary adjacency
        int **best_adj = (int**)calloc(n, sizeof(int*));
        for (int i = 0; i < n; i++) best_adj[i] = (int*)calloc(n, sizeof(int));
        double best_lambda2 = -1.0;

        // Enumerate combinations of size m (m<=3)
        for (int i = 0; i < T; i++) {
            for (int j = i+1; j < T; j++) {
                if (m == 2) {
                    // Build candidate
                    for (int a = 0; a < n; a++) for (int b = 0; b < n; b++) adj_matrix[a][b] = 0;
                    int steps2[2] = {cands[i], cands[j]};
                    for (int si = 0; si < 2; si++) {
                        int step = steps2[si];
                        for (int p = 0; p < n; p++) {
                            int u = order[p];
                            int v = order[(p + step) % n];
                            if (u != v) { adj_matrix[u][v] = adj_matrix[v][u] = 1; }
                        }
                    }
                    double l2 = compute_lambda2_from_adjacency_matrix(n, adj_matrix);
                    if (l2 > best_lambda2) {
                        best_lambda2 = l2;
                        for (int a = 0; a < n; a++) for (int b = 0; b < n; b++) best_adj[a][b] = adj_matrix[a][b];
                    }
                } else if (m == 3) {
                    for (int k3 = j+1; k3 < T; k3++) {
                        for (int a = 0; a < n; a++) for (int b = 0; b < n; b++) adj_matrix[a][b] = 0;
                        int steps3[3] = {cands[i], cands[j], cands[k3]};
                        for (int si = 0; si < 3; si++) {
                            int step = steps3[si];
                            for (int p = 0; p < n; p++) {
                                int u = order[p];
                                int v = order[(p + step) % n];
                                if (u != v) { adj_matrix[u][v] = adj_matrix[v][u] = 1; }
                            }
                        }
                        double l2 = compute_lambda2_from_adjacency_matrix(n, adj_matrix);
                        if (l2 > best_lambda2) {
                            best_lambda2 = l2;
                            for (int a = 0; a < n; a++) for (int b = 0; b < n; b++) best_adj[a][b] = adj_matrix[a][b];
                        }
                    }
                } else if (m == 4) {
                    for (int k3 = j+1; k3 < T; k3++) {
                        for (int k4 = k3+1; k4 < T; k4++) {
                            for (int a = 0; a < n; a++) for (int b = 0; b < n; b++) adj_matrix[a][b] = 0;
                            int steps4[4] = {cands[i], cands[j], cands[k3], cands[k4]};
                            for (int si = 0; si < 4; si++) {
                                int step = steps4[si];
                                for (int p = 0; p < n; p++) {
                                    int u = order[p];
                                    int v = order[(p + step) % n];
                                    if (u != v) { adj_matrix[u][v] = adj_matrix[v][u] = 1; }
                                }
                            }
                            double l2 = compute_lambda2_from_adjacency_matrix(n, adj_matrix);
                            if (l2 > best_lambda2) {
                                best_lambda2 = l2;
                                for (int a = 0; a < n; a++) for (int b = 0; b < n; b++) best_adj[a][b] = adj_matrix[a][b];
                            }
                        }
                    }
                }
            }
        }
        // Copy best found back
        if (best_lambda2 > 0) {
            for (int a = 0; a < n; a++) for (int b = 0; b < n; b++) adj_matrix[a][b] = best_adj[a][b];
        } else {
            // Fallback to top-m greedy if something went wrong
            for (int a = 0; a < n; a++) for (int b = 0; b < n; b++) adj_matrix[a][b] = 0;
            for (int t = 0; t < m && t < smax; t++) {
                int step = scores[t].step;
                for (int p = 0; p < n; p++) {
                    int u = order[p];
                    int v = order[(p + step) % n];
                    if (u != v) { adj_matrix[u][v] = adj_matrix[v][u] = 1; }
                }
            }
        }
        // cleanup
        for (int i = 0; i < n; i++) free(best_adj[i]);
        free(best_adj);
        free(cands);
    } else {
        // Select top m steps with mild constraints
        int *chosen = (int*)calloc(m, sizeof(int));
        int chosen_count = 0;
        for (int idx = 0; idx < smax && chosen_count < m; idx++) {
            int s = scores[idx].step;
            if (gcd_int(s, n) != 1 && chosen_count < m - 1) continue;
            int too_close = 0;
            for (int q = 0; q < chosen_count; q++) {
                int diff = s - chosen[q]; if (diff < 0) diff = -diff;
                if (diff <= 1) { too_close = 1; break; }
            }
            if (too_close) continue;
            chosen[chosen_count++] = s;
        }
        for (int idx = 0; idx < smax && chosen_count < m; idx++) {
            int s = scores[idx].step;
            int dup = 0; for (int q = 0; q < chosen_count; q++) if (chosen[q] == s) { dup = 1; break; }
            if (!dup) chosen[chosen_count++] = s;
        }
        for (int t = 0; t < chosen_count; t++) {
            int step = chosen[t];
            for (int p = 0; p < n; p++) {
                int u = order[p];
                int v = order[(p + step) % n];
                if (u != v && adj_matrix[u][v] == 0) {
                    adj_matrix[u][v] = adj_matrix[v][u] = 1;
                }
            }
        }
        free(chosen);
    }
    free(scores);
    free(order);
    free(evals);
    free(evecs);
}

/* ========================================================================
 * SPECTRAL 2-SWITCH REFINEMENT (DETERMINISTIC)
 * ======================================================================== */

static int cmp_pairs_desc(const void *a, const void *b) {
    const PairScore *pa = (const PairScore*)a;
    const PairScore *pb = (const PairScore*)b;
    if (pa->s < pb->s) return 1;
    if (pa->s > pb->s) return -1;
    if (pa->u != pb->u) return pa->u - pb->u;
    return pa->v - pb->v;
}

static double pair_er_score(int i, int j, int n, int got, const double *evals, const double *evecs) {
    double sum = 0.0;
    for (int p = 0; p < got; p++) {
        double lam = evals[p];
        if (lam <= 1e-12) continue;
        double di = evecs[p*n + i];
        double dj = evecs[p*n + j];
        double d = di - dj;
        sum += (d * d) / lam;
    }
    return sum;
}

static int pick_least_neighbor(int u, int avoid, int n, int **adj, double *rsrow) {
    int best = -1; double bests = 1e300;
    for (int w = 0; w < n; w++) {
        if (w == u || w == avoid) continue;
        if (adj[u][w]) {
            double s = rsrow[w];
            if (s < bests) { bests = s; best = w; }
        }
    }
    return best;
}

static void spectral_refine_by_2switch(int n, int k, int **adj_matrix) {
    // Compute low-k eigenpairs of current graph
    int r = 3;
    double *evals = (double*)malloc(r * sizeof(double));
    double *evecs = (double*)malloc(r * n * sizeof(double));
    int got = compute_low_k_eigenpairs(n, adj_matrix, r, evals, evecs);
    if (got <= 0) { free(evals); free(evecs); return; }

    // Precompute score matrix for all pairs
    double *rs = (double*)malloc((size_t)n * (size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) {
        rs[i*(size_t)n + i] = 0.0;
        for (int j = i+1; j < n; j++) {
            double s = pair_er_score(i, j, n, got, evals, evecs);
            rs[i*(size_t)n + j] = s;
            rs[j*(size_t)n + i] = s;
        }
    }

    // Build list of top candidate non-edges by score
    int max_pairs = n*(n-1)/2;
    PairScore *cands = (PairScore*)malloc(max_pairs * sizeof(PairScore));
    int cc = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            if (adj_matrix[i][j] == 0) {
                cands[cc].u = i; cands[cc].v = j; cands[cc].s = rs[i*(size_t)n + j];
                cc++;
            }
        }
    }
    qsort(cands, cc, sizeof(PairScore), cmp_pairs_desc);

    int max_swaps = (n < 60) ? 60 : 80; // bounded search
    int performed = 0;
    double current_lambda2 = compute_lambda2_from_adjacency_matrix(n, adj_matrix);
    for (int idx = 0; idx < cc && performed < max_swaps; idx++) {
        int u = cands[idx].u, v = cands[idx].v;
        if (adj_matrix[u][v]) continue; // already edge

        // Build small candidate sets of least-beneficial neighbors for u and v
        int candA[4]; int na = 0;
        int candB[4]; int nb = 0;
        // Collect up to 4 least rs neighbors each
        for (int rep = 0; rep < 4; rep++) {
            int pickA = pick_least_neighbor(u, v, n, adj_matrix, &rs[u*(size_t)n]);
            if (pickA >= 0) {
                // Temporarily mark to avoid picking same again
                int tmp = adj_matrix[u][pickA]; adj_matrix[u][pickA] = adj_matrix[pickA][u] = 0;
                candA[na++] = pickA;
                if (na >= 4) { adj_matrix[u][pickA] = adj_matrix[pickA][u] = tmp; break; }
                // restore will be done later in loop
                adj_matrix[u][pickA] = adj_matrix[pickA][u] = tmp;
            }
        }
        for (int rep = 0; rep < 4; rep++) {
            int pickB = pick_least_neighbor(v, u, n, adj_matrix, &rs[v*(size_t)n]);
            if (pickB >= 0) {
                int tmp = adj_matrix[v][pickB]; adj_matrix[v][pickB] = adj_matrix[pickB][v] = 0;
                candB[nb++] = pickB;
                if (nb >= 4) { adj_matrix[v][pickB] = adj_matrix[pickB][v] = tmp; break; }
                adj_matrix[v][pickB] = adj_matrix[pickB][v] = tmp;
            }
        }
        if (na == 0 || nb == 0) continue;

        // Evaluate combinations and pick best positive gain
        double best_gain = 0.0; int bestA = -1, bestB = -1;
        for (int ia = 0; ia < na; ia++) {
            int a = candA[ia];
            if (a == u || a == v) continue;
            double sua = rs[u*(size_t)n + a];
            for (int ib = 0; ib < nb; ib++) {
                int b = candB[ib];
                if (b == u || b == v || b == a) continue;
                if (adj_matrix[a][b]) continue;
                double svb = rs[v*(size_t)n + b];
                double suv = rs[u*(size_t)n + v];
                double sab = rs[a*(size_t)n + b];
                double gain = (suv + sab) - (sua + svb);
                if (gain > best_gain + 1e-12) { best_gain = gain; bestA = a; bestB = b; }
            }
        }
        if (bestA < 0 || bestB < 0) continue; // no improving swap

        // Tentative 2-switch and validate using exact λ₂ improvement
        if (adj_matrix[u][bestA] && adj_matrix[v][bestB] && !adj_matrix[u][v] && !adj_matrix[bestA][bestB]) {
            // Apply
            adj_matrix[u][bestA] = adj_matrix[bestA][u] = 0;
            adj_matrix[v][bestB] = adj_matrix[bestB][v] = 0;
            adj_matrix[u][v] = adj_matrix[v][u] = 1;
            adj_matrix[bestA][bestB] = adj_matrix[bestB][bestA] = 1;

            double new_lambda2 = compute_lambda2_from_adjacency_matrix(n, adj_matrix);
            if (new_lambda2 > current_lambda2 + 1e-9) {
                current_lambda2 = new_lambda2;
                performed++;
            } else {
                // Revert
                adj_matrix[u][v] = adj_matrix[v][u] = 0;
                adj_matrix[bestA][bestB] = adj_matrix[bestB][bestA] = 0;
                adj_matrix[u][bestA] = adj_matrix[bestA][u] = 1;
                adj_matrix[v][bestB] = adj_matrix[bestB][v] = 1;
            }
        }
    }

    free(cands);
    free(rs);
    free(evals);
    free(evecs);
}
