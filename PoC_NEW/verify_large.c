/*
 * verify_large.c — Large-scale verification that random init + degree
 * regularization produces near-Ramanujan d-regular graphs.
 *
 * Optimized for EXTREME sparsity: n up to 2^20+, d=4.
 *   - Adjacency lists: O(nd) memory
 *   - Hash set per node for O(1) edge lookup during init
 *   - Bucket queue for O(1) max/min degree lookup during regularization
 *   - Lanczos with sparse matvec for λ₂: O(k·nd) per run
 *
 * Build:
 *   gcc -O3 -march=native -o verify_large verify_large.c -lm
 *
 * Usage:
 *   ./verify_large <n> <d> [seed]
 *   ./verify_large --sweep           (powers of 2, d=4)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

// ============================================================
// Fast RNG (xoshiro256**)
// ============================================================

static uint64_t rng_s[4];

static inline uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t rng_next(void) {
    uint64_t r = rotl(rng_s[1] * 5, 7) * 9;
    uint64_t t = rng_s[1] << 17;
    rng_s[2] ^= rng_s[0]; rng_s[3] ^= rng_s[1];
    rng_s[1] ^= rng_s[2]; rng_s[0] ^= rng_s[3];
    rng_s[2] ^= t; rng_s[3] = rotl(rng_s[3], 45);
    return r;
}

static void rng_seed(uint64_t seed) {
    rng_s[0] = seed;
    rng_s[1] = seed * 6364136223846793005ULL + 1;
    rng_s[2] = seed * 1103515245ULL + 12345;
    rng_s[3] = seed ^ 0xdeadbeefcafebabeULL;
    for (int i = 0; i < 20; i++) rng_next(); // warm up
}

static inline int rng_int(int n) {
    return (int)(rng_next() % (uint64_t)n);
}

// ============================================================
// Sparse graph: adjacency lists + hash sets for O(1) lookup
// ============================================================

typedef struct {
    int n;
    int d_target;
    int *deg;          // deg[i] = current degree
    int **nbr;         // nbr[i][0..deg[i]-1] = neighbors
    int *nbr_cap;      // capacity

    // Hash set for fast edge lookup during init
    // For d ≤ ~20, linear scan on nbr[] is faster than hash.
    // For init phase with possibly high-degree nodes, use a global
    // hash set approach: hash(min(i,j), max(i,j)) → bucket.
    // Actually at d=4, each node has at most ~12 neighbors during init.
    // Linear scan is fine.
} Graph;

static Graph *graph_new(int n, int d_target) {
    Graph *G = malloc(sizeof(Graph));
    G->n = n;
    G->d_target = d_target;
    G->deg = calloc(n, sizeof(int));
    G->nbr = malloc(n * sizeof(int *));
    G->nbr_cap = malloc(n * sizeof(int));
    int init_cap = d_target * 3 + 2; // enough for init phase
    for (int i = 0; i < n; i++) {
        G->nbr_cap[i] = init_cap;
        G->nbr[i] = malloc(init_cap * sizeof(int));
    }
    return G;
}

static void graph_free(Graph *G) {
    for (int i = 0; i < G->n; i++) free(G->nbr[i]);
    free(G->nbr); free(G->nbr_cap); free(G->deg); free(G);
}

static inline void ensure_cap(Graph *G, int i) {
    if (G->deg[i] >= G->nbr_cap[i]) {
        G->nbr_cap[i] *= 2;
        G->nbr[i] = realloc(G->nbr[i], G->nbr_cap[i] * sizeof(int));
    }
}

static inline int has_edge(Graph *G, int i, int j) {
    // Linear scan — fine for d ≤ ~20
    for (int k = 0; k < G->deg[i]; k++)
        if (G->nbr[i][k] == j) return 1;
    return 0;
}

static void add_edge(Graph *G, int i, int j) {
    ensure_cap(G, i); ensure_cap(G, j);
    G->nbr[i][G->deg[i]++] = j;
    G->nbr[j][G->deg[j]++] = i;
}

static void remove_edge(Graph *G, int i, int j) {
    // Swap-remove j from nbr[i]
    for (int k = 0; k < G->deg[i]; k++) {
        if (G->nbr[i][k] == j) {
            G->nbr[i][k] = G->nbr[i][--G->deg[i]];
            break;
        }
    }
    for (int k = 0; k < G->deg[j]; k++) {
        if (G->nbr[j][k] == i) {
            G->nbr[j][k] = G->nbr[j][--G->deg[j]];
            break;
        }
    }
}

// ============================================================
// Phase 1: Random init with exactly m = nd/2 edges
// ============================================================

static void phase1_random_init(Graph *G, uint64_t seed) {
    int n = G->n;
    int d = G->d_target;
    long long m = (long long)n * d / 2;
    rng_seed(seed);

    long long count = 0;
    while (count < m) {
        int i = rng_int(n);
        int j = rng_int(n);
        if (i != j && !has_edge(G, i, j)) {
            add_edge(G, i, j);
            count++;
        }
    }
}

// ============================================================
// Phase 2: Degree regularization with BUCKET QUEUE
//
// Bucket queue: buckets[deg] = linked list of nodes with this degree.
// O(1) to find max/min degree node. O(1) to update after swap.
// ============================================================

typedef struct BucketNode {
    int node_id;
    struct BucketNode *next, *prev;
} BucketNode;

typedef struct {
    int max_deg;
    int min_deg;
    int num_buckets;
    BucketNode **buckets;   // buckets[deg] = head of doubly-linked list
    BucketNode *nodes;      // nodes[i] = BucketNode for graph node i
} BucketQueue;

static BucketQueue *bq_new(int n, int max_possible_deg) {
    BucketQueue *bq = malloc(sizeof(BucketQueue));
    bq->num_buckets = max_possible_deg + 1;
    bq->buckets = calloc(bq->num_buckets, sizeof(BucketNode *));
    bq->nodes = calloc(n, sizeof(BucketNode));
    bq->max_deg = 0;
    bq->min_deg = max_possible_deg;
    for (int i = 0; i < n; i++) bq->nodes[i].node_id = i;
    return bq;
}

static void bq_free(BucketQueue *bq) {
    free(bq->buckets); free(bq->nodes); free(bq);
}

static void bq_insert(BucketQueue *bq, int node, int deg) {
    BucketNode *bn = &bq->nodes[node];
    bn->prev = NULL;
    bn->next = bq->buckets[deg];
    if (bq->buckets[deg]) bq->buckets[deg]->prev = bn;
    bq->buckets[deg] = bn;
    if (deg > bq->max_deg) bq->max_deg = deg;
    if (deg < bq->min_deg) bq->min_deg = deg;
}

static void bq_remove(BucketQueue *bq, int node, int deg) {
    BucketNode *bn = &bq->nodes[node];
    if (bn->prev) bn->prev->next = bn->next;
    else bq->buckets[deg] = bn->next;
    if (bn->next) bn->next->prev = bn->prev;
    bn->prev = bn->next = NULL;

    // Update max/min if bucket is now empty
    if (!bq->buckets[deg]) {
        if (deg == bq->max_deg) {
            while (bq->max_deg > 0 && !bq->buckets[bq->max_deg])
                bq->max_deg--;
        }
        if (deg == bq->min_deg) {
            while (bq->min_deg < bq->num_buckets - 1 && !bq->buckets[bq->min_deg])
                bq->min_deg++;
        }
    }
}

static void bq_update(BucketQueue *bq, int node, int old_deg, int new_deg) {
    bq_remove(bq, node, old_deg);
    bq_insert(bq, node, new_deg);
}

static int bq_get_max(BucketQueue *bq) {
    if (!bq->buckets[bq->max_deg]) return -1;
    return bq->buckets[bq->max_deg]->node_id;
}

static int bq_get_min(BucketQueue *bq) {
    if (!bq->buckets[bq->min_deg]) return -1;
    return bq->buckets[bq->min_deg]->node_id;
}

static void phase2_regularize(Graph *G) {
    int n = G->n;
    int d = G->d_target;

    // Find max degree for bucket sizing
    int max_d = 0;
    for (int i = 0; i < n; i++)
        if (G->deg[i] > max_d) max_d = G->deg[i];

    BucketQueue *bq = bq_new(n, max_d + 2);
    for (int i = 0; i < n; i++)
        bq_insert(bq, i, G->deg[i]);

    long long swaps = 0;
    long long max_swaps = (long long)n * d * 4;

    while (swaps < max_swaps) {
        if (bq->max_deg <= d && bq->min_deg >= d) break; // done
        if (bq->max_deg <= d || bq->min_deg >= d) break; // can't fix

        int u = bq_get_max(bq); // over-degree
        int w = bq_get_min(bq); // under-degree
        if (u < 0 || w < 0) break;

        // Direct transfer: find v in N(u), v!=w, {w,v} not in E
        int best_v = -1, best_vdeg = -1;
        for (int k = 0; k < G->deg[u]; k++) {
            int v = G->nbr[u][k];
            if (v == w) continue;
            if (has_edge(G, w, v)) continue;
            if (G->deg[v] > best_vdeg) {
                best_vdeg = G->deg[v]; best_v = v;
            }
        }

        if (best_v >= 0) {
            // Transfer: remove {u,v}, add {w,v}
            int old_u = G->deg[u], old_w = G->deg[w]; // v's degree unchanged
            remove_edge(G, u, best_v);
            add_edge(G, w, best_v);
            bq_update(bq, u, old_u, G->deg[u]);
            bq_update(bq, w, old_w, G->deg[w]);
            swaps++;
            continue;
        }

        // Fallback: bridge or indirect
        if (!has_edge(G, u, w)) {
            int old_u = G->deg[u], old_w = G->deg[w];
            add_edge(G, u, w);
            bq_update(bq, u, old_u, G->deg[u]);
            bq_update(bq, w, old_w, G->deg[w]);

            // Remove highest-deg neighbor of u (not w)
            best_v = -1; best_vdeg = -1;
            for (int k = 0; k < G->deg[u]; k++) {
                int v = G->nbr[u][k];
                if (v == w) continue;
                if (G->deg[v] > best_vdeg) { best_vdeg = G->deg[v]; best_v = v; }
            }
            if (best_v >= 0) {
                old_u = G->deg[u]; int old_v = G->deg[best_v];
                remove_edge(G, u, best_v);
                bq_update(bq, u, old_u, G->deg[u]);
                bq_update(bq, best_v, old_v, G->deg[best_v]);
            }
        } else {
            // Already adjacent — remove edge from u, add to w elsewhere
            best_v = -1; best_vdeg = -1;
            for (int k = 0; k < G->deg[u]; k++) {
                int v = G->nbr[u][k];
                if (G->deg[v] > best_vdeg) { best_vdeg = G->deg[v]; best_v = v; }
            }
            if (best_v >= 0) {
                int old_u = G->deg[u], old_v = G->deg[best_v];
                remove_edge(G, u, best_v);
                bq_update(bq, u, old_u, G->deg[u]);
                bq_update(bq, best_v, old_v, G->deg[best_v]);
            }
            // Find lowest-deg non-neighbor of w with deg < d
            for (int x = 0; x < n; x++) {
                if (x != w && G->deg[x] < d && !has_edge(G, w, x)) {
                    int old_w2 = G->deg[w], old_x = G->deg[x];
                    add_edge(G, w, x);
                    bq_update(bq, w, old_w2, G->deg[w]);
                    bq_update(bq, x, old_x, G->deg[x]);
                    break;
                }
            }
        }
        swaps++;
    }

    bq_free(bq);

    if (swaps % 10000 == 0 || swaps < 100)
        fprintf(stderr, "  Regularization: %lld swaps\n", swaps);
}

// ============================================================
// λ₂ via Lanczos on the Laplacian (sparse matvec)
// ============================================================

// Sparse matvec: y = L * x where L = D - A
static void laplacian_matvec(Graph *G, const double *x, double *y) {
    int n = G->n;
    for (int i = 0; i < n; i++) {
        double s = G->deg[i] * x[i];
        for (int k = 0; k < G->deg[i]; k++)
            s -= x[G->nbr[i][k]];
        y[i] = s;
    }
}

static double estimate_lambda2(Graph *G, int lanczos_k) {
    int n = G->n;
    double *v_prev = calloc(n, sizeof(double));
    double *v_curr = malloc(n * sizeof(double));
    double *w = malloc(n * sizeof(double));
    double *alpha = malloc(lanczos_k * sizeof(double));
    double *beta = malloc(lanczos_k * sizeof(double));

    // Random start orthogonal to 1
    rng_seed(999);
    double sum = 0;
    for (int i = 0; i < n; i++) {
        v_curr[i] = (double)(rng_next() & 0xFFFF) / 32768.0 - 1.0;
        sum += v_curr[i];
    }
    sum /= n;
    double norm = 0;
    for (int i = 0; i < n; i++) { v_curr[i] -= sum; norm += v_curr[i] * v_curr[i]; }
    norm = sqrt(norm);
    for (int i = 0; i < n; i++) v_curr[i] /= norm;

    double b = 0;
    int actual_k = 0;

    for (int j = 0; j < lanczos_k; j++) {
        laplacian_matvec(G, v_curr, w);

        double a = 0;
        for (int i = 0; i < n; i++) a += v_curr[i] * w[i];
        alpha[j] = a;

        for (int i = 0; i < n; i++)
            w[i] -= a * v_curr[i] + b * v_prev[i];

        // Re-orthogonalize against 1
        sum = 0;
        for (int i = 0; i < n; i++) sum += w[i];
        sum /= n;
        for (int i = 0; i < n; i++) w[i] -= sum;

        norm = 0;
        for (int i = 0; i < n; i++) norm += w[i] * w[i];
        b = sqrt(norm);
        beta[j] = b;
        actual_k = j + 1;

        if (b < 1e-12) break;

        for (int i = 0; i < n; i++) {
            v_prev[i] = v_curr[i];
            v_curr[i] = w[i] / b;
        }
    }

    // Solve tridiagonal eigenproblem via bisection for λ₂
    // First find λ_max for upper bound
    double hi = 0;
    for (int j = 0; j < actual_k; j++) {
        double bound = fabs(alpha[j]) + (j > 0 ? fabs(beta[j-1]) : 0) +
                        (j < actual_k - 1 ? fabs(beta[j]) : 0);
        if (bound > hi) hi = bound;
    }
    hi += 1.0;
    double lo = 0.0;

    // Bisection: find λ₂ (second smallest eigenvalue of tridiagonal)
    for (int bisect = 0; bisect < 100; bisect++) {
        double mu = (lo + hi) / 2.0;
        // Sturm count
        int count = 0;
        double d_val = alpha[0] - mu;
        if (d_val < 0) count++;
        for (int j = 1; j < actual_k; j++) {
            if (fabs(d_val) < 1e-30) d_val = 1e-30;
            d_val = (alpha[j] - mu) - beta[j-1] * beta[j-1] / d_val;
            if (d_val < 0) count++;
        }
        if (count >= 2) hi = mu;
        else lo = mu;
    }

    free(v_prev); free(v_curr); free(w); free(alpha); free(beta);
    return (lo + hi) / 2.0;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char **argv) {
    if (argc >= 3 && strcmp(argv[1], "--sweep") != 0) {
        int n = atoi(argv[1]);
        int d = atoi(argv[2]);
        uint64_t seed = argc >= 4 ? (uint64_t)atoll(argv[3]) : 42;

        if (n < 4 || d < 3 || ((long long)n * d) % 2 != 0) {
            fprintf(stderr, "Need n>=4, d>=3, n*d even\n"); return 1;
        }

        double ram = d - 2.0 * sqrt(d - 1.0);
        long long m = (long long)n * d / 2;
        long long mem_est = (long long)n * (d * 3 + 10) * 4 / (1024*1024);
        fprintf(stderr, "n=%d d=%d m=%lld seed=%llu Ram=%.4f est_mem=%lldMB\n",
                n, d, m, (unsigned long long)seed, ram, mem_est);

        struct timespec ts0, ts1, ts2, ts3, ts4;
        clock_gettime(CLOCK_MONOTONIC, &ts0);

        fprintf(stderr, "Phase 1: random init...\n");
        Graph *G = graph_new(n, d);
        phase1_random_init(G, seed);
        clock_gettime(CLOCK_MONOTONIC, &ts1);

        // Degree stats
        int min_d = n, max_d = 0;
        for (int i = 0; i < n; i++) {
            if (G->deg[i] < min_d) min_d = G->deg[i];
            if (G->deg[i] > max_d) max_d = G->deg[i];
        }
        fprintf(stderr, "  Init: deg=[%d,%d], edges=%lld (%.2fs)\n",
                min_d, max_d, (long long)n*d/2,
                (ts1.tv_sec-ts0.tv_sec) + (ts1.tv_nsec-ts0.tv_nsec)/1e9);

        fprintf(stderr, "Phase 2: regularize...\n");
        phase2_regularize(G);
        clock_gettime(CLOCK_MONOTONIC, &ts2);
        fprintf(stderr, "  Regularize: %.2fs\n",
                (ts2.tv_sec-ts1.tv_sec) + (ts2.tv_nsec-ts1.tv_nsec)/1e9);

        // Verify
        int reg = 1;
        for (int i = 0; i < n; i++) {
            if (G->deg[i] != d) { reg = 0; break; }
        }
        fprintf(stderr, "  d-regular: %s\n", reg ? "YES" : "NO");

        // λ₂
        int k = 300;
        if (n > 100000) k = 500;
        if (n > 500000) k = 800;
        fprintf(stderr, "Phase 3: Lanczos (%d iters)...\n", k);
        double l2 = estimate_lambda2(G, k);
        clock_gettime(CLOCK_MONOTONIC, &ts3);
        fprintf(stderr, "  λ₂ ≈ %.6f (%.2fs)\n", l2,
                (ts3.tv_sec-ts2.tv_sec) + (ts3.tv_nsec-ts2.tv_nsec)/1e9);

        double total = (ts3.tv_sec-ts0.tv_sec) + (ts3.tv_nsec-ts0.tv_nsec)/1e9;

        // Final output (machine readable)
        printf("n=%d d=%d lambda2=%.6f Ram=%.6f ratio=%.4f regular=%s total=%.2fs\n",
               n, d, l2, ram, l2/ram, reg?"YES":"NO", total);

        graph_free(G);

    } else {
        // Sweep
        int d = 4;
        double ram = d - 2.0 * sqrt(d - 1.0);
        printf("%10s %3s | %8s %8s %8s | %8s %6s | %8s\n",
               "n", "d", "init_s", "reg_s", "eig_s", "lambda2", "ratio", "total");
        printf("-------------------------------------------------------------------\n");

        for (int logn = 5; logn <= 22; logn++) {
            int n = 1 << logn;
            long long m = (long long)n * d / 2;
            long long mem = (long long)n * (d * 3 + 10) * 4 / (1024*1024);
            fprintf(stderr, "n=2^%d=%d m=%lld ~%lldMB\n", logn, n, m, mem);

            struct timespec t0, t1, t2, t3;
            clock_gettime(CLOCK_MONOTONIC, &t0);

            Graph *G = graph_new(n, d);
            phase1_random_init(G, 42);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double init_s = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;

            phase2_regularize(G);
            clock_gettime(CLOCK_MONOTONIC, &t2);
            double reg_s = (t2.tv_sec-t1.tv_sec)+(t2.tv_nsec-t1.tv_nsec)/1e9;

            int reg = 1;
            for (int i = 0; i < n; i++)
                if (G->deg[i] != d) { reg = 0; break; }

            int k = logn <= 14 ? 300 : (logn <= 18 ? 500 : 800);
            double l2 = estimate_lambda2(G, k);
            clock_gettime(CLOCK_MONOTONIC, &t3);
            double eig_s = (t3.tv_sec-t2.tv_sec)+(t3.tv_nsec-t2.tv_nsec)/1e9;
            double total = (t3.tv_sec-t0.tv_sec)+(t3.tv_nsec-t0.tv_nsec)/1e9;

            printf("%10d %3d | %8.2f %8.2f %8.2f | %8.4f %5.2fx | %8.2f%s\n",
                   n, d, init_s, reg_s, eig_s, l2, l2/ram, total,
                   reg ? "" : " NOT-REG");
            fflush(stdout);

            graph_free(G);
            if (total > 3600) { printf("(stopping: >1hr)\n"); break; }
        }
    }
    return 0;
}
