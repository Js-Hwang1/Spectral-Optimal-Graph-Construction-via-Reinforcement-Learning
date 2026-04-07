/*
 * QRS-DR large-n test with eigenvalue computation.
 * Dynamic allocation, int8_t adjacency.
 *
 * Usage: ./algo_large <n> <d>
 *
 * Build:
 *   gcc -O2 -o algo_large algo_large.c -lm -llapacke -llapack -lblas \
 *       -I/opt/homebrew/Cellar/lapack/3.12.1/include \
 *       -L/opt/homebrew/Cellar/lapack/3.12.1/lib
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <lapacke.h>

static int8_t *adj;
static int *degs;
static int N, D;

/* ================================================================
 * Number theory
 * ================================================================ */

static int is_prime(int x) {
    if (x < 2) return 0;
    if (x < 4) return 1;
    if (x % 2 == 0) return 0;
    for (int i = 3; (long long)i * i <= x; i += 2)
        if (x % i == 0) return 0;
    return 1;
}

static int next_prime(int n) {
    int p = n + 1;
    if (p <= 2) return 2;
    if (p % 2 == 0) p++;
    while (!is_prime(p)) p += 2;
    return p;
}

static long long mod_pow(long long base, long long exp, long long m) {
    long long result = 1;
    base %= m;
    if (base < 0) base += m;
    while (exp > 0) {
        if (exp & 1) result = result * base % m;
        base = base * base % m;
        exp >>= 1;
    }
    return result;
}

static int primitive_root(int p) {
    if (p == 2) return 1;
    int pm1 = p - 1;
    int factors[64], nf = 0;
    int tmp = pm1;
    for (int d = 2; (long long)d * d <= tmp; d++) {
        if (tmp % d == 0) {
            factors[nf++] = d;
            while (tmp % d == 0) tmp /= d;
        }
    }
    if (tmp > 1) factors[nf++] = tmp;

    for (int g = 2; g < p; g++) {
        int ok = 1;
        for (int f = 0; f < nf; f++) {
            if (mod_pow(g, pm1 / factors[f], p) == 1) { ok = 0; break; }
        }
        if (ok) return g;
    }
    return 2;
}

/* ================================================================
 * Adjacency helpers
 * ================================================================ */

static inline int8_t get_adj(int i, int j) { return adj[i * N + j]; }
static inline void set_adj(int i, int j, int8_t val) {
    adj[i * N + j] = val;
    adj[j * N + i] = val;
}

static void recompute_degs(void) {
    for (int i = 0; i < N; i++) {
        int s = 0;
        for (int j = 0; j < N; j++) s += get_adj(i, j);
        degs[i] = s;
    }
}

static int count_edges(void) {
    int total = 0;
    for (int i = 0; i < N; i++) total += degs[i];
    return total / 2;
}

/* ================================================================
 * Phase 1: QR Scatter
 * ================================================================ */

static void qr_scatter(void) {
    memset(adj, 0, (size_t)N * N);

    int p = next_prime(N);
    int g = primitive_root(p);

    printf("  p=%d g=%d\n", p, g);

    for (int k = 0; k < D; k++) {
        long long c = mod_pow(g, k + 1, p);
        for (int i = 0; i < N; i++) {
            long long t = (i + 1) % p;
            if (t == 0) t = 1;
            int j = (int)((t * (t + c)) % p % N);
            if (j == i) j = (j + 1) % N;
            set_adj(i, j, 1);
        }
    }

    for (int i = 0; i < N; i++) adj[i * N + i] = 0;
    recompute_degs();
}

/* ================================================================
 * Phase 2: Degree Regularization
 * ================================================================ */

static void degree_regularize(void) {
    int target_edges = N * D / 2;
    int current_edges = count_edges();

    printf("  Scatter: %d edges, target %d\n", current_edges, target_edges);

    /* Step 1: Fix edge count */
    int step1_ops = 0;
    while (current_edges > target_edges) {
        int u = -1, max_deg = -1;
        for (int i = 0; i < N; i++) {
            if (degs[i] > max_deg) { max_deg = degs[i]; u = i; }
        }
        int v = -1, best_deg = -1;
        for (int j = 0; j < N; j++) {
            if (get_adj(u, j) && degs[j] > best_deg) { best_deg = degs[j]; v = j; }
        }
        if (v < 0) break;
        set_adj(u, v, 0);
        degs[u]--; degs[v]--;
        current_edges--;
        step1_ops++;
    }

    while (current_edges < target_edges) {
        int u = -1, min_deg = N + 1;
        for (int i = 0; i < N; i++) {
            if (degs[i] < min_deg) { min_deg = degs[i]; u = i; }
        }
        int v = -1, best_deg = N + 1;
        for (int j = 0; j < N; j++) {
            if (j != u && !get_adj(u, j) && degs[j] < best_deg) { best_deg = degs[j]; v = j; }
        }
        if (v < 0) break;
        set_adj(u, v, 1);
        degs[u]++; degs[v]++;
        current_edges++;
        step1_ops++;
    }

    printf("  Step 1: %d ops (edge count now %d)\n", step1_ops, count_edges());

    /* Step 2: Degree equalization */
    int step2_ops = 0;
    int max_iter = N * D * 2;  /* generous bound */

    for (int iter = 0; iter < max_iter; iter++) {
        /* Find over/under */
        int u = -1, max_over = D;
        int w = -1, min_under = D;

        for (int i = 0; i < N; i++) {
            if (degs[i] > max_over) { max_over = degs[i]; }
            if (degs[i] < min_under) { min_under = degs[i]; }
        }
        if (max_over == D && min_under == D) break;  /* done */
        if (max_over == D || min_under == D) break;   /* shouldn't happen */

        /* Lowest index with max degree > d */
        for (int i = 0; i < N; i++) {
            if (degs[i] == max_over) { u = i; break; }
        }
        /* Lowest index with min degree < d */
        for (int i = 0; i < N; i++) {
            if (degs[i] == min_under) { w = i; break; }
        }

        if (u < 0 || w < 0) break;

        /* Collect and sort N(u) by (-deg, +index) */
        int nb[1024], nb_count = 0;  /* d is small, this is fine */
        for (int j = 0; j < N; j++) {
            if (get_adj(u, j)) nb[nb_count++] = j;
        }
        /* Insertion sort */
        for (int a = 1; a < nb_count; a++) {
            int key = nb[a];
            int b = a - 1;
            while (b >= 0 && (degs[nb[b]] < degs[key] ||
                              (degs[nb[b]] == degs[key] && nb[b] > key))) {
                nb[b + 1] = nb[b]; b--;
            }
            nb[b + 1] = key;
        }

        /* Attempt 1: Direct transfer */
        int transferred = 0;
        for (int idx = 0; idx < nb_count; idx++) {
            int v = nb[idx];
            if (v == w) continue;
            if (!get_adj(w, v)) {
                set_adj(u, v, 0);
                set_adj(w, v, 1);
                degs[u]--; degs[w]++;
                transferred = 1;
                break;
            }
        }
        if (transferred) { step2_ops++; continue; }

        /* Attempt 2: Bridge */
        if (!get_adj(u, w)) {
            set_adj(u, w, 1);
            degs[u]++; degs[w]++;
            int best_v = -1, best_vdeg = -1;
            for (int j = 0; j < N; j++) {
                if (j != w && get_adj(u, j) && degs[j] > best_vdeg) {
                    best_vdeg = degs[j]; best_v = j;
                }
            }
            if (best_v >= 0) {
                /* Tie-break: lowest index */
                for (int j = 0; j < N; j++) {
                    if (j != w && get_adj(u, j) && degs[j] == best_vdeg) {
                        best_v = j; break;
                    }
                }
                set_adj(u, best_v, 0);
                degs[u]--; degs[best_v]--;
            }
        } else {
            /* Attempt 3: Indirect */
            int best_v = -1, best_vdeg = -1;
            for (int j = 0; j < N; j++) {
                if (get_adj(u, j) && degs[j] > best_vdeg) {
                    best_vdeg = degs[j]; best_v = j;
                }
            }
            for (int j = 0; j < N; j++) {
                if (get_adj(u, j) && degs[j] == best_vdeg) {
                    best_v = j; break;
                }
            }
            if (best_v >= 0) {
                set_adj(u, best_v, 0);
                degs[u]--; degs[best_v]--;
            }
            for (int x = 0; x < N; x++) {
                if (x != w && !get_adj(w, x) && degs[x] < D) {
                    set_adj(w, x, 1);
                    degs[w]++; degs[x]++;
                    break;
                }
            }
        }
        step2_ops++;
    }

    printf("  Step 2: %d swaps\n", step2_ops);
}

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <n> <d>\n", argv[0]);
        return 1;
    }

    N = atoi(argv[1]);
    D = atoi(argv[2]);

    if (N < 4 || D < 3 || D >= N || (N * D) % 2 != 0) {
        fprintf(stderr, "Invalid: need n>=4, d>=3, d<n, n*d even\n");
        return 1;
    }

    printf("QRS-DR: n=%d d=%d\n", N, D);
    printf("  Memory: %.1f MB (adj) + %.1f KB (degs)\n",
           (double)N * N / 1e6, (double)N * sizeof(int) / 1e3);

    adj = (int8_t *)calloc((size_t)N * N, sizeof(int8_t));
    degs = (int *)calloc(N, sizeof(int));
    if (!adj || !degs) {
        fprintf(stderr, "OOM: cannot allocate %lld bytes\n",
                (long long)N * N + (long long)N * sizeof(int));
        return 1;
    }

    clock_t t0, t1;

    /* Phase 1 */
    t0 = clock();
    qr_scatter();
    t1 = clock();
    printf("  Phase 1 (scatter): %.3f sec\n", (double)(t1 - t0) / CLOCKS_PER_SEC);

    /* Phase 2 */
    t0 = clock();
    degree_regularize();
    t1 = clock();
    printf("  Phase 2 (regularize): %.3f sec\n", (double)(t1 - t0) / CLOCKS_PER_SEC);

    /* Verify */
    int min_deg = N, max_deg = 0;
    int irregular = 0;
    for (int i = 0; i < N; i++) {
        if (degs[i] < min_deg) min_deg = degs[i];
        if (degs[i] > max_deg) max_deg = degs[i];
        if (degs[i] != D) irregular++;
    }

    int edges = count_edges();
    int expected_edges = N * D / 2;

    printf("  Edges: %d (expected %d)\n", edges, expected_edges);
    printf("  Degree range: [%d, %d] (target %d)\n", min_deg, max_deg, D);
    printf("  Irregular nodes: %d / %d\n", irregular, N);
    printf("  Result: %s\n", (irregular == 0 && edges == expected_edges) ? "d-REGULAR" : "FAIL");

    /* Symmetry check */
    int sym_err = 0;
    for (int i = 0; i < N && sym_err == 0; i++)
        for (int j = i + 1; j < N; j++)
            if (get_adj(i, j) != get_adj(j, i)) { sym_err = 1; break; }
    printf("  Symmetric: %s\n", sym_err ? "NO" : "YES");

    /* Self-loop check */
    int self_loops = 0;
    for (int i = 0; i < N; i++) if (get_adj(i, i)) self_loops++;
    printf("  Self-loops: %d\n", self_loops);

    /* Lambda_2 computation */
    printf("  Computing lambda_2 (LAPACK dsyev, O(n^3))...\n");
    fflush(stdout);

    double *L = (double *)calloc((size_t)N * N, sizeof(double));
    double *evals = (double *)malloc(N * sizeof(double));
    if (!L || !evals) {
        printf("  OOM for eigensolve (need %.1f MB)\n",
               (double)N * N * sizeof(double) / 1e6);
        free(adj); free(degs);
        return 1;
    }

    printf("  Laplacian: %.1f MB allocated\n", (double)N * N * sizeof(double) / 1e6);
    fflush(stdout);

    /* Build Laplacian L = D - A */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j)
                L[i * N + j] = (double)degs[i];
            else
                L[i * N + j] = -(double)get_adj(i, j);
        }
    }

    t0 = clock();
    int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'N', 'U', N, L, N, evals);
    t1 = clock();

    if (info != 0) {
        printf("  LAPACK dsyev FAILED: info=%d\n", info);
    } else {
        double l2 = evals[1];
        double ram = D - 2.0 * sqrt(D - 1.0);
        printf("  Eigensolve: %.3f sec\n", (double)(t1 - t0) / CLOCKS_PER_SEC);
        printf("  lambda_2 = %.6f\n", l2);
        printf("  Ramanujan = %.6f\n", ram);
        printf("  lambda_2 / Ram = %.2fx\n", l2 / ram);
        printf("  Above Ramanujan: %s\n", l2 >= ram * 0.99 ? "YES" : "NO");
    }

    free(L);
    free(evals);
    free(adj);
    free(degs);
    return 0;
}
