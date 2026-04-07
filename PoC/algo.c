/*
 * QRS-DR: Quadratic Residue Scatter + Degree Regularization
 *
 * Fully deterministic d-regular graph construction.
 * All ties broken by lowest node index.
 *
 * Usage:
 *   ./algo <n> <d>           Print lambda_2 for single (n,d)
 *   ./algo <n> <d> --dump    Print adjacency matrix to stdout (row-major, space-separated)
 *   ./algo --sweep           Test all standard configurations
 *
 * Build:
 *   gcc -O2 -o algo algo.c -lm -llapacke -llapack -lblas \
 *       -I/opt/homebrew/Cellar/lapack/3.12.1/include \
 *       -L/opt/homebrew/Cellar/lapack/3.12.1/lib
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <lapacke.h>

#define MAX_N 2048

/* ================================================================
 * Adjacency matrix (flat array, row-major)
 * adj[i*n + j] == 1 means edge {i,j} exists
 * ================================================================ */

static int adj[MAX_N * MAX_N];
static int degs[MAX_N];
static int n_global, d_global;

/* ================================================================
 * Number theory helpers
 * ================================================================ */

/* Trial-division primality test */
static int is_prime(int x) {
    if (x < 2) return 0;
    if (x < 4) return 1;
    if (x % 2 == 0) return 0;
    for (int i = 3; (long long)i * i <= x; i += 2)
        if (x % i == 0) return 0;
    return 1;
}

/* Smallest prime > n */
static int next_prime(int n) {
    int p = n + 1;
    if (p <= 2) return 2;
    if (p % 2 == 0) p++;
    while (!is_prime(p)) p += 2;
    return p;
}

/* Modular exponentiation: base^exp mod m */
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

/* GCD */
static int gcd(int a, int b) {
    while (b) { int t = b; b = a % b; a = t; }
    return a;
}

/* Smallest primitive root mod p (p prime) */
static int primitive_root(int p) {
    if (p == 2) return 1;
    int pm1 = p - 1;

    /* Factor p-1 */
    int factors[64];
    int nf = 0;
    int tmp = pm1;
    for (int d = 2; (long long)d * d <= tmp; d++) {
        if (tmp % d == 0) {
            factors[nf++] = d;
            while (tmp % d == 0) tmp /= d;
        }
    }
    if (tmp > 1) factors[nf++] = tmp;

    /* Test candidates */
    for (int g = 2; g < p; g++) {
        int ok = 1;
        for (int f = 0; f < nf; f++) {
            if (mod_pow(g, pm1 / factors[f], p) == 1) {
                ok = 0;
                break;
            }
        }
        if (ok) return g;
    }
    return 2; /* fallback */
}

/* ================================================================
 * Adjacency helpers
 * ================================================================ */

static inline int get_adj(int i, int j) {
    return adj[i * n_global + j];
}

static inline void set_adj(int i, int j, int val) {
    adj[i * n_global + j] = val;
    adj[j * n_global + i] = val;
}

static void recompute_degs(void) {
    int n = n_global;
    for (int i = 0; i < n; i++) {
        int s = 0;
        for (int j = 0; j < n; j++) s += get_adj(i, j);
        degs[i] = s;
    }
}

static int count_edges(void) {
    int n = n_global;
    int total = 0;
    for (int i = 0; i < n; i++) total += degs[i];
    return total / 2;
}

/* Find lowest-index node with maximum degree among nodes where mask[i] is true.
 * If mask is NULL, consider all nodes. */
static int find_max_deg_node(const int *mask, int n) {
    int best = -1, best_deg = -1;
    for (int i = 0; i < n; i++) {
        if (mask && !mask[i]) continue;
        if (degs[i] > best_deg) {
            best_deg = degs[i];
            best = i;
        }
    }
    return best;
}

/* Find lowest-index node with minimum degree among nodes where mask[i] is true. */
static int find_min_deg_node(const int *mask, int n) {
    int best = -1, best_deg = n + 1;
    for (int i = 0; i < n; i++) {
        if (mask && !mask[i]) continue;
        if (degs[i] < best_deg) {
            best_deg = degs[i];
            best = i;
        }
    }
    return best;
}

/* ================================================================
 * Phase 1: Quadratic Residue Scatter
 * ================================================================ */

static void qr_scatter(int n, int d) {
    n_global = n;
    d_global = d;
    memset(adj, 0, sizeof(int) * n * n);

    int p = next_prime(n);
    int g = primitive_root(p);

    for (int k = 0; k < d; k++) {
        long long c = mod_pow(g, k + 1, p);
        for (int i = 0; i < n; i++) {
            long long t = (i + 1) % p;
            if (t == 0) t = 1;
            int j = (int)((t * (t + c)) % p % n);
            if (j == i) j = (j + 1) % n;
            set_adj(i, j, 1);
        }
    }

    /* Remove self-loops (shouldn't exist, but be safe) */
    for (int i = 0; i < n; i++) adj[i * n + i] = 0;

    recompute_degs();
}

/* ================================================================
 * Phase 2: Degree Regularization (fully deterministic)
 * ================================================================ */

static void degree_regularize(int n, int d) {
    int target_edges = n * d / 2;
    int current_edges = count_edges();

    /* --- Step 1: Fix total edge count --- */

    /* Remove surplus edges */
    while (current_edges > target_edges) {
        /* u = lowest-index node with max degree */
        int u = find_max_deg_node(NULL, n);
        if (u < 0) break;

        /* v = lowest-index max-degree neighbor of u */
        int v = -1, best_deg = -1;
        for (int j = 0; j < n; j++) {
            if (get_adj(u, j) && degs[j] > best_deg) {
                best_deg = degs[j];
                v = j;
            }
        }
        if (v < 0) break;

        set_adj(u, v, 0);
        degs[u]--;
        degs[v]--;
        current_edges--;
    }

    /* Add deficit edges */
    while (current_edges < target_edges) {
        /* u = lowest-index node with min degree */
        int u = find_min_deg_node(NULL, n);
        if (u < 0) break;

        /* v = lowest-index min-degree non-neighbor of u */
        int v = -1, best_deg = n + 1;
        for (int j = 0; j < n; j++) {
            if (j != u && !get_adj(u, j) && degs[j] < best_deg) {
                best_deg = degs[j];
                v = j;
            }
        }
        if (v < 0) break;

        set_adj(u, v, 1);
        degs[u]++;
        degs[v]++;
        current_edges++;
    }

    /* --- Step 2: Degree equalization via edge swaps --- */

    for (int iter = 0; iter < n * n; iter++) {
        /* Find over and under degree nodes */
        int has_over = 0, has_under = 0;
        for (int i = 0; i < n; i++) {
            if (degs[i] > d) has_over = 1;
            if (degs[i] < d) has_under = 1;
        }
        if (!has_over && !has_under) break;
        if (!has_over || !has_under) break;

        /* u = lowest-index node with max degree > d */
        int u = -1, max_deg = d;
        for (int i = 0; i < n; i++) {
            if (degs[i] > max_deg) {
                max_deg = degs[i];
                u = i;
            }
        }
        /* Among ties, find lowest index */
        for (int i = 0; i < n; i++) {
            if (degs[i] == max_deg && degs[i] > d) {
                u = i;
                break;
            }
        }

        /* w = lowest-index node with min degree < d */
        int w = -1, min_deg = d;
        for (int i = 0; i < n; i++) {
            if (degs[i] < min_deg) {
                min_deg = degs[i];
                w = i;
            }
        }
        /* Among ties, find lowest index */
        for (int i = 0; i < n; i++) {
            if (degs[i] == min_deg && degs[i] < d) {
                w = i;
                break;
            }
        }

        if (u < 0 || w < 0) break;

        /* Collect N(u) sorted by (-deg, index) */
        int nb[MAX_N], nb_count = 0;
        for (int j = 0; j < n; j++) {
            if (get_adj(u, j)) nb[nb_count++] = j;
        }

        /* Sort by (-deg, +index): insertion sort is fine for small d */
        for (int a = 1; a < nb_count; a++) {
            int key = nb[a];
            int b = a - 1;
            while (b >= 0 && (degs[nb[b]] < degs[key] ||
                              (degs[nb[b]] == degs[key] && nb[b] > key))) {
                nb[b + 1] = nb[b];
                b--;
            }
            nb[b + 1] = key;
        }

        /* Attempt 1: Direct transfer */
        int transferred = 0;
        for (int idx = 0; idx < nb_count; idx++) {
            int v = nb[idx];
            if (v == w) continue;
            if (!get_adj(w, v)) {
                /* Transfer: remove {u,v}, add {w,v} */
                set_adj(u, v, 0);
                set_adj(w, v, 1);
                degs[u]--;
                degs[w]++;
                transferred = 1;
                break;
            }
        }
        if (transferred) continue;

        /* Attempt 2: Bridge via {u, w} */
        if (!get_adj(u, w)) {
            set_adj(u, w, 1);
            degs[u]++;
            degs[w]++;

            /* Remove edge from u to highest-degree neighbor != w (tie: lowest index) */
            /* Recompute neighbors of u excluding w */
            int best_v = -1, best_vdeg = -1;
            for (int j = 0; j < n; j++) {
                if (j != w && get_adj(u, j) && degs[j] > best_vdeg) {
                    best_vdeg = degs[j];
                    best_v = j;
                }
            }
            /* Tie-break: lowest index */
            if (best_v >= 0) {
                for (int j = 0; j < n; j++) {
                    if (j != w && get_adj(u, j) && degs[j] == best_vdeg) {
                        best_v = j;
                        break;
                    }
                }
                set_adj(u, best_v, 0);
                degs[u]--;
                degs[best_v]--;
            }
        } else {
            /* Attempt 3: Indirect transfer */
            /* Remove: highest-degree neighbor of u (tie: lowest index) */
            int best_v = -1, best_vdeg = -1;
            for (int j = 0; j < n; j++) {
                if (get_adj(u, j) && degs[j] > best_vdeg) {
                    best_vdeg = degs[j];
                    best_v = j;
                }
            }
            /* Tie-break: lowest index */
            for (int j = 0; j < n; j++) {
                if (get_adj(u, j) && degs[j] == best_vdeg) {
                    best_v = j;
                    break;
                }
            }
            if (best_v >= 0) {
                set_adj(u, best_v, 0);
                degs[u]--;
                degs[best_v]--;
            }

            /* Add: connect w to lowest-index node with deg < d, not already connected */
            for (int x = 0; x < n; x++) {
                if (x != w && !get_adj(w, x) && degs[x] < d) {
                    set_adj(w, x, 1);
                    degs[w]++;
                    degs[x]++;
                    break;
                }
            }
        }
    }
}

/* ================================================================
 * Build the full QRS-DR graph
 * ================================================================ */

static void build_qrs_dr(int n, int d) {
    qr_scatter(n, d);
    degree_regularize(n, d);
}

/* ================================================================
 * Lambda_2 computation via LAPACK (dsyev)
 * ================================================================ */

static double compute_lambda2(int n) {
    double *L = (double *)calloc(n * n, sizeof(double));
    double *evals = (double *)malloc(n * sizeof(double));
    if (!L || !evals) { fprintf(stderr, "OOM\n"); exit(1); }

    /* Build Laplacian: L = D - A */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                L[i * n + j] = (double)degs[i];
            } else {
                L[i * n + j] = -(double)get_adj(i, j);
            }
        }
    }

    /* Compute eigenvalues */
    int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'N', 'U', n, L, n, evals);
    if (info != 0) {
        fprintf(stderr, "LAPACK dsyev failed: %d\n", info);
        free(L); free(evals);
        return -1.0;
    }

    double l2 = evals[1]; /* second smallest */
    free(L);
    free(evals);
    return l2;
}

/* ================================================================
 * Verification: check d-regular
 * ================================================================ */

static int verify_regular(int n, int d) {
    for (int i = 0; i < n; i++) {
        if (degs[i] != d) return 0;
    }
    /* Check symmetry */
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (get_adj(i, j) != get_adj(j, i)) return 0;
    /* No self-loops */
    for (int i = 0; i < n; i++)
        if (get_adj(i, i)) return 0;
    return 1;
}

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char **argv) {
    if (argc >= 3 && strcmp(argv[1], "--sweep") != 0) {
        int n = atoi(argv[1]);
        int d = atoi(argv[2]);

        if (n < 4 || d < 3 || d >= n || (n * d) % 2 != 0) {
            fprintf(stderr, "Invalid: need n>=4, d>=3, d<n, n*d even\n");
            return 1;
        }
        if (n > MAX_N) {
            fprintf(stderr, "n=%d exceeds MAX_N=%d\n", n, MAX_N);
            return 1;
        }

        build_qrs_dr(n, d);

        if (argc >= 4 && strcmp(argv[3], "--dump") == 0) {
            /* Print adjacency matrix */
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    printf("%d", get_adj(i, j));
                    if (j < n - 1) printf(" ");
                }
                printf("\n");
            }
        } else {
            int reg = verify_regular(n, d);
            double l2 = compute_lambda2(n);
            double ram = d - 2.0 * sqrt(d - 1.0);
            printf("n=%d d=%d: lambda2=%.6f  Ram=%.4f  ratio=%.2fx  regular=%s\n",
                   n, d, l2, ram, l2 / ram, reg ? "YES" : "NO");
        }
    } else {
        /* Sweep */
        int ns[] = {32, 64, 128, 256, 512, 25, 33, 49, 63, 99, 127, 255,
                    31, 37, 61, 67, 97, 251, 509};
        int n_count = sizeof(ns) / sizeof(ns[0]);
        int ds[] = {3, 4, 5, 6, 7, 8};
        int d_count = sizeof(ds) / sizeof(ds[0]);

        printf("%5s %2s | %8s %7s | %6s %4s\n", "n", "d", "lambda2", "Ram", "ratio", "reg");
        printf("------------------------------------------\n");

        int total = 0, ok = 0;
        for (int ni = 0; ni < n_count; ni++) {
            for (int di = 0; di < d_count; di++) {
                int n = ns[ni], d = ds[di];
                if ((n * d) % 2 != 0 || d >= n || n > MAX_N) continue;

                total++;
                build_qrs_dr(n, d);
                int reg = verify_regular(n, d);
                if (reg) {
                    double l2 = compute_lambda2(n);
                    double ram = d - 2.0 * sqrt(d - 1.0);
                    double ratio = l2 / ram;
                    if (ratio >= 0.99) ok++;
                    printf("%5d %2d | %8.4f %7.4f | %5.2fx %4s\n",
                           n, d, l2, ram, ratio, "YES");
                } else {
                    printf("%5d %2d | %8s %7s | %6s %4s  [%d,%d]\n",
                           n, d, "FAIL", "", "", "NO", degs[0], degs[1]);
                }
            }
        }
        printf("\nAbove Ramanujan: %d/%d\n", ok, total);
    }

    return 0;
}
