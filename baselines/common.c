/*
 * common.c - Shared utilities implementation
 */

#include "common.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
extern void dsyev_(char *jobz, char *uplo, int *n, double *a, int *lda,
                   double *w, double *work, int *lwork, int *info);
#endif

/* ============================================================================
 * RANDOM NUMBER GENERATOR (xorshift64)
 * ============================================================================ */

static uint64_t rng_state = 12345ULL;

void rng_seed(uint64_t seed) {
    rng_state = seed ? seed : 12345ULL;
}

uint64_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

int rng_int(int n) {
    return (int)(rng_next() % (uint64_t)n);
}

double rng_double(void) {
    return (double)(rng_next() & 0x7FFFFFFFULL) / (double)0x7FFFFFFFULL;
}

/* ============================================================================
 * ADJACENCY MATRIX
 * ============================================================================ */

AdjMatrix *adj_create(int n) {
    AdjMatrix *adj = malloc(sizeof(AdjMatrix));
    if (!adj) return NULL;

    adj->n = n;
    adj->data = calloc((size_t)n * n, sizeof(bool));
    if (!adj->data) {
        free(adj);
        return NULL;
    }
    return adj;
}

void adj_free(AdjMatrix *adj) {
    if (adj) {
        free(adj->data);
        free(adj);
    }
}

bool adj_get(const AdjMatrix *adj, int i, int j) {
    return adj->data[i * adj->n + j];
}

void adj_set(AdjMatrix *adj, int i, int j, bool val) {
    adj->data[i * adj->n + j] = val;
}

int adj_edge_count(const AdjMatrix *adj) {
    int count = 0;
    int n = adj->n;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (adj_get(adj, i, j)) count++;
        }
    }
    return count;
}

void adj_copy(AdjMatrix *dst, const AdjMatrix *src) {
    memcpy(dst->data, src->data, (size_t)src->n * src->n * sizeof(bool));
}

/* ============================================================================
 * GRAPH INITIALIZATION
 * ============================================================================ */

void build_ring(AdjMatrix *adj) {
    int n = adj->n;
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        adj_set(adj, i, j, true);
        adj_set(adj, j, i, true);
    }
}

void build_random_tree(AdjMatrix *adj) {
    int n = adj->n;

    /* Fisher-Yates shuffle to get a random permutation */
    int *perm = malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) perm[i] = i;
    for (int i = n - 1; i > 0; i--) {
        int j = rng_int(i + 1);
        int tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
    }

    /* Connect each node to a random already-in-tree node */
    for (int i = 1; i < n; i++) {
        int u = perm[i];
        int v = perm[rng_int(i)];
        adj_set(adj, u, v, true);
        adj_set(adj, v, u, true);
    }

    free(perm);
}

/* ============================================================================
 * LAPLACIAN HELPERS
 * ============================================================================ */

static void build_laplacian(const AdjMatrix *adj, double *L) {
    int n = adj->n;

    for (int i = 0; i < n; i++) {
        double deg = 0.0;
        for (int j = 0; j < n; j++) {
            if (adj_get(adj, i, j)) {
                deg += 1.0;
                L[i * n + j] = -1.0;
            } else {
                L[i * n + j] = 0.0;
            }
        }
        L[i * n + i] = deg;
    }
}

/* ============================================================================
 * EIGENVALUE COMPUTATIONS
 * ============================================================================ */

double compute_algebraic_connectivity(const AdjMatrix *adj) {
    int n = adj->n;
    double *L = malloc((size_t)n * n * sizeof(double));
    double *eigenvalues = malloc((size_t)n * sizeof(double));

    if (!L || !eigenvalues) {
        free(L);
        free(eigenvalues);
        return 0.0;
    }

    build_laplacian(adj, L);

    char jobz = 'N';
    char uplo = 'U';
    int info;
    int lwork = -1;
    double work_query;

    dsyev_(&jobz, &uplo, &n, L, &n, eigenvalues,
           &work_query, &lwork, &info);

    lwork = (int)work_query + 256; /* pad for OpenBLAS overrun bug */
    double *work = malloc((size_t)lwork * sizeof(double));

    if (!work) {
        free(L);
        free(eigenvalues);
        return 0.0;
    }

    dsyev_(&jobz, &uplo, &n, L, &n, eigenvalues,
           work, &lwork, &info);

    double result = (info == 0 && n > 1) ? eigenvalues[1] : 0.0;

    free(L);
    free(eigenvalues);
    free(work);

    return result;
}

void compute_fiedler_vector(const AdjMatrix *adj, double *fiedler) {
    int n = adj->n;
    double *L = malloc((size_t)n * n * sizeof(double));
    double *eigenvalues = malloc((size_t)n * sizeof(double));

    if (!L || !eigenvalues) {
        for (int i = 0; i < n; i++) fiedler[i] = 0.0;
        free(L);
        free(eigenvalues);
        return;
    }

    build_laplacian(adj, L);

    char jobz = 'V';  /* compute eigenvectors */
    char uplo = 'U';
    int info;
    int lwork = -1;
    double work_query;

    dsyev_(&jobz, &uplo, &n, L, &n, eigenvalues,
           &work_query, &lwork, &info);

    lwork = (int)work_query + 256; /* pad for OpenBLAS overrun bug */
    double *work = malloc((size_t)lwork * sizeof(double));

    if (!work) {
        for (int i = 0; i < n; i++) fiedler[i] = 0.0;
        free(L);
        free(eigenvalues);
        return;
    }

    dsyev_(&jobz, &uplo, &n, L, &n, eigenvalues,
           work, &lwork, &info);

    /* Fiedler vector is 2nd eigenvector (column 1 in column-major order) */
    if (info == 0) {
        for (int i = 0; i < n; i++) {
            fiedler[i] = L[1 * n + i];
        }
    } else {
        for (int i = 0; i < n; i++) {
            fiedler[i] = 0.0;
        }
    }

    free(L);
    free(eigenvalues);
    free(work);
}

void compute_laplacian_pinv(const AdjMatrix *adj, double *L_pinv) {
    int n = adj->n;
    double *L = malloc((size_t)n * n * sizeof(double));
    double *eigenvalues = malloc((size_t)n * sizeof(double));

    if (!L || !eigenvalues) {
        for (int i = 0; i < n * n; i++) L_pinv[i] = 0.0;
        free(L);
        free(eigenvalues);
        return;
    }

    build_laplacian(adj, L);

    char jobz = 'V';
    char uplo = 'U';
    int info;
    int lwork = -1;
    double work_query;

    dsyev_(&jobz, &uplo, &n, L, &n, eigenvalues,
           &work_query, &lwork, &info);

    lwork = (int)work_query + 256; /* pad for OpenBLAS overrun bug */
    double *work = malloc((size_t)lwork * sizeof(double));

    if (!work) {
        for (int i = 0; i < n * n; i++) L_pinv[i] = 0.0;
        free(L);
        free(eigenvalues);
        return;
    }

    dsyev_(&jobz, &uplo, &n, L, &n, eigenvalues,
           work, &lwork, &info);

    /* Compute pseudoinverse: sum over non-zero eigenvalues of (1/λ) * v * v^T */
    double rcond = 1e-6;
    double max_eig = 0.0;
    for (int i = 0; i < n; i++) {
        if (eigenvalues[i] > max_eig) max_eig = eigenvalues[i];
    }
    double threshold = rcond * max_eig;

    for (int i = 0; i < n * n; i++) L_pinv[i] = 0.0;

    for (int k = 0; k < n; k++) {
        if (eigenvalues[k] > threshold) {
            double inv_lambda = 1.0 / eigenvalues[k];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    L_pinv[i * n + j] += inv_lambda * L[k * n + i] * L[k * n + j];
                }
            }
        }
    }

    free(L);
    free(eigenvalues);
    free(work);
}
