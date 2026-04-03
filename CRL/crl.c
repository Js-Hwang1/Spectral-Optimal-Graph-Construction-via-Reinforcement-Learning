/*
 * crl.c — Full C PPO Training for G(n,m) algebraic connectivity.
 *
 * Episode collection + MLP backward + AdamW + GAE + PPO update.
 * LAPACK/CBLAS: Accelerate on macOS, OpenBLAS on Linux.
 */

#include "crl.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
/* LAPACK */
extern void dsyev_(char *jobz, char *uplo, int *n, double *a, int *lda,
                   double *w, double *work, int *lwork, int *info);
/* CBLAS */
extern void cblas_sgemm(int Order, int TransA, int TransB,
                        int M, int N, int K,
                        float alpha, const float *A, int lda,
                        const float *B, int ldb,
                        float beta, float *C, int ldc);
#ifndef CblasRowMajor
#define CblasRowMajor 101
#define CblasNoTrans  111
#define CblasTrans    112
#endif
#endif

/* ========================================================================== */
/* RNG (xorshift64)                                                            */
/* ========================================================================== */

void rng_init(RNG *rng, uint64_t seed) {
    rng->state = seed ? seed : 12345ULL;
}

uint64_t rng_next(RNG *rng) {
    rng->state ^= rng->state << 13;
    rng->state ^= rng->state >> 7;
    rng->state ^= rng->state << 17;
    return rng->state;
}

int rng_int(RNG *rng, int n) {
    return (int)(rng_next(rng) % (uint64_t)n);
}

double rng_double(RNG *rng) {
    return (double)(rng_next(rng) & 0x7FFFFFFFULL) / (double)0x7FFFFFFFULL;
}

/* ========================================================================== */
/* Graph operations                                                            */
/* ========================================================================== */

void build_ring_random(uint8_t *adj, int *degrees, int n, int m, RNG *rng) {
    memset(adj, 0, (size_t)n * n);
    memset(degrees, 0, (size_t)n * sizeof(int));

    /* Ring: n edges */
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        adj[i * n + j] = 1;
        adj[j * n + i] = 1;
        degrees[i]++;
        degrees[j]++;
    }

    /* Random residual edges */
    int remaining = m - n;
    if (remaining <= 0) return;

    /* Collect non-edges (upper triangle) */
    int max_ne = n * (n - 1) / 2 - n;  /* total possible - ring edges */
    int *ne_i = (int *)malloc((size_t)max_ne * sizeof(int));
    int *ne_j = (int *)malloc((size_t)max_ne * sizeof(int));
    int ne_count = 0;

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (!adj[i * n + j]) {
                ne_i[ne_count] = i;
                ne_j[ne_count] = j;
                ne_count++;
            }
        }
    }

    /* Fisher-Yates shuffle */
    for (int i = ne_count - 1; i > 0; i--) {
        int k = rng_int(rng, i + 1);
        int ti = ne_i[i]; ne_i[i] = ne_i[k]; ne_i[k] = ti;
        int tj = ne_j[i]; ne_j[i] = ne_j[k]; ne_j[k] = tj;
    }

    int to_add = remaining < ne_count ? remaining : ne_count;
    for (int k = 0; k < to_add; k++) {
        int i = ne_i[k], j = ne_j[k];
        adj[i * n + j] = 1;
        adj[j * n + i] = 1;
        degrees[i]++;
        degrees[j]++;
    }

    free(ne_i);
    free(ne_j);
}

/* ========================================================================== */
/* Eigenvalue computation via LAPACK dsyev_                                    */
/* ========================================================================== */

double exact_lambda2(const uint8_t *adj, int n) {
    if (n < 2) return 0.0;

    double *L = (double *)malloc((size_t)n * n * sizeof(double));
    double *eigvals = (double *)malloc((size_t)n * sizeof(double));
    if (!L || !eigvals) { free(L); free(eigvals); return 0.0; }

    /* Build Laplacian L = D - A */
    for (int i = 0; i < n; i++) {
        double deg = 0.0;
        for (int j = 0; j < n; j++) {
            if (adj[i * n + j]) {
                deg += 1.0;
                L[i * n + j] = -1.0;
            } else {
                L[i * n + j] = 0.0;
            }
        }
        L[i * n + i] = deg;
    }

    char jobz = 'N', uplo = 'U';
    int info, lwork = -1;
    double work_query;

    dsyev_(&jobz, &uplo, &n, L, &n, eigvals, &work_query, &lwork, &info);
    lwork = (int)work_query + 256;
    double *work = (double *)malloc((size_t)lwork * sizeof(double));
    if (!work) { free(L); free(eigvals); return 0.0; }

    dsyev_(&jobz, &uplo, &n, L, &n, eigvals, work, &lwork, &info);

    double result = (info == 0 && n > 1) ? eigvals[1] : 0.0;
    free(L); free(eigvals); free(work);
    return result;
}

/* ========================================================================== */
/* MLP forward (SiLU activation)                                               */
/* ========================================================================== */

static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

float mlp_forward_single(const MLP *mlp, const float *features) {
    int in_dim = mlp->in_dim;
    float h0[HIDDEN_DIM], h1[HIDDEN_DIM];

    /* Layer 0: W0 @ x + b0, SiLU
     * W0 is (HIDDEN_DIM, in_dim) row-major: W0[out * in_dim + in] */
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float sum = mlp->b0[i];
        for (int j = 0; j < in_dim; j++) {
            sum += mlp->W0[i * in_dim + j] * features[j];
        }
        h0[i] = silu(sum);
    }

    /* Layer 1: W1 @ h0 + b1, SiLU */
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float sum = mlp->b1[i];
        for (int j = 0; j < HIDDEN_DIM; j++) {
            sum += mlp->W1[i * HIDDEN_DIM + j] * h0[j];
        }
        h1[i] = silu(sum);
    }

    /* Layer 2: W2 @ h1 + b2 -> scalar */
    float out = mlp->b2[0];
    for (int j = 0; j < HIDDEN_DIM; j++) {
        out += mlp->W2[j] * h1[j];
    }
    return out;
}

void mlp_forward_batch(const MLP *mlp, const float *features,
                       int count, float *logits) {
    if (count <= 0) return;

    /* For tiny batches, scalar loop avoids BLAS call overhead */
    if (count < 8) {
        int in_dim = mlp->in_dim;
        for (int c = 0; c < count; c++)
            logits[c] = mlp_forward_single(mlp, features + c * in_dim);
        return;
    }

    int in_dim = mlp->in_dim;

    /* Intermediate buffers for hidden activations */
    float *H0 = (float *)malloc((size_t)count * HIDDEN_DIM * sizeof(float));
    float *H1 = (float *)malloc((size_t)count * HIDDEN_DIM * sizeof(float));

    /* Layer 0: H0 = SiLU(features @ W0^T + b0)
     * W0 is (HIDDEN_DIM, in_dim) row-major.
     * sgemm: C(count, HIDDEN_DIM) = features(count, in_dim) × W0^T(in_dim, HIDDEN_DIM) */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                count, HIDDEN_DIM, in_dim,
                1.0f, features, in_dim,
                mlp->W0, in_dim,
                0.0f, H0, HIDDEN_DIM);
    for (int i = 0; i < count; i++) {
        float *row = H0 + i * HIDDEN_DIM;
        for (int j = 0; j < HIDDEN_DIM; j++)
            row[j] = silu(row[j] + mlp->b0[j]);
    }

    /* Layer 1: H1 = SiLU(H0 @ W1^T + b1) */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                count, HIDDEN_DIM, HIDDEN_DIM,
                1.0f, H0, HIDDEN_DIM,
                mlp->W1, HIDDEN_DIM,
                0.0f, H1, HIDDEN_DIM);
    for (int i = 0; i < count; i++) {
        float *row = H1 + i * HIDDEN_DIM;
        for (int j = 0; j < HIDDEN_DIM; j++)
            row[j] = silu(row[j] + mlp->b1[j]);
    }

    /* Layer 2: logits = H1 @ W2 + b2
     * W2 is (HIDDEN_DIM,) = column vector, treat as (HIDDEN_DIM, 1).
     * sgemm: C(count, 1) = H1(count, HIDDEN_DIM) × W2(HIDDEN_DIM, 1) */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                count, 1, HIDDEN_DIM,
                1.0f, H1, HIDDEN_DIM,
                mlp->W2, 1,
                0.0f, logits, 1);
    float b2 = mlp->b2[0];
    for (int i = 0; i < count; i++)
        logits[i] += b2;

    free(H0);
    free(H1);
}

float value_mlp_forward(const MLP *mlp, const float *gfeat) {
    /* val_mlp has in_dim = GRAPH_FEAT_DIM (3) but MLP struct uses same W0 layout */
    return mlp_forward_single(mlp, gfeat);
}

/* ========================================================================== */
/* Lanczos with full reorthogonalization                                       */
/* ========================================================================== */

void lanczos_ext_k(const uint8_t *adj, int n,
                   const double *v_init, int k_lanczos, int n_eig,
                   double *V_out, double *lams_out) {
    if (n < 3 || n_eig < 1) {
        memset(V_out, 0, (size_t)n * n_eig * sizeof(double));
        memset(lams_out, 0, (size_t)n_eig * sizeof(double));
        if (n >= 2) {
            V_out[0 * n_eig + 0] = 1.0 / sqrt(2.0);
            V_out[1 * n_eig + 0] = -1.0 / sqrt(2.0);
        }
        return;
    }

    /* Compute degrees */
    double *deg = (double *)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) {
        double d = 0;
        for (int j = 0; j < n; j++) d += adj[i * n + j];
        deg[i] = d;
    }

    int k = k_lanczos < n - 1 ? k_lanczos : n - 1;
    n_eig = n_eig < n - 1 ? n_eig : n - 1;

    /* Lanczos vectors Q: (n, k) column-major for convenience but stored row-major */
    /* Q[i * k + j] = Q_j[i] */
    double *Q = (double *)calloc((size_t)n * k, sizeof(double));
    double *alpha = (double *)calloc((size_t)k, sizeof(double));
    double *beta  = (double *)calloc((size_t)k, sizeof(double));
    double *w     = (double *)malloc((size_t)n * sizeof(double));

    /* Initialize v */
    if (v_init) {
        for (int i = 0; i < n; i++) Q[i * k + 0] = v_init[i];
    } else {
        /* Deterministic pseudo-random init */
        RNG tmp_rng;
        rng_init(&tmp_rng, 0);
        for (int i = 0; i < n; i++)
            Q[i * k + 0] = rng_double(&tmp_rng) * 2.0 - 1.0;
    }

    /* Project out all-ones and normalize */
    double mean_v = 0;
    for (int i = 0; i < n; i++) mean_v += Q[i * k + 0];
    mean_v /= n;
    for (int i = 0; i < n; i++) Q[i * k + 0] -= mean_v;

    double norm_v = 0;
    for (int i = 0; i < n; i++) norm_v += Q[i * k + 0] * Q[i * k + 0];
    norm_v = sqrt(norm_v);
    if (norm_v < 1e-10) {
        /* Fallback */
        RNG tmp_rng;
        rng_init(&tmp_rng, 42);
        for (int i = 0; i < n; i++)
            Q[i * k + 0] = rng_double(&tmp_rng) * 2.0 - 1.0;
        mean_v = 0;
        for (int i = 0; i < n; i++) mean_v += Q[i * k + 0];
        mean_v /= n;
        for (int i = 0; i < n; i++) Q[i * k + 0] -= mean_v;
        norm_v = 0;
        for (int i = 0; i < n; i++) norm_v += Q[i * k + 0] * Q[i * k + 0];
        norm_v = sqrt(norm_v);
    }
    for (int i = 0; i < n; i++) Q[i * k + 0] /= norm_v;

    int actual_k = k;

    for (int j = 0; j < k; j++) {
        /* w = L @ Q_j = deg * Q_j - A @ Q_j */
        for (int i = 0; i < n; i++) {
            double s = deg[i] * Q[i * k + j];
            for (int p = 0; p < n; p++) {
                if (adj[i * n + p]) s -= Q[p * k + j];
            }
            w[i] = s;
        }

        /* alpha[j] = Q_j . w */
        double a = 0;
        for (int i = 0; i < n; i++) a += Q[i * k + j] * w[i];
        alpha[j] = a;

        /* Three-term recurrence */
        if (j > 0) {
            for (int i = 0; i < n; i++)
                w[i] -= beta[j] * Q[i * k + (j - 1)];
        }
        for (int i = 0; i < n; i++)
            w[i] -= alpha[j] * Q[i * k + j];

        /* Full reorthogonalization */
        for (int p = 0; p <= j; p++) {
            double dot = 0;
            for (int i = 0; i < n; i++) dot += w[i] * Q[i * k + p];
            for (int i = 0; i < n; i++) w[i] -= dot * Q[i * k + p];
        }

        /* Project out all-ones */
        double mean_w = 0;
        for (int i = 0; i < n; i++) mean_w += w[i];
        mean_w /= n;
        for (int i = 0; i < n; i++) w[i] -= mean_w;

        double beta_next = 0;
        for (int i = 0; i < n; i++) beta_next += w[i] * w[i];
        beta_next = sqrt(beta_next);

        if (beta_next < 1e-12) {
            actual_k = j + 1;
            break;
        }

        if (j + 1 < k) {
            beta[j + 1] = beta_next;
            for (int i = 0; i < n; i++)
                Q[i * k + (j + 1)] = w[i] / beta_next;
        }
    }

    /* Build tridiagonal matrix T and solve eigenproblem */
    double *T = (double *)calloc((size_t)actual_k * actual_k, sizeof(double));
    for (int j = 0; j < actual_k; j++) {
        T[j * actual_k + j] = alpha[j];
        if (j + 1 < actual_k) {
            T[j * actual_k + (j + 1)] = beta[j + 1];
            T[(j + 1) * actual_k + j] = beta[j + 1];
        }
    }

    double *eig_vals = (double *)malloc((size_t)actual_k * sizeof(double));
    /* dsyev_ overwrites T with eigenvectors (column-major) */
    {
        char jobz = 'V', uplo = 'U';
        int info, lwork = -1, ak = actual_k;
        double work_query;
        dsyev_(&jobz, &uplo, &ak, T, &ak, eig_vals, &work_query, &lwork, &info);
        lwork = (int)work_query + 256;
        double *work_buf = (double *)malloc((size_t)lwork * sizeof(double));
        dsyev_(&jobz, &uplo, &ak, T, &ak, eig_vals, work_buf, &lwork, &info);
        free(work_buf);
    }
    /* T now has eigenvectors in column-major: T[col * actual_k + row] */

    /* Skip trivial eigenvalue ~0 */
    int idx0 = 0;
    if (actual_k > 1 && eig_vals[0] < 0.1 * fmax(eig_vals[1], 1e-10))
        idx0 = 1;

    int n_available = actual_k - idx0;
    int n_extract = n_eig < n_available ? n_eig : n_available;

    memset(V_out, 0, (size_t)n * n_eig * sizeof(double));
    memset(lams_out, 0, (size_t)n_eig * sizeof(double));

    for (int p = 0; p < n_extract; p++) {
        int col_idx = idx0 + p;
        /* V_out[:, p] = Q[:, :actual_k] @ T_eigvec[:, col_idx] */
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int j2 = 0; j2 < actual_k; j2++) {
                /* T eigvec: column col_idx, row j2 -> T[col_idx * actual_k + j2] */
                s += Q[i * k + j2] * T[col_idx * actual_k + j2];
            }
            V_out[i * n_eig + p] = s;
        }
        lams_out[p] = eig_vals[col_idx];

        /* Center and normalize */
        double mv = 0;
        for (int i = 0; i < n; i++) mv += V_out[i * n_eig + p];
        mv /= n;
        for (int i = 0; i < n; i++) V_out[i * n_eig + p] -= mv;

        double nv = 0;
        for (int i = 0; i < n; i++) nv += V_out[i * n_eig + p] * V_out[i * n_eig + p];
        nv = sqrt(nv);
        if (nv > 1e-10) {
            for (int i = 0; i < n; i++) V_out[i * n_eig + p] /= nv;
        }
    }

    /* Fill remaining eigenvalues with last extracted */
    for (int p = n_extract; p < n_eig; p++) {
        lams_out[p] = n_extract > 0 ? lams_out[n_extract - 1] : 0.0;
    }

    /* Sign convention: v2 aligned with v_init or sum positive */
    if (v_init) {
        double dot = 0;
        for (int i = 0; i < n; i++) dot += V_out[i * n_eig + 0] * v_init[i];
        if (dot < 0) {
            for (int i = 0; i < n; i++) V_out[i * n_eig + 0] = -V_out[i * n_eig + 0];
        }
    } else {
        double sum = 0;
        for (int i = 0; i < n; i++) sum += V_out[i * n_eig + 0];
        if (sum < 0) {
            for (int i = 0; i < n; i++) V_out[i * n_eig + 0] = -V_out[i * n_eig + 0];
        }
    }

    /* Sign convention for remaining: sum positive */
    for (int p = 1; p < n_extract; p++) {
        double sum = 0;
        for (int i = 0; i < n; i++) sum += V_out[i * n_eig + p];
        if (sum < 0) {
            for (int i = 0; i < n; i++) V_out[i * n_eig + p] = -V_out[i * n_eig + p];
        }
    }

    free(deg); free(Q); free(alpha); free(beta); free(w);
    free(T); free(eig_vals);
}

/* ========================================================================== */
/* Rayleigh-Ritz update                                                        */
/* ========================================================================== */

void rr_update(double *V, double *lams, int n, int k, int u, int v, double sign) {
    /* delta = V[u,:] - V[v,:] */
    double delta[RR_K];
    for (int p = 0; p < k; p++)
        delta[p] = V[u * k + p] - V[v * k + p];

    /* L_sub = diag(lams) + sign * delta @ delta^T */
    double L_sub[RR_K * RR_K];
    memset(L_sub, 0, sizeof(double) * k * k);
    for (int i = 0; i < k; i++) {
        L_sub[i * k + i] = lams[i];
        for (int j = 0; j < k; j++) {
            L_sub[i * k + j] += sign * delta[i] * delta[j];
        }
    }

    /* Solve k×k eigenproblem */
    double new_lams[RR_K];
    {
        char jobz = 'V', uplo = 'U';
        int info, lwork = -1, kk = k;
        double work_query;
        dsyev_(&jobz, &uplo, &kk, L_sub, &kk, new_lams, &work_query, &lwork, &info);
        lwork = (int)work_query + 256;
        double *work_buf = (double *)malloc((size_t)lwork * sizeof(double));
        dsyev_(&jobz, &uplo, &kk, L_sub, &kk, new_lams, work_buf, &lwork, &info);
        free(work_buf);
    }
    /* L_sub now contains eigenvectors R (column-major): R[col * k + row] */

    /* V_new = V @ R: (n, k) @ (k, k) -> (n, k) */
    double *V_new = (double *)malloc((size_t)n * k * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int p = 0; p < k; p++) {
            double s = 0;
            for (int q = 0; q < k; q++) {
                /* R column p, row q -> L_sub[p * k + q] */
                s += V[i * k + q] * L_sub[p * k + q];
            }
            V_new[i * k + p] = s;
        }
    }

    memcpy(V, V_new, (size_t)n * k * sizeof(double));
    memcpy(lams, new_lams, (size_t)k * sizeof(double));
    free(V_new);
}

/* ========================================================================== */
/* Bridge detection (iterative Tarjan)                                         */
/* ========================================================================== */

void find_bridges(const uint8_t *adj, int n, uint8_t *bridge_mask) {
    memset(bridge_mask, 0, (size_t)n * n);
    if (n <= 1) return;

    int *disc    = (int *)malloc((size_t)n * sizeof(int));
    int *low     = (int *)malloc((size_t)n * sizeof(int));
    int *parent  = (int *)malloc((size_t)n * sizeof(int));
    int timer = 0;

    for (int i = 0; i < n; i++) { disc[i] = -1; low[i] = -1; parent[i] = -1; }

    /* Build adjacency lists */
    int *adj_list = (int *)malloc((size_t)n * n * sizeof(int));
    int *adj_count = (int *)calloc((size_t)n, sizeof(int));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (adj[i * n + j]) {
                adj_list[i * n + adj_count[i]] = j;
                adj_count[i]++;
            }
        }
    }

    /* Stack: (node, neighbor_idx) */
    int *stack_node = (int *)malloc((size_t)n * sizeof(int));
    int *stack_ni   = (int *)malloc((size_t)n * sizeof(int));

    for (int start = 0; start < n; start++) {
        if (disc[start] != -1) continue;

        int sp = 0;
        stack_node[sp] = start;
        stack_ni[sp] = 0;
        disc[start] = low[start] = timer++;

        while (sp >= 0) {
            int u = stack_node[sp];
            int ni = stack_ni[sp];
            int found_child = 0;

            while (ni < adj_count[u]) {
                int v2 = adj_list[u * n + ni];
                ni++;
                stack_ni[sp] = ni;

                if (disc[v2] == -1) {
                    parent[v2] = u;
                    disc[v2] = low[v2] = timer++;
                    sp++;
                    stack_node[sp] = v2;
                    stack_ni[sp] = 0;
                    found_child = 1;
                    break;
                } else if (v2 != parent[u]) {
                    if (disc[v2] < low[u]) low[u] = disc[v2];
                }
            }

            if (!found_child) {
                if (sp > 0) {
                    int p2 = stack_node[sp - 1];
                    if (low[u] < low[p2]) low[p2] = low[u];
                    if (low[u] > disc[p2]) {
                        /* (p2, u) is a bridge */
                        bridge_mask[p2 * n + u] = 1;
                        bridge_mask[u * n + p2] = 1;
                    }
                }
                sp--;
            }
        }
    }

    free(disc); free(low); free(parent);
    free(adj_list); free(adj_count);
    free(stack_node); free(stack_ni);
}

/* ========================================================================== */
/* Feature computation                                                         */
/* ========================================================================== */

void build_edge_features(const double *V_rr, const int *degrees,
                         int n, int step, int k_steps, float *feat_out) {
    /*
     * 4-dim edge features (N,N,4):
     *   0: n * |v₂ᵢ - v₂ⱼ|²  — Fiedler gap
     *   1: degᵢ / (n-1)       — source degree
     *   2: degⱼ / (n-1)       — target degree
     *   3: step / K            — budget awareness
     *
     * V_rr layout: (n, RR_K) row-major: V_rr[i * RR_K + p]
     * feat_out: (N_MAX, N_MAX, EDGE_FEAT_DIM) row-major
     */
    double nm1 = fmax(n - 1, 1);
    float step_frac = (float)step / fmax(k_steps, 1);

    for (int i = 0; i < n; i++) {
        double v2i = V_rr[i * RR_K + 0];
        float degi = (float)(degrees[i] / nm1);

        for (int j = 0; j < n; j++) {
            float *f = feat_out + (i * N_MAX + j) * EDGE_FEAT_DIM;
            double v2j = V_rr[j * RR_K + 0];
            double gap = v2i - v2j;
            f[0] = (float)(n * gap * gap);
            f[1] = degi;
            f[2] = (float)(degrees[j] / nm1);
            f[3] = step_frac;
        }
    }
}

void build_graph_features(const double *lams_rr, const int *degrees,
                          int n, int step, int k_steps, float *gfeat_out) {
    /*
     * 3-dim graph features:
     *   0: mean_degree / (n-1)
     *   1: λ₂ / n
     *   2: step / K
     */
    double nm1 = fmax(n - 1, 1);
    double deg_sum = 0;
    for (int i = 0; i < n; i++) deg_sum += degrees[i];
    gfeat_out[0] = (float)((deg_sum / n) / nm1);
    gfeat_out[1] = (float)(lams_rr[0] / fmax(n, 1));
    gfeat_out[2] = (float)step / fmax(k_steps, 1);
}

/* ========================================================================== */
/* Row/col logit re-scoring                                                    */
/* ========================================================================== */

void rescore_node(const MLP *mlp, const double *V_rr, const int *degrees,
                  int n, int step, int k_steps, int node, float *logits) {
    double nm1 = fmax(n - 1, 1);
    float step_frac = (float)step / fmax(k_steps, 1);
    double v2_nd = V_rr[node * RR_K + 0];
    float deg_nd = (float)(degrees[node] / nm1);

    float feat[EDGE_FEAT_DIM];

    /* Row: (node, j) for all j */
    for (int j = 0; j < n; j++) {
        double v2j = V_rr[j * RR_K + 0];
        double gap = v2_nd - v2j;
        feat[0] = (float)(n * gap * gap);
        feat[1] = deg_nd;
        feat[2] = (float)(degrees[j] / nm1);
        feat[3] = step_frac;
        logits[node * n + j] = mlp_forward_single(mlp, feat);
    }

    /* Col: (j, node) for all j */
    for (int j = 0; j < n; j++) {
        double v2j = V_rr[j * RR_K + 0];
        double gap = v2j - v2_nd;
        feat[0] = (float)(n * gap * gap);
        feat[1] = (float)(degrees[j] / nm1);
        feat[2] = deg_nd;
        feat[3] = step_frac;
        logits[j * n + node] = mlp_forward_single(mlp, feat);
    }
}

/* ========================================================================== */
/* Softmax sampling                                                            */
/* ========================================================================== */

int softmax_sample(const float *logits, const uint8_t *mask, int total,
                   RNG *rng, double *log_prob_out) {
    /* Find valid indices and max logit */
    int *valid = (int *)malloc((size_t)total * sizeof(int));
    int nv = 0;
    float max_logit = -FLT_MAX;

    for (int i = 0; i < total; i++) {
        if (mask[i]) {
            valid[nv++] = i;
            if (logits[i] > max_logit) max_logit = logits[i];
        }
    }

    if (nv == 0) {
        free(valid);
        *log_prob_out = 0.0;
        return -1;
    }

    /* Compute exp and sum */
    double *probs = (double *)malloc((size_t)nv * sizeof(double));
    double sum = 0;
    for (int i = 0; i < nv; i++) {
        probs[i] = exp((double)(logits[valid[i]] - max_logit));
        sum += probs[i];
    }

    /* Sample */
    double r = rng_double(rng);
    double csum = 0;
    int sel = nv - 1;  /* fallback */
    for (int i = 0; i < nv; i++) {
        csum += probs[i] / sum;
        if (r < csum) {
            sel = i;
            break;
        }
    }

    *log_prob_out = log(probs[sel] / sum);
    int result = valid[sel];
    free(valid);
    free(probs);
    return result;
}

/* ========================================================================== */
/* Weight loading                                                              */
/* ========================================================================== */

static void load_mlp(MLP *mlp, int in_dim,
                     const float *W0, const float *b0,
                     const float *W1, const float *b1,
                     const float *W2, const float *b2) {
    mlp->in_dim = in_dim;
    memcpy(mlp->W0, W0, (size_t)HIDDEN_DIM * in_dim * sizeof(float));
    memcpy(mlp->b0, b0, (size_t)HIDDEN_DIM * sizeof(float));
    memcpy(mlp->W1, W1, (size_t)HIDDEN_DIM * HIDDEN_DIM * sizeof(float));
    memcpy(mlp->b1, b1, (size_t)HIDDEN_DIM * sizeof(float));
    memcpy(mlp->W2, W2, (size_t)HIDDEN_DIM * sizeof(float));
    memcpy(mlp->b2, b2, sizeof(float));
}

void crl_load_weights(PolicyWeights *pw,
                      const float *add_W0, const float *add_b0,
                      const float *add_W1, const float *add_b1,
                      const float *add_W2, const float *add_b2,
                      const float *rem_W0, const float *rem_b0,
                      const float *rem_W1, const float *rem_b1,
                      const float *rem_W2, const float *rem_b2,
                      const float *val_W0, const float *val_b0,
                      const float *val_W1, const float *val_b1,
                      const float *val_W2, const float *val_b2) {
    load_mlp(&pw->add_mlp, EDGE_FEAT_DIM, add_W0, add_b0, add_W1, add_b1, add_W2, add_b2);
    load_mlp(&pw->rem_mlp, EDGE_FEAT_DIM, rem_W0, rem_b0, rem_W1, rem_b1, rem_W2, rem_b2);
    load_mlp(&pw->val_mlp, GRAPH_FEAT_DIM, val_W0, val_b0, val_W1, val_b1, val_W2, val_b2);
}

/* ========================================================================== */
/* Episode collection                                                          */
/* ========================================================================== */

int crl_collect_episode(
    const PolicyWeights *pw,
    int n, int m, int k_steps, double swap_frac, uint64_t seed,
    SwapTxn *add_txns, SwapTxn *rem_txns,
    int *out_num_add, int *out_num_rem,
    double *out_metrics,
    int max_txns
) {
    if (n > N_MAX || n < 2) return -1;

    RNG rng;
    rng_init(&rng, seed);

    /* Adjacency and degrees */
    uint8_t adj[N_MAX * N_MAX];
    int degrees[N_MAX];
    build_ring_random(adj, degrees, n, m, &rng);

    double initial_lambda2 = exact_lambda2(adj, n);
    double lambda2_cache = initial_lambda2;

    /* RR subspace: V_rr (n, RR_K), lams_rr (RR_K) */
    double V_rr[N_MAX * RR_K];
    double lams_rr[RR_K];
    double v_warm[N_MAX];  /* warm-start for Lanczos */
    int has_warm = 0;

    /* Working buffers */
    float edge_feat[N_MAX * N_MAX * EDGE_FEAT_DIM];
    float add_logits[N_MAX * N_MAX];
    float rem_logits[N_MAX * N_MAX];
    uint8_t upper_tri[N_MAX * N_MAX];
    uint8_t bridge_mask_buf[N_MAX * N_MAX];

    /* Build upper triangle mask */
    memset(upper_tri, 0, sizeof(upper_tri));
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            upper_tri[i * n + j] = 1;

    int num_swaps = (int)(m * swap_frac);
    if (num_swaps < 1) num_swaps = 1;

    int total_add = 0, total_rem = 0;

    for (int kk = 0; kk < k_steps; kk++) {
        /* Lanczos cold/warm start -> RR subspace */
        lanczos_ext_k(adj, n, has_warm ? v_warm : NULL,
                      LANCZOS_K, RR_K, V_rr, lams_rr);
        /* Update warm-start vector */
        for (int i = 0; i < n; i++) v_warm[i] = V_rr[i * RR_K + 0];
        has_warm = 1;

        /* Build features and initial logits */
        build_edge_features(V_rr, degrees, n, kk, k_steps, edge_feat);

        /* Score all pairs for add and remove */
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float *f = edge_feat + (i * N_MAX + j) * EDGE_FEAT_DIM;
                add_logits[i * n + j] = mlp_forward_single(&pw->add_mlp, f);
                rem_logits[i * n + j] = mlp_forward_single(&pw->rem_mlp, f);
            }
        }

        for (int s = 0; s < num_swaps; s++) {
            if (total_add >= max_txns || total_rem >= max_txns) break;

            /* === Graph features for value === */
            float gfeat[GRAPH_FEAT_DIM];
            build_graph_features(lams_rr, degrees, n, kk, k_steps, gfeat);
            float val = value_mlp_forward(&pw->val_mlp, gfeat);

            /* === ADD phase === */
            uint8_t add_mask[N_MAX * N_MAX];
            memset(add_mask, 0, sizeof(add_mask));
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                    if (!adj[i * n + j])
                        add_mask[i * n + j] = 1;

            /* Store features for PPO replay */
            SwapTxn *at = &add_txns[total_add];
            memset(at, 0, sizeof(SwapTxn));
            /* Copy current features (n,n,4) padded to N_MAX */
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                    float *src = edge_feat + (i * N_MAX + j) * EDGE_FEAT_DIM;
                    float *dst = at->feat + (i * N_MAX + j) * EDGE_FEAT_DIM;
                    memcpy(dst, src, EDGE_FEAT_DIM * sizeof(float));
                    at->mask[i * N_MAX + j] = add_mask[i * n + j];
                }
            memcpy(at->gfeat, gfeat, sizeof(gfeat));
            at->value = (double)val;

            /* Sample add */
            double add_lp;
            int add_flat = softmax_sample(add_logits, add_mask, n * n, &rng, &add_lp);
            if (add_flat < 0) break;  /* no valid adds */

            int ai = add_flat / n, aj = add_flat % n;
            at->flat_idx = ai * N_MAX + aj;  /* N_MAX-based flat index for Python */
            at->log_prob = add_lp;

            /* Add edge */
            adj[ai * n + aj] = 1;
            adj[aj * n + ai] = 1;
            degrees[ai]++;
            degrees[aj]++;

            /* RR update for add */
            rr_update(V_rr, lams_rr, n, RR_K, ai, aj, +1.0);

            /* Rescore affected nodes */
            rescore_node(&pw->add_mlp, V_rr, degrees, n, kk, k_steps, ai, add_logits);
            rescore_node(&pw->add_mlp, V_rr, degrees, n, kk, k_steps, aj, add_logits);
            rescore_node(&pw->rem_mlp, V_rr, degrees, n, kk, k_steps, ai, rem_logits);
            rescore_node(&pw->rem_mlp, V_rr, degrees, n, kk, k_steps, aj, rem_logits);

            /* Rebuild features for remove phase snapshot */
            build_edge_features(V_rr, degrees, n, kk, k_steps, edge_feat);

            /* === REMOVE phase === */
            find_bridges(adj, n, bridge_mask_buf);

            uint8_t rem_mask[N_MAX * N_MAX];
            memset(rem_mask, 0, sizeof(rem_mask));
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                    if (adj[i * n + j] && !bridge_mask_buf[i * n + j]) {
                        /* Exclude just-added edge */
                        int a2 = ai < aj ? ai : aj;
                        int b2 = ai < aj ? aj : ai;
                        if (i == a2 && j == b2) continue;
                        rem_mask[i * n + j] = 1;
                    }

            /* Check if any valid removals exist */
            int has_valid_rem = 0;
            for (int i = 0; i < n * n && !has_valid_rem; i++)
                if (rem_mask[i]) has_valid_rem = 1;

            if (!has_valid_rem) {
                /* Undo add */
                adj[ai * n + aj] = 0;
                adj[aj * n + ai] = 0;
                degrees[ai]--;
                degrees[aj]--;
                rr_update(V_rr, lams_rr, n, RR_K, ai, aj, -1.0);
                rescore_node(&pw->add_mlp, V_rr, degrees, n, kk, k_steps, ai, add_logits);
                rescore_node(&pw->add_mlp, V_rr, degrees, n, kk, k_steps, aj, add_logits);
                rescore_node(&pw->rem_mlp, V_rr, degrees, n, kk, k_steps, ai, rem_logits);
                rescore_node(&pw->rem_mlp, V_rr, degrees, n, kk, k_steps, aj, rem_logits);
                build_edge_features(V_rr, degrees, n, kk, k_steps, edge_feat);
                break;
            }

            /* Store rem features */
            SwapTxn *rt = &rem_txns[total_rem];
            memset(rt, 0, sizeof(SwapTxn));
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                    float *src = edge_feat + (i * N_MAX + j) * EDGE_FEAT_DIM;
                    float *dst = rt->feat + (i * N_MAX + j) * EDGE_FEAT_DIM;
                    memcpy(dst, src, EDGE_FEAT_DIM * sizeof(float));
                    rt->mask[i * N_MAX + j] = rem_mask[i * n + j];
                }
            memcpy(rt->gfeat, gfeat, sizeof(gfeat));
            rt->value = (double)val;

            /* Sample remove */
            double rem_lp;
            int rem_flat = softmax_sample(rem_logits, rem_mask, n * n, &rng, &rem_lp);
            if (rem_flat < 0) {
                /* Shouldn't happen since we checked has_valid_rem */
                adj[ai * n + aj] = 0;
                adj[aj * n + ai] = 0;
                degrees[ai]--;
                degrees[aj]--;
                rr_update(V_rr, lams_rr, n, RR_K, ai, aj, -1.0);
                break;
            }

            int ri = rem_flat / n, rj = rem_flat % n;
            rt->flat_idx = ri * N_MAX + rj;
            rt->log_prob = rem_lp;

            /* Remove edge */
            adj[ri * n + rj] = 0;
            adj[rj * n + ri] = 0;
            degrees[ri]--;
            degrees[rj]--;

            /* RR update for remove */
            rr_update(V_rr, lams_rr, n, RR_K, ri, rj, -1.0);

            /* Rescore affected nodes */
            rescore_node(&pw->add_mlp, V_rr, degrees, n, kk, k_steps, ri, add_logits);
            rescore_node(&pw->add_mlp, V_rr, degrees, n, kk, k_steps, rj, add_logits);
            rescore_node(&pw->rem_mlp, V_rr, degrees, n, kk, k_steps, ri, rem_logits);
            rescore_node(&pw->rem_mlp, V_rr, degrees, n, kk, k_steps, rj, rem_logits);

            /* Rebuild edge features after remove */
            build_edge_features(V_rr, degrees, n, kk, k_steps, edge_feat);

            /* === Per-swap exact reward === */
            double lambda2_after = exact_lambda2(adj, n);
            double reward = lambda2_after - lambda2_cache;
            lambda2_cache = lambda2_after;

            at->reward = reward;
            rt->reward = reward;  /* Same reward for both transitions in the swap */

            total_add++;
            total_rem++;
        }
    }

    *out_num_add = total_add;
    *out_num_rem = total_rem;

    double final_lambda2 = exact_lambda2(adj, n);
    out_metrics[0] = final_lambda2;
    out_metrics[1] = initial_lambda2;
    out_metrics[2] = final_lambda2 - initial_lambda2;
    out_metrics[3] = (double)n;
    out_metrics[4] = (double)m;

    return 0;
}

/* ========================================================================== */
/* SiLU derivative                                                             */
/* ========================================================================== */

static inline float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float silu_deriv(float x) {
    float s = sigmoidf(x);
    return s * (1.0f + x * (1.0f - s));
}

/* ========================================================================== */
/* MLP forward with activation caching                                         */
/* ========================================================================== */

float mlp_forward_cached(const MLP *mlp, const float *features, MLPCache *cache) {
    int in_dim = mlp->in_dim;
    cache->in_dim = in_dim;
    for (int j = 0; j < in_dim; j++) cache->input[j] = features[j];

    /* Layer 0: W0 @ x + b0, SiLU */
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float sum = mlp->b0[i];
        for (int j = 0; j < in_dim; j++)
            sum += mlp->W0[i * in_dim + j] * features[j];
        cache->z0[i] = sum;
        cache->h0[i] = silu(sum);
    }

    /* Layer 1: W1 @ h0 + b1, SiLU */
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float sum = mlp->b1[i];
        for (int j = 0; j < HIDDEN_DIM; j++)
            sum += mlp->W1[i * HIDDEN_DIM + j] * cache->h0[j];
        cache->z1[i] = sum;
        cache->h1[i] = silu(sum);
    }

    /* Layer 2: W2 @ h1 + b2 -> scalar */
    float out = mlp->b2[0];
    for (int j = 0; j < HIDDEN_DIM; j++)
        out += mlp->W2[j] * cache->h1[j];
    cache->out = out;
    return out;
}

/* ========================================================================== */
/* MLP backward pass                                                           */
/* ========================================================================== */

void mlp_backward(const MLP *mlp, const MLPCache *cache,
                  float d_out, MLPGrad *grad) {
    int in_dim = cache->in_dim;

    /* Layer 2 grads: out = W2 @ h1 + b2 */
    grad->db2[0] += d_out;
    float dh1[HIDDEN_DIM];
    for (int j = 0; j < HIDDEN_DIM; j++) {
        grad->dW2[j] += d_out * cache->h1[j];
        dh1[j] = d_out * mlp->W2[j];
    }

    /* Layer 1 grads: h1 = silu(z1), z1 = W1 @ h0 + b1 */
    float dz1[HIDDEN_DIM];
    float dh0[HIDDEN_DIM];
    memset(dh0, 0, sizeof(dh0));
    for (int i = 0; i < HIDDEN_DIM; i++) {
        dz1[i] = dh1[i] * silu_deriv(cache->z1[i]);
        grad->db1[i] += dz1[i];
        for (int j = 0; j < HIDDEN_DIM; j++) {
            grad->dW1[i * HIDDEN_DIM + j] += dz1[i] * cache->h0[j];
            dh0[j] += dz1[i] * mlp->W1[i * HIDDEN_DIM + j];
        }
    }

    /* Layer 0 grads: h0 = silu(z0), z0 = W0 @ input + b0 */
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float dz0 = dh0[i] * silu_deriv(cache->z0[i]);
        grad->db0[i] += dz0;
        for (int j = 0; j < in_dim; j++)
            grad->dW0[i * in_dim + j] += dz0 * cache->input[j];
    }
}

/* ========================================================================== */
/* Gradient utilities                                                          */
/* ========================================================================== */

static void mlp_grad_zero(MLPGrad *g) {
    memset(g, 0, sizeof(MLPGrad));
}

void policy_grad_zero(PolicyGrad *pg) {
    mlp_grad_zero(&pg->add_grad);
    mlp_grad_zero(&pg->rem_grad);
    mlp_grad_zero(&pg->val_grad);
}

/* Collect all params from one MLPGrad into a flat buffer, return count */
static int mlp_grad_flatten(const MLPGrad *g, int in_dim, float *buf) {
    int off = 0;
    memcpy(buf + off, g->dW0, (size_t)(HIDDEN_DIM * in_dim) * sizeof(float)); off += HIDDEN_DIM * in_dim;
    memcpy(buf + off, g->db0, (size_t)HIDDEN_DIM * sizeof(float)); off += HIDDEN_DIM;
    memcpy(buf + off, g->dW1, (size_t)(HIDDEN_DIM * HIDDEN_DIM) * sizeof(float)); off += HIDDEN_DIM * HIDDEN_DIM;
    memcpy(buf + off, g->db1, (size_t)HIDDEN_DIM * sizeof(float)); off += HIDDEN_DIM;
    memcpy(buf + off, g->dW2, (size_t)HIDDEN_DIM * sizeof(float)); off += HIDDEN_DIM;
    memcpy(buf + off, g->db2, sizeof(float)); off += 1;
    return off;
}


void policy_grad_add(PolicyGrad *dst, const PolicyGrad *src) {
    float *d = (float *)dst;
    const float *s = (const float *)src;
    int total = (int)(sizeof(PolicyGrad) / sizeof(float));
    for (int i = 0; i < total; i++) d[i] += s[i];
}

void policy_grad_scale(PolicyGrad *pg, float scale) {
    float *p = (float *)&pg->add_grad;
    int total = (int)(sizeof(PolicyGrad) / sizeof(float));
    for (int i = 0; i < total; i++) p[i] *= scale;
}

float policy_grad_norm(const PolicyGrad *pg) {
    const float *p = (const float *)pg;
    int total = (int)(sizeof(PolicyGrad) / sizeof(float));
    double sum = 0;
    for (int i = 0; i < total; i++) sum += (double)p[i] * p[i];
    return (float)sqrt(sum);
}

void policy_grad_clip(PolicyGrad *pg, float max_norm) {
    float norm = policy_grad_norm(pg);
    if (norm > max_norm) {
        policy_grad_scale(pg, max_norm / norm);
    }
}

/* ========================================================================== */
/* PPO re-evaluation (forward only, no backward)                               */
/* ========================================================================== */

/*
 * Masked log-softmax: compute log_prob for chosen action and entropy.
 * logits: (total,) all logits from MLP
 * mask: (total,) uint8, 1 = valid
 * chosen_flat: the flat index chosen during collection
 * n: graph size (logits are n*n but mask may index into N_MAX*N_MAX)
 */
static void masked_log_softmax(const float *logits, const uint8_t *mask,
                               int n, int chosen_flat_nmax,
                               double *out_lp, double *out_entropy) {
    /* chosen_flat_nmax is N_MAX-based, convert to n-based */
    int ci = chosen_flat_nmax / N_MAX;
    int cj = chosen_flat_nmax % N_MAX;
    int chosen_flat_n = ci * n + cj;

    /* Gather valid logits and find max */
    int valid_idx[N_MAX * N_MAX];
    int nv = 0;
    float max_logit = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (mask[i * N_MAX + j]) {
                valid_idx[nv] = i * n + j;
                if (logits[i * n + j] > max_logit)
                    max_logit = logits[i * n + j];
                nv++;
            }
        }
    }

    if (nv == 0) {
        *out_lp = 0.0;
        *out_entropy = 0.0;
        return;
    }

    /* log-sum-exp */
    double log_sum = 0;
    for (int i = 0; i < nv; i++)
        log_sum += exp((double)(logits[valid_idx[i]] - max_logit));
    log_sum = log(log_sum) + (double)max_logit;

    /* log_prob for chosen action */
    *out_lp = (double)logits[chosen_flat_n] - log_sum;

    /* entropy = -sum(p * log_p) */
    double ent = 0;
    for (int i = 0; i < nv; i++) {
        double lp_i = (double)logits[valid_idx[i]] - log_sum;
        double p_i = exp(lp_i);
        ent -= p_i * lp_i;
    }
    *out_entropy = ent;
}

void ppo_evaluate_swap(const PolicyWeights *pw,
                       const SwapTxn *add_txn, const SwapTxn *rem_txn,
                       int n,
                       double *out_log_prob, double *out_entropy,
                       float *out_value) {
    /* Value from add_txn graph features */
    *out_value = mlp_forward_single(&pw->val_mlp, add_txn->gfeat);

    /* Score all pairs for add */
    float add_logits[N_MAX * N_MAX];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            const float *f = add_txn->feat + (i * N_MAX + j) * EDGE_FEAT_DIM;
            add_logits[i * n + j] = mlp_forward_single(&pw->add_mlp, f);
        }

    double add_lp, add_ent;
    masked_log_softmax(add_logits, add_txn->mask, n, add_txn->flat_idx,
                       &add_lp, &add_ent);

    /* Score all pairs for rem */
    float rem_logits[N_MAX * N_MAX];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            const float *f = rem_txn->feat + (i * N_MAX + j) * EDGE_FEAT_DIM;
            rem_logits[i * n + j] = mlp_forward_single(&pw->rem_mlp, f);
        }

    double rem_lp, rem_ent;
    masked_log_softmax(rem_logits, rem_txn->mask, n, rem_txn->flat_idx,
                       &rem_lp, &rem_ent);

    *out_log_prob = add_lp + rem_lp;
    *out_entropy = add_ent + rem_ent;
}

/* ========================================================================== */
/* PPO backward for one swap                                                   */
/* ========================================================================== */

void ppo_backward_swap(const PolicyWeights *pw,
                       const SwapTxn *add_txn, const SwapTxn *rem_txn,
                       int n,
                       double advantage, double ret, double old_lp,
                       double ent_coef, double clip_eps,
                       PolicyGrad *pg) {
    /*
     * Fused: single forward pass per MLP computes log_prob + entropy
     * AND caches activations for backward. No separate ppo_evaluate_swap.
     */

    /* Value forward + cache for backward */
    MLPCache val_cache;
    float value = mlp_forward_cached(&pw->val_mlp, add_txn->gfeat, &val_cache);

    /* Phase 1: Forward all valid logits with caching, get log_prob */
    double add_lp, rem_lp;

    /* --- ADD MLP: forward + cache --- */
    int add_ci = add_txn->flat_idx / N_MAX;
    int add_cj = add_txn->flat_idx % N_MAX;
    int add_chosen_n = add_ci * n + add_cj;

    int add_valid_n[N_MAX * N_MAX];
    MLPCache *add_caches = (MLPCache *)malloc((size_t)(n * n) * sizeof(MLPCache));
    float add_logits[N_MAX * N_MAX];
    int add_nv = 0;
    float add_max = -FLT_MAX;

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (add_txn->mask[i * N_MAX + j]) {
                const float *f = add_txn->feat + (i * N_MAX + j) * EDGE_FEAT_DIM;
                float logit = mlp_forward_cached(&pw->add_mlp, f, &add_caches[add_nv]);
                add_logits[add_nv] = logit;
                add_valid_n[add_nv] = i * n + j;
                if (logit > add_max) add_max = logit;
                add_nv++;
            }

    /* Softmax + log_prob + entropy for add */
    double add_sm[N_MAX * N_MAX], add_lp_buf[N_MAX * N_MAX];
    int add_chosen_pos = -1;
    if (add_nv > 0) {
        double sum_e = 0;
        for (int i = 0; i < add_nv; i++) {
            add_sm[i] = exp((double)(add_logits[i] - add_max));
            sum_e += add_sm[i];
        }
        for (int i = 0; i < add_nv; i++) add_sm[i] /= sum_e;
        double lse = log(sum_e) + (double)add_max;
        for (int i = 0; i < add_nv; i++) add_lp_buf[i] = (double)add_logits[i] - lse;
        for (int i = 0; i < add_nv; i++)
            if (add_valid_n[i] == add_chosen_n) { add_chosen_pos = i; break; }
        add_lp = (add_chosen_pos >= 0) ? add_lp_buf[add_chosen_pos] : 0.0;
    } else {
        add_lp = 0;
    }

    /* --- REM MLP: forward + cache --- */
    int rem_ci = rem_txn->flat_idx / N_MAX;
    int rem_cj = rem_txn->flat_idx % N_MAX;
    int rem_chosen_n = rem_ci * n + rem_cj;

    int rem_valid_n[N_MAX * N_MAX];
    MLPCache *rem_caches = (MLPCache *)malloc((size_t)(n * n) * sizeof(MLPCache));
    float rem_logits[N_MAX * N_MAX];
    int rem_nv = 0;
    float rem_max = -FLT_MAX;

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (rem_txn->mask[i * N_MAX + j]) {
                const float *f = rem_txn->feat + (i * N_MAX + j) * EDGE_FEAT_DIM;
                float logit = mlp_forward_cached(&pw->rem_mlp, f, &rem_caches[rem_nv]);
                rem_logits[rem_nv] = logit;
                rem_valid_n[rem_nv] = i * n + j;
                if (logit > rem_max) rem_max = logit;
                rem_nv++;
            }

    /* Softmax + log_prob + entropy for rem */
    double rem_sm[N_MAX * N_MAX], rem_lp_buf[N_MAX * N_MAX];
    int rem_chosen_pos = -1;
    if (rem_nv > 0) {
        double sum_e = 0;
        for (int i = 0; i < rem_nv; i++) {
            rem_sm[i] = exp((double)(rem_logits[i] - rem_max));
            sum_e += rem_sm[i];
        }
        for (int i = 0; i < rem_nv; i++) rem_sm[i] /= sum_e;
        double lse = log(sum_e) + (double)rem_max;
        for (int i = 0; i < rem_nv; i++) rem_lp_buf[i] = (double)rem_logits[i] - lse;
        for (int i = 0; i < rem_nv; i++)
            if (rem_valid_n[i] == rem_chosen_n) { rem_chosen_pos = i; break; }
        rem_lp = (rem_chosen_pos >= 0) ? rem_lp_buf[rem_chosen_pos] : 0.0;
    } else {
        rem_lp = 0;
    }

    /* Phase 2: Compute PPO loss scalars using the forward results */
    double new_lp = add_lp + rem_lp;

    double ratio = exp(new_lp - old_lp);
    double surr1 = ratio * advantage;
    double clipped_ratio = ratio;
    if (clipped_ratio < 1.0 - clip_eps) clipped_ratio = 1.0 - clip_eps;
    if (clipped_ratio > 1.0 + clip_eps) clipped_ratio = 1.0 + clip_eps;
    double surr2 = clipped_ratio * advantage;

    float d_lp = 0.0f;
    if (surr1 <= surr2)
        d_lp = (float)(-ratio * advantage);

    float d_value = (float)(value - ret);
    float ec = (float)ent_coef;

    /* Phase 3: Backward pass using cached activations (no redundant forward) */

    /* ADD backward */
    if (add_nv > 0) {
        double add_ews = 0;
        for (int i = 0; i < add_nv; i++)
            add_ews += add_sm[i] * (add_lp_buf[i] + 1.0);
        for (int i = 0; i < add_nv; i++) {
            float d_logit = 0.0f;
            float ind = (i == add_chosen_pos) ? 1.0f : 0.0f;
            d_logit += d_lp * (ind - (float)add_sm[i]);
            d_logit += ec * (float)(add_sm[i] * add_ews - add_sm[i] * (add_lp_buf[i] + 1.0));
            mlp_backward(&pw->add_mlp, &add_caches[i], d_logit, &pg->add_grad);
        }
    }

    /* REM backward */
    if (rem_nv > 0) {
        double rem_ews = 0;
        for (int i = 0; i < rem_nv; i++)
            rem_ews += rem_sm[i] * (rem_lp_buf[i] + 1.0);
        for (int i = 0; i < rem_nv; i++) {
            float d_logit = 0.0f;
            float ind = (i == rem_chosen_pos) ? 1.0f : 0.0f;
            d_logit += d_lp * (ind - (float)rem_sm[i]);
            d_logit += ec * (float)(rem_sm[i] * rem_ews - rem_sm[i] * (rem_lp_buf[i] + 1.0));
            mlp_backward(&pw->rem_mlp, &rem_caches[i], d_logit, &pg->rem_grad);
        }
    }

    /* Value backward */
    mlp_backward(&pw->val_mlp, &val_cache, d_value, &pg->val_grad);

    free(add_caches);
    free(rem_caches);
}

/* ========================================================================== */
/* AdamW optimizer                                                             */
/* ========================================================================== */

void adam_init(AdamState *state) {
    memset(state, 0, sizeof(AdamState));
}

static void adam_step_mlp(MLP *mlp, const MLPGrad *grad, AdamMLPState *state,
                          int t, double lr, double beta1, double beta2,
                          double eps, double weight_decay) {
    /* Flatten weights and gradients */
    int in_dim = mlp->in_dim;
    float grad_buf[MLP_MAX_PARAMS];
    int np = mlp_grad_flatten(grad, in_dim, grad_buf);

    /* Flatten weights into same order */
    float *w_ptrs[] = { mlp->W0, mlp->b0, mlp->W1, mlp->b1, mlp->W2, mlp->b2 };
    int w_sizes[] = {
        HIDDEN_DIM * in_dim, HIDDEN_DIM,
        HIDDEN_DIM * HIDDEN_DIM, HIDDEN_DIM,
        HIDDEN_DIM, 1
    };

    /* Is this a bias (no weight decay)? Track offset */
    int is_bias[MLP_MAX_PARAMS];
    int off = 0;
    for (int layer = 0; layer < 6; layer++) {
        int is_b = (layer % 2 == 1); /* odd indices are biases */
        for (int i = 0; i < w_sizes[layer]; i++)
            is_bias[off++] = is_b;
    }

    /* Create flat weight buffer */
    float w_buf[MLP_MAX_PARAMS];
    off = 0;
    for (int layer = 0; layer < 6; layer++) {
        memcpy(w_buf + off, w_ptrs[layer], (size_t)w_sizes[layer] * sizeof(float));
        off += w_sizes[layer];
    }

    /* Bias correction */
    double bc1 = 1.0 - pow(beta1, t);
    double bc2 = 1.0 - pow(beta2, t);

    /* Adam update */
    for (int i = 0; i < np; i++) {
        float g = grad_buf[i];
        state->m[i] = (float)(beta1 * state->m[i] + (1.0 - beta1) * g);
        state->v[i] = (float)(beta2 * state->v[i] + (1.0 - beta2) * g * g);
        float m_hat = (float)(state->m[i] / bc1);
        float v_hat = (float)(state->v[i] / bc2);
        float wd = is_bias[i] ? 0.0f : (float)weight_decay;
        w_buf[i] -= (float)(lr * (m_hat / (sqrtf(v_hat) + eps) + wd * w_buf[i]));
    }

    /* Write back */
    off = 0;
    for (int layer = 0; layer < 6; layer++) {
        memcpy(w_ptrs[layer], w_buf + off, (size_t)w_sizes[layer] * sizeof(float));
        off += w_sizes[layer];
    }
}

void adam_step(PolicyWeights *pw, const PolicyGrad *pg, AdamState *state,
              double lr, double beta1, double beta2, double eps,
              double weight_decay) {
    state->t++;
    adam_step_mlp(&pw->add_mlp, &pg->add_grad, &state->add_state,
                  state->t, lr, beta1, beta2, eps, weight_decay);
    adam_step_mlp(&pw->rem_mlp, &pg->rem_grad, &state->rem_state,
                  state->t, lr, beta1, beta2, eps, weight_decay);
    adam_step_mlp(&pw->val_mlp, &pg->val_grad, &state->val_state,
                  state->t, lr, beta1, beta2, eps, weight_decay);
}

/* ========================================================================== */
/* GAE computation                                                             */
/* ========================================================================== */

void compute_gae(const double *values, const double *rewards,
                 int num_steps, double gamma, double gae_lambda,
                 double *advantages, double *returns) {
    double gae = 0.0;
    for (int t = num_steps - 1; t >= 0; t--) {
        double next_val = (t + 1 < num_steps) ? values[t + 1] : 0.0;
        double delta = rewards[t] + gamma * next_val - values[t];
        gae = delta + gamma * gae_lambda * gae;
        advantages[t] = gae;
        returns[t] = gae + values[t];
    }
}

void normalize_advantages(double *advantages, int count) {
    if (count <= 1) return;
    double mean = 0, var = 0;
    for (int i = 0; i < count; i++) mean += advantages[i];
    mean /= count;
    for (int i = 0; i < count; i++) {
        double d = advantages[i] - mean;
        var += d * d;
    }
    var /= count;
    double std = sqrt(var) + 1e-8;
    for (int i = 0; i < count; i++)
        advantages[i] = (advantages[i] - mean) / std;
}

/* ========================================================================== */
/* Checkpoint I/O                                                              */
/* ========================================================================== */

static void write_mlp_weights(FILE *f, const MLP *mlp) {
    int in_dim = mlp->in_dim;
    fwrite(mlp->W0, sizeof(float), (size_t)(HIDDEN_DIM * in_dim), f);
    fwrite(mlp->b0, sizeof(float), HIDDEN_DIM, f);
    fwrite(mlp->W1, sizeof(float), (size_t)(HIDDEN_DIM * HIDDEN_DIM), f);
    fwrite(mlp->b1, sizeof(float), HIDDEN_DIM, f);
    fwrite(mlp->W2, sizeof(float), HIDDEN_DIM, f);
    fwrite(mlp->b2, sizeof(float), 1, f);
}

static void read_mlp_weights(FILE *f, MLP *mlp, int in_dim) {
    mlp->in_dim = in_dim;
    fread(mlp->W0, sizeof(float), (size_t)(HIDDEN_DIM * in_dim), f);
    fread(mlp->b0, sizeof(float), HIDDEN_DIM, f);
    fread(mlp->W1, sizeof(float), (size_t)(HIDDEN_DIM * HIDDEN_DIM), f);
    fread(mlp->b1, sizeof(float), HIDDEN_DIM, f);
    fread(mlp->W2, sizeof(float), HIDDEN_DIM, f);
    fread(mlp->b2, sizeof(float), 1, f);
}

int save_checkpoint(const char *path, const PolicyWeights *pw,
                    uint32_t episode, float best_avg_improvement) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    CkptHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = CRL_CKPT_MAGIC;
    hdr.version = CRL_CKPT_VERSION;
    hdr.hidden_dim = HIDDEN_DIM;
    hdr.edge_feat_dim = EDGE_FEAT_DIM;
    hdr.graph_feat_dim = GRAPH_FEAT_DIM;
    hdr.episode = episode;
    hdr.best_avg_improvement = best_avg_improvement;
    fwrite(&hdr, sizeof(hdr), 1, f);

    write_mlp_weights(f, &pw->add_mlp);
    write_mlp_weights(f, &pw->rem_mlp);
    write_mlp_weights(f, &pw->val_mlp);

    fclose(f);
    return 0;
}

int load_checkpoint(const char *path, PolicyWeights *pw,
                    uint32_t *episode, float *best_avg_improvement) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    CkptHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return -1; }
    if (hdr.magic != CRL_CKPT_MAGIC) { fclose(f); return -2; }
    if (hdr.hidden_dim != HIDDEN_DIM) { fclose(f); return -3; }

    if (episode) *episode = hdr.episode;
    if (best_avg_improvement) *best_avg_improvement = hdr.best_avg_improvement;

    read_mlp_weights(f, &pw->add_mlp, EDGE_FEAT_DIM);
    read_mlp_weights(f, &pw->rem_mlp, EDGE_FEAT_DIM);
    read_mlp_weights(f, &pw->val_mlp, GRAPH_FEAT_DIM);

    fclose(f);
    return 0;
}

/* ========================================================================== */
/* Weight initialization (approximate orthogonal via QR of random matrix)      */
/* ========================================================================== */

static void init_linear_orthogonal(float *W, int rows, int cols, RNG *rng, float gain) {
    /* Fill with random normal (Box-Muller) */
    int total = rows * cols;
    for (int i = 0; i < total; i += 2) {
        double u1 = rng_double(rng) * 0.998 + 0.001;
        double u2 = rng_double(rng);
        double r = sqrt(-2.0 * log(u1));
        double t = 2.0 * M_PI * u2;
        W[i] = (float)(r * cos(t));
        if (i + 1 < total) W[i + 1] = (float)(r * sin(t));
    }

    /* Simple orthogonalization via Gram-Schmidt on rows (good enough for small dims) */
    int min_dim = rows < cols ? rows : cols;
    for (int i = 0; i < min_dim; i++) {
        /* Orthogonalize row i against previous rows */
        for (int j = 0; j < i; j++) {
            float dot = 0, norm_j = 0;
            for (int k = 0; k < cols; k++) {
                dot += W[i * cols + k] * W[j * cols + k];
                norm_j += W[j * cols + k] * W[j * cols + k];
            }
            if (norm_j > 1e-10f) {
                float scale = dot / norm_j;
                for (int k = 0; k < cols; k++)
                    W[i * cols + k] -= scale * W[j * cols + k];
            }
        }
        /* Normalize */
        float norm = 0;
        for (int k = 0; k < cols; k++) norm += W[i * cols + k] * W[i * cols + k];
        norm = sqrtf(norm);
        if (norm > 1e-10f) {
            float s = gain / norm;
            for (int k = 0; k < cols; k++) W[i * cols + k] *= s;
        }
    }
}

static void init_mlp_orthogonal(MLP *mlp, int in_dim, RNG *rng, float gain) {
    mlp->in_dim = in_dim;
    init_linear_orthogonal(mlp->W0, HIDDEN_DIM, in_dim, rng, gain);
    memset(mlp->b0, 0, sizeof(mlp->b0));
    init_linear_orthogonal(mlp->W1, HIDDEN_DIM, HIDDEN_DIM, rng, gain);
    memset(mlp->b1, 0, sizeof(mlp->b1));
    init_linear_orthogonal(mlp->W2, 1, HIDDEN_DIM, rng, gain);
    memset(mlp->b2, 0, sizeof(mlp->b2));
}

void policy_init_orthogonal(PolicyWeights *pw, RNG *rng, float gain) {
    init_mlp_orthogonal(&pw->add_mlp, EDGE_FEAT_DIM, rng, gain);
    init_mlp_orthogonal(&pw->rem_mlp, EDGE_FEAT_DIM, rng, gain);
    init_mlp_orthogonal(&pw->val_mlp, GRAPH_FEAT_DIM, rng, gain);
}
