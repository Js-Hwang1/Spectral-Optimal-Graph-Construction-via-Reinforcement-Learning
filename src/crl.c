/*
 * crl.c — Dual-Phase Spectral Refine for G(n,m) algebraic connectivity.
 *
 * Core library: RNG, graph ops, Lanczos, bridges, MLP fwd/bwd,
 * dual-phase episode collection, PPO evaluate/backward, AdamW, checkpoint I/O.
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
extern void dsyev_(char *jobz, char *uplo, int *n, double *a, int *lda,
                   double *w, double *work, int *lwork, int *info);
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

    /* Random spanning tree: shuffle nodes, connect as a path (n-1 edges) */
    int *perm = (int *)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) perm[i] = i;
    for (int i = n - 1; i > 0; i--) {
        int k = rng_int(rng, i + 1);
        int t = perm[i]; perm[i] = perm[k]; perm[k] = t;
    }
    for (int i = 0; i < n - 1; i++) {
        int u = perm[i], v = perm[i + 1];
        adj[u * n + v] = 1;
        adj[v * n + u] = 1;
        degrees[u]++;
        degrees[v]++;
    }
    free(perm);

    int remaining = m - (n - 1);
    if (remaining <= 0) return;

    int max_ne = n * (n - 1) / 2 - (n - 1);
    int *ne_i = (int *)malloc((size_t)max_ne * sizeof(int));
    int *ne_j = (int *)malloc((size_t)max_ne * sizeof(int));
    int ne_count = 0;

    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (!adj[i * n + j]) {
                ne_i[ne_count] = i;
                ne_j[ne_count] = j;
                ne_count++;
            }

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

    for (int i = 0; i < HIDDEN_DIM; i++) {
        float sum = mlp->b0[i];
        for (int j = 0; j < in_dim; j++)
            sum += mlp->W0[i * in_dim + j] * features[j];
        h0[i] = silu(sum);
    }

    for (int i = 0; i < HIDDEN_DIM; i++) {
        float sum = mlp->b1[i];
        for (int j = 0; j < HIDDEN_DIM; j++)
            sum += mlp->W1[i * HIDDEN_DIM + j] * h0[j];
        h1[i] = silu(sum);
    }

    float out = mlp->b2[0];
    for (int j = 0; j < HIDDEN_DIM; j++)
        out += mlp->W2[j] * h1[j];
    return out;
}

void mlp_forward_batch(const MLP *mlp, const float *features,
                       int count, float *logits) {
    if (count <= 0) return;

    if (count < 8) {
        int in_dim = mlp->in_dim;
        for (int c = 0; c < count; c++)
            logits[c] = mlp_forward_single(mlp, features + c * in_dim);
        return;
    }

    int in_dim = mlp->in_dim;
    float *H0 = (float *)malloc((size_t)count * HIDDEN_DIM * sizeof(float));
    float *H1 = (float *)malloc((size_t)count * HIDDEN_DIM * sizeof(float));

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

    double *deg = (double *)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) {
        double d = 0;
        for (int j = 0; j < n; j++) d += adj[i * n + j];
        deg[i] = d;
    }

    int k = k_lanczos < n - 1 ? k_lanczos : n - 1;
    n_eig = n_eig < n - 1 ? n_eig : n - 1;

    double *Q = (double *)calloc((size_t)n * k, sizeof(double));
    double *alpha = (double *)calloc((size_t)k, sizeof(double));
    double *beta  = (double *)calloc((size_t)k, sizeof(double));
    double *w     = (double *)malloc((size_t)n * sizeof(double));

    if (v_init) {
        for (int i = 0; i < n; i++) Q[i * k + 0] = v_init[i];
    } else {
        RNG tmp_rng;
        rng_init(&tmp_rng, 0);
        for (int i = 0; i < n; i++)
            Q[i * k + 0] = rng_double(&tmp_rng) * 2.0 - 1.0;
    }

    double mean_v = 0;
    for (int i = 0; i < n; i++) mean_v += Q[i * k + 0];
    mean_v /= n;
    for (int i = 0; i < n; i++) Q[i * k + 0] -= mean_v;

    double norm_v = 0;
    for (int i = 0; i < n; i++) norm_v += Q[i * k + 0] * Q[i * k + 0];
    norm_v = sqrt(norm_v);
    if (norm_v < 1e-10) {
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
        for (int i = 0; i < n; i++) {
            double s = deg[i] * Q[i * k + j];
            for (int p = 0; p < n; p++)
                if (adj[i * n + p]) s -= Q[p * k + j];
            w[i] = s;
        }

        double a = 0;
        for (int i = 0; i < n; i++) a += Q[i * k + j] * w[i];
        alpha[j] = a;

        if (j > 0)
            for (int i = 0; i < n; i++)
                w[i] -= beta[j] * Q[i * k + (j - 1)];
        for (int i = 0; i < n; i++)
            w[i] -= alpha[j] * Q[i * k + j];

        for (int p = 0; p <= j; p++) {
            double dot = 0;
            for (int i = 0; i < n; i++) dot += w[i] * Q[i * k + p];
            for (int i = 0; i < n; i++) w[i] -= dot * Q[i * k + p];
        }

        double mean_w = 0;
        for (int i = 0; i < n; i++) mean_w += w[i];
        mean_w /= n;
        for (int i = 0; i < n; i++) w[i] -= mean_w;

        double beta_next = 0;
        for (int i = 0; i < n; i++) beta_next += w[i] * w[i];
        beta_next = sqrt(beta_next);

        if (beta_next < 1e-12) { actual_k = j + 1; break; }

        if (j + 1 < k) {
            beta[j + 1] = beta_next;
            for (int i = 0; i < n; i++)
                Q[i * k + (j + 1)] = w[i] / beta_next;
        }
    }

    double *T = (double *)calloc((size_t)actual_k * actual_k, sizeof(double));
    for (int j = 0; j < actual_k; j++) {
        T[j * actual_k + j] = alpha[j];
        if (j + 1 < actual_k) {
            T[j * actual_k + (j + 1)] = beta[j + 1];
            T[(j + 1) * actual_k + j] = beta[j + 1];
        }
    }

    double *eig_vals = (double *)malloc((size_t)actual_k * sizeof(double));
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

    int idx0 = 0;
    if (actual_k > 1 && eig_vals[0] < 0.1 * fmax(eig_vals[1], 1e-10))
        idx0 = 1;

    int n_available = actual_k - idx0;
    int n_extract = n_eig < n_available ? n_eig : n_available;

    memset(V_out, 0, (size_t)n * n_eig * sizeof(double));
    memset(lams_out, 0, (size_t)n_eig * sizeof(double));

    for (int p = 0; p < n_extract; p++) {
        int col_idx = idx0 + p;
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int j2 = 0; j2 < actual_k; j2++)
                s += Q[i * k + j2] * T[col_idx * actual_k + j2];
            V_out[i * n_eig + p] = s;
        }
        lams_out[p] = eig_vals[col_idx];

        double mv = 0;
        for (int i = 0; i < n; i++) mv += V_out[i * n_eig + p];
        mv /= n;
        for (int i = 0; i < n; i++) V_out[i * n_eig + p] -= mv;

        double nv = 0;
        for (int i = 0; i < n; i++) nv += V_out[i * n_eig + p] * V_out[i * n_eig + p];
        nv = sqrt(nv);
        if (nv > 1e-10)
            for (int i = 0; i < n; i++) V_out[i * n_eig + p] /= nv;
    }

    for (int p = n_extract; p < n_eig; p++)
        lams_out[p] = n_extract > 0 ? lams_out[n_extract - 1] : 0.0;

    if (v_init) {
        double dot = 0;
        for (int i = 0; i < n; i++) dot += V_out[i * n_eig + 0] * v_init[i];
        if (dot < 0)
            for (int i = 0; i < n; i++) V_out[i * n_eig + 0] = -V_out[i * n_eig + 0];
    } else {
        double sum = 0;
        for (int i = 0; i < n; i++) sum += V_out[i * n_eig + 0];
        if (sum < 0)
            for (int i = 0; i < n; i++) V_out[i * n_eig + 0] = -V_out[i * n_eig + 0];
    }

    for (int p = 1; p < n_extract; p++) {
        double sum = 0;
        for (int i = 0; i < n; i++) sum += V_out[i * n_eig + p];
        if (sum < 0)
            for (int i = 0; i < n; i++) V_out[i * n_eig + p] = -V_out[i * n_eig + p];
    }

    free(deg); free(Q); free(alpha); free(beta); free(w);
    free(T); free(eig_vals);
}

/* ========================================================================== */
/* Rayleigh-Ritz update (kept for v10_main.c compat)                           */
/* ========================================================================== */

void rr_update(double *V, double *lams, int n, int k, int u, int v, double sign) {
    double delta[RR_K];
    for (int p = 0; p < k; p++)
        delta[p] = V[u * k + p] - V[v * k + p];

    double L_sub[RR_K * RR_K];
    memset(L_sub, 0, sizeof(double) * k * k);
    for (int i = 0; i < k; i++) {
        L_sub[i * k + i] = lams[i];
        for (int j = 0; j < k; j++)
            L_sub[i * k + j] += sign * delta[i] * delta[j];
    }

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

    double *V_new = (double *)malloc((size_t)n * k * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int p = 0; p < k; p++) {
            double s = 0;
            for (int q = 0; q < k; q++)
                s += V[i * k + q] * L_sub[p * k + q];
            V_new[i * k + p] = s;
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

    int *adj_list = (int *)malloc((size_t)n * n * sizeof(int));
    int *adj_count = (int *)calloc((size_t)n, sizeof(int));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (adj[i * n + j]) {
                adj_list[i * n + adj_count[i]] = j;
                adj_count[i]++;
            }

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
/* Softmax sampling (single, for v10_main.c compat)                            */
/* ========================================================================== */

int softmax_sample(const float *logits, const uint8_t *mask, int total,
                   RNG *rng, double *log_prob_out) {
    int *valid = (int *)malloc((size_t)total * sizeof(int));
    int nv = 0;
    float max_logit = -FLT_MAX;

    for (int i = 0; i < total; i++) {
        if (mask[i]) {
            valid[nv++] = i;
            if (logits[i] > max_logit) max_logit = logits[i];
        }
    }

    if (nv == 0) { free(valid); *log_prob_out = 0.0; return -1; }

    double *probs = (double *)malloc((size_t)nv * sizeof(double));
    double sum = 0;
    for (int i = 0; i < nv; i++) {
        probs[i] = exp((double)(logits[valid[i]] - max_logit));
        sum += probs[i];
    }

    double r = rng_double(rng);
    double csum = 0;
    int sel = nv - 1;
    for (int i = 0; i < nv; i++) {
        csum += probs[i] / sum;
        if (r < csum) { sel = i; break; }
    }

    *log_prob_out = log(probs[sel] / sum);
    int result = valid[sel];
    free(valid);
    free(probs);
    return result;
}

/* ========================================================================== */
/* Batch softmax sampling without replacement                                  */
/* ========================================================================== */

int softmax_sample_batch(const double *scores, int num_cand, int B,
                         double tau, RNG *rng,
                         int *out_selected, double *out_log_probs) {
    if (num_cand <= 0 || B <= 0) return 0;
    if (B > num_cand) B = num_cand;
    if (tau < 1e-15) tau = 1e-15;

    /* remaining indices into original scores array */
    int *remaining = (int *)malloc((size_t)num_cand * sizeof(int));
    double *rem_logits = (double *)malloc((size_t)num_cand * sizeof(double));
    int num_rem = num_cand;

    for (int i = 0; i < num_cand; i++) remaining[i] = i;

    int selected = 0;
    for (int b = 0; b < B; b++) {
        /* compute logits / tau and find max for stability */
        double max_l = -1e30;
        for (int i = 0; i < num_rem; i++) {
            rem_logits[i] = scores[remaining[i]] / tau;
            if (rem_logits[i] > max_l) max_l = rem_logits[i];
        }

        /* softmax */
        double sum = 0;
        for (int i = 0; i < num_rem; i++) {
            rem_logits[i] = exp(rem_logits[i] - max_l);
            sum += rem_logits[i];
        }
        if (sum < 1e-30) {
            /* uniform fallback */
            sum = (double)num_rem;
            for (int i = 0; i < num_rem; i++) rem_logits[i] = 1.0;
        }

        /* sample */
        double r = rng_double(rng) * sum;
        double csum = 0;
        int sel = num_rem - 1;
        for (int i = 0; i < num_rem; i++) {
            csum += rem_logits[i];
            if (r <= csum) { sel = i; break; }
        }

        out_selected[selected] = remaining[sel];
        out_log_probs[selected] = log(rem_logits[sel] / sum);
        selected++;

        /* remove from pool */
        remaining[sel] = remaining[num_rem - 1];
        num_rem--;
    }

    free(remaining);
    free(rem_logits);
    return selected;
}

/* ========================================================================== */
/* Dual-phase feature building                                                 */
/* ========================================================================== */

void build_candidate_features(const double *v2, const int *degrees,
                              int n, double progress,
                              const int *cand_i, const int *cand_j,
                              int num_cand,
                              const uint8_t *ref_adj, const double *ref_v2,
                              float *feat_out, double *fv_gaps_out) {
    double nm1 = fmax(n - 1, 1);
    float prog = (float)progress;

    for (int c = 0; c < num_cand; c++) {
        int i = cand_i[c], j = cand_j[c];
        double gap = v2[i] - v2[j];
        double fv_gap = (double)n * gap * gap;

        float *f = feat_out + c * EDGE_FEAT_DIM;
        f[0] = (float)fv_gap;
        f[1] = (float)(degrees[i] / nm1);
        f[2] = (float)(degrees[j] / nm1);
        f[3] = prog;

        /* Reference features: edge existence + FV gap in reference graph */
        if (ref_adj && ref_v2) {
            f[4] = (float)ref_adj[i * n + j];
            double ref_gap = ref_v2[i] - ref_v2[j];
            f[5] = (float)((double)n * ref_gap * ref_gap);
        } else {
            f[4] = 0.0f;
            f[5] = 0.0f;
        }

        fv_gaps_out[c] = fv_gap;
    }
}

void build_graph_features_dp(double lam2_est, const int *degrees,
                             int n, double progress, float *gfeat_out) {
    double nm1 = fmax(n - 1, 1);
    double deg_sum = 0;
    for (int i = 0; i < n; i++) deg_sum += degrees[i];
    gfeat_out[0] = (float)((deg_sum / n) / nm1);
    gfeat_out[1] = (float)(lam2_est / fmax(n, 1));
    gfeat_out[2] = (float)progress;
}

/* ========================================================================== */
/* Score candidates: FV base ± MLP correction                                  */
/* ========================================================================== */

void score_candidates(const MLP *mlp, const float *features,
                      const double *fv_gaps, int num_cand,
                      int is_remove, double *scores_out) {
    if (num_cand <= 0) return;

    float *mlp_out = (float *)malloc((size_t)num_cand * sizeof(float));
    mlp_forward_batch(mlp, features, num_cand, mlp_out);

    for (int i = 0; i < num_cand; i++) {
        double base = is_remove ? -fv_gaps[i] : fv_gaps[i];
        scores_out[i] = base + MLP_SCALE * (double)mlp_out[i];
    }

    free(mlp_out);
}

/* ========================================================================== */
/* Dual-phase episode collection (K=C*N design with Metropolis)                */
/* ========================================================================== */

int dual_phase_collect_episode(
    const PolicyWeights *pw,
    int n, int m, int num_cycles, double swap_frac,
    double tau_start, double tau_end, uint64_t seed,
    const uint8_t *ref_adj,
    PhaseTxn *txns, int max_txns,
    int *out_num_txns, double *out_metrics,
    uint8_t *out_best_adj) {

    RNG rng;
    rng_init(&rng, seed);

    uint8_t adj[N_MAX * N_MAX];
    int degrees[N_MAX];
    build_ring_random(adj, degrees, n, m, &rng);

    double init_l2 = exact_lambda2(adj, n);
    double best_l2 = init_l2;
    double cur_l2 = init_l2;
    uint8_t best_adj[N_MAX * N_MAX];
    memcpy(best_adj, adj, (size_t)n * n);

    /* Compute reference Fiedler once if reference provided */
    double ref_V_out[N_MAX];
    double ref_lam[1];
    const double *ref_v2_ptr = NULL;
    if (ref_adj) {
        lanczos_ext_k(ref_adj, n, NULL, LANCZOS_K, 1, ref_V_out, ref_lam);
        ref_v2_ptr = ref_V_out;
    }

    double V_out[N_MAX], lam_out[1], v_warm[N_MAX];
    int has_warm = 0;

    float  feat_buf[MAX_CAND * EDGE_FEAT_DIM];
    double fv_buf[MAX_CAND];
    int    ci_buf[MAX_CAND], cj_buf[MAX_CAND];
    double score_buf[MAX_CAND];
    uint8_t bridge_mask[N_MAX * N_MAX];
    uint8_t saved_adj[N_MAX * N_MAX];
    int    saved_degrees[N_MAX];

    int total_swaps = (int)(m * swap_frac);
    if (total_swaps < 1) total_swaps = 1;
    int effective_K = num_cycles < total_swaps ? num_cycles : total_swaps;
    int swaps_per_epoch = (total_swaps + effective_K - 1) / effective_K;
    if (swaps_per_epoch < 1) swaps_per_epoch = 1;

    /* Metropolis temperature: geometric cooling */
    double T_start = 1.0, T_end = 0.001;

    int num_txns = 0;
    int swap_count = 0;

    for (int epoch = 0; epoch < effective_K; epoch++) {
        if (num_txns + 2 > max_txns) break;
        if (swap_count >= total_swaps) break;

        double progress = (effective_K > 1) ? (double)epoch / (effective_K - 1) : 0.0;
        double tau = tau_start * pow(tau_end / fmax(tau_start, 1e-15), progress);
        double T = T_start * pow(T_end / fmax(T_start, 1e-15), progress);

        int B = swaps_per_epoch;
        int remaining = total_swaps - swap_count;
        if (B > remaining) B = remaining;

        /* Save state for Metropolis revert */
        memcpy(saved_adj, adj, (size_t)n * n);
        memcpy(saved_degrees, degrees, (size_t)n * sizeof(int));
        double saved_l2 = cur_l2;

        /* === ADD PHASE === */
        lanczos_ext_k(adj, n, has_warm ? v_warm : NULL,
                      LANCZOS_K, 1, V_out, lam_out);
        for (int i = 0; i < n; i++) v_warm[i] = V_out[i];
        has_warm = 1;

        int num_add_cand = 0;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (!adj[i * n + j] && num_add_cand < MAX_CAND) {
                    ci_buf[num_add_cand] = i;
                    cj_buf[num_add_cand] = j;
                    num_add_cand++;
                }

        if (num_add_cand == 0) continue;

        build_candidate_features(V_out, degrees, n, progress,
                                 ci_buf, cj_buf, num_add_cand,
                                 ref_adj, ref_v2_ptr,
                                 feat_buf, fv_buf);
        score_candidates(&pw->add_mlp, feat_buf, fv_buf,
                         num_add_cand, 0, score_buf);

        /* Sample B adds */
        uint8_t added_mask[N_MAX * N_MAX];
        memset(added_mask, 0, sizeof(added_mask));
        int added_i[N_MAX], added_j[N_MAX];
        int n_added = 0;
        int pool[MAX_CAND];
        for (int i = 0; i < num_add_cand; i++) pool[i] = i;
        int pool_size = num_add_cand;

        int add_txn_start = num_txns;

        for (int b = 0; b < B && pool_size > 0; b++) {
            if (num_txns >= max_txns) break;

            double max_s = -1e30;
            for (int i = 0; i < pool_size; i++) {
                double s = score_buf[pool[i]] / fmax(tau, 1e-15);
                if (s > max_s) max_s = s;
            }
            double sum_exp = 0;
            double probs_buf[MAX_CAND];
            for (int i = 0; i < pool_size; i++) {
                probs_buf[i] = exp(score_buf[pool[i]] / fmax(tau, 1e-15) - max_s);
                sum_exp += probs_buf[i];
            }

            double r = rng_double(&rng) * sum_exp;
            double csum = 0;
            int sel = pool_size - 1;
            for (int i = 0; i < pool_size; i++) {
                csum += probs_buf[i];
                if (r <= csum) { sel = i; break; }
            }

            int chosen = pool[sel];
            double log_prob = log(probs_buf[sel] / sum_exp);

            int ai = ci_buf[chosen], aj = cj_buf[chosen];
            adj[ai * n + aj] = adj[aj * n + ai] = 1;
            degrees[ai]++;
            degrees[aj]++;
            added_mask[ai * n + aj] = added_mask[aj * n + ai] = 1;
            added_i[n_added] = ai;
            added_j[n_added] = aj;
            n_added++;

            /* Store add PhaseTxn */
            PhaseTxn *txn = &txns[num_txns];
            memset(txn, 0, sizeof(PhaseTxn));
            txn->n = n;
            txn->phase = 0;
            txn->num_cand = num_add_cand;
            memcpy(txn->features, feat_buf, (size_t)num_add_cand * EDGE_FEAT_DIM * sizeof(float));
            memcpy(txn->fv_gaps, fv_buf, (size_t)num_add_cand * sizeof(double));
            memcpy(txn->cand_i, ci_buf, (size_t)num_add_cand * sizeof(int));
            memcpy(txn->cand_j, cj_buf, (size_t)num_add_cand * sizeof(int));
            txn->selected[0] = chosen;
            txn->log_probs[0] = log_prob;
            txn->num_selected = 1;
            build_graph_features_dp(lam_out[0], degrees, n, progress, txn->gfeat);
            txn->value = (double)mlp_forward_single(&pw->val_mlp, txn->gfeat);
            txn->reward = 0; /* filled after Metropolis */
            num_txns++;

            pool[sel] = pool[pool_size - 1];
            pool_size--;
        }

        if (n_added == 0) continue;

        /* === LANCZOS REFRESH === */
        lanczos_ext_k(adj, n, v_warm, LANCZOS_K, 1, V_out, lam_out);
        for (int i = 0; i < n; i++) v_warm[i] = V_out[i];

        /* === REMOVE PHASE === */
        find_bridges(adj, n, bridge_mask);

        int num_rem_cand = 0;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (adj[i * n + j] && !added_mask[i * n + j] &&
                    !bridge_mask[i * n + j] && num_rem_cand < MAX_CAND) {
                    ci_buf[num_rem_cand] = i;
                    cj_buf[num_rem_cand] = j;
                    num_rem_cand++;
                }

        if (num_rem_cand == 0) {
            /* undo adds, rewind txns */
            for (int s = 0; s < n_added; s++) {
                adj[added_i[s] * n + added_j[s]] = 0;
                adj[added_j[s] * n + added_i[s]] = 0;
                degrees[added_i[s]]--;
                degrees[added_j[s]]--;
            }
            num_txns = add_txn_start;
            continue;
        }

        build_candidate_features(V_out, degrees, n, progress,
                                 ci_buf, cj_buf, num_rem_cand,
                                 ref_adj, ref_v2_ptr,
                                 feat_buf, fv_buf);
        score_candidates(&pw->rem_mlp, feat_buf, fv_buf,
                         num_rem_cand, 1, score_buf);

        int n_to_remove = n_added < num_rem_cand ? n_added : num_rem_cand;
        int n_removed = 0;
        pool_size = num_rem_cand;
        for (int i = 0; i < num_rem_cand; i++) pool[i] = i;


        for (int b = 0; b < n_to_remove && pool_size > 0; b++) {
            if (num_txns >= max_txns) break;

            double max_s = -1e30;
            for (int i = 0; i < pool_size; i++) {
                double s = score_buf[pool[i]] / fmax(tau, 1e-15);
                if (s > max_s) max_s = s;
            }
            double sum_exp = 0;
            double probs_buf[MAX_CAND];
            for (int i = 0; i < pool_size; i++) {
                probs_buf[i] = exp(score_buf[pool[i]] / fmax(tau, 1e-15) - max_s);
                sum_exp += probs_buf[i];
            }

            double r = rng_double(&rng) * sum_exp;
            double csum = 0;
            int sel = pool_size - 1;
            for (int i = 0; i < pool_size; i++) {
                csum += probs_buf[i];
                if (r <= csum) { sel = i; break; }
            }

            int chosen = pool[sel];
            double log_prob = log(probs_buf[sel] / sum_exp);

            int ri = ci_buf[chosen], rj = cj_buf[chosen];
            adj[ri * n + rj] = adj[rj * n + ri] = 0;
            degrees[ri]--;
            degrees[rj]--;
            n_removed++;

            /* Store rem PhaseTxn */
            PhaseTxn *txn = &txns[num_txns];
            memset(txn, 0, sizeof(PhaseTxn));
            txn->n = n;
            txn->phase = 1;
            txn->num_cand = num_rem_cand;
            memcpy(txn->features, feat_buf, (size_t)num_rem_cand * EDGE_FEAT_DIM * sizeof(float));
            memcpy(txn->fv_gaps, fv_buf, (size_t)num_rem_cand * sizeof(double));
            memcpy(txn->cand_i, ci_buf, (size_t)num_rem_cand * sizeof(int));
            memcpy(txn->cand_j, cj_buf, (size_t)num_rem_cand * sizeof(int));
            txn->selected[0] = chosen;
            txn->log_probs[0] = log_prob;
            txn->num_selected = 1;
            build_graph_features_dp(lam_out[0], degrees, n, progress, txn->gfeat);
            txn->value = (double)mlp_forward_single(&pw->val_mlp, txn->gfeat);
            txn->reward = 0;
            num_txns++;

            pool[sel] = pool[pool_size - 1];
            pool_size--;
        }

        /* Undo excess adds */
        if (n_removed < n_added) {
            int excess = n_added - n_removed;
            for (int s = n_added - 1; s >= 0 && excess > 0; s--) {
                adj[added_i[s] * n + added_j[s]] = 0;
                adj[added_j[s] * n + added_i[s]] = 0;
                degrees[added_i[s]]--;
                degrees[added_j[s]]--;
                excess--;
            }
        }

        /* === METROPOLIS ACCEPT/REJECT === */
        double new_l2 = exact_lambda2(adj, n);
        double delta = new_l2 - saved_l2;
        int accept;
        if (delta > 0) {
            accept = 1;
        } else if (T > 1e-15) {
            accept = rng_double(&rng) < exp(delta / T);
        } else {
            accept = 0;
        }

        /* Assign reward to all txns in this epoch */
        double reward = accept ? fmax(delta, 0.0) : 0.0;
        for (int t = add_txn_start; t < num_txns; t++)
            txns[t].reward = reward;

        if (accept) {
            cur_l2 = new_l2;
            if (cur_l2 > best_l2) {
                best_l2 = cur_l2;
                memcpy(best_adj, adj, (size_t)n * n);
            }
        } else {
            /* revert graph */
            memcpy(adj, saved_adj, (size_t)n * n);
            memcpy(degrees, saved_degrees, (size_t)n * sizeof(int));
            cur_l2 = saved_l2;
        }

        swap_count += n_added;
    }

    *out_num_txns = num_txns;
    if (out_metrics) {
        out_metrics[0] = cur_l2;
        out_metrics[1] = init_l2;
        out_metrics[2] = best_l2;
        out_metrics[3] = (double)n;
        out_metrics[4] = (double)m;
    }
    if (out_best_adj)
        memcpy(out_best_adj, best_adj, (size_t)n * n);

    return 0;
}

/* ========================================================================== */
/* MLP forward with activation caching                                         */
/* ========================================================================== */

static inline float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float silu_deriv(float x) {
    float s = sigmoidf(x);
    return s * (1.0f + x * (1.0f - s));
}

float mlp_forward_cached(const MLP *mlp, const float *features, MLPCache *cache) {
    int in_dim = mlp->in_dim;
    cache->in_dim = in_dim;
    for (int j = 0; j < in_dim; j++) cache->input[j] = features[j];

    for (int i = 0; i < HIDDEN_DIM; i++) {
        float sum = mlp->b0[i];
        for (int j = 0; j < in_dim; j++)
            sum += mlp->W0[i * in_dim + j] * features[j];
        cache->z0[i] = sum;
        cache->h0[i] = silu(sum);
    }

    for (int i = 0; i < HIDDEN_DIM; i++) {
        float sum = mlp->b1[i];
        for (int j = 0; j < HIDDEN_DIM; j++)
            sum += mlp->W1[i * HIDDEN_DIM + j] * cache->h0[j];
        cache->z1[i] = sum;
        cache->h1[i] = silu(sum);
    }

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

    grad->db2[0] += d_out;
    float dh1[HIDDEN_DIM];
    for (int j = 0; j < HIDDEN_DIM; j++) {
        grad->dW2[j] += d_out * cache->h1[j];
        dh1[j] = d_out * mlp->W2[j];
    }

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
    if (norm > max_norm) policy_grad_scale(pg, max_norm / norm);
}

/* ========================================================================== */
/* PPO evaluate phase (forward only)                                           */
/* ========================================================================== */

void ppo_evaluate_phase(const PolicyWeights *pw, const PhaseTxn *txn,
                        double *out_log_prob, double *out_entropy,
                        float *out_value) {
    const MLP *mlp = (txn->phase == 0) ? &pw->add_mlp : &pw->rem_mlp;
    int nc = txn->num_cand;
    int ns = txn->num_selected;

    /* Score all candidates under current policy */
    double *scores = (double *)malloc((size_t)nc * sizeof(double));
    score_candidates(mlp, txn->features, txn->fv_gaps, nc,
                     txn->phase, scores);

    /* Independent selection approximation:
     * treat each selection as softmax over ALL candidates */
    double total_lp = 0;
    double total_ent = 0;

    /* Compute log-softmax over all candidates once */
    double max_s = -1e30;
    for (int i = 0; i < nc; i++)
        if (scores[i] > max_s) max_s = scores[i];

    double log_sum = 0;
    for (int i = 0; i < nc; i++)
        log_sum += exp(scores[i] - max_s);
    log_sum = log(log_sum) + max_s;

    /* Entropy */
    for (int i = 0; i < nc; i++) {
        double lp_i = scores[i] - log_sum;
        double p_i = exp(lp_i);
        total_ent -= p_i * lp_i;
    }

    /* Log prob for each selected item */
    for (int s = 0; s < ns; s++) {
        int idx = txn->selected[s];
        double lp = scores[idx] - log_sum;
        total_lp += lp;
    }

    *out_log_prob = total_lp;
    *out_entropy = total_ent;
    *out_value = mlp_forward_single(&pw->val_mlp, txn->gfeat);

    free(scores);
}

/* ========================================================================== */
/* PPO backward phase                                                          */
/* ========================================================================== */

void ppo_backward_phase(const PolicyWeights *pw, const PhaseTxn *txn,
                        double advantage, double ret, double old_lp,
                        double ent_coef, double clip_eps, PolicyGrad *pg) {
    const MLP *mlp = (txn->phase == 0) ? &pw->add_mlp : &pw->rem_mlp;
    MLPGrad *mlp_grad = (txn->phase == 0) ? &pg->add_grad : &pg->rem_grad;
    int nc = txn->num_cand;
    int ns = txn->num_selected;
    int is_rem = txn->phase;

    /* Forward pass with caching for all candidates */
    MLPCache *caches = (MLPCache *)malloc((size_t)nc * sizeof(MLPCache));
    double *scores = (double *)malloc((size_t)nc * sizeof(double));

    for (int i = 0; i < nc; i++) {
        const float *feat = txn->features + i * EDGE_FEAT_DIM;
        float mlp_out = mlp_forward_cached(mlp, feat, &caches[i]);
        double base = is_rem ? -txn->fv_gaps[i] : txn->fv_gaps[i];
        scores[i] = base + MLP_SCALE * (double)mlp_out;
    }

    /* log-softmax (independent approx) */
    double max_s = -1e30;
    for (int i = 0; i < nc; i++)
        if (scores[i] > max_s) max_s = scores[i];

    double *sm = (double *)malloc((size_t)nc * sizeof(double)); /* softmax probs */
    double sum_exp = 0;
    for (int i = 0; i < nc; i++) {
        sm[i] = exp(scores[i] - max_s);
        sum_exp += sm[i];
    }
    for (int i = 0; i < nc; i++) sm[i] /= sum_exp;

    double log_sum = log(sum_exp) + max_s;

    /* New total log prob */
    double new_lp = 0;
    for (int s = 0; s < ns; s++) {
        int idx = txn->selected[s];
        new_lp += scores[idx] - log_sum;
    }

    /* PPO ratio and clipping */
    double ratio = exp(new_lp - old_lp);
    double surr1 = ratio * advantage;
    double surr2 = fmin(fmax(ratio, 1.0 - clip_eps), 1.0 + clip_eps) * advantage;
    double use_surr = (surr1 < surr2) ? surr1 : surr2;

    /* d(loss)/d(new_lp) — policy gradient */
    double d_lp;
    if ((surr1 < surr2 && advantage >= 0) || (surr1 > surr2 && advantage < 0))
        d_lp = -ratio * advantage;
    else if (ratio >= 1.0 - clip_eps && ratio <= 1.0 + clip_eps)
        d_lp = -ratio * advantage;
    else
        d_lp = 0.0;
    (void)use_surr;

    /* Entropy bonus: d(entropy)/d(score_i) */
    double ent_weighted_sum = 0;
    for (int i = 0; i < nc; i++) {
        double lp_i = scores[i] - log_sum;
        ent_weighted_sum += sm[i] * (lp_i + 1.0);
    }

    /* Backward through each candidate's MLP */
    for (int i = 0; i < nc; i++) {
        double d_score = 0;

        /* Policy loss gradient: d_lp * d(new_lp)/d(score_i) */
        /* For selected items: d(lp)/d(score_i) = 1 - sm[i] (for each selection) */
        /* For non-selected: d(lp)/d(score_i) = -sm[i] (for each selection) */
        int is_selected = 0;
        for (int s = 0; s < ns; s++)
            if (txn->selected[s] == i) is_selected++;

        d_score += d_lp * ((double)is_selected - (double)ns * sm[i]);

        /* Entropy gradient */
        double d_ent = ent_coef * (sm[i] * ent_weighted_sum - sm[i] * (scores[i] - log_sum + 1.0));
        d_score += d_ent;

        if (fabs(d_score) > 1e-12)
            mlp_backward(mlp, &caches[i], (float)(d_score * MLP_SCALE), mlp_grad);
    }

    /* Value backward */
    MLPCache val_cache;
    float val_out = mlp_forward_cached(&pw->val_mlp, txn->gfeat, &val_cache);
    float d_value = (float)(val_out - ret);
    mlp_backward(&pw->val_mlp, &val_cache, d_value, &pg->val_grad);

    free(caches);
    free(scores);
    free(sm);
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
    int in_dim = mlp->in_dim;
    float grad_buf[MLP_MAX_PARAMS];
    int np = mlp_grad_flatten(grad, in_dim, grad_buf);

    float *w_ptrs[] = { mlp->W0, mlp->b0, mlp->W1, mlp->b1, mlp->W2, mlp->b2 };
    int w_sizes[] = {
        HIDDEN_DIM * in_dim, HIDDEN_DIM,
        HIDDEN_DIM * HIDDEN_DIM, HIDDEN_DIM,
        HIDDEN_DIM, 1
    };

    int is_bias[MLP_MAX_PARAMS];
    int off = 0;
    for (int layer = 0; layer < 6; layer++) {
        int is_b = (layer % 2 == 1);
        for (int i = 0; i < w_sizes[layer]; i++)
            is_bias[off++] = is_b;
    }

    float w_buf[MLP_MAX_PARAMS];
    off = 0;
    for (int layer = 0; layer < 6; layer++) {
        memcpy(w_buf + off, w_ptrs[layer], (size_t)w_sizes[layer] * sizeof(float));
        off += w_sizes[layer];
    }

    double bc1 = 1.0 - pow(beta1, t);
    double bc2 = 1.0 - pow(beta2, t);

    for (int i = 0; i < np; i++) {
        float g = grad_buf[i];
        state->m[i] = (float)(beta1 * state->m[i] + (1.0 - beta1) * g);
        state->v[i] = (float)(beta2 * state->v[i] + (1.0 - beta2) * g * g);
        float m_hat = (float)(state->m[i] / bc1);
        float v_hat = (float)(state->v[i] / bc2);
        float wd = is_bias[i] ? 0.0f : (float)weight_decay;
        w_buf[i] -= (float)(lr * (m_hat / (sqrtf(v_hat) + eps) + wd * w_buf[i]));
    }

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
    if (hdr.edge_feat_dim != EDGE_FEAT_DIM) {
        fprintf(stderr, "Checkpoint edge_feat_dim=%u, expected %d\n",
                hdr.edge_feat_dim, EDGE_FEAT_DIM);
        fclose(f);
        return -4;
    }

    if (episode) *episode = hdr.episode;
    if (best_avg_improvement) *best_avg_improvement = hdr.best_avg_improvement;

    read_mlp_weights(f, &pw->add_mlp, EDGE_FEAT_DIM);
    read_mlp_weights(f, &pw->rem_mlp, EDGE_FEAT_DIM);
    read_mlp_weights(f, &pw->val_mlp, GRAPH_FEAT_DIM);

    fclose(f);
    return 0;
}

/* ========================================================================== */
/* Weight initialization                                                       */
/* ========================================================================== */

static void init_linear_orthogonal(float *W, int rows, int cols, RNG *rng, float gain) {
    int total = rows * cols;
    for (int i = 0; i < total; i += 2) {
        double u1 = rng_double(rng) * 0.998 + 0.001;
        double u2 = rng_double(rng);
        double r = sqrt(-2.0 * log(u1));
        double t = 2.0 * M_PI * u2;
        W[i] = (float)(r * cos(t));
        if (i + 1 < total) W[i + 1] = (float)(r * sin(t));
    }

    int min_dim = rows < cols ? rows : cols;
    for (int i = 0; i < min_dim; i++) {
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

    /* Zero out final layer of add/rem MLPs so initial MLP output = 0.
     * This makes initial scoring = pure FV (no MLP noise). */
    memset(pw->add_mlp.W2, 0, sizeof(pw->add_mlp.W2));
    memset(pw->add_mlp.b2, 0, sizeof(pw->add_mlp.b2));
    memset(pw->rem_mlp.W2, 0, sizeof(pw->rem_mlp.W2));
    memset(pw->rem_mlp.b2, 0, sizeof(pw->rem_mlp.b2));
}
