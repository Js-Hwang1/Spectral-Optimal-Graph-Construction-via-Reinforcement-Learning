/*
 * sweep_gpu.cu — GPU-accelerated 3-step exhaustive sweep.
 *
 * Flow:
 *   1. CPU enumerates all leaf (a0, a1, a2) sequences
 *   2. CPU builds N×N Laplacian for each leaf
 *   3. GPU batch-eigensolves via cusolverDnSsyevjBatched
 *   4. CPU reduces: for each step-0 action, find max λ₂
 *
 * No bridge detection in sweep — disconnecting removals get λ₂=0
 * and are naturally filtered out.
 */

#include "sweep_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#define MAX_N 24
#define MAX_LEAVES (200 * 200 * 200)  /* generous upper bound */

static cusolverDnHandle_t cusolver_handle;
static syevjInfo_t syevj_params;
static int gpu_initialized = 0;

/* Persistent GPU buffers — resized as needed */
static float *d_A = NULL, *d_W = NULL, *d_work = NULL;
static int *d_info = NULL;
static int buf_capacity = 0;  /* current allocated batch capacity */
static int buf_N = 0;         /* current allocated matrix size */
static int buf_lwork = 0;

void gpu_init(void) {
    if (gpu_initialized) return;
    cusolverDnCreate(&cusolver_handle);
    cusolverDnCreateSyevjInfo(&syevj_params);
    cusolverDnXsyevjSetTolerance(syevj_params, 1e-5);
    cusolverDnXsyevjSetMaxSweeps(syevj_params, 50);
    gpu_initialized = 1;
}

void gpu_cleanup(void) {
    if (!gpu_initialized) return;
    if (d_A) cudaFree(d_A);
    if (d_W) cudaFree(d_W);
    if (d_work) cudaFree(d_work);
    if (d_info) cudaFree(d_info);
    cusolverDnDestroySyevjInfo(syevj_params);
    cusolverDnDestroy(cusolver_handle);
    gpu_initialized = 0;
}

static void ensure_buffers(int n, int batch) {
    if (batch <= buf_capacity && n == buf_N) return;

    if (d_A) cudaFree(d_A);
    if (d_W) cudaFree(d_W);
    if (d_info) cudaFree(d_info);
    if (d_work) cudaFree(d_work);

    cudaMalloc(&d_A, (size_t)batch * n * n * sizeof(float));
    cudaMalloc(&d_W, (size_t)batch * n * sizeof(float));
    cudaMalloc(&d_info, (size_t)batch * sizeof(int));

    /* Query workspace */
    int lwork;
    cusolverDnSsyevjBatched_bufferSize(cusolver_handle,
        CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER,
        n, d_A, n, d_W, &lwork, syevj_params, batch);
    cudaMalloc(&d_work, (size_t)lwork * sizeof(float));

    buf_capacity = batch;
    buf_N = n;
    buf_lwork = lwork;
}

/* Batch eigensolve: h_L is batch × n × n column-major Laplacians.
 * Returns λ₂ for each in h_lambda2. */
static void batch_lambda2(const float *h_L, float *h_lambda2, int n, int batch) {
    if (batch == 0) return;
    ensure_buffers(n, batch);

    cudaMemcpy(d_A, h_L, (size_t)batch * n * n * sizeof(float), cudaMemcpyHostToDevice);

    cusolverDnSsyevjBatched(cusolver_handle,
        CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER,
        n, d_A, n, d_W, d_work, buf_lwork, d_info, syevj_params, batch);
    cudaDeviceSynchronize();

    /* Copy all eigenvalues back, extract λ₂ (index 1) for each */
    float *h_W = (float *)malloc((size_t)batch * n * sizeof(float));
    cudaMemcpy(h_W, d_W, (size_t)batch * n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int b = 0; b < batch; b++)
        h_lambda2[b] = h_W[b * n + 1];  /* eigenvalues sorted ascending, λ₂ is index 1 */
    free(h_W);
}

/* Graph stride for adjacency matrix */
static int g_n = 0;

/* Build column-major Laplacian from adjacency */
static void build_laplacian(const uint8_t *adj, int n, float *L) {
    for (int j = 0; j < n; j++) {
        float rowsum = 0;
        for (int i = 0; i < n; i++) {
            if (i == j) continue;
            float v = adj[i * g_n + j] ? -1.0f : 0.0f;
            L[i + j * n] = v;  /* column-major */
            rowsum -= v;
        }
        L[j + j * n] = rowsum;
    }
}

/* ========================================================================== */
/* Enumerate + sweep                                                           */
/* ========================================================================== */

typedef struct { int src; int tgt; } Action;

/* Enumerate all valid add actions from current adj */
static int enum_adds(const uint8_t *adj, int n, Action *out) {
    int cnt = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (!adj[i * g_n + j]) { out[cnt].src = i; out[cnt].tgt = j; cnt++; }
    return cnt;
}

/* Enumerate all edges (for removal — no bridge check in sweep) */
static int enum_rems(const uint8_t *adj, int n, Action *out) {
    int cnt = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (adj[i * g_n + j]) { out[cnt].src = i; out[cnt].tgt = j; cnt++; }
    return cnt;
}

static void apply_add(uint8_t *adj, int u, int v) {
    adj[u * g_n + v] = adj[v * g_n + u] = 1;
}
static void undo_add(uint8_t *adj, int u, int v) {
    adj[u * g_n + v] = adj[v * g_n + u] = 0;
}
static void apply_rem(uint8_t *adj, int u, int v) {
    adj[u * g_n + v] = adj[v * g_n + u] = 0;
}
static void undo_rem(uint8_t *adj, int u, int v) {
    adj[u * g_n + v] = adj[v * g_n + u] = 1;
}

int gpu_sweep_add(const uint8_t *adj_in, const int *deg, int n, int total_steps,
                   int *out_src, int *out_tgt, double *out_val) {
    (void)deg;
    if (!gpu_initialized) gpu_init();
    g_n = n;

    uint8_t *adj = (uint8_t *)calloc((size_t)n * n, 1);
    memcpy(adj, adj_in, (size_t)n * n);

    Action a0[512], a1[512], a2[512];
    int na0 = enum_adds(adj, n, a0);
    if (na0 == 0) { free(adj); return 0; }

    if (total_steps == 1) {
        /* Just evaluate each step-0 action directly */
        float *h_L = (float *)malloc((size_t)na0 * n * n * sizeof(float));
        float *h_l2 = (float *)malloc((size_t)na0 * sizeof(float));
        for (int i = 0; i < na0; i++) {
            apply_add(adj, a0[i].src, a0[i].tgt);
            build_laplacian(adj, n, h_L + (size_t)i * n * n);
            undo_add(adj, a0[i].src, a0[i].tgt);
        }
        batch_lambda2(h_L, h_l2, n, na0);
        for (int i = 0; i < na0; i++) {
            out_src[i] = a0[i].src; out_tgt[i] = a0[i].tgt;
            out_val[i] = (double)h_l2[i];
        }
        free(h_L); free(h_l2); free(adj);
        return na0;
    }

    /* Enumerate all leaf sequences and batch eigensolve */
    /* First pass: count total leaves */
    int total_leaves = 0;
    int *a0_leaf_start = (int *)malloc((size_t)na0 * sizeof(int));

    for (int i0 = 0; i0 < na0; i0++) {
        a0_leaf_start[i0] = total_leaves;
        apply_add(adj, a0[i0].src, a0[i0].tgt);
        int na1 = enum_adds(adj, n, a1);

        if (total_steps == 2) {
            total_leaves += na1;
        } else { /* total_steps == 3 */
            for (int i1 = 0; i1 < na1; i1++) {
                apply_add(adj, a1[i1].src, a1[i1].tgt);
                int na2 = enum_adds(adj, n, a2);
                total_leaves += na2;
                undo_add(adj, a1[i1].src, a1[i1].tgt);
            }
        }
        undo_add(adj, a0[i0].src, a0[i0].tgt);
    }

    if (total_leaves == 0) { free(a0_leaf_start); free(adj); return 0; }

    /* Second pass: build Laplacians */
    float *h_L = (float *)malloc((size_t)total_leaves * n * n * sizeof(float));
    int leaf = 0;

    for (int i0 = 0; i0 < na0; i0++) {
        apply_add(adj, a0[i0].src, a0[i0].tgt);
        int na1 = enum_adds(adj, n, a1);

        if (total_steps == 2) {
            for (int i1 = 0; i1 < na1; i1++) {
                apply_add(adj, a1[i1].src, a1[i1].tgt);
                build_laplacian(adj, n, h_L + (size_t)leaf * n * n);
                leaf++;
                undo_add(adj, a1[i1].src, a1[i1].tgt);
            }
        } else {
            for (int i1 = 0; i1 < na1; i1++) {
                apply_add(adj, a1[i1].src, a1[i1].tgt);
                int na2 = enum_adds(adj, n, a2);
                for (int i2 = 0; i2 < na2; i2++) {
                    apply_add(adj, a2[i2].src, a2[i2].tgt);
                    build_laplacian(adj, n, h_L + (size_t)leaf * n * n);
                    leaf++;
                    undo_add(adj, a2[i2].src, a2[i2].tgt);
                }
                undo_add(adj, a1[i1].src, a1[i1].tgt);
            }
        }
        undo_add(adj, a0[i0].src, a0[i0].tgt);
    }

    /* GPU batch eigensolve */
    float *h_l2 = (float *)malloc((size_t)total_leaves * sizeof(float));
    batch_lambda2(h_L, h_l2, n, total_leaves);

    /* Reduce: for each step-0 action, find max λ₂ */
    for (int i0 = 0; i0 < na0; i0++) {
        int start = a0_leaf_start[i0];
        int end = (i0 + 1 < na0) ? a0_leaf_start[i0 + 1] : total_leaves;
        double best = -1e30;
        for (int l = start; l < end; l++)
            if ((double)h_l2[l] > best) best = (double)h_l2[l];
        out_src[i0] = a0[i0].src;
        out_tgt[i0] = a0[i0].tgt;
        out_val[i0] = best;
    }

    free(h_L); free(h_l2); free(a0_leaf_start); free(adj);
    return na0;
}

int gpu_sweep_rem(const uint8_t *adj_in, const int *deg, int n, int total_steps,
                   int *out_src, int *out_tgt, double *out_val) {
    (void)deg;
    if (!gpu_initialized) gpu_init();
    g_n = n;

    uint8_t *adj = (uint8_t *)calloc((size_t)n * n, 1);
    memcpy(adj, adj_in, (size_t)n * n);

    Action a0[512], a1[512], a2[512];
    int na0 = enum_rems(adj, n, a0);
    if (na0 == 0) { free(adj); return 0; }

    if (total_steps == 1) {
        float *h_L = (float *)malloc((size_t)na0 * n * n * sizeof(float));
        float *h_l2 = (float *)malloc((size_t)na0 * sizeof(float));
        for (int i = 0; i < na0; i++) {
            apply_rem(adj, a0[i].src, a0[i].tgt);
            build_laplacian(adj, n, h_L + (size_t)i * n * n);
            undo_rem(adj, a0[i].src, a0[i].tgt);
        }
        batch_lambda2(h_L, h_l2, n, na0);
        for (int i = 0; i < na0; i++) {
            out_src[i] = a0[i].src; out_tgt[i] = a0[i].tgt;
            out_val[i] = (double)h_l2[i];
        }
        free(h_L); free(h_l2); free(adj);
        return na0;
    }

    int total_leaves = 0;
    int *a0_leaf_start = (int *)malloc((size_t)na0 * sizeof(int));

    for (int i0 = 0; i0 < na0; i0++) {
        a0_leaf_start[i0] = total_leaves;
        apply_rem(adj, a0[i0].src, a0[i0].tgt);
        int na1 = enum_rems(adj, n, a1);
        if (total_steps == 2) {
            total_leaves += na1;
        } else {
            for (int i1 = 0; i1 < na1; i1++) {
                apply_rem(adj, a1[i1].src, a1[i1].tgt);
                int na2 = enum_rems(adj, n, a2);
                total_leaves += na2;
                undo_rem(adj, a1[i1].src, a1[i1].tgt);
            }
        }
        undo_rem(adj, a0[i0].src, a0[i0].tgt);
    }

    if (total_leaves == 0) { free(a0_leaf_start); free(adj); return 0; }

    float *h_L = (float *)malloc((size_t)total_leaves * n * n * sizeof(float));
    int leaf = 0;

    for (int i0 = 0; i0 < na0; i0++) {
        apply_rem(adj, a0[i0].src, a0[i0].tgt);
        int na1 = enum_rems(adj, n, a1);
        if (total_steps == 2) {
            for (int i1 = 0; i1 < na1; i1++) {
                apply_rem(adj, a1[i1].src, a1[i1].tgt);
                build_laplacian(adj, n, h_L + (size_t)leaf * n * n);
                leaf++;
                undo_rem(adj, a1[i1].src, a1[i1].tgt);
            }
        } else {
            for (int i1 = 0; i1 < na1; i1++) {
                apply_rem(adj, a1[i1].src, a1[i1].tgt);
                int na2 = enum_rems(adj, n, a2);
                for (int i2 = 0; i2 < na2; i2++) {
                    apply_rem(adj, a2[i2].src, a2[i2].tgt);
                    build_laplacian(adj, n, h_L + (size_t)leaf * n * n);
                    leaf++;
                    undo_rem(adj, a2[i2].src, a2[i2].tgt);
                }
                undo_rem(adj, a1[i1].src, a1[i1].tgt);
            }
        }
        undo_rem(adj, a0[i0].src, a0[i0].tgt);
    }

    float *h_l2 = (float *)malloc((size_t)total_leaves * sizeof(float));
    batch_lambda2(h_L, h_l2, n, total_leaves);

    for (int i0 = 0; i0 < na0; i0++) {
        int start = a0_leaf_start[i0];
        int end = (i0 + 1 < na0) ? a0_leaf_start[i0 + 1] : total_leaves;
        double best = -1e30;
        for (int l = start; l < end; l++)
            if ((double)h_l2[l] > best) best = (double)h_l2[l];
        out_src[i0] = a0[i0].src;
        out_tgt[i0] = a0[i0].tgt;
        out_val[i0] = best;
    }

    free(h_L); free(h_l2); free(a0_leaf_start); free(adj);
    return na0;
}
