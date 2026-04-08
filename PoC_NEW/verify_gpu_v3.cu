/*
 * verify_gpu_v3.cu — Maximum-scale verification.
 *
 * Optimizations:
 *   1. Compact storage: after regularization, store exactly d neighbors
 *      per node (not max_deg). GPU only sees the compact array.
 *   2. Init uses full max_deg on CPU, then compacts before GPU transfer.
 *   3. Lanczos vectors use float instead of double (halves GPU memory).
 *      λ₂ accuracy to ~4 decimal places is sufficient for our purpose.
 *   4. Parallel regularization on CPU.
 *
 * Memory per node:
 *   CPU init phase: (d+12) * 4 bytes (generous for Poisson tail)
 *   GPU Lanczos:    d * 4 bytes (adj) + 4 bytes (deg) + 3 * 4 bytes (vectors)
 *                 = (d + 4) * 4 bytes per node
 *
 * Limits (GPU VRAM):
 *   141 GB H200:  d=4 → 141GB / (8*4) = 4.4B nodes → n=2^32
 *                 d=8 → 141GB / (12*4) = 2.9B nodes → n=2^31
 *   102 GB RTX:   d=4 → 102GB / (8*4) = 3.2B nodes → n=2^31
 *                 d=8 → 102GB / (12*4) = 2.1B nodes → n=2^31
 *
 * Build:
 *   nvcc -O3 -arch=sm_90 -o verify_gpu_v3 verify_gpu_v3.cu -lm   # H200
 *   nvcc -O3 -arch=sm_120 -o verify_gpu_v3 verify_gpu_v3.cu -lm  # Blackwell
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

// ============================================================
// CPU RNG
// ============================================================
static uint64_t rng_s;
static void rng_seed(uint64_t s) { rng_s = s; }
static uint32_t rng_next(void) {
    rng_s = rng_s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(rng_s >> 32);
}
static long long rng_ll(long long n) {
    uint64_t r = ((uint64_t)rng_next() << 32) | rng_next();
    return (long long)(r % (uint64_t)n);
}

// ============================================================
// Phase 1 (CPU): Random init — exactly m edges
// Uses generous max_deg for Poisson tail during init
// ============================================================
void phase1_init(int *adj, int *deg, long long n, int d, int max_deg, uint64_t seed) {
    long long m = n * d / 2;
    rng_seed(seed);
    memset(deg, 0, (size_t)n * sizeof(int));
    memset(adj, -1, (size_t)n * max_deg * sizeof(int));

    long long count = 0;
    while (count < m) {
        int u = (int)rng_ll(n), v = (int)rng_ll(n);
        if (u == v) continue;
        if (deg[u] >= max_deg - 1 || deg[v] >= max_deg - 1) continue;
        int exists = 0;
        for (int k = 0; k < deg[u]; k++)
            if (adj[(long long)u * max_deg + k] == v) { exists = 1; break; }
        if (exists) continue;
        adj[(long long)u * max_deg + deg[u]++] = v;
        adj[(long long)v * max_deg + deg[v]++] = u;
        count++;
    }
}

// ============================================================
// Phase 2 (CPU): Parallel degree regularization
// ============================================================
void phase2_regularize(int *adj, int *deg, long long n, int d, int max_deg) {
    int *over = (int*)malloc((size_t)n * sizeof(int));
    int *under = (int*)malloc((size_t)n * sizeof(int));
    int *claimed = (int*)calloc((size_t)n, sizeof(int));

    for (long long round = 0; round < n * d; round++) {
        int n_over = 0, n_under = 0;
        for (long long i = 0; i < n; i++) {
            if (deg[i] > d) over[n_over++] = i;
            else if (deg[i] < d) under[n_under++] = i;
        }
        if (n_over == 0 && n_under == 0) break;
        if (n_over == 0 || n_under == 0) break;

        int num_pairs = n_over < n_under ? n_over : n_under;
        memset(claimed, 0, (size_t)n * sizeof(int));
        int swaps = 0;

        for (int idx = 0; idx < num_pairs; idx++) {
            int u = over[idx], w = under[idx];
            if (claimed[u] || claimed[w]) continue;

            int bv = -1, bd = -1;
            for (int k = 0; k < deg[u]; k++) {
                int v = adj[(long long)u * max_deg + k];
                if (v < 0 || v == w || claimed[v]) continue;
                int skip = 0;
                for (int kk = 0; kk < deg[w]; kk++)
                    if (adj[(long long)w * max_deg + kk] == v) { skip = 1; break; }
                if (skip) continue;
                if (deg[v] > bd) { bd = deg[v]; bv = v; }
            }

            if (bv >= 0) {
                claimed[u] = claimed[w] = claimed[bv] = 1;
                // Remove {u, bv}
                for (int k = 0; k < deg[u]; k++)
                    if (adj[(long long)u * max_deg + k] == bv) {
                        adj[(long long)u * max_deg + k] = adj[(long long)u * max_deg + --deg[u]];
                        adj[(long long)u * max_deg + deg[u]] = -1; break; }
                for (int k = 0; k < deg[bv]; k++)
                    if (adj[(long long)bv * max_deg + k] == u) {
                        adj[(long long)bv * max_deg + k] = adj[(long long)bv * max_deg + --deg[bv]];
                        adj[(long long)bv * max_deg + deg[bv]] = -1; break; }
                // Add {w, bv}
                adj[(long long)w * max_deg + deg[w]++] = bv;
                adj[(long long)bv * max_deg + deg[bv]++] = w;
                swaps++;
            }
        }

        if (swaps == 0) {
            // Sequential fallback for stragglers
            for (int oi = 0; oi < n_over && swaps == 0; oi++) {
                int u = over[oi]; if (deg[u] <= d) continue;
                for (int ui = 0; ui < n_under && swaps == 0; ui++) {
                    int w = under[ui]; if (deg[w] >= d) continue;
                    for (int k = 0; k < deg[u]; k++) {
                        int v = adj[(long long)u * max_deg + k];
                        if (v < 0 || v == w) continue;
                        int skip = 0;
                        for (int kk = 0; kk < deg[w]; kk++)
                            if (adj[(long long)w * max_deg + kk] == v) { skip = 1; break; }
                        if (skip) continue;
                        for (int k2 = 0; k2 < deg[u]; k2++)
                            if (adj[(long long)u * max_deg + k2] == v) {
                                adj[(long long)u * max_deg + k2] = adj[(long long)u * max_deg + --deg[u]];
                                adj[(long long)u * max_deg + deg[u]] = -1; break; }
                        for (int k2 = 0; k2 < deg[v]; k2++)
                            if (adj[(long long)v * max_deg + k2] == u) {
                                adj[(long long)v * max_deg + k2] = adj[(long long)v * max_deg + --deg[v]];
                                adj[(long long)v * max_deg + deg[v]] = -1; break; }
                        adj[(long long)w * max_deg + deg[w]++] = v;
                        adj[(long long)v * max_deg + deg[v]++] = w;
                        swaps++; break;
                    }
                    if (swaps == 0) {
                        int uw = 0;
                        for (int k = 0; k < deg[u]; k++)
                            if (adj[(long long)u * max_deg + k] == w) { uw = 1; break; }
                        if (!uw) {
                            adj[(long long)u * max_deg + deg[u]++] = w;
                            adj[(long long)w * max_deg + deg[w]++] = u;
                            int bv2 = -1, bd2 = -1;
                            for (int k = 0; k < deg[u]; k++) {
                                int v = adj[(long long)u * max_deg + k];
                                if (v < 0 || v == w) continue;
                                if (deg[v] > bd2) { bd2 = deg[v]; bv2 = v; }
                            }
                            if (bv2 >= 0) {
                                for (int k = 0; k < deg[u]; k++)
                                    if (adj[(long long)u * max_deg + k] == bv2) {
                                        adj[(long long)u * max_deg + k] = adj[(long long)u * max_deg + --deg[u]];
                                        adj[(long long)u * max_deg + deg[u]] = -1; break; }
                                for (int k = 0; k < deg[bv2]; k++)
                                    if (adj[(long long)bv2 * max_deg + k] == u) {
                                        adj[(long long)bv2 * max_deg + k] = adj[(long long)bv2 * max_deg + --deg[bv2]];
                                        adj[(long long)bv2 * max_deg + deg[bv2]] = -1; break; }
                            }
                            swaps++;
                        }
                    }
                }
            }
        }
    }
    free(over); free(under); free(claimed);
}

// ============================================================
// Compact: copy exactly d neighbors per node into dense array
// ============================================================
void compact_adj(const int *adj_full, const int *deg, long long n, int d,
                 int max_deg, int *adj_compact) {
    for (long long i = 0; i < n; i++) {
        for (int k = 0; k < d; k++) {
            adj_compact[i * (long long)d + k] = adj_full[i * (long long)max_deg + k];
        }
    }
}

// ============================================================
// Phase 3 (GPU): Lanczos with FLOAT vectors, compact adj
// ============================================================

__global__ void kernel_laplacian_mv_compact(
    const int *adj, const float *x, float *y, long long n, int d)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = d * x[i];
    for (int k = 0; k < d; k++) {
        int j = adj[i * (long long)d + k];
        s -= x[j];
    }
    y[i] = s;
}

__global__ void kernel_axpy_f(float *y, const float *x, float a, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

__global__ void kernel_scale_f(float *x, float a, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= a;
}

__global__ void kernel_dot_f(const float *a, const float *b, double *out, long long n) {
    extern __shared__ char smem[];
    double *sd = (double*)smem;
    int tid = threadIdx.x;
    long long i = (long long)blockIdx.x * blockDim.x + tid;
    sd[tid] = (i < n) ? (double)a[i] * (double)b[i] : 0.0;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sd[tid] += sd[tid+s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sd[0]);
}

double gpu_dot_f(const float *a, const float *b, long long n, double *tmp) {
    CUDA_CHECK(cudaMemset(tmp, 0, sizeof(double)));
    int bl = 256;
    long long gr = (n + bl - 1) / bl;
    kernel_dot_f<<<(int)gr, bl, bl*sizeof(double)>>>(a, b, tmp, n);
    double r;
    CUDA_CHECK(cudaMemcpy(&r, tmp, sizeof(double), cudaMemcpyDeviceToHost));
    return r;
}

double gpu_lanczos_f(int *d_adj, long long n, int d, int iters) {
    int bl = 256;
    long long gr = (n + bl - 1) / bl;

    float *d_v0, *d_v1, *d_w, *d_ones;
    double *d_tmp;
    CUDA_CHECK(cudaMalloc(&d_v0, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v1, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ones, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp, sizeof(double)));

    // Fill ones and random v1 on host, upload once
    float *h = (float*)malloc((size_t)n * sizeof(float));
    for (long long i = 0; i < n; i++) h[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(d_ones, h, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));

    srand(999);
    double sum = 0;
    for (long long i = 0; i < n; i++) {
        h[i] = (float)(rand() % 10000) / 5000.0f - 1.0f;
        sum += h[i];
    }
    sum /= n;
    double nm = 0;
    for (long long i = 0; i < n; i++) { h[i] -= (float)sum; nm += (double)h[i] * h[i]; }
    nm = sqrt(nm);
    for (long long i = 0; i < n; i++) h[i] /= (float)nm;
    CUDA_CHECK(cudaMemcpy(d_v1, h, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_v0, 0, (size_t)n * sizeof(float)));
    free(h);

    double *al = (double*)malloc(iters * sizeof(double));
    double *be = (double*)malloc(iters * sizeof(double));
    double beta = 0;
    int ak = 0;

    for (int j = 0; j < iters; j++) {
        kernel_laplacian_mv_compact<<<(int)gr, bl>>>(d_adj, d_v1, d_w, n, d);

        double alpha = gpu_dot_f(d_v1, d_w, n, d_tmp);
        al[j] = alpha;

        kernel_axpy_f<<<(int)gr, bl>>>(d_w, d_v1, -(float)alpha, n);
        if (beta > 0) kernel_axpy_f<<<(int)gr, bl>>>(d_w, d_v0, -(float)beta, n);

        // Orthogonalize vs 1
        double ws = gpu_dot_f(d_w, d_ones, n, d_tmp);
        kernel_axpy_f<<<(int)gr, bl>>>(d_w, d_ones, -(float)(ws / n), n);

        double wn = gpu_dot_f(d_w, d_w, n, d_tmp);
        beta = sqrt(wn);
        be[j] = beta;
        ak = j + 1;
        if (beta < 1e-6) break;  // float precision limit

        CUDA_CHECK(cudaMemcpy(d_v0, d_v1, (size_t)n * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_v1, d_w, (size_t)n * sizeof(float), cudaMemcpyDeviceToDevice));
        kernel_scale_f<<<(int)gr, bl>>>(d_v1, 1.0f / (float)beta, n);
    }

    // Bisection on tridiagonal
    double hi = 0;
    for (int j = 0; j < ak; j++) {
        double b2 = fabs(al[j]) + (j > 0 ? fabs(be[j-1]) : 0) + (j < ak-1 ? fabs(be[j]) : 0);
        if (b2 > hi) hi = b2;
    }
    hi += 1; double lo = 0;
    for (int b2 = 0; b2 < 100; b2++) {
        double mu = (lo + hi) / 2;
        int cnt = 0; double dv = al[0] - mu;
        if (dv < 0) cnt++;
        for (int j = 1; j < ak; j++) {
            if (fabs(dv) < 1e-30) dv = 1e-30;
            dv = (al[j] - mu) - be[j-1] * be[j-1] / dv;
            if (dv < 0) cnt++;
        }
        if (cnt >= 2) hi = mu; else lo = mu;
    }

    free(al); free(be);
    cudaFree(d_v0); cudaFree(d_v1); cudaFree(d_w); cudaFree(d_ones); cudaFree(d_tmp);
    return (lo + hi) / 2;
}

// ============================================================
// Main
// ============================================================
int main(int argc, char **argv) {
    int d = 4;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    long long vram_mb = prop.totalGlobalMem / (1024 * 1024);
    fprintf(stderr, "GPU: %s (%.1f GB VRAM)\n", prop.name, prop.totalGlobalMem / 1e9);

    if (argc >= 3 && strcmp(argv[1], "--sweep") != 0) {
        int n = atoi(argv[1]);
        d = atoi(argv[2]);
        uint64_t seed = argc >= 4 ? atoll(argv[3]) : 42;
        double ram_bound = d - 2 * sqrt(d - 1);
        int max_deg = d + 12;

        // Memory estimates
        long long cpu_mb = (long long)n * max_deg * 4 / (1024*1024);
        long long gpu_mb = ((long long)n * d * 4 + (long long)n * 3 * 4) / (1024*1024);

        fprintf(stderr, "n=%d d=%d CPU=~%lldMB GPU=~%lldMB\n", n, d, cpu_mb, gpu_mb);

        if (gpu_mb > vram_mb - 500) {
            printf("RESULT: n=%d d=%d lambda2=0 ratio=0 regular=SKIP total=0s\n", n, d);
            fprintf(stderr, "SKIP: GPU OOM (%lld > %lld MB)\n", gpu_mb, vram_mb);
            return 0;
        }

        struct timespec t0, t1, t2, t3;

        // Phase 1
        int *h_adj = (int*)malloc((size_t)n * max_deg * sizeof(int));
        int *h_deg = (int*)malloc((size_t)n * sizeof(int));
        if (!h_adj || !h_deg) {
            printf("RESULT: n=%d d=%d lambda2=0 ratio=0 regular=OOM total=0s\n", n, d);
            return 0;
        }

        clock_gettime(CLOCK_MONOTONIC, &t0);
        phase1_init(h_adj, h_deg, n, d, max_deg, seed);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        // Phase 2
        phase2_regularize(h_adj, h_deg, n, d, max_deg);
        clock_gettime(CLOCK_MONOTONIC, &t2);

        int reg = 1;
        for (long long i = 0; i < n; i++) if (h_deg[i] != d) { reg = 0; break; }

        // Compact
        int *h_compact = (int*)malloc((size_t)n * d * sizeof(int));
        if (!h_compact) {
            printf("RESULT: n=%d d=%d lambda2=0 ratio=0 regular=OOM total=0s\n", n, d);
            free(h_adj); free(h_deg); return 0;
        }
        compact_adj(h_adj, h_deg, n, d, max_deg, h_compact);
        free(h_adj); // release big array immediately

        // GPU transfer (compact only)
        int *d_adj;
        CUDA_CHECK(cudaMalloc(&d_adj, (size_t)n * d * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_adj, h_compact, (size_t)n * d * sizeof(int), cudaMemcpyHostToDevice));
        free(h_compact);

        // Phase 3: Lanczos (float)
        int k = n < 100000 ? 300 : (n < 10000000 ? 500 : 800);
        double l2 = gpu_lanczos_f(d_adj, n, d, k);
        clock_gettime(CLOCK_MONOTONIC, &t3);

        double is_ = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
        double rs = (t2.tv_sec-t1.tv_sec)+(t2.tv_nsec-t1.tv_nsec)/1e9;
        double es = (t3.tv_sec-t2.tv_sec)+(t3.tv_nsec-t2.tv_nsec)/1e9;
        double tot = (t3.tv_sec-t0.tv_sec)+(t3.tv_nsec-t0.tv_nsec)/1e9;

        printf("RESULT: n=%d d=%d lambda2=%.6f ratio=%.4f regular=%s total=%.2fs\n",
               n, d, l2, l2/ram_bound, reg?"YES":"NO", tot);
        fprintf(stderr, "  init=%.1fs reg=%.1fs eig=%.1fs total=%.1fs\n", is_, rs, es, tot);

        free(h_deg); cudaFree(d_adj);

    } else {
        // Sweep
        double ram_bound = d - 2 * sqrt(d - 1);
        printf("%12s %3s | %8s %8s %8s | %8s %6s | %8s\n",
               "n", "d", "init_s", "reg_s", "eig_s", "lambda2", "ratio", "total");
        printf("---------------------------------------------------------------------\n");

        for (int logn = 5; logn <= 32; logn++) {
            long long n = 1LL << logn;
            int max_deg = d + 12;
            long long gpu_need = n * d * 4 + n * 3 * 4;
            if (gpu_need > (long long)(prop.totalGlobalMem - 500LL*1024*1024)) {
                printf("%12lld %3d | GPU OOM (need %lld MB)\n", n, d, gpu_need/(1024*1024));
                break;
            }
            long long cpu_need = n * max_deg * 4 + n * 4;
            int *h_adj = (int*)malloc((size_t)(n * max_deg * sizeof(int)));
            int *h_deg = (int*)malloc((size_t)(n * sizeof(int)));
            if (!h_adj || !h_deg) {
                printf("%12lld %3d | CPU OOM\n", n, d);
                break;
            }

            struct timespec t0,t1,t2,t3;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            phase1_init(h_adj, h_deg, n, d, max_deg, 42);
            clock_gettime(CLOCK_MONOTONIC, &t1);

            phase2_regularize(h_adj, h_deg, n, d, max_deg);
            clock_gettime(CLOCK_MONOTONIC, &t2);

            int reg = 1;
            for (long long i = 0; i < n; i++) if (h_deg[i] != d) { reg = 0; break; }

            int *h_compact = (int*)malloc((size_t)(n * d * sizeof(int)));
            if (!h_compact) { printf("OOM compact\n"); free(h_adj); free(h_deg); break; }
            compact_adj(h_adj, h_deg, n, d, max_deg, h_compact);
            free(h_adj);

            int *d_adj;
            CUDA_CHECK(cudaMalloc(&d_adj, (size_t)(n * d * sizeof(int))));
            CUDA_CHECK(cudaMemcpy(d_adj, h_compact, (size_t)(n * d * sizeof(int)), cudaMemcpyHostToDevice));
            free(h_compact);

            int k = logn <= 14 ? 300 : (logn <= 20 ? 500 : 800);
            double l2 = gpu_lanczos_f(d_adj, n, d, k);
            clock_gettime(CLOCK_MONOTONIC, &t3);

            double is_=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
            double rs=(t2.tv_sec-t1.tv_sec)+(t2.tv_nsec-t1.tv_nsec)/1e9;
            double es=(t3.tv_sec-t2.tv_sec)+(t3.tv_nsec-t2.tv_nsec)/1e9;
            double tot=(t3.tv_sec-t0.tv_sec)+(t3.tv_nsec-t0.tv_nsec)/1e9;

            printf("%12lld %3d | %8.1f %8.1f %8.1f | %8.4f %5.2fx | %8.1f%s\n",
                   n, d, is_, rs, es, l2, l2/ram_bound, tot, reg?"":" BAD");
            fflush(stdout);

            free(h_deg); cudaFree(d_adj);
            if (tot > 14400) { printf("(>4hr)\n"); break; }
        }
    }
    return 0;
}
