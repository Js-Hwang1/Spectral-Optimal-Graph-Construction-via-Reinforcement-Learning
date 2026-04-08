/*
 * verify_gpu_v2.cu — Large-scale verification with PARALLEL regularization.
 *
 * Phase 1 (CPU): Random init with exactly m=nd/2 edges
 * Phase 2 (CPU): PARALLEL degree regularization
 *   - Each round: find all over/under nodes, match pairs, swap in batch
 *   - ~10-20 rounds, each O(n) work → O(n) total (vs O(n²) sequential)
 * Phase 3 (GPU): Lanczos eigensolve
 *
 * Build:
 *   nvcc -O3 -arch=sm_120 -o verify_gpu_v2 verify_gpu_v2.cu -lm
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
static int rng_int(int n) { return (int)(rng_next() % (uint32_t)n); }

// ============================================================
// Phase 1 (CPU): Random init — exactly m edges
// ============================================================
void phase1_init(int *adj, int *deg, int n, int d, int max_deg, uint64_t seed) {
    long long m = (long long)n * d / 2;
    rng_seed(seed);
    memset(deg, 0, n * sizeof(int));
    for (long long i = 0; i < (long long)n * max_deg; i++) adj[i] = -1;

    long long count = 0;
    while (count < m) {
        int u = rng_int(n), v = rng_int(n);
        if (u == v) continue;
        int exists = 0;
        for (int k = 0; k < deg[u]; k++)
            if (adj[u * max_deg + k] == v) { exists = 1; break; }
        if (exists) continue;
        if (deg[u] >= max_deg - 1 || deg[v] >= max_deg - 1) continue; // skip if full
        adj[u * max_deg + deg[u]++] = v;
        adj[v * max_deg + deg[v]++] = u;
        count++;
    }
}

// ============================================================
// Phase 2 (CPU): PARALLEL degree regularization
// ============================================================
void phase2_parallel_regularize(int *adj, int *deg, int n, int d, int max_deg) {
    // Temp arrays
    int *over = (int*)malloc(n * sizeof(int));
    int *under = (int*)malloc(n * sizeof(int));
    int *claimed = (int*)calloc(n, sizeof(int));

    int max_rounds = n * d;
    int total_swaps = 0;

    for (int round = 0; round < max_rounds; round++) {
        // 1. Classify nodes
        int n_over = 0, n_under = 0;
        for (int i = 0; i < n; i++) {
            if (deg[i] > d) over[n_over++] = i;
            else if (deg[i] < d) under[n_under++] = i;
        }
        if (n_over == 0 && n_under == 0) break;
        if (n_over == 0 || n_under == 0) break;

        // 2. Match pairs and swap (independent pairs only)
        int num_pairs = n_over < n_under ? n_over : n_under;
        memset(claimed, 0, n * sizeof(int));
        int swaps_this_round = 0;

        for (int idx = 0; idx < num_pairs; idx++) {
            int u = over[idx];
            int w = under[idx];
            if (claimed[u] || claimed[w]) continue;

            // Find v in N(u): not w, not claimed, not in N(w)
            int best_v = -1, best_vd = -1;
            for (int k = 0; k < deg[u]; k++) {
                int v = adj[u * max_deg + k];
                if (v < 0 || v == w || claimed[v]) continue;
                // Check w~v
                int skip = 0;
                for (int kk = 0; kk < deg[w]; kk++)
                    if (adj[w * max_deg + kk] == v) { skip = 1; break; }
                if (skip) continue;
                if (deg[v] > best_vd) { best_vd = deg[v]; best_v = v; }
            }

            if (best_v >= 0) {
                claimed[u] = claimed[w] = claimed[best_v] = 1;

                // Remove {u, best_v}
                for (int k = 0; k < deg[u]; k++)
                    if (adj[u * max_deg + k] == best_v) {
                        adj[u * max_deg + k] = adj[u * max_deg + --deg[u]];
                        adj[u * max_deg + deg[u]] = -1; break; }
                for (int k = 0; k < deg[best_v]; k++)
                    if (adj[best_v * max_deg + k] == u) {
                        adj[best_v * max_deg + k] = adj[best_v * max_deg + --deg[best_v]];
                        adj[best_v * max_deg + deg[best_v]] = -1; break; }

                // Add {w, best_v}
                adj[w * max_deg + deg[w]++] = best_v;
                adj[best_v * max_deg + deg[best_v]++] = w;

                swaps_this_round++;
            }
        }

        total_swaps += swaps_this_round;

        // If no swaps happened, do one sequential fallback
        if (swaps_this_round == 0) {
            // Find any over/under pair and force a swap
            for (int oi = 0; oi < n_over && swaps_this_round == 0; oi++) {
                int u = over[oi];
                if (deg[u] <= d) continue;
                for (int ui = 0; ui < n_under && swaps_this_round == 0; ui++) {
                    int w = under[ui];
                    if (deg[w] >= d) continue;

                    // Try direct transfer
                    for (int k = 0; k < deg[u]; k++) {
                        int v = adj[u * max_deg + k];
                        if (v < 0 || v == w) continue;
                        int skip = 0;
                        for (int kk = 0; kk < deg[w]; kk++)
                            if (adj[w * max_deg + kk] == v) { skip = 1; break; }
                        if (skip) continue;

                        // Do swap
                        for (int k2 = 0; k2 < deg[u]; k2++)
                            if (adj[u * max_deg + k2] == v) {
                                adj[u * max_deg + k2] = adj[u * max_deg + --deg[u]];
                                adj[u * max_deg + deg[u]] = -1; break; }
                        for (int k2 = 0; k2 < deg[v]; k2++)
                            if (adj[v * max_deg + k2] == u) {
                                adj[v * max_deg + k2] = adj[v * max_deg + --deg[v]];
                                adj[v * max_deg + deg[v]] = -1; break; }
                        adj[w * max_deg + deg[w]++] = v;
                        adj[v * max_deg + deg[v]++] = w;
                        swaps_this_round++;
                        total_swaps++;
                        break;
                    }

                    // Fallback: add {u,w} remove worst from u
                    if (swaps_this_round == 0) {
                        int uw = 0;
                        for (int k = 0; k < deg[u]; k++)
                            if (adj[u * max_deg + k] == w) { uw = 1; break; }
                        if (!uw) {
                            adj[u * max_deg + deg[u]++] = w;
                            adj[w * max_deg + deg[w]++] = u;
                            int bv = -1, bd = -1;
                            for (int k = 0; k < deg[u]; k++) {
                                int v = adj[u * max_deg + k];
                                if (v < 0 || v == w) continue;
                                if (deg[v] > bd) { bd = deg[v]; bv = v; }
                            }
                            if (bv >= 0) {
                                for (int k = 0; k < deg[u]; k++)
                                    if (adj[u * max_deg + k] == bv) {
                                        adj[u * max_deg + k] = adj[u * max_deg + --deg[u]];
                                        adj[u * max_deg + deg[u]] = -1; break; }
                                for (int k = 0; k < deg[bv]; k++)
                                    if (adj[bv * max_deg + k] == u) {
                                        adj[bv * max_deg + k] = adj[bv * max_deg + --deg[bv]];
                                        adj[bv * max_deg + deg[bv]] = -1; break; }
                            }
                            swaps_this_round++;
                            total_swaps++;
                        }
                    }
                }
            }
        }
    }

    free(over); free(under); free(claimed);
}

// ============================================================
// Phase 3 (GPU): Lanczos (same as v1)
// ============================================================

__global__ void kernel_laplacian_mv(
    const int *adj, const int *deg, const double *x, double *y,
    int n, int max_deg)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double s = deg[i] * x[i];
    for (int k = 0; k < deg[i]; k++) {
        int j = adj[i * max_deg + k];
        if (j >= 0) s -= x[j];
    }
    y[i] = s;
}

__global__ void kernel_axpy(double *y, const double *x, double a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

__global__ void kernel_scale(double *x, double a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= a;
}

__global__ void kernel_dot(const double *a, const double *b, double *out, int n) {
    extern __shared__ char smem[];
    double *sd = (double*)smem;
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    sd[tid] = (i < n) ? a[i] * b[i] : 0.0;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sd[tid] += sd[tid+s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sd[0]);
}

double gpu_dot(const double *a, const double *b, int n, double *tmp) {
    CUDA_CHECK(cudaMemset(tmp, 0, sizeof(double)));
    kernel_dot<<<(n+255)/256, 256, 256*sizeof(double)>>>(a, b, tmp, n);
    double r; CUDA_CHECK(cudaMemcpy(&r, tmp, sizeof(double), cudaMemcpyDeviceToHost));
    return r;
}

double gpu_lanczos(int *d_adj, int *d_deg, int n, int max_deg, int iters) {
    int bl = 256, gr = (n+bl-1)/bl;
    double *d_v0, *d_v1, *d_w, *d_ones, *d_tmp;
    CUDA_CHECK(cudaMalloc(&d_v0, n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_v1, n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_w, n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ones, n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tmp, sizeof(double)));

    double *h = (double*)malloc(n*sizeof(double));
    for (int i = 0; i < n; i++) h[i] = 1.0;
    CUDA_CHECK(cudaMemcpy(d_ones, h, n*sizeof(double), cudaMemcpyHostToDevice));

    srand(999);
    for (int i = 0; i < n; i++) h[i] = (double)(rand()%10000)/5000.0 - 1.0;
    double s = 0; for (int i = 0; i < n; i++) s += h[i]; s /= n;
    double nm = 0; for (int i = 0; i < n; i++) { h[i] -= s; nm += h[i]*h[i]; }
    nm = sqrt(nm); for (int i = 0; i < n; i++) h[i] /= nm;
    CUDA_CHECK(cudaMemcpy(d_v1, h, n*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_v0, 0, n*sizeof(double)));

    double *al = (double*)malloc(iters*sizeof(double));
    double *be = (double*)malloc(iters*sizeof(double));
    double beta = 0; int ak = 0;

    for (int j = 0; j < iters; j++) {
        kernel_laplacian_mv<<<gr,bl>>>(d_adj, d_deg, d_v1, d_w, n, max_deg);
        double alpha = gpu_dot(d_v1, d_w, n, d_tmp);
        al[j] = alpha;
        kernel_axpy<<<gr,bl>>>(d_w, d_v1, -alpha, n);
        if (beta > 0) kernel_axpy<<<gr,bl>>>(d_w, d_v0, -beta, n);
        double ws = gpu_dot(d_w, d_ones, n, d_tmp);
        kernel_axpy<<<gr,bl>>>(d_w, d_ones, -ws/n, n);
        double wn = gpu_dot(d_w, d_w, n, d_tmp);
        beta = sqrt(wn); be[j] = beta; ak = j+1;
        if (beta < 1e-12) break;
        CUDA_CHECK(cudaMemcpy(d_v0, d_v1, n*sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_v1, d_w, n*sizeof(double), cudaMemcpyDeviceToDevice));
        kernel_scale<<<gr,bl>>>(d_v1, 1.0/beta, n);
    }

    double hi = 0;
    for (int j = 0; j < ak; j++) {
        double b2 = fabs(al[j]) + (j>0?fabs(be[j-1]):0) + (j<ak-1?fabs(be[j]):0);
        if (b2 > hi) hi = b2;
    }
    hi += 1; double lo = 0;
    for (int b2 = 0; b2 < 100; b2++) {
        double mu = (lo+hi)/2;
        int cnt = 0; double dv = al[0]-mu;
        if (dv < 0) cnt++;
        for (int j = 1; j < ak; j++) {
            if (fabs(dv) < 1e-30) dv = 1e-30;
            dv = (al[j]-mu) - be[j-1]*be[j-1]/dv;
            if (dv < 0) cnt++;
        }
        if (cnt >= 2) hi = mu; else lo = mu;
    }

    free(h); free(al); free(be);
    cudaFree(d_v0); cudaFree(d_v1); cudaFree(d_w); cudaFree(d_ones); cudaFree(d_tmp);
    return (lo+hi)/2;
}

// ============================================================
// Main
// ============================================================
int main(int argc, char **argv) {
    int d = 4, max_deg = d + 12;  // d + ~3*sqrt(d) + safety margin
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    fprintf(stderr, "GPU: %s (%.1f GB)\n", prop.name, prop.totalGlobalMem/1e9);

    if (argc >= 3 && strcmp(argv[1], "--sweep") != 0) {
        int n = atoi(argv[1]); d = atoi(argv[2]); max_deg = d + 12;
        uint64_t seed = argc >= 4 ? atoll(argv[3]) : 42;
        double ram = d - 2*sqrt(d-1);

        printf("n=%d d=%d Ram=%.4f\n", n, d, ram);

        struct timespec t0,t1,t2,t3;
        int *h_adj = (int*)malloc((long long)n*max_deg*sizeof(int));
        int *h_deg = (int*)malloc(n*sizeof(int));

        clock_gettime(CLOCK_MONOTONIC, &t0);
        phase1_init(h_adj, h_deg, n, d, max_deg, seed);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        printf("Phase 1: %.2fs\n", (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9);

        phase2_parallel_regularize(h_adj, h_deg, n, d, max_deg);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        printf("Phase 2: %.2fs\n", (t2.tv_sec-t1.tv_sec)+(t2.tv_nsec-t1.tv_nsec)/1e9);

        int reg = 1;
        for (int i = 0; i < n; i++) if (h_deg[i] != d) { reg = 0; break; }
        printf("Regular: %s\n", reg?"YES":"NO");

        int *d_adj, *d_deg;
        CUDA_CHECK(cudaMalloc(&d_adj, (long long)n*max_deg*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_deg, n*sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_adj, h_adj, (long long)n*max_deg*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_deg, h_deg, n*sizeof(int), cudaMemcpyHostToDevice));

        int k = n < 100000 ? 300 : (n < 1000000 ? 500 : 800);
        double l2 = gpu_lanczos(d_adj, d_deg, n, max_deg, k);
        clock_gettime(CLOCK_MONOTONIC, &t3);
        printf("Phase 3: λ₂≈%.6f ratio=%.2fx (%.2fs)\n", l2, l2/ram,
               (t3.tv_sec-t2.tv_sec)+(t3.tv_nsec-t2.tv_nsec)/1e9);
        printf("RESULT: n=%d d=%d lambda2=%.6f ratio=%.4f regular=%s total=%.2fs\n",
               n, d, l2, l2/ram, reg?"YES":"NO",
               (t3.tv_sec-t0.tv_sec)+(t3.tv_nsec-t0.tv_nsec)/1e9);

        free(h_adj); free(h_deg); cudaFree(d_adj); cudaFree(d_deg);

    } else {
        double ram = d - 2*sqrt(d-1);
        printf("%10s %3s | %8s %8s %8s | %8s %6s | %8s\n",
               "n","d","init_s","reg_s","eig_s","lambda2","ratio","total");
        printf("-------------------------------------------------------------------\n");

        for (int logn = 5; logn <= 30; logn++) {
            int n = 1 << logn;
            max_deg = d*3+4;
            int *h_adj = (int*)malloc((long long)n*max_deg*sizeof(int));
            int *h_deg = (int*)malloc(n*sizeof(int));
            if (!h_adj || !h_deg) { printf("OOM at n=2^%d=%d\n", logn, n); break; }

            struct timespec t0,t1,t2,t3;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            phase1_init(h_adj, h_deg, n, d, max_deg, 42);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double is_ = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;

            phase2_parallel_regularize(h_adj, h_deg, n, d, max_deg);
            clock_gettime(CLOCK_MONOTONIC, &t2);
            double rs = (t2.tv_sec-t1.tv_sec)+(t2.tv_nsec-t1.tv_nsec)/1e9;

            int reg = 1;
            for (int i = 0; i < n; i++) if (h_deg[i] != d) { reg = 0; break; }

            int *d_adj, *d_deg;
            CUDA_CHECK(cudaMalloc(&d_adj, (long long)n*max_deg*sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_deg, n*sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_adj, h_adj, (long long)n*max_deg*sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_deg, h_deg, n*sizeof(int), cudaMemcpyHostToDevice));

            int k = logn <= 14 ? 300 : (logn <= 18 ? 500 : 800);
            double l2 = gpu_lanczos(d_adj, d_deg, n, max_deg, k);
            clock_gettime(CLOCK_MONOTONIC, &t3);
            double es = (t3.tv_sec-t2.tv_sec)+(t3.tv_nsec-t2.tv_nsec)/1e9;
            double tot = (t3.tv_sec-t0.tv_sec)+(t3.tv_nsec-t0.tv_nsec)/1e9;

            printf("%10d %3d | %8.2f %8.2f %8.2f | %8.4f %5.2fx | %8.2f%s\n",
                   n, d, is_, rs, es, l2, l2/ram, tot, reg?"":" BAD");
            fflush(stdout);

            free(h_adj); free(h_deg); cudaFree(d_adj); cudaFree(d_deg);
            if (tot > 7200) { printf("(>2hr)\n"); break; }
        }
    }
    return 0;
}
