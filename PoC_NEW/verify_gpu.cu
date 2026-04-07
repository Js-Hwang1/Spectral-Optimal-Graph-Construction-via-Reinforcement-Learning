/*
 * verify_gpu.cu — GPU-accelerated verification that random init + degree
 * regularization produces near-Ramanujan d-regular graphs.
 *
 * ALL computation on GPU — no host-device transfers for graph data.
 *   Phase 1: Random init (parallel edge generation on GPU)
 *   Phase 2: Degree regularization (GPU kernel)
 *   Phase 3: λ₂ via Lanczos with sparse matvec (GPU)
 *
 * Graph stored as CSR on GPU for sparse matvec.
 * During init/regularization, stored as sorted adjacency arrays.
 *
 * Build:
 *   nvcc -O3 -arch=sm_120 -o verify_gpu verify_gpu.cu -lcusparse -lcublas
 *
 * Usage:
 *   ./verify_gpu <n> <d> [seed]
 *   ./verify_gpu --sweep
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================
// GPU RNG: xoshiro128** per thread
// ============================================================

__device__ uint32_t dev_rotl(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

struct RNGState {
    uint32_t s[4];
};

__device__ uint32_t rng_next(RNGState *st) {
    uint32_t r = dev_rotl(st->s[1] * 5, 7) * 9;
    uint32_t t = st->s[1] << 9;
    st->s[2] ^= st->s[0]; st->s[3] ^= st->s[1];
    st->s[1] ^= st->s[2]; st->s[0] ^= st->s[3];
    st->s[2] ^= t; st->s[3] = dev_rotl(st->s[3], 11);
    return r;
}

__device__ void rng_init(RNGState *st, uint64_t seed, int tid) {
    uint64_t s = seed + tid * 2654435761ULL;
    st->s[0] = (uint32_t)(s);
    st->s[1] = (uint32_t)(s >> 16) ^ 0xABCD1234;
    st->s[2] = (uint32_t)(s * 6364136223846793005ULL);
    st->s[3] = (uint32_t)(s >> 32) ^ 0xDEADBEEF;
    for (int i = 0; i < 10; i++) rng_next(st);
}

// ============================================================
// Graph on GPU: fixed-size adjacency array per node
// adj_flat[i * max_deg + k] = k-th neighbor of node i (-1 if empty)
// deg[i] = current degree of node i
// ============================================================

// Phase 1: Each thread generates edges by picking random (i,j) pairs.
// Uses atomicCAS on adj slots to avoid duplicates.

__device__ int gpu_has_edge(int *adj_flat, int *deg, int i, int j, int max_deg) {
    for (int k = 0; k < deg[i]; k++)
        if (adj_flat[i * max_deg + k] == j) return 1;
    return 0;
}

__device__ int gpu_add_edge_atomic(int *adj_flat, int *deg, int i, int j, int max_deg) {
    // Try to atomically add j to node i's adjacency list
    int pos = atomicAdd(&deg[i], 1);
    if (pos >= max_deg) {
        atomicSub(&deg[i], 1);
        return 0; // overflow
    }
    adj_flat[i * max_deg + pos] = j;
    return 1;
}

__global__ void kernel_random_init(
    int *adj_flat, int *deg, int n, int d, int max_deg,
    long long edges_per_thread, uint64_t seed, int *global_count, long long target_m)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    RNGState rng;
    rng_init(&rng, seed, tid);

    for (long long e = 0; e < edges_per_thread; e++) {
        // Check if we've reached target
        if (atomicAdd(global_count, 0) >= (int)target_m) return;

        int i = rng_next(&rng) % n;
        int j = rng_next(&rng) % n;
        if (i == j) continue;
        // Canonical order to avoid double-insertion races
        int lo = i < j ? i : j;
        int hi = i < j ? j : i;

        // Check if edge exists (approximate — race possible but harmless)
        if (gpu_has_edge(adj_flat, deg, lo, hi, max_deg)) continue;

        // Try to add both directions
        int ok1 = gpu_add_edge_atomic(adj_flat, deg, lo, hi, max_deg);
        if (ok1) {
            int ok2 = gpu_add_edge_atomic(adj_flat, deg, hi, lo, max_deg);
            if (ok2) {
                atomicAdd(global_count, 1);
            } else {
                // Rollback lo side (find and remove hi)
                for (int k = 0; k < deg[lo]; k++) {
                    if (adj_flat[lo * max_deg + k] == hi) {
                        adj_flat[lo * max_deg + k] = -1;
                        atomicSub(&deg[lo], 1);
                        break;
                    }
                }
            }
        }
    }
}

// ============================================================
// Phase 2: Regularization on GPU
// Sequential swap logic in a single-thread kernel operating
// directly on GPU memory — zero PCIe transfer.
// Uses parallel reductions to find max/min degree nodes.
// ============================================================

// Find index of node with max degree (parallel reduction)
__global__ void kernel_find_max_over(
    const int *deg, int n, int d, int *result_idx, int *result_deg)
{
    extern __shared__ char smem_raw[];
    int *s_idx = (int*)smem_raw;
    int *s_deg = s_idx + blockDim.x;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    s_deg[tid] = (i < n && deg[i] > d) ? deg[i] : -1;
    s_idx[tid] = (i < n && deg[i] > d) ? i : -1;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_deg[tid + s] > s_deg[tid]) {
                s_deg[tid] = s_deg[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        // Atomic max across blocks
        int old = atomicMax(result_deg, s_deg[0]);
        if (s_deg[0] > old) {
            *result_idx = s_idx[0]; // race possible but OK for heuristic
        }
    }
}

__global__ void kernel_find_min_under(
    const int *deg, int n, int d, int *result_idx, int *result_deg)
{
    extern __shared__ char smem_raw[];
    int *s_idx = (int*)smem_raw;
    int *s_deg = s_idx + blockDim.x;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    s_deg[tid] = (i < n && deg[i] < d) ? deg[i] : n + 1;
    s_idx[tid] = (i < n && deg[i] < d) ? i : -1;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_deg[tid + s] < s_deg[tid]) {
                s_deg[tid] = s_deg[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        // Atomic min across blocks
        int old = atomicMin(result_deg, s_deg[0]);
        if (s_deg[0] < old) {
            *result_idx = s_idx[0];
        }
    }
}

// Single-thread kernel: perform one edge swap on GPU memory
__global__ void kernel_do_swap(
    int *adj_flat, int *deg, int n, int d, int max_deg,
    int u, int w)
{
    // This runs as a single thread — inherently sequential
    // but operates on GPU memory, no PCIe needed.

    // Find best_v in N(u): highest degree, not w, not adjacent to w
    int best_v = -1, best_vd = -1;
    for (int k = 0; k < deg[u]; k++) {
        int v = adj_flat[u * max_deg + k];
        if (v < 0 || v == w) continue;
        // Check if w~v
        int w_has_v = 0;
        for (int kk = 0; kk < deg[w]; kk++)
            if (adj_flat[w * max_deg + kk] == v) { w_has_v = 1; break; }
        if (w_has_v) continue;
        if (deg[v] > best_vd) { best_vd = deg[v]; best_v = v; }
    }

    if (best_v >= 0) {
        // Direct transfer: remove {u, best_v}, add {w, best_v}
        // Remove best_v from u's list
        for (int k = 0; k < deg[u]; k++) {
            if (adj_flat[u * max_deg + k] == best_v) {
                adj_flat[u * max_deg + k] = adj_flat[u * max_deg + deg[u] - 1];
                adj_flat[u * max_deg + deg[u] - 1] = -1;
                deg[u]--; break;
            }
        }
        // Remove u from best_v's list
        for (int k = 0; k < deg[best_v]; k++) {
            if (adj_flat[best_v * max_deg + k] == u) {
                adj_flat[best_v * max_deg + k] = adj_flat[best_v * max_deg + deg[best_v] - 1];
                adj_flat[best_v * max_deg + deg[best_v] - 1] = -1;
                deg[best_v]--; break;
            }
        }
        // Add best_v to w
        adj_flat[w * max_deg + deg[w]] = best_v; deg[w]++;
        adj_flat[best_v * max_deg + deg[best_v]] = w; deg[best_v]++;
    } else {
        // Check if u~w
        int u_has_w = 0;
        for (int k = 0; k < deg[u]; k++)
            if (adj_flat[u * max_deg + k] == w) { u_has_w = 1; break; }

        if (!u_has_w) {
            // Add {u,w}, then remove highest-deg neighbor of u (not w)
            adj_flat[u * max_deg + deg[u]] = w; deg[u]++;
            adj_flat[w * max_deg + deg[w]] = u; deg[w]++;

            best_v = -1; best_vd = -1;
            for (int k = 0; k < deg[u]; k++) {
                int v = adj_flat[u * max_deg + k];
                if (v < 0 || v == w) continue;
                if (deg[v] > best_vd) { best_vd = deg[v]; best_v = v; }
            }
            if (best_v >= 0) {
                for (int k = 0; k < deg[u]; k++) {
                    if (adj_flat[u * max_deg + k] == best_v) {
                        adj_flat[u * max_deg + k] = adj_flat[u * max_deg + deg[u]-1];
                        adj_flat[u * max_deg + deg[u]-1] = -1;
                        deg[u]--; break;
                    }
                }
                for (int k = 0; k < deg[best_v]; k++) {
                    if (adj_flat[best_v * max_deg + k] == u) {
                        adj_flat[best_v * max_deg + k] = adj_flat[best_v * max_deg + deg[best_v]-1];
                        adj_flat[best_v * max_deg + deg[best_v]-1] = -1;
                        deg[best_v]--; break;
                    }
                }
            }
        } else {
            // Remove highest-deg neighbor of u
            best_v = -1; best_vd = -1;
            for (int k = 0; k < deg[u]; k++) {
                int v = adj_flat[u * max_deg + k];
                if (v < 0) continue;
                if (deg[v] > best_vd) { best_vd = deg[v]; best_v = v; }
            }
            if (best_v >= 0) {
                for (int k = 0; k < deg[u]; k++) {
                    if (adj_flat[u * max_deg + k] == best_v) {
                        adj_flat[u * max_deg + k] = adj_flat[u * max_deg + deg[u]-1];
                        adj_flat[u * max_deg + deg[u]-1] = -1;
                        deg[u]--; break;
                    }
                }
                for (int k = 0; k < deg[best_v]; k++) {
                    if (adj_flat[best_v * max_deg + k] == u) {
                        adj_flat[best_v * max_deg + k] = adj_flat[best_v * max_deg + deg[best_v]-1];
                        adj_flat[best_v * max_deg + deg[best_v]-1] = -1;
                        deg[best_v]--; break;
                    }
                }
            }
            // Add edge from w to some under-degree non-neighbor
            // Scan from node 0 upward
            for (int x = 0; x < n; x++) {
                if (x == w || deg[x] >= d) continue;
                int w_has_x = 0;
                for (int k = 0; k < deg[w]; k++)
                    if (adj_flat[w * max_deg + k] == x) { w_has_x = 1; break; }
                if (!w_has_x) {
                    adj_flat[w * max_deg + deg[w]] = x; deg[w]++;
                    adj_flat[x * max_deg + deg[x]] = w; deg[x]++;
                    break;
                }
            }
        }
    }
}

// Host function that orchestrates GPU regularization
void gpu_regularize(int *d_adj, int *d_deg, int n, int d, int max_deg) {
    int *d_max_idx, *d_max_deg, *d_min_idx, *d_min_deg;
    CUDA_CHECK(cudaMalloc(&d_max_idx, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_max_deg, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_min_idx, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_min_deg, sizeof(int)));

    int block = 256;
    int grid = (n + block - 1) / block;
    size_t smem = 2 * block * sizeof(int);

    long long max_swaps = (long long)n * d * 4;
    int h_max_deg, h_min_deg, h_max_idx, h_min_idx;

    for (long long sw = 0; sw < max_swaps; sw++) {
        // Find max-over and min-under via parallel reduction
        int neg1 = -1, big = n + 1;
        CUDA_CHECK(cudaMemcpy(d_max_deg, &neg1, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_max_idx, &neg1, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_min_deg, &big, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_min_idx, &neg1, sizeof(int), cudaMemcpyHostToDevice));

        kernel_find_max_over<<<grid, block, smem>>>(d_deg, n, d, d_max_idx, d_max_deg);
        kernel_find_min_under<<<grid, block, smem>>>(d_deg, n, d, d_min_idx, d_min_deg);
        cudaDeviceSynchronize();

        CUDA_CHECK(cudaMemcpy(&h_max_deg, d_max_deg, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_min_deg, d_min_deg, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_max_idx, d_max_idx, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_min_idx, d_min_idx, sizeof(int), cudaMemcpyDeviceToHost));

        if (h_max_deg <= d && h_min_deg >= d) break; // done
        if (h_max_idx < 0 || h_min_idx < 0) break;

        // Do the swap on GPU (single thread, but data stays on device)
        kernel_do_swap<<<1, 1>>>(d_adj, d_deg, n, d, max_deg, h_max_idx, h_min_idx);
        // No sync needed here — next iteration's reductions will sync implicitly
    }

    cudaFree(d_max_idx); cudaFree(d_max_deg);
    cudaFree(d_min_idx); cudaFree(d_min_deg);
}

// ============================================================
// Phase 3: Lanczos on GPU — sparse Laplacian matvec
// ============================================================

// Kernel: y = L * x where L = D - A (sparse, adjacency list)
__global__ void kernel_laplacian_matvec(
    const int *adj_flat, const int *deg, const double *x, double *y,
    int n, int max_deg)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double s = deg[i] * x[i];
    for (int k = 0; k < deg[i]; k++) {
        int j = adj_flat[i * max_deg + k];
        if (j >= 0) s -= x[j];
    }
    y[i] = s;
}

// Kernel: various vector ops
__global__ void kernel_axpy(double *y, const double *x, double a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

__global__ void kernel_scale(double *x, double a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= a;
}

__global__ void kernel_init_random(double *x, int n, uint64_t seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    RNGState rng;
    rng_init(&rng, seed, i);
    x[i] = (double)(rng_next(&rng) & 0xFFFF) / 32768.0 - 1.0;
}

// Dot product via reduction
__device__ double warp_reduce_sum(double val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void kernel_dot(const double *a, const double *b, double *result, int n) {
    extern __shared__ char smem_raw[];
    double *smem = (double*)smem_raw;
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    smem[tid] = (i < n) ? a[i] * b[i] : 0.0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(result, smem[0]);
}

double gpu_dot(const double *a, const double *b, int n, double *d_tmp) {
    CUDA_CHECK(cudaMemset(d_tmp, 0, sizeof(double)));
    int block = 256;
    int grid = (n + block - 1) / block;
    kernel_dot<<<grid, block, block * sizeof(double)>>>(a, b, d_tmp, n);
    double result;
    CUDA_CHECK(cudaMemcpy(&result, d_tmp, sizeof(double), cudaMemcpyDeviceToHost));
    return result;
}

double gpu_lanczos_lambda2(
    int *d_adj, int *d_deg, int n, int max_deg, int lanczos_k)
{
    int block = 256;
    int grid = (n + block - 1) / block;

    double *d_v0, *d_v1, *d_w, *d_tmp;
    CUDA_CHECK(cudaMalloc(&d_v0, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_v1, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_w, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tmp, sizeof(double)));

    double *h_alpha = (double*)malloc(lanczos_k * sizeof(double));
    double *h_beta = (double*)malloc(lanczos_k * sizeof(double));

    // Init v1 random, orthogonal to 1
    kernel_init_random<<<grid, block>>>(d_v1, n, 777);
    // Subtract mean
    double sum = gpu_dot(d_v1, d_v1, n, d_tmp); // just to sync
    // Simple: compute sum via dot with ones vector... use host
    double *h_v = (double*)malloc(n * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h_v, d_v1, n * sizeof(double), cudaMemcpyDeviceToHost));
    sum = 0;
    for (int i = 0; i < n; i++) sum += h_v[i];
    sum /= n;
    double norm = 0;
    for (int i = 0; i < n; i++) { h_v[i] -= sum; norm += h_v[i] * h_v[i]; }
    norm = sqrt(norm);
    for (int i = 0; i < n; i++) h_v[i] /= norm;
    CUDA_CHECK(cudaMemcpy(d_v1, h_v, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_v0, 0, n * sizeof(double)));

    double beta = 0;
    int actual_k = 0;

    for (int j = 0; j < lanczos_k; j++) {
        // w = L * v1
        kernel_laplacian_matvec<<<grid, block>>>(d_adj, d_deg, d_v1, d_w, n, max_deg);

        // alpha = v1^T w
        double alpha = gpu_dot(d_v1, d_w, n, d_tmp);
        h_alpha[j] = alpha;

        // w = w - alpha*v1 - beta*v0
        kernel_axpy<<<grid, block>>>(d_w, d_v1, -alpha, n);
        if (beta > 0) kernel_axpy<<<grid, block>>>(d_w, d_v0, -beta, n);

        // Orthogonalize against 1
        CUDA_CHECK(cudaMemcpy(h_v, d_w, n * sizeof(double), cudaMemcpyDeviceToHost));
        sum = 0;
        for (int i = 0; i < n; i++) sum += h_v[i];
        sum /= n;
        for (int i = 0; i < n; i++) h_v[i] -= sum;
        CUDA_CHECK(cudaMemcpy(d_w, h_v, n * sizeof(double), cudaMemcpyHostToDevice));

        // beta = ||w||
        double w_norm_sq = gpu_dot(d_w, d_w, n, d_tmp);
        beta = sqrt(w_norm_sq);
        h_beta[j] = beta;
        actual_k = j + 1;

        if (beta < 1e-12) break;

        // v0 = v1, v1 = w / beta
        CUDA_CHECK(cudaMemcpy(d_v0, d_v1, n * sizeof(double), cudaMemcpyDeviceToDevice));
        double inv_beta = 1.0 / beta;
        CUDA_CHECK(cudaMemcpy(d_v1, d_w, n * sizeof(double), cudaMemcpyDeviceToDevice));
        kernel_scale<<<grid, block>>>(d_v1, inv_beta, n);
    }

    // Bisection on tridiagonal for λ₂
    double hi = 0;
    for (int j = 0; j < actual_k; j++) {
        double bound = fabs(h_alpha[j]) +
            (j > 0 ? fabs(h_beta[j-1]) : 0) +
            (j < actual_k - 1 ? fabs(h_beta[j]) : 0);
        if (bound > hi) hi = bound;
    }
    hi += 1.0;
    double lo = 0.0;

    for (int bisect = 0; bisect < 100; bisect++) {
        double mu = (lo + hi) / 2.0;
        int count = 0;
        double d_val = h_alpha[0] - mu;
        if (d_val < 0) count++;
        for (int j = 1; j < actual_k; j++) {
            if (fabs(d_val) < 1e-30) d_val = 1e-30;
            d_val = (h_alpha[j] - mu) - h_beta[j-1] * h_beta[j-1] / d_val;
            if (d_val < 0) count++;
        }
        if (count >= 2) hi = mu;
        else lo = mu;
    }

    free(h_v); free(h_alpha); free(h_beta);
    cudaFree(d_v0); cudaFree(d_v1); cudaFree(d_w); cudaFree(d_tmp);
    return (lo + hi) / 2.0;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char **argv) {
    int d = 4;
    int max_deg = d * 3 + 4; // enough headroom during init

    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    fprintf(stderr, "GPU: %s (%.1f GB, SM %d.%d)\n",
            prop.name, prop.totalGlobalMem / 1e9,
            prop.major, prop.minor);

    if (argc >= 3 && strcmp(argv[1], "--sweep") != 0) {
        int n = atoi(argv[1]);
        d = atoi(argv[2]);
        max_deg = d * 3 + 4;
        uint64_t seed = argc >= 4 ? (uint64_t)atoll(argv[3]) : 42;
        double ram = d - 2.0 * sqrt(d - 1.0);
        long long m = (long long)n * d / 2;
        long long mem_mb = (long long)n * max_deg * 4 / (1024*1024);

        printf("n=%d d=%d m=%lld Ram=%.4f GPU_mem=~%lldMB\n",
               n, d, m, ram, mem_mb);

        // Allocate on GPU
        int *d_adj, *d_deg, *d_count;
        CUDA_CHECK(cudaMalloc(&d_adj, (long long)n * max_deg * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_deg, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_adj, 0xFF, (long long)n * max_deg * sizeof(int))); // -1
        CUDA_CHECK(cudaMemset(d_deg, 0, n * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

        // Phase 1: random init on GPU
        cudaEvent_t t0, t1, t2, t3;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventCreate(&t2); cudaEventCreate(&t3);

        cudaEventRecord(t0);
        int threads = 256;
        int blocks = 1024;
        long long edges_per_thread = (m * 3) / (blocks * threads) + 1;
        kernel_random_init<<<blocks, threads>>>(
            d_adj, d_deg, n, d, max_deg, edges_per_thread, seed, d_count, m);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);

        int h_count;
        CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
        float init_ms;
        cudaEventElapsedTime(&init_ms, t0, t1);
        printf("Phase 1: %d/%lld edges (%.2fs)\n", h_count, m, init_ms / 1000);

        // Phase 2: regularize on GPU (zero PCIe transfer)
        cudaEventRecord(t1);
        gpu_regularize(d_adj, d_deg, n, d, max_deg);
        cudaEventRecord(t2);
        cudaEventSynchronize(t2);

        // Verify regularity (copy just deg array — tiny)
        int *h_deg = (int*)malloc(n * sizeof(int));
        CUDA_CHECK(cudaMemcpy(h_deg, d_deg, n * sizeof(int), cudaMemcpyDeviceToHost));
        int reg = 1;
        for (int i = 0; i < n; i++) if (h_deg[i] != d) { reg = 0; break; }
        float reg_ms;
        cudaEventElapsedTime(&reg_ms, t1, t2);
        printf("Phase 2: regularize (%.2fs) regular=%s\n",
               reg_ms / 1000, reg ? "YES" : "NO");

        // Phase 3: Lanczos on GPU
        int lanczos_k = 500;
        if (n > 500000) lanczos_k = 800;
        cudaEventRecord(t2);
        double l2 = gpu_lanczos_lambda2(d_adj, d_deg, n, max_deg, lanczos_k);
        cudaEventRecord(t3);
        cudaEventSynchronize(t3);
        float eig_ms;
        cudaEventElapsedTime(&eig_ms, t2, t3);

        float total_ms;
        cudaEventElapsedTime(&total_ms, t0, t3);
        printf("Phase 3: λ₂≈%.6f Ram=%.6f ratio=%.2fx (%.2fs)\n",
               l2, ram, l2/ram, eig_ms / 1000);
        printf("RESULT: n=%d d=%d lambda2=%.6f ratio=%.4f regular=%s total=%.2fs\n",
               n, d, l2, l2/ram, reg?"YES":"NO", total_ms/1000);

        free(h_deg);
        cudaFree(d_adj); cudaFree(d_deg); cudaFree(d_count);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaEventDestroy(t2); cudaEventDestroy(t3);

    } else {
        // Sweep powers of 2
        double ram = d - 2.0 * sqrt(d - 1.0);
        printf("%10s %3s | %8s %8s %8s | %8s %6s | %8s\n",
               "n", "d", "init_s", "reg_s", "eig_s", "lambda2", "ratio", "total");
        printf("-------------------------------------------------------------------\n");

        for (int logn = 5; logn <= 22; logn++) {
            int n = 1 << logn;
            long long m = (long long)n * d / 2;
            long long mem = (long long)n * max_deg * sizeof(int);
            if (mem > 80ULL * 1024 * 1024 * 1024) {
                printf("%10d — skipped (%.1f GB > GPU mem)\n", n, mem / 1e9);
                continue;
            }

            int *d_adj, *d_deg, *d_count;
            CUDA_CHECK(cudaMalloc(&d_adj, (long long)n * max_deg * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_deg, n * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
            CUDA_CHECK(cudaMemset(d_adj, 0xFF, (long long)n * max_deg * sizeof(int)));
            CUDA_CHECK(cudaMemset(d_deg, 0, n * sizeof(int)));
            CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

            cudaEvent_t t0, t1, t2, t3;
            cudaEventCreate(&t0); cudaEventCreate(&t1);
            cudaEventCreate(&t2); cudaEventCreate(&t3);

            // Phase 1
            cudaEventRecord(t0);
            int threads = 256, blocks = min(1024, (int)(m / 256 + 1));
            long long ept = (m * 3) / (blocks * threads) + 1;
            kernel_random_init<<<blocks, threads>>>(
                d_adj, d_deg, n, d, max_deg, ept, 42, d_count, m);
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            float init_ms;
            cudaEventElapsedTime(&init_ms, t0, t1);

            // Phase 2 on GPU
            cudaEventRecord(t1);
            gpu_regularize(d_adj, d_deg, n, d, max_deg);
            cudaEventRecord(t2);
            cudaEventSynchronize(t2);
            float reg_ms;
            cudaEventElapsedTime(&reg_ms, t1, t2);

            int *h_deg = (int*)malloc(n * sizeof(int));
            CUDA_CHECK(cudaMemcpy(h_deg, d_deg, n * sizeof(int), cudaMemcpyDeviceToHost));
            int reg = 1;
            for (int i = 0; i < n; i++) if (h_deg[i] != d) { reg = 0; break; }

            // Phase 3
            int k = logn <= 14 ? 300 : (logn <= 18 ? 500 : 800);
            cudaEventRecord(t2);
            double l2 = gpu_lanczos_lambda2(d_adj, d_deg, n, max_deg, k);
            cudaEventRecord(t3);
            cudaEventSynchronize(t3);
            float eig_ms, total_ms;
            cudaEventElapsedTime(&eig_ms, t2, t3);
            cudaEventElapsedTime(&total_ms, t0, t3);

            printf("%10d %3d | %8.2f %8.2f %8.2f | %8.4f %5.2fx | %8.2f%s\n",
                   n, d, init_ms/1000, reg_ms/1000, eig_ms/1000,
                   l2, l2/ram, total_ms/1000, reg ? "" : " BAD");
            fflush(stdout);

            free(h_deg);
            cudaFree(d_adj); cudaFree(d_deg); cudaFree(d_count);
            cudaEventDestroy(t0); cudaEventDestroy(t1);
            cudaEventDestroy(t2); cudaEventDestroy(t3);

            if (total_ms > 3600000) { printf("(>1hr, stopping)\n"); break; }
        }
    }
    return 0;
}
