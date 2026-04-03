/**
 * Custom CUDA kernels for batched G(n,m) graph RL environment.
 *
 * Supports variable n per graph in a batch. Tensors are padded to max_n.
 * Each thread block handles one graph (B blocks, max_n threads/block).
 *
 * Step kernel matches CPU pipeline EXACTLY:
 *   1. BFS connectivity check  (matches is_connected)
 *   2. Lanczos Fiedler w/ warm start  (matches lanczos_fiedler(adj, k=10, v_init=...))
 *   3. Jacobi eigenvalues-only for exact λ₂  (matches algebraic_connectivity)
 *
 * Target: NVIDIA sm_80+ (Ampere/Ada/Blackwell)
 * Max N: 128 (shared memory limited)
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <float.h>

#define MAX_N 128
#define LANCZOS_K 10

/**
 * Kernel A: fused_step_kernel
 *
 * One block per graph. max_n threads per block.
 * Per-graph n from g_n_vals[b]. Only threads with tid < n_b do work.
 * All threads reach every __syncthreads() (block-uniform control flow).
 *
 * Matches CPU pipeline:
 *   _try_rewire(u, v, w)  →  BFS connectivity
 *   lanczos_fiedler(adj, k=10, v_init=fiedler_approx)  →  warm-started Lanczos
 *   algebraic_connectivity(adj)  →  Jacobi eigenvalues-only for exact λ₂
 *   reward = new_lambda2 - old_lambda2
 *
 * Shared memory layout (max_n-based strides):
 *   s_adj[max_n*max_n]      adjacency matrix (persistent)
 *   s_work[max_n*max_n]     multipurpose: BFS / Lanczos V / Jacobi L
 *   s_deg[max_n]            degrees
 *   s_vec[max_n]            Fiedler vector
 *   s_w[max_n]              temp vector (Lanczos w)
 *   s_alpha[K]              Lanczos tridiagonal diagonal
 *   s_beta[K]               Lanczos tridiagonal off-diagonal
 *   s_T[K*K]                K×K tridiagonal for small eigenproblem
 *   s_TV[K*K]               K×K eigenvectors of T
 *   s_scratch[8]            temporaries
 *
 * Total: 2*max_n² + 3*max_n + 2*K + 2*K² + 8 floats
 */
extern "C" __global__ void fused_step_kernel(
    float* __restrict__ g_adj,
    float* __restrict__ g_degrees,
    float* __restrict__ g_lambda2,
    float* __restrict__ g_fiedler,
    const int* __restrict__ g_nb_actions,
    const int* __restrict__ g_dests,
    int* __restrict__ g_current_node,
    int* __restrict__ g_current_round,
    const int* __restrict__ g_phi,
    int* __restrict__ g_done,
    int* __restrict__ g_total_rewires,
    float* __restrict__ g_rewards,
    const int* __restrict__ g_n_vals,
    const int max_n,
    const int sweeps,
    const int inference_mode
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_b = g_n_vals[b];

    // All threads return if done (block-uniform, no sync issue)
    if (g_done[b]) {
        if (tid == 0) g_rewards[b] = 0.0f;
        return;
    }

    extern __shared__ float smem[];
    float* s_adj     = smem;
    float* s_work    = smem + max_n * max_n;
    float* s_deg     = smem + 2 * max_n * max_n;
    float* s_vec     = smem + 2 * max_n * max_n + max_n;
    float* s_w       = smem + 2 * max_n * max_n + 2 * max_n;
    float* s_alpha   = smem + 2 * max_n * max_n + 3 * max_n;
    float* s_beta    = smem + 2 * max_n * max_n + 3 * max_n + LANCZOS_K;
    float* s_T       = smem + 2 * max_n * max_n + 3 * max_n + 2 * LANCZOS_K;
    float* s_TV      = smem + 2 * max_n * max_n + 3 * max_n + 2 * LANCZOS_K + LANCZOS_K * LANCZOS_K;
    float* s_scratch = smem + 2 * max_n * max_n + 3 * max_n + 2 * LANCZOS_K + 2 * LANCZOS_K * LANCZOS_K;

    // Load adjacency row into shared memory (only real nodes)
    const int adj_offset = b * max_n * max_n;
    if (tid < n_b) {
        for (int j = 0; j < n_b; j++)
            s_adj[tid * max_n + j] = g_adj[adj_offset + tid * max_n + j];
    }
    __syncthreads();

    // Get action (all threads read same values — block-uniform)
    int u = g_current_node[b];
    int v_act = g_nb_actions[b];
    int w_act = g_dests[b];

    float reward = 0.0f;
    bool is_keep = (v_act >= n_b);

    if (!is_keep) {
        // Validity check (block-uniform: all threads compute same result)
        bool valid = (v_act != w_act) && (u != w_act) &&
                     (v_act >= 0) && (v_act < n_b) &&
                     (w_act >= 0) && (w_act < n_b) &&
                     (u >= 0) && (u < n_b);
        if (valid) {
            valid = (s_adj[u * max_n + v_act] == 1.0f) &&
                    (s_adj[u * max_n + w_act] == 0.0f);
        }

        if (valid) {
            // Apply rewire in shared memory
            if (tid == 0) {
                s_adj[u * max_n + v_act] = 0.0f;
                s_adj[v_act * max_n + u] = 0.0f;
                s_adj[u * max_n + w_act] = 1.0f;
                s_adj[w_act * max_n + u] = 1.0f;
            }
            __syncthreads();

            // ============================================================
            // Phase 1: BFS connectivity check (thread 0, reuse s_vec/s_w)
            // Matches CPU is_connected(adj)
            // ============================================================
            int* bfs_visited = (int*)s_vec;
            int* bfs_queue   = (int*)s_w;
            if (tid == 0) {
                for (int i = 0; i < n_b; i++) bfs_visited[i] = 0;
                bfs_visited[0] = 1;
                bfs_queue[0] = 0;
                int front = 0, back = 1, count = 1;
                while (front < back) {
                    int node = bfs_queue[front++];
                    for (int j = 0; j < n_b; j++) {
                        if (s_adj[node * max_n + j] > 0.0f && !bfs_visited[j]) {
                            bfs_visited[j] = 1;
                            bfs_queue[back++] = j;
                            count++;
                        }
                    }
                }
                s_scratch[0] = (count == n_b) ? 1.0f : 0.0f;
            }
            __syncthreads();

            bool connected = (s_scratch[0] == 1.0f);

            if (!connected) {
                // Rollback rewire
                if (tid == 0) {
                    s_adj[u * max_n + v_act] = 1.0f;
                    s_adj[v_act * max_n + u] = 1.0f;
                    s_adj[u * max_n + w_act] = 0.0f;
                    s_adj[w_act * max_n + u] = 0.0f;
                }
                // reward stays 0, no write-back needed
            } else {
                // ========================================================
                // Phase 2a: Recompute degrees
                // Matches CPU: degrees[v] -= 1; degrees[w] += 1
                // (we recompute fully for simplicity, same result)
                // ========================================================
                if (tid < n_b) {
                    float deg = 0.0f;
                    for (int j = 0; j < n_b; j++)
                        deg += s_adj[tid * max_n + j];
                    s_deg[tid] = deg;
                }
                __syncthreads();

                // ========================================================
                // Phase 2b: Lanczos Fiedler with warm start
                // Matches CPU: lanczos_fiedler(adj, k=10, v_init=fiedler_approx)
                // ========================================================
                const int K = (n_b - 1 < LANCZOS_K) ? (n_b - 1) : LANCZOS_K;

                // Load previous Fiedler as warm start
                if (tid < n_b) s_vec[tid] = g_fiedler[b * max_n + tid];
                __syncthreads();

                // Center and normalize (matches CPU: v = v - mean; v = v / norm)
                if (tid == 0) {
                    float sum = 0.0f;
                    for (int i = 0; i < n_b; i++) sum += s_vec[i];
                    float mean = sum / (float)n_b;
                    for (int i = 0; i < n_b; i++) s_vec[i] -= mean;

                    float norm_sq = 0.0f;
                    for (int i = 0; i < n_b; i++) norm_sq += s_vec[i] * s_vec[i];
                    float norm = sqrtf(norm_sq);

                    if (norm < 1e-10f) {
                        // Fallback: deterministic init
                        for (int i = 0; i < n_b; i++)
                            s_vec[i] = (float)i - (float)(n_b - 1) / 2.0f;
                        sum = 0.0f;
                        for (int i = 0; i < n_b; i++) sum += s_vec[i];
                        mean = sum / (float)n_b;
                        for (int i = 0; i < n_b; i++) s_vec[i] -= mean;
                        norm_sq = 0.0f;
                        for (int i = 0; i < n_b; i++) norm_sq += s_vec[i] * s_vec[i];
                        norm = sqrtf(norm_sq);
                    }
                    for (int i = 0; i < n_b; i++) s_vec[i] /= (norm + 1e-10f);
                }
                __syncthreads();

                // V[:, 0] = normalized warm start (column-major in s_work)
                if (tid < n_b) s_work[0 * max_n + tid] = s_vec[tid];
                if (tid < LANCZOS_K) { s_alpha[tid] = 0.0f; s_beta[tid] = 0.0f; }
                __syncthreads();

                // Lanczos iteration (matches CPU lanczos_fiedler exactly)
                int actual_K = K;
                for (int j = 0; j < K; j++) {
                    // w = L @ V[:, j] = deg * V[:, j] - adj @ V[:, j]
                    if (tid < n_b) {
                        float val = s_deg[tid] * s_work[j * max_n + tid];
                        for (int k = 0; k < n_b; k++)
                            val -= s_adj[tid * max_n + k] * s_work[j * max_n + k];
                        s_w[tid] = val;
                    }
                    __syncthreads();

                    // alpha[j] = dot(V[:, j], w)
                    if (tid == 0) {
                        float dot = 0.0f;
                        for (int i = 0; i < n_b; i++)
                            dot += s_work[j * max_n + i] * s_w[i];
                        s_alpha[j] = dot;
                    }
                    __syncthreads();

                    // Three-term recurrence: w -= beta[j]*V[:, j-1] + alpha[j]*V[:, j]
                    if (tid < n_b) {
                        float wval = s_w[tid];
                        if (j > 0)
                            wval -= s_beta[j] * s_work[(j - 1) * max_n + tid];
                        wval -= s_alpha[j] * s_work[j * max_n + tid];
                        s_w[tid] = wval;
                    }
                    __syncthreads();

                    // Full reorthogonalization
                    for (int i = 0; i <= j; i++) {
                        if (tid == 0) {
                            float dot = 0.0f;
                            for (int k = 0; k < n_b; k++)
                                dot += s_w[k] * s_work[i * max_n + k];
                            s_scratch[2] = dot;
                        }
                        __syncthreads();
                        if (tid < n_b)
                            s_w[tid] -= s_scratch[2] * s_work[i * max_n + tid];
                        __syncthreads();
                    }

                    // Project out all-ones direction
                    if (tid == 0) {
                        float sum = 0.0f;
                        for (int i = 0; i < n_b; i++) sum += s_w[i];
                        s_scratch[2] = sum / (float)n_b;
                    }
                    __syncthreads();
                    if (tid < n_b) s_w[tid] -= s_scratch[2];
                    __syncthreads();

                    // beta_next = norm(w)
                    if (tid == 0) {
                        float norm_sq = 0.0f;
                        for (int i = 0; i < n_b; i++) norm_sq += s_w[i] * s_w[i];
                        s_scratch[2] = sqrtf(norm_sq);
                    }
                    __syncthreads();

                    float beta_next = s_scratch[2];
                    if (beta_next < 1e-12f) {
                        actual_K = j + 1;
                        break;  // Block-uniform: all threads break
                    }

                    // V[:, j+1] = w / beta_next
                    if (j + 1 < K) {
                        if (tid == 0) s_beta[j + 1] = beta_next;
                        if (tid < n_b)
                            s_work[(j + 1) * max_n + tid] = s_w[tid] / beta_next;
                    }
                    __syncthreads();
                }

                // Solve K×K tridiagonal eigenproblem (thread 0 only)
                if (tid == 0) {
                    // Build T from alpha, beta
                    for (int i = 0; i < actual_K; i++) {
                        for (int jj = 0; jj < actual_K; jj++)
                            s_T[i * LANCZOS_K + jj] = 0.0f;
                        s_T[i * LANCZOS_K + i] = s_alpha[i];
                    }
                    for (int jj = 0; jj < actual_K - 1; jj++) {
                        s_T[jj * LANCZOS_K + (jj + 1)] = s_beta[jj + 1];
                        s_T[(jj + 1) * LANCZOS_K + jj] = s_beta[jj + 1];
                    }

                    // TV = I (eigenvector matrix)
                    for (int i = 0; i < actual_K; i++)
                        for (int jj = 0; jj < actual_K; jj++)
                            s_TV[i * LANCZOS_K + jj] = (i == jj) ? 1.0f : 0.0f;

                    // Jacobi eigensolver on K×K T (5 sweeps sufficient)
                    for (int sweep = 0; sweep < 5; sweep++) {
                        for (int p = 0; p < actual_K - 1; p++) {
                            for (int q = p + 1; q < actual_K; q++) {
                                float Tpq = s_T[p * LANCZOS_K + q];
                                if (fabsf(Tpq) < 1e-12f) continue;
                                float Tpp = s_T[p * LANCZOS_K + p];
                                float Tqq = s_T[q * LANCZOS_K + q];
                                float tau = (Tqq - Tpp) / (2.0f * Tpq);
                                float t;
                                if (tau >= 0.0f)
                                    t = 1.0f / (tau + sqrtf(1.0f + tau * tau));
                                else
                                    t = -1.0f / (-tau + sqrtf(1.0f + tau * tau));
                                float c = 1.0f / sqrtf(1.0f + t * t);
                                float s = t * c;

                                for (int i = 0; i < actual_K; i++) {
                                    float Tip = s_T[i * LANCZOS_K + p];
                                    float Tiq = s_T[i * LANCZOS_K + q];
                                    s_T[i * LANCZOS_K + p] = c * Tip - s * Tiq;
                                    s_T[i * LANCZOS_K + q] = s * Tip + c * Tiq;
                                }
                                for (int i = 0; i < actual_K; i++) {
                                    float Tpi = s_T[p * LANCZOS_K + i];
                                    float Tqi = s_T[q * LANCZOS_K + i];
                                    s_T[p * LANCZOS_K + i] = c * Tpi - s * Tqi;
                                    s_T[q * LANCZOS_K + i] = s * Tpi + c * Tqi;
                                }
                                for (int i = 0; i < actual_K; i++) {
                                    float TVip = s_TV[i * LANCZOS_K + p];
                                    float TViq = s_TV[i * LANCZOS_K + q];
                                    s_TV[i * LANCZOS_K + p] = c * TVip - s * TViq;
                                    s_TV[i * LANCZOS_K + q] = s * TVip + c * TViq;
                                }
                            }
                        }
                    }

                    // Find 2nd smallest eigenvalue of T
                    // Matches CPU: idx = 1 if eig_vals[0] < 0.1*max(eig_vals[1], 1e-10) else 0
                    float min1 = FLT_MAX, min2 = FLT_MAX;
                    int min1_idx = 0, min2_idx = 0;
                    for (int i = 0; i < actual_K; i++) {
                        float ev = s_T[i * LANCZOS_K + i];
                        if (ev < min1) {
                            min2 = min1; min2_idx = min1_idx;
                            min1 = ev; min1_idx = i;
                        } else if (ev < min2) {
                            min2 = ev; min2_idx = i;
                        }
                    }
                    int ritz_idx;
                    if (actual_K > 1 && min1 < 0.1f * fmaxf(min2, 1e-10f))
                        ritz_idx = min2_idx;  // 2nd smallest = Fiedler
                    else
                        ritz_idx = min1_idx;  // smallest is already Fiedler

                    s_scratch[3] = __int_as_float(ritz_idx);
                    s_scratch[4] = __int_as_float(actual_K);
                }
                __syncthreads();

                // Reconstruct Fiedler: s_vec = V[:, :K] @ TV[:, ritz_idx]
                {
                    int ritz_idx = __float_as_int(s_scratch[3]);
                    int final_K  = __float_as_int(s_scratch[4]);
                    if (tid < n_b) {
                        float val = 0.0f;
                        for (int j = 0; j < final_K; j++)
                            val += s_work[j * max_n + tid] * s_TV[j * LANCZOS_K + ritz_idx];
                        s_vec[tid] = val;
                    }
                }
                __syncthreads();

                // Center, normalize, sign-align with warm start
                if (tid == 0) {
                    float sum = 0.0f;
                    for (int i = 0; i < n_b; i++) sum += s_vec[i];
                    float mean = sum / (float)n_b;
                    for (int i = 0; i < n_b; i++) s_vec[i] -= mean;

                    float norm_sq = 0.0f;
                    for (int i = 0; i < n_b; i++) norm_sq += s_vec[i] * s_vec[i];
                    float norm = sqrtf(norm_sq + 1e-10f);
                    for (int i = 0; i < n_b; i++) s_vec[i] /= norm;

                    // Sign: align with v_init (previous Fiedler from global memory)
                    // Matches CPU: if dot(v2, v_init) < 0: v2 = -v2
                    float dot = 0.0f;
                    for (int i = 0; i < n_b; i++)
                        dot += s_vec[i] * g_fiedler[b * max_n + i];
                    if (dot < 0.0f) {
                        for (int i = 0; i < n_b; i++) s_vec[i] = -s_vec[i];
                    }
                }
                __syncthreads();

                // ========================================================
                // Phase 2c: Exact λ₂ via Jacobi eigenvalues-only
                // Matches CPU: algebraic_connectivity(adj)
                //   = np.linalg.eigvalsh(D - A)[1]
                // SKIPPED in inference_mode (no reward needed, O(N²) only)
                // ========================================================

                if (!inference_mode) {
                    // Build Laplacian in s_work (overwrites Lanczos V)
                    if (tid < n_b) {
                        for (int j = 0; j < n_b; j++)
                            s_work[tid * max_n + j] = -s_adj[tid * max_n + j];
                        s_work[tid * max_n + tid] = s_deg[tid];
                    }
                    __syncthreads();

                    // Jacobi sweeps (eigenvalues only — no eigenvector V)
                    for (int sweep = 0; sweep < sweeps; sweep++) {
                        for (int p = 0; p < n_b - 1; p++) {
                            for (int q = p + 1; q < n_b; q++) {
                                if (tid == 0) {
                                    float Lpq = s_work[p * max_n + q];
                                    if (fabsf(Lpq) > 1e-12f) {
                                        float Lpp = s_work[p * max_n + p];
                                        float Lqq = s_work[q * max_n + q];
                                        float tau = (Lqq - Lpp) / (2.0f * Lpq);
                                        float t;
                                        if (tau >= 0.0f)
                                            t = 1.0f / (tau + sqrtf(1.0f + tau * tau));
                                        else
                                            t = -1.0f / (-tau + sqrtf(1.0f + tau * tau));
                                        float c = 1.0f / sqrtf(1.0f + t * t);
                                        float ss = t * c;
                                        s_scratch[0] = c;
                                        s_scratch[1] = ss;
                                    } else {
                                        s_scratch[0] = 1.0f;
                                        s_scratch[1] = 0.0f;
                                    }
                                }
                                __syncthreads();

                                float c = s_scratch[0];
                                float ss = s_scratch[1];

                                if (ss != 0.0f) {
                                    if (tid < n_b) {
                                        float Lip = s_work[tid * max_n + p];
                                        float Liq = s_work[tid * max_n + q];
                                        s_work[tid * max_n + p] = c * Lip - ss * Liq;
                                        s_work[tid * max_n + q] = ss * Lip + c * Liq;
                                    }
                                    __syncthreads();
                                    if (tid < n_b) {
                                        float Lpi = s_work[p * max_n + tid];
                                        float Lqi = s_work[q * max_n + tid];
                                        s_work[p * max_n + tid] = c * Lpi - ss * Lqi;
                                        s_work[q * max_n + tid] = ss * Lpi + c * Lqi;
                                    }
                                    __syncthreads();
                                } else {
                                    __syncthreads();
                                    __syncthreads();
                                }
                            }
                        }
                    }

                    // Extract λ₂ = 2nd smallest diagonal entry
                    if (tid == 0) {
                        float min1 = FLT_MAX, min2 = FLT_MAX;
                        for (int i = 0; i < n_b; i++) {
                            float ev = s_work[i * max_n + i];
                            if (ev < min1) { min2 = min1; min1 = ev; }
                            else if (ev < min2) { min2 = ev; }
                        }
                        float new_lambda2 = fmaxf(min2, 0.0f);
                        float old_lambda2 = g_lambda2[b];
                        reward = new_lambda2 - old_lambda2;
                        g_lambda2[b] = new_lambda2;
                    }
                }

                // Count rewire (both training and inference)
                if (tid == 0) g_total_rewires[b] += 1;

                // Write back adj, degrees, fiedler to global memory
                if (tid < n_b) {
                    for (int j = 0; j < n_b; j++)
                        g_adj[adj_offset + tid * max_n + j] = s_adj[tid * max_n + j];
                    g_degrees[b * max_n + tid] = s_deg[tid];
                    g_fiedler[b * max_n + tid] = s_vec[tid];
                }
            }
        }
    }

    // Advancement (thread 0 only)
    if (tid == 0) {
        g_rewards[b] = reward;
        int node = g_current_node[b] + 1;
        int round = g_current_round[b];
        if (node >= n_b) {
            round += 1;
            node = 0;
        }
        if (round >= g_phi[b]) {
            g_done[b] = 1;
        }
        g_current_node[b] = node;
        g_current_round[b] = round;
    }
}


/**
 * Kernel B: fused_features_and_masks_kernel
 *
 * Per-graph n from g_n_vals. Threads with tid >= n_b write zeros.
 */
extern "C" __global__ void fused_features_and_masks_kernel(
    const float* __restrict__ g_adj,
    const float* __restrict__ g_degrees,
    const float* __restrict__ g_fiedler,
    const int* __restrict__ g_current_node,
    float* __restrict__ g_features,
    float* __restrict__ g_nb_mask,
    float* __restrict__ g_dest_mask,
    const int* __restrict__ g_n_vals,
    const int max_n
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_b = g_n_vals[b];

    extern __shared__ float smem[];
    float* s_deg = smem;

    // Load degrees (padding = 0)
    if (tid < n_b)
        s_deg[tid] = g_degrees[b * max_n + tid];
    else
        s_deg[tid] = 0.0f;
    __syncthreads();

    int feat_offset = b * max_n * 4;
    int u = g_current_node[b];

    if (tid < n_b) {
        float deg_tid = s_deg[tid];
        float feat0 = deg_tid / (float)(n_b - 1);

        // RWLP
        float rwlp = 0.0f;
        float inv_deg = (deg_tid > 0.0f) ? (1.0f / deg_tid) : 0.0f;
        const int adj_offset = b * max_n * max_n;
        for (int j = 0; j < n_b; j++) {
            float aij = g_adj[adj_offset + tid * max_n + j];
            if (aij > 0.0f && s_deg[j] > 0.0f)
                rwlp += aij / s_deg[j];
        }
        rwlp *= inv_deg;
        float feat1 = logf(rwlp * n_b + 1e-10f);
        float feat2 = (tid == u) ? 1.0f : 0.0f;
        float feat3 = g_fiedler[b * max_n + tid];

        g_features[feat_offset + tid * 4 + 0] = feat0;
        g_features[feat_offset + tid * 4 + 1] = feat1;
        g_features[feat_offset + tid * 4 + 2] = feat2;
        g_features[feat_offset + tid * 4 + 3] = feat3;

        float a_u_tid = g_adj[b * max_n * max_n + u * max_n + tid];
        g_nb_mask[b * max_n + tid] = (a_u_tid > 0.0f) ? 1.0f : 0.0f;
        g_dest_mask[b * max_n + tid] = (a_u_tid == 0.0f && tid != u) ? 1.0f : 0.0f;
    } else {
        // Padding: zero features and masks
        g_features[feat_offset + tid * 4 + 0] = 0.0f;
        g_features[feat_offset + tid * 4 + 1] = 0.0f;
        g_features[feat_offset + tid * 4 + 2] = 0.0f;
        g_features[feat_offset + tid * 4 + 3] = 0.0f;
        g_nb_mask[b * max_n + tid] = 0.0f;
        g_dest_mask[b * max_n + tid] = 0.0f;
    }
}


/**
 * Kernel C: batched_jacobi_eigh_kernel
 *
 * Unchanged — operates on full (B, N, N) matrices.
 * Python pads Laplacian diagonal with 1e6 for unused nodes so their
 * eigenvalues sort to the end, and lambda2 = sorted[1] is correct.
 */
extern "C" __global__ void batched_jacobi_eigh_kernel(
    const float* __restrict__ g_L,
    float* __restrict__ g_evals,
    float* __restrict__ g_evecs,
    const int N,
    const int sweeps
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;

    if (tid >= N) return;

    extern __shared__ float smem[];
    float* s_L = smem;
    float* s_V = smem + N * N;
    float* s_scratch = smem + 2 * N * N;

    const int offset = b * N * N;
    for (int j = 0; j < N; j++)
        s_L[tid * N + j] = g_L[offset + tid * N + j];
    for (int j = 0; j < N; j++)
        s_V[tid * N + j] = (tid == j) ? 1.0f : 0.0f;
    __syncthreads();

    for (int sweep = 0; sweep < sweeps; sweep++) {
        for (int p = 0; p < N - 1; p++) {
            for (int q = p + 1; q < N; q++) {
                if (tid == 0) {
                    float Lpq = s_L[p * N + q];
                    if (fabsf(Lpq) > 1e-12f) {
                        float Lpp = s_L[p * N + p];
                        float Lqq = s_L[q * N + q];
                        float tau = (Lqq - Lpp) / (2.0f * Lpq);
                        float t;
                        if (tau >= 0.0f)
                            t = 1.0f / (tau + sqrtf(1.0f + tau * tau));
                        else
                            t = -1.0f / (-tau + sqrtf(1.0f + tau * tau));
                        float c = 1.0f / sqrtf(1.0f + t * t);
                        float s = t * c;
                        s_scratch[0] = c;
                        s_scratch[1] = s;
                    } else {
                        s_scratch[0] = 1.0f;
                        s_scratch[1] = 0.0f;
                    }
                }
                __syncthreads();

                float c = s_scratch[0];
                float s = s_scratch[1];

                if (s != 0.0f) {
                    float Lip = s_L[tid * N + p];
                    float Liq = s_L[tid * N + q];
                    s_L[tid * N + p] = c * Lip - s * Liq;
                    s_L[tid * N + q] = s * Lip + c * Liq;
                    __syncthreads();
                    float Lpi = s_L[p * N + tid];
                    float Lqi = s_L[q * N + tid];
                    s_L[p * N + tid] = c * Lpi - s * Lqi;
                    s_L[q * N + tid] = s * Lpi + c * Lqi;
                    __syncthreads();
                    float Vip = s_V[tid * N + p];
                    float Viq = s_V[tid * N + q];
                    s_V[tid * N + p] = c * Vip - s * Viq;
                    s_V[tid * N + q] = s * Vip + c * Viq;
                    __syncthreads();
                } else {
                    __syncthreads();
                    __syncthreads();
                    __syncthreads();
                }
            }
        }
    }

    g_evals[b * N + tid] = s_L[tid * N + tid];
    for (int j = 0; j < N; j++)
        g_evecs[offset + tid * N + j] = s_V[tid * N + j];
}


/**
 * Kernel D: build_random_graphs_kernel
 *
 * Per-graph n from g_n_vals. Only builds edges among nodes 0..n_b-1.
 * Pads rest with zeros.
 */
extern "C" __global__ void build_random_graphs_kernel(
    float* __restrict__ g_adj,
    float* __restrict__ g_degrees,
    const int* __restrict__ g_m_vals,
    const int* __restrict__ g_n_vals,
    const int max_n,
    const unsigned long long seed
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_b = g_n_vals[b];

    const int adj_offset = b * max_n * max_n;

    // Zero adj matrix (all threads, full max_n row)
    for (int j = 0; j < max_n; j++)
        g_adj[adj_offset + tid * max_n + j] = 0.0f;
    __syncthreads();

    // Thread 0: build random spanning tree + random edges
    if (tid == 0) {
        curandState_t state;
        curand_init(seed, b, 0, &state);

        int m = g_m_vals[b];

        extern __shared__ float smem[];
        int* perm = (int*)smem;

        for (int i = 0; i < n_b; i++) perm[i] = i;

        // Fisher-Yates shuffle
        for (int i = n_b - 1; i > 0; i--) {
            int j = curand(&state) % (i + 1);
            int tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
        }

        // Spanning tree
        int edges_added = 0;
        for (int i = 1; i < n_b; i++) {
            int new_node = perm[i];
            int parent = perm[curand(&state) % i];
            g_adj[adj_offset + new_node * max_n + parent] = 1.0f;
            g_adj[adj_offset + parent * max_n + new_node] = 1.0f;
            edges_added++;
        }

        // Remaining random edges
        int remaining = m - edges_added;
        int attempts = 0;
        int max_attempts = remaining * 10 + n_b * n_b;
        while (remaining > 0 && attempts < max_attempts) {
            int i = curand(&state) % n_b;
            int j = curand(&state) % n_b;
            if (i != j && g_adj[adj_offset + i * max_n + j] == 0.0f) {
                g_adj[adj_offset + i * max_n + j] = 1.0f;
                g_adj[adj_offset + j * max_n + i] = 1.0f;
                remaining--;
            }
            attempts++;
        }
    }
    __syncthreads();

    // Compute degrees (real nodes sum n_b cols, padding nodes get 0)
    if (tid < n_b) {
        float deg = 0.0f;
        for (int j = 0; j < n_b; j++)
            deg += g_adj[adj_offset + tid * max_n + j];
        g_degrees[b * max_n + tid] = deg;
    } else {
        g_degrees[b * max_n + tid] = 0.0f;
    }
}
