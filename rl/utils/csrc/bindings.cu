/**
 * PyTorch C++ bindings for custom CUDA kernels.
 *
 * Supports variable n per graph — fused_step, fused_features_and_masks,
 * and build_random_graphs accept per-graph n_vals.
 * batched_jacobi_eigh operates on full padded matrices (padding trick).
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel forward declarations (defined in kernels.cu)
extern "C" __global__ void fused_step_kernel(
    float*, float*, float*, float*,
    const int*, const int*, int*, int*,
    const int*, int*, int*, float*,
    const int*, const int, const int, const int);

extern "C" __global__ void fused_features_and_masks_kernel(
    const float*, const float*, const float*, const int*,
    float*, float*, float*,
    const int*, const int);

extern "C" __global__ void batched_jacobi_eigh_kernel(
    const float*, float*, float*, const int, const int);

extern "C" __global__ void build_random_graphs_kernel(
    float*, float*, const int*, const int*, const int,
    const unsigned long long);


torch::Tensor fused_step(
    torch::Tensor adj,
    torch::Tensor degrees,
    torch::Tensor lambda2,
    torch::Tensor fiedler,
    torch::Tensor neighbor_actions,
    torch::Tensor destinations,
    torch::Tensor current_node,
    torch::Tensor current_round,
    torch::Tensor phi,
    torch::Tensor done,
    torch::Tensor total_rewires,
    torch::Tensor n_vals,
    int max_n,
    int sweeps,
    int inference_mode
) {
    TORCH_CHECK(adj.is_cuda(), "adj must be on CUDA");
    int B = adj.size(0);

    auto rewards = torch::zeros({B}, adj.options());

    // Shared memory: adj[N²] + work[N²] + deg[N] + vec[N] + w[N]
    //   + alpha[K] + beta[K] + T[K²] + TV[K²] + scratch[8]
    // where K = LANCZOS_K = 10
    int K = 10;
    int smem_bytes = (2 * max_n * max_n + 3 * max_n + 2 * K + 2 * K * K + 8) * sizeof(float);

    fused_step_kernel<<<B, max_n, smem_bytes>>>(
        adj.data_ptr<float>(),
        degrees.data_ptr<float>(),
        lambda2.data_ptr<float>(),
        fiedler.data_ptr<float>(),
        neighbor_actions.data_ptr<int>(),
        destinations.data_ptr<int>(),
        current_node.data_ptr<int>(),
        current_round.data_ptr<int>(),
        phi.data_ptr<int>(),
        done.data_ptr<int>(),
        total_rewires.data_ptr<int>(),
        rewards.data_ptr<float>(),
        n_vals.data_ptr<int>(),
        max_n,
        sweeps,
        inference_mode
    );

    return rewards;
}


std::vector<torch::Tensor> fused_features_and_masks(
    torch::Tensor adj,
    torch::Tensor degrees,
    torch::Tensor fiedler,
    torch::Tensor current_node,
    torch::Tensor n_vals,
    int max_n
) {
    TORCH_CHECK(adj.is_cuda(), "adj must be on CUDA");
    int B = adj.size(0);

    auto features = torch::zeros({B, max_n, 4}, adj.options());
    auto nb_mask = torch::zeros({B, max_n}, adj.options());
    auto dest_mask = torch::zeros({B, max_n}, adj.options());

    int smem_bytes = max_n * sizeof(float);

    fused_features_and_masks_kernel<<<B, max_n, smem_bytes>>>(
        adj.data_ptr<float>(),
        degrees.data_ptr<float>(),
        fiedler.data_ptr<float>(),
        current_node.data_ptr<int>(),
        features.data_ptr<float>(),
        nb_mask.data_ptr<float>(),
        dest_mask.data_ptr<float>(),
        n_vals.data_ptr<int>(),
        max_n
    );

    return {features, nb_mask, dest_mask};
}


std::vector<torch::Tensor> batched_jacobi_eigh(
    torch::Tensor L,
    int sweeps
) {
    TORCH_CHECK(L.is_cuda(), "L must be on CUDA");
    int B = L.size(0);
    int N = L.size(1);

    auto evals = torch::zeros({B, N}, L.options());
    auto evecs = torch::zeros({B, N, N}, L.options());

    int smem_bytes = (2 * N * N + 4) * sizeof(float);

    batched_jacobi_eigh_kernel<<<B, N, smem_bytes>>>(
        L.data_ptr<float>(),
        evals.data_ptr<float>(),
        evecs.data_ptr<float>(),
        N,
        sweeps
    );

    return {evals, evecs};
}


std::vector<torch::Tensor> build_random_graphs(
    torch::Tensor m_vals,
    torch::Tensor n_vals,
    int B,
    int max_n,
    int64_t seed
) {
    auto options_f = torch::TensorOptions().dtype(torch::kFloat32).device(m_vals.device());
    auto adj = torch::zeros({B, max_n, max_n}, options_f);
    auto degrees = torch::zeros({B, max_n}, options_f);

    int smem_bytes = max_n * sizeof(int);

    build_random_graphs_kernel<<<B, max_n, smem_bytes>>>(
        adj.data_ptr<float>(),
        degrees.data_ptr<float>(),
        m_vals.data_ptr<int>(),
        n_vals.data_ptr<int>(),
        max_n,
        (unsigned long long)seed
    );

    return {adj, degrees};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_step", &fused_step,
          "Fused step kernel: rewire + eigsolve + reward (CUDA)");
    m.def("fused_features_and_masks", &fused_features_and_masks,
          "Fused features + masks computation (CUDA)");
    m.def("batched_jacobi_eigh", &batched_jacobi_eigh,
          "Batched Jacobi eigendecomposition (CUDA)");
    m.def("build_random_graphs", &build_random_graphs,
          "Build random graphs on GPU (CUDA)");
}
