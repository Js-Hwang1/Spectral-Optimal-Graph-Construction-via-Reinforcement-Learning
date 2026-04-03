/*
 * enumerate_g6file.cu -- GPU eigensolve from pre-generated graph6 files.
 *
 * Reads a .g6 file, batches Laplacians to GPU, finds max lambda2.
 * Optimized: pinned memory, CUDA streams, large batches.
 *
 * Usage: crl_enum_g6 <n> <g6_file> <output_csv>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#define BATCH_SIZE 262144
#define MAX_N 16

static double wall_time(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static inline void parse_g6(const char *s, double *L, int n) {
    memset(L, 0, (size_t)(n * n) * sizeof(double));
    int bit_idx = 0, data_idx = 1;
    for (int j = 1; j < n; j++)
        for (int i = 0; i < j; i++) {
            int bv = (unsigned char)s[data_idx + bit_idx / 6] - 63;
            if ((bv >> (5 - bit_idx % 6)) & 1) {
                L[i + j*n] = -1.0; L[j + i*n] = -1.0;
                L[i + i*n] += 1.0; L[j + j*n] += 1.0;
            }
            bit_idx++;
        }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("Usage: %s <n> <input.g6> <output.csv>\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    const char *g6_path = argv[2];
    const char *csv_path = argv[3];

    if (n > MAX_N) { fprintf(stderr, "n=%d > MAX_N\n", n); return 1; }

    FILE *g6 = fopen(g6_path, "r");
    if (!g6) { fprintf(stderr, "Cannot open %s\n", g6_path); return 1; }

    printf("GPU g6 solver: n=%d, file=%s\n", n, g6_path);

    /* Init cuSOLVER */
    cusolverDnHandle_t handle; cusolverDnCreate(&handle);
    syevjInfo_t params; cusolverDnCreateSyevjInfo(&params);
    cusolverDnXsyevjSetTolerance(params, 1e-10);
    cusolverDnXsyevjSetMaxSweeps(params, 100);
    cudaStream_t stream; cudaStreamCreate(&stream);

    int lwork = 0;
    cusolverDnDsyevjBatched_bufferSize(handle, CUSOLVER_EIG_MODE_NOVECTOR,
                                        CUBLAS_FILL_MODE_LOWER,
                                        n, NULL, n, NULL, &lwork, params, BATCH_SIZE);

    size_t mat_bytes = (size_t)BATCH_SIZE * n * n * sizeof(double);
    size_t eig_bytes = (size_t)BATCH_SIZE * n * sizeof(double);

    double *d_A, *d_W, *d_work; int *d_info;
    cudaMalloc(&d_A, mat_bytes);
    cudaMalloc(&d_W, eig_bytes);
    cudaMalloc(&d_work, (size_t)lwork * sizeof(double));
    cudaMalloc(&d_info, (size_t)BATCH_SIZE * sizeof(int));

    double *h_A, *h_W;
    cudaMallocHost(&h_A, mat_bytes);
    cudaMallocHost(&h_W, eig_bytes);

    double best_l2 = 0;
    long long total = 0;
    int bc = 0;
    size_t mat_sz = (size_t)n * n;
    char line[256];

    double t0 = wall_time();
    double t_last = t0;

    while (fgets(line, sizeof(line), g6)) {
        int len = (int)strlen(line);
        if (len > 0 && line[len-1] == '\n') line[len-1] = '\0';
        if (len < 2) continue;

        parse_g6(line, h_A + bc * mat_sz, n);
        bc++; total++;

        if (bc >= BATCH_SIZE) {
            cudaMemcpyAsync(d_A, h_A, bc * mat_sz * sizeof(double), cudaMemcpyHostToDevice, stream);
            cusolverDnSetStream(handle, stream);
            cusolverDnDsyevjBatched(handle, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER,
                                     n, d_A, n, d_W, d_work, lwork, d_info, params, bc);
            cudaMemcpyAsync(h_W, d_W, (size_t)bc * n * sizeof(double), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            for (int i = 0; i < bc; i++) {
                double l2 = h_W[i * n + 1];
                if (l2 > best_l2) best_l2 = l2;
            }
            bc = 0;

            /* Progress every 10M graphs */
            if (total % (10000000LL / BATCH_SIZE * BATCH_SIZE) < BATCH_SIZE) {
                double now = wall_time();
                printf("  %lld graphs, best=%.6f, %.0f g/s\n",
                       total, best_l2, (double)total / (now - t0));
            }
        }
    }

    /* Remainder */
    if (bc > 0) {
        cudaMemcpyAsync(d_A, h_A, bc * mat_sz * sizeof(double), cudaMemcpyHostToDevice, stream);
        cusolverDnSetStream(handle, stream);
        cusolverDnDsyevjBatched(handle, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER,
                                 n, d_A, n, d_W, d_work, lwork, d_info, params, bc);
        cudaMemcpyAsync(h_W, d_W, (size_t)bc * n * sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        for (int i = 0; i < bc; i++) {
            double l2 = h_W[i * n + 1];
            if (l2 > best_l2) best_l2 = l2;
        }
    }

    fclose(g6);
    double elapsed = wall_time() - t0;

    /* Write CSV */
    FILE *out = fopen(csv_path, "w");
    if (out) {
        fprintf(out, "m,score\n");
        /* Extract m from filename or just write the result */
        fprintf(out, "%.10f\n", best_l2);
        fclose(out);
    }

    printf("Result: lambda2=%.10f (%lld graphs in %.1fs, %.0f g/s)\n",
           best_l2, total, elapsed, (double)total / elapsed);

    cudaFreeHost(h_A); cudaFreeHost(h_W);
    cudaFree(d_A); cudaFree(d_W); cudaFree(d_work); cudaFree(d_info);
    cudaStreamDestroy(stream);
    cusolverDnDestroySyevjInfo(params);
    cusolverDnDestroy(handle);
    return 0;
}
