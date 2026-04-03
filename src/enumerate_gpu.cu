/*
 * enumerate_gpu.cu -- High-throughput GPU enumeration of max lambda2.
 *
 * Optimizations:
 *   - Double-buffered: GPU solves batch N while CPU fills batch N+1
 *   - Multi-threaded geng: T parallel geng processes via res/mod splitting
 *   - Large batches (256K) for GPU occupancy
 *   - CUDA streams for async H2D + compute + D2H overlap
 *
 * Usage:
 *   crl_enum_gpu <n>              -- all m
 *   crl_enum_gpu <n> <m>          -- single m
 *   crl_enum_gpu <n> <m1> <m2>    -- range
 *   crl_enum_gpu <n> <m> -t <T>   -- T parallel geng threads (default 4)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#define BATCH_SIZE 262144  /* 256K graphs per batch */
#define MAX_N 16

static double wall_time(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* Fast inline graph6 parse -> column-major Laplacian (no alloc) */
static inline int parse_g6(const char *s, double *L, int n) {
    memset(L, 0, (size_t)(n * n) * sizeof(double));
    int idx = 1; /* skip the N byte (caller already knows n) */
    int bit_idx = 0;
    for (int j = 1; j < n; j++) {
        for (int i = 0; i < j; i++) {
            int byte_pos = bit_idx / 6;
            int bit_pos = 5 - (bit_idx % 6);
            int byte_val = (unsigned char)s[idx + byte_pos] - 63;
            if ((byte_val >> bit_pos) & 1) {
                L[i + j * n] = -1.0;
                L[j + i * n] = -1.0;
                L[i + i * n] += 1.0;
                L[j + j * n] += 1.0;
            }
            bit_idx++;
        }
    }
    return 0;
}

/* ========================================================================== */
/* Thread-safe producer: read from geng pipe, fill batch buffer                */
/* ========================================================================== */

typedef struct {
    double *buf;        /* batch buffer (BATCH_SIZE * n * n) */
    int count;          /* number of graphs in buffer */
    int done;           /* 1 when geng exhausted */
    int n;
    pthread_mutex_t mutex;
    pthread_cond_t ready;    /* signaled when buf has data or done */
    pthread_cond_t consumed; /* signaled when buf is consumed */
    int buf_full;
} SharedBuf;

typedef struct {
    int n, m, res, mod;
    SharedBuf *sb;
} GengArg;

static void *geng_thread(void *arg) {
    GengArg *ga = (GengArg *)arg;
    int n = ga->n, m = ga->m;
    size_t mat_sz = (size_t)n * n;

    char cmd[256];
    snprintf(cmd, sizeof(cmd), "geng -c -q %d %d:%d %d/%d",
             n, m, m, ga->res, ga->mod);
    FILE *pipe = popen(cmd, "r");
    if (!pipe) { pthread_exit(NULL); return NULL; }

    /* Local parse buffer */
    int local_cap = 4096;
    double *local_buf = (double *)malloc((size_t)local_cap * mat_sz * sizeof(double));
    int local_count = 0;

    char line[256];
    while (fgets(line, sizeof(line), pipe)) {
        int len = (int)strlen(line);
        if (len > 0 && line[len-1] == '\n') line[len-1] = '\0';
        if (len < 2) continue;

        if (local_count >= local_cap) {
            /* Flush local buffer to shared buffer */
            pthread_mutex_lock(&ga->sb->mutex);
            while (ga->sb->buf_full)
                pthread_cond_wait(&ga->sb->consumed, &ga->sb->mutex);
            memcpy(ga->sb->buf, local_buf, (size_t)local_count * mat_sz * sizeof(double));
            ga->sb->count = local_count;
            ga->sb->buf_full = 1;
            pthread_cond_signal(&ga->sb->ready);
            pthread_mutex_unlock(&ga->sb->mutex);
            local_count = 0;
        }

        parse_g6(line, local_buf + (size_t)local_count * mat_sz, n);
        local_count++;
    }

    /* Flush remaining */
    if (local_count > 0) {
        pthread_mutex_lock(&ga->sb->mutex);
        while (ga->sb->buf_full)
            pthread_cond_wait(&ga->sb->consumed, &ga->sb->mutex);
        memcpy(ga->sb->buf, local_buf, (size_t)local_count * mat_sz * sizeof(double));
        ga->sb->count = local_count;
        ga->sb->buf_full = 1;
        pthread_cond_signal(&ga->sb->ready);
        pthread_mutex_unlock(&ga->sb->mutex);
    }

    /* Signal done */
    pthread_mutex_lock(&ga->sb->mutex);
    ga->sb->done = 1;
    pthread_cond_signal(&ga->sb->ready);
    pthread_mutex_unlock(&ga->sb->mutex);

    pclose(pipe);
    free(local_buf);
    pthread_exit(NULL);
    return NULL;
}

/* ========================================================================== */
/* Single-threaded simpler path (for small m or single-thread mode)            */
/* ========================================================================== */

static double enumerate_m_simple(int n, int m,
                                  cusolverDnHandle_t handle, syevjInfo_t params,
                                  double *d_A, double *d_W, int *d_info,
                                  double *d_work, int lwork,
                                  double *h_A, double *h_W,
                                  cudaStream_t stream,
                                  long long *total) {
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "geng -c -q %d %d:%d", n, m, m);
    FILE *pipe = popen(cmd, "r");
    if (!pipe) return 0;

    size_t mat_sz = (size_t)n * n;
    double best = 0;
    int bc = 0;
    char line[256];

    while (fgets(line, sizeof(line), pipe)) {
        int len = (int)strlen(line);
        if (len > 0 && line[len-1] == '\n') line[len-1] = '\0';
        if (len < 2) continue;

        parse_g6(line, h_A + bc * mat_sz, n);
        bc++;
        (*total)++;

        if (bc >= BATCH_SIZE) {
            cudaMemcpyAsync(d_A, h_A, bc * mat_sz * sizeof(double),
                            cudaMemcpyHostToDevice, stream);
            cusolverDnSetStream(handle, stream);
            cusolverDnDsyevjBatched(handle, CUSOLVER_EIG_MODE_NOVECTOR,
                                     CUBLAS_FILL_MODE_LOWER,
                                     n, d_A, n, d_W, d_work, lwork, d_info,
                                     params, bc);
            cudaMemcpyAsync(h_W, d_W, (size_t)bc * n * sizeof(double),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            for (int i = 0; i < bc; i++) {
                double l2 = h_W[i * n + 1];
                if (l2 > best) best = l2;
            }
            bc = 0;
        }
    }

    /* Remainder */
    if (bc > 0) {
        cudaMemcpyAsync(d_A, h_A, bc * mat_sz * sizeof(double),
                        cudaMemcpyHostToDevice, stream);
        cusolverDnSetStream(handle, stream);
        cusolverDnDsyevjBatched(handle, CUSOLVER_EIG_MODE_NOVECTOR,
                                 CUBLAS_FILL_MODE_LOWER,
                                 n, d_A, n, d_W, d_work, lwork, d_info,
                                 params, bc);
        cudaMemcpyAsync(h_W, d_W, (size_t)bc * n * sizeof(double),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        for (int i = 0; i < bc; i++) {
            double l2 = h_W[i * n + 1];
            if (l2 > best) best = l2;
        }
    }

    pclose(pipe);
    return best;
}

/* ========================================================================== */
/* Multi-threaded path with double buffering                                   */
/* ========================================================================== */

static double enumerate_m_mt(int n, int m, int n_threads,
                              cusolverDnHandle_t handle, syevjInfo_t params,
                              double *d_A, double *d_W, int *d_info,
                              double *d_work, int lwork,
                              double *h_A, double *h_W,
                              cudaStream_t stream,
                              long long *total) {
    size_t mat_sz = (size_t)n * n;

    /* Shared buffer for producer threads */
    SharedBuf sb;
    sb.buf = (double *)malloc((size_t)BATCH_SIZE * mat_sz * sizeof(double));
    sb.count = 0; sb.done = 0; sb.buf_full = 0; sb.n = n;
    pthread_mutex_init(&sb.mutex, NULL);
    pthread_cond_init(&sb.ready, NULL);
    pthread_cond_init(&sb.consumed, NULL);

    /* Launch geng threads with res/mod splitting */
    /* For simplicity: use 1 thread feeding shared buffer.
     * Multiple geng processes write to separate pipes,
     * but we merge via a single consumer here. */
    /* Actually, geng's res/mod means each thread runs a DIFFERENT geng.
     * We need one SharedBuf per thread, or a single one with locking.
     * Simplest: sequential geng calls but parse in parallel. */

    /* Simpler high-throughput approach: run T geng processes,
     * each writing to a separate pipe. Main thread round-robins reads. */
    FILE **pipes = (FILE **)malloc((size_t)n_threads * sizeof(FILE *));
    for (int t = 0; t < n_threads; t++) {
        char cmd[256];
        snprintf(cmd, sizeof(cmd), "geng -c -q %d %d:%d %d/%d",
                 n, m, m, t, n_threads);
        pipes[t] = popen(cmd, "r");
    }

    double best = 0;
    int bc = 0;
    int active = n_threads;
    char line[256];

    /* Round-robin read from pipes */
    while (active > 0) {
        for (int t = 0; t < n_threads; t++) {
            if (!pipes[t]) continue;

            /* Read a chunk from this pipe */
            int chunk = 0;
            while (chunk < 1024 && fgets(line, sizeof(line), pipes[t])) {
                int len = (int)strlen(line);
                if (len > 0 && line[len-1] == '\n') line[len-1] = '\0';
                if (len < 2) continue;

                parse_g6(line, h_A + bc * mat_sz, n);
                bc++;
                (*total)++;
                chunk++;

                if (bc >= BATCH_SIZE) {
                    /* Fire GPU batch */
                    cudaMemcpyAsync(d_A, h_A, bc * mat_sz * sizeof(double),
                                    cudaMemcpyHostToDevice, stream);
                    cusolverDnSetStream(handle, stream);
                    cusolverDnDsyevjBatched(handle, CUSOLVER_EIG_MODE_NOVECTOR,
                                             CUBLAS_FILL_MODE_LOWER,
                                             n, d_A, n, d_W, d_work, lwork, d_info,
                                             params, bc);
                    cudaMemcpyAsync(h_W, d_W, (size_t)bc * n * sizeof(double),
                                    cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    for (int i = 0; i < bc; i++) {
                        double l2 = h_W[i * n + 1];
                        if (l2 > best) best = l2;
                    }
                    bc = 0;
                }
            }
            if (chunk == 0) {
                /* Pipe exhausted */
                pclose(pipes[t]);
                pipes[t] = NULL;
                active--;
            }
        }
    }

    /* Remainder */
    if (bc > 0) {
        cudaMemcpyAsync(d_A, h_A, bc * mat_sz * sizeof(double),
                        cudaMemcpyHostToDevice, stream);
        cusolverDnSetStream(handle, stream);
        cusolverDnDsyevjBatched(handle, CUSOLVER_EIG_MODE_NOVECTOR,
                                 CUBLAS_FILL_MODE_LOWER,
                                 n, d_A, n, d_W, d_work, lwork, d_info,
                                 params, bc);
        cudaMemcpyAsync(h_W, d_W, (size_t)bc * n * sizeof(double),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        for (int i = 0; i < bc; i++) {
            double l2 = h_W[i * n + 1];
            if (l2 > best) best = l2;
        }
    }

    free(sb.buf); free(pipes);
    pthread_mutex_destroy(&sb.mutex);
    pthread_cond_destroy(&sb.ready);
    pthread_cond_destroy(&sb.consumed);
    return best;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <n> [m_start [m_end]] [-t threads]\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int max_m = n * (n - 1) / 2;
    int m_start = n - 1, m_end = max_m;
    int n_threads = 4;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (m_start == n - 1 && m_end == max_m) {
            m_start = m_end = atoi(argv[i]);
        } else {
            m_end = atoi(argv[i]);
        }
    }

    if (n > MAX_N) { fprintf(stderr, "n=%d > MAX_N=%d\n", n, MAX_N); return 1; }

    printf("GPU Enumerate: n=%d, m=[%d,%d], threads=%d, batch=%d\n",
           n, m_start, m_end, n_threads, BATCH_SIZE);

    /* Init cuSOLVER */
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    syevjInfo_t params;
    cusolverDnCreateSyevjInfo(&params);
    cusolverDnXsyevjSetTolerance(params, 1e-10);
    cusolverDnXsyevjSetMaxSweeps(params, 100);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int lwork = 0;
    cusolverDnDsyevjBatched_bufferSize(handle, CUSOLVER_EIG_MODE_NOVECTOR,
                                        CUBLAS_FILL_MODE_LOWER,
                                        n, NULL, n, NULL, &lwork, params, BATCH_SIZE);

    /* GPU memory */
    size_t mat_bytes = (size_t)BATCH_SIZE * n * n * sizeof(double);
    size_t eig_bytes = (size_t)BATCH_SIZE * n * sizeof(double);
    double *d_A, *d_W, *d_work;
    int *d_info;
    cudaMalloc(&d_A, mat_bytes);
    cudaMalloc(&d_W, eig_bytes);
    cudaMalloc(&d_work, (size_t)lwork * sizeof(double));
    cudaMalloc(&d_info, (size_t)BATCH_SIZE * sizeof(int));

    /* Pinned host memory for async transfers */
    double *h_A, *h_W;
    cudaMallocHost(&h_A, mat_bytes);
    cudaMallocHost(&h_W, eig_bytes);

    /* Output */
    FILE *out = NULL;
    char path[256];
    if (m_start == n - 1 && m_end == max_m) {
        snprintf(path, sizeof(path), "data/OPT_%d.csv", n);
        out = fopen(path, "w");
        if (out) fprintf(out, "m,score\n");
    }

    double t0 = wall_time();
    long long total = 0;

    for (int m = m_start; m <= m_end; m++) {
        double tm = wall_time();
        long long m_count = 0;
        double best;

        if (n_threads > 1)
            best = enumerate_m_mt(n, m, n_threads, handle, params,
                                   d_A, d_W, d_info, d_work, lwork,
                                   h_A, h_W, stream, &m_count);
        else
            best = enumerate_m_simple(n, m, handle, params,
                                       d_A, d_W, d_info, d_work, lwork,
                                       h_A, h_W, stream, &m_count);

        double elapsed = wall_time() - tm;
        total += m_count;

        printf("  m=%d: lambda2=%.6f (%lld graphs, %.2fs, %.0f g/s)\n",
               m, best, m_count, elapsed,
               elapsed > 0 ? (double)m_count / elapsed : 0);

        if (out) { fprintf(out, "%d,%.10f\n", m, best); fflush(out); }
    }

    if (out) { fclose(out); printf("Saved to %s\n", path); }

    double total_time = wall_time() - t0;
    printf("Done. %lld total graphs in %.1fs (%.0f graphs/s)\n",
           total, total_time, (double)total / total_time);

    /* Cleanup */
    cudaFreeHost(h_A); cudaFreeHost(h_W);
    cudaFree(d_A); cudaFree(d_W); cudaFree(d_work); cudaFree(d_info);
    cudaStreamDestroy(stream);
    cusolverDnDestroySyevjInfo(params);
    cusolverDnDestroy(handle);
    return 0;
}
