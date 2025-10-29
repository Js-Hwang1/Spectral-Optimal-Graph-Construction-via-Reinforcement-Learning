#include <errno.h>
#include <lapacke.h>
#include <math.h>
/* OpenMP is optional. If omp.h is not available (macOS without libomp),
 * provide minimal fallbacks so the code can compile and run single-threaded.
 */
#if defined(__has_include)
# if __has_include(<omp.h>)
#  include <omp.h>
# else
#  define omp_get_max_threads() 1
#  define omp_get_thread_num() 0
#  /* no-op stub when OpenMP is not available */
#  define omp_set_num_threads(x) ((void)0)
# endif
#else
# include <omp.h>
#endif
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define INDEX(i, j, n) ((i) * (n) + (j))

typedef enum {
    INIT_EMPTY = 0,
    INIT_PATH = 1
} init_mode_t;

typedef struct {
    int n_min;
    int n_max;
    int reoptimize_every;
    init_mode_t init_mode;
    char *output_dir;
    int threads;
    double pinv_tol;
} config_t;

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s --n-min <min> --n-max <max> [options]\n"
            "Options:\n"
            "  --reoptimize-every <k>   Recompute resistances every k steps (default: 1)\n"
            "  --init <empty|path>      Initial graph structure (default: empty)\n"
            "  --out-dir <path>         Output directory (default: data_ERG_c)\n"
            "  --threads <count>        Number of OpenMP threads to use (default: OMP default)\n"
            "  --tol <value>            Tolerance for pseudoinverse eigenvalues (default: 1e-8)\n"
            "  --help                   Show this message\n",
            prog);
}

static int ensure_directory(const char *path) {
    if (!path || !*path) {
        return -1;
    }
    char buffer[PATH_MAX];
    size_t len = strlen(path);
    if (len >= sizeof(buffer)) {
        fprintf(stderr, "Output path too long\n");
        return -1;
    }
    strcpy(buffer, path);
    for (size_t i = 1; i <= len; ++i) {
        if (buffer[i] == '/' || buffer[i] == '\0') {
            char saved = buffer[i];
            buffer[i] = '\0';
            if (buffer[0] != '\0') {
                if (mkdir(buffer, 0775) != 0 && errno != EEXIST) {
                    if (errno != EEXIST) {
                        perror("mkdir");
                        return -1;
                    }
                }
            }
            buffer[i] = saved;
        }
    }
    return 0;
}

static char *xstrdup(const char *src) {
    if (!src) {
        return NULL;
    }
    size_t len = strlen(src) + 1;
    char *dst = (char *)malloc(len);
    if (!dst) {
        return NULL;
    }
    memcpy(dst, src, len);
    return dst;
}

static void build_empty(double *adj, int n) {
    memset(adj, 0, sizeof(double) * n * n);
}

static void build_path(double *adj, int n) {
    build_empty(adj, n);
    for (int i = 0; i < n - 1; ++i) {
        adj[INDEX(i, i + 1, n)] = 1.0;
        adj[INDEX(i + 1, i, n)] = 1.0;
    }
}

static int count_edges(const double *adj, int n) {
    int count = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (adj[INDEX(i, j, n)] > 0.5) {
                ++count;
            }
        }
    }
    return count;
}

static void compute_laplacian(const double *adj, int n, double *lap) {
    for (int i = 0; i < n; ++i) {
        double deg = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                continue;
            }
            double val = adj[INDEX(i, j, n)];
            deg += val;
            lap[INDEX(i, j, n)] = -val;
        }
        lap[INDEX(i, i, n)] = deg;
    }
}

static double compute_lambda2_only(const double *adj, int n, double *lap, double *eigvals) {
    compute_laplacian(adj, n, lap);
    int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'N', 'U', n, lap, n, eigvals);
    if (info != 0) {
        fprintf(stderr, "LAPACKE_dsyev (lambda-only) failed with info=%d\n", info);
        return 0.0;
    }
    double lambda2 = 0.0;
    for (int i = 0; i < n; ++i) {
        if (eigvals[i] > 1e-10) {  // basic guard
            lambda2 = eigvals[i];
            break;
        }
    }
    return (lambda2 < 0.0) ? 0.0 : lambda2;
}

static int compute_laplacian_info(const double *adj,
                                  int n,
                                  double tol,
                                  double *lap_pinv,
                                  double *diag_pinv,
                                  double *lambda2_out,
                                  double *work_matrix,
                                  double *eigvals,
                                  double *diag_inv) {
    compute_laplacian(adj, n, work_matrix);
    int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, work_matrix, n, eigvals);
    if (info != 0) {
        fprintf(stderr, "LAPACKE_dsyev failed with info=%d\n", info);
        return info;
    }
    int positive_index = -1;
    for (int k = 0; k < n; ++k) {
        if (eigvals[k] > tol) {
            positive_index = k;
            break;
        }
    }
    double lambda2 = 0.0;
    if (positive_index >= 0) {
        lambda2 = eigvals[positive_index];
    }
    if (lambda2_out) {
        *lambda2_out = lambda2 > 0.0 ? lambda2 : 0.0;
    }
    for (int k = 0; k < n; ++k) {
        diag_inv[k] = (eigvals[k] > tol) ? 1.0 / eigvals[k] : 0.0;
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                double vik = work_matrix[INDEX(i, k, n)];
                double vjk = work_matrix[INDEX(j, k, n)];
                sum += diag_inv[k] * vik * vjk;
            }
            lap_pinv[INDEX(i, j, n)] = sum;
        }
    }
    for (int i = 0; i < n; ++i) {
        diag_pinv[i] = lap_pinv[INDEX(i, i, n)];
    }
    return 0;
}

static int compute_components(const double *adj, int n, int *labels, int *stack) {
    for (int i = 0; i < n; ++i) {
        labels[i] = -1;
    }
    int cid = 0;
    for (int v = 0; v < n; ++v) {
        if (labels[v] != -1) {
            continue;
        }
        int top = 0;
        stack[top++] = v;
        labels[v] = cid;
        while (top > 0) {
            int u = stack[--top];
            for (int w = 0; w < n; ++w) {
                if (adj[INDEX(u, w, n)] > 0.5 && labels[w] == -1) {
                    labels[w] = cid;
                    stack[top++] = w;
                }
            }
        }
        ++cid;
    }
    return cid;
}

static char encode_six_bits(uint8_t value) {
    return (char)(value + 63);
}

static char *graph6_encode(const double *adj, int n) {
    size_t total_pairs = (size_t)n * (n - 1) / 2;
    size_t header_len = 0;
    char header[8];
    if (n <= 62) {
        header[header_len++] = encode_six_bits((uint8_t)n);
    } else if (n <= 258047) {
        header[header_len++] = '~';
        header[header_len++] = encode_six_bits((uint8_t)((n >> 12) & 0x3F));
        header[header_len++] = encode_six_bits((uint8_t)((n >> 6) & 0x3F));
        header[header_len++] = encode_six_bits((uint8_t)(n & 0x3F));
    } else {
        header[header_len++] = '~';
        header[header_len++] = '~';
        header[header_len++] = encode_six_bits((uint8_t)((n >> 30) & 0x3F));
        header[header_len++] = encode_six_bits((uint8_t)((n >> 24) & 0x3F));
        header[header_len++] = encode_six_bits((uint8_t)((n >> 18) & 0x3F));
        header[header_len++] = encode_six_bits((uint8_t)((n >> 12) & 0x3F));
        header[header_len++] = encode_six_bits((uint8_t)((n >> 6) & 0x3F));
        header[header_len++] = encode_six_bits((uint8_t)(n & 0x3F));
    }
    size_t groups = (total_pairs + 5) / 6;
    size_t total_len = header_len + groups;
    char *result = (char *)malloc(total_len + 1);
    if (!result) {
        return NULL;
    }
    memcpy(result, header, header_len);
    size_t idx = header_len;
    uint8_t current = 0;
    int bits = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            current <<= 1;
            current |= (uint8_t)(adj[INDEX(i, j, n)] > 0.5);
            bits++;
            if (bits == 6) {
                result[idx++] = encode_six_bits(current);
                current = 0;
                bits = 0;
            }
        }
    }
    if (bits > 0) {
        current <<= (6 - bits);
        result[idx++] = encode_six_bits(current);
    }
    result[idx] = '\0';
    return result;
}

static int select_best_edge(const double *adj,
                            int n,
                            const double *lap_pinv,
                            const double *diag_pinv,
                            const int *labels,
                            int num_comps,
                            int *best_u,
                            int *best_v) {
    double best_r = -1.0;
    int u = -1;
    int v = -1;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (adj[INDEX(i, j, n)] > 0.5) {
                continue;
            }
            if (num_comps > 1 && labels[i] == labels[j]) {
                continue;
            }
            double resistance = diag_pinv[i] + diag_pinv[j] - 2.0 * lap_pinv[INDEX(i, j, n)];
            if (resistance > best_r) {
                best_r = resistance;
                u = i;
                v = j;
            }
        }
    }
    if (u == -1 || v == -1) {
        return -1;
    }
    *best_u = u;
    *best_v = v;
    return 0;
}

static int write_state(FILE *fp, int n, int m, double lambda2, const char *graph6) {
    if (fprintf(fp, "(%d,%d)\n", n, m) < 0) {
        return -1;
    }
    if (fprintf(fp, "lambda2: %.12f\n", lambda2) < 0) {
        return -1;
    }
    if (fprintf(fp, "graph6: %s\n\n", graph6) < 0) {
        return -1;
    }
    return 0;
}

static int process_n(const config_t *cfg, int n) {
    int total_possible = n * (n - 1) / 2;
    int min_record = (n > 1) ? (n - 1) : 0;
    int max_record = total_possible > 0 ? (total_possible - 1) : 0;
    if (max_record < min_record) {
        max_record = min_record;
    }
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/n%d_ERG_data.txt", cfg->output_dir, n);
    FILE *fp = fopen(path, "w");
    if (!fp) {
        perror("fopen");
        return -1;
    }
    double *adj = (double *)malloc(sizeof(double) * n * n);
    double *lap_pinv = (double *)malloc(sizeof(double) * n * n);
    double *diag_pinv = (double *)malloc(sizeof(double) * n);
    double *work_matrix = (double *)malloc(sizeof(double) * n * n);
    double *eigvals = (double *)malloc(sizeof(double) * n);
    double *diag_inv = (double *)malloc(sizeof(double) * n);
    double *lap_tmp = (double *)malloc(sizeof(double) * n * n);
    double *eigvals_tmp = (double *)malloc(sizeof(double) * n);
    int *labels = (int *)malloc(sizeof(int) * n);
    int *stack = (int *)malloc(sizeof(int) * n);
    if (!adj || !lap_pinv || !diag_pinv || !work_matrix || !eigvals || !diag_inv || !lap_tmp || !eigvals_tmp ||
        !labels || !stack) {
        fprintf(stderr, "Allocation failure for n=%d\n", n);
        fclose(fp);
        free(adj);
        free(lap_pinv);
        free(diag_pinv);
        free(work_matrix);
        free(eigvals);
        free(diag_inv);
        free(lap_tmp);
        free(eigvals_tmp);
        free(labels);
        free(stack);
        return -1;
    }
    if (cfg->init_mode == INIT_PATH && n >= 2) {
        build_path(adj, n);
    } else {
        build_empty(adj, n);
    }
    int cur_m = count_edges(adj, n);
    double lambda2 = 0.0;
    int steps_since = cfg->reoptimize_every;
    int info = compute_laplacian_info(adj, n, cfg->pinv_tol, lap_pinv, diag_pinv, &lambda2, work_matrix, eigvals, diag_inv);
    if (info != 0) {
        fprintf(stderr, "Failed to compute Laplacian info for n=%d\n", n);
        goto cleanup;
    }
    steps_since = 0;
    if (cur_m >= min_record && cur_m <= max_record) {
        char *graph6 = graph6_encode(adj, n);
        if (!graph6) {
            fprintf(stderr, "Graph6 encoding failed for n=%d m=%d\n", n, cur_m);
            goto cleanup;
        }
        if (write_state(fp, n, cur_m, lambda2, graph6) != 0) {
            fprintf(stderr, "Write failed for n=%d m=%d\n", n, cur_m);
            free(graph6);
            goto cleanup;
        }
        free(graph6);
    }
    while (cur_m < total_possible) {
        int num_components = compute_components(adj, n, labels, stack);
        int u = -1;
        int v = -1;
        if (select_best_edge(adj, n, lap_pinv, diag_pinv, labels, num_components, &u, &v) != 0) {
            break;
        }
        adj[INDEX(u, v, n)] = 1.0;
        adj[INDEX(v, u, n)] = 1.0;
        ++cur_m;
        ++steps_since;
        bool recomputed = false;
        if (steps_since >= cfg->reoptimize_every) {
            info = compute_laplacian_info(adj, n, cfg->pinv_tol, lap_pinv, diag_pinv, &lambda2, work_matrix, eigvals, diag_inv);
            if (info != 0) {
                fprintf(stderr, "Failed to recompute Laplacian info for n=%d m=%d\n", n, cur_m);
                goto cleanup;
            }
            steps_since = 0;
            recomputed = true;
        }
        if (cur_m >= min_record && cur_m <= max_record) {
            if (!recomputed) {
                lambda2 = compute_lambda2_only(adj, n, lap_tmp, eigvals_tmp);
            }
            char *graph6 = graph6_encode(adj, n);
            if (!graph6) {
                fprintf(stderr, "Graph6 encoding failed for n=%d m=%d\n", n, cur_m);
                goto cleanup;
            }
            if (write_state(fp, n, cur_m, lambda2, graph6) != 0) {
                fprintf(stderr, "Write failed for n=%d m=%d\n", n, cur_m);
                free(graph6);
                goto cleanup;
            }
            free(graph6);
        }
        if (cur_m >= max_record) {
            break;
        }
    }
    fclose(fp);
    free(adj);
    free(lap_pinv);
    free(diag_pinv);
    free(work_matrix);
    free(eigvals);
    free(diag_inv);
    free(lap_tmp);
    free(eigvals_tmp);
    free(labels);
    free(stack);
    return 0;

cleanup:
    fclose(fp);
    free(adj);
    free(lap_pinv);
    free(diag_pinv);
    free(work_matrix);
    free(eigvals);
    free(diag_inv);
    free(lap_tmp);
    free(eigvals_tmp);
    free(labels);
    free(stack);
    unlink(path);
    return -1;
}

int main(int argc, char **argv) {
    config_t cfg;
    cfg.n_min = -1;
    cfg.n_max = -1;
    cfg.reoptimize_every = 1;
    cfg.init_mode = INIT_EMPTY;
    cfg.output_dir = xstrdup("data_ERG_c");
    cfg.threads = 0;
    cfg.pinv_tol = 1e-8;
    if (!cfg.output_dir) {
        fprintf(stderr, "Failed to allocate default output path\n");
        return EXIT_FAILURE;
    }
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--n-min") == 0 && i + 1 < argc) {
            cfg.n_min = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n-max") == 0 && i + 1 < argc) {
            cfg.n_max = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--reoptimize-every") == 0 && i + 1 < argc) {
            cfg.reoptimize_every = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--init") == 0 && i + 1 < argc) {
            ++i;
            if (strcmp(argv[i], "empty") == 0) {
                cfg.init_mode = INIT_EMPTY;
            } else if (strcmp(argv[i], "path") == 0) {
                cfg.init_mode = INIT_PATH;
            } else {
                fprintf(stderr, "Unknown init mode: %s\n", argv[i]);
                free(cfg.output_dir);
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc) {
            free(cfg.output_dir);
            cfg.output_dir = xstrdup(argv[++i]);
            if (!cfg.output_dir) {
                fprintf(stderr, "Failed to set output directory\n");
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            cfg.threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tol") == 0 && i + 1 < argc) {
            cfg.pinv_tol = atof(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            free(cfg.output_dir);
            return EXIT_SUCCESS;
        } else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            print_usage(argv[0]);
            free(cfg.output_dir);
            return EXIT_FAILURE;
        }
    }
    if (cfg.n_min < 1 || cfg.n_max < cfg.n_min) {
        fprintf(stderr, "Both --n-min and --n-max must be provided (n_min >= 1, n_max >= n_min)\n");
        free(cfg.output_dir);
        return EXIT_FAILURE;
    }
    if (cfg.reoptimize_every < 1) {
        cfg.reoptimize_every = 1;
    }
    if (ensure_directory(cfg.output_dir) != 0) {
        fprintf(stderr, "Failed to prepare output directory: %s\n", cfg.output_dir);
        free(cfg.output_dir);
        return EXIT_FAILURE;
    }
    if (cfg.threads > 0) {
        omp_set_num_threads(cfg.threads);
    }
    int status = 0;
#pragma omp parallel for schedule(dynamic) reduction(| : status)
    for (int n = cfg.n_min; n <= cfg.n_max; ++n) {
        int rc = process_n(&cfg, n);
        if (rc != 0) {
#pragma omp critical
            {
                fprintf(stderr, "Processing failed for n=%d\n", n);
            }
            status |= 1;
        } else {
#pragma omp critical
            {
                printf("n=%d done\n", n);
                fflush(stdout);
            }
        }
    }
    free(cfg.output_dir);
    return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
