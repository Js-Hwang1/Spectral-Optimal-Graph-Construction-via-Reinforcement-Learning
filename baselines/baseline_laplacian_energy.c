#include <errno.h>
#include <lapacke.h>
#include <math.h>
#include <omp.h>
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

typedef struct {
    int n_min;
    int n_max;
    char *output_dir;
    int threads;
} config_t;

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s --n-min <min> --n-max <max> [options]\n"
            "Options:\n"
            "  --out-dir <path>   Output directory (default: data_LE_c)\n"
            "  --threads <count>  Number of OpenMP threads (default: OMP default)\n"
            "  --help             Show this message\n",
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
                    perror("mkdir");
                    return -1;
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

static void build_circulant_graph(int n, int m, double *adj) {
    size_t total = (size_t)n * (size_t)n;
    for (size_t idx = 0; idx < total; ++idx) {
        adj[idx] = 0.0;
    }
    int edges = 0;
    int max_edges = n * (n - 1) / 2;
    if (m < 0) {
        m = 0;
    }
    if (m > max_edges) {
        m = max_edges;
    }
    for (int offset = 1; offset < n && edges < m; ++offset) {
        for (int u = 0; u < n && edges < m; ++u) {
            int v = u + offset;
            if (v >= n) {
                v -= n;
            }
            if (u < v && adj[INDEX(u, v, n)] == 0.0) {
                adj[INDEX(u, v, n)] = 1.0;
                adj[INDEX(v, u, n)] = 1.0;
                ++edges;
            }
        }
    }
}

static void compute_laplacian(const double *adj, int n, double *lap) {
    for (int i = 0; i < n; ++i) {
        double deg = 0.0;
        for (int j = 0; j < n; ++j) {
            double val = adj[INDEX(i, j, n)];
            deg += val;
            lap[INDEX(i, j, n)] = (i == j) ? 0.0 : -val;
        }
        lap[INDEX(i, i, n)] = deg;
    }
}

static int laplacian_spectrum(const double *adj,
                              int n,
                              double *lambda2_out,
                              double *energy_out,
                              double *workspace,
                              double *eigvals) {
    compute_laplacian(adj, n, workspace);
    int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'N', 'U', n, workspace, n, eigvals);
    if (info != 0) {
        return info;
    }
    double lambda2 = 0.0;
    if (n >= 2) {
        lambda2 = eigvals[1];
        if (lambda2 < 0.0 && lambda2 > -1e-9) {
            lambda2 = 0.0;
        }
    }
    double energy = 0.0;
    if (energy_out) {
        for (int i = 0; i < n; ++i) {
            energy += eigvals[i] * eigvals[i];
        }
    }
    if (lambda2_out) {
        *lambda2_out = lambda2;
    }
    if (energy_out) {
        *energy_out = energy;
    }
    return 0;
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
            ++bits;
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
    int m_min = (n > 1) ? (n - 1) : 0;
    int m_max = n * (n - 1) / 2 - 1;
    if (m_max < m_min) {
        m_max = m_min;
    }
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/n%d_LE_data.txt", cfg->output_dir, n);
    FILE *fp = fopen(path, "w");
    if (!fp) {
        perror("fopen");
        return -1;
    }
    double *adj = (double *)malloc(sizeof(double) * n * n);
    double *lap = (double *)malloc(sizeof(double) * n * n);
    double *eigvals = (double *)malloc(sizeof(double) * n);
    if (!adj || !lap || !eigvals) {
        fprintf(stderr, "Allocation failure for n=%d\n", n);
        fclose(fp);
        free(adj);
        free(lap);
        free(eigvals);
        unlink(path);
        return -1;
    }
    for (int m = m_min; m <= m_max; ++m) {
        build_circulant_graph(n, m, adj);
        double lambda2 = 0.0;
        int info = laplacian_spectrum(adj, n, &lambda2, NULL, lap, eigvals);
        if (info != 0) {
            fprintf(stderr, "Spectrum computation failed (n=%d, m=%d, info=%d)\n", n, m, info);
            fclose(fp);
            free(adj);
            free(lap);
            free(eigvals);
            unlink(path);
            return -1;
        }
        char *g6 = graph6_encode(adj, n);
        if (!g6) {
            fprintf(stderr, "Graph6 encoding failed for n=%d, m=%d\n", n, m);
            fclose(fp);
            free(adj);
            free(lap);
            free(eigvals);
            unlink(path);
            return -1;
        }
        if (write_state(fp, n, m, lambda2, g6) != 0) {
            fprintf(stderr, "Write failed for n=%d, m=%d\n", n, m);
            free(g6);
            fclose(fp);
            free(adj);
            free(lap);
            free(eigvals);
            unlink(path);
            return -1;
        }
        free(g6);
    }
    fclose(fp);
    free(adj);
    free(lap);
    free(eigvals);
    return 0;
}

int main(int argc, char **argv) {
    config_t cfg;
    cfg.n_min = -1;
    cfg.n_max = -1;
    cfg.output_dir = xstrdup("data_LE_c");
    cfg.threads = 0;
    if (!cfg.output_dir) {
        fprintf(stderr, "Failed to allocate default output directory\n");
        return EXIT_FAILURE;
    }
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--n-min") == 0 && i + 1 < argc) {
            cfg.n_min = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n-max") == 0 && i + 1 < argc) {
            cfg.n_max = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc) {
            free(cfg.output_dir);
            cfg.output_dir = xstrdup(argv[++i]);
            if (!cfg.output_dir) {
                fprintf(stderr, "Failed to allocate output directory string\n");
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            cfg.threads = atoi(argv[++i]);
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
        fprintf(stderr, "Both --n-min and --n-max must be provided with n_min >= 1 and n_max >= n_min\n");
        free(cfg.output_dir);
        return EXIT_FAILURE;
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


./LPE \
  --n-min 6 \
  --n-max 32 \
  --threads 8 \
  --out-dir /gpfs/scratch/jungshwang/data