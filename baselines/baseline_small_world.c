#include <ctype.h>
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
    int tail;
    int head;
} edge_t;

typedef struct {
    int n_min;
    int n_max;
    int k_single;
    int k_min;
    int k_max;
    bool sweep_k;
    double *p_values;
    int p_count;
    char *output_dir;
    int threads;
    uint64_t seed;
} config_t;

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s --n-min <min> --n-max <max> [options]\n"
            "Options:\n"
            "  --n <value>             Evaluate a single network size n (overrides --n-min/--n-max)\n"
            "  --k <value>             Number of lattice neighbors on each side (default: 2)\n"
            "  --k-min <value>         Minimum k for sweeping (enables k sweep)\n"
            "  --k-max <value>         Maximum k for sweeping (enables k sweep)\n"
            "  --k-all                 Sweep k from 1 to floor((n-1)/2)\n"
            "  --p-values <list>       Comma-separated rewiring probabilities (default: 0,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.68,0.8,1)\n"
            "  --p <value>             Single rewiring probability in [0,1] (overrides --p-values)\n"
            "  --seed <value>          RNG seed (default: 1469598103934665603)\n"
            "  --out-dir <path>        Output directory (default: data_SW_c)\n"
            "  --threads <count>       Number of OpenMP threads (default: OMP default)\n"
            "  --help                  Show this message\n",
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

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

static double rand_uniform(uint64_t *state) {
    return (splitmix64(state) >> 11) * (1.0 / (double)(UINT64_C(1) << 53));
}

static int rand_int(uint64_t *state, int bound) {
    if (bound <= 0) {
        return 0;
    }
    return (int)(splitmix64(state) % (uint64_t)bound);
}

static int rand_bit(uint64_t *state) {
    return (int)(splitmix64(state) & UINT64_C(1));
}

static void build_ring_lattice(double *adj, int n, int k) {
    size_t total = (size_t)n * (size_t)n;
    for (size_t idx = 0; idx < total; ++idx) {
        adj[idx] = 0.0;
    }
    for (int i = 0; i < n; ++i) {
        for (int d = 1; d <= k; ++d) {
            int j_forward = (i + d) % n;
            int j_backward = (i - d + n) % n;
            if (i < j_forward) {
                adj[INDEX(i, j_forward, n)] = 1.0;
                adj[INDEX(j_forward, i, n)] = 1.0;
            }
            if (j_backward != j_forward && i < j_backward) {
                adj[INDEX(i, j_backward, n)] = 1.0;
                adj[INDEX(j_backward, i, n)] = 1.0;
            }
        }
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

static int build_edge_list(const double *adj, int n, edge_t *edges, uint64_t *state) {
    int idx = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (adj[INDEX(i, j, n)] > 0.5) {
                if (rand_bit(state)) {
                    edges[idx].tail = i;
                    edges[idx].head = j;
                } else {
                    edges[idx].tail = j;
                    edges[idx].head = i;
                }
                ++idx;
            }
        }
    }
    return idx;
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
                              double *lambda_n_out,
                              double *lap,
                              double *eigvals) {
    compute_laplacian(adj, n, lap);
    int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'N', 'U', n, lap, n, eigvals);
    if (info != 0) {
        return info;
    }
    if (lambda2_out) {
        *lambda2_out = (n >= 2) ? eigvals[1] : 0.0;
    }
    if (lambda_n_out) {
        *lambda_n_out = eigvals[n - 1];
    }
    return 0;
}

static int select_random_non_neighbor(const double *adj, int n, int u, int exclude, uint64_t *state) {
    int count = 0;
    int chosen = -1;
    for (int v = 0; v < n; ++v) {
        if (v == u || v == exclude) {
            continue;
        }
        if (adj[INDEX(u, v, n)] > 0.5) {
            continue;
        }
        ++count;
        if (rand_int(state, count) == 0) {
            chosen = v;
        }
    }
    return chosen;
}

static uint8_t encode_six_bits(uint8_t value) {
    return (uint8_t)(value + 63);
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

static int write_record(FILE *fp, int n, int m, double rho, double lambda2, const char *graph6) {
    if (fprintf(fp, "(%d,%d,%.6f)\n", n, m, rho) < 0) {
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

static void format_probability_tag(double p, char *buffer, size_t len) {
    char tmp[32];
    snprintf(tmp, sizeof(tmp), "%.6f", p);
    size_t end = strlen(tmp);
    while (end > 0 && tmp[end - 1] == '0') {
        --end;
    }
    if (end > 0 && tmp[end - 1] == '.') {
        --end;
    }
    if (end == 0) {
        tmp[0] = '0';
        end = 1;
    }
    tmp[end] = '\0';
    char sanitized[32];
    size_t idx = 0;
    if (idx < sizeof(sanitized)) {
        sanitized[idx++] = 'p';
    }
    for (size_t i = 0; i < end && idx < sizeof(sanitized) - 1; ++i) {
        char c = tmp[i];
        sanitized[idx++] = (c == '.') ? '_' : c;
    }
    sanitized[idx] = '\0';
    strncpy(buffer, sanitized, len);
    if (len > 0) {
        buffer[len - 1] = '\0';
    }
}

static double *default_p_values(int *count) {
    static const double defaults[] = {0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.68, 0.8, 1.0};
    int n = (int)(sizeof(defaults) / sizeof(defaults[0]));
    double *values = (double *)malloc(sizeof(double) * n);
    if (!values) {
        return NULL;
    }
    for (int i = 0; i < n; ++i) {
        values[i] = defaults[i];
    }
    *count = n;
    return values;
}

static double *parse_p_values(const char *arg, int *out_count) {
    if (!arg) {
        return NULL;
    }
    char *copy = xstrdup(arg);
    if (!copy) {
        return NULL;
    }
    int capacity = 8;
    double *values = (double *)malloc(sizeof(double) * capacity);
    if (!values) {
        free(copy);
        return NULL;
    }
    int count = 0;
    char *token = strtok(copy, ",");
    while (token) {
        while (isspace((unsigned char)*token)) {
            ++token;
        }
        char *endptr = token + strlen(token);
        while (endptr > token && isspace((unsigned char)endptr[-1])) {
            --endptr;
        }
        *endptr = '\0';
        if (*token == '\0') {
            free(values);
            free(copy);
            return NULL;
        }
        char *parse_end = NULL;
        double val = strtod(token, &parse_end);
        if (parse_end == token || *parse_end != '\0' || val < 0.0 || val > 1.0) {
            free(values);
            free(copy);
            return NULL;
        }
        if (count >= capacity) {
            capacity *= 2;
            double *tmp = (double *)realloc(values, sizeof(double) * capacity);
            if (!tmp) {
                free(values);
                free(copy);
                return NULL;
            }
            values = tmp;
        }
        values[count++] = val;
        token = strtok(NULL, ",");
    }
    free(copy);
    if (count == 0) {
        free(values);
        return NULL;
    }
    *out_count = count;
    return values;
}

static int process_case(const config_t *cfg, int n, int k) {
    int k_cap = (n - 1) / 2;
    if (k < 1 || k > k_cap) {
        return 0;
    }
    int max_edges = n * k;
    if (max_edges <= 0) {
        return 0;
    }
    double *base_adj = (double *)malloc(sizeof(double) * n * n);
    double *work_adj = (double *)malloc(sizeof(double) * n * n);
    double *lap = (double *)malloc(sizeof(double) * n * n);
    double *eigvals = (double *)malloc(sizeof(double) * n);
    edge_t *edges = (edge_t *)malloc(sizeof(edge_t) * max_edges);
    if (!base_adj || !work_adj || !lap || !eigvals || !edges) {
        fprintf(stderr, "Allocation failure for n=%d k=%d\n", n, k);
        free(base_adj);
        free(work_adj);
        free(lap);
        free(eigvals);
        free(edges);
        return -1;
    }
    build_ring_lattice(base_adj, n, k);
    int actual_edges = count_edges(base_adj, n);
    if (actual_edges != max_edges) {
        actual_edges = max_edges;
    }
    size_t matrix_bytes = sizeof(double) * n * n;
    char path[PATH_MAX];
    char tag[32];
    int status = 0;
    double upper_bound = 0.0;
    if (n > 1) {
        upper_bound = ((double)k * (double)n) / (double)(n - 1);
    }
    for (int p_idx = 0; p_idx < cfg->p_count; ++p_idx) {
        double p = cfg->p_values[p_idx];
        format_probability_tag(p, tag, sizeof(tag));
        snprintf(path, sizeof(path), "%s/n%d_k%d_%s_SW_data.txt", cfg->output_dir, n, k, tag);
        memcpy(work_adj, base_adj, matrix_bytes);
        uint64_t state = cfg->seed;
        state ^= (uint64_t)n * UINT64_C(0x9E3779B97F4A7C15);
        state ^= (uint64_t)k * UINT64_C(0xBF58476D1CE4E5B9);
        state ^= (uint64_t)(p_idx + 1) * UINT64_C(0xD6E8FEB86659FD93);
        int edge_count = build_edge_list(work_adj, n, edges, &state);
        for (int e = 0; e < edge_count; ++e) {
            if (rand_uniform(&state) > p) {
                continue;
            }
            int u = edges[e].tail;
            int v = edges[e].head;
            int w = select_random_non_neighbor(work_adj, n, u, v, &state);
            if (w == -1) {
                continue;
            }
            work_adj[INDEX(u, v, n)] = 0.0;
            work_adj[INDEX(v, u, n)] = 0.0;
            work_adj[INDEX(u, w, n)] = 1.0;
            work_adj[INDEX(w, u, n)] = 1.0;
            edges[e].head = w;
        }
        double lambda2 = 0.0;
        if (laplacian_spectrum(work_adj, n, &lambda2, NULL, lap, eigvals) != 0) {
            fprintf(stderr, "Spectrum computation failed (n=%d k=%d p=%.6f)\n", n, k, p);
            status = -1;
            break;
        }
        if (n > 1 && upper_bound > 0.0 && lambda2 > upper_bound + 1e-9) {
            fprintf(stderr, "lambda2 %.12f exceeds bound %.12f (n=%d k=%d p=%.6f)\n",
                    lambda2, upper_bound, n, k, p);
            status = -1;
            break;
        }
        char *graph6 = graph6_encode(work_adj, n);
        if (!graph6) {
            fprintf(stderr, "Graph6 encoding failed (n=%d k=%d p=%.6f)\n", n, k, p);
            status = -1;
            break;
        }
        FILE *fp = fopen(path, "w");
        if (!fp) {
            perror("fopen");
            free(graph6);
            status = -1;
            break;
        }
        if (write_record(fp, n, actual_edges, p, lambda2, graph6) != 0) {
            fprintf(stderr, "Write failed for %s\n", path);
            fclose(fp);
            free(graph6);
            unlink(path);
            status = -1;
            break;
        }
        fclose(fp);
        free(graph6);
    }
    free(base_adj);
    free(work_adj);
    free(lap);
    free(eigvals);
    free(edges);
    return status;
}

int main(int argc, char **argv) {
    config_t cfg;
    cfg.n_min = -1;
    cfg.n_max = -1;
    cfg.k_single = 2;
    cfg.k_min = -1;
    cfg.k_max = -1;
    cfg.sweep_k = false;
    cfg.p_values = default_p_values(&cfg.p_count);
    cfg.output_dir = xstrdup("data_SW_c");
    cfg.threads = 0;
    cfg.seed = UINT64_C(1469598103934665603);
    if (!cfg.p_values || !cfg.output_dir) {
        fprintf(stderr, "Failed to allocate defaults\n");
        free(cfg.p_values);
        free(cfg.output_dir);
        return EXIT_FAILURE;
    }
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--n-min") == 0 && i + 1 < argc) {
            cfg.n_min = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n-max") == 0 && i + 1 < argc) {
            cfg.n_max = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            char *endptr = NULL;
            long val = strtol(argv[++i], &endptr, 10);
            if (endptr == argv[i] || *endptr != '\0' || val < 1) {
                fprintf(stderr, "Invalid --n argument\n");
                free(cfg.p_values);
                free(cfg.output_dir);
                return EXIT_FAILURE;
            }
            cfg.n_min = (int)val;
            cfg.n_max = (int)val;
        } else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            cfg.k_single = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--k-min") == 0 && i + 1 < argc) {
            cfg.k_min = atoi(argv[++i]);
            cfg.sweep_k = true;
        } else if (strcmp(argv[i], "--k-max") == 0 && i + 1 < argc) {
            cfg.k_max = atoi(argv[++i]);
            cfg.sweep_k = true;
        } else if (strcmp(argv[i], "--k-all") == 0) {
            cfg.sweep_k = true;
            cfg.k_min = 1;
            cfg.k_max = -1;
        } else if (strcmp(argv[i], "--p-values") == 0 && i + 1 < argc) {
            double *values = parse_p_values(argv[++i], &cfg.p_count);
            if (!values) {
                fprintf(stderr, "Invalid --p-values argument\n");
                free(cfg.p_values);
                free(cfg.output_dir);
                return EXIT_FAILURE;
            }
            free(cfg.p_values);
            cfg.p_values = values;
        } else if (strcmp(argv[i], "--p") == 0 && i + 1 < argc) {
            char *endptr = NULL;
            double val = strtod(argv[++i], &endptr);
            if (endptr == argv[i] || *endptr != '\0' || val < 0.0 || val > 1.0) {
                fprintf(stderr, "Invalid --p argument\n");
                free(cfg.p_values);
                free(cfg.output_dir);
                return EXIT_FAILURE;
            }
            double *single = (double *)malloc(sizeof(double));
            if (!single) {
                fprintf(stderr, "Failed to allocate probability array\n");
                free(cfg.p_values);
                free(cfg.output_dir);
                return EXIT_FAILURE;
            }
            *single = val;
            free(cfg.p_values);
            cfg.p_values = single;
            cfg.p_count = 1;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            cfg.seed = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc) {
            free(cfg.output_dir);
            cfg.output_dir = xstrdup(argv[++i]);
            if (!cfg.output_dir) {
                fprintf(stderr, "Failed to set output directory\n");
                free(cfg.p_values);
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            cfg.threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            free(cfg.p_values);
            free(cfg.output_dir);
            return EXIT_SUCCESS;
        } else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            print_usage(argv[0]);
            free(cfg.p_values);
            free(cfg.output_dir);
            return EXIT_FAILURE;
        }
    }
    if (cfg.n_min < 1 || cfg.n_max < cfg.n_min) {
        fprintf(stderr, "Provide --n or both --n-min/--n-max with n >= 1 and n_max >= n_min\n");
        free(cfg.p_values);
        free(cfg.output_dir);
        return EXIT_FAILURE;
    }
    if (cfg.sweep_k) {
        if (cfg.k_min < 1) {
            cfg.k_min = 1;
        }
        if (cfg.k_max >= 0 && cfg.k_max < cfg.k_min) {
            fprintf(stderr, "k-max must be greater than or equal to k-min\n");
            free(cfg.p_values);
            free(cfg.output_dir);
            return EXIT_FAILURE;
        }
    } else {
        if (cfg.k_single < 1) {
            cfg.k_single = 1;
        }
    }
    if (cfg.p_count < 1) {
        fprintf(stderr, "No rewiring probabilities provided\n");
        free(cfg.p_values);
        free(cfg.output_dir);
        return EXIT_FAILURE;
    }
    if (ensure_directory(cfg.output_dir) != 0) {
        fprintf(stderr, "Failed to prepare output directory: %s\n", cfg.output_dir);
        free(cfg.p_values);
        free(cfg.output_dir);
        return EXIT_FAILURE;
    }
    if (cfg.threads > 0) {
        omp_set_num_threads(cfg.threads);
    }
    int status = 0;
#pragma omp parallel for schedule(dynamic) reduction(| : status)
    for (int n = cfg.n_min; n <= cfg.n_max; ++n) {
        int cap = (n - 1) / 2;
        if (cap < 1) {
            continue;
        }
        int k_start = cfg.sweep_k ? cfg.k_min : cfg.k_single;
        int k_end = cfg.sweep_k ? ((cfg.k_max < 0) ? cap : cfg.k_max) : k_start;
        if (k_start < 1) {
            k_start = 1;
        }
        if (k_end > cap) {
            k_end = cap;
        }
        if (k_start > k_end) {
            continue;
        }
        for (int k = k_start; k <= k_end; ++k) {
            int rc = process_case(&cfg, n, k);
            if (rc != 0) {
#pragma omp critical
                {
                    fprintf(stderr, "Processing failed for n=%d k=%d\n", n, k);
                }
                status |= 1;
            } else {
#pragma omp critical
                {
                    printf("n=%d k=%d done\n", n, k);
                    fflush(stdout);
                }
            }
        }
    }
    free(cfg.p_values);
    free(cfg.output_dir);
    return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
