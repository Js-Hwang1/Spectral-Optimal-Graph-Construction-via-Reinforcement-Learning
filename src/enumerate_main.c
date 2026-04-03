/*
 * enumerate_main.c -- Find max lambda2 for each (n,m) using nauty/geng.
 *
 * Pipes non-isomorphic connected graphs from geng, computes lambda2 for each,
 * keeps the maximum per m value.
 *
 * Usage: crl_enum <n> [min_m max_m]
 * Output: data/OPT_{n}.csv with columns: m,score
 */

#include "crl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

static double wall_time(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* Parse graph6 format (nauty output) into adjacency matrix */
static int parse_graph6(const char *s, uint8_t *adj, int max_n) {
    /* graph6 format: N(n) then adjacency bits */
    int idx = 0;
    int n;

    /* Parse n */
    if ((unsigned char)s[idx] >= 63 && (unsigned char)s[idx] <= 126) {
        n = (unsigned char)s[idx] - 63;
        idx++;
    } else {
        return -1; /* multi-byte n not supported for small graphs */
    }

    if (n > max_n) return -1;
    memset(adj, 0, (size_t)n * n);

    /* Parse adjacency bits */
    int bit_idx = 0;
    int data_idx = idx;
    for (int j = 1; j < n; j++) {
        for (int i = 0; i < j; i++) {
            int byte_pos = bit_idx / 6;
            int bit_pos = 5 - (bit_idx % 6);
            int byte_val = (unsigned char)s[data_idx + byte_pos] - 63;
            int bit = (byte_val >> bit_pos) & 1;
            if (bit) {
                adj[i * n + j] = adj[j * n + i] = 1;
            }
            bit_idx++;
        }
    }

    return n;
}

static int count_edges(const uint8_t *adj, int n) {
    int m = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (adj[i * n + j]) m++;
    return m;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <n> [min_m max_m]\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int max_m = n * (n - 1) / 2;
    int min_m = n - 1;
    if (argc >= 3) min_m = atoi(argv[2]);
    if (argc >= 4) max_m = atoi(argv[3]);

    /* Allocate per-m best lambda2 */
    double *best_l2 = (double *)calloc((size_t)(max_m + 1), sizeof(double));

    char path[256];
    snprintf(path, sizeof(path), "data/OPT_%d.csv", n);

    printf("Enumerating optimal lambda2 for n=%d, m=[%d,%d] using nauty/geng\n", n, min_m, max_m);

    double t0 = wall_time();
    long long total_graphs = 0;

    for (int m = min_m; m <= max_m; m++) {
        double tm = wall_time();

        /* Run geng for this (n, m) */
        char cmd[256];
        snprintf(cmd, sizeof(cmd), "geng -c -q %d %d:%d", n, m, m);
        FILE *pipe = popen(cmd, "r");
        if (!pipe) { fprintf(stderr, "Failed to run geng\n"); continue; }

        /* Read all graphs into a buffer first for OMP parallelism */
        char **lines = NULL;
        int n_lines = 0, cap_lines = 1024;
        lines = (char **)malloc((size_t)cap_lines * sizeof(char *));

        char line[1024];
        while (fgets(line, sizeof(line), pipe)) {
            int len = (int)strlen(line);
            if (len > 0 && line[len-1] == '\n') line[len-1] = '\0';
            if (len < 2) continue;
            if (n_lines >= cap_lines) {
                cap_lines *= 2;
                lines = realloc(lines, (size_t)cap_lines * sizeof(char *));
            }
            lines[n_lines] = strdup(line);
            n_lines++;
        }
        pclose(pipe);

        /* Parallel lambda2 evaluation */
        double m_best = 0;
        #pragma omp parallel for reduction(max:m_best) schedule(dynamic, 64)
        for (int i = 0; i < n_lines; i++) {
            uint8_t adj[256]; /* max n=16, 16*16=256 */
            int gn = parse_graph6(lines[i], adj, 16);
            if (gn != n) continue;
            double l2 = exact_lambda2(adj, gn);
            if (l2 > m_best) m_best = l2;
        }

        best_l2[m] = m_best;
        total_graphs += n_lines;

        double elapsed = wall_time() - tm;
        printf("  m=%d: lambda2=%.6f (%d graphs, %.2fs)\n", m, m_best, n_lines, elapsed);

        for (int i = 0; i < n_lines; i++) free(lines[i]);
        free(lines);
    }

    /* Write output */
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return 1; }
    fprintf(f, "m,score\n");
    for (int m = min_m; m <= max_m; m++) {
        fprintf(f, "%d,%.10f\n", m, best_l2[m]);
    }
    fclose(f);

    double total_time = wall_time() - t0;
    printf("Done. %lld total graphs. Saved to %s (%.1fs)\n", total_graphs, path, total_time);

    free(best_l2);
    return 0;
}
