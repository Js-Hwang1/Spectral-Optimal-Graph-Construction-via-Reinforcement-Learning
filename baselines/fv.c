/*
 * fv.c - Fiedler Vector Greedy Algorithm Implementation
 */

#include "fv.h"
#include <stdlib.h>
#include <stdio.h>

FVResult *fv_result_create(int capacity) {
    FVResult *result = malloc(sizeof(FVResult));
    if (!result) return NULL;

    result->m_values = malloc((size_t)capacity * sizeof(int));
    result->scores = malloc((size_t)capacity * sizeof(double));
    result->count = 0;
    result->capacity = capacity;

    if (!result->m_values || !result->scores) {
        free(result->m_values);
        free(result->scores);
        free(result);
        return NULL;
    }

    return result;
}

void fv_result_free(FVResult *result) {
    if (result) {
        free(result->m_values);
        free(result->scores);
        free(result);
    }
}

static void fv_result_add(FVResult *result, int m, double score) {
    if (result->count >= result->capacity) {
        result->capacity *= 2;
        result->m_values = realloc(result->m_values,
                                   (size_t)result->capacity * sizeof(int));
        result->scores = realloc(result->scores,
                                 (size_t)result->capacity * sizeof(double));
    }
    result->m_values[result->count] = m;
    result->scores[result->count] = score;
    result->count++;
}

void fv_run(int n, FVResult *result, InitType init) {
    int max_m = n * (n - 1) / 2;

    AdjMatrix *adj = adj_create(n);
    if (init == INIT_RING)
        build_ring(adj);
    else
        build_random_tree(adj);

    int curr_m = adj_edge_count(adj);
    fv_result_add(result, curr_m, compute_algebraic_connectivity(adj));

    double *fiedler = malloc((size_t)n * sizeof(double));
    if (!fiedler) {
        adj_free(adj);
        return;
    }

    while (curr_m < max_m) {
        compute_fiedler_vector(adj, fiedler);

        /* Find non-edge with maximum (fiedler[i] - fiedler[j])^2 */
        double max_diff_sq = -1.0;
        int best_u = -1, best_v = -1;

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (!adj_get(adj, i, j)) {
                    double diff = fiedler[i] - fiedler[j];
                    double diff_sq = diff * diff;
                    if (diff_sq > max_diff_sq) {
                        max_diff_sq = diff_sq;
                        best_u = i;
                        best_v = j;
                    }
                }
            }
        }

        if (best_u < 0) break;

        adj_set(adj, best_u, best_v, true);
        adj_set(adj, best_v, best_u, true);
        curr_m++;

        fv_result_add(result, curr_m, compute_algebraic_connectivity(adj));
    }

    free(fiedler);
    adj_free(adj);

    printf("  FV done for N=%d\n", n);
}

void fv_run_single(int n, int m, InitType init, AdjMatrix *adj_out) {
    AdjMatrix *adj = adj_create(n);
    if (init == INIT_RING)
        build_ring(adj);
    else
        build_random_tree(adj);

    int curr_m = adj_edge_count(adj);

    double *fiedler = malloc((size_t)n * sizeof(double));
    if (!fiedler) { adj_free(adj); return; }

    while (curr_m < m) {
        compute_fiedler_vector(adj, fiedler);

        double max_diff_sq = -1.0;
        int best_u = -1, best_v = -1;

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (!adj_get(adj, i, j)) {
                    double diff = fiedler[i] - fiedler[j];
                    double diff_sq = diff * diff;
                    if (diff_sq > max_diff_sq) {
                        max_diff_sq = diff_sq;
                        best_u = i;
                        best_v = j;
                    }
                }
            }
        }

        if (best_u < 0) break;

        adj_set(adj, best_u, best_v, true);
        adj_set(adj, best_v, best_u, true);
        curr_m++;
    }

    free(fiedler);
    adj_copy(adj_out, adj);
    adj_free(adj);
}
