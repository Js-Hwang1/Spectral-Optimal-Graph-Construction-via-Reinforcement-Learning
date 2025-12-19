/*
 * er.c - Effective Resistance Greedy Algorithm Implementation
 */

#include "er.h"
#include <stdlib.h>
#include <stdio.h>

ERResult *er_result_create(int capacity) {
    ERResult *result = malloc(sizeof(ERResult));
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

void er_result_free(ERResult *result) {
    if (result) {
        free(result->m_values);
        free(result->scores);
        free(result);
    }
}

static void er_result_add(ERResult *result, int m, double score) {
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

void er_run(int n, ERResult *result) {
    int max_m = n * (n - 1) / 2;

    /* Initialize ring graph */
    AdjMatrix *adj = adj_create(n);
    for (int i = 0; i < n; i++) {
        adj_set(adj, i, (i + 1) % n, true);
        adj_set(adj, (i + 1) % n, i, true);
    }

    int curr_m = n;
    er_result_add(result, curr_m, compute_algebraic_connectivity(adj));

    double *L_pinv = malloc((size_t)n * n * sizeof(double));
    if (!L_pinv) {
        adj_free(adj);
        return;
    }

    while (curr_m < max_m) {
        compute_laplacian_pinv(adj, L_pinv);

        /* Find non-edge with maximum effective resistance */
        double max_R = -1.0;
        int best_u = -1, best_v = -1;

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (!adj_get(adj, i, j)) {
                    double R = L_pinv[i * n + i] + L_pinv[j * n + j]
                               - 2.0 * L_pinv[i * n + j];
                    if (R > max_R) {
                        max_R = R;
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

        er_result_add(result, curr_m, compute_algebraic_connectivity(adj));
    }

    free(L_pinv);
    adj_free(adj);

    printf("  ER done for N=%d\n", n);
}
