/*
 * sw.c - Small World Network Baseline Implementation
 */

#include "sw.h"
#include <stdlib.h>
#include <stdio.h>

SWResult *sw_result_create(int capacity, double rho) {
    SWResult *result = malloc(sizeof(SWResult));
    if (!result) return NULL;

    result->m_values = malloc((size_t)capacity * sizeof(int));
    result->scores = malloc((size_t)capacity * sizeof(double));
    result->count = 0;
    result->capacity = capacity;
    result->rho = rho;

    if (!result->m_values || !result->scores) {
        free(result->m_values);
        free(result->scores);
        free(result);
        return NULL;
    }

    return result;
}

void sw_result_free(SWResult *result) {
    if (result) {
        free(result->m_values);
        free(result->scores);
        free(result);
    }
}

double sw_single_score(int n, int m, double rho) {
    int k = 2 * m / n;
    if (k < 2) k = 2;

    AdjMatrix *adj = adj_create(n);

    /* Build ring lattice */
    for (int i = 0; i < n; i++) {
        for (int j = 1; j <= k / 2; j++) {
            int target = (i + j) % n;
            adj_set(adj, i, target, true);
            adj_set(adj, target, i, true);
        }
    }

    /* Add random edges to reach m */
    int curr = adj_edge_count(adj);
    while (curr < m) {
        int u = rng_int(n);
        int v = rng_int(n);
        if (u != v && !adj_get(adj, u, v)) {
            adj_set(adj, u, v, true);
            adj_set(adj, v, u, true);
            curr++;
        }
    }

    /* Rewiring - only rewire if we can find a valid replacement */
    if (rho > 0.0) {
        int actual_edges = adj_edge_count(adj);
        int *edges_u = malloc((size_t)actual_edges * sizeof(int));
        int *edges_v = malloc((size_t)actual_edges * sizeof(int));
        int num_edges = 0;

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (adj_get(adj, i, j)) {
                    edges_u[num_edges] = i;
                    edges_v[num_edges] = j;
                    num_edges++;
                }
            }
        }

        int num_rewire = (int)(num_edges * rho);
        for (int r = 0; r < num_rewire; r++) {
            int idx = rng_int(num_edges);
            int u = edges_u[idx];
            int v = edges_v[idx];

            /* Find a valid replacement BEFORE removing the edge */
            int nu = -1;
            for (int attempt = 0; attempt < 100; attempt++) {
                int candidate = rng_int(n);
                if (candidate != u && !adj_get(adj, u, candidate)) {
                    nu = candidate;
                    break;
                }
            }

            /* Only rewire if we found a valid replacement */
            if (nu >= 0) {
                adj_set(adj, u, v, false);
                adj_set(adj, v, u, false);
                adj_set(adj, u, nu, true);
                adj_set(adj, nu, u, true);
            }
        }

        free(edges_u);
        free(edges_v);
    }

    double score = compute_algebraic_connectivity(adj);
    adj_free(adj);
    return score;
}

void sw_run(int n, double rho, SWResult *result, int step) {
    int max_m = n * (n - 1) / 2;
    double scores[SW_NUM_SEEDS];

    int last_m = -1;
    for (int m = n - 1; m <= max_m; m += step) {
        /* Run multiple seeds */
        for (int seed = 0; seed < SW_NUM_SEEDS; seed++) {
            rng_seed((uint64_t)(seed * 12345 + m * 67890 + (int)(rho * 1000)));
            scores[seed] = sw_single_score(n, m, rho);
        }

        /* Compute mean */
        double sum = 0.0;
        for (int i = 0; i < SW_NUM_SEEDS; i++) {
            sum += scores[i];
        }
        double mean = sum / SW_NUM_SEEDS;

        /* Store result */
        if (result->count >= result->capacity) {
            result->capacity *= 2;
            result->m_values = realloc(result->m_values,
                                       (size_t)result->capacity * sizeof(int));
            result->scores = realloc(result->scores,
                                     (size_t)result->capacity * sizeof(double));
        }

        result->m_values[result->count] = m;
        result->scores[result->count] = mean;
        result->count++;
        last_m = m;
    }

    /* Always include the complete graph if not already included */
    if (last_m < max_m) {
        for (int seed = 0; seed < SW_NUM_SEEDS; seed++) {
            rng_seed((uint64_t)(seed * 12345 + max_m * 67890 + (int)(rho * 1000)));
            scores[seed] = sw_single_score(n, max_m, rho);
        }
        double sum = 0.0;
        for (int i = 0; i < SW_NUM_SEEDS; i++) sum += scores[i];

        if (result->count >= result->capacity) {
            result->capacity *= 2;
            result->m_values = realloc(result->m_values, (size_t)result->capacity * sizeof(int));
            result->scores = realloc(result->scores, (size_t)result->capacity * sizeof(double));
        }
        result->m_values[result->count] = max_m;
        result->scores[result->count] = sum / SW_NUM_SEEDS;
        result->count++;
    }

    printf("  SW (rho=%.2f) done for N=%d\n", rho, n);
}
