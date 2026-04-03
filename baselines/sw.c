/*
 * sw.c - Small World Network Baseline Implementation
 */

#include "sw.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

SWResult *sw_result_create(int capacity, double rho) {
    SWResult *result = malloc(sizeof(SWResult));
    if (!result) return NULL;

    result->m_values = malloc((size_t)capacity * sizeof(int));
    result->scores = malloc((size_t)capacity * sizeof(double));
    result->stds = malloc((size_t)capacity * sizeof(double));
    result->count = 0;
    result->capacity = capacity;
    result->rho = rho;

    if (!result->m_values || !result->scores || !result->stds) {
        free(result->m_values);
        free(result->scores);
        free(result->stds);
        free(result);
        return NULL;
    }

    return result;
}

void sw_result_free(SWResult *result) {
    if (result) {
        free(result->m_values);
        free(result->scores);
        free(result->stds);
        free(result);
    }
}

void sw_build(int n, int m, double rho, AdjMatrix *adj_out) {
    int max_m = n * (n - 1) / 2;
    if (m > max_m) m = max_m;

    /*
     * Build circulant graph hop-by-hop:
     *   hop 1: edges (i, i+1 mod n)  — the ring          -> n edges
     *   hop 2: edges (i, i+2 mod n)                       -> n edges
     *   ...
     *   hop h: edges (i, i+h mod n)                       -> n edges (or n/2 for h=n/2)
     *
     * Fill complete hops first, then add a partial hop to hit exactly m.
     */
    int curr = 0;
    int hop = 1;
    int max_hop = n / 2;

    /* Add complete hops while budget allows */
    while (hop <= max_hop) {
        int hop_edges = (hop == max_hop && n % 2 == 0) ? n / 2 : n;
        if (curr + hop_edges > m) break;
        for (int i = 0; i < n; i++) {
            int j = (i + hop) % n;
            if (!adj_get(adj_out, i, j)) {
                adj_set(adj_out, i, j, true);
                adj_set(adj_out, j, i, true);
                curr++;
            }
        }
        hop++;
    }

    /* Partial hop: add edges from the next hop until we reach m */
    if (curr < m && hop <= max_hop) {
        for (int i = 0; i < n && curr < m; i++) {
            int j = (i + hop) % n;
            if (!adj_get(adj_out, i, j)) {
                adj_set(adj_out, i, j, true);
                adj_set(adj_out, j, i, true);
                curr++;
            }
        }
    }

    /*
     * Watts-Strogatz rewiring (faithful to original):
     * Single pass, ordered by hop distance. For each hop h = 1..max_hop,
     * sweep nodes i = 0..n-1. For edge (i, i+h mod n), with probability
     * rho, keep i anchored and rewire (i+h) to a uniform random node
     * that is not i and not already a neighbor of i.
     */
    if (rho > 0.0) {
        for (int h = 1; h <= max_hop; h++) {
            for (int i = 0; i < n; i++) {
                int j = (i + h) % n;
                if (!adj_get(adj_out, i, j)) continue;
                if (rng_double() >= rho) continue;

                int nv = -1;
                for (int attempt = 0; attempt < 100; attempt++) {
                    int candidate = rng_int(n);
                    if (candidate != i && !adj_get(adj_out, i, candidate)) {
                        nv = candidate;
                        break;
                    }
                }

                if (nv >= 0) {
                    adj_set(adj_out, i, j, false);
                    adj_set(adj_out, j, i, false);
                    adj_set(adj_out, i, nv, true);
                    adj_set(adj_out, nv, i, true);
                }
            }
        }
    }
}

double sw_single_score(int n, int m, double rho) {
    AdjMatrix *adj = adj_create(n);
    sw_build(n, m, rho, adj);
    double score = compute_algebraic_connectivity(adj);
    adj_free(adj);
    return score;
}

static void sw_run_m(int n, int m, double rho, SWResult *result) {
    double scores[SW_NUM_SEEDS];

    for (int seed = 0; seed < SW_NUM_SEEDS; seed++) {
        rng_seed((uint64_t)(seed * 12345 + m * 67890 + (int)(rho * 1000)));
        scores[seed] = sw_single_score(n, m, rho);
    }

    /* Mean */
    double sum = 0.0;
    for (int i = 0; i < SW_NUM_SEEDS; i++) sum += scores[i];
    double mean = sum / SW_NUM_SEEDS;

    /* Std (population) */
    double sum_sq = 0.0;
    for (int i = 0; i < SW_NUM_SEEDS; i++) {
        double d = scores[i] - mean;
        sum_sq += d * d;
    }
    double std = sqrt(sum_sq / SW_NUM_SEEDS);

    /* Store */
    if (result->count >= result->capacity) {
        result->capacity *= 2;
        result->m_values = realloc(result->m_values, (size_t)result->capacity * sizeof(int));
        result->scores = realloc(result->scores, (size_t)result->capacity * sizeof(double));
        result->stds = realloc(result->stds, (size_t)result->capacity * sizeof(double));
    }
    result->m_values[result->count] = m;
    result->scores[result->count] = mean;
    result->stds[result->count] = std;
    result->count++;
}

void sw_run(int n, double rho, SWResult *result, int step) {
    int max_m = n * (n - 1) / 2;

    int last_m = -1;
    for (int m = n - 1; m <= max_m; m += step) {
        sw_run_m(n, m, rho, result);
        last_m = m;
    }

    if (last_m < max_m)
        sw_run_m(n, max_m, rho, result);

    printf("  SW (rho=%.2f) done for N=%d\n", rho, n);
}
