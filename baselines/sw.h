/*
 * sw.h - Small World Network Baseline
 *
 * Watts-Strogatz small-world model with rewiring.
 * Complexity: O(N^2) per construction.
 */

#ifndef SW_H
#define SW_H

#include "common.h"

#define SW_NUM_SEEDS 10

/* Result for a single rho value */
typedef struct {
    int *m_values;
    double *scores;  /* mean score for each m (averaged over seeds) */
    double *stds;    /* standard deviation for each m */
    int count;
    int capacity;
    double rho;
} SWResult;

SWResult *sw_result_create(int capacity, double rho);
void sw_result_free(SWResult *result);

/*
 * Run Small World benchmark for a specific rho value.
 * Runs SW_NUM_SEEDS seeds and computes mean.
 */
void sw_run(int n, double rho, SWResult *result, int step);

/*
 * Compute single small-world score (for internal use).
 */
double sw_single_score(int n, int m, double rho);

#endif /* SW_H */
