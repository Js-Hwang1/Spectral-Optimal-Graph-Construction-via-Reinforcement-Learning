/*
 * rd.h - Random d-Regular Graph Baseline
 *
 * Generates uniformly random simple d-regular graphs via the pairing model
 * (Bollobas, 1980). Valid only when d = 2m/n is an integer and n*d is even.
 *
 * Reference:
 *   B. Bollobas, "A probabilistic proof of an asymptotic formula for the
 *   number of labelled regular graphs", European J. Combin., 1(4), 1980.
 */

#ifndef RD_H
#define RD_H

#include "common.h"

/*
 * Check if (n, m) admits a d-regular graph.
 * Returns d >= 0 if valid, -1 otherwise.
 */
int rd_check(int n, int m);

/*
 * Build a uniformly random simple d-regular graph on n nodes.
 * adj_out must be pre-allocated with adj_create(n).
 * Returns 0 on success, -1 if (n,m) is not d-regular-able.
 */
int rd_build(int n, int m, AdjMatrix *adj_out);

#endif /* RD_H */
