/*
 * er.h - Effective Resistance Greedy Algorithm
 *
 * Greedily adds edges that maximize effective resistance.
 * Complexity: O(M * N^3) due to pseudoinverse computation per edge.
 */

#ifndef ER_H
#define ER_H

#include "common.h"

typedef struct {
    int *m_values;
    double *scores;
    int count;
    int capacity;
} ERResult;

ERResult *er_result_create(int capacity);
void er_result_free(ERResult *result);

/*
 * Run the Effective Resistance greedy algorithm.
 * init: INIT_TREE (random spanning tree) or INIT_RING (cycle graph).
 */
void er_run(int n, ERResult *result, InitType init);

#endif /* ER_H */
