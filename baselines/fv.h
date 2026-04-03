/*
 * fv.h - Fiedler Vector Greedy Algorithm
 *
 * Greedily adds edges that maximize squared Fiedler vector difference.
 * Complexity: O(M * N^3) due to eigenvector computation per edge.
 */

#ifndef FV_H
#define FV_H

#include "common.h"

typedef struct {
    int *m_values;
    double *scores;
    int count;
    int capacity;
} FVResult;

FVResult *fv_result_create(int capacity);
void fv_result_free(FVResult *result);

/*
 * Run the Fiedler Vector greedy algorithm.
 * init: INIT_TREE (random spanning tree) or INIT_RING (cycle graph).
 */
void fv_run(int n, FVResult *result, InitType init);

#endif /* FV_H */
