/* C2_branch.h — Odd n, High Degree Construction Header
 *
 * DESCRIPTION:
 *   Header file for C2 branch handling odd n vertices with k ≥ (n+1)/2.
 *   Uses progressive construction from complete bipartite base case.
 */

#ifndef C2_BRANCH_H
#define C2_BRANCH_H

/* ========================================================================
 * FUNCTION DECLARATIONS
 * ======================================================================== */

/**
 * Main entry point for C2 branch: Odd n, k ≥ (n+1)/2.
 * Uses progressive construction starting from complete bipartite base case.
 */
void c2_branch_main(int n, int k, int **adj_matrix);

/**
 * Add exactly two edges to each vertex using deterministic chord-based approach.
 * Internal function for progressive degree increment (odd n requires even k).
 */
int add_two_degree_deterministic(int n, int **adj_matrix, int current_k);

/**
 * Build k-regular graph using progressive construction from base case.
 * Internal function implementing the core C2 algorithm.
 */
void build_c2_progressive(int n, int k, int **adj_matrix);

#endif /* C2_BRANCH_H */