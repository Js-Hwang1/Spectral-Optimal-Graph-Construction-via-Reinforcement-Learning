/* A_branch.h — A Branch Header for Trivial Cases
 *
 * DESCRIPTION:
 *   Header file for A branch handling trivial cases (k == n-1 or k < 3).
 *   Uses simple constructions for complete graphs or small graphs.
 */

#ifndef A_BRANCH_H
#define A_BRANCH_H

/* ========================================================================
 * FUNCTION DECLARATIONS
 * ======================================================================== */


/**
 * Branch validation and entry point with error checking.
 * Validates B2 branch conditions and calls build_b2_graph.
 */
void a_branch_main(int n, int k, int **adj_matrix);

#endif /* A_BRANCH_H */
