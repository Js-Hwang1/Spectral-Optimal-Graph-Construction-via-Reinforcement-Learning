/* B1_branch.h — B1 Branch Header for Even n, Low Degree
 *
 * DESCRIPTION:
 *   Header file for B1 branch handling even n with k < n/2.
 *   Uses round-robin chord construction for graph generation.
 */

#ifndef B1_BRANCH_H
#define B1_BRANCH_H

/* ========================================================================
 * FUNCTION DECLARATIONS
 * ======================================================================== */


/**
 * Branch validation and entry point with error checking.
 * Validates B1 branch conditions and calls build_b1_graph.
 */
void b1_branch_main(int n, int k, int **adj_matrix);

#endif /* B1_BRANCH_H */

