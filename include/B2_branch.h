/* B2_branch.h — B2 Branch Header for Even n, High Degree
 *
 * DESCRIPTION:
 *   Header file for B2 branch handling even n with k >= n/2.
 *   Uses generalized n-partite construction with chord-based enhancement.
 */

#ifndef B2_BRANCH_H
#define B2_BRANCH_H

/* ========================================================================
 * FUNCTION DECLARATIONS
 * ======================================================================== */


/**
 * Branch validation and entry point with error checking.
 * Validates B2 branch conditions and calls build_b2_graph.
 */
void b2_branch_main(int n, int k, int **adj_matrix);

#endif /* B2_BRANCH_H */
