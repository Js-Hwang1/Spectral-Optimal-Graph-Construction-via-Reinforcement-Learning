/* Algorithm1.h — Main Algorithm Controller Header
 *
 * DESCRIPTION:
 *   Header file for the main algorithm controller and verification functions.
 *   Provides graph validation and k-regularity checking capabilities.
 */

#ifndef ALGORITHM1_H
#define ALGORITHM1_H

/* ========================================================================
 * FUNCTION DECLARATIONS
 * ======================================================================== */

/**
 * Verify that all vertices in the graph have exactly degree k.
 * Provides detailed verification with optional verbose output.
 */
int verify_k_regular(int n, int k, int **adj_matrix, int verbose);

/**
 * Quick k-regularity check without verbose output.
 */
int is_k_regular(int n, int k, int **adj_matrix);

/**
 * Verify basic graph properties (symmetry, no self-loops, binary values).
 */
int verify_simple_graph(int n, int **adj_matrix, int verbose);

/**
 * Main algorithm entry point that dispatches to appropriate branches.
 * Routes (n,k) to correct branch, verifies result, and computes λ₂.
 */
int algorithm1_main(int n, int k, double *lambda2_out);

/**
 * Convenience function for getting just the lambda2 value.
 */
double algorithm1_get_lambda2(int n, int k);

#endif /* ALGORITHM1_H */