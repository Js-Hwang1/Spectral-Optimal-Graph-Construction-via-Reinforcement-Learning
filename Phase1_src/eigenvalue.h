/* eigenvalue.h — Numerical Eigenvalue Computation Module Header
 *
 * DESCRIPTION:
 *   Header file for numerical eigenvalue computation using LAPACKE.
 *   Provides exact λ₂ computation for general graphs.
 */

#ifndef EIGENVALUE_H
#define EIGENVALUE_H

/* ========================================================================
 * FUNCTION DECLARATIONS
 * ======================================================================== */

/**
 * Compute exact λ₂ from adjacency matrix using LAPACKE.
 * O(n²) complexity, optimized for finding only the second eigenvalue.
 */
double compute_lambda2_from_adjacency_matrix(int n, int **adj_matrix);

/**
 * Validate computed eigenvalue for sanity checking.
 */
int validate_lambda2(double lambda2, int n, int k);

#endif /* EIGENVALUE_H */