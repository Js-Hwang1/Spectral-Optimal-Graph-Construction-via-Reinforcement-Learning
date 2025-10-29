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

/**
 * Compute the Fiedler vector (second smallest eigenvector of the Laplacian).
 * Returns 1 on success and writes the vector into out_vec (size n), 0 on failure.
 * The vector is not normalized in any particular way beyond LAPACK's output.
 */
int compute_fiedler_vector(int n, int **adj_matrix, double *out_vec);

/**
 * Compute the first r positive Laplacian eigenpairs (smallest nonzero eigenvalues)
 * along with their eigenvectors. Returns the number of pairs found (<= r), or 0 on failure.
 * eigvals_out should have size at least r; eigvecs_out should be of size r*n
 * and will store vectors in row-major as eigvecs_out[p*n + i] = u_p[i].
 */
int compute_low_k_eigenpairs(int n, int **adj_matrix, int r, double *eigvals_out, double *eigvecs_out);

#endif /* EIGENVALUE_H */
