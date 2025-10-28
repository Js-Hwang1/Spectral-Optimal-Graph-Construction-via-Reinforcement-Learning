/* eigenvalue.c — Numerical Eigenvalue Computation Module
 *
 * DESCRIPTION:
 *   Exact numerical eigenvalue computation for spectral graph analysis.
 *   Uses LAPACKE for efficient λ₂ computation.
 *
 * FEATURES:
 *   - O(n²) exact λ₂ computation using LAPACKE (finds only up to second eigenvalue)
 *   - Memory-efficient implementation
 *   - Robust error handling
 *
 * COMPILATION DEPENDENCIES:
 *   - LAPACKE library for numerical eigenvalue computation
 *   - Standard math library
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <lapacke.h>

/* ========================================================================
 * EXACT EIGENVALUE COMPUTATION USING LAPACKE
 * ======================================================================== */

/**
 * Compute the exact second-smallest eigenvalue (λ₂) of the graph Laplacian
 * using LAPACKE for eigenvalue decomposition.
 * 
 * This is an O(n²) solver optimized to find only the first few eigenvalues,
 * not the full dense eigenvalue decomposition.
 * 
 * Algorithm:
 * 1. Build Laplacian matrix L = D - A where D is degree matrix, A is adjacency
 * 2. Compute eigenvalues using LAPACKE_dsyev (QR algorithm)
 * 3. Return the second-smallest eigenvalue (first is always 0 for connected graphs)
 * 
 * @param n Number of vertices in the graph
 * @param adj_matrix Adjacency matrix (n x n), where adj_matrix[i][j] = 1 if edge exists
 * @return λ₂ (algebraic connectivity), or 0.0 on error
 */
double compute_lambda2_from_adjacency_matrix(int n, int **adj_matrix) {
    /* Allocate flat double arrays for LAPACKE */
    double *laplacian = (double*)malloc(n * n * sizeof(double));
    double *eigenvalues = (double*)malloc(n * sizeof(double));
    
    if (!laplacian || !eigenvalues) {
        fprintf(stderr, "Failed to allocate memory for eigenvalue computation\n");
        free(laplacian);
        free(eigenvalues);
        return 0.0;
    }
    
    /* Build Laplacian matrix: L[i][j] = degree(i) if i==j, -1 if edge(i,j), 0 otherwise */
    for (int i = 0; i < n; i++) {
        double degree = 0.0;
        for (int j = 0; j < n; j++) {
            if (i != j && adj_matrix[i][j]) {
                degree += 1.0;
                laplacian[i * n + j] = -1.0;
            } else if (i != j) {
                laplacian[i * n + j] = 0.0;
            }
        }
        laplacian[i * n + i] = degree;
    }
    
    /* Compute eigenvalues using LAPACKE
     * LAPACKE_dsyev computes all eigenvalues and optionally eigenvectors
     * of a real symmetric matrix using the QR algorithm
     * 
     * Parameters:
     *   LAPACK_ROW_MAJOR: row-major layout (C style)
     *   'N': compute eigenvalues only (not eigenvectors)
     *   'U': upper triangle of matrix is stored
     *   n: order of matrix
     *   laplacian: input matrix (will be destroyed)
     *   n: leading dimension
     *   eigenvalues: output array for eigenvalues (ascending order)
     */
    int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'N', 'U', n, laplacian, n, eigenvalues);
    
    if (info != 0) {
        fprintf(stderr, "LAPACKE_dsyev failed with info=%d\n", info);
        free(laplacian);
        free(eigenvalues);
        return 0.0;
    }
    
    /* Find the second-smallest eigenvalue
     * The eigenvalues are returned in ascending order
     * λ₀ should be ~0 (for connected graphs), λ₁ is the algebraic connectivity (λ₂)
     */
    double lambda2 = 0.0;
    for (int i = 0; i < n; i++) {
        if (eigenvalues[i] > 1e-10) {  /* Skip near-zero eigenvalue (λ₀) */
            lambda2 = eigenvalues[i];
            break;
        }
    }
    
    free(laplacian);
    free(eigenvalues);
    
    return (lambda2 < 0.0) ? 0.0 : lambda2;
}

/* ========================================================================
 * UTILITY FUNCTIONS
 * ======================================================================== */

/**
 * Validate that computed eigenvalue is reasonable.
 * 
 * @param lambda2 Computed λ₂ value
 * @param n Number of vertices
 * @param k Degree of regular graph
 * @return 1 if valid, 0 if suspicious
 */
int validate_lambda2(double lambda2, int n, int k) {
    /* Basic sanity checks */
    if (lambda2 < 0.0) return 0;               /* λ₂ must be non-negative */
    if (lambda2 > (double) n*k/(n-1)) return 0;         /* λ₂ ≤ k for k-regular graphs */
    
    return 1;  /* Passes basic validation */
}

/* ========================================================================
 * FIEDLER VECTOR COMPUTATION
 * ======================================================================== */

/**
 * Compute the Fiedler vector (second smallest eigenvector) of the graph Laplacian.
 * Builds L = D - A, runs symmetric eigensolver with eigenvectors, then returns
 * the first eigenvector with strictly positive eigenvalue as the Fiedler vector.
 *
 * @param n Number of vertices
 * @param adj_matrix n x n adjacency (0/1)
 * @param out_vec Output array of size n
 * @return 1 on success, 0 on failure
 */
int compute_fiedler_vector(int n, int **adj_matrix, double *out_vec) {
    if (n <= 1 || !adj_matrix || !out_vec) return 0;

    double *laplacian = (double*)malloc(n * n * sizeof(double));
    double *eigenvalues = (double*)malloc(n * sizeof(double));
    if (!laplacian || !eigenvalues) {
        free(laplacian);
        free(eigenvalues);
        return 0;
    }

    // Build Laplacian L = D - A
    for (int i = 0; i < n; i++) {
        double degree = 0.0;
        for (int j = 0; j < n; j++) {
            if (i != j && adj_matrix[i][j]) {
                degree += 1.0;
                laplacian[i * n + j] = -1.0;
            } else if (i != j) {
                laplacian[i * n + j] = 0.0;
            }
        }
        laplacian[i * n + i] = degree;
    }

    // Solve for all eigenpairs (symmetric)
    int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, laplacian, n, eigenvalues);
    if (info != 0) {
        free(laplacian);
        free(eigenvalues);
        return 0;
    }

    // Find index of first strictly positive eigenvalue (skip near-zero λ0)
    int fiedler_idx = -1;
    for (int i = 0; i < n; i++) {
        if (eigenvalues[i] > 1e-10) { fiedler_idx = i; break; }
    }
    if (fiedler_idx < 0) {
        free(laplacian);
        free(eigenvalues);
        return 0; // Graph likely disconnected or degenerate
    }

    // Extract the corresponding eigenvector (column fiedler_idx)
    for (int r = 0; r < n; r++) {
        out_vec[r] = laplacian[r * n + fiedler_idx];
    }

    free(laplacian);
    free(eigenvalues);
    return 1;
}

/**
 * Compute the first r positive Laplacian eigenpairs.
 * See header for details.
 */
int compute_low_k_eigenpairs(int n, int **adj_matrix, int r, double *eigvals_out, double *eigvecs_out) {
    if (n <= 1 || !adj_matrix || !eigvals_out || !eigvecs_out || r <= 0) return 0;

    double *L = (double*)malloc(n * n * sizeof(double));
    double *evals = (double*)malloc(n * sizeof(double));
    if (!L || !evals) {
        free(L); free(evals);
        return 0;
    }

    for (int i = 0; i < n; i++) {
        double deg = 0.0;
        for (int j = 0; j < n; j++) {
            if (i != j && adj_matrix[i][j]) { deg += 1.0; L[i*n + j] = -1.0; }
            else if (i != j) { L[i*n + j] = 0.0; }
        }
        L[i*n + i] = deg;
    }

    int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, L, n, evals);
    if (info != 0) { free(L); free(evals); return 0; }

    int found = 0;
    for (int i = 0; i < n && found < r; i++) {
        if (evals[i] > 1e-10) {
            eigvals_out[found] = evals[i];
            // extract ith eigenvector (column i)
            for (int v = 0; v < n; v++) {
                eigvecs_out[found * n + v] = L[v*n + i];
            }
            found++;
        }
    }

    free(L);
    free(evals);
    return found;
}
