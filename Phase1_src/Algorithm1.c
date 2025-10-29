/* Algorithm1.c — Main Algorithm Controller and Verification
 *
 * DESCRIPTION:
 *   Main controller that receives (n,k) parameters and dispatches to
 *   appropriate algorithm branches (A, B, C, etc.). Also provides
 *   graph verification and validation functions.
 *
 * USAGE:
 *   This module coordinates between different algorithm branches
 *   and ensures graph properties are correctly verified.
 */

/**
 * Convenience function for getting just the lambda2 value.tly verified.
 */

#include <stdio.h>
#include <stdlib.h>
#include "../include/eigenvalue.h"
#include "../include/Special_Builder.h"
#include "../include/A_branch.h"
#include "../include/B1_branch.h"
#include "../include/B2_branch.h"
#include "../include/C2_branch.h"


/* Forward declarations for other branches (to be implemented) */
void c1_branch_main(int n, int k, int **adj_matrix);

/* ========================================================================
 * GRAPH VERIFICATION FUNCTIONS
 * ======================================================================== */

/**
 * Verify that all vertices in the graph have exactly degree k.
 * 
 * This function checks k-regularity by computing the degree of each vertex
 * and ensuring it equals the target degree k.
 * 
 * @param n Number of vertices
 * @param k Expected degree for each vertex
 * @param adj_matrix n×n adjacency matrix to verify
 * @param verbose If 1, print detailed information about degree mismatches
 * @return 1 if graph is k-regular, 0 if not k-regular
 */
int verify_k_regular(int n, int k, int **adj_matrix, int verbose) {
    if(!adj_matrix) {
        if(verbose) printf("Error: NULL adjacency matrix\n");
        return 0;
    }
    
    if(n <= 0 || k < 0) {
        if(verbose) printf("Error: Invalid parameters n=%d, k=%d\n", n, k);
        return 0;
    }
    
    int is_regular = 1;
    int total_edges = 0;
    
    if(verbose) {
        printf("=== K-REGULARITY VERIFICATION ===\n");
        printf("Graph: n=%d vertices, expected degree k=%d\n", n, k);
    }
    
    /* Check degree of each vertex */
    for(int i = 0; i < n; i++) {
        int degree = 0;
        
        /* Count edges for vertex i */
        for(int j = 0; j < n; j++) {
            if(adj_matrix[i][j]) {
                degree++;
            }
        }
        
        /* Check if degree matches expected k */
        if(degree != k) {
            is_regular = 0;
            if(verbose) {
                printf("❌ Vertex %d: degree=%d (expected %d) [MISMATCH]\n", i, degree, k);
            }
        } else if(verbose) {
            printf("✅ Vertex %d: degree=%d ✓\n", i, degree);
        }
        
        total_edges += degree;
    }
    
    /* Total edges should be n*k/2 (each edge counted twice) */
    int expected_total_edges = (n * k) / 2;
    int actual_total_edges = total_edges / 2;
    
    if(verbose) {
        printf("\n=== EDGE COUNT VERIFICATION ===\n");
        printf("Expected total edges: %d\n", expected_total_edges);
        printf("Actual total edges: %d\n", actual_total_edges);
        
        if(actual_total_edges == expected_total_edges) {
            printf("✅ Edge count matches expectation\n");
        } else {
            printf("❌ Edge count mismatch\n");
        }
    }
    
    /* Check if n*k is even (necessary condition for k-regular graphs) */
    if((n * k) % 2 != 0) {
        if(verbose) {
            printf("❌ Error: n*k = %d*%d = %d is odd (impossible for simple graphs)\n", n, k, n*k);
        }
        return 0;
    }
    
    /* Final result */
    if(verbose) {
        printf("\n=== FINAL RESULT ===\n");
        if(is_regular && actual_total_edges == expected_total_edges) {
            printf("✅ Graph is %d-regular ✓\n", k);
        } else {
            printf("❌ Graph is NOT %d-regular\n", k);
        }
        printf("========================\n\n");
    }
    
    return (is_regular && actual_total_edges == expected_total_edges);
}

/**
 * Quick verification without verbose output.
 * 
 * @param n Number of vertices  
 * @param k Expected degree
 * @param adj_matrix n×n adjacency matrix
 * @return 1 if k-regular, 0 otherwise
 */
int is_k_regular(int n, int k, int **adj_matrix) {
    return verify_k_regular(n, k, adj_matrix, 0);
}

/**
 * Verify basic graph properties (symmetry, no self-loops, binary values).
 * 
 * @param n Number of vertices
 * @param adj_matrix n×n adjacency matrix
 * @param verbose If 1, print detailed verification information
 * @return 1 if valid simple graph, 0 otherwise
 */
int verify_simple_graph(int n, int **adj_matrix, int verbose) {
    if(!adj_matrix) {
        if(verbose) printf("Error: NULL adjacency matrix\n");
        return 0;
    }
    
    int is_valid = 1;
    
    if(verbose) {
        printf("=== SIMPLE GRAPH VERIFICATION ===\n");
    }
    
    for(int i = 0; i < n; i++) {
        /* Check for self-loops */
        if(adj_matrix[i][i] != 0) {
            is_valid = 0;
            if(verbose) {
                printf("❌ Self-loop detected at vertex %d\n", i);
            }
        }
        
        for(int j = 0; j < n; j++) {
            /* Check for binary values */
            if(adj_matrix[i][j] != 0 && adj_matrix[i][j] != 1) {
                is_valid = 0;
                if(verbose) {
                    printf("❌ Non-binary value %d at position (%d,%d)\n", adj_matrix[i][j], i, j);
                }
            }
            
            /* Check for symmetry */
            if(adj_matrix[i][j] != adj_matrix[j][i]) {
                is_valid = 0;
                if(verbose) {
                    printf("❌ Asymmetry detected: adj[%d][%d]=%d but adj[%d][%d]=%d\n", 
                           i, j, adj_matrix[i][j], j, i, adj_matrix[j][i]);
                }
            }
        }
    }
    
    if(verbose) {
        if(is_valid) {
            printf("✅ Graph is a valid simple graph ✓\n");
        } else {
            printf("❌ Graph violates simple graph properties\n");
        }
        printf("==================================\n\n");
    }
    
    return is_valid;
}

/* ========================================================================
 * MAIN ALGORITHM DISPATCHER
 * ======================================================================== */

/**
 * Main algorithm entry point that dispatches to appropriate branches.
 * 
 * Routes (n,k) parameters to the correct algorithm branch based on:
 * - A: k==n-1 or k<3 (trivial cases)
 * - B1: EVEN n and k < n/2
 * - B2: EVEN n and k >= n/2  
 * - C1: ODD n and k < (n+1)/2
 * - C2: ODD n and k >= (n+1)/2
 * 
 * @param n Number of vertices
 * @param k Target degree
 * @param lambda2_out Output: computed algebraic connectivity
 * @return 1 if successful, 0 if failed
 */
int algorithm1_main(int n, int k, double *lambda2_out) {
    /* Input validation */
    if(n <= 0) {
        fprintf(stderr, "Error: Invalid n=%d (must be > 0)\n", n);
        return 0;
    }
    
    if(k < 0 || k >= n) {
        fprintf(stderr, "Error: Invalid k=%d (must be 0 <= k < n=%d)\n", k, n);
        return 0;
    }
    
    /* Check handshaking lemma: n*k must be even */
    if((n * k) % 2 != 0) {
        fprintf(stderr, "Error: n*k = %d*%d = %d is odd (impossible for simple graphs)\n", n, k, n*k);
        return 0;
    }
    
    printf("=== ALGORITHM 1 MAIN DISPATCHER ===\n");
    printf("Input: n=%d vertices, k=%d degree\n", n, k);
    
    /* Allocate adjacency matrix */
    int **adj_matrix = (int**)calloc(n, sizeof(int*));
    if(!adj_matrix) {
        fprintf(stderr, "Error: Failed to allocate adjacency matrix\n");
        return 0;
    }
    
    for(int i = 0; i < n; i++) {
        adj_matrix[i] = (int*)calloc(n, sizeof(int));
        if(!adj_matrix[i]) {
            fprintf(stderr, "Error: Failed to allocate adjacency matrix row %d\n", i);
            /* Cleanup allocated rows */
            for(int j = 0; j < i; j++) {
                free(adj_matrix[j]);
            }
            free(adj_matrix);
            return 0;
        }
    }
    
    /* ========================================================================
     * BRANCH DISPATCH LOGIC
     * ======================================================================== */
    
    printf("Determining algorithm branch...\n");
    
    /* Branch A: Trivial cases (k==n-1 or k<3) */
    if(k == n-1 || k < 3) {
        printf("→ Branch A: Trivial cases (k=%d)\n", k);
        a_branch_main(n, k, adj_matrix);
    }
    /* Branch B: Even n */
    else if(n % 2 == 0) {
        if(k < n/2) {
            printf("→ Branch B1: Even n=%d, k=%d < n/2=%d\n", n, k, n/2);
            b1_branch_main(n, k, adj_matrix);
        } else {
            printf("→ Branch B2: Even n=%d, k=%d >= n/2=%d\n", n, k, n/2);
            b2_branch_main(n, k, adj_matrix);
        }
    }
    /* Branch C: Odd n */
    else {
        int ceil_half = (n + 1) / 2;
        if(k < ceil_half) {
            printf("→ Branch C1: Odd n=%d, k=%d < ceil(n/2)=%d\n", n, k, ceil_half);
            c1_branch_main(n, k, adj_matrix);
        } else {
            printf("→ Branch C2: Odd n=%d, k=%d >= ceil(n/2)=%d\n", n, k, ceil_half);
            c2_branch_main(n, k, adj_matrix);
        }
    }
    
    /* ========================================================================
     * VERIFICATION PHASE
     * ======================================================================== */
    
    printf("\n=== VERIFICATION PHASE ===\n");
    
    /* Verify simple graph properties */
    if(!verify_simple_graph(n, adj_matrix, 1)) {
        fprintf(stderr, "❌ Graph verification failed: Invalid simple graph\n");
        goto cleanup_and_fail;
    }
    
    /* Verify k-regularity */
    if(!verify_k_regular(n, k, adj_matrix, 1)) {
        fprintf(stderr, "❌ Graph verification failed: Not %d-regular\n", k);
        goto cleanup_and_fail;
    }
    
    printf("✅ All verifications passed!\n");
    
    /* ========================================================================
     * ALGEBRAIC CONNECTIVITY COMPUTATION
     * ======================================================================== */
    
    printf("\n=== COMPUTING ALGEBRAIC CONNECTIVITY ===\n");
    printf("Using LAPACKE-based exact eigenvalue computation...\n");
    
    double lambda2 = compute_lambda2_from_adjacency_matrix(n, adj_matrix);
    
    if(lambda2 < 0) {
        fprintf(stderr, "❌ Eigenvalue computation failed\n");
        goto cleanup_and_fail;
    }
    
    /* SANITY CHECK: Upper bound for k-regular graphs */
    double theoretical_upper_bound = (double)(n * k) / (n - 1);
    if(lambda2 > theoretical_upper_bound + 1e-6) {  /* Allow small numerical tolerance */
        fprintf(stderr, "❌ SANITY CHECK FAILED: λ₂ = %.6f exceeds theoretical upper bound %.6f for k-regular graph\n", 
                lambda2, theoretical_upper_bound);
        fprintf(stderr, "    Upper bound: λ₂ ≤ nk/(n-1) = %d×%d/(%d-1) = %.6f\n", 
                n, k, n, theoretical_upper_bound);
        goto cleanup_and_fail;
    }
    
    printf("✅ Algebraic connectivity λ₂ = %.12f (upper bound: %.6f)\n", lambda2, theoretical_upper_bound);
    
    /* Validate eigenvalue makes sense */
    if(!validate_lambda2(lambda2, n, k)) {
        fprintf(stderr, "⚠️  Warning: Computed λ₂ = %.6f seems suspicious for n=%d, k=%d\n", lambda2, n, k);
    }
    
    *lambda2_out = lambda2;
    
    /* ========================================================================
     * SUCCESS CLEANUP
     * ======================================================================== */
    
    printf("\n=== ALGORITHM 1 COMPLETE ===\n");
    printf("✅ Successfully constructed %d-regular graph on %d vertices\n", k, n);
    printf("✅ Algebraic connectivity λ₂ = %.12f\n", lambda2);
    printf("=====================================\n\n");
    
    /* Cleanup adjacency matrix */
    for(int i = 0; i < n; i++) {
        free(adj_matrix[i]);
    }
    free(adj_matrix);
    
    return 1;

cleanup_and_fail:
    /* Cleanup adjacency matrix on failure */
    for(int i = 0; i < n; i++) {
        free(adj_matrix[i]);
    }
    free(adj_matrix);
    
    printf("❌ ALGORITHM 1 FAILED\n");
    printf("=====================================\n\n");
    
    return 0;
}

/* ========================================================================
 * PLACEHOLDER BRANCH IMPLEMENTATIONS
 * ======================================================================== */




/**
 * Convenience function for getting just the lambda2 value.
 */
double algorithm1_get_lambda2(int n, int k) {
    double lambda2;
    if(algorithm1_main(n, k, &lambda2)) {
        return lambda2;
    } else {
        return -1.0;  /* Error indicator */
    }
}