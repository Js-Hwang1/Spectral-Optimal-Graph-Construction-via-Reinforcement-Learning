/* A_branch.c — Weird Cases (k < 3 or k = n-1)
 *
 * DESCRIPTION:
 *   Handles the weird/trivial graph construction cases:
 *   - k < 3: Throws an error (not supported)
 *   - k = n-1: Constructs a complete graph
 *
 * USAGE:
 *   Called when: k < 3 OR k == n-1
 */

#include <stdio.h>
#include <stdlib.h>

/**
 * Main entry point for A branch: Weird cases.
 * 
 * @param n Number of vertices
 * @param k Target degree
 * @param adj_matrix Output: n×n adjacency matrix (allocated by caller)
 */
void a_branch_main(int n, int k, int **adj_matrix) {
    printf("A Branch: Handling weird case n=%d, k=%d\n", n, k);
    
    /* Handle k < 3 cases - throw error */
    if(k < 3) {
        fprintf(stderr, "Error: A branch does not support k < 3, got k=%d\n", k);
        exit(1);
    }
    
    /* Handle complete graph case k = n-1 */
    if(k == n-1) {
        printf("Constructing complete graph K_%d...\n", n);
        
        /* Connect every vertex to every other vertex */
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                if(i != j) {
                    adj_matrix[i][j] = 1;
                } else {
                    adj_matrix[i][j] = 0;  /* No self-loops */
                }
            }
        }
        
        printf("✅ Complete graph K_%d constructed\n", n);
    }
    else {
        fprintf(stderr, "Error: A branch received unexpected case k=%d with n=%d\n", k, n);
        exit(1);
    }
}