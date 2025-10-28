/* B2_branch.c — Deterministic N-Partite Construction for Even n, High Degree (k ≥ n/2)
 *
 * DESCRIPTION:
 *   Deterministic construction using n-partite structure for even n graphs.
 *   Each node gets degree k using n-partite base + chord enhancement.
 *
 * STRATEGY:
 *   1. Use generalized n-partite base structure
 *   2. Find maximum beneficial partite number using is_npartite_beneficial()
 *   3. Apply chord-based round-robin enhancement to fill remaining degrees
 *   4. Systematic fallback for edge cases
 *
 * ALGORITHM:
 *   For even n and k >= n/2:
 *   1. Check maximum beneficial n-partite structure
 *   2. Build n-partite base with complete inter-group connections
 *   3. Use round-robin chord logic within groups to reach target degree k
 *   4. Fallback to systematic approach if no n-partite structure is beneficial
 *
 * USAGE:
 *   Called when: n % 2 == 0 && k >= n/2
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "Special_Builder.h"
#include "eigenvalue.h"
#include "Algorithm1.h"

// Forward declarations
static void build_hybrid_b2_graph(int n, int k, int **adj_matrix);
static void build_npartite_b2_graph(int n, int k, int num_groups, int **adj_matrix);


/* ========================================================================
 * SYSTEMATIC DEGREE ENHANCEMENT FROM SPECIAL N-PARTITE BASE
 * ======================================================================== */



/**
 * N-partite approach using generalized Special_Builder foundation + chord enhancement.
 * Uses build_npartite_base() for foundation, then applies chord-based round-robin
 * enhancement to fill remaining degrees within each group.
 * 
 * @param n Number of vertices (must be even)
 * @param k Target degree
 * @param num_groups Number of groups for n-partite division
 * @param adj_matrix Output: adjacency matrix
 */
static void build_npartite_b2_graph(int n, int k, int num_groups, int **adj_matrix) {
    // Step 1: Build n-partite base using EVEN-n specific function
    int base_degree = build_npartite_base_even_n(n, num_groups, k, adj_matrix);
    
    if(base_degree == 0) {
        // Fallback to bipartite if n-partite failed
        build_hybrid_b2_graph(n, k, adj_matrix);
        return;
    }
    
    // Step 2: Calculate remaining degrees needed
    int remaining_degree = k - base_degree;
    if(remaining_degree <= 0) return;  // Already at target degree
    
    // Step 3: Fill remaining degrees using round-robin chord logic WITHIN EACH GROUP
    int *degrees = (int*)calloc(n, sizeof(int));
    
    // Count current degrees
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(adj_matrix[i][j]) degrees[i]++;
        }
    }
    
    // Calculate group parameters for n-partite structure (MATCH build_npartite_base logic exactly)
    int base_group_size = n / num_groups;
    int extra_vertices = n % num_groups;
    printf("DEBUG: base_group_size=%d extra_vertices=%d\n", base_group_size, extra_vertices);
    
    // Calculate group starts and sizes exactly like build_npartite_base does
    int *group_sizes = (int*)malloc(num_groups * sizeof(int));
    int *group_starts = (int*)malloc(num_groups * sizeof(int));
    printf("DEBUG: malloc completed, calculating group assignments\n");
    
    int total_assigned = 0;
    for(int g = 0; g < num_groups; g++) {
        group_sizes[g] = base_group_size + (g < extra_vertices ? 1 : 0);
        group_starts[g] = total_assigned;
        total_assigned += group_sizes[g];
        printf("DEBUG: Group %d: start=%d size=%d (nodes %d to %d)\n", 
               g, group_starts[g], group_sizes[g], group_starts[g], group_starts[g] + group_sizes[g] - 1);
    }
    printf("DEBUG: Group assignment completed\n");
    
    printf("DEBUG: Starting round-robin enhancement for %d groups\n", num_groups);
    
    // Run round-robin within each group separately
    for(int group = 0; group < num_groups; group++) {
        int group_start = group_starts[group];
        int current_group_size = group_sizes[group];
        
        // Round-robin chord logic within this group only
        int max_cl_group = (current_group_size + 1) / 2;
        
        for(int round = 0; round < remaining_degree; round++) {
            int progress_made = 0;
            
            // First pass: C2-style round-robin within this group
            for(int local_i = 0; local_i < current_group_size; local_i++) {
                int i = group_start + local_i;
                if(degrees[i] >= k) continue;
                
                int edge_added = 0;
                int j = 0;
                
                while(!edge_added && j < current_group_size) {
                    int local_pos_target = (local_i + max_cl_group + j) % current_group_size;
                    int local_neg_target = (local_i + max_cl_group - 1 - j + current_group_size) % current_group_size;
                    
                    int pos_target = group_start + local_pos_target;
                    int neg_target = group_start + local_neg_target;
                    
                    if(pos_target != i && !adj_matrix[i][pos_target] && degrees[pos_target] < k) {
                        adj_matrix[i][pos_target] = 1;
                        adj_matrix[pos_target][i] = 1;
                        degrees[i]++;
                        degrees[pos_target]++;
                        edge_added = 1;
                        progress_made = 1;
                    }
                    
                    if(!edge_added && neg_target != i && neg_target != pos_target && 
                       !adj_matrix[i][neg_target] && degrees[neg_target] < k) {
                        adj_matrix[i][neg_target] = 1;
                        adj_matrix[neg_target][i] = 1;
                        degrees[i]++;
                        degrees[neg_target]++;
                        edge_added = 1;
                        progress_made = 1;
                    }
                    
                    j++;
                    if(j >= current_group_size) break;
                }
            }
            
            // If no progress in this round for this group, try direct connections within group
            if(!progress_made) {
                for(int local_i = 0; local_i < current_group_size && !progress_made; local_i++) {
                    int i = group_start + local_i;
                    if(degrees[i] >= k) continue;
                    
                    for(int local_target = 0; local_target < current_group_size; local_target++) {
                        int target = group_start + local_target;
                        if(target == i) continue;
                        if(adj_matrix[i][target]) continue;
                        if(degrees[target] >= k) continue;
                        
                        // Make intra-group connection
                        adj_matrix[i][target] = 1;
                        adj_matrix[target][i] = 1;
                        degrees[i]++;
                        degrees[target]++;
                        progress_made = 1;
                        break;
                    }
                }
            }
            
            if(!progress_made) break;  // No more connections possible in this group
        }
    }
    
    // Clean up allocated memory
    free(group_sizes);
    free(group_starts);
    free(degrees);
}

/**
 * Hybrid approach: Bipartite foundation + chord-based intra-group enhancement.
 * This function creates the bipartite connections between groups A and B,
 * then uses chord-based logic within each group to fill remaining degrees.
 * 
 * @param n Number of vertices (must be even)
 * @param k Target degree 
 * @param adj_matrix Output: adjacency matrix
 */
static void build_hybrid_b2_graph(int n, int k, int **adj_matrix) {
    // Clear matrix
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            adj_matrix[i][j] = 0;
        }
    }
    
    int size_a = n / 2;  // First group: {0, 1, ..., size_a-1}
    int size_b = n / 2;  // Second group: {size_a, ..., n-1}
    
    // Step 1: Create bipartite connections between Group A and Group B
    for(int i = 0; i < size_a; i++) {
        for(int j = size_a; j < n; j++) {
            adj_matrix[i][j] = 1;
            adj_matrix[j][i] = 1;
        }
    }
    
    // Step 2: Calculate current degrees and remaining needed
    int *degrees = (int*)calloc(n, sizeof(int));
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(adj_matrix[i][j]) degrees[i]++;
        }
    }
    
    int remaining_degree_needed = k - size_b;  // Each node needs this many more connections
    
    // Step 3: Add intra-group connections using circulant pattern
    if(remaining_degree_needed > 0) {
        // Fill Group A using chord lengths 1, 2, 3, ... up to remaining_degree_needed
        for(int cl = 1; cl <= remaining_degree_needed && cl <= size_a/2; cl++) {
            for(int i = 0; i < size_a; i++) {
                if(degrees[i] >= k) continue;
                
                int j = (i + cl) % size_a;
                if(!adj_matrix[i][j] && degrees[j] < k) {
                    adj_matrix[i][j] = 1;
                    adj_matrix[j][i] = 1;
                    degrees[i]++;
                    degrees[j]++;
                }
            }
        }
        
        // Fill Group B using chord lengths 1, 2, 3, ... up to remaining_degree_needed  
        for(int cl = 1; cl <= remaining_degree_needed && cl <= size_b/2; cl++) {
            for(int i = size_a; i < n; i++) {
                if(degrees[i] >= k) continue;
                
                int local_i = i - size_a;
                int local_j = (local_i + cl) % size_b;
                int j = size_a + local_j;
                
                if(!adj_matrix[i][j] && degrees[j] < k) {
                    adj_matrix[i][j] = 1;
                    adj_matrix[j][i] = 1;
                    degrees[i]++;
                    degrees[j]++;
                }
            }
        }
    }
    
    free(degrees);
}

/**
 * Build systematic B2 graph using bipartite foundation + circulant enhancement.
 * This is the deterministic approach that handles any k >= n/2 systematically.
 * 
 * @param n Number of vertices (must be even)
 * @param k Target degree 
 * @param adj_matrix Output: adjacency matrix
 */
static void build_systematic_b2_graph(int n, int k, int **adj_matrix) {
    // For even n, use bipartite base with k = n/2, then enhance systematically
    int k_base = n / 2;
    
    // Build the bipartite foundation using Special_Builder EVEN-n function
    build_npartite_base_even_n(n, 2, k_base, adj_matrix);
    
    // Fill remaining degrees using round-robin chord enhancement
    int remaining_degree = k - k_base;
    if(remaining_degree <= 0) return;
    
    int *degrees = (int*)calloc(n, sizeof(int));
    
    // Count current degrees
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(adj_matrix[i][j]) degrees[i]++;
        }
    }
    

    
    // Use round-robin approach to systematically add remaining edges
    int max_cl = (n + 1) / 2;
    
    for(int round = 0; round < remaining_degree; round++) {
        for(int i = 0; i < n; i++) {
            if(degrees[i] >= k) continue;
            
            // Try to give node i one more edge this round
            int edge_added = 0;
            int j = 0;
            
            while(!edge_added && j < n) {
                int pos_target = (i + max_cl + j) % n;
                int neg_target = (i + max_cl - 1 - j + n) % n;
                
                // Attempt positive direction connection
                if(pos_target != i && !adj_matrix[i][pos_target] && degrees[pos_target] < k) {
                    adj_matrix[i][pos_target] = 1;
                    adj_matrix[pos_target][i] = 1;
                    degrees[i]++;
                    degrees[pos_target]++;
                    edge_added = 1;
                }
                
                // Attempt negative direction connection if positive didn't work
                if(!edge_added && neg_target != i && neg_target != pos_target && 
                   !adj_matrix[i][neg_target] && degrees[neg_target] < k) {
                    adj_matrix[i][neg_target] = 1;
                    adj_matrix[neg_target][i] = 1;
                    degrees[i]++;
                    degrees[neg_target]++;
                    edge_added = 1;
                }
                
                j++;
                if(j >= n) break;
            }
            

        }
    }
    

    
    free(degrees);
}


/* ========================================================================
 * MAIN B2 BRANCH ENTRY POINT
 * ======================================================================== */

/**
 * Main entry point for B2 branch: Even n, k ≥ n/2.
 * 
 * Uses deterministic n-partite construction for even n graphs.
 * 
 * @param n Number of vertices (must be even)  
 * @param k Target degree (k >= n/2)
 * @param adj_matrix Output: n×n adjacency matrix (allocated by caller)
 */
void build_b2_graph(int n, int k, int **adj_matrix) {
    /* Get the maximum beneficial partite number FOR EVEN n */
    int max_partite = is_npartite_beneficial_even_n(n, k);

    if(max_partite != 0) {
        /* Use n-partite approach for very high k values */
        build_npartite_b2_graph(n, k, max_partite, adj_matrix);
    } else {
        /* Fallback to systematic approach */
        build_systematic_b2_graph(n, k, adj_matrix);
    }
    
    
    /* Final verification */
    int *degrees = (int*)calloc(n, sizeof(int));
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(adj_matrix[i][j]) degrees[i]++;
        }
    }
    
    int success = 1;
    for(int i = 0; i < n; i++) {
        if(degrees[i] != k) {
            success = 0;
            break;
        }
    }
    
    if(!success) {
        fprintf(stderr, "Error: Final verification failed for n=%d, k=%d\n", n, k);
        for(int i = 0; i < n; i++) {
            if(degrees[i] != k) {
                fprintf(stderr, "Vertex %d has degree %d, expected %d\n", i, degrees[i], k);
            }
        }
        free(degrees);
        exit(1);  // Restore exit for production use
    }
    
    free(degrees);
}

/**
 * Main entry point for B2 branch: Even n, k ≥ n/2.
 * 
 * @param n Number of vertices
 * @param k Target degree
 * @param adj_matrix Output: n×n adjacency matrix (allocated by caller)
 */
void b2_branch_main(int n, int k, int **adj_matrix) {
    
    /* Validate B2 branch conditions */
    if(n % 2 == 1) {
        fprintf(stderr, "Error: B2 branch called with odd n=%d\n", n);
        exit(1);
    }
    
    int half = n / 2;
    if(k < half) {
        fprintf(stderr, "Error: B2 branch called with k=%d < n/2=%d\n", k, half);
        exit(1);
    }
    
    /* Execute deterministic n-partite construction */
    build_b2_graph(n, k, adj_matrix);
}








