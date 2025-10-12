/* C2_branch.c — Deterministic Chord Length Construction for Odd n, High Degree (k ≥ ⌈n/2⌉)
 *
 * DESCRIPTION:
 *   Deterministic construction using chord length analogy for bipartite-like structure.
 *   Each node gets degree k (even) using systematic chord distance connections.
 *
 * STRATEGY:
 *   1. Maximum chord length: max_cl = ceil(n/2)
 *   2. For each node i, connect to i + max_cl + j and i + max_cl - 1 - j
 *   3. Use round-robin or priority-based edge assignment
 *   4. Systematic and deterministic approach ensuring even degree distribution
 *
 * ALGORITHM:
 *   For odd n and even k:
 *   1. Calculate max_cl = ceil(n/2)
 *   2. For each node 0 to n-1, attempt connections using chord distances
 *   3. Priority: node 0 gets full k edges first, then node 1, etc.
 *   4. Alternative: round-robin where each node gets 1 edge per round
 *
 * USAGE:
 *   Called when: n % 2 == 1 && k >= ceil(n/2)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "Special_Builder.h"
#include "eigenvalue.h"
#include "Algorithm1.h"

// Forward declarations
static void build_hybrid_c2_graph(int n, int k, int **adj_matrix);
static void build_npartite_c2_graph(int n, int k, int num_groups, int **adj_matrix);


/* ========================================================================
 * SYSTEMATIC DEGREE ENHANCEMENT FROM SPECIAL BIPARTITE BASE
 * ======================================================================== */

/**
 * Systematically enhance degrees from special bipartite base using circulant patterns.
 * This function takes the special bipartite case as a foundation and adds systematic
 * intra-group connections to reach higher target degrees.
 * 
 * @param n Number of vertices (must be odd)
 * @param k_base Base degree from special bipartite case 
 * @param k_target Target degree to reach
 * @param adj_matrix Input/Output: adjacency matrix with special bipartite base
 */

static void enhance_degrees_systematically(int n, int k_base, int k_target, int **adj_matrix) {
    if(k_target <= k_base) return;  // Nothing to enhance
    
    int size_a = (n + 1) / 2;  // Larger group: {0, 1, ..., size_a-1}
    int size_b = n - size_a;   // Smaller group: {size_a, ..., n-1}
    
    int degree_increment = k_target - k_base;  // 2, 4, 6, ...
    
    if(((n+1)/2) % 2 == 0) {
        /* Case 1: ((n+1)/2) is even
         * Group A is even-sized, Group B is odd-sized
         * Group A had perfect matching (+1), Group B had no extra connections
         * We need to add degree_increment more to both groups
         */
        // Group A enhancement: add circulant patterns starting from offset 2
        for(int j_a = 2; j_a <= degree_increment/2 + 1; j_a++) {
            for(int i = 0; i < size_a; i++) {
                int target = (i + j_a) % size_a;  // wrap around within Group A
                if(i != target && !adj_matrix[i][target]) {
                    adj_matrix[i][target] = 1;
                    adj_matrix[target][i] = 1;
                }
            }
        }
        
        // Group B enhancement: add circulant patterns starting from offset 1  
        for(int j_b = 1; j_b <= degree_increment/2; j_b++) {
            for(int i = size_a; i < n; i++) {
                // Isolate Group B indicies from 0 ... size_b - 1
                int local_i = i - size_a; // 0 ... size_b - 1 
                int target_local = (local_i + j_b) % size_b; // wrap around within Group B
                int target = size_a + target_local;
                if(i != target && !adj_matrix[i][target]) {
                    adj_matrix[i][target] = 1;
                    adj_matrix[target][i] = 1;
                }
            }
        }
        
    } else {
        /* Case 2: ((n+1)/2) is odd  
         * Group A is odd-sized, Group B is even-sized
         * Group A had ring (+2), Group B had perfect matching (+1)
         * We need to add degree_increment more to both groups
         */
        
        // Group A enhancement: add circulant patterns starting from offset 1
        for(int j = 2; j <= degree_increment/2 + 1; j++) {
            for(int i = 0; i < size_a; i++) {
                int target = (i + j) % size_a;
                if(i != target && !adj_matrix[i][target]) {
                    adj_matrix[i][target] = 1;
                    adj_matrix[target][i] = 1;
                }
            }
        }
        
        // Group B enhancement: add circulant patterns starting from offset 2
        for(int j = 2; j <= degree_increment/2 + 1; j++) {
            for(int i = size_a; i < n; i++) {
                int local_i = i - size_a;
                int target_local = (local_i + j) % size_b;
                int target = size_a + target_local;
                if(i != target && !adj_matrix[i][target]) {
                    adj_matrix[i][target] = 1;
                    adj_matrix[target][i] = 1;
                }
            }
        }
    }
}

/**
 * N-partite approach using general n-partite base + chord-based enhancement.
 * This function uses the general n-partite base and fills remaining degrees
 * with chord-based round-robin logic.
 * 
 * @param n Number of vertices (must be odd)
 * @param k Target degree 
 * @param num_groups Number of groups for n-partite division
 * @param adj_matrix Output: adjacency matrix
 */
static void build_npartite_c2_graph(int n, int k, int num_groups, int **adj_matrix) {
    // Step 1: Build n-partite base
    int base_degree = build_npartite_base(n, num_groups, k, adj_matrix);
    
    if(base_degree == 0) {
        // Fallback to bipartite if n-partite failed
        build_hybrid_c2_graph(n, k, adj_matrix);
        return;
    }
    
    // Step 2: Calculate remaining degrees needed
    int remaining_degree = k - base_degree;
    if(remaining_degree <= 0) return;  // Already at target degree
    
    // Step 3: Fill remaining degrees using round-robin chord logic
    int *degrees = (int*)calloc(n, sizeof(int));
    
    // Count current degrees
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(adj_matrix[i][j]) degrees[i]++;
        }
    }
    
    // Use round-robin approach to fill remaining degrees
    int max_cl = (n + 1) / 2;  // chord length for the full graph
    
    for(int round = 0; round < remaining_degree; round++) {
        for(int i = 0; i < n; i++) {
            if(degrees[i] >= k) continue;  // Node already has enough edges
            
            // Try to give node i one more edge this round
            int edge_added = 0;
            int j = 0;
            
            while(!edge_added && j < n) {
                // Chord-based connections
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
                
                // Safety check
                if(j >= n) break;
            }
        }
    }
    
    free(degrees);
}

/**
 * Hybrid approach: Bipartite foundation + chord-based intra-group enhancement.
 * This function creates the bipartite connections between groups A and B,
 * then uses chord-based logic within each group to fill remaining degrees.
 * 
 * @param n Number of vertices (must be odd)
 * @param k Target degree 
 * @param adj_matrix Output: adjacency matrix
 */
static void build_hybrid_c2_graph(int n, int k, int **adj_matrix) {
    // Clear matrix
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            adj_matrix[i][j] = 0;
        }
    }
    
    int size_a = (n + 1) / 2;  // Larger group: {0, 1, ..., size_a-1}
    int size_b = n - size_a;   // Smaller group: {size_a, ..., n-1}
    
    // Step 1: Create bipartite connections between Group A and Group B
    // Connect each vertex in Group A to all vertices in Group B
    for(int i = 0; i < size_a; i++) {
        for(int j = size_a; j < n; j++) {
            adj_matrix[i][j] = 1;
            adj_matrix[j][i] = 1;
        }
    }
    
    // Step 2: Calculate remaining degrees needed for each group
    int *degrees = (int*)calloc(n, sizeof(int));
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(adj_matrix[i][j]) degrees[i]++;
        }
    }
    
    // Step 3: Fill remaining degrees in Group A using round-robin logic
    int max_cl_a = (size_a + 1) / 2;  // Maximum chord length within Group A (proper ceil calculation)
    
    // Round-robin within Group A
    int remaining_rounds_a = k - size_b;  // Remaining degree needed after bipartite connections
    for(int round = 0; round < remaining_rounds_a; round++) {
        for(int i = 0; i < size_a; i++) {
            if(degrees[i] >= k) continue;  // Node already has enough edges
            
            // Try to give node i one more edge this round
            int edge_added = 0;
            int j = 0;
            
            while(!edge_added && j < size_a) {
                // Chord-based connections within Group A
                int pos_target = (i + max_cl_a + j) % size_a;
                int neg_target = (i + max_cl_a - 1 - j + size_a) % size_a;
                
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
                
                // Safety check
                if(j >= size_a) break;
            }
        }
    }
    
    // Step 4: Fill remaining degrees in Group B using round-robin logic
    int max_cl_b = (size_b + 1) / 2;  // Maximum chord length within Group B (proper ceil calculation)
    
    // Round-robin within Group B
    int remaining_rounds_b = k - size_a;  // Remaining degree needed after bipartite connections
    for(int round = 0; round < remaining_rounds_b; round++) {
        for(int i = size_a; i < n; i++) {
            if(degrees[i] >= k) continue;  // Node already has enough edges
            
            // Try to give node i one more edge this round
            int edge_added = 0;
            int j = 0;
            
            while(!edge_added && j < size_b) {
                // Map to local Group B indices for chord calculations
                int local_i = i - size_a;
                
                // Chord-based connections within Group B
                int pos_target_local = (local_i + max_cl_b + j) % size_b;
                int neg_target_local = (local_i + max_cl_b - 1 - j + size_b) % size_b;
                
                // Map back to global indices
                int pos_target = size_a + pos_target_local;
                int neg_target = size_a + neg_target_local;
                
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
                
                // Safety check
                if(j >= size_b) break;
            }
        }
    }
    
    free(degrees);
}

/**
 * Build systematic C2 graph using special bipartite foundation + circulant enhancement.
 * This is the deterministic approach that handles any k >= ceil_half systematically.
 * 
 * @param n Number of vertices (must be odd)
 * @param k Target degree 
 * @param adj_matrix Output: adjacency matrix
 */
static void build_systematic_c2_graph(int n, int k, int **adj_matrix) {
    int ceil_half = (n + 1) / 2;
    
    // Determine the appropriate special bipartite base case
    int k_base;
    if(((n+1)/2) % 2 == 0) {
        k_base = ceil_half;  // Case 1: even-sized larger group
    } else {
        k_base = ceil_half + 1;  // Case 2: odd-sized larger group  
    }
    
    // Build the special bipartite foundation
    build_complete_bipartite_special_case(n, k_base, adj_matrix);
    
    // Enhance degrees systematically to reach target k
    enhance_degrees_systematically(n, k_base, k, adj_matrix);
}


/* ========================================================================
 * MAIN C2 BRANCH ENTRY POINT
 * ======================================================================== */

/**
 * Main entry point for C2 branch: Odd n, k ≥ ⌈n/2⌉.
 * 
 * Uses deterministic chord length construction for bipartite-like structure.
 * 
 * @param n Number of vertices (must be odd)  
 * @param k Target degree (must be even, >= ceil(n/2))
 * @param adj_matrix Output: n×n adjacency matrix (allocated by caller)
 */
void build_c2_graph(int n, int k, int **adj_matrix) {
    
    /* Get the maximum beneficial partite number */
    int max_partite = is_npartite_beneficial(n, k);
    
    if(max_partite != 0) {
        /* Use n-partite approach with the returned partite number */
        build_npartite_c2_graph(n, k, max_partite, adj_matrix);
    } else {
        /* Fallback for cases where no n-partite structure is beneficial */
        build_systematic_c2_graph(n, k, adj_matrix);
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
        exit(1);
    }
    
    free(degrees);
}

/**
 * Main entry point for C2 branch: Odd n, k ≥ (n+1)/2.
 * 
 * @param n Number of vertices
 * @param k Target degree
 * @param adj_matrix Output: n×n adjacency matrix (allocated by caller)
 */
void c2_branch_main(int n, int k, int **adj_matrix) {
    /* Validate C2 branch conditions */
    if(n % 2 == 0) {
        fprintf(stderr, "Error: C2 branch called with even n=%d\n", n);
        exit(1);
    }
    
    int ceil_half = (n + 1) / 2;
    if(k < ceil_half) {
        fprintf(stderr, "Error: C2 branch called with k=%d < ceil(n/2)=%d\n", k, ceil_half);
        exit(1);
    }
    
    /* Execute deterministic chord construction */
    build_c2_graph(n, k, adj_matrix);
}















/* STASH ARTIFACTS*/

/*

 * Optimal chord construction: minimizes maximum center angle on unit circle.
 * Uses geometric optimization to find the best node placement.

static void build_chord_optimal_angle(int n, int k, int **adj_matrix) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            adj_matrix[i][j] = 0;
        }
    }
    
    int *degrees = (int*)calloc(n, sizeof(int));
    
    for(int node = 0; node < n; node++) {
        while(degrees[node] < k) {
            int i = node;  

            int *neighbors = (int*)malloc(n * sizeof(int));
            double *neighbor_angles = (double*)malloc(n * sizeof(double));
            int num_neighbors = 0;
            
            for(int j = 0; j < n; j++) {
                if(adj_matrix[i][j]) {
                    neighbors[num_neighbors] = j;
                    neighbor_angles[num_neighbors] = (2.0 * M_PI * j) / n;
                    num_neighbors++;
                }
            }
            
            int best_target = -1;
            
            if(num_neighbors == 0) {
                for(int j = 0; j < n; j++) {
                    if(j != i && degrees[j] < k) {
                        best_target = j;
                        break;
                    }
                }
            } else {
                for(int a = 0; a < num_neighbors; a++) {
                    for(int b = a + 1; b < num_neighbors; b++) {
                        if(neighbor_angles[a] > neighbor_angles[b]) {
                            double temp_angle = neighbor_angles[a];
                            neighbor_angles[a] = neighbor_angles[b];
                            neighbor_angles[b] = temp_angle;
                            
                            int temp_node = neighbors[a];
                            neighbors[a] = neighbors[b];
                            neighbors[b] = temp_node;
                        }
                    }
                }
                
                typedef struct {
                    double gap_size;
                    int gap_index;
                    double angle1;
                    double angle2;
                } gap_info_t;
                
                gap_info_t *gaps = (gap_info_t*)malloc(num_neighbors * sizeof(gap_info_t));
                
                for(int a = 0; a < num_neighbors; a++) {
                    int next = (a + 1) % num_neighbors;
                    double gap = neighbor_angles[next] - neighbor_angles[a];
                    if(gap < 0) gap += 2 * M_PI; 
                    
                    gaps[a].gap_size = gap;
                    gaps[a].gap_index = a;
                    gaps[a].angle1 = neighbor_angles[a];
                    gaps[a].angle2 = neighbor_angles[next];
                }
                
                for(int a = 0; a < num_neighbors; a++) {
                    for(int b = a + 1; b < num_neighbors; b++) {
                        if(gaps[a].gap_size < gaps[b].gap_size) {
                            gap_info_t temp = gaps[a];
                            gaps[a] = gaps[b];
                            gaps[b] = temp;
                        }
                    }
                }
                
                for(int gap_attempt = 0; gap_attempt < num_neighbors && best_target == -1; gap_attempt++) {
                    double angle1 = gaps[gap_attempt].angle1;
                    double angle2 = gaps[gap_attempt].angle2;
                    
                    double optimal_angle;
                    if(angle2 > angle1) {
                        optimal_angle = (angle1 + angle2) / 2.0;
                    } else {
                        optimal_angle = (angle1 + angle2 + 2 * M_PI) / 2.0;
                        if(optimal_angle >= 2 * M_PI) optimal_angle -= 2 * M_PI;
                    }
                    
                    int optimal_node = (int)((optimal_angle * n) / (2.0 * M_PI) + 0.5) % n;
                    
                    if(optimal_node != i && !adj_matrix[i][optimal_node] && degrees[optimal_node] < k) {
                        best_target = optimal_node;
                        break;
                    }
                    
                    int *candidates_in_gap = (int*)malloc(n * sizeof(int));
                    int num_candidates = 0;
                    
                    for(int candidate = 0; candidate < n; candidate++) {
                        if(candidate == i) continue;  
                        if(adj_matrix[i][candidate]) continue; 
                        if(degrees[candidate] >= k) continue;  
                        
                        double candidate_angle = (2.0 * M_PI * candidate) / n;
                        int candidate_in_gap = 0;
                        
                        if(angle1 < angle2) {
                            if(candidate_angle > angle1 && candidate_angle < angle2) {
                                candidate_in_gap = 1;
                            }
                        } else {
                            if(candidate_angle > angle1 || candidate_angle < angle2) {
                                candidate_in_gap = 1;
                            }
                        }
                        
                        if(candidate_in_gap) {
                            candidates_in_gap[num_candidates++] = candidate;
                        }
                    }
                    
                    for(int a = 0; a < num_candidates; a++) {
                        for(int b = a + 1; b < num_candidates; b++) {
                            if(candidates_in_gap[a] > candidates_in_gap[b]) {
                                int temp = candidates_in_gap[a];
                                candidates_in_gap[a] = candidates_in_gap[b];
                                candidates_in_gap[b] = temp;
                            }
                        }
                    }
                    
                   
                    if(num_candidates > 0) {
                        best_target = candidates_in_gap[0]; 
                    
                    free(candidates_in_gap);
                }
                
                free(gaps);
            }
            
            if(best_target != -1) {
                adj_matrix[i][best_target] = 1;
                adj_matrix[best_target][i] = 1;
                degrees[i]++;
                degrees[best_target]++;
            } else {
             
                free(neighbor_angles);
                break;  
            }
            
            free(neighbors);
            free(neighbor_angles);
        }
    }
    
    free(degrees);
}




* DETERMINISTIC CHORD LENGTH CONSTRUCTION

* Round-robin chord construction: each node gets 1 edge per round.
* More balanced edge assignment across all nodes.
static void build_chord_round_robin(int n, int k, int **adj_matrix) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            adj_matrix[i][j] = 0;
        }
    }
    
    int *degrees = (int*)calloc(n, sizeof(int));

    int max_cl = (n + 1) / 2;  
    
    for(int round = 0; round < k; round++) {
        for(int i = 0; i < n; i++) {
            if(degrees[i] >= k) continue;  
            
            int edge_added = 0;
            int j = 0;
            
            while(!edge_added && j < n) {
                int pos_target = (i + max_cl + j) % n;
                int neg_target = (i + max_cl - 1 - j + n) % n;
                
                if(pos_target != i && !adj_matrix[i][pos_target] && degrees[pos_target] < k) {
                    adj_matrix[i][pos_target] = 1;
                    adj_matrix[pos_target][i] = 1;
                    degrees[i]++;
                    degrees[pos_target]++;
                    edge_added = 1;
                }
                
                if(!edge_added && neg_target != i && neg_target != pos_target && 
                   !adj_matrix[i][neg_target] && degrees[neg_target] < k) {
                    adj_matrix[i][neg_target] = 1;
                    adj_matrix[neg_target][i] = 1;
                    degrees[i]++;
                    degrees[neg_target]++;
                    edge_added = 1;
                }
                
                j++;
                
                if(j >= n) {
                    break;
                }
            }
        }
    }
    
    free(degrees);
}

* Priority-based chord construction: each node gets full k edges before moving to next.
* Node 0 gets all k edges first, then node 1, etc.
static void build_chord_priority_based(int n, int k, int **adj_matrix) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            adj_matrix[i][j] = 0;
        }
    }
    
    int *degrees = (int*)calloc(n, sizeof(int));
    
    int max_cl = (n + 1) / 2;  
    
    for(int i = 0; i < n; i++) {
        int j = 0;
        while(degrees[i] < k && j < n) {
            int pos_target = (i + max_cl + j) % n;
            int neg_target = (i + max_cl - 1 - j + n) % n;
            
            if(pos_target != i && !adj_matrix[i][pos_target] && degrees[i] < k && degrees[pos_target] < k) {
                adj_matrix[i][pos_target] = 1;
                adj_matrix[pos_target][i] = 1;
                degrees[i]++;
                degrees[pos_target]++;
            }
            
            if(neg_target != i && neg_target != pos_target && 
               !adj_matrix[i][neg_target] && degrees[i] < k && degrees[neg_target] < k) {
                adj_matrix[i][neg_target] = 1;
                adj_matrix[neg_target][i] = 1;
                degrees[i]++;
                degrees[neg_target]++;
            }
            
            j++;
            
            if(j >= n) {
                break;
            }
        }
    }
    
    free(degrees);
}
*/