/* Special_Builder.c — Special Graph Construction Cases
 *
 * DESCRIPTION:
 *   Specialized builders for optimal special cases that provide 
 *   exact or near-optimal solutions for specific parameter ranges.
 *   All functions use adjacency matrix representation.
 *
 * SPECIAL CASES:
 *   1. Complete Bipartite (k = n/2): Perfect K_{n/2,n/2} with λ₂ = k
 *
 * USAGE:
 *   These functions are called from various algorithm branches when
 *   specific conditions trigger special case requirements.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "eigenvalue.h"

/* Special_Builder.c — Special Graph Construction Cases
 *
 * DESCRIPTION:
 *   Specialized builders for optimal special cases that provide 
 *   exact or near-optimal solutions for specific parameter ranges.
 *   All functions use adjacency matrix representation.
 *
 * SPECIAL CASES:
 *   1. Complete Bipartite (k = n/2 for even n, k = ⌈n/2⌉ for odd n)
 *
 * USAGE:
 *   These functions are called from various algorithm branches when
 *   specific conditions trigger special case requirements.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "eigenvalue.h"

/* ========================================================================
 * COMPLETE BIPARTITE CONSTRUCTION
 * ======================================================================== */

/**
 * Build complete bipartite graph for optimal special cases.
 * 
 * EVEN n, k = n/2: Complete bipartite K_{n/2,n/2}
 * - Structure: Two equal parts of size n/2
 * - Every vertex in one part connected to every vertex in the other part
 * - λ₂ = k = n/2 (proven optimal)
 * 
 * ODD n, k = ⌈n/2⌉ = (n+1)/2: Complete bipartite + perfect matching
 * - Structure: Parts of size (n+1)/2 and (n-1)/2
 * - Complete bipartite connectivity between parts
 * - Perfect matching within the larger part
 * - λ₂ = k-1 = (n-1)/2 (well-known result)
 * 
 * @param n Number of vertices
 * @param k Degree (k = n/2 for even n, k = (n+1)/2 for odd n)
 * @param adj_matrix Output: n×n adjacency matrix (allocated by caller, modified in-place)
 */
void build_complete_bipartite_special_case(int n, int k, int **adj_matrix) {
    /* Clear adjacency matrix */
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            adj_matrix[i][j] = 0;
        }
    }
    
    if(n % 2 == 0) {
        /* EVEN n case: k = n/2, Complete bipartite K_{n/2,n/2} */
        if(k != n/2) {
            fprintf(stderr, "Error: For even n=%d, expected k=%d but got k=%d\n", n, n/2, k);
            return;
        }
        
        int part_size = n / 2;
        
        /* Connect every vertex in part A (0..part_size-1) to every vertex in part B (part_size..n-1) */
        for(int i = 0; i < part_size; i++) {
            for(int j = part_size; j < n; j++) {
                adj_matrix[i][j] = 1;
                adj_matrix[j][i] = 1;
            }
        }
        
    } else {
        /* ODD n case: two cases: n+1 % 4 == 0 or n+1 % 4== 1
         */

        /* For odd n, create the best possible near-bipartite structure:
         * Split vertices into two groups: A = {0, 1, ..., size_a-1} and B = {size_a, ..., n-1}
         * Group A has size_a vertices, Group B has n-size_a vertices
         * Connect each vertex in A to ALL vertices in B, plus some strategic A-A connections
         */

        int size_a = (n + 1) / 2;  // Larger partition: (n+1)/2
        int size_b = n - size_a;   // Smaller partition: (n-1)/2

        /* Connect every vertex in A to every vertex in B */
        for(int i = 0; i < size_a; i++) {
            for(int j = size_a; j < n; j++) {
                adj_matrix[i][j] = 1;
                adj_matrix[j][i] = 1;
            }
        }

        int additional_a = k - size_b;  // Additional connections needed for Group A
        int additional_b = k - size_a;  // Additional connections needed for Group B

        int ceil_half = (n + 1) / 2;

        if(((n+1)/2)%2 == 0){
            if(k != ceil_half) {
                fprintf(stderr, "Error: For odd n=%d C2 special case, expected k=%d but got k=%d\n", 
                        n, ceil_half, k);
                return;
            }
            // In this case, Group A should be Even and Group B should be Odd, so Group A needs an extra degree, which can be done with a perfect matching
            
            /* Group A (odd size) needs +1 degree: use perfect matching */
            if(additional_a == 1) {
                /* Create perfect matching within Group A */
                for(int i = 0; i < size_a - 1; i += 2) {
                    adj_matrix[i][i+1] = 1;
                    adj_matrix[i+1][i] = 1;
                }
                /* If size_a is odd, connect the last vertex to the first */
                if(size_a % 2 == 1 && size_a > 1) {
                    int last = size_a - 1;
                    adj_matrix[last][0] = 1;
                    adj_matrix[0][last] = 1;
                }
            }
            
            /* Group B (even size) needs 0 additional connections - already at target degree */

        }else if(((n+1)/2)%2 == 1){
            if(k != ceil_half + 1) {
                fprintf(stderr, "Error: For odd n=%d C2 special case, expected k=%d but got k=%d\n", 
                        n, ceil_half+1 , k);
                return;
            }
            // In this case, Group A should be Odd and Group B should be Even, so Group B needs an extra degree, which can be done with a perfect matching, and Group A needs two more edges, so it can be done via giving a ring inside group A.
            
            /* Group A (Odd size) needs +2 degrees: use simple cycle (ring) */
            if(additional_a == 2) {
                /* Create a simple cycle within Group A: each vertex connects to next and previous */
                for(int i = 0; i < size_a; i++) {
                    int next = (i + 1) % size_a;
                    if(!adj_matrix[i][next]) {
                        adj_matrix[i][next] = 1;
                        adj_matrix[next][i] = 1;
                    }
                }
            }
            
            /* Group B (Even size) needs +1 degree: use perfect matching */
            if(additional_b == 1) {
                /* Create perfect matching within Group B */
                for(int i = size_a; i < size_a + size_b - 1; i += 2) {
                    adj_matrix[i][i+1] = 1;
                    adj_matrix[i+1][i] = 1;
                }
                /* If size_b is odd, connect the last B vertex to the first B vertex */
                if(size_b % 2 == 1 && size_b > 1) {
                    int last = size_a + size_b - 1;
                    int first = size_a;
                    adj_matrix[last][first] = 1;
                    adj_matrix[first][last] = 1;
                }
            }
        }
    }
    
    /* Verify k-regularity */
    for(int i = 0; i < n; i++) {
        int degree = 0;
        for(int j = 0; j < n; j++) {
            degree += adj_matrix[i][j];
        }
        if(degree != k) {
            fprintf(stderr, "Error: Vertex %d has degree %d, expected %d\n", i, degree, k);
        }
    }
}

/* ========================================================================
 * N-PARTITE CONSTRUCTION
 * ======================================================================== */

/**
 * Build complete n-partite graph foundation.
 * Creates n groups with complete connections between all pairs of groups.
 * This serves as a high-connectivity base that can be enhanced with additional patterns.
 * 
 * @param n Number of vertices
 * @param num_groups Number of groups to create (e.g., 3 for tri-partite, 4 for quad-partite)
 * @param k Target degree (must be >= (n-1) - max_group_size for n-partite to be possible)
 * @param adj_matrix Output: n×n adjacency matrix (allocated by caller, modified in-place)
 * @return Base degree achieved by n-partite connections (remaining degree = k - base_degree)
 */
int build_npartite_base(int n, int num_groups, int k, int **adj_matrix) {
    /* Clear adjacency matrix */
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            adj_matrix[i][j] = 0;
        }
    }
    
    if(num_groups < 2 || num_groups > n) {
        fprintf(stderr, "Error: Invalid num_groups=%d for n=%d\n", num_groups, n);
        return 0;
    }
    
    /* Calculate group sizes - distribute vertices as evenly as possible */
    int base_group_size = n / num_groups;
    int extra_vertices = n % num_groups;
    
    int *group_sizes = (int*)malloc(num_groups * sizeof(int));
    int *group_starts = (int*)malloc(num_groups * sizeof(int));
    
    /* Assign group sizes (some groups get +1 vertex if n doesn't divide evenly) */
    int total_assigned = 0;
    for(int g = 0; g < num_groups; g++) {
        group_sizes[g] = base_group_size + (g < extra_vertices ? 1 : 0);
        group_starts[g] = total_assigned;
        total_assigned += group_sizes[g];
    }
    
    /* Calculate minimum degree requirement for complete n-partite */
    int largest_group_size = base_group_size + (extra_vertices > 0 ? 1 : 0);
    int min_degree_npartite = n - largest_group_size;  /* connect to all vertices except own group */
    
    if(k < min_degree_npartite) {
        fprintf(stderr, "Warning: k=%d too low for %d-partite (min=%d), falling back to bipartite\n", 
                k, num_groups, min_degree_npartite);
        free(group_sizes);
        free(group_starts);
        return 0;  /* Indicate failure - caller should use different approach */
    }
    
    /* Create complete n-partite connections */
    for(int g1 = 0; g1 < num_groups; g1++) {
        for(int g2 = g1 + 1; g2 < num_groups; g2++) {
            /* Connect every vertex in group g1 to every vertex in group g2 */
            for(int i = group_starts[g1]; i < group_starts[g1] + group_sizes[g1]; i++) {
                for(int j = group_starts[g2]; j < group_starts[g2] + group_sizes[g2]; j++) {
                    adj_matrix[i][j] = 1;
                    adj_matrix[j][i] = 1;
                }
            }
        }
    }
    
    /* Calculate actual base degree achieved */
    int base_degree = min_degree_npartite;
    
    free(group_sizes);
    free(group_starts);
    
    return base_degree;
}

/**
 * Check for the maximum partite division that is beneficial FOR EVEN n.
 * For even n, we ensure that all groups have even sizes to avoid issues
 * with adding exactly 1 degree per node within groups.
 * 
 * @param n Number of vertices (must be even)
 * @param k Degree
 * @return Maximum beneficial partite number for even n
 *         Returns 0 if no n-partite structure is beneficial
 */
int is_npartite_beneficial_even_n(int n, int k) {
    if(n % 2 != 0) {
        fprintf(stderr, "Error: is_npartite_beneficial_even_n called with odd n=%d\n", n);
        return 0;
    }
    
    if(n < 4 || k < 2) return 0;  // Too small for any meaningful partite structure
    
    int max_partite = 0;
    int min_group_size = 2;  // For even n, minimum group size should be 2 (and even)
    
    // Start from maximum possible groups and work down to find the best
    int max_possible_groups = n / min_group_size;  // Can't have more groups than this
    
    for(int num_groups = max_possible_groups; num_groups >= 2; num_groups--) {
        // For even n, we want to ensure all groups have even sizes
        // This is only possible if n/num_groups results in even group sizes
        
        int base_group_size = n / num_groups;
        int extra_vertices = n % num_groups;
        
        // Check if we can create all-even-sized groups
        bool all_groups_even = true;
        
        // Group sizes will be: base_group_size for (num_groups - extra_vertices) groups
        //                     base_group_size + 1 for extra_vertices groups
        
        // Check if base_group_size is even
        if(base_group_size % 2 != 0) {
            all_groups_even = false;  // Base size is odd
        }
        
        // Check if base_group_size + 1 is even (for the extra_vertices groups)
        if(extra_vertices > 0 && (base_group_size + 1) % 2 != 0) {
            all_groups_even = false;  // Extended size is odd
        }
        
        // Skip this partitioning if it creates odd-sized groups
        if(!all_groups_even) continue;
        
        // The smallest group size will be base_group_size
        int smallest_group_size = base_group_size;
        
        // Check if groups are large enough to be meaningful
        if(smallest_group_size < min_group_size) continue;
        
        // For N-partite to be possible: k ≥ (n - smallest_group_size)
        int min_degree_required = n - smallest_group_size;
        
        if(k >= min_degree_required) {
            max_partite = num_groups;
            break;  // Found the maximum beneficial partite number
        }
    }
    
    // Special handling for bipartite case
    if(max_partite == 0) {
        // For even n, bipartite creates two groups of size n/2 each
        // Since n is even, n/2 is always integer, and we need n/2 to be even too
        int group_size = n / 2;
        if(group_size % 2 == 0) {  // Both groups will have even size
            if(k >= group_size) {  // k >= n/2 for bipartite to be possible
                max_partite = 2;  // Bipartite is possible with even-sized groups
            }
        }
    }
    
    return max_partite;
}

/**
 * Build complete n-partite graph foundation FOR EVEN n.
 * Ensures all groups have even sizes to avoid issues with degree-1 bumping.
 * 
 * @param n Number of vertices (must be even)
 * @param num_groups Number of groups to create
 * @param k Target degree
 * @param adj_matrix Output: n×n adjacency matrix
 * @return Base degree achieved by n-partite connections
 */
int build_npartite_base_even_n(int n, int num_groups, int k, int **adj_matrix) {
    if(n % 2 != 0) {
        fprintf(stderr, "Error: build_npartite_base_even_n called with odd n=%d\n", n);
        return 0;
    }
    
    /* Clear adjacency matrix */
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            adj_matrix[i][j] = 0;
        }
    }
    
    if(num_groups < 2 || num_groups > n) {
        fprintf(stderr, "Error: Invalid num_groups=%d for n=%d\n", num_groups, n);
        return 0;
    }
    
    /* Calculate group sizes - ensure all groups have even sizes */
    int base_group_size = n / num_groups;
    int extra_vertices = n % num_groups;
    
    // Verify that this partitioning creates only even-sized groups
    if(base_group_size % 2 != 0) {
        fprintf(stderr, "Error: Partitioning creates odd base group size %d\n", base_group_size);
        return 0;
    }
    
    if(extra_vertices > 0 && (base_group_size + 1) % 2 != 0) {
        fprintf(stderr, "Error: Partitioning creates odd extended group size %d\n", base_group_size + 1);
        return 0;
    }
    
    int *group_sizes = (int*)malloc(num_groups * sizeof(int));
    int *group_starts = (int*)malloc(num_groups * sizeof(int));
    
    /* Assign group sizes (all will be even) */
    int total_assigned = 0;
    for(int g = 0; g < num_groups; g++) {
        group_sizes[g] = base_group_size + (g < extra_vertices ? 1 : 0);
        group_starts[g] = total_assigned;
        total_assigned += group_sizes[g];
        
        // Verify that this group has even size
        if(group_sizes[g] % 2 != 0) {
            fprintf(stderr, "Error: Group %d has odd size %d\n", g, group_sizes[g]);
            free(group_sizes);
            free(group_starts);
            return 0;
        }
    }
    
    /* Calculate minimum degree requirement for complete n-partite */
    int largest_group_size = base_group_size + (extra_vertices > 0 ? 1 : 0);
    int min_degree_npartite = n - largest_group_size;
    
    if(k < min_degree_npartite) {
        fprintf(stderr, "Warning: k=%d too low for %d-partite (min=%d)\n", 
                k, num_groups, min_degree_npartite);
        free(group_sizes);
        free(group_starts);
        return 0;
    }
    
    /* Create complete n-partite connections */
    for(int g1 = 0; g1 < num_groups; g1++) {
        for(int g2 = g1 + 1; g2 < num_groups; g2++) {
            /* Connect every vertex in group g1 to every vertex in group g2 */
            for(int i = group_starts[g1]; i < group_starts[g1] + group_sizes[g1]; i++) {
                for(int j = group_starts[g2]; j < group_starts[g2] + group_sizes[g2]; j++) {
                    adj_matrix[i][j] = 1;
                    adj_matrix[j][i] = 1;
                }
            }
        }
    }
    
    /* Calculate actual base degree achieved */
    int base_degree = min_degree_npartite;
    
    free(group_sizes);
    free(group_starts);
    
    return base_degree;
}


/**
 * Check for the maximum partite division that is beneficial (ORIGINAL FUNCTION).
 * Returns the maximum number of groups that can provide meaningful benefit.
 * 
 * Logic: For N-partite to be possible, we partition n vertices into N groups,
 * find the smallest group size, and check if k ≥ (n - smallest_group_size).
 * This ensures every vertex can connect to all vertices outside its group.
 * 
 * @param n Number of vertices
 * @param k Degree
 * @return Maximum beneficial partite number (2 for bipartite, 3 for tripartite, etc.)
 *         Returns 0 if no n-partite structure is beneficial
 */
int is_npartite_beneficial(int n, int k) {
    if(n < 4 || k < 2) return 0;  // Too small for any meaningful partite structure
    
    int max_partite = 0;
    int min_group_size = 3;  // Minimum meaningful group size
    
    // Start from maximum possible groups and work down to find the best
    int max_possible_groups = n / min_group_size;  // Can't have more groups than this
    
    for(int num_groups = max_possible_groups; num_groups >= 2; num_groups--) {
        // Calculate group sizes when partitioning into num_groups
        int base_group_size = n / num_groups;
        int extra_vertices = n % num_groups;
        
        // The smallest group size will be base_group_size
        // (some groups get +1 vertex, but smallest is always base_group_size)
        int smallest_group_size = base_group_size;
        
        // Check if groups are large enough to be meaningful
        if(smallest_group_size < min_group_size) continue;
        
        // For N-partite to be possible: k ≥ (n - smallest_group_size)
        // This ensures every vertex can connect to all vertices outside its group
        int min_degree_required = n - smallest_group_size;
        
        if(k >= min_degree_required) {
            max_partite = num_groups;
            break;  // Found the maximum beneficial partite number
        }
    }
    
    // Special handling for edge cases - check if basic bipartite is at least possible
    if(max_partite == 0) {
        // For bipartite: partition into 2 groups of sizes ceil(n/2) and floor(n/2)
        int smaller_group = n / 2;  // This is the smallest group size for bipartite
        if(k >= (n - smaller_group)) {
            max_partite = 2;  // Bipartite is possible
        }
    }
    
    // Additional check: avoid bipartite when it leads to odd-group intra-connection issues
    if(max_partite == 2) {
        int group_size = n / 2;
        int inter_group_degree = n - group_size;  // Degree from inter-group connections
        int intra_group_degree_needed = k - inter_group_degree;  // Additional degree needed within groups
        
        // If each node needs exactly 1 more degree and group size is odd, bipartite will fail
        // because odd groups can't have every node get exactly 1 additional connection
        if(intra_group_degree_needed == 1 && group_size % 2 == 1) {
            max_partite = 0;  // Fall back to systematic approach
        }
    }
    
    return max_partite;
}
