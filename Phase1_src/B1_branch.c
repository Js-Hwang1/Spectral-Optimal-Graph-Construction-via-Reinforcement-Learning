/* B1_branch.c — Round-Robin Chord Construction for Even n, Low Degree (3 ≤ k < n/2)
 *
 * DESCRIPTION:
 *   Round-robin chord-based construction for even n graphs with low degree.
 *   Uses systematic chord distribution to achieve target degree k.
 *
 * STRATEGY:
 *   1. Start with no base graph (empty)
 *   2. Apply round-robin chord construction to build k-regular graph
 *   3. Distribute chords evenly across all vertices using round-robin pattern
 *   4. Ensure symmetric edge placement for undirected graphs
 *
 * ALGORITHM:
 *   For even n and 3 ≤ k < n/2:
 *   1. Initialize empty adjacency matrix
 *   2. Use round-robin chord placement:
 *      - For each vertex i, connect to vertices (i+d) mod n for chord distances d
 *      - Distribute chord distances evenly to achieve degree k
 *      - Skip self-loops and duplicate edges
 *   3. Verify degree constraints and connectivity
 *
 * USAGE:
 *   Called when: n % 2 == 0 && 3 <= k < n/2
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "Special_Builder.h"
#include "eigenvalue.h"
#include "Algorithm1.h"

// Forward declarations
static void build_optimized_chord_spacing_graph(int n, int k, int **adj_matrix);
static void add_chord_edge(int n, int i, int j, int **adj_matrix);
static int calculate_chord_distance(int i, int j, int n);
static int gcd(int a, int b);
static void get_optimal_chord_distances(int n, int k, int *distances);



/* ========================================================================
 * OPTIMIZED CHORD SPACING CONSTRUCTION
 * ======================================================================== */

/**
 * Calculate greatest common divisor using Euclidean algorithm.
 */
static int gcd(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

/**
 * Get optimal chord distances that maximize minimum spacing between consecutive chords.
 * Uses mathematical properties to select well-distributed chord distances.
 * Enhanced with tuning parameters for better low-k performance.
 * 
 * @param n Number of vertices
 * @param k Target degree
 * @param distances Output array of chord distances (caller allocates)
 */
static void get_optimal_chord_distances(int n, int k, int *distances) {
    if (k % 2 == 0) {
        // Even k: need k/2 distances
        int num_distances = k / 2;
        
        // Enhanced strategy for low k values (k <= 10)
        if (k <= 10) {
            // For low k, use more aggressive spacing and prefer prime distances
            // Try to use distances that are roughly Fibonacci-spaced for better spectral properties
            int fib_sequence[] = {1, 2, 3, 5, 8, 13, 21, 34, 55, 89};
            int fib_size = sizeof(fib_sequence) / sizeof(fib_sequence[0]);
            
            for (int i = 0; i < num_distances; i++) {
                int candidate;
                
                if (i < fib_size && fib_sequence[i] < n/2) {
                    // Try Fibonacci-based distance
                    candidate = fib_sequence[i];
                } else {
                    // Fall back to optimized spacing
                    int base_spacing = (n/2) / (num_distances + 1);
                    candidate = (i + 1) * base_spacing;
                }
                
                // Ensure candidate is valid and coprime to n when possible
                while (candidate >= n/2 || (gcd(candidate, n) > 2 && candidate < n/4)) {
                    candidate++;
                    if (candidate >= n/2) {
                        candidate = i + 1; // simple fallback
                        break;
                    }
                }
                distances[i] = candidate;
            }
        } else {
            // Original strategy for higher k values (works well for k >= 20)
            int spacing = (n/2) / num_distances;
            if (spacing < 1) spacing = 1;
            
            for (int i = 0; i < num_distances; i++) {
                int candidate = (i + 1) * spacing;
                
                // Ensure candidate is valid and coprime to n for better properties
                while (candidate >= n/2 || gcd(candidate, n) > 1) {
                    candidate++;
                    if (candidate >= n/2) {
                        candidate = i + 1; // fallback to simple increment
                        break;
                    }
                }
                distances[i] = candidate;
            }
        }
        
        // Validation: ensure no duplicates and all distances are valid
        for (int i = 0; i < num_distances; i++) {
            if (distances[i] <= 0 || distances[i] >= n/2) {
                distances[i] = i + 1;
            }
            // Check for duplicates
            for (int j = 0; j < i; j++) {
                if (distances[i] == distances[j]) {
                    distances[i] = i + 1;
                    break;
                }
            }
        }
        
    } else {
        // Odd k: need (k-1)/2 distances + special pattern
        int num_distances = (k - 1) / 2;
        
        // Enhanced strategy for low odd k values
        if (k <= 11) {
            // For low odd k, use golden ratio based spacing
            double golden_ratio = 1.618033988749;
            
            for (int i = 0; i < num_distances; i++) {
                int candidate = (int)((i + 1) * golden_ratio + 0.5);
                
                // Ensure valid range and avoid n/2 (reserved for perfect matching)
                while (candidate >= n/2 || candidate == n/2 || (gcd(candidate, n) > 2 && candidate < n/4)) {
                    candidate++;
                    if (candidate >= n/2) {
                        candidate = i + 1; // simple fallback
                        break;
                    }
                }
                distances[i] = candidate;
            }
        } else {
            // Original strategy for higher odd k values
            int spacing = (n/2) / (num_distances + 1); // +1 to leave room for n/2
            if (spacing < 1) spacing = 1;
            
            for (int i = 0; i < num_distances; i++) {
                int candidate = (i + 1) * spacing;
                
                // Avoid n/2 since that's reserved for the perfect matching
                while (candidate >= n/2 || candidate == n/2 || gcd(candidate, n) > 1) {
                    candidate++;
                    if (candidate >= n/2) {
                        candidate = i + 1; // fallback
                        break;
                    }
                }
                distances[i] = candidate;
            }
        }
        
        // Validation for odd k
        for (int i = 0; i < num_distances; i++) {
            if (distances[i] <= 0 || distances[i] >= n/2) {
                distances[i] = i + 1;
            }
            // Check for duplicates and n/2 conflict
            for (int j = 0; j < i; j++) {
                if (distances[i] == distances[j] || distances[i] == n/2) {
                    distances[i] = i + 1;
                    break;
                }
            }
        }
    }
}

/**
 * Build a k-regular graph using optimized chord spacing.
 * Maximizes minimum distance between consecutive chords for better spectral properties.
 * 
 * @param n Number of vertices (must be even)
 * @param k Target degree (3 ≤ k < n/2)
 * @param adj_matrix Output: adjacency matrix for the constructed graph
 */
static void build_optimized_chord_spacing_graph(int n, int k, int **adj_matrix) {
    // Initialize empty matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            adj_matrix[i][j] = 0;
        }
    }
    
    if (k % 2 == 0) {
        // Even k: use optimally spaced chord distances
        int num_distances = k / 2;
        int distances[num_distances];
        get_optimal_chord_distances(n, k, distances);
        
        for (int d_idx = 0; d_idx < num_distances; d_idx++) {
            int distance = distances[d_idx];
            for (int i = 0; i < n; i++) {
                int j = (i + distance) % n;
                add_chord_edge(n, i, j, adj_matrix);
            }
        }
    } else {
        // Odd k: use optimally spaced chord distances + perfect matching
        int num_distances = (k - 1) / 2;
        int distances[num_distances];
        get_optimal_chord_distances(n, k, distances);
        
        // Add the optimally spaced symmetric chords
        for (int d_idx = 0; d_idx < num_distances; d_idx++) {
            int distance = distances[d_idx];
            for (int i = 0; i < n; i++) {
                int j = (i + distance) % n;
                add_chord_edge(n, i, j, adj_matrix);
            }
        }
        
        // Add perfect matching at distance n/2 for the final edge
        for (int i = 0; i < n/2; i++) {
            int j = (i + n/2) % n;
            add_chord_edge(n, i, j, adj_matrix);
        }
    }
}

/**
 * Add a chord edge between vertices i and j.
 * Ensures symmetric placement for undirected graphs.
 * 
 * @param n Number of vertices
 * @param i First vertex
 * @param j Second vertex
 * @param adj_matrix Adjacency matrix to update
 */
static void add_chord_edge(int n, int i, int j, int **adj_matrix) {
    if (i != j && adj_matrix[i][j] == 0) {
        adj_matrix[i][j] = 1;
        adj_matrix[j][i] = 1;
    }
}

/* ========================================================================
 * BIPARTITE FOUNDATION + SYSTEMATIC REDUCTION CONSTRUCTION
 * ======================================================================== */

/**
 * Build a k-regular graph by starting with bipartite base and systematically removing edges.
 * 
 * @param n Number of vertices (must be even)
 * @param k Target degree (3 ≤ k < n/2)
 * @param adj_matrix Output: adjacency matrix for the constructed graph
 */
/**
 * Calculate chord distance between two vertices on a cycle.
 * 
 * @param i First vertex
 * @param j Second vertex
 * @param n Number of vertices
 * @return Minimum distance between i and j on the cycle
 */
static int calculate_chord_distance(int i, int j, int n) {
    int forward_dist = (j - i + n) % n;
    int backward_dist = (i - j + n) % n;
    return (forward_dist < backward_dist) ? forward_dist : backward_dist;
}

/* ========================================================================
 * MAIN B1 BRANCH ENTRY POINT
 * ======================================================================== */

/**
 * Main entry point for B1 branch.
 * Constructs k-regular graphs for even n with 3 ≤ k < n/2.
 * 
 * @param n Number of vertices (must be even)
 * @param k Target degree (3 ≤ k < n/2)
 * @param adj_matrix Output: n×n adjacency matrix (allocated by caller)
 */
void b1_branch_main(int n, int k, int **adj_matrix) {
    // Validate input parameters
    if (n % 2 != 0) {
        printf("Error: B1 branch requires even n. Got n=%d\n", n);
        return;
    }
    
    if (k < 3 || k >= n/2) {
        printf("Error: B1 branch requires 3 ≤ k < n/2. Got n=%d, k=%d\n", n, k);
        return;
    }
    
    if ((n * k) % 2 != 0) {
        printf("Error: Invalid (n,k) pair for simple graphs. n*k must be even. Got n=%d, k=%d\n", n, k);
        return;
    }
    
    // Build the graph using optimized chord spacing construction
    build_optimized_chord_spacing_graph(n, k, adj_matrix);
}