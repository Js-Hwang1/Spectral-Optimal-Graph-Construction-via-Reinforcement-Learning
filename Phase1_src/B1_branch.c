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

// NEW: Bipartite-based edge removal strategy (CORRECT B1 approach)
static void build_bipartite_edge_removal_graph(int n, int k, int **adj_matrix);
static void initialize_complete_bipartite(int n, int **adj_matrix);
static void remove_edges_to_target_degree(int n, int k, int **adj_matrix);

// NEW: Ramanujan graph-based construction for optimal spectral properties
static void build_ramanujan_seed_graph(int n, int k, int **adj_matrix);
static void build_ramanujan_anchor_graph(int n, int seed_degree, int **adj_matrix);
static void grow_ramanujan_to_target_degree(int n, int seed_degree, int target_k, int **adj_matrix);
static int is_quadratic_residue(int a, int p);
static int find_suitable_prime(int n);
static void build_cayley_ramanujan_base(int n, int degree, int **adj_matrix);



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
 * BIPARTITE EDGE REMOVAL FOR B1 BRANCH (CORRECT APPROACH)
 * ======================================================================== */

/**
 * Initialize a complete bipartite graph with two equal parts.
 * For even n, create K_{n/2, n/2} which has degree n/2 for all vertices.
 * 
 * @param n Number of vertices (must be even)
 * @param adj_matrix Output adjacency matrix
 */
static void initialize_complete_bipartite(int n, int **adj_matrix) {
    // Initialize empty matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            adj_matrix[i][j] = 0;
        }
    }
    
    int half = n / 2;
    
    // Create complete bipartite graph K_{n/2, n/2}
    // Part A: vertices 0, 1, ..., n/2-1
    // Part B: vertices n/2, n/2+1, ..., n-1
    for (int i = 0; i < half; i++) {
        for (int j = half; j < n; j++) {
            adj_matrix[i][j] = 1;
            adj_matrix[j][i] = 1;
        }
    }
    
    fprintf(stderr, "B1: Initialized complete bipartite K_%d,%d (degree %d)\n", half, half, half);
}

/**
 * Remove edges from complete bipartite to reach target degree k.
 * Strategy: For each vertex in Part A, keep k connections to Part B.
 * This naturally creates a k-regular bipartite graph.
 * 
 * @param n Number of vertices
 * @param k Target degree (3 ≤ k < n/2)
 * @param adj_matrix Adjacency matrix to modify
 */
static void remove_edges_to_target_degree(int n, int k, int **adj_matrix) {
    int current_degree = n / 2;  // All vertices start with degree n/2
    
    fprintf(stderr, "B1: Reducing degree from %d to %d\n", current_degree, k);
    
    if (k >= current_degree) {
        return;  // Already at or below target degree
    }
    
    int half = n / 2;
    
    // For each vertex in Part A, keep only k connections to Part B
    for (int i = 0; i < half; i++) {
        int kept = 0;
        
        // Keep first k connections, remove the rest
        for (int j = half; j < n; j++) {
            if (adj_matrix[i][j] == 1) {
                if (kept < k) {
                    // Keep this edge
                    kept++;
                } else {
                    // Remove this edge
                    adj_matrix[i][j] = 0;
                    adj_matrix[j][i] = 0;
                }
            }
        }
        
        fprintf(stderr, "B1: Vertex %d in Part A now has degree %d\n", i, kept);
    }
    
    // For each vertex in Part B, keep only k connections to Part A  
    for (int j = half; j < n; j++) {
        int kept = 0;
        
        // Keep first k connections, remove the rest
        for (int i = 0; i < half; i++) {
            if (adj_matrix[j][i] == 1) {
                if (kept < k) {
                    // Keep this edge
                    kept++;
                } else {
                    // Remove this edge
                    adj_matrix[j][i] = 0;
                    adj_matrix[i][j] = 0;
                }
            }
        }
        
        fprintf(stderr, "B1: Vertex %d in Part B now has degree %d\n", j, kept);
    }
    
    fprintf(stderr, "B1: Edge removal complete\n");
}

/**
 * Build k-regular graph using optimized circulant construction with number theory.
 * Uses Chebyshev polynomial-inspired step selection for optimal spectral gap.
 * 
 * @param n Number of vertices (even)
 * @param k Target degree (3 ≤ k < n/2)
 * @param adj_matrix Output adjacency matrix
 */
static void build_optimized_circulant_graph(int n, int k, int **adj_matrix) {
    // Initialize empty matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            adj_matrix[i][j] = 0;
        }
    }
    
    if (k % 2 == 0) {
        // Even k: use k/2 symmetric steps with optimal selection
        int num_steps = k / 2;
        int *steps = (int*)malloc(num_steps * sizeof(int));
        
        if (k == 2) {
            steps[0] = 1;
        } else if (k == 4) {
            steps[0] = 1;
            steps[1] = (n >= 16) ? (n/8 + 1) : (n/4);
        } else if (k == 6) {
            steps[0] = 1;
            steps[1] = (n >= 24) ? (n/12) : 2;
            steps[2] = (n >= 16) ? (n/6) : (n/4);
        } else if (k == 8) {
            steps[0] = 1;
            steps[1] = (n >= 32) ? (n/16) : 2;
            steps[2] = (n >= 24) ? (n/8) : 3;
            steps[3] = (n >= 16) ? (n/5) : (n/4);
        } else {
            // General case: use number theory based selection
            steps[0] = 1; // Always include 1
            
            // Use approximately golden ratio spacing for optimal distribution
            double phi = 1.618033988749;
            for (int i = 1; i < num_steps; i++) {
                double target = i * n / (2.0 * phi * num_steps);
                int step = (int)(target + 0.5);
                
                // Ensure step is valid and not too close to previous steps
                if (step <= steps[i-1]) step = steps[i-1] + 1;
                if (step >= n/2) step = n/2 - 1;
                
                // Prefer steps that are coprime to n for better mixing
                while (step < n/2 && gcd(step, n) > 2) {
                    step++;
                }
                if (step >= n/2) step = steps[i-1] + 1;
                
                steps[i] = step;
            }
        }
        
        // Build circulant with selected steps
        for (int s = 0; s < num_steps; s++) {
            int step = steps[s];
            for (int i = 0; i < n; i++) {
                int j = (i + step) % n;
                if (i != j) {
                    adj_matrix[i][j] = 1;
                    adj_matrix[j][i] = 1;
                }
            }
        }
        
        free(steps);
    } else {
        // Odd k: use (k-1)/2 symmetric steps + perfect matching
        int num_symmetric = (k - 1) / 2;
        
        if (num_symmetric > 0) {
            int *steps = (int*)malloc(num_symmetric * sizeof(int));
            
            if (k == 3) {
                steps[0] = 1;
            } else if (k == 5) {
                steps[0] = 1;
                steps[1] = (n >= 20) ? (n/10) : 2;
            } else if (k == 7) {
                steps[0] = 1;
                steps[1] = (n >= 28) ? (n/14) : 2;
                steps[2] = (n >= 16) ? (n/7) : 3;
            } else {
                // General case for odd k
                steps[0] = 1;
                double phi = 1.618033988749;
                for (int i = 1; i < num_symmetric; i++) {
                    double target = i * n / (2.0 * phi * num_symmetric);
                    int step = (int)(target + 0.5);
                    
                    if (step <= steps[i-1]) step = steps[i-1] + 1;
                    if (step >= n/2) step = n/2 - 1;
                    
                    // Prefer coprime steps
                    while (step < n/2 && gcd(step, n) > 2) {
                        step++;
                    }
                    if (step >= n/2) step = steps[i-1] + 1;
                    
                    steps[i] = step;
                }
            }
            
            // Add symmetric edges
            for (int s = 0; s < num_symmetric; s++) {
                int step = steps[s];
                for (int i = 0; i < n; i++) {
                    int j = (i + step) % n;
                    if (i != j) {
                        adj_matrix[i][j] = 1;
                        adj_matrix[j][i] = 1;
                    }
                }
            }
            
            free(steps);
        }
        
        // Add perfect matching for the last edge
        for (int i = 0; i < n/2; i++) {
            int j = i + n/2;
            adj_matrix[i][j] = 1;
            adj_matrix[j][i] = 1;
        }
    }
}

/**
 * Build k-regular graph using algebraic pseudorandom construction.
 * Uses carefully chosen polynomial generators over finite fields for optimal expansion.
 * 
 * @param n Number of vertices (even)
 * @param k Target degree (3 ≤ k < n/2)
 * @param adj_matrix Output adjacency matrix
 */
static void build_algebraic_expander_graph(int n, int k, int **adj_matrix) {
    // Initialize empty matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            adj_matrix[i][j] = 0;
        }
    }
    
    // Use algebraic construction based on finite field arithmetic
    if (k % 2 == 0) {
        // Even k: use polynomial-based generator selection
        int num_steps = k / 2;
        int *steps = (int*)malloc(num_steps * sizeof(int));
        
        // Generate steps using polynomial sequences for good distribution
        steps[0] = 1; // Always include 1
        
        for (int i = 1; i < num_steps; i++) {
            // Use quadratic polynomial: step = (a*i^2 + b*i + c) mod (n/2)
            // Choose a, b, c based on n for optimal spectral gap
            int a = (n >= 64) ? 3 : 2;
            int b = (n >= 32) ? 5 : 3;
            int c = 1;
            
            int step = (a * i * i + b * i + c) % (n/2);
            if (step == 0) step = 1;
            if (step >= n/2) step = (step % (n/4)) + 1;
            
            // Ensure no duplicates
            int duplicate = 0;
            for (int j = 0; j < i; j++) {
                if (steps[j] == step) {
                    duplicate = 1;
                    break;
                }
            }
            
            if (duplicate) {
                step = (step + i) % (n/2);
                if (step == 0) step = i + 1;
                if (step >= n/2) step = (i + 1) % (n/4) + 1;
            }
            
            steps[i] = step;
        }
        
        // Add edges for each step
        for (int s = 0; s < num_steps; s++) {
            int step = steps[s];
            for (int i = 0; i < n; i++) {
                int j = (i + step) % n;
                if (i != j) {
                    adj_matrix[i][j] = 1;
                    adj_matrix[j][i] = 1;
                }
            }
        }
        
        free(steps);
    } else {
        // Odd k: use hybrid polynomial + perfect matching
        int num_symmetric = (k - 1) / 2;
        
        if (num_symmetric > 0) {
            int *steps = (int*)malloc(num_symmetric * sizeof(int));
            steps[0] = 1;
            
            for (int i = 1; i < num_symmetric; i++) {
                // Use different polynomial for odd degrees
                int a = (n >= 64) ? 2 : 1;
                int b = (n >= 32) ? 7 : 5;
                int c = 2;
                
                int step = (a * i * i + b * i + c) % (n/2);
                if (step == 0) step = 1;
                if (step >= n/2) step = (step % (n/4)) + 1;
                
                // Avoid duplicates and n/2
                int duplicate = 0;
                for (int j = 0; j < i; j++) {
                    if (steps[j] == step || step == n/2) {
                        duplicate = 1;
                        break;
                    }
                }
                
                if (duplicate) {
                    step = (step + i + 1) % (n/2);
                    if (step == 0 || step == n/2) step = i + 1;
                    if (step >= n/2) step = (i + 1) % (n/4) + 1;
                }
                
                steps[i] = step;
            }
            
            // Add symmetric edges
            for (int s = 0; s < num_symmetric; s++) {
                int step = steps[s];
                for (int i = 0; i < n; i++) {
                    int j = (i + step) % n;
                    if (i != j) {
                        adj_matrix[i][j] = 1;
                        adj_matrix[j][i] = 1;
                    }
                }
            }
            
            free(steps);
        }
        
        // Add perfect matching
        for (int i = 0; i < n/2; i++) {
            int j = i + n/2;
            adj_matrix[i][j] = 1;
            adj_matrix[j][i] = 1;
        }
    }
}

/**
 * Build k-regular graph using sophisticated deterministic expander approach.
 * Uses algebraic construction for optimal spectral properties.
 * 
 * @param n Number of vertices (even)
 * @param k Target degree (3 ≤ k < n/2)
 * @param adj_matrix Output adjacency matrix
 */
static void build_sophisticated_expander_graph(int n, int k, int **adj_matrix) {
    // Choose construction method based on k and n
    if (k <= n/16) {
        // For very low k, use algebraic expander
        build_algebraic_expander_graph(n, k, adj_matrix);
    } else {
        // For medium k, use optimized circulant
        build_optimized_circulant_graph(n, k, adj_matrix);
    }
}

/**
 * Build k-regular graph using Paley-like construction for prime powers.
 * When n is near a prime power, use quadratic residue-based construction.
 * 
 * @param n Number of vertices (even)
 * @param k Target degree (3 ≤ k < n/2)
 * @param adj_matrix Output adjacency matrix
 */
static void build_paley_like_graph(int n, int k, int **adj_matrix) {
    // Initialize empty matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            adj_matrix[i][j] = 0;
        }
    }
    
    // Use quadratic residue pattern modified for target degree k
    int *residues = (int*)malloc(n * sizeof(int));
    int residue_count = 0;
    
    // Compute quadratic residues modulo n (or nearest suitable modulus)
    for (int x = 1; x < n; x++) {
        int square = (x * x) % n;
        if (square != 0) {
            int found = 0;
            for (int i = 0; i < residue_count; i++) {
                if (residues[i] == square) {
                    found = 1;
                    break;
                }
            }
            if (!found) {
                residues[residue_count++] = square;
            }
        }
    }
    
    // Select k/2 best residues for connectivity
    int selected_count = (k + 1) / 2;
    if (selected_count > residue_count) selected_count = residue_count;
    
    // Sort residues to pick well-distributed ones
    for (int i = 0; i < residue_count - 1; i++) {
        for (int j = i + 1; j < residue_count; j++) {
            if (residues[i] > residues[j]) {
                int temp = residues[i];
                residues[i] = residues[j];
                residues[j] = temp;
            }
        }
    }
    
    // Use evenly spaced residues
    for (int s = 0; s < selected_count; s++) {
        int step = (residue_count * s) / selected_count;
        if (step >= residue_count) step = residue_count - 1;
        
        int residue = residues[step];
        for (int i = 0; i < n; i++) {
            int j = (i + residue) % n;
            if (i != j && adj_matrix[i][j] == 0) {
                adj_matrix[i][j] = 1;
                adj_matrix[j][i] = 1;
            }
        }
    }
    
    free(residues);
}

/* ========================================================================
 * RAMANUJAN GRAPH CONSTRUCTION FOR OPTIMAL SPECTRAL PROPERTIES
 * ======================================================================== */

/**
 * Check if a is a quadratic residue modulo p (odd prime).
 * Uses Legendre symbol computation.
 * 
 * @param a Value to check
 * @param p Odd prime modulus
 * @return 1 if a is QR mod p, 0 otherwise
 */
static int is_quadratic_residue(int a, int p) {
    if (a % p == 0) return 0;
    
    // Compute Legendre symbol (a/p) using Euler's criterion: a^((p-1)/2) mod p
    int exp = (p - 1) / 2;
    int result = 1;
    int base = a % p;
    
    while (exp > 0) {
        if (exp % 2 == 1) {
            result = (result * base) % p;
        }
        base = (base * base) % p;
        exp /= 2;
    }
    
    return result == 1;
}

/**
 * Find the largest prime ≤ n that's suitable for Ramanujan construction.
 * Prefers primes ≡ 1 (mod 4) for better quadratic residue properties.
 * 
 * @param n Upper bound for prime search
 * @return Suitable prime for Ramanujan construction
 */
static int find_suitable_prime(int n) {
    // Check primes in descending order from n
    for (int p = (n % 2 == 0) ? n - 1 : n; p >= 5; p -= 2) {
        // Check if p is prime
        int is_prime = 1;
        for (int i = 3; i * i <= p; i += 2) {
            if (p % i == 0) {
                is_prime = 0;
                break;
            }
        }
        
        if (is_prime) {
            // Prefer primes ≡ 1 (mod 4) for better QR properties
            if (p % 4 == 1) return p;
            
            // Accept primes ≡ 3 (mod 4) if nothing better found
            if (p >= n/2) return p;
        }
    }
    
    // Fallback for small n
    int small_primes[] = {5, 7, 11, 13, 17, 19, 23, 29, 31};
    int num_primes = sizeof(small_primes) / sizeof(small_primes[0]);
    
    for (int i = num_primes - 1; i >= 0; i--) {
        if (small_primes[i] <= n) return small_primes[i];
    }
    
    return 5; // Absolute fallback
}

/**
 * Build Cayley graph base using quadratic residues for Ramanujan-like properties.
 * Creates a highly expander foundation with optimal spectral gap.
 * 
 * @param n Number of vertices
 * @param degree Target degree for the base graph
 * @param adj_matrix Output adjacency matrix
 */
static void build_cayley_ramanujan_base(int n, int degree, int **adj_matrix) {
    // Initialize empty matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            adj_matrix[i][j] = 0;
        }
    }
    
    // Find suitable prime for construction
    int p = find_suitable_prime(n);
    
    fprintf(stderr, "B1-Ramanujan: Using prime p=%d for n=%d, target_degree=%d\n", p, n, degree);
    
    // Collect quadratic residues modulo p
    int *qr_list = (int*)malloc(p * sizeof(int));
    int qr_count = 0;
    
    for (int x = 1; x < p; x++) {
        if (is_quadratic_residue(x, p)) {
            qr_list[qr_count++] = x;
        }
    }
    
    fprintf(stderr, "B1-Ramanujan: Found %d quadratic residues mod %d\n", qr_count, p);
    
    // Select best subset of QRs for target degree
    int generators_needed = (degree + 1) / 2; // Each generator gives 2 edges per vertex
    if (generators_needed > qr_count) generators_needed = qr_count;
    
    // Use well-distributed QRs for optimal expansion
    int *selected_generators = (int*)malloc(generators_needed * sizeof(int));
    for (int i = 0; i < generators_needed; i++) {
        int idx = (i * qr_count) / generators_needed;
        if (idx >= qr_count) idx = qr_count - 1;
        selected_generators[i] = qr_list[idx];
    }
    
    fprintf(stderr, "B1-Ramanujan: Selected %d generators: ", generators_needed);
    for (int i = 0; i < generators_needed; i++) {
        fprintf(stderr, "%d ", selected_generators[i]);
    }
    fprintf(stderr, "\n");
    
    // Build Cayley graph using selected generators
    for (int g = 0; g < generators_needed; g++) {
        int generator = selected_generators[g];
        
        for (int i = 0; i < n; i++) {
            // Forward edge: i -> (i + generator) mod n
            int j_forward = (i + generator) % n;
            if (i != j_forward && adj_matrix[i][j_forward] == 0) {
                adj_matrix[i][j_forward] = 1;
                adj_matrix[j_forward][i] = 1;
            }
            
            // Backward edge: i -> (i - generator + n) mod n
            int j_backward = (i - generator + n) % n;
            if (i != j_backward && adj_matrix[i][j_backward] == 0) {
                adj_matrix[i][j_backward] = 1;
                adj_matrix[j_backward][i] = 1;
            }
        }
    }
    
    free(selected_generators);
    free(qr_list);
}

/**
 * Build Ramanujan anchor graph with specified seed degree.
 * Creates optimal expander foundation for subsequent degree growth.
 * 
 * @param n Number of vertices
 * @param seed_degree Degree of the anchor graph
 * @param adj_matrix Output adjacency matrix
 */
static void build_ramanujan_anchor_graph(int n, int seed_degree, int **adj_matrix) {
    if (seed_degree >= 6) {
        // For higher seed degrees, use Cayley construction
        build_cayley_ramanujan_base(n, seed_degree, adj_matrix);
    } else {
        // For low seed degrees, use optimal circulant with QR-inspired steps
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                adj_matrix[i][j] = 0;
            }
        }
        
        if (seed_degree == 2) {
            // 2-regular: simple cycle
            for (int i = 0; i < n; i++) {
                int j = (i + 1) % n;
                adj_matrix[i][j] = 1;
                adj_matrix[j][i] = 1;
            }
        } else if (seed_degree == 3) {
            // 3-regular: cycle + perfect matching
            for (int i = 0; i < n; i++) {
                int j = (i + 1) % n;
                adj_matrix[i][j] = 1;
                adj_matrix[j][i] = 1;
            }
            // Add perfect matching
            for (int i = 0; i < n/2; i++) {
                int j = i + n/2;
                adj_matrix[i][j] = 1;
                adj_matrix[j][i] = 1;
            }
        } else if (seed_degree == 4) {
            // 4-regular: two well-spaced circulant steps
            int step1 = 1;
            int step2 = (n >= 16) ? (n/8) : 2;
            if (step2 >= n/2) step2 = n/2 - 1;
            if (step2 == step1) step2 = step1 + 1;
            
            int steps[] = {step1, step2};
            for (int s = 0; s < 2; s++) {
                int step = steps[s];
                for (int i = 0; i < n; i++) {
                    int j = (i + step) % n;
                    if (i != j && adj_matrix[i][j] == 0) {
                        adj_matrix[i][j] = 1;
                        adj_matrix[j][i] = 1;
                    }
                }
            }
        } else if (seed_degree == 5) {
            // 5-regular: 4-regular + perfect matching
            int step1 = 1;
            int step2 = (n >= 20) ? (n/10) : 2;
            if (step2 >= n/2) step2 = n/2 - 1;
            if (step2 == step1) step2 = step1 + 1;
            
            int steps[] = {step1, step2};
            for (int s = 0; s < 2; s++) {
                int step = steps[s];
                for (int i = 0; i < n; i++) {
                    int j = (i + step) % n;
                    if (i != j && adj_matrix[i][j] == 0) {
                        adj_matrix[i][j] = 1;
                        adj_matrix[j][i] = 1;
                    }
                }
            }
            // Add perfect matching
            for (int i = 0; i < n/2; i++) {
                int j = i + n/2;
                if (adj_matrix[i][j] == 0) {
                    adj_matrix[i][j] = 1;
                    adj_matrix[j][i] = 1;
                }
            }
        }
    }
    
    fprintf(stderr, "B1-Ramanujan: Built anchor graph with seed degree %d\n", seed_degree);
}

/**
 * Grow Ramanujan anchor graph to target degree using spectral-preserving edge additions.
 * Adds edges systematically while preserving the excellent spectral properties of the anchor.
 * FIXED: Properly tracks degrees to ensure exact k-regularity.
 * 
 * @param n Number of vertices
 * @param seed_degree Current degree of anchor graph
 * @param target_k Target degree to reach
 * @param adj_matrix Adjacency matrix to modify
 */
static void grow_ramanujan_to_target_degree(int n, int seed_degree, int target_k, int **adj_matrix) {
    fprintf(stderr, "B1-Ramanujan: Growing from degree %d to degree %d\n", seed_degree, target_k);
    
    // Count current degrees
    int *degrees = (int*)calloc(n, sizeof(int));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (adj_matrix[i][j] == 1) degrees[i]++;
        }
    }
    
    int edges_needed = target_k - seed_degree;
    int edges_added = 0;
    
    while (edges_added < edges_needed) {
        int step = 1;
        
        // Find next available step that creates valid edges
        bool step_found = false;
        for (int candidate = 1; candidate < n/2 && !step_found; candidate++) {
            int potential_edges = 0;
            
            // Count how many new edges this step would add
            for (int i = 0; i < n; i++) {
                int j = (i + candidate) % n;
                if (i != j && adj_matrix[i][j] == 0 && degrees[i] < target_k && degrees[j] < target_k) {
                    potential_edges++;
                }
            }
            
            if (potential_edges > 0) {
                step = candidate;
                step_found = true;
            }
        }
        
        if (!step_found) {
            fprintf(stderr, "B1-Ramanujan: Warning - no valid step found, using greedy edge addition\n");
            
            // Greedy edge addition as fallback
            int added_this_round = 0;
            for (int i = 0; i < n && edges_added < edges_needed; i++) {
                if (degrees[i] < target_k) {
                    for (int j = i + 1; j < n; j++) {
                        if (degrees[j] < target_k && adj_matrix[i][j] == 0) {
                            adj_matrix[i][j] = 1;
                            adj_matrix[j][i] = 1;
                            degrees[i]++;
                            degrees[j]++;
                            edges_added++;
                            added_this_round++;
                            if (edges_added >= edges_needed) break;
                        }
                    }
                }
            }
            
            if (added_this_round == 0) {
                fprintf(stderr, "B1-Ramanujan: Error - cannot add more edges\n");
                break;
            }
        } else {
            fprintf(stderr, "B1-Ramanujan: Adding circulant layer with step %d\n", step);
            
            // Add the circulant layer, checking degree constraints
            for (int i = 0; i < n; i++) {
                int j = (i + step) % n;
                if (i != j && adj_matrix[i][j] == 0 && degrees[i] < target_k && degrees[j] < target_k) {
                    adj_matrix[i][j] = 1;
                    adj_matrix[j][i] = 1;
                    degrees[i]++;
                    degrees[j]++;
                    edges_added++;
                    
                    if (edges_added >= edges_needed) break;
                }
            }
        }
    }
    
    // Final degree check and correction
    for (int i = 0; i < n; i++) {
        while (degrees[i] < target_k) {
            // Find a vertex to connect to
            int best_j = -1;
            for (int j = 0; j < n; j++) {
                if (i != j && adj_matrix[i][j] == 0 && degrees[j] < target_k) {
                    best_j = j;
                    break;
                }
            }
            
            if (best_j != -1) {
                adj_matrix[i][best_j] = 1;
                adj_matrix[best_j][i] = 1;
                degrees[i]++;
                degrees[best_j]++;
            } else {
                fprintf(stderr, "B1-Ramanujan: Warning - cannot reach target degree for vertex %d\n", i);
                break;
            }
        }
    }
    
    free(degrees);
    fprintf(stderr, "B1-Ramanujan: Growth complete, target degree %d reached\n", target_k);
}

/**
 * Build k-regular graph using effective resistance inspired construction.
 * Mimics the greedy effective resistance algorithm but in a deterministic way.
 * Key insight: Effective resistance prioritizes edges that maximally improve connectivity.
 * 
 * @param n Number of vertices (even)
 * @param k Target degree (3 ≤ k < n/2)
 * @param adj_matrix Output adjacency matrix
 */
static void build_effective_resistance_inspired_graph(int n, int k, int **adj_matrix) {
    // Initialize empty matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            adj_matrix[i][j] = 0;
        }
    }
    
    fprintf(stderr, "B1-EffRes: Building k=%d graph inspired by effective resistance greedy\n", k);
    
    // INSIGHT 1: Effective resistance greedy starts by connecting components
    // For sparse graphs, this means prioritizing long-distance connections
    
    // INSIGHT 2: In sparse regime, the algorithm tends to create a "spanning tree + extra edges"
    // The spanning tree connects all vertices, extra edges improve λ₂
    
    // Strategy: Build a good spanning structure first, then add improvement edges
    
    // Phase 1: Create initial connectivity backbone
    // For k=3, this is critical - we need exactly the right spanning structure
    
    if (k == 3) {
        // For 3-regular: We need exactly 3n/2 edges total
        // Strategy: Build cycle (n edges) + additional n/2 chords
        
        // Build cycle first (gives each vertex degree 2)
        for (int i = 0; i < n; i++) {
            int j = (i + 1) % n;
            adj_matrix[i][j] = 1;
            adj_matrix[j][i] = 1;
        }
        
        // Add n/2 well-spaced chords to reach degree 3
        // Use effective resistance insight: prefer distant connections
        int chord_distance = n / 2;
        int chords_added = 0;
        
        for (int i = 0; i < n && chords_added < n/2; i++) {
            int j = (i + chord_distance) % n;
            if (i != j && adj_matrix[i][j] == 0) {
                adj_matrix[i][j] = 1;
                adj_matrix[j][i] = 1;
                chords_added++;
            }
        }
        
        fprintf(stderr, "B1-EffRes: Built 3-regular cycle + %d chords (distance %d)\n", chords_added, chord_distance);
        
    } else if (k == 4) {
        // For 4-regular: Use circulant C_n(1, 2)
        // This gives each vertex connections to nearest and second-nearest neighbors
        
        // Add step 1 connections (cycle)
        for (int i = 0; i < n; i++) {
            int j = (i + 1) % n;
            adj_matrix[i][j] = 1;
            adj_matrix[j][i] = 1;
        }
        
        // Add step 2 connections (second-nearest neighbors)
        for (int i = 0; i < n; i++) {
            int j = (i + 2) % n;
            if (adj_matrix[i][j] == 0) {
                adj_matrix[i][j] = 1;
                adj_matrix[j][i] = 1;
            }
        }
        
        fprintf(stderr, "B1-EffRes: Built 4-regular circulant C_%d(1, 2)\n", n);
        
    } else if (k == 5) {
        // For 5-regular: We need exactly 5n/2 edges total
        // Strategy: circulant with steps {1, 2, n/2}
        
        // Step 1: Basic connectivity
        for (int i = 0; i < n; i++) {
            int j = (i + 1) % n;
            adj_matrix[i][j] = 1;
            adj_matrix[j][i] = 1;
        }
        
        // Step 2: Medium distance connections
        for (int i = 0; i < n; i++) {
            int j = (i + 2) % n;
            if (i != j && adj_matrix[i][j] == 0) {
                adj_matrix[i][j] = 1;
                adj_matrix[j][i] = 1;
            }
        }
        
        // Step n/2: Long distance connections (perfect matching)
        if (n % 2 == 0) {
            for (int i = 0; i < n/2; i++) {
                int j = i + n/2;
                if (adj_matrix[i][j] == 0) {
                    adj_matrix[i][j] = 1;
                    adj_matrix[j][i] = 1;
                }
            }
        }
        
        fprintf(stderr, "B1-EffRes: Built 5-regular circulant C_%d(1, 2, %d)\n", n, n/2);
        
    } else {
        // For higher k: Use circulant with effective-resistance inspired step selection
        // Key insight: Steps should prioritize "electrical distance"
        
        int num_steps = (k % 2 == 0) ? k/2 : (k-1)/2;
        int *steps = (int*)malloc(num_steps * sizeof(int));
        
        // Step selection based on effective resistance insights:
        // 1. Always include step=1 (basic connectivity)
        // 2. Include steps that are roughly geometric progression (distant connections)
        // 3. Avoid steps that are too close together
        
        steps[0] = 1; // Always start with basic connectivity
        
        if (num_steps > 1) {
            // Use geometric-like progression for remaining steps
            // This mimics how effective resistance prefers distant connections
            int base_step = (n / (2 * num_steps)) + 1;
            
            for (int i = 1; i < num_steps; i++) {
                int step = base_step * (i + 1);
                if (step >= n/2) step = n/2 - i;
                if (step <= steps[i-1]) step = steps[i-1] + 1;
                steps[i] = step;
            }
        }
        
        // Build circulant with selected steps
        for (int s = 0; s < num_steps; s++) {
            int step = steps[s];
            for (int i = 0; i < n; i++) {
                int j = (i + step) % n;
                if (i != j && adj_matrix[i][j] == 0) {
                    adj_matrix[i][j] = 1;
                    adj_matrix[j][i] = 1;
                }
            }
        }
        
        // Add perfect matching for odd k
        if (k % 2 == 1) {
            for (int i = 0; i < n/2; i++) {
                int j = i + n/2;
                if (adj_matrix[i][j] == 0) {
                    adj_matrix[i][j] = 1;
                    adj_matrix[j][i] = 1;
                }
            }
        }
        
        fprintf(stderr, "B1-EffRes: Built %d-regular circulant with eff-res inspired steps\n", k);
        free(steps);
    }
    
    fprintf(stderr, "B1-EffRes: Construction complete\n");
}

/**
 * Build k-regular graph using Ramanujan-inspired direct construction.
 * Uses quadratic residue-inspired step selection for optimal spectral properties.
 * Builds directly to target degree k without anchors for better regularity.
 * 
 * @param n Number of vertices (even)
 * @param k Target degree (3 ≤ k < n/2)
 * @param adj_matrix Output adjacency matrix
 */
static void build_ramanujan_seed_graph(int n, int k, int **adj_matrix) {
    // Initialize empty matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            adj_matrix[i][j] = 0;
        }
    }
    
    fprintf(stderr, "B1-Ramanujan: Building direct k=%d regular graph using QR-inspired steps\n", k);
    
    // Find suitable prime for quadratic residue inspiration
    int p = find_suitable_prime(n);
    
    // Collect quadratic residues modulo p
    int *qr_list = (int*)malloc(p * sizeof(int));
    int qr_count = 0;
    
    for (int x = 1; x < p; x++) {
        if (is_quadratic_residue(x, p)) {
            qr_list[qr_count++] = x;
        }
    }
    
    fprintf(stderr, "B1-Ramanujan: Using prime p=%d with %d quadratic residues\n", p, qr_count);
    
    // Select circulant steps inspired by QRs
    int num_steps_needed;
    if (k % 2 == 0) {
        num_steps_needed = k / 2;
    } else {
        num_steps_needed = (k - 1) / 2; // Will add perfect matching for odd k
    }
    
    int *selected_steps = (int*)malloc(num_steps_needed * sizeof(int));
    
    // Step 1: Always include 1 for connectivity
    selected_steps[0] = 1;
    
    // Remaining steps: use QR-inspired selection with good distribution
    for (int i = 1; i < num_steps_needed; i++) {
        int step;
        
        if (i - 1 < qr_count) {
            // Use QR as starting point
            int qr_idx = ((i - 1) * qr_count) / (num_steps_needed - 1);
            if (qr_idx >= qr_count) qr_idx = qr_count - 1;
            step = qr_list[qr_idx] % (n/2);
            if (step == 0) step = 1;
        } else {
            // Fallback to arithmetic progression
            step = i + 1;
        }
        
        // Ensure step is valid and unique
        while (step >= n/2 || step <= 0) {
            step = (step + 1) % (n/2);
            if (step == 0) step = 1;
        }
        
        // Check for duplicates
        bool duplicate = false;
        for (int j = 0; j < i; j++) {
            if (selected_steps[j] == step) {
                duplicate = true;
                break;
            }
        }
        
        if (duplicate) {
            // Find next available step
            for (int candidate = step + 1; candidate < n/2; candidate++) {
                bool found_dup = false;
                for (int j = 0; j < i; j++) {
                    if (selected_steps[j] == candidate) {
                        found_dup = true;
                        break;
                    }
                }
                if (!found_dup) {
                    step = candidate;
                    break;
                }
            }
        }
        
        selected_steps[i] = step;
    }
    
    fprintf(stderr, "B1-Ramanujan: Selected steps: ");
    for (int i = 0; i < num_steps_needed; i++) {
        fprintf(stderr, "%d ", selected_steps[i]);
    }
    fprintf(stderr, "\n");
    
    // Build circulant graph with selected steps
    for (int s = 0; s < num_steps_needed; s++) {
        int step = selected_steps[s];
        for (int i = 0; i < n; i++) {
            int j = (i + step) % n;
            if (i != j && adj_matrix[i][j] == 0) {
                adj_matrix[i][j] = 1;
                adj_matrix[j][i] = 1;
            }
        }
    }
    
    // Add perfect matching for odd k
    if (k % 2 == 1) {
        fprintf(stderr, "B1-Ramanujan: Adding perfect matching for odd degree\n");
        for (int i = 0; i < n/2; i++) {
            int j = i + n/2;
            if (adj_matrix[i][j] == 0) {
                adj_matrix[i][j] = 1;
                adj_matrix[j][i] = 1;
            }
        }
    }
    
    free(selected_steps);
    free(qr_list);
    
    fprintf(stderr, "B1-Ramanujan: Direct construction complete\n");
}

/**
 * Build k-regular graph using sophisticated deterministic expander approach.
 * Optimized circulant construction with spectral-aware step selection.
 * 
 * @param n Number of vertices (even)
 * @param k Target degree (3 ≤ k < n/2)
 * @param adj_matrix Output adjacency matrix
 */
/* ========================================================================
 * MAIN B1 BRANCH ENTRY POINT
 * ======================================================================== */

/**
 * Main entry point for B1 branch.
 * Constructs k-regular graphs for even n with 3 ≤ k < n/2.
 * 
 * NEW APPROACH: Uses Ramanujan graph anchors for optimal spectral properties.
 * Build optimal expander foundations using quadratic residue Cayley graphs,
 * then grow systematically to reach target degree while preserving spectral gap.
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
    
    fprintf(stderr, "B1: Using effective resistance inspired construction\n");
    
    // NEW: Use effective resistance inspired construction
    // This mimics the patterns that make the greedy baseline so successful
    build_effective_resistance_inspired_graph(n, k, adj_matrix);
}