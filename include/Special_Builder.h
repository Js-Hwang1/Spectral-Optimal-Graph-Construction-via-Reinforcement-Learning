/* Special_Builder.h — Special Graph Construction Cases Header
 *
 * DESCRIPTION:
 *   Header file for specialized graph construction functions that handle
 *   optimal special cases using adjacency matrix representation.
 */

#ifndef SPECIAL_BUILDER_H
#define SPECIAL_BUILDER_H

/* ========================================================================
 * FUNCTION DECLARATIONS
 * ======================================================================== */

/**
 * Build complete bipartite graph for optimal special cases.
 * Handles both even n (k=n/2) and odd n (k=⌈n/2⌉) cases.
 * Modifies adjacency matrix in-place.
 */
void build_complete_bipartite_special_case(int n, int k, int **adj_matrix);

/**
 * Check if parameters satisfy complete bipartite case requirements.
 * Works for both even n (k = n/2) and odd n (k = ⌈n/2⌉).
 */
int is_complete_bipartite_case(int n, int k);

/**
 * Build complete n-partite graph foundation.
 * Creates n groups with complete connections between all pairs of groups.
 * Returns the base degree achieved, or 0 if n-partite is not suitable.
 */
int build_npartite_base(int n, int num_groups, int k, int **adj_matrix);

/**
 * Check for the maximum partite division that is beneficial.
 * Uses the logic: partition n vertices into N groups, find smallest group size,
 * and check if k ≥ (n - smallest_group_size).
 * 
 * @param n Number of vertices
 * @param k Degree
 * @return Maximum beneficial partite number (2 for bipartite, 3 for tripartite, etc.)
 *         Returns 0 if no n-partite structure is beneficial
 */
int is_npartite_beneficial(int n, int k);

/**
 * Check for the maximum partite division that is beneficial FOR EVEN n.
 * Ensures all groups have even sizes to avoid degree-1 bumping issues.
 * 
 * @param n Number of vertices (must be even)
 * @param k Degree
 * @return Maximum beneficial partite number for even n
 *         Returns 0 if no n-partite structure is beneficial
 */
int is_npartite_beneficial_even_n(int n, int k);

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
int build_npartite_base_even_n(int n, int num_groups, int k, int **adj_matrix);

#endif /* SPECIAL_BUILDER_H */