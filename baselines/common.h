/*
 * common.h - Shared utilities for algebraic connectivity benchmark
 */

#ifndef COMMON_H
#define COMMON_H

#include <stdbool.h>
#include <stdint.h>

/* ============================================================================
 * ADJACENCY MATRIX
 * ============================================================================ */

typedef struct {
    bool *data;
    int n;
} AdjMatrix;

AdjMatrix *adj_create(int n);
void adj_free(AdjMatrix *adj);
bool adj_get(const AdjMatrix *adj, int i, int j);
void adj_set(AdjMatrix *adj, int i, int j, bool val);
int adj_edge_count(const AdjMatrix *adj);
void adj_copy(AdjMatrix *dst, const AdjMatrix *src);

/* Save adjacency matrix as raw n*n uint8 binary (row-major) */
int adj_save_binary(const AdjMatrix *adj, const char *path);

/* ============================================================================
 * GRAPH INITIALIZATION
 * ============================================================================ */

typedef enum { INIT_TREE, INIT_RING } InitType;

/* Build a random spanning tree (n-1 edges). Requires rng_seed() first. */
void build_random_tree(AdjMatrix *adj);

/* Build a ring/cycle graph (n edges). */
void build_ring(AdjMatrix *adj);

/* ============================================================================
 * EIGENVALUE COMPUTATIONS
 * ============================================================================ */

/* Compute algebraic connectivity (second smallest eigenvalue of Laplacian) */
double compute_algebraic_connectivity(const AdjMatrix *adj);

/* Compute Fiedler vector (eigenvector for second smallest eigenvalue) */
void compute_fiedler_vector(const AdjMatrix *adj, double *fiedler);

/* Compute pseudoinverse of Laplacian matrix */
void compute_laplacian_pinv(const AdjMatrix *adj, double *L_pinv);

/* ============================================================================
 * RANDOM NUMBER GENERATOR
 * ============================================================================ */

void rng_seed(uint64_t seed);
uint64_t rng_next(void);
int rng_int(int n);
double rng_double(void);

#endif /* COMMON_H */
