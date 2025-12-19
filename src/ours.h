/*
 * ours.h - Adaptive Spectral Builder Algorithm
 *
 * O(M) graph builder for maximizing algebraic connectivity.
 * For dense graphs (M ≈ N²), this is O(N²).
 *
 * COMPLEXITY GUARANTEE:
 * - Adjacency matrix: O(N²) space, O(1) access
 * - Degree tracking: O(N) array
 * - Min-degree selection: O(1) amortized via bucket queue
 * - Overlap detection: O(1) via fixed-size sampling
 * - Edge selection: O(1) constant probes
 * - Total: O(M) for M edges
 */

#ifndef OURS_H
#define OURS_H

#include "common.h"

#define ASB_MAX_STRIDES  64
#define ASB_MAX_PROBES   256
#define ASB_SAMPLE_LIMIT 16   /* Fixed sample size for O(1) overlap detection */

typedef struct {
    int n;
    AdjMatrix *adj;
    int *degrees;
    int edge_count;

    /* Bucket queue for O(1) amortized min-degree selection */
    int *bucket_head;   /* bucket_head[d] = first node with degree d, or -1 */
    int *bucket_next;   /* bucket_next[node] = next node in same bucket */
    int *bucket_prev;   /* bucket_prev[node] = prev node in same bucket */
    int min_degree;     /* current minimum degree (monotonically increasing) */

    /* Track active nodes */
    bool *node_active;

    /* Sample neighbors for overlap detection */
    int **sample_neighbors;
    int *sample_sizes;
    int sample_limit;

    /* Strides and probes for edge selection */
    int strides[ASB_MAX_STRIDES];
    int num_strides;
    int probes[ASB_MAX_PROBES];
    int num_probes;
} AdaptiveSpectralBuilder;

/*
 * Create a new builder instance.
 */
AdaptiveSpectralBuilder *asb_create(int n);

/*
 * Free builder resources.
 */
void asb_free(AdaptiveSpectralBuilder *builder);

/*
 * Build a graph with m edges, maximizing algebraic connectivity.
 * Returns pointer to the internal adjacency matrix (do not free separately).
 */
AdjMatrix *asb_build(AdaptiveSpectralBuilder *builder, int m);

/*
 * Convenience function: build graph and compute algebraic connectivity.
 */
double ours_score(int n, int m);

#endif /* OURS_H */
