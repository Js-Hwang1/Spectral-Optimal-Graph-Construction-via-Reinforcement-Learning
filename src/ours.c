/*
 * ours.c - Adaptive Spectral Builder Algorithm Implementation
 *
 * O(M) complexity via:
 * - Bucket queue for O(1) amortized min-degree selection
 * - Fixed-size sampling for O(1) overlap detection
 */

#include "ours.h"
#include <stdlib.h>
#include <math.h>

#define PHI 0.61803398875

/* ============================================================================
 * BUCKET QUEUE OPERATIONS (O(1) amortized min-degree)
 * ============================================================================ */

/* Remove node from its current bucket */
static void bucket_remove(AdaptiveSpectralBuilder *b, int node) {
    int deg = b->degrees[node];
    int prev = b->bucket_prev[node];
    int next = b->bucket_next[node];

    if (prev >= 0) {
        b->bucket_next[prev] = next;
    } else {
        /* node was head of bucket */
        b->bucket_head[deg] = next;
    }

    if (next >= 0) {
        b->bucket_prev[next] = prev;
    }

    b->bucket_prev[node] = -1;
    b->bucket_next[node] = -1;
}

/* Insert node into bucket for given degree */
static void bucket_insert(AdaptiveSpectralBuilder *b, int node, int deg) {
    int old_head = b->bucket_head[deg];
    b->bucket_head[deg] = node;
    b->bucket_next[node] = old_head;
    b->bucket_prev[node] = -1;

    if (old_head >= 0) {
        b->bucket_prev[old_head] = node;
    }
}

/* Move node from degree d to degree d+1 */
static void bucket_increase_degree(AdaptiveSpectralBuilder *b, int node) {
    int old_deg = b->degrees[node];
    bucket_remove(b, node);
    b->degrees[node] = old_deg + 1;
    bucket_insert(b, node, old_deg + 1);
}

/* Find minimum degree active node - O(1) amortized */
static int find_min_degree_node(AdaptiveSpectralBuilder *b) {
    /* Advance min_degree until we find a non-empty bucket */
    while (b->min_degree < b->n && b->bucket_head[b->min_degree] < 0) {
        b->min_degree++;
    }

    if (b->min_degree >= b->n) return -1;

    /* Return first active node in this bucket */
    int node = b->bucket_head[b->min_degree];
    while (node >= 0 && !b->node_active[node]) {
        node = b->bucket_next[node];
    }

    return node;
}

/* ============================================================================
 * INTERNAL HELPERS
 * ============================================================================ */

static void deactivate_node(AdaptiveSpectralBuilder *b, int node) {
    b->node_active[node] = false;
    bucket_remove(b, node);
}

static bool add_edge(AdaptiveSpectralBuilder *b, int u, int v) {
    if (adj_get(b->adj, u, v)) return false;

    adj_set(b->adj, u, v, true);
    adj_set(b->adj, v, u, true);
    b->edge_count++;

    /* Update sample neighbors */
    if (b->sample_sizes[u] < b->sample_limit) {
        b->sample_neighbors[u][b->sample_sizes[u]++] = v;
    }
    if (b->sample_sizes[v] < b->sample_limit) {
        b->sample_neighbors[v][b->sample_sizes[v]++] = u;
    }

    /* Update bucket queue - move both nodes to higher degree bucket */
    bucket_increase_degree(b, u);
    bucket_increase_degree(b, v);

    return true;
}

static int sampled_overlap(AdaptiveSpectralBuilder *b, int u, int v) {
    int count = 0;

    /* Count common neighbors from u's sample */
    for (int i = 0; i < b->sample_sizes[u]; i++) {
        int w = b->sample_neighbors[u][i];
        if (adj_get(b->adj, v, w)) count++;
    }

    /* Count additional from v's sample using adjacency matrix for dedup */
    for (int i = 0; i < b->sample_sizes[v]; i++) {
        int w = b->sample_neighbors[v][i];
        if (!adj_get(b->adj, u, w)) continue;  /* not a common neighbor */
        /* Check if already counted via u's sample */
        bool already_counted = false;
        for (int j = 0; j < b->sample_sizes[u] && j < ASB_SAMPLE_LIMIT; j++) {
            if (b->sample_neighbors[u][j] == w) {
                already_counted = true;
                break;
            }
        }
        if (!already_counted) count++;
    }

    return count;
}

static void expand_sparse(AdaptiveSpectralBuilder *b, int m) {
    while (b->edge_count < m) {
        int u = find_min_degree_node(b);
        if (u < 0) break;

        int best_v = -1;
        int min_overlap = 999;
        int start_rot = (u * 7) % b->num_strides;

        for (int k = 0; k < b->num_strides; k++) {
            int stride = b->strides[(start_rot + k) % b->num_strides];

            int candidates[2] = {
                (u + stride) % b->n,
                (u - stride + b->n) % b->n
            };

            for (int c = 0; c < 2; c++) {
                int v = candidates[c];
                if (v == u || adj_get(b->adj, u, v)) continue;

                int overlap = sampled_overlap(b, u, v);
                if (overlap == 0) {
                    best_v = v;
                    min_overlap = 0;
                    break;
                }
                if (overlap < min_overlap) {
                    min_overlap = overlap;
                    best_v = v;
                }
            }

            if (min_overlap == 0) break;
        }

        /* Fallback: golden ratio probing */
        if (best_v < 0) {
            for (int k = 1; k < 48; k++) {
                int v = (u + (int)(k * PHI * b->n)) % b->n;
                if (v != u && !adj_get(b->adj, u, v)) {
                    best_v = v;
                    break;
                }
            }
        }

        if (best_v >= 0) {
            add_edge(b, u, best_v);
        } else {
            deactivate_node(b, u);
        }
    }
}

static void expand_dense(AdaptiveSpectralBuilder *b, int m) {
    while (b->edge_count < m) {
        int u = find_min_degree_node(b);
        if (u < 0) break;

        int best_v = -1;
        int best_overlap = 999;
        int best_neg_dist = 0;

        int n_probes = b->num_probes;
        int rot = (u * 13) % n_probes;
        int max_probes = n_probes < 128 ? n_probes : 128;

        for (int k = 0; k < max_probes; k++) {
            int idx = (rot + k) % n_probes;
            int stride = b->probes[idx];
            int v = (u + stride) % b->n;

            if (v != u && !adj_get(b->adj, u, v)) {
                int overlap = sampled_overlap(b, u, v);
                int diff = abs(u - v);
                int dist = diff < b->n - diff ? diff : b->n - diff;
                int neg_dist = -dist;

                if (overlap < best_overlap ||
                    (overlap == best_overlap && neg_dist < best_neg_dist)) {
                    best_overlap = overlap;
                    best_neg_dist = neg_dist;
                    best_v = v;

                    if (overlap == 0 && dist >= (b->n / 2) - 1) break;
                }
            }
        }

        if (best_v >= 0) {
            add_edge(b, u, best_v);
        } else {
            deactivate_node(b, u);
        }
    }
}

/* ============================================================================
 * PUBLIC API
 * ============================================================================ */

AdaptiveSpectralBuilder *asb_create(int n) {
    AdaptiveSpectralBuilder *b = malloc(sizeof(AdaptiveSpectralBuilder));
    if (!b) return NULL;

    b->n = n;
    b->adj = adj_create(n);
    b->degrees = calloc((size_t)n, sizeof(int));
    b->edge_count = 0;

    /* Initialize bucket queue */
    b->bucket_head = malloc((size_t)n * sizeof(int));
    b->bucket_next = malloc((size_t)n * sizeof(int));
    b->bucket_prev = malloc((size_t)n * sizeof(int));
    b->min_degree = 0;

    /* All buckets start empty */
    for (int i = 0; i < n; i++) {
        b->bucket_head[i] = -1;
    }

    /* All nodes start as active with degree 0 */
    b->node_active = malloc((size_t)n * sizeof(bool));
    for (int i = 0; i < n; i++) {
        b->node_active[i] = true;
        b->bucket_prev[i] = -1;
        b->bucket_next[i] = -1;
    }

    /* Insert all nodes into bucket 0 (degree 0) */
    for (int i = 0; i < n; i++) {
        bucket_insert(b, i, 0);
    }

    /* Sample neighbors - fixed size for O(1) overlap detection */
    b->sample_limit = ASB_SAMPLE_LIMIT;
    b->sample_neighbors = malloc((size_t)n * sizeof(int *));
    b->sample_sizes = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < n; i++) {
        b->sample_neighbors[i] = malloc((size_t)ASB_SAMPLE_LIMIT * sizeof(int));
    }

    /* Build strides using golden ratio */
    bool *seen = calloc((size_t)n, sizeof(bool));
    b->num_strides = 0;

    for (int k = 2; k < 16; k++) {
        int val = n / k;
        if (val > 1 && !seen[val]) {
            b->strides[b->num_strides++] = val;
            seen[val] = true;
        }
    }

    for (int k = 1; k < 32; k++) {
        int val = (int)(k * PHI * n) % n;
        int dist = val < n - val ? val : n - val;
        if (dist > 1 && !seen[dist]) {
            b->strides[b->num_strides++] = dist;
            seen[dist] = true;
        }
    }

    /* Sort strides descending */
    for (int i = 0; i < b->num_strides - 1; i++) {
        for (int j = i + 1; j < b->num_strides; j++) {
            if (b->strides[j] > b->strides[i]) {
                int tmp = b->strides[i];
                b->strides[i] = b->strides[j];
                b->strides[j] = tmp;
            }
        }
    }

    /* Build probes */
    b->num_probes = b->num_strides;
    for (int i = 0; i < b->num_strides; i++) {
        b->probes[i] = b->strides[i];
    }

    for (int k = 20; k < 128; k++) {
        int val = (int)(k * PHI * n) % n;
        int dist = val < n - val ? val : n - val;
        if (dist > 1 && !seen[dist] && b->num_probes < ASB_MAX_PROBES) {
            b->probes[b->num_probes++] = dist;
            seen[dist] = true;
        }
    }

    free(seen);
    return b;
}

void asb_free(AdaptiveSpectralBuilder *b) {
    if (!b) return;

    adj_free(b->adj);
    free(b->degrees);
    free(b->node_active);
    free(b->bucket_head);
    free(b->bucket_next);
    free(b->bucket_prev);

    for (int i = 0; i < b->n; i++) {
        free(b->sample_neighbors[i]);
    }
    free(b->sample_neighbors);
    free(b->sample_sizes);
    free(b);
}

AdjMatrix *asb_build(AdaptiveSpectralBuilder *b, int m) {
    int max_m = b->n * (b->n - 1) / 2;
    double target_density = (double)m / max_m;
    double sparse_threshold = (b->n >= 128) ? 0.4 : 0.5;

    /* Build initial ring */
    for (int i = 0; i < b->n; i++) {
        if (b->edge_count >= m) break;
        add_edge(b, i, (i + 1) % b->n);
    }

    if (b->edge_count >= m) return b->adj;

    if (target_density < sparse_threshold) {
        expand_sparse(b, m);
    } else {
        expand_dense(b, m);
    }

    return b->adj;
}

double ours_score(int n, int m) {
    AdaptiveSpectralBuilder *builder = asb_create(n);
    asb_build(builder, m);
    double score = compute_algebraic_connectivity(builder->adj);
    asb_free(builder);
    return score;
}
