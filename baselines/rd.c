/*
 * rd.c - Random d-Regular Graph Generator
 *
 * Two-phase approach:
 *   Phase 1: Pairing model (Bollobas 1980) — try pure rejection sampling.
 *            Works well for d << sqrt(n).
 *   Phase 2: Pairing + edge-swap repair — generate a pairing (possibly with
 *            self-loops/multi-edges), then fix violations via random 2-opt swaps
 *            until the graph is simple. This converges for any d < n.
 *
 * References:
 *   B. Bollobas, "A probabilistic proof of an asymptotic formula for the
 *   number of labelled regular graphs", European J. Combin., 1(4), 1980.
 *
 *   A. Steger & N. Wormald, "Generating random regular graphs quickly",
 *   Combinatorics, Probability & Computing, 8(4), 1999.
 */

#include "rd.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define RD_PURE_ATTEMPTS 100
#define RD_SWAP_LIMIT 100000

int rd_check(int n, int m) {
    if (n <= 0 || m < 0) return -1;
    if (m == 0) return 0;

    /* d = 2m/n must be integer */
    if ((2 * m) % n != 0) return -1;
    int d = 2 * m / n;

    /* d must be < n (simple graph) */
    if (d >= n) return -1;

    /* n*d must be even */
    if ((n * d) % 2 != 0) return -1;

    return d;
}

/* Shuffle an array of nd ints */
static void shuffle(int *arr, int len) {
    for (int i = len - 1; i > 0; i--) {
        int j = rng_int(i + 1);
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

/* Fill half-edge array: node i appears d times */
static void fill_halfedges(int *arr, int n, int d) {
    int idx = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++)
            arr[idx++] = i;
}

/* Try pure pairing model: returns 1 on success, 0 on failure */
static int try_pure_pairing(int n, int d, int *perm, AdjMatrix *adj) {
    int nd = n * d;

    fill_halfedges(perm, n, d);
    shuffle(perm, nd);

    memset(adj->data, 0, (size_t)n * n * sizeof(bool));

    for (int i = 0; i < nd; i += 2) {
        int u = perm[i];
        int v = perm[i + 1];
        if (u == v || adj_get(adj, u, v))
            return 0;
        adj_set(adj, u, v, true);
        adj_set(adj, v, u, true);
    }
    return 1;
}

/*
 * Pairing + swap repair:
 *   1. Generate a random pairing (may have self-loops / multi-edges)
 *   2. Build an edge list from the pairing
 *   3. Find bad edges (self-loops or duplicates)
 *   4. Fix each bad edge by swapping with a random good edge
 */
static int pairing_with_repair(int n, int d, int *perm, AdjMatrix *adj) {
    int nd = n * d;
    int m = nd / 2;

    fill_halfedges(perm, n, d);
    shuffle(perm, nd);

    /* Build edge list from pairing */
    int *eu = malloc((size_t)m * sizeof(int));
    int *ev = malloc((size_t)m * sizeof(int));

    for (int i = 0; i < m; i++) {
        eu[i] = perm[2 * i];
        ev[i] = perm[2 * i + 1];
    }

    /* Rebuild adjacency and fix violations */
    for (int iter = 0; iter < RD_SWAP_LIMIT; iter++) {
        /* Rebuild adjacency from edge list */
        memset(adj->data, 0, (size_t)n * n * sizeof(bool));

        int bad = -1;
        for (int i = 0; i < m; i++) {
            int u = eu[i], v = ev[i];
            if (u == v || adj_get(adj, u, v)) {
                bad = i;
                break;
            }
            adj_set(adj, u, v, true);
            adj_set(adj, v, u, true);
        }

        if (bad < 0) {
            /* All edges are good — verify degree */
            free(eu);
            free(ev);
            return 1;
        }

        /* Swap bad edge with a random other edge */
        int other = rng_int(m);
        if (other == bad) other = (other + 1) % m;

        /* Swap one endpoint: (bad.u, bad.v) + (other.u, other.v)
         * becomes (bad.u, other.v) + (other.u, bad.v)
         * with 50% chance of the alternative cross */
        if (rng_int(2)) {
            int tmp = ev[bad];
            ev[bad] = ev[other];
            ev[other] = tmp;
        } else {
            int tmp = ev[bad];
            ev[bad] = eu[other];
            eu[other] = tmp;
        }
    }

    free(eu);
    free(ev);
    fprintf(stderr, "  [RD] Swap repair did not converge after %d iterations\n",
            RD_SWAP_LIMIT);
    return 0;
}

int rd_build(int n, int m, AdjMatrix *adj_out) {
    int d = rd_check(n, m);
    if (d < 0) return -1;
    if (d == 0) return 0;

    int nd = n * d;
    int *perm = malloc((size_t)nd * sizeof(int));

    /* Phase 1: try pure rejection (fast for small d) */
    for (int attempt = 0; attempt < RD_PURE_ATTEMPTS; attempt++) {
        if (try_pure_pairing(n, d, perm, adj_out)) {
            free(perm);
            return 0;
        }
    }

    /* Phase 2: pairing with swap repair (works for any d) */
    if (pairing_with_repair(n, d, perm, adj_out)) {
        free(perm);
        return 0;
    }

    free(perm);
    fprintf(stderr, "  [RD] Failed to generate d=%d regular graph for n=%d\n", d, n);
    return -1;
}
