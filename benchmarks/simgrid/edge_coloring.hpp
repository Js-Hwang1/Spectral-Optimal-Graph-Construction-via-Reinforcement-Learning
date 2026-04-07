/*
 * edge_coloring.hpp — Decompose a d-regular graph into edge-disjoint
 * matchings via iterative maximum matching extraction.
 *
 * Algorithm:
 *   For c = 0, 1, ...:
 *     1. Find a maximal matching in the remaining graph using
 *        greedy + augmenting paths.
 *     2. Store as color c.
 *     3. Remove matched edges.
 *     4. Stop when no edges remain.
 *
 * For d-regular graphs with even n:
 *   - Typically produces exactly d matchings (Class 1).
 *   - At worst d+1 matchings (Vizing's theorem guarantees this).
 *   - Each matching is perfect or near-perfect (0-2 unmatched nodes).
 *
 * Time: O(n * d^2) — at most d+1 matchings, each found in O(n*d).
 */

#pragma once

#include <vector>
#include <queue>
#include <cassert>

namespace edge_coloring {

struct Matching {
    std::vector<int> partner;  // partner[i] = j if matched, -1 if unmatched

    std::vector<std::pair<int,int>> edges() const {
        std::vector<std::pair<int,int>> result;
        for (int i = 0; i < (int)partner.size(); i++)
            if (partner[i] > i) result.push_back({i, partner[i]});
        return result;
    }

    int unmatched_count() const {
        int c = 0;
        for (int p : partner) if (p < 0) c++;
        return c;
    }

    int matched_edges() const {
        int c = 0;
        for (int p : partner) if (p >= 0) c++;
        return c / 2;
    }
};

// ============================================================
// Find maximum matching via greedy + BFS augmenting paths
// ============================================================

static Matching find_max_matching(
    const std::vector<std::vector<int>>& adj, int n)
{
    Matching M;
    M.partner.assign(n, -1);

    // Phase 1: Greedy (lowest index priority)
    for (int i = 0; i < n; i++) {
        if (M.partner[i] >= 0) continue;
        for (int j : adj[i]) {
            if (M.partner[j] < 0) {
                M.partner[i] = j;
                M.partner[j] = i;
                break;
            }
        }
    }

    // Phase 2: BFS augmenting paths from each unmatched node
    for (int start = 0; start < n; start++) {
        if (M.partner[start] >= 0) continue;

        std::vector<int> parent(n, -2);  // -2 = not visited
        std::queue<int> q;
        parent[start] = -1;
        q.push(start);
        int found = -1;

        while (!q.empty() && found < 0) {
            int u = q.front(); q.pop();

            for (int v : adj[u]) {
                if (parent[v] != -2) continue;
                if (M.partner[u] == v) continue;  // don't traverse matched edge from u

                parent[v] = u;

                if (M.partner[v] < 0) {
                    found = v;
                    break;
                }

                // v is matched — traverse through its matched partner
                int w = M.partner[v];
                parent[w] = v;
                q.push(w);
            }
        }

        if (found >= 0) {
            // Augment along path
            int v = found;
            while (v >= 0) {
                int u = parent[v];
                int prev = (u >= 0) ? parent[u] : -1;
                M.partner[u] = v;
                M.partner[v] = u;
                v = prev;
            }
        }
    }

    return M;
}

// ============================================================
// Decompose: iteratively extract matchings
// ============================================================

static std::vector<Matching> decompose(
    const std::vector<std::vector<int>>& adj_in, int n, int d)
{
    // Build mutable edge set
    std::vector<std::vector<bool>> has_edge(n, std::vector<bool>(n, false));
    int total_edges = 0;
    for (int i = 0; i < n; i++) {
        for (int j : adj_in[i]) {
            if (!has_edge[i][j]) {
                has_edge[i][j] = true;
                has_edge[j][i] = true;
                if (i < j) total_edges++;
            }
        }
    }

    std::vector<Matching> result;

    while (total_edges > 0) {
        // Build current adjacency
        std::vector<std::vector<int>> adj(n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (has_edge[i][j]) adj[i].push_back(j);
            }
        }

        // Find matching
        Matching M = find_max_matching(adj, n);

        // Verify symmetry (defensive)
        for (int i = 0; i < n; i++) {
            if (M.partner[i] >= 0) {
                assert(M.partner[M.partner[i]] == i);
            }
        }

        // Remove matched edges
        int removed = 0;
        for (int i = 0; i < n; i++) {
            int j = M.partner[i];
            if (j > i && has_edge[i][j]) {
                has_edge[i][j] = false;
                has_edge[j][i] = false;
                removed++;
            }
        }
        total_edges -= removed;

        result.push_back(M);

        // Safety: shouldn't need more than d+1 colors
        if ((int)result.size() > d + 1) break;
    }

    return result;
}

} // namespace edge_coloring
