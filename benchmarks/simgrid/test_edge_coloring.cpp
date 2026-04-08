/*
 * Test edge coloring: verify decomposition of QRS-DR graphs into
 * d edge-disjoint perfect matchings.
 */

#include <cstdio>
#include <vector>
#include <cassert>
#include <cmath>
#include <set>

#include "topologies.hpp"
#include "edge_coloring.hpp"

int main() {
    struct TestCase { int n; int d; };
    std::vector<TestCase> tests = {
        {16, 4}, {32, 4}, {64, 4}, {128, 4},
        {32, 6}, {64, 6}, {128, 6},
        {64, 8}, {128, 8},
        {25, 4}, {33, 4}, {49, 6}, {63, 4},  // odd n
        {31, 4}, {37, 6}, {97, 4},            // primes
        {32, 3}, {64, 3}, {64, 5}, {64, 7},   // odd d
    };

    int passed = 0, failed = 0;

    for (auto& tc : tests) {
        int n = tc.n, d = tc.d;
        if ((n * d) % 2 != 0) continue;

        // Build QRS-DR graph
        auto topo = topo::qrs_dr(n, d);

        // Get adjacency list (static topology, round 0)
        const auto& adj = topo.get_round(0);

        // Verify d-regular
        bool regular = true;
        for (int i = 0; i < n; i++) {
            if ((int)adj[i].size() != d) { regular = false; break; }
        }

        // Decompose into matchings
        auto matchings = edge_coloring::decompose(adj, n, d);

        // Verify: each matching is valid (no two edges share a node)
        bool matchings_valid = true;
        int total_matched = 0;
        int total_unmatched = 0;

        for (int c = 0; c < d; c++) {
            auto& M = matchings[c];
            // Check no conflicts
            std::vector<bool> used(n, false);
            for (int i = 0; i < n; i++) {
                if (M.partner[i] >= 0) {
                    if (used[i]) { matchings_valid = false; break; }
                    used[i] = true;
                }
            }
            // Check symmetry
            for (int i = 0; i < n; i++) {
                if (M.partner[i] >= 0 && M.partner[M.partner[i]] != i) {
                    matchings_valid = false;
                    break;
                }
            }
            total_unmatched += M.unmatched_count();
        }

        // Verify: all edges are covered (edge-disjoint union = original graph)
        std::set<std::pair<int,int>> original_edges;
        for (int i = 0; i < n; i++) {
            for (int j : adj[i]) {
                if (i < j) original_edges.insert({i, j});
            }
        }

        std::set<std::pair<int,int>> decomposed_edges;
        for (int c = 0; c < d; c++) {
            for (auto& e : matchings[c].edges()) {
                decomposed_edges.insert(e);
            }
        }

        bool edges_match = (original_edges == decomposed_edges);

        // Check: for even n, each matching should be perfect (0 unmatched)
        bool perfect = true;
        if (n % 2 == 0) {
            for (int c = 0; c < d; c++) {
                if (matchings[c].unmatched_count() > 0) {
                    perfect = false;
                    break;
                }
            }
        } else {
            // Odd n: each matching has exactly 1 unmatched node
            for (int c = 0; c < d; c++) {
                if (matchings[c].unmatched_count() > 1) {
                    perfect = false;
                    break;
                }
            }
        }

        bool ok = regular && matchings_valid && edges_match && perfect;

        printf("n=%4d d=%d: reg=%s valid=%s edges=%s perfect=%s  => %s",
               n, d,
               regular ? "Y" : "N",
               matchings_valid ? "Y" : "N",
               edges_match ? "Y" : "N",
               perfect ? "Y" : "N",
               ok ? "PASS" : "FAIL");

        if (!ok) {
            printf("  (unmatched=%d, orig=%d, decomp=%d)",
                   total_unmatched,
                   (int)original_edges.size(),
                   (int)decomposed_edges.size());
            failed++;
        } else {
            passed++;
        }
        printf("\n");
    }

    printf("\n%d/%d PASSED\n", passed, passed + failed);
    return failed > 0 ? 1 : 0;
}
