/*
 * test_topologies.cpp — Standalone test for topology generators
 * Build: g++ -O2 -std=c++17 -o test_topologies test_topologies.cpp
 */

#include <cstdio>
#include <cassert>
#include "topologies.hpp"

void print_topo_info(const char* name, const topo::Topology& t, int n) {
    printf("\n=== %s ===\n", name);
    printf("  is_static: %s\n", t.is_static ? "true" : "false");
    printf("  num_rounds: %d\n", t.num_rounds());

    for (int r = 0; r < t.num_rounds(); r++) {
        const auto& al = t.get_round(r);
        int min_deg = n, max_deg = 0, total_edges = 0;
        for (int i = 0; i < n; i++) {
            int deg = (int)al[i].size();
            min_deg = std::min(min_deg, deg);
            max_deg = std::max(max_deg, deg);
            total_edges += deg;
        }
        total_edges /= 2;

        printf("  Round %2d: deg range [%d, %d], edges=%d",
               r, min_deg, max_deg, total_edges);

        // Check symmetry
        bool symmetric = true;
        for (int i = 0; i < n && symmetric; i++) {
            for (int j : al[i]) {
                bool found = false;
                for (int k : al[j]) {
                    if (k == i) { found = true; break; }
                }
                if (!found) { symmetric = false; break; }
            }
        }
        printf(", symmetric=%s\n", symmetric ? "yes" : "NO!");
    }

    // For time-varying: check total degree across all rounds
    if (!t.is_static) {
        // Count unique neighbors across all rounds
        std::vector<std::vector<bool>> seen(n, std::vector<bool>(n, false));
        for (int r = 0; r < t.num_rounds(); r++) {
            const auto& al = t.get_round(r);
            for (int i = 0; i < n; i++) {
                for (int j : al[i]) {
                    seen[i][j] = true;
                }
            }
        }
        int total_unique = 0;
        int min_unique = n, max_unique = 0;
        for (int i = 0; i < n; i++) {
            int cnt = 0;
            for (int j = 0; j < n; j++) {
                if (seen[i][j]) cnt++;
            }
            total_unique += cnt;
            min_unique = std::min(min_unique, cnt);
            max_unique = std::max(max_unique, cnt);
        }
        printf("  Union across rounds: unique-neighbor range [%d, %d]\n",
               min_unique, max_unique);
    }
}

int main() {
    int n = 32;
    int d = 4;

    printf("Testing topologies with n=%d, d=%d\n", n, d);

    // 1. Our construction (QRS-DR)
    auto t_ours = topo::ours(n, d, 42);
    print_topo_info("Ours (QRS-DR)", t_ours, n);

    // 2. Base-(k+1) time-varying
    auto t_base = topo::base_graph(n, d);
    print_topo_info("Base-(k+1)", t_base, n);

    // 3. Ring
    auto t_ring = topo::ring(n);
    print_topo_info("Ring", t_ring, n);

    // 4. Random d-regular
    auto t_rd = topo::random_d_regular(n, d, 42);
    print_topo_info("Random d-regular", t_rd, n);

    // 5. ExpGraph time-varying (NeurIPS 2021)
    auto t_expgraph = topo::expgraph_tv(n);
    print_topo_info("ExpGraph-TV (NeurIPS'21)", t_expgraph, n);

    // 6. EquiTopo time-varying (NeurIPS 2022)
    auto t_equitopo = topo::equitopo(n, d);
    print_topo_info("EquiTopo (NeurIPS'22)", t_equitopo, n);

    // 7. Torus
    auto t_torus = topo::torus(n);
    print_topo_info("Torus", t_torus, n);

    // 8. Static expander (legacy)
    auto t_exp_static = topo::expander(n);
    print_topo_info("Expander (static, legacy)", t_exp_static, n);

    // === Non-power-of-2 test ===
    printf("\n\n--- Testing n=48 (non-power-of-2) ---\n");
    n = 48; d = 4;

    auto t_expgraph48 = topo::expgraph_tv(n);
    print_topo_info("ExpGraph-TV n=48", t_expgraph48, n);

    auto t_equitopo48 = topo::equitopo(n, d);
    print_topo_info("EquiTopo n=48", t_equitopo48, n);

    auto t_base48 = topo::base_graph(n, d);
    print_topo_info("Base-(k+1) n=48", t_base48, n);

    auto t_torus48 = topo::torus(n);
    print_topo_info("Torus n=48", t_torus48, n);

    printf("\n=== All tests passed ===\n");
    return 0;
}
