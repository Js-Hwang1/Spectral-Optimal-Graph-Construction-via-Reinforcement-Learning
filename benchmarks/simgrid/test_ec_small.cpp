#include <cstdio>
#include <vector>
#include "topologies.hpp"
#include "edge_coloring.hpp"

int main(int argc, char** argv) {
    int n = argc > 1 ? atoi(argv[1]) : 8;
    int d = argc > 2 ? atoi(argv[2]) : 4;
    if ((n*d)%2!=0 || d>=n || n<4 || d<3) { printf("Invalid n=%d d=%d\n",n,d); return 1; }
    auto topo = topo::qrs_dr(n, d);
    const auto& adj = topo.get_round(0);

    printf("n=%d d=%d\n", n, d);
    for (int i = 0; i < n; i++) {
        printf("  %d: [", i);
        for (int j : adj[i]) printf("%d ", j);
        printf("] deg=%d\n", (int)adj[i].size());
    }

    printf("\nDecomposing...\n");
    auto matchings = edge_coloring::decompose(adj, n, d);
    printf("Got %d matchings\n", (int)matchings.size());

    for (int c = 0; c < (int)matchings.size(); c++) {
        printf("  M%d: ", c);
        auto& M = matchings[c];
        for (auto& [u,v] : M.edges()) printf("(%d,%d) ", u, v);
        printf(" unmatched=%d\n", M.unmatched_count());
    }

    return 0;
}
