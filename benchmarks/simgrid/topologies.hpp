/*
 * topologies.hpp — Topology generators for decentralized learning simulation
 *
 * Implements: QRS-DR, Base-(k+1) Graph, Ring, Random d-regular
 * Header-only C++17 library.
 */

#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <numeric>
#include <random>
#include <cstring>

namespace topo {

// ============================================================
// Common types
// ============================================================

// Adjacency list representation: adj[i] = list of neighbors of node i
using AdjList = std::vector<std::vector<int>>;

// A time-varying topology: sequence of adjacency lists (one per round)
struct Topology {
    std::vector<AdjList> rounds; // rounds[r][i] = neighbors of node i in round r
    bool is_static;              // if true, rounds has exactly 1 entry

    const AdjList& get_round(int r) const {
        if (is_static) return rounds[0];
        return rounds[r % (int)rounds.size()];
    }

    int num_rounds() const { return (int)rounds.size(); }
};

// ============================================================
// Number theory helpers (ported from PoC/algo.c)
// ============================================================

namespace detail {

inline bool is_prime(int x) {
    if (x < 2) return false;
    if (x < 4) return true;
    if (x % 2 == 0) return false;
    for (int i = 3; (long long)i * i <= x; i += 2)
        if (x % i == 0) return false;
    return true;
}

inline int next_prime(int n) {
    int p = n + 1;
    if (p <= 2) return 2;
    if (p % 2 == 0) p++;
    while (!is_prime(p)) p += 2;
    return p;
}

inline long long mod_pow(long long base, long long exp, long long m) {
    long long result = 1;
    base %= m;
    if (base < 0) base += m;
    while (exp > 0) {
        if (exp & 1) result = result * base % m;
        base = base * base % m;
        exp >>= 1;
    }
    return result;
}

inline int primitive_root(int p) {
    if (p == 2) return 1;
    int pm1 = p - 1;

    // Factor p-1
    std::vector<int> factors;
    int tmp = pm1;
    for (int d = 2; (long long)d * d <= tmp; d++) {
        if (tmp % d == 0) {
            factors.push_back(d);
            while (tmp % d == 0) tmp /= d;
        }
    }
    if (tmp > 1) factors.push_back(tmp);

    for (int g = 2; g < p; g++) {
        bool ok = true;
        for (int f : factors) {
            if (mod_pow(g, pm1 / f, p) == 1) {
                ok = false;
                break;
            }
        }
        if (ok) return g;
    }
    return 2;
}

} // namespace detail

// ============================================================
// QRS-DR: Quadratic Residue Scatter + Degree Regularization
// ============================================================

inline Topology qrs_dr(int n, int d) {
    assert(n >= 4 && d >= 3 && d < n && (n * d) % 2 == 0);

    // Adjacency matrix (flat)
    std::vector<int> adj(n * n, 0);
    std::vector<int> degs(n, 0);

    auto get = [&](int i, int j) -> int { return adj[i * n + j]; };
    auto set = [&](int i, int j, int val) {
        adj[i * n + j] = val;
        adj[j * n + i] = val;
    };
    auto recompute_degs = [&]() {
        for (int i = 0; i < n; i++) {
            int s = 0;
            for (int j = 0; j < n; j++) s += adj[i * n + j];
            degs[i] = s;
        }
    };
    auto count_edges = [&]() -> int {
        int total = 0;
        for (int i = 0; i < n; i++) total += degs[i];
        return total / 2;
    };

    // Phase 1: QR Scatter
    int p = detail::next_prime(n);
    int g = detail::primitive_root(p);

    for (int k = 0; k < d; k++) {
        long long c = detail::mod_pow(g, k + 1, p);
        for (int i = 0; i < n; i++) {
            long long t = (i + 1) % p;
            if (t == 0) t = 1;
            int j = (int)((t * (t + c)) % p % n);
            if (j == i) j = (j + 1) % n;
            set(i, j, 1);
        }
    }

    // Remove self-loops
    for (int i = 0; i < n; i++) adj[i * n + i] = 0;
    recompute_degs();

    // Phase 2: Degree Regularization
    int target_edges = n * d / 2;
    int current_edges = count_edges();

    // Remove surplus edges
    while (current_edges > target_edges) {
        int u = -1, best_deg = -1;
        for (int i = 0; i < n; i++) {
            if (degs[i] > best_deg) { best_deg = degs[i]; u = i; }
        }
        if (u < 0) break;

        int v = -1, best_vdeg = -1;
        for (int j = 0; j < n; j++) {
            if (get(u, j) && degs[j] > best_vdeg) { best_vdeg = degs[j]; v = j; }
        }
        if (v < 0) break;

        set(u, v, 0);
        degs[u]--;
        degs[v]--;
        current_edges--;
    }

    // Add deficit edges
    while (current_edges < target_edges) {
        int u = -1, best_deg = n + 1;
        for (int i = 0; i < n; i++) {
            if (degs[i] < best_deg) { best_deg = degs[i]; u = i; }
        }
        if (u < 0) break;

        int v = -1, best_vdeg = n + 1;
        for (int j = 0; j < n; j++) {
            if (j != u && !get(u, j) && degs[j] < best_vdeg) { best_vdeg = degs[j]; v = j; }
        }
        if (v < 0) break;

        set(u, v, 1);
        degs[u]++;
        degs[v]++;
        current_edges++;
    }

    // Degree equalization via edge swaps
    for (int iter = 0; iter < n * n; iter++) {
        bool has_over = false, has_under = false;
        for (int i = 0; i < n; i++) {
            if (degs[i] > d) has_over = true;
            if (degs[i] < d) has_under = true;
        }
        if (!has_over || !has_under) break;

        // u = lowest-index node with max degree > d
        int u = -1, max_deg = d;
        for (int i = 0; i < n; i++) {
            if (degs[i] > max_deg) { max_deg = degs[i]; }
        }
        for (int i = 0; i < n; i++) {
            if (degs[i] == max_deg && degs[i] > d) { u = i; break; }
        }

        // w = lowest-index node with min degree < d
        int w = -1, min_deg = d;
        for (int i = 0; i < n; i++) {
            if (degs[i] < min_deg) { min_deg = degs[i]; }
        }
        for (int i = 0; i < n; i++) {
            if (degs[i] == min_deg && degs[i] < d) { w = i; break; }
        }

        if (u < 0 || w < 0) break;

        // Collect neighbors of u, sorted by (-deg, +index)
        std::vector<int> nb;
        for (int j = 0; j < n; j++) {
            if (get(u, j)) nb.push_back(j);
        }
        std::sort(nb.begin(), nb.end(), [&](int a, int b) {
            if (degs[a] != degs[b]) return degs[a] > degs[b];
            return a < b;
        });

        // Attempt 1: Direct transfer
        bool transferred = false;
        for (int v : nb) {
            if (v == w) continue;
            if (!get(w, v)) {
                set(u, v, 0);
                set(w, v, 1);
                degs[u]--;
                degs[w]++;
                transferred = true;
                break;
            }
        }
        if (transferred) continue;

        // Attempt 2: Bridge via {u, w}
        if (!get(u, w)) {
            set(u, w, 1);
            degs[u]++;
            degs[w]++;

            int best_v = -1, best_vdeg = -1;
            for (int j = 0; j < n; j++) {
                if (j != w && get(u, j) && degs[j] > best_vdeg) {
                    best_vdeg = degs[j]; best_v = j;
                }
            }
            if (best_v >= 0) {
                for (int j = 0; j < n; j++) {
                    if (j != w && get(u, j) && degs[j] == best_vdeg) {
                        best_v = j; break;
                    }
                }
                set(u, best_v, 0);
                degs[u]--;
                degs[best_v]--;
            }
        } else {
            // Attempt 3: Indirect transfer
            int best_v = -1, best_vdeg = -1;
            for (int j = 0; j < n; j++) {
                if (get(u, j) && degs[j] > best_vdeg) {
                    best_vdeg = degs[j]; best_v = j;
                }
            }
            for (int j = 0; j < n; j++) {
                if (get(u, j) && degs[j] == best_vdeg) {
                    best_v = j; break;
                }
            }
            if (best_v >= 0) {
                set(u, best_v, 0);
                degs[u]--;
                degs[best_v]--;
            }

            for (int x = 0; x < n; x++) {
                if (x != w && !get(w, x) && degs[x] < d) {
                    set(w, x, 1);
                    degs[w]++;
                    degs[x]++;
                    break;
                }
            }
        }
    }

    // Convert to adjacency list
    AdjList al(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (adj[i * n + j]) al[i].push_back(j);
        }
    }

    Topology topo;
    topo.is_static = true;
    topo.rounds.push_back(std::move(al));
    return topo;
}

// ============================================================
// Base-(k+1) Graph (Time-varying, Algorithm 3)
// ============================================================

namespace detail {

// Build k-peer hypercube on a node set: returns sequence of edge sets (rounds)
// Each round is a set of pairs. Each node participates in at most k edges per round.
// For a set of size m, we need ceil(log_{k+1}(m)) rounds.
inline std::vector<std::vector<std::pair<int,int>>>
k_peer_hypercube(const std::vector<int>& nodes, int k) {
    int m = (int)nodes.size();
    if (m <= 1) return {};

    // Number of rounds needed
    int num_rounds = 1;
    {
        int cap = k + 1;
        while (cap < m) {
            cap *= (k + 1);
            num_rounds++;
        }
    }

    std::vector<std::vector<std::pair<int,int>>> rounds(num_rounds);

    for (int r = 0; r < num_rounds; r++) {
        // In round r, group nodes into groups of (k+1) based on
        // digit r of the base-(k+1) representation
        int stride = 1;
        for (int i = 0; i < r; i++) stride *= (k + 1);
        int group_size = stride * (k + 1);

        for (int g_start = 0; g_start < m; g_start += group_size) {
            // Within this group, pair nodes that differ only in digit r
            for (int offset = 0; offset < stride && g_start + offset < m; offset++) {
                // Collect nodes in this "fiber" (same digits except digit r)
                std::vector<int> fiber;
                for (int d = 0; d < k + 1; d++) {
                    int idx = g_start + offset + d * stride;
                    if (idx < m) fiber.push_back(idx);
                }
                // Connect all pairs in the fiber (complete subgraph)
                for (int a = 0; a < (int)fiber.size(); a++) {
                    for (int b = a + 1; b < (int)fiber.size(); b++) {
                        rounds[r].push_back({nodes[fiber[a]], nodes[fiber[b]]});
                    }
                }
            }
        }
    }

    return rounds;
}

// GCD for integers
inline int gcd(int a, int b) {
    while (b) { int t = b; b = a % b; a = t; }
    return a;
}

// Check if x is coprime to all of 2, 3, ..., k+1
inline bool coprime_to_range(int x, int k) {
    for (int i = 2; i <= k + 1; i++) {
        if (gcd(x, i) != 1) return false;
    }
    return true;
}

} // namespace detail

inline Topology base_graph(int n, int k) {
    assert(n >= 2 && k >= 2);

    // Decompose n = p * q where q is coprime to 2,...,k+1
    // We want p to be a multiple of lcm(2,...,k+1) if possible,
    // or at least divisible by small primes up to k+1.
    // Simple approach: find largest q <= n that is coprime to 2..k+1 and divides n
    int p = 1, q = n;
    for (int candidate_q = n; candidate_q >= 1; candidate_q--) {
        if (n % candidate_q == 0 && detail::coprime_to_range(candidate_q, k)) {
            q = candidate_q;
            p = n / candidate_q;
            break;
        }
    }

    // Split V into p subsets of size q
    // V_l = {l*q, l*q+1, ..., l*q+q-1} for l = 0,...,p-1
    std::vector<std::vector<int>> subsets(p);
    for (int l = 0; l < p; l++) {
        for (int i = 0; i < q; i++) {
            subsets[l].push_back(l * q + i);
        }
    }

    // For each subset, build the Simple Base-(k+1) Graph
    // Simple Base-(k+1) uses k-peer hypercube within each subset
    std::vector<std::vector<std::pair<int,int>>> intra_rounds;
    if (q > 1) {
        // Build k-peer hypercube for each subset and merge rounds
        for (int l = 0; l < p; l++) {
            auto sub_rounds = detail::k_peer_hypercube(subsets[l], k);
            // Extend intra_rounds to match
            while ((int)intra_rounds.size() < (int)sub_rounds.size()) {
                intra_rounds.push_back({});
            }
            for (int r = 0; r < (int)sub_rounds.size(); r++) {
                for (auto& e : sub_rounds[r]) {
                    intra_rounds[r].push_back(e);
                }
            }
        }
    }

    // Build cross-subset edges using k-peer hypercube on the p subsets
    // For cross-subset: connect representative nodes from different subsets
    // Each subset l has a representative = l*q (first node)
    // We do a k-peer hypercube on the subset indices
    std::vector<std::vector<std::pair<int,int>>> cross_rounds;
    if (p > 1) {
        // For each position within subsets, build cross-subset connections
        // Use k-peer hypercube on subset indices, applied to each offset
        std::vector<int> subset_indices(p);
        std::iota(subset_indices.begin(), subset_indices.end(), 0);
        auto idx_rounds = detail::k_peer_hypercube(subset_indices, k);

        for (auto& round_edges : idx_rounds) {
            std::vector<std::pair<int,int>> edges;
            for (auto& [a, b] : round_edges) {
                // Connect corresponding nodes: for each offset in subset
                // To keep degree bounded, just connect the first node of each subset
                // Then average propagates through intra-subset rounds
                for (int offset = 0; offset < q; offset++) {
                    edges.push_back({a * q + offset, b * q + offset});
                }
            }
            cross_rounds.push_back(std::move(edges));
        }
    }

    // Combine into final sequence of rounds
    // First all intra-rounds, then all cross-rounds
    int total_rounds = std::max(1, (int)intra_rounds.size() + (int)cross_rounds.size());
    std::vector<AdjList> all_rounds;

    for (auto& round_edges : intra_rounds) {
        AdjList al(n);
        for (auto& [a, b] : round_edges) {
            al[a].push_back(b);
            al[b].push_back(a);
        }
        all_rounds.push_back(std::move(al));
    }

    for (auto& round_edges : cross_rounds) {
        AdjList al(n);
        for (auto& [a, b] : round_edges) {
            al[a].push_back(b);
            al[b].push_back(a);
        }
        all_rounds.push_back(std::move(al));
    }

    // If no rounds were generated, add a single empty round
    if (all_rounds.empty()) {
        all_rounds.push_back(AdjList(n));
    }

    Topology topo;
    topo.is_static = false;
    topo.rounds = std::move(all_rounds);
    return topo;
}

// ============================================================
// Ring topology (static)
// ============================================================

inline Topology ring(int n) {
    assert(n >= 3);
    AdjList al(n);
    for (int i = 0; i < n; i++) {
        al[i].push_back((i - 1 + n) % n);
        al[i].push_back((i + 1) % n);
    }
    Topology topo;
    topo.is_static = true;
    topo.rounds.push_back(std::move(al));
    return topo;
}

// ============================================================
// Random d-regular graph (static, pairing model)
// ============================================================

inline Topology random_d_regular(int n, int d, unsigned seed = 42) {
    assert(n >= d + 1 && (n * d) % 2 == 0);
    std::mt19937 rng(seed);

    AdjList best_al;
    int best_defects = n * d; // track best attempt

    // Pairing model with retries
    for (int attempt = 0; attempt < 100; attempt++) {
        // Create d copies of each node
        std::vector<int> points;
        points.reserve(n * d);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                points.push_back(i);
            }
        }

        // Random perfect matching
        std::shuffle(points.begin(), points.end(), rng);

        AdjList al(n);
        std::vector<std::vector<bool>> seen(n, std::vector<bool>(n, false));
        int defects = 0;

        for (int i = 0; i < (int)points.size(); i += 2) {
            int u = points[i], v = points[i + 1];
            if (u == v || seen[u][v]) {
                defects++;
                continue;
            }
            seen[u][v] = true;
            seen[v][u] = true;
            al[u].push_back(v);
            al[v].push_back(u);
        }

        if (defects == 0) {
            Topology topo;
            topo.is_static = true;
            topo.rounds.push_back(std::move(al));
            return topo;
        }

        if (defects < best_defects) {
            best_defects = defects;
            best_al = std::move(al);
        }
    }

    // Return best attempt (may not be perfectly d-regular)
    Topology topo;
    topo.is_static = true;
    topo.rounds.push_back(std::move(best_al));
    return topo;
}

} // namespace topo
