/*
 * dsgd_sim.cpp — SimGrid-based decentralized scalar consensus simulation
 *
 * Models decentralized SGD with gossip averaging over different topologies
 * under realistic network conditions (heterogeneous bandwidth, link failures,
 * stragglers).
 *
 * Build:
 *   g++ -O2 -std=c++17 -o dsgd_sim dsgd_sim.cpp \
 *       -I/opt/homebrew/Cellar/simgrid/4.1_1/include \
 *       -L/opt/homebrew/Cellar/simgrid/4.1_1/lib -lsimgrid
 *
 * Usage:
 *   ./dsgd_sim --platform platform.xml --topology ours --n 64 --d 4 \
 *              --rounds 500 --msg-size 1000000
 */

#include <simgrid/s4u.hpp>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include <random>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <mutex>
#include <iostream>

#include "topologies.hpp"
#include "edge_coloring.hpp"

namespace sg4 = simgrid::s4u;

// ============================================================
// Global simulation state
// ============================================================

struct SimConfig {
    std::string topology_name = "ours";
    int n = 64;
    int d = 4;
    int rounds = 500;
    uint64_t msg_size = 1000000;  // 1 MB
    double compute_flops = 1e9;
    double fail_rate = 0.0;
    double straggler_frac = 0.0;
    double straggler_slowdown = 10.0;
    std::string straggler_mode = "drop"; // "sync" = all wait (old), "drop" = skip stragglers
    std::string platform_file;
    std::string output_file = "results.json";
    unsigned seed = 42;
    bool one_peer = false;  // one-peer-per-round via edge coloring
    std::vector<double> weights; // per-round alpha values (empty = use default)
};

struct RoundMetrics {
    int round;
    double wall_clock;
    double consensus_error;
    double max_error;
};

// Shared state protected by mutex
static std::mutex g_mutex;
static std::vector<double> g_values;       // x_i for each worker
static std::vector<double> g_new_values;   // buffer for next round
static std::vector<RoundMetrics> g_metrics;
static topo::Topology g_topology;
static SimConfig g_config;
static std::vector<bool> g_is_straggler;

// Synchronization barrier
static sg4::BarrierPtr g_barrier;

// Per-round link failure mask (recomputed by worker 0 each round)
static std::vector<std::vector<bool>> g_link_failed; // g_link_failed[i][j]
static int g_current_round = 0;

// ============================================================
// Worker Actor
// ============================================================

static void worker_actor(int my_id) {
    double straggler_mult = g_is_straggler[my_id] ? g_config.straggler_slowdown : 1.0;

    for (int round = 0; round < g_config.rounds; round++) {

        // === 1. Local SGD step (compute) ===
        double flops = g_config.compute_flops * straggler_mult;
        sg4::this_actor::execute(flops);

        // === Worker 0: setup link failures for this round ===
        if (my_id == 0) {
            std::mt19937 rng(g_config.seed + round * 1000);
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            g_link_failed.assign(g_config.n, std::vector<bool>(g_config.n, false));
            if (g_config.fail_rate > 0.0) {
                for (int i = 0; i < g_config.n; i++) {
                    for (int j = i + 1; j < g_config.n; j++) {
                        if (dist(rng) < g_config.fail_rate) {
                            g_link_failed[i][j] = true;
                            g_link_failed[j][i] = true;
                        }
                    }
                }
            }
        }

        // Synchronize so all workers see the link failure state
        g_barrier->wait();

        // === 2. Gossip: exchange parameters with neighbors ===
        const auto& neighbors = g_topology.get_round(round)[my_id];

        // Determine active neighbors (not failed, not straggler in drop mode)
        std::vector<int> active_neighbors;
        for (int j : neighbors) {
            if (g_link_failed[my_id][j]) continue;
            // In "drop" mode: skip straggler neighbors (timeout model)
            // Also skip if I am a straggler (my neighbors won't wait for me)
            if (g_config.straggler_mode == "drop") {
                if (g_is_straggler[j] || g_is_straggler[my_id]) continue;
            }
            active_neighbors.push_back(j);
        }

        // Use ASYNC sends to avoid deadlock: issue all put_async first,
        // then do blocking receives, then wait on sends to complete.
        std::vector<sg4::CommPtr> pending_sends;

        // Async send my value to each active neighbor
        for (int j : active_neighbors) {
            auto* payload = new double(g_values[my_id]);
            std::string mbox_name = "mbox_" + std::to_string(j) +
                                    "_r" + std::to_string(round) +
                                    "_from_" + std::to_string(my_id);
            sg4::Mailbox* mbox = sg4::Mailbox::by_name(mbox_name);
            pending_sends.push_back(mbox->put_async(payload, g_config.msg_size));
        }

        // Blocking receive from each active neighbor
        std::vector<double> received;
        for (int j : active_neighbors) {
            std::string mbox_name = "mbox_" + std::to_string(my_id) +
                                    "_r" + std::to_string(round) +
                                    "_from_" + std::to_string(j);
            sg4::Mailbox* mbox = sg4::Mailbox::by_name(mbox_name);
            auto* val = mbox->get<double>();
            received.push_back(*val);
            delete val;
        }

        // Wait for all sends to complete
        for (auto& comm : pending_sends) {
            comm->wait();
        }

        // === 3. Gossip update ===
        double my_val = g_values[my_id];
        double new_val = my_val;

        if (!active_neighbors.empty()) {
            if (!g_config.weights.empty()) {
                // Per-round alpha from --weights (cycles through the list)
                // W = (1-alpha)*I + alpha*P  for one-peer matching
                int w_idx = round % (int)g_config.weights.size();
                double alpha = g_config.weights[w_idx];
                // One-peer: exactly 1 neighbor
                new_val = (1.0 - alpha) * my_val + alpha * received[0];
            } else if (g_topology.is_static) {
                // Static topology: W = (1/2)(I + A/d)
                // w_ij = 1/(2d) for each neighbor, w_ii = 1/2
                int deg = (int)active_neighbors.size();
                double w_neighbor = 1.0 / (2.0 * deg);
                double w_self = 1.0 - w_neighbor * deg;
                new_val = w_self * my_val;
                for (double rv : received) {
                    new_val += w_neighbor * rv;
                }
            } else if (g_config.topology_name == "expgraph" ||
                       g_config.topology_name == "equitopo") {
                // Time-varying one-peer: W = (1/2)(I + P_r)
                // Node averages with self and its one partner equally
                // For ExpGraph non-pow2 (deg=2): W = (1/3)(I + A)
                int deg = (int)active_neighbors.size();
                double w = 1.0 / (deg + 1.0);
                new_val = w * my_val;
                for (double rv : received) {
                    new_val += w * rv;
                }
            } else {
                // Time-varying (Base-(k+1), generic): equal weight averaging
                // w_ij = 1/(deg+1) for neighbors and self
                int deg = (int)active_neighbors.size();
                double w = 1.0 / (deg + 1.0);
                new_val = w * my_val;
                for (double rv : received) {
                    new_val += w * rv;
                }
            }
        }

        // Synchronize before updating values
        g_barrier->wait();

        g_new_values[my_id] = new_val;

        // Synchronize so all new values are written
        g_barrier->wait();

        // Copy new values to current values
        if (my_id == 0) {
            for (int i = 0; i < g_config.n; i++) {
                g_values[i] = g_new_values[i];
            }
        }

        g_barrier->wait();

        // === 4. Record metrics (worker 0 only) ===
        if (my_id == 0) {
            double x_bar = 0.0;
            for (int i = 0; i < g_config.n; i++) x_bar += g_values[i];
            x_bar /= g_config.n;

            double sum_sq = 0.0, max_sq = 0.0;
            for (int i = 0; i < g_config.n; i++) {
                double diff = g_values[i] - x_bar;
                double sq = diff * diff;
                sum_sq += sq;
                max_sq = std::max(max_sq, sq);
            }

            RoundMetrics m;
            m.round = round;
            m.wall_clock = sg4::Engine::get_clock();
            m.consensus_error = sum_sq / g_config.n;
            m.max_error = max_sq;
            g_metrics.push_back(m);
        }

        g_barrier->wait();
    }
}

// ============================================================
// JSON output
// ============================================================

static void write_json(const std::string& filename) {
    std::ofstream f(filename);
    f << "{\n";
    f << "  \"config\": {\n";
    f << "    \"topology\": \"" << g_config.topology_name << "\",\n";
    f << "    \"n\": " << g_config.n << ",\n";
    f << "    \"d\": " << g_config.d << ",\n";
    f << "    \"rounds\": " << g_config.rounds << ",\n";
    f << "    \"msg_size\": " << g_config.msg_size << ",\n";
    f << "    \"compute_flops\": " << g_config.compute_flops << ",\n";
    f << "    \"fail_rate\": " << g_config.fail_rate << ",\n";
    f << "    \"straggler_frac\": " << g_config.straggler_frac << ",\n";
    f << "    \"straggler_slowdown\": " << g_config.straggler_slowdown << ",\n";
    f << "    \"straggler_mode\": \"" << g_config.straggler_mode << "\",\n";
    f << "    \"seed\": " << g_config.seed << ",\n";
    f << "    \"platform\": \"" << g_config.platform_file << "\",\n";
    f << "    \"one_peer\": " << (g_config.one_peer ? "true" : "false") << "\n";
    f << "  },\n";
    f << "  \"rounds\": [\n";
    for (size_t i = 0; i < g_metrics.size(); i++) {
        const auto& m = g_metrics[i];
        f << "    {\"round\": " << m.round
          << ", \"wall_clock\": " << m.wall_clock
          << ", \"consensus_error\": " << m.consensus_error
          << ", \"max_error\": " << m.max_error << "}";
        if (i + 1 < g_metrics.size()) f << ",";
        f << "\n";
    }
    f << "  ]\n";
    f << "}\n";
    f.close();
    std::cerr << "Results written to " << filename << std::endl;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char* argv[]) {
    sg4::Engine e(&argc, argv);

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--topology" && i + 1 < argc) g_config.topology_name = argv[++i];
        else if (arg == "--n" && i + 1 < argc) g_config.n = std::stoi(argv[++i]);
        else if (arg == "--d" && i + 1 < argc) g_config.d = std::stoi(argv[++i]);
        else if (arg == "--rounds" && i + 1 < argc) g_config.rounds = std::stoi(argv[++i]);
        else if (arg == "--msg-size" && i + 1 < argc) g_config.msg_size = std::stoull(argv[++i]);
        else if (arg == "--compute-flops" && i + 1 < argc) g_config.compute_flops = std::stod(argv[++i]);
        else if (arg == "--fail-rate" && i + 1 < argc) g_config.fail_rate = std::stod(argv[++i]);
        else if (arg == "--straggler-frac" && i + 1 < argc) g_config.straggler_frac = std::stod(argv[++i]);
        else if (arg == "--straggler-slowdown" && i + 1 < argc) g_config.straggler_slowdown = std::stod(argv[++i]);
        else if (arg == "--platform" && i + 1 < argc) g_config.platform_file = argv[++i];
        else if (arg == "--output" && i + 1 < argc) g_config.output_file = argv[++i];
        else if (arg == "--seed" && i + 1 < argc) g_config.seed = std::stoul(argv[++i]);
        else if (arg == "--straggler-mode" && i + 1 < argc) g_config.straggler_mode = argv[++i];
        else if (arg == "--one-peer") g_config.one_peer = true;
        else if (arg == "--weights" && i + 1 < argc) {
            // Parse comma-separated alpha values
            std::string wstr = argv[++i];
            std::stringstream ss(wstr);
            std::string token;
            while (std::getline(ss, token, ',')) {
                g_config.weights.push_back(std::stod(token));
            }
        }
    }

    if (g_config.platform_file.empty()) {
        std::cerr << "Error: --platform is required\n";
        return 1;
    }

    // Load platform
    e.load_platform(g_config.platform_file);

    // Build topology
    std::cerr << "Building topology: " << g_config.topology_name
              << " (n=" << g_config.n << ", d=" << g_config.d << ")\n";

    if (g_config.topology_name == "ours") {
        g_topology = topo::ours(g_config.n, g_config.d, g_config.seed);
    } else if (g_config.topology_name == "base") {
        g_topology = topo::base_graph(g_config.n, g_config.d);
    } else if (g_config.topology_name == "ring") {
        g_topology = topo::ring(g_config.n);
    } else if (g_config.topology_name == "random") {
        g_topology = topo::random_d_regular(g_config.n, g_config.d, g_config.seed);
    } else if (g_config.topology_name == "expander") {
        g_topology = topo::expander(g_config.n);
    } else if (g_config.topology_name == "expgraph") {
        // Time-varying one-peer exponential graph (Ying et al., NeurIPS 2021)
        g_topology = topo::expgraph_tv(g_config.n);
    } else if (g_config.topology_name == "equitopo") {
        // Time-varying EquiTopo (Jin et al., NeurIPS 2022)
        g_topology = topo::equitopo(g_config.n, g_config.d);
    } else if (g_config.topology_name == "torus") {
        g_topology = topo::torus(g_config.n);
    } else {
        std::cerr << "Unknown topology: " << g_config.topology_name << "\n";
        return 1;
    }

    // Report topology info
    if (g_topology.is_static) {
        int max_deg = 0, min_deg = g_config.n;
        for (int i = 0; i < g_config.n; i++) {
            int deg = (int)g_topology.get_round(0)[i].size();
            max_deg = std::max(max_deg, deg);
            min_deg = std::min(min_deg, deg);
        }
        std::cerr << "  Static topology: degree range [" << min_deg
                  << ", " << max_deg << "]\n";
    } else {
        std::cerr << "  Time-varying topology: " << g_topology.num_rounds()
                  << " rounds per cycle\n";
        for (int r = 0; r < g_topology.num_rounds(); r++) {
            int max_deg = 0;
            for (int i = 0; i < g_config.n; i++) {
                int deg = (int)g_topology.get_round(r)[i].size();
                max_deg = std::max(max_deg, deg);
            }
            std::cerr << "    Round " << r << ": max degree " << max_deg << "\n";
        }
    }

    // One-peer-per-round: decompose static topology into d matchings
    if (g_config.one_peer && g_topology.is_static) {
        std::cerr << "Decomposing into matchings for one-peer-per-round...\n";
        const auto& adj = g_topology.get_round(0);
        auto matchings = edge_coloring::decompose(adj, g_config.n, g_config.d);
        std::cerr << "  Got " << matchings.size() << " matchings\n";

        // Convert to time-varying topology: each matching is one round
        topo::Topology tv_topo;
        tv_topo.is_static = false;
        for (auto& M : matchings) {
            // Build adjacency list for this matching: each node has 0 or 1 neighbor
            topo::AdjList round_adj(g_config.n);
            for (int i = 0; i < g_config.n; i++) {
                if (M.partner[i] >= 0) {
                    round_adj[i].push_back(M.partner[i]);
                }
            }
            tv_topo.rounds.push_back(round_adj);
        }
        g_topology = tv_topo;
        std::cerr << "  Topology now has " << g_topology.num_rounds()
                  << " rounds (one-peer-per-round)\n";
    }

    // Initialize worker values: x_i ~ N(0,1)
    std::mt19937 rng(g_config.seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    g_values.resize(g_config.n);
    g_new_values.resize(g_config.n);
    for (int i = 0; i < g_config.n; i++) {
        g_values[i] = normal(rng);
    }

    // Designate stragglers
    g_is_straggler.assign(g_config.n, false);
    int num_stragglers = (int)(g_config.straggler_frac * g_config.n);
    // Pick the last num_stragglers workers as stragglers (deterministic)
    for (int i = g_config.n - num_stragglers; i < g_config.n; i++) {
        g_is_straggler[i] = true;
    }

    // Initialize link failure state
    g_link_failed.assign(g_config.n, std::vector<bool>(g_config.n, false));

    // Create barrier
    g_barrier = sg4::Barrier::create(g_config.n);

    // Create worker actors
    for (int i = 0; i < g_config.n; i++) {
        std::string host_name = "worker_" + std::to_string(i);
        sg4::Host::by_name(host_name)->add_actor(
            "worker_" + std::to_string(i), worker_actor, i);
    }

    // Run simulation
    std::cerr << "Running simulation for " << g_config.rounds << " rounds...\n";
    e.run();
    std::cerr << "Simulation complete. Wall clock: " << e.get_clock() << "s\n";

    // Write results
    write_json(g_config.output_file);

    return 0;
}
