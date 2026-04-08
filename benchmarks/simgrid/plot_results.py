#!/usr/bin/env python3
"""
plot_results.py — Generate paper-ready figures from SimGrid experiment results.

Usage:
  python3 plot_results.py [results_dir]

Produces:
  fig1_hetero_bandwidth.pdf   - Consensus error vs wall-clock time (Exp 1)
  fig2_link_failure.pdf       - Final consensus error vs fail_rate (Exp 2)
  fig3_straggler.pdf          - Wall-clock per round vs straggler fraction (Exp 3)
  fig4_fattree.pdf            - Consensus error vs rounds on fat-tree (Exp 4)
"""

import sys
import os
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Paper-quality settings
rcParams["font.family"] = "serif"
rcParams["font.size"] = 11
rcParams["axes.labelsize"] = 12
rcParams["legend.fontsize"] = 10
rcParams["xtick.labelsize"] = 10
rcParams["ytick.labelsize"] = 10
rcParams["figure.figsize"] = (5.5, 4.0)
rcParams["figure.dpi"] = 300
rcParams["savefig.bbox"] = "tight"
rcParams["savefig.pad_inches"] = 0.05

TOPO_NAMES = {
    "ours": "QRS-DR",
    "base": "Base-(k+1)",
    "ring": "Ring",
    "random": "Random d-regular",
}

TOPO_COLORS = {
    "ours": "#d62728",    # red
    "base": "#1f77b4",     # blue
    "ring": "#2ca02c",     # green
    "random": "#ff7f0e",   # orange
}

TOPO_MARKERS = {
    "ours": "o",
    "base": "s",
    "ring": "^",
    "random": "D",
}

TOPO_LINESTYLES = {
    "ours": "-",
    "base": "--",
    "ring": "-.",
    "random": ":",
}

TOPOLOGIES = ["ours", "base", "ring", "random"]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fig1_hetero_bandwidth(results_dir, output_dir):
    """Consensus error vs wall-clock time on 3-tier platform."""
    fig, ax = plt.subplots()

    for topo in TOPOLOGIES:
        path = os.path.join(results_dir, "exp1_hetero", f"{topo}.json")
        if not os.path.exists(path):
            print(f"  Skipping {path} (not found)")
            continue
        data = load_json(path)
        rounds = data["rounds"]
        times = [r["wall_clock"] for r in rounds]
        errors = [r["consensus_error"] for r in rounds]

        ax.semilogy(times, errors,
                    label=TOPO_NAMES[topo],
                    color=TOPO_COLORS[topo],
                    linestyle=TOPO_LINESTYLES[topo],
                    marker=TOPO_MARKERS[topo],
                    markevery=max(1, len(times) // 10),
                    markersize=5,
                    linewidth=1.5)

    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Consensus error")
    ax.set_title("Heterogeneous Bandwidth (3-tier)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    outpath = os.path.join(output_dir, "fig1_hetero_bandwidth.pdf")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved {outpath}")


def fig2_link_failure(results_dir, output_dir):
    """Final consensus error vs fail_rate."""
    fig, ax = plt.subplots()
    fail_rates = [0.0, 0.01, 0.05, 0.10]

    for topo in TOPOLOGIES:
        final_errors = []
        for fr in fail_rates:
            path = os.path.join(results_dir, "exp2_failure",
                                f"{topo}_fr{fr}.json")
            if not os.path.exists(path):
                final_errors.append(np.nan)
                continue
            data = load_json(path)
            rounds = data["rounds"]
            if rounds:
                final_errors.append(rounds[-1]["consensus_error"])
            else:
                final_errors.append(np.nan)

        ax.semilogy(fail_rates, final_errors,
                    label=TOPO_NAMES[topo],
                    color=TOPO_COLORS[topo],
                    linestyle=TOPO_LINESTYLES[topo],
                    marker=TOPO_MARKERS[topo],
                    markersize=6,
                    linewidth=1.5)

    ax.set_xlabel("Link failure probability")
    ax.set_ylabel("Final consensus error (round 500)")
    ax.set_title("Link Failure Resilience")
    ax.legend()
    ax.grid(True, alpha=0.3)

    outpath = os.path.join(output_dir, "fig2_link_failure.pdf")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved {outpath}")


def fig3_straggler(results_dir, output_dir):
    """Wall-clock per round vs straggler fraction."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    strag_fracs = [0.0, 0.05, 0.10, 0.20]

    for topo in TOPOLOGIES:
        avg_times = []
        final_errors = []
        for sf in strag_fracs:
            path = os.path.join(results_dir, "exp3_straggler",
                                f"{topo}_sf{sf}.json")
            if not os.path.exists(path):
                avg_times.append(np.nan)
                final_errors.append(np.nan)
                continue
            data = load_json(path)
            rounds = data["rounds"]
            if len(rounds) >= 2:
                total_time = rounds[-1]["wall_clock"] - rounds[0]["wall_clock"]
                avg_times.append(total_time / (len(rounds) - 1))
            else:
                avg_times.append(np.nan)

            if rounds:
                final_errors.append(rounds[-1]["consensus_error"])
            else:
                final_errors.append(np.nan)

        ax1.plot(strag_fracs, avg_times,
                 label=TOPO_NAMES[topo],
                 color=TOPO_COLORS[topo],
                 linestyle=TOPO_LINESTYLES[topo],
                 marker=TOPO_MARKERS[topo],
                 markersize=6,
                 linewidth=1.5)

        ax2.semilogy(strag_fracs, final_errors,
                     label=TOPO_NAMES[topo],
                     color=TOPO_COLORS[topo],
                     linestyle=TOPO_LINESTYLES[topo],
                     marker=TOPO_MARKERS[topo],
                     markersize=6,
                     linewidth=1.5)

    ax1.set_xlabel("Straggler fraction")
    ax1.set_ylabel("Avg wall-clock per round (s)")
    ax1.set_title("Straggler Impact: Latency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Straggler fraction")
    ax2.set_ylabel("Final consensus error")
    ax2.set_title("Straggler Impact: Convergence")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    outpath = os.path.join(output_dir, "fig3_straggler.pdf")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved {outpath}")


def fig4_fattree(results_dir, output_dir):
    """Consensus error vs rounds on fat-tree."""
    fig, ax = plt.subplots()

    for topo in TOPOLOGIES:
        path = os.path.join(results_dir, "exp4_fattree", f"{topo}.json")
        if not os.path.exists(path):
            print(f"  Skipping {path} (not found)")
            continue
        data = load_json(path)
        rounds_data = data["rounds"]
        round_nums = [r["round"] for r in rounds_data]
        errors = [r["consensus_error"] for r in rounds_data]

        ax.semilogy(round_nums, errors,
                    label=TOPO_NAMES[topo],
                    color=TOPO_COLORS[topo],
                    linestyle=TOPO_LINESTYLES[topo],
                    marker=TOPO_MARKERS[topo],
                    markevery=max(1, len(round_nums) // 10),
                    markersize=5,
                    linewidth=1.5)

    ax.set_xlabel("Communication round")
    ax.set_ylabel("Consensus error")
    ax.set_title("Fat-tree Topology (k=4)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    outpath = os.path.join(output_dir, "fig4_fattree.pdf")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved {outpath}")


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    output_dir = os.path.join(results_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    print("Generating paper figures...")
    print("")

    print("Figure 1: Heterogeneous bandwidth")
    fig1_hetero_bandwidth(results_dir, output_dir)

    print("Figure 2: Link failure resilience")
    fig2_link_failure(results_dir, output_dir)

    print("Figure 3: Straggler robustness")
    fig3_straggler(results_dir, output_dir)

    print("Figure 4: Fat-tree placement")
    fig4_fattree(results_dir, output_dir)

    print("")
    print(f"All figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
