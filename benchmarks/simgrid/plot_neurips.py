#!/usr/bin/env python3
"""
NeurIPS-quality figures from SimGrid experiment results.
Produces 4 figures matching the paper's Section 6.3.
"""

import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

# NeurIPS formatting
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "grid.linewidth": 0.3,
    "grid.alpha": 0.4,
})

TOPOS = ["qrsdr", "base", "ring", "random"]
NAMES = {"qrsdr": "QRS-DR (ours)", "base": r"Base-$(k\!+\!1)$",
         "ring": "Ring", "random": "Random $d$-reg"}
COLORS = {"qrsdr": "#c0392b", "base": "#2980b9",
          "ring": "#27ae60", "random": "#e67e22"}
LS = {"qrsdr": "-", "base": "--", "ring": "-.", "random": ":"}
MK = {"qrsdr": "o", "base": "s", "ring": "^", "random": "D"}
ZO = {"qrsdr": 10, "base": 5, "ring": 3, "random": 4}  # zorder


def load(path):
    with open(path) as f:
        return json.load(f)


def fig1(rdir, odir):
    """Consensus error vs wall-clock time on 3-tier heterogeneous network."""
    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    for topo in TOPOS:
        p = os.path.join(rdir, "exp1_hetero", f"{topo}.json")
        if not os.path.exists(p): continue
        d = load(p)["rounds"]
        t = [r["wall_clock"] for r in d]
        e = [max(r["consensus_error"], 1e-18) for r in d]
        me = max(1, len(t)//8)
        ax.semilogy(t, e, label=NAMES[topo], color=COLORS[topo],
                    ls=LS[topo], marker=MK[topo], markevery=me,
                    zorder=ZO[topo])

    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Consensus error")
    ax.set_title(r"(a) Heterogeneous bandwidth ($n\!=\!64$, $d\!=\!4$)",
                 fontsize=9)
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7",
              loc="upper right")
    ax.grid(True)
    ax.set_ylim(bottom=1e-16)
    fig.savefig(os.path.join(odir, "fig_hetero_bandwidth.pdf"))
    plt.close(fig)
    print("  fig_hetero_bandwidth.pdf")


def fig2(rdir, odir):
    """Link failure: convergence curves at different failure rates."""
    fail_rates = ["0.0", "0.01", "0.05", "0.10"]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2), sharey=True)

    for idx, fr in enumerate(fail_rates):
        ax = axes[idx]
        for topo in TOPOS:
            p = os.path.join(rdir, "exp2_failure", f"{topo}_fr{fr}.json")
            if not os.path.exists(p): continue
            d = load(p)["rounds"]
            rounds = [r["round"] for r in d]
            errors = [max(r["consensus_error"], 1e-18) for r in d]
            me = max(1, len(rounds)//6)
            ax.semilogy(rounds, errors, color=COLORS[topo], ls=LS[topo],
                        marker=MK[topo], markevery=me, zorder=ZO[topo],
                        label=NAMES[topo] if idx == 0 else None)

        ax.set_xlabel("Round")
        ax.set_title(f"$p_{{\\mathrm{{fail}}}}={fr}$", fontsize=8)  # fr is string like "0.10"
        ax.grid(True)
        ax.set_ylim(bottom=1e-16, top=2)

    axes[0].set_ylabel("Consensus error")
    axes[0].legend(frameon=True, fancybox=False, edgecolor="0.7",
                   fontsize=7, loc="upper right")
    fig.suptitle(r"(b) Link failure resilience ($n\!=\!64$, $d\!=\!4$)",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(odir, "fig_link_failure.pdf"))
    plt.close(fig)
    print("  fig_link_failure.pdf")


def fig2_summary(rdir, odir):
    """Link failure: final error vs fail rate (single panel summary)."""
    fail_rates = ["0.0", "0.01", "0.05", "0.10"]
    fail_pcts = [0, 1, 5, 10]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    target_round = 30

    for topo in TOPOS:
        errs = []
        for fr in fail_rates:
            p = os.path.join(rdir, "exp2_failure", f"{topo}_fr{fr}.json")
            if not os.path.exists(p):
                errs.append(np.nan)
                continue
            d = load(p)["rounds"]
            # Find round closest to target
            if target_round < len(d):
                errs.append(max(d[target_round]["consensus_error"], 1e-18))
            else:
                errs.append(max(d[-1]["consensus_error"], 1e-18))

        ax.semilogy(fail_pcts, errs,
                    label=NAMES[topo], color=COLORS[topo], ls=LS[topo],
                    marker=MK[topo], markersize=6, zorder=ZO[topo])

    ax.set_xlabel("Link failure rate (%)")
    ax.set_ylabel(f"Consensus error at round {target_round}")
    ax.set_title(r"(b) Link failure resilience ($n\!=\!64$, $d\!=\!4$)",
                 fontsize=9)
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    ax.grid(True)
    fig.savefig(os.path.join(odir, "fig_link_failure_summary.pdf"))
    plt.close(fig)
    print("  fig_link_failure_summary.pdf")


def fig3(rdir, odir):
    """Straggler: wall-clock time per round and convergence."""
    strag_fracs = [0.0, 0.05, 0.10, 0.20]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))

    for topo in TOPOS:
        times_per_round = []
        final_errors = []
        for sf in strag_fracs:
            p = os.path.join(rdir, "exp3_straggler", f"{topo}_sf{sf}.json")
            if not os.path.exists(p):
                times_per_round.append(np.nan)
                final_errors.append(np.nan)
                continue
            d = load(p)["rounds"]
            if len(d) >= 2:
                total_t = d[-1]["wall_clock"] - d[0]["wall_clock"]
                times_per_round.append(total_t / (len(d) - 1))
            else:
                times_per_round.append(np.nan)
            final_errors.append(max(d[-1]["consensus_error"], 1e-18))

        sf_pct = [s*100 for s in strag_fracs]
        ax1.plot(sf_pct, times_per_round, label=NAMES[topo],
                 color=COLORS[topo], ls=LS[topo], marker=MK[topo],
                 markersize=6, zorder=ZO[topo])
        ax2.semilogy(sf_pct, final_errors, label=NAMES[topo],
                     color=COLORS[topo], ls=LS[topo], marker=MK[topo],
                     markersize=6, zorder=ZO[topo])

    ax1.set_xlabel("Straggler fraction (%)")
    ax1.set_ylabel("Avg. wall-clock per round (s)")
    ax1.set_title("Latency impact", fontsize=9)
    ax1.legend(frameon=True, fancybox=False, edgecolor="0.7", fontsize=7)
    ax1.grid(True)

    ax2.set_xlabel("Straggler fraction (%)")
    ax2.set_ylabel("Final consensus error")
    ax2.set_title("Convergence impact", fontsize=9)
    ax2.legend(frameon=True, fancybox=False, edgecolor="0.7", fontsize=7)
    ax2.grid(True)

    fig.suptitle(r"(c) Straggler robustness ($n\!=\!64$, $d\!=\!4$, slowdown$=10\times$)",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(odir, "fig_straggler.pdf"))
    plt.close(fig)
    print("  fig_straggler.pdf")


def fig4(rdir, odir):
    """Fat-tree: consensus error vs rounds."""
    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    for topo in TOPOS:
        p = os.path.join(rdir, "exp4_fattree", f"{topo}.json")
        if not os.path.exists(p): continue
        d = load(p)["rounds"]
        rounds = [r["round"] for r in d]
        errors = [max(r["consensus_error"], 1e-18) for r in d]
        me = max(1, len(rounds)//8)
        ax.semilogy(rounds, errors, label=NAMES[topo], color=COLORS[topo],
                    ls=LS[topo], marker=MK[topo], markevery=me,
                    zorder=ZO[topo])

    ax.set_xlabel("Communication round")
    ax.set_ylabel("Consensus error")
    ax.set_title(r"(d) Fat-tree datacenter ($n\!=\!64$, $d\!=\!4$, $k\!=\!4$)",
                 fontsize=9)
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    ax.grid(True)
    ax.set_ylim(bottom=1e-16)
    fig.savefig(os.path.join(odir, "fig_fattree.pdf"))
    plt.close(fig)
    print("  fig_fattree.pdf")


def combined_figure(rdir, odir):
    """Combined 2x2 figure for the paper (saves space)."""
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))

    # (a) Heterogeneous bandwidth — consensus error vs wall-clock
    ax = axes[0, 0]
    for topo in TOPOS:
        p = os.path.join(rdir, "exp1_hetero", f"{topo}.json")
        if not os.path.exists(p): continue
        d = load(p)["rounds"]
        t = [r["wall_clock"] for r in d]
        e = [max(r["consensus_error"], 1e-18) for r in d]
        me = max(1, len(t)//6)
        ax.semilogy(t, e, label=NAMES[topo], color=COLORS[topo],
                    ls=LS[topo], marker=MK[topo], markevery=me,
                    zorder=ZO[topo])
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Consensus error")
    ax.set_title("(a) Heterogeneous bandwidth (3-tier)", fontsize=8)
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7", fontsize=6.5,
              loc="lower left")
    ax.grid(True)
    ax.set_ylim(bottom=1e-16)

    # (b) Link failure — convergence curves at 0% vs 10% failure
    ax = axes[0, 1]
    plot_configs = [
        ("base",  "0.0",  1.0, "-",  r"Base-$(k\!+\!1)$, 0\%"),
        ("base",  "0.10", 0.6, ":",  r"Base-$(k\!+\!1)$, 10\%"),
        ("qrsdr", "0.0",  1.0, "-",  "QRS-DR, 0\\%"),
        ("qrsdr", "0.10", 0.6, "--", "QRS-DR, 10\\%"),
    ]
    for topo, fr_str, alpha_val, lstyle, lbl in plot_configs:
        p = os.path.join(rdir, "exp2_failure", f"{topo}_fr{fr_str}.json")
        if not os.path.exists(p): continue
        d = load(p)["rounds"][:50]
        rounds = [r["round"] for r in d]
        errors = [max(r["consensus_error"], 1e-18) for r in d]
        me = max(1, len(rounds)//6)
        ax.semilogy(rounds, errors, color=COLORS[topo], ls=lstyle,
                    marker=MK[topo], markevery=me, zorder=ZO[topo],
                    alpha=alpha_val, label=lbl, markersize=3.5)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Consensus error")
    ax.set_title("(b) Link failure: 0\\% vs 10\\% failure rate", fontsize=8)
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7", fontsize=5.5,
              loc="upper right", ncol=1)
    ax.grid(True)
    ax.set_ylim(bottom=1e-16)

    # (c) Straggler: consensus error at fixed wall-clock budget
    ax = axes[1, 0]
    strag_fracs = [0.0, 0.05, 0.10, 0.20]
    # Fixed budget: pick wall-clock time that allows ~100 rounds at 0% straggler
    # (~120s for homo platform). Show error reached by that time.
    budget = 150.0  # seconds
    for topo in TOPOS:
        errs = []
        for sf in strag_fracs:
            p = os.path.join(rdir, "exp3_straggler", f"{topo}_sf{sf}.json")
            if not os.path.exists(p):
                errs.append(np.nan); continue
            d = load(p)["rounds"]
            # Find last round within budget
            best_err = d[0]["consensus_error"]
            for r in d:
                if r["wall_clock"] <= budget:
                    best_err = r["consensus_error"]
            errs.append(max(best_err, 1e-18))
        ax.semilogy([s*100 for s in strag_fracs], errs, label=NAMES[topo],
                    color=COLORS[topo], ls=LS[topo], marker=MK[topo],
                    markersize=5, zorder=ZO[topo])
    ax.set_xlabel("Straggler fraction (%)")
    ax.set_ylabel(f"Consensus error (at {budget:.0f}s budget)")
    ax.set_title(r"(c) Straggler impact (slowdown$=10\times$)", fontsize=8)
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7", fontsize=6.5)
    ax.grid(True)
    ax.set_xticks([0, 5, 10, 15, 20])

    # (d) Fat-tree — consensus error vs rounds
    ax = axes[1, 1]
    for topo in TOPOS:
        p = os.path.join(rdir, "exp4_fattree", f"{topo}.json")
        if not os.path.exists(p): continue
        d = load(p)["rounds"]
        rounds = [r["round"] for r in d]
        errors = [max(r["consensus_error"], 1e-18) for r in d]
        me = max(1, len(rounds)//6)
        ax.semilogy(rounds, errors, label=NAMES[topo], color=COLORS[topo],
                    ls=LS[topo], marker=MK[topo], markevery=me,
                    zorder=ZO[topo])
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Consensus error")
    ax.set_title(r"(d) Fat-tree datacenter ($k\!=\!4$)", fontsize=8)
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7", fontsize=6.5,
              loc="lower left")
    ax.grid(True)
    ax.set_ylim(bottom=1e-16)

    fig.suptitle("SimGrid systems-level evaluation ($n\\!=\\!64$, $d\\!=\\!4$)",
                 fontsize=10, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(odir, "fig_simgrid_combined.pdf"))
    plt.close(fig)
    print("  fig_simgrid_combined.pdf")


if __name__ == "__main__":
    rdir = sys.argv[1] if len(sys.argv) > 1 else "results"
    odir = os.path.join(rdir, "figures")
    os.makedirs(odir, exist_ok=True)

    print("Generating NeurIPS figures...")
    fig1(rdir, odir)
    fig2(rdir, odir)
    fig2_summary(rdir, odir)
    fig3(rdir, odir)
    fig4(rdir, odir)
    combined_figure(rdir, odir)
    print(f"\nAll figures in {odir}/")
