#!/usr/bin/env python3
"""
Evaluate DFL results and generate NeurIPS paper figures/tables.

Usage:
    python3 eval.py [results_dir]

Produces:
    - Table: Final accuracy by topology x alpha (LaTeX)
    - Figure: Convergence curves (accuracy vs round)
    - Figure: Effect of heterogeneity (accuracy vs alpha)
"""

import sys
import os
import json
import glob
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# NeurIPS formatting
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 10,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "grid.linewidth": 0.3,
    "grid.alpha": 0.4,
})

TOPO_ORDER = ["ring", "torus", "expander", "random", "ours", "base"]
TOPO_DISPLAY = {
    "ring": "Ring",
    "torus": "Torus",
    "expander": "Exp. Graph",
    "random": "Random d-reg",
    "ours": "Ours",
    "base": "Base-(k+1)",
}
TOPO_COLORS = {
    "ring": "#27ae60", "torus": "#8e44ad", "expander": "#2c3e50",
    "random": "#e67e22", "ours": "#c0392b", "base": "#2980b9",
}
TOPO_LS = {
    "ring": "-.", "torus": ":", "expander": "--",
    "random": ":", "ours": "-", "base": "--",
}
TOPO_MK = {
    "ring": "^", "torus": "v", "expander": "x",
    "random": "D", "ours": "o", "base": "s",
}


def load_results(results_dir):
    """Load all JSON result files."""
    results = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            data = json.load(f)
        args = data["args"]
        key = (args["topo"], args["n"], args.get("d", 0),
               args.get("alpha", None))
        results[key].append(data)
    return results


def final_accuracy_table(results, output_dir):
    """LaTeX table: final accuracy by topology x (n, alpha)."""
    table = defaultdict(dict)
    for (topo, n, d, alpha), runs in results.items():
        accs = [r["log"][-1]["accuracy"] for r in runs if r["log"]]
        if accs:
            table[topo][(n, alpha)] = (np.mean(accs) * 100, np.std(accs) * 100)

    configs = sorted(set(c for t in table.values() for c in t.keys()))

    out = os.path.join(output_dir, "table_accuracy.tex")
    with open(out, "w") as f:
        f.write("\\begin{tabular}{l" + "c" * len(configs) + "}\n\\toprule\n")
        header = "Topology"
        for n, alpha in configs:
            a = f"$\\alpha$={alpha}" if alpha else "IID"
            header += f" & $n$={n}, {a}"
        f.write(header + " \\\\\n\\midrule\n")
        for topo in TOPO_ORDER:
            if topo not in table: continue
            row = TOPO_DISPLAY[topo]
            for cfg in configs:
                if cfg in table[topo]:
                    m, s = table[topo][cfg]
                    best = max((table[t].get(cfg, (0,0))[0] for t in table), default=0)
                    bold = "\\textbf{" if abs(m - best) < 0.05 else ""
                    end = "}" if bold else ""
                    row += f" & {bold}{m:.1f}{end}$\\pm${s:.1f}"
                else:
                    row += " & ---"
            f.write(row + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"  {out}")

    # Console output
    print(f"\n  {'Topology':15s}", end="")
    for n, alpha in configs:
        print(f" n={n},a={alpha}", end="")
    print()
    for topo in TOPO_ORDER:
        if topo not in table: continue
        print(f"  {topo:15s}", end="")
        for cfg in configs:
            if cfg in table[topo]:
                m, s = table[topo][cfg]
                print(f" {m:5.1f}±{s:.1f}", end="")
            else:
                print(f" {'---':>9s}", end="")
        print()


def convergence_figure(results, output_dir, n=32, alpha=0.1):
    """Accuracy vs round."""
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    for topo in TOPO_ORDER:
        runs = [r for (t, nn, d, a), rl in results.items()
                for r in rl if t == topo and nn == n and a == alpha]
        if not runs: continue
        all_r = defaultdict(list)
        for run in runs:
            for e in run["log"]:
                all_r[e["round"]].append(e["accuracy"] * 100)
        rounds = sorted(all_r.keys())
        means = [np.mean(all_r[r]) for r in rounds]
        stds = [np.std(all_r[r]) for r in rounds]
        me = max(1, len(rounds) // 8)
        ax.plot(rounds, means, label=TOPO_DISPLAY[topo],
                color=TOPO_COLORS[topo], ls=TOPO_LS[topo],
                marker=TOPO_MK[topo], markevery=me)
        ax.fill_between(rounds, np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.1, color=TOPO_COLORS[topo])
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"CIFAR-100, $n$={n}, $\\alpha$={alpha}", fontsize=9)
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    ax.grid(True)
    out = os.path.join(output_dir, f"fig_conv_n{n}_a{alpha}.pdf")
    fig.savefig(out); plt.close(fig)
    print(f"  {out}")


def heterogeneity_figure(results, output_dir, n=32):
    """Final accuracy vs alpha."""
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    alphas = sorted(set(a for (t, nn, d, a), _ in results.items()
                        if nn == n and a is not None))
    for topo in TOPO_ORDER:
        accs = []
        for alpha in alphas:
            runs = [r for (t, nn, d, a), rl in results.items()
                    for r in rl if t == topo and nn == n and a == alpha]
            if runs:
                accs.append(np.mean([r["log"][-1]["accuracy"] * 100
                                     for r in runs if r["log"]]))
            else:
                accs.append(np.nan)
        if any(not np.isnan(a) for a in accs):
            ax.plot(alphas, accs, label=TOPO_DISPLAY[topo],
                    color=TOPO_COLORS[topo], ls=TOPO_LS[topo],
                    marker=TOPO_MK[topo], markersize=6)
    ax.set_xlabel("Dirichlet $\\alpha$")
    ax.set_ylabel("Final test accuracy (%)")
    ax.set_title(f"CIFAR-100, $n$={n}", fontsize=9)
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    ax.grid(True)
    ax.set_xscale("log")
    out = os.path.join(output_dir, f"fig_hetero_n{n}.pdf")
    fig.savefig(out); plt.close(fig)
    print(f"  {out}")


def main():
    rdir = sys.argv[1] if len(sys.argv) > 1 else "results"
    odir = os.path.join(rdir, "figures")
    os.makedirs(odir, exist_ok=True)
    results = load_results(rdir)
    total = sum(len(v) for v in results.values())
    print(f"Loaded {total} runs across {len(results)} configs")
    if not results:
        print("No results found."); return
    final_accuracy_table(results, odir)
    for n in [32, 64]:
        for alpha in [0.1, 0.3, 1.0]:
            convergence_figure(results, odir, n=n, alpha=alpha)
        heterogeneity_figure(results, odir, n=n)
    print(f"\nAll outputs in {odir}/")


if __name__ == "__main__":
    main()
