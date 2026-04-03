"""
Post-hoc evaluation: load DFL result JSONs, generate plots and tables.

Produces:
  - Figure 1: Accuracy vs communication rounds (per n, fixed m/n and alpha)
  - Figure 2: Rounds to threshold vs n (scaling plot)
  - Figure 3: lambda_2 vs convergence rate (scatter)
  - Table 1: Final accuracy summary

Usage:
    python eval.py --results-dir results/
    python eval.py --results-dir results/ --threshold 0.85
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TOPO_COLORS = {
    "ring":   "#888888",
    "sw_r25": "#2ca02c",
    "sw_r50": "#1f77b4",
    "sw_r75": "#9467bd",
    "er":     "#d62728",
    "fv":     "#ff7f0e",
    "rd":     "#8c564b",
    "refine": "#e377c2",
}

TOPO_LABELS = {
    "ring":   "Ring",
    "sw_r25": "SW (0.25)",
    "sw_r50": "SW (0.50)",
    "sw_r75": "SW (0.75)",
    "er":     "ER",
    "fv":     "FV",
    "rd":     "RD (d-reg)",
    "refine": "REFINE (ours)",
}


def load_results(results_dir):
    """Load all result JSONs from a directory."""
    files = glob.glob(os.path.join(results_dir, "*.json"))
    results = []
    for f in sorted(files):
        with open(f) as fp:
            results.append(json.load(fp))
    return results


def group_results(results):
    """Group results by (topo, n, m, alpha)."""
    groups = {}
    for r in results:
        a = r["args"]
        alpha_str = f"{a['alpha']}" if a["alpha"] is not None else "iid"
        key = (a["topo"], a["n"], a["m"], alpha_str)
        groups.setdefault(key, []).append(r)
    return groups


def extract_curves(group):
    """Extract (rounds, mean_acc, std_acc) from a group of runs."""
    all_rounds = None
    all_accs = []

    for r in group:
        rounds = [e["round"] for e in r["log"]]
        accs = [e["accuracy"] for e in r["log"]]
        if all_rounds is None:
            all_rounds = rounds
        all_accs.append(accs)

    all_accs = np.array(all_accs)
    mean_acc = all_accs.mean(axis=0)
    std_acc = all_accs.std(axis=0)
    return np.array(all_rounds), mean_acc, std_acc


def rounds_to_threshold(group, threshold):
    """Find first round where accuracy >= threshold, per seed. Return mean."""
    vals = []
    for r in group:
        found = None
        for e in r["log"]:
            if e["accuracy"] >= threshold:
                found = e["round"]
                break
        if found is not None:
            vals.append(found)
    if not vals:
        return None
    return np.mean(vals), np.std(vals)


def plot_accuracy_vs_rounds(groups, n_values, m_div_n, alpha_str, output_dir):
    """Figure 1: accuracy vs rounds, one subplot per n."""
    fig, axes = plt.subplots(1, len(n_values), figsize=(5 * len(n_values), 4),
                             sharey=True)
    if len(n_values) == 1:
        axes = [axes]

    for ax, n in zip(axes, n_values):
        m = n * m_div_n
        for topo in TOPO_COLORS:
            key = (topo, n, m, alpha_str)
            if key not in groups:
                continue
            rounds, mean_acc, std_acc = extract_curves(groups[key])
            color = TOPO_COLORS[topo]
            label = TOPO_LABELS.get(topo, topo)
            ax.plot(rounds, mean_acc, color=color, label=label, linewidth=1.5)
            ax.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc,
                            alpha=0.15, color=color)
        ax.set_title(f"n={n}, m={m}")
        ax.set_xlabel("Communication rounds")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Test accuracy")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"DFL Convergence (m/n={m_div_n}, alpha={alpha_str})",
                 fontsize=13)
    fig.tight_layout()

    path = os.path.join(output_dir, f"fig1_acc_rounds_d{m_div_n}_{alpha_str}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_scaling(groups, n_values, m_div_n, alpha_str, threshold, output_dir):
    """Figure 2: rounds to threshold vs n."""
    fig, ax = plt.subplots(figsize=(6, 4))

    for topo in TOPO_COLORS:
        xs, ys, errs = [], [], []
        for n in n_values:
            m = n * m_div_n
            key = (topo, n, m, alpha_str)
            if key not in groups:
                continue
            result = rounds_to_threshold(groups[key], threshold)
            if result is not None:
                xs.append(n)
                ys.append(result[0])
                errs.append(result[1])

        if xs:
            color = TOPO_COLORS[topo]
            label = TOPO_LABELS.get(topo, topo)
            ax.errorbar(xs, ys, yerr=errs, color=color, label=label,
                        marker="o", linewidth=1.5, capsize=3)

    ax.set_xlabel("Number of nodes (n)")
    ax.set_ylabel(f"Rounds to {threshold*100:.0f}% accuracy")
    ax.set_xscale("log", base=2)
    ax.set_xticks(n_values)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Scaling (m/n={m_div_n}, alpha={alpha_str})")
    fig.tight_layout()

    path = os.path.join(output_dir, f"fig2_scaling_d{m_div_n}_{alpha_str}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_lambda2_vs_convergence(groups, threshold, output_dir):
    """Figure 3: scatter of lambda_2 vs rounds to threshold."""
    fig, ax = plt.subplots(figsize=(6, 5))

    for key, group in groups.items():
        topo, n, m, alpha_str = key
        result = rounds_to_threshold(group, threshold)
        if result is None:
            continue

        l2 = group[0]["topology"]["lambda2"]
        rounds_mean = result[0]
        color = TOPO_COLORS.get(topo, "#333333")
        marker = "o" if n <= 64 else ("s" if n <= 128 else "^")
        ax.scatter(l2, rounds_mean, color=color, marker=marker, s=40,
                   alpha=0.7, edgecolors="white", linewidths=0.5)

    # Theoretical curve: rounds ~ C / lambda_2
    l2_range = np.linspace(0.01, ax.get_xlim()[1], 100)
    # Fit C from the data
    all_l2, all_r = [], []
    for key, group in groups.items():
        result = rounds_to_threshold(group, threshold)
        if result is None:
            continue
        all_l2.append(group[0]["topology"]["lambda2"])
        all_r.append(result[0])
    if all_l2:
        C = np.median(np.array(all_l2) * np.array(all_r))
        ax.plot(l2_range, C / l2_range, "k--", alpha=0.4, linewidth=1,
                label=f"~1/lambda_2")

    ax.set_xlabel("Algebraic connectivity (lambda_2)")
    ax.set_ylabel(f"Rounds to {threshold*100:.0f}% accuracy")
    ax.set_title("lambda_2 vs convergence rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "fig3_lambda2_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def print_table(groups, n_values, m_div_n_values, alpha_str):
    """Table 1: final accuracy for each (topo, n, m/n)."""
    topos = list(TOPO_COLORS.keys())

    header = f"{'Topology':<12}"
    for n in n_values:
        for d in m_div_n_values:
            header += f" n={n},d={d:>2}"
    print(f"\nTable 1: Final accuracy (alpha={alpha_str})")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for topo in topos:
        row = f"{TOPO_LABELS.get(topo, topo):<12}"
        for n in n_values:
            for d in m_div_n_values:
                m = n * d
                key = (topo, n, m, alpha_str)
                if key in groups:
                    group = groups[key]
                    accs = [r["log"][-1]["accuracy"] for r in group]
                    mean = np.mean(accs)
                    std = np.std(accs)
                    row += f" {mean:.3f}+{std:.3f}"
                else:
                    row += "      -     "
        print(row)


def main():
    parser = argparse.ArgumentParser(description="DFL Evaluation")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="figures")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--n-values", type=int, nargs="+",
                        default=[32, 64, 128, 256])
    parser.add_argument("--m-div-n", type=int, nargs="+",
                        default=[2, 4, 8, 16])
    parser.add_argument("--alpha", type=str, default="0.1")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}/")
        return

    print(f"Loaded {len(results)} result files")
    groups = group_results(results)

    # Figure 1: accuracy vs rounds (primary m/n)
    for d in args.m_div_n:
        plot_accuracy_vs_rounds(groups, args.n_values, d, args.alpha,
                                args.output_dir)

    # Figure 2: scaling plot
    for d in args.m_div_n:
        plot_scaling(groups, args.n_values, d, args.alpha, args.threshold,
                     args.output_dir)

    # Figure 3: lambda2 scatter
    plot_lambda2_vs_convergence(groups, args.threshold, args.output_dir)

    # Table 1
    print_table(groups, args.n_values, args.m_div_n, args.alpha)


if __name__ == "__main__":
    main()
