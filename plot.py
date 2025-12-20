#!/usr/bin/env python3
"""
Plot benchmark results from individual CSV files.

Usage:
    python plot.py [N1 N2 N3 N4]

Reads from data/{ER,FV,OURS,SW_*}_{N}.csv and generates spectral_benchmark_grid.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

def power_formatter(x, pos):
    """Format tick labels with powers of 10."""
    if x == 0:
        return '0'
    exp = int(np.floor(np.log10(abs(x))))
    coef = x / 10**exp
    if coef == 1:
        return rf'$10^{exp}$'
    elif coef == int(coef):
        return rf'${int(coef)}\times10^{exp}$'
    else:
        return rf'${coef:.1f}\times10^{exp}$'

DATA_DIR = "data"

# SW rho configurations with colors (excluding 0.0 ring and 1.0 pure random)
SW_CONFIGS = [
    ("r25",  0.25, '#9467bd'),   # Purple
    ("r50",  0.50, '#8c564b'),   # Brown
    ("r75",  0.75, '#e377c2'),   # Pink
]


def load_algorithm_data(algo, n):
    """Load data for a single algorithm and N value."""
    path = os.path.join(DATA_DIR, f"{algo}_{n}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_sw_data(n):
    """Load all SW rho variants separately."""
    sw_data = {}
    for name, rho, color in SW_CONFIGS:
        path = os.path.join(DATA_DIR, f"SW_{name}_{n}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            sw_data[name] = {
                'm_values': df['m'].values,
                'scores': df['score'].values,
                'rho': rho,
                'color': color
            }
    return sw_data if sw_data else None


def interpolate(target_m, source_m, source_scores):
    """Interpolate scores to match target m values."""
    return np.interp(target_m, source_m, source_scores)


def main():
    # Parse N values from command line or use defaults
    if len(sys.argv) > 1:
        n_values = [int(x) for x in sys.argv[1:]]
    else:
        # Default: look for available data
        n_values = []
        if os.path.exists(DATA_DIR):
            for f in os.listdir(DATA_DIR):
                # Match OURS_{N}.csv but not OURS_TEST_{N}.csv
                if f.startswith("OURS_") and f.endswith(".csv") and not f.startswith("OURS_TEST_"):
                    n = int(f.replace("OURS_", "").replace(".csv", ""))
                    if n not in n_values:
                        n_values.append(n)
        n_values.sort()

    if not n_values:
        print("Error: No data found in data/ directory.")
        print("Run the benchmark first: make run")
        sys.exit(1)

    # Limit to 4 plots
    n_values = n_values[:4]
    print(f"Plotting for N values: {n_values}")

    # Load data for each N
    results = {}
    for n in n_values:
        er_df = load_algorithm_data("ER", n)
        fv_df = load_algorithm_data("FV", n)
        ours_df = load_algorithm_data("OURS", n)
        sw_data = load_sw_data(n)

        if ours_df is None:
            print(f"Warning: Missing OURS_{n}.csv, skipping N={n}")
            continue

        # Use OURS m_values as reference
        m_values = ours_df['m'].values
        ours_scores = ours_df['score'].values

        # Interpolate ER and FV to match m_values
        if er_df is not None:
            er_scores = interpolate(m_values, er_df['m'].values, er_df['score'].values)
        else:
            er_scores = np.zeros_like(ours_scores)
            print(f"Warning: Missing ER_{n}.csv")

        if fv_df is not None:
            fv_scores = interpolate(m_values, fv_df['m'].values, fv_df['score'].values)
        else:
            fv_scores = np.zeros_like(ours_scores)
            print(f"Warning: Missing FV_{n}.csv")

        # Interpolate SW data
        sw_interpolated = {}
        if sw_data:
            for name, data in sw_data.items():
                sw_interpolated[name] = {
                    'scores': interpolate(m_values, data['m_values'], data['scores']),
                    'rho': data['rho'],
                    'color': data['color']
                }

        results[n] = {
            'm_values': m_values,
            'er': er_scores,
            'fv': fv_scores,
            'ours': ours_scores,
            'sw': sw_interpolated
        }

        # Print summary
        max_m = n * (n - 1) // 2
        print(f"  N={n}: m range [{m_values[0]}..{m_values[-1]}] (max_m={max_m})")
        valid_idx = er_scores > 0
        if np.any(valid_idx):
            pct = np.mean(100 * ours_scores[valid_idx] / er_scores[valid_idx])
            print(f"  N={n}: OURS Avg {pct:.1f}% of ER")

    if not results:
        print("Error: No valid data to plot.")
        sys.exit(1)

    # Create figure - fixed 1x4 layout
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=200)
    num_plots = len(results)

    # Colors for main algorithms
    C_ER = '#1f77b4'    # Blue
    C_FV = '#ff7f0e'    # Orange
    C_OURS = '#d62728'  # Red

    # Hide unused axes
    for idx in range(num_plots, 4):
        axes[idx].set_visible(False)

    for idx, n in enumerate(list(results.keys())[:4]):
        ax = axes[idx]
        data = results[n]
        m_vals = data['m_values']

        # Plot each SW rho with different color
        for name, sw_info in data['sw'].items():
            ax.plot(m_vals, sw_info['scores'], color=sw_info['color'],
                    linewidth=1, linestyle='-', alpha=0.7, zorder=1)

        # ER and FV baselines
        ax.plot(m_vals, data['er'], color=C_ER, linewidth=1.5, linestyle='-', zorder=2)
        ax.plot(m_vals, data['fv'], color=C_FV, linewidth=1.5, linestyle='--', zorder=2)

        # Ours algorithm
        ax.plot(m_vals, data['ours'], color=C_OURS, linewidth=2, zorder=3)

        # Styling
        ax.set_title(f'N = {n}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Edges (M)', fontsize=14)
        if idx == 0:
            ax.set_ylabel('Algebraic Connectivity', fontsize=14)
        ax.grid(True, linestyle='-', alpha=0.25, color='#cccccc')
        ax.set_facecolor('#fafafa')
        for spine in ax.spines.values():
            spine.set_color('#cccccc')
        # Use power notation for x-axis tick labels
        ax.xaxis.set_major_formatter(FuncFormatter(power_formatter))

    # Single-row legend with complexity annotations
    legend_handles = [
        Line2D([0], [0], color=C_ER, linewidth=2, linestyle='-', label=r'ER $O(MN^3)$'),
        Line2D([0], [0], color=C_FV, linewidth=2, linestyle='--', label=r'FV $O(MN^3)$'),
        Line2D([0], [0], color=C_OURS, linewidth=2.5, linestyle='-', label=r'Ours $O(N^2)$'),
    ]
    # Add SW variants - each shows rho and complexity
    for name, rho, color in SW_CONFIGS:
        label = rf'SW $\rho$={rho} $O(N^2)$'
        legend_handles.append(
            Line2D([0], [0], color=color, linewidth=1.5, linestyle='-', alpha=0.7, label=label)
        )

    fig.legend(handles=legend_handles, loc='upper center', ncol=6, fontsize=12,
               frameon=True, framealpha=0.95, edgecolor='#cccccc',
               bbox_to_anchor=(0.5, 1.02), columnspacing=1.0, handletextpad=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_file = 'figure.png'
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()

    print(f"\nPlot saved: {output_file}")


if __name__ == "__main__":
    main()
