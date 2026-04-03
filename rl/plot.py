#!/usr/bin/env python3
"""
Plot RL eval results vs DB baselines from eval CSV files.

Usage:
    python plot.py logs/v7r_swap05_eval_n16.csv logs/v7r_swap05_eval_n24.csv ...
    python plot.py logs/v7r_swap05_eval_n*.csv
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


def load_eval_csv(path):
    """Load eval CSV, converting 'None' strings to NaN."""
    df = pd.read_csv(path, na_values=['None', 'none', ''])
    return df


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot.py <eval_csv1> [eval_csv2] ...")
        sys.exit(1)

    csv_paths = sys.argv[1:]
    dfs = {}
    for p in csv_paths:
        df = load_eval_csv(p)
        n = int(df['n'].iloc[0])
        dfs[n] = df

    n_values = sorted(dfs.keys())
    num_plots = len(n_values)

    fig, axes = plt.subplots(1, num_plots, figsize=(5.5 * num_plots, 5), dpi=150)
    if num_plots == 1:
        axes = [axes]

    C_RL = '#d62728'     # Red
    C_FV = '#ff7f0e'     # Orange
    C_ER = '#1f77b4'     # Blue
    C_SW25 = '#9467bd'   # Purple
    C_SW50 = '#8c564b'   # Brown
    C_SW75 = '#e377c2'   # Pink
    C_BEST = '#2ca02c'   # Green

    for idx, n in enumerate(n_values):
        ax = axes[idx]
        df = dfs[n]

        m = df['m'].values
        density = df['density'].values
        rl = df['rl'].values
        rl_std = df['rl_std'].values

        # DB baselines
        fv = df['fv'].values.astype(float)
        er = df['er'].values.astype(float)
        sw25 = df['sw025'].values.astype(float)
        sw50 = df['sw050'].values.astype(float)
        sw75 = df['sw075'].values.astype(float)

        # Best DB baseline per (n,m)
        db_best = np.nanmax(np.column_stack([fv, er, sw25, sw50, sw75]), axis=1)

        # Plot baselines
        ax.plot(density, db_best, color=C_BEST, linewidth=2, linestyle='-',
                alpha=0.8, zorder=2, label='DB Best')
        ax.plot(density, fv, color=C_FV, linewidth=1, linestyle='--',
                alpha=0.6, zorder=1)
        ax.plot(density, er, color=C_ER, linewidth=1, linestyle='--',
                alpha=0.6, zorder=1)
        ax.plot(density, sw25, color=C_SW25, linewidth=0.8, linestyle=':',
                alpha=0.5, zorder=1)
        ax.plot(density, sw50, color=C_SW50, linewidth=0.8, linestyle=':',
                alpha=0.5, zorder=1)
        ax.plot(density, sw75, color=C_SW75, linewidth=0.8, linestyle=':',
                alpha=0.5, zorder=1)

        # RL with std band
        ax.plot(density, rl, color=C_RL, linewidth=2.5, zorder=3)
        ax.fill_between(density, rl - rl_std, rl + rl_std,
                        color=C_RL, alpha=0.15, zorder=2)

        # Mark wins and ties
        wins = df['win'].values == 1
        ties = df['tie'].values == 1
        if wins.any():
            ax.scatter(density[wins], rl[wins], color=C_RL, s=40,
                       marker='*', zorder=5, label=f'Win ({wins.sum()})')
        if ties.any():
            ax.scatter(density[ties], rl[ties], color=C_RL, s=20,
                       marker='o', zorder=4, edgecolors='black',
                       linewidths=0.5, alpha=0.7)

        # Ratio text
        valid = db_best > 0
        if valid.any():
            ratio = np.mean(rl[valid] / db_best[valid]) * 100
            ax.text(0.97, 0.03, f'Avg ratio: {ratio:.1f}%',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#cccccc', alpha=0.9))

        # Highlight gap regions (where RL < 80% of DB best)
        gap = np.where(valid, rl / db_best, 1.0)
        weak = gap < 0.80
        if weak.any():
            ax.fill_between(density, 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                            where=weak, color='red', alpha=0.04, zorder=0)

        ax.set_title(f'n = {n}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Density (m / max_m)', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Algebraic Connectivity (λ₂)', fontsize=12)
        ax.grid(True, linestyle='-', alpha=0.2, color='#cccccc')
        ax.set_facecolor('#fafafa')
        for spine in ax.spines.values():
            spine.set_color('#cccccc')

    # Legend
    legend_handles = [
        Line2D([0], [0], color=C_RL, linewidth=2.5, label='RL'),
        Line2D([0], [0], color=C_BEST, linewidth=2, label='DB Best'),
        Line2D([0], [0], color=C_FV, linewidth=1, linestyle='--',
               alpha=0.6, label='FV'),
        Line2D([0], [0], color=C_ER, linewidth=1, linestyle='--',
               alpha=0.6, label='ER'),
        Line2D([0], [0], color=C_SW25, linewidth=0.8, linestyle=':',
               alpha=0.5, label='SW ρ=0.25'),
        Line2D([0], [0], color=C_SW50, linewidth=0.8, linestyle=':',
               alpha=0.5, label='SW ρ=0.50'),
        Line2D([0], [0], color=C_SW75, linewidth=0.8, linestyle=':',
               alpha=0.5, label='SW ρ=0.75'),
    ]

    fig.legend(handles=legend_handles, loc='upper center', ncol=7, fontsize=10,
               frameon=True, framealpha=0.95, edgecolor='#cccccc',
               bbox_to_anchor=(0.5, 1.02), columnspacing=0.8, handletextpad=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = 'eval_plot.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
