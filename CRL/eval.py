#!/usr/bin/env python3
"""
Evaluate CRL/REFINE checkpoint against baselines.

Standalone — uses existing rl/eval_refine.py logic.
Checkpoints are compatible: same policy format.

Usage:
    python eval.py best.pt --n 8,10,12,14,16 --trials 5
"""

import sys
from pathlib import Path

# Add rl/ to path so we can reuse eval_refine
sys.path.insert(0, str(Path(__file__).parent.parent / "rl"))

from eval_refine import eval_refine

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate CRL/REFINE checkpoint")
    parser.add_argument("checkpoint", help="Path to checkpoint file (.pt)")
    parser.add_argument("--n", type=str, required=True,
                        help="Comma-separated n values (e.g. 8,10,12,14,16)")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--k-steps", type=int, default=20)
    parser.add_argument("--swap-frac", type=float, default=0.1)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--init", type=str, default="ring", choices=["ring", "fv"])
    args = parser.parse_args()

    n_values = [int(x.strip()) for x in args.n.split(',')]
    eval_refine(args.checkpoint, n_values=n_values,
                num_trials=args.trials, num_workers=args.workers,
                k_steps=args.k_steps, swap_frac=args.swap_frac,
                stochastic=args.stochastic, init_method=args.init)
