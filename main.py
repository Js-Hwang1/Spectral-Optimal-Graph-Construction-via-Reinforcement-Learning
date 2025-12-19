"""
Two-Phase Graph Construction for Maximum Algebraic Connectivity

Usage:
    python main.py build --n 50 --m 200                    # Build graph (heuristic)
    python main.py build --n 50 --m 200 --model best.pt    # Build graph (trained model)
    python main.py compare --n 50 --m 200                  # Compare against baselines
    python main.py train                                    # Train RL agent
    python main.py eval --model checkpoints/best_model.pt  # Evaluate model
"""

import argparse
import numpy as np
import scipy.linalg
import time
from pathlib import Path

from phase1 import AdaptiveGraphBuilder, build_small_world_graph, get_matrix_lambda2
from phase2 import ResidualEdgeFiller, build_graph_with_rl


# =============================================================================
# UTILITIES
# =============================================================================
def compute_lambda2(adj: np.ndarray) -> float:
    """Compute algebraic connectivity (second smallest eigenvalue of Laplacian)."""
    degrees = np.sum(adj, axis=0)
    L = np.diag(degrees) - adj
    eigvals = scipy.linalg.eigh(L, eigvals_only=True)
    eigvals.sort()
    return eigvals[1] if len(eigvals) > 1 else 0.0


def print_graph_stats(adj: np.ndarray, name: str = "Graph"):
    """Print statistics about a graph."""
    n = adj.shape[0]
    m = int(adj.sum() / 2)
    lambda2 = compute_lambda2(adj)
    degrees = adj.sum(axis=1)

    print(f"\n{name} Statistics:")
    print(f"  Nodes: {n}")
    print(f"  Edges: {m}")
    print(f"  λ₂ (algebraic connectivity): {lambda2:.6f}")
    print(f"  Degree - min: {degrees.min():.0f}, max: {degrees.max():.0f}, "
          f"mean: {degrees.mean():.2f}, std: {degrees.std():.2f}")


# =============================================================================
# COMMANDS
# =============================================================================
def cmd_build(args):
    """Build a graph using the two-phase pipeline."""
    print(f"Building graph with N={args.n}, M={args.m}")
    print("=" * 60)

    start_time = time.time()

    if args.model and Path(args.model).exists():
        # Use trained model
        print(f"Using trained model: {args.model}")
        final_adj = build_graph_with_rl(
            args.n, args.m,
            model_path=args.model,
            skeleton_budget_ratio=args.skeleton_ratio
        )
    else:
        # Use heuristic fallback
        if args.model:
            print(f"Warning: Model {args.model} not found, using heuristic")

        # Phase 1
        m_skeleton = int(args.m * args.skeleton_ratio)
        builder = AdaptiveGraphBuilder(args.n)
        builder.build(m_skeleton)
        print(f"Phase 1: Built skeleton with {len(builder.edges)} edges")
        print_graph_stats(builder.adj.astype(float), "Skeleton")

        # Phase 2
        m_remaining = args.m - len(builder.edges)
        if m_remaining > 0:
            print(f"\nPhase 2: Adding {m_remaining} edges with heuristic...")
            filler = ResidualEdgeFiller()
            final_adj = filler.fill_edges(builder.adj.astype(np.float32), m_remaining)
        else:
            final_adj = builder.adj.astype(float)

    elapsed = time.time() - start_time

    print_graph_stats(final_adj, "Final Graph")
    print(f"\nTime elapsed: {elapsed:.3f}s")

    # Save if requested
    if args.output:
        np.save(args.output, final_adj)
        print(f"Saved to: {args.output}")

    return final_adj


def cmd_compare(args):
    """Compare our method against baselines."""
    print(f"Comparing methods for N={args.n}, M={args.m}")
    print("=" * 60)

    results = {}

    # 1. Phase 1 Only
    builder = AdaptiveGraphBuilder(args.n)
    builder.build(args.m)
    results['Phase1 Only'] = builder.get_lambda2()

    # 2. Two-Phase (Heuristic)
    m_skeleton = int(args.m * args.skeleton_ratio)
    builder = AdaptiveGraphBuilder(args.n)
    builder.build(m_skeleton)
    m_remaining = args.m - len(builder.edges)
    if m_remaining > 0:
        filler = ResidualEdgeFiller()
        final_adj = filler.fill_edges(builder.adj.astype(np.float32), m_remaining)
    else:
        final_adj = builder.adj.astype(float)
    results['Two-Phase (Heuristic)'] = compute_lambda2(final_adj)

    # 3. Two-Phase (Model) if available
    if args.model and Path(args.model).exists():
        final_adj = build_graph_with_rl(args.n, args.m, model_path=args.model,
                                         skeleton_budget_ratio=args.skeleton_ratio)
        results['Two-Phase (Model)'] = compute_lambda2(final_adj)

    # 4. Small World baselines
    sw_probs = [0.0, 0.25, 0.5, 0.75, 1.0]
    best_sw = -1
    best_sw_p = 0
    for p in sw_probs:
        adj = build_small_world_graph(args.n, args.m, p=p)
        l2 = get_matrix_lambda2(adj)
        if l2 > best_sw:
            best_sw = l2
            best_sw_p = p
    results[f'Small World (p={best_sw_p})'] = best_sw

    # Print results
    print("\nResults (λ₂):")
    print("-" * 40)
    sorted_results = sorted(results.items(), key=lambda x: -x[1])
    for name, l2 in sorted_results:
        marker = " <-- BEST" if l2 == sorted_results[0][1] else ""
        print(f"  {name:<25} {l2:.6f}{marker}")

    # Winner
    winner = sorted_results[0][0]
    print(f"\nBest method: {winner} (λ₂ = {sorted_results[0][1]:.6f})")


def cmd_train(args):
    """Train the RL agent."""
    from phase2_train import train, argparse as ap

    # Create args namespace for training
    train_args = ap.Namespace(
        num_episodes=args.num_episodes,
        min_n=args.min_n,
        max_n=args.max_n,
        min_density=args.min_density,
        max_density=args.max_density,
        skeleton_ratio=args.skeleton_ratio,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        train_freq=args.train_freq,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        seed=args.seed,
        cpu=args.cpu
    )
    train(train_args)


def cmd_eval(args):
    """Evaluate a trained model."""
    from phase2_train import evaluate, argparse as ap

    eval_args = ap.Namespace(
        model_path=args.model,
        cpu=args.cpu
    )
    evaluate(eval_args)


def cmd_benchmark(args):
    """Run comprehensive benchmark across multiple graph sizes."""
    print("Running Comprehensive Benchmark")
    print("=" * 80)

    test_cases = [
        # (n, m) - various regimes
        (15, 30), (15, 50), (15, 80),
        (20, 40), (20, 80), (20, 120),
        (30, 60), (30, 120), (30, 200),
        (40, 100), (40, 200), (40, 350),
        (50, 150), (50, 300), (50, 500),
    ]

    print(f"{'N':>4} {'M':>5} {'Phase1':>10} {'TwoPhase':>10} {'SmallWorld':>10} {'Winner':>12}")
    print("-" * 60)

    wins = {'Phase1': 0, 'TwoPhase': 0, 'SmallWorld': 0}

    for n, m in test_cases:
        # Phase 1 only
        builder = AdaptiveGraphBuilder(n)
        builder.build(m)
        l2_phase1 = builder.get_lambda2()

        # Two-phase heuristic
        m_skeleton = int(m * args.skeleton_ratio)
        builder = AdaptiveGraphBuilder(n)
        builder.build(m_skeleton)
        m_remaining = m - len(builder.edges)
        if m_remaining > 0:
            filler = ResidualEdgeFiller()
            final_adj = filler.fill_edges(builder.adj.astype(np.float32), m_remaining)
        else:
            final_adj = builder.adj.astype(float)
        l2_twophase = compute_lambda2(final_adj)

        # Best small world
        best_sw = max(get_matrix_lambda2(build_small_world_graph(n, m, p=p))
                      for p in [0.0, 0.25, 0.5, 0.75, 1.0])

        # Determine winner
        scores = {'Phase1': l2_phase1, 'TwoPhase': l2_twophase, 'SmallWorld': best_sw}
        winner = max(scores, key=scores.get)
        wins[winner] += 1

        print(f"{n:>4} {m:>5} {l2_phase1:>10.4f} {l2_twophase:>10.4f} "
              f"{best_sw:>10.4f} {winner:>12}")

    print("-" * 60)
    print(f"Wins: Phase1={wins['Phase1']}, TwoPhase={wins['TwoPhase']}, "
          f"SmallWorld={wins['SmallWorld']}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Two-Phase Graph Construction for Maximum λ₂',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Build command
    build_parser = subparsers.add_parser('build', help='Build a graph')
    build_parser.add_argument('--n', type=int, required=True, help='Number of nodes')
    build_parser.add_argument('--m', type=int, required=True, help='Number of edges')
    build_parser.add_argument('--model', type=str, help='Path to trained model')
    build_parser.add_argument('--skeleton-ratio', type=float, default=0.7,
                              help='Fraction of edges for Phase 1')
    build_parser.add_argument('--output', '-o', type=str, help='Output file (.npy)')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare against baselines')
    compare_parser.add_argument('--n', type=int, required=True)
    compare_parser.add_argument('--m', type=int, required=True)
    compare_parser.add_argument('--model', type=str, help='Path to trained model')
    compare_parser.add_argument('--skeleton-ratio', type=float, default=0.7)

    # Train command
    train_parser = subparsers.add_parser('train', help='Train RL agent')
    train_parser.add_argument('--num-episodes', type=int, default=10000)
    train_parser.add_argument('--min-n', type=int, default=15)
    train_parser.add_argument('--max-n', type=int, default=50)
    train_parser.add_argument('--min-density', type=float, default=0.1)
    train_parser.add_argument('--max-density', type=float, default=0.4)
    train_parser.add_argument('--skeleton-ratio', type=float, default=0.7)
    train_parser.add_argument('--hidden-dim', type=int, default=64)
    train_parser.add_argument('--lr', type=float, default=3e-4)
    train_parser.add_argument('--gamma', type=float, default=0.99)
    train_parser.add_argument('--batch-size', type=int, default=64)
    train_parser.add_argument('--n-epochs', type=int, default=4)
    train_parser.add_argument('--train-freq', type=int, default=10)
    train_parser.add_argument('--log-freq', type=int, default=100)
    train_parser.add_argument('--save-freq', type=int, default=1000)
    train_parser.add_argument('--save-dir', type=str, default='./checkpoints')
    train_parser.add_argument('--log-dir', type=str, default='./logs')
    train_parser.add_argument('--seed', type=int, default=42)
    train_parser.add_argument('--cpu', action='store_true')

    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate trained model')
    eval_parser.add_argument('--model', type=str, required=True)
    eval_parser.add_argument('--cpu', action='store_true')

    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run comprehensive benchmark')
    bench_parser.add_argument('--skeleton-ratio', type=float, default=0.7)

    args = parser.parse_args()

    if args.command == 'build':
        cmd_build(args)
    elif args.command == 'compare':
        cmd_compare(args)
    elif args.command == 'train':
        cmd_train(args)
    elif args.command == 'eval':
        cmd_eval(args)
    elif args.command == 'benchmark':
        cmd_benchmark(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
