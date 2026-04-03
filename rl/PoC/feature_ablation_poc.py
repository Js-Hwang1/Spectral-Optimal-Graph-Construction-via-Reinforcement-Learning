#!/usr/bin/env python3
"""
Feature Ablation PoC — Are v3/v4/v5 eigenvector features helpful or noise?

Hypothesis: The extra spectral features (v3,v4,v5 gaps and spectral gap scalars)
may be adding noise that confuses the MLP, causing it to make worse edge selections
than random rewiring (SW baselines).

Method:
  1. Generate random graphs at various (n, m) configs
  2. For each graph, compute TRUE Δλ₂ for every possible non-edge addition
  3. Rank edges by Δλ₂ (ground truth: which edges actually help most)
  4. Compute features under different feature sets
  5. Train tiny MLPs to predict edge quality from features
  6. Measure: which feature set best identifies the TOP-K best edges?

Feature sets tested:
  A. "Fiedler only" (4 features): v2 gap, deg_i, deg_j, step
  B. "Fiedler + global" (7 features): A + spectral gaps 2-3, 3-4, 4-5
  C. "Full spectrum" (10 features): current full feature set (v2,v3,v4,v5 gaps + all)
  D. "Degree only" (3 features): deg_i, deg_j, step (no spectral info at all)

We also measure raw feature correlation with Δλ₂ — no MLP needed — to see
which individual features actually correlate with edge quality.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.linalg import eigh
from collections import defaultdict
import time


def algebraic_connectivity(adj):
    """Exact λ₂ from adjacency matrix."""
    n = adj.shape[0]
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    evals = eigh(L, eigvals_only=True)
    return float(evals[1])


def full_spectrum(adj, k=4):
    """Get first k+1 eigenvalues and eigenvectors (skip λ₁=0)."""
    n = adj.shape[0]
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    evals, evecs = eigh(L)
    # Return λ₂..λ_{k+1} and v₂..v_{k+1}
    return evals[1:k+1], evecs[:, 1:k+1]


def make_random_graph(n, m):
    """Ring + random edges to reach m edges."""
    adj = np.zeros((n, n), dtype=np.float64)
    # Ring
    for i in range(n):
        j = (i + 1) % n
        adj[i, j] = adj[j, i] = 1
    current_m = n
    upper = np.triu_indices(n, k=1)
    non_edges = [(upper[0][idx], upper[1][idx])
                 for idx in range(len(upper[0]))
                 if adj[upper[0][idx], upper[1][idx]] == 0]
    np.random.shuffle(non_edges)
    for i, j in non_edges:
        if current_m >= m:
            break
        adj[i, j] = adj[j, i] = 1
        current_m += 1
    return adj


def compute_all_add_deltas(adj):
    """For every non-edge (i,j), compute Δλ₂ if we add it. Returns dict."""
    n = adj.shape[0]
    base_l2 = algebraic_connectivity(adj)
    deltas = {}
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] == 0:
                adj[i, j] = adj[j, i] = 1
                new_l2 = algebraic_connectivity(adj)
                deltas[(i, j)] = new_l2 - base_l2
                adj[i, j] = adj[j, i] = 0
    return deltas, base_l2


def build_edge_features(adj, i, j, lams, vecs, step_frac=0.5):
    """
    Build features for edge (i,j) under different feature sets.
    Returns dict of feature-set-name → feature vector.
    """
    n = adj.shape[0]
    nm1 = max(n - 1, 1)
    deg = adj.sum(axis=1)

    v2 = vecs[:, 0]
    v3 = vecs[:, 1] if vecs.shape[1] > 1 else np.zeros(n)
    v4 = vecs[:, 2] if vecs.shape[1] > 2 else np.zeros(n)
    v5 = vecs[:, 3] if vecs.shape[1] > 3 else np.zeros(n)

    lam2 = lams[0]
    lam3 = lams[1] if len(lams) > 1 else lam2
    lam4 = lams[2] if len(lams) > 2 else lam3
    lam5 = lams[3] if len(lams) > 3 else lam4

    # Individual features
    fiedler_gap = n * (v2[i] - v2[j]) ** 2
    v3_gap = n * (v3[i] - v3[j]) ** 2
    v4_gap = n * (v4[i] - v4[j]) ** 2
    v5_gap = n * (v5[i] - v5[j]) ** 2
    deg_i = deg[i] / nm1
    deg_j = deg[j] / nm1
    gap_23 = max(lam3 - lam2, 0) / n
    gap_34 = max(lam4 - lam3, 0) / n
    gap_45 = max(lam5 - lam4, 0) / n

    return {
        'degree_only': np.array([deg_i, deg_j, step_frac]),
        'fiedler_only': np.array([fiedler_gap, deg_i, deg_j, step_frac]),
        'fiedler_global': np.array([fiedler_gap, deg_i, deg_j, gap_23, gap_34, gap_45, step_frac]),
        'full': np.array([fiedler_gap, v3_gap, v4_gap, v5_gap, deg_i, deg_j, gap_23, gap_34, gap_45, step_frac]),
    }


def correlation_analysis(all_features, all_deltas, feature_names_10):
    """Compute Pearson correlation of each feature with Δλ₂."""
    feats = np.array(all_features)  # (N_edges, 10)
    deltas = np.array(all_deltas)   # (N_edges,)

    print("\n" + "=" * 60)
    print("RAW FEATURE CORRELATIONS WITH Δλ₂")
    print("=" * 60)
    print(f"{'Feature':<25} {'Pearson r':>10} {'|r|':>8}")
    print("-" * 45)

    correlations = []
    for f_idx, fname in enumerate(feature_names_10):
        feat_col = feats[:, f_idx]
        if feat_col.std() < 1e-10:
            r = 0.0
        else:
            r = np.corrcoef(feat_col, deltas)[0, 1]
        correlations.append((fname, r))
        print(f"{fname:<25} {r:>10.4f} {abs(r):>8.4f}")

    print("-" * 45)
    # Sort by |r|
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    print("\nRanked by |correlation|:")
    for fname, r in correlations:
        bar = "█" * int(abs(r) * 40)
        print(f"  {fname:<25} |r|={abs(r):.4f} {bar}")


def topk_overlap(predicted_scores, true_deltas, k):
    """What fraction of the top-K predicted edges are in the true top-K?"""
    pred_topk = set(np.argsort(predicted_scores)[-k:])
    true_topk = set(np.argsort(true_deltas)[-k:])
    return len(pred_topk & true_topk) / k


def mlp_experiment(train_feats, train_deltas, test_feats, test_deltas, hidden=32, epochs=200, lr=1e-3):
    """
    Train a tiny MLP to predict Δλ₂ from features, measure top-K overlap on test set.
    Pure numpy implementation (no torch dependency for PoC).
    """
    in_dim = train_feats.shape[1]

    # Simple 2-layer MLP: in → hidden → 1
    np.random.seed(42)
    W1 = np.random.randn(in_dim, hidden) * 0.1
    b1 = np.zeros(hidden)
    W2 = np.random.randn(hidden, 1) * 0.1
    b2 = np.zeros(1)

    # Normalize inputs
    mu = train_feats.mean(axis=0)
    std = train_feats.std(axis=0) + 1e-8
    X_train = (train_feats - mu) / std
    X_test = (test_feats - mu) / std

    # Normalize targets
    y_mu = train_deltas.mean()
    y_std = train_deltas.std() + 1e-8
    y_train = (train_deltas - y_mu) / y_std
    y_test = (test_deltas - y_mu) / y_std

    for epoch in range(epochs):
        # Forward
        h = X_train @ W1 + b1
        h_act = np.maximum(h, 0)  # ReLU
        pred = (h_act @ W2 + b2).squeeze()

        # MSE loss
        loss = np.mean((pred - y_train) ** 2)

        # Backward
        dpred = 2 * (pred - y_train) / len(y_train)
        dW2 = h_act.T @ dpred.reshape(-1, 1)
        db2 = dpred.sum()
        dh_act = dpred.reshape(-1, 1) @ W2.T
        dh = dh_act * (h > 0)
        dW1 = X_train.T @ dh
        db1 = dh.sum(axis=0)

        # SGD
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    # Test prediction
    h_test = X_test @ W1 + b1
    h_test_act = np.maximum(h_test, 0)
    pred_test = (h_test_act @ W2 + b2).squeeze()

    # Measure top-K overlap
    results = {}
    for k_frac in [0.05, 0.10, 0.20]:
        k = max(1, int(len(test_deltas) * k_frac))
        overlap = topk_overlap(pred_test, y_test, k)
        results[f"top_{int(k_frac*100)}%"] = overlap

    # Also measure rank correlation (Spearman)
    from scipy.stats import spearmanr
    rho, _ = spearmanr(pred_test, y_test)
    results['spearman'] = rho

    # MSE on test
    test_loss = np.mean((pred_test - y_test) ** 2)
    results['test_mse'] = test_loss

    return results


def main():
    print("=" * 60)
    print("FEATURE ABLATION PoC")
    print("Are v3/v4/v5 eigenvector features helpful or noise?")
    print("=" * 60)

    feature_names_10 = [
        'v2_gap (Fiedler)', 'v3_gap', 'v4_gap', 'v5_gap',
        'deg_i', 'deg_j',
        'gap_λ₃-λ₂', 'gap_λ₄-λ₃', 'gap_λ₅-λ₄',
        'step_frac'
    ]

    feature_sets = ['degree_only', 'fiedler_only', 'fiedler_global', 'full']
    feature_dims = {'degree_only': 3, 'fiedler_only': 4, 'fiedler_global': 7, 'full': 10}

    # ============================================================
    # Phase 1: Collect data across multiple (n, m) configs
    # ============================================================
    configs = []
    for n in [10, 12, 14, 16]:
        max_m = n * (n - 1) // 2
        min_m = n
        # Sample at low, mid, high density
        for frac in [0.2, 0.35, 0.5, 0.65, 0.8]:
            m = max(min_m, min(max_m, int(frac * max_m)))
            configs.append((n, m))

    print(f"\nTesting {len(configs)} (n,m) configs...")

    all_data = {fs: {'features': [], 'deltas': []} for fs in feature_sets}
    full_features_for_corr = []  # For correlation analysis

    t0 = time.time()
    for ci, (n, m) in enumerate(configs):
        density = m / (n * (n - 1) // 2)
        adj = make_random_graph(n, m)
        deltas, base_l2 = compute_all_add_deltas(adj)

        if not deltas:
            continue

        lams, vecs = full_spectrum(adj, k=4)

        edges = list(deltas.keys())
        delta_vals = np.array([deltas[e] for e in edges])

        for i, j in edges:
            feat_sets = build_edge_features(adj, i, j, lams, vecs)
            for fs in feature_sets:
                all_data[fs]['features'].append(feat_sets[fs])
            full_features_for_corr.append(feat_sets['full'])
            all_data[feature_sets[0]]['deltas'].append(deltas[(i, j)])

        print(f"  [{ci+1}/{len(configs)}] n={n}, m={m} (density={density:.2f}): "
              f"{len(edges)} non-edges, Δλ₂ range [{delta_vals.min():.4f}, {delta_vals.max():.4f}]")

    elapsed = time.time() - t0
    print(f"\nData collection: {elapsed:.1f}s")

    # Shared deltas (same edges for all feature sets)
    all_deltas = np.array(all_data[feature_sets[0]]['deltas'])

    # ============================================================
    # Phase 2: Raw correlation analysis
    # ============================================================
    correlation_analysis(full_features_for_corr, all_deltas, feature_names_10)

    # ============================================================
    # Phase 3: MLP prediction experiment
    # ============================================================
    print("\n" + "=" * 60)
    print("MLP EDGE QUALITY PREDICTION")
    print("=" * 60)
    print("Train tiny MLP to predict Δλ₂ from features.")
    print("Measure: top-K overlap (can the MLP identify the best edges?)")
    print()

    N = len(all_deltas)
    split = int(0.7 * N)
    perm = np.random.RandomState(123).permutation(N)
    train_idx = perm[:split]
    test_idx = perm[split:]

    print(f"Train: {split} edges, Test: {N - split} edges\n")
    print(f"{'Feature Set':<20} {'Dims':>4} {'Top-5%':>8} {'Top-10%':>9} {'Top-20%':>9} {'Spearman':>9} {'Test MSE':>9}")
    print("-" * 70)

    for fs in feature_sets:
        feats = np.array(all_data[fs]['features'])
        train_f = feats[train_idx]
        test_f = feats[test_idx]
        train_d = all_deltas[train_idx]
        test_d = all_deltas[test_idx]

        results = mlp_experiment(train_f, train_d, test_f, test_d,
                                  hidden=32, epochs=300, lr=1e-3)

        print(f"{fs:<20} {feature_dims[fs]:>4} "
              f"{results['top_5%']:>8.1%} {results['top_10%']:>9.1%} "
              f"{results['top_20%']:>9.1%} {results['spearman']:>9.4f} "
              f"{results['test_mse']:>9.4f}")

    # ============================================================
    # Phase 4: Per-density analysis
    # ============================================================
    print("\n" + "=" * 60)
    print("PER-DENSITY CORRELATION ANALYSIS")
    print("=" * 60)
    print("Does v2 gap (Fiedler) correlation change with density?\n")

    # Re-collect per-config stats
    print(f"{'Config':<12} {'Density':>7} {'|r(v2,Δλ₂)|':>12} {'|r(v3,Δλ₂)|':>12} "
          f"{'|r(v4,Δλ₂)|':>12} {'|r(deg,Δλ₂)|':>12} {'Fiedler signal':>14}")
    print("-" * 85)

    for n, m in configs:
        density = m / (n * (n - 1) // 2)
        adj = make_random_graph(n, m)
        deltas, _ = compute_all_add_deltas(adj)
        if not deltas:
            continue

        lams, vecs = full_spectrum(adj, k=4)
        edges = list(deltas.keys())

        v2_gaps, v3_gaps, v4_gaps, deg_feats, dvals = [], [], [], [], []
        for i, j in edges:
            v2_gaps.append(n * (vecs[i, 0] - vecs[j, 0])**2)
            v3_gaps.append(n * (vecs[i, 1] - vecs[j, 1])**2 if vecs.shape[1] > 1 else 0)
            v4_gaps.append(n * (vecs[i, 2] - vecs[j, 2])**2 if vecs.shape[1] > 2 else 0)
            deg = adj.sum(axis=1)
            deg_feats.append((deg[i] + deg[j]) / (2 * (n - 1)))
            dvals.append(deltas[(i, j)])

        v2_r = abs(np.corrcoef(v2_gaps, dvals)[0, 1]) if np.std(v2_gaps) > 1e-10 else 0
        v3_r = abs(np.corrcoef(v3_gaps, dvals)[0, 1]) if np.std(v3_gaps) > 1e-10 else 0
        v4_r = abs(np.corrcoef(v4_gaps, dvals)[0, 1]) if np.std(v4_gaps) > 1e-10 else 0
        deg_r = abs(np.corrcoef(deg_feats, dvals)[0, 1]) if np.std(deg_feats) > 1e-10 else 0

        signal = "STRONG" if v2_r > 0.5 else "MODERATE" if v2_r > 0.3 else "WEAK"
        print(f"n={n:>2},m={m:<4} {density:>7.2f} {v2_r:>12.4f} {v3_r:>12.4f} "
              f"{v4_r:>12.4f} {deg_r:>12.4f} {signal:>14}")

    # ============================================================
    # Phase 5: Noise injection test
    # ============================================================
    print("\n" + "=" * 60)
    print("NOISE TEST: Does replacing v3/v4/v5 with random noise hurt?")
    print("=" * 60)
    print("If v3/v4/v5 are helpful, replacing them with noise should degrade performance.")
    print("If they're noise, performance should be similar.\n")

    # Take full features and replace columns 1,2,3 (v3,v4,v5 gaps) with random noise
    full_feats = np.array(all_data['full']['features'])
    noisy_feats = full_feats.copy()
    rng = np.random.RandomState(999)
    for col in [1, 2, 3]:  # v3, v4, v5 gap columns
        noisy_feats[:, col] = rng.randn(len(noisy_feats)) * full_feats[:, col].std()

    train_f = full_feats[train_idx]
    test_f = full_feats[test_idx]
    train_fn = noisy_feats[train_idx]
    test_fn = noisy_feats[test_idx]
    train_d = all_deltas[train_idx]
    test_d = all_deltas[test_idx]

    results_real = mlp_experiment(train_f, train_d, test_f, test_d, hidden=32, epochs=300, lr=1e-3)
    results_noise = mlp_experiment(train_fn, train_d, test_fn, test_d, hidden=32, epochs=300, lr=1e-3)

    print(f"{'Variant':<25} {'Top-5%':>8} {'Top-10%':>9} {'Top-20%':>9} {'Spearman':>9}")
    print("-" * 55)
    print(f"{'Full (real v3/v4/v5)':<25} {results_real['top_5%']:>8.1%} {results_real['top_10%']:>9.1%} "
          f"{results_real['top_20%']:>9.1%} {results_real['spearman']:>9.4f}")
    print(f"{'Noisy (random v3/v4/v5)':<25} {results_noise['top_5%']:>8.1%} {results_noise['top_10%']:>9.1%} "
          f"{results_noise['top_20%']:>9.1%} {results_noise['spearman']:>9.4f}")

    diff_top5 = results_real['top_5%'] - results_noise['top_5%']
    diff_spear = results_real['spearman'] - results_noise['spearman']
    print(f"\nDifference (real - noise): Top-5% = {diff_top5:+.1%}, Spearman = {diff_spear:+.4f}")

    if abs(diff_top5) < 0.03 and abs(diff_spear) < 0.02:
        print("\n>>> VERDICT: v3/v4/v5 features are NOISE — replacing them with random values")
        print("    doesn't meaningfully change prediction quality.")
    elif diff_top5 > 0.05:
        print("\n>>> VERDICT: v3/v4/v5 features are HELPFUL — real features significantly")
        print("    outperform random noise.")
    else:
        print("\n>>> VERDICT: v3/v4/v5 features have MARGINAL value — small improvement")
        print("    over noise, but may not justify the complexity.")


if __name__ == "__main__":
    main()
