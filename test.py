import numpy as np
import time

def compute_lambda2(adj, verbose=False):
    n = len(adj)
    degrees = np.sum(adj, axis=1)
    L = np.diag(degrees) - adj
    
    eigvals = np.linalg.eigvalsh(L)
    
    if verbose:
        print(f"All eigenvalues: {eigvals[:10]}") 
    
    lambda0 = eigvals[0]
    lambda1 = eigvals[1]
    
    if verbose:
        print(f"λ₀ = {lambda0:.10f} (should be ≈ 0)")
        print(f"λ₁ = {lambda1:.10f} (algebraic connectivity)")
    
    if abs(lambda0) > 1e-6:
        print(f"WARNING: λ₀ = {lambda0:.6f} is not close to 0! Graph may be disconnected.")
    
    return lambda1


def make_graph(n, k, l1, l2):
    """
    Build k-regular graph (k=3) using three-step process:
    Step 1: Connect upper half to lower half with long cords (n//2)
    Step 2: Add circular connections with cord lengths l1 (upper) and l2 (lower)
    Step 3: Fill remaining edges to achieve k-regularity
    
    Args:
        n: Number of nodes (must be even)
        k: Target degree (must be 3)
        l1: Cord length for upper half [0, n//2-1]
        l2: Cord length for lower half [n//2, n-1]
    
    Returns:
        adj: Adjacency matrix if k-regular, None otherwise
    """
    adj = np.zeros((n, n), dtype=int)
    long_cord_length = n // 2
    
    # Step 1: Perfect matching between upper and lower halves
    for i in range(long_cord_length):
        adj[i][i + long_cord_length] = 1
        adj[i + long_cord_length][i] = 1

    # Step 2: Add circular connections WITHIN each half
    # Upper half: connect i to (i + l1) mod (n//2)
    for i in range(long_cord_length):
        j = (i + l1) % long_cord_length  # Stay in upper half [0, n//2-1]
        if i != j and adj[i][j] == 0:  # Avoid self-loops and duplicate edges
            adj[i][j] = 1
            adj[j][i] = 1
    
    # Lower half: connect j to (j + l2) mod (n//2) within lower half
    for j in range(long_cord_length, n):
        local_idx = j - long_cord_length  # Convert to local index [0, n//2-1]
        target_local = (local_idx + l2) % long_cord_length
        target = target_local + long_cord_length  # Convert back to global index
        if j != target and adj[j][target] == 0:  # Avoid self-loops and duplicate edges
            adj[j][target] = 1
            adj[target][j] = 1
    
    # Step 3: Fill remaining edges to achieve k-regularity
    not_regular = is_k_regular(adj, k)
    
    # Pair up nodes that need more edges
    i = 0
    while i < len(not_regular) - 1:
        u, v = not_regular[i], not_regular[i + 1]
        if adj[u][v] == 0:  # Only add if not already connected
            adj[u][v] = 1
            adj[v][u] = 1
        i += 2
    
    # Verify k-regularity
    degrees = np.sum(adj, axis=1)
    if not np.all(degrees == k):
        return None
    
    return adj


def is_k_regular(adj, k):
    """Return list of nodes that don't have degree k."""
    not_reg_idx = []
    n = len(adj)
    for i in range(n):
        degree = np.sum(adj[i])
        if degree != k:
            not_reg_idx.append(i)
    return not_reg_idx


def test_all_cord_combinations(n, k=3):
    """
    Test all possible cord length combinations for given n.
    For k=3 and even n.
    
    Cord lengths:
    - l1: circular offset for upper half [0, n//2-1], valid range [1, n//2-1]
    - l2: circular offset for lower half [n//2, n-1], valid range [1, n//2-1]
    
    Step 1 uses n//2 (cross-connection), Step 2 adds circular connections within each half.
    """
    print(f"\n{'='*80}")
    print(f"Testing all cord combinations for n={n}, k={k}")
    print(f"{'='*80}\n")
    
    half = n // 2
    results = []
    
    # Theoretical upper bound for λ₂ in k-regular graph
    theoretical_max = k * (n / (n - 1))
    print(f"Theoretical upper bound: λ₂ ≤ {theoretical_max:.6f}\n")
    
    # Test all combinations of l1 and l2 (within-half circular offsets)
    # Enforce l1 > l2 to avoid duplicate solutions and exclude circulant graphs
    for l1 in range(1, half):
        for l2 in range(1, l1):  # l2 < l1 ensures l1 > l2
            adj = make_graph(n, k, l1, l2)
            
            if adj is not None:
                # Verify k-regularity
                degrees = np.sum(adj, axis=1)
                if not np.all(degrees == k):
                    print(f"ERROR: l1={l1}, l2={l2} produced non-k-regular graph!")
                    print(f"Degrees: {degrees}")
                    continue
                
                lambda2 = compute_lambda2(adj, verbose=False)
                
                # Sanity check
                if lambda2 > theoretical_max + 1e-6:
                    print(f"ERROR: l1={l1}, l2={l2} produced λ₂={lambda2:.6f} > theoretical max {theoretical_max:.6f}")
                    print("This indicates a bug in the eigenvalue computation!")
                    print(f"Degrees: {degrees}")
                    print(f"Adjacency matrix:\n{adj}")
                    raise ValueError(f"λ₂ exceeds theoretical bound!")
                
                results.append({
                    'l1': l1,
                    'l2': l2,
                    'lambda2': lambda2,
                    'adj': adj
                })
    
    # Sort by lambda2 descending to find BEST
    results.sort(key=lambda x: x['lambda2'], reverse=True)
    
    # Print top 20 results
    print(f"Found {len(results)} valid k-regular graphs\n")
    
    if n <= 50:  # Only print detailed rankings for small n
        print(f"{'Rank':<6} {'l1':<6} {'l2':<6} {'λ₂':<12} {'Check':<10}")
        print("-" * 50)
        
        for rank, result in enumerate(results[:20], 1):
            # Verify each result
            degrees = np.sum(result['adj'], axis=1)
            is_valid = np.all(degrees == k) and result['lambda2'] <= theoretical_max
            status = "✓" if is_valid else "✗"
            print(f"{rank:<6} {result['l1']:<6} {result['l2']:<6} {result['lambda2']:<12.8f} {status:<10}")
    
    if len(results) > 0:
        best = results[0]
        print("\n" + "=" * 80)
        print(f"BEST RESULT for n={n}: l1={best['l1']}, l2={best['l2']}, λ₂={best['lambda2']:.8f}")
        print("=" * 80)
        
        if n <= 32:  # Only print detailed info for small n
            # Detailed verification of best result
            print("\nDetailed verification:")
            degrees = np.sum(best['adj'], axis=1)
            print(f"k-regular: {np.all(degrees == k)} (all degrees = {k})")
            print(f"Degree range: [{degrees.min():.0f}, {degrees.max():.0f}]")
            print(f"λ₂ ≤ theoretical max: {best['lambda2'] <= theoretical_max}")
            
            # Check connectivity
            lambda2_detailed = compute_lambda2(best['adj'], verbose=True)
            print(f"\nConnected: {lambda2_detailed > 1e-6}")
            
            print(f"\nAdjacency matrix:\n{best['adj']}")
        
        # Compare to ERG - real data from baseline
        # ERG values from data_ERG_c directory for k=3 (m=3n/2)
        erg_data = {
            16: 0.96967815, 18: 0.95423583, 20: 0.75880236, 22: 0.66818251,
            24: 0.80012577, 26: 0.70561014, 28: 0.72746872, 30: 0.68929647,
            32: 0.58578644, 34: 0.58184989, 36: 0.60979636, 38: 0.52719544,
            40: 0.54248049, 42: 0.58926807, 44: 0.55011934, 46: 0.50197828,
            48: 0.57731716, 50: 0.53445077, 52: 0.48253120, 54: 0.50428975,
            56: 0.48024434, 58: 0.50865998, 60: 0.48249782, 62: 0.44837054,
            64: 0.47977905, 66: 0.45057421, 68: 0.45497824, 70: 0.46307754,
            72: 0.46356655, 74: 0.45621915, 76: 0.42005914, 78: 0.42632831,
            80: 0.43341707, 82: 0.41817368, 84: 0.42904504, 86: 0.41824015,
            88: 0.41756665, 90: 0.41062413, 92: 0.40642173, 94: 0.41323394,
            96: 0.40226196, 98: 0.41726317, 100: 0.38730488,
        }
        
        if n in erg_data:
            erg_lambda2 = erg_data[n]
            improvement = ((best['lambda2'] - erg_lambda2) / erg_lambda2) * 100
            print(f"\nERG baseline: {erg_lambda2:.8f}")
            print(f"Our result: {best['lambda2']:.8f}")
            print(f"Improvement: {improvement:+.2f}%")
            print(f"Status: {'WIN!' if improvement > 0 else 'LOSS'}")
    
    return results


def main():
    # Test for all even n from 16 to 100
    all_results = []
    
    for n in range(16, 102, 2):  # 16, 18, 20, ..., 100
        results = test_all_cord_combinations(n, k=3)
        if len(results) > 0:
            best = results[0]
            all_results.append({
                'n': n,
                'l1': best['l1'],
                'l2': best['l2'],
                'lambda2': best['lambda2']
            })
    
    # Print summary table
    print("\n\n" + "="*80)
    print("SUMMARY: Best λ₂ for all even n from 16 to 100")
    print("="*80)
    
    # ERG baseline data from data_ERG_c directory for k=3 (m=3n/2)
    erg_data = {
        16: 0.96967815, 18: 0.95423583, 20: 0.75880236, 22: 0.66818251,
        24: 0.80012577, 26: 0.70561014, 28: 0.72746872, 30: 0.68929647,
        32: 0.58578644, 34: 0.58184989, 36: 0.60979636, 38: 0.52719544,
        40: 0.54248049, 42: 0.58926807, 44: 0.55011934, 46: 0.50197828,
        48: 0.57731716, 50: 0.53445077, 52: 0.48253120, 54: 0.50428975,
        56: 0.48024434, 58: 0.50865998, 60: 0.48249782, 62: 0.44837054,
        64: 0.47977905, 66: 0.45057421, 68: 0.45497824, 70: 0.46307754,
        72: 0.46356655, 74: 0.45621915, 76: 0.42005914, 78: 0.42632831,
        80: 0.43341707, 82: 0.41817368, 84: 0.42904504, 86: 0.41824015,
        88: 0.41756665, 90: 0.41062413, 92: 0.40642173, 94: 0.41323394,
        96: 0.40226196, 98: 0.41726317, 100: 0.38730488,
    }
    
    print(f"\n{'n':<6} {'l1':<6} {'l2':<6} {'λ₂':<12} {'ERG':<12} {'Improvement':<12} {'Status':<8}")
    print("-"*70)
    
    for res in all_results:
        n = res['n']
        lambda2 = res['lambda2']
        
        # Use real ERG data
        if n in erg_data:
            erg_value = erg_data[n]
            improvement = ((lambda2 - erg_value) / erg_value) * 100
            status = "WIN" if improvement > 0 else "LOSS"
            
            print(f"{n:<6} {res['l1']:<6} {res['l2']:<6} {lambda2:<12.6f} {erg_value:<12.6f} {improvement:<+11.2f}% {status:<8}")
        else:
            print(f"{n:<6} {res['l1']:<6} {res['l2']:<6} {lambda2:<12.6f} {'N/A':<12} {'N/A':<12} {'N/A':<8}")
    
    # Statistics
    wins = sum(1 for res in all_results if res['n'] in erg_data and res['lambda2'] > erg_data[res['n']])
    losses = sum(1 for res in all_results if res['n'] in erg_data and res['lambda2'] <= erg_data[res['n']])
    print("-"*70)
    print(f"Results vs ERG baseline: {wins} wins, {losses} losses ({wins}/{wins+losses} = {100*wins/(wins+losses):.1f}%)")


if __name__ == "__main__":
    main()