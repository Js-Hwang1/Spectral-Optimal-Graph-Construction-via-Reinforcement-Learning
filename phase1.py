import numpy as np
import scipy.linalg
import math

# =============================================================================
# 1. SPARSE ENGINE: Star-Mesh Builder (For M < N^2/4)
# =============================================================================
class StarMeshBuilder:
    def __init__(self, n):
        self.n = n
        self.adj = np.zeros((n, n), dtype=bool)
        self.edges = []
        # Ordered Dicts for O(1) Deterministic Bucket Queue
        self.buckets = [{} for _ in range(n)]
        self.leaf_degrees = [0] * n 
        for i in range(1, n):
            self.buckets[0][i] = None 
        self.min_ptr = 0
        self.stride = 0

    def _update_structure(self, u, v):
        if self.adj[u][v]: return False
        self.adj[u][v] = True; self.adj[v][u] = True
        self.edges.append((u, v))
        
        deg_u = self.leaf_degrees[u]
        deg_v = self.leaf_degrees[v]
        
        self.buckets[deg_u].pop(u, None)
        self.buckets[deg_v].pop(v, None)
        
        self.leaf_degrees[u] += 1
        self.leaf_degrees[v] += 1
        
        self.buckets[self.leaf_degrees[u]][u] = None
        self.buckets[self.leaf_degrees[v]][v] = None
        return True

    def build(self, m):
        # Phase 1: Star Backbone
        hub = 0
        limit = min(m, self.n - 1)
        for i in range(1, limit + 1):
            self.adj[hub][i] = True; self.adj[i][hub] = True
            self.edges.append((hub, i))
            
        if m <= self.n - 1: return

        # Phase 2: Leaf Mesh
        num_leaves = self.n - 1
        self.stride = int(num_leaves * 0.61803398875) 
        if self.stride % 2 == 0: self.stride += 1
        
        safety = 0
        max_attempts = (m - len(self.edges)) * 10
        
        while len(self.edges) < m and safety < max_attempts:
            if not self._add_mesh_edge():
                self.min_ptr += 1
                if self.min_ptr >= self.n: self.min_ptr = 0
            safety += 1

    def _add_mesh_edge(self):
        while self.min_ptr < len(self.buckets) and not self.buckets[self.min_ptr]:
            self.min_ptr += 1
        if self.min_ptr >= len(self.buckets): return False

        u = next(iter(self.buckets[self.min_ptr]))
        candidates = self.buckets[self.min_ptr]
        
        if len(candidates) < 2:
            target_idx = self.min_ptr + 1
            candidates = self.buckets[target_idx] if target_idx < len(self.buckets) else {}

        v = None
        if candidates:
            cand_list = list(candidates.keys())
            n_cand = len(cand_list)
            start_idx = (u * self.stride) % n_cand
            for k in range(n_cand):
                c = cand_list[(start_idx + k) % n_cand]
                if c != u and not self.adj[u][c]:
                    v = c; break
        
        if v is None: # Global Fallback
            for k in range(1, self.n):
                c = (u + k * self.stride) % self.n
                if c != 0 and c != u and not self.adj[u][c]:
                    v = c; break
                    
        return self._update_structure(u, v) if v is not None else False

# =============================================================================
# 2. DENSE ENGINE: Bipartite-Mesh Builder (For M >= N^2/4)
# =============================================================================
class BipartiteMeshBuilder:
    def __init__(self, n):
        self.n = n
        self.adj = np.zeros((n, n), dtype=bool)
        self.edges = []
        self.buckets = [{} for _ in range(n)]
        self.degrees = [0] * n
        for i in range(n):
            self.buckets[0][i] = None
        self.min_ptr = 0
        
        self.size_a = (n + 1) // 2
        self.size_b = n - self.size_a
        self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        self.prime_ptr = 0

    def _update_structure(self, u, v):
        if self.adj[u][v]: return False
        self.adj[u][v] = True; self.adj[v][u] = True
        self.edges.append((u, v))
        
        self.buckets[self.degrees[u]].pop(u, None)
        self.buckets[self.degrees[v]].pop(v, None)
        self.degrees[u] += 1; self.degrees[v] += 1
        self.buckets[self.degrees[u]][u] = None
        self.buckets[self.degrees[v]][v] = None
        return True

    def build(self, m):
        # Phase 1: Complete Bipartite Base
        for i in range(self.size_a):
            for j in range(self.size_a, self.n):
                self._update_structure(i, j)

        # Phase 2: Odd N Correction (Regularization)
        if self.n % 2 != 0:
            if self.size_a % 2 == 0: # A needs matching
                needed = self.size_a // 2
                if len(self.edges) + needed <= m:
                    for i in range(0, self.size_a, 2):
                        self._update_structure(i, i+1)
            else: # A needs cycle, B needs matching
                needed = self.size_a + (self.size_b // 2)
                if len(self.edges) + needed <= m:
                    for i in range(self.size_a):
                        self._update_structure(i, (i + 1) % self.size_a)
                    for i in range(self.size_a, self.n, 2):
                        self._update_structure(i, i+1)

        # Phase 3: Internal Mesh Fill
        self.min_ptr = 0
        while self.min_ptr < len(self.buckets) and not self.buckets[self.min_ptr]:
            self.min_ptr += 1
            
        safety = 0
        max_attempts = (m - len(self.edges)) * 10
        while len(self.edges) < m and safety < max_attempts:
            if not self._add_constrained_edge():
                 self.min_ptr += 1
                 if self.min_ptr >= self.n: self.min_ptr = 0
            safety += 1

    def _add_constrained_edge(self):
        while self.min_ptr < len(self.buckets) and not self.buckets[self.min_ptr]:
            self.min_ptr += 1
        if self.min_ptr >= len(self.buckets): return False
        
        u = next(iter(self.buckets[self.min_ptr]))
        u_in_a = (u < self.size_a)
        
        candidates = self.buckets[self.min_ptr]
        v = None
        stride = self.primes[self.prime_ptr % len(self.primes)]
        self.prime_ptr += 1
        
        cand_list = list(candidates.keys())
        n_cand = len(cand_list)
        start_idx = (u * stride) % n_cand
        
        for k in range(n_cand):
            c = cand_list[(start_idx + k) % n_cand]
            if (c < self.size_a) == u_in_a and c != u and not self.adj[u][c]:
                v = c; break
        
        if v is None: # Fallback scan in partition
            rng = range(0, self.size_a) if u_in_a else range(self.size_a, self.n)
            g_stride = max(1, len(rng) // 2) | 1
            for k in range(len(rng)):
                idx = rng[(k * g_stride) % len(rng)]
                if idx != u and not self.adj[u][idx]:
                    v = idx; break

        return self._update_structure(u, v) if v is not None else False

# =============================================================================
# 3. UNIFIED ADAPTIVE CONTROLLER
# =============================================================================
class AdaptiveGraphBuilder:
    def __init__(self, n):
        self.n = n
        self.builder = None
        
    def build(self, m):
        # Threshold: Floor(N^2 / 4) is the max edges of a bipartite graph
        threshold = (self.n * self.n) // 4
        
        if m < threshold:
            # Sparse/Medium: Use Star-Mesh
            self.builder = StarMeshBuilder(self.n)
        else:
            # Dense: Use Bipartite-Mesh
            self.builder = BipartiteMeshBuilder(self.n)
            
        self.builder.build(m)
        self.adj = self.builder.adj
        self.edges = self.builder.edges

    def get_lambda2(self):
        degrees = np.sum(self.adj, axis=0)
        L = np.diag(degrees) - self.adj.astype(int)
        if self.n > 1 and np.sum(self.adj[0]) == 0: return 0.0
        eigvals = scipy.linalg.eigh(L, eigvals_only=True)
        eigvals.sort()
        return eigvals[1] if len(eigvals) > 1 else 0.0

# =============================================================================
# 4. BASELINE: Small World (Fixed M)
# =============================================================================
def build_small_world_graph(n, m, p=0.0, seed=42):
    rng = np.random.default_rng(seed)
    adj = np.zeros((n, n), dtype=int)
    edge_count = 0; step = 1
    max_edges = n * (n - 1) // 2
    m = min(m, max_edges)
    
    while edge_count < m and step <= n // 2:
        for i in range(n):
            if edge_count >= m: break
            j = (i + step) % n
            if i != j and adj[i, j] == 0:
                adj[i, j] = 1; adj[j, i] = 1; edge_count += 1
        step += 1
        
    if p > 0:
        edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j]]
        for u, v in edges_list:
            if rng.random() < p:
                adj[u, v] = 0; adj[v, u] = 0
                cands = [k for k in range(n) if k != u and adj[u, k] == 0]
                if cands:
                    new_v = rng.choice(cands)
                    adj[u, new_v] = 1; adj[new_v, u] = 1
                else:
                    adj[u, v] = 1; adj[v, u] = 1
    return adj

def get_matrix_lambda2(adj):
    degrees = np.sum(adj, axis=0)
    L = np.diag(degrees) - adj
    try:
        eigvals = scipy.linalg.eigh(L, eigvals_only=True)
        eigvals.sort()
        return eigvals[1] if len(eigvals) > 1 else 0.0
    except: return 0.0

# =============================================================================
# 5. EXPERIMENT RUNNER
# =============================================================================
# Test across Sparse (k=4) -> Medium -> Dense (k=N/2) -> Very Dense (k near N)
test_cases = [
    # Sparse / Star Regime
    (20, 30), (40, 60), (50, 100),
    # Transition Regime
    (20, 90), (40, 390),
    # Dense / Bipartite Regime (M > N^2/4)
    (20, 105), (20, 150),
    (40, 410), (40, 600),
    (50, 700), (50, 1000)
]

sw_probs = [0.0, 0.25, 0.5, 0.75, 1.0]

print(f"{'PARAMS':<12} | {'MODE':<10} | {'OURS (Adaptive)':<18} | {'SMALL WORLD (Max)':<18} | {'RESULT'}")
print("-" * 80)

wins = 0; losses = 0

for n, m in test_cases:
    # 1. Run Ours
    builder = AdaptiveGraphBuilder(n)
    builder.build(m)
    our_l2 = builder.get_lambda2()
    
    mode = "Star" if isinstance(builder.builder, StarMeshBuilder) else "Bipartite"
    
    # 2. Run Small World
    best_sw = -1.0
    for p in sw_probs:
        try:
            adj = build_small_world_graph(n, m, p=p)
            val = get_matrix_lambda2(adj)
            if val > best_sw: best_sw = val
        except: pass
        
    res = "WIN" if our_l2 >= best_sw - 0.001 else "LOSE"
    if res == "WIN": wins += 1
    else: losses += 1
    
    print(f"N={n:<3} M={m:<4} | {mode:<10} | {our_l2:.6f}           | {best_sw:.6f}           | {res}")

print("-" * 80)
print(f"SCORE: {wins} WINS, {losses} LOSSES")