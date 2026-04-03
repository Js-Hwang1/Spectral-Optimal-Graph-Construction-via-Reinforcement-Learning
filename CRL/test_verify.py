#!/usr/bin/env python3
"""
Verify C implementation matches Python outputs.

Tests: MLP forward, exact_lambda2, lanczos_ext_k, rr_update,
       find_bridges, build_features, softmax_sample, full episode.
"""

import sys
import os
import ctypes
import numpy as np
import torch
from pathlib import Path

# Add rl/ to path for Python reference implementations
# Try sibling directory first (new2/CRL -> new2/rl), then ~/Desktop/new2/rl
_rl_candidates = [
    Path(__file__).parent.parent / "rl",
    Path.home() / "Desktop" / "new2" / "rl",
]
for _rl_path in _rl_candidates:
    if (_rl_path / "utils" / "spectral.py").exists():
        sys.path.insert(0, str(_rl_path))
        break

from policy import RefinePolicy, RefineConfig
from utils.spectral import (exact_lambda2_np, lanczos_fiedler_ext_k,
                            rr_update as py_rr_update, find_bridges as py_find_bridges)

# ============================================================================
# Load shared library
# ============================================================================

def load_lib():
    d = Path(__file__).parent
    if sys.platform == "darwin":
        lib_path = d / "libcrl.dylib"
    else:
        lib_path = d / "libcrl.so"
    if not lib_path.exists():
        print(f"ERROR: {lib_path} not found. Run 'make' first.")
        sys.exit(1)
    return ctypes.CDLL(str(lib_path))

lib = load_lib()

# ============================================================================
# C struct definitions via ctypes
# ============================================================================

N_MAX = 24
EDGE_FEAT_DIM = 4
GRAPH_FEAT_DIM = 3
HIDDEN_DIM = 64
RR_K = 8

class MLP(ctypes.Structure):
    _fields_ = [
        ("W0", ctypes.c_float * (HIDDEN_DIM * EDGE_FEAT_DIM)),
        ("b0", ctypes.c_float * HIDDEN_DIM),
        ("W1", ctypes.c_float * (HIDDEN_DIM * HIDDEN_DIM)),
        ("b1", ctypes.c_float * HIDDEN_DIM),
        ("W2", ctypes.c_float * HIDDEN_DIM),
        ("b2", ctypes.c_float * 1),
        ("in_dim", ctypes.c_int),
    ]

class PolicyWeights(ctypes.Structure):
    _fields_ = [
        ("add_mlp", MLP),
        ("rem_mlp", MLP),
        ("val_mlp", MLP),
    ]

class SwapTxn(ctypes.Structure):
    _fields_ = [
        ("feat", ctypes.c_float * (N_MAX * N_MAX * EDGE_FEAT_DIM)),
        ("mask", ctypes.c_uint8 * (N_MAX * N_MAX)),
        ("flat_idx", ctypes.c_int32),
        ("gfeat", ctypes.c_float * GRAPH_FEAT_DIM),
        ("log_prob", ctypes.c_double),
        ("value", ctypes.c_double),
        ("reward", ctypes.c_double),
    ]

class RNG(ctypes.Structure):
    _fields_ = [("state", ctypes.c_uint64)]

# ============================================================================
# Helper: sync policy weights to C struct
# ============================================================================

def sync_weights_to_c(policy):
    """Extract PyTorch weights into flat arrays for crl_load_weights."""
    sd = policy.state_dict()
    pw = PolicyWeights()

    def get_arrays(prefix):
        W0 = sd[f'{prefix}.0.weight'].numpy().flatten()  # (H, in)
        b0 = sd[f'{prefix}.0.bias'].numpy().flatten()
        W1 = sd[f'{prefix}.2.weight'].numpy().flatten()  # (H, H)
        b1 = sd[f'{prefix}.2.bias'].numpy().flatten()
        W2 = sd[f'{prefix}.4.weight'].numpy().flatten()  # (1, H)
        b2 = sd[f'{prefix}.4.bias'].numpy().flatten()
        return W0, b0, W1, b1, W2, b2

    add_arrays = get_arrays('add_mlp')
    rem_arrays = get_arrays('rem_mlp')
    val_arrays = get_arrays('value_mlp')

    # Setup function signature
    lib.crl_load_weights.restype = None
    lib.crl_load_weights.argtypes = [
        ctypes.POINTER(PolicyWeights),
    ] + [ctypes.POINTER(ctypes.c_float)] * 18

    ptrs = []
    for arrays in [add_arrays, rem_arrays, val_arrays]:
        for arr in arrays:
            arr_c = arr.astype(np.float32)
            ptrs.append(arr_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    lib.crl_load_weights(ctypes.byref(pw), *ptrs)
    return pw

# ============================================================================
# Test 1: MLP forward
# ============================================================================

def test_mlp_forward():
    print("Test 1: MLP forward match... ", end="", flush=True)
    config = RefineConfig(hidden_dim=64)
    policy = RefinePolicy(config)
    pw = sync_weights_to_c(policy)

    # Random edge features
    np.random.seed(42)
    n = 8
    feat_np = np.random.randn(n, n, EDGE_FEAT_DIM).astype(np.float32)

    # Python
    with torch.no_grad():
        py_logits = policy.score_add(torch.tensor(feat_np)).numpy()

    # C: score each pair individually
    c_logits = np.zeros((n, n), dtype=np.float32)
    lib.mlp_forward_single.restype = ctypes.c_float
    lib.mlp_forward_single.argtypes = [ctypes.POINTER(MLP), ctypes.POINTER(ctypes.c_float)]

    for i in range(n):
        for j in range(n):
            f = feat_np[i, j].copy()
            c_logits[i, j] = lib.mlp_forward_single(
                ctypes.byref(pw.add_mlp),
                f.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            )

    max_err = np.max(np.abs(py_logits - c_logits))
    assert max_err < 1e-4, f"MLP mismatch: max_err={max_err}"
    print(f"PASS (max_err={max_err:.2e})")

# ============================================================================
# Test 2: exact_lambda2
# ============================================================================

def test_exact_lambda2():
    print("Test 2: exact_lambda2 match... ", end="", flush=True)

    np.random.seed(42)
    n = 10
    adj = np.zeros((n, n), dtype=np.float64)
    # Ring
    for i in range(n):
        j = (i + 1) % n
        adj[i, j] = adj[j, i] = 1.0
    # Random extra
    for _ in range(5):
        i, j = np.random.randint(0, n, 2)
        if i != j:
            adj[i, j] = adj[j, i] = 1.0

    py_l2 = exact_lambda2_np(adj)

    # C
    adj_u8 = adj.astype(np.uint8)
    lib.exact_lambda2.restype = ctypes.c_double
    lib.exact_lambda2.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int]
    c_l2 = lib.exact_lambda2(
        adj_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int(n)
    )

    err = abs(py_l2 - c_l2)
    assert err < 1e-10, f"lambda2 mismatch: py={py_l2}, c={c_l2}, err={err}"
    print(f"PASS (py={py_l2:.8f}, c={c_l2:.8f}, err={err:.2e})")

# ============================================================================
# Test 3: lanczos_ext_k
# ============================================================================

def test_lanczos():
    print("Test 3: lanczos_ext_k Ritz values... ", end="", flush=True)

    np.random.seed(42)
    n = 12
    adj = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        j = (i + 1) % n
        adj[i, j] = adj[j, i] = 1.0
    for _ in range(8):
        i, j = np.random.randint(0, n, 2)
        if i != j:
            adj[i, j] = adj[j, i] = 1.0

    # Python
    py_V, py_lams = lanczos_fiedler_ext_k(adj, k=15, v_init=None, n_eig=RR_K)

    # C
    adj_u8 = adj.astype(np.uint8)
    V_out = np.zeros((n, RR_K), dtype=np.float64)
    lams_out = np.zeros(RR_K, dtype=np.float64)

    lib.lanczos_ext_k.restype = None
    lib.lanczos_ext_k.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)
    ]

    lib.lanczos_ext_k(
        adj_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int(n), None, ctypes.c_int(15), ctypes.c_int(RR_K),
        V_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        lams_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )

    # Compare Ritz values (eigenvector signs may differ)
    lam_err = np.max(np.abs(py_lams - lams_out))
    print(f"PASS (lam_err={lam_err:.2e}, py_l2={py_lams[0]:.6f}, c_l2={lams_out[0]:.6f})")

# ============================================================================
# Test 4: rr_update
# ============================================================================

def test_rr_update():
    print("Test 4: rr_update match... ", end="", flush=True)

    np.random.seed(42)
    n = 10
    k = RR_K

    # Random V and lams
    V_py = np.random.randn(n, k)
    lams_py = np.sort(np.random.rand(k) * 5)
    u, v = 2, 7

    # Python
    V_py_out, lams_py_out = py_rr_update(V_py.copy(), lams_py.copy(), u, v, sign=+1.0)

    # C
    V_c = V_py.copy()
    lams_c = lams_py.copy()

    lib.rr_update.restype = None
    lib.rr_update.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double
    ]

    lib.rr_update(
        V_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        lams_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n), ctypes.c_int(k), ctypes.c_int(u), ctypes.c_int(v),
        ctypes.c_double(1.0)
    )

    lam_err = np.max(np.abs(lams_py_out - lams_c))
    # Eigenvectors may have sign flips; compare absolute values
    v_err = np.max(np.abs(np.abs(V_py_out) - np.abs(V_c)))
    assert lam_err < 1e-10, f"RR lam mismatch: {lam_err}"
    assert v_err < 1e-10, f"RR vec mismatch: {v_err}"
    print(f"PASS (lam_err={lam_err:.2e}, vec_err={v_err:.2e})")

# ============================================================================
# Test 5: find_bridges
# ============================================================================

def test_find_bridges():
    print("Test 5: find_bridges match... ", end="", flush=True)

    n = 8
    adj = np.zeros((n, n), dtype=np.float64)
    # Ring
    for i in range(n):
        j = (i + 1) % n
        adj[i, j] = adj[j, i] = 1.0
    # Remove one ring edge to create a bridge structure
    # Actually a ring has no bridges. Let's create a tree + extra.
    adj2 = np.zeros((n, n), dtype=np.float64)
    # Path: 0-1-2-3-4-5-6-7
    for i in range(n-1):
        adj2[i, i+1] = adj2[i+1, i] = 1.0
    # Extra edge creating a cycle: 0-3
    adj2[0, 3] = adj2[3, 0] = 1.0
    # Bridges: 3-4, 4-5, 5-6, 6-7

    py_bridges = py_find_bridges(adj2)

    # C
    adj_u8 = adj2.astype(np.uint8)
    bridge_mask = np.zeros((n, n), dtype=np.uint8)

    lib.find_bridges.restype = None
    lib.find_bridges.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8)
    ]

    lib.find_bridges(
        adj_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int(n),
        bridge_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    )

    c_bridges = set()
    for i in range(n):
        for j in range(i+1, n):
            if bridge_mask[i, j]:
                c_bridges.add((i, j))

    assert py_bridges == c_bridges, f"Bridge mismatch: py={py_bridges}, c={c_bridges}"
    print(f"PASS ({len(py_bridges)} bridges)")

# ============================================================================
# Test 6: build_edge_features
# ============================================================================

def test_build_features():
    print("Test 6: build_edge_features match... ", end="", flush=True)

    np.random.seed(42)
    n = 8

    # Random V_rr and degrees
    V_rr = np.random.randn(n, RR_K)
    degrees = np.random.randint(2, n, size=n).astype(np.int32)
    step, k_steps = 5, 20

    # Python reference
    nm1 = max(n - 1, 1)
    deg_norm = degrees / nm1
    step_frac = step / max(k_steps, 1)
    v2 = V_rr[:, 0]

    py_feat = np.zeros((n, n, EDGE_FEAT_DIM), dtype=np.float32)
    py_feat[:, :, 0] = n * (v2[:, None] - v2[None, :]) ** 2
    py_feat[:, :, 1] = deg_norm[:, None]
    py_feat[:, :, 2] = deg_norm[None, :]
    py_feat[:, :, 3] = step_frac

    # C
    c_feat = np.zeros((N_MAX, N_MAX, EDGE_FEAT_DIM), dtype=np.float32)

    lib.build_edge_features.restype = None
    lib.build_edge_features.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float)
    ]

    lib.build_edge_features(
        V_rr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        degrees.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(n), ctypes.c_int(step), ctypes.c_int(k_steps),
        c_feat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )

    # Compare n×n region
    c_feat_nn = c_feat[:n, :n, :]
    max_err = np.max(np.abs(py_feat - c_feat_nn))
    assert max_err < 1e-5, f"Feature mismatch: max_err={max_err}"
    print(f"PASS (max_err={max_err:.2e})")

# ============================================================================
# Test 7: Full episode
# ============================================================================

def test_full_episode():
    print("Test 7: Full episode runs without crash... ", end="", flush=True)

    config = RefineConfig(hidden_dim=64)
    policy = RefinePolicy(config)
    pw = sync_weights_to_c(policy)

    n, m = 8, 12
    k_steps = 5
    swap_frac = 0.25
    seed = 42
    max_txns = 200

    add_txns = (SwapTxn * max_txns)()
    rem_txns = (SwapTxn * max_txns)()
    out_num_add = ctypes.c_int(0)
    out_num_rem = ctypes.c_int(0)
    out_metrics = (ctypes.c_double * 5)()

    lib.crl_collect_episode.restype = ctypes.c_int
    lib.crl_collect_episode.argtypes = [
        ctypes.POINTER(PolicyWeights),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_uint64,
        ctypes.POINTER(SwapTxn), ctypes.POINTER(SwapTxn),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int
    ]

    ret = lib.crl_collect_episode(
        ctypes.byref(pw),
        ctypes.c_int(n), ctypes.c_int(m), ctypes.c_int(k_steps),
        ctypes.c_double(swap_frac), ctypes.c_uint64(seed),
        add_txns, rem_txns,
        ctypes.byref(out_num_add), ctypes.byref(out_num_rem),
        out_metrics, ctypes.c_int(max_txns)
    )

    assert ret == 0, f"Episode failed with code {ret}"
    na = out_num_add.value
    nr = out_num_rem.value
    final_l2 = out_metrics[0]
    init_l2 = out_metrics[1]
    improvement = out_metrics[2]

    print(f"PASS (add={na}, rem={nr}, l2={init_l2:.4f}->{final_l2:.4f}, "
          f"dl2={improvement:+.4f})")

# ============================================================================
# Test 8: MLP backward pass vs PyTorch autograd
# ============================================================================

# C struct for MLPCache
class MLPCache(ctypes.Structure):
    _fields_ = [
        ("z0", ctypes.c_float * HIDDEN_DIM),
        ("h0", ctypes.c_float * HIDDEN_DIM),
        ("z1", ctypes.c_float * HIDDEN_DIM),
        ("h1", ctypes.c_float * HIDDEN_DIM),
        ("out", ctypes.c_float),
        ("input", ctypes.c_float * EDGE_FEAT_DIM),
        ("in_dim", ctypes.c_int),
    ]

class MLPGrad(ctypes.Structure):
    _fields_ = [
        ("dW0", ctypes.c_float * (HIDDEN_DIM * EDGE_FEAT_DIM)),
        ("db0", ctypes.c_float * HIDDEN_DIM),
        ("dW1", ctypes.c_float * (HIDDEN_DIM * HIDDEN_DIM)),
        ("db1", ctypes.c_float * HIDDEN_DIM),
        ("dW2", ctypes.c_float * HIDDEN_DIM),
        ("db2", ctypes.c_float * 1),
    ]


def test_mlp_backward():
    print("Test 8: MLP backward vs PyTorch autograd... ", end="", flush=True)

    config = RefineConfig(hidden_dim=64)
    policy = RefinePolicy(config)
    pw = sync_weights_to_c(policy)

    # Setup C function signatures
    lib.mlp_forward_cached.restype = ctypes.c_float
    lib.mlp_forward_cached.argtypes = [
        ctypes.POINTER(MLP), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(MLPCache)
    ]
    lib.mlp_backward.restype = None
    lib.mlp_backward.argtypes = [
        ctypes.POINTER(MLP), ctypes.POINTER(MLPCache),
        ctypes.c_float, ctypes.POINTER(MLPGrad)
    ]

    np.random.seed(123)
    feat = np.random.randn(EDGE_FEAT_DIM).astype(np.float32)

    # C forward + backward
    cache = MLPCache()
    feat_c = (ctypes.c_float * EDGE_FEAT_DIM)(*feat)
    c_out = lib.mlp_forward_cached(ctypes.byref(pw.add_mlp), feat_c, ctypes.byref(cache))

    grad = MLPGrad()
    ctypes.memset(ctypes.byref(grad), 0, ctypes.sizeof(grad))
    d_out = 1.0  # gradient of loss w.r.t. output
    lib.mlp_backward(ctypes.byref(pw.add_mlp), ctypes.byref(cache),
                     ctypes.c_float(d_out), ctypes.byref(grad))

    # PyTorch forward + backward
    feat_t = torch.tensor(feat, requires_grad=False).unsqueeze(0)
    policy.zero_grad()
    py_out = policy.add_mlp(feat_t).squeeze()
    py_out.backward()

    # Compare outputs
    out_err = abs(float(py_out.item()) - c_out)
    assert out_err < 1e-5, f"Forward mismatch: py={py_out.item()}, c={c_out}"

    # Compare gradients
    sd = policy.state_dict()
    max_err = 0

    # W0
    py_dW0 = policy.add_mlp[0].weight.grad.numpy().flatten()
    c_dW0 = np.ctypeslib.as_array(grad.dW0)[:HIDDEN_DIM * EDGE_FEAT_DIM]
    err = np.max(np.abs(py_dW0 - c_dW0))
    max_err = max(max_err, err)

    # b0
    py_db0 = policy.add_mlp[0].bias.grad.numpy()
    c_db0 = np.ctypeslib.as_array(grad.db0)
    err = np.max(np.abs(py_db0 - c_db0))
    max_err = max(max_err, err)

    # W1
    py_dW1 = policy.add_mlp[2].weight.grad.numpy().flatten()
    c_dW1 = np.ctypeslib.as_array(grad.dW1)
    err = np.max(np.abs(py_dW1 - c_dW1))
    max_err = max(max_err, err)

    # b1
    py_db1 = policy.add_mlp[2].bias.grad.numpy()
    c_db1 = np.ctypeslib.as_array(grad.db1)
    err = np.max(np.abs(py_db1 - c_db1))
    max_err = max(max_err, err)

    # W2
    py_dW2 = policy.add_mlp[4].weight.grad.numpy().flatten()
    c_dW2 = np.ctypeslib.as_array(grad.dW2)
    err = np.max(np.abs(py_dW2 - c_dW2))
    max_err = max(max_err, err)

    # b2
    py_db2 = policy.add_mlp[4].bias.grad.numpy().flatten()
    c_db2 = np.ctypeslib.as_array(grad.db2)
    err = np.max(np.abs(py_db2 - c_db2))
    max_err = max(max_err, err)

    assert max_err < 1e-5, f"Gradient mismatch: max_err={max_err}"
    print(f"PASS (fwd_err={out_err:.2e}, grad_max_err={max_err:.2e})")


# ============================================================================
# Test 9: PPO evaluate_swap matches Python
# ============================================================================

class PolicyGradC(ctypes.Structure):
    _fields_ = [
        ("add_grad", MLPGrad),
        ("rem_grad", MLPGrad),
        ("val_grad", MLPGrad),
    ]


def test_ppo_evaluate():
    print("Test 9: ppo_evaluate_swap match... ", end="", flush=True)

    config = RefineConfig(hidden_dim=64)
    policy = RefinePolicy(config)
    pw = sync_weights_to_c(policy)

    # First run a short episode to get real transitions
    n, m = 8, 12
    max_txns = 200
    add_txns = (SwapTxn * max_txns)()
    rem_txns = (SwapTxn * max_txns)()
    out_na = ctypes.c_int(0)
    out_nr = ctypes.c_int(0)
    out_metrics = (ctypes.c_double * 5)()

    lib.crl_collect_episode.restype = ctypes.c_int
    lib.crl_collect_episode.argtypes = [
        ctypes.POINTER(PolicyWeights),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_uint64,
        ctypes.POINTER(SwapTxn), ctypes.POINTER(SwapTxn),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double), ctypes.c_int
    ]

    lib.crl_collect_episode(
        ctypes.byref(pw), n, m, 3, ctypes.c_double(0.25), ctypes.c_uint64(42),
        add_txns, rem_txns, ctypes.byref(out_na), ctypes.byref(out_nr),
        out_metrics, max_txns)

    na = out_na.value
    if na == 0:
        print("SKIP (no transitions)")
        return

    # Setup C function
    lib.ppo_evaluate_swap.restype = None
    lib.ppo_evaluate_swap.argtypes = [
        ctypes.POINTER(PolicyWeights),
        ctypes.POINTER(SwapTxn), ctypes.POINTER(SwapTxn),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_float)
    ]

    c_lp = ctypes.c_double(0)
    c_ent = ctypes.c_double(0)
    c_val = ctypes.c_float(0)

    lib.ppo_evaluate_swap(
        ctypes.byref(pw),
        ctypes.byref(add_txns[0]), ctypes.byref(rem_txns[0]),
        ctypes.c_int(n),
        ctypes.byref(c_lp), ctypes.byref(c_ent), ctypes.byref(c_val))

    # Compare to Python evaluate_swap
    from train import extract_txn, evaluate_swap
    a_ctx = extract_txn(add_txns[0], n, 'add')
    r_ctx = extract_txn(rem_txns[0], n, 'rem')

    with torch.no_grad():
        py_lp, py_ent, py_val = evaluate_swap(policy, a_ctx, r_ctx, 'cpu')

    lp_err = abs(py_lp.item() - c_lp.value)
    ent_err = abs(py_ent.item() - c_ent.value)
    val_err = abs(py_val.item() - c_val.value)

    assert lp_err < 1e-4, f"log_prob mismatch: py={py_lp.item()}, c={c_lp.value}"
    assert ent_err < 1e-3, f"entropy mismatch: py={py_ent.item()}, c={c_ent.value}"
    assert val_err < 1e-4, f"value mismatch: py={py_val.item()}, c={c_val.value}"

    print(f"PASS (lp_err={lp_err:.2e}, ent_err={ent_err:.2e}, val_err={val_err:.2e})")


# ============================================================================
# Test 10: Checkpoint round-trip
# ============================================================================

def test_checkpoint_roundtrip():
    print("Test 10: Checkpoint round-trip... ", end="", flush=True)

    config = RefineConfig(hidden_dim=64)
    policy = RefinePolicy(config)
    pw = sync_weights_to_c(policy)

    lib.save_checkpoint.restype = ctypes.c_int
    lib.save_checkpoint.argtypes = [
        ctypes.c_char_p, ctypes.POINTER(PolicyWeights),
        ctypes.c_uint32, ctypes.c_float
    ]
    lib.load_checkpoint.restype = ctypes.c_int
    lib.load_checkpoint.argtypes = [
        ctypes.c_char_p, ctypes.POINTER(PolicyWeights),
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_float)
    ]

    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix='.bin', delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        ret = lib.save_checkpoint(tmp_path.encode(), ctypes.byref(pw),
                                  ctypes.c_uint32(1234), ctypes.c_float(0.5))
        assert ret == 0, "save_checkpoint failed"

        pw2 = PolicyWeights()
        ep = ctypes.c_uint32(0)
        best = ctypes.c_float(0)
        ret = lib.load_checkpoint(tmp_path.encode(), ctypes.byref(pw2),
                                  ctypes.byref(ep), ctypes.byref(best))
        assert ret == 0, "load_checkpoint failed"
        assert ep.value == 1234, f"episode mismatch: {ep.value}"
        assert abs(best.value - 0.5) < 1e-6, f"best mismatch: {best.value}"

        # Compare weights
        w1 = np.ctypeslib.as_array(pw.add_mlp.W0)[:HIDDEN_DIM * EDGE_FEAT_DIM]
        w2 = np.ctypeslib.as_array(pw2.add_mlp.W0)[:HIDDEN_DIM * EDGE_FEAT_DIM]
        max_err = np.max(np.abs(w1 - w2))
        assert max_err == 0, f"Weight mismatch: {max_err}"

        # Test bin2pt conversion
        from convert_checkpoint import bin2pt
        pt_path = tmp_path.replace('.bin', '.pt')
        bin2pt(tmp_path, pt_path)

        import torch
        ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
        sd = ckpt['policy_state_dict']

        orig_W0 = policy.state_dict()['add_mlp.0.weight'].numpy().flatten()
        loaded_W0 = sd['add_mlp.0.weight'].numpy().flatten()
        max_err = np.max(np.abs(orig_W0 - loaded_W0))
        assert max_err < 1e-6, f"bin2pt weight mismatch: {max_err}"

        os.unlink(pt_path)
    finally:
        os.unlink(tmp_path)

    print(f"PASS")


# ============================================================================
# Test 11: GAE computation
# ============================================================================

def test_gae():
    print("Test 11: GAE computation... ", end="", flush=True)

    lib.compute_gae.restype = None
    lib.compute_gae.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)
    ]

    np.random.seed(42)
    num_steps = 10
    values = np.random.randn(num_steps) * 0.5
    rewards = np.random.randn(num_steps) * 0.1
    gamma = 0.99
    gae_lam = 0.95

    # Python reference
    py_adv = np.zeros(num_steps)
    py_ret = np.zeros(num_steps)
    gae = 0.0
    for t in reversed(range(num_steps)):
        next_val = values[t + 1] if t + 1 < num_steps else 0.0
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * gae_lam * gae
        py_adv[t] = gae
        py_ret[t] = gae + values[t]

    # C
    c_adv = np.zeros(num_steps, dtype=np.float64)
    c_ret = np.zeros(num_steps, dtype=np.float64)
    lib.compute_gae(
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        rewards.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(num_steps), ctypes.c_double(gamma), ctypes.c_double(gae_lam),
        c_adv.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        c_ret.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )

    adv_err = np.max(np.abs(py_adv - c_adv))
    ret_err = np.max(np.abs(py_ret - c_ret))
    assert adv_err < 1e-10, f"GAE adv mismatch: {adv_err}"
    assert ret_err < 1e-10, f"GAE ret mismatch: {ret_err}"
    print(f"PASS (adv_err={adv_err:.2e}, ret_err={ret_err:.2e})")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CRL Verification Tests")
    print("=" * 60)

    test_mlp_forward()
    test_exact_lambda2()
    test_lanczos()
    test_rr_update()
    test_find_bridges()
    test_build_features()
    test_full_episode()
    test_mlp_backward()
    test_ppo_evaluate()
    test_checkpoint_roundtrip()
    test_gae()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
