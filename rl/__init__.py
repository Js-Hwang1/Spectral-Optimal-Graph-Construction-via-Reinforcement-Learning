"""
G(n,m) Foundational RL Model for High Algebraic Connectivity Graphs

Learns to construct graphs with maximized algebraic connectivity (λ₂).
Given (n, m), outputs a graph with n nodes and m edges.

Key components:
- envs/gnm_env.py: G(n,m) environment with edge rewiring actions
- models/gnm_policy.py: Policy network that learns ρ, φ, and destination
- utils/ours.py: OURS baseline for comparison
- train_gnm.py: Main training script
"""

__version__ = "1.0.0"
