import argparse
import json
import time
import torch

"""
python3 src/evaluate.py --n 32 --m 240 --load_model runs/v11_n96_train_ep08080.pt --topk 8
"""

from train import (
    GraphBuildEnv,
    EnvConfig,
    PolicyValueNet,
    PPOConfig,
    PPOAgent,
    run_episode,
    set_num_threads,
    set_seed,
)


def main():
    p = argparse.ArgumentParser(description="Evaluate a trained graph-construction policy (no training).")

    # Task
    p.add_argument('--n', type=int, default=32, help='Number of nodes')
    p.add_argument('--m', type=int, default=100, help='Number of edges')
    p.add_argument('--init', type=str, default='path', help="Initial graph: 'path' or 'empty'")

    # Candidate generation / spectral
    p.add_argument('--topk', type=int, default=8, help='ER top-k candidate edges per step')
    p.add_argument('--topk_frac', type=float, default=0.2, help='Cap ER candidates to this fraction of non-edges')
    p.add_argument('--spectral_backend', type=str, default='scipy', choices=['scipy', 'torch'])
    p.add_argument('--spectral_refresh_k', type=int, default=1, help='Refresh spectral cache every k steps (ignored if fast_spectral=False)')

    # Reward reporting (for logging only)
    p.add_argument('--reward_mode', type=str, default='ratio', choices=['margin', 'ratio', 'logratio'])
    p.add_argument('--reward_eps', type=float, default=1e-6)

    # Runtime
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='cpu')

    # Model
    p.add_argument('--load_model', type=str, required=True, help='Path to a saved model .pt file')

    # Optional arch overrides (used only if no arch is found in checkpoint)
    p.add_argument('--gat_hidden', type=int, default=64)
    p.add_argument('--gat_heads', type=int, default=6)
    p.add_argument('--gat_layers', type=int, default=3)
    p.add_argument('--edge_mlp_hidden', type=int, default=256)
    p.add_argument('--value_mlp_hidden', type=int, default=128)

    args = p.parse_args()

    set_num_threads(args.threads)
    set_seed(args.seed)

    device = torch.device(args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu')

    # Build a provisional net/agent; will rebuild to match checkpoint arch if needed
    arch = {
        'node_in': 5,
        'hid': args.gat_hidden,
        'heads': args.gat_heads,
        'layers': args.gat_layers,
        'edge_mlp_hidden': args.edge_mlp_hidden,
        'value_mlp_hidden': args.value_mlp_hidden,
    }
    net = PolicyValueNet(
        arch['node_in'], arch['hid'], arch['heads'], arch['layers'],
        arch['edge_mlp_hidden'], arch['value_mlp_hidden'], device=device.type
    ).to(device)
    ppo_cfg = PPOConfig()  # defaults are fine for evaluation
    agent = PPOAgent(net, ppo_cfg, device=device)

    # Load checkpoint and reconcile architecture
    obj = agent.load(args.load_model, device=device)
    chk_arch = obj.get('arch', arch)
    if json.dumps(chk_arch, sort_keys=True) != json.dumps(arch, sort_keys=True):
        arch = chk_arch
        net = PolicyValueNet(
            arch['node_in'], arch['hid'], arch['heads'], arch['layers'],
            arch['edge_mlp_hidden'], arch['value_mlp_hidden'], device=device.type
        ).to(device)
        agent = PPOAgent(net, ppo_cfg, device=device)
    net.load_state_dict(obj['state_dict'], strict=True)
    # Optimizer state is not needed for evaluation; ignore if missing
    try:
        agent.opt.load_state_dict(obj['opt'])
    except Exception:
        pass
    print(f"[load] Loaded model from: {args.load_model}")

    # Environment (no fast_spectral to keep spectral features fresh during evaluation)
    cfg = EnvConfig(
        n=args.n,
        m=args.m,
        init=args.init,
        spectral_backend=args.spectral_backend,
        device=device.type,
        fast_spectral=False,
        spectral_refresh_k=args.spectral_refresh_k,
    )
    env = GraphBuildEnv(cfg)

    t0 = time.time()
    lam2, reward, steps, base_lam2, diag = run_episode(env, net, agent, args, train=False, ep=0)
    dt = time.time() - t0

    # Report
    print(
        f"n={args.n} m={args.m} λ2={lam2:.6f} base={base_lam2:.6f} reward={reward:+.6f} steps={steps} time={dt:.2f}s"
    )
    # Optional: print simple regularity info
    deg = (env.adj.sum(axis=1)).round().astype(int)
    is_reg = (deg.min() == deg.max())
    if is_reg:
        print(f"regular graph with degree k={deg[0]}")


if __name__ == '__main__':
    main()
