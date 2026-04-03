#!/usr/bin/env python3
"""
Supervised Learning pre-training: imitate FV expert's edge rewiring decisions.

Trains the GNMPolicy with cross-entropy loss on two heads:
1. Neighbor+KEEP head: which neighbor to drop (or KEEP)
2. Destination head: where to connect (only when not KEEP)

The value head is not trained (no reward signal in SL).

Usage:
  python train_sl.py --data fv_expert_data.npz --epochs 30 --device cuda
  python train_sl.py --data fv_expert_data.npz --epochs 30 --device cpu --batch-size 512
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent))

from models.gnm_policy import GNMPolicy, GNMPolicyConfig
from envs.gnm_env import NODE_FEAT_DIM


class FVExpertDataset(Dataset):
    """Dataset of FV expert decision traces for imitation learning."""

    def __init__(self, features, nb_actions, dst_actions, nb_masks, dst_masks,
                 n_vals, max_n_policy):
        self.features = features
        self.nb_actions = nb_actions
        self.dst_actions = dst_actions
        self.nb_masks = nb_masks
        self.dst_masks = dst_masks
        self.n_vals = n_vals
        self.max_n_policy = max_n_policy
        self.data_max_n = features.shape[1]
        self.is_keep = nb_actions >= n_vals

    def __len__(self):
        return len(self.nb_actions)

    def __getitem__(self, idx):
        feat = self.features[idx]
        n = int(self.n_vals[idx])

        if self.data_max_n < self.max_n_policy:
            pad_n = self.max_n_policy - self.data_max_n
            feat_padded = np.zeros((self.max_n_policy, NODE_FEAT_DIM), dtype=np.float32)
            feat_padded[:self.data_max_n] = feat
        else:
            feat_padded = feat[:self.max_n_policy]

        nb_mask = np.zeros(self.max_n_policy, dtype=np.bool_)
        nb_mask[:min(self.data_max_n, self.max_n_policy)] = \
            self.nb_masks[idx][:min(self.data_max_n, self.max_n_policy)]

        dst_mask = np.zeros(self.max_n_policy, dtype=np.bool_)
        dst_mask[:min(self.data_max_n, self.max_n_policy)] = \
            self.dst_masks[idx][:min(self.data_max_n, self.max_n_policy)]

        node_mask = np.zeros(self.max_n_policy, dtype=np.bool_)
        node_mask[:n] = True

        u_idx = int(feat_padded[:, 3].argmax())

        if self.is_keep[idx]:
            nb_action = self.max_n_policy
        else:
            nb_action = int(self.nb_actions[idx])

        dst_action = int(self.dst_actions[idx])
        is_rewire = not self.is_keep[idx]

        return (
            torch.from_numpy(feat_padded),
            torch.tensor(u_idx, dtype=torch.long),
            torch.from_numpy(nb_mask),
            torch.from_numpy(dst_mask),
            torch.from_numpy(node_mask),
            torch.tensor(nb_action, dtype=torch.long),
            torch.tensor(dst_action, dtype=torch.long),
            torch.tensor(is_rewire, dtype=torch.bool),
        )


def evaluate(policy, loader, device):
    """Run validation pass. Returns dict of metrics."""
    policy.eval()
    loss_sum = 0.0
    nb_correct = nb_total = 0
    dst_correct = dst_total = 0
    keep_correct = keep_total = 0
    rewire_correct = rewire_total = 0
    batches = 0

    with torch.no_grad():
        for batch in loader:
            feat, u_idx, nb_mask, dst_mask, node_mask, nb_action, dst_action, is_rewire = \
                [b.to(device) for b in batch]

            nk_logits, dest_logits, _ = policy.forward_node(
                feat, u_idx, nb_mask, dst_mask, node_mask=node_mask
            )

            nb_loss = F.cross_entropy(nk_logits, nb_action)
            if is_rewire.any():
                dst_loss = F.cross_entropy(
                    dest_logits[is_rewire], dst_action[is_rewire]
                )
            else:
                dst_loss = torch.tensor(0.0, device=device)

            loss_sum += (nb_loss + dst_loss).item()

            nb_pred = nk_logits.argmax(dim=-1)
            nb_correct += (nb_pred == nb_action).sum().item()
            nb_total += len(nb_action)

            if is_rewire.any():
                dst_pred = dest_logits[is_rewire].argmax(dim=-1)
                dst_correct += (dst_pred == dst_action[is_rewire]).sum().item()
                dst_total += is_rewire.sum().item()

            is_keep = ~is_rewire
            if is_keep.any():
                keep_pred_correct = (nb_pred[is_keep] == nb_action[is_keep])
                keep_correct += keep_pred_correct.sum().item()
                keep_total += is_keep.sum().item()
            if is_rewire.any():
                rew_pred_correct = (nb_pred[is_rewire] == nb_action[is_rewire])
                rewire_correct += rew_pred_correct.sum().item()
                rewire_total += is_rewire.sum().item()

            batches += 1

    return {
        'loss': loss_sum / max(batches, 1),
        'nb_acc': nb_correct / max(nb_total, 1),
        'dst_acc': dst_correct / max(dst_total, 1),
        'keep_acc': keep_correct / max(keep_total, 1),
        'rewire_acc': rewire_correct / max(rewire_total, 1),
    }


def train_sl(
    data_path: str,
    epochs: int = 30,
    batch_size: int = 2048,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    val_fraction: float = 0.1,
    seed: int = 42,
    max_n: int = 32,
    rewire_weight: float = 5.0,
    num_workers: int = 4,
    max_samples: int = 0,
):
    """Train policy via supervised imitation of FV expert."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / f"sl_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    features = data['features']
    nb_actions = data['nb_actions']
    dst_actions = data['dst_actions']
    nb_masks = data['nb_masks']
    dst_masks = data['dst_masks']
    n_vals = data['n_vals']

    total = len(nb_actions)
    is_keep = nb_actions >= n_vals
    num_rewires = int((~is_keep).sum())

    if max_samples > 0 and total > max_samples:
        rng = np.random.RandomState(seed)
        n_keep = int(max_samples * (1 - num_rewires / total))
        n_rewire = max_samples - n_keep
        keep_idx = rng.permutation(np.where(is_keep)[0])[:n_keep]
        rewire_idx = rng.permutation(np.where(~is_keep)[0])[:n_rewire]
        indices = np.concatenate([keep_idx, rewire_idx])
        rng.shuffle(indices)
        features = features[indices]
        nb_actions = nb_actions[indices]
        dst_actions = dst_actions[indices]
        nb_masks = nb_masks[indices]
        dst_masks = dst_masks[indices]
        n_vals = n_vals[indices]
        total = len(nb_actions)
        is_keep = nb_actions >= n_vals
        num_rewires = int((~is_keep).sum())
        print(f"  Subsampled to {total:,} samples (max_samples={max_samples:,})")
    num_keeps = total - num_rewires
    data_max_n = features.shape[1]

    if max_n < data_max_n:
        print(f"WARNING: max_n ({max_n}) < data max_n ({data_max_n}), "
              f"clamping to {data_max_n}")
        max_n = data_max_n

    print(f"  Total samples:  {total:,}")
    print(f"  Rewires:        {num_rewires:,} ({100*num_rewires/total:.1f}%)")
    print(f"  Keeps:          {num_keeps:,} ({100*num_keeps/total:.1f}%)")
    print(f"  Data max_n:     {data_max_n}")
    print(f"  Policy max_n:   {max_n}")
    print(f"  n range:        [{n_vals.min()}, {n_vals.max()}]")

    # Train/val split
    indices = np.random.permutation(total)
    val_size = int(total * val_fraction)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_dataset = FVExpertDataset(
        features[train_idx], nb_actions[train_idx], dst_actions[train_idx],
        nb_masks[train_idx], dst_masks[train_idx], n_vals[train_idx], max_n
    )
    val_dataset = FVExpertDataset(
        features[val_idx], nb_actions[val_idx], dst_actions[val_idx],
        nb_masks[val_idx], dst_masks[val_idx], n_vals[val_idx], max_n
    )

    # Shuffle (WeightedRandomSampler hits 2^24 limit for large datasets)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    policy_config = GNMPolicyConfig(hidden_dim=128, node_feat_dim=NODE_FEAT_DIM)
    policy = GNMPolicy(policy_config).to(device)

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 100
    )

    param_count = sum(p.numel() for p in policy.parameters())
    print(f"\n{'='*80}")
    print("SL Pre-training: Imitate FV Expert Edge Rewiring")
    print(f"{'='*80}")
    print(f"  Policy params:   {param_count:,}")
    print(f"  Train samples:   {len(train_idx):,}")
    print(f"  Val samples:     {len(val_idx):,}")
    print(f"  Batch size:      {batch_size}")
    print(f"  Epochs:          {epochs}")
    print(f"  LR:              {lr} → {lr/100:.2e} (cosine)")
    print(f"  Rewire weight:   {rewire_weight}x (weighted loss)")
    print(f"  Device:          {device}")
    print(f"  Output:          {log_dir}")
    print(f"{'='*80}")
    sys.stdout.flush()

    best_val_loss = float('inf')
    metrics_history = []

    for epoch in range(epochs):
        policy.train()
        train_loss_sum = 0.0
        train_nb_correct = train_nb_total = 0
        train_dst_correct = train_dst_total = 0
        train_batches = 0

        for batch in train_loader:
            feat, u_idx, nb_mask, dst_mask, node_mask, nb_action, dst_action, is_rewire = \
                [b.to(device) for b in batch]

            nk_logits, dest_logits, _ = policy.forward_node(
                feat, u_idx, nb_mask, dst_mask, node_mask=node_mask
            )

            nb_loss_per = F.cross_entropy(nk_logits, nb_action, reduction='none')
            sample_w = torch.where(is_rewire, rewire_weight, 1.0).float()
            nb_loss = (nb_loss_per * sample_w).mean()
            if is_rewire.any():
                dst_loss = F.cross_entropy(
                    dest_logits[is_rewire], dst_action[is_rewire]
                )
            else:
                dst_loss = torch.tensor(0.0, device=device)

            loss = nb_loss + dst_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_nb_correct += (nk_logits.argmax(dim=-1) == nb_action).sum().item()
            train_nb_total += len(nb_action)
            if is_rewire.any():
                train_dst_correct += (
                    dest_logits[is_rewire].argmax(dim=-1) == dst_action[is_rewire]
                ).sum().item()
                train_dst_total += is_rewire.sum().item()
            train_batches += 1

        scheduler.step()

        train_loss = train_loss_sum / max(train_batches, 1)
        train_nb_acc = train_nb_correct / max(train_nb_total, 1)
        train_dst_acc = train_dst_correct / max(train_dst_total, 1)

        val_metrics = evaluate(policy, val_loader, device)

        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_nb_acc': train_nb_acc,
            'train_dst_acc': train_dst_acc,
            'val_loss': val_metrics['loss'],
            'val_nb_acc': val_metrics['nb_acc'],
            'val_dst_acc': val_metrics['dst_acc'],
            'val_keep_acc': val_metrics['keep_acc'],
            'val_rewire_acc': val_metrics['rewire_acc'],
            'lr': optimizer.param_groups[0]['lr'],
        }
        metrics_history.append(metrics)

        marker = ""
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(policy.state_dict(), log_dir / "best_policy.pt")
            marker = " << BEST"

        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train L={train_loss:.4f} nb={train_nb_acc:.3f} dst={train_dst_acc:.3f} | "
            f"Val L={val_metrics['loss']:.4f} nb={val_metrics['nb_acc']:.3f} "
            f"dst={val_metrics['dst_acc']:.3f} "
            f"keep={val_metrics['keep_acc']:.3f} rew={val_metrics['rewire_acc']:.3f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}{marker}",
            flush=True,
        )

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
            }, log_dir / f"checkpoint_{epoch+1}.pt")

    torch.save(policy.state_dict(), log_dir / "final_policy.pt")
    with open(log_dir / "metrics.json", 'w') as f:
        json.dump(metrics_history, f, indent=2)

    best_nb = max(m['val_nb_acc'] for m in metrics_history)
    best_dst = max(m['val_dst_acc'] for m in metrics_history)
    best_rew = max(m['val_rewire_acc'] for m in metrics_history)

    print(f"\n{'='*80}")
    print("SL Pre-training Complete!")
    print(f"  Best val loss:       {best_val_loss:.4f}")
    print(f"  Best val nb_acc:     {best_nb:.3f}")
    print(f"  Best val dst_acc:    {best_dst:.3f}")
    print(f"  Best val rewire_acc: {best_rew:.3f}")
    print(f"  Saved to: {log_dir}")
    print(f"{'='*80}")

    return log_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SL pre-training on FV expert data"
    )
    parser.add_argument("--data", type=str, required=True,
                        help="Path to .npz expert data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-n", type=int, default=32,
                        help="Max n for padding (>= data max_n)")
    parser.add_argument("--rewire-weight", type=float, default=5.0,
                        help="Oversampling weight for rewire vs keep samples")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (use 0 for very large data to avoid OOM)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples for training (0=all). Use 10M-15M for 48M+ datasets to avoid OOM.")
    args = parser.parse_args()

    train_sl(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_n=args.max_n,
        rewire_weight=args.rewire_weight,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
