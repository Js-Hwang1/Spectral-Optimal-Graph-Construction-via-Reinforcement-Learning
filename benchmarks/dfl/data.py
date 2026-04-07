"""
CIFAR-10/100 data partitioning for decentralized federated learning.

Supports:
  - IID: Uniform random assignment of samples to nodes.
  - Non-IID (Dirichlet): Per-class Dirichlet allocation controlling heterogeneity.

Reference for Dirichlet partitioning:
    Hsu, Qi, Brown. "Measuring the Effects of Non-Identical Data Distribution
    for Federated Visual Classification", NeurIPS Workshop 2019.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, test_transform


def load_dataset(name="cifar100", data_dir="./data"):
    """Load CIFAR-10 or CIFAR-100. Returns (train_set, test_set, num_classes)."""
    train_transform, test_transform = get_transforms()
    if name == "cifar10":
        cls = datasets.CIFAR10
        num_classes = 10
    else:
        cls = datasets.CIFAR100
        num_classes = 100
    train_set = cls(data_dir, train=True, download=True, transform=train_transform)
    test_set = cls(data_dir, train=False, download=True, transform=test_transform)
    return train_set, test_set, num_classes


def partition_iid(dataset, n_nodes, seed=0):
    """Uniformly random partition of dataset into n_nodes shards."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(dataset))
    shards = np.array_split(indices, n_nodes)
    return [shard.tolist() for shard in shards]


def partition_dirichlet(dataset, n_nodes, alpha, seed=0):
    """
    Dirichlet non-IID partition.

    For each class c, draw p_c ~ Dir(alpha) over n_nodes, then assign
    each sample of class c to node i with probability p_c[i].

    Lower alpha = more heterogeneous (alpha=0.1: each node has 1-2 classes).
    Higher alpha = more uniform (alpha=100: approaches IID).
    """
    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)
    num_classes = len(set(targets.tolist()))

    # Group indices by class
    class_indices = [np.where(targets == c)[0] for c in range(num_classes)]

    # Allocate samples per class via Dirichlet
    node_indices = [[] for _ in range(n_nodes)]

    for c in range(num_classes):
        idx_c = class_indices[c]
        rng.shuffle(idx_c)

        # Draw Dirichlet proportions for this class
        proportions = rng.dirichlet(np.full(n_nodes, alpha))

        # Convert proportions to counts
        counts = (proportions * len(idx_c)).astype(int)

        # Fix rounding: give leftover to random nodes
        remainder = len(idx_c) - counts.sum()
        for _ in range(remainder):
            counts[rng.randint(n_nodes)] += 1

        # Assign
        start = 0
        for i in range(n_nodes):
            node_indices[i].extend(idx_c[start:start + counts[i]].tolist())
            start += counts[i]

    return node_indices


def create_data_loaders(dataset, partition_indices, batch_size=32):
    """Create one DataLoader per node from partition indices."""
    loaders = []
    for indices in partition_indices:
        subset = Subset(dataset, indices)
        bs = min(batch_size, max(len(indices), 1))
        loader = DataLoader(subset, batch_size=bs, shuffle=True,
                            drop_last=False, num_workers=0)
        loaders.append(loader)
    return loaders


def partition_stats(dataset, partition_indices):
    """Print per-node class distribution for debugging."""
    targets = np.array(dataset.targets)
    n_nodes = len(partition_indices)
    num_classes = len(set(targets.tolist()))

    print(f"{'Node':>6} {'Total':>6}", end="")
    for c in range(num_classes):
        print(f" {'c'+str(c):>5}", end="")
    print()

    for i in range(min(n_nodes, 10)):  # show first 10 nodes
        idx = partition_indices[i]
        t = targets[idx]
        counts = [np.sum(t == c) for c in range(num_classes)]
        print(f"{i:6d} {len(idx):6d}", end="")
        for c in counts:
            print(f" {c:5d}", end="")
        print()

    if n_nodes > 10:
        print(f"  ... ({n_nodes - 10} more nodes)")
