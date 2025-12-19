# Two-Phase Graph Construction for Maximum Algebraic Connectivity

A research-grade implementation for constructing graphs that maximize algebraic connectivity (lambda_2, the second smallest eigenvalue of the graph Laplacian).

## Overview

This project implements a two-phase approach:

1. **Phase 1 (Deterministic Skeleton)**: Fast O(N^2) construction of a base graph using Star-Mesh or Bipartite-Mesh patterns
2. **Phase 2 (RL Residual Filling)**: Learned policy places remaining edges to maximize lambda_2

The key innovation is that Phase 2 inference operates in O(N^2) time without computing eigenvalues, using a Graph Attention Network that learns structural patterns correlated with algebraic connectivity.

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install numpy scipy torch networkx
```

## Quick Start

```bash
# Run Phase 1 benchmarks
python phase1.py

# Test Phase 2 with spectral heuristic
python phase2.py

# Build a graph (uses heuristic without trained model)
python main.py build --n 50 --m 200

# Compare methods
python main.py compare --n 50 --m 200
```

## Training the RL Agent

### Quick Training (for testing)

```bash
python phase2_train.py quick --episodes 500 --save-dir ./quick_checkpoints
```

### Standard Training

```bash
python phase2_train.py train \
    --num-episodes 20000 \
    --min-n 16 \
    --max-n 64 \
    --hidden-dim 128 \
    --save-dir ./checkpoints \
    --log-dir ./logs
```

**Recommended hyperparameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--num-episodes` | 20000 | Total training episodes |
| `--min-n` | 16 | Minimum graph size |
| `--max-n` | 64 | Maximum graph size |
| `--hidden-dim` | 128 | GNN hidden dimension |
| `--num-gat-layers` | 3 | Number of GAT layers |
| `--lr` | 3e-4 | Learning rate |
| `--batch-size` | 32 | Batch size |

### Extended Training (for production)

```bash
python phase2_train.py train \
    --num-episodes 50000 \
    --min-n 10 \
    --max-n 100 \
    --hidden-dim 256 \
    --num-gat-layers 4 \
    --lr 1e-4 \
    --batch-size 64 \
    --save-dir ./production_checkpoints
```

### Monitoring Training

Training logs are saved to `./logs/train.log`. Key metrics to watch:

- **Reward**: Average episode reward (should increase)
- **L2**: Average final lambda_2 achieved (should increase)
- **Val L2**: Validation lambda_2 (most important for generalization)

The best model is automatically saved when validation lambda_2 improves.

### Evaluation

After training:

```bash
python phase2_train.py eval --model ./checkpoints/best_model.pt
```

This compares the trained model against:
- Phase 1 only (deterministic skeleton)
- Spectral heuristic (no learning)

## Architecture Details

### Feature Extraction

Node features (8 dimensions):
1. Normalized degree
2. Clustering coefficient
3. Neighbor average degree (2-hop connectivity)
4. Neighbor degree variance (local regularity)
5. Eigenvector centrality (power iteration approximation)
6. Local edge density
7. Eccentricity proxy
8. PageRank-like centrality

Edge features (10 dimensions):
1. Edge exists indicator
2. Degree product (hub detection)
3. Degree sum
4. Common neighbors count
5. Jaccard similarity
6. Adamic-Adar index
7. Node feature similarity
8. Degree difference
9. Resource allocation index
10. Preferential attachment score

### Graph Attention Network

- Multi-head attention (4 heads by default)
- Residual connections with LayerNorm
- Size-invariant through mean pooling
- Global features for conditioning

### PPO Implementation

- Clipped objective for policy and value
- Entropy bonus with decay
- GAE for advantage estimation
- Size-stratified batching for variable graphs

## Usage Examples

### Building Graphs

```python
from phase2 import build_graph_with_rl

# With trained model
adj = build_graph_with_rl(
    n=50,
    m=200,
    model_path="./checkpoints/best_model.pt",
    skeleton_budget_ratio=0.7
)

# Without model (uses spectral heuristic)
adj = build_graph_with_rl(n=50, m=200)
```

### Custom Inference

```python
from phase1 import AdaptiveGraphBuilder
from phase2 import ResidualEdgeFiller

# Phase 1: Build skeleton
builder = AdaptiveGraphBuilder(n=50)
builder.build(m=140)  # 70% of 200
skeleton = builder.adj.astype(float)

# Phase 2: Fill remaining edges
filler = ResidualEdgeFiller(model_path="./checkpoints/best_model.pt")
final_adj = filler.fill_edges(skeleton, num_edges_to_add=60)
```

## Theoretical Background

### Why Two Phases?

1. **Phase 1 Efficiency**: Deterministic construction is fast and provides good baseline
2. **Phase 2 Refinement**: RL learns subtle patterns that deterministic methods miss
3. **Complexity Constraint**: Both phases maintain O(N^2) complexity

### Algebraic Connectivity

Lambda_2 (algebraic connectivity) measures:
- **Graph connectivity**: Larger values = more connected
- **Mixing time**: Faster information spread
- **Robustness**: Harder to disconnect with edge removal

The Fiedler vector (eigenvector for lambda_2) identifies bottleneck nodes. Our features approximate this vector without O(N^3) eigenvalue computation.

### Feature Selection Rationale

Features correlate with lambda_2 through:
- **Cheeger inequality**: Links lambda_2 to graph expansion
- **Degree distribution**: Regular graphs tend to have higher lambda_2
- **Clustering**: Local structure affects global connectivity
- **Centrality**: Important nodes for connectivity

## Performance Expectations

With proper training, expect:
- **Small graphs (n<30)**: ~10-20% improvement over Phase 1 alone
- **Medium graphs (30<n<60)**: ~5-15% improvement
- **Large graphs (n>60)**: ~3-10% improvement (harder to generalize)

Note: Phase 1 is already quite good, so improvements may be modest but consistent.

## Troubleshooting

### Training doesn't converge
- Reduce learning rate (`--lr 1e-4`)
- Increase hidden dimension (`--hidden-dim 256`)

### Out of memory
- Reduce batch size (`--batch-size 16`)
- Reduce max graph size (`--max-n 40`)
- Use CPU (`--device cpu`)

### Poor generalization
- Train longer (`--num-episodes 50000`)
- Use wider size range during training
- Check validation metrics, not just training reward

## File Structure

```
.
├── CLAUDE.md           # Project instructions
├── README.md           # This file
├── phase1.py           # Deterministic skeleton builders
├── phase2.py           # RL inference module (GNN + policy)
├── phase2_train.py     # RL training pipeline
├── main.py             # CLI interface
├── checkpoints/        # Saved models
└── logs/               # Training logs
```

## License

MIT License
