# Density (ρ) Reference Table

Formula: $m = (n-1) + \rho \cdot \frac{(n-1)(n-2)}{2}$

Where:
- $\rho = \frac{m - (n-1)}{n(n-1)/2 - (n-1)}$ (normalized edge ratio)
- Range: ρ ∈ [0, 1] (0 = tree, 1 = complete graph)

## m values for target densities

| n | 2^x | ρ=0.3 | ρ=0.5 | ρ=0.7 |
|---|-----|-------|-------|-------|
| 1,024 | 2^10 | 157,765 | 262,380 | 366,994 |
| 2,048 | 2^11 | 630,336 | 1,049,088 | 1,467,839 |
| 4,096 | 2^12 | 2,519,773 | 4,194,553 | 5,869,331 |
| 8,192 | 2^13 | 10,080,495 | 16,777,216 | 23,474,343 |
| 16,384 | 2^14 | 40,239,267 | 67,092,865 | 93,945,441 |

## Interpretation

- **ρ=0.3**: Sparse (30% of capacity above minimum spanning tree)
- **ρ=0.5**: Mid-range (50% of capacity)
- **ρ=0.7**: Dense (70% of capacity)
