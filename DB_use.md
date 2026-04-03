# HuggingFace Baselines Database

Dataset: [June30916/algebraic-connectivity-baselines](https://huggingface.co/datasets/June30916/algebraic-connectivity-baselines)

## Quick Start

```python
from database import get_baselines, get_baselines_batch

# Single lookup
b = get_baselines(32, 50)
# {'fv': 0.7067342125, 'er': 0.6514343427, 'sw_025': 0.269..., 'sw_050': 0.388..., 'sw_075': 0.504...}

# Batch lookup (all m values for a given n)
results = get_baselines_batch(32, [50, 100, 200])
# {50: {'fv': ..., 'er': ...}, 100: {...}, 200: {...}}
```

## CLI Commands

```bash
# Show stats (row counts per n, field coverage)
python3 database/db_baselines.py stats

# Query a specific (n, m)
python3 database/db_baselines.py query 32 50

# Load local CSVs from data/ and push to HuggingFace (merges with existing)
python3 database/db_baselines.py load
```

## Direct HuggingFace Access

```python
from datasets import load_dataset

ds = load_dataset("June30916/algebraic-connectivity-baselines", split="train")
df = ds.to_pandas()

# Filter
df[df["n"] == 64].head()
df[(df["n"] == 128) & (df["m"] == 500)]

# Columns: n, m, fv, er, sw_025, sw_050, sw_075
```

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `n` | int | Number of nodes |
| `m` | int | Number of edges |
| `fv` | float | Fiedler Vector baseline λ₂ |
| `er` | float | Effective Resistance baseline λ₂ |
| `sw_025` | float | Small World (ρ=0.25) baseline λ₂ |
| `sw_050` | float | Small World (ρ=0.50) baseline λ₂ |
| `sw_075` | float | Small World (ρ=0.75) baseline λ₂ |

## Dependencies

```bash
pip install datasets pandas huggingface_hub python-dotenv
```

Requires `HF_TOKEN` in `.env` (write token for `load`, read token or none for queries on public dataset).

## Notes

- Dataset is cached locally after first download (~103MB Parquet)
- `get_baselines()` caches the DataFrame in memory for fast repeated lookups
- `load` merges: new CSV data overwrites, existing HF data fills gaps
- The C benchmark (`./benchmark --db`) calls `load` automatically after completion
