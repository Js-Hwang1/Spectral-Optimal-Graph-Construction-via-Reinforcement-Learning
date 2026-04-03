#!/usr/bin/env python3
"""HuggingFace Datasets baseline loader for λ₂ values.

Loads precomputed baseline λ₂ values from CSV files, pushes to
HuggingFace Hub as a Parquet dataset for shared access.

Dataset columns:
  n, m, fv, er, sw_025, sw_050, sw_075

Usage:
  python db_baselines.py load          # Load CSVs and push to HuggingFace Hub
  python db_baselines.py query 32 50   # Query baselines for n=32, m=50
  python db_baselines.py stats         # Show dataset stats
"""

import os
import sys
import csv
from pathlib import Path

import pandas as pd
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

DATA_DIR = Path(__file__).parent.parent / "data"

# CSV filename pattern -> column name
BASELINE_MAP = {
    "FV": "fv",
    "ER": "er",
    "SW_r25": "sw_025",
    "SW_r50": "sw_050",
    "SW_r75": "sw_075",
}

BASELINE_COLS = ["fv", "er", "sw_025", "sw_050", "sw_075"]


def _repo_id():
    """Resolve HuggingFace repo ID from authenticated user."""
    api = HfApi()
    username = api.whoami()["name"]
    return f"{username}/algebraic-connectivity-baselines"


def _load_from_hub():
    """Download dataset from HuggingFace Hub. Returns DataFrame or None."""
    try:
        ds = load_dataset(_repo_id(), split="train")
        return ds.to_pandas()
    except Exception:
        return None


def parse_csv(filepath):
    """Parse a baseline CSV file. Returns list of (m, score) tuples."""
    rows = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = int(row["m"])
            score = float(row["score"])
            rows.append((m, score))
    return rows


def load_all():
    """Load all CSV baseline data and push to HuggingFace Hub."""
    # Discover CSV files
    files_found = {}
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".csv"):
            continue
        stem = fname[:-4]
        for prefix, field in BASELINE_MAP.items():
            if stem.startswith(prefix + "_"):
                n_str = stem[len(prefix) + 1:]
                try:
                    n = int(n_str)
                except ValueError:
                    continue
                files_found[(field, n)] = DATA_DIR / fname
                break

    if not files_found:
        print("No CSV files found in data/")
        return

    # Build DataFrame from CSVs
    data = {}  # (n, m) -> {field: score}
    for (field, n), filepath in sorted(files_found.items()):
        rows = parse_csv(filepath)
        print(f"  Loaded {filepath.name}: {len(rows)} entries (n={n}, field={field})")
        for m, score in rows:
            key = (n, m)
            if key not in data:
                data[key] = {}
            data[key][field] = score

    records = []
    for (n, m), fields in sorted(data.items()):
        row = {"n": n, "m": m}
        row.update(fields)
        records.append(row)

    new_df = pd.DataFrame(records)

    # Merge with existing HF data (new CSV data takes priority)
    existing_df = _load_from_hub()
    if existing_df is not None and len(existing_df) > 0:
        print(f"\n  Existing HF dataset: {len(existing_df)} rows")
        # Set index to (n, m), combine_first: new overwrites, existing fills gaps
        new_df = new_df.set_index(["n", "m"])
        existing_df = existing_df.set_index(["n", "m"])
        merged = new_df.combine_first(existing_df).reset_index()
    else:
        merged = new_df

    # Ensure correct dtypes
    merged["n"] = merged["n"].astype(int)
    merged["m"] = merged["m"].astype(int)
    for col in BASELINE_COLS:
        if col in merged.columns:
            merged[col] = merged[col].astype(float)

    merged = merged.sort_values(["n", "m"]).reset_index(drop=True)

    # Push to HuggingFace Hub
    repo_id = _repo_id()
    print(f"\n  Pushing {len(merged)} rows to {repo_id}...")
    ds = Dataset.from_pandas(merged)
    ds.push_to_hub(repo_id)
    print(f"  Done: {len(merged)} rows pushed to {repo_id}")


def query(n, m):
    """Query baselines for a given (n, m)."""
    df = _load_from_hub()
    if df is None:
        print("Dataset not found on HuggingFace Hub")
        return None

    row = df[(df["n"] == n) & (df["m"] == m)]
    if row.empty:
        print(f"No data for n={n}, m={m}")
        return None

    doc = row.iloc[0].to_dict()
    print(f"Baselines for n={n}, m={m}:")
    for field in BASELINE_COLS:
        val = doc.get(field)
        if val is not None and pd.notna(val):
            print(f"  {field:>6s}: {val:.10f}")
        else:
            print(f"  {field:>6s}: (missing)")
    return {k: v for k, v in doc.items() if k not in ("n", "m")}


def stats():
    """Show dataset statistics."""
    df = _load_from_hub()
    if df is None:
        print("Dataset not found on HuggingFace Hub")
        return

    print(f"Total rows: {len(df)}")
    print(f"\n{'n':>4s} | {'count':>6s} | {'m_range':>14s} | {'FV':>5s} | {'ER':>5s} | {'SW25':>5s} | {'SW50':>5s} | {'SW75':>5s}")
    print("-" * 72)

    for n_val, group in df.groupby("n"):
        counts = {col: group[col].notna().sum() for col in BASELINE_COLS if col in group.columns}
        print(f"{n_val:4d} | {len(group):6d} | {group['m'].min():6d}-{group['m'].max():6d} | "
              f"{counts.get('fv', 0):5d} | {counts.get('er', 0):5d} | "
              f"{counts.get('sw_025', 0):5d} | {counts.get('sw_050', 0):5d} | {counts.get('sw_075', 0):5d}")


# --- Fast lookup API for use from other modules ---

_df_cache = None


def _get_df():
    """Lazy-load and cache the dataset."""
    global _df_cache
    if _df_cache is None:
        _df_cache = _load_from_hub()
    return _df_cache


def get_baselines(n, m):
    """Fast lookup: returns dict with fv, er, sw_025, sw_050, sw_075 or None."""
    df = _get_df()
    if df is None:
        return None
    row = df[(df["n"] == n) & (df["m"] == m)]
    if row.empty:
        return None
    doc = row.iloc[0].to_dict()
    return {k: v for k, v in doc.items() if k not in ("n", "m") and pd.notna(v)}


def get_baselines_batch(n, m_list):
    """Batch lookup: returns dict of {m: {field: score}} for a given n."""
    df = _get_df()
    if df is None:
        return {}
    subset = df[(df["n"] == n) & (df["m"].isin(m_list))]
    result = {}
    for _, row in subset.iterrows():
        m = int(row["m"])
        result[m] = {k: v for k, v in row.to_dict().items() if k not in ("n", "m") and pd.notna(v)}
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "load":
        load_all()
    elif cmd == "query":
        if len(sys.argv) < 4:
            print("Usage: python db_baselines.py query <n> <m>")
            sys.exit(1)
        query(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "stats":
        stats()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)
