#!/usr/bin/env python3
"""Extract baselines from baselines_cache.json to simple CSV for C eval."""
import json
import sys
from pathlib import Path

cache_path = Path(__file__).parent.parent / "rl" / "baselines_cache.json"
if not cache_path.exists():
    print(f"Cache not found: {cache_path}")
    sys.exit(1)

with open(cache_path) as f:
    data = json.load(f)

out = Path(__file__).parent / "baselines.csv"
with open(out, 'w') as f:
    f.write("n,m,fv,er,sw025,sw050,sw075\n")
    for n_key in sorted(data.keys(), key=int):
        for m_key in sorted(data[n_key].keys(), key=int):
            entry = data[n_key][m_key]
            fv = entry.get('fv', 0) or 0
            er = entry.get('er', 0) or 0
            sw025 = entry.get('sw_025', 0) or 0
            sw050 = entry.get('sw_050', 0) or 0
            sw075 = entry.get('sw_075', 0) or 0
            f.write(f"{n_key},{m_key},{fv},{er},{sw025},{sw050},{sw075}\n")

print(f"Extracted to {out}")
