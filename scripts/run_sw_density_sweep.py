#!/usr/bin/env python3
"""Run SW baseline point inference sweep over (n, density, sw_rho, seed).

For each n and density rho_d, m is computed as:
    m = (n - 1) + rho_d * ((n - 1) * (n - 2) / 2)

Then for each sw_rho in {0.25, 0.50, 0.75}, this script runs the C SW scorer,
optionally across multiple seeds and workers, and saves:
- raw rows: one row per (n, density, m, sw_rho, seed)
- mean rows: one row per (n, density, m, sw_rho), averaged over seeds
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "bin" / "sw_point"
SRC = ROOT / "src"


@dataclass(frozen=True)
class Task:
    n: int
    density: float
    m: int
    sw_rho: float
    seed: int


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def density_to_m(n: int, density: float) -> int:
    min_m = n - 1
    max_m = n * (n - 1) // 2
    span = max_m - min_m
    m = int(round(min_m + density * span))
    return max(min_m, min(max_m, m))


def build_binary(force: bool = False) -> None:
    if BINARY.exists() and not force:
        return

    BINARY.parent.mkdir(parents=True, exist_ok=True)

    cc = os.environ.get("CC", "cc")
    cmd = [
        cc,
        "-O3",
        "-Wall",
        "-Wextra",
        "-std=c11",
        "-I",
        str(SRC),
        "-o",
        str(BINARY),
        str(SRC / "sw_point.c"),
        str(SRC / "sw.c"),
        str(SRC / "common.c"),
    ]

    if platform.system() == "Darwin":
        cmd.extend(["-DACCELERATE_NEW_LAPACK", "-framework", "Accelerate", "-lm"])
    else:
        cmd.extend(["-llapack", "-lblas", "-lm"])

    subprocess.run(cmd, check=True)


def run_one(task: Task) -> Tuple[Task, float]:
    cmd = [
        str(BINARY),
        str(task.n),
        str(task.m),
        f"{task.sw_rho:.2f}",
        str(task.seed),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return task, float(out)


def group_mean(rows: Iterable[Tuple[Task, float]]) -> List[Tuple[int, float, int, float, float]]:
    # (n, density, m, sw_rho) -> (sum, count)
    agg = {}
    for task, score in rows:
        key = (task.n, task.density, task.m, task.sw_rho)
        total, cnt = agg.get(key, (0.0, 0))
        agg[key] = (total + score, cnt + 1)

    result = []
    for (n, density, m, sw_rho), (total, cnt) in sorted(agg.items()):
        result.append((n, density, m, sw_rho, total / cnt))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep SW scores over N, density, and SW rho.")
    parser.add_argument("--n-values", default="1024,2048,4096,8192,16384",
                        help="Comma-separated N values")
    parser.add_argument("--densities", default="0.3,0.5,0.7",
                        help="Comma-separated graph densities")
    parser.add_argument("--sw-rhos", default="0.25,0.5,0.75",
                        help="Comma-separated SW rewiring rho values")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of seeds per point")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="Parallel worker processes")
    parser.add_argument("--out", default=str(ROOT / "data" / "SW_density_sweep_raw.csv"),
                        help="Raw output CSV path")
    parser.add_argument("--out-mean", default=str(ROOT / "data" / "SW_density_sweep_mean.csv"),
                        help="Mean-over-seeds output CSV path")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild of bin/sw_point")
    args = parser.parse_args()

    n_values = parse_int_list(args.n_values)
    densities = parse_float_list(args.densities)
    sw_rhos = parse_float_list(args.sw_rhos)

    if args.seeds < 1:
        raise ValueError("--seeds must be >= 1")

    build_binary(force=args.rebuild)

    tasks: List[Task] = []
    for n in n_values:
        for density in densities:
            m = density_to_m(n, density)
            for sw_rho in sw_rhos:
                for seed in range(args.seeds):
                    tasks.append(Task(n=n, density=density, m=m, sw_rho=sw_rho, seed=seed))

    print(f"Running {len(tasks)} tasks with {args.workers} workers...")

    rows: List[Tuple[Task, float]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for idx, item in enumerate(ex.map(run_one, tasks), start=1):
            rows.append(item)
            if idx % max(1, len(tasks) // 20) == 0 or idx == len(tasks):
                print(f"  progress: {idx}/{len(tasks)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "density", "m", "sw_rho", "seed", "score"])
        for task, score in sorted(rows, key=lambda x: (x[0].n, x[0].density, x[0].sw_rho, x[0].seed)):
            w.writerow([task.n, f"{task.density:.6f}", task.m, f"{task.sw_rho:.2f}", task.seed, f"{score:.10f}"])

    mean_rows = group_mean(rows)
    out_mean = Path(args.out_mean)
    out_mean.parent.mkdir(parents=True, exist_ok=True)
    with out_mean.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "density", "m", "sw_rho", "mean_score"])
        for n, density, m, sw_rho, mean_score in mean_rows:
            w.writerow([n, f"{density:.6f}", m, f"{sw_rho:.2f}", f"{mean_score:.10f}"])

    print(f"Saved raw rows to: {out_path}")
    print(f"Saved mean rows to: {out_mean}")


if __name__ == "__main__":
    main()
