#!/bin/bash
#SBATCH --job-name=verify_ram
#SBATCH --partition=b40x4-long
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/verify_ram_%j.log

# Verify: random init + degree regularization → near-Ramanujan
# at scale n = 2^5 to 2^22 (4M nodes), d=4

module load slurm

WORKDIR=/lustre/nvwulf/scratch/jungshwang/Spectral-Optimal-Graph-Construction-via-Reinforcement-Learning
cd $WORKDIR

mkdir -p logs PoC_NEW

# Build
gcc -O3 -march=native -o PoC_NEW/verify_large PoC_NEW/verify_large.c -lm

# Run sweep
echo "=== Large-scale Ramanujan verification ==="
echo "Random init (m=nd/2) + degree regularization → d-regular"
echo ""
PoC_NEW/verify_large --sweep
