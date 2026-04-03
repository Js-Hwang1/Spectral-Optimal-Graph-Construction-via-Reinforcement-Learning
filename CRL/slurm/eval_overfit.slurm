#!/bin/bash
#SBATCH --job-name=crl-overfit
#SBATCH --partition=b40x4-long
#SBATCH --cpus-per-task=62
#SBATCH --ntasks=1
#SBATCH --mem=300G
#SBATCH --time=0-01:00:00
#SBATCH --output=crl_eval_overfit_%j.log

cd ~/CRL

echo "========================================"
echo "CRL Overfitting Check — 3 checkpoints"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo "========================================"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1

NVALS="16,24,36"
TRIALS=5

# === 1. Early checkpoint (ep 8000) ===
echo ""
echo "=== Checkpoint: ep 8000 (early) ==="
./crl_eval logs/crl_200k_k50/checkpoint_8000.bin \
    --n $NVALS --trials $TRIALS --k-steps 50 --swap-frac 0.05 \
    --baselines baselines.csv \
    --out logs/crl_200k_k50/eval_ep8000.csv

# === 2. Best checkpoint (ep 72016) ===
echo ""
echo "=== Checkpoint: best (ep 72016) ==="
./crl_eval logs/crl_200k_k50/best.bin \
    --n $NVALS --trials $TRIALS --k-steps 50 --swap-frac 0.05 \
    --baselines baselines.csv \
    --out logs/crl_200k_k50/eval_best.csv

# === 3. Latest checkpoint (ep 152000) ===
echo ""
echo "=== Checkpoint: ep 152000 (latest) ==="
./crl_eval logs/crl_200k_k50/checkpoint_152000.bin \
    --n $NVALS --trials $TRIALS --k-steps 50 --swap-frac 0.05 \
    --baselines baselines.csv \
    --out logs/crl_200k_k50/eval_ep152000.csv

echo ""
echo "========================================"
echo "Done: $(date)"
echo "========================================"
