#!/bin/bash
#SBATCH --job-name=crl-eval-c
#SBATCH --partition=b40x4-long
#SBATCH --cpus-per-task=62
#SBATCH --ntasks=1
#SBATCH --mem=300G
#SBATCH --time=0-08:00:00
#SBATCH --output=crl_eval_%j.log

cd ~/CRL

echo "========================================"
echo "CRL C-Based Eval (BLAS-batched)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo "========================================"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1

N_VALS="16,24,36,64,128"
TRIALS=5
K_STEPS=20
SWAP_FRAC=0.05

# === Checkpoint 1: K=20 best (frac=0.05, 50k training) ===
CKPT1=logs/crl_50k_fast/checkpoint_40000.bin
OUT1=logs/crl_50k_fast/eval_c_hpc.csv

echo ""
echo "=== Eval: K=20 best checkpoint ==="
echo "Checkpoint: $CKPT1"
echo "N values: $N_VALS | Trials: $TRIALS | K=$K_STEPS | frac=$SWAP_FRAC"
echo ""

./crl_eval $CKPT1 \
    --n $N_VALS \
    --trials $TRIALS \
    --k-steps $K_STEPS \
    --swap-frac $SWAP_FRAC \
    --baselines baselines.csv \
    --out $OUT1

echo ""
echo "=== Eval: K=50 best checkpoint ==="

# === Checkpoint 2: K=50 best (frac=0.05, 200k training) ===
CKPT2=logs/crl_200k_k50/best.bin
OUT2=logs/crl_200k_k50/eval_c_hpc.csv

echo "Checkpoint: $CKPT2"
echo "N values: $N_VALS | Trials: $TRIALS | K=$K_STEPS | frac=$SWAP_FRAC"
echo ""

./crl_eval $CKPT2 \
    --n $N_VALS \
    --trials $TRIALS \
    --k-steps $K_STEPS \
    --swap-frac $SWAP_FRAC \
    --baselines baselines.csv \
    --out $OUT2

echo ""
echo "========================================"
echo "Done: $(date)"
echo "========================================"
