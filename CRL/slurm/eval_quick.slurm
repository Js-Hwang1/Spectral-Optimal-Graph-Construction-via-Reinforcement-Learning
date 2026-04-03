#!/bin/bash
#SBATCH --job-name=crl-eval-quick
#SBATCH --partition=b40x4-long
#SBATCH --cpus-per-task=62
#SBATCH --ntasks=1
#SBATCH --mem=300G
#SBATCH --time=0-01:00:00
#SBATCH --output=crl_eval_quick_%j.log

cd ~/CRL

echo "========================================"
echo "CRL Quick Eval — K=50 new best (ep 72400)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo "========================================"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1

CKPT=logs/crl_200k_k50/best.bin
OUT=logs/crl_200k_k50/eval_quick_best.csv

./crl_eval $CKPT \
    --n 16,24,36 \
    --trials 5 \
    --k-steps 50 \
    --swap-frac 0.05 \
    --baselines baselines.csv \
    --out $OUT

echo ""
echo "========================================"
echo "Done: $(date)"
echo "========================================"
