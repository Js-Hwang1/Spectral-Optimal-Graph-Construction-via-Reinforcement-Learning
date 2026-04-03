#!/bin/bash
#SBATCH --job-name=crl-eval-final
#SBATCH --partition=b40x4-long
#SBATCH --cpus-per-task=62
#SBATCH --ntasks=1
#SBATCH --mem=300G
#SBATCH --time=0-08:00:00
#SBATCH --output=crl_eval_final_%j.log

cd ~/CRL

echo "========================================"
echo "CRL Final Eval — ep 200000 (K=50)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo "========================================"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1

./crl_eval logs/crl_200k_k50/checkpoint_200000.bin \
    --n 16,24,36,64,128 \
    --trials 5 \
    --k-steps 50 \
    --swap-frac 0.05 \
    --baselines baselines.csv \
    --out logs/crl_200k_k50/eval_ep200000.csv

echo ""
echo "========================================"
echo "Done: $(date)"
echo "========================================"
