#!/bin/bash
#SBATCH --job-name=v10-a
#SBATCH --partition=b40x4
#SBATCH --cpus-per-task=62
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=v10_sweep_a_%j.log

module load openblas/dynamic

cd ~/CRL

echo "========================================"
echo "v10 Sweep A: n=16,32,64,128"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo "========================================"

export OPENBLAS_NUM_THREADS=1

gcc -O3 -Wall -Wextra -std=c11 -D_GNU_SOURCE -fopenmp -lopenblas -lm -lpthread -o crl_v10_a v10_main.c crl.c 2>&1
echo "Build done"

stdbuf -oL ./crl_v10_a --n 16,32,64,128 --seeds 3 --passes 10 --greedy --threads 60 --baselines baselines.csv

echo ""
echo "========================================"
echo "Done: $(date)"
echo "========================================"
