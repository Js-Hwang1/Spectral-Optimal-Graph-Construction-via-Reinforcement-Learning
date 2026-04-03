#!/bin/bash
#SBATCH --job-name=v10-n256
#SBATCH --partition=b40x4
#SBATCH --cpus-per-task=62
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=v10_n256_%j.log

module load openblas/dynamic

cd ~/CRL

echo "========================================"
echo "v10 Node-Centric Greedy"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo "========================================"

export OPENBLAS_NUM_THREADS=1

# Build
gcc -O3 -Wall -Wextra -std=c11 -D_GNU_SOURCE -fopenmp -lopenblas -lm -lpthread -o crl_v10 v10_main.c crl.c 2>&1
echo "Build done"

# Run n=256 with baselines, unbuffered output
stdbuf -oL ./crl_v10 --n 256 --seeds 3 --passes 10 --greedy --threads 60 --baselines baselines.csv

echo ""
echo "========================================"
echo "Done: $(date)"
echo "========================================"
