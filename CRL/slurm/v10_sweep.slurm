#!/bin/bash
#SBATCH --job-name=v10-sweep
#SBATCH --partition=b40x4
#SBATCH --nodes=2
#SBATCH --cpus-per-task=62
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=v10_sweep_%j.log

module load openblas/dynamic

cd ~/CRL

echo "========================================"
echo "v10 Node-Centric Greedy — Full Sweep"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "CPUs/node: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo "========================================"

export OPENBLAS_NUM_THREADS=1

# Build
gcc -O3 -Wall -Wextra -std=c11 -D_GNU_SOURCE -fopenmp -lopenblas -lm -lpthread -o crl_v10 v10_main.c crl.c 2>&1
echo "Build done"

# Node 1: n=16,32,64,128 (smaller, faster)
srun --nodes=1 --ntasks=1 --exclusive \
    stdbuf -oL ./crl_v10 --n 16,32,64,128 --seeds 3 --passes 10 --greedy --threads 60 --baselines baselines.csv &

# Node 2: n=256 (largest, slowest)
srun --nodes=1 --ntasks=1 --exclusive \
    stdbuf -oL ./crl_v10 --n 256 --seeds 3 --passes 10 --greedy --threads 60 --baselines baselines.csv &

wait

echo ""
echo "========================================"
echo "Done: $(date)"
echo "========================================"
