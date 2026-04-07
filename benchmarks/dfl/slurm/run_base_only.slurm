#!/bin/bash
#SBATCH --job-name=dfl_base
#SBATCH --partition=b40x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/dfl_base_%j.log

# Run ONLY Base-(k+1) with the REAL implementation.
# CIFAR-100, n=32, d=1 (one-peer), alpha=0.1, seed=0

module load slurm

WORKDIR=/lustre/nvwulf/scratch/jungshwang/Spectral-Optimal-Graph-Construction-via-Reinforcement-Learning
cd $WORKDIR

export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_PYTHONUNBUFFERED=1

mkdir -p logs benchmarks/dfl/results

echo "=== Base-(k+1) ONLY ==="

singularity exec --nv dfl.sif python benchmarks/dfl/train.py \
    --topo base \
    --n 32 \
    --d 1 \
    --alpha 0.1 \
    --seed 0 \
    --dataset cifar100 \
    --tau 5 \
    --rounds 2000 \
    --eval-freq 10 \
    --data-dir ${WORKDIR}/data \
    --output-dir benchmarks/dfl/results

echo "=== Done ==="
