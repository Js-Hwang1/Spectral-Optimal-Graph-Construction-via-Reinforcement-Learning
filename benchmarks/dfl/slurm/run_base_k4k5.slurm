#!/bin/bash
#SBATCH --job-name=dfl_base45
#SBATCH --partition=b40x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/dfl_base45_%j.log

# Base-(k+1) with k=4 and k=5. Same setup as all other runs.
# CIFAR-100, n=32, alpha=0.1, seed=0

module load slurm

WORKDIR=/lustre/nvwulf/scratch/jungshwang/Spectral-Optimal-Graph-Construction-via-Reinforcement-Learning
cd $WORKDIR

export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_PYTHONUNBUFFERED=1

mkdir -p logs benchmarks/dfl/results

echo "=== Base k=4 ==="
singularity exec --nv dfl.sif python benchmarks/dfl/train.py \
    --topo base \
    --n 32 \
    --d 4 \
    --alpha 0.1 \
    --seed 0 \
    --dataset cifar100 \
    --tau 5 \
    --rounds 2000 \
    --eval-freq 10 \
    --data-dir ${WORKDIR}/data \
    --output-dir benchmarks/dfl/results
echo "=== Done k=4 ==="

echo ""

echo "=== Base k=5 ==="
singularity exec --nv dfl.sif python benchmarks/dfl/train.py \
    --topo base \
    --n 32 \
    --d 5 \
    --alpha 0.1 \
    --seed 0 \
    --dataset cifar100 \
    --tau 5 \
    --rounds 2000 \
    --eval-freq 10 \
    --data-dir ${WORKDIR}/data \
    --output-dir benchmarks/dfl/results
echo "=== Done k=5 ==="
