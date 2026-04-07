#!/bin/bash
#SBATCH --job-name=dfl
#SBATCH --partition=b40x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/dfl_%j.log

# Usage:
#   sbatch run_single.slurm er 32 64 0.1 0 cifar100
#   args: <topo> <n> <m> <alpha> <seed> [dataset]

TOPO=${1:?Usage: sbatch run_single.slurm <topo> <n> <m> <alpha> <seed> [dataset]}
N=${2:?}
M=${3:?}
ALPHA=${4:?}
SEED=${5:?}
DATASET=${6:-cifar100}

module load slurm
cd /lustre/nvwulf/scratch/jungshwang/Spectral-Optimal-Graph-Construction-via-Reinforcement-Learning

export SINGULARITYENV_PYTHONNOUSERSITE=1

mkdir -p logs benchmarks/dfl/results

singularity exec --nv dfl.sif python benchmarks/dfl/train.py \
    --topo $TOPO \
    --n $N \
    --m $M \
    --alpha $ALPHA \
    --seed $SEED \
    --dataset $DATASET \
    --tau 5 \
    --rounds 2000 \
    --eval-freq 10 \
    --data-dir /lustre/nvwulf/scratch/jungshwang/Spectral-Optimal-Graph-Construction-via-Reinforcement-Learning/data \
    --output-dir benchmarks/dfl/results

