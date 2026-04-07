#!/bin/bash
#SBATCH --job-name=dfl-multi
#SBATCH --partition=b40x4
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/dfl_multi_%j.log

# Runs multiple topologies in parallel on one node (1 GPU per topo).
#
# Usage:
#   sbatch run_multi.slurm 64 128 0.1 0 cifar100
#   args: <n> <m> <alpha> <seed> [dataset]

N=${1:?Usage: sbatch run_multi.slurm <n> <m> <alpha> <seed> [dataset]}
M=${2:?}
ALPHA=${3:?}
SEED=${4:?}
DATASET=${5:-cifar100}

module load slurm
cd /lustre/nvwulf/scratch/jungshwang/Spectral-Optimal-Graph-Construction-via-Reinforcement-Learning

export SINGULARITYENV_PYTHONNOUSERSITE=1
mkdir -p logs benchmarks/dfl/results

DATA_DIR=/lustre/nvwulf/scratch/jungshwang/Spectral-Optimal-Graph-Construction-via-Reinforcement-Learning/data
OUT_DIR=benchmarks/dfl/results

run_topo() {
    local GPU=$1
    local TOPO=$2
    CUDA_VISIBLE_DEVICES=$GPU singularity exec --nv dfl.sif python benchmarks/dfl/train.py \
        --topo $TOPO --n $N --m $M --alpha $ALPHA --seed $SEED \
        --dataset $DATASET --tau 5 --rounds 2000 --eval-freq 10 \
        --data-dir $DATA_DIR --output-dir $OUT_DIR \
        2>&1 | sed "s/^/[GPU$GPU $TOPO] /" &
}

run_topo 0 er
run_topo 1 fv
run_topo 2 rd
wait

echo ""
echo "All topologies complete for $DATASET n=$N m=$M alpha=$ALPHA seed=$SEED"
