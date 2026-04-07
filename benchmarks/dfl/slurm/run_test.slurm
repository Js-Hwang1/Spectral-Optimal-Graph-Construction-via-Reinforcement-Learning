#!/bin/bash
#SBATCH --job-name=dfl_test
#SBATCH --partition=b40x4-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/dfl_test.log

# Quick test: run all 6 topologies on CIFAR-100, n=32, d=4, alpha=0.1, seed=0
# ~30min per topology, ~3 hours total

module load slurm

WORKDIR=/lustre/nvwulf/scratch/jungshwang/Spectral-Optimal-Graph-Construction-via-Reinforcement-Learning
cd $WORKDIR

export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_PYTHONUNBUFFERED=1

mkdir -p logs benchmarks/dfl/results

for topo in ring torus expander random qrsdr base; do
  echo "=== Running $topo ==="
  singularity exec --nv dfl.sif python benchmarks/dfl/train.py \
      --topo $topo \
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
  echo "=== Done $topo ==="
  echo ""
done
