#!/bin/bash
#SBATCH --job-name=dfl_ours
#SBATCH --partition=b40x4-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/dfl_ours_%j.log

# ============================================================
# Sweep OURS: all (n, d, alpha, seed) for the paper.
# Single job, sequential runs. Skips existing results.
#
# 3 n x 3 d x 4 alpha x 3 seeds = 108 runs
#
# Usage:
#   sbatch slurm/sweep_ours.slurm
# ============================================================

module load slurm

WORKDIR=/lustre/nvwulf/scratch/jungshwang/Spectral-Optimal-Graph-Construction-via-Reinforcement-Learning
cd $WORKDIR

export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_PYTHONUNBUFFERED=1

mkdir -p logs benchmarks/dfl/results

CONTAINER=dfl.sif
TRAIN_SCRIPT=benchmarks/dfl/train.py
DATA_DIR=${WORKDIR}/data
OUT_DIR=benchmarks/dfl/results

DONE=0
SKIPPED=0
TOTAL=108

echo "============================================================"
echo " Sweep OURS — CIFAR-100"
echo " n={32,64,128} x d={4,5,6} x alpha={0.1,0.5,1.0,iid} x seed={0,1,2}"
echo " $TOTAL total runs"
echo "============================================================"
echo ""

for N in 32 64 128; do
  for D in 4 5 6; do
    for ALPHA in 0.1 0.5 1.0 iid; do
      for SEED in 0 1 2; do
        DONE=$((DONE + 1))

        if [ "$ALPHA" = "iid" ]; then
          RUN_NAME="cifar100_ours_n${N}_d${D}_iid_s${SEED}"
          ALPHA_FLAG=""
        else
          RUN_NAME="cifar100_ours_n${N}_d${D}_dir${ALPHA}_s${SEED}"
          ALPHA_FLAG="--alpha $ALPHA"
        fi

        RESULT_FILE="${OUT_DIR}/${RUN_NAME}.json"
        if [ -f "$RESULT_FILE" ]; then
          SKIPPED=$((SKIPPED + 1))
          echo "[$DONE/$TOTAL] SKIP $RUN_NAME (exists)"
          continue
        fi

        echo "[$DONE/$TOTAL] RUN $RUN_NAME"

        singularity exec --nv $CONTAINER python $TRAIN_SCRIPT \
            --topo ours \
            --n $N \
            --d $D \
            $ALPHA_FLAG \
            --seed $SEED \
            --dataset cifar100 \
            --tau 5 \
            --rounds 2000 \
            --batch-size 32 \
            --eval-freq 10 \
            --data-dir $DATA_DIR \
            --output-dir $OUT_DIR

        echo "  DONE $RUN_NAME"
        echo ""
      done
    done
  done
done

echo "============================================================"
echo " Complete. Ran $((DONE - SKIPPED))/$TOTAL, skipped $SKIPPED."
echo " Results: $OUT_DIR/"
echo "============================================================"
