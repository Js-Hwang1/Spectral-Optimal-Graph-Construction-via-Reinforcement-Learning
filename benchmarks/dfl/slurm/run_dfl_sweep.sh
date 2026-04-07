#!/bin/bash
#SBATCH --job-name=dfl_sweep
#SBATCH --partition=b40x4-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/dfl_sweep_%j.log

# ============================================================
# DFL sweep: runs ALL topology x alpha x seed configs
# sequentially on a single GPU.
#
# Usage:
#   sbatch run_dfl_sweep.sh
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

# ============================================================
# Experiment config
# ============================================================
DATASET=cifar100
N=32
D=4
ROUNDS=2000
TAU=5
BATCH_SIZE=32
EVAL_FREQ=10

# Topologies to compare:
#   ring     - d=2 static baseline
#   torus    - d=4 static baseline
#   expander - d=2*ceil(log2(n)) static baseline
#   random   - random d-regular (our main static competitor)
#   qrsdr    - QRS-DR (OURS)
#   base     - Base-(k+1) time-varying (SOTA competitor)
TOPOLOGIES="ring torus expander random qrsdr base"

# Heterogeneity levels
ALPHAS="0.1 0.3 1.0"

# Seeds for averaging
SEEDS="0 1 2"

# ============================================================
# Run loop
# ============================================================
TOTAL=0
DONE=0

# Count total runs
for topo in $TOPOLOGIES; do
  for alpha in $ALPHAS; do
    for seed in $SEEDS; do
      TOTAL=$((TOTAL + 1))
    done
  done
done

echo "============================================================"
echo " DFL Sweep: $DATASET, n=$N, d=$D"
echo " $TOTAL total runs (${#TOPOLOGIES} topos x ${#ALPHAS} alphas x ${#SEEDS} seeds)"
echo "============================================================"
echo ""

for topo in $TOPOLOGIES; do
  for alpha in $ALPHAS; do
    for seed in $SEEDS; do
      DONE=$((DONE + 1))
      RUN_NAME="${DATASET}_${topo}_n${N}_d${D}_dir${alpha}_s${seed}"

      # Skip if result already exists
      RESULT_FILE="${OUT_DIR}/${RUN_NAME}.json"
      if [ -f "$RESULT_FILE" ]; then
        echo "[$DONE/$TOTAL] SKIP $RUN_NAME (exists)"
        continue
      fi

      echo "[$DONE/$TOTAL] RUN  $RUN_NAME"
      echo "  topo=$topo n=$N d=$D alpha=$alpha seed=$seed"

      singularity exec --nv $CONTAINER python $TRAIN_SCRIPT \
          --topo $topo \
          --n $N \
          --d $D \
          --alpha $alpha \
          --seed $seed \
          --dataset $DATASET \
          --tau $TAU \
          --rounds $ROUNDS \
          --batch-size $BATCH_SIZE \
          --eval-freq $EVAL_FREQ \
          --data-dir $DATA_DIR \
          --output-dir $OUT_DIR

      echo "  DONE ($RUN_NAME)"
      echo ""
    done
  done
done

echo "============================================================"
echo " All $TOTAL runs complete."
echo " Results in: $OUT_DIR/"
echo "============================================================"
