#!/usr/bin/env bash
#
# run_dfl_sweep.sh — Launch all DFL training experiments for the NeurIPS paper.
#
# Submits one Slurm job per (topology, n, d, alpha, seed) configuration.
# Results are saved to benchmarks/dfl/results/.
#
# Usage: bash run_dfl_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DFL_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${DFL_DIR}/results"
mkdir -p "$RESULTS_DIR"

# Experiment grid
TOPOLOGIES="ring torus expander random qrsdr base"
N_VALUES="32 64"
D_VALUES="4"
ALPHAS="0.1 0.3 1.0"
SEEDS="0 1 2"

# Training config
DATASET="cifar100"
ROUNDS=2000
TAU=5
BATCH_SIZE=32
EVAL_FREQ=10

# Container path (update for your HPC)
CONTAINER="/lustre/nvwulf/scratch/jungshwang/deeprl-graph/deeprl-graph.sif"

JOB_COUNT=0

for topo in $TOPOLOGIES; do
  for n in $N_VALUES; do
    for d in $D_VALUES; do
      for alpha in $ALPHAS; do
        for seed in $SEEDS; do
          JOB_NAME="dfl_${topo}_n${n}_d${d}_a${alpha}_s${seed}"

          # Skip ring/torus/expander with non-default d (they have fixed degree)
          if [[ "$topo" == "ring" || "$topo" == "torus" || "$topo" == "expander" ]]; then
            if [[ "$d" != "4" ]]; then
              continue
            fi
          fi

          sbatch --job-name="$JOB_NAME" \
                 --output="${RESULTS_DIR}/logs/${JOB_NAME}.out" \
                 --error="${RESULTS_DIR}/logs/${JOB_NAME}.err" \
                 --partition=b40x4-long \
                 --gres=gpu:1 \
                 --time=12:00:00 \
                 --mem=32G \
                 <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME

module load slurm

mkdir -p ${RESULTS_DIR}/logs

singularity exec --nv \
    --bind /lustre:/lustre \
    --bind /home:/home \
    $CONTAINER \
    bash -c "
cd ${DFL_DIR}
PYTHONUNBUFFERED=1 python3 train.py \
    --topo $topo \
    --n $n \
    --d $d \
    --dataset $DATASET \
    --alpha $alpha \
    --rounds $ROUNDS \
    --tau $TAU \
    --batch-size $BATCH_SIZE \
    --eval-freq $EVAL_FREQ \
    --seed $seed \
    --output-dir $RESULTS_DIR
"
EOF
          JOB_COUNT=$((JOB_COUNT + 1))
          echo "Submitted: $JOB_NAME"
        done
      done
    done
  done
done

echo ""
echo "Total jobs submitted: $JOB_COUNT"
echo "Results will be in: $RESULTS_DIR/"
