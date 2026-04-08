#!/bin/bash
# Array worker for OURS sweep. Called by sweep_ours.sh.

CONFIGS_FILE=${1:?}

LINE=$((SLURM_ARRAY_TASK_ID + 1))
CONFIG=$(sed -n "${LINE}p" "$CONFIGS_FILE")
N=$(echo "$CONFIG" | awk '{print $1}')
D=$(echo "$CONFIG" | awk '{print $2}')
ALPHA=$(echo "$CONFIG" | awk '{print $3}')
SEED=$(echo "$CONFIG" | awk '{print $4}')

module load slurm

WORKDIR=/lustre/nvwulf/scratch/jungshwang/Spectral-Optimal-Graph-Construction-via-Reinforcement-Learning
cd $WORKDIR

export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_PYTHONUNBUFFERED=1

CONTAINER=dfl.sif
TRAIN_SCRIPT=benchmarks/dfl/train.py
DATA_DIR=${WORKDIR}/data
OUT_DIR=benchmarks/dfl/results
mkdir -p $OUT_DIR

if [ "$ALPHA" = "iid" ]; then
    RUN_NAME="cifar100_ours_n${N}_d${D}_iid_s${SEED}"
    ALPHA_FLAG=""
else
    RUN_NAME="cifar100_ours_n${N}_d${D}_dir${ALPHA}_s${SEED}"
    ALPHA_FLAG="--alpha $ALPHA"
fi

RESULT_FILE="${OUT_DIR}/${RUN_NAME}.json"
if [ -f "$RESULT_FILE" ]; then
    echo "SKIP: $RESULT_FILE exists"
    exit 0
fi

echo "RUN: $RUN_NAME (task $SLURM_ARRAY_TASK_ID)"

singularity exec --nv $CONTAINER python $TRAIN_SCRIPT \
    --topo ours \
    --n $N \
    --d $D \
    $ALPHA_FLAG \
    --seed $SEED \
    --dataset cifar100 \
    --tau $TAU \
    --rounds 2000 \
    --batch-size 32 \
    --eval-freq 10 \
    --data-dir $DATA_DIR \
    --output-dir $OUT_DIR

echo "DONE: $RUN_NAME"
