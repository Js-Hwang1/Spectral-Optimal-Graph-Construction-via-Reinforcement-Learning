#!/bin/bash
#SBATCH --job-name=deeprl-d${DEPTH:-256}
#SBATCH --partition=b40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_d%j.out
#SBATCH --error=logs/train_d%j.err

# Deep RL Training Script for SLURM
# Trains a single model with specified depth

set -e

# Configuration
DEPTH=${DEPTH:-256}
HIDDEN_DIM=${HIDDEN_DIM:-128}
NUM_ENVS=${NUM_ENVS:-64}
TOTAL_STEPS=${TOTAL_STEPS:-100000000}

echo "=============================================="
echo "Deep RL Graph Construction Training"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Depth: $DEPTH"
echo "Hidden Dim: $HIDDEN_DIM"
echo "=============================================="

# Load modules (adjust for your HPC)
module load cuda/12.4 2>/dev/null || true
module load singularity 2>/dev/null || true

# Navigate to project directory
cd $SLURM_SUBMIT_DIR/..

# Check if using Singularity/Apptainer (common on HPC)
if command -v singularity &> /dev/null; then
    echo "Using Singularity container..."

    # Pull container if not exists
    if [ ! -f deeprl-graph.sif ]; then
        echo "Building Singularity container..."
        singularity build deeprl-graph.sif docker://nvcr.io/nvidia/pytorch:24.08-py3
    fi

    # Run training in container
    singularity exec --nv \
        --bind $PWD:/app \
        --pwd /app \
        deeprl-graph.sif \
        python train.py \
            --num-layers $DEPTH \
            --hidden-dim $HIDDEN_DIM \
            --num-envs $NUM_ENVS \
            --total-timesteps $TOTAL_STEPS \
            --no-progressive \
            --checkpoint-dir checkpoints/depth_${DEPTH} \
            --log-dir logs/depth_${DEPTH} \
            --device cuda

else
    echo "Running without container (native Python)..."

    # Activate virtual environment if exists
    if [ -f venv/bin/activate ]; then
        source venv/bin/activate
    fi

    # Install requirements
    pip install -r requirements.txt

    # Run training
    python train.py \
        --num-layers $DEPTH \
        --hidden-dim $HIDDEN_DIM \
        --num-envs $NUM_ENVS \
        --total-timesteps $TOTAL_STEPS \
        --no-progressive \
        --checkpoint-dir checkpoints/depth_${DEPTH} \
        --log-dir logs/depth_${DEPTH} \
        --device cuda
fi

echo "Training completed!"
