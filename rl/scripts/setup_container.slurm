#!/bin/bash
#SBATCH --job-name=setup-deeprl
#SBATCH --partition=b40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

# ============================================
# Setup Deep RL Graph Construction Environment
# ============================================
# This job:
# 1. Builds the Singularity container
# 2. Installs Python dependencies
# 3. Tests the setup
# ============================================

set -e

echo "=============================================="
echo "Deep RL Graph Construction - Setup Job"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo "=============================================="

# Navigate to project directory
cd /lustre/nvwulf/scratch/jungshwang/deeprl-graph

# Create directories
mkdir -p checkpoints logs data

# ============================================
# Step 1: Build Singularity Container
# ============================================
echo ""
echo "[1/3] Building Singularity container..."
echo "      Base image: nvcr.io/nvidia/pytorch:25.01-py3 (Blackwell support)"
echo "      This may take 10-20 minutes..."

# Force rebuild with new Blackwell-compatible container
FORCE_REBUILD=${FORCE_REBUILD:-0}
if [ "$FORCE_REBUILD" -eq 1 ] && [ -f "deeprl-graph.sif" ]; then
    echo "      Removing old container for rebuild..."
    rm -f deeprl-graph.sif
fi

if [ -f "deeprl-graph.sif" ]; then
    echo "      Container already exists, skipping build."
    echo "      To rebuild, use: sbatch --export=FORCE_REBUILD=1 scripts/setup_container.slurm"
else
    singularity build deeprl-graph.sif docker://nvcr.io/nvidia/pytorch:25.01-py3
    echo "      Container built successfully!"
fi

ls -lh deeprl-graph.sif

# ============================================
# Step 2: Install Python Dependencies
# ============================================
echo ""
echo "[2/3] Installing Python dependencies..."

# Create a pip cache directory in scratch
export PIP_CACHE_DIR=/lustre/nvwulf/scratch/jungshwang/.pip_cache
mkdir -p $PIP_CACHE_DIR

# Install dependencies inside container (to user site-packages)
singularity exec --nv deeprl-graph.sif pip install --user \
    networkx>=3.0 \
    tqdm>=4.65.0 \
    tensorboard>=2.14.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    pandas>=2.0.0

echo "      Dependencies installed!"

# ============================================
# Step 3: Test the Setup
# ============================================
echo ""
echo "[3/3] Testing setup..."

singularity exec --nv deeprl-graph.sif python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

import numpy as np
print(f'NumPy version: {np.__version__}')

import networkx as nx
print(f'NetworkX version: {nx.__version__}')

print()
print('All tests passed!')
"

# Test project imports
echo ""
echo "Testing project imports..."
singularity exec --nv --bind $(pwd):/app --pwd /app deeprl-graph.sif python -c "
import sys
sys.path.insert(0, '.')
from config import Config, get_default_config
print('Config: OK')
# Note: Full model tests require more memory, skip for now
print('Project imports: OK')
"

echo ""
echo "=============================================="
echo "Setup completed successfully!"
echo "=============================================="
echo ""
echo "To run training:"
echo "  sbatch scripts/train_depth.slurm DEPTH=256"
echo ""
echo "To run interactively:"
echo "  srun --partition=b40x4 --gres=gpu:4 --mem=256G --cpus-per-task=32 --pty bash"
echo "  singularity exec --nv --bind \$(pwd):/app --pwd /app deeprl-graph.sif python train.py --debug"
echo ""
