#!/bin/bash
# ============================================
# Depth Ablation Study
# ============================================
# Submits 5 training jobs: 64, 128, 256, 512, 1024 layers
# Each job uses 4 GPUs and runs for up to 8 hours
# ============================================

set -e

# Load SLURM module
source /etc/profile
module load slurm/slurm/23.02.8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Create directories
mkdir -p checkpoints logs

echo "=============================================="
echo "Deep RL Graph Construction - Depth Ablation"
echo "=============================================="
echo "Submitting 5 training jobs for depths: 64, 128, 256, 512, 1024"
echo ""

# Submit jobs for each depth
for DEPTH in 64 128 256 512 1024; do
    echo "Submitting depth=$DEPTH..."
    JOB_ID=$(sbatch --export=ALL,DEPTH=$DEPTH --parsable scripts/train_depth.slurm)
    echo "  Job ID: $JOB_ID"
done

echo ""
echo "=============================================="
echo "All jobs submitted!"
echo "=============================================="
echo ""
echo "Check status: squeue -u \$USER"
echo "Cancel all:   scancel -u \$USER"
echo ""
