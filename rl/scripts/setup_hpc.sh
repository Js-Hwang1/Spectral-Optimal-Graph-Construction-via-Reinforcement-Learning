#!/bin/bash
# Setup script for HPC environment
# Run this once after rsync to set up the environment

set -e

echo "Setting up Deep RL Graph Construction on HPC..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Create directories
mkdir -p checkpoints logs data scripts

# Option 1: Build Singularity container (recommended for HPC)
if command -v singularity &> /dev/null; then
    echo "Building Singularity container from NGC PyTorch image..."
    echo "This may take 10-20 minutes on first run..."

    if [ ! -f deeprl-graph.sif ]; then
        singularity build deeprl-graph.sif docker://nvcr.io/nvidia/pytorch:24.08-py3
        echo "Container built: deeprl-graph.sif"
    else
        echo "Container already exists: deeprl-graph.sif"
    fi
fi

# Option 2: Create Python virtual environment (fallback)
if [ ! -f deeprl-graph.sif ]; then
    echo "Setting up Python virtual environment..."

    module load python/3.11 2>/dev/null || true
    module load cuda/12.4 2>/dev/null || true

    python3 -m venv venv
    source venv/bin/activate

    pip install --upgrade pip
    pip install -r requirements.txt

    echo "Virtual environment created: venv/"
fi

# Make scripts executable
chmod +x scripts/*.sh
chmod +x scripts/*.slurm 2>/dev/null || true

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To run the depth ablation study:"
echo "  cd $PROJECT_DIR"
echo "  ./scripts/run_ablation.sh"
echo ""
echo "To run a single training:"
echo "  sbatch --export=DEPTH=256 scripts/train_single.slurm"
echo ""
echo "To run interactively:"
echo "  srun --partition=b40x4 --gres=gpu:1 --pty bash"
echo "  singularity exec --nv deeprl-graph.sif python train.py --debug"
echo ""
