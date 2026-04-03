#!/bin/bash
# Alternative: Build container from Docker image (simpler, no .def file needed)
# Use this if the .def build fails

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=============================================="
echo "Building Singularity Container from NGC Image"
echo "=============================================="

# Remove old container if exists
if [ -f "deeprl-graph.sif" ]; then
    echo "Removing old container..."
    rm -f deeprl-graph.sif
fi

# Pull and convert NGC PyTorch container
echo "Pulling NGC PyTorch 24.08 (CUDA 12.4, Blackwell support)..."
singularity build --fakeroot deeprl-graph.sif docker://nvcr.io/nvidia/pytorch:24.08-py3

echo ""
echo "Container built: deeprl-graph.sif"
ls -lh deeprl-graph.sif

echo ""
echo "To install project dependencies inside container:"
echo "  singularity exec --nv deeprl-graph.sif pip install --user -r requirements.txt"
