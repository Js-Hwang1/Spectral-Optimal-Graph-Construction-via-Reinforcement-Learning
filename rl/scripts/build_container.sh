#!/bin/bash
# Build the dedicated Singularity container for Deep RL Graph Construction
# Run this on a compute node or with sufficient memory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=============================================="
echo "Building Deep RL Graph Singularity Container"
echo "=============================================="
echo "Project dir: $PROJECT_DIR"
echo "Definition file: deeprl-graph.def"
echo ""

# Check if definition file exists
if [ ! -f "deeprl-graph.def" ]; then
    echo "Error: deeprl-graph.def not found!"
    exit 1
fi

# Remove old container if exists
if [ -f "deeprl-graph.sif" ]; then
    echo "Removing old container..."
    rm -f deeprl-graph.sif
fi

# Build container
echo "Building container (this may take 10-20 minutes)..."
singularity build --fakeroot deeprl-graph.sif deeprl-graph.def

echo ""
echo "=============================================="
echo "Container built successfully!"
echo "=============================================="
ls -lh deeprl-graph.sif

echo ""
echo "Test the container:"
echo "  singularity run --nv deeprl-graph.sif train.py --debug"
