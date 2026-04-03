#!/bin/bash
#SBATCH --job-name=build-container
#SBATCH --partition=b40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/build_container_%j.out
#SBATCH --error=logs/build_container_%j.err

# Build Singularity container on compute node
# (Login nodes have memory limits that kill the build)

set -e

cd $SLURM_SUBMIT_DIR/..

echo "Building Singularity container..."
echo "Node: $SLURMD_NODENAME"
echo "Working dir: $(pwd)"

# Build container
singularity build deeprl-graph.sif docker://nvcr.io/nvidia/pytorch:24.08-py3

echo "Container built successfully: deeprl-graph.sif"
ls -lh deeprl-graph.sif
