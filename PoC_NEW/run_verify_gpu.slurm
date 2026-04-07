#!/bin/bash
#SBATCH --job-name=verify_ram_gpu
#SBATCH --partition=b40x4-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/verify_ram_gpu_%j.log

module load slurm

WORKDIR=/lustre/nvwulf/scratch/jungshwang/Spectral-Optimal-Graph-Construction-via-Reinforcement-Learning
cd $WORKDIR
mkdir -p logs

CONTAINER=dfl.sif

echo "=== Building verify_gpu inside container ==="
singularity exec --nv $CONTAINER \
    nvcc -O3 -arch=sm_120 -o PoC_NEW/verify_gpu PoC_NEW/verify_gpu.cu -lm

echo ""
echo "=== GPU Large-scale Ramanujan Verification ==="
echo ""
singularity exec --nv $CONTAINER \
    PoC_NEW/verify_gpu --sweep

echo ""
echo "=== Also test d=6 and d=8 at select sizes ==="
for d in 6 8; do
    for logn in 10 14 18 20; do
        n=$((1 << logn))
        echo "--- n=$n d=$d ---"
        singularity exec --nv $CONTAINER \
            PoC_NEW/verify_gpu $n $d
    done
done
