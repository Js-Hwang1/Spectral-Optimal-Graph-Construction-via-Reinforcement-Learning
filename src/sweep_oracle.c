#!/bin/bash
#SBATCH --job-name=crl-oracle
#SBATCH --partition=b40x4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --output=sweep_oracle_%j.log

source /etc/profile
module load slurm 2>/dev/null
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

cd /lustre/nvwulf/scratch/jungshwang/CRL

echo "========================================"
echo "Sweep Oracle — Absolute Ceiling"
echo "Start: $(date)"
echo "========================================"

echo "Building..."
nvcc -O3 -Xcompiler -fPIC -c -o bin/sweep_gpu.o src/sweep_gpu.cu 2>&1 | tail -3
gcc -O3 -Wall -std=c11 -D_GNU_SOURCE -DUSE_GPU_SWEEP -Isrc -o bin/sweep_oracle \
    src/sweep_oracle.c src/crl.c bin/sweep_gpu.o \
    /usr/lib64/libopenblaso.so.0 -lm -lgomp -lcusolver -lcudart -L/usr/local/cuda/lib64 -lstdc++ 2>&1
if [ $? -ne 0 ]; then echo "BUILD FAILED"; exit 1; fi
echo "Build OK"

echo ""
bin/sweep_oracle --n 8,10,12,16 --baselines baselines.csv --swaps 3 --epoch-C 1.0

echo ""
echo "Done: $(date)"
