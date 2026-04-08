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

echo "=== Building verify_gpu_v2 (parallel regularizer) ==="
singularity exec --nv $CONTAINER \
    nvcc -O3 -arch=sm_120 -o PoC_NEW/verify_gpu_v2 PoC_NEW/verify_gpu_v2.cu -lm

echo ""
echo "=== d=4 sweep: 5 seeds per size ==="
echo ""
echo "        n   d | seed0  seed1  seed2  seed3  seed4 |   avg    min    max    Ram  ratio"
echo "------------------------------------------------------------------------------------"

for logn in 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22; do
    n=$((1 << logn))
    results=""
    for seed in 0 1 2 3 4; do
        r=$(singularity exec --nv $CONTAINER \
            PoC_NEW/verify_gpu_v2 $n 4 $seed 2>/dev/null | grep "^RESULT:" | awk '{print $4}' | cut -d= -f2)
        results="$results $r"
    done
    # Compute stats in python
    singularity exec --nv $CONTAINER python3 -c "
import sys
vals = [float(x) for x in '$results'.split() if x]
if vals:
    import math
    ram = 4 - 2*math.sqrt(3)
    avg = sum(vals)/len(vals)
    mn = min(vals)
    mx = max(vals)
    seeds_str = ' '.join(f'{v:.4f}' for v in vals)
    print(f'$n:>8 4   | {seeds_str} | {avg:.4f} {mn:.4f} {mx:.4f} {ram:.4f} {avg/ram:.2f}x')
"
done

echo ""
echo "=== d=6 sweep: 5 seeds per size ==="
for logn in 8 10 12 14 16 18 20; do
    n=$((1 << logn))
    results=""
    for seed in 0 1 2 3 4; do
        r=$(singularity exec --nv $CONTAINER \
            PoC_NEW/verify_gpu_v2 $n 6 $seed 2>/dev/null | grep "^RESULT:" | awk '{print $4}' | cut -d= -f2)
        results="$results $r"
    done
    singularity exec --nv $CONTAINER python3 -c "
import sys, math
vals = [float(x) for x in '$results'.split() if x]
if vals:
    ram = 6 - 2*math.sqrt(5)
    avg = sum(vals)/len(vals)
    print(f'$n:>8 6   | {\" \".join(f\"{v:.4f}\" for v in vals)} | {avg:.4f} {min(vals):.4f} {max(vals):.4f} {ram:.4f} {avg/ram:.2f}x')
"
done

echo ""
echo "=== d=8 sweep: 5 seeds per size ==="
for logn in 8 10 12 14 16 18 20; do
    n=$((1 << logn))
    results=""
    for seed in 0 1 2 3 4; do
        r=$(singularity exec --nv $CONTAINER \
            PoC_NEW/verify_gpu_v2 $n 8 $seed 2>/dev/null | grep "^RESULT:" | awk '{print $4}' | cut -d= -f2)
        results="$results $r"
    done
    singularity exec --nv $CONTAINER python3 -c "
import sys, math
vals = [float(x) for x in '$results'.split() if x]
if vals:
    ram = 8 - 2*math.sqrt(7)
    avg = sum(vals)/len(vals)
    print(f'$n:>8 8   | {\" \".join(f\"{v:.4f}\" for v in vals)} | {avg:.4f} {min(vals):.4f} {max(vals):.4f} {ram:.4f} {avg/ram:.2f}x')
"
done
