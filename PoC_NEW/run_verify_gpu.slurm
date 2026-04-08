#!/bin/bash
#SBATCH --job-name=verify_ram_gpu
#SBATCH --partition=b40x4-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=300G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/verify_ram_gpu_%j.log

module load slurm

WORKDIR=/lustre/nvwulf/scratch/jungshwang/Spectral-Optimal-Graph-Construction-via-Reinforcement-Learning
cd $WORKDIR
mkdir -p logs

CONTAINER=dfl.sif

echo "=== Building v2 and v3 ==="
singularity exec --nv $CONTAINER \
    nvcc -O3 -arch=sm_120 -o PoC_NEW/verify_gpu_v2 PoC_NEW/verify_gpu_v2.cu -lm
singularity exec --nv $CONTAINER \
    nvcc -O3 -arch=sm_120 -o PoC_NEW/verify_gpu_v3 PoC_NEW/verify_gpu_v3.cu -lm

for d in 4 6 8; do
    echo ""
    echo "=== d=$d sweep: 5 seeds per size ==="
    echo ""
    echo "          n   d | seed0  seed1  seed2  seed3  seed4 |   avg    min    max    Ram  ratio"
    echo "------------------------------------------------------------------------------------"

    for logn in 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
        n=$((1 << logn))

        # Skip if n*d is odd
        nd=$((n * d))
        if [ $((nd % 2)) -ne 0 ]; then
            continue
        fi

        results=""
        all_ok=1
        for seed in 0 1 2 3 4; do
            line=$(singularity exec --nv $CONTAINER \
                PoC_NEW/verify_gpu_v3 $n $d $seed 2>/dev/null | grep "^RESULT:")
            if [ -z "$line" ]; then
                all_ok=0
                break
            fi
            r=$(echo "$line" | awk '{print $4}' | cut -d= -f2)
            results="$results $r"
        done

        if [ $all_ok -eq 0 ]; then
            echo "  n=$n d=$d — FAILED or OOM, stopping d=$d sweep"
            break
        fi

        singularity exec --nv $CONTAINER python3 -c "
import math
vals = [float(x) for x in '$results'.split() if x]
if vals:
    ram = $d - 2*math.sqrt($d - 1)
    avg = sum(vals)/len(vals)
    mn = min(vals)
    mx = max(vals)
    seeds_str = ' '.join(f'{v:.4f}' for v in vals)
    print(f'{$n:>10} {$d:>3} | {seeds_str} | {avg:.4f} {mn:.4f} {mx:.4f} {ram:.4f} {avg/ram:.2f}x')
"
    done
done
