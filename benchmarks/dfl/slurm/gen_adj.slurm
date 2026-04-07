#!/bin/bash
#SBATCH --job-name=gen-adj
#SBATCH --partition=b40x4
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/gen_adj.log

# Generates adjacency matrices for all (n, m) configs.
# Sparse regime: d = avg degree = {3, 4, 5}, m = n*d/2
# All configs are RD-compatible (2m/n is integer, n*d even).
#
# No GPU needed — pure C code with LAPACK.

module load slurm
cd /lustre/nvwulf/scratch/jungshwang/Spectral-Optimal-Graph-Construction-via-Reinforcement-Learning

mkdir -p data logs

# Load OpenBLAS module and build baselines
module load openblas/dynamic
make bin/baselines LDFLAGS="/usr/lib64/libopenblaso.so.0 -lm -lgomp"

SEED=42
W=24

# m = n * d / 2 for avg degree d
# d=3: n must be even (n*3 even) — all our n are even, OK
# d=4: always valid
# d=5: n must be even — OK
NODES="32 64 128 256"
DEGREES="3 4 5"

for N in $NODES; do
    for D in $DEGREES; do
        M=$((N * D / 2))
        echo "=== n=$N, m=$M (d=$D) ==="
        bin/baselines --config $N $M --seed $SEED --w $W
    done
done

echo ""
echo "All adjacency matrices generated."
ls data/adj_*.bin 2>/dev/null | wc -l
echo "total .bin files"
