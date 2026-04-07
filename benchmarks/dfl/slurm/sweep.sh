#!/bin/bash
# Launch full DFL benchmark sweep.
#
# Density = (m - (n-1)) / (n(n-1)/2 - (n-1))
# m values are RD-compatible (2m/n is integer).
#
# Usage:
#   bash sweep.sh          # launch all configs
#   bash sweep.sh --dry    # print commands without submitting

DRY=${1:-""}
COUNT=0

TOPOS="ring sw_r25 sw_r50 sw_r75 er fv rd"
SEEDS="0 1 2 3 4"
ALPHAS="1.0 0.5 0.1"

# (n, m) pairs for density ≈ 0.3, 0.5, 0.7
CONFIGS="
32 176
32 256
32 352
64 640
64 1024
64 1440
128 2496
128 4096
128 5696
256 9984
256 16384
256 22912
"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

while read -r N M; do
    [ -z "$N" ] && continue

    for TOPO in $TOPOS; do
        for ALPHA in $ALPHAS; do
            for SEED in $SEEDS; do
                # Skip RD for non-regular configs
                if [ "$TOPO" = "rd" ]; then
                    D=$((2 * M / N))
                    CHECK=$((D * N / 2))
                    if [ "$CHECK" -ne "$M" ]; then
                        continue
                    fi
                fi

                CMD="sbatch ${SCRIPT_DIR}/run_single.slurm $TOPO $N $M $ALPHA $SEED"

                if [ "$DRY" = "--dry" ]; then
                    echo "$CMD"
                else
                    $CMD
                fi
                COUNT=$((COUNT + 1))
            done
        done
    done
done <<< "$CONFIGS"

echo ""
echo "Total jobs: $COUNT"
