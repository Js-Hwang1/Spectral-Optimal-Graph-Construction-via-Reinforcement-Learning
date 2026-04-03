/*
 * sweep_gpu.h — GPU-accelerated 3-step exhaustive sweep.
 *
 * CPU enumerates all leaf sequences, GPU batch-eigensolves,
 * CPU reduces to per-action values.
 */

#ifndef SWEEP_GPU_H
#define SWEEP_GPU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void gpu_init(void);
void gpu_cleanup(void);

/* Sweep for ADD phase: enumerate all 3-step add sequences from current graph.
 * Returns number of step-0 actions. For each, action_val[i] = best λ₂
 * achievable by choosing (action_src[i], action_tgt[i]) at step 0
 * and playing optimally for steps 1-2. */
int gpu_sweep_add(const uint8_t *adj, const int *deg, int n, int total_steps,
                   int *action_src, int *action_tgt, double *action_val);

/* Sweep for REM phase: same but for edge removal.
 * Bridge detection NOT done in sweep — disconnecting sequences get λ₂=0. */
int gpu_sweep_rem(const uint8_t *adj, const int *deg, int n, int total_steps,
                   int *action_src, int *action_tgt, double *action_val);

#ifdef __cplusplus
}
#endif

#endif /* SWEEP_GPU_H */
