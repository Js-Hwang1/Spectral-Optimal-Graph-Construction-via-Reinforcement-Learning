/*
 * crl.h — Dual-Phase Spectral Refine for G(n,m) algebraic connectivity.
 *
 * Batched add/remove phases with warm Lanczos signal, MLP-augmented scoring,
 * PPO training, all in C.
 */

#ifndef CRL_H
#define CRL_H

#include <stdint.h>
#include <stdbool.h>

/* ========================================================================== */
/* Constants                                                                   */
/* ========================================================================== */

#define N_MAX       24
#define EDGE_FEAT_DIM  6
#define GRAPH_FEAT_DIM 3
#define HIDDEN_DIM  64
#define RR_K        8       /* kept for v10_main.c compat */
#define LANCZOS_K   15
#define MAX_CAND    300     /* n*(n-1)/2 for n=24 = 276, with margin */
#define MLP_SCALE   0.1f   /* scale MLP correction relative to FV base */

/* ========================================================================== */
/* MLP Weights                                                                 */
/* ========================================================================== */

typedef struct {
    float W0[HIDDEN_DIM * EDGE_FEAT_DIM];  /* max input dim */
    float b0[HIDDEN_DIM];
    float W1[HIDDEN_DIM * HIDDEN_DIM];
    float b1[HIDDEN_DIM];
    float W2[HIDDEN_DIM];
    float b2[1];
    int in_dim;
} MLP;

typedef struct {
    MLP add_mlp;   /* in_dim = EDGE_FEAT_DIM (6) */
    MLP rem_mlp;   /* in_dim = EDGE_FEAT_DIM (6) */
    MLP val_mlp;   /* in_dim = GRAPH_FEAT_DIM (3) */
} PolicyWeights;

/* ========================================================================== */
/* Phase transaction (one ADD or REMOVE batch)                                 */
/* ========================================================================== */

typedef struct {
    float  features[MAX_CAND * EDGE_FEAT_DIM];  /* candidate edge features */
    double fv_gaps[MAX_CAND];                    /* FV base scores */
    int    cand_i[MAX_CAND];                     /* candidate pair nodes */
    int    cand_j[MAX_CAND];
    int    num_cand;

    int    selected[N_MAX];      /* indices into cand arrays */
    double log_probs[N_MAX];     /* log prob per selection */
    int    num_selected;

    float  gfeat[GRAPH_FEAT_DIM];
    double value;
    double reward;
    int    n;
    int    phase;                /* 0 = add, 1 = remove */
} PhaseTxn;

/* ========================================================================== */
/* RNG (xorshift64)                                                            */
/* ========================================================================== */

typedef struct {
    uint64_t state;
} RNG;

void     rng_init(RNG *rng, uint64_t seed);
uint64_t rng_next(RNG *rng);
int      rng_int(RNG *rng, int n);
double   rng_double(RNG *rng);

/* ========================================================================== */
/* Graph operations                                                            */
/* ========================================================================== */

void   build_ring_random(uint8_t *adj, int *degrees, int n, int m, RNG *rng);
double exact_lambda2(const uint8_t *adj, int n);

/* ========================================================================== */
/* MLP forward                                                                 */
/* ========================================================================== */

float mlp_forward_single(const MLP *mlp, const float *features);
void  mlp_forward_batch(const MLP *mlp, const float *features,
                        int count, float *logits);

/* ========================================================================== */
/* Lanczos + RR subspace                                                       */
/* ========================================================================== */

void lanczos_ext_k(const uint8_t *adj, int n,
                   const double *v_init, int k_lanczos, int n_eig,
                   double *V_out, double *lams_out);

/* Kept for v10_main.c compat */
void rr_update(double *V, double *lams, int n, int k, int u, int v, double sign);

/* ========================================================================== */
/* Bridge detection                                                            */
/* ========================================================================== */

void find_bridges(const uint8_t *adj, int n, uint8_t *bridge_mask);

/* ========================================================================== */
/* Softmax sampling                                                            */
/* ========================================================================== */

/* Single masked softmax sample (used by v10_main.c) */
int softmax_sample(const float *logits, const uint8_t *mask, int total,
                   RNG *rng, double *log_prob_out);

/* Batch softmax sampling without replacement.
 * Returns actual number selected (may be < B). */
int softmax_sample_batch(const double *scores, int num_cand, int B,
                         double tau, RNG *rng,
                         int *out_selected, double *out_log_probs);

/* ========================================================================== */
/* Dual-phase feature building and scoring                                     */
/* ========================================================================== */

/*
 * Build features for candidate edges from Fiedler vector.
 * v2: (n,) Fiedler vector (first column of Lanczos output)
 * feat_out: (num_cand, EDGE_FEAT_DIM) row-major
 * fv_gaps_out: (num_cand,) raw FV gap values n*(v2i-v2j)^2
 */
void build_candidate_features(const double *v2, const int *degrees,
                              int n, double progress,
                              const int *cand_i, const int *cand_j,
                              int num_cand,
                              const uint8_t *ref_adj, const double *ref_v2,
                              float *feat_out, double *fv_gaps_out);

/*
 * Build graph-level features for value function.
 * gfeat_out: (GRAPH_FEAT_DIM,)
 */
void build_graph_features_dp(double lam2_est, const int *degrees,
                             int n, double progress, float *gfeat_out);

/*
 * Score candidates: final = (±fv_gap) + mlp(features).
 * is_remove: if true, negate fv_gap base.
 */
void score_candidates(const MLP *mlp, const float *features,
                      const double *fv_gaps, int num_cand,
                      int is_remove, double *scores_out);

/* ========================================================================== */
/* Dual-phase episode collection (training)                                    */
/* ========================================================================== */

/*
 * Collect one dual-phase training episode.
 *
 * Returns 0 on success. txns must hold at least max_txns PhaseTxns.
 * out_metrics: [final_l2, init_l2, best_l2, n, m]
 */
int dual_phase_collect_episode(
    const PolicyWeights *pw,
    int n, int m, int num_cycles, double swap_frac,
    double tau_start, double tau_end, uint64_t seed,
    const uint8_t *ref_adj,
    PhaseTxn *txns, int max_txns,
    int *out_num_txns, double *out_metrics,
    uint8_t *out_best_adj);

/* ========================================================================== */
/* MLP activation cache (for backward pass)                                    */
/* ========================================================================== */

typedef struct {
    float z0[HIDDEN_DIM];
    float h0[HIDDEN_DIM];
    float z1[HIDDEN_DIM];
    float h1[HIDDEN_DIM];
    float out;
    float input[EDGE_FEAT_DIM]; /* max dim */
    int   in_dim;
} MLPCache;

/* ========================================================================== */
/* MLP Gradients                                                               */
/* ========================================================================== */

typedef struct {
    float dW0[HIDDEN_DIM * EDGE_FEAT_DIM];
    float db0[HIDDEN_DIM];
    float dW1[HIDDEN_DIM * HIDDEN_DIM];
    float db1[HIDDEN_DIM];
    float dW2[HIDDEN_DIM];
    float db2[1];
} MLPGrad;

typedef struct {
    MLPGrad add_grad;
    MLPGrad rem_grad;
    MLPGrad val_grad;
} PolicyGrad;

/* ========================================================================== */
/* MLP forward/backward with caching                                           */
/* ========================================================================== */

float mlp_forward_cached(const MLP *mlp, const float *features, MLPCache *cache);
void  mlp_backward(const MLP *mlp, const MLPCache *cache,
                   float d_out, MLPGrad *grad);

/* ========================================================================== */
/* PPO for dual-phase transactions                                             */
/* ========================================================================== */

/*
 * Re-evaluate one phase transaction under current policy.
 * out_log_prob: sum of log probs for all selected items.
 * out_entropy: entropy of the candidate distribution.
 * out_value: value from val_mlp on stored graph features.
 */
void ppo_evaluate_phase(const PolicyWeights *pw, const PhaseTxn *txn,
                        double *out_log_prob, double *out_entropy,
                        float *out_value);

/*
 * Backward pass for one phase transaction's PPO loss.
 * Accumulates gradients into pg.
 */
void ppo_backward_phase(const PolicyWeights *pw, const PhaseTxn *txn,
                        double advantage, double ret, double old_lp,
                        double ent_coef, double clip_eps, PolicyGrad *pg);

/* ========================================================================== */
/* Gradient utilities                                                          */
/* ========================================================================== */

void  policy_grad_zero(PolicyGrad *pg);
void  policy_grad_add(PolicyGrad *dst, const PolicyGrad *src);
void  policy_grad_scale(PolicyGrad *pg, float scale);
float policy_grad_norm(const PolicyGrad *pg);
void  policy_grad_clip(PolicyGrad *pg, float max_norm);

/* ========================================================================== */
/* AdamW optimizer                                                             */
/* ========================================================================== */

#define MLP_MAX_PARAMS (HIDDEN_DIM * EDGE_FEAT_DIM + HIDDEN_DIM + \
                        HIDDEN_DIM * HIDDEN_DIM + HIDDEN_DIM + \
                        HIDDEN_DIM + 1)

typedef struct {
    float m[MLP_MAX_PARAMS];
    float v[MLP_MAX_PARAMS];
} AdamMLPState;

typedef struct {
    AdamMLPState add_state;
    AdamMLPState rem_state;
    AdamMLPState val_state;
    int t;
} AdamState;

void adam_init(AdamState *state);
void adam_step(PolicyWeights *pw, const PolicyGrad *pg, AdamState *state,
              double lr, double beta1, double beta2, double eps,
              double weight_decay);

/* ========================================================================== */
/* GAE computation                                                             */
/* ========================================================================== */

void compute_gae(const double *values, const double *rewards,
                 int num_steps, double gamma, double gae_lambda,
                 double *advantages, double *returns);
void normalize_advantages(double *advantages, int count);

/* ========================================================================== */
/* Checkpoint I/O                                                              */
/* ========================================================================== */

#define CRL_CKPT_MAGIC   0x43524C42
#define CRL_CKPT_VERSION 1

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t hidden_dim;
    uint32_t edge_feat_dim;
    uint32_t graph_feat_dim;
    uint32_t episode;
    float    best_avg_improvement;
    uint32_t reserved[5];
} CkptHeader;

int save_checkpoint(const char *path, const PolicyWeights *pw,
                    uint32_t episode, float best_avg_improvement);
int load_checkpoint(const char *path, PolicyWeights *pw,
                    uint32_t *episode, float *best_avg_improvement);

/* ========================================================================== */
/* Weight initialization                                                       */
/* ========================================================================== */

void policy_init_orthogonal(PolicyWeights *pw, RNG *rng, float gain);

/* ========================================================================== */
/* Training configuration                                                      */
/* ========================================================================== */

typedef struct {
    int min_n;
    int max_n;
    int num_episodes;
    int batch_size;
    int ppo_epochs;
    double cycle_mult;       /* K = cycle_mult * n Lanczos epochs */
    double swap_frac;        /* total_swaps = m * swap_frac */
    double tau_start;
    double tau_end;
    double lr;
    double gamma;
    double gae_lambda;
    double clip_eps;
    double entropy_coef_start;
    double entropy_coef_end;
    double grad_clip;
    double weight_decay;
    uint64_t seed;
    int log_freq;
    int ckpt_freq;
    char save_dir[256];
} TrainConfig;

#endif /* CRL_H */
