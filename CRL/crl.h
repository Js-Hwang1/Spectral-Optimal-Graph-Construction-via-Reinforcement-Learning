/*
 * crl.h — Full C PPO Training for G(n,m) algebraic connectivity.
 *
 * Episode collection + MLP backward + AdamW + GAE + PPO update, all in C.
 */

#ifndef CRL_H
#define CRL_H

#include <stdint.h>
#include <stdbool.h>

/* ========================================================================== */
/* Constants                                                                   */
/* ========================================================================== */

#define N_MAX       24
#define EDGE_FEAT_DIM  4
#define GRAPH_FEAT_DIM 3
#define HIDDEN_DIM  64
#define RR_K        8
#define LANCZOS_K   15

/* ========================================================================== */
/* MLP Weights                                                                 */
/* ========================================================================== */

typedef struct {
    /* Layer 0: in_dim -> HIDDEN_DIM */
    float W0[HIDDEN_DIM * EDGE_FEAT_DIM];  /* max input dim */
    float b0[HIDDEN_DIM];
    /* Layer 1: HIDDEN_DIM -> HIDDEN_DIM */
    float W1[HIDDEN_DIM * HIDDEN_DIM];
    float b1[HIDDEN_DIM];
    /* Layer 2: HIDDEN_DIM -> 1 */
    float W2[HIDDEN_DIM];
    float b2[1];
    int in_dim;
} MLP;

typedef struct {
    MLP add_mlp;   /* in_dim = EDGE_FEAT_DIM (4) */
    MLP rem_mlp;   /* in_dim = EDGE_FEAT_DIM (4) */
    MLP val_mlp;   /* in_dim = GRAPH_FEAT_DIM (3) */
} PolicyWeights;

/* ========================================================================== */
/* Per-swap transition (output to Python)                                      */
/* ========================================================================== */

/* Flat indices into N_MAX×N_MAX arrays */
typedef struct {
    float  feat[N_MAX * N_MAX * EDGE_FEAT_DIM];   /* (N,N,4) edge features */
    uint8_t mask[N_MAX * N_MAX];                    /* (N,N) valid mask */
    int32_t flat_idx;                               /* chosen flat index */
    float  gfeat[GRAPH_FEAT_DIM];                   /* (3,) graph features */
    double log_prob;
    double value;
    double reward;
} SwapTxn;

/* Episode output */
typedef struct {
    SwapTxn *add_txns;
    SwapTxn *rem_txns;
    int num_add;
    int num_rem;
    /* Metrics */
    double final_lambda2;
    double initial_lambda2;
    double improvement;
    int    n;
    int    m;
} EpisodeResult;

/* ========================================================================== */
/* RNG (xorshift64, per-thread safe via explicit state)                        */
/* ========================================================================== */

typedef struct {
    uint64_t state;
} RNG;

void   rng_init(RNG *rng, uint64_t seed);
uint64_t rng_next(RNG *rng);
int    rng_int(RNG *rng, int n);
double rng_double(RNG *rng);

/* ========================================================================== */
/* Graph operations                                                            */
/* ========================================================================== */

/* Build ring + random edges. adj: (n,n) row-major, degrees: (n,) */
void build_ring_random(uint8_t *adj, int *degrees, int n, int m, RNG *rng);

/* Exact lambda2 via LAPACK dsyev_ */
double exact_lambda2(const uint8_t *adj, int n);

/* ========================================================================== */
/* MLP forward                                                                 */
/* ========================================================================== */

/* Single MLP forward: features (in_dim,) -> scalar */
float mlp_forward_single(const MLP *mlp, const float *features);

/* Batch MLP forward: features (count, in_dim) -> logits (count,) */
void mlp_forward_batch(const MLP *mlp, const float *features,
                       int count, float *logits);

/* Value MLP forward: gfeat (GRAPH_FEAT_DIM,) -> scalar */
float value_mlp_forward(const MLP *mlp, const float *gfeat);

/* ========================================================================== */
/* Lanczos + RR subspace                                                       */
/* ========================================================================== */

/*
 * Lanczos k-step with full reorthogonalization.
 * Extracts n_eig eigenpairs [v2..v_{n_eig+1}].
 *
 * adj: (n,n) uint8 adjacency
 * v_init: (n,) warm-start vector, or NULL for random init
 * V_out: (n, n_eig) output eigenvectors (row-major: V_out[i*n_eig + p])
 * lams_out: (n_eig,) output eigenvalues
 */
void lanczos_ext_k(const uint8_t *adj, int n,
                   const double *v_init, int k_lanczos, int n_eig,
                   double *V_out, double *lams_out);

/*
 * Rayleigh-Ritz subspace update after add/remove edge (u,v).
 * V: (n, k) subspace, lams: (k,) eigenvalues. Updated in-place.
 * sign: +1 for add, -1 for remove.
 */
void rr_update(double *V, double *lams, int n, int k, int u, int v, double sign);

/* ========================================================================== */
/* Bridge detection                                                            */
/* ========================================================================== */

/* Find bridges via iterative Tarjan. bridge_mask: (n,n) uint8, set to 1 for bridges. */
void find_bridges(const uint8_t *adj, int n, uint8_t *bridge_mask);

/* ========================================================================== */
/* Feature computation                                                         */
/* ========================================================================== */

/*
 * Build (n,n,4) edge features from current state.
 * feat_out: (N_MAX, N_MAX, EDGE_FEAT_DIM) pre-allocated
 */
void build_edge_features(const double *V_rr, const int *degrees,
                         int n, int step, int k_steps, float *feat_out);

/* Build (3,) graph features */
void build_graph_features(const double *lams_rr, const int *degrees,
                          int n, int step, int k_steps, float *gfeat_out);

/* ========================================================================== */
/* Softmax sampling                                                            */
/* ========================================================================== */

/*
 * Masked softmax sample from logits.
 * Returns flat index, writes log_prob.
 * mask: (n*n) uint8, 1 = valid
 */
int softmax_sample(const float *logits, const uint8_t *mask, int total,
                   RNG *rng, double *log_prob_out);

/* ========================================================================== */
/* Episode collection                                                          */
/* ========================================================================== */

/*
 * Collect one full episode. Main C entry point.
 *
 * pw: policy weights
 * n, m: graph parameters
 * k_steps: Lanczos snapshots per episode
 * swap_frac: fraction of edges to swap per step
 * seed: RNG seed for this episode
 *
 * add_txns, rem_txns: pre-allocated output arrays (size max_txns each)
 * out_num_add, out_num_rem: actual counts written
 * out_metrics: [5] array: final_l2, init_l2, improvement, n, m
 *
 * Returns 0 on success.
 */
int crl_collect_episode(
    const PolicyWeights *pw,
    int n, int m, int k_steps, double swap_frac, uint64_t seed,
    /* Output buffers */
    SwapTxn *add_txns, SwapTxn *rem_txns,
    int *out_num_add, int *out_num_rem,
    double *out_metrics,
    int max_txns
);

/* ========================================================================== */
/* Weight loading                                                              */
/* ========================================================================== */

/*
 * Load policy weights from flat arrays (called from Python ctypes).
 * Each MLP has: W0 (H*in), b0 (H), W1 (H*H), b1 (H), W2 (H), b2 (1)
 * Weights are in row-major order matching PyTorch's Linear(in, out).weight
 * which stores (out, in) — so W[out_idx * in_dim + in_idx].
 */
void crl_load_weights(PolicyWeights *pw,
                      const float *add_W0, const float *add_b0,
                      const float *add_W1, const float *add_b1,
                      const float *add_W2, const float *add_b2,
                      const float *rem_W0, const float *rem_b0,
                      const float *rem_W1, const float *rem_b1,
                      const float *rem_W2, const float *rem_b2,
                      const float *val_W0, const float *val_b0,
                      const float *val_W1, const float *val_b1,
                      const float *val_W2, const float *val_b2);

/* ========================================================================== */
/* Row update for logits                                                       */
/* ========================================================================== */

/*
 * Re-score rows and columns of node `node` in (n,n) logits matrix.
 * Uses current V_rr state. O(N) per node.
 */
void rescore_node(const MLP *mlp, const double *V_rr, const int *degrees,
                  int n, int step, int k_steps, int node, float *logits);

/* ========================================================================== */
/* MLP activation cache (for backward pass)                                    */
/* ========================================================================== */

typedef struct {
    float z0[HIDDEN_DIM];      /* pre-activation layer 0 */
    float h0[HIDDEN_DIM];      /* post-SiLU layer 0 */
    float z1[HIDDEN_DIM];      /* pre-activation layer 1 */
    float h1[HIDDEN_DIM];      /* post-SiLU layer 1 */
    float out;                  /* final scalar output */
    float input[EDGE_FEAT_DIM]; /* cached input (max dim) */
    int   in_dim;               /* actual input dimension */
} MLPCache;

/* ========================================================================== */
/* MLP Gradients                                                               */
/* ========================================================================== */

typedef struct {
    float dW0[HIDDEN_DIM * EDGE_FEAT_DIM]; /* max input dim */
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
/* AdamW optimizer state                                                       */
/* ========================================================================== */

/* Max parameter count for a single MLP buffer */
#define MLP_MAX_PARAMS (HIDDEN_DIM * EDGE_FEAT_DIM + HIDDEN_DIM + \
                        HIDDEN_DIM * HIDDEN_DIM + HIDDEN_DIM + \
                        HIDDEN_DIM + 1)

typedef struct {
    float m[MLP_MAX_PARAMS];   /* first moment */
    float v[MLP_MAX_PARAMS];   /* second moment */
} AdamMLPState;

typedef struct {
    AdamMLPState add_state;
    AdamMLPState rem_state;
    AdamMLPState val_state;
    int t;   /* timestep counter */
} AdamState;

/* ========================================================================== */
/* Training configuration                                                      */
/* ========================================================================== */

typedef struct {
    int min_n;
    int max_n;
    int num_episodes;
    int k_steps;
    int batch_size;
    double swap_frac;
    double lr;
    double gamma;
    double gae_lambda;
    double clip_eps;
    int ppo_epochs;
    double entropy_coef_start;
    double entropy_coef_end;
    double grad_clip;
    double weight_decay;
    uint64_t seed;
    int log_freq;      /* updates between logs */
    int ckpt_freq;     /* updates between checkpoints */
    char save_dir[256];
} TrainConfig;

/* ========================================================================== */
/* Checkpoint binary format                                                    */
/* ========================================================================== */

#define CRL_CKPT_MAGIC  0x43524C42  /* "CRLB" */
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

/* ========================================================================== */
/* MLP forward with activation caching                                         */
/* ========================================================================== */

float mlp_forward_cached(const MLP *mlp, const float *features, MLPCache *cache);

/* ========================================================================== */
/* MLP backward pass                                                           */
/* ========================================================================== */

/*
 * Backprop d_out through MLP, accumulating gradients into grad.
 * Uses cached activations from mlp_forward_cached().
 */
void mlp_backward(const MLP *mlp, const MLPCache *cache,
                  float d_out, MLPGrad *grad);

/* ========================================================================== */
/* PPO re-evaluation and backward                                              */
/* ========================================================================== */

/*
 * Re-evaluate one swap (add+rem) under current policy.
 * Returns new log_prob (add+rem), entropy, value.
 */
void ppo_evaluate_swap(const PolicyWeights *pw,
                       const SwapTxn *add_txn, const SwapTxn *rem_txn,
                       int n,
                       double *out_log_prob, double *out_entropy,
                       float *out_value);

/*
 * Backward pass for one swap's PPO loss contribution.
 * Accumulates gradients into pg.
 *
 * advantage, ret, old_lp, ent_coef come from the PPO loss.
 * clip_eps is the PPO clipping epsilon.
 */
void ppo_backward_swap(const PolicyWeights *pw,
                       const SwapTxn *add_txn, const SwapTxn *rem_txn,
                       int n,
                       double advantage, double ret, double old_lp,
                       double ent_coef, double clip_eps,
                       PolicyGrad *pg);

/* ========================================================================== */
/* Gradient utilities                                                          */
/* ========================================================================== */

void policy_grad_zero(PolicyGrad *pg);
void policy_grad_add(PolicyGrad *dst, const PolicyGrad *src);
void policy_grad_scale(PolicyGrad *pg, float scale);
float policy_grad_norm(const PolicyGrad *pg);
void policy_grad_clip(PolicyGrad *pg, float max_norm);

/* ========================================================================== */
/* AdamW optimizer                                                             */
/* ========================================================================== */

void adam_init(AdamState *state);
void adam_step(PolicyWeights *pw, const PolicyGrad *pg, AdamState *state,
              double lr, double beta1, double beta2, double eps,
              double weight_decay);

/* ========================================================================== */
/* GAE computation                                                             */
/* ========================================================================== */

/*
 * Compute GAE advantages and returns for one episode.
 * values, rewards: arrays of length num_steps
 * advantages, returns: output arrays of length num_steps
 */
void compute_gae(const double *values, const double *rewards,
                 int num_steps, double gamma, double gae_lambda,
                 double *advantages, double *returns);

/* Normalize advantages in-place (mean=0, std=1) */
void normalize_advantages(double *advantages, int count);

/* ========================================================================== */
/* Checkpoint I/O                                                              */
/* ========================================================================== */

int save_checkpoint(const char *path, const PolicyWeights *pw,
                    uint32_t episode, float best_avg_improvement);
int load_checkpoint(const char *path, PolicyWeights *pw,
                    uint32_t *episode, float *best_avg_improvement);

/* ========================================================================== */
/* Weight initialization                                                       */
/* ========================================================================== */

void policy_init_orthogonal(PolicyWeights *pw, RNG *rng, float gain);

#endif /* CRL_H */
