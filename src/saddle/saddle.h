/*
 * saddle.h -- Saddle-aware RL system for lambda2 maximization.
 *
 * Two agents:
 *   Climber:   scores edge swaps (ADD + REM heads) for greedy hill climbing
 *   Navigator: detects saddle points, manages DFS backtracking stack
 *
 * See BUILD.md for architecture details.
 */

#ifndef SADDLE_H
#define SADDLE_H

#include "../crl.h"
#include <stdint.h>
#include <math.h>

/* ========================================================================== */
/* Constants                                                                   */
/* ========================================================================== */

#define SD_H          64    /* hidden dimension */
#define SD_NE_IN      5     /* node features: deg, tri, density, v2_sq, er_cent */
#define SD_ES_IN      (2 * SD_H + 3)  /* edge scorer: embed_i(H) || embed_j(H) || cn || fv_gap || reff */
#define SD_LANCZOS_K  10    /* warm Lanczos iterations */
#define SD_MAX_STACK  64    /* max saddle stack depth */
#define SD_MAX_CAND   6     /* max branches per saddle */

/* Navigator: detection features */
#define SD_NAV_NH     16    /* small hidden dim -- simple decision */
#define SD_NAV_IN     2     /* spec_gap, rho */

/* Checkpoint magic numbers */
#define SD_MAGIC_CLIMBER   0x53434C42  /* 'SCLB' */
#define SD_MAGIC_NAVIGATOR 0x534E4156  /* 'SNAV' */
#define SD_CKPT_VERSION    1

/* ========================================================================== */
/* Activation functions                                                        */
/* ========================================================================== */

static inline float sd_silu(float x) { return x / (1.0f + expf(-x)); }
static inline float sd_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float sd_silu_d(float x) {
    float s = sd_sigmoid(x);
    return s * (1.0f + x * (1.0f - s));
}

/* ========================================================================== */
/* Bridge state (incremental bridge detection)                                 */
/* ========================================================================== */

typedef struct {
    int *parent;
    int *depth;
    uint8_t *is_tree;
    int *cover;
    uint8_t *bridge;
    int n;
} SdBridgeState;

/* ========================================================================== */
/* Graph state (all incrementally maintained)                                  */
/* ========================================================================== */

typedef struct {
    uint8_t *adj;       /* n*n adjacency matrix */
    int *deg;           /* n degrees */
    int *tri;           /* n triangle counts per node */
    int *snd;           /* n sum-of-neighbor-degrees */
    int *cn;            /* n*n common neighbor counts */
    double *lpinv;      /* n*n Laplacian pseudoinverse */
    double *v2;         /* n Fiedler vector */
    double *v2_warm;    /* n warm Lanczos state */
    double v2_lam;      /* lambda2 estimate */
    double v3_lam;      /* lambda3 estimate (for spectral gap) */
    SdBridgeState bs;
    int n, m;
} GraphState;

/* ========================================================================== */
/* Edge swap candidate                                                         */
/* ========================================================================== */

typedef struct {
    int add_i, add_j;
    int rem_i, rem_j;
    double score;       /* climber's net FV score */
    double add_gap;     /* n * (v2[i] - v2[j])^2 for added edge */
    double rem_gap;     /* n * (v2[i] - v2[j])^2 for removed edge */
    double reff_add;    /* n * R_eff(add_i, add_j) */
    double reff_rem;    /* n * R_eff(rem_i, rem_j) */
} EdgeCandidate;

/* ========================================================================== */
/* Climber weights (shared embedder + 4 scorer heads)                          */
/* ========================================================================== */

typedef struct {
    /* Shared node embedder: NE_IN -> H -> H */
    float ne_w0[SD_H * SD_NE_IN]; float ne_b0[SD_H];
    float ne_w1[SD_H * SD_H];     float ne_b1[SD_H];
    /* ADD edge scorer: ES_IN -> H -> 1 */
    float ae_w0[SD_H * SD_ES_IN]; float ae_b0[SD_H]; float ae_w1[SD_H]; float ae_b1[1];
    /* REM edge scorer: ES_IN -> H -> 1 */
    float re_w0[SD_H * SD_ES_IN]; float re_b0[SD_H]; float re_w1[SD_H]; float re_b1[1];
} ClimberWeights;

typedef struct {
    float ne_w0[SD_H * SD_NE_IN]; float ne_b0[SD_H];
    float ne_w1[SD_H * SD_H];     float ne_b1[SD_H];
    float ae_w0[SD_H * SD_ES_IN]; float ae_b0[SD_H]; float ae_w1[SD_H]; float ae_b1[1];
    float re_w0[SD_H * SD_ES_IN]; float re_b0[SD_H]; float re_w1[SD_H]; float re_b1[1];
} ClimberGrads;

/* Caches for backward pass */
typedef struct { float in_val[SD_NE_IN]; float pre0[SD_H]; float h0[SD_H]; float pre1[SD_H]; } NECache;
typedef struct { float input[SD_ES_IN]; float pre[SD_H]; float hid[SD_H]; } ESCache;

/* ========================================================================== */
/* Navigator weights (small MLP: NAV_IN -> NH -> NH -> 1)                      */
/* ========================================================================== */

typedef struct {
    float w0[SD_NAV_NH * SD_NAV_IN]; float b0[SD_NAV_NH];
    float w1[SD_NAV_NH * SD_NAV_NH]; float b1[SD_NAV_NH];
    float w2[SD_NAV_NH];             float b2[1];
} NavigatorWeights;

typedef struct {
    float w0[SD_NAV_NH * SD_NAV_IN]; float b0[SD_NAV_NH];
    float w1[SD_NAV_NH * SD_NAV_NH]; float b1[SD_NAV_NH];
    float w2[SD_NAV_NH];             float b2[1];
} NavigatorGrads;

/* Navigator detection features */
typedef struct {
    float entropy;     /* normalized softmax entropy of ADD scores */
    float l2_n;        /* v2_lam / n (graph quality) */
    float mean_deg_n;  /* rho = (m - m_min) / (m_max - m_min) */
    float spec_gap;    /* (lambda3 - lambda2) / n */
    float d_entropy;   /* entropy[t] - entropy[t-1] (transition signal) */
    int valid;
} NavFeatures;

/* ========================================================================== */
/* Saddle stack (DFS backtracking)                                             */
/* ========================================================================== */

typedef struct {
    GraphState *saved_gs;
    EdgeCandidate cands[SD_MAX_CAND];
    int n_cands;
    int tried;
} SaddleFrame;

typedef struct {
    SaddleFrame frames[SD_MAX_STACK];
    int sp;
} SaddleStack;

/* ========================================================================== */
/* Training config                                                             */
/* ========================================================================== */

typedef struct {
    int min_n, max_n;
    int num_episodes;
    int budget_mult;        /* epochs = budget_mult * n */
    double lr;
    double tau_kl;          /* temperature for softmax target */
    double weight_decay;
    double grad_clip;
    uint64_t seed;
    char save_dir[256];
    int log_freq;
    int ckpt_freq;
} SaddleTrainConfig;

/* ========================================================================== */
/* features.c                                                                  */
/* ========================================================================== */

GraphState *graph_state_alloc(int n);
void        graph_state_free(GraphState *gs);
void        graph_state_init(GraphState *gs, const uint8_t *adj, int n, int m);
GraphState *graph_state_clone(const GraphState *gs);
void        graph_state_restore(GraphState *dst, const GraphState *src);
void        graph_state_add_edge(GraphState *gs, int u, int v);
void        graph_state_rem_edge(GraphState *gs, int u, int v);
void        lpinv_refresh(GraphState *gs);

/* ========================================================================== */
/* climber.c                                                                   */
/* ========================================================================== */

void  climber_init_weights(ClimberWeights *w, uint64_t seed);
void  climber_embed_all(const ClimberWeights *w, const GraphState *gs, float *hh);

/* Edge-level scoring: returns count of candidates, fills scores/edges arrays.
 * edges[k][0]=i, edges[k][1]=j for k-th candidate. */
int   climber_score_add_edges(const ClimberWeights *w, const float *hh,
                              const GraphState *gs, float *scores, int edges[][2]);
int   climber_score_rem_edges(const ClimberWeights *w, const float *hh,
                              const GraphState *gs, float *scores, int edges[][2]);
int   climber_greedy_pick(const float *scores, int count);

/* Cached versions for training */
void  climber_embed_all_cached(const ClimberWeights *w, const GraphState *gs,
                               float *hh, NECache *nec);
float climber_edge_fwd_cached(const float *hi, const float *hj,
                               float cn, float fv_gap, float reff,
                               const float *W0, const float *b0,
                               const float *W1, const float *b1, ESCache *ec);
void  climber_edge_bwd(const ESCache *c, const float *W0, const float *W1,
                       float d_out, float *dW0, float *db0, float *dW1, float *db1,
                       float *d_hi, float *d_hj);
void  climber_embed_bwd(const ClimberWeights *w, const NECache *c,
                        const float *d_h, ClimberGrads *pg);

/* ========================================================================== */
/* navigator.c                                                                 */
/* ========================================================================== */

void  navigator_init_weights(NavigatorWeights *w, uint64_t seed);
void  navigator_compute_features(const float *add_scores, int n_add,
                                 const float *rem_scores, int n_rem,
                                 const GraphState *gs, const NavFeatures *prev,
                                 NavFeatures *out);
float navigator_score_candidate(const NavigatorWeights *w, const NavFeatures *nf,
                                const EdgeCandidate *cand, int rank, int n_cands);
int   navigator_detect_saddle(const float *saddle_scores, int n_cands, float threshold);

void  stack_init(SaddleStack *st);
void  stack_push(SaddleStack *st, const GraphState *gs,
                 const EdgeCandidate *cands, int n_cands);
int   stack_backtrack(SaddleStack *st, GraphState *gs, EdgeCandidate *next_cand);
void  stack_free(SaddleStack *st);

/* ========================================================================== */
/* episode.c                                                                   */
/* ========================================================================== */

double saddle_episode(const ClimberWeights *cw, const NavigatorWeights *nw,
                      int n, int m, int budget_mult, float nav_threshold,
                      uint64_t seed);

/* God navigator: exhaustive branch eval at every step. Measures climber ceiling. */
double god_episode(const ClimberWeights *cw, int n, int m,
                   int budget_mult, int n_branches, uint64_t seed);

/* ========================================================================== */
/* train_climber.c                                                             */
/* ========================================================================== */

void train_climber(SaddleTrainConfig *cfg);

/* ========================================================================== */
/* train_navigator.c                                                           */
/* ========================================================================== */

void train_navigator(const ClimberWeights *frozen_cw, SaddleTrainConfig *cfg);

/* ========================================================================== */
/* weights.c                                                                   */
/* ========================================================================== */

int  save_climber(const char *path, const ClimberWeights *w, uint32_t episode);
int  load_climber(const char *path, ClimberWeights *w, uint32_t *episode);
int  save_navigator(const char *path, const NavigatorWeights *w, uint32_t episode);
int  load_navigator(const char *path, NavigatorWeights *w, uint32_t *episode);

#endif /* SADDLE_H */
