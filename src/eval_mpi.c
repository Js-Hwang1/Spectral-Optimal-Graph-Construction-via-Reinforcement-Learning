/*
 * eval_mpi.c — MPI-parallel inference for CRL.
 *
 * Master-worker dynamic dispatch: master distributes (n,m) jobs to workers.
 * Workers run mlp_episode and report results back.
 *
 * Usage:
 *   mpirun -np 64 ./crl_eval_mpi --checkpoint model.bin --n 16 --baselines bl.csv --swaps 3 --epoch-C 1.0 --tabu 1
 *
 * Master (rank 0): reads baselines, builds job list, dispatches to workers, collects results.
 * Workers (rank 1..NP-1): receive (n,m), run inference, send back l2.
 */

#include "crl.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

#define H 64
#define NE_IN 5
#define TS_IN (H + 1)
#define MLP_MAGIC 0x4D4C503F

static inline float silu_f(float x) { return x / (1.0f + expf(-x)); }

static double wall_time(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* ========================================================================== */
/* Weights (must match test.c exactly)                                         */
/* ========================================================================== */

typedef struct {
    float ne_w0[H * NE_IN]; float ne_b0[H];
    float ne_w1[H * H]; float ne_b1[H];
    float an_w0[H * H]; float an_b0[H]; float an_w1[H]; float an_b1[1];
    float at_w0[H * TS_IN]; float at_b0[H]; float at_w1[H]; float at_b1[1];
    float rn_w0[H * H]; float rn_b0[H]; float rn_w1[H]; float rn_b1[1];
    float rt_w0[H * TS_IN]; float rt_b0[H]; float rt_w1[H]; float rt_b1[1];
} MLPWeights;

static int load_weights(const char *path, MLPWeights *w) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    uint32_t magic; fread(&magic, 4, 1, f);
    if (magic != MLP_MAGIC) { fclose(f); return -2; }
    fread(w, sizeof(MLPWeights), 1, f); fclose(f); return 0;
}

/* ========================================================================== */
/* Forward functions (copied from test.c — must stay in sync)                  */
/* ========================================================================== */

/* --- Graph state --- */
static void init_tri_snd(const uint8_t *adj, const int *deg, int n, int *tri, int *snd) {
    memset(tri, 0, (size_t)n * sizeof(int));
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            if (!adj[i * n + j]) continue;
            for (int k = j + 1; k < n; k++)
                if (adj[i * n + k] && adj[j * n + k]) { tri[i]++; tri[j]++; tri[k]++; }
        }
    for (int i = 0; i < n; i++) {
        snd[i] = 0;
        for (int j = 0; j < n; j++) if (adj[i * n + j]) snd[i] += deg[j];
    }
}

static void update_tri_snd_add(const uint8_t *adj, const int *deg, int n, int u, int v, int *tri, int *snd) {
    for (int k = 0; k < n; k++) if (k!=u && k!=v && adj[u*n+k] && adj[v*n+k]) { tri[u]++; tri[v]++; tri[k]++; }
    snd[u] += deg[v]; snd[v] += deg[u];
    for (int k = 0; k < n; k++) { if (k==u||k==v) continue; if (adj[u*n+k]) snd[k]++; if (adj[v*n+k]) snd[k]++; }
}

static void update_tri_snd_rem(const uint8_t *adj, const int *deg, int n, int u, int v, int *tri, int *snd) {
    for (int k = 0; k < n; k++) if (k!=u && k!=v && adj[u*n+k] && adj[v*n+k]) { tri[u]--; tri[v]--; tri[k]--; }
    snd[u] -= deg[v]; snd[v] -= deg[u];
    for (int k = 0; k < n; k++) { if (k==u||k==v) continue; if (adj[u*n+k]) snd[k]--; if (adj[v*n+k]) snd[k]--; }
}

static void init_cn(const uint8_t *adj, int n, int *cn) {
    memset(cn, 0, (size_t)n * n * sizeof(int));
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            int c = 0;
            for (int k = 0; k < n; k++) if (adj[i*n+k] && adj[j*n+k]) c++;
            cn[i*n+j] = cn[j*n+i] = c;
        }
}

static void update_cn_add(const uint8_t *adj, int n, int u, int v, int *cn) {
    for (int k = 0; k < n; k++) {
        if (k==u||k==v) continue;
        if (adj[v*n+k]) { cn[u*n+k]++; cn[k*n+u]++; }
        if (adj[u*n+k]) { cn[v*n+k]++; cn[k*n+v]++; }
    }
}

static void update_cn_rem(const uint8_t *adj, int n, int u, int v, int *cn) {
    for (int k = 0; k < n; k++) {
        if (k==u||k==v) continue;
        if (adj[v*n+k]) { cn[u*n+k]--; cn[k*n+u]--; }
        if (adj[u*n+k]) { cn[v*n+k]--; cn[k*n+v]--; }
    }
}

/* --- Bridge state (same as test.c) --- */
typedef struct {
    int *parent, *depth, *cover, n;
    uint8_t *is_tree, *bridge;
} BridgeState;

static void bridge_alloc(int n, BridgeState *bs) {
    bs->n = n;
    bs->parent = (int *)malloc((size_t)n * sizeof(int));
    bs->depth = (int *)malloc((size_t)n * sizeof(int));
    bs->is_tree = (uint8_t *)calloc((size_t)n * n, 1);
    bs->cover = (int *)calloc((size_t)n, sizeof(int));
    bs->bridge = (uint8_t *)calloc((size_t)n * n, 1);
}

static void bridge_free(BridgeState *bs) {
    free(bs->parent); free(bs->depth); free(bs->is_tree); free(bs->cover); free(bs->bridge);
}

static void bridge_init(const uint8_t *adj, int n, BridgeState *bs) {
    memset(bs->is_tree, 0, (size_t)n*n); memset(bs->cover, 0, (size_t)n*sizeof(int)); memset(bs->bridge, 0, (size_t)n*n);
    for (int i = 0; i < n; i++) { bs->parent[i] = -1; bs->depth[i] = -1; }
    int *queue = (int *)malloc((size_t)n * sizeof(int));
    int qh = 0, qt = 0;
    bs->depth[0] = 0; bs->parent[0] = -1; queue[qt++] = 0;
    while (qh < qt) {
        int u = queue[qh++];
        for (int v = 0; v < n; v++)
            if (adj[u*n+v] && bs->depth[v] < 0) {
                bs->parent[v] = u; bs->depth[v] = bs->depth[u]+1;
                bs->is_tree[u*n+v] = bs->is_tree[v*n+u] = 1; queue[qt++] = v;
            }
    }
    free(queue);
    for (int i = 0; i < n; i++)
        for (int j = i+1; j < n; j++) {
            if (!adj[i*n+j] || bs->is_tree[i*n+j]) continue;
            int u = i, v = j;
            while (bs->depth[u] > bs->depth[v]) { bs->cover[u]++; u = bs->parent[u]; }
            while (bs->depth[v] > bs->depth[u]) { bs->cover[v]++; v = bs->parent[v]; }
            while (u != v) { bs->cover[u]++; bs->cover[v]++; u = bs->parent[u]; v = bs->parent[v]; }
        }
    for (int v = 1; v < n; v++) {
        int p = bs->parent[v];
        if (bs->cover[v] == 0) bs->bridge[p*n+v] = bs->bridge[v*n+p] = 1;
    }
}

static void bridge_add_edge(int n, int u, int v, BridgeState *bs) {
    int a = u, b = v;
    while (bs->depth[a] > bs->depth[b]) { bs->cover[a]++; if (bs->cover[a]==1) { int p=bs->parent[a]; bs->bridge[p*n+a]=bs->bridge[a*n+p]=0; } a=bs->parent[a]; }
    while (bs->depth[b] > bs->depth[a]) { bs->cover[b]++; if (bs->cover[b]==1) { int p=bs->parent[b]; bs->bridge[p*n+b]=bs->bridge[b*n+p]=0; } b=bs->parent[b]; }
    while (a != b) {
        bs->cover[a]++; if (bs->cover[a]==1) { int p=bs->parent[a]; bs->bridge[p*n+a]=bs->bridge[a*n+p]=0; }
        bs->cover[b]++; if (bs->cover[b]==1) { int p=bs->parent[b]; bs->bridge[p*n+b]=bs->bridge[b*n+p]=0; }
        a=bs->parent[a]; b=bs->parent[b];
    }
}

static void bridge_rem_edge(const uint8_t *adj, int n, int u, int v, BridgeState *bs) {
    if (!bs->is_tree[u*n+v]) {
        int a = u, b = v;
        while (bs->depth[a] > bs->depth[b]) { bs->cover[a]--; if (bs->cover[a]==0) { int p=bs->parent[a]; bs->bridge[p*n+a]=bs->bridge[a*n+p]=1; } a=bs->parent[a]; }
        while (bs->depth[b] > bs->depth[a]) { bs->cover[b]--; if (bs->cover[b]==0) { int p=bs->parent[b]; bs->bridge[p*n+b]=bs->bridge[b*n+p]=1; } b=bs->parent[b]; }
        while (a != b) {
            bs->cover[a]--; if (bs->cover[a]==0) { int p=bs->parent[a]; bs->bridge[p*n+a]=bs->bridge[a*n+p]=1; }
            bs->cover[b]--; if (bs->cover[b]==0) { int p=bs->parent[b]; bs->bridge[p*n+b]=bs->bridge[b*n+p]=1; }
            a=bs->parent[a]; b=bs->parent[b];
        }
    } else {
        bridge_init(adj, n, bs);
    }
}

/* --- MLP forward --- */
static void node_embed_all(const MLPWeights *w, const int *deg, const int *tri,
                            const int *snd, int n, float nm1, float step_f,
                            float tgt_dens, float *hh) {
    float nm1sq = nm1 * nm1;
    for (int i = 0; i < n; i++) {
        float d = (float)deg[i] / nm1;
        float mt = (float)(deg[i]*(deg[i]-1)) / 2.0f;
        float tc = mt > 0.5f ? (float)tri[i]/mt : 0.0f;
        float sn = nm1sq > 0.5f ? (float)snd[i]/nm1sq : 0.0f;
        float feats[NE_IN] = {d, step_f, tc, sn, tgt_dens};
        float h0[H];
        for (int k = 0; k < H; k++) { float s = w->ne_b0[k]; for (int dd=0; dd<NE_IN; dd++) s += w->ne_w0[k*NE_IN+dd]*feats[dd]; h0[k] = silu_f(s); }
        for (int k = 0; k < H; k++) { float s = w->ne_b1[k]; for (int j=0; j<H; j++) s += w->ne_w1[k*H+j]*h0[j]; hh[i*H+k] = silu_f(s); }
    }
}

static float scorer_fwd(const float *h, const float *W0, const float *b0, const float *W1, const float *b1) {
    float hid[H];
    for (int i = 0; i < H; i++) { float s = b0[i]; for (int j=0; j<H; j++) s += W0[i*H+j]*h[j]; hid[i] = silu_f(s); }
    float out = b1[0]; for (int j=0; j<H; j++) out += W1[j]*hid[j]; return out;
}

static float target_scorer_fwd(const float *h_tgt, float cn_norm, const float *W0, const float *b0, const float *W1, const float *b1) {
    float input[TS_IN]; memcpy(input, h_tgt, H*sizeof(float)); input[H] = cn_norm;
    float hid[H];
    for (int i = 0; i < H; i++) { float s = b0[i]; for (int j=0; j<TS_IN; j++) s += W0[i*TS_IN+j]*input[j]; hid[i] = silu_f(s); }
    float out = b1[0]; for (int j=0; j<H; j++) out += W1[j]*hid[j]; return out;
}

/* ========================================================================== */
/* Inference episode (same logic as test.c mlp_episode)                        */
/* ========================================================================== */

static double mlp_episode(const MLPWeights *w, int n, int m,
                           int swaps, double epoch_C, int tabu_mult, uint64_t seed) {
    int nn = n * n;
    uint8_t *adj = (uint8_t *)calloc((size_t)nn, 1);
    int *degrees = (int *)calloc((size_t)n, sizeof(int));
    int *tri = (int *)calloc((size_t)n, sizeof(int));
    int *snd_arr = (int *)calloc((size_t)n, sizeof(int));
    int *cn = (int *)calloc((size_t)nn, sizeof(int));
    RNG rng; rng_init(&rng, seed);
    build_ring_random(adj, degrees, n, m, &rng);
    init_tri_snd(adj, degrees, n, tri, snd_arr);
    init_cn(adj, n, cn);
    float nm1 = (float)(n-1); if (nm1 < 1.0f) nm1 = 1.0f;
    float spe_inv = 1.0f / (float)swaps;
    float inv_n = 1.0f / (float)n;
    int max_m = n*(n-1)/2;
    float tgt_dens = (float)(m-(n-1)) / (float)(max_m-(n-1));
    int num_epochs = (int)(epoch_C * (max_m - m));
    if (num_epochs < 4) num_epochs = 4;

    float *hh = (float *)malloc((size_t)n * H * sizeof(float));
    float *scores = (float *)malloc((size_t)n * sizeof(float));
    int *tabu = (int *)calloc((size_t)nn, sizeof(int));
    int tabu_len = tabu_mult > 0 ? tabu_mult * n : 0;
    BridgeState bs; bridge_alloc(n, &bs); bridge_init(adj, n, &bs);

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        int added_edges[3][2]; int n_added = 0;
        /* ADD */
        for (int s = 0; s < swaps; s++) {
            float step_f = (float)s * spe_inv;
            node_embed_all(w, degrees, tri, snd_arr, n, nm1, step_f, tgt_dens, hh);
            for (int i = 0; i < n; i++) scores[i] = scorer_fwd(hh+i*H, w->an_w0, w->an_b0, w->an_w1, w->an_b1);
            int sel = -1; float bv = -FLT_MAX;
            for (int i = 0; i < n; i++) { if (degrees[i] < n-1 && scores[i] > bv) { bv = scores[i]; sel = i; } }
            if (sel < 0) break;
            float tscores[1024]; int tn[1024]; int nt = 0;
            for (int j = 0; j < n; j++) {
                if (j==sel || adj[sel*n+j] || tabu[sel*n+j] > epoch) continue;
                float cn_norm = (float)cn[sel*n+j] * inv_n;
                tscores[nt] = target_scorer_fwd(hh+j*H, cn_norm, w->at_w0, w->at_b0, w->at_w1, w->at_b1);
                tn[nt++] = j;
            }
            if (nt == 0) break;
            int best_t = 0; for (int t=1; t<nt; t++) if (tscores[t]>tscores[best_t]) best_t=t;
            int tgt = tn[best_t];
            int u = sel<tgt?sel:tgt, v = sel<tgt?tgt:sel;
            adj[u*n+v] = adj[v*n+u] = 1; degrees[u]++; degrees[v]++;
            update_tri_snd_add(adj, degrees, n, u, v, tri, snd_arr);
            update_cn_add(adj, n, u, v, cn);
            bridge_add_edge(n, u, v, &bs);
            if (n_added<3) { added_edges[n_added][0]=u; added_edges[n_added][1]=v; n_added++; }
        }
        /* REM */
        for (int s = 0; s < swaps; s++) {
            float step_f = (float)s * spe_inv;
            node_embed_all(w, degrees, tri, snd_arr, n, nm1, step_f, tgt_dens, hh);
            for (int i = 0; i < n; i++) scores[i] = scorer_fwd(hh+i*H, w->rn_w0, w->rn_b0, w->rn_w1, w->rn_b1);
            int sel = -1; float bv = -FLT_MAX;
            for (int i = 0; i < n; i++) { if (degrees[i]>0 && scores[i]>bv) { bv=scores[i]; sel=i; } }
            if (sel < 0) break;
            float tscores[1024]; int tn[1024]; int nt = 0;
            for (int j = 0; j < n; j++) {
                if (j==sel || !adj[sel*n+j] || bs.bridge[sel*n+j]) continue;
                float cn_norm = (float)cn[sel*n+j] * inv_n;
                tscores[nt] = target_scorer_fwd(hh+j*H, cn_norm, w->rt_w0, w->rt_b0, w->rt_w1, w->rt_b1);
                tn[nt++] = j;
            }
            if (nt == 0) break;
            int best_t = 0; for (int t=1; t<nt; t++) if (tscores[t]>tscores[best_t]) best_t=t;
            int tgt = tn[best_t];
            int u = sel<tgt?sel:tgt, v = sel<tgt?tgt:sel;
            bridge_rem_edge(adj, n, u, v, &bs);
            update_cn_rem(adj, n, u, v, cn);
            update_tri_snd_rem(adj, degrees, n, u, v, tri, snd_arr);
            adj[u*n+v] = adj[v*n+u] = 0; degrees[u]--; degrees[v]--;
        }
        /* Tabu */
        for (int k = 0; k < n_added; k++) {
            int u = added_edges[k][0], v = added_edges[k][1];
            if (!adj[u*n+v]) tabu[u*n+v] = tabu[v*n+u] = epoch + tabu_len;
        }
    }
    double final_l2 = exact_lambda2(adj, n);
    free(adj); free(degrees); free(tri); free(snd_arr); free(cn);
    free(hh); free(scores); free(tabu); bridge_free(&bs);
    return final_l2;
}

/* ========================================================================== */
/* Baselines                                                                   */
/* ========================================================================== */

typedef struct { int n, m; double fv, er, sw025, sw050, sw075, best; } BL;
static BL *bls = NULL; static int nbl = 0;

static void load_bl(const char *path) {
    FILE *f = fopen(path, "r"); if (!f) return;
    char line[1024]; fgets(line, sizeof(line), f);
    int cap = 8192; bls = (BL *)malloc((size_t)cap * sizeof(BL));
    while (fgets(line, sizeof(line), f)) {
        BL e = {0};
        sscanf(line, "%d,%d,%lf,%lf,%lf,%lf,%lf", &e.n, &e.m, &e.fv, &e.er, &e.sw025, &e.sw050, &e.sw075);
        e.best = e.fv; if (e.er>e.best) e.best=e.er; if (e.sw025>e.best) e.best=e.sw025;
        if (e.sw050>e.best) e.best=e.sw050; if (e.sw075>e.best) e.best=e.sw075;
        if (nbl >= cap) { cap *= 2; bls = realloc(bls, (size_t)cap * sizeof(BL)); }
        bls[nbl++] = e;
    }
    fclose(f);
}

static BL *find_bl(int n, int m) {
    for (int i = 0; i < nbl; i++) if (bls[i].n==n && bls[i].m==m) return &bls[i];
    return NULL;
}

/* ========================================================================== */
/* MPI tags                                                                    */
/* ========================================================================== */

#define TAG_JOB   1   /* master → worker: (n, m) job */
#define TAG_DONE  2   /* worker → master: (job_idx, l2) result */
#define TAG_STOP  3   /* master → worker: no more jobs, exit */

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    setbuf(stdout, NULL);  /* unbuffered stdout for MPI */
    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    /* Parse args (all ranks) */
    int n_values[64], num_n = 0;
    char *ckpt = NULL, *bl_path = NULL, *csv_path = NULL;
    int swaps = 3, tabu_mult = 1;
    double epoch_C = 1.0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--n") == 0 && i+1 < argc) {
            char *tok = strtok(argv[++i], ",");
            while (tok && num_n < 64) { n_values[num_n++] = atoi(tok); tok = strtok(NULL, ","); }
        }
        else if (strcmp(argv[i], "--checkpoint") == 0 && i+1 < argc) ckpt = argv[++i];
        else if (strcmp(argv[i], "--baselines") == 0 && i+1 < argc) bl_path = argv[++i];
        else if (strcmp(argv[i], "--swaps") == 0 && i+1 < argc) swaps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--epoch-C") == 0 && i+1 < argc) epoch_C = atof(argv[++i]);
        else if (strcmp(argv[i], "--tabu") == 0 && i+1 < argc) tabu_mult = atoi(argv[++i]);
        else if (strcmp(argv[i], "--csv") == 0 && i+1 < argc) csv_path = argv[++i];
    }

    /* Load weights (all ranks) */
    MLPWeights w;
    if (!ckpt) { if (rank==0) fprintf(stderr, "Need --checkpoint\n"); MPI_Finalize(); return 1; }
    if (load_weights(ckpt, &w) != 0) { if (rank==0) fprintf(stderr, "Failed to load %s\n", ckpt); MPI_Finalize(); return 1; }

    if (rank == 0) {
        /* === MASTER === */
        if (bl_path) load_bl(bl_path);

        /* Build job list: all (n,m) pairs */
        int max_jobs = 600000;
        int *job_n = (int *)malloc((size_t)max_jobs * sizeof(int));
        int *job_m = (int *)malloc((size_t)max_jobs * sizeof(int));
        double *job_l2 = (double *)calloc((size_t)max_jobs, sizeof(double));
        int njobs = 0;

        for (int ni = 0; ni < num_n; ni++) {
            int n = n_values[ni], mx = n*(n-1)/2;
            for (int mv = n-1; mv <= mx; mv++) {
                BL *b = find_bl(n, mv);
                if (b && b->best > 0) {
                    job_n[njobs] = n; job_m[njobs] = mv; njobs++;
                }
            }
        }

        printf("Master: %d jobs, %d workers\n", njobs, nproc-1); fflush(stdout);
        double t0 = wall_time();

        /* Track which workers have been stopped */
        int *stopped = (int *)calloc((size_t)nproc, sizeof(int));

        /* Initial dispatch: send one job to each worker */
        int next_job = 0;
        for (int r = 1; r < nproc; r++) {
            if (next_job < njobs) {
                int buf[3] = {job_n[next_job], job_m[next_job], next_job};
                MPI_Send(buf, 3, MPI_INT, r, TAG_JOB, MPI_COMM_WORLD);
                next_job++;
            } else {
                /* No job for this worker, stop immediately */
                int buf[3] = {0, 0, -1};
                MPI_Send(buf, 3, MPI_INT, r, TAG_STOP, MPI_COMM_WORLD);
                stopped[r] = 1;
            }
        }

        /* Collect results and dispatch more */
        int completed = 0;
        while (completed < njobs) {
            double result[2];
            MPI_Status status;
            MPI_Recv(result, 2, MPI_DOUBLE, MPI_ANY_SOURCE, TAG_DONE, MPI_COMM_WORLD, &status);
            int src = status.MPI_SOURCE;
            int idx = (int)result[0];
            job_l2[idx] = result[1];
            completed++;

            if (completed % 500 == 0)
                printf("  %d/%d completed (%.1fs)\n", completed, njobs, wall_time() - t0);

            /* Send next job or stop signal */
            if (next_job < njobs) {
                int buf[3] = {job_n[next_job], job_m[next_job], next_job};
                MPI_Send(buf, 3, MPI_INT, src, TAG_JOB, MPI_COMM_WORLD);
                next_job++;
            } else {
                int buf[3] = {0, 0, -1};
                MPI_Send(buf, 3, MPI_INT, src, TAG_STOP, MPI_COMM_WORLD);
                stopped[src] = 1;
            }
        }
        free(stopped);

        double elapsed = wall_time() - t0;

        /* Report results per n */
        FILE *csv_f = NULL;
        if (csv_path) { csv_f = fopen(csv_path, "w"); if (csv_f) fprintf(csv_f, "n,m,rl_l2,fv_l2,best_l2,density\n"); }

        for (int ni = 0; ni < num_n; ni++) {
            int n = n_values[ni], mx = n*(n-1)/2;
            double sr = 0, sf = 0, sb = 0; int wf = 0, wb = 0, tot = 0;
            double bin_rl[10]={0}, bin_best[10]={0}; int bin_w[10]={0}, bin_cnt[10]={0};

            for (int idx = 0; idx < njobs; idx++) {
                if (job_n[idx] != n) continue;
                BL *b = find_bl(n, job_m[idx]); if (!b) continue;
                double l2 = job_l2[idx];
                sr += l2; sf += b->fv; sb += b->best;
                if (l2 > b->fv + 1e-6) wf++;
                if (l2 > b->best + 1e-6) wb++;
                tot++;
                double density = (double)(job_m[idx]-(n-1)) / (double)(mx-(n-1));
                int bi = (int)(density * 10); if (bi >= 10) bi = 9; if (bi < 0) bi = 0;
                bin_rl[bi] += l2; bin_best[bi] += b->best;
                if (l2 > b->best + 1e-6) bin_w[bi]++;
                bin_cnt[bi]++;
                if (csv_f) fprintf(csv_f, "%d,%d,%.6f,%.6f,%.6f,%.4f\n", n, job_m[idx], l2, b->fv, b->best, density);
            }
            printf("\nn=%d (%d configs)\n", n, tot);
            printf("  vs FV: %.1f%% (%dW/%d) | vs Best: %.1f%% (%dW/%d)\n",
                   sf>0?sr/sf*100:0, wf, tot, sb>0?sr/sb*100:0, wb, tot);
            printf("  Density: ");
            for (int bi = 0; bi < 10; bi++) {
                if (bin_cnt[bi] == 0) continue;
                printf("%d0-%d0%%:%.1f%%(%dW/%d) ", bi, bi+1, bin_best[bi]>0?bin_rl[bi]/bin_best[bi]*100:0, bin_w[bi], bin_cnt[bi]);
            }
            printf("\n");
        }
        printf("\nTotal: %.1fs with %d workers\n", elapsed, nproc-1);
        if (csv_f) { fclose(csv_f); printf("Results saved to %s\n", csv_path); }

        free(job_n); free(job_m); free(job_l2);
        if (bls) free(bls);

    } else {
        /* === WORKER === */
        fprintf(stderr, "Worker %d ready\n", rank);
        while (1) {
            int buf[3];
            MPI_Status status;
            MPI_Recv(buf, 3, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == TAG_STOP) break;

            int n = buf[0], m = buf[1], idx = buf[2];
            fprintf(stderr, "Worker %d: job idx=%d n=%d m=%d\n", rank, idx, n, m);
            uint64_t seed = (uint64_t)(n * 100 + m);
            double l2 = mlp_episode(&w, n, m, swaps, epoch_C, tabu_mult, seed);
            fprintf(stderr, "Worker %d: done idx=%d l2=%.4f\n", rank, idx, l2);

            double result[2] = {(double)idx, l2};
            MPI_Send(result, 2, MPI_DOUBLE, 0, TAG_DONE, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
