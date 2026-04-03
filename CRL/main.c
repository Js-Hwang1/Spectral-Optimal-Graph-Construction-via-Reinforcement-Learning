/*
 * main.c — Full C PPO Training for G(n,m) algebraic connectivity.
 *
 * Usage: ./crl_train [options]
 *   --min-n 8 --max-n 16 --episodes 200000 --batch-size 8
 *   --k-steps 20 --swap-frac 0.25 --lr 3e-4 --gamma 0.99
 *   --gae-lambda 0.95 --clip-eps 0.2 --ppo-epochs 4
 *   --entropy-coef-start 0.1 --entropy-coef-end 0.01
 *   --grad-clip 1.0 --weight-decay 1e-4 --seed 42
 *   --save-dir logs/crl --checkpoint path/to/checkpoint.bin
 */

#include "crl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========================================================================== */
/* Timing                                                                      */
/* ========================================================================== */

static double wall_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* ========================================================================== */
/* Arg parsing                                                                 */
/* ========================================================================== */

static TrainConfig default_config(void) {
    TrainConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.min_n = 8;
    cfg.max_n = 16;
    cfg.num_episodes = 200000;
    cfg.k_steps = 20;
    cfg.batch_size = 8;
    cfg.swap_frac = 0.25;
    cfg.lr = 3e-4;
    cfg.gamma = 0.99;
    cfg.gae_lambda = 0.95;
    cfg.clip_eps = 0.2;
    cfg.ppo_epochs = 4;
    cfg.entropy_coef_start = 0.1;
    cfg.entropy_coef_end = 0.01;
    cfg.grad_clip = 1.0;
    cfg.weight_decay = 1e-4;
    cfg.seed = 42;
    cfg.log_freq = 10;
    cfg.ckpt_freq = 100;
    strcpy(cfg.save_dir, "logs/crl");
    return cfg;
}

static void parse_args(int argc, char **argv, TrainConfig *cfg, char *checkpoint_path) {
    checkpoint_path[0] = '\0';
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--min-n") == 0 && i+1 < argc) cfg->min_n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-n") == 0 && i+1 < argc) cfg->max_n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--episodes") == 0 && i+1 < argc) cfg->num_episodes = atoi(argv[++i]);
        else if (strcmp(argv[i], "--k-steps") == 0 && i+1 < argc) cfg->k_steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch-size") == 0 && i+1 < argc) cfg->batch_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--swap-frac") == 0 && i+1 < argc) cfg->swap_frac = atof(argv[++i]);
        else if (strcmp(argv[i], "--lr") == 0 && i+1 < argc) cfg->lr = atof(argv[++i]);
        else if (strcmp(argv[i], "--gamma") == 0 && i+1 < argc) cfg->gamma = atof(argv[++i]);
        else if (strcmp(argv[i], "--gae-lambda") == 0 && i+1 < argc) cfg->gae_lambda = atof(argv[++i]);
        else if (strcmp(argv[i], "--clip-eps") == 0 && i+1 < argc) cfg->clip_eps = atof(argv[++i]);
        else if (strcmp(argv[i], "--ppo-epochs") == 0 && i+1 < argc) cfg->ppo_epochs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--entropy-coef-start") == 0 && i+1 < argc) cfg->entropy_coef_start = atof(argv[++i]);
        else if (strcmp(argv[i], "--entropy-coef-end") == 0 && i+1 < argc) cfg->entropy_coef_end = atof(argv[++i]);
        else if (strcmp(argv[i], "--grad-clip") == 0 && i+1 < argc) cfg->grad_clip = atof(argv[++i]);
        else if (strcmp(argv[i], "--weight-decay") == 0 && i+1 < argc) cfg->weight_decay = atof(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc) cfg->seed = (uint64_t)atol(argv[++i]);
        else if (strcmp(argv[i], "--save-dir") == 0 && i+1 < argc) strncpy(cfg->save_dir, argv[++i], 255);
        else if (strcmp(argv[i], "--log-freq") == 0 && i+1 < argc) cfg->log_freq = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ckpt-freq") == 0 && i+1 < argc) cfg->ckpt_freq = atoi(argv[++i]);
        else if (strcmp(argv[i], "--checkpoint") == 0 && i+1 < argc) strncpy(checkpoint_path, argv[++i], 511);
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --min-n N          Min graph size (default: 8)\n");
            printf("  --max-n N          Max graph size (default: 16)\n");
            printf("  --episodes N       Total episodes (default: 200000)\n");
            printf("  --batch-size N     Episodes per update (default: 8)\n");
            printf("  --k-steps N        Lanczos snapshots per episode (default: 20)\n");
            printf("  --swap-frac F      Fraction of edges to swap (default: 0.25)\n");
            printf("  --lr F             Learning rate (default: 3e-4)\n");
            printf("  --gamma F          Discount factor (default: 0.99)\n");
            printf("  --gae-lambda F     GAE lambda (default: 0.95)\n");
            printf("  --clip-eps F       PPO clip epsilon (default: 0.2)\n");
            printf("  --ppo-epochs N     PPO epochs per update (default: 4)\n");
            printf("  --entropy-coef-start F  (default: 0.1)\n");
            printf("  --entropy-coef-end F    (default: 0.01)\n");
            printf("  --grad-clip F      Max gradient norm (default: 1.0)\n");
            printf("  --weight-decay F   AdamW weight decay (default: 1e-4)\n");
            printf("  --seed N           Random seed (default: 42)\n");
            printf("  --save-dir DIR     Output directory (default: logs/crl)\n");
            printf("  --log-freq N       Updates between log lines (default: 10)\n");
            printf("  --ckpt-freq N      Updates between checkpoints (default: 100)\n");
            printf("  --checkpoint PATH  Resume from checkpoint\n");
            exit(0);
        }
    }
}

/* ========================================================================== */
/* Recursive mkdir                                                             */
/* ========================================================================== */

static void mkdirs(const char *path) {
    char tmp[512];
    strncpy(tmp, path, sizeof(tmp) - 1);
    tmp[sizeof(tmp) - 1] = '\0';
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    mkdir(tmp, 0755);
}

/* ========================================================================== */
/* Main training loop                                                          */
/* ========================================================================== */

int main(int argc, char **argv) {
    TrainConfig cfg = default_config();
    char checkpoint_path[512] = {0};
    parse_args(argc, argv, &cfg, checkpoint_path);

    mkdirs(cfg.save_dir);

    /* Init policy */
    PolicyWeights pw;
    uint32_t start_episode = 0;
    float best_avg_improvement = -1e30f;

    if (checkpoint_path[0]) {
        int ret = load_checkpoint(checkpoint_path, &pw, &start_episode, &best_avg_improvement);
        if (ret != 0) {
            fprintf(stderr, "Failed to load checkpoint: %s (error %d)\n", checkpoint_path, ret);
            return 1;
        }
        printf("Resumed from %s (episode %u, best_imp=%.4f)\n",
               checkpoint_path, start_episode, best_avg_improvement);
    } else {
        RNG init_rng;
        rng_init(&init_rng, cfg.seed);
        policy_init_orthogonal(&pw, &init_rng, 0.5f);
    }

    /* Init optimizer */
    AdamState adam;
    adam_init(&adam);

    int num_updates = cfg.num_episodes / cfg.batch_size;
    int start_update = (int)start_episode / cfg.batch_size;

    /* Pre-allocate episode buffers */
    int max_n = cfg.max_n;
    int max_txns = cfg.k_steps * (max_n * (max_n - 1) / 2) + 100;
    int num_threads = omp_get_max_threads();
    printf("OpenMP threads: %d\n", num_threads);

    /* Print config */
    printf("======================================================================\n");
    printf("CRL — Full C PPO Training\n");
    printf("======================================================================\n");
    printf("Graph size: n in [%d, %d]\n", cfg.min_n, cfg.max_n);
    printf("K steps: %d | swap_frac: %.2f\n", cfg.k_steps, cfg.swap_frac);
    printf("Episodes: %d | Batch: %d | Updates: %d\n",
           cfg.num_episodes, cfg.batch_size, num_updates);
    printf("LR: %.1e | PPO epochs: %d | Clip: %.2f\n",
           cfg.lr, cfg.ppo_epochs, cfg.clip_eps);
    printf("Entropy coef: %.3f -> %.3f\n", cfg.entropy_coef_start, cfg.entropy_coef_end);
    printf("Grad clip: %.1f | Weight decay: %.1e\n", cfg.grad_clip, cfg.weight_decay);
    printf("Max txns/episode: %d\n", max_txns);
    printf("======================================================================\n");
    fflush(stdout);

    /* Allocate batch storage */
    int max_batch_txns = cfg.batch_size * max_txns;
    SwapTxn *all_add_txns = (SwapTxn *)malloc((size_t)max_batch_txns * sizeof(SwapTxn));
    SwapTxn *all_rem_txns = (SwapTxn *)malloc((size_t)max_batch_txns * sizeof(SwapTxn));
    double *all_values   = (double *)malloc((size_t)max_batch_txns * sizeof(double));
    double *all_old_lps  = (double *)malloc((size_t)max_batch_txns * sizeof(double));
    double *all_rewards  = (double *)malloc((size_t)max_batch_txns * sizeof(double));
    int    *all_n        = (int *)malloc((size_t)max_batch_txns * sizeof(int));
    int    *ep_starts    = (int *)malloc((size_t)(cfg.batch_size + 1) * sizeof(int));

    double *all_advantages = (double *)malloc((size_t)max_batch_txns * sizeof(double));
    double *all_returns    = (double *)malloc((size_t)max_batch_txns * sizeof(double));

    /* Per-thread episode buffers for parallel collection */
    SwapTxn **thread_ep_add = (SwapTxn **)malloc((size_t)cfg.batch_size * sizeof(SwapTxn *));
    SwapTxn **thread_ep_rem = (SwapTxn **)malloc((size_t)cfg.batch_size * sizeof(SwapTxn *));
    for (int b = 0; b < cfg.batch_size; b++) {
        thread_ep_add[b] = (SwapTxn *)malloc((size_t)max_txns * sizeof(SwapTxn));
        thread_ep_rem[b] = (SwapTxn *)malloc((size_t)max_txns * sizeof(SwapTxn));
    }

    if (!all_add_txns || !all_rem_txns || !all_values || !all_old_lps ||
        !all_rewards || !all_n || !ep_starts || !all_advantages ||
        !all_returns || !thread_ep_add || !thread_ep_rem) {
        fprintf(stderr, "Failed to allocate batch storage\n");
        return 1;
    }

    /* Rolling metrics */
    double rolling_l2_sum = 0, rolling_imp_sum = 0;
    int rolling_count = 0;

    RNG train_rng;
    rng_init(&train_rng, cfg.seed + 1000);

    double t_start = wall_time();
    uint32_t episodes_done = start_episode;

    PolicyGrad pg;

    for (int update = start_update; update < num_updates; update++) {
        double progress = (double)update / fmax(num_updates - 1, 1);

        /* Cosine LR schedule */
        double lr = cfg.lr * (0.1 + 0.9 * 0.5 * (1.0 + cos(M_PI * progress)));

        /* Linear entropy coefficient decay */
        double ent_coef = cfg.entropy_coef_start +
            (cfg.entropy_coef_end - cfg.entropy_coef_start) * progress;

        /* ============================================================ */
        /* 1. Collect batch of episodes (OpenMP parallel)                */
        /* ============================================================ */
        double t_collect = wall_time();
        int total_txns = 0;
        int valid_episodes = 0;
        double batch_l2_sum = 0, batch_imp_sum = 0;

        /* Pre-generate n, m, seed per episode (sequential — RNG is shared) */
        int batch_n[cfg.batch_size], batch_m[cfg.batch_size];
        uint64_t batch_seed[cfg.batch_size];
        for (int b = 0; b < cfg.batch_size; b++) {
            batch_n[b] = cfg.min_n + rng_int(&train_rng, cfg.max_n - cfg.min_n + 1);
            int max_m_b = batch_n[b] * (batch_n[b] - 1) / 2;
            int min_m_b = batch_n[b] - 1;
            batch_m[b] = min_m_b + rng_int(&train_rng, max_m_b - min_m_b + 1);
            batch_seed[b] = cfg.seed + episodes_done + (uint32_t)b;
        }

        /* Parallel episode collection */
        int ep_num_swaps[cfg.batch_size];
        double ep_metrics[cfg.batch_size][5];
        int ep_ret[cfg.batch_size];

        #pragma omp parallel for schedule(dynamic)
        for (int b = 0; b < cfg.batch_size; b++) {
            int out_na = 0, out_nr = 0;
            ep_ret[b] = crl_collect_episode(
                &pw, batch_n[b], batch_m[b], cfg.k_steps, cfg.swap_frac,
                batch_seed[b],
                thread_ep_add[b], thread_ep_rem[b],
                &out_na, &out_nr, ep_metrics[b], max_txns);
            ep_num_swaps[b] = out_na < out_nr ? out_na : out_nr;
        }

        /* Sequential merge into batch arrays */
        for (int b = 0; b < cfg.batch_size; b++) {
            ep_starts[b] = total_txns;
            if (ep_ret[b] != 0) continue;

            int ns = ep_num_swaps[b];
            for (int i = 0; i < ns && total_txns < max_batch_txns; i++) {
                memcpy(&all_add_txns[total_txns], &thread_ep_add[b][i], sizeof(SwapTxn));
                memcpy(&all_rem_txns[total_txns], &thread_ep_rem[b][i], sizeof(SwapTxn));
                all_values[total_txns] = thread_ep_add[b][i].value;
                all_old_lps[total_txns] = thread_ep_add[b][i].log_prob + thread_ep_rem[b][i].log_prob;
                all_rewards[total_txns] = thread_ep_add[b][i].reward;
                all_n[total_txns] = batch_n[b];
                total_txns++;
            }

            batch_l2_sum += ep_metrics[b][0];
            batch_imp_sum += ep_metrics[b][2];
            valid_episodes++;
        }
        ep_starts[cfg.batch_size] = total_txns;
        episodes_done += (uint32_t)cfg.batch_size;
        double t_collect_end = wall_time();

        if (total_txns == 0) continue;

        /* ============================================================ */
        /* 2. GAE per episode                                            */
        /* ============================================================ */
        for (int b = 0; b < cfg.batch_size; b++) {
            int start = ep_starts[b];
            int end = ep_starts[b + 1];
            int ep_len = end - start;
            if (ep_len == 0) continue;

            compute_gae(all_values + start, all_rewards + start, ep_len,
                        cfg.gamma, cfg.gae_lambda,
                        all_advantages + start, all_returns + start);
        }
        normalize_advantages(all_advantages, total_txns);

        /* ============================================================ */
        /* 3. PPO epochs (OpenMP parallel gradient accumulation)         */
        /* ============================================================ */
        double t_ppo = wall_time();
        for (int epoch = 0; epoch < cfg.ppo_epochs; epoch++) {
            policy_grad_zero(&pg);

            #pragma omp parallel
            {
                PolicyGrad local_pg;
                policy_grad_zero(&local_pg);

                #pragma omp for schedule(static)
                for (int t = 0; t < total_txns; t++) {
                    ppo_backward_swap(&pw,
                                      &all_add_txns[t], &all_rem_txns[t],
                                      all_n[t],
                                      all_advantages[t], all_returns[t],
                                      all_old_lps[t], ent_coef, cfg.clip_eps,
                                      &local_pg);
                }

                #pragma omp critical
                {
                    policy_grad_add(&pg, &local_pg);
                }
            }

            /* Average gradients */
            policy_grad_scale(&pg, 1.0f / total_txns);

            /* Clip gradients */
            policy_grad_clip(&pg, (float)cfg.grad_clip);

            /* AdamW step */
            adam_step(&pw, &pg, &adam, lr, 0.9, 0.999, 1e-8, cfg.weight_decay);
        }
        double t_ppo_end = wall_time();

        /* ============================================================ */
        /* 4. Metrics and logging                                        */
        /* ============================================================ */
        if (valid_episodes > 0) {
            rolling_l2_sum += batch_l2_sum;
            rolling_imp_sum += batch_imp_sum;
            rolling_count += valid_episodes;
        }

        /* Best checkpoint */
        if (rolling_count > 0) {
            float avg_imp = (float)(rolling_imp_sum / rolling_count);
            if (avg_imp > best_avg_improvement) {
                best_avg_improvement = avg_imp;
                char path[600];
                snprintf(path, sizeof(path), "%s/best.bin", cfg.save_dir);
                save_checkpoint(path, &pw, episodes_done, best_avg_improvement);
            }
        }

        if ((update + 1) % cfg.log_freq == 0 && rolling_count > 0) {
            double elapsed = wall_time() - t_start;
            double eps_per_sec = (double)(episodes_done - start_episode) / elapsed;
            double avg_l2 = rolling_l2_sum / rolling_count;
            double avg_imp = rolling_imp_sum / rolling_count;

            printf("[%5d/%d] ep=%u | l2=%.4f | dl2=%+.4f | best=%+.4f | "
                   "txns=%d | %.1f ep/s | collect=%.2fs ppo=%.2fs | "
                   "lr=%.1e | ent=%.3f | %.1fm\n",
                   update + 1, num_updates, episodes_done,
                   avg_l2, avg_imp, (double)best_avg_improvement,
                   total_txns, eps_per_sec,
                   t_collect_end - t_collect, t_ppo_end - t_ppo,
                   lr, ent_coef, elapsed / 60.0);
            fflush(stdout);

            /* Reset rolling metrics */
            rolling_l2_sum = 0;
            rolling_imp_sum = 0;
            rolling_count = 0;
        }

        /* Periodic checkpoint */
        if ((update + 1) % cfg.ckpt_freq == 0) {
            char path[600];
            snprintf(path, sizeof(path), "%s/checkpoint_%u.bin",
                     cfg.save_dir, episodes_done);
            save_checkpoint(path, &pw, episodes_done, best_avg_improvement);
        }
    }

    /* Final checkpoint */
    {
        char path[600];
        snprintf(path, sizeof(path), "%s/final.bin", cfg.save_dir);
        save_checkpoint(path, &pw, episodes_done, best_avg_improvement);
    }

    double total_time = wall_time() - t_start;
    printf("======================================================================\n");
    printf("CRL Training Complete!\n");
    printf("  Episodes: %u | Updates: %d | Time: %.1fm\n",
           episodes_done, num_updates, total_time / 60.0);
    printf("  Best avg improvement: %.4f\n", (double)best_avg_improvement);
    printf("  Saved to: %s\n", cfg.save_dir);
    printf("======================================================================\n");

    /* Cleanup */
    free(all_add_txns); free(all_rem_txns);
    free(all_values); free(all_old_lps); free(all_rewards);
    free(all_n); free(ep_starts);
    free(all_advantages); free(all_returns);
    for (int b = 0; b < cfg.batch_size; b++) {
        free(thread_ep_add[b]); free(thread_ep_rem[b]);
    }
    free(thread_ep_add); free(thread_ep_rem);

    return 0;
}
