/*
 * main.c — Dual-Phase Spectral Refine PPO Training.
 *
 * Trains add/remove MLPs via PPO on dual-phase episodes at n=8-24.
 *
 * Usage:
 *   ./crl_train --min-n 8 --max-n 16 --episodes 50000
 *   ./crl_train --checkpoint logs/best.bin --episodes 100000
 */

#include "crl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <sys/stat.h>

static double wall_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static TrainConfig default_config(void) {
    TrainConfig c;
    memset(&c, 0, sizeof(c));
    c.min_n = 8;
    c.max_n = 16;
    c.num_episodes = 50000;
    c.batch_size = 8;
    c.ppo_epochs = 4;
    c.cycle_mult = 20.0;
    c.swap_frac = 0.05;
    c.tau_start = 0.2;
    c.tau_end = 0.002;
    c.lr = 3e-4;
    c.gamma = 0.99;
    c.gae_lambda = 0.95;
    c.clip_eps = 0.2;
    c.entropy_coef_start = 0.05;
    c.entropy_coef_end = 0.005;
    c.grad_clip = 1.0;
    c.weight_decay = 1e-4;
    c.seed = 42;
    c.log_freq = 10;
    c.ckpt_freq = 100;
    snprintf(c.save_dir, sizeof(c.save_dir), "logs/dp_train");
    return c;
}

static void parse_args(TrainConfig *c, int argc, char **argv, char **ckpt_path) {
    *ckpt_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--min-n") == 0 && i+1 < argc) c->min_n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-n") == 0 && i+1 < argc) c->max_n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--episodes") == 0 && i+1 < argc) c->num_episodes = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch-size") == 0 && i+1 < argc) c->batch_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ppo-epochs") == 0 && i+1 < argc) c->ppo_epochs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--cycle-mult") == 0 && i+1 < argc) c->cycle_mult = atof(argv[++i]);
        else if (strcmp(argv[i], "--batch-frac") == 0 && i+1 < argc) c->swap_frac = atof(argv[++i]);
        else if (strcmp(argv[i], "--tau-start") == 0 && i+1 < argc) c->tau_start = atof(argv[++i]);
        else if (strcmp(argv[i], "--tau-end") == 0 && i+1 < argc) c->tau_end = atof(argv[++i]);
        else if (strcmp(argv[i], "--lr") == 0 && i+1 < argc) c->lr = atof(argv[++i]);
        else if (strcmp(argv[i], "--gamma") == 0 && i+1 < argc) c->gamma = atof(argv[++i]);
        else if (strcmp(argv[i], "--clip-eps") == 0 && i+1 < argc) c->clip_eps = atof(argv[++i]);
        else if (strcmp(argv[i], "--entropy-start") == 0 && i+1 < argc) c->entropy_coef_start = atof(argv[++i]);
        else if (strcmp(argv[i], "--entropy-end") == 0 && i+1 < argc) c->entropy_coef_end = atof(argv[++i]);
        else if (strcmp(argv[i], "--grad-clip") == 0 && i+1 < argc) c->grad_clip = atof(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc) c->seed = (uint64_t)atol(argv[++i]);
        else if (strcmp(argv[i], "--save-dir") == 0 && i+1 < argc) snprintf(c->save_dir, sizeof(c->save_dir), "%s", argv[++i]);
        else if (strcmp(argv[i], "--checkpoint") == 0 && i+1 < argc) *ckpt_path = argv[++i];
        else if (strcmp(argv[i], "--log-freq") == 0 && i+1 < argc) c->log_freq = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ckpt-freq") == 0 && i+1 < argc) c->ckpt_freq = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --min-n N          Min graph size (default: %d)\n", c->min_n);
            printf("  --max-n N          Max graph size (default: %d)\n", c->max_n);
            printf("  --episodes N       Total episodes (default: %d)\n", c->num_episodes);
            printf("  --batch-size N     Episodes per update (default: %d)\n", c->batch_size);
            printf("  --cycle-mult F     Cycles = F*n (default: %.1f)\n", c->cycle_mult);
            printf("  --batch-frac F     Batch = F*n per phase (default: %.2f)\n", c->swap_frac);
            printf("  --tau-start F      Start temperature (default: %.3f)\n", c->tau_start);
            printf("  --tau-end F        End temperature (default: %.4f)\n", c->tau_end);
            printf("  --lr F             Learning rate (default: %.1e)\n", c->lr);
            printf("  --ppo-epochs N     PPO epochs per update (default: %d)\n", c->ppo_epochs);
            printf("  --checkpoint PATH  Resume from checkpoint\n");
            printf("  --save-dir PATH    Output directory (default: %s)\n", c->save_dir);
            printf("  --seed N           RNG seed (default: %llu)\n", (unsigned long long)c->seed);
            exit(0);
        }
    }
}

int main(int argc, char **argv) {
    TrainConfig cfg = default_config();
    char *ckpt_path = NULL;
    parse_args(&cfg, argc, argv, &ckpt_path);

    mkdir(cfg.save_dir, 0755);

    PolicyWeights pw;
    AdamState adam;
    adam_init(&adam);
    uint32_t start_episode = 0;

    if (ckpt_path) {
        float best_imp;
        if (load_checkpoint(ckpt_path, &pw, &start_episode, &best_imp) != 0) {
            fprintf(stderr, "Failed to load checkpoint: %s\n", ckpt_path);
            return 1;
        }
        printf("Resumed from %s (episode %u)\n", ckpt_path, start_episode);
    } else {
        RNG init_rng;
        rng_init(&init_rng, cfg.seed);
        policy_init_orthogonal(&pw, &init_rng, 0.5f);
    }

    int num_updates = cfg.num_episodes / cfg.batch_size;
    int max_cycles_per_ep = (int)(cfg.cycle_mult * cfg.max_n) + 1;
    int max_txns_per_ep = 2 * max_cycles_per_ep;
    /* 2 trajectories per episode: traj1 (no ref) + traj2 (with ref) */
    int max_txns_batch = 2 * cfg.batch_size * max_txns_per_ep;
    int max_segments = 2 * cfg.batch_size; /* each episode = 2 GAE segments */

    PhaseTxn *all_txns = (PhaseTxn *)malloc((size_t)max_txns_batch * sizeof(PhaseTxn));
    int *ep_starts = (int *)calloc((size_t)(max_segments + 1), sizeof(int));
    double *all_values = (double *)malloc((size_t)max_txns_batch * sizeof(double));
    double *all_rewards = (double *)malloc((size_t)max_txns_batch * sizeof(double));
    double *all_advantages = (double *)malloc((size_t)max_txns_batch * sizeof(double));
    double *all_returns = (double *)malloc((size_t)max_txns_batch * sizeof(double));
    double *all_old_lps = (double *)malloc((size_t)max_txns_batch * sizeof(double));

    int param_count = 3 * (EDGE_FEAT_DIM * HIDDEN_DIM + HIDDEN_DIM +
                           HIDDEN_DIM * HIDDEN_DIM + HIDDEN_DIM +
                           HIDDEN_DIM + 1);

    printf("======================================================================\n");
    printf("CRL Dual-Phase PPO Training\n");
    printf("======================================================================\n");
    printf("  n: [%d, %d] | episodes: %d | batch: %d | updates: %d\n",
           cfg.min_n, cfg.max_n, cfg.num_episodes, cfg.batch_size, num_updates);
    printf("  K: %.0f*N | swap_frac: %.2f | tau: %.3f->%.4f\n",
           cfg.cycle_mult, cfg.swap_frac, cfg.tau_start, cfg.tau_end);
    printf("  lr: %.1e | ppo_epochs: %d | clip: %.2f | params: %d\n",
           cfg.lr, cfg.ppo_epochs, cfg.clip_eps, param_count);
    printf("  save_dir: %s\n", cfg.save_dir);
    printf("======================================================================\n\n");

    RNG train_rng;
    rng_init(&train_rng, cfg.seed + start_episode);

    double best_avg_l2 = -1e30;
    double t_start = wall_time();
    uint32_t episodes_done = start_episode;
    int start_update = (int)(start_episode / (uint32_t)cfg.batch_size);

    for (int update = start_update; update < num_updates; update++) {
        double progress = (double)update / fmax(num_updates - 1, 1);

        double lr = cfg.lr * (0.1 + 0.9 * 0.5 * (1.0 + cos(M_PI * progress)));
        double ent_coef = cfg.entropy_coef_start +
            (cfg.entropy_coef_end - cfg.entropy_coef_start) * progress;

        /* Collect batch of episodes (2 trajectories each) */
        int total_txns = 0;
        int num_segments = 0;
        double batch_l2_sum = 0, batch_best_sum = 0;
        int batch_valid = 0;

        for (int ep = 0; ep < cfg.batch_size; ep++) {
            int n = cfg.min_n + rng_int(&train_rng, cfg.max_n - cfg.min_n + 1);
            int max_m = n * (n - 1) / 2;
            int min_m = n;
            int m_val = min_m + rng_int(&train_rng, max_m - min_m + 1);

            int num_cycles = (int)(cfg.cycle_mult * n);
            if (num_cycles < 1) num_cycles = 1;

            uint64_t ep_seed = cfg.seed + episodes_done + (uint64_t)ep;

            int space = max_txns_batch - total_txns;
            if (space < 4) break;

            /* Trajectory 1: no reference */
            uint8_t ref_graph[N_MAX * N_MAX];
            ep_starts[num_segments] = total_txns;
            int t1_num = 0;
            double m1[5] = {0};

            dual_phase_collect_episode(
                &pw, n, m_val, num_cycles, cfg.swap_frac,
                cfg.tau_start, cfg.tau_end, ep_seed,
                NULL,
                all_txns + total_txns, space,
                &t1_num, m1, ref_graph);

            for (int t = 0; t < t1_num; t++) {
                PhaseTxn *txn = &all_txns[total_txns + t];
                all_values[total_txns + t] = txn->value;
                all_rewards[total_txns + t] = txn->reward;
                double olp = 0;
                for (int s = 0; s < txn->num_selected; s++)
                    olp += txn->log_probs[s];
                all_old_lps[total_txns + t] = olp;
            }
            total_txns += t1_num;
            num_segments++;

            /* Trajectory 2: with reference = trajectory 1's best graph */
            space = max_txns_batch - total_txns;
            if (space < 2) break;

            ep_starts[num_segments] = total_txns;
            int t2_num = 0;
            double m2[5] = {0};

            dual_phase_collect_episode(
                &pw, n, m_val, num_cycles, cfg.swap_frac,
                cfg.tau_start, cfg.tau_end, ep_seed + 1000000,
                ref_graph,
                all_txns + total_txns, space,
                &t2_num, m2, NULL);

            for (int t = 0; t < t2_num; t++) {
                PhaseTxn *txn = &all_txns[total_txns + t];
                all_values[total_txns + t] = txn->value;
                all_rewards[total_txns + t] = txn->reward;
                double olp = 0;
                for (int s = 0; s < txn->num_selected; s++)
                    olp += txn->log_probs[s];
                all_old_lps[total_txns + t] = olp;
            }
            total_txns += t2_num;
            num_segments++;

            double best_of_two = fmax(m1[2], m2[2]);
            batch_l2_sum += fmax(m1[0], m2[0]);
            batch_best_sum += best_of_two;
            batch_valid++;
        }
        ep_starts[num_segments] = total_txns;
        episodes_done += (uint32_t)cfg.batch_size;

        if (total_txns == 0) continue;

        /* GAE per trajectory segment */
        for (int seg = 0; seg < num_segments; seg++) {
            int start = ep_starts[seg];
            int end = ep_starts[seg + 1];
            int len = end - start;
            if (len <= 0) continue;

            compute_gae(all_values + start, all_rewards + start, len,
                        cfg.gamma, cfg.gae_lambda,
                        all_advantages + start, all_returns + start);
        }
        normalize_advantages(all_advantages, total_txns);

        /* PPO epochs */
        for (int ppo_ep = 0; ppo_ep < cfg.ppo_epochs; ppo_ep++) {
            PolicyGrad pg;
            policy_grad_zero(&pg);

            for (int t = 0; t < total_txns; t++) {
                ppo_backward_phase(&pw, &all_txns[t],
                                   all_advantages[t], all_returns[t],
                                   all_old_lps[t],
                                   ent_coef, cfg.clip_eps, &pg);
            }

            policy_grad_scale(&pg, 1.0f / (float)total_txns);
            policy_grad_clip(&pg, (float)cfg.grad_clip);
            adam_step(&pw, &pg, &adam, lr, 0.9, 0.999, 1e-8, cfg.weight_decay);
        }

        /* Logging */
        if ((update + 1) % cfg.log_freq == 0 || update == num_updates - 1) {
            double elapsed = wall_time() - t_start;
            double avg_l2 = batch_valid > 0 ? batch_l2_sum / batch_valid : 0;
            double avg_best = batch_valid > 0 ? batch_best_sum / batch_valid : 0;
            double eps = (episodes_done - start_episode) / fmax(elapsed, 0.001);

            printf("update %5d/%d | ep %6u | l2 %.3f | best %.3f | "
                   "txns %4d | lr %.1e | ent %.3f | %.1f ep/s | %.1fm\n",
                   update + 1, num_updates, episodes_done,
                   avg_l2, avg_best, total_txns, lr, ent_coef,
                   eps, elapsed / 60.0);
        }

        /* Best checkpoint */
        double avg_best = batch_valid > 0 ? batch_best_sum / batch_valid : 0;
        if (avg_best > best_avg_l2) {
            best_avg_l2 = avg_best;
            char path[512];
            snprintf(path, sizeof(path), "%s/best.bin", cfg.save_dir);
            save_checkpoint(path, &pw, episodes_done, (float)best_avg_l2);
        }

        /* Periodic checkpoint */
        if ((update + 1) % cfg.ckpt_freq == 0) {
            char path[512];
            snprintf(path, sizeof(path), "%s/ckpt_%u.bin",
                     cfg.save_dir, episodes_done);
            save_checkpoint(path, &pw, episodes_done, (float)best_avg_l2);
        }
    }

    /* Final checkpoint */
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/final.bin", cfg.save_dir);
        save_checkpoint(path, &pw, episodes_done, (float)best_avg_l2);
    }

    double total_time = wall_time() - t_start;
    printf("\n======================================================================\n");
    printf("Training complete: %u episodes, %d updates, %.1f min\n",
           episodes_done, num_updates, total_time / 60.0);
    printf("Best avg l2: %.4f\n", best_avg_l2);
    printf("Saved to: %s\n", cfg.save_dir);
    printf("======================================================================\n");

    free(all_txns); free(ep_starts);
    free(all_values); free(all_rewards);
    free(all_advantages); free(all_returns);
    free(all_old_lps);

    return 0;
}
