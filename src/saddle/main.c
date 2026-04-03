/*
 * main.c -- CLI entry for saddle-aware RL graph rewiring.
 *
 * Modes:
 *   --train-climber    Train the climber agent
 *   --train-navigator  Train the navigator (requires --climber-ckpt)
 *   --eval             Evaluate (requires --climber-ckpt, optional --navigator-ckpt)
 */

#include "saddle.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>

static double wall_time(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
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
        e.best = e.fv; if (e.er > e.best) e.best = e.er;
        if (e.sw025 > e.best) e.best = e.sw025;
        if (e.sw050 > e.best) e.best = e.sw050;
        if (e.sw075 > e.best) e.best = e.sw075;
        if (nbl >= cap) { cap *= 2; bls = realloc(bls, (size_t)cap * sizeof(BL)); }
        bls[nbl++] = e;
    }
    fclose(f); printf("Loaded %d baselines\n", nbl);
}

static BL *find_bl(int n, int m) {
    for (int i = 0; i < nbl; i++) if (bls[i].n == n && bls[i].m == m) return &bls[i];
    return NULL;
}

/* ========================================================================== */
/* Eval mode                                                                   */
/* ========================================================================== */

static void run_eval(const ClimberWeights *cw, const NavigatorWeights *nw,
                     int *n_values, int num_n, int budget_mult,
                     float nav_threshold, int god_mode, const char *csv_path) {
    FILE *csv_f = NULL;
    if (csv_path) { csv_f = fopen(csv_path, "w"); if (csv_f) fprintf(csv_f, "m,score\n"); }

    printf("Saddle RL Eval | budget=%d*n | threads=%d\n", budget_mult, omp_get_max_threads());

    for (int ni = 0; ni < num_n; ni++) {
        int n = n_values[ni]; int mx = n * (n - 1) / 2;
        int *m_list = (int *)malloc((size_t)(mx + 1) * sizeof(int)); int nm = 0;
        for (int mv = n - 1; mv <= mx; mv++) {
            BL *b = find_bl(n, mv);
            if (b && b->best > 0) m_list[nm++] = mv;
            else if (nbl == 0) m_list[nm++] = mv;
        }
        if (nm == 0) { free(m_list); continue; }
        printf("\nn=%d (%d configs)\n", n, nm);

        double t0 = wall_time();
        double *results = (double *)malloc((size_t)nm * sizeof(double));

        #pragma omp parallel for schedule(dynamic, 1)
        for (int mi = 0; mi < nm; mi++) {
            uint64_t s = (uint64_t)(n * 100 + m_list[mi]);
            results[mi] = god_mode
                ? god_episode(cw, n, m_list[mi], budget_mult, SD_MAX_CAND, s)
                : saddle_episode(cw, nw, n, m_list[mi], budget_mult, nav_threshold, s);
        }

        double sr = 0, sf = 0, sb = 0; int wf = 0, wb = 0, tot = 0;
        for (int mi = 0; mi < nm; mi++) {
            BL *b = find_bl(n, m_list[mi]);
            double fv_val = b ? b->fv : 0, best_val = b ? b->best : 0;
            sr += results[mi]; sf += fv_val; sb += best_val;
            if (results[mi] > fv_val * 1.001) wf++;
            if (results[mi] > best_val * 1.001) wb++;
            tot++;
            if (csv_f) fprintf(csv_f, "%d,%.10f\n", m_list[mi], results[mi]);
        }

        double elapsed = wall_time() - t0;
        printf("  vs FV: %.1f%% (%dW/%d) | vs Best: %.1f%% (%dW/%d) | %.1fs\n",
               sf > 0 ? sr / sf * 100 : 0, wf, tot,
               sb > 0 ? sr / sb * 100 : 0, wb, tot, elapsed);

        free(results); free(m_list);
    }

    if (csv_f) fclose(csv_f);
}

/* ========================================================================== */
/* Main                                                                        */
/* ========================================================================== */

int main(int argc, char **argv) {
    int mode = 0; /* 0=none, 1=train_climber, 2=train_navigator, 3=eval */
    char *climber_ckpt = NULL, *navigator_ckpt = NULL;
    char *bl_path = NULL, *csv_path = NULL;
    int n_values[64]; int num_n = 0;

    SaddleTrainConfig cfg = {
        .min_n = 8, .max_n = 10,
        .num_episodes = 5000,
        .budget_mult = 4,
        .lr = 1e-3,
        .tau_kl = 0.1,
        .weight_decay = 1e-4,
        .grad_clip = 1.0,
        .seed = 42,
        .log_freq = 50,
        .ckpt_freq = 500,
    };
    snprintf(cfg.save_dir, sizeof(cfg.save_dir), "logs/saddle_rl");

    float nav_threshold = 0.0f;
    int god_mode = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--train-climber") == 0) mode = 1;
        else if (strcmp(argv[i], "--train-navigator") == 0) mode = 2;
        else if (strcmp(argv[i], "--eval") == 0) mode = 3;
        else if (strcmp(argv[i], "--climber-ckpt") == 0 && i + 1 < argc) climber_ckpt = argv[++i];
        else if (strcmp(argv[i], "--navigator-ckpt") == 0 && i + 1 < argc) navigator_ckpt = argv[++i];
        else if (strcmp(argv[i], "--baselines") == 0 && i + 1 < argc) bl_path = argv[++i];
        else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) csv_path = argv[++i];
        else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            char *tok = strtok(argv[++i], ",");
            while (tok && num_n < 64) { n_values[num_n++] = atoi(tok); tok = strtok(NULL, ","); }
        }
        else if (strcmp(argv[i], "--min-n") == 0 && i + 1 < argc) cfg.min_n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-n") == 0 && i + 1 < argc) cfg.max_n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--episodes") == 0 && i + 1 < argc) cfg.num_episodes = atoi(argv[++i]);
        else if (strcmp(argv[i], "--budget-mult") == 0 && i + 1 < argc) cfg.budget_mult = atoi(argv[++i]);
        else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) cfg.lr = atof(argv[++i]);
        else if (strcmp(argv[i], "--tau") == 0 && i + 1 < argc) cfg.tau_kl = atof(argv[++i]);
        else if (strcmp(argv[i], "--save-dir") == 0 && i + 1 < argc) snprintf(cfg.save_dir, sizeof(cfg.save_dir), "%s", argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) cfg.seed = (uint64_t)atol(argv[++i]);
        else if (strcmp(argv[i], "--nav-threshold") == 0 && i + 1 < argc) nav_threshold = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--god") == 0) god_mode = 1;
    }

    if (mode == 1) {
        train_climber(&cfg);
        return 0;
    }

    if (mode == 2) {
        if (!climber_ckpt) {
            fprintf(stderr, "Need --climber-ckpt for navigator training\n");
            return 1;
        }
        ClimberWeights cw;
        if (load_climber(climber_ckpt, &cw, NULL) != 0) {
            fprintf(stderr, "Failed to load climber: %s\n", climber_ckpt);
            return 1;
        }
        printf("Loaded climber: %s\n", climber_ckpt);
        train_navigator(&cw, &cfg);
        return 0;
    }

    if (mode == 3) {
        if (!climber_ckpt || num_n == 0) {
            printf("Usage: %s --eval --climber-ckpt X --n 8,16 [--navigator-ckpt X] [--baselines X]\n", argv[0]);
            return 1;
        }
        ClimberWeights cw;
        if (load_climber(climber_ckpt, &cw, NULL) != 0) {
            fprintf(stderr, "Failed to load climber: %s\n", climber_ckpt);
            return 1;
        }
        printf("Loaded climber: %s\n", climber_ckpt);

        NavigatorWeights nw;
        int has_nav = 0;
        if (navigator_ckpt) {
            if (load_navigator(navigator_ckpt, &nw, NULL) != 0) {
                fprintf(stderr, "Failed to load navigator: %s\n", navigator_ckpt);
                return 1;
            }
            printf("Loaded navigator: %s\n", navigator_ckpt);
            has_nav = 1;
        } else {
            navigator_init_weights(&nw, 0);
        }

        if (bl_path) load_bl(bl_path);
        run_eval(&cw, &nw, n_values, num_n, cfg.budget_mult, nav_threshold, god_mode, csv_path);
        free(bls);
        return 0;
    }

    printf("Saddle-Aware RL for Lambda2 Maximization\n");
    printf("Usage:\n");
    printf("  %s --train-climber --min-n 8 --max-n 10 --episodes 5000 --save-dir logs/saddle\n", argv[0]);
    printf("  %s --train-navigator --climber-ckpt logs/saddle/climber_final.bin --save-dir logs/saddle\n", argv[0]);
    printf("  %s --eval --climber-ckpt X [--navigator-ckpt X] --n 8,16 --baselines baselines.csv\n", argv[0]);
    return 1;
}
