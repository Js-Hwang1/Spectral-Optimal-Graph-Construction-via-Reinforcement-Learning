/* Test_C2_fixed.c — Clean, single-definition C2 test harness (fixed)
 *
 * Same behavior as the previous Test_C2.c but placed in a new file to avoid
 * corruption in the original. This file calls algorithm1_main, runs the ERG
 * baseline executable, and prints a standardized comparison table.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#include "Algorithm1.h"

static double get_lambda2_quiet(int n, int k) {
    double lambda2 = -1.0;
    FILE *saved = stdout;
    FILE *f = fopen("/dev/null", "w");
    if (!f) return -1.0;
    stdout = f;
    int rc = algorithm1_main(n, k, &lambda2);
    fclose(f);
    stdout = saved;
    if (rc == 0 || lambda2 < 0) return -1.0;
    return lambda2;
}

static double run_erg_baseline(int n, int target_edges) {
    char tmpdir[] = "/tmp/erg_test_XXXXXX";
    if (mkdtemp(tmpdir) == NULL) return 0.0;
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "%s/baselines/baseline_effective_resistance --n-min %d --n-max %d --out-dir %s > /dev/null 2>&1",
             "/Users/j/Desktop/Projects/Spectral-Optimal-Graph-Construction-via-Reinforcement-Learning", n, n, tmpdir);
    if (system(cmd) != 0) { snprintf(cmd, sizeof(cmd), "rm -rf %s", tmpdir); system(cmd); return 0.0; }

    char path[512];
    snprintf(path, sizeof(path), "%s/n%d_ERG_data.txt", tmpdir, n);
    FILE *fp = fopen(path, "r");
    if (!fp) { snprintf(cmd, sizeof(cmd), "rm -rf %s", tmpdir); system(cmd); return 0.0; }

    double best = 0.0;
    char line[512];
    int cn, cm;
    double lam;
    while (fgets(line, sizeof(line), fp)) {
        if (sscanf(line, "(%d,%d)", &cn, &cm) == 2) {
            if (fgets(line, sizeof(line), fp)) {
                if (sscanf(line, "lambda2: %lf", &lam) == 1) {
                    if (cm == target_edges) { best = lam; break; }
                    if (cm <= target_edges && lam > best) best = lam;
                }
            }
            fgets(line, sizeof(line), fp);
            fgets(line, sizeof(line), fp);
        }
    }
    fclose(fp);
    snprintf(cmd, sizeof(cmd), "rm -rf %s", tmpdir); system(cmd);
    return best;
}

static void print_header(void) {
    printf(" (n,k) | Algorithm1 |   Baseline |      Diff |    Improv | Timing\n");
    printf("-------|------------|------------|-----------|-----------|------------------\n");
}

static void test_single(int n, int k) {
    if ((n * k) % 2 != 0) return;
    clock_t s = clock();
    double a = get_lambda2_quiet(n, k);
    clock_t e = clock();
    double ta = (double)(e - s) / CLOCKS_PER_SEC;

    s = clock();
    double b = run_erg_baseline(n, (n*k)/2);
    e = clock();
    double tb = (double)(e - s) / CLOCKS_PER_SEC;

    double diff = 0.0, imp = 0.0;
    if (a > 0 && b > 0) { diff = a - b; imp = (diff / b) * 100.0; }

    printf("(%2d,%2d) ", n, k);
    if (a>0) printf("| %10.6f ", a); else printf("| %10s ", "FAIL");
    if (b>0) printf("| %10.6f ", b); else printf("| %10s ", "FAIL");
    if (a>0 && b>0) {
        printf("| %+9.6f ", diff);
        if (imp >= 0.1) printf("| %+7.2f%% ✓", imp);
        else if (imp <= -0.1) printf("| %+7.2f%% ✗", imp);
        else printf("| %+7.2f%%  ", imp);
    } else {
        printf("| %9s | %9s ", "---", "---");
    }
    printf(" | C2: %.3fs ERG: %.3fs\n", ta, tb);
}

static void test_comprehensive(int *ns, int cnt) {
    print_header();
    for (int i=0;i<cnt;++i) {
        int n = ns[i];
        if (n % 2 == 0) continue;
        int lo = (n+1)/2, hi = n-1;
        if (lo % 2 != 0) ++lo; if (hi % 2 != 0) --hi; if (hi < lo) hi = lo;
        for (int k = lo; k <= hi; k += 2) test_single(n,k);
    }
}

int main(int argc, char **argv) {
    int default_ns[] = {7,9,11,13,15,17,19,21,23,25};
    int dn = sizeof(default_ns)/sizeof(default_ns[0]);
    if (argc == 1) test_comprehensive(default_ns, dn);
    else {
        int *arr = malloc(sizeof(int)*(argc-1)); int c=0;
        for (int i=1;i<argc;++i) { int v = atoi(argv[i]); if (v>0) arr[c++]=v; }
        if (c>0) test_comprehensive(arr,c); else test_comprehensive(default_ns,dn);
        free(arr);
    }
    return 0;
}
