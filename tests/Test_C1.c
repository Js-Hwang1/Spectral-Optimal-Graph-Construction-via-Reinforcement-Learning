/* Test_C1.c — Comprehensive C1 Branch Testing
 *
 * DESCRIPTION:
 *   Comprehensive test suite for C1 branch (odd n, even k, 4 ≤ k < (n+1)/2) with
 *   side-by-side comparison against ERG baseline.
 *   Tests all relevant even k values for specified odd n values.
 *   
 *   C1: Uses circulant construction (ring + steps) for odd n with even k
 *   ERG: Random graph baseline using effective resistance
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "Algorithm1.h"

// Include the Algorithm1 and Supporting functions
double algorithm1_get_lambda2_quiet(int n, int k);

// Quiet version of algorithm1_main that suppresses ALL output
double algorithm1_get_lambda2_quiet(int n, int k) {
    double lambda2 = -1.0;
    
    // Redirect stdout to null to suppress all output
    FILE *original_stdout = stdout;
    stdout = fopen("/dev/null", "w");
    
    // Call the actual algorithm1_main function  
    int result = algorithm1_main(n, k, &lambda2);
    
    // Restore stdout
    fclose(stdout);
    stdout = original_stdout;
    
    if (result == 0 || lambda2 < 0) {  // result=0 means failure, result=1 means success
        return -1.0;  // Error case
    }
    
    return lambda2;
}

/* ========================================================================
 * CONFIGURATION & CONSTANTS
 * ======================================================================== */

#define MAX_N 200
#define MAX_K 200

/* ========================================================================
 * BASELINE FUNCTIONS
 * ======================================================================== */

// Call the existing ERG baseline executable and parse result for target edges
static double run_erg_baseline(int n, int target_edges) {
    char command[512];
    char temp_dir[] = "/tmp/erg_test_XXXXXX";
    
    // Create temporary directory
    if (mkdtemp(temp_dir) == NULL) {
        perror("mkdtemp");
        return 0.0;
    }
    
    // Run ERG baseline for single n value
    snprintf(command, sizeof(command), 
             "%s/baselines/baseline_effective_resistance --n-min %d --n-max %d --out-dir %s > /dev/null 2>&1",
             "/Users/j/Desktop/Projects/Spectral-Optimal-Graph-Construction-via-Reinforcement-Learning",
             n, n, temp_dir);
    
    int ret = system(command);
    if (ret != 0) {
        // Cleanup and return
        snprintf(command, sizeof(command), "rm -rf %s", temp_dir);
        system(command);
        return 0.0;
    }
    
    // Read result file and find the lambda2 for target_edges
    char result_file[512];
    snprintf(result_file, sizeof(result_file), "%s/n%d_ERG_data", temp_dir, n);  // No .txt extension
    
    FILE *fp = fopen(result_file, "r");
    if (!fp) {
        // Cleanup and return
        snprintf(command, sizeof(command), "rm -rf %s", temp_dir);
        system(command);
        return 0.0;
    }
    
    double best_lambda2 = 0.0;
    char line[256];
    int current_n, current_m;
    double lambda2;
    
    // Parse the ERG output file to find lambda2 for target_edges
    while (fgets(line, sizeof(line), fp)) {
        if (sscanf(line, "(%d,%d)", &current_n, &current_m) == 2) {
            // Read next line for lambda2
            if (fgets(line, sizeof(line), fp)) {
                if (sscanf(line, "lambda2: %lf", &lambda2) == 1) {
                    if (current_m == target_edges) {
                        best_lambda2 = lambda2;
                        break;
                    }
                    // Keep track of best lambda2 if exact match not found
                    if (current_m <= target_edges && lambda2 > best_lambda2) {
                        best_lambda2 = lambda2;
                    }
                }
            }
        }
    }
    
    fclose(fp);
    
    // Cleanup temporary directory
    snprintf(command, sizeof(command), "rm -rf %s", temp_dir);
    system(command);
    
    return best_lambda2;
}

/* ========================================================================
 * TEST FUNCTIONS
 * ======================================================================== */

static void test_c1_single_n(int n) {
    printf("\n=== C1 Branch Comprehensive Testing ===\n");
    printf("Testing odd n=%d with even k values, 4 ≤ k < (n+1)/2\n", n);
    printf("\n (n,k) | Algorithm1 |   Baseline |      Diff |    Improv | Timing\n");
    printf("-------|------------|------------|-----------|----------|------------------\n");
    
    int ceil_half = (n + 1) / 2;
    
    // Test all even k values from 4 to just under ceil(n/2)
    for (int k = 4; k < ceil_half; k += 2) {  // Only even k values
        int target_edges = (n * k) / 2;
        
        // Skip impossible cases (n*k must be even for simple graphs)
        if ((n * k) % 2 != 0) {
            continue;
        }
        
        // Time Algorithm1
        clock_t start_alg = clock();
        double lambda2_alg = algorithm1_get_lambda2_quiet(n, k);
        clock_t end_alg = clock();
        double time_alg = ((double)(end_alg - start_alg)) / CLOCKS_PER_SEC;
        
        // Time ERG baseline
        clock_t start_erg = clock();
        double lambda2_erg = run_erg_baseline(n, target_edges);
        clock_t end_erg = clock();
        double time_erg = ((double)(end_erg - start_erg)) / CLOCKS_PER_SEC;
        
        // Calculate improvement
        if (lambda2_alg < 0) {
            printf("(%2d,%2d) |       FAIL |   %8.6f |       --- |     ---   | C1: %.3fs ERG: %.3fs\n",
                   n, k, lambda2_erg, time_alg, time_erg);
        } else if (lambda2_erg <= 0) {
            printf("(%2d,%2d) |   %8.6f |       FAIL |       --- |     ---   | C1: %.3fs ERG: %.3fs\n",
                   n, k, lambda2_alg, time_alg, time_erg);
        } else {
            double diff = lambda2_alg - lambda2_erg;
            double improvement = (diff / lambda2_erg) * 100.0;
            
            printf("(%2d,%2d) |   %8.6f |   %8.6f | %+9.6f |", n, k, lambda2_alg, lambda2_erg, diff);
            
            if (improvement >= 0.1) {
                printf(" %+7.2f%% ✓", improvement);
            } else if (improvement >= -0.1) {
                printf(" %+7.2f%%  ", improvement);
            } else {
                printf(" %+7.2f%% ✗", improvement);
            }
            
            printf(" | C1: %.3fs ERG: %.3fs\n", time_alg, time_erg);
        }
    }
    
    printf("\nLegend:\n");
    printf("  ✓ = Algorithm1 outperforms baseline by ≥0.1%%\n");
    printf("  ✗ = Algorithm1 underperforms baseline by ≥0.1%%\n");
    printf("  Diff = Algorithm1 λ₂ - Baseline λ₂\n");
    printf("  Improv = Performance improvement percentage\n");
}

/* ========================================================================
 * MAIN FUNCTION
 * ======================================================================== */

int main(int argc, char *argv[]) {
    printf("C1 Branch Test Suite - Odd n, Even k, 4 ≤ k < (n+1)/2\n");
    printf("=======================================================\n");
    
    if (argc < 2) {
        printf("Usage: %s <n1> [n2] [n3] ...\n", argv[0]);
        printf("Tests C1 branch for odd n with even k from 4 to (n+1)/2-1\n");
        printf("Example: %s 9 11 13\n\n", argv[0]);
        return 1;
    }
    
    int num_tests = argc - 1;
    printf("Running custom test suite for %d values...\n", num_tests);
    
    for (int i = 1; i < argc; i++) {
        int n = atoi(argv[i]);
        
        // Validate that n is odd
        if (n % 2 == 0) {
            printf("Error: C1 branch requires odd n. Skipping n=%d\n", n);
            continue;
        }
        
        // Validate minimum constraints
        if (n < 5) {
            printf("Error: C1 branch requires n ≥ 5 for meaningful k values. Skipping n=%d\n", n);
            continue;
        }
        
        test_c1_single_n(n);
    }
    
    printf("=== Test Suite Complete ===\n");
    
    return 0;
}