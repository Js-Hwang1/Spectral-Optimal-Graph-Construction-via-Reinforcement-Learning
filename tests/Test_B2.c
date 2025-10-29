/* Test_B2.c — Comprehensive B2 Branch Testing
 *
 * DESCRIPTION:
 *   Comprehensive test suite for B2 branch (even n, k ≥ n/2) with
 *   side-by-side comparison against ERG baseline.
 *   Tests all relevant k values for specified n values.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
// #include <lapacke.h>

// Include our modules
#include "Algorithm1.h"
// #include "../Phase1_src/eigenvalue.h"

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
 * ERG BASELINE INTEGRATION - Using Existing Executable
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
    snprintf(result_file, sizeof(result_file), "%s/n%d_ERG_data.txt", temp_dir, n);
    
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
            // Skip the graph6 line
            fgets(line, sizeof(line), fp);
            // Skip empty line
            fgets(line, sizeof(line), fp);
        }
    }
    
    fclose(fp);
    
    // Cleanup
    snprintf(command, sizeof(command), "rm -rf %s", temp_dir);
    system(command);
    
    return best_lambda2;
}

/* ========================================================================
 * TEST FUNCTIONS
 * ======================================================================== */

static void test_b2_single(int n, int k) {
    // Check if (n,k) is valid for simple graphs
    if ((n * k) % 2 != 0) {
        return;  // Skip invalid cases silently
    }
    
    // Run Algorithm1 (B2 branch) - quiet mode
    clock_t start_b2 = clock();
    double lambda2_b2 = algorithm1_get_lambda2_quiet(n, k);
    clock_t end_b2 = clock();
    double time_b2 = ((double)(end_b2 - start_b2)) / CLOCKS_PER_SEC;
    
    // Run ERG baseline
    clock_t start_erg = clock();
    int target_edges = (n * k) / 2;  // Total edges for k-regular graph
    double lambda2_erg = run_erg_baseline(n, target_edges);
    clock_t end_erg = clock();
    double time_erg = ((double)(end_erg - start_erg)) / CLOCKS_PER_SEC;
    
    // Calculate improvement
    double improvement = 0.0;
    double difference = 0.0;
    if (lambda2_b2 > 0 && lambda2_erg > 0) {
        improvement = ((lambda2_b2 - lambda2_erg) / lambda2_erg) * 100.0;
        difference = lambda2_b2 - lambda2_erg;
    }
    
    // Print results with proper alignment
    printf("(%2d,%2d) ", n, k);
    
    if (lambda2_b2 > 0) {
        printf("| %10.6f ", lambda2_b2);
    } else {
        printf("| %10s ", "FAIL");
    }
    
    if (lambda2_erg > 0) {
        printf("| %10.6f ", lambda2_erg);
    } else {
        printf("| %10s ", "FAIL");
    }
    
    if (lambda2_b2 > 0 && lambda2_erg > 0) {
        printf("| %+9.6f ", difference);
        if (improvement >= 0.1) {
            printf("| %+7.2f%% ✓", improvement);
        } else if (improvement >= -0.1) {
            printf("| %+7.2f%%  ", improvement);
        } else {
            printf("| %+7.2f%% ✗", improvement);
        }
    } else {
        printf("| %10s | %8s ", "---", "---");
    }
    
    printf(" | B2:%6.3fs ERG:%6.3fs", time_b2, time_erg);
    printf("\n");
}

static void test_b2_comprehensive(int *test_n_values, int num_n) {
    printf(" (n,k) | Algorithm1 |   Baseline |      Diff |    Improv | Timing\n");
    printf("-------|------------|------------|-----------|-----------|------------------\n");
    
    for (int i = 0; i < num_n; i++) {
        int n = test_n_values[i];
        
        // Skip odd n (B2 is for even n only)
        if (n % 2 == 1) {
            continue;
        }
        
        // Test all relevant k values for B2 branch
        int min_k = (n  / 2);  // Minimum k for B2 branch
        int max_k = n - 1;        // Maximum possible k (complete graph)
        
        // Ensure min_k is even (since k must be even for odd n)
        if (min_k % 2 != 0) {
            min_k += 1;  // Round up to next even number
        }
        
        // Ensure max_k is even (since k must be even for odd n)
        if (max_k % 2 != 0) {
            max_k -= 1;
        }
        
        // Ensure max_k is at least min_k
        if (max_k < min_k) {
            max_k = min_k;
        }
        
        for (int k = min_k; k <= max_k; k += 2) {  // k must be even for odd n
            test_b2_single(n, k);
        }
    }
}

/* ========================================================================
 * MAIN FUNCTION
 * ======================================================================== */

int main(int argc, char *argv[]) {
    printf("B2 Branch Test Suite - Even n, k >= n/2\n");
    printf("========================================\n\n");
    
    // Help message
    if (argc > 1 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
        printf("USAGE:\n");
        printf("  %s                  - Run default test suite (even n: 6,8,10,12,14,16,18,20,22,24)\n", argv[0]);
        printf("  %s n1 n2 n3 ...     - Run tests for specified n values\n", argv[0]);
        printf("  %s --help           - Show this help message\n\n", argv[0]);
        printf("DESCRIPTION:\n");
        printf("  Tests B2 branch (even n, k >= n/2) against ERG baseline.\n");
        printf("  For each even n, tests all valid even k values in the B2 range.\n");
        printf("  Skips impossible (n,k) combinations where n*k is odd.\n\n");
        return 0;
    }
    
    // Default test values (even n only)
    int default_n_values[] = {6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96};
    int num_default = sizeof(default_n_values) / sizeof(default_n_values[0]);
    
    if (argc == 1) {
        // No arguments - run default test suite
        printf("Running default test suite...\n");
        test_b2_comprehensive(default_n_values, num_default);
    } else {
        // Parse n values from command line
        int *custom_n_values = malloc(sizeof(int) * (argc - 1));
        int num_custom = 0;
        
        for (int i = 1; i < argc; i++) {
            int n = atoi(argv[i]);
            if (n > 0) {
                custom_n_values[num_custom++] = n;
            } else {
                printf("Warning: Invalid n value '%s' ignored\n", argv[i]);
            }
        }
        
        if (num_custom > 0) {
            printf("Running custom test suite for %d values...\n", num_custom);
            test_b2_comprehensive(custom_n_values, num_custom);
        } else {
            printf("No valid n values provided, running default suite...\n");
            test_b2_comprehensive(default_n_values, num_default);
        }
        
        free(custom_n_values);
    }
    
    printf("=== Test Suite Complete ===\n");
    return 0;
}
