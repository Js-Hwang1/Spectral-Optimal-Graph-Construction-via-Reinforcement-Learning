/* Test_C2.c — Comprehensive C2 Branch Testing
 *
 * DESCRIPTION:
 *   Comprehensive test suite for C2 branch (odd n, k ≥ (n+1)/2) with
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

static void test_c2_single(int n, int k) {
    // Check if (n,k) is valid for simple graphs
    if ((n * k) % 2 != 0) {
        return;  // Skip invalid cases silently
    }
    

    double lambda2_c2 = algorithm1_get_lambda2_quiet(n, k);

    

    int target_edges = (n * k) / 2;  // Total edges for k-regular graph
    double lambda2_erg = run_erg_baseline(n, target_edges);
    
    double difference = 0.0;
    if (lambda2_c2 > 0 && lambda2_erg > 0) {
        difference = lambda2_c2 - lambda2_erg;
    }
    
    // Clean output format: (n,k) | Alg1's lambda2 | Baseline's lambda2 | Difference
    printf("(%2d,%2d) | %10.6f | %10.6f | %+10.6f\n", n, k, lambda2_c2, lambda2_erg, difference);
}

static void test_c2_comprehensive(int *test_n_values, int num_n) {
    printf(" (n,k) |  Alg1's λ₂  | Baseline λ₂  |  Difference \n");
    printf("=======|============|==============|=============\n");
    
    for (int i = 0; i < num_n; i++) {
        int n = test_n_values[i];
        
        // Skip even n (C2 is for odd n only)
        if (n % 2 == 0) {
            continue;
        }
        
        // Test all relevant k values for C2 branch
        int min_k = (n + 1) / 2;  // Minimum k for C2 branch
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
            test_c2_single(n, k);
        }
    }
}

/* ========================================================================
 * MAIN PROGRAM
 * ======================================================================== */

int main(int argc, char *argv[]) {
    
    // Help message
    if (argc > 1 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
        printf("USAGE:\n");
        printf("  %s                  - Run default test suite (odd n: 7,9,11,13,15,17,19,21,23,25)\n", argv[0]);
        printf("  %s n1 n2 n3 ...     - Run tests for specified n values\n", argv[0]);
        printf("  %s --help           - Show this help message\n\n", argv[0]);
        printf("DESCRIPTION:\n");
        printf("  Tests C2 branch (odd n, k >= (n+1)/2) against ERG baseline.\n");
        printf("  For each odd n, tests all valid even k values in the C2 range.\n");
        printf("  Skips impossible (n,k) combinations where n*k is odd.\n\n");
        return 0;
    }
    
    // Default test values (odd n only)
    int default_n_values[] = {7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95};
    int num_default = sizeof(default_n_values) / sizeof(default_n_values[0]);
    
    if (argc == 1) {
        // No arguments - run default test suite
        printf("Running default test suite...\n");
        test_c2_comprehensive(default_n_values, num_default);
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
            test_c2_comprehensive(custom_n_values, num_custom);
        } else {
            printf("No valid n values provided, running default suite...\n");
            test_c2_comprehensive(default_n_values, num_default);
        }
        
        free(custom_n_values);
    }
    
    printf("=== Test Suite Complete ===\n");
    return 0;
}