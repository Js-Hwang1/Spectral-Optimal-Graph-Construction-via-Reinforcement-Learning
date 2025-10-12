/* Test_B1.c — Comprehensive B1 Branch Testing
 *
 * DESCRIPTION:
 *   Comprehensive test suite for B1 branch (even n, 3 ≤ k < n/2) with
 *   side-by-side comparison against ERG baseline.
 *   Tests all relevant k values for specified n values.
 *   
 *   B1: Uses round-robin chord construction for low-degree even n graphs
 *   ERG: Random graph baseline using effective resistance
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "../Phase1_src/Algorithm1.h"

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

static void test_b1_single(int n, int k) {
    // Check if (n,k) is valid for simple graphs
    if ((n * k) % 2 != 0) {
        return;  // Skip invalid cases silently
    }
    
    // Run Algorithm1 (B1 branch) - quiet mode
    clock_t start_b1 = clock();
    double lambda2_b1 = algorithm1_get_lambda2_quiet(n, k);
    clock_t end_b1 = clock();
    double time_b1 = ((double)(end_b1 - start_b1)) / CLOCKS_PER_SEC;
    
    // Run ERG baseline
    clock_t start_erg = clock();
    int target_edges = (n * k) / 2;  // Total edges for k-regular graph
    double lambda2_erg = run_erg_baseline(n, target_edges);
    clock_t end_erg = clock();
    double time_erg = ((double)(end_erg - start_erg)) / CLOCKS_PER_SEC;
    
    // Calculate improvement
    double improvement = 0.0;
    double difference = 0.0;
    if (lambda2_b1 > 0 && lambda2_erg > 0) {
        improvement = ((lambda2_b1 - lambda2_erg) / lambda2_erg) * 100.0;
        difference = lambda2_b1 - lambda2_erg;
    }
    
    // Print results with proper alignment
    printf("(%2d,%2d) ", n, k);
    
    if (lambda2_b1 > 0) {
        printf("| %10.6f ", lambda2_b1);
    } else {
        printf("| %10s ", "FAIL");
    }
    
    if (lambda2_erg > 0) {
        printf("| %10.6f ", lambda2_erg);
    } else {
        printf("| %10s ", "FAIL");
    }
    
    if (lambda2_b1 > 0 && lambda2_erg > 0) {
        printf("| %+9.6f ", difference);
        
        if (improvement >= 0.1) {
            printf("| %+7.2f%% ✓", improvement);
        } else if (improvement <= -0.1) {
            printf("| %+7.2f%% ✗", improvement);
        } else {
            printf("| %+7.2f%%  ", improvement);
        }
    } else {
        printf("| %9s | %7s  ", "---", "---");
    }
    
    printf(" | B1: %.3fs ERG: %.3fs\n", time_b1, time_erg);
}

static void test_b1_comprehensive(int n) {
    printf("\n=== B1 Branch Comprehensive Testing ===\n");
    printf("Testing even n values with 3 ≤ k < n/2\n\n");
    printf(" (n,k) | Algorithm1 |   Baseline |      Diff |    Improv | Timing\n");
    printf("-------|------------|------------|-----------|----------|------------------\n");
    
    // Test all valid k values for B1 branch: 3 ≤ k < n/2
    int max_k = n / 2 - 1;  // B1 handles k < n/2
    
    for (int k = 3; k <= max_k; k++) {
        // Only test even k for even n to ensure simple graphs
        if ((n * k) % 2 == 0) {
            test_b1_single(n, k);
        }
    }
}

/* ========================================================================
 * MAIN FUNCTION
 * ======================================================================== */

int main(int argc, char *argv[]) {
    printf("B1 Branch Test Suite - Even n, 3 ≤ k < n/2\n");
    printf("===========================================\n");
    
    if (argc < 2) {
        printf("Usage: %s <n1> [n2] [n3] ...\n", argv[0]);
        printf("Tests B1 branch for even n with k from 3 to n/2-1\n");
        printf("Example: %s 8 10 12\n", argv[0]);
        return 1;
    }
    
    int num_tests = argc - 1;
    printf("\nRunning custom test suite for %d values...\n", num_tests);
    
    for (int i = 1; i < argc; i++) {
        int n = atoi(argv[i]);
        
        // Validate that n is even
        if (n % 2 != 0) {
            printf("Warning: Skipping n=%d (B1 branch requires even n)\n", n);
            continue;
        }
        
        // Validate that n allows for meaningful B1 testing (need k ≥ 3 and k < n/2)
        if (n < 8) {
            printf("Warning: Skipping n=%d (too small for B1 range 3 ≤ k < n/2)\n", n);
            continue;
        }
        
        test_b1_comprehensive(n);
    }
    
    printf("\nLegend:\n");
    printf("  ✓ = Algorithm1 outperforms baseline by ≥0.1%%\n");
    printf("  ✗ = Algorithm1 underperforms baseline by ≥0.1%%\n");
    printf("  Diff = Algorithm1 λ₂ - Baseline λ₂\n");
    printf("  Improv = Performance improvement percentage\n");
    printf("=== Test Suite Complete ===\n");
    
    return 0;
}