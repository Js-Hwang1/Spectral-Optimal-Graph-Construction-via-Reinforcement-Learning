#include <stdio.h>
#include "../include/Algorithm1.h"

int main(int argc, char **argv) {
    printf("Phase1 Algorithm Demo - Testing A Branch\n");
    
    // Test A branch with complete graph: n=4, k=3 (k = n-1)
    double lambda2;
    printf("\n=== Testing A Branch: Complete Graph K_4 ===\n");
    if(algorithm1_main(4, 3, &lambda2)) {
        printf("Success! Complete graph K_4 constructed with λ₂ = %.6f\n", lambda2);
        printf("Expected λ₂ for K_4: 4.0\n");
    } else {
        printf("Failed to construct complete graph K_4\n");
    }
    
    // Test A branch error case: k < 3
    printf("\n=== Testing A Branch: Error Case k < 3 ===\n");
    printf("Attempting n=5, k=2 (should throw error)...\n");
    if(algorithm1_main(5, 2, &lambda2)) {
        printf("Unexpected success! λ₂ = %.6f\n", lambda2);
    } else {
        printf("Expected failure occurred\n");
    }
    
    return 0;
}
