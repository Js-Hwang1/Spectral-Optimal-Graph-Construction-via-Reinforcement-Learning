// gcc -O3 -std=c11 -Wall -Wextra circ_graph.c -o circ_graph -lm

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

// forward declarations for builders used later
static void steps_circ_anchors_evenmid(int n, int kk, const double *anchors, int na, int *out_h, int *out_t, int **out_steps);
static void build_adj_from_circ_full(int n, const int* steps, int t, int h, double* A);
static void build_adj_from_bip_shifts(int n, const int* Sh, int t, double* A);

// Local spectral refiner for circulant steps (odd-n improvement)




#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int is_prime_int_local(int n){ if(n<2) return 0; if(n%2==0) return (n==2); for(int d=3; d*d<=n; d+=2) if(n%d==0) return 0; return 1; }

static long long gcdll(long long a, long long b) {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b) { long long t = a % b; a = b; b = t; }
    return a ? a : 1;
}

static long long gcd_with_all(int n, const int *steps, int t) {
    long long g = n;
    for (int i = 0; i < t; ++i) g = gcdll(g, steps[i]);
    return g;
}

static int clamp_int(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}


// Deterministic coprime correction: adjust largest steps downward first (then upward) minimally.
static void enforce_coprime(int n, int *steps, int t, int hi, unsigned char *used) {
    long long g = gcd_with_all(n, steps, t);
    if (g == 1) return;

    // Precompute order in which to try indices: largest to smallest.
    int order[t];
    for (int i = 0; i < t; ++i) order[i] = i;
    // Simple insertion sort by value (ascending), then traverse descending
    for (int i = 1; i < t; ++i) {
        int j = i, key = steps[i], idx = order[i];
        while (j > 0 && steps[j-1] > key) {
            steps[j] = steps[j-1];
            order[j] = order[j-1];
            --j;
        }
        steps[j] = key;
        order[j] = idx;
    }
    // Rebuild original steps from sorted copy
    int sorted[t]; for (int i = 0; i < t; ++i) sorted[i] = steps[i];
    // Map back: we don’t actually need original indices for connectivity; we’ll modify values directly.

    // Try modifying from largest downward.
    for (int pass = 0; pass < t && g != 1; ++pass) {
        int idx = t - 1 - pass;
        int s0 = sorted[idx];
        used[s0] = 0; // temporarily free it
        // Try minimal delta adjustments: prefer downward then upward.
        int chosen = -1;
        for (int d = 1; d <= hi && chosen < 0; ++d) {
            int cand = s0 - d;
            if (cand >= 1 && !used[cand]) {
                // test gcd if we take cand
                long long gg = n;
                for (int j = 0; j < t; ++j) {
                    gg = gcdll(gg, (j == idx ? cand : sorted[j]));
                }
                if (gg == 1) { chosen = cand; break; }
            }
            cand = s0 + d;
            if (cand <= hi && !used[cand]) {
                long long gg = n;
                for (int j = 0; j < t; ++j) {
                    gg = gcdll(gg, (j == idx ? cand : sorted[j]));
                }
                if (gg == 1) { chosen = cand; break; }
            }
        }
        if (chosen < 0) {
            // As a last resort, pick the smallest free integer that yields gcd 1.
            for (int cand = 1; cand <= hi && chosen < 0; ++cand) {
                if (used[cand]) continue;
                long long gg = n;
                for (int j = 0; j < t; ++j) gg = gcdll(gg, (j == idx ? cand : sorted[j]));
                if (gg == 1) chosen = cand;
            }
        }
        if (chosen >= 0) {
            sorted[idx] = chosen;
            used[chosen] = 1;
            g = gcd_with_all(n, sorted, t);
        } else {
            // Should not occur; restore and continue (graph may be disconnected in pathological case).
            used[s0] = 1;
        }
    }
    // Copy back corrected, but keep ascending order property irrelevant; preserve as chosen.
    for (int i = 0; i < t; ++i) steps[i] = sorted[i];
}

// ---------- Analytic step/shifts constructors and exact spectra ----------

// Compute Laplacian nontrivial extrema for a circulant defined by t positive steps in [1, floor(n/2)],
// and an optional single antipodal step if h==1 (only when n even and kk odd in our usage).
static void extrema_circulant(int n, const int *steps, int t, int h, double *lam_min, double *lam_max) {
    double lmin = INFINITY, lmax = -INFINITY;
    for (int i = 1; i <= n-1; ++i) {
        double val = 0.0;
        for (int j = 0; j < t; ++j) {
            double theta = 2.0 * M_PI * (double)i * (double)steps[j] / (double)n;
            val += 2.0 * (1.0 - cos(theta));
        }
        if (h == 1) {
            val += 1.0 - cos(M_PI * (double)i);
        }
        if (val < lmin) lmin = val;
        if (val > lmax) lmax = val;
    }
    *lam_min = lmin; *lam_max = lmax;
}

// Compute minimizing Fourier mode j* for current circulant (1..n-1)
static int floor_mode_circulant(int n, const int *steps, int t, int h) {
    double lmin = INFINITY; int jstar = 1;
    for (int i = 1; i <= n-1; ++i) {
        double val = 0.0;
        for (int m = 0; m < t; ++m) {
            double theta = 2.0 * M_PI * (double)i * (double)steps[m] / (double)n;
            val += 2.0 * (1.0 - cos(theta));
        }
        if (h == 1) val += 1.0 - cos(M_PI * (double)i);
        if (val < lmin) { lmin = val; jstar = i; }
    }
    return jstar;
}

// One-shot, analytic adjustment toward the bottleneck mode j*:
// - Build a candidate from a base seed (already unique and coprime),
// - Replace one step with the nearest available to round(n/(2*j*)) deterministically.
static void steps_circ_bottleneck_targeted(int n, int kk, int *out_h, int *out_t, int **out_steps,
                                           void (*base_builder)(int,int,int*,int*,int**)) {
    int h=0,t=0; int *steps=NULL; base_builder(n, kk, &h, &t, &steps);
    if (t <= 0) { *out_h=h; *out_t=t; *out_steps=steps; return; }
    int Rmax = n/2 - h;
    unsigned char *used = (unsigned char*)calloc((Rmax>0? Rmax+1:1), 1);
    for (int i=0;i<t;i++) { int r=steps[i]; if (r>=1 && r<=Rmax) used[r]=1; }
    int jstar = floor_mode_circulant(n, steps, t, h);
    int target = (int)llround((double)n / (2.0 * (double)jstar));
    target = clamp_int(target, 1, Rmax);
    // choose victim: the step farthest from target
    int victim = 0; int best_dist = -1;
    for (int i=0;i<t;i++){
        int d = abs(steps[i] - target);
        if (d > best_dist) { best_dist = d; victim = i; }
    }
    // find nearest free position to target (prefer lower)
    int chosen = -1;
    if (!used[target] || steps[victim]==target) chosen = target;
    else{
        for (int d=1; d<=Rmax; ++d){
            int a = target - d; if (a>=1 && (!used[a] || steps[victim]==a)) { chosen=a; break; }
            int b = target + d; if (b<=Rmax && (!used[b] || steps[victim]==b)) { chosen=b; break; }
        }
    }
    if (chosen >= 1){
        used[steps[victim]] = 0;
        steps[victim] = chosen;
        used[chosen] = 1;
        // ensure gcd(n, steps)=1 (deterministic correction if needed)
        if (gcd_with_all(n, steps, t) != 1) enforce_coprime(n, steps, t, Rmax, used);
    }
    free(used);
    *out_h=h; *out_t=t; *out_steps=steps;
}
// Helper: compute lmin and the minimizing Fourier mode j* in 1..n-1

// Evenly spaced midpoints for circulant: r_j ≈ (2j-1)/(2s) * Rmax (rounded), j=1..s.
static void steps_circ_even_midpoints(int n, int kk, int *out_h, int *out_t, int **out_steps) {
    int h = 0; if ((n % 2 == 0) && (kk % 2 == 1)) h = 1;
    int t = (kk - h) / 2;
    int Rmax = n/2 - h;
    int *steps = (t > 0 ? (int*)malloc(sizeof(int)*t) : NULL);
    unsigned char *used = (unsigned char*)calloc((Rmax > 0 ? Rmax+1 : 1), 1);
    for (int j = 1; j <= t; ++j) {
        double x = ((2.0*j - 1.0) * (double)Rmax) / (2.0 * (double)t);
        int r = (int)llround(x);
        r = clamp_int(r, 1, Rmax);
        if (used[r]) {
            // deterministically move to the nearest free, prefer lower
            int rr = r;
            for (int d = 1; d <= Rmax; ++d) {
                int a = r - d; if (a >= 1 && !used[a]) { rr = a; break; }
                int b = r + d; if (b <= Rmax && !used[b]) { rr = b; break; }
            }
            r = rr;
        }
        steps[j-1] = r; used[r] = 1;
    }
    // enforce gcd(n, steps) = 1 ignoring antipode
    if (t > 0) enforce_coprime(n, steps, t, Rmax, used);
    free(used);
    *out_h = h; *out_t = t; *out_steps = steps;
}

// Chebyshev-mapped nodes on [1, Rmax]: r_j ≈ round((theta_j/pi)*Rmax), theta_j = j*pi/(s+1)
static void steps_circ_chebyshev(int n, int kk, int *out_h, int *out_t, int **out_steps) {
    int h = 0; if ((n % 2 == 0) && (kk % 2 == 1)) h = 1;
    int t = (kk - h) / 2;
    int Rmax = n/2 - h;
    int *steps = (t > 0 ? (int*)malloc(sizeof(int)*t) : NULL);
    unsigned char *used = (unsigned char*)calloc((Rmax > 0 ? Rmax+1 : 1), 1);
    for (int j = 1; j <= t; ++j) {
        double theta = M_PI * (double)j / (double)(t + 1);
        int r = (int)llround( (theta / M_PI) * (double)Rmax );
        r = clamp_int(r, 1, Rmax);
        if (used[r]) {
            int rr = r;
            for (int d = 1; d <= Rmax; ++d) {
                int a = r - d; if (a >= 1 && !used[a]) { rr = a; break; }
                int b = r + d; if (b <= Rmax && !used[b]) { rr = b; break; }
            }
            r = rr;
        }
        steps[j-1] = r; used[r] = 1;
    }
    if (t > 0) enforce_coprime(n, steps, t, Rmax, used);
    free(used);
    *out_h = h; *out_t = t; *out_steps = steps;
}

// Golden-Beatty steps for circulant
static void steps_circ_golden(int n, int kk, int *out_h, int *out_t, int **out_steps) {
    int h = 0; if ((n % 2 == 0) && (kk % 2 == 1)) h = 1;
    int t = (kk - h) / 2;
    int Rmax = n/2 - h;
    int *steps = (t > 0 ? (int*)malloc(sizeof(int)*t) : NULL);
    unsigned char *used = (unsigned char*)calloc((Rmax > 0 ? Rmax+1 : 1), 1);
    const double phi = (sqrt(5.0)-1.0)/2.0;
    for (int j = 1; j <= t; ++j) {
        double f = fmod(j*phi, 1.0);
        int r = 1 + (int)floor(f * (double)Rmax);
        r = clamp_int(r, 1, Rmax);
        if (used[r]) {
            int rr = r;
            for (int d = 1; d <= Rmax; ++d) {
                int a = r - d; if (a >= 1 && !used[a]) { rr = a; break; }
                int b = r + d; if (b <= Rmax && !used[b]) { rr = b; break; }
            }
            r = rr;
        }
        steps[j-1] = r; used[r] = 1;
    }
    if (t > 0) enforce_coprime(n, steps, t, Rmax, used);
    free(used);
    *out_h = h; *out_t = t; *out_steps = steps;
}

// Dyadic steps for circulant
static void steps_circ_dyadic(int n, int kk, int *out_h, int *out_t, int **out_steps) {
    int h = 0; if ((n % 2 == 0) && (kk % 2 == 1)) h = 1;
    int t = (kk - h) / 2;
    int Rmax = n/2 - h;
    int *steps = (t > 0 ? (int*)malloc(sizeof(int)*t) : NULL);
    unsigned char *used = (unsigned char*)calloc((Rmax > 0 ? Rmax+1 : 1), 1);
    int cnt = 0;
    for (int r = 1; r <= Rmax && cnt < t; r <<= 1) {
        if (!used[r]) { steps[cnt++] = r; used[r] = 1; }
    }
    for (int j = cnt+1; j <= t; ++j) {
        double x = ((2.0*j - 1.0) * (double)Rmax) / (2.0 * (double)t);
        int r = (int)llround(x);
        r = clamp_int(r, 1, Rmax);
        if (used[r]) {
            int rr = r;
            for (int d = 1; d <= Rmax; ++d) {
                int a = r - d; if (a >= 1 && !used[a]) { rr = a; break; }
                int b = r + d; if (b <= Rmax && !used[b]) { rr = b; break; }
            }
            r = rr;
        }
        steps[j-1] = r; used[r] = 1;
    }
    if (t > 0) enforce_coprime(n, steps, t, Rmax, used);
    free(used);
    *out_h = h; *out_t = t; *out_steps = steps;
}

// Prefix steps: 1,2,3,... then fill as needed (simple ring lattice bias)
static void steps_circ_prefix(int n, int kk, int *out_h, int *out_t, int **out_steps) {
    int h = 0; if ((n % 2 == 0) && (kk % 2 == 1)) h = 1;
    int t = (kk - h) / 2;
    int Rmax = n/2 - h;
    int *steps = (t > 0 ? (int*)malloc(sizeof(int)*t) : NULL);
    int cnt=0; for(int r=1; r<=Rmax && cnt<t; ++r) steps[cnt++]=r;
    *out_h=h; *out_t=t; *out_steps=steps;
}

// Long-jump circulant for odd n, even k: take the r=t largest distances below n/2.
static void steps_circ_long_jumps(int n, int kk, int *out_h, int *out_t, int **out_steps) {
    int h = 0; if ((n % 2 == 0) && (kk % 2 == 1)) h = 1; // standard antipode bit for even n
    int t = (kk - h) / 2;
    int Rmax = n/2 - h;
    int *steps = (t > 0 ? (int*)malloc(sizeof(int)*t) : NULL);
    if (t <= 0) { *out_h=h; *out_t=t; *out_steps=steps; return; }
    // For odd n, this yields T = {(n-1)/2, (n-3)/2, ..., (n-(2t-1))/2}.
    // For even n, we still take descending from Rmax.
    int idx = 0;
    for (int p = 0; p < t; ++p) {
        int s = Rmax - p; // descending: Rmax, Rmax-1, ...
        if (s < 1) s = 1;
        steps[idx++] = s;
    }
    // Ensure uniqueness/connectivity deterministically
    unsigned char *used = (unsigned char*)calloc((Rmax > 0 ? Rmax+1 : 1), 1);
    for (int i=0;i<t;i++){ int r=steps[i]; if (r>=1 && r<=Rmax) used[r]=1; }
    if (t > 0) enforce_coprime(n, steps, t, Rmax, used);
    free(used);
    *out_h = h; *out_t = t; *out_steps = steps;
}

// Target low Fourier modes by placing steps near their antipodes: r_i ≈ round(n/(2*i)), i=1..t.
// Deterministic, analytic, no local refinement.
static void steps_circ_lowmode_antipodes(int n, int kk, int *out_h, int *out_t, int **out_steps) {
    int h = 0; if ((n % 2 == 0) && (kk % 2 == 1)) h = 1;
    int t = (kk - h) / 2;
    int Rmax = n/2 - h;
    int *steps = (t > 0 ? (int*)malloc(sizeof(int)*t) : NULL);
    if (t <= 0) { *out_h=h; *out_t=t; *out_steps=steps; return; }
    unsigned char *used = (unsigned char*)calloc((Rmax > 0 ? Rmax+1 : 1), 1);
    for (int i = 1; i <= t; ++i) {
        double rstar = (double)n / (2.0 * (double)i);
        int r = (int)llround(rstar);
        r = clamp_int(r, 1, Rmax);
        if (used[r]) {
            int rr = r;
            for (int d = 1; d <= Rmax; ++d) {
                int a = r - d; if (a >= 1 && !used[a]) { rr = a; break; }
                int b = r + d; if (b <= Rmax && !used[b]) { rr = b; break; }
            }
            r = rr;
        }
        steps[i-1] = r; used[r] = 1;
    }
    // Ensure connectivity deterministically
    if (t > 0) enforce_coprime(n, steps, t, Rmax, used);
    free(used);
    *out_h = h; *out_t = t; *out_steps = steps;
}

// Diameter-biased builder for odd n: include r = (n-1)/2, then fill remaining with even-midpoints.
static void steps_circ_diameter_bias(int n, int kk, int *out_h, int *out_t, int **out_steps) {
    int h = 0; if ((n % 2 == 0) && (kk % 2 == 1)) h = 1;
    int t = (kk - h) / 2;
    int Rmax = n/2 - h;
    int *steps = (t > 0 ? (int*)malloc(sizeof(int)*t) : NULL);
    if (t <= 0) { *out_h = h; *out_t = t; *out_steps = steps; return; }
    unsigned char *used = (unsigned char*)calloc((Rmax > 0 ? Rmax+1 : 1), 1);

    int idx = 0;
    if ((n % 2 == 1) && Rmax >= 1) {
        int r_d = Rmax; // (n-1)/2 when n odd
        steps[idx++] = r_d; used[r_d] = 1;
    }
    int remain = t - idx;
    if (remain > 0) {
        for (int j = 1; j <= remain; ++j) {
            double x = ((2.0*j - 1.0) * (double)Rmax) / (2.0 * (double)remain);
            int r = (int)llround(x);
            r = clamp_int(r, 1, Rmax);
            if (used[r]) {
                int rr = r;
                for (int d = 1; d <= Rmax; ++d) {
                    int a = r - d; if (a >= 1 && !used[a]) { rr = a; break; }
                    int b = r + d; if (b <= Rmax && !used[b]) { rr = b; break; }
                }
                r = rr;
            }
            steps[idx++] = r; used[r] = 1;
        }
    }
    if (t > 0) enforce_coprime(n, steps, t, Rmax, used);
    free(used);
    *out_h = h; *out_t = t; *out_steps = steps;
}

// Harmonic-Quenching Cayley (HQC): place r=k/2 steps near midpoints of 2k equal arcs.
// Works best for odd n, even k; deterministic rounding + uniqueness + gcd(n,steps)=1.
static void steps_circ_hqc(int n, int kk, int *out_h, int *out_t, int **out_steps) {
    int h = 0; if ((n % 2 == 0) && (kk % 2 == 1)) h = 1;
    int t = (kk - h) / 2; // number of undirected step lengths
    int B = n/2 - h;      // max step
    int *steps = (t > 0 ? (int*)malloc(sizeof(int)*t) : NULL);
    if (t <= 0) { *out_h = h; *out_t = t; *out_steps = steps; return; }
    unsigned char *used = (unsigned char*)calloc((B > 0 ? B+1 : 1), 1);

    // 1) Rounding to midpoints: s_i ≈ round(((2i-1) n)/(2k)) mapped to [1..B]
    int idx = 0;
    for (int i = 1; i <= t; ++i) {
        double approx = ((2.0*(double)i - 1.0) * (double)n) / (2.0 * (double)kk);
        int s = (int)llround(approx);
        s %= n; if (s <= 0) s += n; // map to 1..n-1
        if (s > B) s = n - s;      // map to 1..B
        if (s < 1) s = 1; if (s > B) s = B;
        if (used[s]) {
            int chosen = -1;
            for (int d = 1; d <= B; ++d) {
                int a = s - d; if (a >= 1 && !used[a]) { chosen = a; break; }
                int b = s + d; if (b <= B && !used[b]) { chosen = b; break; }
            }
            if (chosen >= 1) s = chosen;
        }
        steps[idx++] = s; used[s] = 1;
    }

    // 2) Ensure connectivity: gcd(n, steps)=1
    if (t > 0 && gcd_with_all(n, steps, t) != 1) enforce_coprime(n, steps, t, B, used);

    // 3) Optional micro-tune: one nearest ±1 swap to reduce dominant μ_j (j in {1,2})
    if (t > 0) {
        double theta = 2.0 * M_PI / (double)n;
        // compute μ1 and μ2
        double mu1 = 0.0, mu2 = 0.0;
        for (int i = 0; i < t; ++i) {
            mu1 += cos(theta * (double)steps[i]);
            mu2 += cos(2.0 * theta * (double)steps[i]);
        }
        mu1 *= 2.0; mu2 *= 2.0;
        int jtar = (mu2 > mu1) ? 2 : 1;
        double best_gain = 0.0; int best_i = -1; int best_new = -1;
        for (int i = 0; i < t; ++i) {
            int s = steps[i];
            int cands[2] = { s-1, s+1 };
            for (int ci = 0; ci < 2; ++ci) {
                int cand = cands[ci];
                if (cand < 1 || cand > B) continue;
                if (used[cand] && cand != s) continue;
                double cs = cos(jtar * theta * (double)s);
                double ct = cos(jtar * theta * (double)cand);
                double gain = 2.0*cs - 2.0*ct; // decrease in μ_j
                if (gain > best_gain) {
                    // test gcd preservation
                    int tmp = steps[i]; steps[i] = cand;
                    int ok = (gcd_with_all(n, steps, t) == 1);
                    steps[i] = tmp; // restore
                    if (ok) { best_gain = gain; best_i = i; best_new = cand; }
                }
            }
        }
        if (best_i >= 0 && best_new >= 1) {
            int old = steps[best_i]; used[old] = 0; steps[best_i] = best_new; used[best_new] = 1;
        }
    }

    *out_h = h; *out_t = t; *out_steps = steps;
    free(used);
}

// Fourier-minimax circulant builder (deterministic):
// Pick t steps in [1..Rmax] that (approximately) minimize max_{1<=j<=J} sum_{r in S} cos(2π j r / n)
// for a small fixed J (emphasizing the lowest modes). Greedy, deterministic tie-breaking.
static void steps_circ_fourier_minimax(int n, int kk, int *out_h, int *out_t, int **out_steps) {
    int h = 0; if ((n % 2 == 0) && (kk % 2 == 1)) h = 1;
    int t = (kk - h) / 2;
    int Rmax = n/2 - h;
    int *steps = (t > 0 ? (int*)malloc(sizeof(int)*t) : NULL);
    if (t <= 0) { *out_h=h; *out_t=t; *out_steps=steps; return; }

    int J = kk; if (J > n) J = n; if (J > 96) J = 96; if (J < 1) J = 1; // larger low-mode band
    double *S = (double*)calloc(J, sizeof(double)); // current cos sums per mode
    unsigned char *used = (unsigned char*)calloc((Rmax > 0 ? Rmax+1 : 1), 1);

    for (int sel = 0; sel < t; ++sel) {
        double best_val = INFINITY; int best_r = -1;
        // Try candidates in descending r to bias toward longer jumps on ties
        for (int r = Rmax; r >= 1; --r) {
            if (used[r]) continue;
            // compute new max across j if r is added
            double new_max = -INFINITY;
            for (int j = 1; j <= J; ++j) {
                double c = cos(2.0 * M_PI * (double)(j * r) / (double)n);
                double v = S[j-1] + c;
                if (v > new_max) new_max = v;
            }
            if (new_max < best_val || (new_max == best_val && r > best_r)) {
                best_val = new_max; best_r = r;
            }
        }
        steps[sel] = best_r; used[best_r] = 1;
        for (int j = 1; j <= J; ++j) S[j-1] += cos(2.0 * M_PI * (double)(j * best_r) / (double)n);
    }

    // Ensure connectivity
    if (gcd_with_all(n, steps, t) != 1) enforce_coprime(n, steps, t, Rmax, used);

    free(S); free(used);
    *out_h = h; *out_t = t; *out_steps = steps;
}


// Bipartite-circulant shifts: n even, N=n/2, degree t=|Sh|. Include shift 0 if include0=1.
static void shifts_bip_even_spacing(int N, int t, int include0, int **out_shifts, int *out_s) {
    if (t == N) include0 = 1; // ensure K_{N,N} case includes shift 0
    int need = t - (include0 ? 1 : 0);
    int *Sh = (int*)malloc(sizeof(int)*t);
    int idx = 0;
    if (include0) Sh[idx++] = 0;
    if (need > 0) {
        unsigned char *used = (unsigned char*)calloc((N > 0 ? N : 1), 1);
        if (include0) used[0] = 1;
        for (int j = 1; j <= need; ++j) {
            double x = ((double)j) * (double)N / (double)(need + 1);
            int s = (int)llround(x);
            if (s < 1) s = 1; if (s >= N) s = N-1;
            if (used[s]) {
                int ss = s;
                for (int d = 1; d < N; ++d) {
                    int a = s - d; if (a >= 1 && !used[a]) { ss = a; break; }
                    int b = s + d; if (b <= N-1 && !used[b]) { ss = b; break; }
                }
                s = ss;
            }
            Sh[idx++] = s; used[s] = 1;
        }
        // Ensure connectivity: require gcd(N, set_without_zero) = 1; if not, set first nonzero to 1 deterministically.
        if (need > 0) {
            int g = N;
            for (int i = 0; i < idx; ++i) if (Sh[i] != 0) g = (int)gcdll(g, Sh[i]);
            if (g != 1) {
                // Replace the largest nonzero by 1 (if not already present)
                int has1 = 0, pos_largest = -1, val_largest = -1;
                for (int i = 0; i < idx; ++i) {
                    if (Sh[i] == 1) { has1 = 1; break; }
                    if (Sh[i] != 0 && Sh[i] > val_largest) { val_largest = Sh[i]; pos_largest = i; }
                }
                if (!has1) {
                    Sh[pos_largest] = 1;
                }
            }
        }
        free(used);
    }
    *out_shifts = Sh; *out_s = idx;
}

// Chebyshev-like shifts for bipartite
static void shifts_bip_chebyshev(int N, int t, int include0, int **out_shifts, int *out_s) {
    if (t == N) include0 = 1;
    int need = t - (include0 ? 1 : 0);
    int *Sh = (int*)malloc(sizeof(int)*t);
    int idx = 0;
    if (include0) Sh[idx++] = 0;
    if (need > 0) {
        unsigned char *used = (unsigned char*)calloc((N > 0 ? N : 1), 1);
        if (include0) used[0] = 1;
        for (int j = 1; j <= need; ++j) {
            double theta = M_PI * (double)j / (double)(need + 1);
            int s = (int)llround( (theta / M_PI) * (double)(N-1) );
            if (s < 1) s = 1; if (s >= N) s = N-1;
            if (used[s]) {
                int ss = s;
                for (int d = 1; d < N; ++d) {
                    int a = s - d; if (a >= 1 && !used[a]) { ss = a; break; }
                    int b = s + d; if (b <= N-1 && !used[b]) { ss = b; break; }
                }
                s = ss;
            }
            Sh[idx++] = s; used[s] = 1;
        }
        // ensure shift 1 for connectivity if missing
        if (need > 0) {
            int has1 = 0; for (int i = 0; i < idx; ++i) if (Sh[i] == 1) { has1 = 1; break; }
            if (!has1) {
                int pos=-1, val=-1; for (int i=0;i<idx;i++){ if (Sh[i]!=0 && Sh[i]>val){ val=Sh[i]; pos=i; } }
                if (pos>=0) Sh[pos]=1;
            }
        }
        free(used);
    }
    *out_shifts = Sh; *out_s = idx;
}

// Golden-beatty shifts for bipartite
static void shifts_bip_golden(int N, int t, int include0, int **out_shifts, int *out_s) {
    if (t == N) include0 = 1;
    int need = t - (include0 ? 1 : 0);
    int *Sh = (int*)malloc(sizeof(int)*t);
    int idx = 0;
    if (include0) Sh[idx++] = 0;
    if (need > 0) {
        unsigned char *used = (unsigned char*)calloc((N > 0 ? N : 1), 1);
        if (include0) used[0] = 1;
        const double phi = (sqrt(5.0)-1.0)/2.0;
        for (int j = 1; j <= need; ++j) {
            double f = fmod(j*phi, 1.0);
            int s = 1 + (int)floor(f * (double)(N-1));
            if (s < 1) s = 1; if (s >= N) s = N-1;
            if (used[s]) {
                int ss = s;
                for (int d = 1; d < N; ++d) {
                    int a = s - d; if (a >= 1 && !used[a]) { ss = a; break; }
                    int b = s + d; if (b <= N-1 && !used[b]) { ss = b; break; }
                }
                s = ss;
            }
            Sh[idx++] = s; used[s] = 1;
        }
        // ensure shift 1 for connectivity if missing
        if (need > 0) {
            int has1 = 0; for (int i = 0; i < idx; ++i) if (Sh[i] == 1) { has1 = 1; break; }
            if (!has1) {
                int pos=-1, val=-1; for (int i=0;i<idx;i++){ if (Sh[i]!=0 && Sh[i]>val){ val=Sh[i]; pos=i; } }
                if (pos>=0) Sh[pos]=1;
            }
        }
        free(used);
    }
    *out_shifts = Sh; *out_s = idx;
}

// Exact lambda2 for bipartite-circulant with shifts Sh (size t).
static void extrema_bipcirc(int n, const int *Sh, int t, double *lam_min, double *lam_max) {
    int N = n/2; int k = t;
    double lmin = INFINITY; double lmax = 0.0;
    // j in 1..N-1 are nontrivial; adjacency eigenvalues ±|beta_j|; Laplacian: k ± |beta_j|.
    double max_beta = 0.0; double min_beta = INFINITY;
    for (int j = 1; j < N; ++j) {
        double re = 0.0, im = 0.0;
        for (int i = 0; i < t; ++i) {
            int s = Sh[i];
            double ang = 2.0 * M_PI * (double)(j * s) / (double)N;
            re += cos(ang); im += sin(ang);
        }
        double beta = sqrt(re*re + im*im);
        if (beta > max_beta) max_beta = beta;
        if (beta < min_beta) min_beta = beta;
        double lam1 = (double)k - beta; // small branch
        double lam2 = (double)k + beta; // large branch
        if (lam1 < lmin) lmin = lam1;
        if (lam2 > lmax) lmax = lam2;
    }
    // Also consider j=0: eigenvalues 0 and 2k (nontrivial max might be 2k)
    if (2.0*(double)k > lmax) lmax = 2.0*(double)k;
    // lmin cannot be negative due to Laplacian PSD; clamp tiny negatives
    if (lmin < 0.0 && lmin > -1e-12) lmin = 0.0;
    *lam_min = lmin; *lam_max = lmax;
}

// ---------- Adjacency builders ----------

static void build_adj_from_circ_full(int n, const int* steps, int t, int h, double* A){
    for(int i=0;i<n;i++) for(int j=0;j<n;j++) A[i*(size_t)n + j] = 0.0;
    int even=(n%2==0), half=n/2;
    for(int u=0; u<t; ++u){
        int r = steps[u];
        for(int i=0;i<n;i++){ int j=(i+r)%n; A[i*(size_t)n + j]=A[j*(size_t)n + i]=1.0; }
    }
    if(h==1 && even){ for(int i=0;i<n/2;i++){ int j=i+half; A[i*(size_t)n + j]=A[j*(size_t)n + i]=1.0; } }
}

static void build_adj_from_bip_shifts(int n,const int* Sh,int t,double* A){
    int N=n/2; for(int i=0;i<n;i++) for(int j=0;j<n;j++) A[i*(size_t)n + j]=0.0;
    for(int sidx=0; sidx<t; ++sidx){ int s=Sh[sidx]; for(int i=0;i<N;i++){ int u=i, v=N + ((i + s) % N); A[u*(size_t)n + v]=A[v*(size_t)n + u]=1.0; } }
}

// Build adjacency for 2-periodic alternating Cayley: even vertices use S0, odd use S1
// (AC2 removed)

// ---------- Numeric eigen fallback for general adjacency ----------

static int cmpd_increasing(const void* a, const void* b){ double x=*(const double*)a, y=*(const double*)b; return (x<y)?-1:(x>y); }

// Simple Jacobi eigenvalue solver (values only) for symmetric matrices (row-major).
static void jacobi_eigenvalues(double* A, int n, double* evals){
    const int maxsw = 50 * n * n;
    const double eps = 1e-12;
    for(int sw=0; sw<maxsw; ++sw){
        int p=0, q=1; double maxa=fabs(A[p*(size_t)n+q]);
        for(int i=0;i<n;i++) for(int j=i+1;j<n;j++){ double v=fabs(A[i*(size_t)n + j]); if(v>maxa){maxa=v;p=i;q=j;} }
        if(maxa < eps) break;
        double app=A[p*(size_t)n+p], aqq=A[q*(size_t)n+q], apq=A[p*(size_t)n+q];
        double phi=0.5*atan2(2.0*apq, (aqq-app)); double c=cos(phi), s=sin(phi);
        for(int i=0;i<n;i++) if(i!=p && i!=q){
            double aip=A[i*(size_t)n+p], aiq=A[i*(size_t)n+q];
            double nip=c*aip - s*aiq, niq=s*aip + c*aiq;
            A[i*(size_t)n+p]=A[p*(size_t)n+i]=nip;
            A[i*(size_t)n+q]=A[q*(size_t)n+i]=niq;
        }
        double app2=c*c*app - 2.0*s*c*apq + s*s*aqq;
        double aqq2=s*s*app + 2.0*s*c*apq + c*c*aqq;
        A[p*(size_t)n+p]=app2; A[q*(size_t)n+q]=aqq2; A[p*(size_t)n+q]=A[q*(size_t)n+p]=0.0;
    }
    for(int i=0;i<n;i++) evals[i]=A[i*(size_t)n+i];
    qsort(evals, n, sizeof(double), cmpd_increasing);
}

static double lambda2_from_adj_numeric(int n, const double* A){
    double* L=(double*)malloc(sizeof(double)*(size_t)n*(size_t)n);
    for(int i=0;i<n;i++){
        double d=0.0; for(int j=0;j<n;j++) d += A[i*(size_t)n + j];
        for(int j=0;j<n;j++) L[i*(size_t)n + j] = (i==j)? d : -A[i*(size_t)n + j];
    }
    double* W=(double*)malloc(sizeof(double)*(size_t)n*(size_t)n);
    memcpy(W, L, sizeof(double)*(size_t)n*(size_t)n);
    double* e=(double*)malloc(sizeof(double)*n);
    jacobi_eigenvalues(W, n, e);
    double l2 = e[1];
    free(L); free(W); free(e);
    return l2;
}

// (AC2 numeric extremes helper removed)

// ---------- 2-lift deterministic expansion ----------

static inline int sign_pattern_var(int u,int v,int level,int variant){
    if(variant==0){
        return ((u + v + level) & 1) ? -1 : +1;
    } else {
        unsigned int x = (unsigned int)(u ^ v);
        int parity = __builtin_parity(x);
        int s = (parity ^ (level & 1));
        return s ? -1 : +1;
    }
}

// (Removed) Conjugate-circulant union branch to avoid numeric path for large odd n.

// ---------- Paley augmentation for odd prime n (n ≡ 1 mod 4), k >= (n-1)/2 ----------

static int modpow_int(int a,int e,int m){ long long res=1, base=((a%m)+m)%m; while(e>0){ if(e&1) res=(res*base)%m; base=(base*base)%m; e>>=1; } return (int)res; }

static void paley_build_steps(int n,int** out_steps,int* out_t){
    int Rmax=(n-1)/2; int cap=Rmax; int* S=(int*)malloc(sizeof(int)*cap); int t=0;
    for(int r=1;r<=Rmax;r++){
        int val = modpow_int(r, (n-1)/2, n);
        if(val==1){ S[t++]=r; }
    }
    *out_steps=S; *out_t=t;
}

static double paley_augment_lambda2(int n,int k){
    if(!is_prime_int_local(n) || (n%4)!=1) return -1e300;
    int kp=(n-1)/2; if(k<kp) return -1e300; if((k%2)!=0) return -1e300; // odd n implies even k
    int* S0=NULL; int t0=0; paley_build_steps(n,&S0,&t0);
    int Rmax=(n-1)/2;
    unsigned char* used=(unsigned char*)calloc(Rmax+1,1);
    for(int i=0;i<t0;i++) used[S0[i]]=1;
    int need=(k-kp)/2; // extra pairs needed
    // candidate order: dyadic powers, then midpoints fill
    int* extra=(need>0)? (int*)malloc(sizeof(int)*need) : NULL; int ce=0;
    for(int r=1; r<=Rmax && ce<need; r<<=1){ if(r<=Rmax && !used[r]){ extra[ce++]=r; used[r]=1; } if(r==0) break; }
    if(ce<need){
        int sguess = need*4 + 8; if(sguess<need) sguess=need;
        for(int j=1; ce<need && j<=sguess; ++j){ double x=((2.0*j-1.0)*Rmax)/(2.0*sguess); int r=(int)llround(x); if(r<1) r=1; if(r>Rmax) r=Rmax; if(!used[r]){ extra[ce++]=r; used[r]=1; } }
        for(int r=1; ce<need && r<=Rmax; ++r){ if(!used[r]){ extra[ce++]=r; used[r]=1; } }
    }
    // build combined steps
    int tall=t0+ce; int* Sall=(int*)malloc(sizeof(int)*tall);
    for(int i=0;i<t0;i++) Sall[i]=S0[i]; for(int i=0;i<ce;i++) Sall[t0+i]=extra[i];
    double l2; double lmax;
    extrema_circulant(n,Sall,tall,0,&l2,&lmax);
    free(S0); free(Sall); free(used); if(extra) free(extra);
    return l2;
}

static double* two_lift_once(const double* A, int m, int level, int variant){
    int n2 = 2*m;
    double* B=(double*)calloc((size_t)n2*(size_t)n2, sizeof(double));
    for(int i=0;i<m;i++){
        for(int j=i+1;j<m;j++){
            if(A[i*(size_t)m + j] > 0.5){
                int s = sign_pattern_var(i,j,level,variant);
                if(s>0){
                    B[(i+0*m)*(size_t)n2 + (j+0*m)] = 1.0; B[(j+0*m)*(size_t)n2 + (i+0*m)] = 1.0;
                    B[(i+1*m)*(size_t)n2 + (j+1*m)] = 1.0; B[(j+1*m)*(size_t)n2 + (i+1*m)] = 1.0;
                }else{
                    B[(i+0*m)*(size_t)n2 + (j+1*m)] = 1.0; B[(j+1*m)*(size_t)n2 + (i+0*m)] = 1.0;
                    B[(i+1*m)*(size_t)n2 + (j+0*m)] = 1.0; B[(j+0*m)*(size_t)n2 + (i+1*m)] = 1.0;
                }
            }
        }
    }
    return B;
}

// Build best base adjacency among our analytic families (returns allocated A of size n0*n0)
static double* build_best_base_adj(int n0,int k0){
    double best=-1e300; double* bestA=NULL;
    if((n0%2)==0 && k0<=n0/2){
        // bipartite seeds
        for(int inc0=0; inc0<=1; ++inc0){
            int *Sh=NULL, tSh=0; double lmin,lmax;
            shifts_bip_even_spacing(n0/2, k0, inc0, &Sh, &tSh);
            extrema_bipcirc(n0, Sh, tSh, &lmin, &lmax);
            if(lmin>best){ best=lmin; if(bestA) free(bestA); bestA=(double*)malloc(sizeof(double)*(size_t)n0*(size_t)n0); build_adj_from_bip_shifts(n0, Sh, tSh, bestA);} free(Sh);
            Sh=NULL; tSh=0; shifts_bip_chebyshev(n0/2, k0, inc0, &Sh, &tSh);
            extrema_bipcirc(n0, Sh, tSh, &lmin, &lmax);
            if(lmin>best){ best=lmin; if(bestA) free(bestA); bestA=(double*)malloc(sizeof(double)*(size_t)n0*(size_t)n0); build_adj_from_bip_shifts(n0, Sh, tSh, bestA);} free(Sh);
            Sh=NULL; tSh=0; shifts_bip_golden(n0/2, k0, inc0, &Sh, &tSh);
            extrema_bipcirc(n0, Sh, tSh, &lmin, &lmax);
            if(lmin>best){ best=lmin; if(bestA) free(bestA); bestA=(double*)malloc(sizeof(double)*(size_t)n0*(size_t)n0); build_adj_from_bip_shifts(n0, Sh, tSh, bestA);} free(Sh);
        }
    }
    // circulants
    int h=0,t=0; int* S=NULL; double lmin,lmax;
    steps_circ_even_midpoints(n0, k0, &h, &t, &S); extrema_circulant(n0, S, t, h, &lmin, &lmax);
    if(lmin>best){ best=lmin; if(bestA) free(bestA); bestA=(double*)malloc(sizeof(double)*(size_t)n0*(size_t)n0); build_adj_from_circ_full(n0, S, t, h, bestA);} free(S);
    steps_circ_chebyshev(n0, k0, &h, &t, &S); extrema_circulant(n0, S, t, h, &lmin, &lmax);
    if(lmin>best){ best=lmin; if(bestA) free(bestA); bestA=(double*)malloc(sizeof(double)*(size_t)n0*(size_t)n0); build_adj_from_circ_full(n0, S, t, h, bestA);} free(S);
    steps_circ_golden(n0, k0, &h, &t, &S); extrema_circulant(n0, S, t, h, &lmin, &lmax);
    if(lmin>best){ best=lmin; if(bestA) free(bestA); bestA=(double*)malloc(sizeof(double)*(size_t)n0*(size_t)n0); build_adj_from_circ_full(n0, S, t, h, bestA);} free(S);
    steps_circ_dyadic(n0, k0, &h, &t, &S); extrema_circulant(n0, S, t, h, &lmin, &lmax);
    if(lmin>best){ best=lmin; if(bestA) free(bestA); bestA=(double*)malloc(sizeof(double)*(size_t)n0*(size_t)n0); build_adj_from_circ_full(n0, S, t, h, bestA);} free(S);
    // anchors few
    {
        double anch[2]={0.25, 1.0/3.0}; steps_circ_anchors_evenmid(n0,k0,anch,2,&h,&t,&S); extrema_circulant(n0,S,t,h,&lmin,&lmax);
        if(lmin>best){ best=lmin; if(bestA) free(bestA); bestA=(double*)malloc(sizeof(double)*(size_t)n0*(size_t)n0); build_adj_from_circ_full(n0, S, t, h, bestA);} free(S);
    }
    return bestA;
}

static int try_two_lift_lambda2(int n,int k,double* out_l2){
    if(n%2!=0) return 0; // need even n to lift at least once
    int s=0, n0=n;
    while((n0%2==0) && (n0/2 >= k+1)) { n0/=2; s++; }
    if(s==0) return 0;
    double* Ab = build_best_base_adj(n0, k);
    if(!Ab) return 0;
    double best = -1e300;
    for(int variant=0; variant<2; ++variant){
        double* Acur = Ab; int curN = n0;
        for(int lev=0; lev<s; ++lev){
            double* Anext = two_lift_once(Acur, curN, lev, variant);
            if(Acur!=Ab) free(Acur);
            Acur = Anext; curN *= 2;
        }
        double l2 = lambda2_from_adj_numeric(curN, Acur);
        if(l2 > best) best = l2;
        free(Acur);
    }
    free(Ab);
    *out_l2 = best;
    return 1;
}

// Build bipartite shifts by removing t = N-k nearly Chebyshev-distributed shifts from full set {0..N-1}.
static void shifts_bip_knn_minus_t(int N, int k, int **out_shifts, int *out_s) {
    int t = N - k;
    int *keep = (int*)calloc(N, sizeof(int));
    for (int i=0;i<N;i++) keep[i]=1; // start with all shifts
    int removed = 0;
    if (t > 0) {
        unsigned char *rem = (unsigned char*)calloc(N,1);
        for (int i=1; i<=t; ++i) {
            double theta = M_PI * (double)i / (double)(t+1);
            int s = (int)llround( (theta / M_PI) * (double)(N-1) );
            if (s < 0) s = 0; if (s >= N) s = N-1;
            if (!rem[s]) { rem[s]=1; removed++; }
        }
        // fill remainder if duplicates
        for (int s=0; removed<t && s<N; ++s) if (!rem[s]) { rem[s]=1; removed++; }
        for (int s=0; s<N; ++s) if (rem[s]) keep[s]=0;
        free(rem);
    }
    // collect kept shifts
    int *Sh = (int*)malloc(sizeof(int)*k);
    int idx=0; for (int s=0; s<N; ++s) if (keep[s]) Sh[idx++]=s;
    // connectivity: ensure a 1 shift present among nonzero
    int has1=0, posnz=-1; for (int i=0;i<idx;i++){ if (Sh[i]==1) has1=1; if (Sh[i]!=0) posnz=i; }
    if (!has1 && posnz>=0) Sh[posnz]=1;
    free(keep);
    *out_shifts = Sh; *out_s = idx;
}

// Remove t=N-k shifts via evenly spaced indices across 0..N-1
static void shifts_bip_remove_even(int N, int k, int **out_shifts, int *out_s) {
    int t = N - k;
    int *keep = (int*)calloc(N, sizeof(int));
    for (int i=0;i<N;i++) keep[i]=1;
    if (t > 0) {
        unsigned char *rem = (unsigned char*)calloc(N,1);
        for (int i=1; i<=t; ++i) {
            double x = ((double)i) * (double)N / (double)(t + 1);
            int s = (int)llround(x);
            if (s < 0) s = 0; if (s >= N) s = N-1;
            if (!rem[s]) rem[s]=1; else {
                // fill next free
                for (int d=1; d<N; ++d) {
                    int a = s - d; if (a>=0 && !rem[a]) { rem[a]=1; break; }
                    int b = s + d; if (b<N && !rem[b]) { rem[b]=1; break; }
                }
            }
        }
        for (int s=0; s<N; ++s) if (rem[s]) keep[s]=0;
        free(rem);
    }
    int *Sh = (int*)malloc(sizeof(int)*k);
    int idx=0; for (int s=0; s<N; ++s) if (keep[s]) Sh[idx++]=s;
    int has1=0, posnz=-1; for (int i=0;i<idx;i++){ if (Sh[i]==1) has1=1; if (Sh[i]!=0) posnz=i; }
    if (!has1 && posnz>=0) Sh[posnz]=1;
    free(keep);
    *out_shifts = Sh; *out_s = idx;
}

// Remove t=N-k shifts via golden-Beatty positions over 0..N-1
static void shifts_bip_remove_golden(int N, int k, int **out_shifts, int *out_s) {
    int t = N - k;
    int *keep = (int*)calloc(N, sizeof(int));
    for (int i=0;i<N;i++) keep[i]=1;
    if (t > 0) {
        unsigned char *rem = (unsigned char*)calloc(N,1);
        const double phi = (sqrt(5.0)-1.0)/2.0;
        for (int i=1; i<=t; ++i) {
            double f = fmod(i*phi, 1.0);
            int s = (int)floor(f * (double)N);
            if (s < 0) s=0; if (s>=N) s=N-1;
            if (!rem[s]) rem[s]=1; else {
                for (int d=1; d<N; ++d) {
                    int a = s - d; if (a>=0 && !rem[a]) { rem[a]=1; break; }
                    int b = s + d; if (b<N && !rem[b]) { rem[b]=1; break; }
                }
            }
        }
        for (int s=0; s<N; ++s) if (rem[s]) keep[s]=0;
        free(rem);
    }
    int *Sh = (int*)malloc(sizeof(int)*k);
    int idx=0; for (int s=0; s<N; ++s) if (keep[s]) Sh[idx++]=s;
    int has1=0, posnz=-1; for (int i=0;i<idx;i++){ if (Sh[i]==1) has1=1; if (Sh[i]!=0) posnz=i; }
    if (!has1 && posnz>=0) Sh[posnz]=1;
    free(keep);
    *out_shifts = Sh; *out_s = idx;
}

// Prefix-stable dyadic-order shifts for bipartite-circulant (deterministic).
// Generates an ordering of distinct shifts in 0..N-1 by dyadic midpoints on the circle.
// Taking the first t of this order yields a monotone edge-addition chain in k, hence λ2 is nondecreasing in k.
static void shifts_bip_prefix_dyadic_order(int N, int include0, int **out_order, int *out_len) {
    int cap = N;
    int *ord = (int*)malloc(sizeof(int)*cap);
    unsigned char *used = (unsigned char*)calloc((N>0?N:1), 1);
    int cnt = 0;
    if (include0) { ord[cnt++] = 0; used[0]=1; }
    // Generate fractions a/2^{m+1} with odd a, increasing in m; map to s = round(a*N/2^{m+1}) mod N
    for (int m = 0; cnt < cap && m < 20; ++m) { // m up to 20 suffices for N<=1e6
        int denom = 1 << (m+1);
        for (int a = 1; a < denom; a += 2) {
            long long num = (long long)a * (long long)N;
            int s = (int)((num + (denom/2)) / denom); // round
            if (s >= N) s %= N;
            if (s < 0) s += N;
            if (!used[s]) { ord[cnt++] = s; used[s]=1; if (cnt>=cap) break; }
        }
    }
    // Fill any remaining slots linearly (shouldn't happen)
    for (int s=0; cnt<cap && s<N; ++s) if (!used[s]) { ord[cnt++]=s; used[s]=1; }
    free(used);
    *out_order = ord; *out_len = cnt;
}

// Circulant steps with anchor fractions (e.g., 1/4, 1/3), then fill by even-midpoints
static void steps_circ_anchors_evenmid(int n, int kk, const double *anchors, int na, int *out_h, int *out_t, int **out_steps) {
    int h = 0; if ((n % 2 == 0) && (kk % 2 == 1)) h = 1;
    int t = (kk - h) / 2;
    int Rmax = n/2 - h;
    int *steps = (t > 0 ? (int*)malloc(sizeof(int)*t) : NULL);
    unsigned char *used = (unsigned char*)calloc((Rmax > 0 ? Rmax+1 : 1), 1);
    int idx=0;
    for (int i=0; i<na && idx<t; ++i) {
        int r = (int)llround(anchors[i] * (double)n);
        r = clamp_int(r, 1, Rmax);
        if (!used[r]) { steps[idx++]=r; used[r]=1; }
    }
    for (int j=idx+1; j<=t; ++j) {
        double x = ((2.0*j - 1.0) * (double)Rmax) / (2.0 * (double)t);
        int r = (int)llround(x);
        r = clamp_int(r, 1, Rmax);
        if (used[r]) {
            int rr=r;
            for (int d=1; d<=Rmax; ++d) {
                int a=r-d; if (a>=1 && !used[a]) { rr=a; break; }
                int b=r+d; if (b<=Rmax && !used[b]) { rr=b; break; }
            }
            r=rr;
        }
        steps[j-1]=r; used[r]=1;
    }
    if (t > 0) enforce_coprime(n, steps, t, Rmax, used);
    free(used);
    *out_h=h; *out_t=t; *out_steps=steps;
}

// Bilayer-circulant (even n): within-partition circulant steps on length N=n/2, cross-part shifts; exact λ2.
static double lambda2_bilayer_circ_exact(int n, const int* side_steps, int s, int hside, const int* Sh, int t){
    int N = n/2;
    int deg_side = (hside?1:0) + 2*s;
    int k = deg_side + t;
    double best = 1e300;
    for(int j=1;j<N;j++){
        double mu=0.0;
        for(int i=0;i<s;i++){
            int r=side_steps[i];
            double ang=2.0*M_PI*(double)(j*r)/(double)N;
            mu += 2.0*cos(ang);
        }
        if(hside){ mu += (j%2==0)? 1.0 : -1.0; }
        double re=0.0, im=0.0;
        for(int i=0;i<t;i++){
            int sft=Sh[i]; double ang=2.0*M_PI*(double)(j*sft)/(double)N; re += cos(ang); im += sin(ang);
        }
        double beta = sqrt(re*re + im*im);
        double lam = (double)k - (mu + beta);
        if(lam < best) best = lam;
    }
    if(best < 0.0 && best > -1e-12) best = 0.0;
    return best;
}

static double lambda_max_bilayer_circ_exact(int n, const int* side_steps, int s, int hside, const int* Sh, int t){
    int N = n/2;
    int d = ((hside?1:0) + 2*s) + t;
    double worst = -1e300; // we will compute max Laplacian eigenvalue
    for(int j=0;j<N;j++){
        // include j=0 as well because A's min eigenvalue can be at 0
        double mu=0.0;
        if(j==0){
            mu = (hside?1.0:0.0) + 2.0*s; // at j=0, cos(0)=1
        }else{
            for(int i=0;i<s;i++){
                int r=side_steps[i]; double ang=2.0*M_PI*(double)(j*r)/(double)N; mu += 2.0*cos(ang);
            }
            if(hside){ mu += (j%2==0)? 1.0 : -1.0; }
        }
        double re=0.0, im=0.0;
        for(int i=0;i<t;i++){ int sft=Sh[i]; double ang=2.0*M_PI*(double)(j*sft)/(double)N; re += cos(ang); im += sin(ang); }
        double beta = sqrt(re*re + im*im);
        // adjacency eigen candidates: mu ± beta; minimum of A is mu - beta
        double a_min = mu - beta;
        double lap_max = (double)d - a_min;
        if(lap_max > worst) worst = lap_max;
    }
    return worst;
}

// Choose the best analytic construction deterministically and compute lambda2.
static void best_lambda2_analytic(int n, int k, double *lambda2_out) {
    // Special Paley case: n prime, n ≡ 1 (mod 4), k=(n-1)/2
    if (k == (n-1)/2) {
        int is_prime = 1;
        if (n < 2) is_prime = 0; else if (n % 2 == 0) is_prime = (n==2);
        for (int d=3; is_prime && d*d<=n; d+=2) if (n % d == 0) is_prime = 0;
        if (is_prime && (n % 4 == 1)) {
            double l2 = 0.5 * ((double)n - sqrt((double)n));
            *lambda2_out = l2; return;
        }
    }
    if ((n % 2 == 0) && (k <= n/2)) {
        // Even n, not dense: Monotone prefix-stable bipartite-circulant chain.
        int N = n/2;
        int *order=NULL; int ordlen=0; shifts_bip_prefix_dyadic_order(N, 1, &order, &ordlen);
        if (k > ordlen) k = ordlen;
        int *Sh = (int*)malloc(sizeof(int)*k);
        unsigned char *used = (unsigned char*)calloc((N>0?N:1),1);
        for (int i=0;i<k;i++) { Sh[i]=order[i]; if (Sh[i]>=0 && Sh[i]<N) used[Sh[i]]=1; }
        // Ensure connectivity: require gcd(N, nonzero shifts) = 1
        long long g = N;
        for (int i=0;i<k;i++){ int s=Sh[i]; if (s!=0) g = gcdll(g, s); }
        if (g != 1) {
            if (!used[1]) {
                // replace the largest nonzero shift by 1 deterministically
                int pos=-1, val=-1;
                for (int i=0;i<k;i++){ if (Sh[i]!=0 && Sh[i]>val){ val=Sh[i]; pos=i; } }
                if (pos>=0) { used[Sh[pos]]=0; Sh[pos]=1; used[1]=1; }
            } else {
                // find smallest cand coprime to N not used
                int cand=2; while (cand<N && (used[cand] || gcdll(N,cand)!=1)) cand++;
                if (cand<N){ int pos=-1, val=-1; for (int i=0;i<k;i++){ if (Sh[i]!=0 && Sh[i]>val){ val=Sh[i]; pos=i; } } if(pos>=0){ used[Sh[pos]]=0; Sh[pos]=cand; used[cand]=1; } }
            }
        }
        double lmin_bip, lmax_bip; extrema_bipcirc(n, Sh, k, &lmin_bip, &lmax_bip);
        free(order); free(Sh);
        free(used);
        *lambda2_out = (lmin_bip < 0.0 && lmin_bip > -1e-12) ? 0.0 : lmin_bip;
        return;

        // unreachable legacy even-n code
#if 0

        // bipartite-circulant with three shift rules; try include0 in {0,1}
        for (int inc0 = 0; inc0 <= 1; ++inc0) {
            int include0 = inc0;
            int *Sh = NULL; int tSh = 0; double lmin_bip, lmax_bip;
            shifts_bip_even_spacing(n/2, k, include0, &Sh, &tSh);
            extrema_bipcirc(n, Sh, tSh, &lmin_bip, &lmax_bip);
            if (lmin_bip > best) best = lmin_bip; free(Sh);

            Sh = NULL; tSh = 0; shifts_bip_chebyshev(n/2, k, include0, &Sh, &tSh);
            extrema_bipcirc(n, Sh, tSh, &lmin_bip, &lmax_bip);
            if (lmin_bip > best) best = lmin_bip; free(Sh);

            Sh = NULL; tSh = 0; shifts_bip_golden(n/2, k, include0, &Sh, &tSh);
            extrema_bipcirc(n, Sh, tSh, &lmin_bip, &lmax_bip);
            if (lmin_bip > best) best = lmin_bip; free(Sh);
        }

        // K_{N,N} minus t perfect matchings (remove-shift design)
        {
            int *Sh=NULL; int tSh=0; double lmin_bip,lmax_bip;
            shifts_bip_knn_minus_t(n/2, k, &Sh, &tSh);
            extrema_bipcirc(n, Sh, tSh, &lmin_bip, &lmax_bip);
            if (lmin_bip > best) best = lmin_bip;
            free(Sh);
        }

        // Bilayer-circulant candidates: split k = k_side + t_cross (expanded splits)
        {
            int N = n/2;
            int base = k/2;
            int deltas[] = {-5,-4,-3,-2,-1,0,1,2,3,4,5};
            int nd = (int)(sizeof(deltas)/sizeof(deltas[0]));
            for(int ti=0; ti<nd; ++ti){
                int t_cross = base + deltas[ti]; if(t_cross<0) t_cross=0; if(t_cross>N) t_cross=N;
                int kk_side = k - t_cross; if(kk_side<0) continue;
                // side families
                for(int sf=0; sf<5; ++sf){
                    int hs=0, ts=0; int* S=NULL;
                    if(sf==0) steps_circ_even_midpoints(N, kk_side, &hs, &ts, &S);
                    else if(sf==1) steps_circ_chebyshev(N, kk_side, &hs, &ts, &S);
                    else if(sf==2) steps_circ_golden(N, kk_side, &hs, &ts, &S);
                    else if(sf==3) steps_circ_dyadic(N, kk_side, &hs, &ts, &S);
                    else {
                        double anch[2] = {0.25, 1.0/3.0};
                        steps_circ_anchors_evenmid(N, kk_side, anch, 2, &hs, &ts, &S);
                    }
                    for(int inc0=0; inc0<=1; ++inc0){
                        // shift families
                        for(int shf=0; shf<3; ++shf){
                            int *Sh=NULL; int tSh=0;
                            if(shf==0) shifts_bip_even_spacing(N, t_cross, inc0, &Sh, &tSh);
                            else if(shf==1) shifts_bip_chebyshev(N, t_cross, inc0, &Sh, &tSh);
                            else shifts_bip_golden(N, t_cross, inc0, &Sh, &tSh);
                            double l2 = lambda2_bilayer_circ_exact(n, S, ts, hs, Sh, tSh);
                            if(l2 > best) best = l2;
                            free(Sh);
                        }
                    }
                    free(S);
                }
            }
        }

        // circulant even-midpoints
        int h=0,t=0; int *steps=NULL; steps_circ_even_midpoints(n, k, &h, &t, &steps);
        double lmin1,lmax1; extrema_circulant(n, steps, t, h, &lmin1, &lmax1);
        if (lmin1 > best) best = lmin1; free(steps);

        // circulant chebyshev
        h=0; t=0; steps=NULL; steps_circ_chebyshev(n, k, &h, &t, &steps);
        double lmin2,lmax2; extrema_circulant(n, steps, t, h, &lmin2, &lmax2);
        if (lmin2 > best) best = lmin2; free(steps);

        // circulant golden-beatty
        h=0; t=0; steps=NULL; steps_circ_golden(n, k, &h, &t, &steps);
        double lmin3,lmax3; extrema_circulant(n, steps, t, h, &lmin3, &lmax3);
        if (lmin3 > best) best = lmin3; free(steps);

        // circulant dyadic
        h=0; t=0; steps=NULL; steps_circ_dyadic(n, k, &h, &t, &steps);
        double lmin4,lmax4; extrema_circulant(n, steps, t, h, &lmin4, &lmax4);
        if (lmin4 > best) best = lmin4; free(steps);

        // circulant with anchor fractions {1/4, 1/3}
        {
            double anch[2] = {0.25, 1.0/3.0};
            h=0; t=0; steps=NULL; steps_circ_anchors_evenmid(n, k, anch, 2, &h, &t, &steps);
            double la,ua; extrema_circulant(n, steps, t, h, &la, &ua); if (la > best) best = la; free(steps);
        }
        // circulant with anchor fractions {1/5, 2/5}
        {
            double anch[2] = {0.2, 0.4};
            h=0; t=0; steps=NULL; steps_circ_anchors_evenmid(n, k, anch, 2, &h, &t, &steps);
            double la,ua; extrema_circulant(n, steps, t, h, &la, &ua); if (la > best) best = la; free(steps);
        }
        // circulant with anchor fractions {1/6, 1/3}
        {
            double anch[2] = {1.0/6.0, 1.0/3.0};
            h=0; t=0; steps=NULL; steps_circ_anchors_evenmid(n, k, anch, 2, &h, &t, &steps);
            double la,ua; extrema_circulant(n, steps, t, h, &la, &ua); if (la > best) best = la; free(steps);
        }
        // circulant with anchor fractions {1/8, 3/8}
        {
            double anch[2] = {1.0/8.0, 3.0/8.0};
            h=0; t=0; steps=NULL; steps_circ_anchors_evenmid(n, k, anch, 2, &h, &t, &steps);
            double la,ua; extrema_circulant(n, steps, t, h, &la, &ua); if (la > best) best = la; free(steps);
        }
        // circulant with anchor fractions {1/9, 2/9, 4/9}
        {
            double anch3[3] = {1.0/9.0, 2.0/9.0, 4.0/9.0};
            h=0; t=0; steps=NULL; steps_circ_anchors_evenmid(n, k, anch3, 3, &h, &t, &steps);
            double la,ua; extrema_circulant(n, steps, t, h, &la, &ua); if (la > best) best = la; free(steps);
        }
        // circulant with anchor fractions {1/10, 3/10}
        {
            double anch[2] = {1.0/10.0, 3.0/10.0};
            h=0; t=0; steps=NULL; steps_circ_anchors_evenmid(n, k, anch, 2, &h, &t, &steps);
            double la,ua; extrema_circulant(n, steps, t, h, &la, &ua); if (la > best) best = la; free(steps);
        }

        // circulant prefix steps 1,2,3,...
        h=0; t=0; steps=NULL; steps_circ_prefix(n, k, &h, &t, &steps);
        double lp,up; extrema_circulant(n, steps, t, h, &lp, &up); if (lp > best) best = lp; free(steps);
        // circulant with anchor fractions {1/7,2/7,3/7}
        {
            double anch3[3] = {1.0/7.0, 2.0/7.0, 3.0/7.0};
            h=0; t=0; steps=NULL; steps_circ_anchors_evenmid(n, k, anch3, 3, &h, &t, &steps);
            double la,ua; extrema_circulant(n, steps, t, h, &la, &ua); if (la > best) best = la; free(steps);
        }

        // Also consider complement-based candidates even when k <= n/2
        int d = (n - 1) - k;
        if (d >= 1) {
            double lam2_comp_best = -INFINITY;
            // circulant complement variants
            h=0; t=0; steps=NULL; steps_circ_even_midpoints(n, d, &h, &t, &steps);
            double cminA,cmaxA; extrema_circulant(n, steps, t, h, &cminA, &cmaxA);
            double lam2A = (double)n - cmaxA; if (lam2A > lam2_comp_best) lam2_comp_best = lam2A; free(steps);
            h=0; t=0; steps=NULL; steps_circ_chebyshev(n, d, &h, &t, &steps);
            double cminB,cmaxB; extrema_circulant(n, steps, t, h, &cminB, &cmaxB);
            double lam2B = (double)n - cmaxB; if (lam2B > lam2_comp_best) lam2_comp_best = lam2B; free(steps);
            h=0; t=0; steps=NULL; steps_circ_golden(n, d, &h, &t, &steps);
            double cminC,cmaxC; extrema_circulant(n, steps, t, h, &cminC, &cmaxC);
            double lam2C = (double)n - cmaxC; if (lam2C > lam2_comp_best) lam2_comp_best = lam2C; free(steps);
            h=0; t=0; steps=NULL; steps_circ_dyadic(n, d, &h, &t, &steps);
            double cminD,cmaxD; extrema_circulant(n, steps, t, h, &cminD, &cmaxD);
            double lam2D = (double)n - cmaxD; if (lam2D > lam2_comp_best) lam2_comp_best = lam2D; free(steps);
            // bipartite complement variants (only when n even)
            int *Shc=NULL; int tShc=0; double lminCmpl,lmaxCmpl;
            shifts_bip_even_spacing(n/2, d, 0, &Shc, &tShc);
            extrema_bipcirc(n, Shc, tShc, &lminCmpl, &lmaxCmpl);
            double lam2E = (double)n - lmaxCmpl; if (lam2E > lam2_comp_best) lam2_comp_best = lam2E; free(Shc);
            Shc=NULL; tShc=0; shifts_bip_chebyshev(n/2, d, 0, &Shc, &tShc);
            extrema_bipcirc(n, Shc, tShc, &lminCmpl, &lmaxCmpl);
            double lam2F = (double)n - lmaxCmpl; if (lam2F > lam2_comp_best) lam2_comp_best = lam2F; free(Shc);
            Shc=NULL; tShc=0; shifts_bip_golden(n/2, d, 0, &Shc, &tShc);
            extrema_bipcirc(n, Shc, tShc, &lminCmpl, &lmaxCmpl);
            double lam2G = (double)n - lmaxCmpl; if (lam2G > lam2_comp_best) lam2_comp_best = lam2G; free(Shc);
            Shc=NULL; tShc=0; shifts_bip_knn_minus_t(n/2, d, &Shc, &tShc);
            extrema_bipcirc(n, Shc, tShc, &lminCmpl, &lmaxCmpl);
            double lam2H = (double)n - lmaxCmpl; if (lam2H > lam2_comp_best) lam2_comp_best = lam2H; free(Shc);
            Shc=NULL; tShc=0; shifts_bip_remove_even(n/2, d, &Shc, &tShc);
            extrema_bipcirc(n, Shc, tShc, &lminCmpl, &lmaxCmpl);
            double lam2I = (double)n - lmaxCmpl; if (lam2I > lam2_comp_best) lam2_comp_best = lam2I; free(Shc);
            Shc=NULL; tShc=0; shifts_bip_remove_golden(n/2, d, &Shc, &tShc);
            extrema_bipcirc(n, Shc, tShc, &lminCmpl, &lmaxCmpl);
            double lam2J = (double)n - lmaxCmpl; if (lam2J > lam2_comp_best) lam2_comp_best = lam2J; free(Shc);

            // bilayer complement variants (exact λmax)
            {
                int N=n/2;
                int base = d/2; int deltas2[] = {-5,-4,-3,-2,-1,0,1,2,3,4,5}; int nd2=(int)(sizeof(deltas2)/sizeof(deltas2[0]));
                for(int ti=0; ti<nd2; ++ti){
                    int t_cross = base + deltas2[ti]; if(t_cross<0) t_cross=0; if(t_cross>N) t_cross=N;
                    int kk_side = d - t_cross; if(kk_side<0) continue; if(kk_side > N-1){ t_cross = d - (N-1); if(t_cross<0) continue; if(t_cross>N) t_cross=N; kk_side = d - t_cross; }
                    for(int sf=0; sf<4; ++sf){
                        int hs2=0, ts2=0; int* S2=NULL;
                        if(sf==0) steps_circ_even_midpoints(N, kk_side, &hs2, &ts2, &S2);
                        else if(sf==1) steps_circ_chebyshev(N, kk_side, &hs2, &ts2, &S2);
                        else if(sf==2) steps_circ_golden(N, kk_side, &hs2, &ts2, &S2);
                        else steps_circ_dyadic(N, kk_side, &hs2, &ts2, &S2);
                        for(int inc0=0; inc0<=1; ++inc0){
                            for(int shf=0; shf<3; ++shf){
                                int *Sh=NULL; int tSh=0;
                                if(shf==0) shifts_bip_even_spacing(N, t_cross, inc0, &Sh, &tSh);
                                else if(shf==1) shifts_bip_chebyshev(N, t_cross, inc0, &Sh, &tSh);
                                else shifts_bip_golden(N, t_cross, inc0, &Sh, &tSh);
                                double lmaxH = lambda_max_bilayer_circ_exact(n, S2, ts2, hs2, Sh, tSh);
                                double lam2G = (double)n - lmaxH; if(lam2G > lam2_comp_best) lam2_comp_best = lam2G;
                                free(Sh);
                            }
                        }
                        free(S2);
                    }
                }
            }
            if (lam2_comp_best > best) best = lam2_comp_best;
        }

        // k=3 special families (even n only): prism and Möbius ladder
        if (k == 3) {
            int N = n/2;
            double lam_prism = 2.0 - 2.0 * cos(2.0 * M_PI / (double)N);
            double lam_mobius = 4.0 - 2.0 * cos(2.0 * M_PI / (double)n);
            if (lam_prism > best) best = lam_prism;
            if (lam_mobius > best) best = lam_mobius;
        }

        *lambda2_out = (best < 0.0 && best > -1e-12) ? 0.0 : best;
        return;
#endif
    }

    if (k > n/2) {
        // Dense: use complement reduction to degree d=n-1-k with circulant (non-bipartite) constructions.
        int d = (n - 1) - k;
        double lam2_best = -INFINITY;

        // Augment strong base K_{N,N} with intra-layer circulants (monotone add)
        {
            int N = n/2;
            int k_side = k - N;
            if (k_side > 0) {
                int *Sh_full=(int*)malloc(sizeof(int)*N);
                for(int i=0;i<N;i++) Sh_full[i]=i; // full cross shifts = K_{N,N}
                int hs=0, ts=0; int* S=NULL; double l2;
                steps_circ_even_midpoints(N, k_side, &hs, &ts, &S);
                l2 = lambda2_bilayer_circ_exact(n, S, ts, hs, Sh_full, N);
                if(l2 > lam2_best) lam2_best = l2; free(S);
                steps_circ_chebyshev(N, k_side, &hs, &ts, &S);
                l2 = lambda2_bilayer_circ_exact(n, S, ts, hs, Sh_full, N);
                if(l2 > lam2_best) lam2_best = l2; free(S);
                steps_circ_golden(N, k_side, &hs, &ts, &S);
                l2 = lambda2_bilayer_circ_exact(n, S, ts, hs, Sh_full, N);
                if(l2 > lam2_best) lam2_best = l2; free(S);
                steps_circ_dyadic(N, k_side, &hs, &ts, &S);
                l2 = lambda2_bilayer_circ_exact(n, S, ts, hs, Sh_full, N);
                if(l2 > lam2_best) lam2_best = l2; free(S);
                free(Sh_full);
            }
        }

        // complement via circulant even-midpoints
        int h=0,t=0; int *steps=NULL; steps_circ_even_midpoints(n, d, &h, &t, &steps);
        double cmin1,cmax1; extrema_circulant(n, steps, t, h, &cmin1, &cmax1);
        double lam2_1 = (double)n - cmax1; if (lam2_1 > lam2_best) lam2_best = lam2_1; free(steps);

        // complement via circulant chebyshev
        h=0; t=0; steps=NULL; steps_circ_chebyshev(n, d, &h, &t, &steps);
        double cmin2,cmax2; extrema_circulant(n, steps, t, h, &cmin2, &cmax2);
        double lam2_2 = (double)n - cmax2; if (lam2_2 > lam2_best) lam2_best = lam2_2; free(steps);

        // complement via circulant golden
        h=0; t=0; steps=NULL; steps_circ_golden(n, d, &h, &t, &steps);
        double cmin3,cmax3; extrema_circulant(n, steps, t, h, &cmin3, &cmax3);
        double lam2_3 = (double)n - cmax3; if (lam2_3 > lam2_best) lam2_best = lam2_3; free(steps);

        // complement via circulant dyadic
        h=0; t=0; steps=NULL; steps_circ_dyadic(n, d, &h, &t, &steps);
        double cmin4,cmax4; extrema_circulant(n, steps, t, h, &cmin4, &cmax4);
        double lam2_4 = (double)n - cmax4; if (lam2_4 > lam2_best) lam2_best = lam2_4; free(steps);

        // complement via circulant anchors {1/4, 1/3}
        {
            double anch[2] = {0.25, 1.0/3.0};
            h=0; t=0; steps=NULL; steps_circ_anchors_evenmid(n, d, anch, 2, &h, &t, &steps);
            double cm, cM; extrema_circulant(n, steps, t, h, &cm, &cM);
            double lam2A = (double)n - cM; if (lam2A > lam2_best) lam2_best = lam2A; free(steps);
        }
        // complement via circulant anchors {1/5, 2/5}
        {
            double anch[2] = {0.2, 0.4};
            h=0; t=0; steps=NULL; steps_circ_anchors_evenmid(n, d, anch, 2, &h, &t, &steps);
            double cm, cM; extrema_circulant(n, steps, t, h, &cm, &cM);
            double lam2A = (double)n - cM; if (lam2A > lam2_best) lam2_best = lam2A; free(steps);
        }
        // complement via circulant anchors {1/6, 1/3}
        {
            double anch[2] = {1.0/6.0, 1.0/3.0};
            h=0; t=0; steps=NULL; steps_circ_anchors_evenmid(n, d, anch, 2, &h, &t, &steps);
            double cm, cM; extrema_circulant(n, steps, t, h, &cm, &cM);
            double lam2A = (double)n - cM; if (lam2A > lam2_best) lam2_best = lam2A; free(steps);
        }
        // complement via circulant anchors {1/8, 3/8}
        {
            double anch[2] = {1.0/8.0, 3.0/8.0};
            h=0; t=0; steps=NULL; steps_circ_anchors_evenmid(n, d, anch, 2, &h, &t, &steps);
            double cm, cM; extrema_circulant(n, steps, t, h, &cm, &cM);
            double lam2A = (double)n - cM; if (lam2A > lam2_best) lam2_best = lam2A; free(steps);
        }
        // complement via circulant anchors {1/9, 2/9, 4/9}
        {
            double anch3[3] = {1.0/9.0, 2.0/9.0, 4.0/9.0};
            h=0; t=0; steps=NULL; steps_circ_anchors_evenmid(n, d, anch3, 3, &h, &t, &steps);
            double cm, cM; extrema_circulant(n, steps, t, h, &cm, &cM);
            double lam2A = (double)n - cM; if (lam2A > lam2_best) lam2_best = lam2A; free(steps);
        }
        // complement via circulant anchors {1/10, 3/10}
        {
            double anch[2] = {1.0/10.0, 3.0/10.0};
            h=0; t=0; steps=NULL; steps_circ_anchors_evenmid(n, d, anch, 2, &h, &t, &steps);
            double cm, cM; extrema_circulant(n, steps, t, h, &cm, &cM);
            double lam2A = (double)n - cM; if (lam2A > lam2_best) lam2_best = lam2A; free(steps);
        }

        // complement via circulant prefix steps
        h=0; t=0; steps=NULL; steps_circ_prefix(n, d, &h, &t, &steps);
        double cmP,cMP; extrema_circulant(n, steps, t, h, &cmP, &cMP);
        double lam2P = (double)n - cMP; if (lam2P > lam2_best) lam2_best = lam2P; free(steps);
        // complement via anchors {1/7,2/7,3/7}
        {
            double anch3[3] = {1.0/7.0, 2.0/7.0, 3.0/7.0};
            h=0; t=0; steps=NULL; steps_circ_anchors_evenmid(n, d, anch3, 3, &h, &t, &steps);
            double cm, cM; extrema_circulant(n, steps, t, h, &cm, &cM);
            double lam2A = (double)n - cM; if (lam2A > lam2_best) lam2_best = lam2A; free(steps);
        }

        // Small-d complement special cases (deterministic)
        if (d == 3) {
            if (n % 2 == 0) {
                // H: 3-regular circulant with antipode + one step r
                int Rmax = n/2 - 1;
                int rcand_raw[] = {1, n/8, n/6, n/5, n/4, n/3};
                int cand[8]; int cc=0;
                for(int i=0;i<(int)(sizeof(rcand_raw)/sizeof(rcand_raw[0]));++i){
                    int r = rcand_raw[i]; if(r<1) r=1; if(r>Rmax) r=Rmax;
                    int seen=0; for(int j=0;j<cc;j++){ if(cand[j]==r){seen=1;break;} }
                    if(!seen) cand[cc++]=r;
                }
                for(int i=0;i<cc;i++){
                    int r=cand[i]; int S1[1]={r}; int hh=1, tt=1; double cm,cM;
                    extrema_circulant(n, S1, tt, hh, &cm, &cM);
                    double lam2G = (double)n - cM; if(lam2G > lam2_best) lam2_best = lam2G;
                }
            }
        } else if (d == 4) {
            // H: 4-regular circulant candidates {1,2} and {1,3}
            int S2a[2]={1,2}; int hh=0, tt=2; double cm,cM;
            extrema_circulant(n, S2a, tt, hh, &cm, &cM);
            double lam2G = (double)n - cM; if(lam2G > lam2_best) lam2_best = lam2G;
            int S2b[2]={1,3}; extrema_circulant(n, S2b, tt, hh, &cm, &cM);
            lam2G = (double)n - cM; if(lam2G > lam2_best) lam2_best = lam2G;
        }

        // complement via bilayer-circulant seeds (exact)
        {
            int N=n/2;
            int t_choices[3] = { d/2, (d/2>0? d/2 - 1 : 0), (d/2 < N ? d/2 + 1 : N) };
            for(int ti=0; ti<3; ++ti){
                int t_cross = t_choices[ti]; if(t_cross<0) t_cross=0; if(t_cross>N) t_cross=N;
                int kk_side = d - t_cross; if(kk_side<0) continue; if(kk_side > N-1){ t_cross = d - (N-1); if(t_cross<0) continue; if(t_cross>N) t_cross=N; kk_side = d - t_cross; }
                for(int sf=0; sf<4; ++sf){
                    int hs=0, ts=0; int* S=NULL;
                    if(sf==0) steps_circ_even_midpoints(N, kk_side, &hs, &ts, &S);
                    else if(sf==1) steps_circ_chebyshev(N, kk_side, &hs, &ts, &S);
                    else if(sf==2) steps_circ_golden(N, kk_side, &hs, &ts, &S);
                    else steps_circ_dyadic(N, kk_side, &hs, &ts, &S);
                    for(int inc0=0; inc0<=1; ++inc0){
                        for(int shf=0; shf<3; ++shf){
                            int *Sh=NULL; int tSh=0;
                            if(shf==0) shifts_bip_even_spacing(N, t_cross, inc0, &Sh, &tSh);
                            else if(shf==1) shifts_bip_chebyshev(N, t_cross, inc0, &Sh, &tSh);
                            else shifts_bip_golden(N, t_cross, inc0, &Sh, &tSh);
                            double lmaxH = lambda_max_bilayer_circ_exact(n, S, ts, hs, Sh, tSh);
                            double lam2G = (double)n - lmaxH; if(lam2G > lam2_best) lam2_best = lam2G;
                            free(Sh);
                        }
                    }
                    free(S);
                }
            }
        }

        // trivial complement bound: λ2(G) ≥ n - 2d
        {
            double trivial_lb = (double)n - 2.0 * (double)d;
            if (trivial_lb > lam2_best) lam2_best = trivial_lb;
        }
        if (lam2_best < 0.0 && lam2_best > -1e-12) lam2_best = 0.0;
        *lambda2_out = lam2_best;
        return;
    }

    // Odd n or even n with k>n/2 handled; remaining: odd n (k even) or even n small k but prefer circulant
    double best = -INFINITY;
    // Paley augmentation first if applicable (odd prime n ≡ 1 mod 4)
    if((n%2)==1 && (n%4)==1 && is_prime_int_local(n)){
        int kp=(n-1)/2; if(k>=kp){ double l2p = paley_augment_lambda2(n,k); if(l2p>best) best=l2p; }
    }
    int h=0,t=0; int *steps=NULL;
    // Analytic seeds targeting odd-n weakness (AC2 removed)
    // Fourier-minimax single-set circulant
    steps_circ_fourier_minimax(n, k, &h, &t, &steps);
    double lmin1,lmax1; extrema_circulant(n, steps, t, h, &lmin1, &lmax1); if (lmin1 > best) best = lmin1; free(steps);
    // Harmonic-Quenching Cayley
    steps_circ_hqc(n, k, &h, &t, &steps);
    double lminH,lmaxH; extrema_circulant(n, steps, t, h, &lminH, &lmaxH); if (lminH > best) best = lminH; free(steps);
    // Long-jump circulant
    steps_circ_long_jumps(n, k, &h, &t, &steps);
    double lminL,lmaxL; extrema_circulant(n, steps, t, h, &lminL, &lmaxL); if (lminL > best) best = lminL; free(steps);
    // Analytic seeds targeting odd-n weakness
    steps_circ_lowmode_antipodes(n, k, &h, &t, &steps);
    double lminA,lmaxA; extrema_circulant(n, steps, t, h, &lminA, &lmaxA); if (lminA > best) best = lminA; free(steps);
    steps_circ_diameter_bias(n, k, &h, &t, &steps);
    double lmind,lmaxd; extrema_circulant(n, steps, t, h, &lmind, &lmaxd); if (lmind > best) best = lmind; free(steps);
    // One-shot bottleneck-targeted variants of baseline seeds
    steps_circ_bottleneck_targeted(n, k, &h, &t, &steps, steps_circ_even_midpoints);
    double lbt1,ubt1; extrema_circulant(n, steps, t, h, &lbt1, &ubt1); if (lbt1 > best) best = lbt1; free(steps);
    steps_circ_bottleneck_targeted(n, k, &h, &t, &steps, steps_circ_chebyshev);
    double lbt2,ubt2; extrema_circulant(n, steps, t, h, &lbt2, &ubt2); if (lbt2 > best) best = lbt2; free(steps);
    steps_circ_bottleneck_targeted(n, k, &h, &t, &steps, steps_circ_golden);
    double lbt3,ubt3; extrema_circulant(n, steps, t, h, &lbt3, &ubt3); if (lbt3 > best) best = lbt3; free(steps);
    steps_circ_bottleneck_targeted(n, k, &h, &t, &steps, steps_circ_dyadic);
    double lbt4,ubt4; extrema_circulant(n, steps, t, h, &lbt4, &ubt4); if (lbt4 > best) best = lbt4; free(steps);
    steps_circ_chebyshev(n, k, &h, &t, &steps);
    double lmin2,lmax2; extrema_circulant(n, steps, t, h, &lmin2, &lmax2); if (lmin2 > best) best = lmin2; free(steps);
    steps_circ_golden(n, k, &h, &t, &steps);
    double lmin3,lmax3; extrema_circulant(n, steps, t, h, &lmin3, &lmax3); if (lmin3 > best) best = lmin3; free(steps);
    steps_circ_dyadic(n, k, &h, &t, &steps);
    double lmin4,lmax4; extrema_circulant(n, steps, t, h, &lmin4, &lmax4); if (lmin4 > best) best = lmin4; free(steps);
    // prefix steps as an additional seed
    steps_circ_prefix(n, k, &h, &t, &steps);
    double lpfx,upfx; extrema_circulant(n, steps, t, h, &lpfx, &upfx); if (lpfx > best) best = lpfx; free(steps);
    // anchors {1/4,1/3}
    { double anch[2]={0.25, 1.0/3.0}; h=0; t=0; steps=NULL; steps_circ_anchors_evenmid(n, k, anch, 2, &h, &t, &steps); double la,ua; extrema_circulant(n, steps, t, h, &la, &ua); if (la>best) best=la; free(steps);}    
    // anchors {1/5,2/5}
    { double anch[2]={0.2, 0.4}; h=0; t=0; steps=NULL; steps_circ_anchors_evenmid(n, k, anch, 2, &h, &t, &steps); double la,ua; extrema_circulant(n, steps, t, h, &la, &ua); if (la>best) best=la; free(steps);}    
    // anchors {1/6,1/3}
    { double anch[2]={1.0/6.0, 1.0/3.0}; h=0; t=0; steps=NULL; steps_circ_anchors_evenmid(n, k, anch, 2, &h, &t, &steps); double la,ua; extrema_circulant(n, steps, t, h, &la, &ua); if (la>best) best=la; free(steps);}    
    // anchors {1/7,2/7,3/7}
    { double anch3[3]={1.0/7.0, 2.0/7.0, 3.0/7.0}; h=0; t=0; steps=NULL; steps_circ_anchors_evenmid(n, k, anch3, 3, &h, &t, &steps); double la,ua; extrema_circulant(n, steps, t, h, &la, &ua); if (la>best) best=la; free(steps);}    
    // anchors including diameter-ish {1/2,1/3}
    { double anch2[2]={0.5, 1.0/3.0}; h=0; t=0; steps=NULL; steps_circ_anchors_evenmid(n, k, anch2, 2, &h, &t, &steps); double la,ua; extrema_circulant(n, steps, t, h, &la, &ua); if (la>best) best=la; free(steps);}    
    // anchors including diameter-ish {1/2,1/4}
    { double anch2b[2]={0.5, 0.25}; h=0; t=0; steps=NULL; steps_circ_anchors_evenmid(n, k, anch2b, 2, &h, &t, &steps); double la,ua; extrema_circulant(n, steps, t, h, &la, &ua); if (la>best) best=la; free(steps);}    
    // anchors including diameter-ish {1/2,2/5}
    { double anch2c[2]={0.5, 0.4}; h=0; t=0; steps=NULL; steps_circ_anchors_evenmid(n, k, anch2c, 2, &h, &t, &steps); double la,ua; extrema_circulant(n, steps, t, h, &la, &ua); if (la>best) best=la; free(steps);}    
    // anchors {1/8,3/8}
    { double anch[2]={1.0/8.0, 3.0/8.0}; h=0; t=0; steps=NULL; steps_circ_anchors_evenmid(n, k, anch, 2, &h, &t, &steps); double la,ua; extrema_circulant(n, steps, t, h, &la, &ua); if (la>best) best=la; free(steps);}    
    // anchors {1/9,2/9,4/9}
    { double anch3b[3]={1.0/9.0, 2.0/9.0, 4.0/9.0}; h=0; t=0; steps=NULL; steps_circ_anchors_evenmid(n, k, anch3b, 3, &h, &t, &steps); double la,ua; extrema_circulant(n, steps, t, h, &la, &ua); if (la>best) best=la; free(steps);}    
    // anchors {1/10,3/10}
    { double anchb[2]={1.0/10.0, 3.0/10.0}; h=0; t=0; steps=NULL; steps_circ_anchors_evenmid(n, k, anchb, 2, &h, &t, &steps); double la,ua; extrema_circulant(n, steps, t, h, &la, &ua); if (la>best) best=la; free(steps);}    

    // Also consider complement designs for d = n-1-k
    int d = (n - 1) - k;
    if (d >= 1) {
        double lam2_comp_best = -INFINITY;
        // AC2 complement removed
        h=0; t=0; steps=NULL; steps_circ_even_midpoints(n, d, &h, &t, &steps);
        double cminA,cmaxA; extrema_circulant(n, steps, t, h, &cminA, &cmaxA);
        double lam2A = (double)n - cmaxA; if (lam2A > lam2_comp_best) lam2_comp_best = lam2A; free(steps);
        h=0; t=0; steps=NULL; steps_circ_chebyshev(n, d, &h, &t, &steps);
        double cminB,cmaxB; extrema_circulant(n, steps, t, h, &cminB, &cmaxB);
        double lam2B = (double)n - cmaxB; if (lam2B > lam2_comp_best) lam2_comp_best = lam2B; free(steps);
        h=0; t=0; steps=NULL; steps_circ_golden(n, d, &h, &t, &steps);
        double cminC,cmaxC; extrema_circulant(n, steps, t, h, &cminC, &cmaxC);
        double lam2C = (double)n - cmaxC; if (lam2C > lam2_comp_best) lam2_comp_best = lam2C; free(steps);
        h=0; t=0; steps=NULL; steps_circ_dyadic(n, d, &h, &t, &steps);
        double cminD,cmaxD; extrema_circulant(n, steps, t, h, &cminD, &cmaxD);
        double lam2D = (double)n - cmaxD; if (lam2D > lam2_comp_best) lam2_comp_best = lam2D; free(steps);
        // small-d specials in sparse complement path
        if (d == 3) {
            if (n % 2 == 0) {
                int Rmax = n/2 - 1;
                int rcand_raw[] = {1, n/8, n/6, n/5, n/4, n/3};
                int cand[8]; int cc=0;
                for(int i=0;i<(int)(sizeof(rcand_raw)/sizeof(rcand_raw[0]));++i){
                    int r = rcand_raw[i]; if(r<1) r=1; if(r>Rmax) r=Rmax;
                    int seen=0; for(int j=0;j<cc;j++){ if(cand[j]==r){seen=1;break;} }
                    if(!seen) cand[cc++]=r;
                }
                for(int i=0;i<cc;i++){
                    int r=cand[i]; int S1[1]={r}; int hh=1, tt=1; double cm,cM;
                    extrema_circulant(n, S1, tt, hh, &cm, &cM);
                    double lam2G = (double)n - cM; if(lam2G > lam2_comp_best) lam2_comp_best = lam2G;
                }
            }
        } else if (d == 4) {
            int S2a[2]={1,2}; int hh=0, tt=2; double cm2,cM2;
            extrema_circulant(n, S2a, tt, hh, &cm2, &cM2);
            double lam2G = (double)n - cM2; if(lam2G > lam2_comp_best) lam2_comp_best = lam2G;
            int S2b[2]={1,3}; extrema_circulant(n, S2b, tt, hh, &cm2, &cM2);
            lam2G = (double)n - cM2; if(lam2G > lam2_comp_best) lam2_comp_best = lam2G;
        }
        if (lam2_comp_best > best) best = lam2_comp_best;
    }
    // Note: conjugate-circulant union (numeric) removed to keep odd-n fast.
    // optional two-lift deterministic expansion to improve scaling
    double l2lift=best;
    double l2tmp=0.0; if(try_two_lift_lambda2(n,k,&l2tmp)){ if(l2tmp > l2lift) l2lift = l2tmp; }
    *lambda2_out = (l2lift < 0.0 && l2lift > -1e-12) ? 0.0 : l2lift;
}

// Backward-compat shim: compute using new analytic cases and also provide max eigen if needed via a second pass.
static void build_steps_and_extrema(int n, int k, double *lambda_min_out, double *lambda_max_out) {
    // For external callers, we only need lambda2 (min nonzero). Provide lambda_max_out as 0.
    double l2 = 0.0; best_lambda2_analytic(n, k, &l2);
    *lambda_min_out = l2; *lambda_max_out = 0.0;
}

static void usage(const char* prog) {
    fprintf(stderr, "Usage: %s -n <num_vertices> -d <regular_degree> [--quiet]\n", prog);
}

int main(int argc, char **argv) {
    int n = -1, k = -1;
    int quiet = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i+1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-d") == 0 && i+1 < argc) {
            k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--quiet") == 0) {
            quiet = 1;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (n < 3 || k < 0 || k >= n) {
        fprintf(stderr, "Invalid input: require n>=3 and 0<=k<=n-1.\n");
        return 2;
    }
    if (((long long)n * (long long)k) % 2 != 0) {
        fprintf(stderr, "Invalid input: n*k must be even for a simple k-regular graph.\n");
        return 2;
    }
    if (k == 0) { printf("0.000000000000\n"); return 0; }
    if (k == n-1) { // complete graph: Laplacian eigenvalues are 0 and n (mult n-1)
        printf("%.12f\n", (double)n);
        return 0;
    }

    double lambda2, _;
    build_steps_and_extrema(n, k, &lambda2, &_);
    // print λ2 only
    if (fabs(lambda2) < 1e-12) lambda2 = 0.0;
    if (quiet) {
        printf("lambda2= %.12f\n", lambda2);
    } else {
        printf("%.12f\n", lambda2);
    }
    return 0;
}
