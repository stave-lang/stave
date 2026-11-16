/* The C twin of src/main.stv: same steps, same output. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define N 65536u
#define REPS 2000u

static uint64_t next(uint64_t *s) {
    *s = *s * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    return *s;
}

static uint64_t mix(uint64_t h, uint64_t x) {
    return (h ^ x) * UINT64_C(1099511628211);
}

static void histogram(const unsigned char *xs, size_t n, uint32_t *bins) {
    for (size_t i = 0; i < n; i++) {
        bins[xs[i]] += 1;
    }
}

int main(void) {
    uint64_t s = 7;
    unsigned char *xs = malloc(N);
    uint32_t bins[256] = {0};
    if (xs == NULL) {
        return 1;
    }
    for (size_t i = 0; i < N; i++) {
        xs[i] = (unsigned char)(next(&s) >> 56);
    }
    for (size_t r = 0; r < REPS; r++) {
        xs[r % N] = (unsigned char)(xs[r % N] + 1);
        histogram(xs, N, bins);
    }
    uint64_t h = UINT64_C(14695981039346656037);
    for (int b = 0; b < 256; b++) {
        h = mix(h, bins[b]);
    }
    printf("%llu\n", (unsigned long long)h);
    free(xs);
    return 0;
}
