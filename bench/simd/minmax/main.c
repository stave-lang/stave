/* The C twin of src/main.stv: same steps, same output. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 32768u
#define REPS 40000u

static uint64_t next(uint64_t *s) {
    *s = *s * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    return *s;
}

static float unit(uint64_t *s) {
    uint32_t x = (uint32_t)(next(s) >> 40);
    return (float)x * (1.0f / 16777216.0f) - 0.5f;
}

static uint64_t mix(uint64_t h, uint64_t x) {
    return (h ^ x) * UINT64_C(1099511628211);
}

static uint32_t bits(float x) {
    uint32_t b;
    memcpy(&b, &x, sizeof b);
    return b;
}

static void minmax(const float *xs, size_t n, float *lo, float *hi) {
    float l = xs[0];
    float g = xs[0];
    for (size_t i = 0; i < n; i++) {
        l = xs[i] < l ? xs[i] : l;
        g = xs[i] > g ? xs[i] : g;
    }
    *lo = l;
    *hi = g;
}

int main(void) {
    uint64_t s = 3;
    float *xs = malloc(N * sizeof *xs);
    if (xs == NULL) {
        return 1;
    }
    for (size_t i = 0; i < N; i++) {
        xs[i] = unit(&s);
    }
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t r = 0; r < REPS; r++) {
        float lo, hi;
        xs[r % N] = xs[r % N] + 0.25f;
        minmax(xs, N, &lo, &hi);
        h = mix(mix(h, bits(lo)), bits(hi));
    }
    printf("%llu\n", (unsigned long long)h);
    free(xs);
    return 0;
}
