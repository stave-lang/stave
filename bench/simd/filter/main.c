/* The C twin of src/main.stv: same steps, same output. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 32768u
#define REPS 20000u

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

static size_t filter(const float *xs, size_t n, float k, float *out) {
    size_t m = 0;
    for (size_t i = 0; i < n; i++) {
        out[m] = xs[i];
        m += xs[i] > k;
    }
    return m;
}

int main(void) {
    uint64_t s = 4;
    float *xs = malloc(N * sizeof *xs);
    float *out = malloc(N * sizeof *out);
    if (xs == NULL || out == NULL) {
        return 1;
    }
    for (size_t i = 0; i < N; i++) {
        xs[i] = unit(&s);
        out[i] = 0.0f;
    }
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t r = 0; r < REPS; r++) {
        float k = (float)(uint32_t)(r % 64) * (1.0f / 64.0f) - 0.5f;
        size_t n = filter(xs, N, k, out);
        h = mix(h, n);
        if (n > 0) {
            h = mix(h, bits(out[n - 1]));
        }
    }
    printf("%llu\n", (unsigned long long)h);
    free(xs);
    free(out);
    return 0;
}
