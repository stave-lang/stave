/* The C twin of src/main.stv: same steps, same output. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 32768u
#define REPS 10000u

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

static void transform(float *xs, size_t n, const float *m) {
    for (size_t p = 0; p + 4 <= n; p += 4) {
        float x0 = xs[p], x1 = xs[p + 1], x2 = xs[p + 2], x3 = xs[p + 3];
        for (int r = 0; r < 4; r++) {
            xs[p + r] = ((m[r] * x0 + m[4 + r] * x1) + m[8 + r] * x2) + m[12 + r] * x3;
        }
    }
}

int main(void) {
    uint64_t s = 8;
    float *xs = malloc(N * sizeof *xs);
    float m[16] = {0};
    if (xs == NULL) {
        return 1;
    }
    for (size_t i = 0; i < N; i++) {
        xs[i] = unit(&s);
    }
    for (int k = 0; k < 4; k++) {
        m[4 * k + k] = 0.5f;
        m[4 * k + (k + 1) % 4] = 0.5f;
    }
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t r = 0; r < REPS; r++) {
        transform(xs, N, m);
        h = mix(h, bits(xs[(r * 31) % N]));
    }
    printf("%llu\n", (unsigned long long)h);
    free(xs);
    return 0;
}
