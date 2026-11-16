/* The C twin of src/main.stv: same steps, same output. Eight lane accumulators, reduced as the
   Stave lane tree: lane k + lane k+4, then k + k+2, then 0 + 1. */
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

static float dot(const float *a, const float *b, size_t n) {
    float acc[8] = {0};
    for (size_t i = 0; i + 8 <= n; i += 8) {
        for (int k = 0; k < 8; k++) {
            acc[k] = acc[k] + a[i + k] * b[i + k];
        }
    }
    for (int k = 0; k < 4; k++) {
        acc[k] = acc[k] + acc[k + 4];
    }
    acc[0] = acc[0] + acc[2];
    acc[1] = acc[1] + acc[3];
    return acc[0] + acc[1];
}

int main(void) {
    uint64_t s = 2;
    float *a = malloc(N * sizeof *a);
    float *b = malloc(N * sizeof *b);
    if (a == NULL || b == NULL) {
        return 1;
    }
    for (size_t i = 0; i < N; i++) {
        a[i] = unit(&s);
        b[i] = unit(&s);
    }
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t r = 0; r < REPS; r++) {
        a[r % N] = a[r % N] + 0.25f;
        h = mix(h, bits(dot(a, b, N)));
    }
    printf("%llu\n", (unsigned long long)h);
    free(a);
    free(b);
    return 0;
}
