/* The C twin of src/main.stv: same steps, same output. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define N 32768u
#define REPS 20000u

static uint64_t next(uint64_t *s) {
    *s = *s * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    return *s;
}

static uint64_t mix(uint64_t h, uint64_t x) {
    return (h ^ x) * UINT64_C(1099511628211);
}

static void prefix(const uint32_t *xs, size_t n, uint32_t *ys) {
    uint32_t s = 0;
    for (size_t i = 0; i < n; i++) {
        s += xs[i];
        ys[i] = s;
    }
}

int main(void) {
    uint64_t s = 6;
    uint32_t *xs = malloc(N * sizeof *xs);
    uint32_t *ys = malloc(N * sizeof *ys);
    if (xs == NULL || ys == NULL) {
        return 1;
    }
    for (size_t i = 0; i < N; i++) {
        xs[i] = (uint32_t)(next(&s) >> 32);
        ys[i] = 0;
    }
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t r = 0; r < REPS; r++) {
        xs[r % N] += 1;
        prefix(xs, N, ys);
        h = mix(mix(h, ys[N - 1]), ys[(r * 31) % N]);
    }
    printf("%llu\n", (unsigned long long)h);
    free(xs);
    free(ys);
    return 0;
}
