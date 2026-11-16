/* The C twin of src/main.stv: same steps, same output. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define N 65536u
#define REPS 20000u

static uint64_t next(uint64_t *s) {
    *s = *s * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    return *s;
}

static uint64_t mix(uint64_t h, uint64_t x) {
    return (h ^ x) * UINT64_C(1099511628211);
}

static size_t count_byte(const unsigned char *xs, size_t n, unsigned char b) {
    size_t c = 0;
    for (size_t i = 0; i < n; i++) {
        c += xs[i] == b;
    }
    return c;
}

static size_t find_byte(const unsigned char *xs, size_t n, unsigned char b) {
    for (size_t i = 0; i < n; i++) {
        if (xs[i] == b) {
            return i;
        }
    }
    return n;
}

int main(void) {
    uint64_t s = 5;
    unsigned char *xs = malloc(N);
    if (xs == NULL) {
        return 1;
    }
    for (size_t i = 0; i < N; i++) {
        xs[i] = (unsigned char)(32 + (next(&s) >> 58));
    }
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t r = 0; r < REPS; r++) {
        unsigned char key = (unsigned char)(96 + r % 32);
        size_t at = N - 1 - (r * 97) % 1024;
        unsigned char was = xs[at];
        xs[at] = key;
        h = mix(mix(h, count_byte(xs, N, key)), find_byte(xs, N, key));
        xs[at] = was;
    }
    printf("%llu\n", (unsigned long long)h);
    free(xs);
    return 0;
}
