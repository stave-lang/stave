/* The C twin of src/main.stv: same operations in the same order, same output. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define N 16777216u
#define REPS 4u

static void sieve(unsigned char *f, size_t n) {
    for (size_t i = 0; i < n; i++) {
        f[i] = 0;
    }
    f[0] = 1;
    f[1] = 1;
    size_t i = 2;
    while (i * i < n) {
        if (f[i] == 0) {
            size_t j = i * i;
            while (j < n) {
                f[j] = 1;
                j = j + i;
            }
        }
        i = i + 1;
    }
}

static uint64_t count(const unsigned char *f, size_t n) {
    uint64_t c = 0;
    for (size_t i = 0; i < n; i++) {
        if (f[i] == 0) {
            c = c + 1;
        }
    }
    return c;
}

static uint64_t sum(const unsigned char *f, size_t n) {
    uint64_t s = 0;
    for (size_t i = 0; i < n; i++) {
        if (f[i] == 0) {
            s = s + (uint64_t)i;
        }
    }
    return s;
}

static uint64_t gaps(const unsigned char *f, size_t n) {
    uint64_t h = 14695981039346656037ull;
    uint64_t last = 2;
    for (size_t i = 3; i < n; i++) {
        if (f[i] == 0) {
            uint64_t p = (uint64_t)i;
            h = (h ^ (p - last)) * 1099511628211ull;
            last = p;
        }
    }
    return h;
}

int main(void) {
    unsigned char *flags = malloc(N);
    if (flags == NULL) {
        fputs("out of memory\n", stderr);
        return 1;
    }
    for (size_t i = 0; i < N; i++) {
        flags[i] = 1;
    }
    uint64_t total = 0;
    for (uint32_t r = 0; r < REPS; r++) {
        sieve(flags, N);
        uint64_t c = count(flags, N);
        uint64_t s = sum(flags, N);
        uint64_t g = gaps(flags, N);
        total = total ^ g ^ s ^ c;
        printf("rep %u: primes %llu sum %llu gaps %llu\n", (unsigned)r, (unsigned long long)c,
               (unsigned long long)s, (unsigned long long)g);
    }
    printf("checksum %llu\n", (unsigned long long)total);
    free(flags);
    return 0;
}
