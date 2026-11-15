/* The C twin of src/main.stv: same operations in the same order, same output. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1048576u
#define SLOTS 2097152u
#define MASK 2097151ull
#define ROUNDS 4u
#define PASSES 3u

static uint64_t mix64(uint64_t x0) {
    uint64_t x = x0;
    x = x ^ (x >> 30);
    x = x * 0xBF58476D1CE4E5B9ull;
    x = x ^ (x >> 27);
    x = x * 0x94D049BB133111EBull;
    x = x ^ (x >> 31);
    return x;
}

static uint64_t *zeros(size_t n) {
    uint64_t *v = calloc(n, sizeof(uint64_t));
    if (v == NULL) {
        fputs("out of memory\n", stderr);
        exit(1);
    }
    return v;
}

static void fill(uint64_t *xs, size_t n, uint64_t seed) {
    uint64_t s = seed;
    for (size_t i = 0; i < n; i++) {
        s = s ^ (s << 13);
        s = s ^ (s >> 7);
        s = s ^ (s << 17);
        xs[i] = s;
    }
}

/* n is a power of two. */
static void scramble(uint64_t *xs, size_t n, uint32_t rounds) {
    for (uint32_t r = 0; r < rounds; r++) {
        for (size_t i = 0; i < n; i++) {
            size_t j = (i * 7 + 1) & (n - 1);
            xs[i] = mix64(xs[i] ^ xs[j]) + (uint64_t)r;
        }
    }
}

/* Inserts the nonzero key k; returns 1 if it was not in the table yet. */
static uint64_t insert(uint64_t *t, uint64_t k) {
    uint64_t i = mix64(k) & MASK;
    while (t[i] != 0) {
        if (t[i] == k) {
            return 0;
        }
        i = (i + 1) & MASK;
    }
    t[i] = k;
    return 1;
}

static uint64_t contains(const uint64_t *t, uint64_t k) {
    uint64_t i = mix64(k) & MASK;
    while (t[i] != 0) {
        if (t[i] == k) {
            return 1;
        }
        i = (i + 1) & MASK;
    }
    return 0;
}

static uint64_t build(uint64_t *t, size_t slots, const uint64_t *keys, size_t n) {
    for (size_t i = 0; i < slots; i++) {
        t[i] = 0;
    }
    uint64_t fresh = 0;
    for (size_t i = 0; i < n; i++) {
        fresh = fresh + insert(t, keys[i] | 1);
    }
    return fresh;
}

/* Even probes are inserted keys, odd ones come from another stream. */
static uint64_t probe(const uint64_t *t, const uint64_t *keys, const uint64_t *other, size_t n) {
    uint64_t hits = 0;
    for (size_t i = 0; i < n; i++) {
        uint64_t k = (i & 1) == 0 ? (keys[i] | 1) : (other[i] | 1);
        hits = hits + contains(t, k);
    }
    return hits;
}

static uint64_t digest(const uint64_t *t, size_t n) {
    uint64_t c = 14695981039346656037ull;
    for (size_t i = 0; i < n; i++) {
        c = (c ^ t[i]) * 1099511628211ull;
    }
    return c;
}

int main(void) {
    uint64_t *keys = zeros(N);
    uint64_t *other = zeros(N);
    uint64_t *table = zeros(SLOTS);
    uint64_t sum = 0;
    for (uint32_t p = 0; p < PASSES; p++) {
        fill(keys, N, 88172645463325252ull + (uint64_t)p);
        fill(other, N, 2463534242ull + (uint64_t)p);
        scramble(keys, N, ROUNDS);
        uint64_t fresh = build(table, SLOTS, keys, N);
        uint64_t hits = probe(table, keys, other, N);
        uint64_t c = digest(table, SLOTS);
        printf("pass %u: inserted %llu hits %llu digest %llu\n", (unsigned)p, (unsigned long long)fresh,
               (unsigned long long)hits, (unsigned long long)c);
        sum = sum ^ c;
    }
    printf("checksum %llu\n", (unsigned long long)sum);
    free(keys);
    free(other);
    free(table);
    return 0;
}
