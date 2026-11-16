/* The C twin of src/main.stv: same data, same output; the validator decodes byte by byte. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define N 65536u
#define REPS 4000u

static uint64_t next(uint64_t *s) {
    *s = *s * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    return *s;
}

static uint64_t mix(uint64_t h, uint64_t x) {
    return (h ^ x) * UINT64_C(1099511628211);
}

static unsigned char *buf;
static size_t len;

static void push(unsigned char b) {
    buf[len++] = b;
}

static void push_char(uint64_t *s) {
    uint64_t r = next(s);
    size_t room = N - len;
    size_t kind = (size_t)(r >> 62);
    uint32_t c = (uint32_t)(r >> 8);
    if (kind == 0 || room < kind + 1) {
        push((unsigned char)(32 + c % 95));
    } else if (kind == 1) {
        uint32_t x = 0x80 + c % 0x780;
        push((unsigned char)(0xc0 | (x >> 6)));
        push((unsigned char)(0x80 | (x & 0x3f)));
    } else if (kind == 2) {
        uint32_t x = 0xe000 + c % 0x1000;
        push((unsigned char)(0xe0 | (x >> 12)));
        push((unsigned char)(0x80 | ((x >> 6) & 0x3f)));
        push((unsigned char)(0x80 | (x & 0x3f)));
    } else {
        uint32_t x = 0x10000 + c % 0x100000;
        push((unsigned char)(0xf0 | (x >> 18)));
        push((unsigned char)(0x80 | ((x >> 12) & 0x3f)));
        push((unsigned char)(0x80 | ((x >> 6) & 0x3f)));
        push((unsigned char)(0x80 | (x & 0x3f)));
    }
}

static int in_range(unsigned char x, unsigned char lo, unsigned char hi) {
    return x >= lo && x <= hi;
}

static int valid_utf8(const unsigned char *xs, size_t n) {
    size_t i = 0;
    while (i < n) {
        unsigned char b = xs[i];
        if (b < 0x80) {
            i++;
            continue;
        }
        size_t need = in_range(b, 0xc2, 0xdf) ? 1 : in_range(b, 0xe0, 0xef) ? 2 : in_range(b, 0xf0, 0xf4) ? 3 : 0;
        if (need == 0 || i + need >= n) {
            return 0;
        }
        unsigned char lo = b == 0xe0 ? 0xa0 : b == 0xf0 ? 0x90 : 0x80;
        unsigned char hi = b == 0xed ? 0x9f : b == 0xf4 ? 0x8f : 0xbf;
        if (!in_range(xs[i + 1], lo, hi)) {
            return 0;
        }
        for (size_t k = 2; k <= need; k++) {
            if (!in_range(xs[i + k], 0x80, 0xbf)) {
                return 0;
            }
        }
        i += need + 1;
    }
    return 1;
}

int main(void) {
    uint64_t s = 9;
    buf = malloc(N);
    if (buf == NULL) {
        return 1;
    }
    while (len < N) {
        push_char(&s);
    }
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t r = 0; r < REPS; r++) {
        size_t at = (r * 7919) % N;
        unsigned char was = buf[at];
        if (r % 3 == 0) {
            buf[at] = 0xff;
        }
        h = mix(h, (uint64_t)valid_utf8(buf, N));
        buf[at] = was;
    }
    printf("%llu\n", (unsigned long long)h);
    free(buf);
    return 0;
}
