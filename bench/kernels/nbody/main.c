/* The C twin of src/main.stv: same operations in the same order, same output. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define N 256
#define STEPS 600u
#define DT 0.001
#define EPS 0.01

static uint64_t next(uint64_t s) {
    uint64_t x = s;
    x = x ^ (x << 13);
    x = x ^ (x >> 7);
    x = x ^ (x << 17);
    return x;
}

static double unit(uint64_t s) {
    uint32_t hi = (uint32_t)(s >> 32);
    return ((double)hi) / 4294967296.0;
}

/* Stave's `as| i64`: truncate toward zero, saturate, NaN -> 0. */
static int64_t sat_i64(double d) {
    if (d != d) return 0;
    if (d >= 9223372036854775807.0) return INT64_MAX;
    if (d <= -9223372036854775808.0) return INT64_MIN;
    return (int64_t)d;
}

static double *zeros(size_t n) {
    double *v = calloc(n, sizeof(double));
    if (v == NULL) {
        fputs("out of memory\n", stderr);
        exit(1);
    }
    return v;
}

static void init(double *x, double *y, double *z, double *vx, double *vy, double *vz, double *m) {
    uint64_t s = 88172645463325252ull;
    for (size_t i = 0; i < N; i++) {
        s = next(s);
        x[i] = 2.0 * unit(s) - 1.0;
        s = next(s);
        y[i] = 2.0 * unit(s) - 1.0;
        s = next(s);
        z[i] = 2.0 * unit(s) - 1.0;
        s = next(s);
        vx[i] = 0.1 * (unit(s) - 0.5);
        s = next(s);
        vy[i] = 0.1 * (unit(s) - 0.5);
        s = next(s);
        vz[i] = 0.1 * (unit(s) - 0.5);
        s = next(s);
        m[i] = (0.5 + unit(s)) / 256.0;
    }
}

static double energy(const double *x, const double *y, const double *z, const double *vx, const double *vy,
                     const double *vz, const double *m) {
    double e = 0.0;
    for (size_t i = 0; i < N; i++) {
        e = e + 0.5 * m[i] * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
        for (size_t j = i + 1; j < N; j++) {
            double dx = x[i] - x[j];
            double dy = y[i] - y[j];
            double dz = z[i] - z[j];
            double r2 = dx * dx + dy * dy + dz * dz + EPS;
            e = e - 0.5 * m[i] * m[j] / r2;
        }
    }
    return e;
}

static void step(double *x, double *y, double *z, double *vx, double *vy, double *vz, const double *m) {
    for (size_t i = 0; i < N; i++) {
        double ax = 0.0;
        double ay = 0.0;
        double az = 0.0;
        for (size_t j = 0; j < N; j++) {
            if (j != i) {
                double dx = x[j] - x[i];
                double dy = y[j] - y[i];
                double dz = z[j] - z[i];
                double r2 = dx * dx + dy * dy + dz * dz + EPS;
                double f = m[j] / (r2 * r2);
                ax = ax + f * dx;
                ay = ay + f * dy;
                az = az + f * dz;
            }
        }
        vx[i] = vx[i] + DT * ax;
        vy[i] = vy[i] + DT * ay;
        vz[i] = vz[i] + DT * az;
    }
    for (size_t i = 0; i < N; i++) {
        x[i] = x[i] + DT * vx[i];
        y[i] = y[i] + DT * vy[i];
        z[i] = z[i] + DT * vz[i];
    }
}

static double position_sum(const double *x, const double *y, const double *z) {
    double s = 0.0;
    for (size_t i = 0; i < N; i++) {
        s = s + x[i] * 3.0 + y[i] * 5.0 + z[i] * 7.0;
    }
    return s;
}

int main(void) {
    double *x = zeros(N);
    double *y = zeros(N);
    double *z = zeros(N);
    double *vx = zeros(N);
    double *vy = zeros(N);
    double *vz = zeros(N);
    double *m = zeros(N);
    init(x, y, z, vx, vy, vz, m);
    double e0 = energy(x, y, z, vx, vy, vz, m);
    for (uint32_t k = 0; k < STEPS; k++) {
        step(x, y, z, vx, vy, vz, m);
    }
    double e1 = energy(x, y, z, vx, vy, vz, m);
    double p = position_sum(x, y, z);
    long long q0 = (long long)sat_i64(e0 * 1000000000000.0);
    long long q1 = (long long)sat_i64(e1 * 1000000000000.0);
    long long qp = (long long)sat_i64(p * 1000000000000.0);
    printf("energy %lld -> %lld positions %lld\n", q0, q1, qp);
    free(x);
    free(y);
    free(z);
    free(vx);
    free(vy);
    free(vz);
    free(m);
    return 0;
}
