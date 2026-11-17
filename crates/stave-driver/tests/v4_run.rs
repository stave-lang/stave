//! The `v4` leg end to end: AVX-512 code in static ELF executables, debug and release, run on
//! an AVX-512 CPU or under Intel SDE (`$STAVE_SDE -icx --`, tests/support/legs.rs):
//!
//! - a default-target `dispatch` with a 512-bit arm takes that arm under the v4 set and the
//!   scalar `_` arm elsewhere, with the same output;
//! - the same program built for `x86_64-linux-v4` (std and runtime too) prints the same;
//! - lane casts, widen and narrow in a v4 build print what the IR interpreter prints, whose
//!   lane casts are the scalar ones;
//! - `std.cpu.features()` has the five v4 bits exactly where the CPU (or SDE's CPUID) does;
//! - a v4 binary on a CPU without the v4 set stops with `panic: UnsupportedCpu`.
//!
//! A leg that cannot run here is skipped with a message, or fails when `STAVE_REQUIRE_LEGS`
//! lists it. SDE emulates CPUID, not `/proc/cpuinfo`: the native expectations come from
//! `exec::can_run` and the standard library's detection.

use std::path::{Path, PathBuf};

#[path = "../../../tests/support/legs.rs"]
mod legs;

use legs::V4Run;
use stave_driver::exec::{self, Executable, RunSpec, TempDir};
use stave_driver::interp::{self, Limits};
use stave_driver::pipeline::{self, Goal, Options, Output};
use stave_driver::shipped::Library;
use stave_ir::{Module, Op, Operand};
use stave_target::Triple;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A `[f32]` of 37 elements (two full 16-lane chunks and a tail of 5) through `scale`: each
/// element above 0.5 doubled, the others less 1; prints the sum and the output array. The
/// inputs are odd quarters in -4.25..4.75, so every output and every partial sum is exact in
/// f32 whatever the summation order.
const KERNEL: &str = r#"import std.io;

const THR: f32 = 0.5;

fn scale_v4(xs: &[f32], out: &mut [f32], thr: f32) -> f32 needs(avx512f, avx512bw, avx512vl, avx512dq, avx512cd) {
    let t = f32x16.splat(thr);
    let two = f32x16.splat(2.0);
    let one = f32x16.splat(1.0);
    let zero = f32x16.splat(0.0);
    let mut acc = zero;
    let n = xs.len();
    let mut i: usize = 0;
    while i + 16 <= n {
        let v = f32x16.load(xs, i);
        let r = (v > t).select(v * two, v - one);
        r.store(out, i);
        acc = acc + r;
        i = i + 16;
    }
    if i < n {
        let m = mask32x16.first_n(n - i);
        let v = f32x16.load_masked(xs, i, m);
        let r = (v > t).select(v * two, v - one);
        r.store_masked(out, i, m);
        acc = acc + m.select(r, zero);
    }
    return acc.reduce_add();
}

fn scale_base(xs: &[f32], out: &mut [f32], thr: f32) -> f32 {
    let mut acc: f32 = 0.0;
    for i in 0..xs.len() {
        let x = xs[i];
        let r = if x > thr then x * 2.0 else x - 1.0;
        out[i] = r;
        acc = acc + r;
    }
    return acc;
}

dispatch fn scale(xs: &[f32], out: &mut [f32], thr: f32) -> f32 {
    avx512f + avx512bw + avx512vl + avx512dq + avx512cd => scale_v4,
    _ => scale_base,
}

fn main() -> i32 {
    let mut xs: [f32; 37] = [0.0; 37];
    let mut out: [f32; 37] = [0.0; 37];
    let mut k: u32 = 0;
    let mut x: f32 = -4.25;
    for i in 0usize..37 {
        xs[i] = x;
        k = k + 7;
        x = x + 3.5;
        if k >= 19 {
            k = k - 19;
            x = x - 9.5;
        }
    }
    let s = scale(&xs[0..37], &mut out[0..37], THR);
    io.print(f"sum {s}\n");
    io.print("out");
    for i in 0usize..37 {
        let y = out[i];
        io.print(f" {y}");
    }
    io.print("\n");
    return 0;
}
"#;

/// Prints `<feature> <0|1>` for the five features of the v4 set.
const FEATURES: &str = r#"import std.cpu;
import std.io;

fn bit(name: &str, w: u64, m: u64) {
    io.print(name);
    if w & m != 0 { io.print(" 1\n"); } else { io.print(" 0\n"); }
}

fn main() -> i32 {
    let w = cpu.features();
    bit("avx512f", w, cpu.AVX512F);
    bit("avx512bw", w, cpu.AVX512BW);
    bit("avx512vl", w, cpu.AVX512VL);
    bit("avx512dq", w, cpu.AVX512DQ);
    bit("avx512cd", w, cpu.AVX512CD);
    return 0;
}
"#;

/// Every family of lane cast (`as`, `as%`, `as|`, `as~`, `as?`; int and float lanes both ways),
/// `widen_lo` / `widen_hi`, `narrow_wrap` / `narrow_sat`, mostly on 512-bit vectors, a few at
/// 128 and 256 bits, over edge values: NaN of either sign, infinities, -0, the bounds of the
/// target type and the values just beyond them, fractions, rounding ties, the unsigned lanes
/// whose conversion rounds up to 2^bits. `as?` runs once with every lane exact and once per
/// kind of failing lane.
const CASTS: &str = r#"import std.io;

fn p_i8(name: &str, xs: &[i8]) {
    io.print(name);
    for i in 0..xs.len() {
        let x = xs[i];
        io.print(f" {x}");
    }
    io.print("\n");
}

fn p_u8(name: &str, xs: &[u8]) {
    io.print(name);
    for i in 0..xs.len() {
        let x = xs[i];
        io.print(f" {x}");
    }
    io.print("\n");
}

fn p_i16(name: &str, xs: &[i16]) {
    io.print(name);
    for i in 0..xs.len() {
        let x = xs[i];
        io.print(f" {x}");
    }
    io.print("\n");
}

fn p_u16(name: &str, xs: &[u16]) {
    io.print(name);
    for i in 0..xs.len() {
        let x = xs[i];
        io.print(f" {x}");
    }
    io.print("\n");
}

fn p_i32(name: &str, xs: &[i32]) {
    io.print(name);
    for i in 0..xs.len() {
        let x = xs[i];
        io.print(f" {x}");
    }
    io.print("\n");
}

fn p_u32(name: &str, xs: &[u32]) {
    io.print(name);
    for i in 0..xs.len() {
        let x = xs[i];
        io.print(f" {x}");
    }
    io.print("\n");
}

fn p_i64(name: &str, xs: &[i64]) {
    io.print(name);
    for i in 0..xs.len() {
        let x = xs[i];
        io.print(f" {x}");
    }
    io.print("\n");
}

fn p_u64(name: &str, xs: &[u64]) {
    io.print(name);
    for i in 0..xs.len() {
        let x = xs[i];
        io.print(f" {x}");
    }
    io.print("\n");
}

fn p_f32(name: &str, xs: &[f32]) {
    io.print(name);
    for i in 0..xs.len() {
        let x = xs[i];
        io.print(f" {x}");
    }
    io.print("\n");
}

fn p_f64(name: &str, xs: &[f64]) {
    io.print(name);
    for i in 0..xs.len() {
        let x = xs[i];
        io.print(f" {x}");
    }
    io.print("\n");
}

fn none(name: &str) {
    io.print(name);
    io.print(" None\n");
}

// `as?`: the lanes, or None when one lane does not convert exactly

fn try_f32_i32(name: &str, bits: &[u32]) {
    let mut o: [i32; 16] = [0; 16];
    match u32x16.load(bits, 0).bitcast::<f32x16>() as? i32x16 {
        Option.Some(r) => { r.store(&mut o, 0); p_i32(name, &o); }
        Option.None => { none(name); }
    }
}

fn try_f32_u32(name: &str, bits: &[u32]) {
    let mut o: [u32; 16] = [0; 16];
    match u32x16.load(bits, 0).bitcast::<f32x16>() as? u32x16 {
        Option.Some(r) => { r.store(&mut o, 0); p_u32(name, &o); }
        Option.None => { none(name); }
    }
}

fn try_f32_u8(name: &str, bits: &[u32]) {
    let mut o: [u8; 16] = [0; 16];
    match u32x16.load(bits, 0).bitcast::<f32x16>() as? u8x16 {
        Option.Some(r) => { r.store(&mut o, 0); p_u8(name, &o); }
        Option.None => { none(name); }
    }
}

fn try_f64_u64(name: &str, bits: &[u64]) {
    let mut o: [u64; 8] = [0; 8];
    match u64x8.load(bits, 0).bitcast::<f64x8>() as? u64x8 {
        Option.Some(r) => { r.store(&mut o, 0); p_u64(name, &o); }
        Option.None => { none(name); }
    }
}

fn try_f64_u32(name: &str, bits: &[u64]) {
    let mut o: [u32; 8] = [0; 8];
    match u64x8.load(bits, 0).bitcast::<f64x8>() as? u32x8 {
        Option.Some(r) => { r.store(&mut o, 0); p_u32(name, &o); }
        Option.None => { none(name); }
    }
}

fn try_f64_i16(name: &str, bits: &[u64]) {
    let mut o: [i16; 8] = [0; 8];
    match u64x8.load(bits, 0).bitcast::<f64x8>() as? i16x8 {
        Option.Some(r) => { r.store(&mut o, 0); p_i16(name, &o); }
        Option.None => { none(name); }
    }
}

fn try_f64_f32(name: &str, bits: &[u64]) {
    let mut o: [f32; 8] = [0.0; 8];
    match u64x8.load(bits, 0).bitcast::<f64x8>() as? f32x8 {
        Option.Some(r) => { r.store(&mut o, 0); p_f32(name, &o); }
        Option.None => { none(name); }
    }
}

fn try_u32_f32(name: &str, xs: &[u32]) {
    let mut o: [f32; 16] = [0.0; 16];
    match u32x16.load(xs, 0) as? f32x16 {
        Option.Some(r) => { r.store(&mut o, 0); p_f32(name, &o); }
        Option.None => { none(name); }
    }
}

fn try_i32_f32(name: &str, xs: &[i32]) {
    let mut o: [f32; 16] = [0.0; 16];
    match i32x16.load(xs, 0) as? f32x16 {
        Option.Some(r) => { r.store(&mut o, 0); p_f32(name, &o); }
        Option.None => { none(name); }
    }
}

fn try_u64_f64(name: &str, xs: &[u64]) {
    let mut o: [f64; 8] = [0.0; 8];
    match u64x8.load(xs, 0) as? f64x8 {
        Option.Some(r) => { r.store(&mut o, 0); p_f64(name, &o); }
        Option.None => { none(name); }
    }
}

fn try_i64_f32(name: &str, xs: &[i64]) {
    let mut o: [f32; 8] = [0.0; 8];
    match i64x8.load(xs, 0) as? f32x8 {
        Option.Some(r) => { r.store(&mut o, 0); p_f32(name, &o); }
        Option.None => { none(name); }
    }
}

fn try_i32_i16(name: &str, xs: &[i32]) {
    let mut o: [i16; 16] = [0; 16];
    match i32x16.load(xs, 0) as? i16x16 {
        Option.Some(r) => { r.store(&mut o, 0); p_i16(name, &o); }
        Option.None => { none(name); }
    }
}

fn try_u32_i32(name: &str, xs: &[u32]) {
    let mut o: [i32; 16] = [0; 16];
    match u32x16.load(xs, 0) as? i32x16 {
        Option.Some(r) => { r.store(&mut o, 0); p_i32(name, &o); }
        Option.None => { none(name); }
    }
}

fn try_i64_u16(name: &str, xs: &[i64]) {
    let mut o: [u16; 8] = [0; 8];
    match i64x8.load(xs, 0) as? u16x8 {
        Option.Some(r) => { r.store(&mut o, 0); p_u16(name, &o); }
        Option.None => { none(name); }
    }
}

fn try_i8_u16(name: &str, xs: &[i8]) {
    let mut o: [u16; 32] = [0; 32];
    match i8x32.load(xs, 0) as? u16x32 {
        Option.Some(r) => { r.store(&mut o, 0); p_u16(name, &o); }
        Option.None => { none(name); }
    }
}

fn main() -> i32 {
    let mut ob: [i8; 64] = [0; 64];
    let mut oub: [u8; 64] = [0; 64];
    let mut ow: [i16; 32] = [0; 32];
    let mut ouw: [u16; 32] = [0; 32];
    let mut od: [i32; 16] = [0; 16];
    let mut oud: [u32; 16] = [0; 16];
    let mut oq: [i64; 8] = [0; 8];
    let mut ouq: [u64; 8] = [0; 8];
    let mut ofs: [f32; 16] = [0.0; 16];
    let mut ofd: [f64; 8] = [0.0; 8];

    // f32: NaN, -NaN, inf, -inf, 2^31, the next below -2^31, the largest below 2^31, -0.5, 0.5,
    // -1, 1.5, 2.5, -2.5, 3e9, -0, 65535.9
    let fbits: [u32; 16] = [
        0x7fc00000, 0xffc00001, 0x7f800000, 0xff800000, 0x4f000000, 0xcf000001, 0x4effffff,
        0xbf000000, 0x3f000000, 0xbf800000, 0x3fc00000, 0x40200000, 0xc0200000, 0x4f32d05e,
        0x80000000, 0x477fffe6
    ];
    let f = u32x16.load(&fbits, 0).bitcast::<f32x16>();
    (f as| i32x16).store(&mut od, 0);
    p_i32("f32 as| i32", &od);
    (f as| u32x16).store(&mut oud, 0);
    p_u32("f32 as| u32", &oud);
    (f as| i16x16).store(&mut ow, 0);
    p_i16("f32 as| i16", &ow[0..16]);
    (f as| u8x16).store(&mut oub, 0);
    p_u8("f32 as| u8", &oub[0..16]);
    let flo = u32x8.load(&fbits, 0).bitcast::<f32x8>();
    let fhi = u32x8.load(&fbits, 8).bitcast::<f32x8>();
    (flo as| i64x8).store(&mut oq, 0);
    p_i64("f32 lo as| i64", &oq);
    (fhi as| i64x8).store(&mut oq, 0);
    p_i64("f32 hi as| i64", &oq);
    (flo as| u64x8).store(&mut ouq, 0);
    p_u64("f32 lo as| u64", &ouq);
    (fhi as| u64x8).store(&mut ouq, 0);
    p_u64("f32 hi as| u64", &ouq);
    (fhi as f64x8).store(&mut ofd, 0);
    p_f64("f32 hi as f64", &ofd);
    f.widen_lo().store(&mut ofd, 0);
    p_f64("f32 widen_lo", &ofd);
    f.widen_hi().store(&mut ofd, 0);
    p_f64("f32 widen_hi", &ofd);
    let f4 = u32x4.load(&fbits, 4).bitcast::<f32x4>();
    (f4 as| i32x4).store(&mut od, 0);
    p_i32("f32x4 as| i32", &od[0..4]);
    (flo as| u16x8).store(&mut ouw, 0);
    p_u16("f32x8 as| u16", &ouw[0..8]);

    // f64: NaN, -inf, 2^63, -2^63, 2^64, -0.5, 2^32 - 1, 2.5
    let dbits: [u64; 8] = [
        0x7ff8000000000000, 0xfff0000000000000, 0x43e0000000000000, 0xc3e0000000000000,
        0x43f0000000000000, 0xbfe0000000000000, 0x41efffffffe00000, 0x4004000000000000
    ];
    let d = u64x8.load(&dbits, 0).bitcast::<f64x8>();
    (d as| i64x8).store(&mut oq, 0);
    p_i64("f64 as| i64", &oq);
    (d as| u64x8).store(&mut ouq, 0);
    p_u64("f64 as| u64", &ouq);
    (d as| i32x8).store(&mut od, 0);
    p_i32("f64 as| i32", &od[0..8]);
    (d as| u32x8).store(&mut oud, 0);
    p_u32("f64 as| u32", &oud[0..8]);
    (d as| i8x8).store(&mut ob, 0);
    p_i8("f64 as| i8", &ob[0..8]);
    (d as| u16x8).store(&mut ouw, 0);
    p_u16("f64 as| u16", &ouw[0..8]);

    // f64 to f32: overflow, a tie to even below, above a tie, half the least subnormal (a tie
    // to 0), NaN, -inf, the tie between MAX and 2^128 (to inf), -0
    let rbits: [u64; 8] = [
        0x7e37e43c8800759c, 0x3ff0000010000000, 0x3ff0000018000000, 0x3690000000000000,
        0x7ff8000000000001, 0xfff0000000000000, 0x47effffff0000000, 0x8000000000000000
    ];
    let r = u64x8.load(&rbits, 0).bitcast::<f64x8>();
    (r as~ f32x8).store(&mut ofs, 0);
    p_f32("f64 as~ f32", &ofs[0..8]);
    try_f64_f32("f64 as? f32 inexact", &rbits);
    // exact: 1.5, NaN, inf, -0, 0.1f32, f32 MAX, the least f32 subnormal, -2.5
    let mut xbits: [u64; 8] = [
        0x3ff8000000000000, 0x7ff8000000000000, 0x7ff0000000000000, 0x8000000000000000,
        0x3fb99999a0000000, 0x47efffffe0000000, 0x36a0000000000000, 0xc004000000000000
    ];
    try_f64_f32("f64 as? f32 exact", &xbits);
    xbits[4] = 0x3fb999999999999a;
    try_f64_f32("f64 as? f32 0.1", &xbits);

    // as? f32 to i32: integral and in range; then 2^31, 0.5, NaN in one lane
    let mut g: [u32; 16] = [
        0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x4effffff, 0xcf000000, 0x42c80000,
        0x4b800001, 0xcb800001, 0x40400000, 0xc0e00000, 0x477fff00, 0x437f0000, 0x44800000,
        0xc4800000, 0x4640e400
    ];
    try_f32_i32("f32 as? i32", &g);
    g[15] = 0x4f000000;
    try_f32_i32("f32 as? i32 2^31", &g);
    g[15] = 0x4640e400;
    g[0] = 0x3f000000;
    try_f32_i32("f32 as? i32 0.5", &g);
    g[0] = 0x00000000;
    g[7] = 0x7fc00000;
    try_f32_i32("f32 as? i32 NaN", &g);
    // as? f32 to u32: up to the largest below 2^32; then 2^32, -1
    let mut gu: [u32; 16] = [
        0x00000000, 0x80000000, 0x3f800000, 0x4f7fffff, 0x4f000000, 0x4effffff, 0x42c80000,
        0x4b800001, 0x477fff00, 0x40400000, 0x437f0000, 0x44800000, 0x4640e400, 0x4f7ffffe,
        0x4f400000, 0x3f800000
    ];
    try_f32_u32("f32 as? u32", &gu);
    gu[3] = 0x4f800000;
    try_f32_u32("f32 as? u32 2^32", &gu);
    gu[3] = 0x4f7fffff;
    gu[14] = 0xbf800000;
    try_f32_u32("f32 as? u32 -1", &gu);
    // as? f32 to u8: 0..255, then 256
    let mut gb: [u32; 16] = [
        0x00000000, 0x80000000, 0x3f800000, 0x437f0000, 0x42fe0000, 0x43000000, 0x40400000,
        0x41200000, 0x42c80000, 0x43480000, 0x437e0000, 0x40000000, 0x40a00000, 0x41000000,
        0x41800000, 0x42000000
    ];
    try_f32_u8("f32 as? u8", &gb);
    gb[2] = 0x43800000;
    try_f32_u8("f32 as? u8 256", &gb);
    // as? f64 to u64: 0, 2^63, the largest below 2^64, 1, 2^53, 12345, -0, 2^32 - 1; then 2^64
    let mut gq: [u64; 8] = [
        0x0000000000000000, 0x43e0000000000000, 0x43efffffffffffff, 0x3ff0000000000000,
        0x4340000000000000, 0x40c81c8000000000, 0x8000000000000000, 0x41efffffffe00000
    ];
    try_f64_u64("f64 as? u64", &gq);
    try_f64_u32("f64 as? u32 big", &gq);
    gq[1] = 0x41e0000000000000;
    gq[2] = 0x4000000000000000;
    gq[4] = 0x4130000000000000;
    try_f64_u32("f64 as? u32", &gq);
    try_f64_i16("f64 as? i16 big", &gq);
    // as? f64 to i16: 0, 32767, -32768, 1, -0, 12345, -1, 2; then 2.5, 32768
    let mut gw: [u64; 8] = [
        0x0000000000000000, 0x40dfffc000000000, 0xc0e0000000000000, 0x3ff0000000000000,
        0x8000000000000000, 0x40c81c8000000000, 0xbff0000000000000, 0x4000000000000000
    ];
    try_f64_i16("f64 as? i16", &gw);
    gw[7] = 0x4004000000000000;
    try_f64_i16("f64 as? i16 2.5", &gw);
    gw[7] = 0x40e0000000000000;
    try_f64_i16("f64 as? i16 32768", &gw);
    gq[1] = 0x43f0000000000000;
    try_f64_u64("f64 as? u64 2^64", &gq);

    // int to float, rounding: ties and the values next to them
    let mut di: [i32; 16] = [
        16777217, -16777217, 2147483647, -2147483648, 0x7fffffbf, 0x7fffffc0, 33554435,
        -33554435, 16777219, 0, 1, -1, 0x01000003, 0x0001ffff, 0x7fffff80, 123456789
    ];
    (i32x16.load(&di, 0) as~ f32x16).store(&mut ofs, 0);
    p_f32("i32 as~ f32", &ofs);
    (i32x8.load(&di, 0) as f64x8).store(&mut ofd, 0);
    p_f64("i32 as f64", &ofd);
    try_i32_f32("i32 as? f32 inexact", &di);
    di[0] = 16777216;
    di[1] = -16777216;
    di[2] = 2147483520;
    di[4] = 0x7fffff80;
    di[5] = 2;
    di[6] = 33554436;
    di[7] = -33554436;
    di[8] = 16777220;
    di[12] = 0x01000004;
    di[15] = 123456784;
    try_i32_f32("i32 as? f32", &di);
    let mut du: [u32; 16] = [
        0xffffffff, 0xffffff80, 0xffffff7f, 0x80000080, 0x80000081, 0x80000180, 0x80000000, 0,
        1, 16777217, 0x7fffffbf, 0xfffffe80, 0x01000003, 0xfffffffe, 0x7fffffff, 4294967040
    ];
    (u32x16.load(&du, 0) as~ f32x16).store(&mut ofs, 0);
    p_f32("u32 as~ f32", &ofs);
    (u32x8.load(&du, 0) as f64x8).store(&mut ofd, 0);
    p_f64("u32 as f64", &ofd);
    try_u32_f32("u32 as? f32 inexact", &du);
    try_u32_i32("u32 as? i32 big", &du);
    du[0] = 0xffffff00;
    du[1] = 0xfffffe00;
    du[2] = 0x80000100;
    du[3] = 0x80000000;
    du[4] = 0x00ffffff;
    du[5] = 0x40000000;
    du[9] = 16777216;
    du[10] = 0x7fffff80;
    du[11] = 0xfffffd00;
    du[12] = 0x01000004;
    du[13] = 3;
    du[14] = 0x7fffff00;
    try_u32_f32("u32 as? f32", &du);
    du[0] = 0xffffffff;
    try_u32_f32("u32 as? f32 MAX", &du);
    let mut dq: [i64; 8] = [
        9223372036854775807, -9223372036854775808, 9007199254740993, -9007199254740993,
        4611686018427387905, 1, -1, 27021597764222977
    ];
    (i64x8.load(&dq, 0) as~ f64x8).store(&mut ofd, 0);
    p_f64("i64 as~ f64", &ofd);
    (i64x8.load(&dq, 0) as~ f32x8).store(&mut ofs, 0);
    p_f32("i64 as~ f32", &ofs[0..8]);
    try_i64_f32("i64 as? f32 MAX", &dq);
    dq[0] = 4611686018427387904;
    dq[2] = 16777216;
    dq[3] = -16777216;
    dq[4] = 1099511627776;
    dq[7] = -1099511627776;
    try_i64_f32("i64 as? f32", &dq);
    let mut duq: [u64; 8] = [
        18446744073709551615, 9223372036854776833, 9223372036854775808, 9007199254740993, 0, 1,
        18446744073709549568, 9223372586610589697
    ];
    (u64x8.load(&duq, 0) as~ f64x8).store(&mut ofd, 0);
    p_f64("u64 as~ f64", &ofd);
    (u64x8.load(&duq, 0) as~ f32x8).store(&mut ofs, 0);
    p_f32("u64 as~ f32", &ofs[0..8]);
    try_u64_f64("u64 as? f64 MAX", &duq);
    duq[0] = 18446744073709549568;
    duq[1] = 9223372036854777856;
    duq[3] = 9007199254740992;
    duq[7] = 9223372586610589696;
    try_u64_f64("u64 as? f64", &duq);

    // int to float, exact: 8- and 16-bit lanes
    let mut bs: [i8; 64] = [0; 64];
    let mut us: [u8; 64] = [0; 64];
    for i in 0usize..64 {
        let k = (i as% i32) *% 37 -% 1000;
        bs[i] = k as% i8;
        us[i] = k as% u8;
    }
    bs[0] = -128;
    bs[1] = 127;
    us[0] = 255;
    (i8x16.load(&bs, 0) as f32x16).store(&mut ofs, 0);
    p_f32("i8 as f32", &ofs);
    (u8x16.load(&us, 0) as f32x16).store(&mut ofs, 0);
    p_f32("u8 as f32", &ofs);
    (i8x8.load(&bs, 0) as~ f64x8).store(&mut ofd, 0);
    p_f64("i8 as~ f64", &ofd);
    let mut ws: [i16; 32] = [0; 32];
    let mut uws: [u16; 32] = [0; 32];
    for i in 0usize..32 {
        let k = (i as% i32) *% 2111 -% 30000;
        ws[i] = k as% i16;
        uws[i] = (k *% 3) as% u16;
    }
    ws[0] = -32768;
    ws[1] = 32767;
    uws[0] = 65535;
    uws[1] = 128;
    uws[2] = 127;
    uws[3] = 255;
    uws[4] = 256;
    (i16x16.load(&ws, 0) as f32x16).store(&mut ofs, 0);
    p_f32("i16 as f32", &ofs);
    (u16x8.load(&uws, 0) as f64x8).store(&mut ofd, 0);
    p_f64("u16 as f64", &ofd);

    // int to int
    let e32: [i32; 16] = [
        -2147483648, -32769, -32768, -129, -128, -1, 0, 1, 127, 128, 255, 256, 32767, 32768,
        65535, 2147483647
    ];
    let a = i32x16.load(&e32, 0);
    (a as% i8x16).store(&mut ob, 0);
    p_i8("i32 as% i8", &ob[0..16]);
    (a as% u16x16).store(&mut ouw, 0);
    p_u16("i32 as% u16", &ouw[0..16]);
    (a as| u8x16).store(&mut oub, 0);
    p_u8("i32 as| u8", &oub[0..16]);
    (a as| i8x16).store(&mut ob, 0);
    p_i8("i32 as| i8", &ob[0..16]);
    (a as| i16x16).store(&mut ow, 0);
    p_i16("i32 as| i16", &ow[0..16]);
    (a as| u16x16).store(&mut ouw, 0);
    p_u16("i32 as| u16", &ouw[0..16]);
    (a as| u32x16).store(&mut oud, 0);
    p_u32("i32 as| u32", &oud);
    (i32x8.load(&e32, 0) as| u64x8).store(&mut ouq, 0);
    p_u64("i32 as| u64", &ouq);
    (i32x8.load(&e32, 8) as i64x8).store(&mut oq, 0);
    p_i64("i32 as i64", &oq);
    try_i32_i16("i32 as? i16 range", &e32);
    let mut e16: [i32; 16] = [0; 16];
    for i in 0usize..16 {
        let x = e32[i];
        e16[i] = if x < -32768 then -32768 else if x > 32767 then 32767 else x;
    }
    try_i32_i16("i32 as? i16", &e16);
    let ue: [u32; 16] = [
        0x80000000, 0xffffffff, 0x7fffffff, 0, 1, 127, 128, 255, 256, 32767, 32768, 65535,
        65536, 0x7fff0000, 0xffff8000, 2147483647
    ];
    let b = u32x16.load(&ue, 0);
    (b as| i32x16).store(&mut od, 0);
    p_i32("u32 as| i32", &od);
    (b as| i8x16).store(&mut ob, 0);
    p_i8("u32 as| i8", &ob[0..16]);
    (b as| u16x16).store(&mut ouw, 0);
    p_u16("u32 as| u16", &ouw[0..16]);
    (u32x8.load(&ue, 0) as u64x8).store(&mut ouq, 0);
    p_u64("u32 as u64", &ouq);
    let mut ui: [u32; 16] = [0; 16];
    for i in 0usize..16 {
        let x = ue[i];
        ui[i] = if x > 2147483647 then 2147483647 else x;
    }
    try_u32_i32("u32 as? i32", &ui);
    let w = u16x32.load(&uws, 0);
    (w as| i8x32).store(&mut ob, 0);
    p_i8("u16 as| i8", &ob[0..32]);
    (w as| u8x32).store(&mut oub, 0);
    p_u8("u16 as| u8", &oub[0..32]);
    (w as% i8x32).store(&mut ob, 0);
    p_i8("u16 as% i8", &ob[0..32]);
    (w as| i16x32).store(&mut ow, 0);
    p_i16("u16 as| i16", &ow);
    let s16 = i16x32.load(&ws, 0);
    (s16 as| u8x32).store(&mut oub, 0);
    p_u8("i16 as| u8", &oub[0..32]);
    (s16 as| u16x32).store(&mut ouw, 0);
    p_u16("i16 as| u16", &ouw);
    let e64: [i64; 8] = [
        -9223372036854775808, -2147483649, -2147483648, -1, 0, 4294967295, 4294967296,
        9223372036854775807
    ];
    let q = i64x8.load(&e64, 0);
    (q as| i32x8).store(&mut od, 0);
    p_i32("i64 as| i32", &od[0..8]);
    (q as| u32x8).store(&mut oud, 0);
    p_u32("i64 as| u32", &oud[0..8]);
    (q as% i16x8).store(&mut ow, 0);
    p_i16("i64 as% i16", &ow[0..8]);
    (q as| u8x8).store(&mut oub, 0);
    p_u8("i64 as| u8", &oub[0..8]);
    (q as| u64x8).store(&mut ouq, 0);
    p_u64("i64 as| u64", &ouq);
    try_i64_u16("i64 as? u16 range", &e64);
    let small: [i64; 8] = [0, 1, 65535, 32768, 255, 256, 12345, 65534];
    try_i64_u16("i64 as? u16", &small);
    let uq = q.bitcast::<u64x8>();
    (uq as| i64x8).store(&mut oq, 0);
    p_i64("u64 as| i64", &oq);
    (uq as| u16x8).store(&mut ouw, 0);
    p_u16("u64 as| u16", &ouw[0..8]);
    (uq as| i32x8).store(&mut od, 0);
    p_i32("u64 as| i32", &od[0..8]);
    let sb = i8x32.load(&bs, 0);
    (sb as i16x32).store(&mut ow, 0);
    p_i16("i8 as i16", &ow);
    (sb as| u16x32).store(&mut ouw, 0);
    p_u16("i8 as| u16", &ouw);
    (sb as% u16x32).store(&mut ouw, 0);
    p_u16("i8 as% u16", &ouw);
    (i8x16.load(&bs, 0) as i32x16).store(&mut od, 0);
    p_i32("i8 as i32", &od);
    (i8x8.load(&bs, 0) as| u64x8).store(&mut ouq, 0);
    p_u64("i8 as| u64", &ouq);
    (u8x64.load(&us, 0) as| i8x64).store(&mut ob, 0);
    p_i8("u8 as| i8", &ob);
    (i8x64.load(&bs, 0) as| u8x64).store(&mut oub, 0);
    p_u8("i8 as| u8", &oub);
    try_i8_u16("i8 as? u16 negative", &bs);
    let mut pos: [i8; 32] = [0; 32];
    for i in 0usize..32 {
        let x = bs[i];
        pos[i] = if x < 0 then 0 else x;
    }
    try_i8_u16("i8 as? u16", &pos);

    // widen and narrow
    let v8 = i8x64.load(&bs, 0);
    v8.widen_lo().store(&mut ow, 0);
    p_i16("i8 widen_lo", &ow);
    v8.widen_hi().store(&mut ow, 0);
    p_i16("i8 widen_hi", &ow);
    u8x64.load(&us, 0).widen_hi().store(&mut ouw, 0);
    p_u16("u8 widen_hi", &ouw);
    s16.widen_hi().store(&mut od, 0);
    p_i32("i16 widen_hi", &od);
    b.widen_lo().store(&mut ouq, 0);
    p_u64("u32 widen_lo", &ouq);
    i8x16.load(&bs, 0).widen_hi().store(&mut ow, 0);
    p_i16("i8x16 widen_hi", &ow[0..8]);
    let c = i32x16.load(&di, 0);
    a.narrow_sat(c).store(&mut ow, 0);
    p_i16("i32 narrow_sat", &ow);
    a.narrow_wrap(c).store(&mut ow, 0);
    p_i16("i32 narrow_wrap", &ow);
    b.narrow_sat(u32x16.load(&du, 0)).store(&mut ouw, 0);
    p_u16("u32 narrow_sat", &ouw);
    s16.narrow_sat(w.bitcast::<i16x32>()).store(&mut ob, 0);
    p_i8("i16 narrow_sat", &ob);
    s16.narrow_wrap(s16).store(&mut ob, 0);
    p_i8("i16 narrow_wrap", &ob);
    w.narrow_sat(w).store(&mut oub, 0);
    p_u8("u16 narrow_sat", &oub);
    q.narrow_sat(i64x8.load(&dq, 0)).store(&mut od, 0);
    p_i32("i64 narrow_sat", &od);
    q.narrow_wrap(q).store(&mut od, 0);
    p_i32("i64 narrow_wrap", &od);
    uq.narrow_sat(u64x8.load(&duq, 0)).store(&mut oud, 0);
    p_u32("u64 narrow_sat", &oud);
    u16x8.load(&uws, 0).narrow_sat(u16x8.load(&uws, 8)).store(&mut oub, 0);
    p_u8("u16x8 narrow_sat", &oub[0..16]);
    i32x4.load(&e32, 0).narrow_wrap(i32x4.load(&e32, 12)).store(&mut ow, 0);
    p_i16("i32x4 narrow_wrap", &ow[0..8]);
    i64x2.load(&e64, 0).narrow_sat(i64x2.load(&e64, 6)).store(&mut od, 0);
    p_i32("i64x2 narrow_sat", &od[0..4]);
    return 0;
}
"#;

const V4_SET: [&str; 5] = ["avx512f", "avx512bw", "avx512vl", "avx512dq", "avx512cd"];

/// [`KERNEL`]'s stdout.
fn kernel_stdout() -> String {
    let mut xs = [0f32; 37];
    let (mut k, mut x) = (0u32, -4.25f32);
    for v in &mut xs {
        *v = x;
        k += 7;
        x += 3.5;
        if k >= 19 {
            k -= 19;
            x -= 9.5;
        }
    }
    let out: Vec<f32> = xs.iter().map(|&x| if x > 0.5 { x * 2.0 } else { x - 1.0 }).collect();
    let sum: f32 = out.iter().sum();
    // Stave's float text and Rust's `{:?}` agree on these values: a decimal point, no exponent
    let mut s = format!("sum {sum:?}\nout");
    for y in &out {
        s.push_str(&format!(" {y:?}"));
    }
    s.push('\n');
    s
}

fn build(p: &Path, target: Triple, release: bool) -> Output {
    let mut o = Options::new(Goal::Build).threads(Some(4));
    o.target = Some(target);
    o.release = release;
    let out = pipeline::run(p, &Library::in_dir(&repo()), &o).unwrap();
    assert!(!out.has_errors(), "{} ({target}): diagnostics {:?}", p.display(), out.diagnostics);
    let mode = if release { "release" } else { "debug" };
    assert!(out.failure.is_none(), "{} ({target}, {mode}): {}", p.display(), out.failure.as_ref().unwrap());
    assert!(out.image.is_some());
    out
}

/// The paths of the fns `path`'s body calls directly, without the dispatch trace calls (`None`:
/// no such fn with a body).
fn callees(m: &Module, path: &str) -> Option<Vec<String>> {
    let f = m.funcs.iter().find(|f| f.path == path)?;
    let b = f.body.as_ref()?;
    let mut out = Vec::new();
    for (i, op) in b.insts.op.iter().enumerate() {
        if *op == Op::Call
            && let Some(Operand::Func(g)) = b.args(i as u32).first()
            && let Some(g) = m.func(*g)
            && g.path != "runtime.cpu.trace"
        {
            out.push(g.path.clone());
        }
    }
    Some(out)
}

fn has_body(m: &Module, path: &str) -> bool {
    m.funcs.iter().any(|f| f.path == path && f.body.is_some())
}

struct Ran {
    exit: i32,
    stdout: String,
    stderr: String,
}

impl Ran {
    fn of(r: exec::Ran) -> Ran {
        Ran {
            exit: r.exit,
            stdout: String::from_utf8_lossy(&r.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&r.stderr).into_owned(),
        }
    }

    /// The `dispatch: ` lines of stderr (SDE may add its own lines).
    fn trace(&self) -> Vec<&str> {
        self.stderr.lines().filter(|l| l.starts_with("dispatch: ")).collect()
    }

    fn show(&self) -> String {
        format!("exit {}\n--- stdout\n{}--- stderr\n{}", self.exit, self.stdout, self.stderr)
    }
}

/// Runs the file `bin` (under `runner`) with `STAVE_DISPATCH_TRACE=1` when `trace`, without
/// `STAVE_CPU`. ETXTBSY (another test thread's fork still holds a just-written binary open) is
/// retried.
fn run(bin: &Path, runner: Option<&str>, trace: bool) -> Ran {
    let spec = RunSpec { args: &[], runner, cwd: None, inherit: false };
    let mut tries = 0;
    loop {
        let mut cmd = exec::command(bin, &spec).unwrap();
        cmd.env_remove("STAVE_CPU").env_remove("STAVE_DISPATCH_TRACE");
        if trace {
            cmd.env("STAVE_DISPATCH_TRACE", "1");
        }
        match cmd.stdin(std::process::Stdio::null()).output() {
            Err(e) if e.raw_os_error() == Some(26) && tries < 50 => {
                tries += 1;
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(e) => panic!("{}: {e}", bin.display()),
            Ok(o) => {
                return Ran {
                    exit: exec::status_code(o.status),
                    stdout: String::from_utf8_lossy(&o.stdout).into_owned(),
                    stderr: String::from_utf8_lossy(&o.stderr).into_owned(),
                };
            }
        }
    }
}

fn triple(name: &str) -> Triple {
    Triple::parse(name).unwrap()
}

/// Whether default-target binaries run natively here (the `v2` leg).
fn native_v2() -> bool {
    let v2 = triple("x86_64-linux");
    legs::gate(v2.name(), exec::can_run(v2))
}

/// How the `v4` leg runs here (`None`: skipped, with a note).
fn v4_leg(what: &str) -> Option<V4Run> {
    let how = legs::v4(exec::can_run(triple("x86_64-linux-v4")));
    if how.is_none() {
        eprintln!("note: no AVX-512 CPU and no STAVE_SDE: {what} skipped");
    }
    how
}

fn runner_of(how: &V4Run) -> Option<&str> {
    match how {
        V4Run::Native => None,
        V4Run::Runner(r) => Some(r.as_str()),
    }
}

/// The default-target dispatch takes the 512-bit arm exactly where the CPU has the v4 set, and
/// both arms print the same.
#[test]
fn dispatch_v4_arm() {
    let tmp = TempDir::new("v4-disp").unwrap();
    let src = tmp.path().join("kernel.stv");
    std::fs::write(&src, KERNEL).unwrap();
    let want = kernel_stdout();
    let v4_cpu = exec::can_run(triple("x86_64-linux-v4"));
    // the IR interpreter sees no CPU features: it takes `_`
    let mut o = Options::new(Goal::Check).threads(Some(4));
    o.lower = true;
    o.target = Some(triple("x86_64-linux"));
    let checked = pipeline::run(&src, &Library::in_dir(&repo()), &o).unwrap();
    assert!(!checked.has_errors(), "{:?}", checked.diagnostics);
    let m = checked.module.as_ref().unwrap();
    let r = interp::run(m, "main.main", triple("x86_64-linux"), &[], &[], Limits::default()).unwrap().ran;
    assert_eq!((r.exit, String::from_utf8_lossy(&r.stdout).into_owned()), (0, want.clone()), "interpreter");

    let arm =
        |v4: bool| format!("dispatch: main.scale -> main.{}", if v4 { "scale_v4" } else { "scale_base" });

    // (runner, whether the run sees the v4 set, label)
    let mut cases: Vec<(Option<String>, bool, &str)> = Vec::new();
    if native_v2() {
        cases.push((None, v4_cpu, "native"));
    } else {
        eprintln!("note: x86_64-linux binaries cannot run here: native runs skipped");
    }
    match v4_leg("the 512-bit arm's run") {
        Some(V4Run::Runner(r)) => cases.push((Some(r), true, "sde")),
        // the native case runs the 512-bit arm
        Some(V4Run::Native) => {}
        None => {}
    }

    let mut failures = Vec::new();
    let mut runs = 0;
    for release in [false, true] {
        let mode = if release { "release" } else { "debug" };
        let out = build(&src, triple("x86_64-linux"), release);
        let m = out.module.as_ref().unwrap();
        let calls = callees(m, "main.scale").expect("no `main.scale` with a body");
        assert_eq!(calls, ["main.scale_v4", "main.scale_base"], "{mode}");
        let bin = tmp.path().join(format!("kernel-{mode}"));
        out.image.as_ref().unwrap().write_file(&bin).unwrap();
        for (runner, v4, label) in &cases {
            let got = run(&bin, runner.as_deref(), true);
            runs += 1;
            let trace = arm(*v4);
            if got.exit != 0 || got.stdout != want || got.trace() != [trace.as_str()] {
                failures
                    .push(format!("[{mode}, {label}] want `{trace}`\n{}--- want stdout\n{want}", got.show()));
            }
        }
    }
    eprintln!("{runs} runs");
    assert!(failures.is_empty(), "{} failure(s):\n{}", failures.len(), failures.join("\n"));
}

/// [`KERNEL`] built for `x86_64-linux-v4`: the baseline covers the 512-bit arm, so `main` calls
/// it directly, and the program prints what the default-target build prints.
#[test]
fn whole_program_v4() {
    let tmp = TempDir::new("v4-whole").unwrap();
    let src = tmp.path().join("kernel.stv");
    std::fs::write(&src, KERNEL).unwrap();
    let want = kernel_stdout();
    let how = v4_leg("the x86_64-linux-v4 runs");
    let mut failures = Vec::new();
    for release in [false, true] {
        let mode = if release { "release" } else { "debug" };
        let out = build(&src, triple("x86_64-linux-v4"), release);
        let m = out.module.as_ref().unwrap();
        let main_calls = callees(m, "main.main").expect("no `main.main` with a body");
        assert!(main_calls.iter().any(|c| c == "main.scale_v4"), "{mode}: main calls {main_calls:?}");
        assert!(!has_body(m, "main.scale"), "{mode}: the resolved dispatch is built");
        let Some(how) = &how else { continue };
        let bin = tmp.path().join(format!("kernel-v4-{mode}"));
        out.image.as_ref().unwrap().write_file(&bin).unwrap();
        let got = run(&bin, runner_of(how), true);
        if got.exit != 0 || got.stdout != want || !got.trace().is_empty() {
            failures.push(format!("[{mode}, {how:?}]\n{}--- want stdout\n{want}", got.show()));
        }
    }
    assert!(failures.is_empty(), "{} failure(s):\n{}", failures.len(), failures.join("\n"));
}

/// [`CASTS`] built for `x86_64-linux-v4`, debug and release, prints what the IR interpreter
/// prints for it.
#[test]
fn casts_v4() {
    let tmp = TempDir::new("v4-casts").unwrap();
    let src = tmp.path().join("casts.stv");
    std::fs::write(&src, CASTS).unwrap();
    let v4 = triple("x86_64-linux-v4");
    let mut o = Options::new(Goal::Check).threads(Some(4));
    o.lower = true;
    o.target = Some(v4);
    let checked = pipeline::run(&src, &Library::in_dir(&repo()), &o).unwrap();
    assert!(!checked.has_errors(), "{:?}", checked.diagnostics);
    let m = checked.module.as_ref().unwrap();
    let r = interp::run(m, "main.main", v4, &[], &[], Limits::default()).unwrap().ran;
    let want = String::from_utf8_lossy(&r.stdout).into_owned();
    assert_eq!(r.exit, 0, "interpreter:\n{want}");
    assert_eq!(want.lines().count(), 122, "interpreter:\n{want}");
    let how = v4_leg("the cast runs");
    let mut failures = Vec::new();
    for release in [false, true] {
        let mode = if release { "release" } else { "debug" };
        let out = build(&src, v4, release);
        let Some(how) = &how else { continue };
        let bin = tmp.path().join(format!("casts-{mode}"));
        out.image.as_ref().unwrap().write_file(&bin).unwrap();
        let got = run(&bin, runner_of(how), false);
        if got.exit != 0 || got.stdout != want {
            let diff: Vec<String> = got
                .stdout
                .lines()
                .zip(want.lines())
                .filter(|(g, w)| g != w)
                .map(|(g, w)| format!("got  {g}\nwant {w}"))
                .collect();
            failures.push(format!("[{mode}, {how:?}] exit {}\n{}", got.exit, diff.join("\n")));
        }
    }
    assert!(failures.is_empty(), "{} failure(s):\n{}", failures.len(), failures.join("\n"));
}

/// The v4 features of this CPU as the standard library detects them, in [`V4_SET`] order.
fn host_v4_bits() -> [bool; 5] {
    #[cfg(target_arch = "x86_64")]
    {
        [
            std::arch::is_x86_feature_detected!("avx512f"),
            std::arch::is_x86_feature_detected!("avx512bw"),
            std::arch::is_x86_feature_detected!("avx512vl"),
            std::arch::is_x86_feature_detected!("avx512dq"),
            std::arch::is_x86_feature_detected!("avx512cd"),
        ]
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        [false; 5]
    }
}

fn features_stdout(bits: [bool; 5]) -> String {
    V4_SET.iter().zip(bits).map(|(f, b)| format!("{f} {}\n", u8::from(b))).collect()
}

/// `std.cpu.features()` of a default-target program: all five v4 bits under SDE; natively the
/// CPU's own (all five set iff `can_run(x86_64-linux-v4)`). The SDE run goes both through a
/// file and through an in-memory [`Executable`] (the runner gets `/dev/fd/N`).
#[test]
fn cpu_features_under_sde() {
    let tmp = TempDir::new("v4-feat").unwrap();
    let src = tmp.path().join("features.stv");
    std::fs::write(&src, FEATURES).unwrap();
    let native = native_v2();
    let how = v4_leg("the SDE runs");
    let host = host_v4_bits();
    assert_eq!(
        host.iter().all(|b| *b),
        exec::can_run(triple("x86_64-linux-v4")),
        "can_run(x86_64-linux-v4) disagrees with the detected features {host:?}"
    );
    let mut failures = Vec::new();
    let mut runs = Vec::new();
    for release in [false, true] {
        let mode = if release { "release" } else { "debug" };
        let out = build(&src, triple("x86_64-linux"), release);
        let image = out.image.as_ref().unwrap();
        let bin = tmp.path().join(format!("features-{mode}"));
        image.write_file(&bin).unwrap();
        let mut check = |label: &str, got: Ran, want: &str| {
            runs.push(format!("{mode}, {label}"));
            if got.exit != 0 || got.stdout != want {
                failures.push(format!("[{mode}, {label}]\n{}--- want stdout\n{want}", got.show()));
            }
        };
        if native {
            check("native", run(&bin, None, false), &features_stdout(host));
        }
        if let Some(V4Run::Runner(r)) = &how {
            let all = features_stdout([true; 5]);
            check("sde, file", run(&bin, Some(r), false), &all);
            let mut exe = Executable::new("features", image.bytes()).unwrap();
            let spec = RunSpec { args: &[], runner: Some(r), cwd: None, inherit: false };
            let label = if exe.in_memory() { "sde, memory" } else { "sde, executable on disk" };
            check(label, Ran::of(exe.run(&spec).unwrap()), &all);
        }
    }
    eprintln!("runs: {runs:?}");
    assert!(failures.is_empty(), "{} failure(s):\n{}", failures.len(), failures.join("\n"));
}

/// A v4 binary started on a CPU without the v4 set stops before `main`.
#[test]
fn v4_binary_on_plain_cpu() {
    if exec::can_run(triple("x86_64-linux-v4")) {
        eprintln!("note: this CPU has the v4 set: the UnsupportedCpu run is skipped");
        return;
    }
    if !native_v2() {
        eprintln!("note: x86_64-linux binaries cannot run here: the UnsupportedCpu run is skipped");
        return;
    }
    let tmp = TempDir::new("v4-plain").unwrap();
    let src = tmp.path().join("features.stv");
    std::fs::write(&src, FEATURES).unwrap();
    let unsupported = "panic: UnsupportedCpu at <unknown>:0:0 in <unknown> (thread 0)\n";
    for release in [false, true] {
        let out = build(&src, triple("x86_64-linux-v4"), release);
        let bin = tmp.path().join(format!("plain-{release}"));
        out.image.as_ref().unwrap().write_file(&bin).unwrap();
        let got = run(&bin, None, false);
        let ok = got.exit == 101
            && got.stdout.is_empty()
            && got.stderr.starts_with(unsupported)
            && got.stderr.lines().count() == 2;
        assert!(ok, "release {release}: {}", got.show());
    }
}

/// The ops whose 512-bit forms take multi-instruction AVX-512 sequences (scans, streaming
/// stores, gathers and scatters, division by a constant, saturating and checked multiplies,
/// bit counts, permutes, lookups, compress / expand, 8-bit shifts by a vector), in a program
/// built for `x86_64-linux-v4`, debug and release: each family's vector results against the
/// program's own scalar references on edge values (every line `bad 0`), and the whole output
/// against the IR interpreter's.
#[test]
fn k_ops_match_scalar() {
    let tmp = TempDir::new("v4-ops").unwrap();
    let src = tmp.path().join("ops.stv");
    std::fs::write(&src, include_str!("v4_run/ops.stv")).unwrap();
    let v4 = triple("x86_64-linux-v4");
    let mut o = Options::new(Goal::Check).threads(Some(4));
    o.lower = true;
    o.target = Some(v4);
    let checked = pipeline::run(&src, &Library::in_dir(&repo()), &o).unwrap();
    assert!(!checked.has_errors(), "{:?}", checked.diagnostics);
    let m = checked.module.as_ref().unwrap();
    let r = interp::run(m, "main.main", v4, &[], &[], Limits::default()).unwrap().ran;
    let want = String::from_utf8_lossy(&r.stdout).into_owned();
    assert_eq!(r.exit, 0, "interpreter: {want}");
    let families: Vec<&str> = want.lines().filter(|l| l.contains(" bad ")).collect();
    assert_eq!(families.len(), 12, "interpreter:\n{want}");
    assert!(families.iter().all(|l| l.contains(" bad 0 ")), "interpreter:\n{want}");
    let how = v4_leg("the x86_64-linux-v4 runs of the K-mode ops");
    let mut failures = Vec::new();
    for release in [false, true] {
        let mode = if release { "release" } else { "debug" };
        let out = build(&src, v4, release);
        let Some(how) = &how else { continue };
        let bin = tmp.path().join(format!("ops-{mode}"));
        out.image.as_ref().unwrap().write_file(&bin).unwrap();
        let got = run(&bin, runner_of(how), false);
        if got.exit != 0 || got.stdout != want {
            failures.push(format!("[{mode}, {how:?}]\n{}--- want stdout\n{want}", got.show()));
        }
    }
    assert!(failures.is_empty(), "{} failure(s):\n{}", failures.len(), failures.join("\n"));
}
