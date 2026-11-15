//! Executes NEON code under `qemu-aarch64` (or natively on an AArch64 host): each result vector is
//! stored to the stack, written to stdout, and compared with a scalar model of the operation.

use std::process::Command;

use stave_obj::{DataObj, ImageBuilder, Machine, Options, stub};

use super::{have, output_retry, tmp};
use crate::enc::*;

const X: Width = Width::X;
const W: Width = Width::W;
const SP: Reg = Reg::SP;

fn x(n: u8) -> Reg {
    Reg::x(n).unwrap()
}

fn vr(n: u8) -> VReg {
    VReg::new(n).unwrap()
}

fn b32(l: [u32; 4]) -> [u8; 16] {
    let mut o = [0; 16];
    for (i, v) in l.iter().enumerate() {
        o[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
    }
    o
}

fn b16(l: [u16; 8]) -> [u8; 16] {
    let mut o = [0; 16];
    for (i, v) in l.iter().enumerate() {
        o[2 * i..2 * i + 2].copy_from_slice(&v.to_le_bytes());
    }
    o
}

fn b64(l: [u64; 2]) -> [u8; 16] {
    let mut o = [0; 16];
    o[..8].copy_from_slice(&l[0].to_le_bytes());
    o[8..].copy_from_slice(&l[1].to_le_bytes());
    o
}

fn bf(l: [f32; 4]) -> [u8; 16] {
    b32(l.map(f32::to_bits))
}

fn l32(b: [u8; 16]) -> [u32; 4] {
    std::array::from_fn(|i| u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap()))
}

fn l16(b: [u8; 16]) -> [u16; 8] {
    std::array::from_fn(|i| u16::from_le_bytes(b[2 * i..2 * i + 2].try_into().unwrap()))
}

fn l64(b: [u8; 16]) -> [u64; 2] {
    std::array::from_fn(|i| u64::from_le_bytes(b[8 * i..8 * i + 8].try_into().unwrap()))
}

fn bits(b: [u8; 16]) -> u128 {
    u128::from_le_bytes(b)
}

fn from_bits(v: u128) -> [u8; 16] {
    v.to_le_bytes()
}

fn zip4(f: impl Fn(u32, u32) -> u32, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let (a, b) = (l32(a), l32(b));
    b32(std::array::from_fn(|i| f(a[i], b[i])))
}

/// `sshl`/`ushl` lane: the shift is the signed low byte of the second operand.
fn shift_lane(v: u32, amount: u32, signed: bool) -> u32 {
    let s = amount as u8 as i8 as i32;
    if s >= 0 {
        if s >= 32 { 0 } else { v << s }
    } else if signed {
        ((v as i32) >> (-s).min(31)) as u32
    } else if -s >= 32 {
        0
    } else {
        v >> -s
    }
}

struct Prog {
    a: Asm,
    want: Vec<u8>,
}

impl Prog {
    /// Stores `v<reg>` to the next stack slot and records its expected contents.
    fn st(&mut self, reg: u8, want: [u8; 16]) {
        let off = self.want.len() as i64;
        self.a.str_fp(VecSize::Q, vr(reg), Addr::Off(SP, off)).unwrap();
        self.want.extend(want);
    }
}

#[test]
fn neon_runs() {
    let a_l: [u32; 4] = [0x8000_0001, 0xfedc_ba98, 0x1111_1111, 0x7fff_ffff];
    let b_l: [u32; 4] = [0x1234_5603, 0x89ab_cdfd, 0x0f0f_0fe0, 0x8000_001f];
    let f_l = [1.5f32, -2.25, 100.0, 0.5];
    let g_l = [2.0f32, 4.0, -0.5, 3.0];
    let idx: [u8; 16] = [0, 31, 16, 15, 32, 255, 1, 17, 5, 6, 7, 8, 9, 10, 11, 12];
    let (ab, bb, fb, gb) = (b32(a_l), b32(b_l), bf(f_l), bf(g_l));
    let mut data = Vec::new();
    for v in [ab, bb, fb, gb, idx] {
        data.extend(v);
    }

    let mut img = ImageBuilder::new(Machine::Aarch64, Options::release(Machine::Aarch64));
    let entry = img.declare(stub::RUNTIME_ENTRY);
    let sym = img.rodata("simd.data", DataObj::new(data, 16)).unwrap();
    img.add_start_stub(entry).unwrap();

    let mut p = Prog { a: Asm::new(), want: Vec::new() };
    p.a.addr_of_sym(x(9), sym, 0).unwrap();
    for (reg, off) in [(0, 0), (1, 16), (16, 32), (17, 48), (8, 64)] {
        p.a.ldr_fp(VecSize::Q, vr(reg), Addr::Off(x(9), off)).unwrap();
    }
    p.a.addsub_imm(AddSub::Sub, X, SP, SP, 0x1000).unwrap();
    p.a.mov_imm(W, x(10), 0xbeef).unwrap();

    // Integer three-same: operand order, wrapping, saturation, signedness.
    p.a.vbin(VBin::Sub, Arr::S4, vr(2), vr(0), vr(1)).unwrap();
    p.st(2, zip4(u32::wrapping_sub, ab, bb));
    p.a.vbin(VBin::Mul, Arr::S4, vr(2), vr(0), vr(1)).unwrap();
    p.st(2, zip4(u32::wrapping_mul, ab, bb));
    p.a.vbin(VBin::Umin, Arr::S4, vr(2), vr(0), vr(1)).unwrap();
    p.st(2, zip4(u32::min, ab, bb));
    p.a.vbin(VBin::UqAdd, Arr::B16, vr(2), vr(0), vr(1)).unwrap();
    p.st(
        2,
        from_bits((0..16).fold(0u128, |acc, i| acc | u128::from(ab[i].saturating_add(bb[i])) << (8 * i))),
    );
    p.a.vbin(VBin::SqSub, Arr::H8, vr(2), vr(0), vr(1)).unwrap();
    p.st(2, b16(std::array::from_fn(|i| (l16(ab)[i] as i16).saturating_sub(l16(bb)[i] as i16) as u16)));
    p.a.vbin(VBin::Sshl, Arr::S4, vr(2), vr(0), vr(1)).unwrap();
    p.st(2, zip4(|a, s| shift_lane(a, s, true), ab, bb));
    p.a.vbin(VBin::Ushl, Arr::S4, vr(2), vr(0), vr(1)).unwrap();
    p.st(2, zip4(|a, s| shift_lane(a, s, false), ab, bb));
    p.a.vbin(VBin::Cmgt, Arr::S4, vr(2), vr(0), vr(1)).unwrap();
    p.st(2, zip4(|a, b| if (a as i32) > (b as i32) { !0 } else { 0 }, ab, bb));
    p.a.vbin(VBin::Cmhi, Arr::S4, vr(2), vr(0), vr(1)).unwrap();
    p.st(2, zip4(|a, b| if a > b { !0 } else { 0 }, ab, bb));
    p.a.vbin(VBin::Cmtst, Arr::S4, vr(2), vr(0), vr(1)).unwrap();
    p.st(2, zip4(|a, b| if a & b != 0 { !0 } else { 0 }, ab, bb));

    // Bitwise select: the destination of `bsl` is the mask, of `bit`/`bif` the base value.
    let (ua, ub, ug) = (bits(ab), bits(bb), bits(gb));
    p.a.vmov(Arr::B16, vr(3), vr(1)).unwrap();
    p.a.vlogic(VLogic::Bsl, Arr::B16, vr(3), vr(0), vr(17)).unwrap();
    p.st(3, from_bits((ub & ua) | (!ub & ug)));
    p.a.vmov(Arr::B16, vr(3), vr(0)).unwrap();
    p.a.vlogic(VLogic::Bit, Arr::B16, vr(3), vr(1), vr(17)).unwrap();
    p.st(3, from_bits((ua & !ug) | (ub & ug)));
    p.a.vmov(Arr::B16, vr(3), vr(0)).unwrap();
    p.a.vlogic(VLogic::Bif, Arr::B16, vr(3), vr(1), vr(17)).unwrap();
    p.st(3, from_bits((ua & ug) | (ub & !ug)));
    p.a.vlogic(VLogic::Bic, Arr::B16, vr(3), vr(0), vr(1)).unwrap();
    p.st(3, from_bits(ua & !ub));
    p.a.vlogic(VLogic::Orn, Arr::B16, vr(3), vr(0), vr(1)).unwrap();
    p.st(3, from_bits(ua | !ub));

    // Integer unary.
    p.a.vint_un(VIntUn::Abs, Arr::S4, vr(2), vr(0)).unwrap();
    p.st(2, b32(a_l.map(|a| (a as i32).wrapping_abs() as u32)));
    p.a.vint_un(VIntUn::Neg, Arr::S4, vr(2), vr(0)).unwrap();
    p.st(2, b32(a_l.map(|a| (a as i32).wrapping_neg() as u32)));
    p.a.vint_un(VIntUn::Clz, Arr::S4, vr(2), vr(1)).unwrap();
    p.st(2, b32(b_l.map(u32::leading_zeros)));
    p.a.vint_un(VIntUn::Cnt, Arr::B16, vr(2), vr(0)).unwrap();
    p.st(2, from_bits((0..16).fold(0u128, |acc, i| acc | u128::from(ab[i].count_ones()) << (8 * i))));
    p.a.vrev(Rev::R32, Arr::B16, vr(2), vr(0)).unwrap();
    p.st(2, b32(a_l.map(u32::swap_bytes)));

    // Permutes and table lookup.
    let cat: Vec<u8> = ab.iter().chain(bb.iter()).copied().collect();
    p.a.vtbl(Tbl::Tbl, Arr::B16, vr(2), vr(0), 2, vr(8)).unwrap();
    p.st(2, std::array::from_fn(|i| if idx[i] < 32 { cat[usize::from(idx[i])] } else { 0 }));
    p.a.vperm(Perm::Zip1, Arr::H8, vr(2), vr(0), vr(1)).unwrap();
    let (ah, bh) = (l16(ab), l16(bb));
    p.st(2, b16([ah[0], bh[0], ah[1], bh[1], ah[2], bh[2], ah[3], bh[3]]));
    p.a.vperm(Perm::Uzp2, Arr::H8, vr(2), vr(0), vr(1)).unwrap();
    p.st(2, b16([ah[1], ah[3], ah[5], ah[7], bh[1], bh[3], bh[5], bh[7]]));
    p.a.vext(Arr::B16, vr(2), vr(0), vr(1), 5).unwrap();
    p.st(2, std::array::from_fn(|i| cat[i + 5]));

    // Widening, narrowing, multiplies.
    p.a.uxtl(Arr::H8, vr(2), vr(0)).unwrap();
    p.st(2, b32([4, 5, 6, 7].map(|i| u32::from(ah[i]))));
    p.a.sxtl(Arr::B8, vr(2), vr(0)).unwrap();
    p.st(2, b16(std::array::from_fn(|i| ab[i] as i8 as i16 as u16)));
    p.a.vmull(Sign::Signed, Arr::H8, vr(2), vr(0), vr(1)).unwrap();
    p.st(2, b32([4, 5, 6, 7].map(|i| (i32::from(ah[i] as i16) * i32::from(bh[i] as i16)) as u32)));
    p.a.vmull(Sign::Unsigned, Arr::S2, vr(2), vr(0), vr(1)).unwrap();
    p.st(2, b64([0, 1].map(|i| u64::from(a_l[i]) * u64::from(b_l[i]))));
    p.a.vshrn(false, Arr::H4, vr(2), vr(0), 8).unwrap();
    let lo64 = |h: [u16; 4]| b16([h[0], h[1], h[2], h[3], 0, 0, 0, 0]);
    p.st(2, lo64(a_l.map(|a| (a >> 8) as u16)));
    p.a.vnarrow(Narrow::Sqxtn, Arr::H4, vr(2), vr(0)).unwrap();
    p.st(2, lo64(a_l.map(|a| (a as i32).clamp(-32768, 32767) as i16 as u16)));
    p.a.vnarrow(Narrow::Sqxtun, Arr::H4, vr(2), vr(0)).unwrap();
    p.st(2, lo64(a_l.map(|a| (a as i32).clamp(0, 65535) as u16)));
    p.a.vnarrow(Narrow::Xtn, Arr::H4, vr(2), vr(0)).unwrap();
    p.a.vnarrow(Narrow::Xtn, Arr::H8, vr(2), vr(1)).unwrap();
    p.st(2, b16([a_l[0], a_l[1], a_l[2], a_l[3], b_l[0], b_l[1], b_l[2], b_l[3]].map(|v| v as u16)));

    // Shifts by immediate.
    p.a.vshift(VShift::Shl, Arr::S4, vr(2), vr(0), 4).unwrap();
    p.st(2, b32(a_l.map(|a| a << 4)));
    p.a.vshift(VShift::Ushr, Arr::S4, vr(2), vr(0), 28).unwrap();
    p.st(2, b32(a_l.map(|a| a >> 28)));
    p.a.vshift(VShift::Sshr, Arr::S4, vr(2), vr(0), 31).unwrap();
    p.st(2, b32(a_l.map(|a| ((a as i32) >> 31) as u32)));

    // Floating point.
    p.a.vmov(Arr::B16, vr(18), vr(17)).unwrap();
    p.a.vfp_bin(VFpBin::Fmla, Arr::S4, vr(18), vr(16), vr(17)).unwrap();
    p.st(18, bf(std::array::from_fn(|i| g_l[i] + f_l[i] * g_l[i])));
    p.a.vmov(Arr::B16, vr(18), vr(17)).unwrap();
    p.a.vfp_bin(VFpBin::Fmls, Arr::S4, vr(18), vr(16), vr(17)).unwrap();
    p.st(18, bf(std::array::from_fn(|i| g_l[i] - f_l[i] * g_l[i])));
    p.a.vfp_bin(VFpBin::Div, Arr::S4, vr(18), vr(16), vr(17)).unwrap();
    p.st(18, bf(std::array::from_fn(|i| f_l[i] / g_l[i])));
    p.a.vfp_bin(VFpBin::Sub, Arr::S4, vr(18), vr(16), vr(17)).unwrap();
    p.st(18, bf(std::array::from_fn(|i| f_l[i] - g_l[i])));
    p.a.vfp_un(VFpUn::Abs, Arr::S4, vr(18), vr(16)).unwrap();
    p.a.vfp_un(VFpUn::Sqrt, Arr::S4, vr(18), vr(18)).unwrap();
    p.st(18, bf(f_l.map(|f| f.abs().sqrt())));
    p.a.fcvt_v(FcvtRound::Zs, Arr::S4, vr(18), vr(16)).unwrap();
    p.st(18, b32(f_l.map(|f| f as i32 as u32)));
    p.a.scvtf_v(Arr::S4, vr(18), vr(0)).unwrap();
    p.st(18, bf(a_l.map(|a| a as i32 as f32)));
    p.a.vfp_un(VFpUn::RintP, Arr::S4, vr(18), vr(16)).unwrap();
    p.st(18, bf(f_l.map(f32::ceil)));
    p.a.vfp_bin(VFpBin::Cmgt, Arr::S4, vr(18), vr(16), vr(17)).unwrap();
    p.st(18, b32(std::array::from_fn(|i| if f_l[i] > g_l[i] { !0 } else { 0 })));
    p.a.vfcmp_zero(CmpZ::Ge, Arr::S4, vr(18), vr(16)).unwrap();
    p.st(18, b32(f_l.map(|f| if f >= 0.0 { !0 } else { 0 })));

    // Reductions write the scalar and clear the rest of the register.
    p.a.vacross(Across::Addv, Arr::S4, vr(2), vr(0)).unwrap();
    p.st(2, b32([a_l.iter().fold(0u32, |s, v| s.wrapping_add(*v)), 0, 0, 0]));
    p.a.vacross(Across::Smaxv, Arr::S4, vr(2), vr(0)).unwrap();
    p.st(2, b32([a_l.iter().map(|v| *v as i32).max().unwrap() as u32, 0, 0, 0]));
    p.a.vacross(Across::Uaddlv, Arr::S4, vr(2), vr(0)).unwrap();
    p.st(2, b64([a_l.iter().map(|v| u64::from(*v)).sum(), 0]));
    p.a.vacross(Across::Fmaxv, Arr::S4, vr(2), vr(16)).unwrap();
    p.st(2, bf([100.0, 0.0, 0.0, 0.0]));
    p.a.addp_scalar(vr(2), vr(0)).unwrap();
    p.st(2, b64([l64(ab)[0].wrapping_add(l64(ab)[1]), 0]));

    // Element moves, immediates.
    p.a.dup_gpr(Arr::H8, vr(2), x(10)).unwrap();
    p.st(2, b16([0xbeef; 8]));
    p.a.movi_2d(vr(2), 0).unwrap();
    p.a.ins_gpr(Lane::S, vr(2), 2, x(10)).unwrap();
    p.a.ins_elem(Lane::S, vr(2), 1, vr(0), 3).unwrap();
    p.st(2, b32([0, a_l[3], 0xbeef, 0]));
    p.a.dup_elem(Arr::S4, vr(2), vr(1), 2).unwrap();
    p.st(2, b32([b_l[2]; 4]));
    p.a.movi(Arr::B16, vr(2), 0xa5, ImmShift::Lsl(0)).unwrap();
    p.st(2, [0xa5; 16]);
    p.a.mvni(Arr::S4, vr(2), 0x12, ImmShift::Lsl(8)).unwrap();
    p.st(2, b32([!0x1200; 4]));
    p.a.movi(Arr::S4, vr(2), 0x34, ImmShift::Msl(8)).unwrap();
    p.st(2, b32([0x34ff; 4]));
    p.a.movi_2d(vr(2), 0xff00_ff00_0000_ffff).unwrap();
    p.st(2, b64([0xff00_ff00_0000_ffff; 2]));
    p.a.fmov_vec(Arr::D2, vr(2), -0.5).unwrap();
    p.st(2, b64([(-0.5f64).to_bits(); 2]));
    p.a.fmov_vec(Arr::S4, vr(2), 1.0).unwrap();
    p.st(2, bf([1.0; 4]));
    p.a.umov(Lane::D, x(11), vr(0), 1).unwrap();
    p.a.smov(Lane::S, X, x(12), vr(0), 0).unwrap();
    p.a.addsub_imm(AddSub::Add, X, x(14), SP, p.want.len() as u64).unwrap();
    p.a.stp(X, x(11), x(12), Addr::Off(x(14), 0)).unwrap();
    p.want.extend(b64([l64(ab)[1], a_l[0] as i32 as i64 as u64]));

    // Memory forms.
    p.a.ld1r(Arr::S4, vr(2), VecAddr::Base(x(9))).unwrap();
    p.st(2, b32([a_l[0]; 4]));
    p.a.mov(X, x(13), x(9)).unwrap();
    p.a.ld1(Arr::B16, vr(28), 2, VecAddr::PostImm(x(13))).unwrap();
    p.st(28, ab);
    p.st(29, bb);
    p.a.ld1(Arr::S4, vr(2), 1, VecAddr::Base(x(13))).unwrap();
    p.st(2, fb);
    let at = p.want.len() as u64;
    p.a.addsub_imm(AddSub::Add, X, x(14), SP, at).unwrap();
    p.a.stnp_fp(VecSize::Q, vr(0), vr(1), x(14), 0).unwrap();
    p.want.extend(ab);
    p.want.extend(bb);
    p.a.prfm(Prefetch { kind: PrefKind::Pld, level: PrefLevel::L1, stream: false }, Addr::Off(x(9), 0))
        .unwrap();
    p.a.prfm(Prefetch { kind: PrefKind::Pst, level: PrefLevel::L2, stream: true }, Addr::Off(SP, 0)).unwrap();

    // write(1, sp, len); exit(0).
    let len = p.want.len() as u64;
    assert!(len < 0x1000);
    p.a.mov(X, x(1), SP).unwrap();
    p.a.mov_imm(X, x(0), 1).unwrap();
    p.a.mov_imm(X, x(2), len).unwrap();
    p.a.mov_imm(X, x(8), 64).unwrap();
    p.a.svc(0);
    p.a.mov_imm(X, x(0), 0).unwrap();
    p.a.mov_imm(X, x(8), 94).unwrap();
    p.a.svc(0);
    img.define_func(entry, p.a.finish().unwrap()).unwrap();
    let img = img.link().unwrap();

    let path = tmp("neon");
    img.write_file(&path).unwrap();
    let mut cmd = if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        Command::new(&path)
    } else if have("qemu-aarch64") {
        let mut c = Command::new("qemu-aarch64");
        c.arg(&path);
        c
    } else {
        eprintln!("SKIP running AArch64 ELF: no qemu-aarch64 and not an AArch64 host");
        return;
    };
    let out = output_retry(&mut cmd);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(out.stdout.len(), p.want.len());
    for (i, (got, want)) in out.stdout.chunks(16).zip(p.want.chunks(16)).enumerate() {
        assert_eq!(got, want, "result {i} (stack offset {:#x})", i * 16);
    }
    eprintln!("simd run: {} result vectors", p.want.len() / 16);
}
