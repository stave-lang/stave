//! `__stave_fma32` / `__stave_fma64`: `a * b + c` with one rounding (round to nearest even) in
//! integer arithmetic, for targets without an FMA instruction.
//!
//! Convention: `a`, `b`, `c` in the low lanes of `xmm0`, `xmm1`, `xmm2`; the result in `xmm0`.
//! Clobbers `rax rcx rdx rsi rdi r8`–`r11` and `xmm0`–`xmm2`; uses no stack.
//!
//! A NaN, infinity, or zero among `a`, `b` gives `a * b + c` with the ordinary instructions (the
//! product is exact or special). With finite nonzero `a`, `b`: a NaN or infinity `c` is the
//! result (quieted; the exact product is finite even where its rounding overflows), a zero `c`
//! gives `a * b`.
//! Otherwise the exact product (106 bits) and `c` are aligned in 128 bits with a sticky bit,
//! added or subtracted, normalized, and rounded once to the format (the f32 entry widens its
//! operands to f64 first, exactly, and rounds the 128-bit sum to 24 bits).

use stave_obj::{ImageBuilder, SymbolId};

use super::Res;
use crate::enc::*;

const RAX: Gpr = Gpr::RAX;
const RCX: Gpr = Gpr::RCX;
const RDX: Gpr = Gpr::RDX;
const RSI: Gpr = Gpr::RSI;
const RDI: Gpr = Gpr::RDI;
const R8: Gpr = Gpr::R8;
const R9: Gpr = Gpr::R9;
const R10: Gpr = Gpr::R10;
const R11: Gpr = Gpr::R11;

/// The result format: bits of the stored mantissa, the exponent field of infinity.
#[derive(Clone, Copy)]
struct Format {
    prec: Prec,
    mant: u8,
    emax: i64,
}

const F32: Format = Format { prec: Prec::F32, mant: 23, emax: 255 };
const F64: Format = Format { prec: Prec::F64, mant: 52, emax: 2047 };

/// Defines both functions: `[f32, f64]`.
pub(crate) fn define(b: &mut ImageBuilder) -> Res<[SymbolId; 2]> {
    let f32_ = b.func("__stave_fma32", emit(F32)?.finish()?)?;
    let f64_ = b.func("__stave_fma64", emit(F64)?.finish()?)?;
    Ok([f32_, f64_])
}

fn x(n: u8) -> Xmm {
    Xmm::masked(n)
}

/// Splits the finite nonzero f64 bits in `bits` into the mantissa with its leading bit at bit 52
/// (left in `bits`) and the exponent `e` with value = mantissa * 2^(e - 1075). `one` holds 2^52.
/// Clobbers `rcx`.
fn decompose(a: &mut Asm, bits: Gpr, e: Gpr, one: Gpr) -> Result<(), EncError> {
    let (normal, done) = (a.new_label(), a.new_label());
    a.mov_rr(Size::Q, e, bits);
    a.shift_ri(ShiftOp::Shr, Size::Q, e, 52)?;
    a.alu_ri(AluOp::And, Size::D, e, 0x7ff)?;
    a.shift_ri(ShiftOp::Shl, Size::Q, bits, 12)?;
    a.shift_ri(ShiftOp::Shr, Size::Q, bits, 12)?;
    a.test_rr(Size::D, e, e);
    a.jcc(Cond::Ne, normal)?;
    // subnormal: the top bit p moves to bit 52, the exponent is 1 - (52 - p)
    a.bsr(Size::Q, RCX, bits)?;
    a.lea(Size::Q, e, Mem::base(RCX, -51))?;
    a.unary(UnaryOp::Neg, Size::D, RCX)?;
    a.alu_ri(AluOp::Add, Size::D, RCX, 52)?;
    a.shift_cl(ShiftOp::Shl, Size::Q, bits)?;
    a.jmp(done)?;
    a.bind(normal)?;
    a.alu_rr(AluOp::Or, Size::Q, bits, one);
    a.bind(done)?;
    Ok(())
}

/// `r = (r != 0) as u64`.
fn nonzero(a: &mut Asm, r: Gpr) -> Result<(), EncError> {
    a.test_rr(Size::Q, r, r);
    a.setcc(Cond::Ne, r);
    a.movzx(Size::D, r, Size::B, r)
}

fn emit(f: Format) -> Result<Asm, EncError> {
    let mut a = Asm::new();
    let single = f.prec == Prec::F32;
    let (special, cspecial, czero) = (a.new_label(), a.new_label(), a.new_label());
    // Operand classes from the raw bits: `bits << 1` is at least `top` for a NaN or infinity.
    let (s, top) = if single { (Size::D, 0xff00_0000u32 as i32 as i64) } else { (Size::Q, -(1i64 << 53)) };
    a.sse_movd_r(s, RAX, x(0))?;
    a.sse_movd_r(s, RSI, x(1))?;
    a.sse_movd_r(s, RDI, x(2))?;
    a.mov_ri(s, R11, top)?;
    for (r, naninf, zero) in [(RAX, special, special), (RSI, special, special), (RDI, cspecial, czero)] {
        a.lea(s, R8, Mem::index(r, r, Scale::S1, 0))?;
        a.alu_rr(AluOp::Cmp, s, R8, R11);
        a.jcc(Cond::Ae, naninf)?;
        a.test_rr(s, R8, R8);
        a.jcc(Cond::E, zero)?;
    }
    if single {
        for n in 0..3 {
            a.cvts2s(Prec::F32, x(n), x(n))?;
        }
        a.sse_movd_r(Size::Q, RAX, x(0))?;
        a.sse_movd_r(Size::Q, RSI, x(1))?;
        a.sse_movd_r(Size::Q, RDI, x(2))?;
    }
    // Mantissas ma (rax), mb (rsi), mc (rdi); exponents ea + eb (r9), ec (r11).
    a.mov_ri(Size::Q, R8, 1 << 52)?;
    decompose(&mut a, RAX, R9, R8)?;
    decompose(&mut a, RSI, R10, R8)?;
    a.alu_rr(AluOp::Add, Size::Q, R9, R10);
    decompose(&mut a, RDI, R11, R8)?;
    // P = ma * mb << 20 in rdx:rax, in [2^124, 2^126), value P * 2^(ea + eb - 2170).
    a.unary(UnaryOp::Mul, Size::Q, RSI)?;
    a.mov_rr(Size::Q, RCX, RAX);
    a.shift_ri(ShiftOp::Shr, Size::Q, RCX, 44)?;
    a.shift_ri(ShiftOp::Shl, Size::Q, RDX, 20)?;
    a.alu_rr(AluOp::Or, Size::Q, RDX, RCX);
    a.shift_ri(ShiftOp::Shl, Size::Q, RAX, 20)?;
    // C = mc << 73 in r8:rsi, in [2^125, 2^126), value C * 2^(ec - 1148).
    a.mov_rr(Size::Q, R8, RDI);
    a.shift_ri(ShiftOp::Shl, Size::Q, R8, 9)?;
    a.alu_rr(AluOp::Xor, Size::D, RSI, RSI);
    // d = exponent of P - exponent of C. The operand with the larger exponent stays in rdx:rax
    // (X), the other goes to r8:rsi (Y) and is shifted right by |d| into X's scale. r9 = the
    // exponent of X's unit, edi = 1 when X is C.
    let (cbig, shift) = (a.new_label(), a.new_label());
    a.mov_rr(Size::Q, R10, R9);
    a.alu_rr(AluOp::Sub, Size::Q, R10, R11);
    a.alu_ri(AluOp::Sub, Size::Q, R10, 1022)?;
    a.jcc(Cond::S, cbig)?;
    a.lea(Size::Q, R9, Mem::base(R9, -2170))?;
    a.alu_rr(AluOp::Xor, Size::D, RDI, RDI);
    a.mov_rr(Size::Q, RCX, R10);
    a.jmp(shift)?;
    a.bind(cbig)?;
    a.lea(Size::Q, R9, Mem::base(R11, -1148))?;
    a.mov_ri(Size::D, RDI, 1)?;
    a.mov_rr(Size::Q, RCX, R10);
    a.unary(UnaryOp::Neg, Size::Q, RCX)?;
    a.mov_rr(Size::Q, R11, RDX);
    a.mov_rr(Size::Q, RDX, R8);
    a.mov_rr(Size::Q, R8, R11);
    a.mov_rr(Size::Q, R11, RAX);
    a.mov_rr(Size::Q, RAX, RSI);
    a.mov_rr(Size::Q, RSI, R11);
    a.bind(shift)?;
    // Y >>= rcx; a lost nonzero bit sets bit 0 (sticky).
    let (mid, all, shifted) = (a.new_label(), a.new_label(), a.new_label());
    a.test_rr(Size::Q, RCX, RCX);
    a.jcc(Cond::E, shifted)?;
    a.alu_ri(AluOp::Cmp, Size::Q, RCX, 128)?;
    a.jcc(Cond::Ae, all)?;
    a.alu_ri(AluOp::Cmp, Size::Q, RCX, 64)?;
    a.jcc(Cond::Ae, mid)?;
    a.mov_rr(Size::Q, R10, RSI);
    a.mov_rr(Size::Q, R11, R8);
    a.shift_cl(ShiftOp::Shr, Size::Q, RSI)?;
    a.shift_cl(ShiftOp::Shr, Size::Q, R8)?;
    a.unary(UnaryOp::Neg, Size::D, RCX)?;
    a.alu_ri(AluOp::Add, Size::D, RCX, 64)?;
    a.shift_cl(ShiftOp::Shl, Size::Q, R11)?;
    a.alu_rr(AluOp::Or, Size::Q, RSI, R11);
    a.shift_cl(ShiftOp::Shl, Size::Q, R10)?;
    nonzero(&mut a, R10)?;
    a.alu_rr(AluOp::Or, Size::Q, RSI, R10);
    a.jmp(shifted)?;
    a.bind(mid)?;
    a.mov_rr(Size::Q, R10, RSI);
    a.mov_rr(Size::Q, R11, R8);
    a.mov_rr(Size::Q, RSI, R8);
    a.alu_ri(AluOp::Sub, Size::D, RCX, 64)?;
    a.shift_cl(ShiftOp::Shr, Size::Q, RSI)?;
    a.mov_rr(Size::Q, R8, RSI);
    a.shift_cl(ShiftOp::Shl, Size::Q, R8)?;
    a.alu_rr(AluOp::Xor, Size::Q, R11, R8);
    a.alu_rr(AluOp::Or, Size::Q, R10, R11);
    a.alu_rr(AluOp::Xor, Size::D, R8, R8);
    nonzero(&mut a, R10)?;
    a.alu_rr(AluOp::Or, Size::Q, RSI, R10);
    a.jmp(shifted)?;
    a.bind(all)?;
    a.alu_rr(AluOp::Xor, Size::D, R8, R8);
    a.mov_ri(Size::D, RSI, 1)?;
    a.bind(shifted)?;
    // Signs (bit 63): r10 = X's, r11 = Y's.
    let (ordered, sub, signed, zero) = (a.new_label(), a.new_label(), a.new_label(), a.new_label());
    a.sse_movd_r(Size::Q, R10, x(0))?;
    a.sse_movd_r(Size::Q, R11, x(1))?;
    a.alu_rr(AluOp::Xor, Size::Q, R10, R11);
    a.sse_movd_r(Size::Q, R11, x(2))?;
    a.test_rr(Size::D, RDI, RDI);
    a.jcc(Cond::E, ordered)?;
    a.mov_rr(Size::Q, RCX, R10);
    a.mov_rr(Size::Q, R10, R11);
    a.mov_rr(Size::Q, R11, RCX);
    a.bind(ordered)?;
    a.mov_rr(Size::Q, RCX, R10);
    a.alu_rr(AluOp::Xor, Size::Q, RCX, R11);
    a.jcc(Cond::S, sub)?;
    a.alu_rr(AluOp::Add, Size::Q, RAX, RSI);
    a.alu_rr(AluOp::Adc, Size::Q, RDX, R8);
    a.jmp(signed)?;
    a.bind(sub)?;
    a.alu_rr(AluOp::Sub, Size::Q, RAX, RSI);
    a.alu_rr(AluOp::Sbb, Size::Q, RDX, R8);
    a.jcc(Cond::NC, signed)?;
    // Y was larger: the sum is -(X - Y) with Y's sign
    a.unary(UnaryOp::Not, Size::Q, RDX)?;
    a.unary(UnaryOp::Neg, Size::Q, RAX)?;
    a.alu_ri(AluOp::Sbb, Size::Q, RDX, -1)?;
    a.mov_rr(Size::Q, R10, R11);
    a.bind(signed)?;
    a.mov_rr(Size::Q, RCX, RAX);
    a.alu_rr(AluOp::Or, Size::Q, RCX, RDX);
    a.jcc(Cond::E, zero)?;
    // Normalize: leading bit of rdx:rax to bit 127.
    let (high, norm) = (a.new_label(), a.new_label());
    a.test_rr(Size::Q, RDX, RDX);
    a.jcc(Cond::Ne, high)?;
    a.mov_rr(Size::Q, RDX, RAX);
    a.alu_rr(AluOp::Xor, Size::D, RAX, RAX);
    a.alu_ri(AluOp::Sub, Size::Q, R9, 64)?;
    a.bind(high)?;
    a.bsr(Size::Q, RCX, RDX)?;
    a.alu_ri(AluOp::Xor, Size::D, RCX, 63)?;
    a.alu_rr(AluOp::Sub, Size::Q, R9, RCX);
    a.test_rr(Size::D, RCX, RCX);
    a.jcc(Cond::E, norm)?;
    a.mov_rr(Size::Q, R8, RAX);
    a.shift_cl(ShiftOp::Shl, Size::Q, RDX)?;
    a.shift_cl(ShiftOp::Shl, Size::Q, RAX)?;
    a.unary(UnaryOp::Neg, Size::D, RCX)?;
    a.alu_ri(AluOp::Add, Size::D, RCX, 64)?;
    a.shift_cl(ShiftOp::Shr, Size::Q, R8)?;
    a.alu_rr(AluOp::Or, Size::Q, RDX, R8);
    a.bind(norm)?;
    // The sum is rdx:rax * 2^r9 with bit 127 set. Its exponent field E: the mantissa is the top
    // `mant + 1` bits, so E - (bias + mant) = r9 + 127 - mant.
    let mant = i64::from(f.mant);
    let bias = f.emax / 2;
    let (inf, tiny, normal, up, rounded, sign) =
        (a.new_label(), a.new_label(), a.new_label(), a.new_label(), a.new_label(), a.new_label());
    a.alu_ri(AluOp::Add, Size::Q, R9, 127 + bias)?;
    a.alu_ri(AluOp::Cmp, Size::Q, R9, f.emax)?;
    a.jcc(Cond::Ge, inf)?;
    // rcx = how far the mantissa's last bit is above bit 64; r8 = the result bits without it.
    a.mov_ri(Size::D, RCX, 63 - mant)?;
    a.lea(Size::Q, R8, Mem::base(R9, -1))?;
    a.shift_ri(ShiftOp::Shl, Size::Q, R8, f.mant)?;
    a.test_rr(Size::Q, R9, R9);
    a.jcc(Cond::G, normal)?;
    // subnormal: 1 - E more bits are dropped and the exponent field is 0
    a.mov_ri(Size::D, R8, 1)?;
    a.alu_rr(AluOp::Sub, Size::Q, R8, R9);
    a.alu_rr(AluOp::Add, Size::Q, RCX, R8);
    a.alu_rr(AluOp::Xor, Size::D, R8, R8);
    a.alu_ri(AluOp::Cmp, Size::Q, RCX, 65)?;
    a.jcc(Cond::Ae, tiny)?;
    a.bind(normal)?;
    // rsi = mantissa and round bit, rdi != 0 iff a lower bit is set
    a.alu_ri(AluOp::Sub, Size::D, RCX, 1)?;
    a.mov_rr(Size::Q, RSI, RDX);
    a.shift_cl(ShiftOp::Shr, Size::Q, RSI)?;
    a.mov_rr(Size::Q, RDI, RSI);
    a.shift_cl(ShiftOp::Shl, Size::Q, RDI)?;
    a.alu_rr(AluOp::Xor, Size::Q, RDI, RDX);
    a.alu_rr(AluOp::Or, Size::Q, RDI, RAX);
    a.mov_rr(Size::Q, RAX, RSI);
    a.shift_ri(ShiftOp::Shr, Size::Q, RAX, 1)?;
    a.test_ri(Size::D, RSI, 1)?;
    a.jcc(Cond::E, rounded)?;
    a.test_rr(Size::Q, RDI, RDI);
    a.jcc(Cond::Ne, up)?;
    a.test_ri(Size::D, RAX, 1)?;
    a.jcc(Cond::E, rounded)?;
    a.bind(up)?;
    a.alu_ri(AluOp::Add, Size::Q, RAX, 1)?;
    a.bind(rounded)?;
    // the mantissa's leading bit carries into the exponent field (and a round-up carry too)
    a.alu_rr(AluOp::Add, Size::Q, RAX, R8);
    a.jmp(sign)?;
    a.bind(inf)?;
    a.mov_ri(Size::Q, RAX, f.emax << f.mant)?;
    a.jmp(sign)?;
    a.bind(tiny)?;
    a.alu_rr(AluOp::Xor, Size::D, RAX, RAX);
    a.bind(sign)?;
    a.shift_ri(ShiftOp::Shr, Size::Q, R10, 63)?;
    a.shift_ri(ShiftOp::Shl, Size::Q, R10, if single { 31 } else { 63 })?;
    a.alu_rr(AluOp::Or, Size::Q, RAX, R10);
    a.sse_movd_x(if single { Size::D } else { Size::Q }, x(0), RAX)?;
    a.ret();
    a.bind(zero)?;
    a.logic_p(PLogic::Xor, f.prec, x(0), x(0))?;
    a.ret();
    a.bind(special)?;
    a.sse(SseOp::Mul, f.prec, x(0), x(1))?;
    a.sse(SseOp::Add, f.prec, x(0), x(2))?;
    a.ret();
    a.bind(cspecial)?;
    a.sse(SseOp::Add, f.prec, x(2), x(2))?;
    a.movaps(x(0), x(2));
    a.ret();
    a.bind(czero)?;
    a.sse(SseOp::Mul, f.prec, x(0), x(1))?;
    a.ret();
    Ok(a)
}
