//! Hand-written encodings checked against objdump.

use super::check;
use crate::enc::*;

fn x(n: u8) -> Reg {
    Reg::x(n).unwrap()
}

fn vr(n: u8) -> VReg {
    VReg::new(n).unwrap()
}

const W: Width = Width::W;
const X: Width = Width::X;
const SP: Reg = Reg::SP;
const ZR: Reg = Reg::ZR;

#[test]
fn moves() {
    check(
        |a| {
            a.mov(X, x(0), x(1))?;
            a.mov(X, x(0), SP)?;
            a.mov(X, SP, x(29))?;
            a.mov(W, x(3), x(4))?;
            a.movz(X, x(0), 0x1234, 0)?;
            a.movz(X, x(1), 0x1234, 16)?;
            a.movk(X, x(1), 0x5678, 48)?;
            a.movn(X, x(2), 0x1234, 0)?;
            a.movz(W, x(5), 0xffff, 16)?;
            a.mov_imm(X, x(0), 0)?;
            a.mov_imm(X, x(0), u64::MAX)?;
            a.mov_imm(W, x(0), 0xffff_ffff)?;
            a.mov_imm(X, x(0), 0x5555_5555_5555_5555)?;
            a.mov_imm(X, x(0), 0x1234_5678_9abc_def0)?;
            a.mov_imm(X, x(9), 0xffff_ffff_ffff_1234)?;
            a.mov_imm(W, x(9), 0x0001_0002)?;
            Ok(())
        },
        &[
            "mov x0, x1",
            "mov x0, sp",
            "mov sp, x29",
            "mov w3, w4",
            "mov x0, #0x1234",
            "mov x1, #0x12340000",
            "movk x1, #0x5678, lsl #48",
            "mov x2, #0xffffffffffffedcb",
            "mov w5, #0xffff0000",
            "mov x0, #0x0",
            "mov x0, #0xffffffffffffffff",
            "mov w0, #0xffffffff",
            "mov x0, #0x5555555555555555",
            "mov x0, #0xdef0",
            "movk x0, #0x9abc, lsl #16",
            "movk x0, #0x5678, lsl #32",
            "movk x0, #0x1234, lsl #48",
            "mov x9, #0xffffffffffff1234",
            "mov w9, #0x2",
            "movk w9, #0x1, lsl #16",
        ],
    );
}

#[test]
fn add_sub() {
    check(
        |a| {
            a.addsub_imm(AddSub::Add, X, x(0), x(1), 0x10)?;
            a.addsub_imm(AddSub::Add, X, x(0), x(1), 0x1000)?;
            a.addsub_imm(AddSub::Sub, X, SP, SP, 0x20)?;
            a.addsub_imm(AddSub::Adds, W, x(2), x(3), 1)?;
            a.addsub_imm(AddSub::Subs, X, x(2), SP, 4095)?;
            a.cmp_imm(X, x(0), 5)?;
            a.cmn_imm(W, x(0), 1)?;
            a.add(X, x(0), x(1), x(2))?;
            a.addsub_reg(AddSub::Add, X, x(0), x(1), x(2), Shift::Lsl, 3)?;
            a.addsub_reg(AddSub::Sub, W, x(0), x(1), x(2), Shift::Asr, 31)?;
            a.addsub_reg(AddSub::Subs, X, x(0), x(1), x(2), Shift::Lsr, 1)?;
            a.addsub_reg(AddSub::Adds, X, x(0), x(1), x(2), Shift::Lsl, 0)?;
            a.add(X, SP, SP, x(1))?;
            a.add(X, x(0), SP, x(1))?;
            a.addsub_reg(AddSub::Add, X, x(0), SP, x(1), Shift::Lsl, 3)?;
            a.sub(X, x(4), x(5), x(6))?;
            a.cmp(X, x(0), x(1))?;
            a.neg(X, x(0), x(1))?;
            Ok(())
        },
        &[
            "add x0, x1, #0x10",
            "add x0, x1, #0x1, lsl #12",
            "sub sp, sp, #0x20",
            "adds w2, w3, #0x1",
            "subs x2, sp, #0xfff",
            "cmp x0, #0x5",
            "cmn w0, #0x1",
            "add x0, x1, x2",
            "add x0, x1, x2, lsl #3",
            "sub w0, w1, w2, asr #31",
            "subs x0, x1, x2, lsr #1",
            "adds x0, x1, x2",
            "add sp, sp, x1",
            "add x0, sp, x1",
            "add x0, sp, x1, lsl #3",
            "sub x4, x5, x6",
            "cmp x0, x1",
            "neg x0, x1",
        ],
    );
}

/// `add/sub` with `sp` as `rd` or `rn` take the extended-register form (`uxtx #0`, bit 21 set),
/// where register 31 is `sp`; the shifted-register form would read it as `xzr`. Words from the
/// field layout `sf op S 01011 00 1 Rm option imm3 Rn Rd`; the last, without `sp`, is the
/// shifted-register form.
#[test]
fn addsub_reg_with_sp_is_extended() {
    let mut a = Asm::new();
    a.add(X, SP, SP, x(17)).unwrap();
    a.sub(X, SP, SP, x(17)).unwrap();
    a.add(X, x(0), SP, x(17)).unwrap();
    a.sub(X, x(0), SP, x(17)).unwrap();
    a.add(X, x(17), SP, x(17)).unwrap();
    a.add(X, x(0), x(1), x(17)).unwrap();
    let w: Vec<u32> = a.bytes().chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(w, [0x8B31_63FF, 0xCB31_63FF, 0x8B31_63E0, 0xCB31_63E0, 0x8B31_63F1, 0x8B11_0020]);
}

#[test]
fn logical_and_shifts() {
    check(
        |a| {
            a.logic_reg(Logic::And, X, x(0), x(1), x(2), Shift::Lsl, 0)?;
            a.logic_reg(Logic::Orr, W, x(0), x(1), x(2), Shift::Ror, 4)?;
            a.logic_reg(Logic::Eor, X, x(0), x(1), x(2), Shift::Lsr, 63)?;
            a.logic_reg(Logic::Ands, X, x(0), x(1), x(2), Shift::Lsl, 0)?;
            a.logic_reg(Logic::Bic, X, x(0), x(1), x(2), Shift::Lsl, 0)?;
            a.logic_reg(Logic::Orn, X, x(0), x(1), x(2), Shift::Lsl, 0)?;
            a.logic_reg(Logic::Eon, X, x(0), x(1), x(2), Shift::Lsl, 0)?;
            a.logic_reg(Logic::Bics, W, x(0), x(1), x(2), Shift::Lsl, 0)?;
            a.mvn(X, x(0), x(1))?;
            a.tst(X, x(0), x(1))?;
            a.logic_imm(Logic::And, X, x(0), x(1), 0xff)?;
            a.logic_imm(Logic::Orr, W, x(0), x(1), 0xffff_fff0)?;
            a.logic_imm(Logic::Eor, X, x(0), x(1), 0x8000_0000_0000_0000)?;
            a.logic_imm(Logic::And, X, SP, x(1), !0xfu64)?;
            a.tst_imm(W, x(0), 1)?;
            a.shift_reg(Shift::Lsl, X, x(0), x(1), x(2))?;
            a.shift_reg(Shift::Lsr, W, x(0), x(1), x(2))?;
            a.shift_reg(Shift::Asr, X, x(0), x(1), x(2))?;
            a.shift_reg(Shift::Ror, X, x(0), x(1), x(2))?;
            a.shift_imm(Shift::Lsl, X, x(0), x(1), 3)?;
            a.shift_imm(Shift::Lsr, X, x(0), x(1), 3)?;
            a.shift_imm(Shift::Asr, W, x(0), x(1), 31)?;
            a.shift_imm(Shift::Ror, X, x(0), x(1), 8)?;
            a.shift_imm(Shift::Lsl, W, x(0), x(1), 0)?;
            a.sxt(X, x(0), x(1), Ext::B)?;
            a.sxt(X, x(0), x(1), Ext::H)?;
            a.sxt(X, x(0), x(1), Ext::W)?;
            a.sxt(W, x(0), x(1), Ext::B)?;
            a.uxt(x(0), x(1), Ext::B)?;
            a.uxt(x(0), x(1), Ext::H)?;
            a.ubfm(X, x(0), x(1), 4, 11)?;
            a.sbfm(X, x(0), x(1), 4, 11)?;
            Ok(())
        },
        &[
            "and x0, x1, x2",
            "orr w0, w1, w2, ror #4",
            "eor x0, x1, x2, lsr #63",
            "ands x0, x1, x2",
            "bic x0, x1, x2",
            "orn x0, x1, x2",
            "eon x0, x1, x2",
            "bics w0, w1, w2",
            "mvn x0, x1",
            "tst x0, x1",
            "and x0, x1, #0xff",
            "orr w0, w1, #0xfffffff0",
            "eor x0, x1, #0x8000000000000000",
            "and sp, x1, #0xfffffffffffffff0",
            "tst w0, #0x1",
            "lsl x0, x1, x2",
            "lsr w0, w1, w2",
            "asr x0, x1, x2",
            "ror x0, x1, x2",
            "lsl x0, x1, #3",
            "lsr x0, x1, #3",
            "asr w0, w1, #31",
            "ror x0, x1, #8",
            "lsr w0, w1, #0", // ubfm #0, #31: objdump picks the lsr alias
            "sxtb x0, w1",
            "sxth x0, w1",
            "sxtw x0, w1",
            "sxtb w0, w1",
            "uxtb w0, w1",
            "uxth w0, w1",
            "ubfx x0, x1, #4, #8",
            "sbfx x0, x1, #4, #8",
        ],
    );
}

#[test]
fn mul_div_cond() {
    check(
        |a| {
            a.mul(X, x(0), x(1), x(2))?;
            a.madd(X, x(0), x(1), x(2), x(3))?;
            a.msub(W, x(0), x(1), x(2), x(3))?;
            a.smulh(x(0), x(1), x(2))?;
            a.umulh(x(0), x(1), x(2))?;
            a.smull(x(0), x(1), x(2))?;
            a.umull(x(0), x(1), x(2))?;
            a.udiv(X, x(0), x(1), x(2))?;
            a.sdiv(W, x(0), x(1), x(2))?;
            a.csel(X, x(0), x(1), x(2), Cond::Eq)?;
            a.csinc(X, x(0), x(1), x(2), Cond::Lt)?;
            a.csinv(W, x(0), x(1), x(2), Cond::Hi)?;
            a.csneg(X, x(0), x(1), x(2), Cond::Ge)?;
            a.cset(X, x(0), Cond::Ne)?;
            a.cset(W, x(3), Cond::Vs)?;
            a.csetm(X, x(0), Cond::Gt)?;
            a.ccmp(X, x(0), x(1), 0, Cond::Ne)?;
            a.ccmp_imm(W, x(0), 31, 4, Cond::Eq)?;
            a.ccmn(X, x(0), x(1), 15, Cond::Le)?;
            Ok(())
        },
        &[
            "mul x0, x1, x2",
            "madd x0, x1, x2, x3",
            "msub w0, w1, w2, w3",
            "smulh x0, x1, x2",
            "umulh x0, x1, x2",
            "smull x0, w1, w2",
            "umull x0, w1, w2",
            "udiv x0, x1, x2",
            "sdiv w0, w1, w2",
            "csel x0, x1, x2, eq",
            "csinc x0, x1, x2, lt",
            "csinv w0, w1, w2, hi",
            "csneg x0, x1, x2, ge",
            "cset x0, ne",
            "cset w3, vs",
            "csetm x0, gt",
            "ccmp x0, x1, #0x0, ne",
            "ccmp w0, #0x1f, #0x4, eq",
            "ccmn x0, x1, #0xf, le",
        ],
    );
}

const CC: [&str; 14] = ["eq", "ne", "cs", "cc", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le"];

#[test]
fn branches() {
    let mut want: Vec<String> = Vec::new();
    let mut a = Asm::new();
    let top = a.new_label();
    let end = a.new_label();
    a.bind(top).unwrap();
    for (i, n) in CC.iter().enumerate() {
        a.b_cond(Cond::ALL[i], top).unwrap();
        want.push(format!("b.{n} 0x0"));
    }
    a.b(top).unwrap();
    a.bl(top).unwrap();
    a.cbz(X, x(0), top).unwrap();
    a.cbnz(W, x(1), top).unwrap();
    a.tbz(x(2), 3, top).unwrap();
    a.tbnz(x(2), 63, top).unwrap();
    a.b(end).unwrap();
    a.b_cond(Cond::Vs, end).unwrap();
    a.adr(x(5), end).unwrap();
    a.adr(x(6), top).unwrap();
    a.br(x(16)).unwrap();
    a.blr(x(16)).unwrap();
    a.ret();
    a.ret_reg(x(1)).unwrap();
    a.bind(end).unwrap();
    a.svc(0);
    a.brk(0);
    a.brk(0x3e8);
    a.udf(0);
    a.nop();
    let e = 4 * (CC.len() + 14);
    for s in [
        "b 0x0".to_string(),
        "bl 0x0".into(),
        "cbz x0, 0x0".into(),
        "cbnz w1, 0x0".into(),
        "tbz w2, #3, 0x0".into(),
        "tbnz x2, #63, 0x0".into(),
        format!("b {e:#x}"),
        format!("b.vs {e:#x}"),
        format!("adr x5, {e:#x}"),
        "adr x6, 0x0".into(),
        "br x16".into(),
        "blr x16".into(),
        "ret".into(),
        "ret x1".into(),
        "svc #0x0".into(),
        "brk #0x0".into(),
        "brk #0x3e8".into(),
        "udf #0".into(),
        "nop".into(),
    ] {
        want.push(s);
    }
    super::check_bytes(&a.finish().unwrap().code, &want);
    for c in Cond::ALL {
        assert_eq!(c.negate().negate(), c);
    }
}

#[test]
fn loads_stores() {
    check(
        |a| {
            a.ldr(MemSize::X, x(0), Addr::Off(x(1), 0))?;
            a.ldr(MemSize::X, x(0), Addr::Off(x(1), 8))?;
            a.ldr(MemSize::W, x(0), Addr::Off(SP, 16))?;
            a.ldr(MemSize::B, x(0), Addr::Off(x(1), 1))?;
            a.ldr(MemSize::H, x(0), Addr::Off(x(1), 4094))?;
            a.ldr(MemSize::X, x(0), Addr::Off(x(1), 32760))?;
            a.ldr(MemSize::X, x(0), Addr::Off(x(1), -8))?;
            a.ldr(MemSize::X, x(0), Addr::Off(x(1), 3))?;
            a.ldr(MemSize::X, x(0), Addr::Pre(x(1), 8))?;
            a.ldr(MemSize::X, x(0), Addr::Post(x(1), -256))?;
            a.ldr(MemSize::X, x(0), Addr::Idx { base: x(1), index: x(2), ext: IdxExt::Lsl, scaled: false })?;
            a.ldr(MemSize::X, x(0), Addr::Idx { base: x(1), index: x(2), ext: IdxExt::Lsl, scaled: true })?;
            a.ldr(MemSize::W, x(0), Addr::Idx { base: x(1), index: x(2), ext: IdxExt::Uxtw, scaled: true })?;
            a.ldr(MemSize::B, x(0), Addr::Idx { base: x(1), index: x(2), ext: IdxExt::Sxtw, scaled: false })?;
            a.ldr(MemSize::H, x(0), Addr::Idx { base: SP, index: x(2), ext: IdxExt::Sxtx, scaled: true })?;
            a.ldrs(MemSize::B, X, x(0), Addr::Off(x(1), 0))?;
            a.ldrs(MemSize::H, W, x(0), Addr::Off(x(1), 2))?;
            a.ldrs(MemSize::W, X, x(0), Addr::Off(x(1), 4))?;
            a.ldrs(MemSize::W, X, x(0), Addr::Off(x(1), -4))?;
            a.str(MemSize::X, x(0), Addr::Off(SP, 8))?;
            a.str(MemSize::B, ZR, Addr::Off(x(1), 0))?;
            a.str(MemSize::H, x(3), Addr::Off(x(1), -2))?;
            a.str(MemSize::W, x(3), Addr::Pre(SP, -16))?;
            a.str(MemSize::X, x(3), Addr::Post(x(4), 16))?;
            a.ldr_fp(VecSize::D, vr(0), Addr::Off(x(0), 8))?;
            a.str_fp(VecSize::S, vr(1), Addr::Off(SP, 4))?;
            a.ldr_fp(VecSize::Q, vr(2), Addr::Off(x(0), 16))?;
            a.str_fp(VecSize::Q, vr(31), Addr::Pre(SP, -32))?;
            a.ldr_fp(VecSize::D, vr(0), Addr::Off(x(0), -8))?;
            a.ldr_fp(
                VecSize::D,
                vr(3),
                Addr::Idx { base: x(0), index: x(1), ext: IdxExt::Lsl, scaled: true },
            )?;
            a.stp(X, Reg::FP, Reg::LR, Addr::Pre(SP, -16))?;
            a.ldp(X, Reg::FP, Reg::LR, Addr::Post(SP, 16))?;
            a.stp(X, x(0), x(1), Addr::Off(SP, 16))?;
            a.ldp(W, x(0), x(1), Addr::Off(x(2), 0))?;
            a.stp(W, x(0), x(1), Addr::Off(x(2), -256))?;
            a.stp_fp(VecSize::D, vr(8), vr(9), Addr::Pre(SP, -16))?;
            a.ldp_fp(VecSize::Q, vr(0), vr(1), Addr::Off(x(0), 32))?;
            a.ldp_fp(VecSize::S, vr(0), vr(1), Addr::Post(x(0), 8))?;
            Ok(())
        },
        &[
            "ldr x0, [x1]",
            "ldr x0, [x1, #8]",
            "ldr w0, [sp, #16]",
            "ldrb w0, [x1, #1]",
            "ldrh w0, [x1, #4094]",
            "ldr x0, [x1, #32760]",
            "ldur x0, [x1, #-8]",
            "ldur x0, [x1, #3]",
            "ldr x0, [x1, #8]!",
            "ldr x0, [x1], #-256",
            "ldr x0, [x1, x2]",
            "ldr x0, [x1, x2, lsl #3]",
            "ldr w0, [x1, w2, uxtw #2]",
            "ldrb w0, [x1, w2, sxtw]",
            "ldrh w0, [sp, x2, sxtx #1]",
            "ldrsb x0, [x1]",
            "ldrsh w0, [x1, #2]",
            "ldrsw x0, [x1, #4]",
            "ldursw x0, [x1, #-4]",
            "str x0, [sp, #8]",
            "strb wzr, [x1]",
            "sturh w3, [x1, #-2]",
            "str w3, [sp, #-16]!",
            "str x3, [x4], #16",
            "ldr d0, [x0, #8]",
            "str s1, [sp, #4]",
            "ldr q2, [x0, #16]",
            "str q31, [sp, #-32]!",
            "ldur d0, [x0, #-8]",
            "ldr d3, [x0, x1, lsl #3]",
            "stp x29, x30, [sp, #-16]!",
            "ldp x29, x30, [sp], #16",
            "stp x0, x1, [sp, #16]",
            "ldp w0, w1, [x2]",
            "stp w0, w1, [x2, #-256]",
            "stp d8, d9, [sp, #-16]!",
            "ldp q0, q1, [x0, #32]",
            "ldp s0, s1, [x0], #8",
        ],
    );
}

#[test]
fn symbol_relocs() {
    use stave_obj::{RelocKind, SymbolId};
    let s = SymbolId(3);
    let mut a = Asm::new();
    a.bl_sym(s);
    a.b_sym(s);
    a.addr_of_sym(x(0), s, 8).unwrap();
    a.adrp_sym(x(1), s, 0).unwrap();
    a.ldr_lo12_sym(MemSize::X, x(2), x(1), s, 0).unwrap();
    let f = a.finish().unwrap();
    let r: Vec<_> = f.relocs.iter().map(|r| (r.offset, r.kind, r.addend)).collect();
    assert_eq!(
        r,
        [
            (0, RelocKind::A64Branch26, 0),
            (4, RelocKind::A64Branch26, 0),
            (8, RelocKind::A64AdrPage21, 8),
            (12, RelocKind::A64AddLo12, 8),
            (16, RelocKind::A64AdrPage21, 0),
            (20, RelocKind::A64LdstLo12 { shift: 3 }, 0),
        ]
    );
    super::check_bytes(
        &f.code,
        &["bl 0x0", "b 0x4", "adrp x0, 0x0", "add x0, x0, #0x0", "adrp x1, 0x0", "ldr x2, [x1]"]
            .map(String::from),
    );
}

#[test]
fn scalar_fp() {
    check(
        |a| {
            a.fmov(Fp::D, vr(0), vr(1));
            a.fmov(Fp::S, vr(2), vr(3));
            a.fmov_to_gpr(Fp::D, x(0), vr(1))?;
            a.fmov_to_gpr(Fp::S, x(0), vr(1))?;
            a.fmov_from_gpr(Fp::D, vr(0), x(1))?;
            a.fmov_from_gpr(Fp::S, vr(0), ZR)?;
            a.fmov_imm(Fp::D, vr(0), 1.0)?;
            a.fmov_imm(Fp::S, vr(1), -0.5)?;
            a.fmov_imm(Fp::D, vr(2), 31.0)?;
            a.fp_bin(FpBin::Add, Fp::D, vr(0), vr(1), vr(2));
            a.fp_bin(FpBin::Sub, Fp::S, vr(0), vr(1), vr(2));
            a.fp_bin(FpBin::Mul, Fp::D, vr(0), vr(1), vr(2));
            a.fp_bin(FpBin::Div, Fp::D, vr(31), vr(30), vr(29));
            a.fp_bin(FpBin::Max, Fp::D, vr(0), vr(1), vr(2));
            a.fp_bin(FpBin::Min, Fp::S, vr(0), vr(1), vr(2));
            a.fp_bin(FpBin::MaxNm, Fp::D, vr(0), vr(1), vr(2));
            a.fp_bin(FpBin::MinNm, Fp::D, vr(0), vr(1), vr(2));
            a.fp_bin(FpBin::Nmul, Fp::D, vr(0), vr(1), vr(2));
            a.fp_un(FpUn::Abs, Fp::S, vr(0), vr(1));
            a.fp_un(FpUn::Neg, Fp::D, vr(0), vr(1));
            a.fp_un(FpUn::Sqrt, Fp::D, vr(0), vr(1));
            a.fp_un(FpUn::RintN, Fp::D, vr(0), vr(1));
            a.fp_un(FpUn::RintP, Fp::D, vr(0), vr(1));
            a.fp_un(FpUn::RintM, Fp::D, vr(0), vr(1));
            a.fp_un(FpUn::RintZ, Fp::D, vr(0), vr(1));
            a.fp_un(FpUn::RintA, Fp::D, vr(0), vr(1));
            a.fp_un(FpUn::RintX, Fp::S, vr(0), vr(1));
            a.fp_un(FpUn::RintI, Fp::S, vr(0), vr(1));
            a.fcvt(Fp::D, vr(0), Fp::S, vr(1))?;
            a.fcvt(Fp::S, vr(0), Fp::D, vr(1))?;
            a.fcmp(Fp::D, vr(0), vr(1));
            a.fcmp_zero(Fp::S, vr(0));
            a.fcmpe(Fp::D, vr(0), vr(1));
            a.fcsel(Fp::D, vr(0), vr(1), vr(2), Cond::Gt);
            a.fma(FpFma::Madd, Fp::D, vr(0), vr(1), vr(2), vr(3));
            a.fma(FpFma::Msub, Fp::D, vr(0), vr(1), vr(2), vr(3));
            a.fma(FpFma::Nmadd, Fp::S, vr(0), vr(1), vr(2), vr(3));
            a.fma(FpFma::Nmsub, Fp::S, vr(0), vr(1), vr(2), vr(3));
            a.scvtf(Fp::D, vr(0), X, x(1))?;
            a.ucvtf(Fp::S, vr(0), W, x(1))?;
            a.scvtf(Fp::S, vr(0), X, x(1))?;
            a.fcvtzs(X, x(0), Fp::D, vr(1))?;
            a.fcvtzu(W, x(0), Fp::S, vr(1))?;
            a.fcvt_int(FcvtRound::Ns, X, x(0), Fp::D, vr(1))?;
            a.fcvt_int(FcvtRound::Ms, X, x(0), Fp::D, vr(1))?;
            a.fcvt_int(FcvtRound::Ps, X, x(0), Fp::D, vr(1))?;
            a.fcvt_int(FcvtRound::As, X, x(0), Fp::D, vr(1))?;
            a.fcvt_int(FcvtRound::Au, W, x(0), Fp::D, vr(1))?;
            Ok(())
        },
        &[
            "fmov d0, d1",
            "fmov s2, s3",
            "fmov x0, d1",
            "fmov w0, s1",
            "fmov d0, x1",
            "fmov s0, wzr",
            "fmov d0, #1.000000000000000000e+00",
            "fmov s1, #-5.000000000000000000e-01",
            "fmov d2, #3.100000000000000000e+01",
            "fadd d0, d1, d2",
            "fsub s0, s1, s2",
            "fmul d0, d1, d2",
            "fdiv d31, d30, d29",
            "fmax d0, d1, d2",
            "fmin s0, s1, s2",
            "fmaxnm d0, d1, d2",
            "fminnm d0, d1, d2",
            "fnmul d0, d1, d2",
            "fabs s0, s1",
            "fneg d0, d1",
            "fsqrt d0, d1",
            "frintn d0, d1",
            "frintp d0, d1",
            "frintm d0, d1",
            "frintz d0, d1",
            "frinta d0, d1",
            "frintx s0, s1",
            "frinti s0, s1",
            "fcvt d0, s1",
            "fcvt s0, d1",
            "fcmp d0, d1",
            "fcmp s0, #0.0",
            "fcmpe d0, d1",
            "fcsel d0, d1, d2, gt",
            "fmadd d0, d1, d2, d3",
            "fmsub d0, d1, d2, d3",
            "fnmadd s0, s1, s2, s3",
            "fnmsub s0, s1, s2, s3",
            "scvtf d0, x1",
            "ucvtf s0, w1",
            "scvtf s0, x1",
            "fcvtzs x0, d1",
            "fcvtzu w0, s1",
            "fcvtns x0, d1",
            "fcvtms x0, d1",
            "fcvtps x0, d1",
            "fcvtas x0, d1",
            "fcvtau w0, d1",
        ],
    );
}

#[test]
fn overflow_trap_sequence() {
    // checked add (signed: b.vs; unsigned: b.cs) and checked unsigned multiply (umulh != 0)
    check(
        |a| {
            let trap = a.new_label();
            a.addsub_reg(AddSub::Adds, X, x(0), x(0), x(1), Shift::Lsl, 0)?; // 0
            a.b_cond(Cond::Vs, trap)?; // 4
            a.addsub_reg(AddSub::Adds, W, x(0), x(0), x(1), Shift::Lsl, 0)?; // 8
            a.b_cond(Cond::Hs, trap)?; // 12
            a.umulh(x(2), x(0), x(1))?; // 16
            a.mul(X, x(0), x(0), x(1))?; // 20
            a.cbnz(X, x(2), trap)?; // 24
            a.ret(); // 28
            a.bind(trap)?; // 32
            a.brk(1);
            Ok(())
        },
        &[
            "adds x0, x0, x1",
            "b.vs 0x20",
            "adds w0, w0, w1",
            "b.cs 0x20",
            "umulh x2, x0, x1",
            "mul x0, x0, x1",
            "cbnz x2, 0x20",
            "ret",
            "brk #0x1",
        ],
    );
}

#[test]
fn invalid_operands() {
    let mut a = Asm::new();
    // sp/zr misuse
    assert!(a.add(X, x(0), x(1), SP).is_err(), "rm cannot be sp");
    assert!(a.addsub_imm(AddSub::Add, X, ZR, x(1), 1).is_err(), "add imm rd is sp-or-reg");
    assert!(a.addsub_imm(AddSub::Adds, X, SP, x(1), 1).is_err(), "adds rd is zr-or-reg");
    assert!(a.addsub_imm(AddSub::Add, X, x(0), ZR, 1).is_err());
    assert!(a.logic_reg(Logic::And, X, SP, x(1), x(2), Shift::Lsl, 0).is_err());
    assert!(a.logic_imm(Logic::Ands, X, SP, x(1), 0xff).is_err());
    assert!(a.mul(X, SP, x(1), x(2)).is_err());
    assert!(a.ldr(MemSize::X, SP, Addr::Off(x(1), 0)).is_err(), "rt cannot be sp");
    assert!(a.ldr(MemSize::X, x(0), Addr::Off(ZR, 0)).is_err(), "base cannot be zr");
    assert!(a.br(ZR).is_err() && a.blr(SP).is_err());
    assert!(a.addr_of_sym(SP, stave_obj::SymbolId(0), 0).is_err());
    // immediates
    assert!(a.addsub_imm(AddSub::Add, X, x(0), x(1), 4097).is_err());
    assert!(a.addsub_imm(AddSub::Add, X, x(0), x(1), 1 << 24).is_err());
    assert!(a.logic_imm(Logic::And, X, x(0), x(1), 0).is_err());
    assert!(a.logic_imm(Logic::And, X, x(0), x(1), 0x1234).is_err());
    assert!(a.logic_imm(Logic::Bic, X, x(0), x(1), 0xff).is_err());
    assert!(a.logic_imm(Logic::And, W, x(0), x(1), 0x1_0000_0000).is_err());
    assert!(a.movz(X, x(0), 0x1_0000, 0).is_err());
    assert!(a.movz(W, x(0), 1, 32).is_err());
    assert!(a.movz(X, x(0), 1, 8).is_err());
    assert!(a.mov_imm(W, x(0), 0x1_0000_0000).is_err());
    assert!(a.shift_imm(Shift::Lsl, W, x(0), x(1), 32).is_err());
    assert!(a.addsub_reg(AddSub::Add, X, x(0), x(1), x(2), Shift::Ror, 1).is_err());
    assert!(a.addsub_reg(AddSub::Add, X, SP, x(1), x(2), Shift::Lsl, 5).is_err());
    assert!(a.ldr(MemSize::X, x(0), Addr::Off(x(1), 32768)).is_err());
    assert!(a.ldr(MemSize::X, x(0), Addr::Off(x(1), -257)).is_err());
    assert!(a.ldr(MemSize::X, x(0), Addr::Pre(x(1), 256)).is_err());
    assert!(a.ldr(MemSize::X, x(1), Addr::Pre(x(1), 8)).is_err(), "writeback to the transfer register");
    assert!(a.ldrs(MemSize::W, W, x(0), Addr::Off(x(1), 0)).is_err());
    assert!(a.ldrs(MemSize::X, X, x(0), Addr::Off(x(1), 0)).is_err());
    assert!(a.ldp(X, x(0), x(0), Addr::Off(x(1), 0)).is_err());
    assert!(a.ldp(X, x(0), x(1), Addr::Off(x(2), 4)).is_err(), "misaligned pair offset");
    assert!(a.ldp(X, x(0), x(1), Addr::Off(x(2), 512)).is_err());
    assert!(a.stp(X, x(0), x(1), Addr::Pre(x(0), -16)).is_err());
    assert!(
        a.ldp(X, x(0), x(1), Addr::Idx { base: x(2), index: x(3), ext: IdxExt::Lsl, scaled: false }).is_err()
    );
    assert!(a.ccmp(X, x(0), x(1), 16, Cond::Eq).is_err());
    assert!(a.ccmp_imm(X, x(0), 32, 0, Cond::Eq).is_err());
    assert!(a.cset(X, x(0), Cond::Al).is_err());
    assert!(a.tbz(x(0), 64, Label(0)).is_err());
    assert!(a.fmov_imm(Fp::D, vr(0), 0.1).is_err());
    assert!(a.fmov_imm(Fp::D, vr(0), f64::NAN).is_err());
    assert!(a.fcvt(Fp::D, vr(0), Fp::D, vr(1)).is_err());
    assert!(a.sxt(W, x(0), x(1), Ext::W).is_err());
    assert_eq!(a.offset(), 0, "failed instructions emit nothing");
    // labels and ranges
    let mut b = Asm::new();
    let l = b.new_label();
    assert!(b.b(Label(9)).is_err());
    b.tbz(x(0), 0, l).unwrap();
    for _ in 0..8192 {
        b.nop();
    }
    b.bind(l).unwrap();
    assert_eq!(b.finish().unwrap_err(), EncError::BranchRange(l), "tbz reaches ±32 KiB");
    let mut c = Asm::new();
    let m = c.new_label();
    c.b(m).unwrap();
    assert_eq!(c.finish().unwrap_err(), EncError::LabelUnbound(m));
    assert!(Reg::x(31).is_none() && VReg::new(32).is_none());
}

#[test]
fn bit_counts() {
    check(
        |a| {
            a.clz(X, x(0), x(1))?;
            a.clz(W, x(2), x(3))?;
            a.rbit(X, x(4), x(5))?;
            a.rbit(W, x(6), x(7))?;
            a.cnt_8b(vr(31), vr(30));
            a.addv_8b(vr(1), vr(2));
            Ok(())
        },
        &["clz x0, x1", "clz w2, w3", "rbit x4, x5", "rbit w6, w7", "cnt v31.8b, v30.8b", "addv b1, v2.8b"],
    );
}

#[test]
fn long_branches() {
    let mut a = Asm::new();
    a.set_long_branches(true);
    let l = a.new_label();
    a.b_cond(Cond::Eq, l).unwrap();
    a.cbz(X, x(0), l).unwrap();
    a.cbnz(W, x(1), l).unwrap();
    a.tbz(x(2), 3, l).unwrap();
    a.tbnz(x(2), 40, l).unwrap();
    a.b_cond(Cond::Al, l).unwrap();
    let far = 300_000;
    for _ in 0..far {
        a.nop();
    }
    a.bind(l).unwrap();
    let code = a.finish().unwrap().code;
    let w = |i: usize| u32::from_le_bytes([code[4 * i], code[4 * i + 1], code[4 * i + 2], code[4 * i + 3]]);
    let target = 11 + far;
    // each: the inverted branch over one word, then `b label`
    let inv = [0x5400_0041, 0xb500_0040, 0x3400_0041, 0x3718_0042, 0xb640_0042];
    for (k, &want) in inv.iter().enumerate() {
        assert_eq!(w(2 * k), want, "inverted branch {k}");
        let b = w(2 * k + 1);
        assert_eq!(b & 0xfc00_0000, 0x1400_0000, "b after branch {k}");
        assert_eq!((b & 0x03ff_ffff) as usize, target - (2 * k + 1), "b offset {k}");
    }
    assert_eq!(w(10), 0x1400_0000 | (1 + far as u32), "b.al is a plain b");
    // the short form does not reach
    let mut b = Asm::new();
    let l = b.new_label();
    b.b_cond(Cond::Eq, l).unwrap();
    for _ in 0..far {
        b.nop();
    }
    b.bind(l).unwrap();
    assert_eq!(b.finish().unwrap_err(), EncError::BranchRange(l));
}

/// The instructions of a branch-free `select`: `cmp`, `csel` on X and W registers, `fcsel` on S
/// and D, the post-indexed word loop of a long aggregate with its `cbnz` counter, and the narrow
/// tail loads and stores.
#[test]
fn select_sequences() {
    check(
        |a| {
            a.cmp_imm(W, x(3), 0)?;
            a.csel(X, x(15), x(15), x(16), Cond::Ne)?;
            a.csel(W, x(15), x(15), x(16), Cond::Ne)?;
            a.fcsel(Fp::S, vr(0), vr(1), vr(2), Cond::Ne);
            a.fcsel(Fp::D, vr(3), vr(4), vr(5), Cond::Ne);
            let top = a.new_label();
            a.bind(top)?;
            a.ldr(MemSize::X, x(15), Addr::Post(x(9), 8))?;
            a.ldr(MemSize::X, x(16), Addr::Post(x(10), 8))?;
            a.csel(X, x(15), x(15), x(16), Cond::Ne)?;
            a.str(MemSize::X, x(15), Addr::Post(x(11), 8))?;
            a.addsub_imm(AddSub::Sub, X, x(17), x(17), 1)?;
            a.cbnz(X, x(17), top)?;
            a.ldr(MemSize::H, x(16), Addr::Off(SP, 6))?;
            a.ldr(MemSize::B, x(15), Addr::Off(x(9), -3))?;
            a.str(MemSize::H, x(15), Addr::Off(SP, 22))?;
            a.str(MemSize::B, x(15), Addr::Off(x(11), 2))?;
            Ok(())
        },
        &[
            "cmp w3, #0x0",
            "csel x15, x15, x16, ne",
            "csel w15, w15, w16, ne",
            "fcsel s0, s1, s2, ne",
            "fcsel d3, d4, d5, ne",
            "ldr x15, [x9], #8",
            "ldr x16, [x10], #8",
            "csel x15, x15, x16, ne",
            "str x15, [x11], #8",
            "sub x17, x17, #0x1",
            "cbnz x17, 0x14",
            "ldrh w16, [sp, #6]",
            "ldurb w15, [x9, #-3]",
            "strh w15, [sp, #22]",
            "strb w15, [x11, #2]",
        ],
    );
}
