//! Property tests: immediate encodings decode back to their values (logical immediates, wide
//! moves, add/sub immediates, load/store offsets, branch displacements), and no input makes the
//! encoder panic (a failed instruction emits nothing).

use proptest::prelude::*;

use crate::enc::*;

fn reg() -> impl Strategy<Value = Reg> {
    prop_oneof![(0u8..31).prop_map(|n| Reg::x(n).unwrap()), Just(Reg::SP), Just(Reg::ZR)]
}

fn width() -> impl Strategy<Value = Width> {
    prop_oneof![Just(Width::W), Just(Width::X)]
}

fn words_of(b: &[u8]) -> Vec<u32> {
    b.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn words(a: &Asm) -> Vec<u32> {
    words_of(a.bytes())
}

/// Executes a `movz`/`movn`/`movk`/`orr #imm` sequence on one register.
fn simulate(ws: &[u32], bits: u32) -> u64 {
    let mut x = 0u64;
    for &w in ws {
        let opc = (w >> 29) & 3;
        if (w >> 23) & 0x3f == 0b100101 {
            let hw = (w >> 21) & 3;
            let imm = u64::from((w >> 5) & 0xffff) << (16 * hw);
            x = match opc {
                0 => !imm,
                2 => imm,
                _ => (x & !(0xffff << (16 * hw))) | imm,
            };
        } else {
            assert_eq!((w >> 23) & 0x3f, 0b100100, "orr immediate");
            let v = decode_logical((w >> 22) & 1, (w >> 16) & 0x3f, (w >> 10) & 0x3f, bits).unwrap();
            x = v;
        }
    }
    if bits == 32 { x & 0xffff_ffff } else { x }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(3000))]

    #[test]
    fn logical_round_trip(v in any::<u64>(), wide in any::<bool>(), esize in 1u32..7, ones in 1u32..64, rot in 0u32..64) {
        let bits = if wide { 64 } else { 32 };
        // arbitrary values: whatever encodes must decode to itself
        let v = if wide { v } else { v & 0xffff_ffff };
        if let Some((n, r, s)) = encode_logical(v, bits) {
            prop_assert_eq!(decode_logical(n, r, s, bits), Some(v));
        }
        // constructed bitmasks always encode
        let es = 1u32 << esize;
        if es <= bits {
            let ones = 1 + ones % (es - 1);
            let rot = rot % es;
            let mask = if es == 64 { u64::MAX } else { (1u64 << es) - 1 };
            let run = (1u64 << ones) - 1;
            let elt = if rot == 0 { run } else { ((run >> rot) | (run << (es - rot))) & mask };
            let mut m = 0u64;
            let mut i = 0;
            while i < bits { m |= elt << i; i += es; }
            let enc = encode_logical(m, bits);
            prop_assert!(enc.is_some(), "{:#x} {}", m, bits);
            let (n, r, s) = enc.unwrap();
            prop_assert_eq!(decode_logical(n, r, s, bits), Some(m));
        }
    }

    #[test]
    fn mov_imm_materializes(v in any::<u64>(), shape in 0u8..4, wide in any::<bool>()) {
        let v = match shape { 0 => v, 1 => v & 0xffff_0000_ffff, 2 => !(v & 0xffff_0000), _ => v >> (v % 64) };
        let (w, bits) = if wide { (Width::X, 64) } else { (Width::W, 32) };
        let v = if wide { v } else { v & 0xffff_ffff };
        let mut a = Asm::new();
        a.mov_imm(w, Reg::x(3).unwrap(), v).unwrap();
        let ws = words(&a);
        prop_assert!(ws.len() <= (bits / 16) as usize);
        prop_assert_eq!(simulate(&ws, bits), v);
    }

    #[test]
    fn add_imm_field(imm in 0u64..(1 << 25), wide in any::<bool>()) {
        let w = if wide { Width::X } else { Width::W };
        let mut a = Asm::new();
        let ok = a.addsub_imm(AddSub::Add, w, Reg::x(1).unwrap(), Reg::SP, imm).is_ok();
        let encodable = imm < 4096 || (imm & 0xfff == 0 && imm < (1 << 24));
        prop_assert_eq!(ok, encodable);
        if ok {
            let word = words(&a)[0];
            let v = u64::from((word >> 10) & 0xfff) << (12 * ((word >> 22) & 1));
            prop_assert_eq!(v, imm);
        }
    }

    #[test]
    fn ldst_offset_field(off in -300i64..40000, sz in 0u8..4) {
        let sz = [MemSize::B, MemSize::H, MemSize::W, MemSize::X][sz as usize];
        let scale = sz as u32;
        let mut a = Asm::new();
        let res = a.ldr(sz, Reg::x(0).unwrap(), Addr::Off(Reg::x(1).unwrap(), off));
        let unit = 1i64 << scale;
        let scaled = off >= 0 && off % unit == 0 && off / unit < 4096;
        let unscaled = (-256..=255).contains(&off);
        prop_assert_eq!(res.is_ok(), scaled || unscaled);
        if let Ok(()) = res {
            let w = words(&a)[0];
            let got = if (w >> 24) & 1 == 1 {
                i64::from((w >> 10) & 0xfff) << scale
            } else {
                i64::from(((w >> 12) & 0x1ff) as i32) << 55 >> 55
            };
            prop_assert_eq!(got, off);
        }
    }

    #[test]
    fn branch_displacement(fwd in 0u32..3000, back in 0u32..3000) {
        // b, b.cond, cbz, tbz, adr to labels before and after
        let mut a = Asm::new();
        let top = a.new_label();
        let end = a.new_label();
        a.bind(top).unwrap();
        for _ in 0..back { a.nop(); }
        let at = a.offset();
        a.b(top).unwrap();
        a.b_cond(Cond::Ne, end).unwrap();
        a.cbz(Width::X, Reg::x(0).unwrap(), top).unwrap();
        a.tbz(Reg::x(0).unwrap(), 5, end).unwrap();
        a.adr(Reg::x(1).unwrap(), end).unwrap();
        for _ in 0..fwd { a.nop(); }
        a.bind(end).unwrap();
        let endo = a.offset() as i64;
        let ws = words_of(&a.finish().unwrap().code);
        let i = (at / 4) as usize;
        let sx = |v: u32, bits: u32| (i64::from(v) << (64 - bits)) >> (64 - bits);
        let p = |k: usize| (at as i64) + 4 * k as i64;
        prop_assert_eq!(p(0) + 4 * sx(ws[i] & 0x3ff_ffff, 26), 0);
        prop_assert_eq!(p(1) + 4 * sx((ws[i + 1] >> 5) & 0x7ffff, 19), endo);
        prop_assert_eq!(p(2) + 4 * sx((ws[i + 2] >> 5) & 0x7ffff, 19), 0);
        prop_assert_eq!(p(3) + 4 * sx((ws[i + 3] >> 5) & 0x3fff, 14), endo);
        let adr = ws[i + 4];
        let (immhi, immlo) = ((adr >> 5) & 0x7ffff, (adr >> 29) & 3);
        prop_assert_eq!(p(4) + sx((immhi << 2) | immlo, 21), endo);
    }

    /// Any operands: Ok or Err, never a panic; an Err leaves the buffer as it was.
    #[test]
    fn no_panics(ops in prop::collection::vec((0u8..26, width(), reg(), reg(), reg(), any::<i64>(), any::<u32>()), 1..40)) {
        let mut a = Asm::new();
        let l = a.new_label();
        for (k, w, r1, r2, r3, imm, small) in ops {
            let before = a.offset();
            let v1 = VReg::new((small % 32) as u8).unwrap();
            let sz = [MemSize::B, MemSize::H, MemSize::W, MemSize::X][(small % 4) as usize];
            let addr = match small % 4 {
                0 => Addr::Off(r2, imm >> 40),
                1 => Addr::Pre(r2, imm >> 52),
                2 => Addr::Post(r2, imm >> 52),
                _ => Addr::Idx { base: r2, index: r3, ext: IdxExt::Sxtw, scaled: small & 8 != 0 },
            };
            let res: Result<(), EncError> = match k {
                0 => a.mov_imm(w, r1, imm as u64),
                1 => a.addsub_imm(AddSub::Subs, w, r1, r2, imm as u64 >> (small % 64)),
                2 => a.addsub_reg(AddSub::Add, w, r1, r2, r3, Shift::Lsl, small % 70),
                3 => a.logic_imm(Logic::And, w, r1, r2, imm as u64),
                4 => a.logic_reg(Logic::Orr, w, r1, r2, r3, Shift::Ror, small % 70),
                5 => a.shift_imm(Shift::Asr, w, r1, r2, small % 70),
                6 => a.ldr(sz, r1, addr),
                7 => a.str(sz, r1, addr),
                8 => a.ldrs(sz, w, r1, addr),
                9 => a.ldp(w, r1, r3, addr),
                10 => a.stp_fp(VecSize::Q, v1, v1, addr),
                11 => a.ldr_fp(VecSize::D, v1, addr),
                12 => a.madd(w, r1, r2, r3, r1),
                13 => a.csel(w, r1, r2, r3, Cond::ALL[(small % 16) as usize]),
                14 => a.ccmp_imm(w, r1, small % 40, small % 20, Cond::Eq),
                15 => a.movz(w, r1, imm as u64 >> 40, small % 70),
                16 => a.cbz(w, r1, l),
                17 => a.tbnz(r1, small % 70, l),
                18 => a.br(r1),
                19 => a.fmov_imm(Fp::D, v1, f64::from_bits(imm as u64)),
                20 => a.fcvtzs(w, r1, Fp::S, v1),
                21 => a.scvtf(Fp::D, v1, w, r1),
                22 => a.addr_of_sym(r1, stave_obj::SymbolId(0), imm),
                23 => a.ldr_lo12_sym(sz, r1, r2, stave_obj::SymbolId(0), imm),
                24 => a.mov(w, r1, r2),
                _ => a.sxt(w, r1, r2, [Ext::B, Ext::H, Ext::W][(small % 3) as usize]),
            };
            if res.is_err() {
                prop_assert_eq!(a.offset(), before);
            }
        }
        let _ = a.bind(l);
        prop_assert!(a.finish().is_ok());
    }
}
