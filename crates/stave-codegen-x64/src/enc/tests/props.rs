//! Property tests: ModRM/SIB/displacement encodings decode back to the operand, and no input
//! makes the encoder panic (a failed instruction emits nothing).

use proptest::prelude::*;

use crate::enc::*;

fn gpr() -> impl Strategy<Value = Gpr> {
    (0u8..16).prop_map(|n| Gpr::new(n).unwrap())
}

fn scale() -> impl Strategy<Value = Scale> {
    prop_oneof![Just(Scale::S1), Just(Scale::S2), Just(Scale::S4), Just(Scale::S8)]
}

fn mem() -> impl Strategy<Value = Mem> {
    (prop::option::of(gpr()), prop::option::of((gpr(), scale())), any::<i32>(), any::<bool>()).prop_map(
        |(base, index, disp, small)| Mem::Sib { base, index, disp: if small { disp >> 24 } else { disp } },
    )
}

fn size() -> impl Strategy<Value = Size> {
    prop_oneof![Just(Size::B), Just(Size::W), Just(Size::D), Just(Size::Q)]
}

/// Decodes `REX.W 8B /r` and returns (reg, base, index, disp, mod).
fn decode_load(b: &[u8]) -> (u8, Option<u8>, Option<(u8, u8)>, i32, u8) {
    let rex = b[0];
    assert_eq!(rex & 0xf8, 0x48);
    assert_eq!(b[1], 0x8b);
    let modrm = b[2];
    let (md, reg, rm) = (modrm >> 6, ((modrm >> 3) & 7) | ((rex & 4) << 1), modrm & 7);
    let mut i = 3;
    let (mut base, mut index) = (Some(rm | ((rex & 1) << 3)), None);
    if rm == 4 {
        let sib = b[3];
        i = 4;
        let (ss, idx, bs) = (sib >> 6, ((sib >> 3) & 7) | ((rex & 2) << 2), sib & 7);
        if idx != 4 {
            index = Some((idx, ss));
        }
        base = if md == 0 && bs == 5 { None } else { Some(bs | ((rex & 1) << 3)) };
    }
    let disp = match md {
        0 if base.is_none() => i32::from_le_bytes(b[i..i + 4].try_into().unwrap()),
        0 => 0,
        1 => b[i] as i8 as i32,
        _ => i32::from_le_bytes(b[i..i + 4].try_into().unwrap()),
    };
    (reg, base, index, disp, md)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn modrm_round_trip(m in mem(), reg in gpr()) {
        let mut a = Asm::new();
        let r = a.mov_rm(Size::Q, reg, m);
        let Mem::Sib { base, index, disp } = m else { unreachable!() };
        if index.is_some_and(|(i, _)| i == Gpr::RSP) {
            prop_assert!(r.is_err());
            prop_assert_eq!(a.offset(), 0);
            return Ok(());
        }
        prop_assert!(r.is_ok());
        let (dreg, dbase, dindex, ddisp, md) = decode_load(a.bytes());
        prop_assert_eq!(dreg, reg.num());
        prop_assert_eq!(dbase, base.map(Gpr::num));
        prop_assert_eq!(dindex, index.map(|(i, s)| (i.num(), s as u8)));
        prop_assert_eq!(ddisp, disp);
        // shortest displacement
        if let Some(b) = base {
            let want = if disp == 0 && b.num() & 7 != 5 { 0 } else if (-128..=127).contains(&disp) { 1 } else { 2 };
            prop_assert_eq!(md, want);
        }
    }

    #[test]
    fn rip_label_displacement(disp in any::<i32>() , pad in 0usize..300, imm in -128i64..128) {
        // value = label - end_of_instruction + disp, with the trailing immediate counted
        let mut a = Asm::new();
        let l = a.new_label();
        a.alu_mi(AluOp::Cmp, Size::D, Mem::rip_label(l, disp >> 8), imm).unwrap();
        let end = a.offset();
        for _ in 0..pad { a.nop(); }
        a.bind(l).unwrap();
        let code = a.finish().unwrap().code;
        let field = i32::from_le_bytes(code[2..6].try_into().unwrap());
        prop_assert_eq!(i64::from(field), (end as i64 + pad as i64) - end as i64 + i64::from(disp >> 8));
    }

    /// Any operands: Ok or Err, never a panic; an Err leaves the buffer as it was.
    #[test]
    fn no_panics(ops in prop::collection::vec((0u8..24, size(), gpr(), gpr(), mem(), any::<i64>(), any::<u8>()), 1..40)) {
        let mut a = Asm::new();
        let l = a.new_label();
        for (k, s, r1, r2, m, imm, sh) in ops {
            let before = a.offset();
            let x1 = Xmm::new(r1.num()).unwrap();
            let res: Result<(), EncError> = match k {
                0 => a.mov_rm(s, r1, m),
                1 => a.mov_mr(s, m, r2),
                2 => a.mov_ri(s, r1, imm),
                3 => a.mov_mi(s, m, imm),
                4 => a.lea(s, r1, m),
                5 => a.alu_ri(AluOp::Sub, s, r1, imm),
                6 => a.alu_mi(AluOp::Add, s, m, imm),
                7 => a.alu_rm(AluOp::Xor, s, r1, m),
                8 => a.imul_rri(s, r1, m, imm),
                9 => a.unary(UnaryOp::Idiv, s, m),
                10 => a.shift_ri(ShiftOp::Sar, s, r1, sh),
                11 => a.movzx(s, r1, Size::B, m),
                12 => a.movsx(s, r1, [Size::B, Size::W, Size::D, Size::Q][usize::from(sh & 3)], r2),
                13 => a.cmov(Cond::L, s, r1, m),
                14 => a.test_ri(s, m, imm),
                15 => a.sse(SseOp::Add, Prec::F64, x1, m),
                16 => a.cvtsi2s(Prec::F32, x1, s, m),
                17 => a.cvtts2si(Prec::F64, s, r1, m),
                18 => a.mov_xg(s, x1, r2),
                19 => a.call_rm(m),
                20 => a.jmp_rm(m),
                21 => a.imul_rr(s, r1, r2),
                22 => a.align(u32::from(sh)),
                _ => a.jcc(Cond::O, l),
            };
            if res.is_err() {
                prop_assert_eq!(a.offset(), before);
            }
        }
        let _ = a.bind(l);
        prop_assert!(a.finish().is_ok());
    }
}
