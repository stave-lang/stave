//! EVEX and mask-register encodings checked against objdump.
//!
//! Every [`EOp`] is encoded at each vector length it has, with register and memory operands and
//! with a write mask; the expected mnemonic is the variant name and the operand shape is listed
//! here independently of the encoder's table. A memory displacement that is and one that is not a
//! multiple of the operand size shows a wrong `disp8 * N` scale as a wrong address.

use super::objdump;
use crate::enc::*;

const RIP: &str = "@RIP@";

/// The operands of an instruction, in objdump's (Intel) order.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// `v, v, v/m`
    Bin,
    /// `v, v, v/m, imm`
    BinI,
    /// `v, v/m`
    Un,
    /// `v, v/m, imm`
    UnI,
    /// `v/m, v`
    Store,
    /// `k, v, v/m`
    Cmp,
    /// `k, v, v/m` with the predicate 1 printed as `lt` in the mnemonic
    CmpLt,
    /// `v, k`
    MaskToVec,
    /// `k, v`
    VecToMask,
    /// `v, xmm/m`
    Broadcast,
    /// `v, r32/r64`
    BroadcastR,
    /// `v, v/m` with the source 1/2, 1/4, 1/8 of the destination
    Extend(u32),
    /// `narrow/m, wide, imm` (the wide register is this many times the narrow one)
    Extract(u32),
    /// `wide, wide, narrow/m, imm`
    Insert(u32),
    /// `v, v, xmm/m`
    ShiftX,
}

fn shape(op: EOp) -> Shape {
    let n = format!("{op:?}");
    let has = |p: &str| n.starts_with(p);
    if n.ends_with("St") {
        Shape::Store
    } else if has("Movdq") || has("Mova") || has("Movshdup") || has("Sqrt") || has("Pabs") {
        Shape::Un
    } else if has("Rndscale") {
        Shape::UnI
    } else if has("Pternlog") {
        Shape::BinI
    } else if has("Pcmpeq") || has("Pcmpgt") || has("Ptestm") {
        Shape::Cmp
    } else if has("Pcmp") || has("Cmpp") {
        Shape::CmpLt
    } else if has("Pmovm2") {
        Shape::MaskToVec
    } else if has("Pmov") && n.ends_with("2m") {
        Shape::VecToMask
    } else if has("Pbroadcast") && n.ends_with('R') {
        Shape::BroadcastR
    } else if has("Pbroadcast") || has("Broadcast") {
        Shape::Broadcast
    } else if has("Pmovsx") || has("Pmovzx") {
        match &n[6..] {
            "bw" | "wd" | "dq" => Shape::Extend(2),
            "bd" | "wq" => Shape::Extend(4),
            _ => Shape::Extend(8),
        }
    } else if has("Extract") {
        Shape::Extract(if n.ends_with("64x4") { 2 } else { 4 })
    } else if has("Insert") {
        Shape::Insert(if n.ends_with("64x4") { 2 } else { 4 })
    } else if n.ends_with('X') {
        Shape::ShiftX
    } else {
        Shape::Bin
    }
}

/// objdump's mnemonic.
fn mnemonic(op: EOp) -> String {
    let n = format!("{op:?}").to_lowercase();
    let n = n.strip_suffix("st").unwrap_or(&n);
    let n = match shape(op) {
        Shape::BroadcastR => n.strip_suffix('r').unwrap_or(n),
        Shape::ShiftX => n.strip_suffix('x').unwrap_or(n),
        _ => n,
    };
    match shape(op) {
        // the predicate is part of the name: `vpcmpltud`, `vcmpltps`
        Shape::CmpLt if n.starts_with("pcmpu") => format!("vpcmpltu{}", &n[5..]),
        Shape::CmpLt if n.starts_with("pcmp") => format!("vpcmplt{}", &n[4..]),
        Shape::CmpLt => format!("vcmplt{}", &n[3..]),
        _ => format!("v{n}"),
    }
}

fn lens(op: EOp) -> Vec<EVL> {
    match (op, shape(op)) {
        (EOp::Broadcastsd, _) => vec![EVL::L256, EVL::L512],
        (_, Shape::Extract(2) | Shape::Insert(2)) => vec![EVL::L512],
        (_, Shape::Extract(_) | Shape::Insert(_)) => vec![EVL::L256, EVL::L512],
        _ => vec![EVL::L128, EVL::L256, EVL::L512],
    }
}

fn vreg(bytes: u32, n: u8) -> String {
    match bytes {
        0..=16 => format!("xmm{n}"),
        32 => format!("ymm{n}"),
        _ => format!("zmm{n}"),
    }
}

fn mems(end: Label) -> Vec<(Mem, String)> {
    vec![
        (Mem::base(Gpr::RAX, 0), "[rax]".into()),
        (Mem::base(Gpr::RBX, 0x40), "[rbx+0x40]".into()),
        (Mem::base(Gpr::RSP, 0x24), "[rsp+0x24]".into()),
        (Mem::base(Gpr::R13, 0), "[r13+0x0]".into()),
        (Mem::base(Gpr::RBP, -0x80), "[rbp-0x80]".into()),
        (Mem::base(Gpr::RBP, -0x7), "[rbp-0x7]".into()),
        (Mem::base(Gpr::R12, 0x2000), "[r12+0x2000]".into()),
        (Mem::index(Gpr::R9, Gpr::R10, Scale::S4, 0x1000), "[r9+r10*4+0x1000]".into()),
        (Mem::index(Gpr::RAX, Gpr::RCX, Scale::S8, -0x200), "[rax+rcx*8-0x200]".into()),
        (Mem::rip_label(end, 0), format!("[rip+0x{RIP}]")),
    ]
}

/// Removes objdump's size keywords and its `{evex}` mark on an instruction that also has a VEX
/// encoding.
fn strip(s: &str) -> String {
    let mut s = s.strip_prefix("{evex} ").unwrap_or(s).to_string();
    for k in ["ZMMWORD", "YMMWORD", "XMMWORD", "DWORD", "QWORD", "WORD", "BYTE"] {
        s = s.replace(&format!("{k} PTR "), "");
    }
    s
}

struct Batch {
    a: Asm,
    end: Label,
    ends: Vec<u32>,
    want: Vec<String>,
}

impl Batch {
    fn new() -> Batch {
        let mut a = Asm::new();
        let end = a.new_label();
        Batch { a, end, ends: Vec::new(), want: Vec::new() }
    }

    fn add(&mut self, want: String, f: impl FnOnce(&mut Asm) -> Result<(), EncError>) {
        if let Err(e) = f(&mut self.a) {
            panic!("`{want}` does not encode: {e}");
        }
        self.ends.push(self.a.offset());
        self.want.push(want);
    }

    fn run(mut self) -> usize {
        let total = self.a.offset();
        self.a.bind(self.end).unwrap();
        self.a.nop();
        let code = self.a.finish().unwrap().code;
        let n = self.want.len();
        let Some(got) = objdump(&code) else {
            eprintln!("SKIP objdump comparison: objdump not installed");
            return n;
        };
        let mut start = 0;
        for i in 0..n {
            let want = self.want[i].replace(RIP, &format!("{:x}", total - self.ends[i]));
            let Some((at, text)) = got.get(i) else {
                panic!("objdump stopped after {i} of {n} instructions")
            };
            let bytes: Vec<String> =
                code[start as usize..self.ends[i] as usize].iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(*at, start, "instruction {i} (`{want}`, {}): wrong start", bytes.join(" "));
            assert_eq!(strip(text), want, "instruction {i}: bytes {}", bytes.join(" "));
            start = self.ends[i];
        }
        assert_eq!(got.len(), n + 1, "instruction count");
        n
    }
}

const TRIPLES: [(u8, u8, u8); 6] = [(0, 1, 2), (7, 6, 5), (8, 1, 2), (1, 9, 2), (1, 2, 10), (15, 14, 13)];

/// The text of the write mask `k` (0: none) after the destination.
fn mask(k: u8, z: bool) -> String {
    match (k, z) {
        (0, _) => String::new(),
        (k, false) => format!("{{k{k}}}"),
        (k, true) => format!("{{k{k}}}{{z}}"),
    }
}

/// Adds `op` at `vl` with registers `(d, a, b)`, the r/m operand a register or `m`, under the
/// write mask `k`.
#[allow(clippy::too_many_arguments)]
fn add(bt: &mut Batch, op: EOp, vl: EVL, (d, a, b): (u8, u8, u8), m: Option<&(Mem, String)>, k: u8, z: bool) {
    let name = mnemonic(op);
    let n = vl.bytes();
    let kr = KReg::masked(k);
    let (v, kk) = (|r: u8| vreg(n, r), mask(k, z));
    let rm = |narrow: u32| match m {
        Some((mem, text)) => (ERm::M(*mem), text.clone()),
        None => (ERm::R(b), vreg(narrow, b)),
    };
    let (ev, text) = match shape(op) {
        Shape::Bin => {
            let (rm, t) = rm(n);
            (Ev::new(op, vl, d, a, rm), format!("{name} {}{kk},{},{t}", v(d), v(a)))
        }
        Shape::BinI => {
            let (rm, t) = rm(n);
            (Ev::new(op, vl, d, a, rm).imm(0x5a), format!("{name} {}{kk},{},{t},0x5a", v(d), v(a)))
        }
        Shape::Un => {
            let (rm, t) = rm(n);
            (Ev::new(op, vl, d, 0, rm), format!("{name} {}{kk},{t}", v(d)))
        }
        Shape::UnI => {
            let (rm, t) = rm(n);
            (Ev::new(op, vl, d, 0, rm).imm(9), format!("{name} {}{kk},{t},0x9", v(d)))
        }
        Shape::Store => {
            let (rm, t) = rm(n);
            (Ev::new(op, vl, d, 0, rm), format!("{name} {t}{kk},{}", v(d)))
        }
        Shape::Cmp => {
            let (rm, t) = rm(n);
            (Ev::new(op, vl, d & 7, a, rm), format!("{name} k{}{kk},{},{t}", d & 7, v(a)))
        }
        Shape::CmpLt => {
            let (rm, t) = rm(n);
            (Ev::new(op, vl, d & 7, a, rm).imm(1), format!("{name} k{}{kk},{},{t}", d & 7, v(a)))
        }
        Shape::MaskToVec => (Ev::new(op, vl, d, 0, ERm::R(b & 7)), format!("{name} {},k{}", v(d), b & 7)),
        Shape::VecToMask => (Ev::new(op, vl, d & 7, 0, ERm::R(b)), format!("{name} k{},{}", d & 7, v(b))),
        Shape::Broadcast => {
            let (rm, t) = rm(16);
            (Ev::new(op, vl, d, 0, rm), format!("{name} {}{kk},{t}", v(d)))
        }
        Shape::BroadcastR => {
            let g = Gpr::new(b).unwrap();
            let s = if op == EOp::PbroadcastqR { Size::Q } else { Size::D };
            (Ev::new(op, vl, d, 0, g), format!("{name} {}{kk},{}", v(d), g.name(s)))
        }
        Shape::Extend(r) => {
            let (rm, t) = rm(n / r);
            (Ev::new(op, vl, d, 0, rm), format!("{name} {}{kk},{t}", v(d)))
        }
        Shape::Extract(r) => {
            let (rm, t) = rm(n / r);
            (Ev::new(op, vl, d, 0, rm).imm(1), format!("{name} {t}{kk},{},0x1", v(d)))
        }
        Shape::Insert(r) => {
            let (rm, t) = rm(n / r);
            (Ev::new(op, vl, d, a, rm).imm(1), format!("{name} {}{kk},{},{t},0x1", v(d), v(a)))
        }
        Shape::ShiftX => {
            let (rm, t) = rm(16);
            (Ev::new(op, vl, d, a, rm), format!("{name} {}{kk},{},{t}", v(d), v(a)))
        }
    };
    bt.add(text, |asm| asm.evex(ev.mask(kr, z)));
}

/// Whether `op` takes a write mask.
fn masked(op: EOp) -> bool {
    op != EOp::Psadbw && !matches!(shape(op), Shape::MaskToVec | Shape::VecToMask)
}

#[test]
fn evex_registers() {
    let mut bt = Batch::new();
    for &op in EOp::ALL {
        // objdump marks the register form of a store opcode (`vmovdqu8.s`); it is never emitted
        if shape(op) == Shape::Store {
            continue;
        }
        for vl in lens(op) {
            for t in TRIPLES {
                add(&mut bt, op, vl, t, None, 0, false);
            }
            if masked(op) {
                add(&mut bt, op, vl, (3, 4, 5), None, 1, false);
                add(&mut bt, op, vl, (9, 10, 11), None, 7, false);
                // a compare or a store has no zeroing form
                if !matches!(shape(op), Shape::Cmp | Shape::CmpLt | Shape::Store | Shape::Extract(_)) {
                    add(&mut bt, op, vl, (12, 2, 6), None, 5, true);
                }
            }
        }
    }
    assert!(bt.run() > 2000);
}

#[test]
fn evex_memory() {
    let mut bt = Batch::new();
    let ms = mems(bt.end);
    for &op in EOp::ALL {
        if matches!(shape(op), Shape::MaskToVec | Shape::VecToMask | Shape::BroadcastR) {
            continue;
        }
        for vl in lens(op) {
            for m in &ms {
                add(&mut bt, op, vl, (1, 9, 0), Some(m), 0, false);
                if masked(op) {
                    add(&mut bt, op, vl, (14, 2, 0), Some(m), 3, false);
                }
            }
        }
    }
    assert!(bt.run() > 5000);
}

#[test]
fn evex_shift_imm() {
    let mut bt = Batch::new();
    let ms = mems(bt.end);
    for &op in EShiftI::ALL {
        let name = format!("v{op:?}").to_lowercase();
        for vl in [EVL::L128, EVL::L256, EVL::L512] {
            let n = vl.bytes();
            for (d, s) in [(0u8, 1u8), (9, 2), (3, 12), (15, 15)] {
                bt.add(format!("{name} {},{},0x7", vreg(n, d), vreg(n, s)), |a| {
                    a.evex_shift_i(op, vl, d, ERm::R(s), 7, KReg::masked(0), false)
                });
                bt.add(format!("{name} {}{{k2}}{{z}},{},0x1", vreg(n, d), vreg(n, s)), |a| {
                    a.evex_shift_i(op, vl, d, ERm::R(s), 1, KReg::masked(2), true)
                });
            }
            for (m, text) in &ms {
                bt.add(format!("{name} {},{text},0x3", vreg(n, 5)), |a| {
                    a.evex_shift_i(op, vl, 5, *m, 3, KReg::masked(0), false)
                });
            }
        }
    }
    bt.run();
}

#[test]
fn evex_rejects() {
    let mut a = Asm::new();
    let zero = Ev::new(EOp::Paddd, EVL::L512, 1, 2, ERm::R(3)).mask(KReg::masked(0), true);
    assert!(a.evex(zero).is_err());
    assert!(a.evex(Ev::new(EOp::Paddd, EVL::L512, 16, 2, ERm::R(3))).is_err());
    assert!(a.evex(Ev::new(EOp::Paddd, EVL::L512, 1, 2, ERm::R(16))).is_err());
    assert_eq!(a.offset(), 0);
}

#[test]
fn evex_features() {
    use stave_target::Feature as F;
    let mut a = Asm::new();
    a.evex(Ev::new(EOp::Paddd, EVL::L512, 1, 2, ERm::R(3))).unwrap();
    assert!(a.features().contains(F::Avx512f));
    assert!(!a.features().contains(F::Avx512vl) && !a.features().contains(F::Avx512bw));
    a.evex(Ev::new(EOp::Paddb, EVL::L256, 1, 2, ERm::R(3))).unwrap();
    assert!(a.features().contains(F::Avx512vl) && a.features().contains(F::Avx512bw));
    assert!(!a.features().contains(F::Avx512dq));
    a.kmov_kk(KSize::B, KReg::masked(1), KReg::masked(2));
    assert!(a.features().contains(F::Avx512dq));
}

#[test]
fn mask_registers() {
    let k = KReg::masked;
    let sfx = |s: KSize| format!("{s:?}").to_lowercase();
    let mut bt = Batch::new();
    let ms = mems(bt.end);
    for s in KSize::ALL {
        let x = sfx(s);
        for (d, a, b) in [(1u8, 2u8, 3u8), (7, 0, 5), (0, 7, 7), (4, 4, 4)] {
            bt.add(format!("kmov{x} k{d},k{a}"), |asm| {
                asm.kmov_kk(s, k(d), k(a));
                Ok(())
            });
            for op in KBin::ALL {
                let name = format!("k{op:?}").to_lowercase();
                bt.add(format!("{name}{x} k{d},k{a},k{b}"), |asm| {
                    asm.kbin(op, s, k(d), k(a), k(b));
                    Ok(())
                });
            }
            bt.add(format!("knot{x} k{d},k{a}"), |asm| {
                asm.knot(s, k(d), k(a));
                Ok(())
            });
            bt.add(format!("kortest{x} k{d},k{a}"), |asm| {
                asm.kortest(s, k(d), k(a));
                Ok(())
            });
            bt.add(format!("kshiftl{x} k{d},k{a},0x5"), |asm| {
                asm.kshift(true, s, k(d), k(a), 5);
                Ok(())
            });
            bt.add(format!("kshiftr{x} k{d},k{a},0x3f"), |asm| {
                asm.kshift(false, s, k(d), k(a), 63);
                Ok(())
            });
        }
        let gs = if s == KSize::Q { Size::Q } else { Size::D };
        for (kr, g) in [(1u8, Gpr::RAX), (6, Gpr::R11), (3, Gpr::RDI), (7, Gpr::R15)] {
            bt.add(format!("kmov{x} k{kr},{}", g.name(gs)), |asm| {
                asm.kmov_kg(s, k(kr), g);
                Ok(())
            });
            bt.add(format!("kmov{x} {},k{kr}", g.name(gs)), |asm| {
                asm.kmov_gk(s, g, k(kr));
                Ok(())
            });
        }
        for (m, text) in &ms {
            bt.add(format!("kmov{x} k2,{text}"), |asm| asm.kmov_km(s, k(2), *m));
            bt.add(format!("kmov{x} {text},k5"), |asm| asm.kmov_mk(s, *m, k(5)));
        }
    }
    bt.run();
}
