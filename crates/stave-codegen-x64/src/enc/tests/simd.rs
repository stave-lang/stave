//! Packed-vector encodings (legacy SSE and VEX) checked against objdump.
//!
//! Tests iterate every table variant (`ALL`); the expected mnemonic is the variant name, so a
//! wrong opcode, prefix, or operand placement shows up as a different instruction or operand.
//! Which instructions exist at which widths, and which operands they accept, is listed here
//! independently of the tables. objdump prints a size keyword before memory operands
//! (`XMMWORD PTR`); the size is implied by the opcode rather than encoded, so it is stripped from
//! both sides, except where the keyword is the only sign of the vector length ([`cvt_n`]).

use std::fmt::Debug;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::objdump;
use crate::enc::*;

static CHECKED: AtomicUsize = AtomicUsize::new(0);

const RIP: &str = "@RIP@";

fn x(n: u8) -> Xmm {
    Xmm::new(n).unwrap()
}

fn y(n: u8) -> Ymm {
    Ymm::new(n).unwrap()
}

fn g(n: u8) -> Gpr {
    Gpr::new(n).unwrap()
}

fn name<T: Debug>(op: T) -> String {
    format!("{op:?}").to_lowercase()
}

fn rname(l: VLen, n: u8) -> String {
    match l {
        VLen::X128 => format!("xmm{n}"),
        VLen::X256 => format!("ymm{n}"),
    }
}

fn vr(l: VLen, n: u8) -> VReg {
    match l {
        VLen::X128 => VReg::X(x(n)),
        VLen::X256 => VReg::Y(y(n)),
    }
}

const LENS: [VLen; 2] = [VLen::X128, VLen::X256];
const PAIRS: [(u8, u8); 5] = [(0, 1), (7, 6), (8, 1), (1, 9), (15, 14)];
const TRIPLES: [(u8, u8, u8); 7] =
    [(0, 1, 2), (7, 6, 5), (8, 1, 2), (1, 9, 2), (1, 2, 10), (15, 14, 13), (2, 15, 8)];
const MEM_REGS: [(u8, u8); 3] = [(0, 1), (8, 9), (15, 7)];

/// Memory operands: base only, rsp/r12/r13/rbp bases (SIB and forced-displacement cases), disp8,
/// disp32, base+index*scale with extended registers, index only, and RIP-relative.
fn mems(end: Label) -> Vec<(Mem, String)> {
    vec![
        (Mem::base(Gpr::RAX, 0), "[rax]".into()),
        (Mem::base(Gpr::RBX, 0x10), "[rbx+0x10]".into()),
        (Mem::base(Gpr::RSP, 0x20), "[rsp+0x20]".into()),
        (Mem::base(Gpr::R12, 0), "[r12]".into()),
        (Mem::base(Gpr::R13, 0), "[r13+0x0]".into()),
        (Mem::base(Gpr::RBP, -0x80), "[rbp-0x80]".into()),
        (Mem::index(Gpr::R9, Gpr::R10, Scale::S4, 0x1000), "[r9+r10*4+0x1000]".into()),
        (Mem::index(Gpr::RAX, Gpr::RCX, Scale::S8, -0x200), "[rax+rcx*8-0x200]".into()),
        (Mem::Sib { base: None, index: Some((Gpr::R11, Scale::S2)), disp: 0x40 }, "[r11*2+0x40]".into()),
        (Mem::rip_label(end, 0), format!("[rip+0x{RIP}]")),
    ]
}

/// objdump's size keywords, removed from memory operands.
fn strip(s: &str) -> String {
    let mut s = s.to_string();
    for k in ["XMMWORD", "YMMWORD", "DWORD", "QWORD", "WORD", "BYTE"] {
        s = s.replace(&format!("{k} PTR "), "");
    }
    s
}

/// A run of instructions assembled into one buffer, with the text objdump must print for each.
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

    fn mems(&self) -> Vec<(Mem, String)> {
        mems(self.end)
    }

    fn add(&mut self, want: impl Into<String>, f: impl FnOnce(&mut Asm) -> Result<(), EncError>) {
        let want = want.into();
        if let Err(e) = f(&mut self.a) {
            panic!("`{want}` does not encode: {e}");
        }
        self.ends.push(self.a.offset());
        self.want.push(want);
    }

    /// Compares every instruction's text and extent with objdump's; returns the count.
    fn run(mut self, strip_sizes: bool) -> usize {
        let total = self.a.offset();
        self.a.bind(self.end).unwrap();
        self.a.nop();
        let code = self.a.finish().unwrap().code;
        let n = self.want.len();
        CHECKED.fetch_add(n, Ordering::Relaxed);
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
            let text = if strip_sizes { strip(text) } else { text.clone() };
            let bytes: Vec<String> =
                code[start as usize..self.ends[i] as usize].iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(*at, start, "instruction {i} (`{want}`, {}): wrong start", bytes.join(" "));
            assert_eq!(text, want, "instruction {i}: bytes {}", bytes.join(" "));
            start = self.ends[i];
        }
        assert_eq!(got.len(), n + 1, "instruction count");
        n
    }
}

/// Asserts that `f` fails and emits nothing.
fn rejects(f: impl FnOnce(&mut Asm) -> Result<(), EncError>, what: &str) {
    let mut a = Asm::new();
    assert!(f(&mut a).is_err(), "{what} should be rejected");
    assert_eq!(a.offset(), 0, "{what}: rejected encoding left bytes behind");
}

/// What a table entry accepts; written out per instruction, not read from the tables.
#[derive(Default, Clone, Copy)]
struct Shape {
    /// No legacy SSE encoding.
    vonly: bool,
    /// 128-bit only.
    x: bool,
    /// 256-bit only.
    y: bool,
    /// The r/m register is an `xmm` at every width.
    rx: bool,
    /// The r/m operand must be a register.
    reg: bool,
    /// The r/m operand must be memory.
    mem: bool,
    /// Immediate is a compare predicate (tested separately).
    cmp: bool,
}

fn shape_bin(op: VBin) -> Shape {
    use VBin::*;
    let n = name(op);
    let fma = n.starts_with('f');
    Shape {
        vonly: fma
            || matches!(
                op,
                Psllvd
                    | Psllvq
                    | Psrlvd
                    | Psrlvq
                    | Psravd
                    | Permd
                    | Permps
                    | Permilps
                    | Permilpd
                    | Maskmovps
                    | Maskmovpd
                    | Pmaskmovd
                    | Pmaskmovq
            ),
        x: (fma && (n.ends_with("ss") || n.ends_with("sd"))) || matches!(op, Movhlps | Movlhps),
        y: matches!(op, Permd | Permps),
        rx: matches!(op, Psllw | Pslld | Psllq | Psrlw | Psrld | Psrlq | Psraw | Psrad),
        reg: matches!(op, Movhlps | Movlhps),
        mem: matches!(op, Maskmovps | Maskmovpd | Pmaskmovd | Pmaskmovq),
        cmp: false,
    }
}

fn shape_un(op: VUn) -> Shape {
    use VUn::*;
    let n = name(op);
    let widen = n.starts_with("pmovsx") || n.starts_with("pmovzx");
    let bcast = n.contains("broadcast");
    Shape {
        vonly: bcast || op == Cvtph2ps,
        x: op == Movq,
        y: op == Broadcastsd,
        rx: widen || bcast || matches!(op, Cvtph2ps | Cvtdq2pd | Cvtps2pd),
        ..Shape::default()
    }
}

fn shape_bin_i(op: VBinI) -> Shape {
    use VBinI::*;
    Shape {
        vonly: matches!(op, Pblendd | Perm2f128 | Perm2i128 | Insertf128 | Inserti128),
        x: matches!(op, Insertps | Roundss | Roundsd),
        y: matches!(op, Perm2f128 | Perm2i128 | Insertf128 | Inserti128),
        rx: matches!(op, Insertf128 | Inserti128),
        cmp: matches!(op, Cmpps | Cmppd),
        ..Shape::default()
    }
}

fn shape_un_i(op: VUnI) -> Shape {
    use VUnI::*;
    Shape {
        vonly: matches!(op, Permq | Permpd | Permilps | Permilpd),
        y: matches!(op, Permq | Permpd),
        ..Shape::default()
    }
}

type LegFn<Op> = fn(&mut Asm, Op, u8, XRm, Option<u8>) -> Result<(), EncError>;
type VexFn<Op> = fn(&mut Asm, Op, VLen, u8, u8, VRm, Option<u8>) -> Result<(), EncError>;

struct Family<Op: 'static> {
    ops: &'static [Op],
    shape: fn(Op) -> Shape,
    leg: LegFn<Op>,
    vex: VexFn<Op>,
    /// The VEX form has a separate first source (`vop dst, a, b`).
    three: bool,
    /// Immediates to encode (empty: the family has none).
    imms: &'static [u8],
}

fn run_family<Op: Copy + Debug>(f: Family<Op>) -> usize {
    let mut b = Batch::new();
    for &op in f.ops {
        let sh = (f.shape)(op);
        if sh.cmp {
            continue;
        }
        let mn = name(op);
        let imms: Vec<Option<u8>> =
            if f.imms.is_empty() { vec![None] } else { f.imms.iter().map(|&i| Some(i)).collect() };
        let mems = b.mems();
        for &imm in &imms {
            let tail = imm.map_or(String::new(), |i| format!(",0x{i:x}"));
            if !sh.vonly {
                if !sh.mem {
                    for (d, s) in PAIRS {
                        b.add(format!("{mn} xmm{d},xmm{s}{tail}"), |a| (f.leg)(a, op, d, XRm::X(x(s)), imm));
                    }
                }
                if !sh.reg {
                    for (d, _) in MEM_REGS {
                        for (m, text) in &mems {
                            b.add(format!("{mn} xmm{d},{text}{tail}"), |a| {
                                (f.leg)(a, op, d, XRm::M(*m), imm)
                            });
                        }
                    }
                }
            }
            for l in LENS {
                if (sh.x && l == VLen::X256) || (sh.y && l == VLen::X128) {
                    continue;
                }
                let rm_len = if sh.rx { VLen::X128 } else { l };
                if !sh.mem {
                    for (d, a1, s) in TRIPLES {
                        let (dn, an, sn) = (rname(l, d), rname(l, a1), rname(rm_len, s));
                        let want = if f.three {
                            format!("v{mn} {dn},{an},{sn}{tail}")
                        } else {
                            format!("v{mn} {dn},{sn}{tail}")
                        };
                        b.add(want, |a| (f.vex)(a, op, l, d, a1, VRm::R(vr(rm_len, s)), imm));
                    }
                }
                if !sh.reg {
                    for (d, a1) in MEM_REGS {
                        for (m, text) in &mems {
                            let (dn, an) = (rname(l, d), rname(l, a1));
                            let want = if f.three {
                                format!("v{mn} {dn},{an},{text}{tail}")
                            } else {
                                format!("v{mn} {dn},{text}{tail}")
                            };
                            b.add(want, |a| (f.vex)(a, op, l, d, a1, VRm::M(*m), imm));
                        }
                    }
                }
            }
        }
        // operand classes the hardware does not have
        let m = Mem::base(Gpr::RAX, 0);
        if sh.vonly {
            rejects(|a| (f.leg)(a, op, 0, XRm::X(x(1)), imm_of(&f)), &format!("legacy {mn}"));
        }
        if sh.x {
            rejects(
                |a| (f.vex)(a, op, VLen::X256, 0, 1, VRm::R(vr(VLen::X256, 2)), imm_of(&f)),
                &format!("256-bit {mn}"),
            );
        }
        if sh.y {
            rejects(
                |a| (f.vex)(a, op, VLen::X128, 0, 1, VRm::R(vr(VLen::X128, 2)), imm_of(&f)),
                &format!("128-bit {mn}"),
            );
        }
        if sh.reg {
            rejects(|a| (f.vex)(a, op, VLen::X128, 0, 1, VRm::M(m), imm_of(&f)), &format!("memory {mn}"));
        }
        if sh.mem {
            rejects(
                |a| (f.vex)(a, op, VLen::X128, 0, 1, VRm::R(vr(VLen::X128, 2)), imm_of(&f)),
                &format!("register {mn}"),
            );
        }
        if !sh.x && !sh.y {
            // the r/m register must have the width the instruction expects
            let wrong = if sh.rx { VLen::X256 } else { VLen::X128 };
            rejects(
                |a| (f.vex)(a, op, VLen::X256, 0, 1, VRm::R(vr(wrong, 2)), imm_of(&f)),
                &format!("{mn} with a mismatched r/m width"),
            );
        }
    }
    b.run(true)
}

fn imm_of<Op>(f: &Family<Op>) -> Option<u8> {
    f.imms.first().copied()
}

#[test]
fn binary_ops() {
    let n = run_family(Family {
        ops: VBin::ALL,
        shape: shape_bin,
        leg: |a, op, d, s, _| a.sse_bin(op, x(d), s),
        vex: |a, op, l, d, a1, b, _| match l {
            VLen::X128 => a.vex_bin(op, x(d), x(a1), b),
            VLen::X256 => a.vex_bin(op, y(d), y(a1), b),
        },
        three: true,
        imms: &[],
    });
    eprintln!("binary_ops: {n} encodings");
}

#[test]
fn unary_ops() {
    let n = run_family(Family {
        ops: VUn::ALL,
        shape: shape_un,
        leg: |a, op, d, s, _| a.sse_un(op, x(d), s),
        vex: |a, op, l, d, _, b, _| match l {
            VLen::X128 => a.vex_un(op, x(d), b),
            VLen::X256 => a.vex_un(op, y(d), b),
        },
        three: false,
        imms: &[],
    });
    eprintln!("unary_ops: {n} encodings");
}

#[test]
fn binary_imm_ops() {
    let n = run_family(Family {
        ops: VBinI::ALL,
        shape: shape_bin_i,
        leg: |a, op, d, s, i| a.sse_bin_i(op, x(d), s, i.unwrap()),
        vex: |a, op, l, d, a1, b, i| match l {
            VLen::X128 => a.vex_bin_i(op, x(d), x(a1), b, i.unwrap()),
            VLen::X256 => a.vex_bin_i(op, y(d), y(a1), b, i.unwrap()),
        },
        three: true,
        imms: &[0, 0x1b, 0xff],
    });
    eprintln!("binary_imm_ops: {n} encodings");
}

#[test]
fn unary_imm_ops() {
    let n = run_family(Family {
        ops: VUnI::ALL,
        shape: shape_un_i,
        leg: |a, op, d, s, i| a.sse_un_i(op, x(d), s, i.unwrap()),
        vex: |a, op, l, d, _, b, i| match l {
            VLen::X128 => a.vex_un_i(op, x(d), b, i.unwrap()),
            VLen::X256 => a.vex_un_i(op, y(d), b, i.unwrap()),
        },
        three: false,
        imms: &[0, 0x1b, 0xff],
    });
    eprintln!("unary_imm_ops: {n} encodings");
}

/// Compare predicates print as part of the mnemonic (`cmpltps`, `vcmpnge_uqpd`).
#[test]
fn compare_predicates() {
    const SSE: [&str; 8] = ["eq", "lt", "le", "unord", "neq", "nlt", "nle", "ord"];
    const AVX: [&str; 32] = [
        "eq", "lt", "le", "unord", "neq", "nlt", "nle", "ord", "eq_uq", "nge", "ngt", "false", "neq_oq",
        "ge", "gt", "true", "eq_os", "lt_oq", "le_oq", "unord_s", "neq_us", "nlt_uq", "nle_uq", "ord_s",
        "eq_us", "nge_uq", "ngt_uq", "false_os", "neq_os", "ge_oq", "gt_oq", "true_us",
    ];
    assert_eq!(CmpPred::Ord as u8, 7);
    let mut b = Batch::new();
    for (op, sfx) in [(VBinI::Cmpps, "ps"), (VBinI::Cmppd, "pd")] {
        for (i, p) in SSE.iter().enumerate() {
            b.add(format!("cmp{p}{sfx} xmm1,xmm2"), |a| a.sse_bin_i(op, x(1), x(2), i as u8));
            b.add(format!("cmp{p}{sfx} xmm9,[r10+0x8]"), |a| {
                a.sse_bin_i(op, x(9), Mem::base(Gpr::R10, 8), i as u8)
            });
        }
        for (i, p) in AVX.iter().enumerate() {
            let i = i as u8;
            b.add(format!("vcmp{p}{sfx} xmm1,xmm2,xmm3"), |a| a.vex_bin_i(op, x(1), x(2), x(3), i));
            b.add(format!("vcmp{p}{sfx} ymm8,ymm9,ymm15"), |a| a.vex_bin_i(op, y(8), y(9), y(15), i));
            b.add(format!("vcmp{p}{sfx} ymm0,ymm7,[rax+0x20]"), |a| {
                a.vex_bin_i(op, y(0), y(7), Mem::base(Gpr::RAX, 0x20), i)
            });
        }
        rejects(|a| a.sse_bin_i(op, x(0), x(1), 8), "legacy predicate 8");
        rejects(|a| a.vex_bin_i(op, x(0), x(1), x(2), 32), "VEX predicate 32");
    }
    let n = b.run(true);
    eprintln!("compare_predicates: {n} encodings");
}

#[test]
fn stores() {
    let mut b = Batch::new();
    let mems = b.mems();
    for &op in VStore::ALL {
        let mn = name(op);
        let x_only = op == VStore::Movq;
        for (m, text) in &mems {
            for d in [0, 7, 8, 15] {
                b.add(format!("{mn} {text},xmm{d}"), |a| a.sse_store(op, *m, x(d)));
                b.add(format!("v{mn} {text},xmm{d}"), |a| a.vex_store(op, *m, x(d)));
                if !x_only {
                    b.add(format!("v{mn} {text},ymm{d}"), |a| a.vex_store(op, *m, y(d)));
                }
            }
        }
        if x_only {
            rejects(|a| a.vex_store(op, Mem::base(Gpr::RAX, 0), y(0)), "256-bit vmovq store");
        }
    }
    let n = b.run(true);
    eprintln!("stores: {n} encodings");
}

/// The source width is the only sign of the vector length in a memory operand, so sizes stay.
#[test]
fn cvt_n() {
    let mut b = Batch::new();
    let mems = b.mems();
    for &op in VCvtN::ALL {
        let mn = name(op);
        for (d, s) in PAIRS {
            b.add(format!("{mn} xmm{d},xmm{s}"), |a| a.sse_cvt_n(op, x(d), x(s)));
            b.add(format!("v{mn} xmm{d},xmm{s}"), |a| a.vex_cvt_n(op, x(d), x(s), VLen::X128));
            b.add(format!("v{mn} xmm{d},ymm{s}"), |a| a.vex_cvt_n(op, x(d), y(s), VLen::X256));
        }
        for (m, text) in &mems {
            let t128 = format!("XMMWORD PTR {text}");
            let t256 = format!("YMMWORD PTR {text}");
            b.add(format!("{mn} xmm3,{t128}"), |a| a.sse_cvt_n(op, x(3), *m));
            b.add(format!("v{mn} xmm11,{t128}"), |a| a.vex_cvt_n(op, x(11), *m, VLen::X128));
            b.add(format!("v{mn} xmm11,{t256}"), |a| a.vex_cvt_n(op, x(11), *m, VLen::X256));
        }
        rejects(|a| a.vex_cvt_n(op, x(0), x(1), VLen::X256), "128-bit source with 256-bit length");
    }
    let n = b.run(false);
    eprintln!("cvt_n: {n} encodings");
}

#[test]
fn movmsk() {
    let mut b = Batch::new();
    for &op in VMovMsk::ALL {
        let mn = name(op);
        for (gr, xr) in [(0, 1), (8, 9), (15, 0), (7, 15), (3, 8)] {
            let gn = g(gr).name(Size::D);
            b.add(format!("{mn} {gn},xmm{xr}"), |a| {
                a.sse_movmsk(op, g(gr), x(xr));
                Ok(())
            });
            b.add(format!("v{mn} {gn},xmm{xr}"), |a| {
                a.vex_movmsk(op, g(gr), x(xr));
                Ok(())
            });
            b.add(format!("v{mn} {gn},ymm{xr}"), |a| {
                a.vex_movmsk(op, g(gr), y(xr));
                Ok(())
            });
        }
    }
    let n = b.run(true);
    eprintln!("movmsk: {n} encodings");
}

#[test]
fn blendv() {
    let mut b = Batch::new();
    let mems = b.mems();
    for &op in VBlendV::ALL {
        let mn = name(op);
        for (d, s) in PAIRS {
            b.add(format!("{mn} xmm{d},xmm{s},xmm0"), |a| a.sse_blendv(op, x(d), x(s)));
        }
        for (m, text) in &mems {
            b.add(format!("{mn} xmm9,{text},xmm0"), |a| a.sse_blendv(op, x(9), *m));
        }
        for l in LENS {
            for (d, a1, s) in TRIPLES {
                for mask in [0, 4, 8, 15] {
                    let (dn, an, sn, kn) = (rname(l, d), rname(l, a1), rname(l, s), rname(l, mask));
                    b.add(format!("v{mn} {dn},{an},{sn},{kn}"), |a| match l {
                        VLen::X128 => a.vex_blendv(op, x(d), x(a1), x(s), x(mask)),
                        VLen::X256 => a.vex_blendv(op, y(d), y(a1), y(s), y(mask)),
                    });
                }
            }
            for (m, text) in &mems {
                let (dn, an, kn) = (rname(l, 3), rname(l, 12), rname(l, 9));
                b.add(format!("v{mn} {dn},{an},{text},{kn}"), |a| match l {
                    VLen::X128 => a.vex_blendv(op, x(3), x(12), *m, x(9)),
                    VLen::X256 => a.vex_blendv(op, y(3), y(12), *m, y(9)),
                });
            }
        }
    }
    let n = b.run(true);
    eprintln!("blendv: {n} encodings");
}

#[test]
fn shifts_by_immediate() {
    let mut b = Batch::new();
    for &op in VShiftI::ALL {
        let mn = name(op);
        for imm in [0u8, 1, 7, 0x3f, 0xff] {
            for d in [0, 7, 8, 15] {
                b.add(format!("{mn} xmm{d},0x{imm:x}"), |a| {
                    a.sse_shift_i(op, x(d), imm);
                    Ok(())
                });
            }
            for l in LENS {
                for (d, _, s) in TRIPLES {
                    b.add(format!("v{mn} {},{},0x{imm:x}", rname(l, d), rname(l, s)), |a| match l {
                        VLen::X128 => a.vex_shift_i(op, x(d), x(s), imm),
                        VLen::X256 => a.vex_shift_i(op, y(d), y(s), imm),
                    });
                }
            }
        }
    }
    let n = b.run(true);
    eprintln!("shifts_by_immediate: {n} encodings");
}

#[test]
fn gpr_lane_moves() {
    let mut b = Batch::new();
    let mems = b.mems();
    for s in [Size::D, Size::Q] {
        let mn = if s == Size::D { "movd" } else { "movq" };
        for (xr, gr) in [(0, 0), (8, 1), (7, 8), (15, 15), (1, 7)] {
            let gn = g(gr).name(s);
            b.add(format!("{mn} xmm{xr},{gn}"), |a| a.sse_movd_x(s, x(xr), g(gr)));
            b.add(format!("{mn} {gn},xmm{xr}"), |a| a.sse_movd_r(s, g(gr), x(xr)));
            b.add(format!("v{mn} xmm{xr},{gn}"), |a| a.vex_movd_x(s, x(xr), g(gr)));
            b.add(format!("v{mn} {gn},xmm{xr}"), |a| a.vex_movd_r(s, g(gr), x(xr)));
        }
        for (m, text) in &mems {
            b.add(format!("{mn} xmm9,{text}"), |a| a.sse_movd_x(s, x(9), *m));
            b.add(format!("{mn} {text},xmm2"), |a| a.sse_movd_r(s, *m, x(2)));
            b.add(format!("v{mn} xmm9,{text}"), |a| a.vex_movd_x(s, x(9), *m));
            b.add(format!("v{mn} {text},xmm2"), |a| a.vex_movd_r(s, *m, x(2)));
        }
    }
    rejects(|a| a.sse_movd_x(Size::W, x(0), Gpr::RAX), "16-bit movd");
    rejects(|a| a.vex_movd_r(Size::B, Gpr::RAX, x(0)), "8-bit vmovd");
    let n = b.run(true);
    eprintln!("gpr_lane_moves: {n} encodings");
}

#[test]
fn lane_extract_insert() {
    let mut b = Batch::new();
    let mems = b.mems();
    for &op in PExtr::ALL {
        let mn = name(op);
        let s = if op == PExtr::Pextrq { Size::Q } else { Size::D };
        for imm in [0u8, 3, 0xf] {
            for (xr, gr) in [(0, 0), (8, 1), (7, 8), (15, 15), (1, 7)] {
                let gn = g(gr).name(s);
                b.add(format!("{mn} {gn},xmm{xr},0x{imm:x}"), |a| a.sse_pextr(op, g(gr), x(xr), imm));
                b.add(format!("v{mn} {gn},xmm{xr},0x{imm:x}"), |a| a.vex_pextr(op, g(gr), x(xr), imm));
            }
            for (m, text) in &mems {
                b.add(format!("{mn} {text},xmm10,0x{imm:x}"), |a| a.sse_pextr(op, *m, x(10), imm));
                b.add(format!("v{mn} {text},xmm10,0x{imm:x}"), |a| a.vex_pextr(op, *m, x(10), imm));
            }
        }
    }
    for &op in PInsr::ALL {
        let mn = name(op);
        let s = if op == PInsr::Pinsrq { Size::Q } else { Size::D };
        for imm in [0u8, 3, 0xf] {
            for (xr, gr) in [(0, 0), (8, 1), (7, 8), (15, 15), (1, 7)] {
                let gn = g(gr).name(s);
                b.add(format!("{mn} xmm{xr},{gn},0x{imm:x}"), |a| a.sse_pinsr(op, x(xr), g(gr), imm));
                b.add(format!("v{mn} xmm{xr},xmm{},{gn},0x{imm:x}", (xr + 5) & 15), |a| {
                    a.vex_pinsr(op, x(xr), x((xr + 5) & 15), g(gr), imm)
                });
            }
            for (m, text) in &mems {
                b.add(format!("{mn} xmm10,{text},0x{imm:x}"), |a| a.sse_pinsr(op, x(10), *m, imm));
                b.add(format!("v{mn} xmm10,xmm13,{text},0x{imm:x}"), |a| {
                    a.vex_pinsr(op, x(10), x(13), *m, imm)
                });
            }
        }
    }
    let n = b.run(true);
    eprintln!("lane_extract_insert: {n} encodings");
}

#[test]
fn extract_128_and_f16_store() {
    let mut b = Batch::new();
    let mems = b.mems();
    for &op in VExtract::ALL {
        let mn = name(op);
        let x_ok = op == VExtract::Cvtps2ph;
        for imm in [0u8, 1, 4] {
            for (d, s) in PAIRS {
                b.add(format!("v{mn} xmm{d},ymm{s},0x{imm:x}"), |a| a.vex_extract(op, x(d), y(s), imm));
                if x_ok {
                    b.add(format!("v{mn} xmm{d},xmm{s},0x{imm:x}"), |a| a.vex_extract(op, x(d), x(s), imm));
                }
            }
            for (m, text) in &mems {
                b.add(format!("v{mn} {text},ymm12,0x{imm:x}"), |a| a.vex_extract(op, *m, y(12), imm));
                if x_ok {
                    b.add(format!("v{mn} {text},xmm12,0x{imm:x}"), |a| a.vex_extract(op, *m, x(12), imm));
                }
            }
        }
        if !x_ok {
            rejects(|a| a.vex_extract(op, x(0), x(1), 1), "128-bit vextract");
        }
    }
    let n = b.run(true);
    eprintln!("extract_128_and_f16_store: {n} encodings");
}

#[test]
fn masked_stores() {
    let mut b = Batch::new();
    let mems = b.mems();
    for &op in VMaskStore::ALL {
        let mn = name(op);
        for l in LENS {
            for (mask, src) in [(1, 2), (8, 9), (15, 0), (3, 12)] {
                for (m, text) in &mems {
                    let (kn, sn) = (rname(l, mask), rname(l, src));
                    b.add(format!("v{mn} {text},{kn},{sn}"), |a| match l {
                        VLen::X128 => a.vex_maskstore(op, *m, x(mask), x(src)),
                        VLen::X256 => a.vex_maskstore(op, *m, y(mask), y(src)),
                    });
                }
            }
        }
    }
    let n = b.run(true);
    eprintln!("masked_stores: {n} encodings");
}

/// Vector-indexed memory: with and without a base, every displacement size, extended registers.
#[test]
fn gathers() {
    let mut b = Batch::new();
    let bases: [(Option<Gpr>, i32, &str); 8] = [
        (Some(Gpr::RAX), 0, "rax"),
        (Some(Gpr::RSP), 0x10, "rsp+0x10"),
        (Some(Gpr::R13), 0, "r13+0x0"),
        (Some(Gpr::R12), -0x20, "r12-0x20"),
        (Some(Gpr::RBX), 0x1000, "rbx+0x1000"),
        (Some(Gpr::R9), 0x7f, "r9+0x7f"),
        (None, 0x100, ""),
        (Some(Gpr::RBP), 0, "rbp+0x0"),
    ];
    for &op in VGather::ALL {
        let mn = name(op);
        let n = mn.as_str();
        // (dst, index, mask) widths the instruction accepts
        let x_ = VLen::X128;
        let y_ = VLen::X256;
        let shapes: &[(VLen, VLen, VLen)] =
            if n.contains("gatherd") && (n.ends_with("dd") || n.ends_with("ps")) {
                &[(x_, x_, x_), (y_, y_, y_)]
            } else if n.ends_with("dpd") || n.ends_with("dq") {
                &[(x_, x_, x_), (y_, x_, y_)]
            } else if n.ends_with("qps") || n.ends_with("qd") {
                &[(x_, x_, x_), (x_, y_, x_)]
            } else {
                &[(x_, x_, x_), (y_, y_, y_)]
            };
        for &(dl, il, ml) in shapes {
            for (dn, inn, kn) in [(1, 2, 3), (8, 9, 10), (15, 7, 0), (4, 12, 5)] {
                for scale in [Scale::S1, Scale::S2, Scale::S4, Scale::S8] {
                    let sc = 1 << (scale as u32);
                    for (base, disp, btext) in bases {
                        let idx = format!("{}{inn}*{sc}", if il == y_ { "ymm" } else { "xmm" });
                        let addr = match (base, btext) {
                            (None, _) => format!("[{idx}+0x{disp:x}]"),
                            (Some(_), t) if t.contains('+') || t.contains('-') => {
                                let (r, d) = t.split_at(t.find(['+', '-']).unwrap());
                                format!("[{r}+{idx}{d}]")
                            }
                            (Some(_), t) => format!("[{t}+{idx}]"),
                        };
                        let want = format!("v{mn} {},{addr},{}", rname(dl, dn), rname(ml, kn));
                        b.add(want, |a| {
                            a.vex_gather(op, vr(dl, dn), base, vr(il, inn), scale, disp, vr(ml, kn))
                        });
                    }
                }
            }
        }
        // repeated and mismatched registers
        rejects(|a| a.vex_gather(op, x(1), Some(Gpr::RAX), x(1), Scale::S4, 0, x(3)), "dst == index");
        rejects(|a| a.vex_gather(op, x(1), Some(Gpr::RAX), x(2), Scale::S4, 0, x(1)), "dst == mask");
        rejects(|a| a.vex_gather(op, x(1), Some(Gpr::RAX), x(2), Scale::S4, 0, x(2)), "index == mask");
        rejects(|a| a.vex_gather(op, y(1), Some(Gpr::RAX), x(2), Scale::S4, 0, x(3)), "ymm dst, xmm mask");
    }
    let n = b.run(true);
    eprintln!("gathers: {n} encodings");
}

#[test]
fn memory_hints_and_barriers() {
    let mut b = Batch::new();
    let mems = b.mems();
    for (h, mn) in [
        (Prefetch::T0, "prefetcht0"),
        (Prefetch::T1, "prefetcht1"),
        (Prefetch::T2, "prefetcht2"),
        (Prefetch::Nta, "prefetchnta"),
    ] {
        for (m, text) in &mems {
            b.add(format!("{mn} {text}"), |a| a.prefetch(h, *m));
        }
    }
    b.add("sfence", |a| {
        a.sfence();
        Ok(())
    });
    b.add("vzeroupper", |a| {
        a.vzeroupper();
        Ok(())
    });
    b.add("vzeroall", |a| {
        a.vzeroall();
        Ok(())
    });
    let n = b.run(true);
    eprintln!("memory_hints_and_barriers: {n} encodings");
}

#[test]
fn popcnt_and_bmi2() {
    let mut b = Batch::new();
    let mems = b.mems();
    for s in [Size::W, Size::D, Size::Q] {
        for (d, r) in [(0, 1), (8, 1), (1, 9), (15, 14), (7, 6)] {
            b.add(format!("popcnt {},{}", g(d).name(s), g(r).name(s)), |a| a.popcnt(s, g(d), g(r)));
        }
        for (m, text) in &mems {
            b.add(format!("popcnt {},{text}", g(9).name(s)), |a| a.popcnt(s, g(9), *m));
        }
    }
    rejects(|a| a.popcnt(Size::B, Gpr::RAX, Gpr::RCX), "8-bit popcnt");
    for s in [Size::D, Size::Q] {
        for (d, a1, r) in [(0, 1, 2), (8, 1, 2), (1, 9, 2), (1, 2, 10), (15, 14, 13), (7, 15, 0)] {
            let (dn, an, rn) = (g(d).name(s), g(a1).name(s), g(r).name(s));
            b.add(format!("pext {dn},{an},{rn}"), |a| a.pext(s, g(d), g(a1), g(r)));
            b.add(format!("pdep {dn},{an},{rn}"), |a| a.pdep(s, g(d), g(a1), g(r)));
            b.add(format!("bzhi {dn},{rn},{an}"), |a| a.bzhi(s, g(d), g(r), g(a1)));
        }
        for (m, text) in &mems {
            let (dn, an) = (g(9).name(s), g(12).name(s));
            b.add(format!("pext {dn},{an},{text}"), |a| a.pext(s, g(9), g(12), *m));
            b.add(format!("pdep {dn},{an},{text}"), |a| a.pdep(s, g(9), g(12), *m));
            b.add(format!("bzhi {dn},{text},{an}"), |a| a.bzhi(s, g(9), *m, g(12)));
        }
    }
    rejects(|a| a.pext(Size::W, Gpr::RAX, Gpr::RCX, Gpr::RDX), "16-bit pext");
    let n = b.run(true);
    eprintln!("popcnt_and_bmi2: {n} encodings");
}

type Encoder = Box<dyn Fn(&mut Asm) -> Result<(), EncError>>;

/// Exact bytes of the smallest valid VEX prefix: 2 bytes only for map `0F`, W0, and no extended
/// index/base register.
#[test]
fn vex_prefix_selection() {
    let cases: Vec<(Encoder, &[u8])> = vec![
        (Box::new(|a| a.vex_bin(VBin::Addps, x(0), x(1), x(2))), &[0xc5, 0xf0, 0x58, 0xc2]),
        (Box::new(|a| a.vex_bin(VBin::Addps, y(0), y(1), y(2))), &[0xc5, 0xf4, 0x58, 0xc2]),
        (Box::new(|a| a.vex_bin(VBin::Addpd, x(0), x(1), x(2))), &[0xc5, 0xf1, 0x58, 0xc2]),
        (Box::new(|a| a.vex_bin(VBin::Addps, x(8), x(1), x(2))), &[0xc5, 0x70, 0x58, 0xc2]),
        (Box::new(|a| a.vex_bin(VBin::Addps, x(0), x(9), x(2))), &[0xc5, 0xb0, 0x58, 0xc2]),
        (Box::new(|a| a.vex_bin(VBin::Addps, x(0), x(1), x(10))), &[0xc4, 0xc1, 0x70, 0x58, 0xc2]),
        (Box::new(|a| a.vex_bin(VBin::Pshufb, x(0), x(1), x(2))), &[0xc4, 0xe2, 0x71, 0x00, 0xc2]),
        (Box::new(|a| a.vex_bin(VBin::Fmadd231pd, x(0), x(1), x(2))), &[0xc4, 0xe2, 0xf1, 0xb8, 0xc2]),
        (
            Box::new(|a| a.vex_bin(VBin::Addps, x(0), x(1), Mem::index(Gpr::R9, Gpr::R10, Scale::S4, 0))),
            &[0xc4, 0x81, 0x70, 0x58, 0x04, 0x91],
        ),
        (Box::new(|a| a.vex_un(VUn::Movups, x(1), Mem::base(Gpr::RAX, 0))), &[0xc5, 0xf8, 0x10, 0x08]),
        (Box::new(|a| a.vex_un(VUn::Movups, x(1), Mem::base(Gpr::R8, 0))), &[0xc4, 0xc1, 0x78, 0x10, 0x08]),
        (Box::new(|a| a.vex_movd_x(Size::Q, x(0), Gpr::RAX)), &[0xc4, 0xe1, 0xf9, 0x6e, 0xc0]),
        (
            Box::new(|a| a.vex_blendv(VBlendV::Blendvps, x(1), x(2), x(3), x(15))),
            &[0xc4, 0xe3, 0x69, 0x4a, 0xcb, 0xf0],
        ),
    ];
    for (f, want) in cases {
        let mut a = Asm::new();
        f(&mut a).unwrap();
        assert_eq!(a.bytes(), want);
    }
}

/// The legacy forms never carry a VEX prefix.
#[test]
fn legacy_forms_are_not_vex() {
    for &op in VBin::ALL {
        if shape_bin(op).vonly || shape_bin(op).mem {
            continue;
        }
        let mut a = Asm::new();
        a.sse_bin(op, x(9), x(12)).unwrap();
        assert!(!matches!(a.bytes()[0], 0xc4 | 0xc5), "{op:?}");
    }
}
