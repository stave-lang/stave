//! Advanced SIMD encodings checked against objdump: every valid arrangement of every operation,
//! and the rejection of every invalid one.

use super::check_bytes;
use crate::enc::*;

const B: u8 = 1;
const H: u8 = 2;
const S: u8 = 4;
const D: u8 = 8;
const BHS: u8 = B | H | S;
const BHSD: u8 = BHS | D;
const FP: u8 = S | D;

const REGS: [u8; 4] = [0, 7, 16, 31];

fn vr(n: u8) -> VReg {
    VReg::new(n).unwrap()
}

fn x(n: u8) -> Reg {
    Reg::x(n).unwrap()
}

fn tx(a: Arr) -> &'static str {
    match a {
        Arr::B8 => "8b",
        Arr::B16 => "16b",
        Arr::H4 => "4h",
        Arr::H8 => "8h",
        Arr::S2 => "2s",
        Arr::S4 => "4s",
        Arr::D2 => "2d",
    }
}

fn lane_bit(a: Arr) -> u8 {
    1 << a.lane() as u32
}

/// The wide arrangement of the narrow arrangement `a` (same lane count).
fn wide(a: Arr) -> &'static str {
    match a {
        Arr::B8 | Arr::B16 => "8h",
        Arr::H4 | Arr::H8 => "4s",
        Arr::S2 | Arr::S4 => "2d",
        Arr::D2 => unreachable!(),
    }
}

/// The arrangement of `a` with adjacent lanes merged (half the lanes, twice as wide).
fn pair(a: Arr) -> &'static str {
    match a {
        Arr::B8 => "4h",
        Arr::B16 => "8h",
        Arr::H4 => "2s",
        Arr::H8 => "4s",
        Arr::S2 => "1d",
        Arr::S4 => "2d",
        Arr::D2 => unreachable!(),
    }
}

/// `"2"` for the upper-half forms.
fn two(a: Arr) -> &'static str {
    if a.is_q() { "2" } else { "" }
}

fn lane_ch(l: Lane) -> char {
    match l {
        Lane::B => 'b',
        Lane::H => 'h',
        Lane::S => 's',
        Lane::D => 'd',
    }
}

/// Assembles cases and compares them with objdump in one call.
struct Batch {
    a: Asm,
    want: Vec<String>,
    next: usize,
}

type Res = Result<(), EncError>;

impl Batch {
    fn new() -> Batch {
        Batch { a: Asm::new(), want: Vec::new(), next: 0 }
    }

    /// Three distinct registers from {v0, v7, v16, v31}, rotating.
    fn regs(&mut self) -> (u8, u8, u8) {
        let i = self.next;
        self.next += 1;
        (REGS[i % 4], REGS[(i + 1) % 4], REGS[(i + 2) % 4])
    }

    /// A valid case must encode and disassemble to the text; an invalid one must be an error that
    /// emits nothing.
    fn case(
        &mut self,
        valid: bool,
        f: impl FnOnce(&mut Asm, u8, u8, u8) -> Res,
        text: impl FnOnce(u8, u8, u8) -> String,
    ) {
        let (d, n, m) = self.regs();
        let before = self.a.offset();
        let r = f(&mut self.a, d, n, m);
        if valid {
            r.unwrap();
            self.want.push(text(d, n, m));
        } else {
            assert!(r.is_err(), "expected an encoding error: {}", text(d, n, m));
            assert_eq!(self.a.offset(), before, "an invalid case emitted code");
        }
    }

    fn ok(&mut self, f: impl FnOnce(&mut Asm, u8, u8, u8) -> Res, text: impl FnOnce(u8, u8, u8) -> String) {
        self.case(true, f, text);
    }

    fn err(&mut self, f: impl FnOnce(&mut Asm, u8, u8, u8) -> Res) {
        self.case(false, f, |_, _, _| "(error expected)".into());
    }

    fn run(self, name: &str) {
        let code = self.a.finish().unwrap().code;
        eprintln!("simd {name}: {} encodings", self.want.len());
        check_bytes(&code, &self.want);
    }
}

#[test]
fn int_three_same() {
    let ops: [(VBin, &str, u8); 24] = [
        (VBin::Add, "add", BHSD),
        (VBin::Sub, "sub", BHSD),
        (VBin::Mul, "mul", BHS),
        (VBin::SqAdd, "sqadd", BHSD),
        (VBin::UqAdd, "uqadd", BHSD),
        (VBin::SqSub, "sqsub", BHSD),
        (VBin::UqSub, "uqsub", BHSD),
        (VBin::Smax, "smax", BHS),
        (VBin::Umax, "umax", BHS),
        (VBin::Smin, "smin", BHS),
        (VBin::Umin, "umin", BHS),
        (VBin::Addp, "addp", BHSD),
        (VBin::Smaxp, "smaxp", BHS),
        (VBin::Umaxp, "umaxp", BHS),
        (VBin::Sminp, "sminp", BHS),
        (VBin::Uminp, "uminp", BHS),
        (VBin::Cmeq, "cmeq", BHSD),
        (VBin::Cmgt, "cmgt", BHSD),
        (VBin::Cmge, "cmge", BHSD),
        (VBin::Cmhi, "cmhi", BHSD),
        (VBin::Cmhs, "cmhs", BHSD),
        (VBin::Cmtst, "cmtst", BHSD),
        (VBin::Sshl, "sshl", BHSD),
        (VBin::Ushl, "ushl", BHSD),
    ];
    let mut b = Batch::new();
    for (op, name, mask) in ops {
        for arr in Arr::ALL {
            b.case(
                lane_bit(arr) & mask != 0,
                |a, d, n, m| a.vbin(op, arr, vr(d), vr(n), vr(m)),
                |d, n, m| format!("{name} v{d}.{0}, v{n}.{0}, v{m}.{0}", tx(arr)),
            );
        }
    }
    b.run("int three-same");
}

#[test]
fn bitwise() {
    let ops = [
        (VLogic::And, "and"),
        (VLogic::Bic, "bic"),
        (VLogic::Orr, "orr"),
        (VLogic::Orn, "orn"),
        (VLogic::Eor, "eor"),
        (VLogic::Bsl, "bsl"),
        (VLogic::Bit, "bit"),
        (VLogic::Bif, "bif"),
    ];
    let mut b = Batch::new();
    for (op, name) in ops {
        for arr in Arr::ALL {
            b.case(
                matches!(arr, Arr::B8 | Arr::B16),
                |a, d, n, m| a.vlogic(op, arr, vr(d), vr(n), vr(m)),
                |d, n, m| format!("{name} v{d}.{0}, v{n}.{0}, v{m}.{0}", tx(arr)),
            );
        }
    }
    for arr in Arr::ALL {
        b.case(
            matches!(arr, Arr::B8 | Arr::B16),
            |a, d, n, _| a.vmov(arr, vr(d), vr(n)),
            |d, n, _| format!("mov v{d}.{0}, v{n}.{0}", tx(arr)),
        );
        b.case(
            matches!(arr, Arr::B8 | Arr::B16),
            |a, d, n, _| a.vint_un(VIntUn::Not, arr, vr(d), vr(n)),
            |d, n, _| format!("mvn v{d}.{0}, v{n}.{0}", tx(arr)),
        );
    }
    b.run("bitwise");
}

#[test]
fn fp_three_same() {
    let ops = [
        (VFpBin::Add, "fadd"),
        (VFpBin::Sub, "fsub"),
        (VFpBin::Mul, "fmul"),
        (VFpBin::Div, "fdiv"),
        (VFpBin::Max, "fmax"),
        (VFpBin::Min, "fmin"),
        (VFpBin::MaxNm, "fmaxnm"),
        (VFpBin::MinNm, "fminnm"),
        (VFpBin::Fmla, "fmla"),
        (VFpBin::Fmls, "fmls"),
        (VFpBin::Addp, "faddp"),
        (VFpBin::Maxp, "fmaxp"),
        (VFpBin::Minp, "fminp"),
        (VFpBin::MaxNmp, "fmaxnmp"),
        (VFpBin::MinNmp, "fminnmp"),
        (VFpBin::Cmeq, "fcmeq"),
        (VFpBin::Cmge, "fcmge"),
        (VFpBin::Cmgt, "fcmgt"),
        (VFpBin::Acge, "facge"),
        (VFpBin::Acgt, "facgt"),
        (VFpBin::Abd, "fabd"),
    ];
    let mut b = Batch::new();
    for (op, name) in ops {
        for arr in Arr::ALL {
            b.case(
                lane_bit(arr) & FP != 0,
                |a, d, n, m| a.vfp_bin(op, arr, vr(d), vr(n), vr(m)),
                |d, n, m| format!("{name} v{d}.{0}, v{n}.{0}, v{m}.{0}", tx(arr)),
            );
        }
    }
    b.run("fp three-same");
}

#[test]
fn two_register_misc() {
    let int = [
        (VIntUn::Abs, "abs", BHSD),
        (VIntUn::Neg, "neg", BHSD),
        (VIntUn::Cls, "cls", BHS),
        (VIntUn::Clz, "clz", BHS),
        (VIntUn::Cnt, "cnt", B),
    ];
    let rev = [(Rev::R16, "rev16", B), (Rev::R32, "rev32", B | H), (Rev::R64, "rev64", BHS)];
    let fp = [
        (VFpUn::Abs, "fabs"),
        (VFpUn::Neg, "fneg"),
        (VFpUn::Sqrt, "fsqrt"),
        (VFpUn::RintN, "frintn"),
        (VFpUn::RintM, "frintm"),
        (VFpUn::RintP, "frintp"),
        (VFpUn::RintZ, "frintz"),
        (VFpUn::RintA, "frinta"),
        (VFpUn::RintX, "frintx"),
        (VFpUn::RintI, "frinti"),
    ];
    let cmp = [(CmpZ::Eq, "eq"), (CmpZ::Gt, "gt"), (CmpZ::Ge, "ge"), (CmpZ::Lt, "lt"), (CmpZ::Le, "le")];
    let mut b = Batch::new();
    for (op, name, mask) in int {
        for arr in Arr::ALL {
            b.case(
                lane_bit(arr) & mask != 0,
                |a, d, n, _| a.vint_un(op, arr, vr(d), vr(n)),
                |d, n, _| format!("{name} v{d}.{0}, v{n}.{0}", tx(arr)),
            );
        }
    }
    for (sign, name) in [(Sign::Signed, "saddlp"), (Sign::Unsigned, "uaddlp")] {
        for arr in Arr::ALL {
            b.case(
                lane_bit(arr) & BHS != 0,
                |a, d, n, _| a.vaddlp(sign, arr, vr(d), vr(n)),
                |d, n, _| format!("{name} v{d}.{}, v{n}.{}", pair(arr), tx(arr)),
            );
        }
    }
    for (op, name, mask) in rev {
        for arr in Arr::ALL {
            b.case(
                lane_bit(arr) & mask != 0,
                |a, d, n, _| a.vrev(op, arr, vr(d), vr(n)),
                |d, n, _| format!("{name} v{d}.{0}, v{n}.{0}", tx(arr)),
            );
        }
    }
    for (op, name) in fp {
        for arr in Arr::ALL {
            b.case(
                lane_bit(arr) & FP != 0,
                |a, d, n, _| a.vfp_un(op, arr, vr(d), vr(n)),
                |d, n, _| format!("{name} v{d}.{0}, v{n}.{0}", tx(arr)),
            );
        }
    }
    for (op, name) in cmp {
        for arr in Arr::ALL {
            b.ok(
                |a, d, n, _| a.vcmp_zero(op, arr, vr(d), vr(n)),
                |d, n, _| format!("cm{name} v{d}.{0}, v{n}.{0}, #0", tx(arr)),
            );
            b.case(
                lane_bit(arr) & FP != 0,
                |a, d, n, _| a.vfcmp_zero(op, arr, vr(d), vr(n)),
                |d, n, _| format!("fcm{name} v{d}.{0}, v{n}.{0}, #0.0", tx(arr)),
            );
        }
    }
    b.run("two-register misc");
}

#[test]
fn float_integer_conversions() {
    let rounds = [
        (FcvtRound::Ns, "fcvtns"),
        (FcvtRound::Nu, "fcvtnu"),
        (FcvtRound::As, "fcvtas"),
        (FcvtRound::Au, "fcvtau"),
        (FcvtRound::Ps, "fcvtps"),
        (FcvtRound::Pu, "fcvtpu"),
        (FcvtRound::Ms, "fcvtms"),
        (FcvtRound::Mu, "fcvtmu"),
        (FcvtRound::Zs, "fcvtzs"),
        (FcvtRound::Zu, "fcvtzu"),
    ];
    let mut b = Batch::new();
    for arr in Arr::ALL {
        let fp = lane_bit(arr) & FP != 0;
        b.case(
            fp,
            |a, d, n, _| a.scvtf_v(arr, vr(d), vr(n)),
            |d, n, _| format!("scvtf v{d}.{0}, v{n}.{0}", tx(arr)),
        );
        b.case(
            fp,
            |a, d, n, _| a.ucvtf_v(arr, vr(d), vr(n)),
            |d, n, _| format!("ucvtf v{d}.{0}, v{n}.{0}", tx(arr)),
        );
        for (rnd, name) in rounds {
            b.case(
                fp,
                |a, d, n, _| a.fcvt_v(rnd, arr, vr(d), vr(n)),
                |d, n, _| format!("{name} v{d}.{0}, v{n}.{0}", tx(arr)),
            );
        }
    }
    // Precision conversions: the narrow side names the arrangement.
    let fcvtl = [
        (Arr::H4, "fcvtl", "4s"),
        (Arr::H8, "fcvtl2", "4s"),
        (Arr::S2, "fcvtl", "2d"),
        (Arr::S4, "fcvtl2", "2d"),
    ];
    let fcvtn = [
        (Arr::H4, "fcvtn", "4s"),
        (Arr::H8, "fcvtn2", "4s"),
        (Arr::S2, "fcvtn", "2d"),
        (Arr::S4, "fcvtn2", "2d"),
    ];
    for arr in Arr::ALL {
        let l = fcvtl.iter().find(|e| e.0 == arr);
        b.case(
            l.is_some(),
            |a, d, n, _| a.fcvtl(arr, vr(d), vr(n)),
            |d, n, _| {
                let (_, name, w) = l.unwrap();
                format!("{name} v{d}.{w}, v{n}.{}", tx(arr))
            },
        );
        let nn = fcvtn.iter().find(|e| e.0 == arr);
        b.case(
            nn.is_some(),
            |a, d, n, _| a.fcvtn(arr, vr(d), vr(n)),
            |d, n, _| {
                let (_, name, w) = nn.unwrap();
                format!("{name} v{d}.{}, v{n}.{w}", tx(arr))
            },
        );
    }
    b.run("float conversions");
}

#[test]
fn widen_narrow() {
    let mut b = Batch::new();
    for arr in Arr::ALL {
        let ok = arr.lane() != Lane::D;
        for (sign, name) in [(Sign::Signed, "s"), (Sign::Unsigned, "u")] {
            b.case(
                ok,
                |a, d, n, _| a.vwiden(sign, arr, vr(d), vr(n)),
                |d, n, _| format!("{name}xtl{} v{d}.{}, v{n}.{}", two(arr), wide(arr), tx(arr)),
            );
            let esize = arr.lane().bits();
            for shift in [0, 1, esize / 2, esize - 1, esize] {
                let valid = ok && shift < esize && shift != 0;
                if shift == 0 {
                    continue;
                }
                b.case(
                    valid,
                    |a, d, n, _| a.vshll(sign, arr, vr(d), vr(n), shift),
                    |d, n, _| {
                        format!("{name}shll{} v{d}.{}, v{n}.{}, #{shift}", two(arr), wide(arr), tx(arr))
                    },
                );
            }
        }
        b.case(
            ok,
            |a, d, n, _| a.sxtl(arr, vr(d), vr(n)),
            |d, n, _| format!("sxtl{} v{d}.{}, v{n}.{}", two(arr), wide(arr), tx(arr)),
        );
        b.case(
            ok,
            |a, d, n, _| a.uxtl(arr, vr(d), vr(n)),
            |d, n, _| format!("uxtl{} v{d}.{}, v{n}.{}", two(arr), wide(arr), tx(arr)),
        );
        for (op, name) in [
            (Narrow::Xtn, "xtn"),
            (Narrow::Sqxtn, "sqxtn"),
            (Narrow::Uqxtn, "uqxtn"),
            (Narrow::Sqxtun, "sqxtun"),
        ] {
            b.case(
                ok,
                |a, d, n, _| a.vnarrow(op, arr, vr(d), vr(n)),
                |d, n, _| format!("{name}{} v{d}.{}, v{n}.{}", two(arr), tx(arr), wide(arr)),
            );
        }
        for (round, name) in [(false, "shrn"), (true, "rshrn")] {
            let esize = arr.lane().bits();
            for shift in [0, 1, esize / 2, esize, esize + 1] {
                let valid = ok && (1..=esize).contains(&shift);
                b.case(
                    valid,
                    |a, d, n, _| a.vshrn(round, arr, vr(d), vr(n), shift),
                    |d, n, _| format!("{name}{} v{d}.{}, v{n}.{}, #{shift}", two(arr), tx(arr), wide(arr)),
                );
            }
        }
        b.case(
            ok,
            |a, d, n, m| a.addhn(arr, vr(d), vr(n), vr(m)),
            |d, n, m| format!("addhn{} v{d}.{}, v{n}.{w}, v{m}.{w}", two(arr), tx(arr), w = wide(arr)),
        );
        for (sign, name) in [(Sign::Signed, "smull"), (Sign::Unsigned, "umull")] {
            b.case(
                ok,
                |a, d, n, m| a.vmull(sign, arr, vr(d), vr(n), vr(m)),
                |d, n, m| format!("{name}{} v{d}.{}, v{n}.{t}, v{m}.{t}", two(arr), wide(arr), t = tx(arr)),
            );
        }
    }
    b.run("widen/narrow");
}

#[test]
fn reductions() {
    let int = [
        (Across::Addv, "addv", false),
        (Across::Saddlv, "saddlv", true),
        (Across::Uaddlv, "uaddlv", true),
        (Across::Smaxv, "smaxv", false),
        (Across::Umaxv, "umaxv", false),
        (Across::Sminv, "sminv", false),
        (Across::Uminv, "uminv", false),
    ];
    let fp = [
        (Across::Fmaxv, "fmaxv"),
        (Across::Fminv, "fminv"),
        (Across::FmaxNmv, "fmaxnmv"),
        (Across::FminNmv, "fminnmv"),
    ];
    let mut b = Batch::new();
    for (op, name, widening) in int {
        for arr in Arr::ALL {
            let l = if widening { arr.lane() as u32 + 1 } else { arr.lane() as u32 };
            let dest = ['b', 'h', 's', 'd'][l as usize % 4];
            b.case(
                !matches!(arr, Arr::S2 | Arr::D2),
                |a, d, n, _| a.vacross(op, arr, vr(d), vr(n)),
                |d, n, _| format!("{name} {dest}{d}, v{n}.{}", tx(arr)),
            );
        }
    }
    for (op, name) in fp {
        for arr in Arr::ALL {
            b.case(
                arr == Arr::S4,
                |a, d, n, _| a.vacross(op, arr, vr(d), vr(n)),
                |d, n, _| format!("{name} s{d}, v{n}.4s"),
            );
        }
    }
    b.ok(|a, d, n, _| a.addp_scalar(vr(d), vr(n)), |d, n, _| format!("addp d{d}, v{n}.2d"));
    b.ok(|a, d, n, _| a.faddp_scalar(Fp::S, vr(d), vr(n)), |d, n, _| format!("faddp s{d}, v{n}.2s"));
    b.ok(|a, d, n, _| a.faddp_scalar(Fp::D, vr(d), vr(n)), |d, n, _| format!("faddp d{d}, v{n}.2d"));
    b.run("reductions");
}

#[test]
fn shifts_by_immediate() {
    let ops = [
        (VShift::Shl, "shl", true),
        (VShift::Sshr, "sshr", false),
        (VShift::Ushr, "ushr", false),
        (VShift::Sli, "sli", true),
        (VShift::Sri, "sri", false),
    ];
    let mut b = Batch::new();
    for (op, name, left) in ops {
        for arr in Arr::ALL {
            let e = arr.lane().bits();
            for imm in [0, 1, e / 2, e - 1, e, e + 1] {
                let valid = if left { imm < e } else { (1..=e).contains(&imm) };
                b.case(
                    valid,
                    |a, d, n, _| a.vshift(op, arr, vr(d), vr(n), imm),
                    |d, n, _| format!("{name} v{d}.{0}, v{n}.{0}, #{imm}", tx(arr)),
                );
            }
        }
    }
    b.run("shifts");
}

#[test]
fn element_moves() {
    let mut b = Batch::new();
    for arr in Arr::ALL {
        let l = arr.lane();
        let gp = if l == Lane::D { 'x' } else { 'w' };
        b.ok(
            |a, d, n, _| a.dup_gpr(arr, vr(d), x(n.min(30))),
            |d, n, _| format!("dup v{d}.{}, {gp}{}", tx(arr), n.min(30)),
        );
        b.ok(|a, d, _, _| a.dup_gpr(arr, vr(d), Reg::ZR), |d, _, _| format!("dup v{d}.{}, {gp}zr", tx(arr)));
        b.err(|a, d, _, _| a.dup_gpr(arr, vr(d), Reg::SP));
        let max = 15 >> l as u32;
        for idx in [0, 1.min(max), max, max + 1] {
            b.case(
                idx <= max,
                |a, d, n, _| a.dup_elem(arr, vr(d), vr(n), idx),
                |d, n, _| format!("dup v{d}.{}, v{n}.{}[{idx}]", tx(arr), lane_ch(l)),
            );
        }
    }
    for l in [Lane::B, Lane::H, Lane::S, Lane::D] {
        let c = lane_ch(l);
        let gp = if l == Lane::D { 'x' } else { 'w' };
        let max = 15 >> l as u32;
        for idx in [0, 1, max, max + 1] {
            let valid = idx <= max;
            b.case(
                valid,
                |a, d, n, _| a.dup_scalar(l, vr(d), vr(n), idx),
                |d, n, _| format!("mov {c}{d}, v{n}.{c}[{idx}]"),
            );
            b.case(
                valid,
                |a, d, n, _| a.ins_gpr(l, vr(d), idx, x(n.min(30))),
                |d, n, _| format!("mov v{d}.{c}[{idx}], {gp}{}", n.min(30)),
            );
            b.case(
                valid,
                |a, d, _, _| a.ins_gpr(l, vr(d), idx, Reg::ZR),
                |d, _, _| format!("mov v{d}.{c}[{idx}], {gp}zr"),
            );
            let rev = max.saturating_sub(idx);
            b.case(
                valid,
                |a, d, n, _| a.ins_elem(l, vr(d), idx, vr(n), rev),
                |d, n, _| format!("mov v{d}.{c}[{idx}], v{n}.{c}[{rev}]"),
            );
            b.case(
                valid,
                |a, d, n, _| a.ins_elem(l, vr(d), rev, vr(n), idx),
                |d, n, _| format!("mov v{d}.{c}[{rev}], v{n}.{c}[{idx}]"),
            );
            let umov = match l {
                Lane::B | Lane::H => "umov",
                _ => "mov",
            };
            b.case(
                valid,
                |a, d, n, _| a.umov(l, x(n.min(30)), vr(d), idx),
                |d, n, _| format!("{umov} {gp}{}, v{d}.{c}[{idx}]", n.min(30)),
            );
            b.case(
                valid,
                |a, d, _, _| a.umov(l, Reg::ZR, vr(d), idx),
                |d, _, _| format!("{umov} {gp}zr, v{d}.{c}[{idx}]"),
            );
            for (w, g) in [(Width::W, 'w'), (Width::X, 'x')] {
                let ok = valid && l != Lane::D && !(l == Lane::S && w == Width::W);
                b.case(
                    ok,
                    |a, d, n, _| a.smov(l, w, x(n.min(30)), vr(d), idx),
                    |d, n, _| format!("smov {g}{}, v{d}.{c}[{idx}]", n.min(30)),
                );
            }
        }
    }
    b.err(|a, d, _, _| a.ins_gpr(Lane::S, vr(d), 0, Reg::SP));
    b.err(|a, d, _, _| a.umov(Lane::S, Reg::SP, vr(d), 0));
    b.run("element moves");
}

#[test]
fn immediates() {
    use ImmShift::{Lsl, Msl};
    let mut b = Batch::new();
    let imms = [0u8, 1, 0x7f, 0x80, 0xff];
    for arr in Arr::ALL {
        let t = tx(arr);
        let lane = arr.lane();
        let shifts = [Lsl(0), Lsl(8), Lsl(16), Lsl(24), Lsl(4), Msl(8), Msl(16), Msl(0), Msl(24)];
        for imm in imms {
            for sh in shifts {
                let valid = matches!(
                    (lane, sh),
                    (Lane::B, Lsl(0)) | (Lane::H, Lsl(0 | 8)) | (Lane::S, Lsl(0 | 8 | 16 | 24) | Msl(8 | 16))
                );
                let sh_text = match sh {
                    Lsl(0) => String::new(),
                    Lsl(n) => format!(", lsl #{n}"),
                    Msl(n) => format!(", msl #{n}"),
                };
                b.case(
                    valid,
                    |a, d, _, _| a.movi(arr, vr(d), imm, sh),
                    |d, _, _| format!("movi v{d}.{t}, #{imm:#x}{sh_text}"),
                );
                b.case(
                    valid && lane != Lane::B,
                    |a, d, _, _| a.mvni(arr, vr(d), imm, sh),
                    |d, _, _| format!("mvni v{d}.{t}, #{imm:#x}{sh_text}"),
                );
            }
        }
    }
    let masks: [u64; 6] = [
        0,
        u64::MAX,
        0xff00_ff00_ff00_ff00,
        0x0000_0000_ffff_ffff,
        0x00ff_0000_0000_00ff,
        0x8000_0000_0000_0001,
    ];
    for m in masks {
        let valid = m != 0x8000_0000_0000_0001;
        b.case(valid, |a, d, _, _| a.movi_2d(vr(d), m), |d, _, _| format!("movi v{d}.2d, #{m:#x}"));
    }
    b.err(|a, d, _, _| a.movi_2d(vr(d), 1));
    b.err(|a, d, _, _| a.movi_2d(vr(d), 0x0100_0000_0000_0000));
    for arr in Arr::ALL {
        for val in [1.0, -0.5, 31.0, 0.125, 2.0, -31.0, 0.0, 0.1, 32.0] {
            let enc = lane_bit(arr) & FP != 0 && matches!(val, 1.0 | -0.5 | 31.0 | 0.125 | 2.0 | -31.0);
            b.case(
                enc,
                |a, d, _, _| a.fmov_vec(arr, vr(d), val),
                |d, _, _| format!("fmov v{d}.{}, #{}", tx(arr), c_exp(val)),
            );
        }
    }
    b.run("immediates");
}

#[test]
fn permutes_and_tables() {
    let perms = [
        (Perm::Uzp1, "uzp1"),
        (Perm::Trn1, "trn1"),
        (Perm::Zip1, "zip1"),
        (Perm::Uzp2, "uzp2"),
        (Perm::Trn2, "trn2"),
        (Perm::Zip2, "zip2"),
    ];
    let mut b = Batch::new();
    for (op, name) in perms {
        for arr in Arr::ALL {
            b.ok(
                |a, d, n, m| a.vperm(op, arr, vr(d), vr(n), vr(m)),
                |d, n, m| format!("{name} v{d}.{0}, v{n}.{0}, v{m}.{0}", tx(arr)),
            );
        }
    }
    for arr in Arr::ALL {
        let bytes = arr.lanes();
        for idx in [0, 1, bytes / 2, bytes - 1, bytes, 15, 16] {
            let valid = matches!(arr, Arr::B8 | Arr::B16) && idx < bytes;
            b.case(
                valid,
                |a, d, n, m| a.vext(arr, vr(d), vr(n), vr(m), idx),
                |d, n, m| format!("ext v{d}.{0}, v{n}.{0}, v{m}.{0}, #{idx}", tx(arr)),
            );
        }
    }
    for (op, name) in [(Tbl::Tbl, "tbl"), (Tbl::Tbx, "tbx")] {
        for arr in Arr::ALL {
            for n_regs in 0u8..=5 {
                let valid = matches!(arr, Arr::B8 | Arr::B16) && (1..=4).contains(&n_regs);
                b.case(
                    valid,
                    |a, d, n, m| a.vtbl(op, arr, vr(d), vr(n), n_regs, vr(m)),
                    |d, n, m| {
                        format!(
                            "{name} v{d}.{t}, {{{}}}, v{m}.{t}",
                            vec_list(n, n_regs, Arr::B16),
                            t = tx(arr)
                        )
                    },
                );
            }
        }
    }
    b.run("permutes and tables");
}

/// objdump's `%.18e` spelling of a float immediate.
fn c_exp(val: f64) -> String {
    let s = format!("{val:.18e}");
    let (mant, exp) = s.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    format!("{mant}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
}

/// `v0.8b-v1.8b`: the register range of a multi-register `ld1`/`st1` list.
fn vec_list(first: u8, n: u8, arr: Arr) -> String {
    if n == 1 {
        return reg_list(first, 1, arr);
    }
    format!("v{first}.{t}-v{}.{t}", (first + n - 1) % 32, t = tx(arr))
}

/// `v7.16b, v8.16b`: consecutive registers wrapping from v31 to v0.
fn reg_list(first: u8, n: u8, arr: Arr) -> String {
    (0..n).map(|k| format!("v{}.{}", (first + k) % 32, tx(arr))).collect::<Vec<_>>().join(", ")
}

#[test]
fn structure_loads_and_stores() {
    let mut b = Batch::new();
    type Case = (fn(u32) -> VecAddr, fn(u32) -> String);
    let addrs: [Case; 5] = [
        (|_| VecAddr::Base(x(1)), |_| "[x1]".into()),
        (|_| VecAddr::Base(Reg::SP), |_| "[sp]".into()),
        (|_| VecAddr::PostImm(x(2)), |bytes| format!("[x2], #{bytes}")),
        (|_| VecAddr::PostImm(Reg::SP), |bytes| format!("[sp], #{bytes}")),
        (|_| VecAddr::PostReg { base: x(3), index: x(30) }, |_| "[x3], x30".into()),
    ];
    for arr in Arr::ALL {
        for n_regs in 0u8..=5 {
            for (mk, txt) in addrs {
                let valid = (1..=4).contains(&n_regs);
                let bytes = u32::from(n_regs) * if arr.is_q() { 16 } else { 8 };
                b.case(
                    valid,
                    |a, d, _, _| a.ld1(arr, vr(d), n_regs, mk(bytes)),
                    |d, _, _| format!("ld1 {{{}}}, {}", vec_list(d, n_regs, arr), txt(bytes)),
                );
                b.case(
                    valid,
                    |a, d, _, _| a.st1(arr, vr(d), n_regs, mk(bytes)),
                    |d, _, _| format!("st1 {{{}}}, {}", vec_list(d, n_regs, arr), txt(bytes)),
                );
            }
        }
        for (mk, txt) in addrs {
            let bytes = 1u32 << arr.lane() as u32;
            b.ok(
                |a, d, _, _| a.ld1r(arr, vr(d), mk(bytes)),
                |d, _, _| format!("ld1r {{v{d}.{}}}, {}", tx(arr), txt(bytes)),
            );
        }
        b.err(|a, d, _, _| a.ld1(arr, vr(d), 1, VecAddr::Base(Reg::ZR)));
        b.err(|a, d, _, _| a.st1(arr, vr(d), 1, VecAddr::PostReg { base: x(1), index: Reg::SP }));
        b.err(|a, d, _, _| a.ld1r(arr, vr(d), VecAddr::PostReg { base: x(1), index: Reg::ZR }));
    }
    b.run("structure loads and stores");
}

#[test]
fn pairs_and_quad_registers() {
    let mut b = Batch::new();
    for (vs, c, unit) in [(VecSize::S, 's', 4i64), (VecSize::D, 'd', 8), (VecSize::Q, 'q', 16)] {
        for off in [0, unit, 63 * unit, -64 * unit, 64 * unit, -65 * unit, unit + 1] {
            let valid = off % unit == 0 && (-64 * unit..=63 * unit).contains(&off);
            let o = if off == 0 { String::new() } else { format!(", #{off}") };
            b.case(
                valid,
                |a, d, n, _| a.ldnp_fp(vs, vr(d), vr(n), x(1), off),
                |d, n, _| format!("ldnp {c}{d}, {c}{n}, [x1{o}]"),
            );
            b.case(
                valid,
                |a, d, n, _| a.stnp_fp(vs, vr(d), vr(n), Reg::SP, off),
                |d, n, _| format!("stnp {c}{d}, {c}{n}, [sp{o}]"),
            );
            b.case(
                valid,
                |a, d, n, _| a.ldp_fp(vs, vr(d), vr(n), Addr::Off(x(1), off)),
                |d, n, _| format!("ldp {c}{d}, {c}{n}, [x1{o}]"),
            );
            b.case(
                valid,
                |a, d, n, _| a.stp_fp(vs, vr(d), vr(n), Addr::Off(Reg::SP, off)),
                |d, n, _| format!("stp {c}{d}, {c}{n}, [sp{o}]"),
            );
        }
        b.ok(
            |a, d, n, _| a.ldp_fp(vs, vr(d), vr(n), Addr::Pre(x(2), unit)),
            |d, n, _| format!("ldp {c}{d}, {c}{n}, [x2, #{unit}]!"),
        );
        b.ok(
            |a, d, n, _| a.stp_fp(vs, vr(d), vr(n), Addr::Post(x(2), -unit)),
            |d, n, _| format!("stp {c}{d}, {c}{n}, [x2], #{}", -unit),
        );
        b.err(|a, d, n, _| a.ldnp_fp(vs, vr(d), vr(n), Reg::ZR, 0));
    }
    // Single `q` registers in every addressing form.
    for off in [0i64, 16, 65520, 1, -256, 255] {
        let o = if off == 0 { String::new() } else { format!(", #{off}") };
        let name = |l: &str| if off % 16 == 0 && off >= 0 { l.to_string() } else { format!("{}ur", &l[..2]) };
        b.ok(
            |a, d, _, _| a.ldr_fp(VecSize::Q, vr(d), Addr::Off(x(1), off)),
            |d, _, _| format!("{} q{d}, [x1{o}]", name("ldr")),
        );
        b.ok(
            |a, d, _, _| a.str_fp(VecSize::Q, vr(d), Addr::Off(Reg::SP, off)),
            |d, _, _| format!("{} q{d}, [sp{o}]", name("str")),
        );
    }
    b.ok(
        |a, d, _, _| a.ldr_fp(VecSize::Q, vr(d), Addr::Pre(x(1), -16)),
        |d, _, _| format!("ldr q{d}, [x1, #-16]!"),
    );
    b.ok(
        |a, d, _, _| a.str_fp(VecSize::Q, vr(d), Addr::Post(x(1), 16)),
        |d, _, _| format!("str q{d}, [x1], #16"),
    );
    let idx = |ext, scaled| Addr::Idx { base: x(1), index: x(2), ext, scaled };
    b.ok(
        |a, d, _, _| a.ldr_fp(VecSize::Q, vr(d), idx(IdxExt::Lsl, false)),
        |d, _, _| format!("ldr q{d}, [x1, x2]"),
    );
    b.ok(
        |a, d, _, _| a.str_fp(VecSize::Q, vr(d), idx(IdxExt::Lsl, true)),
        |d, _, _| format!("str q{d}, [x1, x2, lsl #4]"),
    );
    b.ok(
        |a, d, _, _| a.ldr_fp(VecSize::Q, vr(d), idx(IdxExt::Uxtw, true)),
        |d, _, _| format!("ldr q{d}, [x1, w2, uxtw #4]"),
    );
    b.ok(
        |a, d, _, _| a.str_fp(VecSize::Q, vr(d), idx(IdxExt::Sxtw, false)),
        |d, _, _| format!("str q{d}, [x1, w2, sxtw]"),
    );
    b.run("pairs and quad registers");
}

#[test]
fn prefetch() {
    let mut b = Batch::new();
    let kinds = [(PrefKind::Pld, "pld"), (PrefKind::Pli, "pli"), (PrefKind::Pst, "pst")];
    let levels = [(PrefLevel::L1, "l1"), (PrefLevel::L2, "l2"), (PrefLevel::L3, "l3")];
    for (kind, k) in kinds {
        for (level, l) in levels {
            for (stream, p) in [(false, "keep"), (true, "strm")] {
                let op = Prefetch { kind, level, stream };
                b.ok(|a, _, _, _| a.prfm(op, Addr::Off(x(1), 0)), |_, _, _| format!("prfm {k}{l}{p}, [x1]"));
            }
        }
    }
    let op = Prefetch { kind: PrefKind::Pld, level: PrefLevel::L1, stream: false };
    let ps = Prefetch { kind: PrefKind::Pst, level: PrefLevel::L3, stream: true };
    let offs = [
        (8i64, "prfm pldl1keep, [x1, #8]"),
        (32760, "prfm pldl1keep, [x1, #32760]"),
        (3, "prfum pldl1keep, [x1, #3]"),
        (-256, "prfum pldl1keep, [x1, #-256]"),
        (255, "prfum pldl1keep, [x1, #255]"),
    ];
    for (o, text) in offs {
        b.ok(|a, _, _, _| a.prfm(op, Addr::Off(x(1), o)), |_, _, _| text.to_string());
    }
    b.ok(|a, _, _, _| a.prfm(ps, Addr::Off(Reg::SP, 16)), |_, _, _| "prfm pstl3strm, [sp, #16]".into());
    b.err(|a, _, _, _| a.prfm(op, Addr::Off(x(1), 32768)));
    b.err(|a, _, _, _| a.prfm(op, Addr::Off(x(1), -257)));
    b.err(|a, _, _, _| a.prfm(op, Addr::Pre(x(1), 8)));
    b.err(|a, _, _, _| a.prfm(op, Addr::Post(x(1), 8)));
    let idx = |ext, scaled| Addr::Idx { base: x(1), index: x(2), ext, scaled };
    b.ok(|a, _, _, _| a.prfm(op, idx(IdxExt::Lsl, false)), |_, _, _| "prfm pldl1keep, [x1, x2]".into());
    b.ok(
        |a, _, _, _| a.prfm(op, idx(IdxExt::Lsl, true)),
        |_, _, _| "prfm pldl1keep, [x1, x2, lsl #3]".into(),
    );
    b.ok(
        |a, _, _, _| a.prfm(ps, idx(IdxExt::Uxtw, false)),
        |_, _, _| "prfm pstl3strm, [x1, w2, uxtw]".into(),
    );
    b.ok(
        |a, _, _, _| a.prfm(ps, idx(IdxExt::Sxtw, true)),
        |_, _, _| "prfm pstl3strm, [x1, w2, sxtw #3]".into(),
    );
    b.ok(
        |a, _, _, _| a.prfm(op, idx(IdxExt::Sxtx, false)),
        |_, _, _| "prfm pldl1keep, [x1, x2, sxtx]".into(),
    );
    b.run("prefetch");
}
