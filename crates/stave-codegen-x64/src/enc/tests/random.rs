//! Randomized batches (fixed seed): thousands of memory operands, immediates, and register
//! combinations, disassembled by one objdump call and compared with expected Intel text.

use super::check_bytes;
use crate::enc::*;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn gpr(&mut self) -> Gpr {
        Gpr::new(self.below(16) as u8).unwrap()
    }
    /// A displacement biased towards the disp8/disp32 boundaries.
    fn disp(&mut self) -> i32 {
        match self.below(5) {
            0 => 0,
            1 => self.below(256) as i32 - 128,
            2 => [127, 128, -128, -129, 255, 256][self.below(6) as usize],
            3 => self.next() as i32,
            _ => (self.below(0x10000) as i32) - 0x8000,
        }
    }
    fn size(&mut self) -> Size {
        [Size::B, Size::W, Size::D, Size::Q][self.below(4) as usize]
    }
    fn imm(&mut self, s: Size) -> i64 {
        let v = match self.below(4) {
            0 => self.below(256) as i64 - 128,
            1 => self.next() as i64,
            2 => [0, 1, -1, 127, 128, -128, -129, 0x7fff, 0x8000, 0x7fff_ffff, -0x8000_0000]
                [self.below(11) as usize],
            _ => self.below(0x1_0000_0000) as i64,
        };
        match s {
            Size::B => v as i8 as i64,
            Size::W => v as i16 as i64,
            Size::D | Size::Q => v as i32 as i64,
        }
    }
}

fn hex(v: i64) -> String {
    if v < 0 { format!("-{:#x}", v.unsigned_abs()) } else { format!("+{v:#x}") }
}

fn ptr(s: Size) -> &'static str {
    match s {
        Size::B => "BYTE PTR ",
        Size::W => "WORD PTR ",
        Size::D => "DWORD PTR ",
        Size::Q => "QWORD PTR ",
    }
}

/// objdump's Intel rendering of a memory operand.
fn mem_text(m: &Mem) -> String {
    let Mem::Sib { base, index, disp } = *m else { unreachable!() };
    let d = i64::from(disp);
    let idx = index.map(|(i, s)| format!("{}*{}", i.name(Size::Q), 1 << (s as u8)));
    match (base, idx) {
        (None, None) => format!("ds:{:#x}", disp as u32),
        (None, Some(i)) => format!("[{i}{}]", hex(d)),
        (Some(b), i) => {
            let shown = d != 0 || b.num() & 7 == 5;
            let mut s = format!("[{}", b.name(Size::Q));
            if let Some(i) = i {
                s.push('+');
                s.push_str(&i);
            }
            if shown {
                s.push_str(&hex(d));
            }
            s.push(']');
            s
        }
    }
}

fn rand_mem(r: &mut Rng) -> Mem {
    let base = if r.below(8) == 0 { None } else { Some(r.gpr()) };
    let index = if r.below(2) == 0 {
        None
    } else {
        let mut i = r.gpr();
        if i == Gpr::RSP {
            i = Gpr::R12;
        }
        Some((i, [Scale::S1, Scale::S2, Scale::S4, Scale::S8][r.below(4) as usize]))
    };
    let mut disp = r.disp();
    if base.is_none() && index.is_none() && disp < 0 {
        disp = disp.wrapping_neg().max(0);
    }
    Mem::Sib { base, index, disp }
}

/// The value objdump prints for an immediate of operand size `s` (sign-extended to the size).
fn imm_text(v: i64, s: Size) -> String {
    let bits = s.bits();
    let m = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
    format!("{:#x}", (v as u64) & m)
}

#[test]
fn random_memory_operands() {
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    let mut a = Asm::new();
    let mut want = Vec::new();
    for _ in 0..3000 {
        let m = rand_mem(&mut r);
        let reg = r.gpr();
        let mt = mem_text(&m);
        match r.below(5) {
            0 => {
                let s = r.size();
                a.mov_rm(s, reg, m).unwrap();
                want.push(format!("mov {},{}{mt}", reg.name(s), ptr(s)));
            }
            1 => {
                let s = r.size();
                a.mov_mr(s, m, reg).unwrap();
                want.push(format!("mov {}{mt},{}", ptr(s), reg.name(s)));
            }
            2 => {
                a.lea(Size::Q, reg, m).unwrap();
                want.push(format!("lea {},{mt}", reg.name(Size::Q)));
            }
            3 => {
                a.movs(Prec::F64, Xmm::new(reg.num()).unwrap(), m).unwrap();
                want.push(format!("movsd xmm{},QWORD PTR {mt}", reg.num()));
            }
            _ => {
                let s = r.size();
                let imm = r.imm(s);
                a.alu_mi(AluOp::Cmp, s, m, imm).unwrap();
                want.push(format!("cmp {}{mt},{}", ptr(s), imm_text(imm, s)));
            }
        }
    }
    check_bytes(&a.finish().unwrap().code, &want);
}

#[test]
fn random_immediates() {
    let mut r = Rng(0x1234_5678_9abc_def1);
    let mut a = Asm::new();
    let mut want = Vec::new();
    let ops = [
        (AluOp::Add, "add"),
        (AluOp::Or, "or"),
        (AluOp::Adc, "adc"),
        (AluOp::Sbb, "sbb"),
        (AluOp::And, "and"),
        (AluOp::Sub, "sub"),
        (AluOp::Xor, "xor"),
        (AluOp::Cmp, "cmp"),
    ];
    for _ in 0..3000 {
        let reg = r.gpr();
        let s = r.size();
        match r.below(4) {
            0 => {
                let (op, n) = ops[r.below(8) as usize];
                let imm = r.imm(s);
                a.alu_ri(op, s, reg, imm).unwrap();
                want.push(format!("{n} {},{}", reg.name(s), imm_text(imm, s)));
            }
            1 => {
                let imm = match s {
                    Size::Q => r.next() as i64 >> r.below(64),
                    _ => r.imm(s),
                };
                a.mov_ri(s, reg, imm).unwrap();
                want.push(match s {
                    Size::Q if (0..=0xffff_ffff).contains(&imm) => {
                        format!("mov {},{imm:#x}", reg.name(Size::D))
                    }
                    Size::Q if i32::try_from(imm).is_ok() => format!("mov {},{:#x}", reg.name(s), imm as u64),
                    Size::Q => format!("movabs {},{:#x}", reg.name(s), imm as u64),
                    _ => format!("mov {},{}", reg.name(s), imm_text(imm, s)),
                });
            }
            2 => {
                let ops = [
                    (ShiftOp::Shl, "shl"),
                    (ShiftOp::Shr, "shr"),
                    (ShiftOp::Sar, "sar"),
                    (ShiftOp::Rol, "rol"),
                ];
                let (op, n) = ops[r.below(4) as usize];
                let k = r.below(u64::from(s.bits())) as u8;
                a.shift_ri(op, s, reg, k).unwrap();
                want.push(format!("{n} {},{k:#x}", reg.name(s)));
            }
            _ => {
                let src = r.gpr();
                let (op, n) = ops[r.below(8) as usize];
                a.alu_rr(op, s, reg, src);
                want.push(format!("{n} {},{}", reg.name(s), src.name(s)));
            }
        }
    }
    check_bytes(&a.finish().unwrap().code, &want);
}
