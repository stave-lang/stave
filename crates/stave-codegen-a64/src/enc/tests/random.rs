//! Randomized batches (fixed seed): load/store offsets, add/sub and logical immediates, pairs,
//! and register combinations, disassembled by one objdump call and compared with expected text.

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
    fn reg(&mut self) -> Reg {
        Reg::x(self.below(31) as u8).unwrap()
    }
    fn reg_or_sp(&mut self) -> Reg {
        if self.below(6) == 0 { Reg::SP } else { self.reg() }
    }
    fn width(&mut self) -> Width {
        if self.below(2) == 0 { Width::W } else { Width::X }
    }
}

fn addr_text(base: Reg, off: i64) -> String {
    let b = base.name(Width::X);
    if off == 0 { format!("[{b}]") } else { format!("[{b}, #{off}]") }
}

#[test]
fn random_loads_stores() {
    let mut r = Rng(0x0123_4567_89ab_cdef);
    let mut a = Asm::new();
    let mut want = Vec::new();
    let sizes = [(MemSize::B, "b", 0), (MemSize::H, "h", 1), (MemSize::W, "", 2), (MemSize::X, "", 3)];
    for _ in 0..3000 {
        let (sz, suf, scale) = sizes[r.below(4) as usize];
        let rt = r.reg();
        let base = r.reg_or_sp();
        let tn = if sz == MemSize::X { rt.name(Width::X) } else { rt.name(Width::W) };
        let store = r.below(2) == 0;
        let op = if store { "str" } else { "ldr" };
        match r.below(4) {
            0 => {
                // scaled unsigned offset
                let off = (r.below(4096) as i64) << scale;
                if store { a.str(sz, rt, Addr::Off(base, off)) } else { a.ldr(sz, rt, Addr::Off(base, off)) }
                    .unwrap();
                want.push(format!("{op}{suf} {tn}, {}", addr_text(base, off)));
            }
            1 => {
                // unscaled: negative or misaligned
                let mut off = r.below(512) as i64 - 256;
                if off >= 0 && off % (1 << scale) == 0 {
                    off = -off - 1;
                }
                if store { a.str(sz, rt, Addr::Off(base, off)) } else { a.ldr(sz, rt, Addr::Off(base, off)) }
                    .unwrap();
                let u = if store { "stur" } else { "ldur" };
                want.push(format!("{u}{suf} {tn}, [{}, #{off}]", base.name(Width::X)));
            }
            2 => {
                let off = r.below(512) as i64 - 256;
                let pre = r.below(2) == 0;
                let base = if base.num() == rt.num() { Reg::SP } else { base };
                let ad = if pre { Addr::Pre(base, off) } else { Addr::Post(base, off) };
                if store { a.str(sz, rt, ad) } else { a.ldr(sz, rt, ad) }.unwrap();
                let b = base.name(Width::X);
                want.push(if pre {
                    format!("{op}{suf} {tn}, [{b}, #{off}]!")
                } else {
                    format!("{op}{suf} {tn}, [{b}], #{off}")
                });
            }
            _ => {
                // pairs (W or X)
                let w = if sz == MemSize::X || sz == MemSize::H { Width::X } else { Width::W };
                let unit = if w == Width::X { 8 } else { 4 };
                let off = (r.below(128) as i64 - 64) * unit;
                let mut rt2 = r.reg();
                if rt2 == rt {
                    rt2 = Reg::x((rt.num() + 1) % 31).unwrap();
                }
                let mode = r.below(3);
                let base = if mode != 0 && (base.num() == rt.num() || base.num() == rt2.num()) {
                    Reg::SP
                } else {
                    base
                };
                let ad = match mode {
                    0 => Addr::Off(base, off),
                    1 => Addr::Pre(base, off),
                    _ => Addr::Post(base, off),
                };
                if store { a.stp(w, rt, rt2, ad) } else { a.ldp(w, rt, rt2, ad) }.unwrap();
                let p = if store { "stp" } else { "ldp" };
                let (n1, n2, b) = (rt.name(w), rt2.name(w), base.name(Width::X));
                want.push(match mode {
                    0 => format!("{p} {n1}, {n2}, {}", addr_text(base, off)),
                    1 => format!("{p} {n1}, {n2}, [{b}, #{off}]!"),
                    _ => format!("{p} {n1}, {n2}, [{b}], #{off}"),
                });
            }
        }
    }
    check_bytes(&a.finish().unwrap().code, &want);
}

/// A random encodable bitmask for `bits`: a rotated run of ones in a repeated element.
fn bitmask(r: &mut Rng, bits: u32) -> u64 {
    let esize = [2u32, 4, 8, 16, 32, 64][r.below(if bits == 32 { 5 } else { 6 }) as usize];
    let ones = 1 + r.below(u64::from(esize - 1)) as u32;
    let rot = r.below(u64::from(esize)) as u32;
    let mask = if esize == 64 { u64::MAX } else { (1u64 << esize) - 1 };
    let run = (1u64 << ones) - 1;
    let elt = if rot == 0 { run } else { ((run >> rot) | (run << (esize - rot))) & mask };
    let mut v = 0u64;
    let mut i = 0;
    while i < bits {
        v |= elt << i;
        i += esize;
    }
    v
}

#[test]
fn random_immediates() {
    let mut r = Rng(0xfeed_face_cafe_beef);
    let mut a = Asm::new();
    let mut want = Vec::new();
    for _ in 0..3000 {
        let w = r.width();
        match r.below(4) {
            0 => {
                let (rd, rn) = (r.reg_or_sp(), r.reg_or_sp());
                let sub = r.below(2) == 0;
                let shifted = r.below(3) == 0;
                let imm12 = r.below(4096);
                let imm = if shifted { imm12 << 12 } else { imm12 };
                let op = if sub { AddSub::Sub } else { AddSub::Add };
                a.addsub_imm(op, w, rd, rn, imm).unwrap();
                let (d, n) = (rd.name(w), rn.name(w));
                want.push(if !sub && imm == 0 && !shifted && (rd.is_sp() || rn.is_sp()) {
                    format!("mov {d}, {n}")
                } else if shifted {
                    format!("{} {d}, {n}, #{imm12:#x}, lsl #12", if sub { "sub" } else { "add" })
                } else {
                    format!("{} {d}, {n}, #{imm12:#x}", if sub { "sub" } else { "add" })
                });
            }
            1 => {
                let ops =
                    [(Logic::And, "and"), (Logic::Eor, "eor"), (Logic::Ands, "ands"), (Logic::Orr, "orr")];
                let (op, name) = ops[r.below(4) as usize];
                let rd = if op == Logic::Ands { r.reg() } else { r.reg_or_sp() };
                let rn = r.reg();
                let imm = bitmask(&mut r, w.bits());
                a.logic_imm(op, w, rd, rn, imm).unwrap();
                want.push(format!("{name} {}, {}, #{imm:#x}", rd.name(w), rn.name(w)));
            }
            2 => {
                let rd = r.reg();
                let shift = 16 * r.below(u64::from(w.bits() / 16)) as u32;
                let imm16 = r.below(0x10000);
                a.movk(w, rd, imm16, shift).unwrap();
                want.push(if shift == 0 {
                    format!("movk {}, #{imm16:#x}", rd.name(w))
                } else {
                    format!("movk {}, #{imm16:#x}, lsl #{shift}", rd.name(w))
                });
            }
            _ => {
                let (rd, rn, rm) = (r.reg(), r.reg(), r.reg());
                let ops =
                    [(Logic::And, "and"), (Logic::Bic, "bic"), (Logic::Eor, "eor"), (Logic::Eon, "eon")];
                let (op, name) = ops[r.below(4) as usize];
                let sh = [(Shift::Lsl, "lsl"), (Shift::Lsr, "lsr"), (Shift::Asr, "asr"), (Shift::Ror, "ror")];
                let (s, sn) = sh[r.below(4) as usize];
                let amt = r.below(u64::from(w.bits())) as u32;
                a.logic_reg(op, w, rd, rn, rm, s, amt).unwrap();
                let base = format!("{name} {}, {}, {}", rd.name(w), rn.name(w), rm.name(w));
                want.push(if amt == 0 && s == Shift::Lsl { base } else { format!("{base}, {sn} #{amt}") });
            }
        }
    }
    check_bytes(&a.finish().unwrap().code, &want);
}
