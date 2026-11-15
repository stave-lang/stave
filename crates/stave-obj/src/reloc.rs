//! Relocation arithmetic (the ELF psABI formulas for the kinds in [`RelocKind`]).

use crate::RelocKind;

/// Why a relocation could not be applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelocFail {
    OutOfBounds,
    Overflow,
}

fn rd32(buf: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([buf[at], buf[at + 1], buf[at + 2], buf[at + 3]])
}

fn wr32(buf: &mut [u8], at: usize, v: u32) {
    buf[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

/// Patches the field at `buf[at..]` for target `s + a` seen from field address `p`. Instruction
/// fields keep their opcode bits and replace only the immediate bits.
pub fn apply_reloc(
    kind: RelocKind,
    buf: &mut [u8],
    at: usize,
    p: u64,
    s: u64,
    a: i64,
) -> Result<(), RelocFail> {
    let end = at.checked_add(kind.width()).ok_or(RelocFail::OutOfBounds)?;
    if end > buf.len() {
        return Err(RelocFail::OutOfBounds);
    }
    let target = (s as i128) + (a as i128);
    let pc = p as i128;
    match kind {
        RelocKind::Abs64 | RelocKind::Abs64Body => {
            if !(0..=u64::MAX as i128).contains(&target) {
                return Err(RelocFail::Overflow);
            }
            buf[at..end].copy_from_slice(&(target as u64).to_le_bytes());
        }
        RelocKind::X86Pc32 => {
            let d = target - pc;
            let v = i32::try_from(d).map_err(|_| RelocFail::Overflow)?;
            wr32(buf, at, v as u32);
        }
        RelocKind::A64Branch26 => {
            let d = target - pc;
            if d & 3 != 0 || !(-(1i128 << 27)..(1i128 << 27)).contains(&d) {
                return Err(RelocFail::Overflow);
            }
            let ins = rd32(buf, at) & !0x03ff_ffff;
            wr32(buf, at, ins | ((d >> 2) as u32 & 0x03ff_ffff));
        }
        RelocKind::A64AdrPage21 => {
            let d = (target & !0xfff) - (pc & !0xfff);
            let pages = d >> 12;
            if !(-(1i128 << 20)..(1i128 << 20)).contains(&pages) {
                return Err(RelocFail::Overflow);
            }
            let imm = pages as u32 & 0x1f_ffff;
            let ins = rd32(buf, at) & !((3 << 29) | (0x7ffff << 5));
            wr32(buf, at, ins | ((imm & 3) << 29) | ((imm >> 2) << 5));
        }
        RelocKind::A64AddLo12 => {
            let lo = (target & 0xfff) as u32;
            let ins = rd32(buf, at) & !(0xfff << 10);
            wr32(buf, at, ins | (lo << 10));
        }
        RelocKind::A64LdstLo12 { shift } => {
            if shift > 4 {
                return Err(RelocFail::Overflow);
            }
            let lo = (target & 0xfff) as u32;
            if lo & ((1 << shift) - 1) != 0 {
                return Err(RelocFail::Overflow);
            }
            let ins = rd32(buf, at) & !(0xfff << 10);
            wr32(buf, at, ins | ((lo >> shift) << 10));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pc32() {
        let mut b = [0xe8, 0, 0, 0, 0];
        apply_reloc(RelocKind::X86Pc32, &mut b, 1, 0x1001, 0x2000, -4).unwrap();
        assert_eq!(i32::from_le_bytes([b[1], b[2], b[3], b[4]]), 0x2000 - 4 - 0x1001);
        assert_eq!(apply_reloc(RelocKind::X86Pc32, &mut b, 1, 0, 1 << 40, 0), Err(RelocFail::Overflow));
        assert_eq!(apply_reloc(RelocKind::X86Pc32, &mut b, 2, 0, 0, 0), Err(RelocFail::OutOfBounds));
    }

    #[test]
    fn a64() {
        let mut b = 0x9400_0000u32.to_le_bytes();
        apply_reloc(RelocKind::A64Branch26, &mut b, 0, 0x1000, 0x0ff8, 0).unwrap();
        assert_eq!(u32::from_le_bytes(b), 0x97ff_fffe);
        let mut b = 0x9000_0000u32.to_le_bytes();
        apply_reloc(RelocKind::A64AdrPage21, &mut b, 0, 0x40_1234, 0x40_4000, 0).unwrap();
        // 3 pages: immlo = 3, immhi = 0
        assert_eq!(u32::from_le_bytes(b), 0x9000_0000 | (3 << 29));
        let mut b = 0xf940_0000u32.to_le_bytes();
        apply_reloc(RelocKind::A64LdstLo12 { shift: 3 }, &mut b, 0, 0, 0x40_0018, 0).unwrap();
        assert_eq!(u32::from_le_bytes(b), 0xf940_0000 | (3 << 10));
        assert_eq!(
            apply_reloc(RelocKind::A64LdstLo12 { shift: 3 }, &mut b, 0, 0, 0x40_0014, 0),
            Err(RelocFail::Overflow)
        );
    }
}
