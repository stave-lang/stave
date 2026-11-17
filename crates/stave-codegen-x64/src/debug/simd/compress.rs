//! `compress` / `expand` with shuffle-control tables in `.rodata`: the mask bits index a table of
//! lane indices, and one shuffle (`vpermd` for 256-bit vectors of 32- and 64-bit lanes, `pshufb`
//! for 128 bits) moves every lane at once. A table index whose top bit is set selects a zero
//! lane. A module gets the tables its `compress` / `expand` shapes need and no others.
//!
//! - 32/64-bit lanes, 256 bits: `vpmovsxbd` of an 8-byte entry gives the `vpermd` indices (dword
//!   pairs for 64-bit lanes); the sign of an index lane clears the lane.
//! - 16/32/64-bit lanes, up to 128 bits: a 16-byte `pshufb` control per mask.
//! - 8-bit lanes: 8 lanes per entry of the 8-lane table; the high 8 lanes of a 16-lane vector
//!   are shuffled on their own and slid up by the count of the low ones.
//! - 8/16-bit lanes, 256 bits: each 128-bit half as above, the high half slid by the byte count
//!   of the low one.

use stave_ir::simd::{self as vs, VecOp};
use stave_ir::{Module, Op, Operand, TypeId, Value};
use stave_obj::{DataObj, ImageBuilder, SymbolId};

use super::super::func::*;
use super::super::{Feats, Res, bad};
use super::{Sh, y};
use crate::enc::*;

/// A table; `expand`: the inverse permutation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Tab {
    /// 256 entries of 8 bytes: per 8-bit mask, the source lane of each of 8 lanes.
    Lanes8 { expand: bool },
    /// 16 entries of 8 bytes: per 4-bit mask, the source dword pair of each of four 64-bit lanes.
    Pairs4 { expand: bool },
    /// `2^(16 / w)` entries of 16 bytes: the `pshufb` control of a 128-bit vector of `w`-byte
    /// lanes (`w` = 2, 4, 8).
    Shuf { w: u32, expand: bool },
    /// `0x80` × 16, 0..16, `0x80` × 16, 0..16: `pshufb` controls for a byte shift by `k` ≤ 16.
    /// The 16 bytes at `16 - k` shift a vector up by `k`, those at `32 - k` give the bytes
    /// shifted out at the top; at `16 + k` and `32 + k`, the low and the high half's share of a
    /// 32-byte vector shifted down by `k`.
    Slide,
}

pub(crate) const NTAB: usize = 11;

impl Tab {
    fn index(self) -> usize {
        match self {
            Tab::Lanes8 { expand } => usize::from(expand),
            Tab::Pairs4 { expand } => 2 + usize::from(expand),
            Tab::Shuf { w, expand } => 4 + 2 * (w.trailing_zeros() as usize - 1) + usize::from(expand),
            Tab::Slide => 10,
        }
    }

    fn all() -> [Tab; NTAB] {
        let mut t = [Tab::Slide; NTAB];
        for expand in [false, true] {
            for x in [Tab::Lanes8 { expand }, Tab::Pairs4 { expand }]
                .into_iter()
                .chain([2, 4, 8].map(|w| Tab::Shuf { w, expand }))
            {
                t[x.index()] = x;
            }
        }
        t
    }

    fn name(self) -> String {
        let kind = |e: bool| if e { "expand" } else { "compress" };
        match self {
            Tab::Lanes8 { expand } => format!("__stave_{}.lanes8", kind(expand)),
            Tab::Pairs4 { expand } => format!("__stave_{}.pairs4", kind(expand)),
            Tab::Shuf { w, expand } => format!("__stave_{}.shuf{}", kind(expand), w * 8),
            Tab::Slide => "__stave_compress.slide".to_owned(),
        }
    }

    fn bytes(self) -> Vec<u8> {
        match self {
            Tab::Lanes8 { expand } => entries(8, 1, expand),
            Tab::Pairs4 { expand } => entries(4, 2, expand),
            Tab::Shuf { w, expand } => entries(16 / w, w, expand),
            Tab::Slide => {
                let mut v = Vec::with_capacity(64);
                for _ in 0..2 {
                    v.extend([0x80; 16]);
                    v.extend(0..16u8);
                }
                v
            }
        }
    }
}

/// One entry per `n`-bit mask, `unit` index bytes per lane: lane `p` of the result takes source
/// lane `s` as bytes `s * unit ..`, or `0x80` (zero) where it takes none.
fn entries(n: u32, unit: u32, expand: bool) -> Vec<u8> {
    let mut v = Vec::with_capacity(((n * unit) as usize) << n);
    for m in 0..1u32 << n {
        let mut src = vec![None; n as usize];
        let mut k = 0;
        for j in 0..n {
            if m >> j & 1 != 0 {
                if expand {
                    src[j as usize] = Some(k);
                } else {
                    src[k as usize] = Some(j);
                }
                k += 1;
            }
        }
        for s in src {
            for b in 0..unit {
                v.push(s.map_or(0x80, |s| (s * unit + b) as u8));
            }
        }
    }
    v
}

/// The tables `compress` (`expand`) of `n` lanes of `w` bytes uses, or `None`: lane by lane.
/// The 256-bit forms shuffle with AVX2 instructions.
fn tabs(avx2: bool, w: u32, n: u32, expand: bool) -> Option<Vec<Tab>> {
    let size = w * n;
    if size > 32 || (size > 16 && !avx2) {
        return None;
    }
    Some(match (w, size > 16) {
        (4, true) => vec![Tab::Lanes8 { expand }],
        (8, true) => vec![Tab::Pairs4 { expand }],
        (2, true) => vec![Tab::Shuf { w, expand }, Tab::Slide],
        (1, true) => vec![Tab::Lanes8 { expand }, Tab::Slide],
        (1, false) if n > 8 && !expand => vec![Tab::Lanes8 { expand }, Tab::Slide],
        (1, false) => vec![Tab::Lanes8 { expand }],
        _ => vec![Tab::Shuf { w, expand }],
    })
}

/// Defines the tables of the module's `vec.compress` / `vec.expand` shapes, each as the
/// function it is in lowers it (`baseline`: the module's features).
pub(crate) fn define(
    m: &Module,
    baseline: stave_target::FeatureSet,
    b: &mut ImageBuilder,
) -> Res<[Option<SymbolId>; NTAB]> {
    let mut used = [false; NTAB];
    for f in &m.funcs {
        let Some(body) = &f.body else { continue };
        let avx2 = Feats::of(baseline.union(f.features)).avx2;
        for (op, ty) in body.insts.op.iter().zip(&body.insts.ty) {
            let Op::Vec(op @ (VecOp::Compress | VecOp::Expand)) = op else { continue };
            let Some((_, lane, n)) = ty.and_then(|t| vs::vec_shape(&m.types, t)) else { continue };
            for t in tabs(avx2, lane.bits() / 8, n, *op == VecOp::Expand).unwrap_or_default() {
                used[t.index()] = true;
            }
        }
    }
    let mut syms = [None; NTAB];
    for t in Tab::all() {
        if used[t.index()] {
            // 16: a legacy SSE `pshufb` reads a 16-byte entry as an aligned memory operand
            syms[t.index()] = Some(b.rodata(&t.name(), DataObj::new(t.bytes(), 16))?);
        }
    }
    Ok(syms)
}

/// Placeholder symbols for every table, from `first` on (codegen without an image).
pub(crate) fn placeholders(first: u32) -> [Option<SymbolId>; NTAB] {
    std::array::from_fn(|i| Some(SymbolId(first + i as u32)))
}

impl Cg<'_, '_> {
    /// `g` = the address of table `t`.
    fn tab_addr(&mut self, t: Tab, g: Gpr) -> Res<()> {
        let s = self.cx.ctab[t.index()].ok_or_else(|| bad("internal: no compress table in the image"))?;
        Ok(self.a.lea(Size::Q, g, Mem::rip_sym(s, 0))?)
    }

    /// `d = pshufb(v, ctl)` (128 bits).
    fn shuf(&mut self, d: Xmm, v: Xmm, ctl: impl Into<XRm> + Into<VRm> + Copy) -> Res<()> {
        if self.vex() {
            self.a.vex_bin(VBin::Pshufb, d, v, ctl)?;
        } else {
            self.vmov(false, d, v)?;
            self.a.sse_bin(VBin::Pshufb, d, ctl)?;
        }
        Ok(())
    }

    /// `x = [m]`, 8 bytes, zero above.
    fn movq_load(&mut self, x: Xmm, m: Mem) -> Res<()> {
        if self.vex() {
            self.a.vex_un(VUn::Movq, x, m)?
        } else {
            self.a.sse_un(VUn::Movq, x, m)?
        }
        Ok(())
    }

    /// `x = [m]`, 16 bytes, unaligned.
    fn movdqu_load(&mut self, x: Xmm, m: Mem) -> Res<()> {
        if self.vex() {
            self.a.vex_un(VUn::Movdqu, x, m)?
        } else {
            self.a.sse_un(VUn::Movdqu, x, m)?
        }
        Ok(())
    }

    /// The table-driven form, where the shape has one (else `false`: nothing emitted).
    pub(super) fn v_compress_packed(
        &mut self,
        expand: bool,
        a: Operand,
        m: Operand,
        rt: TypeId,
        rv: Value,
    ) -> Res<bool> {
        let sh = self.vshape(rt)?;
        if tabs(self.fx.avx2, sh.w, sh.n, expand).is_none() {
            return Ok(false);
        }
        let mx = self.vread(m, sh.size)?;
        let bits = self.alloc_g()?;
        self.vmask_bits(bits, mx, sh)?;
        if sh.wide() && sh.w >= 4 {
            let x = self.vget(a, sh.size)?;
            let t = if sh.w == 4 { Tab::Lanes8 { expand } } else { Tab::Pairs4 { expand } };
            self.tab_addr(t, S1)?;
            self.vform(true)?;
            self.a.vex_un(VUn::Pmovsxbd, y(XS0), Mem::index(S1, bits, Scale::S8, 0))?;
            self.a.vex_bin(VBin::Permd, y(x), y(XS0), y(x))?;
            self.a.vex_shift_i(VShiftI::Psrad, y(XS0), y(XS0), 31)?;
            self.a.vex_bin(VBin::Pandn, y(x), y(XS0), y(x))?;
            self.vput(rv, x, sh.size)?;
            return Ok(true);
        }
        if !sh.wide() {
            let v = self.vread(a, sh.size)?;
            let d = self.alloc_x()?;
            self.part128(expand, sh, d, v, bits)?;
            self.vput(rv, d, sh.size)?;
            return Ok(true);
        }
        // 8/16-bit lanes in two 128-bit halves
        let v = self.vread(a, sh.size)?;
        let nh = sh.n / 2;
        let half = Sh { n: nh, size: 16, ..sh };
        let (hb, cnt) = (self.alloc_g()?, self.alloc_g()?);
        self.a.mov_rr(Size::D, hb, bits);
        self.a.shift_ri(ShiftOp::Shr, Size::D, hb, nh as u8)?;
        self.a.alu_ri(AluOp::And, Size::D, bits, (1i64 << nh) - 1)?;
        self.a.popcnt(Size::D, cnt, bits)?;
        let scale = if sh.w == 2 { Scale::S2 } else { Scale::S1 };
        let (d, vh, rh) = (self.alloc_x()?, self.alloc_x()?, self.alloc_x()?);
        self.vhigh(vh, v)?;
        let hi = if expand {
            self.part128(true, half, d, v, bits)?;
            // the source lanes from the low count on: `v` shifted down by their bytes
            self.tab_addr(Tab::Slide, S1)?;
            self.movdqu_load(XS0, Mem::index(S1, cnt, scale, 16))?;
            self.shuf(rh, v, XS0)?;
            self.movdqu_load(XS0, Mem::index(S1, cnt, scale, 32))?;
            self.shuf(XS1, vh, XS0)?;
            self.vb(VBin::Por, false, rh, XS1)?;
            self.part128(true, half, vh, rh, hb)?;
            vh
        } else {
            self.part128(false, half, d, v, bits)?;
            self.part128(false, half, rh, vh, hb)?;
            // the high half's lanes go up by the low count, across the 128-bit boundary
            self.a.unary(UnaryOp::Neg, Size::Q, cnt)?;
            self.tab_addr(Tab::Slide, S1)?;
            self.movdqu_load(XS0, Mem::index(S1, cnt, scale, 16))?;
            self.shuf(XS1, rh, XS0)?;
            self.vb(VBin::Por, false, d, XS1)?;
            self.movdqu_load(XS0, Mem::index(S1, cnt, scale, 32))?;
            self.shuf(rh, rh, XS0)?;
            rh
        };
        self.vform(true)?;
        self.a.vex_bin_i(VBinI::Inserti128, y(d), y(d), hi, 1)?;
        self.vput(rv, d, sh.size)?;
        Ok(true)
    }

    /// `d` = the compress (expand) of the vector of at most 128 bits in `v` under the mask bits
    /// `bits` (clobbered), zero above its lanes. Clobbers `XS0`, `XS1`, `S0`, `S1`.
    fn part128(&mut self, expand: bool, sh: Sh, d: Xmm, v: Xmm, bits: Gpr) -> Res<()> {
        if sh.w >= 2 {
            self.tab_addr(Tab::Shuf { w: sh.w, expand }, S1)?;
            self.a.shift_ri(ShiftOp::Shl, Size::D, bits, 4)?;
            // 16-byte entries of a 16-aligned table: an aligned operand for legacy SSE
            return self.shuf(d, v, Mem::index(S1, bits, Scale::S1, 0));
        }
        self.tab_addr(Tab::Lanes8 { expand }, S1)?;
        if sh.n <= 8 {
            self.movq_load(XS0, Mem::index(S1, bits, Scale::S8, 0))?;
            self.shuf(d, v, XS0)?;
            // the entry's upper 8 control bytes were zero: they picked lane 0
            return self.vu(VUn::Movq, false, d, d);
        }
        let hi = self.alloc_g()?;
        self.a.mov_rr(Size::D, hi, bits);
        self.a.shift_ri(ShiftOp::Shr, Size::D, hi, 8)?;
        self.a.movzx(Size::D, bits, Size::B, bits)?;
        if expand {
            // high entry + the low count in every byte: the source lanes after the low ones
            // (an inactive `0x80` stays at or above `0x80`)
            let c = self.alloc_g()?;
            self.a.popcnt(Size::D, c, bits)?;
            self.a.mov_ri(Size::Q, S0, 0x0101_0101_0101_0101)?;
            self.a.imul_rr(Size::Q, c, S0)?;
            self.a.alu_rm(AluOp::Add, Size::Q, c, Mem::index(S1, hi, Scale::S8, 0))?;
            self.vmovd_xg(Size::Q, XS1, c)?;
            self.movq_load(XS0, Mem::index(S1, bits, Scale::S8, 0))?;
            self.vb(VBin::Punpcklqdq, false, XS0, XS1)?;
            return self.shuf(d, v, XS0);
        }
        self.movq_load(XS0, Mem::index(S1, bits, Scale::S8, 0))?;
        self.movq_load(XS1, Mem::index(S1, hi, Scale::S8, 0))?;
        self.shuf(d, v, XS0)?;
        self.vu(VUn::Movq, false, d, d)?;
        let t = self.alloc_x()?;
        if self.vex() {
            self.a.vex_shift_i(VShiftI::Psrldq, t, v, 8)?;
        } else {
            self.vmov(false, t, v)?;
            self.vsh(VShiftI::Psrldq, false, t, 8)?;
        }
        self.shuf(t, t, XS1)?;
        self.vu(VUn::Movq, false, t, t)?;
        self.a.popcnt(Size::D, bits, bits)?;
        self.a.unary(UnaryOp::Neg, Size::Q, bits)?;
        self.tab_addr(Tab::Slide, S1)?;
        self.movdqu_load(XS0, Mem::index(S1, bits, Scale::S1, 16))?;
        self.shuf(t, t, XS0)?;
        self.vb(VBin::Por, false, d, t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_examples() {
        let c = Tab::Lanes8 { expand: false }.bytes();
        assert_eq!(c.len(), 2048);
        assert_eq!(&c[0b1010_0110 * 8..][..8], &[1, 2, 5, 7, 0x80, 0x80, 0x80, 0x80]);
        let e = Tab::Lanes8 { expand: true }.bytes();
        assert_eq!(&e[0b1010_0110 * 8..][..8], &[0x80, 0, 1, 0x80, 0x80, 2, 0x80, 3]);
        let p = Tab::Pairs4 { expand: false }.bytes();
        assert_eq!(&p[0b1010 * 8..][..8], &[2, 3, 6, 7, 0x80, 0x80, 0x80, 0x80]);
        let s = Tab::Shuf { w: 4, expand: false }.bytes();
        assert_eq!(s.len(), 256);
        assert_eq!(&s[0b0100 * 16..][..8], &[8, 9, 10, 11, 0x80, 0x80, 0x80, 0x80]);
        assert_eq!(Tab::Shuf { w: 2, expand: true }.bytes().len(), 4096);
        assert_eq!(Tab::Shuf { w: 8, expand: true }.bytes().len(), 64);
        let idx: Vec<usize> = Tab::all().iter().map(|t| t.index()).collect();
        assert_eq!(idx, (0..NTAB).collect::<Vec<_>>());
    }
}
