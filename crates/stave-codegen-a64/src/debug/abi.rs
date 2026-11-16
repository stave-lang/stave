//! The Stave calling convention (module doc of [`super`]): argument and result locations.
//!
//! Vectors and masks (at most 128 bits) travel like AAPCS64 short vectors. An argument takes the
//! next free `v0..v7` (the counter floats use) as a whole register: lanes in order from bit 0,
//! bits past the vector's size unspecified; a mask is its full-width lanes, each all ones or
//! zero. Without a free register it goes to the stack by value: 16 bytes at a 16-byte aligned
//! offset for a 128-bit vector, one 8-byte word for a narrower one. A vector result is in `v0`.
//! No V register is preserved across a call as a vector: `v8–v15` keep their low 64 bits only,
//! so a vector live across a call is in its frame home.

use super::ty::Ty;

/// Where one argument travels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Loc {
    /// `x<n>`.
    X(u8),
    /// `v<n>` (`s`/`d`; the whole register for a vector).
    V(u8),
    /// `x<n>, x<n+1>` (fat pointer).
    XPair(u8),
    /// Stack word(s) at this offset from the outgoing `sp` (incoming: `x29 + 16 + off`).
    Stack(u32),
}

/// One argument: its location, and whether it is an aggregate passed by reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Arg {
    pub loc: Loc,
    pub by_ref: bool,
}

/// Assigns locations in order (AAPCS64 NGRN / NSRN / NSAA). Returns the stack bytes used.
pub(crate) fn classify(params: impl Iterator<Item = Ty>, out: &mut Vec<Arg>) -> u32 {
    out.clear();
    let (mut ngrn, mut nsrn, mut nsaa) = (0u8, 0u8, 0u32);
    for t in params {
        let a = match t {
            Ty::Sc(s) if s.is_fp() => {
                if nsrn < 8 {
                    nsrn += 1;
                    Arg { loc: Loc::V(nsrn - 1), by_ref: false }
                } else {
                    nsaa += 8;
                    Arg { loc: Loc::Stack(nsaa - 8), by_ref: false }
                }
            }
            Ty::Vec(v) => {
                if nsrn < 8 {
                    nsrn += 1;
                    Arg { loc: Loc::V(nsrn - 1), by_ref: false }
                } else if v.size == 16 {
                    nsaa = nsaa.div_ceil(16) * 16 + 16;
                    Arg { loc: Loc::Stack(nsaa - 16), by_ref: false }
                } else {
                    nsaa += 8;
                    Arg { loc: Loc::Stack(nsaa - 8), by_ref: false }
                }
            }
            Ty::Fat => {
                if ngrn <= 6 {
                    ngrn += 2;
                    Arg { loc: Loc::XPair(ngrn - 2), by_ref: false }
                } else {
                    ngrn = 8;
                    nsaa += 16;
                    Arg { loc: Loc::Stack(nsaa - 16), by_ref: false }
                }
            }
            _ => {
                let by_ref = matches!(t, Ty::Mem { .. });
                if ngrn < 8 {
                    ngrn += 1;
                    Arg { loc: Loc::X(ngrn - 1), by_ref }
                } else {
                    nsaa += 8;
                    Arg { loc: Loc::Stack(nsaa - 8), by_ref }
                }
            }
        };
        out.push(a);
    }
    nsaa
}

/// Where a result travels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Ret {
    None,
    X0,
    V0,
    /// A vector or mask in `v0`.
    Q0,
    X0X1,
    /// Written through the `x8` buffer.
    Indirect,
}

pub(crate) fn ret(t: Option<Ty>) -> Ret {
    match t {
        None => Ret::None,
        Some(Ty::Sc(s)) if s.is_fp() => Ret::V0,
        Some(Ty::Sc(_)) => Ret::X0,
        Some(Ty::Fat) => Ret::X0X1,
        Some(Ty::Vec(_)) => Ret::Q0,
        Some(_) => Ret::Indirect,
    }
}
