//! Value classes and the Stave calling convention on x86-64 (System V AMD64).
//!
//! # The Stave internal calling convention (x86-64)
//!
//! Stave functions call each other with the System V AMD64 ABI, with these definitions for Stave
//! types (C has no such types, so these are choices, not deviations):
//!
//! - **Scalars of integer class** (`iN uN isize usize bool char`, thin `&T`/`*T`, and
//!   `f16`/`bf16` held as their bits): one INTEGER eightbyte. They are passed and returned
//!   **normalized to 64 bits** (sign-extended for signed types, zero-extended otherwise; `bool` is
//!   0/1). System V leaves the upper bits of narrow arguments undefined; Stave defines them.
//! - `f32`/`f64`: one SSE eightbyte (`xmm0`–`xmm7`, return `xmm0`).
//! - **Aggregates** (tuples, structs, arrays, enums, fat pointers `&[T]`/`&str` = `(ptr, len)`, fn
//!   values = `(code, environment)`) are
//!   classified per eightbyte exactly as System V classifies a C struct with the same layout
//!   (stave-ir `layout`): an eightbyte holding any integer-class leaf is INTEGER, one holding only
//!   floats is SSE. Up to 16 bytes go in registers when all eightbytes fit in the remaining
//!   argument registers (else the whole aggregate goes on the stack); larger ones are MEMORY
//!   (copied to the argument area). Zero-sized values take no register and no stack.
//! - **Returns**: integer class in `rax`, float in `xmm0`; aggregates up to 16 bytes in
//!   `rax`/`rdx` and `xmm0`/`xmm1` by eightbyte class; larger aggregates are returned through a
//!   hidden **sret pointer**: the caller passes the address of the result memory in `rdi` (taking
//!   the first integer register) and the callee returns that address in `rax`.
//! - Registers: `rbx rbp r12–r15` callee-saved, everything else (all of `xmm0`–`xmm15`)
//!   caller-saved; `rsp` 16-byte aligned at every `call`; direction flag clear. No red zone use.
//! - **Fn-value calls** (`call.ind`) pass the value's environment word as one more INTEGER
//!   argument after the declared ones; a closure body takes it as its last param, and a fn item
//!   ignores it (System V: the caller owns the argument area).
//! - Stack arguments: 8-byte slots (aggregates rounded up to 8), first argument at the lowest
//!   address, area size rounded up to 16.

use stave_ir::layout::{Layouts, Shape};
use stave_ir::{FloatBits, TypeId, TypeKey, TypeTable};

use crate::enc::{Gpr, Prec, Xmm};

/// How a value of a type is held by the debug backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VCls {
    /// An integer-class scalar of `bytes` bytes, held normalized in a GPR.
    G { bytes: u8, signed: bool },
    /// `f32` / `f64` in the low lane of an XMM register.
    X(Prec),
    /// A value in memory (aggregate or fat pointer): its home holds `size` bytes.
    M { size: u32, align: u32 },
}

/// Largest aggregate the backend keeps by value (frame offsets are `i32`).
const MAX_VALUE_SIZE: u64 = 1 << 28;

/// The class of every type of the table (`None`: no value of the type can be held: unsized,
/// vector, generic, too large).
pub(crate) fn classify_all(types: &TypeTable, layouts: &Layouts) -> Vec<Option<VCls>> {
    (0..types.len()).map(|i| classify(types, layouts, TypeId(i as u32))).collect()
}

fn classify(types: &TypeTable, layouts: &Layouts, t: TypeId) -> Option<VCls> {
    let key = types.key(t)?;
    let g = |bytes: u8, signed: bool| Some(VCls::G { bytes, signed });
    match key {
        TypeKey::Int { signed, bits } => g(if bits.width() == 64 { 8 } else { bits.width() / 8 }, *signed),
        TypeKey::Bool => g(1, false),
        TypeKey::Char => g(4, false),
        TypeKey::Float(FloatBits::F32) => Some(VCls::X(Prec::F32)),
        TypeKey::Float(FloatBits::F64) => Some(VCls::X(Prec::F64)),
        TypeKey::Float(FloatBits::F16 | FloatBits::B16) => g(2, false),
        TypeKey::FnType { .. }
        | TypeKey::Ref { .. }
        | TypeKey::RawPtr { .. }
        | TypeKey::Tuple(_)
        | TypeKey::Array { .. }
        | TypeKey::Adt { .. } => {
            let l = layouts.get_layout(t).ok()?;
            match l.shape {
                Shape::Scalar => g(8, false),
                Shape::Fat | Shape::Fields(_) | Shape::Array { .. } | Shape::Enum(_) => {
                    if l.size > MAX_VALUE_SIZE || l.align > 16 {
                        return None;
                    }
                    Some(VCls::M { size: l.size as u32, align: l.align.max(1) as u32 })
                }
                Shape::Bytes => None,
            }
        }
        _ => None,
    }
}

/// Class of one eightbyte of an aggregate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Eb {
    Int,
    Sse,
}

/// The eightbyte classes of an aggregate of at most 16 bytes (`None` if larger or unclassifiable).
pub(crate) fn eightbytes(types: &TypeTable, layouts: &Layouts, t: TypeId, size: u32) -> Option<Vec<Eb>> {
    if size == 0 || size > 16 {
        return None;
    }
    let n = size.div_ceil(8) as usize;
    let mut cls: [Option<Eb>; 2] = [None, None];
    leaves(types, layouts, t, 0, &mut cls)?;
    Some((0..n).map(|i| cls[i].unwrap_or(Eb::Int)).collect())
}

fn mark(cls: &mut [Option<Eb>; 2], off: u64, e: Eb) {
    if let Some(c) = cls.get_mut((off / 8) as usize) {
        *c = match (*c, e) {
            (Some(Eb::Int), _) | (_, Eb::Int) => Some(Eb::Int),
            _ => Some(Eb::Sse),
        };
    }
}

/// Marks the eightbytes of `t` at offset `off`. An explicit work list instead of recursion, so a
/// type of any nesting depth classifies (only leaves below offset 16 matter, and zero-size parts
/// have none).
fn leaves(
    types: &TypeTable,
    layouts: &Layouts,
    t: TypeId,
    off: u64,
    cls: &mut [Option<Eb>; 2],
) -> Option<()> {
    let mut work = vec![(t, off)];
    while let Some((t, off)) = work.pop() {
        if off >= 16 {
            continue;
        }
        let l = layouts.get_layout(t).ok()?;
        if l.size == 0 {
            continue;
        }
        match &l.shape {
            Shape::Scalar => {
                let e = match types.key(t)? {
                    TypeKey::Float(FloatBits::F32 | FloatBits::F64) => Eb::Sse,
                    _ => Eb::Int,
                };
                mark(cls, off, e);
            }
            Shape::Fat => {
                mark(cls, off, Eb::Int);
                mark(cls, off + 8, Eb::Int);
            }
            Shape::Fields(fs) => work.extend(fs.iter().map(|&(o, ft)| (ft, off + o))),
            Shape::Array { elem, stride, len } => {
                work.extend((0..(*len).min(16)).map(|k| (*elem, off + k * stride)));
            }
            Shape::Enum(vs) => {
                mark(cls, off, Eb::Int);
                for v in vs {
                    work.extend(v.iter().map(|&(o, ft)| (ft, off + o)));
                }
            }
            Shape::Bytes => return None,
        }
    }
    Some(())
}

/// A register holding one eightbyte (or a scalar).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Piece {
    G(Gpr),
    X(Xmm),
}

/// Where one argument (or the return value) lives at a call boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Loc {
    /// Zero-sized: nothing is passed.
    Ignore,
    /// A scalar in one register.
    Reg(Piece),
    /// An aggregate in one or two registers, eightbyte by eightbyte.
    Regs(Vec<Piece>),
    /// At this offset of the stack argument area (`[rsp + off]` at the call,
    /// `[rbp + 16 + off]` in the callee).
    Stack(u32),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum RetLoc {
    None,
    Reg(Piece),
    Regs(Vec<Piece>),
    /// Through the hidden pointer in `rdi`; the callee returns it in `rax`.
    Sret,
}

#[derive(Clone, Debug)]
pub(crate) struct CallAbi {
    pub args: Vec<Loc>,
    pub ret: RetLoc,
    /// Size of the stack argument area, a multiple of 16.
    pub stack: u32,
}

pub(crate) const ARG_GPRS: [Gpr; 6] = [Gpr::RDI, Gpr::RSI, Gpr::RDX, Gpr::RCX, Gpr::R8, Gpr::R9];

pub(crate) const fn xmm(n: u8) -> Xmm {
    Xmm::masked(n)
}

/// Computes the call ABI of a signature. `None` when a parameter or the return type has no class.
/// `env`: one more INTEGER argument after `params` (a fn value's environment, `call.ind`).
pub(crate) fn call_abi(
    types: &TypeTable,
    layouts: &Layouts,
    cls: &[Option<VCls>],
    params: &[TypeId],
    ret: Option<TypeId>,
    env: bool,
) -> Option<CallAbi> {
    let get = |t: TypeId| cls.get(t.index()).copied().flatten();
    let mut ng = 0usize;
    let mut nx = 0u8;
    let ret = match ret {
        None => RetLoc::None,
        Some(t) => match get(t)? {
            VCls::G { .. } => RetLoc::Reg(Piece::G(Gpr::RAX)),
            VCls::X(_) => RetLoc::Reg(Piece::X(xmm(0))),
            VCls::M { size: 0, .. } => RetLoc::None,
            VCls::M { size, .. } if size <= 16 => {
                let ebs = eightbytes(types, layouts, t, size)?;
                let (mut gi, mut xi) = (0usize, 0u8);
                let ps = ebs
                    .iter()
                    .map(|e| match e {
                        Eb::Int => {
                            gi += 1;
                            Piece::G(if gi == 1 { Gpr::RAX } else { Gpr::RDX })
                        }
                        Eb::Sse => {
                            xi += 1;
                            Piece::X(xmm(xi - 1))
                        }
                    })
                    .collect();
                RetLoc::Regs(ps)
            }
            VCls::M { .. } => {
                ng = 1;
                RetLoc::Sret
            }
        },
    };
    let mut stack = 0u32;
    let mut on_stack = |size: u32, align: u32| -> u32 {
        let a = align.clamp(8, 16);
        stack = stack.div_ceil(a).saturating_mul(a);
        let off = stack;
        // saturates: an area this large is rejected at the call
        stack = stack.saturating_add(size.div_ceil(8) * 8);
        off
    };
    let mut args = Vec::with_capacity(params.len());
    for &p in params {
        let loc = match get(p)? {
            VCls::G { .. } => {
                if ng < 6 {
                    ng += 1;
                    Loc::Reg(Piece::G(ARG_GPRS[ng - 1]))
                } else {
                    Loc::Stack(on_stack(8, 8))
                }
            }
            VCls::X(_) => {
                if nx < 8 {
                    nx += 1;
                    Loc::Reg(Piece::X(xmm(nx - 1)))
                } else {
                    Loc::Stack(on_stack(8, 8))
                }
            }
            VCls::M { size: 0, .. } => Loc::Ignore,
            VCls::M { size, align } => match eightbytes(types, layouts, p, size) {
                Some(ebs) => {
                    let need_g = ebs.iter().filter(|e| **e == Eb::Int).count();
                    let need_x = ebs.len() - need_g;
                    if ng + need_g <= 6 && nx as usize + need_x <= 8 {
                        let ps = ebs
                            .iter()
                            .map(|e| match e {
                                Eb::Int => {
                                    ng += 1;
                                    Piece::G(ARG_GPRS[ng - 1])
                                }
                                Eb::Sse => {
                                    nx += 1;
                                    Piece::X(xmm(nx - 1))
                                }
                            })
                            .collect();
                        Loc::Regs(ps)
                    } else {
                        Loc::Stack(on_stack(size, align))
                    }
                }
                None => Loc::Stack(on_stack(size, align)),
            },
        };
        args.push(loc);
    }
    if env {
        let loc = if ng < 6 {
            ng += 1;
            Loc::Reg(Piece::G(ARG_GPRS[ng - 1]))
        } else {
            Loc::Stack(on_stack(8, 8))
        };
        args.push(loc);
    }
    let stack = stack.saturating_add(15) & !15;
    Some(CallAbi { args, ret, stack })
}
