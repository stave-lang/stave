//! The trapping instructions of a checked-IR function: what makes it have the `trap` effect, read
//! from the IR after the removal of proven bounds and contract checks (the same IR in every build
//! mode).
//!
//! An instruction traps when its op can raise a panic kind other than `StackOverflow` (a call or
//! stack slot overflows the stack only where call recursion is reachable; the call graph decides
//! that): `panic.<kind>`, `check.bounds`, `add sub mul neg` on ints, `div rem` on ints unless the
//! divisor is a constant other than 0 (and other than -1 for a signed type), `shl shr` unless the
//! amount is a constant below the width, `get`/`set` of an array element unless the index is a
//! constant below the length, the vector ops that trap, and a contract check that the removal kept
//! (its call of `runtime.panic.requires_failed` / `ensures_failed`, `Contract`). `panic.out_of_memory`
//! also counts as an allocation.

use crate::ir::{Body, Const, FuncId, Module, Op, Operand, PanicKind, Site};
use crate::types::TypeKey;
use crate::witness::{ENSURES_FAILED, REQUIRES_FAILED};

/// The first trapping instruction of a function and its first `panic.out_of_memory`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Traps {
    pub trap: Option<(Site, PanicKind)>,
    pub out_of_memory: Option<Site>,
}

/// [`Traps`] of function `f` (none for a function without a body).
pub fn scan(m: &Module, f: FuncId) -> Traps {
    let mut out = Traps::default();
    let Some(b) = m.func(f).and_then(|x| x.body.as_ref()) else { return out };
    for i in 0..b.insts.len() as u32 {
        let Some(kind) = trap_of(m, b, i) else { continue };
        let site = b.insts.site.get(i as usize).copied().unwrap_or(Site::NONE);
        if out.trap.is_none() {
            out.trap = Some((site, kind));
        }
        if kind == PanicKind::OutOfMemory && out.out_of_memory.is_none() {
            out.out_of_memory = Some(site);
        }
        if out.out_of_memory.is_some() {
            break;
        }
    }
    out
}

/// The panic kind instruction `i` of `b` can raise, if any (not `StackOverflow`).
pub fn trap_of(m: &Module, b: &Body, i: u32) -> Option<PanicKind> {
    let op = *b.insts.op.get(i as usize)?;
    let args = b.args(i);
    let int_of = |t: Option<crate::types::TypeId>| match t.and_then(|t| m.types.key(t)) {
        Some(TypeKey::Int { signed, bits }) => Some((*signed, bits.width())),
        _ => None,
    };
    let konst = |o: Option<&Operand>| match o {
        Some(Operand::Const(c)) => match b.konst(*c) {
            Some(Const::Int(v)) => Some(v),
            _ => None,
        },
        _ => None,
    };
    let result_ty = b.insts.ty.get(i as usize).copied().flatten();
    let value_ty = |o: Option<&Operand>| match o {
        Some(Operand::Value(v)) => b.values.get(v.index()).map(|d| d.ty),
        _ => None,
    };
    match op {
        Op::Panic(PanicKind::StackOverflow) => None,
        Op::Panic(k) => Some(k),
        Op::BoundsCheck => Some(PanicKind::Bounds),
        Op::Add | Op::Sub | Op::Mul | Op::Neg => int_of(result_ty).map(|_| PanicKind::Overflow),
        Op::Div | Op::Rem => {
            let (signed, _) = int_of(result_ty)?;
            match konst(args.get(1)) {
                Some(c) if c != 0 && (!signed || c != -1) => None,
                _ => Some(PanicKind::DivByZero),
            }
        }
        Op::Shl | Op::Shr => {
            let (_, width) = int_of(result_ty)?;
            match konst(args.get(1)) {
                Some(c) if (0..i128::from(width)).contains(&c) => None,
                _ => Some(PanicKind::Overflow),
            }
        }
        Op::Get | Op::Set => {
            let len = match value_ty(args.first()).and_then(|t| m.types.key(t)) {
                Some(TypeKey::Array { len, .. }) => *len,
                _ => return None,
            };
            match konst(args.get(1)) {
                Some(c) if (0..i128::from(len)).contains(&c) => None,
                _ => Some(PanicKind::Bounds),
            }
        }
        Op::Vec(v) => v.trap_kinds().iter().copied().find(|&k| k != PanicKind::StackOverflow),
        Op::Call => match args.first() {
            Some(Operand::Func(f)) => m
                .func(*f)
                .filter(|f| f.path == REQUIRES_FAILED || f.path == ENSURES_FAILED)
                .map(|_| PanicKind::Contract),
            _ => None,
        },
        _ => None,
    }
}
