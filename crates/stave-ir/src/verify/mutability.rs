//! Mutability of pointers (contracts/checked-ir.md "Mutability").
//!
//! A pointer type is mutable (`ref mut`, `ptr mut`) or shared (`ref`, `ptr const`); a string
//! operand is a shared pointer. Every write goes through a mutable pointer, and no instruction
//! makes a mutable pointer from a shared one. Exempt are the instructions whose pointer operands
//! are not where the result points: a call's arguments and an intrinsic's operands (the result
//! has the callee's or the signature's type, and the callee's body is verified on its own), and a
//! `load`, which reads a stored pointer; and the casts `unsafe` code writes: a cast to or from a
//! raw pointer, or of an int to a pointer. Every other instruction with a mutable pointer result
//! and a shared pointer operand is refused, so an op added later is held to the rule until it is
//! listed here.

use super::FnVerifier;
use crate::ir::{Body, Op, Operand};
use crate::types::{TypeId, TypeKey};

type R = Result<(), String>;

/// A pointer type's mutability and whether it is a raw pointer.
#[derive(Clone, Copy)]
struct Access {
    mutable: bool,
    raw: bool,
}

impl FnVerifier<'_> {
    /// The access of pointer type `t`; `None` for any other type.
    fn access(&self, t: TypeId) -> Option<Access> {
        match self.key(t) {
            Some(TypeKey::Ref { mutable, .. }) => Some(Access { mutable: *mutable, raw: false }),
            Some(TypeKey::RawPtr { mutable, .. }) => Some(Access { mutable: *mutable, raw: true }),
            _ => None,
        }
    }

    /// The access of operand `o` when it is a pointer: a value of a pointer type, or a string
    /// (its bytes are read-only data).
    fn operand_access(&self, b: &Body, o: Operand) -> Option<Access> {
        match o {
            Operand::Value(v) => b.value_ty(v).and_then(|t| self.access(t)),
            Operand::Str(_) => Some(Access { mutable: false, raw: false }),
            Operand::Const(_) | Operand::Func(_) | Operand::Type(_) => None,
        }
    }

    /// Instruction `i`, already well typed, against the mutability rule.
    pub(super) fn check_mutability(&self, b: &Body, i: u32) -> R {
        let op = b.insts.op[i as usize];
        let args = b.args(i);
        let write = match op {
            Op::Store => Some("store"),
            Op::StoreStream => Some("mem.store_stream"),
            _ => None,
        };
        if let Some(name) = write {
            return match args.first().and_then(|&p| self.operand_access(b, p)) {
                Some(Access { mutable: true, .. }) => Ok(()),
                _ => Err(format!("{name} through a shared pointer")),
            };
        }
        let Some(result) = b.insts.ty[i as usize].and_then(|t| self.access(t)) else { return Ok(()) };
        if !result.mutable {
            return Ok(());
        }
        let from = args.first().and_then(|&a| self.operand_access(b, a));
        match op {
            Op::Call | Op::CallInd | Op::Intrinsic(_) | Op::Load => return Ok(()),
            Op::Cast(_) if result.raw || from.is_none_or(|f| f.raw) => return Ok(()),
            _ => {}
        }
        if args.iter().any(|&a| self.operand_access(b, a).is_some_and(|x| !x.mutable)) {
            return Err("mutable pointer from a shared pointer operand".into());
        }
        Ok(())
    }
}
