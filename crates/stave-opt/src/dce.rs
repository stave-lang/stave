//! Dead code: removes instructions that only compute a value nobody reads.
//!
//! Only ops that cannot trap and have no effect besides their result go (wrapping and saturating
//! integer arithmetic, bit ops, compares, `select`, casts, float arithmetic, aggregate and enum
//! construction and field reads, address arithmetic, slice parts, vector and mask ops that neither
//! trap nor touch memory). Trapping arithmetic, checks, loads, stores, memory hints (prefetches,
//! streaming stores, stream fences), calls, intrinsics, and slots stay.

use stave_ir::{Body, Op, Operand, TypeKey, TypeTable};

use crate::cfg;
use crate::edit::Edit;

/// Whether instruction `i` computes only its result and cannot trap.
fn removable(types: &TypeTable, b: &Body, i: usize) -> bool {
    let op = b.insts.op[i];
    match op {
        Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Rem | Op::Neg => {
            // IEEE on floats; ints trap on overflow or a zero divisor
            matches!(b.insts.ty[i].and_then(|t| types.key(t)), Some(TypeKey::Float(_)))
        }
        // a dynamic index traps `Bounds`
        Op::Get | Op::Set => b.args(i as u32).get(1).is_some_and(|o| b.int_const(*o).is_some()),
        Op::AddWrap
        | Op::SubWrap
        | Op::MulWrap
        | Op::ShlWrap
        | Op::AddSat
        | Op::SubSat
        | Op::MulSat
        | Op::And
        | Op::Or
        | Op::Xor
        | Op::Not
        | Op::Popcount
        | Op::Clz
        | Op::Ctz
        | Op::MulWide
        | Op::Eq
        | Op::Ne
        | Op::Lt
        | Op::Le
        | Op::Gt
        | Op::Ge
        | Op::Select
        | Op::Cast(_)
        | Op::Make
        | Op::EnumMake
        | Op::EnumTag
        | Op::EnumGet
        | Op::FieldAddr
        | Op::EnumFieldAddr
        | Op::ElemAddr
        | Op::SliceMake
        | Op::SliceLen
        | Op::SlicePtr
        | Op::FnMake => true,
        Op::Vec(v) => v.trap_kinds().is_empty() && !v.is_memory(),
        Op::Shl
        | Op::Shr
        | Op::Slot
        | Op::Load
        | Op::Store
        | Op::BoundsCheck
        | Op::Prefetch(_)
        | Op::StoreStream
        | Op::FenceStream
        | Op::Call
        | Op::CallInd
        | Op::Intrinsic(_)
        | Op::Panic(_) => false,
    }
}

/// Removes the dead instructions of `b`, transitively. Returns whether the body changed.
pub fn remove_dead(types: &TypeTable, b: &mut Body) -> bool {
    let n = b.insts.len();
    let mut uses: Vec<u32> = cfg::use_counts(b).into_iter().map(u32::from).collect();
    // A saturated count stays above zero: such a value is simply kept.
    let saturated = |c: u32| c >= u32::from(u8::MAX);
    let mut def = vec![cfg::NONE; b.values.len()];
    for (i, r) in b.insts.result.iter().enumerate() {
        if let Some(v) = r
            && let Some(d) = def.get_mut(v.index())
        {
            *d = i as u32;
        }
    }
    let mut edit = Edit::new(b);
    let mut work: Vec<u32> = (0..n as u32)
        .filter(|&i| {
            b.insts.result[i as usize].is_some_and(|v| uses.get(v.index()).copied() == Some(0))
                && removable(types, b, i as usize)
        })
        .collect();
    while let Some(i) = work.pop() {
        if edit.removed(i) {
            continue;
        }
        edit.remove(i);
        for &o in b.args(i) {
            let Operand::Value(v) = o else { continue };
            let Some(c) = uses.get_mut(v.index()) else { continue };
            if *c == 0 || saturated(*c) {
                continue;
            }
            *c -= 1;
            let d = def[v.index()];
            if *c == 0 && d != cfg::NONE && removable(types, b, d as usize) {
                work.push(d);
            }
        }
    }
    let changed = !edit.is_empty();
    edit.apply(b);
    changed
}
