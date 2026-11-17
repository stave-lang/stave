//! Slice parts as values: the address and the length of a slice are read once, where the slice
//! is defined, instead of at every element access.
//!
//! - [`unify_parts`]: all `slice.len s` (or `slice.ptr s`) of one slice value `s` become a single
//!   instruction right after the definition of `s`: inside a loop the part is a value defined
//!   outside it, which the backends keep in a register.
//! - [`split_elem_addrs`]: `addr.elem s i`, `vec.load s i`, and `vec.store s i v` on a slice
//!   reference become `p = slice.ptr s` and the same op on the element pointer `p`, so the
//!   address is such a part too. It runs after the removal of bounds checks, which reads an
//!   access by its slice operand; the bounds audit no longer sees these accesses afterwards.
//!
//! - [`unify_chunk_bounds`]: all `n -| k` of one value `n` and one constant `k` (the bound of a
//!   vector access of `k + 1` lanes, checked against the length `n`) become a single instruction
//!   right after the definition of `n`, so a loop compares its index with a register.
//!
//! All of these are pure and trap-free, so moving them up to the definition changes no behavior.

use stave_ir::ir::ValueDef;
use stave_ir::{Body, Op, Operand, TypeId, TypeKey, TypeTable, Value, VecOp};

use crate::cfg::{self, Edges, NONE};
use crate::edit::{At, Edit};

/// Interns the element pointer type `*mut T` of every slice type `[T]`, which
/// [`split_elem_addrs`] needs. Call it once before the per-function passes.
pub fn declare_elem_pointers(types: &mut TypeTable) {
    let elems: Vec<TypeId> = types
        .keys()
        .iter()
        .filter_map(|k| match k {
            TypeKey::Slice { elem, .. } => Some(*elem),
            _ => None,
        })
        .collect();
    for e in elems {
        types.ptr(true, e);
    }
}

/// Unifies the `op` (`slice.len` or `slice.ptr`) instructions per slice value. A part is moved
/// only when that removes work: two or more instructions read it, or one inside a loop the slice
/// is defined outside of. Returns whether the body changed.
pub fn unify_parts(b: &mut Body, op: Op) -> bool {
    debug_assert!(matches!(op, Op::SliceLen | Op::SlicePtr));
    // (slice value, result type, 0, instruction), grouped by sorting.
    let mut parts: Vec<(u32, u32, i128, u32)> = Vec::new();
    for (i, &o) in b.insts.op.iter().enumerate() {
        if o == op
            && let [Operand::Value(s)] = b.args(i as u32)
            && let Some(Some(t)) = b.insts.ty.get(i)
            && b.insts.result[i].is_some()
        {
            parts.push((s.0, t.0, 0, i as u32));
        }
    }
    unify(b, parts)
}

/// Unifies the `n -| k` instructions (a constant `k`) per value `n` and `k`, under the rule of
/// [`unify_parts`]. Returns whether the body changed.
pub fn unify_chunk_bounds(b: &mut Body) -> bool {
    // (value, result type, constant, instruction), grouped by sorting.
    let mut parts: Vec<(u32, u32, i128, u32)> = Vec::new();
    for (i, &o) in b.insts.op.iter().enumerate() {
        if o == Op::SubSat
            && let [Operand::Value(n), k] = b.args(i as u32)
            && let Some(k) = b.int_const(*k)
            && let Some(Some(t)) = b.insts.ty.get(i)
            && b.insts.result[i].is_some()
        {
            parts.push((n.0, t.0, k, i as u32));
        }
    }
    unify(b, parts)
}

/// Makes every group of instructions with one operand value, type, and constant a single
/// instruction right after the definition of the operand value: the group's first instruction,
/// moved there. `parts`: (operand value, result type, constant, instruction).
fn unify(b: &mut Body, mut parts: Vec<(u32, u32, i128, u32)>) -> bool {
    if parts.is_empty() {
        return false;
    }
    parts.sort_unstable();
    let block_of = cfg::block_of_insts(b);
    let edges = Edges::new(b);
    let depth = edges.loop_depths();
    let depth_of = |blk: u32| depth.get(blk as usize).copied().unwrap_or(0);
    let mut edit = Edit::new(b);
    let mut k = 0;
    while k < parts.len() {
        let (s, t, c) = (parts[k].0, parts[k].1, parts[k].2);
        let mut end = k;
        while end < parts.len() && parts[end].0 == s && parts[end].1 == t && parts[end].2 == c {
            end += 1;
        }
        let group = &parts[k..end];
        k = end;
        let Some(vd) = b.values.get(s as usize) else { continue };
        let (at, def_block) = match vd.def {
            ValueDef::Param { block, .. } => (At::BlockStart(block.0), block.0),
            ValueDef::Inst(d) => match block_of.get(d as usize) {
                Some(&blk) if blk != NONE => (At::After(d), blk),
                _ => continue,
            },
        };
        if !edges.reachable.get(def_block as usize).copied().unwrap_or(false) {
            continue;
        }
        let deeper = group.iter().any(|p| {
            let blk = block_of[p.3 as usize];
            blk != NONE && depth_of(blk) > depth_of(def_block)
        });
        if group.len() < 2 && !deeper {
            continue;
        }
        // The first instruction moves to the definition; the others read its value.
        let first = group[0].3;
        let Some(v) = b.insts.result[first as usize] else { continue };
        let site = b.insts.site[first as usize];
        let (op, args) = (b.insts.op[first as usize], b.args(first).to_vec());
        edit.insert(at, op, Some(TypeId(t)), &args, site, Some(v));
        for p in group {
            edit.remove(p.3);
            if p.3 != first
                && let Some(w) = b.insts.result[p.3 as usize]
            {
                edit.replace(w, Operand::Value(v));
            }
        }
    }
    let changed = !edit.is_empty();
    edit.apply(b);
    changed
}

/// Rewrites every `addr.elem s i`, `vec.load s i`, and `vec.store s i v` whose `s` is a reference
/// to a slice into `slice.ptr s` plus the same op on the element pointer. Returns whether the body
/// changed.
///
/// A slice whose elements are arrays is left alone: `addr.elem` on a pointer to an array steps
/// by the array's element, so on the element pointer `*mut [T; N]` it would address `T`s, not
/// the slice's elements.
pub fn split_elem_addrs(types: &TypeTable, b: &mut Body) -> bool {
    let mut edit = Edit::new(b);
    let mut rewire: Vec<(u32, Value)> = Vec::new();
    for (i, &op) in b.insts.op.iter().enumerate() {
        let arity = match op {
            Op::ElemAddr | Op::Vec(VecOp::Load) => 2,
            Op::Vec(VecOp::Store) => 3,
            _ => continue,
        };
        let Some(span) = b.insts.args.get(i) else { continue };
        let Some(Operand::Value(s)) = b.operands.get(span.start as usize).copied() else { continue };
        if span.len != arity {
            continue;
        }
        let Some(TypeKey::Ref { inner, .. } | TypeKey::RawPtr { inner, .. }) =
            b.value_ty(s).and_then(|t| types.key(t))
        else {
            continue;
        };
        let Some(TypeKey::Slice { elem, .. }) = types.key(*inner) else { continue };
        if matches!(types.key(*elem), Some(TypeKey::Array { .. } | TypeKey::Slice { .. })) {
            continue;
        }
        let Some(pt) = types.find(&TypeKey::RawPtr { mutable: true, inner: *elem }) else { continue };
        let p = edit.value(pt);
        edit.insert(
            At::Before(i as u32),
            Op::SlicePtr,
            Some(pt),
            &[Operand::Value(s)],
            b.insts.site[i],
            Some(p),
        );
        rewire.push((span.start, p));
    }
    if rewire.is_empty() {
        return false;
    }
    for (at, p) in rewire {
        b.operands[at as usize] = Operand::Value(p);
    }
    edit.apply(b);
    true
}
