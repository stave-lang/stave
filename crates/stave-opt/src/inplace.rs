//! Loads a backend reads in place: a `load` of a whole `slot` whose value is copied out right
//! away needs no copy of its own.
//!
//! The value of `%v = load %s` (`%s` a `slot`) qualifies when its one use is the next thing its
//! block runs and that use copies the value out as it runs: the stored value of a `store`, an
//! argument of a `call` / `call.ind`, or the operand of the block's `ret`. Nothing runs between
//! the load and that copy, so the slot's bytes are still the loaded value: the backend may give
//! `%v` the slot's bytes as its home. Whether `%v` is a value kept in memory at all (an
//! aggregate) is the backend's to decide.

use stave_ir::{Body, Op, Operand, Term, Value};

/// Per value: the slot a load that qualifies reads, else `None`.
pub fn in_place_loads(b: &Body) -> Vec<Option<Value>> {
    let nv = b.values.len();
    let mut out = vec![None; nv];
    let mut uses = vec![0u8; nv];
    let mut add = |o: &Operand| {
        if let Operand::Value(v) = o
            && let Some(c) = uses.get_mut(v.index())
        {
            *c = c.saturating_add(1);
        }
    };
    b.operands.iter().for_each(&mut add);
    for bd in &b.blocks {
        match &bd.term {
            Term::Cbr { cond: o, .. } | Term::Switch { scrut: o, .. } | Term::Ret(Some(o)) => add(o),
            _ => {}
        }
    }
    let mut slot = vec![false; nv];
    for (op, r) in b.insts.op.iter().zip(&b.insts.result) {
        if *op == Op::Slot
            && let Some(r) = r
            && let Some(s) = slot.get_mut(r.index())
        {
            *s = true;
        }
    }
    for bd in &b.blocks {
        let mut ks = bd.insts.indices().peekable();
        while let Some(k) = ks.next() {
            if b.insts.op[k as usize] != Op::Load {
                continue;
            }
            let Some(Some(v)) = b.insts.result.get(k as usize).copied() else { continue };
            let Some(&Operand::Value(s)) = b.args(k).first() else { continue };
            if !slot.get(s.index()).copied().unwrap_or(false) || uses.get(v.index()) != Some(&1) {
                continue;
            }
            let me = Operand::Value(v);
            let copied_out = match ks.peek() {
                Some(&n) => match b.insts.op[n as usize] {
                    Op::Store => b.args(n).get(1) == Some(&me),
                    Op::Call | Op::CallInd => b.args(n).iter().skip(1).any(|o| *o == me),
                    _ => false,
                },
                None => matches!(bd.term, Term::Ret(Some(o)) if o == me),
            };
            if copied_out {
                out[v.index()] = Some(s);
            }
        }
    }
    out
}
