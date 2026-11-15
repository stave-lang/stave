//! Batched edits of a function body: remove instructions, insert new ones, replace values.
//!
//! A body keeps its instructions as arrays in block order, so edits are collected first and
//! [`Edit::apply`] rebuilds the arrays once.

use stave_ir::ir::{Insts, Span, ValueData, ValueDef};
use stave_ir::{Body, Op, Operand, Site, Term, TypeId, Value};

/// Where a new instruction goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum At {
    /// First in the block.
    BlockStart(u32),
    /// Right before this (old) instruction.
    Before(u32),
    /// Right after this (old) instruction.
    After(u32),
}

struct NewInst {
    at: At,
    op: Op,
    ty: Option<TypeId>,
    args: Vec<Operand>,
    site: Site,
    result: Option<Value>,
}

pub(crate) struct Edit {
    remove: Vec<bool>,
    replace: Vec<Option<Operand>>,
    new: Vec<NewInst>,
    /// Types of the values created by this edit (ids follow the body's values).
    new_values: Vec<TypeId>,
    first_new_value: u32,
    changed: bool,
}

impl Edit {
    pub(crate) fn new(b: &Body) -> Edit {
        Edit {
            remove: vec![false; b.insts.len()],
            replace: vec![None; b.values.len()],
            new: Vec::new(),
            new_values: Vec::new(),
            first_new_value: b.values.len() as u32,
            changed: false,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        !self.changed
    }

    /// A fresh value of type `ty`, to be defined by an inserted instruction.
    pub(crate) fn value(&mut self, ty: TypeId) -> Value {
        self.new_values.push(ty);
        Value(self.first_new_value + self.new_values.len() as u32 - 1)
    }

    pub(crate) fn remove(&mut self, inst: u32) {
        if let Some(r) = self.remove.get_mut(inst as usize) {
            *r = true;
            self.changed = true;
        }
    }

    pub(crate) fn removed(&self, inst: u32) -> bool {
        self.remove.get(inst as usize).copied().unwrap_or(false)
    }

    /// Every use of `v` reads `by` instead.
    pub(crate) fn replace(&mut self, v: Value, by: Operand) {
        if let Some(r) = self.replace.get_mut(v.index()) {
            *r = Some(by);
            self.changed = true;
        }
    }

    /// Inserts an instruction. `result` is a value of [`Edit::value`], or the result of a
    /// removed instruction (the instruction moves).
    pub(crate) fn insert(
        &mut self,
        at: At,
        op: Op,
        ty: Option<TypeId>,
        args: &[Operand],
        site: Site,
        result: Option<Value>,
    ) {
        self.new.push(NewInst { at, op, ty, args: args.to_vec(), site, result });
        self.changed = true;
    }

    fn subst(&self, mut o: Operand) -> Operand {
        // replacements may chain; a cycle cannot arise from dominance-ordered rewrites, the bound
        // only keeps a malformed edit from looping
        for _ in 0..16 {
            match o {
                Operand::Value(v) => match self.replace.get(v.index()).copied().flatten() {
                    Some(by) => o = by,
                    None => break,
                },
                _ => break,
            }
        }
        o
    }

    pub(crate) fn apply(self, b: &mut Body) {
        if !self.changed {
            return;
        }
        for &ty in &self.new_values {
            b.values.push(ValueData { ty, def: ValueDef::Inst(u32::MAX) });
        }
        // Insertions by anchor, each list in insertion order.
        let mut starts: Vec<(u32, usize)> = Vec::new();
        let mut befores: Vec<(u32, usize)> = Vec::new();
        let mut afters: Vec<(u32, usize)> = Vec::new();
        for (k, n) in self.new.iter().enumerate() {
            match n.at {
                At::BlockStart(blk) => starts.push((blk, k)),
                At::Before(i) => befores.push((i, k)),
                At::After(i) => afters.push((i, k)),
            }
        }
        starts.sort_by_key(|x| x.0);
        befores.sort_by_key(|x| x.0);
        afters.sort_by_key(|x| x.0);
        let old = std::mem::take(&mut b.insts);
        // A removed instruction's result has no definition unless an insertion redefines it.
        for (i, r) in old.result.iter().enumerate() {
            if self.remove[i]
                && let Some(v) = r
                && let Some(vd) = b.values.get_mut(v.index())
            {
                vd.def = ValueDef::Inst(u32::MAX);
            }
        }
        let mut out = Insts::default();
        let (mut si, mut bi, mut ai) = (0usize, 0usize, 0usize);
        let emit_new =
            |k: usize, out: &mut Insts, operands: &mut Vec<Operand>, values: &mut Vec<ValueData>| {
                let n = &self.new[k];
                let span = Span { start: operands.len() as u32, len: n.args.len() as u32 };
                operands.extend_from_slice(&n.args);
                if let Some(v) = n.result
                    && let Some(vd) = values.get_mut(v.index())
                {
                    vd.def = ValueDef::Inst(out.len() as u32);
                }
                out.op.push(n.op);
                out.ty.push(n.ty);
                out.result.push(n.result);
                out.args.push(span);
                out.site.push(n.site);
            };
        for (blk, bd) in b.blocks.iter_mut().enumerate() {
            let start = out.len() as u32;
            while si < starts.len() && starts[si].0 as usize == blk {
                emit_new(starts[si].1, &mut out, &mut b.operands, &mut b.values);
                si += 1;
            }
            for i in bd.insts.indices() {
                // anchors outside every block are skipped
                while bi < befores.len() && befores[bi].0 < i {
                    bi += 1;
                }
                while ai < afters.len() && afters[ai].0 < i {
                    ai += 1;
                }
                while bi < befores.len() && befores[bi].0 == i {
                    emit_new(befores[bi].1, &mut out, &mut b.operands, &mut b.values);
                    bi += 1;
                }
                let iu = i as usize;
                if iu < old.len() && !self.remove[iu] {
                    if let Some(v) = old.result[iu]
                        && let Some(vd) = b.values.get_mut(v.index())
                    {
                        vd.def = ValueDef::Inst(out.len() as u32);
                    }
                    out.op.push(old.op[iu]);
                    out.ty.push(old.ty[iu]);
                    out.result.push(old.result[iu]);
                    out.args.push(old.args[iu]);
                    out.site.push(old.site[iu]);
                }
                while ai < afters.len() && afters[ai].0 == i {
                    emit_new(afters[ai].1, &mut out, &mut b.operands, &mut b.values);
                    ai += 1;
                }
            }
            bd.insts = Span { start, len: out.len() as u32 - start };
        }
        b.insts = out;
        self.rebuild_operands(b);
        compact_values(b);
    }

    /// Rebuilds the operand pool from the spans in use (instruction and branch arguments), with
    /// the replacements applied: operands of removed instructions do not stay behind.
    fn rebuild_operands(&self, b: &mut Body) {
        let old = std::mem::take(&mut b.operands);
        let mut pool: Vec<Operand> = Vec::with_capacity(old.len());
        let mut copy = |s: Span| {
            let start = pool.len() as u32;
            pool.extend(s.of(&old).iter().map(|o| self.subst(*o)));
            Span { start, len: pool.len() as u32 - start }
        };
        for s in &mut b.insts.args {
            *s = copy(*s);
        }
        for bd in &mut b.blocks {
            match &mut bd.term {
                Term::Br(t) => t.args = copy(t.args),
                Term::Cbr { cond, then, els } => {
                    *cond = self.subst(*cond);
                    then.args = copy(then.args);
                    els.args = copy(els.args);
                }
                Term::Switch { scrut, cases, default } => {
                    *scrut = self.subst(*scrut);
                    default.args = copy(default.args);
                    let (lo, n) = (cases.start as usize, cases.len as usize);
                    for c in b.cases.iter_mut().skip(lo).take(n) {
                        c.target.args = copy(c.target.args);
                    }
                }
                Term::Ret(Some(o)) => *o = self.subst(*o),
                Term::Ret(None) | Term::Unreachable => {}
            }
        }
        b.operands = pool;
    }
}

/// Drops the values without a definition (results of removed instructions) and renumbers the
/// rest: a verified body defines every value.
fn compact_values(b: &mut Body) {
    let nv = b.values.len();
    let mut keep = vec![false; nv];
    for v in b.block_params.iter().chain(b.insts.result.iter().flatten()) {
        if let Some(k) = keep.get_mut(v.index()) {
            *k = true;
        }
    }
    if keep.iter().all(|k| *k) {
        return;
    }
    let mut new_id = vec![u32::MAX; nv];
    let mut next = 0u32;
    for (v, k) in keep.iter().enumerate() {
        if *k {
            new_id[v] = next;
            next += 1;
        }
    }
    let map = |v: Value| Value(new_id.get(v.index()).copied().unwrap_or(u32::MAX));
    let mut k = 0;
    b.values.retain(|_| {
        k += 1;
        keep[k - 1]
    });
    b.block_params.iter_mut().for_each(|v| *v = map(*v));
    b.insts.result.iter_mut().flatten().for_each(|v| *v = map(*v));
    let fix = |o: &mut Operand| {
        if let Operand::Value(v) = o {
            *v = map(*v);
        }
    };
    b.operands.iter_mut().for_each(fix);
    for bd in &mut b.blocks {
        match &mut bd.term {
            Term::Cbr { cond: o, .. } | Term::Switch { scrut: o, .. } | Term::Ret(Some(o)) => fix(o),
            _ => {}
        }
    }
}
