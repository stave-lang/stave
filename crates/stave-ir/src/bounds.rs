//! Bounds checks of element accesses (panic `Bounds`; no silent check removal): the audit that
//! every access keeps its check, and the elision of a check only with a recorded proof.
//!
//! **Insertion** happens in the front end's lowering (stave-sema), so the checked IR
//! (contracts/checked-ir.md) already carries every check:
//!
//! - `xs[i]` on a place in memory (a slice, or an array behind a reference or in a `slot`):
//!   `check.bounds i n` right before `addr.elem p i`, with `n = slice.len p` or the array length.
//! - `a[i]` on an array value (SSA): `get a i` / `set a i v`, which trap `Bounds` themselves for a
//!   dynamic index.
//! - a constant index ≥ the array length: an unconditional `panic.bounds` (the trap is kept, the
//!   access is never emitted).
//! - compiler-made element accesses (`for` over an array or slice, drop loops over arrays, the
//!   array → slice coercion) index below a length the generated code itself compares against.
//! - `vec.load` / `vec.store` (and their aligned forms) of N lanes at `i`: `check.bounds i lim`
//!   with `lim = sub.sat n N-1` (so `i < lim` iff `i + N <= n`), or the `ChunkLoop` proof: a chunk
//!   loop's own head test `cbr (lt i lim)` with the same `lim`. Masked loads and stores, gathers and
//!   scatters check their active lanes themselves.
//!
//! **Audit** ([`audit`]): every element access is classified with its [`Guard`]: a dominating
//! `check.bounds` on the same index and length, the op's own trap, or a [`Proof`]. An access with
//! none is [`Unguarded`], a compiler bug (the driver reports it as an internal error). Accesses
//! through raw pointers (`addr.elem` on `*T`) are pointer arithmetic, the `unsafe` code's own
//! obligation, and are not element accesses.
//!
//! **Elision** ([`elide`]): a `check.bounds` is removed only when one of these proofs holds, and
//! each removal is returned as an [`Elided`] record (never silent):
//!
//! | Proof | Holds when |
//! |---|---|
//! | `ConstInRange` | index and length are constants and index < length |
//! | `Dominated` | an identical `check.bounds` (same index and length operands) dominates it |
//! | `LoopBound` | a dominating `cbr (lt i n') T F` reaches the check only through `T` (T's only predecessor), `i` unsigned, and `n'` is the same length operand (or a constant ≤ a constant length) |
//! | `LenEq` | the facts below prove `i < n` through an equality of two lengths, e.g. `i < len(a)` and `len(a) == len(b)` for `check.bounds i len(b)` |
//! | `Offset` | the facts below prove `i + k < n` for the index `i + k` (`k` a constant, possibly negative or 0) without an equality |
//!
//! `LenEq` and `Offset` come from one difference-constraint graph per function (`bounds/facts.rs`):
//! facts `x <= y + c` over SSA values from dominating branches (unsigned compares, a loop's
//! `lt i n` among them) and asserting calls (`std.debug.assert(a.len() == b.len(), ..)`), with
//! trapping `add`/`sub` by a constant folded into offsets. Each check is a query on it, explored
//! from the index for at most [`BOUNDS_STEP_LIMIT`] edges.
//!
//! The checked IR is printed before elision (it is pre-optimization IR); stage0 runs [`elide`] for
//! `--release` code generation only.

mod facts;

use crate::ir::{Block, Body, Func, FuncId, Module, Op, Operand, Term, Value, ValueDef};
use crate::simd::VecOp;
use crate::types::{TypeId, TypeKey, TypeTable};
use crate::verify::Dominators;

/// The most graph edges one bounds query follows before it gives up and keeps the check: compile
/// time stays linear in the number of checks.
pub const BOUNDS_STEP_LIMIT: u32 = 64;

/// What kind of element access an instruction is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AccessKind {
    /// `addr.elem p i` with `p` a reference to an array or slice.
    Elem,
    /// `get a i` on an array value.
    Get,
    /// `set a i v` on an array value.
    Set,
    /// A vector load or store (`vec.load`, `vec.store_masked`, `vec.gather`, ...).
    Vector,
}

/// Why an index is known to be in range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Proof {
    /// Constant index below a constant length.
    ConstInRange { index: u64, len: u64 },
    /// `addr.elem p 0` whose result only feeds `slice.make` (the array → slice coercion: the
    /// base address, never dereferenced by itself).
    SliceBase,
    /// The index was compared `lt i n` by the dominating branch ending block `branch` (stage0
    /// block index), and only its true edge leads here.
    LoopBound { branch: u32 },
    /// An identical `check.bounds` (instruction `first`) dominates this one.
    Dominated { first: u32 },
    /// A chunk loop's head test `lt i (sub.sat n N-1)` covers the whole vector.
    ChunkLoop,
    /// The index is below the length by facts that include the equality tested by instruction
    /// `cmp` (an `eq` or `ne`, through a branch or an asserting call).
    LenEq { cmp: u32 },
    /// The index is `base + k` (`base` an SSA value number, `None` for a constant index) and the
    /// facts prove it below the length without an equality.
    Offset { base: Option<u32>, k: i64 },
}

/// How an access is guarded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Guard {
    /// A dominating `check.bounds` (its instruction index).
    Check(u32),
    /// The op traps `Bounds` itself (`get`/`set` with a dynamic index).
    SelfChecked,
    Proof(Proof),
}

/// One element access of a function.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Access {
    pub func: FuncId,
    /// Instruction index in the function's body.
    pub inst: u32,
    pub kind: AccessKind,
    pub guard: Guard,
}

/// An access with no guard (a compiler bug).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unguarded {
    pub func: FuncId,
    /// The function's checked-IR path and instance.
    pub path: String,
    pub inst: u32,
    pub kind: AccessKind,
}

impl std::fmt::Display for Unguarded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: inst {}: {:?} access without a bounds check or proof", self.path, self.inst, self.kind)
    }
}

/// A removed `check.bounds` and the proof that allowed it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Elided {
    pub func: FuncId,
    pub path: String,
    /// Instruction index before the removal.
    pub inst: u32,
    pub proof: Proof,
}

/// Every element access of the module with its guard, or the unguarded ones.
pub fn audit(m: &Module) -> Result<Vec<Access>, Vec<Unguarded>> {
    let mut ok = Vec::new();
    let mut bad = Vec::new();
    for fi in 0..m.funcs.len() {
        let (o, b) = audit_func(m, FuncId(fi as u32));
        ok.extend(o);
        bad.extend(b);
    }
    if bad.is_empty() { Ok(ok) } else { Err(bad) }
}

/// [`audit`] for one function: its guarded and its unguarded element accesses, in instruction
/// order (none for an unknown id or a function without a body). Functions are independent, so a
/// caller may audit them in parallel; [`audit`]'s lists are every function's in id order.
pub fn audit_func(m: &Module, fid: FuncId) -> (Vec<Access>, Vec<Unguarded>) {
    let mut ok = Vec::new();
    let mut bad = Vec::new();
    let Some(f) = m.func(fid) else { return (ok, bad) };
    let Some(b) = &f.body else { return (ok, bad) };
    // Only these ops can be element accesses: without one there is nothing to analyse.
    let vops = m.vec_ops();
    let is_vec_access = |op: &Op| matches!(op, Op::Intrinsic(id) if vops.get(id.index()).copied().flatten().is_some_and(|v| v.is_memory()));
    if !b.insts.op.iter().any(|op| matches!(op, Op::ElemAddr | Op::Get | Op::Set) || is_vec_access(op)) {
        return (ok, bad);
    }
    let cx = Cx::new(m, b);
    for i in 0..b.insts.len() as u32 {
        let Some(kind) = cx.access_kind(i) else { continue };
        if !cx.reachable_inst(i) {
            continue;
        }
        match cx.guard(i, kind) {
            Some(guard) => ok.push(Access { func: fid, inst: i, kind, guard }),
            None => bad.push(Unguarded { func: fid, path: label(f), inst: i, kind }),
        }
    }
    (ok, bad)
}

/// Removes every `check.bounds` that has a proof; returns the records, in (function, instruction)
/// order. Accesses stay guarded ([`audit`] holds before and after).
pub fn elide(m: &mut Module) -> Vec<Elided> {
    let mut out = Vec::new();
    for fi in 0..m.funcs.len() {
        let removed: Vec<(u32, Proof)> = {
            let f = &m.funcs[fi];
            let Some(b) = &f.body else { continue };
            if !b.insts.op.contains(&Op::BoundsCheck) {
                continue;
            }
            let cx = Cx::new(m, b);
            (0..b.insts.len() as u32)
                .filter(|&i| b.insts.op[i as usize] == Op::BoundsCheck && cx.reachable_inst(i))
                .filter_map(|i| cx.check_proof(i).map(|p| (i, p)))
                .collect()
        };
        if removed.is_empty() {
            continue;
        }
        let path = label(&m.funcs[fi]);
        // A check proven only by another removed check stays proven: `Dominated` points at a
        // check that dominates it, and that one is either kept or itself proven.
        if let Some(b) = &mut m.funcs[fi].body {
            remove_insts(b, &removed.iter().map(|r| r.0).collect::<Vec<_>>());
        }
        out.extend(removed.into_iter().map(|(inst, proof)| Elided {
            func: FuncId(fi as u32),
            path: path.clone(),
            inst,
            proof,
        }));
    }
    out
}

fn label(f: &Func) -> String {
    format!("@{} {}", f.instance, f.path)
}

/// Per-function analysis context.
struct Cx<'a> {
    types: &'a TypeTable,
    /// The module's functions (callee bodies for asserting calls).
    funcs: &'a [Func],
    b: &'a Body,
    dom: Dominators,
    /// Block of each instruction.
    block_of: Vec<u32>,
    /// Predecessor edges per block: (reachable predecessor, is the true edge of a `cbr`).
    preds: Vec<Vec<(u32, bool)>>,
    /// The reachable `check.bounds` instructions, in order.
    checks: Vec<u32>,
    facts: facts::Facts,
    /// The vector op of each intrinsic id.
    vops: Vec<Option<VecOp>>,
}

impl<'a> Cx<'a> {
    fn new(m: &'a Module, b: &'a Body) -> Cx<'a> {
        let (types, funcs) = (&m.types, &m.funcs[..]);
        let mut block_of = vec![u32::MAX; b.insts.len()];
        for (bi, bd) in b.blocks.iter().enumerate() {
            for i in bd.insts.indices() {
                if let Some(s) = block_of.get_mut(i as usize) {
                    *s = bi as u32;
                }
            }
        }
        let dom = Dominators::compute(b);
        let preds = preds_of(b, &dom);
        let checks = (0..b.insts.len() as u32)
            .filter(|&i| {
                b.insts.op[i as usize] == Op::BoundsCheck
                    && block_of.get(i as usize).is_some_and(|&bk| dom.reachable(bk as usize))
            })
            .collect();
        Cx { types, funcs, b, dom, block_of, preds, checks, facts: facts::Facts::default(), vops: m.vec_ops() }
    }

    fn reachable_inst(&self, i: u32) -> bool {
        self.block_of.get(i as usize).is_some_and(|&bk| self.dom.reachable(bk as usize))
    }

    fn key(&self, t: TypeId) -> Option<&TypeKey> {
        self.types.key(t)
    }

    fn ty(&self, o: Operand) -> Option<TypeId> {
        match o {
            Operand::Value(v) => self.b.value_ty(v),
            _ => None,
        }
    }

    /// The instruction defining `v`.
    fn def(&self, v: Value) -> Option<u32> {
        match self.b.values.get(v.index())?.def {
            ValueDef::Inst(i) => Some(i),
            ValueDef::Param { .. } => None,
        }
    }

    fn konst(&self, o: Operand) -> Option<u64> {
        self.b.int_const(o).and_then(|v| u64::try_from(v).ok())
    }

    /// The array length, or `None` for a slice, of what pointer `p` points to; `Err` if `p` is not
    /// a reference to an array or slice.
    fn len_of_ptr(&self, p: Operand) -> Result<Option<u64>, ()> {
        let pt = self.ty(p).ok_or(())?;
        let inner = match self.key(pt) {
            Some(TypeKey::Ref { inner, .. }) => *inner,
            _ => return Err(()),
        };
        match self.key(inner) {
            Some(TypeKey::Array { len, .. }) => Ok(Some(*len)),
            Some(TypeKey::Slice { .. }) => Ok(None),
            _ => Err(()),
        }
    }

    fn access_kind(&self, i: u32) -> Option<AccessKind> {
        let op = *self.b.insts.op.get(i as usize)?;
        let args = self.b.args(i);
        match op {
            Op::ElemAddr => self.len_of_ptr(*args.first()?).ok().map(|_| AccessKind::Elem),
            Op::Get | Op::Set => {
                let at = self.ty(*args.first()?)?;
                matches!(self.key(at), Some(TypeKey::Array { .. })).then_some(if op == Op::Get {
                    AccessKind::Get
                } else {
                    AccessKind::Set
                })
            }
            Op::Intrinsic(_) if self.vop(i).is_some_and(|v| v.is_memory()) => {
                crate::simd::mem_operand(self.types, self.ty(*args.first()?)?).map(|_| AccessKind::Vector)
            }
            _ => None,
        }
    }

    /// The vector op of instruction `i`, if it is one.
    fn vop(&self, i: u32) -> Option<VecOp> {
        match self.b.insts.op.get(i as usize)? {
            Op::Intrinsic(id) => self.vops.get(id.index()).copied().flatten(),
            _ => None,
        }
    }

    /// Lane count of vector access `i`.
    fn vec_lanes(&self, i: u32) -> Option<u64> {
        let t = if self.vop(i)?.is_store() {
            self.ty(*self.b.args(i).get(2)?)?
        } else {
            (*self.b.insts.ty.get(i as usize)?)?
        };
        crate::simd::vec_shape(self.types, t).map(|s| s.2 as u64)
    }

    /// Whether `lim` is `sub.sat n N-1` with `n` the length of what `p` points to.
    fn is_vec_lim(&self, lim: Operand, p: Operand, len: Option<u64>, lanes: u64) -> bool {
        if let (Some(l), Some(k)) = (len, self.konst(lim)) {
            return k.saturating_add(lanes - 1) <= l;
        }
        let Operand::Value(v) = lim else { return false };
        self.def(v).is_some_and(|d| {
            let a = self.b.args(d);
            self.b.insts.op.get(d as usize) == Some(&Op::SubSat)
                && a.len() == 2
                && self.konst(a[1]) == Some(lanes - 1)
                && self.is_len_of(a[0], p, len)
        })
    }

    /// Whether length operand `n` is the length of the object `p` points to (`len`: its array
    /// length, `None` = a slice): `slice.len p`, or a constant ≤ the array length.
    fn is_len_of(&self, n: Operand, p: Operand, len: Option<u64>) -> bool {
        match (len, self.konst(n)) {
            (Some(l), Some(k)) => k <= l,
            (None, _) => match n {
                Operand::Value(v) => self.def(v).is_some_and(|d| {
                    self.b.insts.op.get(d as usize) == Some(&Op::SliceLen)
                        && self.b.args(d).first() == Some(&p)
                }),
                _ => false,
            },
            _ => false,
        }
    }

    /// `a` dominates `b` (instruction indices): an earlier instruction of the same block, or any
    /// instruction of a strictly dominating block.
    fn inst_dominates(&self, a: u32, b: u32) -> bool {
        let (ba, bb) = (self.block_of[a as usize], self.block_of[b as usize]);
        if ba == bb { a < b } else { self.dom.dominates(ba as usize, bb as usize) }
    }

    fn guard(&self, i: u32, kind: AccessKind) -> Option<Guard> {
        let args = self.b.args(i);
        match kind {
            AccessKind::Elem => {
                let (p, idx) = (args[0], *args.get(1)?);
                let len = self.len_of_ptr(p).ok()?;
                if let (Some(k), Some(l)) = (self.konst(idx), len)
                    && k < l
                {
                    return Some(Guard::Proof(Proof::ConstInRange { index: k, len: l }));
                }
                if let Some(c) = self.dominating_check(i, idx, |n| self.is_len_of(n, p, len)) {
                    return Some(Guard::Check(c));
                }
                if self.konst(idx) == Some(0) && self.only_slice_base(i) {
                    return Some(Guard::Proof(Proof::SliceBase));
                }
                if let Some(br) = self.loop_bound(i, idx, |n| self.is_len_of(n, p, len)) {
                    return Some(Guard::Proof(Proof::LoopBound { branch: br }));
                }
                let n = match len {
                    Some(l) => facts::Term::constant(l)?,
                    None => self.slice_len_term(p)?,
                };
                self.prove_lt(i, self.term(idx)?, n).map(Guard::Proof)
            }
            AccessKind::Get | AccessKind::Set => {
                let (a, idx) = (args[0], *args.get(1)?);
                let Some(TypeKey::Array { len, .. }) = self.ty(a).and_then(|t| self.key(t)) else {
                    return None;
                };
                match self.konst(idx) {
                    Some(k) if k < *len => Some(Guard::Proof(Proof::ConstInRange { index: k, len: *len })),
                    Some(_) => None,
                    None => Some(Guard::SelfChecked),
                }
            }
            AccessKind::Vector => {
                if matches!(
                    self.vop(i),
                    Some(VecOp::LoadMasked | VecOp::StoreMasked | VecOp::Gather | VecOp::Scatter)
                ) {
                    return Some(Guard::SelfChecked);
                }
                let (p, idx) = (args[0], *args.get(1)?);
                let len = self.len_of_ptr(p).ok()?;
                let lanes = self.vec_lanes(i)?;
                if let (Some(k), Some(l)) = (self.konst(idx), len)
                    && k.checked_add(lanes).is_some_and(|e| e <= l)
                {
                    return Some(Guard::Proof(Proof::ConstInRange { index: k, len: l }));
                }
                if let Some(c) = self.dominating_check(i, idx, |n| self.is_vec_lim(n, p, len, lanes)) {
                    return Some(Guard::Check(c));
                }
                self.loop_bound(i, idx, |n| self.is_vec_lim(n, p, len, lanes))
                    .map(|_| Guard::Proof(Proof::ChunkLoop))
            }
        }
    }

    /// A `check.bounds idx n` dominating instruction `i` whose `n` satisfies `len_ok`.
    fn dominating_check(&self, i: u32, idx: Operand, len_ok: impl Fn(Operand) -> bool) -> Option<u32> {
        self.checks.iter().copied().find(|&c| {
            self.b.args(c).first() == Some(&idx)
                && self.b.args(c).get(1).is_some_and(|&n| len_ok(n))
                && self.inst_dominates(c, i)
        })
    }

    /// Every use of `addr.elem`'s result is the pointer operand of `slice.make`.
    fn only_slice_base(&self, i: u32) -> bool {
        let Some(Some(r)) = self.b.insts.result.get(i as usize) else { return false };
        let r = Operand::Value(*r);
        let mut used = false;
        for k in 0..self.b.insts.len() as u32 {
            let args = self.b.args(k);
            for (pos, &a) in args.iter().enumerate() {
                if a == r {
                    if self.b.insts.op[k as usize] != Op::SliceMake || pos != 0 {
                        return false;
                    }
                    used = true;
                }
            }
        }
        // Terminator and branch-argument uses are not `slice.make`.
        for bd in &self.b.blocks {
            let in_term = match bd.term {
                Term::Cbr { cond, .. } => cond == r,
                Term::Switch { scrut, .. } => scrut == r,
                Term::Ret(Some(o)) => o == r,
                _ => false,
            };
            if in_term {
                return false;
            }
        }
        for bi in 0..self.b.blocks.len() as u32 {
            for t in self.b.successors(Block(bi)) {
                if self.b.target_args(&t).contains(&r) {
                    return false;
                }
            }
        }
        used
    }

    /// The block ending in `cbr (lt idx n) T F` such that `T` dominates instruction `i` and `T`'s
    /// only predecessor edge is that true edge; `idx` unsigned, `len_ok(n)`.
    fn loop_bound(&self, i: u32, idx: Operand, len_ok: impl Fn(Operand) -> bool) -> Option<u32> {
        let unsigned = self
            .ty(idx)
            .and_then(|t| self.key(t))
            .is_some_and(|k| matches!(k, TypeKey::Int { signed: false, .. }));
        if !unsigned {
            return None;
        }
        let target = self.block_of[i as usize] as usize;
        let mut walk = Some(target);
        while let Some(t) = walk {
            if let [(d, true)] = self.preds[t][..]
                && let Term::Cbr { cond: Operand::Value(c), then, .. } = self.b.blocks[d as usize].term
                && then.block.index() == t
                && let Some(ci) = self.def(c)
                && self.b.insts.op[ci as usize] == Op::Lt
                && self.b.args(ci).first() == Some(&idx)
                && self.b.args(ci).get(1).is_some_and(|&n| len_ok(n))
            {
                return Some(d);
            }
            walk = self.dom.idom(t).filter(|&p| p != t);
        }
        None
    }

    /// The proof that `check.bounds` instruction `i` is redundant, if any.
    fn check_proof(&self, i: u32) -> Option<Proof> {
        let args = self.b.args(i);
        let (idx, n) = (*args.first()?, *args.get(1)?);
        if let (Some(k), Some(l)) = (self.konst(idx), self.konst(n))
            && k < l
        {
            return Some(Proof::ConstInRange { index: k, len: l });
        }
        if let Some(first) = self
            .checks
            .iter()
            .copied()
            .find(|&c| c != i && self.b.args(c) == args && self.inst_dominates(c, i))
        {
            return Some(Proof::Dominated { first });
        }
        let same_len =
            |m: Operand| m == n || matches!((self.konst(m), self.konst(n)), (Some(a), Some(b)) if a <= b);
        if let Some(branch) = self.loop_bound(i, idx, same_len) {
            return Some(Proof::LoopBound { branch });
        }
        self.prove_lt(i, self.term(idx)?, self.term(n)?)
    }
}

/// Predecessor edges per block: (reachable predecessor, is the true edge of a `cbr`).
fn preds_of(b: &Body, dom: &Dominators) -> Vec<Vec<(u32, bool)>> {
    let mut preds: Vec<Vec<(u32, bool)>> = vec![Vec::new(); b.blocks.len()];
    for (bi, bd) in b.blocks.iter().enumerate() {
        if !dom.reachable(bi) {
            continue;
        }
        let mut add = |t: Block, is_then: bool| {
            if let Some(p) = preds.get_mut(t.index()) {
                p.push((bi as u32, is_then));
            }
        };
        match bd.term {
            Term::Br(t) => add(t.block, false),
            Term::Cbr { then, els, .. } => {
                add(then.block, true);
                add(els.block, false);
            }
            Term::Switch { cases, default, .. } => {
                for c in b.switch_cases(cases) {
                    add(c.target.block, false);
                }
                add(default.block, false);
            }
            Term::Ret(_) | Term::Unreachable => {}
        }
    }
    preds
}

/// Removes instructions `dead` (sorted, none with a result) from `b`, keeping block order.
fn remove_insts(b: &mut Body, dead: &[u32]) {
    let n = b.insts.len();
    let mut keep = vec![true; n];
    for &d in dead {
        if let Some(k) = keep.get_mut(d as usize) {
            *k = false;
        }
    }
    let mut new_idx = vec![u32::MAX; n];
    let mut next = 0u32;
    for (i, k) in keep.iter().enumerate() {
        if *k {
            new_idx[i] = next;
            next += 1;
        }
    }
    let ins = std::mem::take(&mut b.insts);
    let pick = |i: usize| keep[i];
    b.insts.op = ins.op.into_iter().enumerate().filter(|(i, _)| pick(*i)).map(|x| x.1).collect();
    b.insts.ty = ins.ty.into_iter().enumerate().filter(|(i, _)| pick(*i)).map(|x| x.1).collect();
    b.insts.result = ins.result.into_iter().enumerate().filter(|(i, _)| pick(*i)).map(|x| x.1).collect();
    b.insts.args = ins.args.into_iter().enumerate().filter(|(i, _)| pick(*i)).map(|x| x.1).collect();
    b.insts.site = ins.site.into_iter().enumerate().filter(|(i, _)| pick(*i)).map(|x| x.1).collect();
    for bd in &mut b.blocks {
        let lo = bd.insts.start as usize;
        let hi = lo + bd.insts.len as usize;
        let kept = (lo..hi.min(n)).filter(|&i| keep[i]).count() as u32;
        let start = (lo..n).map(|i| new_idx[i]).find(|&x| x != u32::MAX).unwrap_or(next);
        bd.insts.start = start;
        bd.insts.len = kept;
    }
    for v in &mut b.values {
        if let ValueDef::Inst(i) = &mut v.def
            && let Some(&ni) = new_idx.get(*i as usize)
        {
            *i = ni;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FuncBuilder, InstanceId};

    fn module() -> (Module, TypeId, TypeId, TypeId) {
        let mut m = Module::new();
        let us = m.types.usize();
        let u32t = m.types.u32();
        let arr = m.types.array(u32t, 4);
        let pa = m.types.ref_(false, arr);
        (m, us, u32t, pa)
    }

    /// `fn f(a: &[u32; 4], i: usize) -> u32 { return (*a)[i]; }` with and without its check.
    #[test]
    fn checked_unchecked_and_const() {
        for checked in [true, false] {
            let (mut m, us, u32t, pa) = module();
            let pe = m.types.ref_(false, u32t);
            let f = m.declare(InstanceId(vec![0]), "app.f", vec![pa, us], Some(u32t));
            let mut b = FuncBuilder::new(&[pa, us]);
            let (a, i) = (b.param(0).unwrap(), b.param(1).unwrap());
            if checked {
                let four = b.int(4);
                b.inst(Op::BoundsCheck, None, &[i.into(), four]);
            }
            let p = b.inst(Op::ElemAddr, Some(pe), &[a.into(), i.into()]).unwrap();
            let two = b.int(2);
            let q = b.inst(Op::ElemAddr, Some(pe), &[a.into(), two]).unwrap();
            let x = b.load(u32t, p);
            let y = b.load(u32t, q);
            let s = b.inst(Op::AddWrap, Some(u32t), &[x.into(), y.into()]).unwrap();
            b.ret(Some(s.into()));
            m.define(f, b.finish().unwrap());
            let r = audit(&m);
            if checked {
                let acc = r.unwrap();
                assert!(matches!(acc[0].guard, Guard::Check(0)));
                assert_eq!(acc[1].guard, Guard::Proof(Proof::ConstInRange { index: 2, len: 4 }));
            } else {
                let bad = r.unwrap_err();
                assert_eq!(bad.len(), 1);
                assert!(bad[0].to_string().contains("app.f"), "{}", bad[0]);
            }
        }
    }

    #[test]
    fn const_check_is_elided_with_proof() {
        let (mut m, us, u32t, pa) = module();
        let pe = m.types.ref_(false, u32t);
        let f = m.declare(InstanceId(vec![0]), "app.g", vec![pa], Some(u32t));
        let mut b = FuncBuilder::new(&[pa]);
        let a = b.param(0).unwrap();
        let (one, four) = (b.int(1), b.int(4));
        b.inst(Op::BoundsCheck, None, &[one, four]);
        let p = b.inst(Op::ElemAddr, Some(pe), &[a.into(), one]).unwrap();
        let x = b.load(u32t, p);
        b.ret(Some(x.into()));
        let _ = us;
        m.define(f, b.finish().unwrap());
        let e = elide(&mut m);
        assert_eq!(e.len(), 1);
        assert_eq!(e[0].proof, Proof::ConstInRange { index: 1, len: 4 });
        let body = m.funcs[0].body.as_ref().unwrap();
        assert!(!body.insts.op.contains(&Op::BoundsCheck));
        crate::verify_module(&m).unwrap();
        audit(&m).unwrap();
    }
}
