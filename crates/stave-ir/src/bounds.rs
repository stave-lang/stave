//! Bounds checks of element accesses (panic `Bounds`; no silent check removal): the audit that
//! every access keeps its check, and the removal of a check only with a witness.
//!
//! **Insertion** happens in the front end's lowering (stave-sema), which gives every access its
//! check:
//!
//! - `xs[i]` on a place in memory (a slice, or an array behind a reference or in a `slot`):
//!   `check.bounds i n` right before `addr.elem p i`, with `n = slice.len p` or the array length.
//! - `a[i]` on an array value (SSA): `get a i` / `set a i v`, which trap `Bounds` themselves for a
//!   dynamic index.
//! - a constant index ≥ the array length: an unconditional `panic.bounds` (the trap is kept, the
//!   access is never emitted).
//! - compiler-made element accesses (`for` over an array or slice, drop loops over arrays, the
//!   array → slice coercion) index below a length the generated code itself compares against.
//! - a sub-slice `xs[a..b]`: branches to `panic.bounds` unless `b <= n` and `a <= b`, then
//!   `slice.make (addr.elem p a) (b - a)`; its start may be `n` (an empty slice at the end).
//! - `vec.load` / `vec.store` (and their aligned forms) of N lanes at `i`: `check.bounds i lim`
//!   with `lim = sub.sat n N-1` (so `i < lim` iff `i + N <= n`), or the `ChunkLoop` proof: a chunk
//!   loop's own head test `cbr (lt i lim)` with the same `lim`, or the facts below proving
//!   `i + N-1 < n` (`while i + N <= n`). Masked loads and stores, gathers and scatters check their
//!   active lanes themselves.
//!
//! **Audit** ([`audit`]): every element access is classified with its [`Guard`]: a dominating
//! `check.bounds` on the same index and length, the op's own trap, or a [`Proof`]. An access with
//! none is [`Unguarded`], a compiler bug (the driver reports it as an internal error). Accesses
//! through raw pointers (`addr.elem` on `*T`) are pointer arithmetic, the `unsafe` code's own
//! obligation, and are not element accesses.
//!
//! **Removal** ([`prove_func`], [`elide`]): the one pass that decides which checks a build
//! keeps. It runs on the lowered IR of every build, debug and release alike, and depends on
//! nothing but the function's body and its callees' bodies, so a function keeps the same checks
//! in every build mode. A `check.bounds` goes only when one of these proofs holds:
//!
//! | Proof | Holds when |
//! |---|---|
//! | `ConstInRange` | index and length are constants and index < length |
//! | `Dominated` | a `check.bounds` on the same index and length dominates it |
//! | `LoopBound` | a dominating `cbr (lt i n') T F` reaches the check only through `T` (T's only predecessor), `i` unsigned, and `n'` is the same length (or a constant ≤ a constant length) |
//! | `LenEq` | the facts below prove `i < n` through an equality of two lengths, e.g. `i < len(a)` and `len(a) == len(b)` for `check.bounds i len(b)` |
//! | `Offset` | the facts below prove `i + k < n` for the index `i + k` (`k` a constant, possibly negative or 0) without an equality; for a check against `n -| c` (a vector access), `k` includes `c`: the last lane |
//!
//! "The same" index or length is the same operand, or two `slice.len` of one slice value (the
//! lowering reads the length again at every use). `Dominated` and `LoopBound` are found for all
//! checks of a function at once, by sorting them with the candidate sources and one scan in
//! dominator-tree order ([`Cx::twins`], [`Cx::loop_heads`]).
//!
//! `LenEq` and `Offset` come from one difference-constraint graph per function (`bounds/facts.rs`):
//! facts `x <= y + c` over SSA values from dominating branches (unsigned compares, a loop's
//! `lt i n` among them) and asserting calls (`std.debug.assert(a.len() == b.len(), ..)`), with
//! trapping `add`/`sub` by a constant folded into offsets. Each check is a query on it, explored
//! from the index for at most [`BOUNDS_STEP_LIMIT`] edges.
//!
//! The pass deletes nothing. Its result is a witness per check ([`crate::witness`]): the proof
//! written out as a chain of inequalities, each naming the place of the IR it is read from. The
//! separate checker [`crate::witness::check`] validates every chain and then deletes the checks
//! it accepted; a chain it refuses is a compiler bug. The [`Proof`] is the label of the rule
//! that found the chain, for reports and tests.

mod facts;

use crate::ir::{Block, Body, Func, FuncId, Module, Op, Operand, Span, Term, Value, ValueDef};
use crate::simd::VecOp;
use crate::types::{TypeId, TypeKey, TypeTable};
use crate::verify::Dominators;
use crate::witness::check::{Rejected, check_module, remove};
use crate::witness::{Edge, FnWitness, NO_BLOCK, Node, Removed, Source};

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
    /// `addr.elem p i` whose result only feeds `slice.make` (a sub-slice's start, never
    /// dereferenced by itself), with `i <= len` proven by the facts (`i == len` starts an empty
    /// slice).
    SliceStart,
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
    /// The facts prove `base + k` below the length without an equality (`base` an SSA value
    /// number, `None` for a constant index): the index, or a vector access's last lane.
    Offset { base: Option<u32>, k: i128 },
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

/// A removed `check.bounds` and the rule that proved it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Elided {
    pub func: FuncId,
    pub path: String,
    /// Instruction index before the removal.
    pub inst: u32,
    pub proof: Proof,
}

/// The checks of one function the pass proved: their witness, and per row of
/// [`FnWitness::removed`] the rule that found it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proved {
    pub witness: FnWitness,
    pub proofs: Vec<Proof>,
}

/// What [`analyze_func`] finds in one function.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Analysis {
    /// The checks to remove (`None`: none).
    pub proved: Option<Proved>,
    /// The element accesses without a check or proof.
    pub unguarded: Vec<Unguarded>,
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
    let Some(f) = m.func(fid) else { return (Vec::new(), Vec::new()) };
    let Some(b) = &f.body else { return (Vec::new(), Vec::new()) };
    if !has_access(b) {
        return (Vec::new(), Vec::new());
    }
    Cx::new(m, b).audit(f, fid)
}

/// Only these ops can be element accesses: without one there is nothing to audit.
fn has_access(b: &Body) -> bool {
    b.insts.op.iter().any(|op| match op {
        Op::ElemAddr | Op::Get | Op::Set => true,
        Op::Vec(v) => v.is_memory(),
        _ => false,
    })
}

/// The witnesses of the `check.bounds` of function `fid` that have a proof (`None`: no check
/// has one). Nothing is removed. Functions are independent: a caller may run them in parallel.
pub fn prove_func(m: &Module, fid: FuncId) -> Option<Proved> {
    let b = m.func(fid)?.body.as_ref()?;
    if !b.insts.op.contains(&Op::BoundsCheck) {
        return None;
    }
    Cx::new(m, b).prove(fid)
}

/// [`prove_func`] and the unguarded accesses of [`audit_func`] from one analysis of the
/// function. The audit sees the body with every check in place: an access behind a check that
/// has a witness stays guarded by that check, whose removal the witness checker answers for.
pub fn analyze_func(m: &Module, fid: FuncId) -> Analysis {
    let Some(f) = m.func(fid) else { return Analysis::default() };
    let Some(b) = &f.body else { return Analysis::default() };
    let (checks, accesses) = (b.insts.op.contains(&Op::BoundsCheck), has_access(b));
    if !checks && !accesses {
        return Analysis::default();
    }
    let cx = Cx::new(m, b);
    Analysis {
        proved: if checks { cx.prove(fid) } else { None },
        unguarded: if accesses { cx.audit(f, fid).1 } else { Vec::new() },
    }
}

/// The whole removal step on a module, function by function: proves, has the witness checker
/// validate every chain, and lets it delete the checks. Returns the records in (function,
/// instruction) order, or the witnesses the checker refused (then nothing is removed).
/// Accesses stay guarded ([`audit`] holds before and after).
pub fn elide(m: &mut Module) -> Result<Vec<Elided>, Vec<Rejected>> {
    let proved: Vec<Proved> = (0..m.funcs.len() as u32).filter_map(|f| prove_func(m, FuncId(f))).collect();
    let witnesses: Vec<FnWitness> = proved.iter().map(|p| p.witness.clone()).collect();
    let checked = check_module(m, &witnesses)?;
    let mut out = Vec::new();
    for p in &proved {
        let path = m.func(p.witness.func).map_or(String::new(), label);
        out.extend(p.witness.removed.iter().zip(&p.proofs).map(|(r, &proof)| Elided {
            func: p.witness.func,
            path: path.clone(),
            inst: r.inst,
            proof,
        }));
    }
    remove(m, &checked);
    Ok(out)
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
        Cx { types, funcs, b, dom, block_of, preds, checks, facts: facts::Facts::default() }
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
            Op::Vec(v) if v.is_memory() => {
                crate::simd::mem_operand(self.types, self.ty(*args.first()?)?).map(|_| AccessKind::Vector)
            }
            _ => None,
        }
    }

    /// The vector op of instruction `i`, if it is one.
    fn vop(&self, i: u32) -> Option<VecOp> {
        match self.b.insts.op.get(i as usize)? {
            Op::Vec(v) => Some(*v),
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
                    Some(l) => facts::Term::constant(l),
                    None => self.slice_len_term(p)?,
                };
                let t = self.term(idx)?;
                if let Some(proof) = self.prove_lt(i, t, n, None) {
                    return Some(Guard::Proof(proof));
                }
                (self.only_slice_base(i) && self.prove_le(i, t, n).is_some())
                    .then_some(Guard::Proof(Proof::SliceStart))
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
                if self.loop_bound(i, idx, |n| self.is_vec_lim(n, p, len, lanes)).is_some() {
                    return Some(Guard::Proof(Proof::ChunkLoop));
                }
                let n = match len {
                    Some(l) => facts::Term::constant(l),
                    None => self.slice_len_term(p)?,
                };
                let last = self.term(idx)?.plus(i128::from(lanes) - 1)?;
                self.prove_lt(i, last, n, None).map(Guard::Proof)
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

    /// The audit of the function: its guarded and its unguarded element accesses.
    fn audit(&self, f: &Func, fid: FuncId) -> (Vec<Access>, Vec<Unguarded>) {
        let mut ok = Vec::new();
        let mut bad = Vec::new();
        for i in 0..self.b.insts.len() as u32 {
            let Some(kind) = self.access_kind(i) else { continue };
            if !self.reachable_inst(i) {
                continue;
            }
            match self.guard(i, kind) {
                Some(guard) => ok.push(Access { func: fid, inst: i, kind, guard }),
                None => bad.push(Unguarded { func: fid, path: label(f), inst: i, kind }),
            }
        }
        (ok, bad)
    }

    /// The witnesses of every check that has a proof.
    fn prove(&self, func: FuncId) -> Option<Proved> {
        let twins = self.twins();
        let heads = self.loop_heads();
        let mut w = FnWitness { func, dom: Vec::new(), removed: Vec::new(), edges: Vec::new() };
        let mut proofs = Vec::new();
        for (k, &i) in self.checks.iter().enumerate() {
            let start = w.edges.len();
            match self.check_proof(i, twins[k], heads[k], &mut w.edges) {
                Some(proof) => {
                    let edges = Span { start: start as u32, len: (w.edges.len() - start) as u32 };
                    w.removed.push(Removed { inst: i, site: self.b.insts.site[i as usize], edges });
                    proofs.push(proof);
                }
                None => w.edges.truncate(start),
            }
        }
        if proofs.is_empty() {
            return None;
        }
        w.dom = (0..self.b.blocks.len()).map(|bk| self.dom.idom(bk).map_or(NO_BLOCK, |d| d as u32)).collect();
        Some(Proved { witness: w, proofs })
    }

    /// The key two operands share exactly when they are "the same": the operand itself, or the
    /// slice value whose length it reads.
    fn same_key(&self, o: Operand) -> u64 {
        let (tag, id) = match o {
            Operand::Value(v) => match self.len_read(v) {
                Some((_, s)) => (5u64, s.0),
                None => (0, v.0),
            },
            Operand::Const(c) => (1, c.0),
            Operand::Str(x) => (2, x.0),
            Operand::Func(f) => (3, f.0),
            Operand::Type(t) => (4, t.0),
        };
        tag << 32 | u64::from(id)
    }

    /// `(d, s)` when `v` is defined by instruction `d` as `slice.len s`, `s` a value.
    fn len_read(&self, v: Value) -> Option<(u32, Value)> {
        let d = self.def(v)?;
        match (self.b.insts.op[d as usize], self.b.args(d)) {
            (Op::SliceLen, &[Operand::Value(s)]) => Some((d, s)),
            _ => None,
        }
    }

    /// The edges from operand `a` to "the same" operand `b`: none, or the two `slice.len`
    /// readings of their slice.
    fn bridge(&self, a: Operand, b: Operand, out: &mut Vec<Edge>) -> Option<()> {
        if a == b {
            return Some(());
        }
        let (Operand::Value(va), Operand::Value(vb)) = (a, b) else { return None };
        let ((da, sa), (db, sb)) = (self.len_read(va)?, self.len_read(vb)?);
        if sa != sb {
            return None;
        }
        let len = Node::Len(sa);
        out.push(Edge { from: Node::Val(va), to: len, w: 0, src: Source::SliceLen { inst: da } });
        out.push(Edge { from: len, to: Node::Val(vb), w: 0, src: Source::SliceLen { inst: db } });
        Some(())
    }

    /// Per check (position in `checks`), the nearest `check.bounds` on the same index and length
    /// that dominates it. Checks are sorted by operands, then by the dominator-tree preorder of
    /// their place; one scan per group keeps the group's checks that dominate the current one
    /// on a stack.
    fn twins(&self) -> Vec<Option<u32>> {
        let mut rows: Vec<(u64, u64, u32, u32, u32)> = self
            .checks
            .iter()
            .enumerate()
            .filter_map(|(k, &i)| {
                let &[idx, n] = self.b.args(i) else { return None };
                let order = self.dom.preorder(self.block_of[i as usize] as usize);
                Some((self.same_key(idx), self.same_key(n), order, i, k as u32))
            })
            .collect();
        rows.sort_unstable();
        let mut twin = vec![None; self.checks.len()];
        let mut above: Vec<u32> = Vec::new();
        let mut group = None;
        for &(idx, n, _, i, k) in &rows {
            if group != Some((idx, n)) {
                group = Some((idx, n));
                above.clear();
            }
            while above.last().is_some_and(|&c| !self.inst_dominates(c, i)) {
                above.pop();
            }
            twin[k as usize] = above.last().copied();
            above.push(i);
        }
        twin
    }

    /// Per check (position in `checks`), a loop bound that proves it: `(branch, head)` where
    /// block `branch` ends in `cbr (lt i n') head F`, that true edge is the only way into
    /// `head`, `head` dominates the check, `i` is the check's (unsigned) index, and `n'` is the
    /// check's length or a constant ≤ its constant length. Found like [`Cx::twins`]: the heads
    /// and the checks of one index and length sorted by dominator-tree preorder, one scan with
    /// the stack of heads that dominate the current row.
    fn loop_heads(&self) -> Vec<Option<(u32, u32)>> {
        // A constant length is its own class: any head testing a constant at most it will do.
        const CONSTANT: u64 = u64::MAX;
        let len_key = |n: Operand| if self.konst(n).is_some() { CONSTANT } else { self.same_key(n) };
        // (index key, length key, preorder, 0 = head / 1 = check, block or position in `checks`,
        // branch block, constant bound)
        let mut rows: Vec<(u64, u64, u32, u8, u32, u32, u64)> = Vec::new();
        for t in 0..self.b.blocks.len() {
            if let [(d, true)] = self.preds[t][..]
                && let Term::Cbr { cond: Operand::Value(c), then, .. } = self.b.blocks[d as usize].term
                && then.block.index() == t
                && let Some(ci) = self.def(c)
                && self.b.insts.op[ci as usize] == Op::Lt
                && let &[idx, n] = self.b.args(ci)
                && self
                    .ty(idx)
                    .and_then(|ty| self.key(ty))
                    .is_some_and(|k| matches!(k, TypeKey::Int { signed: false, .. }))
            {
                let bound = self.konst(n).unwrap_or(0);
                rows.push((self.same_key(idx), len_key(n), self.dom.preorder(t), 0, t as u32, d, bound));
            }
        }
        let mut found = vec![None; self.checks.len()];
        if rows.is_empty() {
            return found;
        }
        for (k, &i) in self.checks.iter().enumerate() {
            let &[idx, n] = self.b.args(i) else { continue };
            let order = self.dom.preorder(self.block_of[i as usize] as usize);
            rows.push((self.same_key(idx), len_key(n), order, 1, k as u32, 0, self.konst(n).unwrap_or(0)));
        }
        rows.sort_unstable();
        // The heads above the current row, each with the head of the smallest constant bound at
        // or below it on the stack: (head block, branch block, bound, best head, best branch,
        // best bound).
        let mut above: Vec<(u32, u32, u64, u32, u32, u64)> = Vec::new();
        let mut group = None;
        for &(idx, n, _, kind, at, branch, bound) in &rows {
            if group != Some((idx, n)) {
                group = Some((idx, n));
                above.clear();
            }
            let block = if kind == 0 { at } else { self.block_of[self.checks[at as usize] as usize] };
            while above.last().is_some_and(|h| !self.dom.dominates(h.0 as usize, block as usize)) {
                above.pop();
            }
            if kind == 0 {
                let best = match above.last() {
                    Some(&(_, _, _, t, d, b)) if b <= bound => (t, d, b),
                    _ => (at, branch, bound),
                };
                above.push((at, branch, bound, best.0, best.1, best.2));
            } else if let Some(&(t, d, _, best_t, best_d, best_bound)) = above.last() {
                if n != CONSTANT {
                    found[at as usize] = Some((d, t));
                } else if best_bound <= bound {
                    found[at as usize] = Some((best_d, best_t));
                }
            }
        }
        found
    }

    /// The proof that `check.bounds` instruction `i` is redundant, if any, with its chain of
    /// witness edges appended to `out`. `twin` and `head`: the check's rows of [`Cx::twins`] and
    /// [`Cx::loop_heads`].
    fn check_proof(
        &self,
        i: u32,
        twin: Option<u32>,
        head: Option<(u32, u32)>,
        out: &mut Vec<Edge>,
    ) -> Option<Proof> {
        let args = self.b.args(i);
        let (idx, n) = (*args.first()?, *args.get(1)?);
        if let (Some(k), Some(l)) = (self.konst(idx), self.konst(n))
            && k < l
        {
            return Some(Proof::ConstInRange { index: k, len: l });
        }
        let start = out.len();
        if let Some(first) = twin
            && let &[x, y] = self.b.args(first)
        {
            if self.less_edge((idx, n), (x, y), Source::Check { inst: first }, out).is_some() {
                return Some(Proof::Dominated { first });
            }
            out.truncate(start);
        }
        if let Some((branch, head)) = head
            && let Term::Cbr { cond: Operand::Value(c), .. } = self.b.blocks[branch as usize].term
            && let Some(ci) = self.def(c)
            && let &[x, y] = self.b.args(ci)
        {
            if self.less_edge((idx, n), (x, y), Source::Branch { block: head }, out).is_some() {
                return Some(Proof::LoopBound { branch });
            }
            out.truncate(start);
        }
        let t = self.term(idx)?;
        let mut path = Vec::new();
        if let Some(tn) = self.term(n)
            && let Some(p) = self.prove_lt(i, t, tn, Some(&mut path))
        {
            return self.chain(idx, &path, n, out).map(|()| p);
        }
        // `idx < n -| c` iff `idx + c < n` (the limit of a vector access of `c + 1` lanes)
        let (len, c) = self.sat_bound(n)?;
        let p = self.prove_lt(i, t.plus(c)?, self.term(len)?, Some(&mut path))?;
        self.chain(idx, &path, len, out)?;
        let (Operand::Value(lim), Some((from, k))) = (n, self.written(len)) else { return None };
        out.push(Edge { from, to: Node::Val(lim), w: c - k, src: Source::SubSat { inst: self.def(lim)? } });
        Some(p)
    }

    /// The chain from `idx` to `n` through `x < y` read at `src`, for a check `(idx, n)` whose
    /// index is the same as `x` and whose length is the same as `y`, or a constant like `y`.
    fn less_edge(
        &self,
        check: (Operand, Operand),
        less: (Operand, Operand),
        src: Source,
        out: &mut Vec<Edge>,
    ) -> Option<()> {
        let ((idx, n), (x, y)) = (check, less);
        self.bridge(idx, x, out)?;
        let ((from, kx), (to, ky)) = (self.written(x)?, self.written(y)?);
        out.push(Edge { from, to, w: ky - kx - 1, src });
        if self.konst(y).is_none() || self.konst(n).is_none() {
            self.bridge(y, n, out)?;
        }
        Some(())
    }

    /// The chain from `idx` to `n` along the facts `path`: the index folded down to its term,
    /// each fact between its terms, and the length's term unfolded up to the length.
    fn chain(&self, idx: Operand, path: &[u32], n: Operand, out: &mut Vec<Edge>) -> Option<()> {
        self.down(idx, out);
        for &f in path {
            self.fact_edges(f, out)?;
        }
        self.up(n, out);
        Some(())
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

    /// `&(*a)[i..j]`: the start `addr.elem a i` may be one past the end; `j <= 4` and `i <= j`
    /// guard it (`SliceStart`), `j <= 4` alone does not.
    #[test]
    fn sub_slice_start() {
        for tested in [true, false] {
            let (mut m, us, u32t, pa) = module();
            let pe = m.types.ref_(false, u32t);
            let sl = m.types.slice(u32t);
            let ps = m.types.ref_(false, sl);
            let bt = m.types.bool();
            let f = m.declare(InstanceId(vec![0]), "app.s", vec![pa, us, us], Some(ps));
            let mut b = FuncBuilder::new(&[pa, us, us]);
            let (a, i, j) = (b.param(0).unwrap(), b.param(1).unwrap(), b.param(2).unwrap());
            let trap_if_gt = |b: &mut FuncBuilder, x: Operand, y: Operand| {
                let c = b.inst(Op::Gt, Some(bt), &[x, y]).unwrap();
                let (trap, ok) = (b.create_block(), b.create_block());
                b.cbr(c, trap, &[], ok, &[]);
                b.switch_to(trap);
                b.inst(Op::Panic(crate::PanicKind::Bounds), None, &[]);
                b.unreachable();
                b.switch_to(ok);
            };
            let four = b.int(4);
            trap_if_gt(&mut b, j.into(), four);
            if tested {
                trap_if_gt(&mut b, i.into(), j.into());
            }
            let n = b.inst(Op::SubWrap, Some(us), &[j.into(), i.into()]).unwrap();
            let p = b.inst(Op::ElemAddr, Some(pe), &[a.into(), i.into()]).unwrap();
            let s = b.inst(Op::SliceMake, Some(ps), &[p.into(), n.into()]).unwrap();
            b.ret(Some(s.into()));
            m.define(f, b.finish().unwrap());
            crate::verify_module(&m).unwrap();
            let r = audit(&m);
            if tested {
                assert_eq!(r.unwrap()[0].guard, Guard::Proof(Proof::SliceStart));
            } else {
                assert_eq!(r.unwrap_err().len(), 1);
            }
        }
    }

    /// `&s[c..c + 1]` with `c` far above any length: `c + 1 <= len` from the branch guards the
    /// start (constants of every `u64` value are terms of the facts).
    #[test]
    fn sub_slice_huge_constant_start() {
        let (mut m, us, u32t, _) = module();
        let pe = m.types.ref_(false, u32t);
        let sl = m.types.slice(u32t);
        let ps = m.types.ref_(false, sl);
        let bt = m.types.bool();
        let f = m.declare(InstanceId(vec![0]), "app.h", vec![ps], Some(ps));
        let mut b = FuncBuilder::new(&[ps]);
        let s = b.param(0).unwrap();
        let n = b.inst(Op::SliceLen, Some(us), &[s.into()]).unwrap();
        let (c, end, one) = (b.int(1 << 62), b.int((1 << 62) + 1), b.int(1));
        let bad = b.inst(Op::Gt, Some(bt), &[end, n.into()]).unwrap();
        let (trap, ok) = (b.create_block(), b.create_block());
        b.cbr(bad, trap, &[], ok, &[]);
        b.switch_to(trap);
        b.inst(Op::Panic(crate::PanicKind::Bounds), None, &[]);
        b.unreachable();
        b.switch_to(ok);
        let p = b.inst(Op::ElemAddr, Some(pe), &[s.into(), c]).unwrap();
        let r = b.inst(Op::SliceMake, Some(ps), &[p.into(), one]).unwrap();
        b.ret(Some(r.into()));
        m.define(f, b.finish().unwrap());
        crate::verify_module(&m).unwrap();
        let acc = audit(&m).unwrap();
        assert!(matches!(acc[0].guard, Guard::Proof(Proof::Offset { base: None, k: 4611686018427387904 })));
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
        let e = elide(&mut m).unwrap();
        assert_eq!(e.len(), 1);
        assert_eq!(e[0].proof, Proof::ConstInRange { index: 1, len: 4 });
        let body = m.funcs[0].body.as_ref().unwrap();
        assert!(!body.insts.op.contains(&Op::BoundsCheck));
        crate::verify_module(&m).unwrap();
        audit(&m).unwrap();
    }
}
