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
//! - a constant index ≥ the array length: an unconditional `panic.bounds` (the trap is kept; the
//!   access and the code after it are lowered into a block nothing reaches, which the lowering's
//!   post-pass deletes).
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
//! none is [`Unguarded`], a compiler bug (the driver reports it as an internal error). What an
//! access indexes is decided in one place ([`Cx::elem_base`]): the slice `s` of an element
//! pointer `p = slice.ptr s` (nothing, when the elements of `s` are arrays or slices), else the
//! array or slice a reference or raw pointer points to, by its type. Every use of a `slice.ptr`
//! result is audited ([`AccessKind::ElemPtr`]): it must be an element access of that slice, or
//! read no element (a prefetch, a cast to an int, `slice.make` of it with `slice.len s` and the
//! type of `s`), so within a function a slice's elements are reached only through accesses of
//! the slice; an element pointer stored to memory and loaded back, or passed through a call, is
//! a raw pointer the audit does not follow. The one `addr.elem` that is no access is pointer
//! arithmetic of `unsafe` code: on a raw pointer `*T` (`T` not an array or slice) that is,
//! through pointer-to-pointer casts, the result of a `load` or a call. On any other pointer to
//! one element (the result of an element access, a parameter, ...) it is an access with nothing
//! to check against, so it is refused.
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
//! lowering reads the length again at every use). For `LenEq` and `Offset`, two reads of one int
//! field behind a `&` parameter are one node too (a `std.soa` column's length is its Soa's `len`).
//! `Dominated` and `LoopBound` are found for all checks of a function at once, by sorting them with
//! the candidate sources and one scan in dominator-tree order ([`Cx::twins`], [`Cx::loop_heads`]).
//!
//! `LenEq` and `Offset` come from one difference-constraint graph per function (`bounds/facts.rs`):
//! facts `x <= y + c` over SSA values from dominating branches (unsigned compares, a loop's
//! `lt i n` among them), asserting calls (`std.debug.assert(a.len() == b.len(), ..)`), unsigned
//! bit ops (`h & (n - 1) <= n - 1`, `h % n <= n - 1`, `x >> k <= x`), and the range of an
//! unsigned type (`b <= 255` for a `u8`), with trapping `add`/`sub` by a constant folded into
//! offsets and a widening unsigned cast into its operand. Each check is a query on it, explored
//! from the index for at most [`BOUNDS_STEP_LIMIT`] edges.
//!
//! **Contract checks** go the same way. A `requires` / `ensures` clause is one check per conjunct
//! of its top-level `&&`: `cbr c ok fail` with `fail` holding only the call of
//! `runtime.panic.requires_failed` / `ensures_failed`. Its goal is the inequality `c` states at the
//! end of the block that branches (a compare of ints under `not`s; `==` is two goals, `!=` none),
//! asked of the same graph; a goal between a constant and a value its type range decides is one
//! `Range` fact. Three answers: proved (every goal has a chain: the check goes, like a bounds
//! check, [`Proof::Clause`]); refuted (not proved, and a chain proves the negation of one goal:
//! the check fails whenever control gets there, [`Proved::refuted`]); unknown (the check stays). A
//! `requires` check sees the clauses before it and the parameters' type ranges; an `ensures`
//! check at a return sees everything that dominates that return.
//!
//! "Fails whenever reached" is vacuous where nothing reaches: there the goal holds too. So before
//! a refutation the pass looks for dead code, and finding it, reports none: (a) facts holding at
//! the point that contradict each other (a walk of negative weight from one of the first 16
//! nodes of the conditions holding there back to itself); every goal linked to that walk is then
//! proved with the walk counted as often as it needs (a multiplier), so a proof wins over the
//! refutation there; (b) a dominating branch on a compare of two constants, entered
//! on the side the compare never takes; (c) a dominating call that cannot return: its callee has
//! no `ret`, or asserts a parameter the call passes the constant `false`. Dead code of any other
//! shape (a loop that never ends, a callee that always panics through a path with a `ret`) is
//! not recognized, and a check there that the facts refute stays an error. Only the pass decides
//! this, the checker certifies no dead code: leaving a refutation out keeps a runtime check and
//! removes nothing.
//!
//! **Calls.** Functions meet only through summaries ([`crate::witness::Summary`], `bounds/summary.rs`):
//! rows over a function's parameters and result, read off its own body by folding alone
//! ([`claim_func`]), so the function's witness and all its callers read the same rows. After a
//! call, the callee's `post` rows and the `pre` rows that read a `requires` check are facts of
//! the graph. A call of a function whose summary has an entry is one more query, right before
//! the call: every `pre` row about the arguments proved (the call goes to the unchecked entry),
//! or the negation of a row that reads a check proved at a point not shown dead (the call always
//! fails: [`call_findings`]), or neither (the call keeps the checked entry). All or nothing per
//! call.
//!
//! The pass deletes nothing. Its result is a witness per check ([`crate::witness`]): the proof
//! written out as a chain of inequalities, each naming the place of the IR it is read from. The
//! separate checker [`crate::witness::check`] validates every chain, refutations included, and
//! then deletes the checks it accepted; a chain it refuses is a compiler bug. The [`Proof`] is the
//! label of the rule that found the chain, for reports and tests.

mod facts;
mod summary;

use std::sync::Arc;

use crate::ir::{Block, Body, Func, FuncId, Module, Op, Operand, Site, Span, Term, Value, ValueDef};
use crate::simd::VecOp;
use crate::types::{IntBits, TypeId, TypeKey, TypeTable};
use crate::verify::Dominators;
use crate::witness::check::{Rejected, check_module, remove};
use crate::witness::{
    CallProved, CallRefuted, Calls, Edge, FnWitness, NO_BLOCK, NOT_DEPTH, Node, Refuted, Removed, Source,
    Summary,
};

/// The most graph edges one bounds query follows before it gives up and keeps the check: compile
/// time stays linear in the number of checks.
pub const BOUNDS_STEP_LIMIT: u32 = 64;

/// What kind of element access an instruction is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AccessKind {
    /// `addr.elem p i`, unless `p` is a raw pointer from memory or a call ([`Cx::elem_base`]).
    Elem,
    /// `get a i` on an array value.
    Get,
    /// `set a i v` on an array value.
    Set,
    /// A vector load or store (`vec.load`, `vec.store_masked`, `vec.gather`, ...).
    Vector,
    /// `slice.ptr s`: the element pointer of slice `s`, guarded when it points to `s`'s element
    /// type and its every use is an element access of `s` or reads no element.
    ElemPtr,
}

impl AccessKind {
    /// What the audit reports for an unguarded access of this kind.
    pub fn unguarded_msg(self) -> String {
        match self {
            AccessKind::ElemPtr => concat!(
                "ElemPtr access: a slice.ptr result must point to the slice's element type and be used ",
                "only by an element access of that slice, a prefetch, slice.make with the slice's ",
                "slice.len and type, or a cast to an int"
            )
            .to_string(),
            k => format!("{k:?} access without a bounds check or proof"),
        }
    }
}

/// What the pointer operand of an element access points into ([`Cx::elem_base`]).
enum ElemBase {
    /// An array or slice object (the operand itself, or the slice whose element pointer it is)
    /// and its array length (`None`: a slice).
    Object(Operand, Option<u64>),
    /// A raw pointer to one element from memory or a call: pointer arithmetic, no access.
    Raw,
    /// Any other pointer: an access through it has nothing to be checked against.
    Unknown,
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
    /// A contract check whose condition the facts prove.
    Clause(Clause),
}

/// The kind of a contract check.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Clause {
    Requires,
    Ensures,
}

/// How an access is guarded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Guard {
    /// A dominating `check.bounds` (its instruction index).
    Check(u32),
    /// The op traps `Bounds` itself (`get`/`set` with a dynamic index).
    SelfChecked,
    /// A `slice.ptr` whose uses are each audited as an access of the slice or read no element.
    Uses,
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
        write!(f, "{}: inst {}: {}", self.path, self.inst, self.kind.unguarded_msg())
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

/// The checks of one function the pass proved or refuted: their witness, per row of
/// [`FnWitness::removed`] the rule that found it, and per row of [`FnWitness::refuted`] the kind
/// of the contract check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proved {
    pub witness: FnWitness,
    pub proofs: Vec<Proof>,
    pub refuted: Vec<Clause>,
}

impl Proved {
    /// The removed `check.bounds` (not contract checks).
    pub fn bounds_removed(&self) -> usize {
        self.proofs.iter().filter(|p| !matches!(p, Proof::Clause(_))).count()
    }
}

/// A contract check the pass proved or refuted, for diagnostics: read off a [`Proved`] and the
/// module it was proved on, before any removal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Finding {
    pub func: FuncId,
    pub clause: Clause,
    /// Proved (removed) or refuted (fails whenever reached).
    pub refuted: bool,
    /// The site of the check's failure call: the conjunct.
    pub site: Site,
    /// An `ensures` check: the site of the return it checks (its failure call's first argument).
    pub ret: Option<Site>,
    /// The sites of the facts of its proof, in order, each once.
    pub facts: Vec<Site>,
    /// The evaluation of the conjunct itself can trap or calls a fn: deleting the clause from
    /// the source would change the program even where its check is proved.
    pub traps: bool,
}

/// The contract checks of `p` (proved on `m`, unchanged since), in instruction order: removed,
/// then refuted.
pub fn findings(m: &Module, p: &Proved) -> Vec<Finding> {
    let w = &p.witness;
    let Some(f) = m.func(w.func) else { return Vec::new() };
    let Some(b) = &f.body else { return Vec::new() };
    let site = |i: u32| b.insts.site.get(i as usize).copied().unwrap_or(Site::NONE);
    let make = |inst: u32, clause: Clause, refuted: bool, spans: &[Span]| {
        let ret = match (clause, b.args(inst).get(1).and_then(|&o| b.int_const(o))) {
            (Clause::Ensures, Some(s)) => u32::try_from(s).ok().map(Site),
            _ => None,
        };
        Finding {
            func: w.func,
            clause,
            refuted,
            site: site(inst),
            ret,
            facts: fact_sites(m, f, b, w, spans),
            traps: conjunct_traps(m, b, inst),
        }
    };
    let mut out = Vec::new();
    for (r, proof) in w.removed.iter().zip(&p.proofs) {
        if let Proof::Clause(c) = *proof {
            out.push(make(r.inst, c, false, &[r.edges, r.also]));
        }
    }
    for (r, &c) in w.refuted.iter().zip(&p.refuted) {
        out.push(make(r.inst, c, true, &[r.edges]));
    }
    out
}

/// The sites of the facts of the proofs `spans` of `w` (the witness of `f`, body `b`), in order,
/// each once.
fn fact_sites(m: &Module, f: &Func, b: &Body, w: &FnWitness, spans: &[Span]) -> Vec<Site> {
    let site = |i: u32| b.insts.site.get(i as usize).copied().unwrap_or(Site::NONE);
    let mut facts: Vec<Site> = Vec::new();
    for e in spans.iter().flat_map(|s| s.of(&w.edges)) {
        let s = match (contract_branch(&m.funcs, b, e.src), source_inst(b, e.src)) {
            // a branch past an earlier contract check: that check's conjunct
            (Some(call), _) | (None, Some(call)) => site(call),
            (None, None) => f.name_site,
        };
        if s != Site::NONE && !facts.contains(&s) {
            facts.push(s);
        }
    }
    facts
}

/// A call the pass refuted, for diagnostics: read off a [`Proved`] and the module it was proved
/// on, before any removal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallFinding {
    /// The caller.
    pub func: FuncId,
    pub callee: FuncId,
    /// The site of the call.
    pub site: Site,
    /// The site of the callee's `requires` conjunct that the call fails (a site of the callee).
    pub clause: Site,
    /// The sites of the facts of the proof, in order, each once.
    pub facts: Vec<Site>,
}

/// The refuted calls of `p` (proved on `m`, unchanged since), in instruction order.
pub fn call_findings(m: &Module, p: &Proved) -> Vec<CallFinding> {
    let w = &p.witness;
    let Some(f) = m.func(w.func) else { return Vec::new() };
    let Some(b) = &f.body else { return Vec::new() };
    let mut out = Vec::new();
    for r in &w.calls.refuted {
        let Some(&Operand::Func(callee)) = b.args(r.inst).first() else { continue };
        let clause = claim_func(m, callee)
            .and_then(|s| {
                let entry = s.pre.get(r.row as usize)?.entry;
                s.entry.get(entry as usize).map(|e| e.inst)
            })
            .and_then(|q| m.func(callee)?.body.as_ref()?.insts.site.get(q as usize).copied())
            .unwrap_or(Site::NONE);
        out.push(CallFinding {
            func: w.func,
            callee,
            site: r.site,
            clause,
            facts: fact_sites(m, f, b, w, &[r.edges]),
        });
    }
    out
}

/// Whether the evaluation of the conjunct whose failure call is `call` has a trapping
/// instruction or a call: the blocks from where the conjunct starts (the true side of the
/// contract check before it, or the entry) up to the block branching on it.
fn conjunct_traps(m: &Module, b: &Body, call: u32) -> bool {
    let is_fail_block = |t: usize| {
        b.blocks.get(t).is_some_and(|bd| {
            bd.insts.len == 1
                && b.insts.op.get(bd.insts.start as usize) == Some(&Op::Call)
                && fail_clause(&m.funcs, b, bd.insts.start).is_some()
        })
    };
    let Some(fb) = b.blocks.iter().position(|bd| bd.insts.indices().contains(&call)) else { return true };
    let Some(p) =
        b.blocks.iter().position(|bd| matches!(bd.term, Term::Cbr { els, .. } if els.block.index() == fb))
    else {
        return true;
    };
    let dom = Dominators::compute(b);
    // the start: the nearest dominator entered by the true edge of an earlier contract check
    let mut start = 0;
    let mut walk = Some(p);
    while let Some(t) = walk {
        let entered = b.blocks.iter().any(|bd| match bd.term {
            Term::Cbr { then, els, .. } => then.block.index() == t && is_fail_block(els.block.index()),
            _ => false,
        });
        if entered {
            start = t;
            break;
        }
        walk = dom.idom(t).filter(|&d| d != t);
    }
    let mut seen = vec![false; b.blocks.len()];
    let mut work = vec![start];
    seen[start] = true;
    while let Some(t) = work.pop() {
        for i in b.blocks[t].insts.indices() {
            if b.insts.op.get(i as usize) == Some(&Op::Call) || crate::traps::trap_of(m, b, i).is_some() {
                return true;
            }
        }
        if t == p {
            continue;
        }
        for s in b.successors(Block(t as u32)) {
            let s = s.block.index();
            if s < seen.len() && !seen[s] && !is_fail_block(s) {
                seen[s] = true;
                work.push(s);
            }
        }
    }
    false
}

/// The failure call of the contract check whose true edge source `src` (a branch) reads.
fn contract_branch(funcs: &[Func], b: &Body, src: Source) -> Option<u32> {
    let Source::Branch { block } = src else { return None };
    b.blocks.iter().find_map(|bd| match bd.term {
        Term::Cbr { then, els, .. } if then.block.0 == block => {
            let fail = b.blocks.get(els.block.index())?;
            let q = fail.insts.start;
            (fail.insts.len == 1
                && b.insts.op.get(q as usize) == Some(&Op::Call)
                && fail_clause(funcs, b, q).is_some())
            .then_some(q)
        }
        _ => None,
    })
}

/// The instruction a witness source is read at, for reports: the compare (or `not`) a branch
/// tests, the defining instruction of a ranged value, the instruction itself; `None` for the
/// range of a block parameter.
pub fn source_inst(b: &Body, src: Source) -> Option<u32> {
    let def = |v: Value| match b.values.get(v.index())?.def {
        ValueDef::Inst(i) => Some(i),
        ValueDef::Param { .. } => None,
    };
    match src {
        Source::Branch { block } => b.blocks.iter().find_map(|bd| match bd.term {
            Term::Cbr { cond: Operand::Value(c), then, els }
                if then.block.0 == block || els.block.0 == block =>
            {
                def(c)
            }
            _ => None,
        }),
        Source::Assert { call: i, .. }
        | Source::Check { inst: i }
        | Source::Arith { inst: i }
        | Source::SliceLen { inst: i }
        | Source::SliceMake { inst: i }
        | Source::SubSat { inst: i }
        | Source::Bits { inst: i }
        | Source::Cast { inst: i }
        | Source::FieldLoad { inst: i }
        | Source::Getter { call: i }
        | Source::Ensures { call: i, .. }
        | Source::Arg { call: i, .. } => Some(i),
        Source::Range { value } => def(value),
        Source::Requires { .. } => None,
    }
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
    let mut cx = Cx::new(m, b, None);
    // After the removal the callees' own checks may be gone: read calls as the checker certified
    // them.
    if !m.stated.is_empty() {
        cx.claims = Some(&m.stated);
    }
    cx.audit(f, fid)
}

/// Only these ops can be element accesses: without one there is nothing to audit.
fn has_access(b: &Body) -> bool {
    b.insts.op.iter().any(|op| match op {
        Op::ElemAddr | Op::Get | Op::Set | Op::SlicePtr => true,
        Op::Vec(v) => v.is_memory(),
        _ => false,
    })
}

/// The witnesses of the `check.bounds` of function `fid` that have a proof (`None`: no check
/// has one). Nothing is removed. Functions are independent: a caller may run them in parallel.
pub fn prove_func(m: &Module, fid: FuncId) -> Option<Proved> {
    prove_func_in(m, fid, None)
}

/// [`prove_func`] reading the callees' summaries from `claims` (`None`: each computed here).
pub fn prove_func_in(m: &Module, fid: FuncId, claims: Option<&Claims>) -> Option<Proved> {
    let b = m.func(fid)?.body.as_ref()?;
    if !has_check(m, b, claims) {
        return None;
    }
    Cx::new(m, b, claims).prove(fid)
}

/// Whether `b` has a `check.bounds`, a call of a contract failure fn, or a call of a fn that
/// states a summary: without one there is nothing to prove.
fn has_check(m: &Module, b: &Body, claims: Option<&Claims>) -> bool {
    (0..b.insts.len() as u32).any(|i| match b.insts.op[i as usize] {
        Op::BoundsCheck => true,
        Op::Call => {
            fail_clause(&m.funcs, b, i).is_some()
                || match (b.args(i).first(), claims) {
                    (Some(&Operand::Func(f)), Some(c)) => c.0.get(f.index()).is_some_and(|s| s.is_some()),
                    // No table: the callee's body tells.
                    (Some(&Operand::Func(f)), None) => {
                        m.func(f).and_then(|f| f.body.as_ref()).is_some_and(|cb| has_contract(m, cb))
                    }
                    _ => false,
                }
        }
        _ => false,
    })
}

/// The kind of contract check whose failure call instruction `i` is, by its callee.
fn fail_clause(funcs: &[Func], b: &Body, i: u32) -> Option<Clause> {
    let Some(&Operand::Func(f)) = b.args(i).first() else { return None };
    match funcs.get(f.index())?.known? {
        crate::Known::RequiresFailed => Some(Clause::Requires),
        crate::Known::EnsuresFailed => Some(Clause::Ensures),
        _ => None,
    }
}

/// [`prove_func`] and the unguarded accesses of [`audit_func`] from one analysis of the
/// function. The audit sees the body with every check in place: an access behind a check that
/// has a witness stays guarded by that check, whose removal the witness checker answers for.
pub fn analyze_func(m: &Module, fid: FuncId) -> Analysis {
    analyze_func_in(m, fid, None)
}

/// [`analyze_func`] reading the callees' summaries from `claims` (`None`: each computed here).
pub fn analyze_func_in(m: &Module, fid: FuncId, claims: Option<&Claims>) -> Analysis {
    let Some(f) = m.func(fid) else { return Analysis::default() };
    let Some(b) = &f.body else { return Analysis::default() };
    let (checks, accesses) = (has_check(m, b, claims), has_access(b));
    if !checks && !accesses {
        return Analysis::default();
    }
    let cx = Cx::new(m, b, claims);
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
    let stated = claims(m);
    let proved: Vec<Proved> =
        (0..m.funcs.len() as u32).filter_map(|f| prove_func_in(m, FuncId(f), Some(&stated))).collect();
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

/// The summaries the functions of a module state ([`crate::witness::Summary`], rows only: the
/// spans of their proofs mean nothing here), by function; `None`: nothing stated. A summary is
/// read off the function's own body alone, so every function that calls it reads the same rows
/// its own witness states.
#[derive(Clone, Debug, Default)]
pub struct Claims(Vec<Option<Arc<Summary>>>);

/// The [`Claims`] of `m`. Functions are independent: [`claim_func`] may run in parallel.
pub fn claims(m: &Module) -> Claims {
    Claims((0..m.funcs.len() as u32).map(|f| claim_func(m, FuncId(f))).collect())
}

impl Claims {
    pub fn from_rows(rows: Vec<Option<Arc<Summary>>>) -> Claims {
        Claims(rows)
    }
}

/// The summary function `fid` states, if it states anything.
pub fn claim_func(m: &Module, fid: FuncId) -> Option<Arc<Summary>> {
    let b = m.func(fid)?.body.as_ref()?;
    if !has_contract(m, b) {
        return None;
    }
    let s = Cx::new(m, b, None).summarize(&mut Vec::new());
    states(&s).then(|| Arc::new(s))
}

/// Whether summary `s` states anything.
fn states(s: &Summary) -> bool {
    !(s.pre.is_empty() && s.entry.is_empty() && s.safe.is_empty() && s.post.is_empty())
}

/// Whether `b` calls a contract failure fn.
fn has_contract(m: &Module, b: &Body) -> bool {
    (0..b.insts.len() as u32)
        .any(|i| b.insts.op[i as usize] == Op::Call && fail_clause(&m.funcs, b, i).is_some())
}

/// Per-function analysis context.
struct Cx<'a> {
    m: &'a Module,
    /// The summaries of the module's functions, by function, when the caller has them.
    claims: Option<&'a [Option<Arc<Summary>>]>,
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
    /// The reachable contract checks, in order.
    contracts: Vec<ContractCheck>,
    facts: facts::Facts,
    /// The `slice.ptr` instructions with a use [`Cx::elem_ptr_use`] refuses, ascending.
    elem_ptr_misuse: std::cell::OnceCell<Vec<u32>>,
    /// Per value: [`Cx::unsafe_root`]'s answer, filled as values are asked.
    unsafe_roots: std::cell::RefCell<Vec<u8>>,
    /// Per value: its uses ([`Cx::sole_use`]), computed once per body.
    uses: std::cell::OnceCell<Vec<Uses>>,
}

/// How a value is used: a terminator or branch-argument use counts as more than one.
#[derive(Clone, Copy)]
enum Uses {
    None,
    One(u32, u32),
    Many,
}

/// A contract check as the pass reads it.
#[derive(Clone, Copy)]
struct ContractCheck {
    /// The failure call, alone in its block.
    call: u32,
    /// The block ending in `cbr cond ok fail`.
    branch: u32,
    cond: Operand,
    clause: Clause,
}

/// What a contract check's goals come to.
enum Verdict {
    /// Every goal has a chain; the second goal's (of an `==`) starts at this edge.
    Proved { also: Option<usize> },
    /// Goal `half` is false whenever the check is reached.
    Refuted { half: u8 },
}

/// One goal `lo < hi` (`strict`) or `lo <= hi`, the operands as written.
#[derive(Clone, Copy)]
struct Goal {
    lo: Operand,
    hi: Operand,
    strict: bool,
}

impl Goal {
    /// The goal that holds exactly when this one does not.
    fn negated(self) -> Goal {
        Goal { lo: self.hi, hi: self.lo, strict: !self.strict }
    }
}

/// The reachable contract checks of `b`: a block holding only a call of a contract failure fn
/// and ending in `unreachable`, entered only by the false edge of a `cbr` whose true edge goes
/// elsewhere.
fn contract_checks(
    funcs: &[Func],
    b: &Body,
    dom: &Dominators,
    preds: &[Vec<(u32, bool)>],
) -> Vec<ContractCheck> {
    let mut out = Vec::new();
    for (fb, bd) in b.blocks.iter().enumerate().skip(1) {
        let q = bd.insts.start;
        if !dom.reachable(fb)
            || bd.insts.len != 1
            || !matches!(bd.term, Term::Unreachable)
            || b.insts.op.get(q as usize) != Some(&Op::Call)
        {
            continue;
        }
        let Some(clause) = fail_clause(funcs, b, q) else { continue };
        let [(p, false)] = preds[fb][..] else { continue };
        if let Term::Cbr { cond, then, els } = b.blocks[p as usize].term
            && els.block.index() == fb
            && then.block.index() != fb
        {
            out.push(ContractCheck { call: q, branch: p, cond, clause });
        }
    }
    out.sort_unstable_by_key(|c| c.call);
    out
}

impl<'a> Cx<'a> {
    fn new(m: &'a Module, b: &'a Body, claims: Option<&'a Claims>) -> Cx<'a> {
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
        let contracts = contract_checks(funcs, b, &dom, &preds);
        Cx {
            m,
            claims: claims.map(|c| &c.0[..]),
            types,
            funcs,
            b,
            dom,
            block_of,
            preds,
            checks,
            contracts,
            facts: facts::Facts::default(),
            elem_ptr_misuse: std::cell::OnceCell::new(),
            unsafe_roots: std::cell::RefCell::new(Vec::new()),
            uses: std::cell::OnceCell::new(),
        }
    }

    /// The summary function `f` states: from the module's table, or computed here once.
    fn claim(&self, f: FuncId) -> Option<Arc<Summary>> {
        if let Some(c) = self.claims {
            return c.get(f.index()).cloned().flatten();
        }
        if let Some(s) = self.facts.claims.borrow().get(&f) {
            return s.clone();
        }
        let s = claim_func(self.m, f);
        self.facts.claims.borrow_mut().insert(f, s.clone());
        s
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
    /// a reference or raw pointer to an array or slice.
    fn len_of_ptr(&self, p: Operand) -> Result<Option<u64>, ()> {
        let pt = self.ty(p).ok_or(())?;
        crate::simd::mem_operand(self.types, pt).map(|mo| mo.len).ok_or(())
    }

    /// What the pointer operand `p` of an element access points into, decided in this order:
    /// - `p = slice.ptr s`, the element pointer of `s` ([`Cx::elem_ptr_of`]): the slice `s`, whose
    ///   elements it counts. When the elements of `s` are arrays or slices, an index through `p`
    ///   would count the inner elements of `s[0]`, which need not exist: nothing to check against.
    /// - `p` a reference or raw pointer to an array or slice: that object, by its type.
    /// - `p` a raw pointer to one element whose definition, through pointer-to-pointer casts, is
    ///   a `load`, `call`, or `call.ind`: pointer arithmetic of `unsafe` code, no access.
    /// - any other pointer (one derived from an element access, a parameter, a block parameter,
    ///   ...): nothing to check against.
    fn elem_base(&self, p: Operand) -> ElemBase {
        if self.def_op(p) == Some(Op::SlicePtr) {
            return match self.elem_ptr_of(p) {
                Some(s) if !self.has_aggregate_elems(s) => ElemBase::Object(s, None),
                _ => ElemBase::Unknown,
            };
        }
        if let Ok(len) = self.len_of_ptr(p) {
            return ElemBase::Object(p, len);
        }
        let one = self.ty(p).is_some_and(|t| crate::simd::elem_ptr_operand(self.types, t).is_some());
        if one && self.unsafe_root(p) { ElemBase::Raw } else { ElemBase::Unknown }
    }

    /// Whether pointer `p` is, through pointer-to-pointer casts, the result of a `load`, `call`, or
    /// `call.ind`: a pointer the function did not derive from an object it can see.
    fn unsafe_root(&self, p: Operand) -> bool {
        const UNKNOWN: u8 = 0;
        const WALKING: u8 = 1;
        const YES: u8 = 2;
        const NO: u8 = 3;
        let Operand::Value(mut v) = p else { return false };
        let mut memo = self.unsafe_roots.borrow_mut();
        if memo.is_empty() {
            memo.resize(self.b.values.len(), UNKNOWN);
        }
        // Each value's answer is decided once: a cast chain is walked once, iteratively, and
        // every value on it gets the answer of its end.
        let mut chain = Vec::new();
        let answer = loop {
            match memo.get(v.index()).copied() {
                Some(UNKNOWN) => {}
                Some(YES) => break true,
                _ => break false,
            }
            memo[v.index()] = WALKING;
            chain.push(v);
            let Some(d) = self.def(v) else { break false };
            match self.b.insts.op[d as usize] {
                Op::Load | Op::Call | Op::CallInd => break true,
                Op::Cast(_) => {
                    let from = self.b.args(d).first().copied().filter(|&a| {
                        self.ty(a)
                            .and_then(|t| self.key(t))
                            .is_some_and(|k| matches!(k, TypeKey::Ref { .. } | TypeKey::RawPtr { .. }))
                    });
                    match from {
                        Some(Operand::Value(a)) => v = a,
                        _ => break false,
                    }
                }
                _ => break false,
            }
        };
        let mark = if answer { YES } else { NO };
        for c in chain {
            memo[c.index()] = mark;
        }
        answer
    }

    /// The instruction defining operand `o`.
    fn def_inst(&self, o: Operand) -> Option<u32> {
        match o {
            Operand::Value(v) => self.def(v),
            _ => None,
        }
    }

    /// The op of the instruction defining operand `o`.
    fn def_op(&self, o: Operand) -> Option<Op> {
        self.def_inst(o).and_then(|d| self.b.insts.op.get(d as usize).copied())
    }

    /// Whether the elements of the slice `s` points to are arrays or slices.
    fn has_aggregate_elems(&self, s: Operand) -> bool {
        let elem = match self.ty(s).and_then(|t| self.key(t)) {
            Some(TypeKey::Ref { inner, .. } | TypeKey::RawPtr { inner, .. }) => match self.key(*inner) {
                Some(TypeKey::Slice { elem, .. }) => *elem,
                _ => return false,
            },
            _ => return false,
        };
        matches!(self.key(elem), Some(TypeKey::Array { .. } | TypeKey::Slice { .. }))
    }

    /// The slice `s` whose element pointer `p` is: `p = slice.ptr s`, typed as a pointer to the
    /// element type of `s`.
    fn elem_ptr_of(&self, p: Operand) -> Option<Operand> {
        let Operand::Value(v) = p else { return None };
        let d = self.def(v)?;
        if self.b.insts.op.get(d as usize) != Some(&Op::SlicePtr) {
            return None;
        }
        let s = *self.b.args(d).first()?;
        let pointee = match self.key(self.b.insts.ty.get(d as usize).copied().flatten()?)? {
            TypeKey::Ref { inner, .. } | TypeKey::RawPtr { inner, .. } => *inner,
            _ => return None,
        };
        let slice = match self.key(self.ty(s)?)? {
            TypeKey::Ref { inner, .. } | TypeKey::RawPtr { inner, .. } => *inner,
            _ => return None,
        };
        let elem_ok = match self.key(slice)? {
            TypeKey::Slice { elem, .. } => *elem == pointee,
            TypeKey::Str => self.key(pointee) == Some(&TypeKey::Int { signed: false, bits: IntBits::B8 }),
            _ => false,
        };
        elem_ok.then_some(s)
    }

    /// Whether operand `pos` of instruction `user` is a use of the element pointer of slice `s`
    /// that the audit accepts: the pointer operand of an element access (`addr.elem`, `vec.load`,
    /// `vec.store`, each audited as an access of `s`), of a prefetch, of a cast to an int, or of
    /// `slice.make` with `slice.len s` that keeps the slice's type as [`Cx::rebuild_type_ok`]
    /// says. Every other use is refused.
    fn elem_ptr_use(&self, user: u32, pos: usize, s: Operand) -> bool {
        let Some(&op) = self.b.insts.op.get(user as usize) else { return false };
        match (op, pos) {
            (Op::ElemAddr | Op::Vec(VecOp::Load | VecOp::Store) | Op::Prefetch(_), 0) => true,
            (Op::Cast(_), 0) => {
                let t = self.b.insts.ty.get(user as usize).copied().flatten();
                matches!(t.and_then(|t| self.key(t)), Some(TypeKey::Int { .. }))
            }
            (Op::SliceMake, 0) => {
                self.b.args(user).get(1).is_some_and(|&n| self.is_len_of(n, s, None))
                    && self.rebuild_type_ok(user, s)
            }
            _ => false,
        }
    }

    /// Whether `slice.make` instruction `m`, rebuilding slice `s` from its element pointer and
    /// length, keeps the type of `s` up to what lowering changes: the same element type (`str`
    /// stays `str`), mutability kept or dropped, alignment kept or dropped, or raised only in the
    /// form of `align_to` ([`Cx::align_to_form`]).
    fn rebuild_type_ok(&self, m: u32, s: Operand) -> bool {
        let parts = |t: TypeId| match self.key(t)? {
            TypeKey::Ref { mutable, inner } | TypeKey::RawPtr { mutable, inner } => {
                Some((*mutable, self.key(*inner)?))
            }
            _ => None,
        };
        let from = self.ty(s).and_then(parts);
        let to = self.b.insts.ty.get(m as usize).copied().flatten().and_then(parts);
        let (Some((from_mut, from)), Some((to_mut, to))) = (from, to) else { return false };
        if to_mut && !from_mut {
            return false;
        }
        match (from, to) {
            (TypeKey::Str, TypeKey::Str) => true,
            (TypeKey::Slice { elem: a, align: x }, TypeKey::Slice { elem: b, align: y }) => {
                a == b
                    && match y {
                        None => true,
                        Some(y) => x == &Some(*y) || self.align_to_form(m, s, *y),
                    }
            }
            _ => false,
        }
    }

    /// Whether the slice `slice.make` instruction `m` makes is only the `Some` of `align_to`:
    /// its one use is an `enum.make`, whose one use is the second operand of `select c yes no`
    /// with `c = eq (rem (cast (slice.ptr s)) align) 0`.
    fn align_to_form(&self, m: u32, s: Operand, align: u32) -> bool {
        let sole_use_of =
            |i: u32| self.b.insts.result.get(i as usize).copied().flatten().and_then(|r| self.sole_use(r));
        let Some((e, _)) = sole_use_of(m) else { return false };
        if self.b.insts.op[e as usize] != Op::EnumMake {
            return false;
        }
        let Some((sel, 1)) = sole_use_of(e) else { return false };
        if self.b.insts.op[sel as usize] != Op::Select {
            return false;
        }
        let arg = |i: u32, k: usize| self.b.args(i).get(k).copied();
        let Some(eq) = arg(sel, 0).and_then(|c| self.def_inst(c)) else { return false };
        let Some(rem) = arg(eq, 0).and_then(|x| self.def_inst(x)) else { return false };
        let Some(addr) = arg(rem, 0).and_then(|x| self.def_inst(x)) else { return false };
        self.b.insts.op[eq as usize] == Op::Eq
            && arg(eq, 1).and_then(|z| self.konst(z)) == Some(0)
            && self.b.insts.op[rem as usize] == Op::Rem
            && arg(rem, 1).and_then(|a| self.konst(a)) == Some(u64::from(align))
            && matches!(self.b.insts.op[addr as usize], Op::Cast(_))
            && arg(addr, 0).and_then(|p| self.elem_ptr_of(p)) == Some(s)
    }

    /// The one use of value `v` (instruction, operand position); `None` when it has none, more
    /// than one, or a terminator or branch-argument use.
    fn sole_use(&self, v: Value) -> Option<(u32, usize)> {
        let uses = self.uses.get_or_init(|| {
            let mut uses = vec![Uses::None; self.b.values.len()];
            let mut add = |o: Operand, at: Option<(u32, u32)>| {
                if let Operand::Value(x) = o
                    && let Some(u) = uses.get_mut(x.index())
                {
                    *u = match (*u, at) {
                        (Uses::None, Some((i, pos))) => Uses::One(i, pos),
                        _ => Uses::Many,
                    };
                }
            };
            for k in 0..self.b.insts.len() as u32 {
                for (pos, &a) in self.b.args(k).iter().enumerate() {
                    add(a, Some((k, pos as u32)));
                }
            }
            for (bi, bd) in self.b.blocks.iter().enumerate() {
                match bd.term {
                    Term::Cbr { cond: o, .. } | Term::Switch { scrut: o, .. } | Term::Ret(Some(o)) => {
                        add(o, None)
                    }
                    // no operand besides the target arguments, read below
                    Term::Br(_) | Term::Ret(None) | Term::Unreachable => {}
                }
                for t in self.b.successors(Block(bi as u32)) {
                    for &a in self.b.target_args(&t) {
                        add(a, None);
                    }
                }
            }
            uses
        });
        match uses.get(v.index())? {
            Uses::One(i, pos) => Some((*i, *pos as usize)),
            Uses::None | Uses::Many => None,
        }
    }

    /// The `slice.ptr` instructions with a use that [`Cx::elem_ptr_use`] refuses (a terminator or
    /// branch-argument use is refused too), ascending; computed once per body.
    fn elem_ptr_misuse(&self) -> &[u32] {
        self.elem_ptr_misuse.get_or_init(|| {
            let mut bad = Vec::new();
            let def_of = |o: Operand| match o {
                Operand::Value(v) => self.def(v).filter(|&d| self.b.insts.op[d as usize] == Op::SlicePtr),
                _ => None,
            };
            for k in 0..self.b.insts.len() as u32 {
                for (pos, &a) in self.b.args(k).iter().enumerate() {
                    if let Some(d) = def_of(a)
                        && !self.b.args(d).first().is_some_and(|&s| self.elem_ptr_use(k, pos, s))
                    {
                        bad.push(d);
                    }
                }
            }
            for (bi, bd) in self.b.blocks.iter().enumerate() {
                match bd.term {
                    Term::Cbr { cond: o, .. } | Term::Switch { scrut: o, .. } | Term::Ret(Some(o)) => {
                        bad.extend(def_of(o));
                    }
                    // no operand besides the target arguments, read below
                    Term::Br(_) | Term::Ret(None) | Term::Unreachable => {}
                }
                for t in self.b.successors(Block(bi as u32)) {
                    bad.extend(self.b.target_args(&t).iter().filter_map(|&o| def_of(o)));
                }
            }
            bad.sort_unstable();
            bad.dedup();
            bad
        })
    }

    fn access_kind(&self, i: u32) -> Option<AccessKind> {
        let op = *self.b.insts.op.get(i as usize)?;
        let args = self.b.args(i);
        match op {
            Op::ElemAddr => {
                (!matches!(self.elem_base(*args.first()?), ElemBase::Raw)).then_some(AccessKind::Elem)
            }
            Op::SlicePtr => Some(AccessKind::ElemPtr),
            Op::Get | Op::Set => {
                let at = self.ty(*args.first()?)?;
                matches!(self.key(at), Some(TypeKey::Array { .. })).then_some(if op == Op::Get {
                    AccessKind::Get
                } else {
                    AccessKind::Set
                })
            }
            Op::Vec(v) if v.is_memory() => Some(AccessKind::Vector),
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
                let idx = *args.get(1)?;
                let ElemBase::Object(p, len) = self.elem_base(*args.first()?) else { return None };
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
            AccessKind::ElemPtr => {
                let to_elem = self
                    .b
                    .insts
                    .result
                    .get(i as usize)
                    .copied()
                    .flatten()
                    .is_some_and(|r| self.elem_ptr_of(Operand::Value(r)).is_some());
                (to_elem && self.elem_ptr_misuse().binary_search(&i).is_err()).then_some(Guard::Uses)
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
                let idx = *args.get(1)?;
                let ElemBase::Object(p, len) = self.elem_base(*args.first()?) else { return None };
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

    /// The witnesses of every check that has a proof, and of every contract check that has a
    /// refutation.
    fn prove(&self, func: FuncId) -> Option<Proved> {
        let mut w = FnWitness {
            func,
            dom: Vec::new(),
            removed: Vec::new(),
            refuted: Vec::new(),
            edges: Vec::new(),
            summary: Summary::default(),
            calls: Calls::default(),
        };
        let mut rows: Vec<(Removed, Proof)> = Vec::new();
        let span = |from: usize, to: usize| Span { start: from as u32, len: (to - from) as u32 };
        if !self.checks.is_empty() {
            let twins = self.twins();
            let heads = self.loop_heads();
            for (k, &i) in self.checks.iter().enumerate() {
                let start = w.edges.len();
                match self.check_proof(i, twins[k], heads[k], &mut w.edges) {
                    Some(proof) => {
                        let edges = span(start, w.edges.len());
                        let site = self.b.insts.site[i as usize];
                        rows.push((Removed { inst: i, site, edges, also: Span::default() }, proof));
                    }
                    None => w.edges.truncate(start),
                }
            }
        }
        let mut refuted = Vec::new();
        for c in &self.contracts {
            let start = w.edges.len();
            let site = self.b.insts.site[c.call as usize];
            match self.contract_proof(c, &mut w.edges) {
                Some(Verdict::Proved { also }) => {
                    let (edges, also) = match also {
                        Some(mid) => (span(start, mid), span(mid, w.edges.len())),
                        None => (span(start, w.edges.len()), Span::default()),
                    };
                    rows.push((Removed { inst: c.call, site, edges, also }, Proof::Clause(c.clause)));
                }
                Some(Verdict::Refuted { half }) => {
                    w.refuted.push(Refuted { inst: c.call, site, edges: span(start, w.edges.len()), half });
                    refuted.push(c.clause);
                }
                None => w.edges.truncate(start),
            }
        }
        if !self.contracts.is_empty() {
            w.summary = self.summarize(&mut w.edges);
        }
        self.decide_calls(&mut w);
        if rows.is_empty()
            && refuted.is_empty()
            && !states(&w.summary)
            && w.calls.proved.is_empty()
            && w.calls.refuted.is_empty()
        {
            return None;
        }
        rows.sort_by_key(|r| r.0.inst);
        let proofs = rows.iter().map(|r| r.1).collect();
        w.removed = rows.into_iter().map(|r| r.0).collect();
        w.dom = (0..self.b.blocks.len()).map(|bk| self.dom.idom(bk).map_or(NO_BLOCK, |d| d as u32)).collect();
        Some(Proved { witness: w, proofs, refuted })
    }

    /// The calls of functions with an unchecked entry: proved when the facts right before the
    /// call prove every `pre` row of the callee about its arguments; refuted when they prove the
    /// negation of a row that reads a `requires` check and the call is at no point shown dead
    /// (as a contract check); else left alone.
    fn decide_calls(&self, w: &mut FnWitness) {
        let span = |from: usize, to: usize| Span { start: from as u32, len: (to - from) as u32 };
        for i in 0..self.b.insts.len() as u32 {
            if self.b.insts.op[i as usize] != Op::Call || !self.reachable_inst(i) {
                continue;
            }
            let Some(&Operand::Func(f)) = self.b.args(i).first() else { continue };
            let Some(s) = self.claim(f).filter(|s| !s.entry.is_empty()) else { continue };
            let (block, site) = (self.block_of[i as usize], self.b.insts.site[i as usize]);
            let (start, first) = (w.edges.len(), w.calls.goals.len());
            let proved = s.pre.iter().all(|p| {
                let from = w.edges.len();
                let done = self
                    .call_sides(i, &p.row)
                    .and_then(|(a, b)| self.side_goal(block, i, (a, b, p.row.w), &mut w.edges));
                w.calls.goals.push(span(from, w.edges.len()));
                done.is_some()
            });
            if proved {
                w.calls.proved.push(CallProved { inst: i, site, rows: span(first, w.calls.goals.len()) });
                continue;
            }
            w.edges.truncate(start);
            w.calls.goals.truncate(first);
            if self.contradiction(block, i).is_some() || self.dead_end(block, i) {
                continue;
            }
            for (k, p) in s.pre.iter().enumerate() {
                if p.entry == NO_BLOCK {
                    continue;
                }
                // not `a <= b + w`: `b <= a - w - 1`
                let negated = self.call_sides(i, &p.row).and_then(|(a, b)| {
                    let w_neg = p.row.w.checked_neg()?.checked_sub(1)?;
                    self.side_goal(block, i, (b, a, w_neg), &mut w.edges)
                });
                if negated.is_some() {
                    let edges = span(start, w.edges.len());
                    w.calls.refuted.push(CallRefuted { inst: i, site, row: k as u32, edges });
                    break;
                }
                w.edges.truncate(start);
            }
        }
    }

    /// The verdict on contract check `c`, with the edges of its proof or refutation appended to
    /// `out` (nothing when unknown).
    fn contract_proof(&self, c: &ContractCheck, out: &mut Vec<Edge>) -> Option<Verdict> {
        let goals = self.goals_of(c.cond)?;
        let start = out.len();
        let prove_all = |cycle: Option<&(u32, Vec<u32>)>, out: &mut Vec<Edge>| {
            let mut second = None;
            for (k, &goal) in goals.iter().enumerate() {
                if k == 1 {
                    second = Some(out.len());
                }
                let mark = out.len();
                if self.goal_chain(c.branch, goal, out).is_some() {
                    continue;
                }
                out.truncate(mark);
                cycle.and_then(|cy| self.goal_by_cycle(c.branch, goal, cy, out))?;
            }
            Some(Verdict::Proved { also: second })
        };
        if let Some(v) = prove_all(None, out) {
            return Some(v);
        }
        out.truncate(start);
        // A point no execution reaches holds the goal and its negation alike: there a proof may
        // need the contradiction several times (multipliers), and no refutation is reported.
        if let Some(cycle) = self.contradiction(c.branch, u32::MAX) {
            let v = prove_all(Some(&cycle), out);
            if v.is_none() {
                out.truncate(start);
            }
            return v;
        }
        if self.dead_end(c.branch, u32::MAX) {
            return None;
        }
        for (half, &goal) in goals.iter().enumerate() {
            if self.goal_chain(c.branch, goal.negated(), out).is_some() {
                return Some(Verdict::Refuted { half: half as u8 });
            }
            out.truncate(start);
        }
        None
    }

    /// The goals of the bool `cond` being true: the compare it tests under its `not`s, as one
    /// inequality, or two for an `==`; `None` for `!=` and for anything that is not a compare.
    fn goals_of(&self, cond: Operand) -> Option<Vec<Goal>> {
        // the compare under the `not`s, as it reads when the condition is true
        let (mut o, mut holds) = (cond, true);
        let (op, x, y) = 'cmp: {
            for _ in 0..=NOT_DEPTH {
                let Operand::Value(v) = o else { return None };
                let d = self.def(v)?;
                match (self.b.insts.op[d as usize], self.b.args(d)) {
                    (Op::Not, &[a]) => {
                        o = a;
                        holds = !holds;
                    }
                    (op, &[x, y]) => break 'cmp (op, x, y),
                    _ => return None,
                }
            }
            return None;
        };
        let op = if holds {
            op
        } else {
            match op {
                Op::Lt => Op::Ge,
                Op::Le => Op::Gt,
                Op::Gt => Op::Le,
                Op::Ge => Op::Lt,
                Op::Eq => Op::Ne,
                Op::Ne => Op::Eq,
                _ => return None,
            }
        };
        let g = |lo, hi, strict| Goal { lo, hi, strict };
        let goals = match op {
            Op::Lt => vec![g(x, y, true)],
            Op::Le => vec![g(x, y, false)],
            Op::Gt => vec![g(y, x, true)],
            Op::Ge => vec![g(y, x, false)],
            Op::Eq => vec![g(x, y, false), g(y, x, false)],
            _ => return None,
        };
        Some(goals)
    }

    /// The chain of `goal` at the end of block `block`, appended to `out`, if the facts prove it.
    fn goal_chain(&self, block: u32, goal: Goal, out: &mut Vec<Edge>) -> Option<()> {
        if self.range_goal(goal, out).is_some() {
            return Some(());
        }
        let (lo, hi) = (self.term(goal.lo)?, self.term(goal.hi)?);
        let hi_bound = if goal.strict { hi } else { hi.plus(1)? };
        let mut path = Vec::new();
        self.prove_at(block, u32::MAX, lo, hi_bound, Some(&mut path))?;
        let from = out.len();
        if self.chain(goal.lo, &path, goal.hi, out).is_none() {
            out.truncate(from);
            return None;
        }
        Some(())
    }

    /// `goal` between a constant and an int value that the value's type range alone decides
    /// (`x >= 0` on an unsigned `x`, `b <= 255` on a `u8`): one `Range` edge.
    fn range_goal(&self, goal: Goal, out: &mut Vec<Edge>) -> Option<()> {
        let strict = i128::from(goal.strict);
        let range = |v: Value| -> Option<(i128, i128)> {
            match self.key(self.b.value_ty(v)?)? {
                TypeKey::Int { signed: true, bits } => {
                    let h = 1i128 << (bits.width() - 1);
                    Some((-h, h - 1))
                }
                TypeKey::Int { signed: false, bits } => Some((0, (1i128 << bits.width()) - 1)),
                _ => None,
            }
        };
        let konst = |o: Operand| self.b.int_const(o).filter(|c| c.abs() <= crate::witness::MAX_CONST);
        let edge = match (goal.lo, goal.hi) {
            // c <= v (+ strict): the type's least value is at least c + strict
            (lo, Operand::Value(v)) if konst(lo).is_some() => {
                let (min, _) = range(v)?;
                (konst(lo)? + strict <= min).then_some(Edge {
                    from: Node::Zero,
                    to: Node::Val(v),
                    w: -min,
                    src: Source::Range { value: v },
                    mul: 1,
                })?
            }
            // v + strict <= c: the type's greatest value is at most c - strict
            (Operand::Value(v), hi) if konst(hi).is_some() => {
                let (_, max) = range(v)?;
                (max + strict <= konst(hi)?).then_some(Edge {
                    from: Node::Val(v),
                    to: Node::Zero,
                    w: max,
                    src: Source::Range { value: v },
                    mul: 1,
                })?
            }
            _ => return None,
        };
        out.push(edge);
        Some(())
    }

    /// The chain of `goal` at the end of `block` through the contradictory `cycle` (a node and
    /// facts leading back to it with negative weight), the cycle's edges counted as often as the
    /// goal needs: valid at a point no execution reaches.
    fn goal_by_cycle(
        &self,
        block: u32,
        goal: Goal,
        cycle: &(u32, Vec<u32>),
        out: &mut Vec<Edge>,
    ) -> Option<()> {
        let (lo, hi) = (self.term(goal.lo)?, self.term(goal.hi)?);
        let to_cycle = self.walk(block, lo, cycle.0)?;
        let from_cycle = self.walk(block, Self::node_term(cycle.0), Self::node_of(hi))?;
        let start = out.len();
        let built = (|| {
            self.down(goal.lo, out);
            for &f in &to_cycle {
                self.fact_edges(f, out)?;
            }
            let ring = out.len();
            for &f in &cycle.1 {
                self.fact_edges(f, out)?;
            }
            let after = out.len();
            for &f in &from_cycle {
                self.fact_edges(f, out)?;
            }
            self.up(goal.hi, out);
            let sum = |es: &[Edge]| es.iter().try_fold(0i128, |s, e| s.checked_add(e.w));
            let (path, around) =
                (sum(&out[start..ring])?.checked_add(sum(&out[after..])?)?, sum(&out[ring..after])?);
            let ((_, klo), (_, khi)) = (self.written(goal.lo)?, self.written(goal.hi)?);
            let want = khi - klo - i128::from(goal.strict);
            let short = path.checked_sub(want)?;
            if short <= 0 {
                out.drain(ring..after);
                return Some(());
            }
            if around >= 0 {
                return None;
            }
            let times = (short + (-around) - 1) / (-around);
            let times = u32::try_from(times).ok().filter(|&t| t <= crate::witness::MAX_MUL)?;
            for e in &mut out[ring..after] {
                e.mul = times;
            }
            Some(())
        })();
        if built.is_none() {
            out.truncate(start);
        }
        built
    }

    /// Whether the point right before instruction `before` of `block` (`u32::MAX`: its end) is
    /// dead in a way the facts do not show: a dominating branch on a constant condition entered
    /// on the side it never takes, or a call that runs before the point and cannot return (a
    /// callee with no `ret`, or an asserting callee given the constant `false` for the parameter
    /// it asserts).
    fn dead_end(&self, block: u32, before: u32) -> bool {
        let mut walk = Some(block as usize);
        while let Some(t) = walk {
            if let Some((cond, pol)) = facts::edge_cond(self.b, &self.preds, t)
                && self.constant_cond(cond).is_some_and(|v| v != pol)
            {
                return true;
            }
            let here = t == block as usize;
            for i in self.b.blocks[t].insts.indices() {
                if (!here || i < before)
                    && self.b.insts.op.get(i as usize) == Some(&Op::Call)
                    && self.never_returns(i)
                {
                    return true;
                }
            }
            walk = self.dom.idom(t).filter(|&p| p != t);
        }
        false
    }

    /// The value of `cond` when it is a bool constant or a compare of two int constants, under
    /// `not`s.
    fn constant_cond(&self, cond: Operand) -> Option<bool> {
        let (mut o, mut flip) = (cond, false);
        for _ in 0..=NOT_DEPTH {
            if let Some(c) = self.b.int_const(o) {
                return Some((c != 0) != flip);
            }
            let Operand::Value(v) = o else { return None };
            let d = self.def(v)?;
            let (x, y) = match (self.b.insts.op[d as usize], self.b.args(d)) {
                (Op::Not, &[a]) => {
                    o = a;
                    flip = !flip;
                    continue;
                }
                (_, &[x, y]) => (self.b.int_const(x)?, self.b.int_const(y)?),
                _ => return None,
            };
            let r = match self.b.insts.op[d as usize] {
                Op::Lt => x < y,
                Op::Le => x <= y,
                Op::Gt => x > y,
                Op::Ge => x >= y,
                Op::Eq => x == y,
                Op::Ne => x != y,
                _ => return None,
            };
            return Some(r != flip);
        }
        None
    }

    /// Whether call `i` cannot return: its callee's body has no `ret`, or the callee asserts a
    /// parameter that the call passes the constant `false`.
    fn never_returns(&self, i: u32) -> bool {
        let args = self.b.args(i);
        let Some(&Operand::Func(f)) = args.first() else { return false };
        let Some(body) = self.funcs.get(f.index()).and_then(|f| f.body.as_ref()) else { return false };
        if !body.blocks.iter().any(|bd| matches!(bd.term, Term::Ret(_))) {
            return true;
        }
        let mask = self.asserted(f);
        args[1..]
            .iter()
            .enumerate()
            .take(64)
            .any(|(k, &a)| mask & (1u64 << k) != 0 && self.b.int_const(a) == Some(0))
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
        out.push(Edge { from: Node::Val(va), to: len, w: 0, src: Source::SliceLen { inst: da }, mul: 1 });
        out.push(Edge { from: len, to: Node::Val(vb), w: 0, src: Source::SliceLen { inst: db }, mul: 1 });
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
        out.push(Edge {
            from,
            to: Node::Val(lim),
            w: c - k,
            src: Source::SubSat { inst: self.def(lim)? },
            mul: 1,
        });
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
        out.push(Edge { from, to, w: ky - kx - 1, src, mul: 1 });
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

    /// Every node kind round-trips through the numbering, and no two nodes share a number.
    #[test]
    fn node_numbering_round_trips() {
        let mut seen = std::collections::HashMap::new();
        let mut nodes = vec![Node::Zero];
        for v in [0u32, 1, 2, 3, 4, 5, 7, 4095, 4096, 1 << 20] {
            nodes.extend([Node::Val(Value(v)), Node::Len(Value(v))]);
            for k in [0u32, 1, 2, 4095] {
                nodes.push(Node::Field(Value(v), k));
            }
        }
        for n in nodes {
            let Some(id) = facts::node_id(n) else {
                assert!(matches!(n, Node::Field(p, _) if p.0 >= 4096), "{n:?}");
                continue;
            };
            assert_eq!(facts::node_of(id), n);
            assert_eq!(seen.insert(id, n), None, "{n:?} shares {id}");
        }
        assert_eq!(facts::node_id(Node::Val(Value(u32::MAX))), None);
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
