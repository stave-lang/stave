//! The checker of check-removal witnesses, and the only code that deletes a `check.bounds` or a
//! contract check.
//!
//! [`check_func`] validates every row of one function against the IR: it reads the row's check
//! (a `check.bounds`, or a contract check: its failure call, that call's block, and the branch
//! into it), derives the goals from the IR, looks each edge up at its source and compares it with
//! what that source states, and adds the edges up, each times its multiplier, against the goal.
//! Nothing is searched for, and nothing here comes from the pass that wrote the witness: this
//! file reads the IR tables and the witness tables only, with its own reading of the
//! control-flow graph, of dominance (the certificate of [`FnWitness::dom`]), of asserting
//! callees, of contract checks, and of int types. A rejection is a compiler bug. [`remove`] then
//! deletes exactly the checks that were accepted; a refuted check is validated the same way and
//! stays (the driver reports it).
//!
//! What an accepted witness rests on: the IR verifier (the module must have passed it: an
//! operand is defined on every path to its use, and ops have their operand types), the op
//! semantics of the checked IR, and this file. An inequality read at a source is about the SSA
//! values it names; it still holds at the check because the source runs before the check on
//! every path to it and every value it names is defined before the source, so no path from the
//! source to the check can define one of them again without passing the source again. A source
//! may be another removed check: in any execution, the first removed check that would fail has
//! only true facts before it, so its proof holds, a contradiction.
//!
//! Cost: linear in the function (blocks, CFG edges, instructions) plus the witness rows, each
//! goal's nodes sorted once. On IR that is not well-formed the checker refuses or accepts, but
//! does not panic.

use super::{
    ASSERT_BLOCKS, ENSURES_FAILED, Edge, FnWitness, MAX_CONST, MAX_MUL, NO_BLOCK, NOT_DEPTH, Node,
    REQUIRES_FAILED, Refuted, Removed, Source,
};
use crate::ir::{Block, Body, CastKind, FuncId, Module, Op, Operand, Span, Target, Term, Value, ValueDef};
use crate::types::{IntBits, TypeKey};

/// Why a witness does not hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Reject {
    /// The function has no body, or the dominance table does not fit its control flow.
    Cfg,
    /// A row does not name a reachable `check.bounds` or contract check, the rows are not in
    /// instruction order, or a refuted row names a check that is not a contract check or is
    /// also removed.
    NotACheck,
    /// An operand is not an int of the type the op reads it in.
    Operand,
    /// The check's condition states no inequality (`!=`), the row has a proof for a goal the
    /// check does not have, or a refuted row names a goal the check does not have.
    Goal,
    /// The source is not the kind of instruction or branch the edge says it is.
    Source,
    /// The source does not state the edge's inequality, or the multiplier is out of range.
    Edge,
    /// The source does not hold at the check: it does not dominate it.
    Region,
    /// The edges do not add up to the goal's two nodes.
    Chain,
    /// The weights do not prove the goal.
    Weak,
    /// A weight times its multiplier, or the sum, is beyond `i128`.
    Overflow,
}

impl Reject {
    fn text(self) -> &'static str {
        match self {
            Reject::Cfg => "the dominance table does not fit the function",
            Reject::NotACheck => "not a reachable check.bounds or contract check",
            Reject::Operand => "an operand is not an int of the op's type",
            Reject::Goal => "the check's condition has no such goal",
            Reject::Source => "the source is not what the edge names",
            Reject::Edge => "the source does not state this inequality",
            Reject::Region => "the source does not dominate the check",
            Reject::Chain => "the edges do not add up to the goal's two sides",
            Reject::Weak => "the weights do not prove the goal",
            Reject::Overflow => "the weights overflow",
        }
    }
}

/// A witness the checker refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rejected {
    pub func: FuncId,
    /// The function's checked-IR instance and path.
    pub path: String,
    /// The check (`None`: the function's tables as a whole).
    pub inst: Option<u32>,
    /// The edge of the check's proof, counted from 0 (a second goal's edges after the first's).
    pub edge: Option<u32>,
    pub why: Reject,
}

impl std::fmt::Display for Rejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.path)?;
        if let Some(i) = self.inst {
            write!(f, ": inst {i}")?;
        }
        if let Some(e) = self.edge {
            write!(f, ": edge {e}")?;
        }
        write!(f, ": {}", self.why.text())
    }
}

/// The checks of one function whose witnesses hold. Only [`check_func`] makes one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Checked {
    func: FuncId,
    insts: Vec<u32>,
    /// The blocks whose `cbr` into an accepted contract check's failure block becomes a `br`.
    jumps: Vec<u32>,
    refuted: Vec<u32>,
}

impl Checked {
    pub fn func(&self) -> FuncId {
        self.func
    }

    /// The accepted removals, by instruction index (a contract check: its failure call).
    pub fn insts(&self) -> &[u32] {
        &self.insts
    }

    /// The accepted refutations, by the instruction index of the failure call.
    pub fn refuted(&self) -> &[u32] {
        &self.refuted
    }
}

/// Validates every row of `w` against its function in `m`. Functions are independent: a caller
/// may check them in parallel.
pub fn check_func(m: &Module, w: &FnWitness) -> Result<Checked, Rejected> {
    let path = m.func(w.func).map_or(String::new(), |f| format!("@{} {}", f.instance, f.path));
    let fail = |inst: Option<u32>, edge: Option<u32>, why: Reject| Rejected {
        func: w.func,
        path: path.clone(),
        inst,
        edge,
        why,
    };
    let Some(b) = m.func(w.func).and_then(|f| f.body.as_ref()) else {
        return Err(fail(None, None, Reject::Cfg));
    };
    let mut cx = Cx::new(m, b, &w.dom).map_err(|why| fail(None, None, why))?;
    let mut prev = None;
    let mut jumps = Vec::new();
    for r in &w.removed {
        let jump = cx.row(w, r, prev).map_err(|(edge, why)| fail(Some(r.inst), edge, why))?;
        jumps.extend(jump);
        prev = Some(r.inst);
    }
    let mut prev = None;
    for r in &w.refuted {
        cx.refuted(w, r, prev).map_err(|(edge, why)| fail(Some(r.inst), edge, why))?;
        prev = Some(r.inst);
    }
    Ok(Checked {
        func: w.func,
        insts: w.removed.iter().map(|r| r.inst).collect(),
        jumps,
        refuted: w.refuted.iter().map(|r| r.inst).collect(),
    })
}

/// [`check_func`] for every function of `ws`: all accepted, or every rejection.
pub fn check_module(m: &Module, ws: &[FnWitness]) -> Result<Vec<Checked>, Vec<Rejected>> {
    let mut ok = Vec::with_capacity(ws.len());
    let mut bad = Vec::new();
    for w in ws {
        match check_func(m, w) {
            Ok(c) => ok.push(c),
            Err(r) => bad.push(r),
        }
    }
    if bad.is_empty() { Ok(ok) } else { Err(bad) }
}

/// Deletes the accepted checks: a `check.bounds` goes; a contract check's `cbr` becomes a `br` to
/// its true side, its failure call goes, and so does the failure block once nothing branches to
/// it. `m` must be the module they were checked against, unchanged; an index that is not a
/// `check.bounds` or a call is left alone.
pub fn remove(m: &mut Module, checked: &[Checked]) {
    for c in checked {
        if let Some(b) = m.func_mut(c.func).and_then(|f| f.body.as_mut()) {
            remove_checks(b, &c.insts, &c.jumps);
        }
    }
}

/// Removes the checks `dead` (instructions) and the branches `jumps` (blocks) from `b`, keeping
/// the order of the rest.
fn remove_checks(b: &mut Body, dead: &[u32], jumps: &[u32]) {
    let mut failed = Vec::with_capacity(jumps.len());
    for &p in jumps {
        if let Some(bd) = b.blocks.get_mut(p as usize)
            && let Term::Cbr { then, els, .. } = bd.term
        {
            bd.term = Term::Br(then);
            failed.push(els.block);
        }
    }
    let n = b.insts.len();
    let mut keep = vec![true; n];
    for &d in dead {
        if matches!(b.insts.op.get(d as usize), Some(Op::BoundsCheck | Op::Call)) {
            keep[d as usize] = false;
        }
    }
    // before[i]: how many of the first `i` instructions stay, which is where a kept instruction
    // `i` moves to.
    let mut before = Vec::with_capacity(n + 1);
    let mut kept = 0u32;
    for &k in &keep {
        before.push(kept);
        kept += u32::from(k);
    }
    before.push(kept);
    fn compact<T>(column: &mut Vec<T>, keep: &[bool]) {
        let mut i = 0;
        column.retain(|_| {
            i += 1;
            keep.get(i - 1).copied().unwrap_or(true)
        });
    }
    compact(&mut b.insts.op, &keep);
    compact(&mut b.insts.ty, &keep);
    compact(&mut b.insts.result, &keep);
    compact(&mut b.insts.args, &keep);
    compact(&mut b.insts.site, &keep);
    for bd in &mut b.blocks {
        let lo = (bd.insts.start as usize).min(n);
        let hi = lo.saturating_add(bd.insts.len as usize).min(n);
        bd.insts.start = before[lo];
        bd.insts.len = before[hi] - before[lo];
    }
    for v in &mut b.values {
        if let ValueDef::Inst(i) = &mut v.def
            && let Some(&to) = before.get(*i as usize)
        {
            *i = to;
        }
    }
    drop_blocks(b, &failed);
}

/// Deletes the blocks of `candidates` that are empty, have no params, and are no branch's
/// target, renumbering the rest in order.
fn drop_blocks(b: &mut Body, candidates: &[Block]) {
    let n = b.blocks.len();
    let mut targeted = vec![false; n];
    if let Some(t) = targeted.first_mut() {
        *t = true;
    }
    let mut mark = |t: Target| {
        if let Some(x) = targeted.get_mut(t.block.index()) {
            *x = true;
        }
    };
    for bd in &b.blocks {
        match bd.term {
            Term::Br(t) => mark(t),
            Term::Cbr { then, els, .. } => {
                mark(then);
                mark(els);
            }
            Term::Switch { cases, default, .. } => {
                for c in b.switch_cases(cases) {
                    mark(c.target);
                }
                mark(default);
            }
            Term::Ret(_) | Term::Unreachable => {}
        }
    }
    let mut gone = vec![false; n];
    for &f in candidates {
        let i = f.index();
        if i < n && !targeted[i] && b.blocks[i].insts.len == 0 && b.blocks[i].params.len == 0 {
            gone[i] = true;
        }
    }
    if !gone.contains(&true) {
        return;
    }
    let mut renumber = vec![0u32; n];
    let mut k = 0u32;
    for i in 0..n {
        renumber[i] = k;
        k += u32::from(!gone[i]);
    }
    let mut i = 0;
    b.blocks.retain(|_| {
        i += 1;
        !gone[i - 1]
    });
    let re = |t: &mut Target| {
        if let Some(&to) = renumber.get(t.block.index()) {
            t.block = Block(to);
        }
    };
    for bd in &mut b.blocks {
        match &mut bd.term {
            Term::Br(t) => re(t),
            Term::Cbr { then, els, .. } => {
                re(then);
                re(els);
            }
            Term::Switch { default, .. } => re(default),
            Term::Ret(_) | Term::Unreachable => {}
        }
    }
    for c in &mut b.cases {
        re(&mut c.target);
    }
    for v in &mut b.values {
        if let ValueDef::Param { block, .. } = &mut v.def
            && let Some(&to) = renumber.get(block.index())
        {
            *block = Block(to);
        }
    }
}

/// A compare of two operands.
#[derive(Clone, Copy)]
enum Rel {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl Rel {
    fn negated(self) -> Rel {
        match self {
            Rel::Lt => Rel::Ge,
            Rel::Le => Rel::Gt,
            Rel::Gt => Rel::Le,
            Rel::Ge => Rel::Lt,
            Rel::Eq => Rel::Ne,
            Rel::Ne => Rel::Eq,
        }
    }
}

/// The inequalities one source states (at most four).
#[derive(Default)]
struct Stated {
    n: usize,
    at: [Option<(Node, Node, i128)>; 4],
}

impl Stated {
    fn push(&mut self, from: Node, to: Node, w: i128) {
        if let Some(slot) = self.at.get_mut(self.n) {
            *slot = Some((from, to, w));
            self.n += 1;
        }
    }

    fn has(&self, e: &Edge) -> bool {
        self.at[..self.n].contains(&Some((e.from, e.to, e.w)))
    }
}

/// `a <= b + w`.
#[derive(Clone, Copy)]
struct Goal {
    a: Node,
    b: Node,
    w: i128,
}

/// The goals of one check: one, or two for an `==`.
#[derive(Clone, Copy)]
struct Goals {
    n: usize,
    at: [Goal; 2],
}

impl Goals {
    fn one(g: Goal) -> Goals {
        Goals { n: 1, at: [g, g] }
    }
}

/// `a + ka <= b + kb - strict` as a goal on the two nodes.
fn le(a: (Node, i128), b: (Node, i128), strict: i128) -> Result<Goal, Reject> {
    let w = b.1.checked_sub(a.1).and_then(|x| x.checked_sub(strict)).ok_or(Reject::Operand)?;
    Ok(Goal { a: a.0, b: b.0, w })
}

/// Where a check's goals must hold: right before instruction `inst` of `block`, or at the end of
/// `block` ([`END`]).
#[derive(Clone, Copy)]
struct At {
    block: u32,
    inst: u32,
}

const END: u32 = u32::MAX;

/// An int type: (signed, bits).
type IntTy = (bool, u32);

/// The width the checker reads an int type at. `usize` / `isize` are 64 bits on every target; a
/// test holds this to the type and layout layers.
pub fn int_width(bits: IntBits) -> u32 {
    match bits {
        IntBits::B8 => 8,
        IntBits::B16 => 16,
        IntBits::B32 => 32,
        IntBits::B64 | IntBits::Size => 64,
    }
}

/// The smallest and the largest value the checker reads an int type of `(signed, bits)` to have
/// (`bits` from [`int_width`]).
pub fn int_range((signed, bits): (bool, u32)) -> (i128, i128) {
    if signed { (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1) } else { (0, (1i128 << bits) - 1) }
}

fn fits(t: IntTy, c: i128) -> bool {
    let (lo, hi) = int_range(t);
    lo <= c && c <= hi
}

/// Calls `f` with every successor of block `u` and whether the edge is the true side of a `cbr`.
fn successors(b: &Body, u: usize, mut f: impl FnMut(Block, bool)) {
    match b.blocks[u].term {
        Term::Br(t) => f(t.block, false),
        Term::Cbr { then, els, .. } => {
            f(then.block, true);
            f(els.block, false);
        }
        Term::Switch { cases, default, .. } => {
            for c in b.switch_cases(cases) {
                f(c.target.block, false);
            }
            f(default.block, false);
        }
        Term::Ret(_) | Term::Unreachable => {}
    }
}

/// One function's tables.
struct Cx<'a> {
    m: &'a Module,
    b: &'a Body,
    /// Block of each instruction ([`NO_BLOCK`]: in none).
    block_of: Vec<u32>,
    /// Whether a block can be reached from the entry.
    reach: Vec<bool>,
    /// Per block, the edges into it from reachable blocks: how many, and for the last one seen
    /// its origin and whether it is the true side of a `cbr`.
    in_count: Vec<u32>,
    in_from: Vec<u32>,
    in_then: Vec<bool>,
    /// Entry and exit numbers of each reachable block in a walk of the certified tree: block
    /// `a` is an ancestor of `b` (or `b` itself) iff `a`'s interval encloses `b`'s.
    pre: Vec<u32>,
    post: Vec<u32>,
    /// Callees already examined, with the parameters they assert.
    asserted: Vec<(FuncId, u64)>,
}

impl<'a> Cx<'a> {
    fn new(m: &'a Module, b: &'a Body, dom: &[u32]) -> Result<Cx<'a>, Reject> {
        let n = b.blocks.len();
        if n == 0 || dom.len() != n {
            return Err(Reject::Cfg);
        }
        let mut block_of = vec![NO_BLOCK; b.insts.len()];
        for (bi, bd) in b.blocks.iter().enumerate() {
            for i in bd.insts.indices() {
                if let Some(s) = block_of.get_mut(i as usize) {
                    *s = bi as u32;
                }
            }
        }
        let mut reach = vec![false; n];
        let (mut in_count, mut in_from, mut in_then) = (vec![0u32; n], vec![NO_BLOCK; n], vec![false; n]);
        let mut work = vec![0usize];
        reach[0] = true;
        let mut well_formed = true;
        while let Some(u) = work.pop() {
            successors(b, u, |v, then| {
                let v = v.index();
                if v >= n {
                    well_formed = false;
                    return;
                }
                in_count[v] += 1;
                in_from[v] = u as u32;
                in_then[v] = then;
                if !reach[v] {
                    reach[v] = true;
                    work.push(v);
                }
            });
        }
        if !well_formed || dom[0] != 0 {
            return Err(Reject::Cfg);
        }
        // The children of each block in the certified tree, grouped by a counting sort.
        let mut first = vec![0u32; n + 1];
        for v in 1..n {
            if reach[v] {
                let p = dom[v] as usize;
                if p >= n || p == v || !reach[p] {
                    return Err(Reject::Cfg);
                }
                first[p + 1] += 1;
            }
        }
        for p in 0..n {
            first[p + 1] += first[p];
        }
        let mut fill = first.clone();
        let mut kids = vec![0u32; first[n] as usize];
        for v in 1..n {
            if reach[v] {
                let p = dom[v] as usize;
                kids[fill[p] as usize] = v as u32;
                fill[p] += 1;
            }
        }
        let (mut pre, mut post) = (vec![0u32; n], vec![0u32; n]);
        let mut clock = 1u32;
        let mut numbered = 1usize;
        let mut stack = vec![(0usize, first[0])];
        while let Some((u, next)) = stack.last_mut() {
            if *next < first[*u + 1] {
                let c = kids[*next as usize] as usize;
                *next += 1;
                pre[c] = clock;
                clock += 1;
                numbered += 1;
                stack.push((c, first[c]));
            } else {
                post[*u] = clock;
                clock += 1;
                stack.pop();
            }
        }
        // A cycle of parents hangs off no path from the entry and stays unnumbered.
        if numbered != reach.iter().filter(|r| **r).count() {
            return Err(Reject::Cfg);
        }
        let cx = Cx { m, b, block_of, reach, in_count, in_from, in_then, pre, post, asserted: Vec::new() };
        // Every edge `u → v` enters `v` from below `dom[v]`: by induction along any path from the
        // entry, each tree ancestor of a block is on the path.
        let mut certified = true;
        for u in (0..n).filter(|&u| cx.reach[u]) {
            successors(b, u, |v, _| certified &= cx.encloses(dom[v.index()] as usize, u));
        }
        if !certified {
            return Err(Reject::Cfg);
        }
        Ok(cx)
    }

    /// Block `a` is `b` or one of its ancestors in the certified tree: `a` dominates `b`.
    fn encloses(&self, a: usize, b: usize) -> bool {
        let live = |x: usize| self.reach.get(x).copied().unwrap_or(false);
        live(a) && live(b) && self.pre[a] <= self.pre[b] && self.post[b] <= self.post[a]
    }

    /// Instruction `i` runs before point `at` on every path to it.
    fn before(&self, i: u32, at: At) -> bool {
        match self.block_of.get(i as usize).copied().filter(|&bk| bk != NO_BLOCK) {
            Some(bi) if bi == at.block => i < at.inst && self.encloses(bi as usize, bi as usize),
            Some(bi) => self.encloses(bi as usize, at.block as usize),
            None => false,
        }
    }

    fn op(&self, i: u32) -> Option<Op> {
        self.b.insts.op.get(i as usize).copied()
    }

    fn def(&self, v: Value) -> Option<u32> {
        match self.b.values.get(v.index())?.def {
            ValueDef::Inst(i) => Some(i),
            ValueDef::Param { .. } => None,
        }
    }

    /// The int type of `v`, if it is an int.
    fn int_ty(&self, v: Value) -> Option<IntTy> {
        match self.b.value_ty(v).and_then(|t| self.m.types.key(t)) {
            Some(TypeKey::Int { signed, bits }) => Some((*signed, int_width(*bits))),
            _ => None,
        }
    }

    /// The largest value of `v`'s type when it is an unsigned int.
    fn top(&self, v: Value) -> Option<i128> {
        match self.int_ty(v)? {
            t @ (false, _) => Some(int_range(t).1),
            _ => None,
        }
    }

    /// The node and offset of an unsigned operand of an op whose type has the largest value
    /// `top`: a constant must be a value of that type.
    fn typed(&self, o: Operand, top: i128) -> Result<(Node, i128), Reject> {
        match self.unsigned(o)? {
            (Node::Zero, c) if c > top => Err(Reject::Operand),
            nk => Ok(nk),
        }
    }

    /// `v` has its value on every path to point `at`: its instruction runs before it, or it is
    /// a parameter of a block that dominates it.
    fn defined_at(&self, v: Value, at: At) -> bool {
        match self.b.values.get(v.index()).map(|d| d.def) {
            Some(ValueDef::Inst(i)) => self.before(i, at),
            Some(ValueDef::Param { block, .. }) => self.encloses(block.index(), at.block as usize),
            None => false,
        }
    }

    /// The constant an operand denotes, if it is one and within [`MAX_CONST`].
    fn konst(&self, o: Operand) -> Option<i128> {
        self.b.int_const(o).filter(|c| (-MAX_CONST..=MAX_CONST).contains(c))
    }

    /// The node and offset an operand of an unsigned op denotes: an unsigned int value, or a
    /// constant `0 ..=` [`MAX_CONST`].
    fn unsigned(&self, o: Operand) -> Result<(Node, i128), Reject> {
        match o {
            Operand::Value(v) if matches!(self.int_ty(v), Some((false, _))) => Ok((Node::Val(v), 0)),
            Operand::Const(_) => {
                self.konst(o).filter(|&c| c >= 0).map(|c| (Node::Zero, c)).ok_or(Reject::Operand)
            }
            _ => Err(Reject::Operand),
        }
    }

    /// The node and offset an operand of an op on int type `t` denotes: a value of type `t`, or
    /// a constant that is a value of `t`.
    fn of_type(&self, o: Operand, t: IntTy) -> Result<(Node, i128), Reject> {
        match o {
            Operand::Value(v) if self.int_ty(v) == Some(t) => Ok((Node::Val(v), 0)),
            Operand::Const(_) => {
                self.konst(o).filter(|&c| fits(t, c)).map(|c| (Node::Zero, c)).ok_or(Reject::Operand)
            }
            _ => Err(Reject::Operand),
        }
    }

    /// The two operands of a compare, read in their common int type (two constants: as
    /// themselves), and whether that type is unsigned.
    #[allow(clippy::type_complexity)]
    fn pair(&self, x: Operand, y: Operand) -> Result<((Node, i128), (Node, i128), bool), Reject> {
        let ty = |o: Operand| match o {
            Operand::Value(v) => self.int_ty(v).map(Some).ok_or(Reject::Operand),
            Operand::Const(_) => Ok(None),
            _ => Err(Reject::Operand),
        };
        let t = match (ty(x)?, ty(y)?) {
            (Some(a), Some(b)) if a != b => return Err(Reject::Operand),
            (Some(a), _) | (_, Some(a)) => Some(a),
            (None, None) => None,
        };
        match t {
            Some(t) => Ok((self.of_type(x, t)?, self.of_type(y, t)?, !t.0)),
            None => {
                let c = |o| self.konst(o).map(|c| (Node::Zero, c)).ok_or(Reject::Operand);
                Ok((c(x)?, c(y)?, false))
            }
        }
    }

    /// `(n, c)` when `o` is an unsigned `sub.sat n c` with a constant `c`.
    fn sub_sat(&self, o: Operand) -> Option<(Operand, i128)> {
        let Operand::Value(h) = o else { return None };
        let d = self.def(h)?;
        match (self.op(d)?, self.b.args(d)) {
            (Op::SubSat, &[n, c]) if self.top(h).is_some() => Some((n, self.konst(c).filter(|&c| c >= 0)?)),
            _ => None,
        }
    }

    /// The result and operands of instruction `i`, which must run before point `at`.
    fn defined(&self, i: u32, at: At) -> Result<(Value, &'a [Operand]), Reject> {
        let v = self.b.insts.result.get(i as usize).copied().flatten().ok_or(Reject::Source)?;
        if !self.before(i, at) {
            return Err(Reject::Region);
        }
        Ok((v, self.b.args(i)))
    }

    /// The block ending in the branch and the condition of contract check `q` (its failure
    /// call), if `q` is one.
    fn contract(&self, q: u32) -> Option<(u32, Operand)> {
        let (Some(Op::Call), [Operand::Func(f), ..]) = (self.op(q), self.b.args(q)) else { return None };
        let path = self.m.func(*f)?.path.as_str();
        if path != REQUIRES_FAILED && path != ENSURES_FAILED {
            return None;
        }
        let fb = *self.block_of.get(q as usize)?;
        let bd = self.b.blocks.get(fb as usize)?;
        let alone = bd.insts.start == q && bd.insts.len == 1 && matches!(bd.term, Term::Unreachable);
        let fb = fb as usize;
        // The entry is also entered by the call.
        if fb == 0 || !alone || !self.reach[fb] || self.in_count[fb] != 1 || self.in_then[fb] {
            return None;
        }
        let p = self.in_from[fb];
        match self.b.blocks.get(p as usize)?.term {
            Term::Cbr { cond, then, els } if els.block.index() == fb && then.block.index() != fb => {
                Some((p, cond))
            }
            _ => None,
        }
    }

    /// The point, the goals, and (a contract check) the branching block of check `q`.
    fn read_check(&self, q: u32) -> Result<(At, Goals, Option<u32>), Reject> {
        let bk = self.block_of.get(q as usize).copied().unwrap_or(NO_BLOCK);
        if bk == NO_BLOCK || !self.encloses(bk as usize, bk as usize) {
            return Err(Reject::NotACheck);
        }
        match self.op(q) {
            Some(Op::BoundsCheck) => {
                let &[i, n] = self.b.args(q) else { return Err(Reject::NotACheck) };
                let g = le(self.unsigned(i)?, self.unsigned(n)?, 1)?;
                Ok((At { block: bk, inst: q }, Goals::one(g), None))
            }
            Some(Op::Call) => {
                let (p, cond) = self.contract(q).ok_or(Reject::NotACheck)?;
                let (rel, x, y) = self.compare(cond, true)?;
                let (x, y, _) = self.pair(x, y)?;
                let goals = match rel {
                    Rel::Lt => Goals::one(le(x, y, 1)?),
                    Rel::Le => Goals::one(le(x, y, 0)?),
                    Rel::Gt => Goals::one(le(y, x, 1)?),
                    Rel::Ge => Goals::one(le(y, x, 0)?),
                    Rel::Eq => Goals { n: 2, at: [le(x, y, 0)?, le(y, x, 0)?] },
                    Rel::Ne => return Err(Reject::Goal),
                };
                Ok((At { block: p, inst: END }, goals, Some(p)))
            }
            _ => Err(Reject::NotACheck),
        }
    }

    /// One removed row: its check's goals each proved. Returns the block whose branch goes.
    fn row(
        &mut self,
        w: &FnWitness,
        r: &Removed,
        prev: Option<u32>,
    ) -> Result<Option<u32>, (Option<u32>, Reject)> {
        let q = r.inst;
        let at_check = |why| (None, why);
        if prev.is_some_and(|p| p >= q) {
            return Err(at_check(Reject::NotACheck));
        }
        let (at, goals, jump) = self.read_check(q).map_err(at_check)?;
        self.proves(w, goals.at[0], r.edges, at, 0)?;
        if goals.n == 2 {
            self.proves(w, goals.at[1], r.also, at, r.edges.len)?;
        } else if r.also.len != 0 {
            return Err(at_check(Reject::Goal));
        }
        Ok(jump)
    }

    /// One refuted row: the negation of one goal of a contract check that is not removed.
    fn refuted(
        &mut self,
        w: &FnWitness,
        r: &Refuted,
        prev: Option<u32>,
    ) -> Result<(), (Option<u32>, Reject)> {
        let q = r.inst;
        let at_check = |why| (None, why);
        if prev.is_some_and(|p| p >= q) || w.removed.iter().any(|x| x.inst == q) {
            return Err(at_check(Reject::NotACheck));
        }
        let (at, goals, jump) = self.read_check(q).map_err(at_check)?;
        if jump.is_none() {
            return Err(at_check(Reject::NotACheck));
        }
        let g = *goals
            .at
            .get(usize::from(r.half))
            .filter(|_| usize::from(r.half) < goals.n)
            .ok_or(at_check(Reject::Goal))?;
        // not `a <= b + w`: `b <= a - w - 1`
        let w_neg = g.w.checked_neg().and_then(|x| x.checked_sub(1)).ok_or(at_check(Reject::Operand))?;
        self.proves(w, Goal { a: g.b, b: g.a, w: w_neg }, r.edges, at, 0)
    }

    /// The edges `rows` of `w`, each validated at `at`, add up to goal `g`: `Σ mul·(from - to)`
    /// is `a - b` (zero's coefficient aside) and `Σ mul·w <= g.w`. `first`: the number of the
    /// first edge in reports.
    fn proves(
        &mut self,
        w: &FnWitness,
        g: Goal,
        rows: Span,
        at: At,
        first: u32,
    ) -> Result<(), (Option<u32>, Reject)> {
        if !rows.in_range(w.edges.len()) {
            return Err((None, Reject::Chain));
        }
        let mut coef: Vec<(Node, i128)> = Vec::with_capacity(2 * rows.len as usize + 2);
        let mut sum: i128 = 0;
        for (k, e) in rows.of(&w.edges).iter().enumerate() {
            let at_edge = |why| (Some(first.saturating_add(k as u32)), why);
            if e.mul == 0 || e.mul > MAX_MUL {
                return Err(at_edge(Reject::Edge));
            }
            let m = i128::from(e.mul);
            sum = m.checked_mul(e.w).and_then(|x| sum.checked_add(x)).ok_or(at_edge(Reject::Overflow))?;
            self.edge(e, at).map_err(at_edge)?;
            coef.push((e.from, m));
            coef.push((e.to, -m));
        }
        coef.push((g.a, -1));
        coef.push((g.b, 1));
        coef.sort_unstable_by_key(|c| c.0);
        let mut i = 0;
        while i < coef.len() {
            let node = coef[i].0;
            let mut c: i128 = 0;
            while i < coef.len() && coef[i].0 == node {
                c += coef[i].1;
                i += 1;
            }
            if node != Node::Zero && c != 0 {
                return Err((None, Reject::Chain));
            }
        }
        if sum > g.w {
            return Err((None, Reject::Weak));
        }
        Ok(())
    }

    /// One edge: its source exists, holds at point `at`, and states the edge.
    fn edge(&mut self, e: &Edge, at: At) -> Result<(), Reject> {
        let mut stated = Stated::default();
        match e.src {
            Source::Branch { block } => {
                let t = block as usize;
                // The entry is also entered by the call; any other block only by its edges.
                if t == 0 || !self.reach.get(t).copied().unwrap_or(false) || self.in_count[t] != 1 {
                    return Err(Reject::Source);
                }
                let Term::Cbr { cond, .. } = self.b.blocks[self.in_from[t] as usize].term else {
                    return Err(Reject::Source);
                };
                if !self.encloses(t, at.block as usize) {
                    return Err(Reject::Region);
                }
                self.cond(cond, self.in_then[t], &mut stated)?;
            }
            Source::Assert { call, arg } => {
                if self.op(call) != Some(Op::Call) {
                    return Err(Reject::Source);
                }
                let args = self.b.args(call);
                let (Some(&Operand::Func(f)), Some(&cond)) = (args.first(), args.get(arg as usize + 1))
                else {
                    return Err(Reject::Source);
                };
                if !self.before(call, at) {
                    return Err(Reject::Region);
                }
                if arg >= 64 || self.asserts(f) & (1u64 << arg) == 0 {
                    return Err(Reject::Source);
                }
                self.cond(cond, true, &mut stated)?;
            }
            Source::Check { inst } => {
                let (Some(Op::BoundsCheck), &[x, y]) = (self.op(inst), self.b.args(inst)) else {
                    return Err(Reject::Source);
                };
                if !self.before(inst, at) {
                    return Err(Reject::Region);
                }
                // `check.bounds` compares unsigned
                self.unsigned(x)?;
                self.unsigned(y)?;
                self.less(x, y, &mut stated)?;
            }
            Source::Arith { inst } => {
                let (v, args) = self.defined(inst, at)?;
                let t = self.int_ty(v).ok_or(Reject::Operand)?;
                let konst = |o: Operand| self.konst(o).filter(|&c| fits(t, c));
                match (self.op(inst), args) {
                    (Some(Op::Add), &[x, y]) => {
                        if let Some(c) = konst(y) {
                            self.equal(Node::Val(v), self.of_type(x, t)?, c, &mut stated);
                        }
                        if let Some(c) = konst(x) {
                            self.equal(Node::Val(v), self.of_type(y, t)?, c, &mut stated);
                        }
                        if stated.n == 0 {
                            return Err(Reject::Source);
                        }
                    }
                    (Some(Op::Sub), &[x, y]) => {
                        let c = konst(y).ok_or(Reject::Source)?;
                        self.equal(Node::Val(v), self.of_type(x, t)?, -c, &mut stated);
                    }
                    _ => return Err(Reject::Source),
                }
            }
            Source::SliceLen { inst } => {
                let (v, args) = self.defined(inst, at)?;
                let (Some(Op::SliceLen), &[Operand::Value(s)]) = (self.op(inst), args) else {
                    return Err(Reject::Source);
                };
                if self.top(v).is_none() {
                    return Err(Reject::Operand);
                }
                stated.push(Node::Val(v), Node::Len(s), 0);
                stated.push(Node::Len(s), Node::Val(v), 0);
            }
            Source::SliceMake { inst } => {
                let (s, args) = self.defined(inst, at)?;
                let (Some(Op::SliceMake), &[_, len]) = (self.op(inst), args) else {
                    return Err(Reject::Source);
                };
                self.equal(Node::Len(s), self.unsigned(len)?, 0, &mut stated);
            }
            Source::SubSat { inst } => {
                let (h, args) = self.defined(inst, at)?;
                let (Some(Op::SubSat), &[n, c]) = (self.op(inst), args) else {
                    return Err(Reject::Source);
                };
                let c = self.konst(c).filter(|&c| c >= 0).ok_or(Reject::Source)?;
                if self.top(h).is_none() {
                    return Err(Reject::Operand);
                }
                let (node, k) = self.unsigned(n)?;
                // h = max(n - c, 0) >= n - c
                stated.push(node, Node::Val(h), c - k);
            }
            Source::Range { value } => {
                let t = self.int_ty(value).ok_or(Reject::Operand)?;
                if !self.defined_at(value, at) {
                    return Err(Reject::Region);
                }
                let (lo, hi) = int_range(t);
                // lo <= value: 0 <= value - lo
                stated.push(Node::Zero, Node::Val(value), -lo);
                stated.push(Node::Val(value), Node::Zero, hi);
            }
            Source::Bits { inst } => {
                let (v, args) = self.defined(inst, at)?;
                let top = self.top(v).ok_or(Reject::Operand)?;
                match (self.op(inst), args) {
                    // v = x & y <= min(x, y)
                    (Some(Op::And), &[x, y]) => {
                        let ((nx, kx), (ny, ky)) = (self.typed(x, top)?, self.typed(y, top)?);
                        stated.push(Node::Val(v), nx, kx);
                        stated.push(Node::Val(v), ny, ky);
                    }
                    // v = x mod y, y >= 1 (y = 0 traps): v <= y - 1 and v <= x
                    (Some(Op::Rem), &[x, y]) => {
                        let ((nx, kx), (ny, ky)) = (self.typed(x, top)?, self.typed(y, top)?);
                        if ny == Node::Zero && ky < 1 {
                            return Err(Reject::Source);
                        }
                        stated.push(Node::Val(v), nx, kx);
                        stated.push(Node::Val(v), ny, ky - 1);
                    }
                    // v = floor(x / 2^k) <= x
                    (Some(Op::Shr), &[x, _]) => {
                        let (nx, kx) = self.typed(x, top)?;
                        stated.push(Node::Val(v), nx, kx);
                    }
                    _ => return Err(Reject::Source),
                }
            }
            Source::Cast { inst } => {
                let (v, args) = self.defined(inst, at)?;
                let (Some(Op::Cast(CastKind::Exact | CastKind::Wrap | CastKind::Sat)), &[Operand::Value(x)]) =
                    (self.op(inst), args)
                else {
                    return Err(Reject::Source);
                };
                let (Some((sv, wv)), Some((sx, wx))) = (self.int_ty(v), self.int_ty(x)) else {
                    return Err(Reject::Operand);
                };
                // only a target type holding every value of the source type keeps the value: a
                // narrower one truncates or saturates, a sign change moves negative or large values
                if !((sv == sx && wv >= wx) || (!sx && sv && wv > wx)) {
                    return Err(Reject::Source);
                }
                stated.push(Node::Val(v), Node::Val(x), 0);
                stated.push(Node::Val(x), Node::Val(v), 0);
            }
        }
        if stated.has(e) { Ok(()) } else { Err(Reject::Edge) }
    }

    /// `a = o + c` (`o` read as `(node, k)`), as both inequalities.
    fn equal(&self, a: Node, (node, k): (Node, i128), c: i128, stated: &mut Stated) {
        stated.push(a, node, k + c);
        stated.push(node, a, -(k + c));
    }

    /// The compare `bool` `cond` tests under its `not`s, as it reads when `cond` is `holds`.
    fn compare(&self, cond: Operand, holds: bool) -> Result<(Rel, Operand, Operand), Reject> {
        let (mut o, mut holds) = (cond, holds);
        for _ in 0..=NOT_DEPTH {
            let Operand::Value(v) = o else { return Err(Reject::Source) };
            let d = self.def(v).ok_or(Reject::Source)?;
            let args = self.b.args(d);
            let rel = match self.op(d) {
                Some(Op::Not) => {
                    let &[x] = args else { return Err(Reject::Source) };
                    o = x;
                    holds = !holds;
                    continue;
                }
                Some(Op::Lt) => Rel::Lt,
                Some(Op::Le) => Rel::Le,
                Some(Op::Gt) => Rel::Gt,
                Some(Op::Ge) => Rel::Ge,
                Some(Op::Eq) => Rel::Eq,
                Some(Op::Ne) => Rel::Ne,
                _ => return Err(Reject::Source),
            };
            let &[x, y] = args else { return Err(Reject::Source) };
            return Ok((if holds { rel } else { rel.negated() }, x, y));
        }
        Err(Reject::Source)
    }

    /// What the bool `cond` being `holds` states: a compare under `not`s.
    fn cond(&self, cond: Operand, holds: bool, stated: &mut Stated) -> Result<(), Reject> {
        let (rel, x, y) = self.compare(cond, holds)?;
        match rel {
            Rel::Lt => self.less(x, y, stated),
            Rel::Le => self.at_most(x, y, stated),
            Rel::Gt => self.less(y, x, stated),
            Rel::Ge => self.at_most(y, x, stated),
            Rel::Eq => self.at_most(x, y, stated).and_then(|()| self.at_most(y, x, stated)),
            Rel::Ne => Ok(()),
        }
    }

    /// `x < y`.
    fn less(&self, x: Operand, y: Operand, stated: &mut Stated) -> Result<(), Reject> {
        let ((nx, kx), (ny, ky), unsigned) = self.pair(x, y)?;
        stated.push(nx, ny, ky - kx - 1);
        // x < max(n - c, 0): n - c is positive, so x + c < n
        if unsigned
            && let Some((n, c)) = self.sub_sat(y)
            && let Ok((nn, kn)) = self.unsigned(n)
        {
            stated.push(nx, nn, kn - c - kx - 1);
        }
        Ok(())
    }

    /// `x <= y`.
    fn at_most(&self, x: Operand, y: Operand, stated: &mut Stated) -> Result<(), Reject> {
        let ((nx, kx), (ny, ky), unsigned) = self.pair(x, y)?;
        stated.push(nx, ny, ky - kx);
        // max(n - c, 0) <= y: n - c <= y
        if unsigned
            && let Some((n, c)) = self.sub_sat(x)
            && let Ok((nn, kn)) = self.unsigned(n)
        {
            stated.push(nn, ny, ky + c - kn);
        }
        Ok(())
    }

    /// The parameters (bit `k` = parameter `k`) that are true whenever `f` returns.
    fn asserts(&mut self, f: FuncId) -> u64 {
        if let Some(&(_, mask)) = self.asserted.iter().find(|(g, _)| *g == f) {
            return mask;
        }
        let mask = self.m.func(f).and_then(|f| f.body.as_ref()).map_or(0, asserting_params);
        self.asserted.push((f, mask));
        mask
    }
}

/// The parameters of `b` every return has seen tested true: a forward pass keeping, per block,
/// the parameters tested true on every path into it. A test is the true side of `cbr p` or the
/// false side of `cbr (not p)`.
fn asserting_params(b: &Body) -> u64 {
    let n = b.blocks.len();
    if n == 0 || n > ASSERT_BLOCKS {
        return 0;
    }
    let params = b.params(Block(0));
    let bit = |o: Operand| match o {
        Operand::Value(v) => params.iter().take(64).position(|&p| p == v).map_or(0, |k| 1u64 << k),
        _ => 0,
    };
    let tested = |u: usize, then: bool| -> u64 {
        let Term::Cbr { cond, then: t, els: e } = b.blocks[u].term else { return 0 };
        if t.block == e.block {
            return 0;
        }
        if then {
            return bit(cond);
        }
        let Operand::Value(c) = cond else { return 0 };
        match b.values.get(c.index()).map(|d| d.def) {
            Some(ValueDef::Inst(i)) if b.insts.op.get(i as usize) == Some(&Op::Not) => {
                b.args(i).first().map_or(0, |&p| bit(p))
            }
            _ => 0,
        }
    };
    // Greatest fixpoint: blocks start at "all", the entry at "none", and each edge lowers its
    // target. Blocks no path reaches stay at "all" and lower nothing.
    let mut known = vec![u64::MAX; n];
    known[0] = 0;
    let mut reach = vec![false; n];
    reach[0] = true;
    let mut changed = true;
    while changed {
        changed = false;
        for u in 0..n {
            if !reach[u] {
                continue;
            }
            let here = known[u];
            successors(b, u, |v, then| {
                let v = v.index();
                if v >= n {
                    return;
                }
                let new = if v == 0 { 0 } else { known[v] & (here | tested(u, then)) };
                if new != known[v] || !reach[v] {
                    known[v] = new;
                    reach[v] = true;
                    changed = true;
                }
            });
        }
    }
    let mut mask = u64::MAX;
    let mut returns = false;
    for u in (0..n).filter(|&u| reach[u]) {
        if matches!(b.blocks[u].term, Term::Ret(_)) {
            returns = true;
            mask &= known[u];
        }
    }
    if returns { mask } else { 0 }
}
