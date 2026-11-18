//! The checker of bounds-check witnesses, and the only code that deletes a `check.bounds`.
//!
//! [`check_func`] validates every chain of one function against the IR: each edge is looked up
//! at its source, compared with what that source states, and the weights are summed. Nothing is
//! searched for, and nothing here comes from the pass that wrote the witness: this file reads
//! the IR tables and the witness tables only, with its own reading of the control-flow graph,
//! of dominance (the certificate of [`FnWitness::dom`]), of asserting callees, and of int
//! widths. A rejection is a compiler bug. [`remove`] then deletes exactly the checks that were
//! accepted.
//!
//! What an accepted witness rests on: the IR verifier (the module must have passed it: an
//! operand is defined on every path to its use, and ops have their operand types), the op
//! semantics of the checked IR, and this file. An inequality read at a source is about the SSA
//! values it names; it still holds at the check because the source dominates the check and
//! every value it names is defined before the source, so no path from the source to the check
//! can define one of them again without passing the source again.
//!
//! Cost: linear in the function (blocks, CFG edges, instructions) plus the witness rows. On IR
//! that is not well-formed the checker refuses or accepts, but does not panic.

use super::{ASSERT_BLOCKS, Edge, FnWitness, MAX_CONST, NO_BLOCK, NOT_DEPTH, Node, Removed, Source};
use crate::ir::{Block, Body, CastKind, FuncId, Module, Op, Operand, Term, Value, ValueDef};
use crate::types::{IntBits, TypeKey};

/// Why a witness does not hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Reject {
    /// The function has no body, or the dominance table does not fit its control flow.
    Cfg,
    /// A row does not name a reachable `check.bounds`, or the rows are not in instruction order.
    NotACheck,
    /// A check operand or a compared operand is not an unsigned int or a constant in range.
    Operand,
    /// The source is not the kind of instruction or branch the edge says it is.
    Source,
    /// The source does not state the edge's inequality.
    Edge,
    /// The source does not hold at the check: it does not dominate it.
    Region,
    /// The edges do not lead from the index to the length.
    Chain,
    /// The weights do not put the index below the length.
    Weak,
}

impl Reject {
    fn text(self) -> &'static str {
        match self {
            Reject::Cfg => "the dominance table does not fit the function",
            Reject::NotACheck => "not a reachable check.bounds",
            Reject::Operand => "an operand is not an unsigned int",
            Reject::Source => "the source is not what the edge names",
            Reject::Edge => "the source does not state this inequality",
            Reject::Region => "the source does not dominate the check",
            Reject::Chain => "the edges do not lead from the index to the length",
            Reject::Weak => "the weights do not prove index < length",
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
    /// The edge of the check's chain, counted from 0.
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
}

impl Checked {
    pub fn func(&self) -> FuncId {
        self.func
    }

    /// The accepted checks, by instruction index.
    pub fn insts(&self) -> &[u32] {
        &self.insts
    }
}

/// Validates every chain of `w` against its function in `m`. Functions are independent: a
/// caller may check them in parallel.
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
    for r in &w.removed {
        cx.row(w, r, prev).map_err(|(edge, why)| fail(Some(r.inst), edge, why))?;
        prev = Some(r.inst);
    }
    Ok(Checked { func: w.func, insts: w.removed.iter().map(|r| r.inst).collect() })
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

/// Deletes the accepted checks. `m` must be the module they were checked against, unchanged; an
/// index that is not a `check.bounds` is left alone.
pub fn remove(m: &mut Module, checked: &[Checked]) {
    for c in checked {
        if let Some(b) = m.func_mut(c.func).and_then(|f| f.body.as_mut()) {
            remove_checks(b, &c.insts);
        }
    }
}

/// Removes the `check.bounds` instructions `dead` from `b`, keeping the order of the rest.
fn remove_checks(b: &mut Body, dead: &[u32]) {
    let n = b.insts.len();
    let mut keep = vec![true; n];
    for &d in dead {
        if b.insts.op.get(d as usize) == Some(&Op::BoundsCheck) {
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

    /// Instruction `i` runs before instruction `q` on every path to `q`.
    fn before(&self, i: u32, q: u32) -> bool {
        let block = |x: u32| self.block_of.get(x as usize).copied().filter(|&bk| bk != NO_BLOCK);
        match (block(i), block(q)) {
            (Some(bi), Some(bq)) if bi == bq => i < q,
            (Some(bi), Some(bq)) => self.encloses(bi as usize, bq as usize),
            _ => false,
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

    fn unsigned(&self, v: Value) -> bool {
        let key = self.b.value_ty(v).and_then(|t| self.m.types.key(t));
        matches!(key, Some(TypeKey::Int { signed: false, .. }))
    }

    /// The width in bits of `v`'s type when it is an unsigned int.
    fn width(&self, v: Value) -> Option<u32> {
        match self.b.value_ty(v).and_then(|t| self.m.types.key(t)) {
            Some(TypeKey::Int { signed: false, bits }) => Some(match bits {
                IntBits::B8 => 8,
                IntBits::B16 => 16,
                IntBits::B32 => 32,
                IntBits::B64 | IntBits::Size => 64,
            }),
            _ => None,
        }
    }

    /// The largest value of `v`'s type when it is an unsigned int.
    fn top(&self, v: Value) -> Option<i128> {
        self.width(v).map(|w| (1i128 << w) - 1)
    }

    /// The node and offset of operand `o` of an op whose type has the largest value `top`: a
    /// constant must be a value of that type.
    fn typed(&self, o: Operand, top: i128) -> Result<(Node, i128), Reject> {
        match self.operand(o)? {
            (Node::Zero, c) if c > top => Err(Reject::Operand),
            nk => Ok(nk),
        }
    }

    /// `v` has its value on every path to instruction `q`: its instruction runs before `q`, or
    /// it is a parameter of a block that dominates `q`'s.
    fn defined_at(&self, v: Value, q: u32) -> bool {
        match self.b.values.get(v.index()).map(|d| d.def) {
            Some(ValueDef::Inst(i)) => self.before(i, q),
            Some(ValueDef::Param { block, .. }) => {
                let at = self.block_of.get(q as usize).copied().unwrap_or(NO_BLOCK);
                at != NO_BLOCK && self.encloses(block.index(), at as usize)
            }
            None => false,
        }
    }

    fn konst(&self, o: Operand) -> Option<i128> {
        self.b.int_const(o).filter(|c| (0..=MAX_CONST).contains(c))
    }

    /// The node and offset an operand denotes.
    fn operand(&self, o: Operand) -> Result<(Node, i128), Reject> {
        match o {
            Operand::Value(v) if self.unsigned(v) => Ok((Node::Val(v), 0)),
            Operand::Const(_) => self.konst(o).map(|c| (Node::Zero, c)).ok_or(Reject::Operand),
            _ => Err(Reject::Operand),
        }
    }

    /// `(n, c)` when `o` is an unsigned `sub.sat n c` with a constant `c`.
    fn sub_sat(&self, o: Operand) -> Option<(Operand, i128)> {
        let Operand::Value(h) = o else { return None };
        let d = self.def(h)?;
        match (self.op(d)?, self.b.args(d)) {
            (Op::SubSat, &[n, c]) if self.unsigned(h) => Some((n, self.konst(c)?)),
            _ => None,
        }
    }

    /// The result and operands of instruction `i`, which must run before the check `q`.
    fn defined(&self, i: u32, q: u32) -> Result<(Value, &'a [Operand]), Reject> {
        let v = self.b.insts.result.get(i as usize).copied().flatten().ok_or(Reject::Source)?;
        if !self.before(i, q) {
            return Err(Reject::Region);
        }
        Ok((v, self.b.args(i)))
    }

    /// One row: the chain of check `r` proves its index below its length.
    fn row(&mut self, w: &FnWitness, r: &Removed, prev: Option<u32>) -> Result<(), (Option<u32>, Reject)> {
        let q = r.inst;
        let at_check = |why| (None, why);
        let placed = self.block_of.get(q as usize).is_some_and(|&bk| self.encloses(bk as usize, bk as usize));
        if prev.is_some_and(|p| p >= q) || self.op(q) != Some(Op::BoundsCheck) || !placed {
            return Err(at_check(Reject::NotACheck));
        }
        let &[i, n] = self.b.args(q) else { return Err(at_check(Reject::NotACheck)) };
        let (mut at, ki) = self.operand(i).map_err(at_check)?;
        let (end, kn) = self.operand(n).map_err(at_check)?;
        if !r.edges.in_range(w.edges.len()) {
            return Err(at_check(Reject::Chain));
        }
        let mut sum: i128 = 0;
        for (k, e) in r.edges.of(&w.edges).iter().enumerate() {
            let at_edge = |why| (Some(k as u32), why);
            if e.from != at {
                return Err(at_edge(Reject::Chain));
            }
            self.edge(e, q).map_err(at_edge)?;
            sum = sum.checked_add(e.w).ok_or(at_edge(Reject::Edge))?;
            at = e.to;
        }
        if at != end {
            return Err(at_check(Reject::Chain));
        }
        // i = at_i + ki <= at_n + sum + ki = n - kn + sum + ki
        let slack = sum.checked_add(ki).and_then(|s| s.checked_sub(kn));
        if slack.is_none_or(|s| s > -1) {
            return Err(at_check(Reject::Weak));
        }
        Ok(())
    }

    /// One edge: its source exists, holds at check `q`, and states the edge.
    fn edge(&mut self, e: &Edge, q: u32) -> Result<(), Reject> {
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
                if !self.encloses(t, self.block_of[q as usize] as usize) {
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
                if !self.before(call, q) {
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
                if !self.before(inst, q) {
                    return Err(Reject::Region);
                }
                self.less(x, y, &mut stated)?;
            }
            Source::Arith { inst } => {
                let (v, args) = self.defined(inst, q)?;
                if !self.unsigned(v) {
                    return Err(Reject::Operand);
                }
                match (self.op(inst), args) {
                    (Some(Op::Add), &[x, y]) => {
                        if let Some(c) = self.konst(y) {
                            self.equal(Node::Val(v), x, c, &mut stated)?;
                        }
                        if let Some(c) = self.konst(x) {
                            self.equal(Node::Val(v), y, c, &mut stated)?;
                        }
                        if stated.n == 0 {
                            return Err(Reject::Source);
                        }
                    }
                    (Some(Op::Sub), &[x, y]) => {
                        let c = self.konst(y).ok_or(Reject::Source)?;
                        self.equal(Node::Val(v), x, -c, &mut stated)?;
                    }
                    _ => return Err(Reject::Source),
                }
            }
            Source::SliceLen { inst } => {
                let (v, args) = self.defined(inst, q)?;
                let (Some(Op::SliceLen), &[Operand::Value(s)]) = (self.op(inst), args) else {
                    return Err(Reject::Source);
                };
                if !self.unsigned(v) {
                    return Err(Reject::Operand);
                }
                stated.push(Node::Val(v), Node::Len(s), 0);
                stated.push(Node::Len(s), Node::Val(v), 0);
            }
            Source::SliceMake { inst } => {
                let (s, args) = self.defined(inst, q)?;
                let (Some(Op::SliceMake), &[_, len]) = (self.op(inst), args) else {
                    return Err(Reject::Source);
                };
                self.equal(Node::Len(s), len, 0, &mut stated)?;
            }
            Source::SubSat { inst } => {
                let (h, args) = self.defined(inst, q)?;
                let (Some(Op::SubSat), &[n, c]) = (self.op(inst), args) else {
                    return Err(Reject::Source);
                };
                let c = self.konst(c).ok_or(Reject::Source)?;
                if !self.unsigned(h) {
                    return Err(Reject::Operand);
                }
                let (node, k) = self.operand(n)?;
                // h = max(n - c, 0) >= n - c
                stated.push(node, Node::Val(h), c - k);
            }
            Source::Range { value } => {
                let top = self.top(value).ok_or(Reject::Operand)?;
                if !self.defined_at(value, q) {
                    return Err(Reject::Region);
                }
                stated.push(Node::Zero, Node::Val(value), 0);
                stated.push(Node::Val(value), Node::Zero, top);
            }
            Source::Bits { inst } => {
                let (v, args) = self.defined(inst, q)?;
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
                let (v, args) = self.defined(inst, q)?;
                let (Some(Op::Cast(CastKind::Exact | CastKind::Wrap | CastKind::Sat)), &[Operand::Value(x)]) =
                    (self.op(inst), args)
                else {
                    return Err(Reject::Source);
                };
                let (Some(wv), Some(wx)) = (self.width(v), self.width(x)) else {
                    return Err(Reject::Operand);
                };
                // a narrower target truncates or saturates: only a widening cast keeps the value
                if wv < wx {
                    return Err(Reject::Source);
                }
                stated.push(Node::Val(v), Node::Val(x), 0);
                stated.push(Node::Val(x), Node::Val(v), 0);
            }
        }
        if stated.has(e) { Ok(()) } else { Err(Reject::Edge) }
    }

    /// `a = o + c`, as both inequalities.
    fn equal(&self, a: Node, o: Operand, c: i128, stated: &mut Stated) -> Result<(), Reject> {
        let (node, k) = self.operand(o)?;
        stated.push(a, node, k + c);
        stated.push(node, a, -(k + c));
        Ok(())
    }

    /// What the bool `cond` being `holds` states: a compare under `not`s.
    fn cond(&self, cond: Operand, holds: bool, stated: &mut Stated) -> Result<(), Reject> {
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
            return match if holds { rel } else { rel.negated() } {
                Rel::Lt => self.less(x, y, stated),
                Rel::Le => self.at_most(x, y, stated),
                Rel::Gt => self.less(y, x, stated),
                Rel::Ge => self.at_most(y, x, stated),
                Rel::Eq => self.at_most(x, y, stated).and_then(|()| self.at_most(y, x, stated)),
                Rel::Ne => Ok(()),
            };
        }
        Err(Reject::Source)
    }

    /// `x < y`.
    fn less(&self, x: Operand, y: Operand, stated: &mut Stated) -> Result<(), Reject> {
        let ((nx, kx), (ny, ky)) = (self.operand(x)?, self.operand(y)?);
        stated.push(nx, ny, ky - kx - 1);
        // x < max(n - c, 0): n - c is positive, so x + c < n
        if let Some((n, c)) = self.sub_sat(y)
            && let Ok((nn, kn)) = self.operand(n)
        {
            stated.push(nx, nn, kn - c - kx - 1);
        }
        Ok(())
    }

    /// `x <= y`.
    fn at_most(&self, x: Operand, y: Operand, stated: &mut Stated) -> Result<(), Reject> {
        let ((nx, kx), (ny, ky)) = (self.operand(x)?, self.operand(y)?);
        stated.push(nx, ny, ky - kx);
        // max(n - c, 0) <= y: n - c <= y
        if let Some((n, c)) = self.sub_sat(x)
            && let Ok((nn, kn)) = self.operand(n)
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
