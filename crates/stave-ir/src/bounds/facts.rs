//! The difference-constraint graph behind the `LenEq` and `Offset` proofs.
//!
//! Values are unsigned integers, read as mathematical integers. A term is `node + k`: `k` a
//! constant, `node` the constant zero, an SSA value, or the length of an SSA slice value. Trapping
//! `add x c` / `sub x c` (constant `c`) are folded into the term of `x`: a defined result is exactly
//! `x ± c` (overflow traps), and the index's own trap stays in the code, so no wrap reasoning is
//! needed. `slice.len (slice.make p n)` is the term of `n`. Wrapping arithmetic, casts, loads, and
//! everything else are opaque nodes.
//!
//! A fact `x <= y + w` is an edge `x → y` of weight `w`, valid in a region of the function:
//!
//! - a branch: `cbr c T F` where `T` (or `F`) is the target's only predecessor edge; `c` (or `!c`)
//!   holds in every block that target dominates. `c` is an unsigned `lt le gt ge eq ne`, or `not`
//!   of one.
//! - an asserting call: `call @f .. c ..` where every `ret` of `f` is reached only through a branch
//!   on that parameter being true (`std.debug.assert`); `c` holds after the call.
//!
//! Every fact is about the current values of its operands wherever its region dominates the query
//! (an SSA value redefined in a later loop iteration passes the region's head again). A query
//! `i < n` searches for a path `i → .. → n` of total weight `≤ n.k - i.k - 1`, from the index, on
//! demand, over facts whose region dominates the check, and gives up after [`BOUNDS_STEP_LIMIT`]
//! edges. Any path found is a chain of true inequalities, so the proof is sound however the search
//! ends; giving up only keeps a check.

use std::cell::{OnceCell, RefCell};
use std::collections::VecDeque;

use rustc_hash::FxHashMap;

use super::{BOUNDS_STEP_LIMIT, Cx, Proof, preds_of};
use crate::ir::{Block, Body, FuncId, Op, Operand, Term as IrTerm, Value, ValueDef};
use crate::types::TypeKey;
use crate::verify::Dominators;

/// Folding depth of `add`/`sub` chains and `not` chains.
const DEPTH: u32 = 8;
/// Constants and offsets beyond this are opaque (keeps every weight sum far from `i64` overflow).
const MAX_K: i64 = 1 << 40;
/// An asserting callee is recognized only in bodies of at most this many blocks.
const ASSERT_BLOCKS: usize = 16;

const ZERO: u32 = 0;

fn val_node(v: Value) -> u32 {
    1 + 2 * v.0
}

fn len_node(s: Value) -> u32 {
    2 + 2 * s.0
}

/// `node + k`; `root` is the value the term was read from after folding (`None`: a constant).
#[derive(Clone, Copy, Debug)]
pub(super) struct Term {
    node: u32,
    k: i64,
    root: Option<Value>,
}

impl Term {
    pub(super) fn constant(c: u64) -> Option<Term> {
        let k = i64::try_from(c).ok().filter(|&k| k <= MAX_K)?;
        Some(Term { node: ZERO, k, root: None })
    }
}

/// Where a fact holds: block `block` and every block it dominates, from instruction `after`
/// on (exclusive) when `after` is set.
#[derive(Clone, Copy)]
struct Region {
    block: u32,
    after: Option<u32>,
}

/// `from <= to + w` (stored at `from`).
#[derive(Clone, Copy)]
struct Fact {
    to: u32,
    w: i64,
    at: Region,
    /// The `eq`/`ne` instruction when the fact is one half of a proved equality.
    eq: Option<u32>,
}

#[derive(Default)]
pub(super) struct Graph {
    adj: FxHashMap<u32, Vec<Fact>>,
}

/// The lazily built per-function graph and the callee summaries it needed.
#[derive(Default)]
pub(super) struct Facts {
    graph: OnceCell<Graph>,
    asserted: RefCell<FxHashMap<FuncId, u64>>,
}

impl Cx<'_> {
    fn unsigned(&self, v: Value) -> bool {
        self.b
            .value_ty(v)
            .and_then(|t| self.key(t))
            .is_some_and(|k| matches!(k, TypeKey::Int { signed: false, .. }))
    }

    fn small_const(&self, o: Operand) -> Option<i64> {
        self.b.int_const(o).and_then(|c| i64::try_from(c).ok()).filter(|c| (0..=MAX_K).contains(c))
    }

    /// The term of an unsigned integer operand.
    pub(super) fn term(&self, o: Operand) -> Option<Term> {
        let mut o = o;
        let mut k: i64 = 0;
        for _ in 0..DEPTH {
            if let Some(c) = self.small_const(o) {
                return Some(Term { node: ZERO, k: k.checked_add(c)?, root: None });
            }
            let Operand::Value(v) = o else { return None };
            if !self.unsigned(v) {
                return None;
            }
            let opaque = Term { node: val_node(v), k, root: Some(v) };
            let Some(d) = self.def(v) else { return Some(opaque) };
            let args = self.b.args(d);
            match (self.b.insts.op[d as usize], args) {
                (Op::Add, &[x, y]) => {
                    let (x, c) = match (self.small_const(x), self.small_const(y)) {
                        (_, Some(c)) => (x, c),
                        (Some(c), _) => (y, c),
                        _ => return Some(opaque),
                    };
                    k = k.checked_add(c)?;
                    o = x;
                }
                (Op::Sub, &[x, y]) => {
                    let Some(c) = self.small_const(y) else { return Some(opaque) };
                    k = k.checked_sub(c)?;
                    o = x;
                }
                (Op::SliceLen, &[s]) => return self.slice_len(s, k, v),
                _ => return Some(opaque),
            }
            if k.abs() > MAX_K {
                return None;
            }
        }
        None
    }

    /// `slice.len s + k` (the value `root` reads it).
    fn slice_len(&self, s: Operand, k: i64, root: Value) -> Option<Term> {
        let Operand::Value(sv) = s else { return None };
        if let Some(d) = self.def(sv)
            && self.b.insts.op[d as usize] == Op::SliceMake
            && let Some(&n) = self.b.args(d).get(1)
        {
            let t = self.term(n)?;
            return Some(Term { k: t.k.checked_add(k)?, ..t });
        }
        Some(Term { node: len_node(sv), k, root: Some(root) })
    }

    /// The length term of the slice `p` points to (`p` itself a slice reference).
    pub(super) fn slice_len_term(&self, p: Operand) -> Option<Term> {
        let Operand::Value(pv) = p else { return None };
        if let Some(d) = self.def(pv)
            && self.b.insts.op[d as usize] == Op::SliceMake
            && let Some(&n) = self.b.args(d).get(1)
        {
            return self.term(n);
        }
        Some(Term { node: len_node(pv), k: 0, root: None })
    }

    fn graph(&self) -> &Graph {
        self.facts.graph.get_or_init(|| self.build())
    }

    fn build(&self) -> Graph {
        let mut g = Graph::default();
        for t in 1..self.b.blocks.len() {
            if let Some((cond, pol)) = edge_cond(self.b, &self.preds, t) {
                self.cond_facts(&mut g, cond, pol, Region { block: t as u32, after: None }, DEPTH);
            }
        }
        for i in 0..self.b.insts.len() as u32 {
            if self.b.insts.op[i as usize] != Op::Call || !self.reachable_inst(i) {
                continue;
            }
            let args = self.b.args(i);
            let Some(&Operand::Func(f)) = args.first() else { continue };
            // Only a comparison argument can give a fact: skip the callee summary otherwise.
            if !args[1..].iter().any(|&a| self.is_cmp(a)) {
                continue;
            }
            let mask = self.asserted(f);
            for (k, &a) in args[1..].iter().enumerate().take(64) {
                if mask & (1u64 << k) != 0 {
                    let at = Region { block: self.block_of[i as usize], after: Some(i) };
                    self.cond_facts(&mut g, a, true, at, DEPTH);
                }
            }
        }
        g
    }

    fn is_cmp(&self, o: Operand) -> bool {
        let Operand::Value(v) = o else { return false };
        self.def(v).is_some_and(|d| {
            matches!(
                self.b.insts.op[d as usize],
                Op::Lt | Op::Le | Op::Gt | Op::Ge | Op::Eq | Op::Ne | Op::Not
            )
        })
    }

    /// The facts of condition `c` being `pol`.
    fn cond_facts(&self, g: &mut Graph, c: Operand, pol: bool, at: Region, depth: u32) {
        let Operand::Value(v) = c else { return };
        let Some(d) = self.def(v) else { return };
        let (op, args) = (self.b.insts.op[d as usize], self.b.args(d));
        if op == Op::Not {
            if let (&[x], 1..) = (args, depth) {
                self.cond_facts(g, x, !pol, at, depth - 1);
            }
            return;
        }
        let &[x, y] = args else { return };
        let mut le = |a: Operand, b: Operand, w: i64, eq: Option<u32>| self.add_le(g, a, b, w, at, eq);
        match (op, pol) {
            // x < y
            (Op::Lt, true) | (Op::Ge, false) => le(x, y, -1, None),
            // x >= y
            (Op::Lt, false) | (Op::Ge, true) => le(y, x, 0, None),
            // x > y
            (Op::Gt, true) | (Op::Le, false) => le(y, x, -1, None),
            // x <= y
            (Op::Gt, false) | (Op::Le, true) => le(x, y, 0, None),
            (Op::Eq, true) | (Op::Ne, false) => {
                le(x, y, 0, Some(d));
                le(y, x, 0, Some(d));
            }
            _ => {}
        }
    }

    /// Records `a <= b + w`.
    fn add_le(&self, g: &mut Graph, a: Operand, b: Operand, w: i64, at: Region, eq: Option<u32>) {
        let (Some(ta), Some(tb)) = (self.term(a), self.term(b)) else { return };
        if ta.node == tb.node {
            return;
        }
        // ta.node + ta.k <= tb.node + tb.k + w
        let Some(w) = tb.k.checked_add(w).and_then(|s| s.checked_sub(ta.k)) else { return };
        g.adj.entry(ta.node).or_default().push(Fact { to: tb.node, w, at, eq });
    }

    /// The bit set of `f`'s parameters that are true whenever `f` returns.
    fn asserted(&self, f: FuncId) -> u64 {
        if let Some(&m) = self.facts.asserted.borrow().get(&f) {
            return m;
        }
        let m = self.funcs.get(f.index()).and_then(|f| f.body.as_ref()).map_or(0, asserted_params);
        self.facts.asserted.borrow_mut().insert(f, m);
        m
    }

    fn holds(&self, at: Region, block: u32, inst: u32) -> bool {
        if at.block == block {
            at.after.is_none_or(|a| a < inst)
        } else {
            self.dom.dominates(at.block as usize, block as usize)
        }
    }

    /// The `LenEq` or `Offset` proof that `idx < n` holds right before instruction `at`.
    pub(super) fn prove_lt(&self, at: u32, idx: Term, n: Term) -> Option<Proof> {
        let bound = n.k.checked_sub(idx.k)?.checked_sub(1)?;
        let offset = Proof::Offset { base: idx.root.map(|v| v.0), k: idx.k };
        if idx.node == n.node {
            return (bound >= 0).then_some(offset);
        }
        let g = self.graph();
        let block = self.block_of[at as usize];
        let mut dist: FxHashMap<u32, (i64, Option<u32>)> = FxHashMap::default();
        dist.insert(idx.node, (0, None));
        let mut work = VecDeque::from([idx.node]);
        let mut steps = 0u32;
        while let Some(u) = work.pop_front() {
            let Some(&(du, eq_u)) = dist.get(&u) else { continue };
            for f in g.adj.get(&u).map_or(&[][..], |v| &v[..]) {
                steps += 1;
                if steps > BOUNDS_STEP_LIMIT {
                    return None;
                }
                if !self.holds(f.at, block, at) {
                    continue;
                }
                let Some(d) = du.checked_add(f.w) else { continue };
                let eq = eq_u.or(f.eq);
                if f.to == n.node && d <= bound {
                    return Some(match eq {
                        Some(cmp) => Proof::LenEq { cmp },
                        None => offset,
                    });
                }
                if dist.get(&f.to).is_none_or(|&(old, _)| d < old) {
                    dist.insert(f.to, (d, eq));
                    work.push_back(f.to);
                }
            }
        }
        None
    }
}

/// `(cond, pol)` when block `t` (not the entry) is entered only through the `pol` edge of a
/// `cbr cond`.
fn edge_cond(b: &Body, preds: &[Vec<(u32, bool)>], t: usize) -> Option<(Operand, bool)> {
    let &[(d, pol)] = preds.get(t)?.as_slice() else { return None };
    if t == 0 {
        return None;
    }
    let IrTerm::Cbr { cond, then, els } = b.blocks.get(d as usize)?.term else { return None };
    let (then, els) = (then.block.index(), els.block.index());
    let ok = if pol { then == t && els != t } else { els == t && then != t };
    ok.then_some((cond, pol))
}

/// The parameters (bit `k` = parameter `k`) every `ret` of `b` is reached only through a branch
/// on that parameter being true: `cbr p T F` with only `T` leading there, or `cbr (not p) T F`
/// with only `F`.
fn asserted_params(b: &Body) -> u64 {
    if b.blocks.len() > ASSERT_BLOCKS {
        return 0;
    }
    let params = b.params(Block(0));
    let param_bit = |o: Operand| match o {
        Operand::Value(v) => params.iter().position(|&p| p == v).filter(|&k| k < 64).map(|k| 1u64 << k),
        _ => None,
    };
    let dom = Dominators::compute(b);
    let preds = preds_of(b, &dom);
    let mut mask = u64::MAX;
    let mut returns = false;
    for (r, bd) in b.blocks.iter().enumerate() {
        if !dom.reachable(r) || !matches!(bd.term, IrTerm::Ret(_)) {
            continue;
        }
        returns = true;
        let mut here = 0u64;
        let mut walk = Some(r);
        while let Some(t) = walk {
            if let Some((cond, pol)) = edge_cond(b, &preds, t) {
                let negated = match cond {
                    Operand::Value(v) => match b.values.get(v.index()).map(|d| d.def) {
                        Some(ValueDef::Inst(i)) if b.insts.op.get(i as usize) == Some(&Op::Not) => {
                            b.args(i).first().copied()
                        }
                        _ => None,
                    },
                    _ => None,
                };
                let bit = match (pol, negated) {
                    (true, None) => param_bit(cond),
                    (false, Some(p)) => param_bit(p),
                    _ => None,
                };
                here |= bit.unwrap_or(0);
            }
            walk = dom.idom(t).filter(|&p| p != t);
        }
        mask &= here;
    }
    if returns { mask } else { 0 }
}
