//! The summary a function states to its callers ([`crate::witness::Summary`]), with the proofs
//! the checker certifies it by.
//!
//! It is read off the function's own body by folding alone, never through the fact graph, so
//! that every reader gets the same rows: the function's own witness (which carries the proofs)
//! and each caller (which only reads the rows, [`super::claim_func`]).
//!
//! - **Entry.** Every reachable `requires` check must test an inequality whose two sides fold to
//!   zero, a parameter, or a slice parameter's length, plus constants. Each goal is one `pre`
//!   row and its proof is the fold of the left side, the row (`Requires`), and the fold of the
//!   right side backwards. One check that does not fit, and the function states no entry.
//! - **Safe.** A trapping `add`/`sub` by a constant that such a fold went through stays in its
//!   type's range when its operand does: shown from the operand's own type range, from a row
//!   that bounds the operand by zero or by a parameter whose type range then bounds it, or else
//!   by one more `pre` row stating the bound, which the caller proves.
//! - **Post.** The goals of the `ensures` checks whose true side dominates a return, with the
//!   returned value kept as a node of its own, are that return's rows. A row is stated when
//!   every reachable return has it (or makes it a true statement about constants); its proof at
//!   a return is the check's branch between the two folds.

use super::facts::{self, Side, ZERO, len_node, node_of, val_node};
use super::{Clause, ContractCheck, Cx, Goal};
use crate::ir::{Block, Op, Operand, Span, Term as IrTerm, Value};
use crate::witness::{
    Edge, NO_BLOCK, Node, PostAt, Pre, Removed, Row, SUMMARY_RETURNS, Safe, Source, Summary, Term,
};

/// One goal of an `ensures` check that holds at a return: the block its true side enters, and
/// its two sides folded to (node, offset) with the returned value a node of its own.
struct Held {
    block: u32,
    goal: Goal,
    a: (u32, i128),
    b: (u32, i128),
}

impl Held {
    /// `a.node <= b.node + w`.
    fn w(&self) -> Option<i128> {
        self.b.1.checked_sub(self.a.1)?.checked_sub(i128::from(self.goal.strict))
    }
}

fn span(from: usize, to: usize) -> Span {
    Span { start: from as u32, len: (to - from) as u32 }
}

impl Cx<'_> {
    /// The function's summary, the edges of its proofs appended to `out`.
    pub(super) fn summarize(&self, out: &mut Vec<Edge>) -> Summary {
        let mut s = Summary::default();
        let start = out.len();
        if self.entry_rows(&mut s, out).is_none() {
            out.truncate(start);
            s = Summary::default();
        }
        self.post_rows(&mut s, out);
        s
    }

    /// The summary term of a term node: zero, a parameter, or a parameter's length.
    fn param_term(&self, node: u32) -> Option<Term> {
        let params = self.b.params(Block(0));
        let index = |v: Value| params.iter().position(|&p| p == v).map(|i| i as u32);
        match node_of(node) {
            Node::Zero => Some(Term::Zero),
            Node::Val(v) => index(v).map(Term::Param),
            Node::Len(v) => index(v).map(Term::LenParam),
        }
    }

    /// The witness node of a summary term about the parameters.
    fn own_node(&self, t: Term) -> Option<Node> {
        let param = |i: u32| self.b.params(Block(0)).get(i as usize).copied();
        match t {
            Term::Zero => Some(Node::Zero),
            Term::Param(i) => param(i).map(Node::Val),
            Term::LenParam(i) => param(i).map(Node::Len),
            Term::Result | Term::LenResult => None,
        }
    }

    /// The value, least, and greatest value of the type of a value node.
    fn range_of(&self, node: u32) -> Option<(Value, i128, i128)> {
        let Node::Val(v) = node_of(node) else { return None };
        let (signed, bits) = self.int_ty(v)?;
        Some(if signed {
            (v, -(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
        } else {
            (v, 0, (1i128 << bits) - 1)
        })
    }

    fn param_range(&self, i: u32) -> Option<(Value, i128, i128)> {
        self.range_of(val_node(*self.b.params(Block(0)).get(i as usize)?))
    }

    /// Whether every reachable `ret` is reached only through the true side of check `c`.
    fn passed_at_returns(&self, c: &ContractCheck) -> bool {
        let IrTerm::Cbr { then, .. } = self.b.blocks[c.branch as usize].term else { return false };
        let t = then.block.index();
        matches!(facts::edge_cond(self.b, &self.preds, t), Some((_, true)))
            && (0..self.b.blocks.len())
                .filter(|&u| self.dom.reachable(u) && matches!(self.b.blocks[u].term, IrTerm::Ret(_)))
                .all(|u| self.dom.dominates(t, u))
    }

    /// The entry of `s`: one proof per `requires` check, each goal a `pre` row. `None`: a check
    /// whose condition is no inequality between zero, parameters, and parameter lengths.
    fn entry_rows(&self, s: &mut Summary, out: &mut Vec<Edge>) -> Option<()> {
        let first = out.len();
        for c in self.contracts.iter().filter(|c| c.clause == Clause::Requires) {
            let goals = self.goals_of(c.cond)?;
            let passed = self.passed_at_returns(c);
            let mut proofs = [Span::default(); 2];
            for (half, g) in goals.iter().enumerate().take(2) {
                let from = out.len();
                let (na, ka) = self.fold(g.lo, &mut |e| out.push(e))?.parts();
                let (nb, kb) = self.term(g.hi)?.parts();
                let w = kb.checked_sub(ka)?.checked_sub(i128::from(g.strict))?;
                let row = Row { a: self.param_term(na)?, b: self.param_term(nb)?, w };
                let clause = s.pre.len() as u32;
                let entry = if passed { s.entry.len() as u32 } else { NO_BLOCK };
                s.pre.push(Pre { row, entry, half: half as u8 });
                out.push(Edge {
                    from: node_of(na),
                    to: node_of(nb),
                    w,
                    src: Source::Requires { clause },
                    mul: 1,
                });
                self.up(g.hi, out);
                proofs[half] = span(from, out.len());
            }
            let site = self.b.insts.site[c.call as usize];
            s.entry.push(Removed { inst: c.call, site, edges: proofs[0], also: proofs[1] });
        }
        if !s.entry.is_empty() {
            self.safe_rows(s, first, out);
        }
        Some(())
    }

    /// The `safe` rows of `s`: the trapping `add`/`sub` instructions the entry proofs
    /// (`out[first..]`) read.
    fn safe_rows(&self, s: &mut Summary, first: usize, out: &mut Vec<Edge>) {
        let mut insts: Vec<u32> = out[first..]
            .iter()
            .filter_map(|e| match e.src {
                Source::Arith { inst } => Some(inst),
                _ => None,
            })
            .collect();
        insts.sort_unstable();
        insts.dedup();
        for d in insts {
            if let Some(x) = self.safe_row(s, d, out) {
                s.safe.push(x);
            }
        }
    }

    /// The proofs that `v = add x c` / `v = sub x c` (instruction `d`) has its exact result
    /// within `v`'s type, from the `pre` rows of `s` (one is added where none bounds `x`).
    /// `None`, with nothing added: `x` is no parameter term, or a constant out of range.
    fn safe_row(&self, s: &mut Summary, d: u32, out: &mut Vec<Edge>) -> Option<Safe> {
        let v = self.b.insts.result.get(d as usize).copied().flatten()?;
        let (_, lo, hi) = self.range_of(val_node(v))?;
        let konst = |o: Operand| self.small_const(o).filter(|c| (lo..=hi).contains(c));
        let (x, c) = match (self.b.insts.op[d as usize], self.b.args(d)) {
            (Op::Add, &[x, y]) => match (konst(x), konst(y)) {
                (_, Some(c)) => (x, c),
                (Some(c), None) => (y, c),
                (None, None) => return None,
            },
            (Op::Sub, &[x, y]) => (x, -konst(y)?),
            _ => return None,
        };
        let (node, k) = self.term(x)?.parts();
        let me = self.param_term(node)?;
        // node <= zero + above, and zero <= node + below
        let (above, below) = (hi.checked_sub(k)?.checked_sub(c)?, k.checked_add(c)?.checked_sub(lo)?);
        if node == ZERO && (above < 0 || below < 0) {
            return None;
        }
        let from = out.len();
        self.down(x, out);
        self.to_zero(s, node, me, above, out);
        let mid = out.len();
        self.from_zero(s, node, me, below, out);
        self.up(x, out);
        Some(Safe { inst: d, hi: span(from, mid), lo: span(mid, out.len()) })
    }

    /// Edges from parameter node `node` (summary term `me`) to zero that prove
    /// `node <= bound`.
    fn to_zero(&self, s: &mut Summary, node: u32, me: Term, bound: i128, out: &mut Vec<Edge>) {
        if node == ZERO {
            return;
        }
        let top = |v: Value, w: i128| Edge { from: Node::Val(v), to: Node::Zero, w, src: Source::Range { value: v }, mul: 1 };
        if let Some((v, _, max)) = self.range_of(node)
            && max <= bound
        {
            out.push(top(v, max));
            return;
        }
        for (clause, p) in s.pre.iter().enumerate() {
            let (Row { a, b, w }, src) = (p.row, Source::Requires { clause: clause as u32 });
            let Some(to) = self.own_node(b).filter(|_| a == me) else { continue };
            let via = Edge { from: node_of(node), to, w, src, mul: 1 };
            match b {
                Term::Zero if w <= bound => {
                    out.push(via);
                    return;
                }
                Term::Param(i) => {
                    if let Some((q, _, max)) = self.param_range(i)
                        && w.checked_add(max).is_some_and(|x| x <= bound)
                    {
                        out.extend([via, top(q, max)]);
                        return;
                    }
                }
                _ => {}
            }
        }
        let clause = s.pre.len() as u32;
        s.pre.push(Pre { row: Row { a: me, b: Term::Zero, w: bound }, entry: NO_BLOCK, half: 0 });
        out.push(Edge {
            from: node_of(node),
            to: Node::Zero,
            w: bound,
            src: Source::Requires { clause },
            mul: 1,
        });
    }

    /// Edges from zero to parameter node `node` (summary term `me`) that prove
    /// `0 <= node + bound`.
    fn from_zero(&self, s: &mut Summary, node: u32, me: Term, bound: i128, out: &mut Vec<Edge>) {
        if node == ZERO {
            return;
        }
        let bottom = |v: Value, w: i128| Edge { from: Node::Zero, to: Node::Val(v), w, src: Source::Range { value: v }, mul: 1 };
        if let Some((v, min, _)) = self.range_of(node)
            && -min <= bound
        {
            out.push(bottom(v, -min));
            return;
        }
        for (clause, p) in s.pre.iter().enumerate() {
            let (Row { a, b, w }, src) = (p.row, Source::Requires { clause: clause as u32 });
            let Some(from) = self.own_node(a).filter(|_| b == me) else { continue };
            let via = Edge { from, to: node_of(node), w, src, mul: 1 };
            match a {
                Term::Zero if w <= bound => {
                    out.push(via);
                    return;
                }
                Term::Param(i) => {
                    if let Some((q, min, _)) = self.param_range(i)
                        && w.checked_sub(min).is_some_and(|x| x <= bound)
                    {
                        out.extend([bottom(q, -min), via]);
                        return;
                    }
                }
                _ => {}
            }
        }
        let clause = s.pre.len() as u32;
        s.pre.push(Pre { row: Row { a: Term::Zero, b: me, w: bound }, entry: NO_BLOCK, half: 0 });
        out.push(Edge {
            from: Node::Zero,
            to: node_of(node),
            w: bound,
            src: Source::Requires { clause },
            mul: 1,
        });
    }

    /// The value block `r` returns, if it returns one.
    fn ret_value(&self, r: usize) -> Option<Value> {
        match self.b.blocks[r].term {
            IrTerm::Ret(Some(Operand::Value(v))) => Some(v),
            _ => None,
        }
    }

    /// The goals of the `ensures` checks whose true side is the only way to return block `r`.
    fn held_at(&self, r: usize) -> Vec<Held> {
        let stop = self.ret_value(r);
        let mut out = Vec::new();
        for c in self.contracts.iter().filter(|c| c.clause == Clause::Ensures) {
            let IrTerm::Cbr { then, .. } = self.b.blocks[c.branch as usize].term else { continue };
            let t = then.block.index();
            if !matches!(facts::edge_cond(self.b, &self.preds, t), Some((_, true)))
                || !self.dom.dominates(t, r)
            {
                continue;
            }
            for goal in self.goals_of(c.cond).unwrap_or_default() {
                let sides = (
                    self.side_term_to(Side::Val(goal.lo), stop),
                    self.side_term_to(Side::Val(goal.hi), stop),
                );
                if let (Some(a), Some(b)) = sides {
                    out.push(Held { block: t as u32, goal, a: a.parts(), b: b.parts() });
                }
            }
        }
        out
    }

    /// The `post` rows of `s`, each with its proof at every reachable return.
    fn post_rows(&self, s: &mut Summary, out: &mut Vec<Edge>) {
        let rets: Vec<usize> = (0..self.b.blocks.len())
            .filter(|&u| self.dom.reachable(u) && matches!(self.b.blocks[u].term, IrTerm::Ret(_)))
            .collect();
        if rets.is_empty() || rets.len() > SUMMARY_RETURNS {
            return;
        }
        let held: Vec<Vec<Held>> = rets.iter().map(|&r| self.held_at(r)).collect();
        // The rows are read at the first return of a value that is no parameter: there the
        // result is told apart from the parameters.
        let own = |r: usize| self.ret_value(r).filter(|&v| self.param_term(val_node(v)).is_none());
        let base = rets.iter().position(|&r| own(r).is_some()).unwrap_or(0);
        let result = self.ret_value(rets[base]);
        let term = |node: u32| match result {
            Some(v) if node == val_node(v) => Some(Term::Result),
            Some(v) if node == len_node(v) => Some(Term::LenResult),
            _ => self.param_term(node),
        };
        let mut rows: Vec<Row> = Vec::new();
        for h in &held[base] {
            if let (Some(a), Some(b), Some(w)) = (term(h.a.0), term(h.b.0), h.w()) {
                let row = Row { a, b, w };
                if a != b && !rows.contains(&row) {
                    rows.push(row);
                }
            }
        }
        for row in rows {
            let from = out.len();
            let k = s.post.len() as u32;
            let mut proofs = Vec::with_capacity(rets.len());
            let all = rets.iter().zip(&held).all(|(&r, held)| {
                let at = out.len();
                let ok = self.post_at(row, r, held, out).is_some();
                proofs.push(PostAt { row: k, block: r as u32, edges: span(at, out.len()) });
                ok
            });
            if all {
                s.post.push(row);
                s.post_at.extend(proofs);
            } else {
                out.truncate(from);
            }
        }
    }

    /// The proof of `row` at the end of return block `r`, appended to `out`: nothing when the
    /// row there is a true statement about one node, else the branch of a goal of `held` that
    /// states it, between the folds of its two sides.
    fn post_at(&self, row: Row, r: usize, held: &[Held], out: &mut Vec<Edge>) -> Option<()> {
        let IrTerm::Ret(ret) = self.b.blocks[r].term else { return None };
        let params = self.b.params(Block(0));
        let side = |t: Term| -> Option<(u32, i128)> {
            let param = |i: u32| params.get(i as usize).copied();
            match (t, ret) {
                (Term::Zero, _) => Some((ZERO, 0)),
                (Term::Param(i), _) => Some((val_node(param(i)?), 0)),
                (Term::LenParam(i), _) => Some((len_node(param(i)?), 0)),
                (Term::Result, Some(Operand::Value(v))) => Some((val_node(v), 0)),
                (Term::Result, Some(o)) => self.small_const(o).map(|c| (ZERO, c)),
                (Term::LenResult, Some(Operand::Value(v))) => Some((len_node(v), 0)),
                (Term::Result | Term::LenResult, _) => None,
            }
        };
        let (a, b) = (side(row.a)?, side(row.b)?);
        let w = row.w.checked_add(b.1)?.checked_sub(a.1)?;
        if a.0 == b.0 {
            return (w >= 0).then_some(());
        }
        let h = held.iter().find(|h| h.a.0 == a.0 && h.b.0 == b.0 && h.w().is_some_and(|x| x <= w))?;
        let stop = self.ret_value(r);
        let ((from, klo), (to, khi)) = (self.written(h.goal.lo)?, self.written(h.goal.hi)?);
        self.side_up(Side::Val(h.goal.lo), stop, out);
        out.push(Edge {
            from,
            to,
            w: khi - klo - i128::from(h.goal.strict),
            src: Source::Branch { block: h.block },
            mul: 1,
        });
        self.side_down(Side::Val(h.goal.hi), stop, out);
        Some(())
    }
}
