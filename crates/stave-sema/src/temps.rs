//! Where a temporary ends: the one answer the borrow checker (stave-borrow `cfg`) and the
//! lowering (`lower_fn` / `lower_stmt`) both read, decided here once per body
//! ([`crate::TypedBody::stmt_ends`]).
//!
//! A temporary ends with its *statement*: the innermost of these that holds the node making it.
//!
//! - a statement (a child of a block);
//! - the condition of an `if` or `while` statement (each evaluation);
//! - a `match` guard;
//! - a branch of a value `if` (`then` and `else`; an `else if` is a value `if` inside the
//!   `else` branch);
//! - the body of an arm of a value `match`;
//! - an operand of the top-level `&&` of a `requires` / `ensures` clause, or the clause when it
//!   has none ([`conjuncts`]);
//! - the operand `e` of an `old(e)`.
//!
//! Everything else belongs to the statement around it: the scrutinee of a `match`, the header
//! of a `for`, the condition of a value `if`, the right side of `&&` / `||`.
//!
//! Each walker opens a statement only at a node this table names, and checks every temporary it
//! makes against the table ([`StmtEnds::of`] of the node making it is the innermost statement
//! it has open). Neither walker decides the end itself. Two values are not temporaries of a
//! statement: the iterator value a `for` is given ends when the loop ends or is left (with its
//! drop, if it has drop glue), and an `old(e)` value lives to the `ensures` clauses that read it.

use stave_syntax::{NodeKind, Term};

use crate::contracts::Contracts;
use crate::itemcx::ItemCx;

/// No statement holds the node (a `const` initializer).
pub const NONE: u32 = u32::MAX;

/// The kinds of statement a temporary can end with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bound {
    Stmt,
    /// The condition of an `if` or `while` statement.
    Cond,
    Guard,
    /// A branch of a value `if`.
    Branch,
    /// The body of an arm of a value `match`.
    Arm,
    /// A check of a contract clause.
    Clause,
    /// The operand `e` of an `old(e)`, evaluated at fn entry.
    Old,
}

/// The statement of every node of one body, by node index from the body's first node.
#[derive(Clone, Debug, Default)]
pub struct StmtEnds {
    lo: u32,
    of: Vec<u32>,
    /// The statements, sorted by node.
    kinds: Vec<(u32, Bound)>,
}

impl StmtEnds {
    /// The table of the body rooted at `root`, whose first node is `lo`.
    pub fn new(icx: &ItemCx<'_>, root: u32, lo: u32, contracts: Option<&Contracts>) -> StmtEnds {
        let n = (root + 1).saturating_sub(lo) as usize;
        let mut bound: Vec<Option<Bound>> = vec![None; n];
        let mark = |b: &mut Vec<Option<Bound>>, x: u32, k: Bound| {
            if let Some(s) = x.checked_sub(lo).and_then(|i| b.get_mut(i as usize)) {
                *s = Some(k);
            }
        };
        if let Some(c) = contracts {
            for cl in c.requires.iter().chain(&c.ensures) {
                for e in conjuncts(icx, cl.expr) {
                    mark(&mut bound, e, Bound::Clause);
                }
            }
            for &(_, e) in &c.olds {
                mark(&mut bound, e, Bound::Old);
            }
        }
        let mut of = vec![NONE; n];
        let mut kinds = Vec::new();
        // the nodes are in postorder: from the root down, a parent comes before its children,
        // and node `x`'s subtree is `descendants(x).start ..= x`
        let mut open: Vec<(u32, u32)> = Vec::new();
        for x in (lo..=root).rev() {
            for (k, b) in bounds_below(icx, x) {
                mark(&mut bound, k, b);
            }
            while open.last().is_some_and(|&(_, start)| start > x) {
                open.pop();
            }
            if let Some(b) = bound[(x - lo) as usize] {
                open.push((x, icx.ast.descendants(x).start));
                kinds.push((x, b));
            }
            of[(x - lo) as usize] = open.last().map_or(NONE, |o| o.0);
        }
        kinds.reverse();
        StmtEnds { lo, of, kinds }
    }

    /// The statement whose end ends a temporary made at node `n` ([`NONE`]: none).
    pub fn of(&self, n: u32) -> u32 {
        n.checked_sub(self.lo).and_then(|i| self.of.get(i as usize)).copied().unwrap_or(NONE)
    }

    /// Whether node `n` is a statement of its own for the temporaries made inside it.
    pub fn is_stmt(&self, n: u32) -> bool {
        n != NONE && self.of(n) == n
    }

    /// The kind of the statement a temporary made at node `n` ends with.
    pub fn bound(&self, n: u32) -> Option<Bound> {
        let s = self.of(n);
        self.kinds.binary_search_by_key(&s, |x| x.0).ok().map(|i| self.kinds[i].1)
    }
}

/// The children of `x` that are statements of their own, with their kind.
fn bounds_below(icx: &ItemCx<'_>, x: u32) -> Vec<(u32, Bound)> {
    let kids = icx.kids_vec(x);
    let arms = |out: &mut Vec<(u32, Bound)>, value: bool| {
        for &arm in kids.iter().skip(1) {
            // [ArmPattern, guard?, body]
            let ak = icx.kids_vec(arm);
            if ak.len() >= 3 {
                out.push((ak[1], Bound::Guard));
            }
            if value && let Some(&b) = ak.last() {
                out.push((b, Bound::Arm));
            }
        }
    };
    let mut out = Vec::new();
    match icx.kind(x) {
        Some(NodeKind::Block) => out.extend(kids.iter().map(|&k| (k, Bound::Stmt))),
        Some(NodeKind::IfStmt) => out.extend(kids.first().map(|&k| (k, Bound::Cond))),
        Some(NodeKind::IfExpr) => out.extend(kids.iter().skip(1).take(2).map(|&k| (k, Bound::Branch))),
        Some(NodeKind::MatchExpr) => arms(&mut out, true),
        Some(NodeKind::Stmt) => {
            let a = icx.a(x);
            if icx.is(a, Term::KwWhile) {
                out.extend(kids.first().map(|&k| (k, Bound::Cond)));
            } else if icx.is(a, Term::KwMatch) {
                arms(&mut out, false);
            }
        }
        _ => {}
    }
    out
}

/// The checks of a contract clause: the operands of its top-level `&&`, else the clause. Each is
/// its own check, run in order (a false one fails before the next is evaluated) and its own
/// statement for the temporaries it makes.
pub fn conjuncts(icx: &ItemCx<'_>, e: u32) -> Vec<u32> {
    if icx.kind(e) == Some(NodeKind::AndExpr) { icx.kids_vec(e).to_vec() } else { vec![e] }
}
