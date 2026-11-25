//! Constant conditions: the one rule for which `if` runs only one arm of an instance
//! (contracts/lang-details.md, "§1 Type queries").
//!
//! **Forms.** The condition of an `if` (statement, `else if`, or value `if`) is constant when it
//! is `true`, `false`, `needs_drop::<T>()` (the built-in, not a declaration of that name), or `!`
//! of one of those. Nothing else is: no parentheses, `const` items, locals, `&&` / `||`, or
//! comparisons. Typeck finds the forms once per body ([`TypedBody::const_ifs`], [`classify`]).
//!
//! **Answers.** Per fn instance, after monomorphization, the call graph evaluates each form once
//! (`needs_drop` of the instance's type, [`crate::mono::needs_drop`]) and stores the answers
//! ([`crate::callgraph::CallGraph::taken`]). Every consumer reads them, none decides again:
//!
//! - lowering branches to the taken arm only; the other arm is lowered into blocks nothing
//!   reaches, which the post-pass deletes;
//! - the call graph makes no edge from the other arm (calls, drops, spawns, fn values made,
//!   flows), but still adds the instances it names as nodes: lowering lowers the arm before
//!   deleting it and asks for them;
//! - the effect pass counts no source in the other arm (`unsafe`, `scope`, `spawn`, intrinsic
//!   calls); its traps are gone with the deleted blocks.
//!
//! A dead arm is still type- and borrow-checked as written. Code in a closure literal inside a
//! dead arm belongs to the closure's own node, which stays as written (nothing reaches it).

use stave_syntax::{NodeKind, Term};

use crate::itemcx::ItemCx;
use crate::prim::Builtin;
use crate::typeck::{Target, TypedBody};
use crate::types::TypeId;

/// A constant condition, `!`s folded in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstCond {
    /// `true` / `false`.
    Known(bool),
    /// `needs_drop::<ty>()` (`ty` as written in the body), negated if `neg`.
    NeedsDrop { ty: TypeId, neg: bool },
}

impl ConstCond {
    fn not(self) -> ConstCond {
        match self {
            ConstCond::Known(b) => ConstCond::Known(!b),
            ConstCond::NeedsDrop { ty, neg } => ConstCond::NeedsDrop { ty, neg: !neg },
        }
    }

    /// The value in one instance: `needs_drop` answers for the type as written (the caller
    /// substitutes the instance's args).
    pub fn value(self, needs_drop: impl FnOnce(TypeId) -> bool) -> bool {
        match self {
            ConstCond::Known(b) => b,
            ConstCond::NeedsDrop { ty, neg } => needs_drop(ty) != neg,
        }
    }
}

/// An `if` whose condition is constant, with the token ranges (first, last) of its arms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConstIf {
    /// The `IfStmt` or `IfExpr` node.
    pub node: u32,
    /// Its `if` token: its owner (the body or a closure) is the owner of the arms' own code.
    pub tok: u32,
    pub cond: ConstCond,
    pub then: (u32, u32),
    /// The `else` arm (a block, an `else if`, or a value `if`'s second value), if any.
    pub els: Option<(u32, u32)>,
}

impl ConstIf {
    /// The arm an instance whose answer is `taken` never runs.
    pub fn dead_arm(&self, taken: bool) -> Option<(u32, u32)> {
        if taken { self.els } else { Some(self.then) }
    }
}

/// The `if`s with a constant condition among nodes `lo..=hi` of a body, by node. `targets`:
/// the body's, sorted by token.
pub(crate) fn classify(icx: &ItemCx<'_>, targets: &[(u32, Target)], lo: u32, hi: u32) -> Vec<ConstIf> {
    let mut out = Vec::new();
    for n in lo..=hi {
        if !matches!(icx.kind(n), Some(NodeKind::IfStmt | NodeKind::IfExpr)) {
            continue;
        }
        let kids = icx.kids_vec(n);
        let (Some(&c), Some(&t)) = (kids.first(), kids.get(1)) else { continue };
        let Some(cond) = cond_of(icx, targets, c) else { continue };
        let range = |k: u32| (icx.a(k), icx.end_tok(k));
        out.push(ConstIf { node: n, tok: icx.a(n), cond, then: range(t), els: kids.get(2).map(|&e| range(e)) });
    }
    out
}

/// The constant form of condition node `c`, if it is one.
fn cond_of(icx: &ItemCx<'_>, targets: &[(u32, Target)], c: u32) -> Option<ConstCond> {
    let mut c = c;
    let mut neg = false;
    // `!` nests by one `Unary` node per operator; bounded by the body's nesting limit
    for _ in 0..=crate::typeck::MAX_DEPTH {
        let a = icx.a(c);
        match icx.kind(c)? {
            NodeKind::Unary if icx.is(a, Term::Bang) => {
                c = icx.last_child(c)?;
                neg = !neg;
            }
            NodeKind::Primary
                if (icx.is(a, Term::KwTrue) || icx.is(a, Term::KwFalse)) && icx.end_tok(c) == a =>
            {
                let k = ConstCond::Known(icx.is(a, Term::KwTrue));
                return Some(if neg { k.not() } else { k });
            }
            NodeKind::Primary | NodeKind::Postfix => {
                // `needs_drop::<T>()`: the built-in's call is the whole node (its `()` ends it)
                let end = icx.end_tok(c);
                let paren = end.checked_sub(1)?;
                if !icx.is(a, Term::Ident) || !icx.is(paren, Term::LParen) || icx.close(paren) != end {
                    return None;
                }
                let i = targets.binary_search_by_key(&paren, |x| x.0).ok()?;
                let &Target::TypeQuery { ty, op: Builtin::NeedsDrop } = &targets[i].1 else { return None };
                let k = ConstCond::NeedsDrop { ty, neg: false };
                return Some(if neg { k.not() } else { k });
            }
            _ => return None,
        }
    }
    None
}

impl TypedBody {
    /// The constant-condition `if` at node `n`: its index in [`TypedBody::const_ifs`].
    pub fn const_if(&self, n: u32) -> Option<usize> {
        self.const_ifs.binary_search_by_key(&n, |c| c.node).ok()
    }
}

/// The arms instance answers `taken` make dead in a body with `ifs`: (owner, first, last)
/// token ranges; `owner(t)` is the body or closure that holds token `t` (as the caller numbers
/// them). A token is dead when a dead arm of its own owner holds it.
pub fn dead_arms(ifs: &[ConstIf], taken: &[bool], owner: impl Fn(u32) -> u32) -> Vec<(u32, u32, u32)> {
    ifs.iter()
        .zip(taken)
        .filter_map(|(c, &t)| c.dead_arm(t).map(|(lo, hi)| (owner(c.tok), lo, hi)))
        .collect()
}

/// Whether token `tok` of owner `owner` is in one of the `dead` arms ([`dead_arms`]).
pub fn is_dead(dead: &[(u32, u32, u32)], owner: u32, tok: u32) -> bool {
    dead.iter().any(|&(o, lo, hi)| o == owner && lo <= tok && tok <= hi)
}
