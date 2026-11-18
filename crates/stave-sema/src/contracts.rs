//! Contracts: `requires(e)` and `ensures(e)` clauses on fn items and inherent methods.
//!
//! **Typing** ([`Ck::contract_clauses`]): each clause is a `bool` expression typed in the fn's
//! scope, after the parameters and before the body. Inside a clause `old(e)` is the value of `e`
//! at entry ([`Target::Old`]; E0213 in a `requires` or nested); inside an `ensures`, `result` is
//! the returned value, a local whose binding node is the fn's first `Ensures` node (resolution
//! binds both names there, whatever else they name).
//!
//! **Operand rules** ([`Ck::contract_rules`], E0211, at most one per clause, the first in source
//! order): no `unsafe`, closure literal, `spawn`, f-string, `?`, `&mut` borrow, `static`, or
//! intrinsic; a call (fn, method, operator, trait method) only of a safe fn whose every parameter
//! is a shared reference or a Copy type, so a clause neither writes nor moves; a call through a fn
//! value only if its type denies `io`, `lock`, and `block`. A parameter of type `&mut T` is not
//! read in a `requires`; an `ensures` may read through it (never write: the call rule). An
//! `ensures` runs after the body, which may have moved a by-value parameter, so there a parameter
//! must be a reference or Copy. E0212 / E0213 (case `param`): a parameter named `result` / `old`
//! in a fn with an `ensures`. E0213: `old(e)` with a non-Copy `e`, or `result` inside it (case
//! `entry`). E0211 case `trait-impl`: contracts on a trait impl method.
//!
//! **Purity** ([`purity`], E0211 case `impure-call`): a callee (or a drop) in a clause whose effect
//! set ([`crate::effects`]) has `io`, `lock`, or `block`. Calls in clauses are call-graph edges of
//! the fn (the check runs in the callee), and a clause precedes the body, so the edge of a callee
//! called there is at the clause's site.
//!
//! **Checks** (lowering): the `requires` clauses at entry in order (on a `test fn` they never
//! panic: with parameters they filter its inputs, [`crate::testgen`]; without, they are not
//! evaluated), then the `old(e)` snapshots, then the `ensures` clauses in order at each return; a
//! false clause calls `runtime.panic.requires_failed` (located at the caller's user site, which
//! every call that can reach a `requires` stores first) or `runtime.panic.ensures_failed` (at the
//! return). A fn with a checked clause has `trap` ([`Contracts::trap_span`]).

use rustc_hash::FxHashSet;
use stave_diag::{Diagnostic, Span};
use stave_syntax::{NodeKind, Term};

use crate::Program;
use crate::callgraph::{EdgeKind, NodeKind as GNode};
use crate::decls::{DeclFlags, DeclKind};
use crate::diag::mk;
use crate::effects::{EffectSets, Effects, OpKind};
use crate::resolve::Res;
use crate::traits::MethodRef;
use crate::typeck::{Ck, Target, Ty, Typeck};
use crate::types::{GArg, Interner, TypeId, TypeKind, ids};

/// One clause: its node, its expression, and its source span (keyword to `)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContractClause {
    pub node: u32,
    pub expr: u32,
    pub span: Span,
}

/// The contract clauses of a fn body ([`crate::TypedBody::contracts`]).
#[derive(Clone, Debug, Default)]
pub struct Contracts {
    pub requires: Vec<ContractClause>,
    pub ensures: Vec<ContractClause>,
    /// `old(e)` in the `ensures` clauses, in source order: (the call's `(` token, `e`'s node).
    pub olds: Vec<(u32, u32)>,
    /// The binding node of `result` (the first `Ensures` node; `u32::MAX` without one).
    pub result: u32,
    /// The `requires` clauses run at entry: not on a `test fn`.
    pub check_requires: bool,
}

impl Contracts {
    /// The `requires` clauses lowering checks.
    pub fn checked_requires(&self) -> &[ContractClause] {
        if self.check_requires { &self.requires } else { &[] }
    }

    /// Whether the fn has a check that can fail (its `trap` op): the first such clause's span.
    pub fn trap_span(&self) -> Option<Span> {
        self.checked_requires().first().or(self.ensures.first()).map(|c| c.span)
    }

    /// Whether span `s` starts inside a clause.
    pub fn holds(&self, s: Span) -> bool {
        self.requires.iter().chain(&self.ensures).any(|c| {
            c.span.file == s.file && s.offset >= c.span.offset && s.offset < c.span.offset + c.span.len
        })
    }
}

/// The clause being typed.
#[derive(Clone, Copy, Debug)]
pub struct ClauseCx {
    pub ensures: bool,
    /// How many `old(..)` calls enclose the current expression.
    pub old_depth: u32,
}

/// The clause's source text as one line (a run of white space holding a line break becomes one
/// space): the second line of a `Contract` panic report.
pub fn clause_line(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut ws = String::new();
    for ch in src.chars() {
        if ch.is_ascii_whitespace() {
            ws.push(ch);
            continue;
        }
        if !ws.is_empty() {
            if ws.contains(['\n', '\r']) {
                out.push(' ');
            } else {
                out.push_str(&ws);
            }
            ws.clear();
        }
        out.push(ch);
    }
    out
}

impl Ck<'_> {
    /// Types the `requires` and `ensures` clauses of fn node `n` (return type `ret`) and records
    /// them in [`Ck::contracts`].
    pub(crate) fn contract_clauses(&mut self, n: u32, ret: TypeId) {
        let icx = self.icx;
        let clauses: Vec<(u32, bool)> = icx
            .kids_vec(n)
            .iter()
            .filter_map(|&k| match icx.kind(k) {
                Some(NodeKind::Requires) => Some((k, false)),
                Some(NodeKind::Ensures) => Some((k, true)),
                _ => None,
            })
            .collect();
        let Some(&(first, _)) = clauses.first() else { return };
        let decl = self.prog.decl(self.decl).copied();
        if decl.is_some_and(|d| d.kind == DeclKind::Method && d.flags.has(DeclFlags::TRAIT_IMPL)) {
            let d = mk("E0211", Some("trait-impl"), &[], icx.tok_span(icx.a(first)), &[]);
            self.report(d);
        }
        let mut c = Contracts {
            result: u32::MAX,
            check_requires: !decl.is_some_and(|d| d.flags.has(DeclFlags::TEST)),
            ..Contracts::default()
        };
        let (diverges, reachable) = (self.diverges, self.reachable);
        for &(k, ensures) in &clauses {
            let Some(e) = icx.first_child(k) else { continue };
            let a = icx.a(k);
            let cl = ContractClause { node: k, expr: e, span: icx.range_span(a, icx.close(a + 1)) };
            if ensures {
                if c.ensures.is_empty() {
                    c.result = k;
                    self.locals.insert(k, ret);
                }
                c.ensures.push(cl);
            } else {
                c.requires.push(cl);
            }
            self.contract = Some(ClauseCx { ensures, old_depth: 0 });
            self.check_expr(e, ids::BOOL, None);
            self.contract = None;
            self.diverges = diverges;
            self.reachable = reachable;
        }
        // `result` is no local of the body
        if !c.ensures.is_empty() {
            self.locals.remove(&c.result);
        }
        c.olds = std::mem::take(&mut self.olds);
        self.contracts = Some(Box::new(c));
    }

    /// The operand rules of the clauses of fn node `n`, once the body is typed (literal types are
    /// final): E0211, E0212, E0213.
    pub(crate) fn contract_rules(&mut self, n: u32) {
        let Some(c) = self.contracts.take() else { return };
        let icx = self.icx;
        if !c.ensures.is_empty()
            && let Some(ps) = icx.child(n, NodeKind::Params)
        {
            for &p in icx.kids_vec(ps).iter() {
                let a = icx.a(p);
                if !icx.is(a, Term::Ident) {
                    continue;
                }
                let d = match icx.tok_str(a) {
                    "result" => mk("E0212", None, &[], icx.tok_span(a), &[]),
                    "old" => mk("E0213", Some("param"), &[], icx.tok_span(a), &[]),
                    _ => continue,
                };
                self.report(d);
            }
        }
        let olds: Vec<(u32, u32)> =
            c.olds.iter().map(|&(paren, _)| (paren + 1, icx.close(paren).saturating_sub(1))).collect();
        for cl in &c.requires {
            self.clause_rules(cl, false, c.result, &olds);
        }
        for cl in &c.ensures {
            self.clause_rules(cl, true, c.result, &olds);
        }
        for &(_, e) in &c.olds {
            let Some(&t) = e.checked_sub(self.lo).and_then(|i| self.node_ty.get(i as usize)) else {
                continue;
            };
            let Ty::T(t) = self.resolve(t) else { continue };
            if !self.contract_value_ok(t, false) {
                let tn = self.show(t);
                let d = mk("E0213", None, &[("type", &tn)], icx.node_span(e), &[]);
                self.report(d);
            }
        }
        self.contracts = Some(c);
    }

    /// Whether a value of type `t` may be passed or read in a clause: a shared reference (or,
    /// with `mut_ref`, any reference) or a Copy type. Error types pass (no cascades).
    fn contract_value_ok(&mut self, t: TypeId, mut_ref: bool) -> bool {
        if t.is_none() || t == ids::ERROR || self.tys.has_error(t) {
            return true;
        }
        let shape = match self.tys.kind(t) {
            TypeKind::Ref { mutable, .. } => Some(mut_ref || !*mutable),
            TypeKind::Never => Some(true),
            _ => None,
        };
        if let Some(ok) = shape {
            return ok;
        }
        let env = self.genv();
        self.w.is_copy(&mut self.tys, t, env)
    }

    /// Whether every parameter of callee `decl` (substituted by `args`, `Self` = `self_ty`) takes
    /// a shared reference or a Copy value, and the callee is safe.
    fn contract_call_ok(&mut self, decl: crate::ids::DeclId, args: &[GArg], self_ty: Option<TypeId>) -> bool {
        let Some(sig) = self.sigs.fn_sig(decl) else { return true };
        if sig.is_unsafe {
            return false;
        }
        let params = sig.params.clone();
        params.iter().all(|&p| {
            let t = self.tys.subst(p, args, self_ty);
            self.contract_value_ok(t, false)
        })
    }

    /// E0211 for clause `cl`: the first violation of the operand rules, if any. `result` is the
    /// binding node of `result`; `olds` the token ranges of the `old(..)` arguments.
    fn clause_rules(&mut self, cl: &ContractClause, ensures: bool, result: u32, olds: &[(u32, u32)]) {
        let icx = self.icx;
        let open = icx.a(cl.node) + 1;
        let (lo, hi) = (open + 1, icx.close(open).saturating_sub(1));
        let mut bad: Option<u32> = None;
        let mut flag = |t: u32| {
            if bad.is_none_or(|b| t < b) {
                bad = Some(t);
            }
        };
        for t in lo..=hi {
            let k = icx.tk(t);
            let banned = [Term::KwUnsafe, Term::KwFn, Term::KwSpawn, Term::FmtLit, Term::Question]
                .iter()
                .any(|&x| k == x as u8);
            if banned || (k == Term::Amp as u8 && icx.is(t + 1, Term::KwMut)) {
                flag(t);
            }
        }
        let ts: Vec<(u32, Target)> =
            self.targets.iter().filter(|(t, _)| (lo..=hi).contains(t)).cloned().collect();
        for (t, tg) in ts {
            let ok = match &tg {
                Target::Global { decl } => !self.prog.decl(*decl).is_some_and(|d| d.kind == DeclKind::Static),
                Target::Intrinsic { .. } => false,
                Target::FnValue { ty } => {
                    let pure = Effects::IO.0 | Effects::LOCK.0 | Effects::BLOCK.0;
                    matches!(self.tys.kind(*ty), TypeKind::Fn { deny, .. } if deny & pure == pure)
                }
                Target::Fn { decl, args } if icx.is(t, Term::LParen) => {
                    self.contract_call_ok(*decl, args, None)
                }
                Target::Method { method, args, adjust } => match method {
                    MethodRef::Impl { method, .. } => self.contract_call_ok(*method, args, None),
                    MethodRef::Bound { method, self_ty, .. } => {
                        self.contract_call_ok(*method, args, Some(*self_ty))
                    }
                    MethodRef::Builtin(_) => adjust.autoref != Some(true) && !adjust.reborrow,
                },
                Target::Op { method, args } => match method {
                    MethodRef::Impl { method, .. } => self.contract_call_ok(*method, args, None),
                    MethodRef::Bound { method, self_ty, .. } => {
                        self.contract_call_ok(*method, args, Some(*self_ty))
                    }
                    MethodRef::Builtin(_) => true,
                },
                Target::TraitMethod { method, self_ty, args } => {
                    self.contract_call_ok(*method, args, Some(*self_ty))
                }
                Target::Builtin { adjust, .. } => adjust.autoref != Some(true) && !adjust.reborrow,
                _ => true,
            };
            if !ok {
                flag(if icx.is(t, Term::LParen) { t.saturating_sub(1) } else { t });
            }
        }
        // Where a value is consumed by value other than by a call (whose params are checked
        // above): a `match` scrutinee, an `if` / `match` value, an aggregate element, a variant
        // payload. A non-Copy value there may be a place moved out of a param, `result`, or a
        // reference (no borrow check runs on a clause), so it must be Copy or a shared reference.
        for m in icx.ast.descendants(cl.node) {
            let consumed: Vec<u32> = match icx.kind(m) {
                Some(NodeKind::MatchExpr) => icx.first_child(m).into_iter().chain([m]).collect(),
                Some(NodeKind::IfExpr) => vec![m],
                Some(NodeKind::StructLit) => {
                    icx.kids_vec(m).iter().copied().filter(|&e| icx.is_expr(e)).collect()
                }
                Some(NodeKind::Primary)
                    if icx.is(icx.a(m), Term::LParen) || icx.is(icx.a(m), Term::LBracket) =>
                {
                    icx.kids_vec(m).iter().copied().filter(|&e| icx.is_expr(e)).collect()
                }
                Some(NodeKind::Postfix) => {
                    let mut v = Vec::new();
                    for &k in icx.kids_vec(m).iter().skip(1) {
                        let lp = icx.a(k).saturating_sub(1);
                        if icx.is(lp, Term::LParen)
                            && matches!(
                                self.targets.iter().find(|x| x.0 == lp),
                                Some((_, Target::Variant { .. }))
                            )
                        {
                            v.extend(icx.kids_vec(k).iter().copied().filter(|&e| icx.is_expr(e)));
                        }
                    }
                    v
                }
                _ => continue,
            };
            for e in consumed {
                let Some(&t) = e.checked_sub(self.lo).and_then(|i| self.node_ty.get(i as usize)) else {
                    continue;
                };
                if let Ty::T(t) = self.resolve(t)
                    && !self.contract_value_ok(t, false)
                {
                    flag(icx.a(e));
                }
            }
        }
        let in_old = |t: u32| olds.iter().any(|&(a, b)| a <= t && t <= b);
        for u in icx.res.uses.iter() {
            if !(lo..=hi).contains(&u.token) {
                continue;
            }
            let (Res::Local(b) | Res::SelfValue(b)) = u.res else { continue };
            if b.node == result && ensures {
                if in_old(u.token) {
                    let d = mk("E0213", Some("entry"), &[], icx.tok_span(u.token), &[]);
                    self.report(d);
                }
                continue;
            }
            if !matches!(icx.kind(b.node), Some(NodeKind::Param | NodeKind::SelfParam)) {
                continue;
            }
            let Some(&t) = self.locals.get(&b.node) else { continue };
            let ok = if ensures {
                self.contract_value_ok(t, true)
            } else {
                !matches!(self.tys.kind(t), TypeKind::Ref { mutable: true, .. })
            };
            if !ok {
                flag(u.token);
            }
        }
        if let Some(t) = bad {
            let d = mk("E0211", None, &[], icx.tok_span(t), &[]);
            self.report(d);
        }
    }
}

/// E0211 case `impure-call`: an edge of a fn with contracts whose site is in a clause and whose
/// callee's effect set has `io`, `lock`, or `block` (one diagnostic per site and effect; the call
/// chain and the op as E1401).
pub fn purity(prog: &Program, tc: &Typeck, sets: &EffectSets) -> Vec<Diagnostic> {
    let g = &tc.call_graph;
    let impure = Effects(Effects::IO.0 | Effects::LOCK.0 | Effects::BLOCK.0);
    let mut out = Vec::new();
    let mut seen: FxHashSet<(u32, u32, u8)> = FxHashSet::default();
    for (i, nd) in g.nodes.iter().enumerate() {
        let GNode::Fn { inst, .. } = &nd.kind else { continue };
        let Some(c) = tc.body(inst.decl).and_then(|b| b.contracts.as_deref()) else { continue };
        for edge in g.succ(i as u32) {
            if edge.kind == EdgeKind::Member || !c.holds(edge.site) {
                continue;
            }
            let Some(set) = sets.set.get(edge.to as usize) else { continue };
            let have = Effects(set.0 & impure.0 & !edge.deny);
            for e in have.each() {
                if !seen.insert((edge.site.file.0, edge.site.offset, e.0)) {
                    continue;
                }
                let (chain, end) = sets.chain(g, edge.to, e).unwrap_or((Vec::new(), edge.to));
                let names = g.source_names(prog, tc, &[edge.to, end]);
                let (Some(callee), Some(at)) = (names.first(), names.get(1)) else { continue };
                let op = sets.op(end, e);
                let op_text = op.map_or(OpKind::Unseen.text(), |o| o.kind.text());
                let mut roles: Vec<(&str, Span)> = Vec::new();
                for &ei in &chain {
                    let x = &g.out[ei as usize];
                    roles.push((if x.kind == EdgeKind::Drop { "drop" } else { "call" }, x.site));
                }
                if let Some(o) = op
                    && o.span != g.nodes[end as usize].span
                {
                    roles.push(("op", o.span));
                }
                let effect = e.names()[0];
                let holes =
                    [("fn", callee.as_str()), ("effect", effect), ("at", at.as_str()), ("op", op_text)];
                out.extend(mk("E0211", Some("impure-call"), &holes, edge.site, &roles));
            }
        }
    }
    out
}
