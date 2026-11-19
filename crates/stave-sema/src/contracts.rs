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
//! is a shared reference or a Copy type, so a clause neither writes nor moves. A clause may read
//! through a parameter of type `&mut T` (never write or pass it on as `&mut`: the call rule). An
//! `ensures` runs after the body, which may have moved a by-value parameter, so there a parameter
//! must be a reference or Copy. E0212 / E0213 (case `param`): a parameter named `result` / `old`
//! in a fn with an `ensures`. E0213: `old(e)` with a non-Copy `e`, or `result` inside it (case
//! `entry`). E0211 case `trait-impl`: contracts on a trait impl method.
//!
//! **Shared state** (E0211 cases `frozen` / `frozen-self`, [`Ck::clause_frozen`]; once the
//! operand rules pass): every value a clause touches has a `Frozen` type, so no call in it can
//! change anything through a shared reference (an atomic, `Cell`, `Mutex`, or a fn value's
//! captures); this bans `load()` too. Case `static` ([`shared_statics`]): no callee reaches a
//! `static` whose type is not Frozen.
//!
//! **Purity** ([`purity`], E0211 case `impure-call`): a callee (or a drop) in a clause whose effect
//! set ([`crate::effects`]) has `io`, `lock`, or `block`. Calls in clauses are call-graph edges of
//! the fn (the check runs in the callee), and a clause precedes the body, so the edge of a callee
//! called there is at the clause's site.
//!
//! **Checks** (lowering): the `requires` clauses at entry in order (on a `test fn` they never
//! panic: with parameters they filter its inputs, [`crate::testgen`]; without, they are not
//! evaluated), then the `old(e)` snapshots, then the `ensures` clauses in order at each return;
//! each operand of a clause's top-level `&&` is its own check. A false one calls
//! `runtime.panic.requires_failed` (located at the caller's user site, which every call that can
//! reach a `requires` stores first) or `runtime.panic.ensures_failed` (at the return). The removal
//! of proven checks (stave-ir `bounds`) deletes the checks the facts prove and reports those that
//! always fail, and sends a call that proves every `requires` of its callee past the callee's
//! checks. A kept `ensures` check is a trapping instruction of the fn's IR (`trap`); a kept
//! `requires` check is the fn's entry effect, which only the calls that do not prove it carry
//! ([`crate::effects`]).

use rustc_hash::{FxHashMap, FxHashSet};
use stave_diag::{Diagnostic, Span};
use stave_syntax::{NodeKind, Term};

use crate::Program;
use crate::callgraph::{EdgeKind, NodeKind as GNode};
use crate::decls::{DeclFlags, DeclKind};
use crate::diag::mk;
use crate::effects::{EffectSets, Effects, OpKind};
use crate::ids::DeclId;
use crate::resolve::Res;
use crate::traits::{Derived, MethodRef, NotFlat};
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
            if !self.clause_rules(cl, false, c.result, &olds) {
                self.clause_frozen(cl);
            }
        }
        for cl in &c.ensures {
            if !self.clause_rules(cl, true, c.result, &olds) {
                self.clause_frozen(cl);
            }
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

    /// E0211 case `frozen` / `frozen-self` for clause `cl`: the first value the clause touches
    /// whose type is not Frozen, by source position (the innermost expression first). A clause
    /// touches every expression in it (each param, `self`, `result` it names, each operand,
    /// receiver, argument, place, and value inside `old(..)`), each call's result and built-in
    /// method receiver (a step inside a chain such as `a.f().g()` is no expression), and every fn
    /// value it calls. A field reached from a touched value is part of it.
    fn clause_frozen(&mut self, cl: &ContractClause) {
        let icx = self.icx;
        let open = icx.a(cl.node) + 1;
        let (lo, hi) = (open + 1, icx.close(open).saturating_sub(1));
        // (first token, span length, text, type)
        let mut touched: Vec<(u32, u32, String, TypeId)> = Vec::new();
        for e in icx.ast.descendants(cl.node) {
            if !icx.is_expr(e) {
                continue;
            }
            let Some(&t) = e.checked_sub(self.lo).and_then(|i| self.node_ty.get(i as usize)) else {
                continue;
            };
            let Ty::T(t) = self.resolve(t) else { continue };
            touched.push((icx.a(e), icx.node_span(e).len, clause_line(icx.node_text(e)), t));
        }
        let ts: Vec<(u32, Target)> =
            self.targets.iter().filter(|(t, _)| (lo..=hi).contains(t)).cloned().collect();
        for (t, tg) in ts {
            let name = icx.tok_str(if icx.is(t, Term::LParen) { t.saturating_sub(1) } else { t });
            let call = format!("{name}(..)");
            let (decl, args, self_ty) = match &tg {
                Target::FnValue { ty } => {
                    touched.push((t, 0, call, *ty));
                    continue;
                }
                Target::Builtin { recv, ret, .. } => {
                    touched.push((t, 0, format!("receiver of {call}"), *recv));
                    touched.push((t, 0, call, *ret));
                    continue;
                }
                Target::Fn { decl, args } if icx.is(t, Term::LParen) => (*decl, args.clone(), None),
                Target::Method { method: MethodRef::Impl { method, .. }, args, .. }
                | Target::Op { method: MethodRef::Impl { method, .. }, args } => {
                    (*method, args.clone(), None)
                }
                Target::Method { method: MethodRef::Bound { method, self_ty, .. }, args, .. }
                | Target::Op { method: MethodRef::Bound { method, self_ty, .. }, args } => {
                    (*method, args.clone(), Some(*self_ty))
                }
                Target::TraitMethod { method, self_ty, args } => (*method, args.clone(), Some(*self_ty)),
                _ => continue,
            };
            let Some(sig) = self.sigs.fn_sig(decl) else { continue };
            let ret = sig.ret;
            let r = self.tys.subst(ret, &args, self_ty);
            touched.push((t, 0, call, r));
        }
        touched.sort_by_key(|x| (x.0, x.1));
        let env = self.genv();
        for (tok, _, text, t) in touched {
            if t.is_none() || t == ids::ERROR || self.tys.has_error(t) {
                continue;
            }
            let d = Derived::Frozen;
            let Some(wit) = self.w.derived_witness(&mut self.tys, t, env, d) else { continue };
            let tn = self.show(t);
            let at = icx.tok_span(tok);
            let diag = match wit.why {
                NotFlat::Unknown => return,
                NotFlat::Depth => {
                    let r = "nested deeper than the limit of the `Frozen` check";
                    mk(
                        "E0211",
                        Some("frozen-self"),
                        &[("value", &text), ("type", &tn), ("reason", r)],
                        at,
                        &[],
                    )
                }
                why if wit.path.is_empty() => {
                    let r = why.text_for(d);
                    mk(
                        "E0211",
                        Some("frozen-self"),
                        &[("value", &text), ("type", &tn), ("reason", r)],
                        at,
                        &[],
                    )
                }
                why => {
                    let pn = self.show(wit.part);
                    let r = why.text_for(d);
                    let holes = [
                        ("value", text.as_str()),
                        ("type", &tn),
                        ("path", &wit.path),
                        ("part", &pn),
                        ("reason", r),
                    ];
                    mk("E0211", Some("frozen"), &holes, at, &[])
                }
            };
            self.report(diag);
            return;
        }
    }

    /// E0211 for clause `cl`: the first violation of the operand rules, if any. `result` is the
    /// binding node of `result`; `olds` the token ranges of the `old(..)` arguments.
    fn clause_rules(&mut self, cl: &ContractClause, ensures: bool, result: u32, olds: &[(u32, u32)]) -> bool {
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
        // The params a `&mut` inside the result borrows from (`from v`, `from copy v`): an
        // `ensures` reading one would alias that live `&mut` (planned noalias), except inside
        // `old(..)`, which reads at entry.
        let mut_from: Vec<u32> = if ensures { self.mut_result_params() } else { Vec::new() };
        let mut mut_read: Option<(u32, u32)> = None;
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
            // A `requires` runs before the body: every param is still there to be read (a
            // `&mut` one too: writing or passing it on as `&mut` fails the call rule above).
            if !ensures || !matches!(icx.kind(b.node), Some(NodeKind::Param | NodeKind::SelfParam)) {
                continue;
            }
            if mut_from.contains(&b.node) && !in_old(u.token) && mut_read.is_none_or(|(t, _)| u.token < t) {
                mut_read = Some((u.token, b.node));
            }
            let Some(&t) = self.locals.get(&b.node) else { continue };
            if !self.contract_value_ok(t, true) {
                flag(u.token);
            }
        }
        if let Some(t) = bad {
            let d = mk("E0211", None, &[], icx.tok_span(t), &[]);
            self.report(d);
            return true;
        }
        if let Some((t, p)) = mut_read {
            let name = icx.tok_str(t).to_string();
            let d = mk(
                "E0211",
                Some("mut-result"),
                &[("name", &name)],
                icx.tok_span(t),
                &[("param", icx.node_span(p))],
            );
            self.report(d);
            return true;
        }
        false
    }

    /// The param nodes of the checked fn that a `&mut` inside its return type may borrow from.
    fn mut_result_params(&mut self) -> Vec<u32> {
        let Some(sig) = self.sigs.fn_sig(self.decl) else { return Vec::new() };
        let (origins, ret, node) = (sig.origins.clone(), sig.ret, sig.node);
        if origins.is_empty() || !self.has_mut_ref(ret) {
            return Vec::new();
        }
        let icx = self.icx;
        let Some(ps) = icx.child(node, NodeKind::Params) else { return Vec::new() };
        let params = icx.kids_vec(ps);
        // a param binds at its Param node, `self` at the SelfParam node inside it
        origins
            .iter()
            .filter_map(|o| {
                params.get(o.param as usize).map(|&p| icx.child(p, NodeKind::SelfParam).unwrap_or(p))
            })
            .collect()
    }

    /// Whether a `&mut` is reachable in type `t` (through references, elements, fields, and type
    /// arguments; a recursive type is visited once).
    fn has_mut_ref(&mut self, t: TypeId) -> bool {
        let mut seen: FxHashSet<TypeId> = FxHashSet::default();
        let mut work = vec![t];
        while let Some(t) = work.pop() {
            if !seen.insert(t) || seen.len() > 4096 {
                continue;
            }
            match self.tys.kind(t).clone() {
                TypeKind::Ref { mutable: true, .. } => return true,
                TypeKind::Adt { def, args } => {
                    let fields: Vec<TypeId> = self
                        .sigs
                        .adt(def)
                        .map(|a| a.variants.iter().flat_map(|v| v.fields.iter().map(|f| f.ty)).collect())
                        .unwrap_or_default();
                    for f in fields {
                        work.push(self.tys.subst(f, &args, None));
                    }
                    work.extend(
                        args.iter().filter_map(|a| if let GArg::Type(x) = a { Some(*x) } else { None }),
                    );
                }
                k => {
                    let mut kids = Vec::new();
                    k.children(&mut kids);
                    work.extend(kids);
                }
            }
        }
        false
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

/// E0211 case `static`: a callee in a clause that reaches, through calls and drops, a body that
/// names a `static` whose type is not Frozen (an atomic, `Mutex`, `Once`, ...): shared state the
/// clause could change. A body counts with every static it names, its closures' included.
/// `static mut` (built-in packages only, reached through `unsafe`) does not count.
pub fn shared_statics(prog: &Program, tc: &Typeck) -> Vec<Diagnostic> {
    let g = &tc.call_graph;
    let frozen = tc.sigs.lang.frozen;
    if frozen.is_none() {
        return Vec::new();
    }
    let w = crate::traits::World { prog, sigs: &tc.sigs };
    let mut cx = crate::types::Overlay::new(&tc.types);
    let tr = crate::sigs::TraitRef { trait_decl: frozen, args: Vec::new() };
    let mut verdict: FxHashMap<DeclId, bool> = FxHashMap::default();
    let mut shared = |d: DeclId| -> bool {
        if let Some(&v) = verdict.get(&d) {
            return v;
        }
        let v = prog.decl(d).is_some_and(|x| x.kind == DeclKind::Static && !x.flags.has(DeclFlags::MUT))
            && tc.sigs.konst(d).is_some_and(|c| !w.implements(&mut cx, c.ty, &tr, &[]));
        verdict.insert(d, v);
        v
    };
    // The decl whose body holds node `n`'s code (a closure's: its enclosing fn's).
    let body_of = |mut n: u32| -> Option<DeclId> {
        for _ in 0..g.nodes.len() {
            match &g.nodes[n as usize].kind {
                GNode::Fn { inst, .. } => return Some(inst.decl),
                GNode::Closure { parent, .. } => n = *parent,
                _ => return None,
            }
        }
        None
    };
    // Per node: the first shared static its body names.
    let mut uses: Vec<Option<DeclId>> = vec![None; g.nodes.len()];
    for (i, u) in uses.iter_mut().enumerate() {
        let Some(b) = body_of(i as u32).and_then(|d| tc.body(d)) else { continue };
        *u = b.targets.iter().find_map(|(_, t)| match t {
            Target::Global { decl } if shared(*decl) => Some(*decl),
            _ => None,
        });
    }
    // Nodes that reach such a use: backwards from the users.
    let mut reach = vec![false; g.nodes.len()];
    let mut work: Vec<u32> = (0..g.nodes.len() as u32).filter(|&n| uses[n as usize].is_some()).collect();
    for &n in &work {
        reach[n as usize] = true;
    }
    while let Some(n) = work.pop() {
        for (c, _) in g.pred(n) {
            if !reach[c as usize] {
                reach[c as usize] = true;
                work.push(c);
            }
        }
    }
    let mut out = Vec::new();
    let mut seen: FxHashSet<(u32, u32)> = FxHashSet::default();
    for (i, nd) in g.nodes.iter().enumerate() {
        let GNode::Fn { inst, .. } = &nd.kind else { continue };
        let Some(c) = tc.body(inst.decl).and_then(|b| b.contracts.as_deref()) else { continue };
        for edge in g.succ(i as u32) {
            if edge.kind == EdgeKind::Member || !c.holds(edge.site) || !reach[edge.to as usize] {
                continue;
            }
            if !seen.insert((edge.site.file.0, edge.site.offset)) {
                continue;
            }
            // a shortest chain to a user, through reaching nodes only
            let mut parent: FxHashMap<u32, Option<u32>> = FxHashMap::default();
            parent.insert(edge.to, None);
            let mut queue = std::collections::VecDeque::from([edge.to]);
            let mut end = None;
            while let Some(m) = queue.pop_front() {
                if uses[m as usize].is_some() {
                    end = Some(m);
                    break;
                }
                let base = g.out_start[m as usize];
                for (k, e) in g.succ(m).iter().enumerate() {
                    if reach[e.to as usize]
                        && let std::collections::hash_map::Entry::Vacant(v) = parent.entry(e.to)
                    {
                        v.insert(Some(base + k as u32));
                        queue.push_back(e.to);
                    }
                }
            }
            let Some(end) = end else { continue };
            let (Some(st), Some(sd)) = (uses[end as usize], uses[end as usize].and_then(|d| prog.decl(d)))
            else {
                continue;
            };
            let mut chain = Vec::new();
            let mut cur = end;
            while let Some(&Some(ei)) = parent.get(&cur) {
                chain.push(ei);
                cur = g.caller(ei);
            }
            chain.reverse();
            let names = g.source_names(prog, tc, &[edge.to, end]);
            let (Some(callee), Some(at)) = (names.first(), names.get(1)) else { continue };
            let sname = prog.decl_path(st);
            let nx = crate::sigs::NameCx { prog, generics: &[] };
            let tn =
                tc.sigs.konst(st).map_or_else(String::new, |k| crate::types::display(&tc.types, &nx, k.ty));
            let mut roles: Vec<(&str, Span)> = Vec::new();
            for &ei in &chain {
                let x = &g.out[ei as usize];
                roles.push((if x.kind == EdgeKind::Drop { "drop" } else { "call" }, x.site));
            }
            roles.push(("static", sd.name_span));
            let holes = [("fn", callee.as_str()), ("at", at.as_str()), ("name", &sname), ("type", &tn)];
            out.extend(mk("E0211", Some("static"), &holes, edge.site, &roles));
        }
    }
    out
}
