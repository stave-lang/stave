//! Statement lowering: blocks and scopes (drops), `let`, assignments, `if`,
//! `while`, `loop`, `for` (ranges, arrays and slices, iterators), `match` (statement and value
//! form, one `switch` on the discriminant when the arms test one, or-patterns: a test per
//! alternative into the arm's one guard and body), `return`/`break`/`continue`,
//! patterns, closures, the fn body driver, drop glue, and the body pre-pass (locals, slots,
//! assignment sites, closures).

use stave_ir::{Block, FuncBuilder, Op, Operand, Value, VecOp};
use stave_syntax::{NodeKind, Term};

use crate::decls::DeclKind;
use crate::lower_fn::{
    Base, E, Env, FnCx, FnOut, FnState, LocalInfo, LoopCx, LoopKind, MState, Place, Proj, Ty, WriteBack,
};
use crate::mono::{InstKey, Req, Resolved};
use crate::resolve::Res;
use crate::typeck::Target;
use crate::types::{ConstArg, GArg, Interner, TypeId, TypeKind};

/// A closure environment: (tuple type, `&mut` tuple type, per capture (local, `&mut T` type)).
pub(crate) type EnvType = (Ty, Ty, Vec<(usize, Ty)>);

/// How a pattern binds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Move,
    Ref(bool),
}

/// The top-level test of a match arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Top {
    Value(i128),
    Any,
    Complex,
}

impl<'a> FnCx<'a> {
    // ---------- pre-pass ----------

    /// The local a place expression is rooted at (no dereference on the way).
    pub fn place_root(&self, n: u32) -> Option<usize> {
        let icx = self.icx;
        let c = n;
        match icx.kind(c)? {
            NodeKind::Unary => {
                if icx.is(icx.a(c), Term::KwUnsafe) {
                    return self.place_root(icx.last_child(c)?);
                }
                None
            }
            NodeKind::Postfix => self.chain_root(c, None),
            NodeKind::Primary => {
                let a = icx.a(c);
                if icx.is(a, Term::Ident) {
                    // a bare name: a chain without steps
                    return self.chain_root(c, None);
                }
                if icx.is(a, Term::KwSelf) {
                    return icx.uses(c).first().and_then(|u| self.local_of(u.res));
                }
                if icx.is(a, Term::LParen) {
                    let kids = icx.kids_vec(c);
                    let close = icx.close(a);
                    if kids.len() == 1 && !icx.is(close - 1, Term::Comma) {
                        return self.place_root(kids[0]);
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Walks a postfix chain's steps: the root local of the place it denotes; method calls whose
    /// receiver is auto-referenced at a local root, and sub-slices of a local root, are reported
    /// to `autoref`.
    fn chain_root(&self, n: u32, mut autoref: Option<&mut Vec<usize>>) -> Option<usize> {
        let icx = self.icx;
        let prim = icx.chain_head(n)?;
        let pa = icx.a(prim);
        let mut root = if icx.is(pa, Term::Ident) {
            icx.use_at(prim, pa).and_then(|u| self.local_of(u.res))
        } else if icx.is(pa, Term::KwSelf) {
            icx.uses(prim).first().and_then(|u| self.local_of(u.res))
        } else if icx.is(pa, Term::LParen) {
            self.place_root(prim)
        } else {
            None
        };
        let mut p = icx.primary_end(prim) + 1;
        for _ in 0..10_000_000 {
            let t = icx.tk(p);
            if t == Term::Dot as u8 && icx.is(p + 1, Term::Ident) {
                let seg = p + 1;
                let mut q = p + 2;
                if icx.is(q, Term::ColonColon) {
                    let (e, partial) = icx.angle_end(q + 1);
                    q = if partial { e - 1 } else { e };
                }
                if icx.use_at(prim, seg).is_some() {
                    root = None;
                    p = q;
                    continue;
                }
                if icx.is(q, Term::LParen) {
                    if let Some(Target::Method { adjust, .. }) = self.target(seg)
                        && adjust.autoref.is_some()
                        && adjust.derefs == 0
                        && let (Some(r), Some(out)) = (root, autoref.as_deref_mut())
                    {
                        out.push(r);
                    }
                    root = None;
                    p = icx.close(q) + 1;
                    continue;
                }
                match self.target(seg) {
                    Some(Target::Field { derefs, .. }) if *derefs == 0 => {}
                    _ => root = None,
                }
                p = q;
                continue;
            }
            if t == Term::Dot as u8 && icx.is(p + 1, Term::IntLit) {
                match self.target(p + 1) {
                    Some(Target::TupleField { derefs, .. }) if *derefs == 0 => {}
                    _ => root = None,
                }
                p += 2;
                continue;
            }
            if t == Term::LBracket as u8 {
                match self.target(p) {
                    Some(Target::Index { derefs }) if *derefs == 0 => {}
                    // a sub-slice is formed from the root's address
                    Some(Target::Slice { derefs }) if *derefs == 0 => {
                        if let (Some(r), Some(out)) = (root, autoref.as_deref_mut()) {
                            out.push(r);
                        }
                    }
                    _ => root = None,
                }
                p = icx.close(p) + 1;
                continue;
            }
            if t == Term::LParen as u8 || t == Term::DotLBrace as u8 {
                root = None;
                p = icx.close(p) + 1;
                continue;
            }
            if t == Term::Question as u8 {
                root = None;
                p += 1;
                continue;
            }
            break;
        }
        root
    }

    /// The innermost closure containing node `n`.
    fn owner(&self, n: u32) -> Option<u32> {
        let icx = self.icx;
        let mut best: Option<(u32, u32)> = None;
        for &c in &self.closure_nodes {
            let r = icx.ast.descendants(c);
            if n >= r.start && n < c {
                let size = c - r.start;
                if best.is_none_or(|(_, s)| size < s) {
                    best = Some((c, size));
                }
            }
        }
        best.map(|b| b.0)
    }

    /// Locals, slots, assignment sites, and closures of the body at `root`.
    pub fn setup(&mut self, root: u32) {
        let icx = self.icx;
        let tb = self.tb();
        let mut binds: Vec<(u32, u32, Ty)> = Vec::new();
        for &(node, t) in &tb.locals {
            if icx.kind(node) == Some(NodeKind::SelfParam) {
                continue;
            }
            let t = self.sub(t);
            binds.push((icx.a(node), node, t));
        }
        binds.sort_unstable();
        for (_, node, ty) in binds {
            let drop = self.needs_drop(ty);
            let idx = self.locals.len();
            self.locals.push(LocalInfo { node, ty, mem: drop, drop, flag: None });
            self.by_node.insert(node, idx);
            if icx.kind(node) == Some(NodeKind::Param)
                && let Some(sp) = icx.child(node, NodeKind::SelfParam)
            {
                self.by_node.insert(sp, idx);
            }
        }
        // a later alternative of an or-pattern binds the first alternative's locals
        for &(alt, first) in &icx.res.aliases {
            if let Some(&l) = self.by_node.get(&first) {
                self.by_node.insert(alt, l);
            }
        }
        let lo = icx.ast.descendants(root).start;
        let mut autoref: Vec<usize> = Vec::new();
        let mut assigns: Vec<(u32, usize)> = Vec::new();
        // a postfix chain adds to `autoref` only at a method call with an auto-referenced
        // receiver or at a sub-slice of the root: without one in the body the chains (most nodes
        // of a body) are not walked
        let autoref_calls = tb.targets.iter().any(|(_, t)| {
            matches!(t, Target::Method { adjust, .. } if adjust.autoref.is_some() && adjust.derefs == 0)
                || matches!(t, Target::Slice { derefs: 0 })
        });
        // the three kinds of interest are picked by their raw ids
        let hi = (root as usize + 1).min(icx.ast.nodes.len());
        let nodes = &icx.ast.nodes[(lo as usize).min(hi)..hi];
        let (unary, postfix, expr_stmt) =
            (NodeKind::Unary as u8, NodeKind::Postfix as u8, NodeKind::ExprStmt as u8);
        for (i, node) in nodes.iter().enumerate() {
            if node.kind != unary && node.kind != postfix && node.kind != expr_stmt {
                continue;
            }
            let m = lo + i as u32;
            match icx.kind(m) {
                Some(NodeKind::Unary) => {
                    let a = icx.a(m);
                    if (icx.is(a, Term::Amp) || icx.is(a, Term::AmpAmp))
                        && let Some(c) = icx.last_child(m)
                        && let Some(l) = self.place_root(c)
                    {
                        autoref.push(l);
                    }
                }
                Some(NodeKind::Postfix) if autoref_calls => {
                    let _ = self.chain_root(m, Some(&mut autoref));
                }
                Some(NodeKind::ExprStmt) => {
                    let kids = icx.kids_vec(m);
                    if kids.len() >= 3
                        && let Some(l) = self.place_root(kids[0])
                    {
                        assigns.push((m, l));
                    }
                }
                _ => {}
            }
        }
        for l in autoref {
            if let Some(li) = self.locals.get_mut(l) {
                li.mem = true;
            }
        }
        assigns.sort_unstable();
        self.assigns = assigns;
        // closure bodies and their captures: a captured local lives in a slot (the environment
        // holds its address)
        let tys = &self.tys;
        let infos = crate::closures::closures(&icx, tb, |t| crate::closures::is_mut_ref(tys, t), root);
        for c in &infos {
            for cap in &c.captures {
                if let Some(&l) = self.by_node.get(&cap.binding) {
                    self.locals[l].mem = true;
                }
            }
        }
        self.closure_nodes = infos.iter().map(|c| c.node).collect();
        self.closure_info = infos;
        // dynamic drop flags
        let n = self.locals.len();
        for l in 0..n {
            if self.locals[l].drop && self.flagged.contains(&self.locals[l].node) {
                let fl = self.locals.len();
                let bt = self.t_bool();
                let node = self.locals[l].node;
                self.locals.push(LocalInfo { node, ty: bt, mem: false, drop: false, flag: None });
                self.locals[l].flag = Some(fl);
            }
        }
    }

    fn fresh_env(&mut self, params: &[Ty], ret: Ty) {
        self.f = FnState::new(params, ret);
        let n = self.locals.len();
        self.f.slot = vec![None; n];
        self.f.env = Env { vals: vec![None; n], st: vec![MState::Uninit; n] };
    }

    /// Slots of the mem locals owned by `owner` (`None`: the fn itself), in local order.
    fn make_slots(&mut self, owner: Option<u32>) {
        for l in 0..self.locals.len() {
            let li = self.locals[l].clone();
            if li.mem && self.owner(li.node) == owner {
                let p = self.new_slot(li.ty);
                self.f.slot[l] = Some(p);
            }
        }
    }

    fn bind_params(&mut self, params_node: Option<u32>) {
        let Some(ps) = params_node else { return };
        let kids = self.icx.kids_vec(ps);
        for (i, &p) in kids.iter().enumerate() {
            let Some(&l) = self.by_node.get(&p) else { continue };
            let v = self.f.b.param(i).map(Operand::Value);
            self.bind(l, v);
        }
    }

    fn finish_fn(&mut self) -> Option<stave_ir::Body> {
        let mut b = std::mem::replace(&mut self.f.b, FuncBuilder::new(&[]));
        b.seal_unreached();
        let borrows = std::mem::take(&mut self.f.borrows);
        match b.finish() {
            Ok(mut body) => {
                body.borrows = borrows;
                let mut body = crate::lower_post::finalize(body);
                stave_ir::fence::place_stream_fences(&mut body);
                Some(body)
            }
            Err(e) => {
                self.fail_at(None, format!("internal: {e}"));
                None
            }
        }
    }

    fn end_fn(&mut self) {
        if self.open() {
            while !self.f.scopes.is_empty() {
                self.pop_scope();
            }
            let r = self.f.ret;
            if self.open() {
                if self.ret_of(r).is_none() && !self.is_never(r) {
                    if let Some(at) = self.f.ensures_end {
                        self.ensures_checks(None, at);
                    }
                    self.f.b.ret(None);
                } else {
                    self.f.b.unreachable();
                }
            }
        }
    }

    /// Lowers the fn body at `fn_node` (a `FnDecl`, `ExternItem` export, or `KernelDecl`).
    pub fn lower_body(&mut self, fn_node: u32, params: &[Ty], ret: Ty) -> FnOut {
        self.result_local(ret);
        self.fresh_env(params, ret);
        self.make_slots(None);
        self.push_scope();
        let icx = self.icx;
        self.bind_params(icx.child(fn_node, NodeKind::Params));
        let block = icx.child(fn_node, NodeKind::Block);
        self.contract_entry(block.map(|b| icx.close(icx.a(b))));
        if let Some(b) = block {
            self.block(b);
        }
        self.end_fn();
        let site_consts = std::mem::take(&mut self.f.site_consts);
        let body = self.finish_fn();
        FnOut {
            closure: None,
            params: params.to_vec(),
            ret: self.ret_of(ret),
            body,
            site_consts,
            name_off: None,
        }
    }

    // ---------- contracts ----------

    fn contracts(&self) -> Option<&'a crate::contracts::Contracts> {
        self.tb().contracts.as_deref()
    }

    /// The local `result` names in the `ensures` clauses (returned type `ret`), never dropped (the
    /// returned value owns its contents): an SSA value when it is an int or `bool` no clause
    /// takes the address of (so the checks compare the returned value itself), else in a slot
    /// unless it holds no data.
    fn result_local(&mut self, ret: Ty) {
        let Some(c) = self.contracts() else { return };
        if c.ensures.is_empty() {
            return;
        }
        let idx = self.locals.len();
        self.locals.push(LocalInfo { node: c.result, ty: ret, mem: false, drop: false, flag: None });
        self.by_node.insert(c.result, idx);
        self.result_local = Some(idx);
        let scalar = matches!(self.tys.kind(ret), TypeKind::Int { .. } | TypeKind::Bool);
        let clauses: Vec<u32> = c.ensures.iter().map(|cl| cl.node).collect();
        let mem = if scalar {
            self.address_taken(&clauses, idx)
        } else {
            !self.is_unit(ret) && !self.is_never(ret)
        };
        self.locals[idx].mem = mem;
    }

    /// Whether an expression under one of the nodes `roots` borrows local `l` (`&l`, or a method
    /// call or sub-slice that auto-references it), as [`Self::setup`] finds them in a body.
    fn address_taken(&self, roots: &[u32], l: usize) -> bool {
        let icx = self.icx;
        let mut autoref: Vec<usize> = Vec::new();
        for &root in roots {
            for m in icx.ast.descendants(root).start..=root {
                match icx.kind(m) {
                    Some(NodeKind::Unary) => {
                        let a = icx.a(m);
                        if (icx.is(a, Term::Amp) || icx.is(a, Term::AmpAmp))
                            && let Some(c) = icx.last_child(m)
                            && let Some(x) = self.place_root(c)
                        {
                            autoref.push(x);
                        }
                    }
                    Some(NodeKind::Postfix) => {
                        let _ = self.chain_root(m, Some(&mut autoref));
                    }
                    _ => {}
                }
            }
        }
        autoref.contains(&l)
    }

    /// At entry, after the params are bound: the `requires` checks in order, then the `old(e)`
    /// snapshots. `end`: the body's closing `}` (where a unit body ends).
    fn contract_entry(&mut self, end: Option<u32>) {
        if self.tb().test_domain.is_some() {
            self.test_entry();
        }
        let Some(c) = self.contracts() else { return };
        let checked = c.checked_requires();
        let calls = checked.iter().any(|cl| self.clause_calls(cl.expr))
            || c.olds.iter().any(|&(_, e)| self.clause_calls(e));
        let trace = if calls { self.trace_save() } else { None };
        for cl in checked {
            self.contract_check(cl, None, trace);
        }
        for &(paren, e) in &c.olds {
            if !self.open() {
                return;
            }
            let t = self.nty(e);
            let Some(v) = self.val(e, Some(t)) else { return };
            self.drop_temps();
            self.old_vals.insert(paren, (v, t));
        }
        if let Some(sv) = trace {
            self.trace_restore(sv);
        }
        if !c.ensures.is_empty() {
            self.f.ensures_end = end;
            if end.is_none() {
                self.fail_at(None, "`ensures` on a fn without a body");
            }
        }
    }

    /// The entry of a `test fn` with parameters: the input marker, then its `requires` clauses
    /// in order as filters. The runner is the caller: an input a clause rejects writes the skip
    /// marker and returns at once (every parameter type the runner generates is Copy, so there
    /// is nothing to drop); it never panics `Contract`.
    fn test_entry(&mut self) {
        let (Some(input), Some(skip)) = (self.g.lang.test_input, self.g.lang.test_skip) else {
            self.fail_at(None, "`runtime.panic` has no `test fn` input markers");
            return;
        };
        let fi = self.req(Req::Inst(InstKey { decl: input, args: Vec::new() }));
        self.f.last_site = None;
        self.f.b.call(fi, None, &[]);
        let Some(c) = self.contracts() else { return };
        let trace =
            if c.requires.iter().any(|cl| self.clause_calls(cl.expr)) { self.trace_save() } else { None };
        for cl in &c.requires {
            if !self.open() {
                return;
            }
            let saved = std::mem::take(&mut self.f.temps);
            let bt = self.t_bool();
            self.hold_site = true;
            let v = self.val(cl.expr, Some(bt));
            self.drop_temps();
            self.hold_site = false;
            self.f.temps = saved;
            let Some(v) = v else { return };
            if !self.open() {
                return;
            }
            let (no, ok) = (self.f.b.create_block(), self.f.b.create_block());
            self.f.b.cbr(v, ok, &[], no, &[]);
            self.f.b.switch_to(no);
            let fs = self.req(Req::Inst(InstKey { decl: skip, args: Vec::new() }));
            self.f.last_site = None;
            self.f.b.call(fs, None, &[]);
            self.f.b.ret(None);
            self.f.b.switch_to(ok);
        }
        if let Some(sv) = trace {
            self.trace_restore(sv);
        }
    }

    /// The `ensures` checks of a return of `v` (`None`: unit) at token `at`, after the drops of
    /// the scopes it leaves.
    pub fn ensures_checks(&mut self, v: Option<Operand>, at: u32) {
        if self.f.ensures_end.is_none() || !self.open() {
            return;
        }
        let Some(c) = self.contracts() else { return };
        if let Some(l) = self.result_local {
            let v = match v {
                Some(v) => v,
                None => self.unit_val(),
            };
            self.bind(l, Some(v));
        }
        let trace =
            if c.ensures.iter().any(|cl| self.clause_calls(cl.expr)) { self.trace_save() } else { None };
        for cl in &c.ensures {
            self.contract_check(cl, Some(at), trace);
        }
        if let Some(sv) = trace {
            self.trace_restore(sv);
        }
    }

    /// Whether the clause expression `e` calls anything or makes a value of an enum variant: only
    /// then can its evaluation change the error trace (`Result.Err(..)` restarts it, `?` in a
    /// callee extends it).
    fn clause_calls(&self, e: u32) -> bool {
        let icx = self.icx;
        let (lo, hi) = (icx.a(e), icx.end_tok(e));
        let tb = self.tb();
        let from = tb.targets.partition_point(|(t, _)| *t < lo);
        tb.targets[from..].iter().take_while(|(t, _)| *t <= hi).any(|(_, t)| {
            matches!(
                t,
                Target::Fn { .. }
                    | Target::Method { .. }
                    | Target::TraitMethod { .. }
                    | Target::Op { .. }
                    | Target::FnValue { .. }
                    | Target::Variant { .. }
            )
        })
    }

    /// Saves the error trace into a fresh slot (`runtime.errtrace.save`): the slot's pointer and
    /// type, `None` without the runtime fns.
    fn trace_save(&mut self) -> Option<(Operand, Ty)> {
        let (save, restore) = (self.g.lang.errtrace_save?, self.g.lang.errtrace_restore?);
        if !self.open() {
            return None;
        }
        let (_, trace) = self.inst_sig(&InstKey { decl: save, args: Vec::new() })?;
        self.inst_sig(&InstKey { decl: restore, args: Vec::new() })?;
        let f = self.req(Req::Inst(InstKey { decl: save, args: Vec::new() }));
        let v = self.f.b.call(f, Some(Self::ir(trace)), &[])?;
        let (p, _) = self.spill(Operand::Value(v), trace, false);
        Some((p, trace))
    }

    /// Puts back the error trace [`FnCx::trace_save`] saved.
    fn trace_restore(&mut self, (p, trace): (Operand, Ty)) {
        let Some(restore) = self.g.lang.errtrace_restore else { return };
        if !self.open() {
            return;
        }
        let Some((ps, _)) = self.inst_sig(&InstKey { decl: restore, args: Vec::new() }) else { return };
        let Some(&want) = ps.first() else { return };
        let from = self.t_ref(true, trace);
        let a = self.convert_ptr(p, from, want);
        let f = self.req(Req::Inst(InstKey { decl: restore, args: Vec::new() }));
        self.f.b.call(f, None, &[a]);
    }

    /// Branches to the failure call of clause `cl` when it is false: a `requires` (`at` `None`)
    /// or an `ensures` reported at token `at`. Each operand of the clause's top-level `&&` is its
    /// own check, in order (a false one fails before the next is evaluated, as `&&` would); each
    /// reports the whole clause. A check's temporaries are dropped before its branch; those of the
    /// enclosing statement are left as they are. `trace`: the error trace saved before the
    /// checks, put back before a failure is reported.
    fn contract_check(
        &mut self,
        cl: &crate::contracts::ContractClause,
        at: Option<u32>,
        trace: Option<(Operand, Ty)>,
    ) {
        let conjuncts = if self.nkind(cl.expr) == Some(NodeKind::AndExpr) {
            self.kids(cl.expr).to_vec()
        } else {
            vec![cl.expr]
        };
        for e in conjuncts {
            if !self.open() {
                return;
            }
            let saved = std::mem::take(&mut self.f.temps);
            let bt = self.t_bool();
            self.hold_site = at.is_none();
            let c = self.val(e, Some(bt));
            self.drop_temps();
            self.hold_site = false;
            self.f.temps = saved;
            let Some(c) = c else { return };
            if !self.open() {
                return;
            }
            let (fail, ok) = (self.f.b.create_block(), self.f.b.create_block());
            self.f.b.cbr(c, ok, &[], fail, &[]);
            self.f.b.switch_to(fail);
            if let Some(sv) = trace {
                self.trace_restore(sv);
            }
            self.contract_fail(cl, e, at);
            self.f.b.switch_to(ok);
        }
    }

    /// The call of `runtime.panic.requires_failed(clause)` or
    /// `runtime.panic.ensures_failed(site, clause)` (never returns) for conjunct `e` of `cl`, at
    /// the site of `e`'s first token (where diagnostics about the check point).
    fn contract_fail(&mut self, cl: &crate::contracts::ContractClause, e: u32, at: Option<u32>) {
        let line = crate::contracts::clause_line(self.icx.span_text(cl.span));
        let s = Operand::Str(self.string(line.as_bytes()));
        let decl = if at.is_some() { self.g.lang.ensures_failed } else { self.g.lang.requires_failed };
        let Some(d) = decl else {
            self.fail_at(at, "`runtime.panic` has no contract failure fn");
            return;
        };
        let f = self.req(Req::Inst(InstKey { decl: d, args: Vec::new() }));
        self.f.last_site = None;
        let here = self.site(self.icx.a(e));
        let saved = self.f.b.site();
        self.f.b.set_site(here);
        self.contract_fail_call(f, s, at);
        self.f.b.set_site(saved);
    }

    fn contract_fail_call(&mut self, f: stave_ir::FuncId, s: Operand, at: Option<u32>) {
        match at {
            None => {
                self.f.b.call(f, None, &[s]);
            }
            Some(tok) => {
                let site = self.site(tok);
                let k = self.f.b.int(site.0 as i128);
                if let Operand::Const(ci) = k {
                    self.f.site_consts.push(ci);
                }
                self.f.b.call(f, None, &[k, s]);
            }
        }
        self.f.b.unreachable();
    }

    /// The `old(e)` snapshot of the call whose `(` is `paren`.
    pub fn old_value(&mut self, paren: u32) -> E {
        match self.old_vals.get(&paren) {
            Some(&(v, t)) => E::V(v, t),
            None => self.fail_at(Some(paren), "`old(..)` without an entry snapshot"),
        }
    }

    /// The environment type of closure body `k`: `&mut (&mut T0, &mut T1, ..)`, one element per
    /// capture (crate::closures); `None` without captures.
    pub fn env_type(&mut self, k: usize) -> Option<EnvType> {
        let caps = self.closure_info.get(k)?.captures.clone();
        if caps.is_empty() {
            return None;
        }
        let mut elems = Vec::with_capacity(caps.len());
        let mut locals = Vec::with_capacity(caps.len());
        for c in &caps {
            let l = *self.by_node.get(&c.binding)?;
            let t = self.locals[l].ty;
            let rt = self.t_ref(true, t);
            elems.push(rt);
            locals.push((l, rt));
        }
        let tup = self.tys.intern(TypeKind::Tuple(elems.into_boxed_slice()));
        let pt = self.t_ref(true, tup);
        Some((tup, pt, locals))
    }

    pub fn lower_closure(&mut self, n: u32, k: u32, ty: Ty) -> Result<(), String> {
        let TypeKind::Fn { params, ret, .. } = self.kind(ty) else { return Err("closure type".into()) };
        let env = self.env_type(k as usize);
        let mut all = params.to_vec();
        if let Some((_, pt, _)) = &env {
            all.push(*pt);
        }
        let saved = std::mem::replace(&mut self.f, FnState::new(&all, ret));
        let nl = self.locals.len();
        self.f.slot = vec![None; nl];
        self.f.env = Env { vals: vec![None; nl], st: vec![MState::Uninit; nl] };
        // captured locals: their slots are the addresses in the environment
        if let Some((_, _, locals)) = &env
            && let Some(ep) = self.f.b.param(params.len()).map(Operand::Value)
        {
            for (i, &(l, rt)) in locals.iter().enumerate() {
                let fp = self.t_ref(true, rt);
                let ix = self.int(i as i128);
                let a = self.op2(Op::FieldAddr, fp, ep, ix);
                let p = Operand::Value(self.f.b.load(FnCx::ir(rt), a));
                self.f.slot[l] = Some(p);
            }
        }
        self.make_slots(Some(n));
        self.push_scope();
        let icx = self.icx;
        self.bind_params(icx.child(n, NodeKind::Params));
        let body = match icx.kind(n) {
            Some(NodeKind::Stmt) => icx.first_child(n).filter(|&b| icx.kind(b) == Some(NodeKind::Block)),
            _ => icx.child(n, NodeKind::Block),
        };
        if let Some(b) = body {
            self.block(b);
        }
        self.end_fn();
        let site_consts = std::mem::take(&mut self.f.site_consts);
        let body = self.finish_fn();
        let name_off = Some((self.icx.fid(), self.icx.tok_start(self.icx.a(n))));
        let out = FnOut { closure: Some(k), params: all, ret: self.ret_of(ret), body, site_consts, name_off };
        self.f = saved;
        self.outs.push(out);
        Ok(())
    }

    // ---------- dispatch ----------

    /// The body of a `dispatch fn`: `%p = intrinsic.static_addr <runtime.cpu.WORD>`,
    /// `%w = load t<u64> %p` (no call: start code wrote the word before main and before any other
    /// thread, and nothing writes it later), then per tested arm top-down `(%w & mask) == mask` →
    /// return the arm target's result; at the end the plan's last arm. Loading the word requests
    /// `runtime.cpu.init`, which fills it. Without tested arms the body is the one call, with no
    /// feature-word load; so is it with a runtime library that has no `runtime.cpu.WORD` (no
    /// word: as if it were 0). With a word, each arm's block first tests the word's trace bit
    /// ([`crate::dispatch::TRACE_BIT`]) and, when set, calls `runtime.cpu.trace` with the line
    /// `dispatch: <dispatch> -> <arm>\n`.
    pub fn lower_dispatch(&mut self, plan: &crate::dispatch::Plan, params: &[Ty], ret: Ty) -> FnOut {
        self.fresh_env(params, ret);
        let args: Vec<Operand> =
            (0..params.len()).filter_map(|i| self.f.b.param(i).map(Operand::Value)).collect();
        if let Some(wd) = self.g.lang.cpu_word
            && !plan.tests.is_empty()
        {
            let u64t = self.t_int(false, crate::types::IntBits::B64);
            let bt = self.t_bool();
            let Some(&idx) = self.g.statics.get(&wd) else {
                self.fail_at(None, "runtime.cpu.WORD without a value");
                return self.dispatch_out(params, ret);
            };
            if self.g.sigs.konst(wd).map(|c| c.ty) != Some(u64t) {
                self.fail_at(None, "runtime.cpu.WORD is not a u64");
                return self.dispatch_out(params, ret);
            }
            if let Some(init) = self.g.lang.cpu_init {
                self.req(Req::Inst(InstKey { decl: init, args: Vec::new() }));
            }
            let pt = self.t_ptr(false, u64t);
            let sa = self.intrinsic("static_addr");
            let i = self.int(i128::from(idx));
            let p = self.f.b.op(Op::Intrinsic(sa), FnCx::ir(pt), &[i]);
            let w = Operand::Value(self.f.b.load(FnCx::ir(u64t), p));
            for (mask, k) in &plan.tests {
                let (hit, next) = (self.f.b.create_block(), self.f.b.create_block());
                let m = self.int(i128::from(*mask));
                let x = self.op2(Op::And, u64t, w, m);
                let c = self.op2(Op::Eq, bt, x, m);
                self.f.b.cbr(c, hit, &[], next, &[]);
                self.f.b.switch_to(hit);
                self.trace_arm(w, k);
                self.tail_call(k, &args, ret);
                self.f.b.switch_to(next);
            }
            self.trace_arm(w, &plan.last);
        }
        self.tail_call(&plan.last, &args, ret);
        self.dispatch_out(params, ret)
    }

    /// `if (%w & TRACE_BIT) != 0 { runtime.cpu.trace("dispatch: <this fn> -> <arm>\n") }`: one
    /// branch on the word already in a register, so tracing costs no load when off.
    fn trace_arm(&mut self, w: Operand, arm: &InstKey) {
        let Some(t) = self.g.lang.cpu_trace else { return };
        let u64t = self.t_int(false, crate::types::IntBits::B64);
        let bt = self.t_bool();
        let (on, go) = (self.f.b.create_block(), self.f.b.create_block());
        let bit = self.int(i128::from(crate::dispatch::TRACE_BIT));
        let zero = self.int(0);
        let x = self.op2(Op::And, u64t, w, bit);
        let c = self.op2(Op::Ne, bt, x, zero);
        self.f.b.cbr(c, on, &[], go, &[]);
        self.f.b.switch_to(on);
        let line = format!(
            "dispatch: {} -> {}\n",
            self.path,
            crate::lower::fn_path(self.g.prog, self.g.sigs, arm.decl)
        );
        let s = self.string(line.as_bytes());
        let f = self.req(Req::Inst(InstKey { decl: t, args: Vec::new() }));
        self.f.b.call(f, None, &[Operand::Str(s)]);
        self.f.b.br(go, &[]);
        self.f.b.switch_to(go);
    }

    fn tail_call(&mut self, k: &InstKey, args: &[Operand], ret: Ty) {
        let f = self.req(Req::Inst(k.clone()));
        let rt = self.ret_of(ret);
        let r = self.f.b.call(f, rt.map(FnCx::ir), args);
        if self.is_never(ret) {
            self.f.b.unreachable();
        } else {
            self.f.b.ret(r.map(Operand::Value));
        }
    }

    fn dispatch_out(&mut self, params: &[Ty], ret: Ty) -> FnOut {
        let body = if self.err.is_none() { self.finish_fn() } else { None };
        FnOut {
            closure: None,
            params: params.to_vec(),
            ret: self.ret_of(ret),
            body,
            site_consts: Vec::new(),
            name_off: None,
        }
    }

    // ---------- drop glue ----------

    /// The drop glue of `t`: `fn(p: &mut T)`.
    pub fn lower_glue(&mut self, t: Ty) -> FnOut {
        let pt = self.t_ref(true, t);
        self.fresh_env(&[pt], TypeId::NONE);
        let unit = self.t_unit();
        self.f.ret = unit;
        let Some(p) = self.f.b.param(0).map(Operand::Value) else {
            return FnOut {
                closure: None,
                params: vec![pt],
                ret: None,
                body: None,
                site_consts: Vec::new(),
                name_off: None,
            };
        };
        let w = self.g.world();
        if let Some(k) = crate::mono::drop_method(&w, &mut self.tys, t) {
            let f = self.req(Req::Inst(k));
            self.f.b.call(f, None, &[p]);
        }
        match self.kind(t) {
            TypeKind::Adt { def, .. } => {
                let is_enum = self.g.sigs.adt(def).is_some_and(|a| a.is_enum);
                let vs = self.adt_fields(t).unwrap_or_default();
                if !is_enum {
                    let fs = vs.into_iter().next().unwrap_or_default();
                    for (i, f) in fs.into_iter().enumerate() {
                        if self.needs_drop(f) {
                            let fp = self.t_ref(true, f);
                            let i = self.int(i as i128);
                            let a = self.op2(Op::FieldAddr, fp, p, i);
                            self.drop_ptr(a, f);
                        }
                    }
                } else {
                    let mut blocks = Vec::new();
                    for (vi, fs) in vs.iter().enumerate() {
                        if fs.iter().any(|&f| self.needs_drop(f)) {
                            blocks.push((vi, self.f.b.create_block()));
                        }
                    }
                    let end = self.f.b.create_block();
                    let v = Operand::Value(self.f.b.load(FnCx::ir(t), p));
                    let u32t = self.t_u32();
                    let tag = self.op1(Op::EnumTag, u32t, v);
                    let cases: Vec<(i128, Block, &[Operand])> =
                        blocks.iter().map(|&(vi, b)| (vi as i128, b, &[][..])).collect();
                    self.f.b.switch(tag, &cases, end, &[]);
                    for (vi, b) in blocks {
                        self.f.b.switch_to(b);
                        for (fi, &f) in vs[vi].iter().enumerate() {
                            if self.needs_drop(f) {
                                let fp = self.t_ref(true, f);
                                let (a, c) = (self.int(vi as i128), self.int(fi as i128));
                                let addr = self.opn(Op::EnumFieldAddr, fp, &[p, a, c]);
                                self.drop_ptr(addr, f);
                            }
                        }
                        self.f.b.br(end, &[]);
                    }
                    self.f.b.switch_to(end);
                }
            }
            TypeKind::Tuple(ts) => {
                for (i, &f) in ts.iter().enumerate() {
                    if self.needs_drop(f) {
                        let fp = self.t_ref(true, f);
                        let i = self.int(i as i128);
                        let a = self.op2(Op::FieldAddr, fp, p, i);
                        self.drop_ptr(a, f);
                    }
                }
            }
            TypeKind::Array { elem, len: ConstArg::Value(n) } => {
                let us = self.t_usize();
                let (head, hv) = self.f.b.create_block_with(&[FnCx::ir(us)]);
                let body = self.f.b.create_block();
                let exit = self.f.b.create_block();
                let z = self.int(0);
                self.f.b.br(head, &[z]);
                self.f.b.switch_to(head);
                let i = Operand::Value(hv[0]);
                let nn = self.int(n as i128);
                let bt = self.t_bool();
                let c = self.op2(Op::Lt, bt, i, nn);
                self.f.b.cbr(c, body, &[], exit, &[]);
                self.f.b.switch_to(body);
                let ep = self.t_ref(true, elem);
                let a = self.op2(Op::ElemAddr, ep, p, i);
                self.drop_ptr(a, elem);
                let one = self.int(1);
                let next = self.op2(Op::AddWrap, us, i, one);
                self.f.b.br(head, &[next]);
                self.f.b.switch_to(exit);
            }
            _ => {}
        }
        self.f.b.ret(None);
        let body = self.finish_fn();
        FnOut { closure: None, params: vec![pt], ret: None, body, site_consts: Vec::new(), name_off: None }
    }

    // ---------- statements ----------

    pub fn block(&mut self, b: u32) {
        self.push_scope();
        let kids = self.kids(b);
        for s in kids {
            if !self.open() {
                break;
            }
            self.stmt(s);
        }
        self.pop_scope();
    }

    fn stmt(&mut self, s: u32) {
        let icx = self.icx;
        let a = icx.a(s);
        if self.g.borrows {
            self.f.stmt_site = self.site(a);
            self.f.b.set_site(self.f.stmt_site);
        }
        let t = icx.tk(a);
        let kids = self.kids(s);
        if t == Term::KwLet as u8 {
            self.let_stmt(&kids);
        } else if t == Term::KwIf as u8 {
            if let Some(&i) = kids.first() {
                self.if_stmt(i);
            }
        } else if t == Term::KwWhile as u8 {
            if let (Some(&c), Some(&b)) = (kids.first(), kids.get(1)) {
                self.while_stmt(s, c, b);
            }
        } else if t == Term::KwLoop as u8 {
            if let Some(&b) = kids.first() {
                self.loop_stmt(s, b);
            }
        } else if t == Term::KwFor as u8 {
            if let Some(&f) = kids.first() {
                self.for_stmt(f);
            }
        } else if t == Term::KwMatch as u8 {
            self.lower_match(s, &kids, false);
        } else if t == Term::KwReturn as u8 {
            self.return_stmt(a, kids.first().copied());
        } else if t == Term::KwBreak as u8 || t == Term::KwContinue as u8 {
            self.break_continue(a, t == Term::KwBreak as u8);
        } else if t == Term::KwSpawn as u8 {
            self.spawn_stmt(s);
        } else if t == Term::KwUnsafe as u8 {
            if let Some(&b) = kids.iter().find(|&&c| icx.kind(c) == Some(NodeKind::Block)) {
                self.block(b);
            }
        } else if t == Term::KwScope as u8 {
            if let Some(&b) = kids.first() {
                self.scope_stmt(b);
            }
        } else if t == Term::KwStrict as u8 || t == Term::KwEmulate as u8 || t == Term::LBrace as u8 {
            if let Some(&b) = kids.first() {
                self.block(b);
            }
        } else if let Some(&e) = kids.first() {
            self.expr_stmt(e);
        }
        self.drop_temps();
    }

    /// `scope { .. }`: a thread-list handle (`usize`, 0 = empty) in a slot; the block's spawns
    /// add to it, and `runtime.thread.scope_join` waits for them when the block ends or is left.
    fn scope_stmt(&mut self, b: u32) {
        let us = self.t_usize();
        let h = self.new_slot(us);
        let z = self.int(0);
        self.f.b.store(h, z);
        self.push_scope();
        let d = self.f.scopes.len();
        self.f.scope_handles.push((d, h));
        for s in self.kids(b) {
            if !self.open() {
                break;
            }
            self.stmt(s);
        }
        self.pop_scope();
        // a handle left behind by a block that ended early (closed): already joined on the exit
        self.f.scope_handles.retain(|x| x.0 < d);
    }

    /// `spawn { .. }`: its body is closure `@<enclosing>.<k>` of type `fn()`;
    /// `runtime.thread.scope_spawn(handle, f)` runs it on a new thread.
    fn spawn_stmt(&mut self, s: u32) {
        let Some(&(_, h)) = self.f.scope_handles.last() else {
            self.fail(s, "spawn outside scope");
            return;
        };
        let Some(k) = self.closure_nodes.iter().position(|&c| c == s) else {
            self.fail(s, "spawn body");
            return;
        };
        let unit = self.t_unit();
        let ft = self.tys.intern(TypeKind::Fn { params: Box::new([]), ret: unit, features: 0, deny: 0 });
        let Some(fv) = self.closure_value(s, k, ft) else { return };
        let Some(d) = crate::mono::item_at(self.g.prog, "runtime.thread", "scope_spawn") else {
            self.fail(s, "`runtime.thread.scope_spawn` not found");
            return;
        };
        let tok = self.icx.a(s);
        let _ = self.call_inst(InstKey { decl: d, args: Vec::new() }, &[h, fv], tok);
    }

    /// Lowers closure body `k` (node `n`, fn type `ty`) and returns its fn value: `@f` without
    /// captures, else `fn.make @f env` with the environment (the captured locals' addresses) in a
    /// slot of the current fn.
    pub fn closure_value(&mut self, n: u32, k: usize, ty: Ty) -> Option<Operand> {
        if let Err(e) = self.lower_closure(n, k as u32, ty) {
            self.fail(n, e);
            return None;
        }
        let f = self.req(Req::Closure(k as u32));
        let Some((tup, pt, locals)) = self.env_type(k) else { return Some(Operand::Func(f)) };
        let env = self.new_slot(tup);
        for (i, &(l, rt)) in locals.iter().enumerate() {
            let Some(p) = self.f.slot.get(l).copied().flatten() else {
                self.fail(n, "captured local without a slot");
                return None;
            };
            let fp = self.t_ref(true, rt);
            let ix = self.int(i as i128);
            let a = self.op2(Op::FieldAddr, fp, env, ix);
            self.f.b.store(a, p);
        }
        let _ = pt;
        Some(Operand::Value(self.f.b.fn_make(FnCx::ir(ty), f, env)))
    }

    /// The type of a binding pattern (a single name), for the coercion site of its initializer.
    fn binding_ty(&self, pat: u32) -> Option<Ty> {
        let icx = self.icx;
        let a = icx.a(pat);
        let single = icx.is(a, Term::KwMut)
            || (icx.is(a, Term::Ident)
                && icx.chain_end(a) == a
                && !icx.is(a + 1, Term::LParen)
                && !icx.is(a + 1, Term::DotLBrace));
        if !single {
            return None;
        }
        self.by_node.get(&pat).map(|&l| self.locals[l].ty)
    }

    fn let_stmt(&mut self, kids: &[u32]) {
        let icx = self.icx;
        let Some(&pat) = kids.first() else { return };
        let Some(init) = kids.iter().copied().find(|&c| icx.is_expr(c)) else { return };
        let want = self.binding_ty(pat);
        let e = self.expr(init, want);
        if matches!(e, E::Never) {
            return;
        }
        let Some(p) = self.place_of(e) else { return };
        self.bind_irrefutable(pat, p);
    }

    fn bind_irrefutable(&mut self, pat: u32, p: Place) {
        let icx = self.icx;
        if icx.is(icx.a(pat), Term::Underscore) {
            // `let _ = e;`: an owned temporary is dropped at the end of the statement
            if let Base::Val(v, t) = p.base
                && p.proj.is_empty()
                && self.needs_drop(t)
            {
                let _ = self.spill(v, t, true);
            }
            return;
        }
        let _ = self.pat(pat, p, Mode::Move, None, false);
    }

    fn expr_stmt(&mut self, es: u32) {
        let icx = self.icx;
        let kids = self.kids(es);
        if kids.len() >= 3 {
            let (lhs, op, rhs) = (kids[0], kids[1], kids[2]);
            let opt = icx.tk(icx.a(op));
            let le = self.expr(lhs, None);
            let Some(lp) = self.place_of(le) else { return };
            if opt == Term::Eq as u8 {
                let lt = lp.ty;
                let Some(v) = self.val(rhs, Some(lt)) else { return };
                self.assign(&lp, v);
            } else {
                let bin = compound_op(opt);
                let lt = lp.ty;
                let Some(old) = self.peek(&lp) else { return };
                let Some(r) = self.val(rhs, Some(lt)) else { return };
                let tok = icx.a(op);
                let Some(nv) = self.binop(tok, bin as u8, old, r, lt) else { return };
                self.assign(&lp, nv);
            }
            return;
        }
        let Some(&e) = kids.first() else { return };
        let ex = self.expr(e, None);
        match ex {
            E::V(v, t) => {
                if self.needs_drop(t) {
                    let _ = self.spill(v, t, true);
                }
            }
            E::P(p) => {
                // a place statement reads nothing; an owned f-string temporary is dropped
                if let Base::Mem(m, _) = p.base
                    && p.root.is_none()
                    && p.proj.is_empty()
                    && self.is_fmt_temp(e)
                {
                    let t = p.ty;
                    self.f.temps.push((m, t));
                }
            }
            _ => {}
        }
    }

    fn is_fmt_temp(&self, e: u32) -> bool {
        self.down_to_primary(e).is_some_and(|p| self.icx.is(self.icx.a(p), Term::FmtLit))
    }

    fn if_stmt(&mut self, n: u32) {
        let kids = self.kids(n);
        let (Some(&c), Some(&tb)) = (kids.first(), kids.get(1)) else { return };
        let els = kids.get(2).copied();
        let then = self.f.b.create_block();
        let eb = els.map(|_| self.f.b.create_block());
        let j = self.join(n, &[]);
        let bt = self.t_bool();
        let Some(cv) = self.val(c, Some(bt)) else { return };
        let snap = self.f.env.clone();
        match eb {
            Some(eb) => self.f.b.cbr(cv, then, &[], eb, &[]),
            None => {
                let args = self.edge(j, &[]);
                let jb = self.f.joins[j].block;
                self.f.b.cbr(cv, then, &[], jb, &args);
            }
        }
        self.f.b.switch_to(then);
        self.block(tb);
        self.br_join(j, &[]);
        if let (Some(eb), Some(e)) = (eb, els) {
            self.f.env = snap;
            self.f.b.switch_to(eb);
            if self.nkind(e) == Some(NodeKind::IfStmt) {
                self.if_stmt(e);
            } else {
                self.block(e);
            }
            self.br_join(j, &[]);
        }
        self.enter_live(j);
    }

    /// Enters join `j`; if no edge reached it, the code after it is unreachable.
    fn enter_live(&mut self, j: usize) -> bool {
        let reached = self.f.joins[j].st.is_some();
        self.enter(j);
        if !reached {
            self.f.b.unreachable();
        }
        reached
    }

    fn while_stmt(&mut self, s: u32, c: u32, b: u32) {
        let head = self.join(s, &[]);
        let body = self.f.b.create_block();
        let exit = self.join(s, &[]);
        self.br_join(head, &[]);
        self.enter(head);
        let bt = self.t_bool();
        let Some(cv) = self.val(c, Some(bt)) else { return };
        let args = self.edge(exit, &[]);
        let eb = self.f.joins[exit].block;
        self.f.b.cbr(cv, body, &[], eb, &args);
        self.f.b.switch_to(body);
        let scope = self.f.scopes.len();
        self.f.loops.push(LoopCx { head, exit, scope, kind: LoopKind::Plain });
        self.block(b);
        self.br_join(head, &[]);
        self.f.loops.pop();
        self.enter_live(exit);
    }

    fn loop_stmt(&mut self, s: u32, b: u32) {
        let body = self.join(s, &[]);
        let exit = self.join(s, &[]);
        self.br_join(body, &[]);
        self.enter(body);
        let scope = self.f.scopes.len();
        self.f.loops.push(LoopCx { head: body, exit, scope, kind: LoopKind::Plain });
        self.block(b);
        self.br_join(body, &[]);
        self.f.loops.pop();
        self.enter_live(exit);
    }

    fn step_counter(&mut self, kind: &LoopKind) -> Vec<Operand> {
        match kind {
            LoopKind::Plain => Vec::new(),
            LoopKind::Count { i, ty, end, step } => {
                let one = self.int(*step as i128);
                let next = self.op2(Op::AddWrap, *ty, *i, one);
                match end {
                    Some(e) => {
                        let bt = self.t_bool();
                        let more = self.op2(Op::Ne, bt, *i, *e);
                        vec![next, more]
                    }
                    None => vec![next],
                }
            }
        }
    }

    fn break_continue(&mut self, tok: u32, is_break: bool) {
        let Some(lp) = self.f.loops.last().cloned() else {
            self.fail_at(Some(tok), "break outside a loop");
            return;
        };
        self.drop_temps_keep();
        self.drop_scopes(lp.scope);
        if is_break {
            self.br_join(lp.exit, &[]);
        } else {
            let pre = self.step_counter(&lp.kind);
            self.br_join(lp.head, &pre);
        }
        if self.open() {
            self.f.b.unreachable();
        }
    }

    /// `return e;` at token `at`.
    fn return_stmt(&mut self, at: u32, e: Option<u32>) {
        let rt = self.f.ret;
        let v = match e {
            Some(e) => {
                let ex = self.expr(e, Some(rt));
                if matches!(ex, E::Never) {
                    return;
                }
                if self.ret_of(rt).is_some() {
                    match self.operand_of(ex) {
                        Some(v) => Some(v),
                        None => return,
                    }
                } else {
                    None
                }
            }
            None => None,
        };
        self.drop_temps_keep();
        self.drop_scopes(0);
        self.ensures_checks(v, at);
        if self.open() {
            self.f.b.ret(v);
        }
    }

    // ---------- for ----------

    fn for_stmt(&mut self, f: u32) {
        let icx = self.icx;
        let kids = self.kids(f);
        let Some(&pat) = kids.first() else { return };
        let Some(iter) = kids.iter().copied().find(|&c| icx.is_expr(c)) else { return };
        let Some(body) = kids.iter().copied().find(|&c| icx.kind(c) == Some(NodeKind::Block)) else { return };
        let tail = kids.iter().copied().find(|&c| icx.kind(c) == Some(NodeKind::TailClause));
        if let Some(bt) = kids.iter().copied().find(|&c| icx.kind(c) == Some(NodeKind::Type)) {
            let t = self.lower_type(bt);
            let vt = match self.kind(t) {
                TypeKind::Ref { inner, .. } => inner,
                _ => t,
            };
            if matches!(self.kind(vt), TypeKind::Vector { .. }) {
                self.chunk_loop(f, pat, t, iter, body, tail);
                return;
            }
        }
        if tail.is_some() {
            self.unsupported(f, "a tail clause on an element loop is not supported");
            return;
        }
        let it_ty = self.nty(iter);
        // ranges: counted loop
        if let TypeKind::Range { elem, inclusive } = self.kind(it_ty) {
            if !matches!(self.kind(elem), TypeKind::Int { .. }) {
                self.unsupported(f, "char ranges in for loops are not supported yet");
                return;
            }
            let bounds = if icx.kind(iter) == Some(NodeKind::RangeExpr) {
                let rk = self.kids(iter);
                let (Some(&l), Some(&r)) = (rk.first(), rk.get(1)) else { return };
                let Some(a) = self.val(l, Some(elem)) else { return };
                let Some(b) = self.val(r, Some(elem)) else { return };
                (a, b)
            } else {
                let Some(rv) = self.val(iter, None) else { return };
                let z = self.int(0);
                let o = self.int(1);
                (self.op2(Op::Get, elem, rv, z), self.op2(Op::Get, elem, rv, o))
            };
            self.counted_loop(f, pat, body, elem, bounds.0, bounds.1, inclusive, None);
            return;
        }
        // arrays and slices
        let e = self.expr(iter, None);
        let Some(p) = self.place_of(e) else { return };
        match self.kind(p.ty) {
            TypeKind::Ref { inner, .. } => {
                let (elem, len) = match self.kind(inner) {
                    TypeKind::Array { elem, len } => (elem, len.value()),
                    TypeKind::Slice { elem, .. } => (elem, None),
                    _ => {
                        self.iter_loop(f, pat, body, p);
                        return;
                    }
                };
                let pt = p.ty;
                let Some(v) = self.read(&p) else { return };
                let us = self.t_usize();
                let n = match len {
                    Some(n) => self.int(n as i128),
                    None => self.op1(Op::SliceLen, us, v),
                };
                let z = self.int(0);
                let _ = elem;
                self.counted_loop(f, pat, body, us, z, n, false, Some(Seq::Ref(v, pt)));
            }
            TypeKind::Array { len, .. } => {
                let us = self.t_usize();
                let n = self.int(len.value().unwrap_or(0) as i128);
                let z = self.int(0);
                self.counted_loop(f, pat, body, us, z, n, false, Some(Seq::Val(p, iter)));
            }
            _ => self.iter_loop(f, pat, body, p),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn counted_loop(
        &mut self,
        f: u32,
        pat: u32,
        body: u32,
        ty: Ty,
        start: Operand,
        end: Operand,
        inclusive: bool,
        seq: Option<Seq>,
    ) {
        let bt = self.t_bool();
        let prefix: Vec<Ty> = if inclusive { vec![ty, bt] } else { vec![ty] };
        let head = self.join(f, &prefix);
        let bb = self.f.b.create_block();
        let exit = self.join(f, &[]);
        let first: Vec<Operand> = if inclusive {
            let more = self.op2(Op::Le, bt, start, end);
            vec![start, more]
        } else {
            vec![start]
        };
        self.br_join(head, &first);
        self.enter(head);
        let i = Operand::Value(self.f.joins[head].prefix[0]);
        let c = if inclusive {
            Operand::Value(self.f.joins[head].prefix[1])
        } else {
            self.op2(Op::Lt, bt, i, end)
        };
        let args = self.edge(exit, &[]);
        let eb = self.f.joins[exit].block;
        self.f.b.cbr(c, bb, &[], eb, &args);
        self.f.b.switch_to(bb);
        let scope = self.f.scopes.len();
        let kind = LoopKind::Count { i, ty, end: inclusive.then_some(end), step: 1 };
        self.f.loops.push(LoopCx { head, exit, scope, kind: kind.clone() });
        self.push_scope();
        let item = match seq {
            None => Some(Place { base: Base::Val(i, ty), proj: Vec::new(), ty, root: None }),
            Some(Seq::Ref(v, pt)) => {
                let (inner, m, _) = self.pointee(pt).unwrap_or((TypeId::NONE, false, false));
                let elem = match self.kind(inner) {
                    TypeKind::Array { elem, .. } | TypeKind::Slice { elem, .. } => elem,
                    _ => TypeId::NONE,
                };
                let et = self.t_ref(m, elem);
                let a = self.op2(Op::ElemAddr, et, v, i);
                Some(Place { base: Base::Val(a, et), proj: Vec::new(), ty: et, root: None })
            }
            Some(Seq::Val(mut p, iter)) => {
                let elem = match self.kind(p.ty) {
                    TypeKind::Array { elem, .. } => elem,
                    _ => TypeId::NONE,
                };
                let tok = self.icx.a(iter);
                p.proj.push(Proj::Elem(i, elem, tok));
                p.ty = elem;
                Some(p)
            }
        };
        if let Some(item) = item {
            let _ = self.pat(pat, item, Mode::Move, None, false);
        }
        self.block(body);
        self.pop_scope();
        if self.open() {
            let pre = self.step_counter(&kind);
            self.br_join(head, &pre);
        }
        self.f.loops.pop();
        self.enter_live(exit);
    }

    /// `for v: TxN in s { .. } tail ..` (contracts/simd-ops.md "Chunk loops and tails"). Blocks:
    /// head, body, the tail's blocks, exit. The head tests `i < lim` with
    /// `lim = sub.sat (slice.len s) N-1` (`i + N <= len`, the chunk-loop proof of the body's
    /// `vec.load s i`). A `&mut` binding is a slot written back to `s[i..i+N]` whenever the body
    /// scope ends or is left. Tails:
    /// - `masked (v, m)`: when `r = len - i` is not 0, `m = mask.first_n r` and
    ///   `v = vec.load_masked s i m` (inactive lanes 0, nothing read past the slice); a `&mut v`
    ///   writes back its active lanes only (`vec.store_masked`).
    /// - `scalar (x)`: a counted loop `i..len` binding `x = s[j]`.
    /// - `exact;`: `check.bounds (len - i) 1` (traps `Bounds` unless `len % N == 0`).
    fn chunk_loop(&mut self, f: u32, pat: u32, bind_t: Ty, iter: u32, body: u32, tail: Option<u32>) {
        let icx = self.icx;
        let (vt, by_ref) = match self.kind(bind_t) {
            TypeKind::Ref { inner, .. } => (inner, true),
            _ => (bind_t, false),
        };
        let (lane, lanes) = match self.kind(vt) {
            TypeKind::Vector { lane, n: crate::prim::Lanes::Fixed(k) } => (lane, k as u32),
            _ => {
                self.vector_unsupported(Some(icx.a(f)), "sizeless chunk loops");
                return;
            }
        };
        // the sequence as a slice reference
        let e = self.expr(iter, None);
        let Some(p) = self.place_of(e) else { return };
        let (s, mutable) = match self.kind(p.ty) {
            TypeKind::Ref { mutable, inner } => {
                let Some(v) = self.read(&p) else { return };
                match self.kind(inner) {
                    TypeKind::Slice { .. } => {
                        let st = self.slice_ref_ty(mutable, lane);
                        (self.convert_ptr(v, p.ty, st), mutable)
                    }
                    TypeKind::Array { len, .. } => {
                        (self.array_slice(v, mutable, lane, len.value().unwrap_or(0)), mutable)
                    }
                    _ => {
                        self.fail(f, "chunk loop over a non-sequence");
                        return;
                    }
                }
            }
            TypeKind::Array { len, .. } => {
                let n = len.value().unwrap_or(0);
                let Some(a) = self.addr(&p, None) else { return };
                (self.array_slice(a, true, lane, n), true)
            }
            // a `[T]` place (a sub-slice, a `std.soa` column): its address is the slice
            TypeKind::Slice { .. } => {
                let Some((a, at)) = self.addr_t(&p, None) else { return };
                let mutable = self.pointee(at).is_some_and(|x| x.1);
                let st = self.slice_ref_ty(mutable, lane);
                (self.convert_ptr(a, at, st), mutable)
            }
            _ => {
                self.fail(f, "chunk loop over a non-sequence");
                return;
            }
        };
        let _ = mutable;
        let us = self.t_usize();
        let bt = self.t_bool();
        let n = self.op1(Op::SliceLen, us, s);
        let k = self.int(lanes as i128 - 1);
        let lim = self.op2(Op::SubSat, us, n, k);
        let tail_kind = tail.map(|tc| {
            let a = icx.a(tc);
            if icx.is(a + 1, Term::KwMasked) {
                1
            } else if icx.is(a + 1, Term::KwScalar) {
                2
            } else {
                3
            }
        });
        // blocks in contract order: head, body, tail blocks, exit
        let head = self.join(f, &[us]);
        let bb = self.f.b.create_block();
        let tb = match tail_kind {
            Some(1) => TailBlocks::Masked(self.f.b.create_block(), self.f.b.create_block()),
            Some(2) => {
                let th = self.join(f, &[us]);
                TailBlocks::Scalar(th, self.f.b.create_block())
            }
            Some(_) => TailBlocks::Exact(self.f.b.create_block()),
            None => TailBlocks::None,
        };
        let exit = self.join(f, &[]);
        let z = self.int(0);
        self.br_join(head, &[z]);
        self.enter(head);
        let i = Operand::Value(self.f.joins[head].prefix[0]);
        let c = self.op2(Op::Lt, bt, i, lim);
        let snap = self.f.env.clone();
        match tb {
            TailBlocks::Scalar(th, _) => {
                let args = self.edge(th, &[i]);
                let hb = self.f.joins[th].block;
                self.f.b.cbr(c, bb, &[], hb, &args);
            }
            TailBlocks::Masked(t, _) | TailBlocks::Exact(t) => self.f.b.cbr(c, bb, &[], t, &[]),
            TailBlocks::None => {
                let args = self.edge(exit, &[]);
                let eb = self.f.joins[exit].block;
                self.f.b.cbr(c, bb, &[], eb, &args);
            }
        }
        // full chunks
        self.f.b.switch_to(bb);
        let scope = self.f.scopes.len();
        let kind = LoopKind::Count { i, ty: us, end: None, step: lanes };
        self.f.loops.push(LoopCx { head, exit, scope, kind: kind.clone() });
        self.push_scope();
        let tok = icx.a(f);
        let Some(x) = self.vop(tok, VecOp::Load, Some(vt), &[s, i]) else { return };
        let item = self.chunk_item(x, vt, bind_t, by_ref, s, i, None, tok);
        let _ = self.pat(pat, item, Mode::Move, None, false);
        self.block(body);
        self.pop_scope();
        if self.open() {
            let pre = self.step_counter(&kind);
            self.br_join(head, &pre);
        }
        self.f.loops.pop();
        // the tail
        let Some(tc) = tail else {
            self.enter_live(exit);
            return;
        };
        let tkids = self.kids(tc);
        let binds: Vec<u32> =
            tkids.iter().copied().filter(|&c| icx.kind(c) == Some(NodeKind::TailBind)).collect();
        let tbody = tkids.iter().copied().find(|&c| icx.kind(c) == Some(NodeKind::Block));
        let ttok = icx.a(tc);
        match tb {
            TailBlocks::Masked(tb, tm) => {
                self.f.env = snap;
                self.f.b.switch_to(tb);
                let r = self.op2(Op::SubWrap, us, n, i);
                let z = self.int(0);
                let any = self.op2(Op::Ne, bt, r, z);
                let args = self.edge(exit, &[]);
                let eb = self.f.joins[exit].block;
                self.f.b.cbr(any, tm, &[], eb, &args);
                self.f.b.switch_to(tm);
                let w = crate::types::lane_bits(&self.kind(lane)) as u8;
                let mt =
                    self.tys.intern(TypeKind::Mask { width: w, n: crate::prim::Lanes::Fixed(lanes as u8) });
                let Some(m) = self.vop(ttok, VecOp::FirstN, Some(mt), &[r]) else { return };
                let Some(x) = self.vop(ttok, VecOp::LoadMasked, Some(vt), &[s, i, m]) else { return };
                self.push_scope();
                let item = self.chunk_item(x, vt, bind_t, by_ref, s, i, Some(m), ttok);
                if let Some(&vb) = binds.first() {
                    let _ = self.pat(vb, item, Mode::Move, None, false);
                }
                if let Some(&mb) = binds.get(1) {
                    let mp = Place { base: Base::Val(m, mt), proj: Vec::new(), ty: mt, root: None };
                    let _ = self.pat(mb, mp, Mode::Move, None, false);
                }
                if let Some(b) = tbody {
                    self.block(b);
                }
                self.pop_scope();
                self.br_join(exit, &[]);
            }
            TailBlocks::Scalar(th, tl) => {
                self.enter(th);
                let j = Operand::Value(self.f.joins[th].prefix[0]);
                let c2 = self.op2(Op::Lt, bt, j, n);
                let args = self.edge(exit, &[]);
                let eb = self.f.joins[exit].block;
                self.f.b.cbr(c2, tl, &[], eb, &args);
                self.f.b.switch_to(tl);
                self.push_scope();
                let ep = self.t_ref(false, lane);
                let a = self.op2(Op::ElemAddr, ep, s, j);
                let xv = Operand::Value(self.f.b.load(Self::ir(lane), a));
                if let Some(&xb) = binds.first() {
                    let xp = Place { base: Base::Val(xv, lane), proj: Vec::new(), ty: lane, root: None };
                    let _ = self.pat(xb, xp, Mode::Move, None, false);
                }
                if let Some(b) = tbody {
                    self.block(b);
                }
                self.pop_scope();
                if self.open() {
                    let one = self.int(1);
                    let next = self.op2(Op::AddWrap, us, j, one);
                    self.br_join(th, &[next]);
                }
            }
            TailBlocks::Exact(tb) => {
                self.f.env = snap;
                self.f.b.switch_to(tb);
                let r = self.op2(Op::SubWrap, us, n, i);
                let one = self.int(1);
                self.trapping(ttok, Op::BoundsCheck, None, &[r, one]);
                self.br_join(exit, &[]);
            }
            TailBlocks::None => {}
        }
        self.enter_live(exit);
    }

    /// The chunk binding's value: the vector `x`, or for a `&mut` binding a slot holding it,
    /// written back to `s[i..]` (active lanes of `mask`) when the current scope ends or is left.
    #[allow(clippy::too_many_arguments)]
    fn chunk_item(
        &mut self,
        x: Operand,
        vt: Ty,
        bind_t: Ty,
        by_ref: bool,
        s: Operand,
        i: Operand,
        mask: Option<Operand>,
        tok: u32,
    ) -> Place {
        if !by_ref {
            return Place { base: Base::Val(x, vt), proj: Vec::new(), ty: vt, root: None };
        }
        let slot = self.new_slot(vt);
        self.f.b.store(slot, x);
        let d = self.f.scopes.len();
        self.f.write_backs.push((d, WriteBack { slot, vt, s, i, mask, tok }));
        Place { base: Base::Val(slot, bind_t), proj: Vec::new(), ty: bind_t, root: None }
    }

    fn slice_ref_ty(&mut self, mutable: bool, elem: Ty) -> Ty {
        let s = self.tys.intern(TypeKind::Slice { elem, align: 0 });
        self.t_ref(mutable, s)
    }

    /// `&[T]` over the `len` elements of the array at pointer `a`.
    fn array_slice(&mut self, a: Operand, mutable: bool, elem: Ty, len: u64) -> Operand {
        let ep = self.t_ref(mutable, elem);
        let z = self.int(0);
        let p = self.op2(Op::ElemAddr, ep, a, z);
        let st = self.slice_ref_ty(mutable, elem);
        let n = self.int(len as i128);
        self.op2(Op::SliceMake, st, p, n)
    }

    fn iter_loop(&mut self, f: u32, pat: u32, body: u32, it: Place) {
        let it = match it.base {
            Base::Val(..) | Base::Local(_) => {
                let t = it.ty;
                let Some(v) = self.read(&it) else { return };
                let (s, st) = self.spill(v, t, false);
                Place { base: Base::Mem(s, st), proj: Vec::new(), ty: t, root: None }
            }
            _ => it,
        };
        let w = self.g.world();
        let Some(next) = self.g.prog.strings.lookup("next") else {
            self.fail(f, "no `next` method");
            return;
        };
        let m = match w.lookup_method(&mut self.tys, it.ty, next, &[]) {
            crate::traits::Lookup::Found { m, .. } => m,
            _ => {
                self.fail(f, "no `next` method");
                return;
            }
        };
        let gargs: Vec<GArg> = match &m {
            crate::traits::MethodRef::Impl { impl_args, .. } => impl_args.clone(),
            _ => Vec::new(),
        };
        let Some(Resolved::Inst(k)) = self.resolve_method(&m, &gargs) else {
            self.fail(f, "unresolved `next`");
            return;
        };
        let Some((ps, ret)) = self.inst_sig(&k) else { return };
        let head = self.join(f, &[]);
        let bb = self.f.b.create_block();
        let exit = self.join(f, &[]);
        self.br_join(head, &[]);
        self.enter(head);
        let want = ps.first().copied();
        let Some(a) = self.addr(&it, want) else { return };
        let tok = self.icx.a(f);
        let r = self.call_inst(k, &[a], tok);
        let Some(rv) = self.operand_of(r) else { return };
        let u32t = self.t_u32();
        let tag = self.op1(Op::EnumTag, u32t, rv);
        let some = self.int(self.g.opt_some as i128);
        let bt = self.t_bool();
        let c = self.op2(Op::Eq, bt, tag, some);
        let args = self.edge(exit, &[]);
        let eb = self.f.joins[exit].block;
        self.f.b.cbr(c, bb, &[], eb, &args);
        self.f.b.switch_to(bb);
        let scope = self.f.scopes.len();
        self.f.loops.push(LoopCx { head, exit, scope, kind: LoopKind::Plain });
        self.push_scope();
        let xt = match self.kind(ret) {
            TypeKind::Adt { args, .. } => match args.first() {
                Some(GArg::Type(t)) => *t,
                _ => TypeId::NONE,
            },
            _ => TypeId::NONE,
        };
        let (s, z) = (self.int(self.g.opt_some as i128), self.int(0));
        let x = self.opn(Op::EnumGet, xt, &[rv, s, z]);
        let item = Place { base: Base::Val(x, xt), proj: Vec::new(), ty: xt, root: None };
        let _ = self.pat(pat, item, Mode::Move, None, false);
        self.block(body);
        self.pop_scope();
        self.br_join(head, &[]);
        self.f.loops.pop();
        self.enter_live(exit);
    }

    // ---------- match ----------

    pub fn match_expr(&mut self, n: u32) -> E {
        let kids = self.kids(n);
        self.lower_match(n, &kids, true)
    }

    /// The top-level test of pattern `p` against a value of (peeled) type `t`.
    fn top_of(&self, p: u32, t: Ty) -> Top {
        let icx = self.icx;
        let a = icx.a(p);
        if icx.is(a, Term::Underscore) || icx.is(a, Term::KwMut) {
            return Top::Any;
        }
        let kids = icx.kids_vec(p);
        if let Some(&l) = kids.first()
            && icx.kind(l) == Some(NodeKind::Literal)
        {
            if kids.len() != 1 {
                return Top::Complex;
            }
            return match lit_int(icx, l) {
                Some(v) if !matches!(self.tys.kind(t), TypeKind::Float(_)) => Top::Value(v),
                _ => Top::Complex,
            };
        }
        if icx.is(a, Term::LParen) {
            let close = icx.close(a);
            if kids.len() == 1 && !icx.is(close - 1, Term::Comma) {
                return self.top_of(kids[0], t);
            }
            return Top::Complex;
        }
        if !icx.is(a, Term::Ident) {
            return Top::Complex;
        }
        let e = icx.chain_end(a);
        if e == a && !icx.is(e + 1, Term::LParen) && !icx.is(e + 1, Term::DotLBrace) {
            return Top::Any;
        }
        let Some(u) = icx.uses(p).last() else { return Top::Complex };
        let Res::Decl(d) = u.res else { return Top::Complex };
        match self.decl_kind(d) {
            Some(DeclKind::Variant) => Top::Value(self.g.prog.decl(d).map_or(0, |x| x.index) as i128),
            Some(DeclKind::Const) => match self.g.sigs.konst(d).and_then(|c| c.value.clone()) {
                Some(crate::consteval::Val::Int(v)) => Top::Value(v),
                Some(crate::consteval::Val::Bool(b)) => Top::Value(i128::from(b)),
                Some(crate::consteval::Val::Char(c)) => Top::Value(c as i128),
                _ => Top::Complex,
            },
            _ => Top::Complex,
        }
    }

    /// Peels references of a scrutinee place (pattern binding modes, patterns.rs).
    fn peel(&mut self, mut p: Place, mut mode: Mode) -> Option<(Place, Mode)> {
        for _ in 0..1_000_000 {
            match self.kind(p.ty) {
                TypeKind::Ref { mutable, .. } => {
                    let v = self.peek(&p)?;
                    let t = p.ty;
                    p = self.deref(v, t);
                    mode = match mode {
                        Mode::Move => Mode::Ref(mutable),
                        Mode::Ref(m) => Mode::Ref(m && mutable),
                    };
                }
                _ => break,
            }
        }
        Some((p, mode))
    }

    /// The SSA locals the first alternative `p` of an or-pattern binds, in local order.
    fn ssa_bindings(&self, p: u32) -> Vec<usize> {
        let icx = self.icx;
        let lo = icx.ast.descendants(p).start;
        let bs = &icx.res.bindings;
        let i = bs.partition_point(|x| x.at.node < lo);
        let j = bs.partition_point(|x| x.at.node <= p);
        let mut ls: Vec<usize> = bs[i..j]
            .iter()
            .filter_map(|b| self.by_node.get(&b.at.node).copied())
            .filter(|&l| self.f.slot.get(l).copied().flatten().is_none())
            .collect();
        ls.sort_unstable();
        ls.dedup();
        ls
    }

    pub fn lower_match(&mut self, n: u32, kids: &[u32], is_expr: bool) -> E {
        let icx = self.icx;
        let Some(&scrut) = kids.first() else { return self.fail(n, "match") };
        let arms = &kids[1..];
        let ty = if is_expr { self.nty(n) } else { TypeId::NONE };
        let vt = if !is_expr || ty.is_none() || self.is_never(ty) { self.t_unit() } else { ty };
        // blocks: per arm a test per alternative, guard, body; then join
        struct ArmB {
            guard: Option<u32>,
            body: u32,
            /// Its alternatives are rows `lo..hi`.
            lo: usize,
            hi: usize,
            gb: Option<Block>,
            bb: Block,
        }
        // one alternative of an arm: its pattern and test block
        let mut rows: Vec<(u32, Block)> = Vec::with_capacity(arms.len());
        let mut ab: Vec<ArmB> = Vec::with_capacity(arms.len());
        for &arm in arms {
            // [ArmPattern, guard?, body]
            let ak = icx.kids_vec(arm);
            let (Some(&ap), Some(&body)) = (ak.first(), ak.last()) else { continue };
            let guard = if ak.len() >= 3 { Some(ak[1]) } else { None };
            let lo = rows.len();
            match icx.only_child(ap) {
                Some(pat) => rows.push((pat, self.f.b.create_block())),
                None => {
                    for &pat in &icx.kids_vec(ap) {
                        rows.push((pat, self.f.b.create_block()));
                    }
                }
            }
            let gb = guard.map(|_| self.f.b.create_block());
            let bb = self.f.b.create_block();
            ab.push(ArmB { guard, body, lo, hi: rows.len(), gb, bb });
        }
        let pre: Vec<Ty> = if is_expr { vec![vt] } else { Vec::new() };
        let j = self.join(n, &pre);
        let e = self.expr(scrut, None);
        let Some(sp) = self.place_of(e) else { return E::Never };
        let Some((dp, _)) = self.peel(sp.clone(), Mode::Move) else { return E::Never };
        let dk = self.kind(dp.ty);
        let tops: Vec<Top> = rows.iter().map(|r| self.top_of(r.0, dp.ty)).collect();
        let nr = rows.len();
        let discr = matches!(dk, TypeKind::Int { .. } | TypeKind::Bool | TypeKind::Char)
            || matches!(&dk, TypeKind::Adt { def, .. } if self.g.sigs.adt(*def).is_some_and(|a| a.is_enum));
        // the value every alternative of the arm tests, if it is one value
        let arm_top = |a: &ArmB| match tops[a.lo] {
            Top::Value(_) if tops[a.lo..a.hi].iter().all(|t| *t == tops[a.lo]) => tops[a.lo],
            _ => Top::Any,
        };
        // After a failed guard the rows of the later arms are tried. Their tests rely on the
        // switch having picked them, so the next row must be known without the discriminant:
        // either the arm tests one value, or the row after it accepts every value.
        let switch_mode = discr
            && nr > 0
            && tops.iter().all(|t| *t != Top::Complex)
            && tops.iter().any(|t| matches!(t, Top::Value(_)))
            && ab
                .iter()
                .all(|a| a.guard.is_none() || arm_top(a) != Top::Any || a.hi == nr || tops[a.hi] == Top::Any);
        // the first row from `from` on that a value passing a row with top `top` can still match
        let next_row = |from: usize, top: Top| {
            if !switch_mode {
                return (from < nr).then_some(from);
            }
            (from..nr).find(|&m| match (top, tops[m]) {
                (_, Top::Any) => true,
                (Top::Value(a), Top::Value(b)) => a == b,
                _ => false,
            })
        };
        let snap = self.f.env.clone();
        let scope_base = self.f.scopes.len();
        let mut unreach: Option<Block> = None;
        if switch_mode {
            let d = if matches!(dk, TypeKind::Adt { .. }) {
                let Some(v) = self.peek(&dp) else { return E::Never };
                let u32t = self.t_u32();
                self.op1(Op::EnumTag, u32t, v)
            } else {
                let Some(v) = self.peek(&dp) else { return E::Never };
                // a constant scrutinee: `switch` takes a value
                match (v, &dk) {
                    (Operand::Const(_), TypeKind::Bool) => {
                        let one = self.int(1);
                        let bt = self.t_bool();
                        self.op2(Op::Eq, bt, v, one)
                    }
                    (Operand::Const(_), _) => {
                        let t = dp.ty;
                        self.op1(Op::Cast(stave_ir::CastKind::Exact), t, v)
                    }
                    _ => v,
                }
            };
            let default = tops.iter().position(|t| *t == Top::Any).unwrap_or(nr - 1);
            let mut cases: Vec<(i128, Block, &[Operand])> = Vec::new();
            for (k, t) in tops.iter().enumerate().take(default) {
                if let Top::Value(v) = t
                    && !cases.iter().any(|c| c.0 == *v)
                {
                    cases.push((*v, rows[k].1, &[]));
                }
            }
            self.f.b.switch(d, &cases, rows[default].1, &[]);
        } else {
            match rows.first() {
                Some(r0) => self.f.b.br(r0.1, &[]),
                None => self.f.b.unreachable(),
            }
        }
        for a in &ab {
            let mut target = |s: &mut Self, row: Option<usize>| match row {
                Some(m) => rows[m].1,
                None => *unreach.get_or_insert_with(|| s.f.b.create_block()),
            };
            let guard_fail = target(self, next_row(a.hi, arm_top(a)));
            let ok = if a.hi - a.lo == 1 {
                let fail = target(self, next_row(a.lo + 1, tops[a.lo]));
                self.f.env = snap.clone();
                self.f.b.switch_to(rows[a.lo].1);
                self.push_scope();
                let ok = self.pat(rows[a.lo].0, sp.clone(), Mode::Move, Some(fail), switch_mode);
                if ok {
                    match a.gb {
                        Some(gb) => self.f.b.br(gb, &[]),
                        None if self.open() => self.f.b.br(a.bb, &[]),
                        None => {}
                    }
                }
                ok
            } else {
                // An or-pattern: every alternative binds the same locals and branches to the
                // one block the arm continues in, which takes the SSA locals as params.
                let shared = a.gb.unwrap_or(a.bb);
                let ls = self.ssa_bindings(rows[a.lo].0);
                let params: Vec<Value> = ls
                    .iter()
                    .map(|&l| {
                        let t = self.locals[l].ty;
                        self.f.b.add_param(shared, Self::ir(t))
                    })
                    .collect();
                // the env and the scope's locals after the first alternative that lowers
                let mut bound: Option<(Env, Vec<usize>)> = None;
                for k in a.lo..a.hi {
                    let fail = target(self, next_row(k + 1, tops[k]));
                    self.f.env = snap.clone();
                    self.f.b.switch_to(rows[k].1);
                    self.push_scope();
                    let ok = self.pat(rows[k].0, sp.clone(), Mode::Move, Some(fail), switch_mode);
                    if !ok || !self.open() {
                        self.f.scopes.truncate(scope_base);
                        continue;
                    }
                    let mut args: Vec<Operand> = Vec::with_capacity(ls.len());
                    for &l in &ls {
                        let v = match self.f.env.vals.get(l).copied().flatten() {
                            Some(v) => v,
                            None => {
                                let t = self.locals[l].ty;
                                self.dummy(t)
                            }
                        };
                        args.push(v);
                    }
                    self.f.b.br(shared, &args);
                    let mut scope = self.f.scopes.pop().unwrap_or_default();
                    if let Some((env, first)) = &bound {
                        // the alternatives bind the same names with the same types and modes,
                        // so they initialize and move the same locals
                        let mut first = first.clone();
                        first.sort_unstable();
                        scope.sort_unstable();
                        if env.st != self.f.env.st || first != scope {
                            self.fail(rows[k].0, "or-pattern alternatives bind differently");
                        }
                    } else {
                        bound = Some((self.f.env.clone(), scope));
                    }
                }
                match bound {
                    Some((env, scope)) => {
                        self.f.env = env;
                        self.f.scopes.push(scope);
                        for (&l, &p) in ls.iter().zip(&params) {
                            self.f.env.vals[l] = Some(Operand::Value(p));
                        }
                        true
                    }
                    None => false,
                }
            };
            if ok {
                if let (Some(g), Some(gb)) = (a.guard, a.gb) {
                    self.f.b.switch_to(gb);
                    let bt = self.t_bool();
                    if let Some(gv) = self.val(g, Some(bt)) {
                        self.f.b.cbr(gv, a.bb, &[], guard_fail, &[]);
                    }
                }
                self.f.b.switch_to(a.bb);
                if is_expr {
                    let w = Some(vt);
                    if let Some(v) = self.val(a.body, w) {
                        self.drop_temps_keep();
                        self.pop_scope();
                        self.br_join(j, &[v]);
                    }
                } else {
                    self.block(a.body);
                    self.pop_scope();
                    self.br_join(j, &[]);
                }
            }
            // keep the scope stack balanced on diverging arms
            self.f.scopes.truncate(scope_base);
        }
        if let Some(u) = unreach {
            self.f.b.switch_to(u);
            self.f.b.unreachable();
        }
        let reached = self.enter_live(j);
        if !reached {
            return E::Never;
        }
        if is_expr { E::V(Operand::Value(self.f.joins[j].prefix[0]), vt) } else { E::Unit }
    }

    /// Tests pattern `p` against place `place` (branching to `fail` on mismatch) and binds its
    /// names. `skip_top`: the top-level discriminant was already tested (switch).
    fn pat(&mut self, p: u32, place: Place, mode: Mode, fail: Option<Block>, skip_top: bool) -> bool {
        let icx = self.icx;
        let a = icx.a(p);
        if icx.is(a, Term::Underscore) {
            return true;
        }
        if icx.is(a, Term::KwMut) {
            return self.bind_name(p, place, mode);
        }
        let kids = icx.kids_vec(p);
        if let Some(&l0) = kids.first()
            && icx.kind(l0) == Some(NodeKind::Literal)
        {
            if icx.is(icx.a(l0), Term::StrLit) {
                self.unsupported(p, "string literal patterns are not supported yet");
                return false;
            }
            let Some((dp, _)) = self.peel(place, mode) else { return false };
            if skip_top {
                return true;
            }
            return self.lit_test(p, &kids, dp, fail);
        }
        if icx.is(a, Term::LParen) {
            let close = icx.close(a);
            if kids.len() == 1 && !icx.is(close - 1, Term::Comma) {
                return self.pat(kids[0], place, mode, fail, skip_top);
            }
            let Some((dp, pm)) = self.peel(place, mode) else { return false };
            let ets = match self.kind(dp.ty) {
                TypeKind::Tuple(ts) => ts.to_vec(),
                _ => {
                    self.fail(p, "tuple pattern type");
                    return false;
                }
            };
            for (i, &k) in kids.iter().enumerate() {
                let mut sp = dp.clone();
                let et = ets.get(i).copied().unwrap_or(TypeId::NONE);
                sp.proj.push(Proj::Field(i as u32, et));
                sp.ty = et;
                if !self.pat(k, sp, pm, fail, false) {
                    return false;
                }
            }
            return true;
        }
        if !icx.is(a, Term::Ident) {
            self.fail(p, "pattern");
            return false;
        }
        let e = icx.chain_end(a);
        let has_paren = icx.is(e + 1, Term::LParen);
        let has_rec = icx.is(e + 1, Term::DotLBrace);
        if e == a && !has_paren && !has_rec {
            return self.bind_name(p, place, mode);
        }
        let Some(u) = icx.uses(p).last() else {
            self.fail(p, "unresolved pattern");
            return false;
        };
        let Res::Decl(d) = u.res else {
            self.fail(p, "pattern");
            return false;
        };
        let Some((dp, pm)) = self.peel(place, mode) else { return false };
        match self.decl_kind(d) {
            Some(DeclKind::Variant | DeclKind::Struct) => {
                let is_variant = self.decl_kind(d) == Some(DeclKind::Variant);
                let vi = if is_variant { self.g.prog.decl(d).map_or(0, |x| x.index) } else { 0 };
                let Some(fts) = self.adt_fields(dp.ty).and_then(|v| v.get(vi as usize).cloned()) else {
                    self.fail(p, "pattern type");
                    return false;
                };
                if is_variant && !skip_top {
                    let Some(v) = self.peek(&dp) else { return false };
                    let u32t = self.t_u32();
                    let tag = self.op1(Op::EnumTag, u32t, v);
                    let want = self.int(vi as i128);
                    let bt = self.t_bool();
                    let c = self.op2(Op::Eq, bt, tag, want);
                    if !self.test(c, fail) {
                        return false;
                    }
                }
                let sub = |s: &mut Self, fi: usize| -> Place {
                    let mut sp = dp.clone();
                    let ft = fts.get(fi).copied().unwrap_or(TypeId::NONE);
                    if is_variant {
                        sp.proj.push(Proj::EnumField(vi, fi as u32, ft));
                    } else {
                        sp.proj.push(Proj::Field(fi as u32, ft));
                    }
                    sp.ty = ft;
                    let _ = s;
                    sp
                };
                if has_rec {
                    let adt = match self.kind(dp.ty) {
                        TypeKind::Adt { def, .. } => def,
                        _ => return false,
                    };
                    let names: Vec<crate::ids::Sym> = self
                        .g
                        .sigs
                        .adt(adt)
                        .and_then(|ad| ad.variants.get(vi as usize))
                        .map(|v| v.fields.iter().map(|f| f.name).collect())
                        .unwrap_or_default();
                    for fp in kids {
                        if icx.kind(fp) != Some(NodeKind::FieldPat) {
                            continue;
                        }
                        let name = icx.sym(icx.a(fp));
                        let Some(fi) = names.iter().position(|&x| x == name) else {
                            self.fail(fp, "unknown field");
                            return false;
                        };
                        let sp = sub(self, fi);
                        let ok = match icx.child(fp, NodeKind::Pattern) {
                            Some(sub_p) => self.pat(sub_p, sp, pm, fail, false),
                            None => self.bind_name(fp, sp, pm),
                        };
                        if !ok {
                            return false;
                        }
                    }
                } else if has_paren {
                    for (fi, &k) in kids.iter().enumerate() {
                        let sp = sub(self, fi);
                        if !self.pat(k, sp, pm, fail, false) {
                            return false;
                        }
                    }
                }
                true
            }
            Some(DeclKind::Const) => {
                if skip_top {
                    return true;
                }
                let Some(c) = self.g.sigs.konst(d) else { return false };
                let (ct, Some(cv)) = (c.ty, c.value.clone()) else { return false };
                let ce = self.const_val(&cv, ct, a);
                let Some(k) = self.operand_of(ce) else { return false };
                let Some(v) = self.peek(&dp) else { return false };
                let bt = self.t_bool();
                let c = self.op2(Op::Eq, bt, v, k);
                self.test(c, fail)
            }
            _ => {
                self.fail(p, "pattern");
                false
            }
        }
    }

    /// Branches to `fail` unless `c`; continues in a fresh block.
    fn test(&mut self, c: Operand, fail: Option<Block>) -> bool {
        let Some(fail) = fail else {
            // an irrefutable context: the test cannot fail
            return true;
        };
        let cont = self.f.b.create_block();
        self.f.b.cbr(c, cont, &[], fail, &[]);
        self.f.b.switch_to(cont);
        true
    }

    fn lit_test(&mut self, p: u32, lits: &[u32], dp: Place, fail: Option<Block>) -> bool {
        let icx = self.icx;
        let t = dp.ty;
        let Some(v) = self.peek(&dp) else { return false };
        let bt = self.t_bool();
        let lit = |s: &mut Self, l: u32| -> Option<Operand> {
            let a = s.icx.a(l);
            let (neg, tok) = if s.icx.is(a, Term::Minus) { (true, a + 1) } else { (false, a) };
            if s.icx.is(tok, Term::KwTrue) || s.icx.is(tok, Term::KwFalse) {
                return Some(s.int(i128::from(s.icx.is(tok, Term::KwTrue))));
            }
            if s.icx.is(tok, Term::CharLit) {
                let c = crate::literals::parse_char(s.icx.tok_str(tok))?;
                return Some(s.int(c as i128));
            }
            let e = s.literal(tok, neg, t);
            s.operand_of(e)
        };
        let c = if lits.len() == 2 {
            let inclusive = icx.is(icx.end_tok(lits[0]) + 1, Term::DotDotEq);
            let Some(lo) = lit(self, lits[0]) else { return false };
            let Some(hi) = lit(self, lits[1]) else { return false };
            let a = self.op2(Op::Ge, bt, v, lo);
            let b = self.op2(if inclusive { Op::Le } else { Op::Lt }, bt, v, hi);
            self.op2(Op::And, bt, a, b)
        } else {
            let Some(k) = lit(self, lits[0]) else { return false };
            self.op2(Op::Eq, bt, v, k)
        };
        let _ = p;
        self.test(c, fail)
    }

    /// Binds the local of binding node `b` to the place (by value or by reference).
    fn bind_name(&mut self, b: u32, place: Place, mode: Mode) -> bool {
        let Some(&l) = self.by_node.get(&b) else {
            self.fail(b, "binding without a local");
            return false;
        };
        let lt = self.locals[l].ty;
        let v = match mode {
            Mode::Move => self.read(&place),
            Mode::Ref(_) => self.addr(&place, Some(lt)).map(|v| self.borrowed(v, lt)),
        };
        let Some(v) = v else { return false };
        self.bind(l, Some(v));
        true
    }
}

/// Element source of a counted `for` loop.
/// The tail blocks of a chunk loop: masked (the `len - i != 0` test, the body), scalar (the
/// counted loop's head join and body), exact (the check), or none.
#[derive(Clone, Copy)]
enum TailBlocks {
    None,
    Masked(Block, Block),
    Scalar(usize, Block),
    Exact(Block),
}

enum Seq {
    /// `&[T]` / `&[T; N]`: the pointer and its type; the binding is `&T` (`&mut T`).
    Ref(Operand, Ty),
    /// An array value (place) and the iterable's node; the binding is `T`.
    Val(Place, u32),
}

fn compound_op(opt: u8) -> Term {
    let m = |t: Term| t as u8 == opt;
    if m(Term::PlusEq) {
        Term::Plus
    } else if m(Term::MinusEq) {
        Term::Minus
    } else if m(Term::StarEq) {
        Term::Star
    } else if m(Term::SlashEq) {
        Term::Slash
    } else if m(Term::PercentEq) {
        Term::Percent
    } else if m(Term::PlusPercentEq) {
        Term::PlusPercent
    } else if m(Term::MinusPercentEq) {
        Term::MinusPercent
    } else if m(Term::StarPercentEq) {
        Term::StarPercent
    } else if m(Term::PlusPipeEq) {
        Term::PlusPipe
    } else if m(Term::MinusPipeEq) {
        Term::MinusPipe
    } else if m(Term::StarPipeEq) {
        Term::StarPipe
    } else if m(Term::AmpEq) {
        Term::Amp
    } else if m(Term::PipeEq) {
        Term::Pipe
    } else if m(Term::CaretEq) {
        Term::Caret
    } else if m(Term::LtLtEq) {
        Term::LtLt
    } else if m(Term::GtGtEq) {
        Term::GtGt
    } else {
        Term::LtLtPercent
    }
}

/// The int value of an int, char, or bool literal node.
fn lit_int(icx: crate::itemcx::ItemCx<'_>, l: u32) -> Option<i128> {
    let a = icx.a(l);
    let (neg, t) = if icx.is(a, Term::Minus) { (true, a + 1) } else { (false, a) };
    if icx.is(t, Term::IntLit) {
        let (mag, _) = crate::literals::parse_int(icx.tok_str(t))?;
        let v = mag as i128;
        return Some(if neg { -v } else { v });
    }
    if icx.is(t, Term::CharLit) {
        return crate::literals::parse_char(icx.tok_str(t)).map(|c| c as i128);
    }
    if icx.is(t, Term::KwTrue) {
        return Some(1);
    }
    if icx.is(t, Term::KwFalse) {
        return Some(0);
    }
    None
}
