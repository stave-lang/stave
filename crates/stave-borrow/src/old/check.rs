//! The dense dataflow's checks: [`crate::check`] with a full state cloned per statement and
//! live sets as bitsets per statement.

use stave_diag::Span;

use crate::bits::Bits;
use crate::cfg::{LoanId, LocalId, OpKind, Operand, Place, Proj, RetCheck, Rvalue, Stmt, StmtKind};
use crate::check::{
    Access, Checked, DumpAccess, Finding, Unfit, back_edge, bad_moves, guard_effects, in_loop, mutability,
    places_conflict, stmt_def, stmt_moves, stmt_uses,
};
use crate::loans::for_each_op;
use crate::old::loans::{CallStore, Flow, State};
use crate::ty::DerefKind;

/// Liveness: live-out sets per block over tracked locals (path exploration per local).
pub fn live_out(flow: &Flow<'_>) -> Vec<Bits> {
    let body = flow.body;
    let nb = body.blocks.len();
    let nt = flow.ntracked;
    let mut ue = vec![Bits::new(nt); nb];
    let mut kill = vec![Bits::new(nt); nb];
    for (b, blk) in body.blocks.iter().enumerate() {
        for s in &blk.stmts {
            stmt_uses(s, |l, _| {
                let t = flow.tracked[l as usize];
                if t != u32::MAX && !kill[b].contains(t as usize) {
                    ue[b].insert(t as usize);
                }
            });
            if let Some(d) = stmt_def(s) {
                let t = flow.tracked[d as usize];
                if t != u32::MAX {
                    kill[b].insert(t as usize);
                }
            }
        }
    }
    let mut out = vec![Bits::new(nt); nb];
    let mut work: Vec<u32> = Vec::new();
    for x in 0..nt {
        work.clear();
        for (b, u) in ue.iter().enumerate() {
            if u.contains(x) {
                work.push(b as u32);
            }
        }
        while let Some(b) = work.pop() {
            for &p in &flow.preds[b as usize] {
                if out[p as usize].insert(x) && !kill[p as usize].contains(x) {
                    work.push(p);
                }
            }
        }
    }
    out
}

/// Live-after sets of every statement of block `b` (from its live-out).
pub fn live_after(flow: &Flow<'_>, b: usize, out: &Bits) -> Vec<Bits> {
    let stmts = &flow.body.blocks[b].stmts;
    let mut la = vec![out.clone(); stmts.len()];
    let mut cur = out.clone();
    for (i, s) in stmts.iter().enumerate().rev() {
        la[i] = cur.clone();
        if let Some(d) = stmt_def(s) {
            let t = flow.tracked[d as usize];
            if t != u32::MAX {
                cur.remove(t as usize);
            }
        }
        stmt_uses(s, |l, _| {
            let t = flow.tracked[l as usize];
            if t != u32::MAX {
                cur.insert(t as usize);
            }
        });
    }
    la
}

/// The live-in set of a statement given its live-after set.
fn live_in(flow: &Flow<'_>, s: &Stmt, after: &Bits) -> Bits {
    let mut li = after.clone();
    if let Some(d) = stmt_def(s) {
        let t = flow.tracked[d as usize];
        if t != u32::MAX {
            li.remove(t as usize);
        }
    }
    stmt_uses(s, |l, _| {
        let t = flow.tracked[l as usize];
        if t != u32::MAX {
            li.insert(t as usize);
        }
    });
    li
}

struct Cx<'f, 'b> {
    flow: &'f mut Flow<'b>,
    findings: Vec<Finding>,
    dump: Vec<DumpAccess>,
    want_dump: bool,
    /// The moves the statement being checked has made so far (local, site): earlier operands of
    /// its rvalue, and the whole rvalue before its write.
    pending: Vec<(LocalId, u32)>,
}

impl Cx<'_, '_> {
    #[allow(clippy::too_many_arguments)]
    fn access(
        &mut self,
        st: &State,
        live: &Bits,
        at: (u32, u32),
        kind: Access,
        place: &Place,
        span: Span,
        node: u32,
        exclude: &[LoanId],
    ) {
        let mut loans = Vec::new();
        for l in live.iter() {
            if l >= self.flow.atoms.nloans || exclude.contains(&(l as LoanId)) {
                continue;
            }
            let loan = self.flow.loan(l as LoanId);
            let loan_mut = loan.mutable && !loan.reserved;
            if !(kind.exclusive() || loan_mut) {
                continue;
            }
            if places_conflict(place, &loan.place, kind) {
                loans.push(l as LoanId);
            }
        }
        let mut verdict = None;
        let mut reported = None;
        if !loans.is_empty() {
            verdict = Some("E0501");
            reported = crate::report::pick_loan(self.flow.body, &loans).map(|l| self.flow.loan(l).node);
            self.findings.push(Finding::Conflict {
                block: at.0,
                stmt: at.1,
                access: kind,
                place: place.clone(),
                span,
                loans,
            });
        }
        // use after move (implicit drops of moved values are no-ops); a write of the whole local
        // initializes it again, a write of a part of it uses it
        if !matches!(kind, Access::Drop | Access::StorageDead)
            && st.moved.contains(place.local as usize)
            && !(kind == Access::Write && place.proj.is_empty())
            && let Some(site) = self.move_site(at, place.local)
        {
            verdict.get_or_insert("E0502");
            self.findings.push(Finding::Moved { site, place: place.clone(), span });
        }
        if self.want_dump && node != u32::MAX {
            self.dump.push(DumpAccess { node, kind: kind.dump_name(), verdict, loan: reported });
        }
    }

    /// The move of local `l` a use at statement `at` sees: one made earlier in the statement, else
    /// the one with the smallest offset among the moves that reach `at` without a whole write of
    /// `l` between (blocks walked backward).
    fn move_site(&self, at: (u32, u32), l: LocalId) -> Option<u32> {
        if let Some(&(_, s)) = self.pending.iter().find(|p| p.0 == l) {
            return Some(s);
        }
        let body = self.flow.body;
        let mut seen = vec![false; body.blocks.len()];
        let mut work = vec![(at.0, at.1 as usize)];
        let mut best: Option<u32> = None;
        while let Some((b, end)) = work.pop() {
            let stmts = &body.blocks[b as usize].stmts;
            let mut reached_start = true;
            for s in stmts[..end.min(stmts.len())].iter().rev() {
                if stmt_def(s) == Some(l) && !matches!(s.kind, StmtKind::StorageDead(_)) {
                    reached_start = false;
                    break;
                }
                let mut site = None;
                stmt_moves(s, |o| {
                    if o.place.local == l {
                        site = Some(o.site);
                    }
                });
                if let Some(site) = site {
                    let off = body.move_sites.get(site as usize).map_or(u32::MAX, |m| m.1.offset);
                    if best.is_none_or(|x| off < body.move_sites[x as usize].1.offset) {
                        best = Some(site);
                    }
                    reached_start = false;
                    break;
                }
            }
            if reached_start {
                for &p in &self.flow.preds[b as usize] {
                    if !seen[p as usize] {
                        seen[p as usize] = true;
                        work.push((p, body.blocks[p as usize].stmts.len()));
                    }
                }
            }
        }
        best
    }

    fn operand(&mut self, st: &State, live: &Bits, at: (u32, u32), o: &Operand) {
        let kind = match o.kind {
            OpKind::Const => return,
            OpKind::Copy => Access::Read,
            OpKind::Move => Access::Move,
        };
        self.access(st, live, at, kind, &o.place, o.span, o.node, &[]);
    }

    /// `live`: loans live at the statement (its operands are read); `after`: loans live after it
    /// (the assignment's write happens last).
    fn stmt(&mut self, st: &State, live: &Bits, after: &Bits, at: (u32, u32), s: &Stmt) {
        let body = self.flow.body;
        match &s.kind {
            StmtKind::Assign { dest, rv, init } => {
                match rv {
                    Rvalue::Ref { place, mutable, loan } => {
                        let l = self.flow.loan(*loan);
                        let kind = if l.reserved {
                            Access::Reserve
                        } else if *mutable {
                            Access::MutBorrow
                        } else {
                            Access::SharedBorrow
                        };
                        let (capture, cast) = (l.capture, l.cast);
                        self.access(st, live, at, kind, place, s.span, s.node, &[]);
                        if *mutable
                            && !capture
                            && let Err((shared, blame)) = mutability(body, place)
                        {
                            self.findings.push(Finding::Immutable {
                                place: place.clone(),
                                span: s.span,
                                shared,
                                blame,
                                cast,
                            });
                        }
                    }
                    _ => {
                        // operands are evaluated left to right: each sees the moves of the ones
                        // before it
                        let mut moved: Option<State> = None;
                        for_each_op(rv, |o| {
                            self.operand(moved.as_ref().unwrap_or(st), live, at, o);
                            if o.kind == OpKind::Move {
                                moved.get_or_insert_with(|| st.clone()).moved.insert(o.place.local as usize);
                                self.pending.push((o.place.local, o.site));
                            }
                        });
                    }
                }
                if !*init {
                    // the write happens after the right side: its moves count (`s.a = f(s)`)
                    let mut moves = Vec::new();
                    for_each_op(rv, |o| {
                        if o.kind == OpKind::Move {
                            moves.push((o.place.local, o.site));
                        }
                    });
                    if moves.is_empty() {
                        self.access(st, after, at, Access::Write, dest, s.span, s.node, &[]);
                    } else {
                        let mut st2 = st.clone();
                        for &(l, _) in &moves {
                            st2.moved.insert(l as usize);
                        }
                        self.pending.extend(moves);
                        self.access(&st2, after, at, Access::Write, dest, s.span, s.node, &[]);
                    }
                    if let Err((shared, blame)) = mutability(body, dest) {
                        self.findings.push(Finding::Immutable {
                            place: dest.clone(),
                            span: s.span,
                            shared,
                            blame,
                            cast: false,
                        });
                    }
                }
                // what the callee may store through its `&mut` arguments, per target argument,
                // with the values as they are once all of the call's stores are made
                if let Rvalue::Call { args, stores, .. } = rv
                    && !stores.is_empty()
                {
                    let mut made = st.clone();
                    let (groups, _) = self.flow.settle_call(&mut made, args, stores);
                    for CallStore { target, into_view, ptr, val } in groups {
                        let Some(arg) = args.get(target as usize) else { continue };
                        if let Err(m) = self.flow.store_fits(&made, &ptr, &val, None, Some(into_view)) {
                            // a pointer held in a named local is written through; a temporary is
                            // named by its expression
                            let mut place = arg.place.clone();
                            if !body.locals[place.local as usize].temp {
                                place.proj.push(Proj::Deref(DerefKind::Mut));
                            }
                            let why = Unfit::of(m, val.is_subset(&self.flow.contents_of(st, &ptr)));
                            self.findings.push(Finding::ViewWrite { place, span: s.span, why });
                        }
                    }
                }
                // a write through a dereference stores into the view objects the pointer may
                // point to; a write into a closure's capture (a reference to an outer local
                // whose origins the outer body fixed when it created the closure) may store only
                // origins the capture already has
                let capture = body.locals[dest.local as usize].capture.is_some();
                if (dest.has_deref() || capture)
                    && self.flow.is_ref_like(dest.ty)
                    && !matches!(rv, Rvalue::Pure(_))
                {
                    let (val, _) = self.flow.rv_origins(st, rv, dest.ty);
                    let why = if dest.has_deref() {
                        let (target, ptr) = self.flow.place_origins(st, dest);
                        let ptr = ptr.unwrap_or_else(|| self.flow.empty());
                        let direct =
                            if dest.derefs() == 1 { body.locals[dest.local as usize].param } else { None };
                        let fit = self.flow.store_fits(st, &ptr, &val, direct, None);
                        fit.err().map(|m| Unfit::of(m, val.is_subset(&target)))
                    } else {
                        let mut t = self.flow.empty();
                        if let Some(p) = body.locals[dest.local as usize].param {
                            t.insert(self.flow.atoms.pb(p));
                        }
                        (!val.is_subset(&t)).then_some(Unfit::Origins)
                    };
                    if let Some(why) = why {
                        self.findings.push(Finding::ViewWrite { place: dest.clone(), span: s.span, why });
                    }
                }
            }
            StmtKind::Activate { reserved, active, .. } => {
                let p = self.flow.loan(*reserved).place.clone();
                self.access(st, live, at, Access::Activate, &p, s.span, s.node, &[*reserved, *active]);
            }
            StmtKind::Read(o) => self.operand(st, live, at, o),
            StmtKind::Drop(p) => self.access(st, live, at, Access::Drop, p, s.span, s.node, &[]),
            StmtKind::StorageDead(l) => {
                let p = Place::local(*l, body.locals[*l as usize].ty);
                self.access(st, live, at, Access::StorageDead, &p, s.span, s.node, &[]);
            }
            StmtKind::Return(o) => {
                self.operand(st, live, at, o);
                self.ret(st, o);
            }
        }
    }

    fn ret(&mut self, st: &State, o: &Operand) {
        let body = self.flow.body;
        let (from, from_span) = match &body.ret {
            RetCheck::None => return,
            RetCheck::LocalsOnly => (None, None),
            RetCheck::From(v, s, clause) => (Some(v), Some((*s, *clause))),
        };
        let s = self.flow.op_origins(st, o);
        let mut escape: Option<LoanId> = None;
        let mut uncovered: Option<u32> = None;
        for a in s.iter() {
            if a < self.flow.atoms.nloans {
                let l = self.flow.loan(a as LoanId);
                let root = &body.locals[l.place.local as usize];
                // A closure may return a borrow of a capture it holds by `&` (the closure value
                // holds its loan, and no call of the closure changes the capture). Of a capture
                // held by `&mut`, through a dereference too, it may not: the next call can write
                // the capture or lend it again while this result lives. Its own params are
                // locals like any other.
                if !root.capture_mut && (l.place.has_deref() || (from.is_none() && root.capture.is_some())) {
                    continue;
                }
                if escape.is_none_or(|e| l.span.offset < self.flow.loan(e).span.offset) {
                    escape = Some(a as LoanId);
                }
            } else if let (Some(from), Some((p, copy))) = (from, self.flow.atoms.param(a)) {
                let ok = from.iter().any(|&(i, c)| i == p && (copy || !c));
                if !ok && uncovered.is_none_or(|u| p < u) {
                    uncovered = Some(p);
                }
            }
        }
        if let Some(l) = escape {
            self.findings.push(Finding::Escape { span: o.span, loan: l });
        } else if let (Some(p), Some(fs)) = (uncovered, from_span) {
            let copy = from.is_some_and(|f| f.iter().any(|&(i, _)| i == p));
            self.findings.push(Finding::Uncovered { span: o.span, param: p, from: fs.0, clause: fs.1, copy });
        }
    }
}

/// Checks a solved body.
pub fn check(flow: &mut Flow<'_>, want_dump: bool) -> Checked {
    let body = flow.body;
    if flow.stateless() && !want_dump {
        // Nothing is tracked ([`Flow::stateless`]): the only findings left are the ones that need
        // no dataflow state, writes to immutable places (in the reachable blocks) and the bad
        // moves. The dump lists every access, so it takes the general path.
        let mut findings = Vec::new();
        for (b, blk) in body.blocks.iter().enumerate() {
            if flow.entry[b].is_none() {
                continue;
            }
            for s in &blk.stmts {
                if let StmtKind::Assign { dest, init: false, .. } = &s.kind
                    && let Err((shared, blame)) = mutability(body, dest)
                {
                    findings.push(Finding::Immutable {
                        place: dest.clone(),
                        span: s.span,
                        shared,
                        blame,
                        cast: false,
                    });
                }
            }
        }
        bad_moves(body, &mut findings);
        guard_effects(body, &mut findings);
        return Checked { findings, dump: Vec::new() };
    }
    let lo = live_out(flow);
    let mut cx = Cx { flow, findings: Vec::new(), dump: Vec::new(), want_dump, pending: Vec::new() };
    for b in 0..body.blocks.len() {
        let Some(mut st) = cx.flow.entry[b].clone() else { continue };
        let la = live_after(cx.flow, b, &lo[b]);
        for (i, s) in body.blocks[b].stmts.iter().enumerate() {
            let li = live_in(cx.flow, s, &la[i]);
            let live = cx.flow.live_loans(&st, &li);
            // The loans live after the statement are read from the state after it: the value
            // the statement gives its destination may hold a loan that no other live local
            // holds (`x = V.{ r: &x.f }`: the write of `x` ends the value the new one borrows).
            let mut post = st.clone();
            cx.flow.apply(&s.kind, &mut post);
            let after = cx.flow.live_loans(&post, &la[i]);
            cx.pending.clear();
            cx.stmt(&st, &live, &after, (b as u32, i as u32), s);
            st = post;
        }
    }
    let mut findings = cx.findings;
    bad_moves(body, &mut findings);
    guard_effects(body, &mut findings);
    Checked { findings, dump: cx.dump }
}

/// The first use, in the order the code runs, after statement `at` (inclusive) of a value whose
/// origin set holds `loan` (E0501 "later use"): the use the fewest statements away on a path from
/// `at`, the first in source order among those. `true` with it: the path to it goes back to the
/// head of a loop that holds `at` and the use is in that loop: it is in a later iteration
/// ([`Body::loops`]).
pub fn later_use(flow: &mut Flow<'_>, at: (u32, u32), loan: LoanId) -> Option<(Span, bool)> {
    use std::cmp::Reverse;
    let body = flow.body;
    let nb = body.blocks.len();
    let mut seen = vec![false; nb];
    // (statements from `at`, the loop whose head the path went back to, block, first statement)
    let mut work = std::collections::BinaryHeap::new();
    work.push(Reverse((0usize, usize::MAX, at.0, at.1)));
    let mut best: Option<(usize, Span, bool)> = None;
    while let Some(Reverse((d, again, b, from))) = work.pop() {
        if best.is_some_and(|x| x.0 < d) {
            break;
        }
        if from == 0 {
            if seen[b as usize] {
                continue;
            }
            seen[b as usize] = true;
        }
        let Some(mut st) = flow.entry[b as usize].clone() else { continue };
        let stmts = &body.blocks[b as usize].stmts;
        let mut found: Option<(usize, Span)> = None;
        for (i, s) in stmts.iter().enumerate() {
            if i as u32 >= from {
                let mut here: Option<Span> = None;
                stmt_uses(s, |l, span| {
                    if flow.holds(&st, l, loan) && here.is_none_or(|x| span.offset < x.offset) {
                        here = Some(span);
                    }
                });
                if let Some(sp) = here {
                    found = Some((d + i - from as usize, sp));
                    break;
                }
            }
            flow.apply(&s.kind, &mut st);
        }
        if let Some((dd, sp)) = found {
            let again = body.loops.get(again).is_some_and(|&l| in_loop(l, b));
            if best.is_none_or(|x| (dd, again, sp.offset) < (x.0, x.2, x.1.offset)) {
                best = Some((dd, sp, again));
            }
            continue;
        }
        let len = stmts.len().saturating_sub(from as usize);
        for &succ in body.blocks[b as usize].term.succs() {
            let next = if again == usize::MAX { back_edge(body, b, succ) } else { again };
            work.push(Reverse((d + len + 1, next, succ, 0)));
        }
    }
    best.map(|x| (x.1, x.2))
}
