//! Loan liveness and access checks.
//!
//! **Liveness by path exploration**: a loan is live at a
//! point iff some value whose origin set contains it may still be used. Per tracked local, every
//! use is walked backward through the CFG until the local's definition (for a borrow temporary,
//! that is the loan's creation), marking the local live on the way; the loans live at a point are
//! the union of the origin sets of the locals live there. This is the SSA-value form of "walk
//! backward from the use to the loan's creation": a path on which the local was redefined does
//! not carry the old loan (so `r = &a; ... r = &b; a = 1; use(r)` is accepted, as in NLL).
//!
//! **Access checks**: every access (read, write, move, drop, storage-dead, and the borrows) is
//! checked against the live loans whose places overlap it (E0501); reads and moves against the
//! maybe-moved sites (E0502); writes and `&mut` borrows against binding and reference mutability
//! (E0503); stores of a ref-like value (an assignment through a dereference, a call whose
//! callee has a `from` clause on a `&mut` param) against the view objects the pointer may point
//! to (E0513, [`crate::loans`] **Stores**); returns against the `from` clause (E0510) and local
//! borrows (E0512).

use stave_diag::Span;

use crate::bits::Bits;
use crate::cfg::{Body, LoanId, LocalId, OpKind, Operand, Place, Proj, RetCheck, Rvalue, Stmt, StmtKind};
use crate::loans::{Flow, Misfit, State, for_each_op};
use crate::ty::DerefKind;

/// The kind of an access.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Access {
    Read,
    SharedBorrow,
    /// A two-phase reservation (shared until activated).
    Reserve,
    MutBorrow,
    /// The activation of a two-phase loan.
    Activate,
    Write,
    Move,
    Drop,
    StorageDead,
}

impl Access {
    /// Conflicts with shared loans too (not only `&mut` ones).
    fn exclusive(self) -> bool {
        !matches!(self, Access::Read | Access::SharedBorrow | Access::Reserve)
    }

    /// A shallow access: a loan of a place below it reached through a dereference is not affected.
    fn shallow(self) -> bool {
        matches!(self, Access::Write | Access::Drop | Access::StorageDead)
    }

    /// The borrow dump's access kind (stage-gates `borrow`).
    pub fn dump_name(self) -> &'static str {
        match self {
            Access::Read | Access::SharedBorrow | Access::Reserve => "read",
            Access::MutBorrow | Access::Activate | Access::Write => "write",
            Access::Move => "move",
            Access::Drop => "drop",
            Access::StorageDead => "storage-dead",
        }
    }

    /// The E0501 `access` hole.
    pub fn verb(self) -> &'static str {
        match self {
            Access::Read | Access::SharedBorrow | Access::Reserve => "borrow",
            Access::MutBorrow | Access::Activate => "borrow mutably",
            Access::Write => "assign to",
            Access::Move => "move",
            Access::Drop | Access::StorageDead => "drop",
        }
    }
}

/// What the checker found (turned into diagnostics by [`crate::report`]).
#[derive(Clone, Debug)]
pub enum Finding {
    /// E0501: the access conflicts with these live loans.
    Conflict { block: u32, stmt: u32, access: Access, place: Place, span: Span, loans: Vec<LoanId> },
    /// E0502: a use of a moved place (the move site).
    Moved { site: u32, place: Place, span: Span },
    /// E0503: an assignment or `&mut` borrow of an immutable place; `shared`: through `&`.
    Immutable { place: Place, span: Span, shared: bool, blame: LocalId },
    /// E0512: a returned value borrows local storage (the loan).
    Escape { span: Span, loan: LoanId },
    /// E0510: a returned value has an origin of param `param` not in the `from` clause (`from`:
    /// the clause, or the return type when `clause` is false); `copy`: the clause names the param
    /// as `from copy`, which covers only the origins of the view it points to.
    Uncovered { span: Span, param: u32, from: Span, clause: bool, copy: bool },
    /// E0504: a by-value use of a non-Copy place behind a dereference or an index, or of a
    /// capture (`capture`: the closure's span).
    MoveOut { place: Place, span: Span, capture: Option<Span> },
    /// E0513: a ref-like value written through a dereference, or stored by a call through a
    /// `&mut` argument, has origins a view object it may land in cannot take.
    ViewWrite { place: Place, span: Span, why: Unfit },
}

/// Why a store is E0513.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unfit {
    /// The target lacks the value's origins.
    Origins,
    /// The objects the pointer may point to have them together, one of them may not.
    Several,
    /// Only the `from` clause of param `param` lacks them: the entries `add` (param, `copy`)
    /// declare the store.
    Clause { param: u32, add: Vec<(u32, bool)> },
}

impl Unfit {
    /// `together`: the value's origins are in the union of what the pointer's objects hold.
    fn of(m: Misfit, together: bool) -> Unfit {
        match m {
            Misfit::Clause { param, add } => Unfit::Clause { param, add },
            Misfit::Object if together => Unfit::Several,
            Misfit::Object => Unfit::Origins,
        }
    }
}

/// One borrow-dump line of an access: node, kind, verdict code, reported loan's node.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DumpAccess {
    pub node: u32,
    pub kind: &'static str,
    pub verdict: Option<&'static str>,
    pub loan: Option<u32>,
}

/// Locals a statement uses (their value or origins), with the use's span.
pub fn stmt_uses(s: &Stmt, mut f: impl FnMut(LocalId, Span)) {
    match &s.kind {
        StmtKind::Assign { dest, rv, .. } => {
            for_each_op(rv, |o| {
                if o.kind != OpKind::Const {
                    f(o.place.local, o.span)
                }
            });
            if let Rvalue::Ref { place, .. } = rv {
                f(place.local, s.span);
            }
            if dest.has_deref() {
                f(dest.local, s.span);
            }
        }
        StmtKind::Activate { local, .. } => f(*local, s.span),
        StmtKind::Read(o) | StmtKind::Return(o) => {
            if o.kind != OpKind::Const {
                f(o.place.local, o.span)
            }
        }
        StmtKind::Drop(p) => f(p.local, s.span),
        StmtKind::StorageDead(_) => {}
    }
}

/// The local a statement redefines entirely.
pub fn stmt_def(s: &Stmt) -> Option<LocalId> {
    match &s.kind {
        StmtKind::Assign { dest, .. } if dest.proj.is_empty() => Some(dest.local),
        StmtKind::StorageDead(l) => Some(*l),
        _ => None,
    }
}

/// Whether access `kind` of place `p` conflicts with a loan of place `q`.
pub fn places_conflict(p: &Place, q: &Place, kind: Access) -> bool {
    if p.local != q.local {
        return false;
    }
    let n = p.proj.len().min(q.proj.len());
    for i in 0..n {
        match (p.proj[i], q.proj[i]) {
            (Proj::Field(a), Proj::Field(b)) | (Proj::Column(a), Proj::Column(b)) if a != b => return false,
            (Proj::EnumField(v, a), Proj::EnumField(w, b)) if v == w && a != b => return false,
            _ => {}
        }
    }
    if p.proj.len() >= q.proj.len() {
        return true;
    }
    // the access is above the loan: a shallow access does not reach through a dereference
    !(kind.shallow() && q.proj[n..].iter().any(|x| matches!(x, Proj::Deref(_))))
}

/// E0503: whether `p` may be written (or mutably borrowed); `Err((shared, blame))` otherwise.
fn mutability(body: &Body, p: &Place) -> Result<(), (bool, LocalId)> {
    let raw = p.proj.iter().any(|x| matches!(x, Proj::Deref(DerefKind::Raw)));
    if body.locals[p.local as usize].guard && !raw {
        return Err((true, p.local));
    }
    let mut saw_deref = false;
    for pr in p.proj.iter().rev() {
        if let Proj::Deref(k) = pr {
            saw_deref = true;
            match k {
                DerefKind::Raw => return Ok(()),
                DerefKind::Shared => return Err((true, p.local)),
                DerefKind::Mut => {}
            }
        }
    }
    if saw_deref {
        return Ok(());
    }
    let l = &body.locals[p.local as usize];
    if l.mutable || l.temp { Ok(()) } else { Err((false, p.local)) }
}

pub struct Checked {
    pub findings: Vec<Finding>,
    pub dump: Vec<DumpAccess>,
}

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
        // use after move (implicit drops of moved values are no-ops); a write re-initializes the
        // place it names, but a write strictly inside a moved place uses it (and would drop a
        // part the move took)
        if !matches!(kind, Access::Drop | Access::StorageDead) && !st.moved.is_empty() {
            let mut best: Option<(u32, u32)> = None;
            for s in st.moved.iter() {
                let Some((m, ms)) = self.flow.body.move_sites.get(s) else { continue };
                let overlap = m.local == place.local
                    && if kind == Access::Write {
                        place.proj.len() > m.proj.len() && place.proj.starts_with(&m.proj)
                    } else {
                        m.proj.starts_with(&place.proj) || place.proj.starts_with(&m.proj)
                    };
                if overlap && best.is_none_or(|(o, _)| ms.offset < o) {
                    best = Some((ms.offset, s as u32));
                }
            }
            if let Some((_, site)) = best {
                verdict.get_or_insert("E0502");
                self.findings.push(Finding::Moved { site, place: place.clone(), span });
            }
        }
        if self.want_dump && node != u32::MAX {
            self.dump.push(DumpAccess { node, kind: kind.dump_name(), verdict, loan: reported });
        }
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
                        let capture = l.capture;
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
                            });
                        }
                    }
                    _ => for_each_op(rv, |o| self.operand(st, live, at, o)),
                }
                if !*init {
                    // the write happens after the right side: its moves count (`s.a = f(s)`)
                    let mut moves = Vec::new();
                    for_each_op(rv, |o| {
                        if o.kind == OpKind::Move {
                            moves.push(o.site as usize);
                        }
                    });
                    if moves.is_empty() {
                        self.access(st, after, at, Access::Write, dest, s.span, s.node, &[]);
                    } else {
                        let mut st2 = st.clone();
                        for m in moves {
                            st2.moved.insert(m);
                        }
                        self.access(&st2, after, at, Access::Write, dest, s.span, s.node, &[]);
                    }
                    if let Err((shared, blame)) = mutability(body, dest) {
                        self.findings.push(Finding::Immutable {
                            place: dest.clone(),
                            span: s.span,
                            shared,
                            blame,
                        });
                    }
                }
                // what the callee may store through its `&mut` arguments, per target argument
                if let Rvalue::Call { args, stores, .. } = rv {
                    let mut i = 0;
                    while i < stores.len() {
                        let target = stores[i].target;
                        let mut ptr = None;
                        let mut val = self.flow.empty();
                        while i < stores.len() && stores[i].target == target {
                            if let Some((p, v)) = self.flow.store_sets(st, args, &stores[i]) {
                                val.union_with(&v);
                                ptr = Some(p);
                            }
                            i += 1;
                        }
                        let (Some(ptr), Some(arg)) = (ptr, args.get(target as usize)) else { continue };
                        if let Err(m) = self.flow.store_fits(st, &ptr, &val, None, true) {
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
                        let fit = self.flow.store_fits(st, &ptr, &val, direct, false);
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
                // a closure may return a borrow of a capture (the closure value holds its loan);
                // its own params are locals like any other
                if l.place.has_deref() || (from.is_none() && root.capture.is_some()) {
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
                    findings.push(Finding::Immutable { place: dest.clone(), span: s.span, shared, blame });
                }
            }
        }
        bad_moves(body, &mut findings);
        return Checked { findings, dump: Vec::new() };
    }
    let lo = live_out(flow);
    let mut cx = Cx { flow, findings: Vec::new(), dump: Vec::new(), want_dump };
    for b in 0..body.blocks.len() {
        let Some(mut st) = cx.flow.entry[b].clone() else { continue };
        let la = live_after(cx.flow, b, &lo[b]);
        for (i, s) in body.blocks[b].stmts.iter().enumerate() {
            let li = live_in(cx.flow, s, &la[i]);
            let live = cx.flow.live_loans(&st, &li);
            let mut la_i = la[i].clone();
            if let Some(d) = stmt_def(s) {
                let t = cx.flow.tracked[d as usize];
                if t != u32::MAX {
                    la_i.remove(t as usize);
                }
            }
            let after = cx.flow.live_loans(&st, &la_i);
            cx.stmt(&st, &live, &after, (b as u32, i as u32), s);
            cx.flow.apply(&s.kind, &mut st);
        }
    }
    let mut findings = cx.findings;
    bad_moves(body, &mut findings);
    Checked { findings, dump: cx.dump }
}

/// E0504: one finding per span of the body's bad moves.
fn bad_moves(body: &Body, findings: &mut Vec<Finding>) {
    let mut seen: Vec<Span> = Vec::new();
    for (place, span, capture) in &body.bad_moves {
        if seen.contains(span) {
            continue;
        }
        seen.push(*span);
        findings.push(Finding::MoveOut { place: place.clone(), span: *span, capture: *capture });
    }
}

/// The span of the first use (smallest offset) reachable from statement `at` (inclusive) of a
/// value whose origin set holds `loan` (E0501 "later use").
pub fn later_use(flow: &mut Flow<'_>, at: (u32, u32), loan: LoanId) -> Option<Span> {
    let body = flow.body;
    let nb = body.blocks.len();
    let mut seen = vec![false; nb];
    let mut best: Option<Span> = None;
    let mut work: Vec<(u32, u32)> = vec![at];
    while let Some((b, from)) = work.pop() {
        let Some(mut st) = flow.entry[b as usize].clone() else { continue };
        let stmts = &body.blocks[b as usize].stmts;
        for (i, s) in stmts.iter().enumerate() {
            if i as u32 >= from {
                stmt_uses(s, |l, span| {
                    if flow.holds(&st, l, loan) && best.is_none_or(|x| span.offset < x.offset) {
                        best = Some(span);
                    }
                });
            }
            flow.apply(&s.kind, &mut st);
        }
        for &succ in body.blocks[b as usize].term.succs() {
            if !seen[succ as usize] {
                seen[succ as usize] = true;
                work.push((succ, 0));
            }
        }
    }
    best
}
