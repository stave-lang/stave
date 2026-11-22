//! Loan liveness and access checks.
//!
//! **Liveness by path exploration**: a loan is live at a
//! point iff some value whose origin set contains it may still be used. Per tracked local, every
//! use is walked backward through the CFG until the local's definition (for a borrow temporary,
//! that is the loan's creation), marking the local live on the way ([`crate::live`]); the loans
//! live at a point are the union of the origin sets of the locals live there. This is the
//! SSA-value form of "walk backward from the use to the loan's creation": a path on which the
//! local was redefined does not carry the old loan (so `r = &a; ... r = &b; a = 1; use(r)` is
//! accepted, as in NLL).
//!
//! The live loans are kept as a count per loan of the live locals that hold it, moved from
//! statement to statement by what the statement changes: the sets it writes, the locals whose
//! last use it is, the local it defines. After a store has grown a local the live loans are
//! closed under what their places hold ([`crate::loans`] **Stores**): the loans reached through
//! the counted ones are found again only when a counted loan that leads further comes or goes,
//! or the set of a borrowed local changes. The live loans of each local are chained, the
//! `&mut` ones apart from the others, and an access looks only at those of its own local that
//! its kind can conflict with.
//!
//! **Access checks**: every access (read, write, move, drop, storage-dead, and the borrows) is
//! checked against the live loans whose places overlap it (E0501; the storage-dead of a
//! temporary at the end of its statement is case `temporary`); reads and moves against the
//! maybe-moved sites (E0502); writes and `&mut` borrows against binding and reference mutability
//! (E0503); stores of a ref-like value (an assignment through a dereference, a call whose
//! callee has a `from` clause on a `&mut` param) against the view objects the pointer may point
//! to (E0513, [`crate::loans`] **Stores**); returns against the `from` clause (E0510), local
//! borrows, and a closure's borrows of its `&mut` captures (E0512).

use stave_diag::Span;

use rustc_hash::FxHashMap;

use crate::cfg::{
    Body, GuardEffect, LoanId, LocalId, MoveOut, OpKind, Operand, Place, Proj, RetCheck, Rvalue, Stmt,
    StmtKind,
};
use crate::loans::{CallStore, Flow, Misfit, Saved, State, for_each_op};
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
    pub(crate) fn exclusive(self) -> bool {
        !matches!(self, Access::Read | Access::SharedBorrow | Access::Reserve)
    }

    /// A shallow access: a loan of a place below it reached through a dereference is not affected.
    pub(crate) fn shallow(self) -> bool {
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
    /// E0503: an assignment or `&mut` borrow of an immutable place; `shared`: through `&`;
    /// `cast`: the borrow is a `&mut` place cast to `*mut`.
    Immutable { place: Place, span: Span, shared: bool, blame: LocalId, cast: bool },
    /// E0512: a returned value borrows local storage, or a closure's returned value borrows a
    /// capture the closure holds by `&mut` (the loan).
    Escape { span: Span, loan: LoanId },
    /// E0510: a returned value has an origin of param `param` not in the `from` clause (`from`:
    /// the clause, or the return type when `clause` is false); `copy`: the clause names the param
    /// as `from copy`, which covers only the origins of the view it points to.
    Uncovered { span: Span, param: u32, from: Span, clause: bool, copy: bool },
    /// E0504: a by-value use of a non-Copy place that cannot be moved from.
    MoveOut { place: Place, span: Span, why: MoveOut },
    /// E0505: a match guard (or a closure written in one) does more than read.
    GuardEffect { span: Span, what: GuardEffect },
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
    pub(crate) fn of(m: Misfit, together: bool) -> Unfit {
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

/// The move operands of a statement.
pub(crate) fn stmt_moves(s: &Stmt, mut f: impl FnMut(&Operand)) {
    match &s.kind {
        StmtKind::Assign { rv, .. } => for_each_op(rv, |o| {
            if o.kind == OpKind::Move {
                f(o)
            }
        }),
        StmtKind::Read(o) | StmtKind::Return(o) if o.kind == OpKind::Move => f(o),
        _ => {}
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

/// Whether an access that conflicts with a live loan is the end of a temporary (its drop or
/// storage-dead where its statement ends, or where a `break` / `continue` leaves it): the
/// borrow outlives the temporary (E0501 case `temporary`).
pub fn temp_end(body: &Body, kind: Access, p: &Place) -> bool {
    let l = &body.locals[p.local as usize];
    matches!(kind, Access::Drop | Access::StorageDead) && p.proj.is_empty() && l.temp && !l.static_root
}

/// E0503: whether `p` may be written (or mutably borrowed); `Err((shared, blame))` otherwise.
pub(crate) fn mutability(body: &Body, p: &Place) -> Result<(), (bool, LocalId)> {
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

struct Cx<'f, 'b> {
    flow: &'f mut Flow<'b>,
    findings: Vec<Finding>,
    dump: Vec<DumpAccess>,
    want_dump: bool,
    /// The moves the statement being checked has made so far (local, site): earlier operands of
    /// its rvalue, and the whole rvalue before its write.
    pending: Vec<(LocalId, u32)>,
    /// The locals those moves moved: moved for the statement's later accesses, beside the ones
    /// the state marks.
    moved_now: Vec<LocalId>,
    /// Per loan: how many of the live tracked locals hold it at the point reached. A loan is
    /// live while that is not zero, or while it is `reached`.
    holders: Vec<u32>,
    /// Whether a store has grown a local ([`Flow::grown`]): the live loans are then closed
    /// under [`Flow::loan_holds`], and the fields up to `stale` are in use.
    grown: bool,
    /// Per loan: it leads further (its place is a borrowed local whose set is read from the
    /// state, or it has contents).
    leads: Vec<bool>,
    /// The loans with a holder that lead further, each once; per loan its place in the list
    /// plus one (0: not in it).
    roots: Vec<LoanId>,
    root_at: Vec<u32>,
    /// Per loan: it is reached through the loans of `roots`; the reached ones. Valid unless
    /// `stale`.
    reached: Vec<bool>,
    reached_list: Vec<LoanId>,
    /// Per loan: the last search that met it (a count of the searches).
    met: Vec<u32>,
    /// Per local: the last search that read its set ([`Flow::read_local`]).
    read_at: Vec<u32>,
    searches: u32,
    stale: bool,
    /// Loans counted in or out and loans met by the searches; the searches.
    counted_loans: u64,
    rebuilt: u64,
    /// Per tracked local: the id of the set whose loans are counted for it in `holders` (0:
    /// none). At a point reached that is its set if it is live.
    counted_id: Vec<u32>,
    /// The tracked locals that may have a set counted, each once (`in_counted`).
    counted: Vec<u32>,
    in_counted: Vec<bool>,
    /// The live loans of each local, chained: per local the first one that is not `&mut` and
    /// the first `&mut` one, per loan the next and the one before (`u32::MAX`: none).
    first: Vec<LoanId>,
    next: Vec<LoanId>,
    prev: Vec<LoanId>,
    /// The block being checked, and its number among the blocks checked.
    block: usize,
    block_no: u32,
    /// Per tracked local: whether it is live at the point reached. It holds for the locals
    /// with `known_at == block_no`; any other is not named by the block, and is live in all
    /// of it or in none of it.
    live: Vec<bool>,
    known_at: Vec<u32>,
    /// Per statement of the block being checked: the tracked locals it uses last, as a range
    /// of `dies`, and the tracked local it defines that is used later (`u32::MAX`: none).
    last_uses: Vec<(u32, u32)>,
    dies: Vec<u32>,
    born: Vec<u32>,
}

/// The chain of loan `l`'s local that the loan is in while it is live ([`Cx::first`]).
fn chain(body: &Body, l: usize) -> usize {
    let loan = &body.loans[l];
    2 * loan.place.local as usize + usize::from(loan.mutable && !loan.reserved)
}

impl Cx<'_, '_> {
    /// Loan `l` becomes live: it joins the chain of its local.
    fn link(&mut self, l: usize) {
        let c = chain(self.flow.body, l);
        let head = self.first[c];
        self.next[l] = head;
        self.prev[l] = u32::MAX;
        if head != u32::MAX {
            self.prev[head as usize] = l as LoanId;
        }
        self.first[c] = l as LoanId;
    }

    /// Loan `l` is live no more.
    fn unlink(&mut self, l: usize) {
        let (next, prev) = (self.next[l], self.prev[l]);
        if next != u32::MAX {
            self.prev[next as usize] = prev;
        }
        if prev != u32::MAX {
            self.next[prev as usize] = next;
        } else {
            let c = chain(self.flow.body, l);
            self.first[c] = next;
        }
    }

    /// One more live local holds loan `l`.
    fn hold(&mut self, l: usize) {
        self.counted_loans += 1;
        self.holders[l] += 1;
        if self.holders[l] > 1 {
            return;
        }
        if self.grown && self.leads[l] {
            self.roots.push(l as LoanId);
            self.root_at[l] = self.roots.len() as u32;
            self.stale = true;
        }
        if !(self.grown && self.reached[l]) {
            self.link(l);
        }
    }

    /// One live local less holds loan `l`.
    fn release(&mut self, l: usize) {
        self.counted_loans += 1;
        self.holders[l] -= 1;
        if self.holders[l] > 0 {
            return;
        }
        if self.grown && self.leads[l] {
            let at = self.root_at[l] as usize - 1;
            let last = self.roots.pop().unwrap_or(l as LoanId);
            if at < self.roots.len() {
                self.roots[at] = last;
                self.root_at[last as usize] = at as u32 + 1;
            }
            self.root_at[l] = 0;
            self.stale = true;
        }
        if !(self.grown && self.reached[l]) {
            self.unlink(l);
        }
    }

    /// Finds the loans reached in state `st` through the loans live locals hold: what
    /// [`Flow::reach`] adds to those.
    fn search(&mut self, st: &State) {
        self.stale = false;
        self.rebuilt += 1;
        self.searches += 1;
        let nloans = self.flow.atoms.nloans;
        let mut found: Vec<LoanId> = Vec::new();
        let mut todo = self.roots.clone();
        while let Some(l) = todo.pop() {
            // every loan of a local whose set is read holds that set: one look meets them all
            if let Some(x) = self.flow.read_local(l as usize) {
                if self.read_at[x as usize] == self.searches {
                    continue;
                }
                self.read_at[x as usize] = self.searches;
            }
            let held = self.flow.loan_holds(st, l as usize);
            for a in held.iter().take_while(|&a| a < nloans) {
                self.counted_loans += 1;
                if self.met[a] == self.searches {
                    continue;
                }
                self.met[a] = self.searches;
                found.push(a as LoanId);
                // one with a holder is searched from as a root
                if self.leads[a] && self.holders[a] == 0 {
                    todo.push(a as LoanId);
                }
            }
        }
        // the new ones first: a loan reached before and now stays live
        for &l in &found {
            let l = l as usize;
            if !self.reached[l] {
                self.reached[l] = true;
                if self.holders[l] == 0 {
                    self.link(l);
                }
            }
        }
        for l in std::mem::take(&mut self.reached_list) {
            let l = l as usize;
            if self.met[l] != self.searches {
                self.reached[l] = false;
                if self.holders[l] == 0 {
                    self.unlink(l);
                }
            }
        }
        self.reached_list = found;
    }

    /// Counts the loans of set `id` for tracked local `t` in place of the ones counted for
    /// it (0: none).
    fn count(&mut self, t: u32, id: u32) {
        let old = self.counted_id[t as usize];
        if old == id {
            return;
        }
        let nloans = self.flow.atoms.nloans;
        // the new ones first: a loan both sets hold stays live
        for k in 0..self.flow.sets.get(id).len() {
            let a = self.flow.sets.get(id).atoms()[k] as usize;
            if a >= nloans {
                break;
            }
            self.hold(a);
        }
        for k in 0..self.flow.sets.get(old).len() {
            let a = self.flow.sets.get(old).atoms()[k] as usize;
            if a >= nloans {
                break;
            }
            self.release(a);
        }
        self.counted_id[t as usize] = id;
        if id != 0 && !self.in_counted[t as usize] {
            self.in_counted[t as usize] = true;
            self.counted.push(t);
        }
    }

    /// Whether tracked local `t` is live at the point reached.
    fn alive(&mut self, t: u32) -> bool {
        if self.known_at[t as usize] != self.block_no {
            self.known_at[t as usize] = self.block_no;
            self.live[t as usize] = self.flow.live.live_in(t, self.block as u32);
        }
        self.live[t as usize]
    }

    /// Applies statement `i` of the block being checked to `st` and moves the live locals and
    /// loans to the point after it.
    fn step(&mut self, st: &mut State, s: &Stmt, i: usize) {
        let (lo, hi) = self.last_uses[i];
        let born = self.born[i];
        st.record();
        self.flow.apply(&s.kind, st);
        for k in lo..hi {
            let t = self.dies[k as usize];
            self.live[t as usize] = false;
        }
        if born != u32::MAX {
            self.live[born as usize] = true;
        }
        // the locals whose set or liveness the statement changed
        for k in 0..st.recorded().len() {
            let t = st.recorded()[k].0;
            let id = if self.alive(t) { st.id(t) } else { 0 };
            self.count(t, id);
            // what a loan of this local leads to has changed
            self.stale |= self.grown && self.flow.is_lent(t);
        }
        for k in lo..hi {
            self.count(self.dies[k as usize], 0);
        }
        if born != u32::MAX {
            self.count(born, st.id(born));
        }
    }

    /// Checks the statements of block `b`, entered in state `st`.
    fn block(&mut self, st: &mut State, b: usize) {
        self.block = b;
        self.block_no += 1;
        let flow = &*self.flow;
        let blk = &flow.body.blocks[b];
        let n = blk.stmts.len();
        // backward from the block's end: what each statement ends and starts. A local is first
        // met with its liveness at the end, and has its liveness at the entry after the walk.
        self.dies.clear();
        self.last_uses.clear();
        self.last_uses.resize(n, (0, 0));
        self.born.clear();
        self.born.resize(n, u32::MAX);
        let block_no = self.block_no;
        let (live, known_at) = (&mut self.live, &mut self.known_at);
        for (i, s) in blk.stmts.iter().enumerate().rev() {
            let def = stmt_def(s).map(|d| flow.tracked[d as usize]).filter(|&t| t != u32::MAX);
            let mut after = false;
            if let Some(t) = def {
                if known_at[t as usize] != block_no {
                    known_at[t as usize] = block_no;
                    live[t as usize] = flow.live.live_out(t, b as u32);
                }
                after = live[t as usize];
                live[t as usize] = false;
            }
            let lo = self.dies.len() as u32;
            stmt_uses(s, |l, _| {
                let t = flow.tracked[l as usize];
                if t == u32::MAX {
                    return;
                }
                if known_at[t as usize] != block_no {
                    known_at[t as usize] = block_no;
                    live[t as usize] = flow.live.live_out(t, b as u32);
                }
                if !live[t as usize] {
                    live[t as usize] = true;
                    if !(after && def == Some(t)) {
                        self.dies.push(t);
                    }
                }
            });
            self.last_uses[i] = (lo, self.dies.len() as u32);
            if let Some(t) = def
                && after
                && !live[t as usize]
            {
                self.born[i] = t;
            }
        }
        self.recount(st, b);
        let body = self.flow.body;
        for (i, s) in body.blocks[b].stmts.iter().enumerate() {
            self.pending.clear();
            self.moved_now.clear();
            self.stmt(st, (b as u32, i as u32), s, i);
        }
    }

    /// Makes the counted sets those of the locals live at the entry of block `b`, entered in
    /// state `st`, from what the block checked before left counted.
    fn recount(&mut self, st: &State, b: usize) {
        // a block entered in its entry state: that keeps the sets of the live locals, and of
        // the borrowed locals a loan's contents may be read from
        let fresh = !self.flow.carried(b);
        if !fresh {
            let before = self.flow.preds[b][0] as usize;
            if self.flow.body.blocks[before].term.succs().len() == 1 {
                // the point after the block before: nothing has changed
                return;
            }
        }
        self.stale |= self.grown && fresh;
        let mut k = 0;
        for i in 0..self.counted.len() {
            let t = self.counted[i];
            let live = (fresh && !self.flow.is_lent(t)) || self.alive(t);
            let id = if live { st.id(t) } else { 0 };
            self.count(t, id);
            if id == 0 {
                self.in_counted[t as usize] = false;
            } else {
                self.counted[k] = t;
                k += 1;
            }
        }
        self.counted.truncate(k);
        for i in 0..st.held().len() {
            let t = st.held()[i];
            if self.counted_id[t as usize] == 0 && ((fresh && !self.flow.is_lent(t)) || self.alive(t)) {
                self.count(t, st.id(t));
            }
        }
    }

    /// Checks an access in state `st` against the live loans and the moved places.
    #[allow(clippy::too_many_arguments)]
    fn access(
        &mut self,
        st: &State,
        at: (u32, u32),
        kind: Access,
        place: &Place,
        span: Span,
        node: u32,
        exclude: &[LoanId],
    ) {
        if self.stale {
            self.search(st);
        }
        let mut loans = Vec::new();
        // only a loan of the same local can conflict, and with an access that is not
        // exclusive only a `&mut` one
        for of_mut in [true, false] {
            if !of_mut && !kind.exclusive() {
                break;
            }
            let mut l = self.first[2 * place.local as usize + usize::from(of_mut)];
            while l != u32::MAX {
                if !exclude.contains(&l) && places_conflict(place, &self.flow.loan(l).place, kind) {
                    loans.push(l);
                }
                l = self.next[l as usize];
            }
        }
        loans.sort_unstable();
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
            && (st.is_moved(place.local) || self.moved_now.contains(&place.local))
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

    fn operand(&mut self, st: &State, at: (u32, u32), o: &Operand) {
        let kind = match o.kind {
            OpKind::Const => return,
            OpKind::Copy => Access::Read,
            OpKind::Move => Access::Move,
        };
        self.access(st, at, kind, &o.place, o.span, o.node, &[]);
    }

    /// Checks statement `i` of the block, `s`, and steps `st` over it. Its operands are read
    /// under the loans live before it; an assignment's write happens last, under the loans live
    /// after it. Those are read from the state after the statement: the value the statement
    /// gives its destination may hold a loan that no other live local holds
    /// (`x = V.{ r: &x.f }`: the write of `x` ends the value the new one borrows).
    fn stmt(&mut self, st: &mut State, at: (u32, u32), s: &Stmt, i: usize) {
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
                        self.access(st, at, kind, place, s.span, s.node, &[]);
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
                        for_each_op(rv, |o| {
                            self.operand(st, at, o);
                            if o.kind == OpKind::Move {
                                self.moved_now.push(o.place.local);
                                self.pending.push((o.place.local, o.site));
                            }
                        });
                    }
                }
                // the stores are judged in the state before the statement, and reported after
                // its write
                let stores = self.stores(st, s, dest, rv);
                self.step(st, s, i);
                if !*init {
                    // the write happens after the right side: its moves count (`s.a = f(s)`)
                    let mut moves = Vec::new();
                    for_each_op(rv, |o| {
                        if o.kind == OpKind::Move {
                            moves.push((o.place.local, o.site));
                        }
                    });
                    self.moved_now.extend(moves.iter().map(|m| m.0));
                    self.pending.extend(moves);
                    self.access(st, at, Access::Write, dest, s.span, s.node, &[]);
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
                self.findings.extend(stores);
                return;
            }
            StmtKind::Activate { reserved, active, .. } => {
                let p = self.flow.loan(*reserved).place.clone();
                self.access(st, at, Access::Activate, &p, s.span, s.node, &[*reserved, *active]);
            }
            StmtKind::Read(o) => self.operand(st, at, o),
            StmtKind::Drop(p) => self.access(st, at, Access::Drop, p, s.span, s.node, &[]),
            StmtKind::StorageDead(l) => {
                let p = Place::local(*l, body.locals[*l as usize].ty);
                self.access(st, at, Access::StorageDead, &p, s.span, s.node, &[]);
            }
            StmtKind::Return(o) => {
                self.operand(st, at, o);
                self.ret(st, o);
            }
        }
        self.step(st, s, i);
    }

    /// E0513: the stores of assignment `s` that do not fit, judged in the state `st` before
    /// it (which is left as it was).
    fn stores(&mut self, st: &mut State, s: &Stmt, dest: &Place, rv: &Rvalue) -> Vec<Finding> {
        let body = self.flow.body;
        let mut out = Vec::new();
        // what the callee may store through its `&mut` arguments, per target argument,
        // with the values as they are once all of the call's stores are made
        if let Rvalue::Call { args, stores, .. } = rv
            && !stores.is_empty()
        {
            st.record();
            let (groups, _) = self.flow.settle_call(st, args, stores);
            let mut unfit = Vec::new();
            for CallStore { target, into_view, ptr, val } in groups {
                let Some(arg) = args.get(target as usize) else { continue };
                if let Err(m) = self.flow.store_fits(st, &ptr, &val, None, Some(into_view)) {
                    unfit.push((arg, m, ptr, val));
                }
            }
            st.undo();
            for (arg, m, ptr, val) in unfit {
                // a pointer held in a named local is written through; a temporary is
                // named by its expression
                let mut place = arg.place.clone();
                if !body.locals[place.local as usize].temp {
                    place.proj.push(Proj::Deref(DerefKind::Mut));
                }
                let why = Unfit::of(m, val.is_subset(&self.flow.contents_of(st, &ptr)));
                out.push(Finding::ViewWrite { place, span: s.span, why });
            }
        }
        // a write through a dereference stores into the view objects the pointer may
        // point to; a write into a closure's capture (a reference to an outer local
        // whose origins the outer body fixed when it created the closure) may store only
        // origins the capture already has
        let capture = body.locals[dest.local as usize].capture.is_some();
        if (dest.has_deref() || capture) && self.flow.is_ref_like(dest.ty) && !matches!(rv, Rvalue::Pure(_)) {
            let (val, _) = self.flow.rv_origins(st, rv, dest.ty);
            let why = if dest.has_deref() {
                let (target, ptr) = self.flow.place_origins(st, dest);
                let ptr = ptr.unwrap_or_else(|| self.flow.empty());
                let direct = if dest.derefs() == 1 { body.locals[dest.local as usize].param } else { None };
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
                out.push(Finding::ViewWrite { place: dest.clone(), span: s.span, why });
            }
        }
        out
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
            if !flow.reached(b) {
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
    let (nt, nloans) = (flow.ntracked, flow.atoms.nloans);
    // after a store has grown a local: the loans whose places or contents lead to more loans
    let grown = flow.grown;
    let leads: Vec<bool> = if grown {
        (0..nloans).map(|l| flow.read_local(l).is_some() || !flow.contents[l].is_empty()).collect()
    } else {
        Vec::new()
    };
    let per_loan = if grown { nloans } else { 0 };
    let mut cx = Cx {
        holders: vec![0; nloans],
        grown,
        leads,
        roots: Vec::new(),
        root_at: vec![0; per_loan],
        reached: vec![false; per_loan],
        reached_list: Vec::new(),
        met: vec![0; per_loan],
        read_at: vec![0; if grown { body.locals.len() } else { 0 }],
        searches: 0,
        stale: false,
        counted_loans: 0,
        rebuilt: 0,
        counted_id: vec![0; nt],
        counted: Vec::new(),
        in_counted: vec![false; nt],
        first: vec![u32::MAX; 2 * body.locals.len()],
        next: vec![u32::MAX; nloans],
        prev: vec![u32::MAX; nloans],
        block: 0,
        block_no: 0,
        live: vec![false; nt],
        known_at: vec![0; nt],
        last_uses: Vec::new(),
        dies: Vec::new(),
        born: Vec::new(),
        flow,
        findings: Vec::new(),
        dump: Vec::new(),
        want_dump,
        pending: Vec::new(),
        moved_now: Vec::new(),
    };
    // The walk follows the reverse postorder, where a block without an entry state comes right
    // after the block whose end state it takes; the findings are then put in block order.
    let mut st = cx.flow.state();
    let order = cx.flow.rpo.clone();
    let mut have = false;
    let mut found: Vec<(u32, Vec<Finding>)> = Vec::new();
    let mut dumped: Vec<(u32, Vec<DumpAccess>)> = Vec::new();
    for &b in &order {
        if !cx.flow.carried(b as usize) {
            have = cx.flow.load(b as usize, &mut st);
        }
        if !have {
            continue;
        }
        cx.block(&mut st, b as usize);
        if body.blocks[b as usize].term.succs().len() > 1 {
            // the walk goes on in one successor: what is dead in all of them is not looked
            // at again
            cx.flow.prune(b as usize, &mut st);
        }
        if !cx.findings.is_empty() {
            found.push((b, std::mem::take(&mut cx.findings)));
        }
        if !cx.dump.is_empty() {
            dumped.push((b, std::mem::take(&mut cx.dump)));
        }
    }
    cx.flow.recycle(st);
    cx.flow.cost.counted += cx.counted_loans;
    cx.flow.cost.rebuilt += cx.rebuilt;
    found.sort_by_key(|x| x.0);
    dumped.sort_by_key(|x| x.0);
    let mut findings: Vec<Finding> = found.into_iter().flat_map(|x| x.1).collect();
    bad_moves(body, &mut findings);
    guard_effects(body, &mut findings);
    Checked { findings, dump: dumped.into_iter().flat_map(|x| x.1).collect() }
}

/// E0505: one finding per span of what the body's guards do besides reading.
pub(crate) fn guard_effects(body: &Body, findings: &mut Vec<Finding>) {
    let mut seen: Vec<Span> = Vec::new();
    for (span, what) in &body.guard_effects {
        if seen.contains(span) {
            continue;
        }
        seen.push(*span);
        findings.push(Finding::GuardEffect { span: *span, what: what.clone() });
    }
}

/// E0504: one finding per span of the body's bad moves.
pub(crate) fn bad_moves(body: &Body, findings: &mut Vec<Finding>) {
    let mut seen: Vec<Span> = Vec::new();
    for (place, span, why) in &body.bad_moves {
        if seen.contains(span) {
            continue;
        }
        seen.push(*span);
        findings.push(Finding::MoveOut { place: place.clone(), span: *span, why: why.clone() });
    }
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
    let mut st = flow.state();
    // the state at the end of a block walked, for the successor that has no entry state
    let mut handed: FxHashMap<u32, Saved> = FxHashMap::default();
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
        let ready = match handed.remove(&b) {
            Some(e) if from == 0 => {
                flow.restore(&e, &mut st);
                true
            }
            _ => flow.enter(b as usize, &mut st),
        };
        if !ready {
            continue;
        }
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
            if flow.carried(succ as usize) && !seen[succ as usize] {
                let mut e = Saved::default();
                flow.keep(succ as usize, &st, &mut e);
                handed.insert(succ, e);
            }
            work.push(Reverse((d + len + 1, next, succ, 0)));
        }
    }
    flow.recycle(st);
    best.map(|x| (x.1, x.2))
}

/// The loop whose head the edge `from -> to` goes back to, from inside it (`usize::MAX`: none).
pub(crate) fn back_edge(body: &Body, from: u32, to: u32) -> usize {
    body.loops.iter().position(|&l| to == l.0 && in_loop(l, from)).unwrap_or(usize::MAX)
}

/// Whether block `b` is one of the blocks of loop `(head, exit, end)`.
pub(crate) fn in_loop((head, exit, end): (u32, u32, u32), b: u32) -> bool {
    b >= head && b < end && b != exit
}
