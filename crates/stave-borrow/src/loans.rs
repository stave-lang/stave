//! Loans and origin sets.
//!
//! An *origin atom* is a loan (one per `&`/`&mut` rvalue, [`crate::cfg::Rvalue::Ref`]; a
//! two-phase receiver has a reservation loan and an activation loan) or a parameter origin:
//! `Pb(i)` (what param `i` itself borrows: `from i`) and `Pc(i)` (the origins of the view `*i`
//! points to: `from copy i`). Every tracked local (one whose type is ref-like) holds an origin set
//! per program point; the sets are unioned through copies, moves, aggregates, field assignments,
//! and call results ([`Flows`]: only the named arguments' origins reach the result; all other
//! argument loans end at the call).
//!
//! **Contents.** Reading a ref-like value through `k` dereferences of a local `x` yields
//! `contents^k(O(x))`, where `contents(L)` is the origin set of the borrowed place's value (taken
//! when L is created: the place cannot be written while L lives), `contents(Pb(i)) = {Pc(i)}`,
//! and `contents(Pc(i)) = {Pc(i)}`. A borrow `&P` carries its loan, the origins of the pointer it
//! reborrows through (a reborrow keeps the outer loan live), and the origins of the borrowed
//! value (a view object borrowed keeps its own origins live).
//!
//! **Fixpoint.** Forward dataflow in reverse postorder to a fixpoint (union at joins). The
//! usual bound (loop depth + 1 rounds) holds for gen/kill problems; copies between locals
//! inside a loop can need more, so the iteration runs to the fixpoint (monotone, finite) and the
//! round count is reported ([`Flow::rounds`]).

use rustc_hash::FxHashMap;

use crate::bits::Bits;
use crate::cfg::{Body, LoanId, LocalId, OpKind, Operand, Place, Proj, Rvalue, StmtKind};
use crate::ty::{DerefKind, Flows, Tys};
use stave_sema::types::TypeId;

/// Origin atoms of one body: loans first, then two per param.
#[derive(Clone, Copy, Debug)]
pub struct Atoms {
    pub nloans: usize,
    pub nparams: usize,
}

impl Atoms {
    pub fn len(&self) -> usize {
        self.nloans + 2 * self.nparams
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn pb(&self, i: u32) -> usize {
        self.nloans + 2 * i as usize
    }

    pub fn pc(&self, i: u32) -> usize {
        self.nloans + 2 * i as usize + 1
    }

    /// The param atom `a` as (param, `copy` origin).
    pub fn param(&self, a: usize) -> Option<(u32, bool)> {
        let k = a.checked_sub(self.nloans)?;
        Some(((k / 2) as u32, k % 2 == 1))
    }
}

/// The dataflow state at a point: origin sets of tracked locals, maybe-moved sites.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct State {
    pub o: Vec<Bits>,
    pub moved: Bits,
}

impl State {
    pub fn union_with(&mut self, other: &State) -> bool {
        let mut ch = false;
        for (a, b) in self.o.iter_mut().zip(&other.o) {
            ch |= a.union_with(b);
        }
        ch | self.moved.union_with(&other.moved)
    }
}

/// The solved dataflow of one body.
pub struct Flow<'b> {
    pub body: &'b Body,
    pub atoms: Atoms,
    /// Per local: its tracked index or `u32::MAX`.
    pub tracked: Vec<u32>,
    pub ntracked: usize,
    /// Per loan: the origin set of the borrowed value.
    pub contents: Vec<Bits>,
    /// Ref-likeness of every type the body's places use.
    pub ref_like: FxHashMap<TypeId, bool>,
    /// Block entry states (`None`: unreachable).
    pub entry: Vec<Option<State>>,
    pub preds: Vec<Vec<u32>>,
    pub rpo: Vec<u32>,
    pub rounds: u32,
}

impl<'b> Flow<'b> {
    pub fn new(body: &'b Body, tys: &mut Tys<'_>) -> Flow<'b> {
        let nparams = body.n_params();
        let atoms = Atoms { nloans: body.loans.len(), nparams };
        let mut ref_like: FxHashMap<TypeId, bool> = FxHashMap::default();
        // Notes type `t` and returns its ref-likeness. Consecutive locals and places mostly have
        // the same type: the one noted last is not looked up again.
        let mut last: Option<(TypeId, bool)> = None;
        let mut note = |t: TypeId, tys: &mut Tys<'_>| -> bool {
            if let Some((lt, r)) = last
                && lt == t
            {
                return r;
            }
            let r = *ref_like.entry(t).or_insert_with(|| tys.ref_like(t));
            last = Some((t, r));
            r
        };
        let mut tracked = vec![u32::MAX; body.locals.len()];
        let mut ntracked = 0usize;
        for (i, l) in body.locals.iter().enumerate() {
            if note(l.ty, tys) {
                tracked[i] = ntracked as u32;
                ntracked += 1;
            }
        }
        for b in &body.blocks {
            for s in &b.stmts {
                for_each_place(&s.kind, |p| {
                    note(p.ty, tys);
                });
            }
        }
        for l in &body.loans {
            note(l.place.ty, tys);
        }
        let n = body.blocks.len();
        let mut preds: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (i, b) in body.blocks.iter().enumerate() {
            for &s in b.term.succs() {
                if let Some(p) = preds.get_mut(s as usize) {
                    p.push(i as u32);
                }
            }
        }
        let rpo = rpo(body);
        Flow {
            body,
            atoms,
            tracked,
            ntracked,
            contents: vec![Bits::new(atoms.len()); body.loans.len()],
            ref_like,
            entry: vec![None; n],
            preds,
            rpo,
            rounds: 0,
        }
    }

    pub fn is_ref_like(&self, t: TypeId) -> bool {
        self.ref_like.get(&t).copied().unwrap_or(true)
    }

    pub fn empty(&self) -> Bits {
        Bits::new(self.atoms.len())
    }

    fn initial(&self) -> State {
        let mut o = vec![self.empty(); self.ntracked];
        for (i, l) in self.body.locals.iter().enumerate() {
            if let (Some(p), t) = (l.param, self.tracked[i])
                && t != u32::MAX
            {
                o[t as usize].insert(self.atoms.pb(p));
            }
        }
        State { o, moved: Bits::new(self.body.move_sites.len()) }
    }

    /// Whether the body has no state to track: no ref-like local, no loan, no move site. Then no
    /// statement changes the dataflow state (origin sets belong to tracked locals, loan contents
    /// to loans, moved sites to move operands), so every reachable block starts in the initial
    /// state, every origin set read from it is empty, and no access conflicts with a loan or
    /// uses a moved place.
    pub fn stateless(&self) -> bool {
        self.ntracked == 0 && self.body.loans.is_empty() && self.body.move_sites.is_empty()
    }

    /// Solves the forward dataflow.
    pub fn solve(&mut self) {
        if self.body.blocks.is_empty() {
            return;
        }
        if self.stateless() {
            // The fixpoint without iterating: the first round reaches every block of the reverse
            // postorder, and a second one finds nothing new if it set any entry but block 0's.
            let init = self.initial();
            for &b in &self.rpo {
                self.entry[b as usize] = Some(init.clone());
            }
            self.rounds = if self.rpo.len() > 1 { 2 } else { 1 };
            return;
        }
        self.entry[0] = Some(self.initial());
        let order = self.rpo.clone();
        for round in 1..=100_000u32 {
            self.rounds = round;
            let mut changed = false;
            for &b in &order {
                let Some(mut st) = self.entry[b as usize].clone() else { continue };
                for s in &self.body.blocks[b as usize].stmts {
                    changed |= self.apply(&s.kind, &mut st);
                }
                for &succ in self.body.blocks[b as usize].term.succs() {
                    match &mut self.entry[succ as usize] {
                        Some(e) => changed |= e.union_with(&st),
                        slot @ None => {
                            *slot = Some(st.clone());
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    pub fn local_o(&self, st: &State, l: LocalId) -> Bits {
        match self.tracked.get(l as usize) {
            Some(&t) if t != u32::MAX => st.o[t as usize].clone(),
            _ => self.empty(),
        }
    }

    /// `contents(S)`.
    pub fn contents_of(&self, s: &Bits) -> Bits {
        let mut out = self.empty();
        for a in s.iter() {
            if a < self.atoms.nloans {
                out.union_with(&self.contents[a]);
            } else if let Some((i, _)) = self.atoms.param(a) {
                out.insert(self.atoms.pc(i));
            }
        }
        out
    }

    /// The origins of the value stored at `p` (after all its dereferences), and of the pointer
    /// of its last dereference (`None` without one).
    pub fn place_origins(&self, st: &State, p: &Place) -> (Bits, Option<Bits>) {
        let mut cur = self.local_o(st, p.local);
        let mut ptr = None;
        for pr in &p.proj {
            if let Proj::Deref(k) = pr {
                if *k == DerefKind::Raw {
                    return (self.empty(), Some(self.empty()));
                }
                ptr = Some(cur.clone());
                cur = self.contents_of(&cur);
            }
        }
        (cur, ptr)
    }

    pub fn op_origins(&self, st: &State, op: &Operand) -> Bits {
        if op.kind == OpKind::Const || !self.is_ref_like(op.place.ty) {
            return self.empty();
        }
        self.place_origins(st, &op.place).0
    }

    /// The origin set of an rvalue; `contents` of a created loan is recorded.
    pub fn rv_origins(&mut self, st: &State, rv: &Rvalue, dest_ty: TypeId) -> (Bits, bool) {
        let mut changed = false;
        let s = match rv {
            Rvalue::Use(op) => self.op_origins(st, op),
            Rvalue::Ref { place, loan, .. } => {
                let mut s = self.empty();
                s.insert(*loan as usize);
                let (val, ptr) = self.place_origins(st, place);
                if let Some(p) = ptr {
                    s.union_with(&p);
                }
                if self.is_ref_like(place.ty) {
                    s.union_with(&val);
                    if let Some(c) = self.contents.get_mut(*loan as usize) {
                        changed |= c.union_with(&val);
                    }
                }
                s
            }
            Rvalue::Agg(ops) => {
                let mut s = self.empty();
                for o in ops {
                    s.union_with(&self.op_origins(st, o));
                }
                s
            }
            Rvalue::Call { args, flows } => match flows {
                Flows::None => self.empty(),
                Flows::AllArgs => {
                    let mut s = self.empty();
                    for o in args {
                        s.union_with(&self.op_origins(st, o));
                    }
                    s
                }
                Flows::Named(v) => {
                    let mut s = self.empty();
                    for &(i, copy) in v {
                        let Some(o) = args.get(i as usize) else { continue };
                        let x = self.op_origins(st, o);
                        if copy {
                            s.union_with(&self.contents_of(&x));
                        } else {
                            s.union_with(&x);
                        }
                    }
                    s
                }
            },
            Rvalue::Pure(_) => self.empty(),
        };
        if !self.is_ref_like(dest_ty) {
            return (self.empty(), changed);
        }
        (s, changed)
    }

    /// Applies statement `k` to `st`; whether a loan's contents changed.
    pub fn apply(&mut self, k: &StmtKind, st: &mut State) -> bool {
        let mut changed = false;
        match k {
            StmtKind::Assign { dest, rv, .. } => {
                let (s, ch) = self.rv_origins(st, rv, dest.ty);
                changed |= ch;
                for_each_op(rv, |o| {
                    if o.kind == OpKind::Move {
                        st.moved.insert(o.site as usize);
                    }
                });
                // re-initialized: moves of the place or below it are undone
                self.reinit(st, dest);
                if let Some(&t) = self.tracked.get(dest.local as usize)
                    && t != u32::MAX
                {
                    if dest.proj.is_empty() {
                        st.o[t as usize] = s;
                    } else if !dest.has_deref() {
                        st.o[t as usize].union_with(&s);
                    }
                }
            }
            StmtKind::Activate { local, reserved, active } => {
                if let Some(&t) = self.tracked.get(*local as usize)
                    && t != u32::MAX
                {
                    let o = &mut st.o[t as usize];
                    if o.contains(*reserved as usize) {
                        o.remove(*reserved as usize);
                        o.insert(*active as usize);
                    }
                }
                let c = self.contents.get(*reserved as usize).cloned();
                if let (Some(c), Some(d)) = (c, self.contents.get_mut(*active as usize)) {
                    changed |= d.union_with(&c);
                }
            }
            StmtKind::StorageDead(l) => {
                if let Some(&t) = self.tracked.get(*l as usize)
                    && t != u32::MAX
                {
                    st.o[t as usize].clear();
                }
                let p = Place::local(*l, TypeId::NONE);
                self.reinit(st, &p);
            }
            StmtKind::Return(o) | StmtKind::Read(o) => {
                if o.kind == OpKind::Move {
                    st.moved.insert(o.site as usize);
                }
            }
            StmtKind::Drop(_) => {}
        }
        changed
    }

    fn reinit(&self, st: &mut State, dest: &Place) {
        if st.moved.is_empty() {
            return;
        }
        let ids: Vec<usize> = st.moved.iter().collect();
        for i in ids {
            let Some((m, _)) = self.body.move_sites.get(i) else { continue };
            if m.local == dest.local && m.proj.starts_with(&dest.proj) {
                st.moved.remove(i);
            }
        }
    }

    /// The loans in the origin sets of the given live locals.
    pub fn live_loans(&self, st: &State, live: &Bits) -> Bits {
        let mut out = self.empty();
        for t in live.iter() {
            if let Some(o) = st.o.get(t) {
                out.union_with(o);
            }
        }
        out
    }

    pub fn loan(&self, l: LoanId) -> &crate::cfg::Loan {
        &self.body.loans[l as usize]
    }
}

/// Visits every operand of an rvalue.
pub fn for_each_op(rv: &Rvalue, mut f: impl FnMut(&Operand)) {
    match rv {
        Rvalue::Use(o) => f(o),
        Rvalue::Agg(v) | Rvalue::Call { args: v, .. } | Rvalue::Pure(v) => v.iter().for_each(f),
        Rvalue::Ref { .. } => {}
    }
}

/// Visits every place a statement names.
pub fn for_each_place(k: &StmtKind, mut f: impl FnMut(&Place)) {
    match k {
        StmtKind::Assign { dest, rv, .. } => {
            f(dest);
            if let Rvalue::Ref { place, .. } = rv {
                f(place);
            }
            for_each_op(rv, |o| {
                if o.kind != OpKind::Const {
                    f(&o.place)
                }
            });
        }
        StmtKind::Read(o) | StmtKind::Return(o) => {
            if o.kind != OpKind::Const {
                f(&o.place)
            }
        }
        StmtKind::Drop(p) => f(p),
        StmtKind::Activate { .. } | StmtKind::StorageDead(_) => {}
    }
}

/// Reverse postorder of the reachable blocks from block 0.
fn rpo(body: &Body) -> Vec<u32> {
    let n = body.blocks.len();
    let mut seen = vec![false; n];
    let mut post = Vec::with_capacity(n);
    let mut stack: Vec<(u32, usize)> = vec![(0, 0)];
    if n == 0 {
        return post;
    }
    seen[0] = true;
    while let Some(&mut (b, ref mut i)) = stack.last_mut() {
        let succs = body.blocks[b as usize].term.succs();
        if *i < succs.len() {
            let s = succs[*i];
            *i += 1;
            if !seen[s as usize] {
                seen[s as usize] = true;
                stack.push((s, 0));
            }
        } else {
            post.push(b);
            stack.pop();
        }
    }
    post.reverse();
    post
}
