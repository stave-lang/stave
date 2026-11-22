//! The dense dataflow's origin sets: the model of [`crate::loans`] with one full state (a
//! bitset over all atoms per tracked local, a bitset of moved locals) per block.

use rustc_hash::FxHashMap;

use crate::bits::Bits;
use crate::cfg::{Body, LoanId, LocalId, OpKind, Operand, Place, Proj, Rvalue, StmtKind, Store};
use crate::check::{Access, places_conflict};
use crate::loans::{Atoms, Misfit, for_each_op, for_each_place};
use crate::ty::{DerefKind, Flows, Tys};
use stave_sema::types::TypeId;

/// The dataflow state at a point: origin sets of tracked locals, maybe-moved locals (a value
/// moves whole: one bit per local).
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

/// The store a call makes through one target argument ([`Flow::call_stores`]).
#[derive(Clone, Debug)]
pub struct CallStore {
    /// The target's operand index.
    pub target: u32,
    /// Whether the target points to a ref-like type at this instance.
    pub into_view: bool,
    /// The target pointer's origins (empty when it points to no ref-like type).
    pub ptr: Bits,
    /// The value's origins: the union over the clause's entries. One set per target, because
    /// the callee may store what any source borrows through any other source it keeps.
    pub val: Bits,
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
    /// Per loan: the origins of the pointers it was made through (empty for a loan of a place
    /// without a dereference). While a value holds the loan, those keep the borrowed place from
    /// being reached another way.
    pub made: Vec<Bits>,
    /// Ref-likeness of every type the body's places use.
    pub ref_like: FxHashMap<TypeId, bool>,
    /// Ref-like params whose type is a reference: `Pb` names one object.
    pub ref_param: Bits,
    /// Ref-like params that reach a `&mut` to a ref-like type through a reference: `Pc` may be
    /// a pointer a store goes through.
    pub deep_param: Bits,
    /// Ref-like params that are or hold a `&mut` to a ref-like type: `Pb` may be a pointer a
    /// store goes through.
    pub store_param: Bits,
    /// Params that take a `from` clause: `&mut` to a ref-like type, in a fn body.
    pub clause_param: Bits,
    /// Per param: the atoms its `from` clause covers.
    pub decl: Vec<Bits>,
    /// Whether a call's store added origins to a loan's contents: only then can a loan's
    /// contents hold what the locals holding the loan do not.
    pub grown: bool,
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
        let mut ref_param = Bits::new(nparams);
        let mut deep_param = Bits::new(nparams);
        let mut store_param = Bits::new(nparams);
        let mut clause_param = Bits::new(nparams);
        for (i, l) in body.locals.iter().enumerate() {
            let Some(p) = l.param else { continue };
            if tracked[i] == u32::MAX || p as usize >= nparams {
                continue;
            }
            if !tys.is_unknown(l.ty) && tys.deref(l.ty).is_some_and(|(_, k)| k != DerefKind::Raw) {
                ref_param.insert(p as usize);
            }
            if tys.holds_inner_mut(l.ty) {
                deep_param.insert(p as usize);
            }
            if tys.holds_mut(l.ty) {
                store_param.insert(p as usize);
            }
            if !body.is_nested
                && !tys.is_unknown(l.ty)
                && let Some((inner, DerefKind::Mut)) = tys.deref(l.ty)
                && tys.ref_like(inner)
            {
                clause_param.insert(p as usize);
            }
        }
        let mut decl = vec![Bits::new(atoms.len()); nparams];
        for s in &body.declared {
            if s.target as usize >= nparams || s.source as usize >= nparams {
                continue;
            }
            let d = &mut decl[s.target as usize];
            if !s.copy {
                d.insert(atoms.pb(s.source));
            }
            d.insert(atoms.pc(s.source));
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
            made: vec![Bits::new(atoms.len()); body.loans.len()],
            ref_like,
            ref_param,
            deep_param,
            store_param,
            clause_param,
            decl,
            grown: false,
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
        State { o, moved: Bits::new(self.body.locals.len()) }
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

    /// The origins of the value in the place loan `l` borrows: the set taken when the loan was
    /// made, and once a store has grown a local, for a borrowed local what its set says in `st`
    /// (a store adds to it from the store on: what is stored later on the path is not in the
    /// place yet).
    fn loan_holds(&self, st: &State, l: usize) -> Bits {
        let place = &self.body.loans[l].place;
        if self.grown && !place.has_deref() && self.is_ref_like(place.ty) {
            self.local_o(st, place.local)
        } else {
            self.contents[l].clone()
        }
    }

    /// `contents(S)` in state `st`.
    pub fn contents_of(&self, st: &State, s: &Bits) -> Bits {
        let mut out = self.empty();
        for a in s.iter() {
            if a < self.atoms.nloans {
                out.union_with(&self.loan_holds(st, a));
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
                cur = self.contents_of(st, &cur);
            }
        }
        (cur, ptr)
    }

    /// `s` closed under what its loans' places hold: the origins of everything a value with
    /// origins `s` reaches in state `st`. A borrowed local holds what its set says in `st` (a
    /// store adds to it from the store on, so a borrow stored later on the path is not reached
    /// yet); a place behind a dereference, what it held when it was borrowed.
    pub fn reach(&self, st: &State, s: &Bits) -> Bits {
        let mut out = s.clone();
        if !self.grown {
            // every borrow carries the origins of the value it borrows: `s` is closed
            return out;
        }
        let nloans = self.atoms.nloans;
        let mut todo: Vec<usize> = s.iter().filter(|&a| a < nloans).collect();
        while let Some(l) = todo.pop() {
            for a in self.loan_holds(st, l).iter() {
                if out.insert(a) && a < nloans {
                    todo.push(a);
                }
            }
        }
        out
    }

    /// Whether the clause of some `&mut` param names param `i` and covers the origins `held`:
    /// the caller's store through that param's argument then grows every view the argument of
    /// `i` reaches, with at least `held`.
    fn covered(&self, i: u32, held: &Bits) -> bool {
        let pb = self.atoms.pb(i);
        self.clause_param.iter().any(|t| self.decl[t].contains(pb) && held.is_subset(&self.decl[t]))
    }

    /// The stores of a call with operands `args`, one per target argument whose clause carries
    /// an origin, with the sets of state `st`.
    pub fn call_stores(&self, st: &State, args: &[Operand], stores: &[Store]) -> Vec<CallStore> {
        let mut out: Vec<CallStore> = Vec::new();
        for s in stores {
            let (Some(t), Some(q)) = (args.get(s.target as usize), args.get(s.source as usize)) else {
                continue;
            };
            let mut val = self.op_origins(st, q);
            if s.copy {
                val = self.contents_of(st, &val);
            }
            if val.is_empty() {
                continue;
            }
            if out.last().is_none_or(|last| last.target != s.target) {
                let ptr = if s.into_view { self.op_origins(st, t) } else { self.empty() };
                out.push(CallStore { target: s.target, into_view: s.into_view, ptr, val: self.empty() });
            }
            if let Some(last) = out.last_mut() {
                last.val.union_with(&val);
            }
        }
        out
    }

    /// Whether a value with origins `val` may be stored through a pointer with origins `ptr`
    /// (module doc, **Stores**). `direct`: the param whose own dereference is the only one of
    /// the place an assignment writes. `call`: the store of a call, which also reaches what
    /// its sources reach; `Some(false)`: its target argument points to a type that holds no
    /// borrow at this instance, so only what the sources reach can take the store. The local
    /// objects are grown by [`Flow::apply`].
    pub fn store_fits(
        &self,
        st: &State,
        ptr: &Bits,
        val: &Bits,
        direct: Option<u32>,
        call: Option<bool>,
    ) -> Result<(), Misfit> {
        if val.is_empty() {
            return Ok(());
        }
        let into_view = call != Some(false);
        let call = call.is_some();
        let body = self.body;
        let pointed = self.reach(st, ptr);
        // a callee may keep a pointer it is given as a source and store through it too
        let mut targets = pointed.clone();
        if call {
            targets.union_with(&self.reach(st, val));
        }
        // what a caller's object must account for: the passive loans keep nothing in this body
        // from changing, and what they borrow from is in the value's set beside them
        let mut held = val.clone();
        for m in val.iter().filter(|&m| self.passive(m)) {
            held.remove(m);
        }
        let mut allowed = self.contents_of(st, &pointed);
        let mut object = false;
        // whether an atom names a view object the pointer may point to
        let mut placed = false;
        // the one param whose clause lacks origins, with them
        let mut want: Option<(u32, Bits)> = None;
        for a in targets.iter() {
            if let Some(l) = body.loans.get(a) {
                if !l.mutable || l.place.has_deref() || !self.is_ref_like(l.place.ty) {
                    continue;
                }
                placed |= pointed.contains(a);
                let captured = body.locals[l.place.local as usize].capture.is_some();
                if captured && !val.is_subset(&self.contents[a]) {
                    object = true;
                }
            } else if let Some((i, copy)) = self.atoms.param(a) {
                let iu = i as usize;
                let deep = self.deep_param.contains(iu);
                let through = if copy { deep } else { self.store_param.contains(iu) };
                if !through {
                    continue;
                }
                placed |= pointed.contains(a);
                if self.covered(i, &held) {
                    continue;
                }
                if !copy && !self.ref_param.contains(iu) {
                    object = true;
                    continue;
                }
                allowed.union_with(&self.decl[iu]);
                let mut missing = held.clone();
                missing.subtract(&self.decl[iu]);
                if !copy && (!deep || direct == Some(i)) {
                    missing.remove(self.atoms.pc(i));
                }
                if missing.is_empty() {
                    continue;
                }
                let fixable = self.clause_param.contains(iu)
                    && missing.iter().all(|m| self.atoms.param(m).is_some_and(|(q, _)| q != i));
                want = match want.take() {
                    Some((p, mut m)) if fixable && p == i => {
                        m.union_with(&missing);
                        Some((p, m))
                    }
                    None if fixable => Some((i, missing)),
                    _ => {
                        object = true;
                        None
                    }
                };
            }
        }
        if object {
            return Err(Misfit::Object);
        }
        if let Some((param, missing)) = want {
            return Err(Misfit::Clause { param, add: self.clause_entries(&missing) });
        }
        // a pointer none of whose origins names a view object (a raw dereference first, a result
        // `from` a shared anchor): nothing is grown for it, and nothing it points to is known to
        // hold the value's origins
        let unplaced = !placed && into_view && (call || !held.is_subset(&allowed));
        if unplaced {
            return Err(Misfit::Object);
        }
        Ok(())
    }

    /// Whether loan `a` is passive: a `&` loan of a place whose last dereference is of a `&`,
    /// with no raw dereference. Nothing is written through a `&`, so the loan conflicts with no
    /// access the body can make; the value it is in borrows what that `&` pointer borrows, and
    /// the pointer's origins are in the value's set with the loan (module doc, **Contents**).
    fn passive(&self, a: usize) -> bool {
        self.body.loans.get(a).is_some_and(|l| {
            let mut last = None;
            for p in &l.place.proj {
                match p {
                    Proj::Deref(DerefKind::Raw) => return false,
                    Proj::Deref(k) => last = Some(*k),
                    _ => {}
                }
            }
            !l.mutable && !l.reserved && last == Some(DerefKind::Shared)
        })
    }

    /// The `from` entries (param, `copy`) that cover the param atoms `missing`.
    fn clause_entries(&self, missing: &Bits) -> Vec<(u32, bool)> {
        let mut out: Vec<(u32, bool)> = Vec::new();
        for m in missing.iter() {
            let Some((q, copy)) = self.atoms.param(m) else { continue };
            // `Pb(q)` comes before `Pc(q)`, and `from q` covers both
            if out.last().is_some_and(|&(p, _)| p == q) {
                continue;
            }
            out.push((q, copy && self.ref_param.contains(q as usize)));
        }
        out
    }

    /// Applies the stores of a call with operands `args` to `st` until nothing changes, and
    /// returns them as they are then ([`Flow::call_stores`]), with whether a loan's contents
    /// changed. A store into one target can feed the `from copy` entry of another, in either
    /// order of the params, and the callee reads a `from copy` source after its own stores:
    /// the values are those of the state after all of them. Sets only grow, so this ends.
    pub fn settle_call(
        &mut self,
        st: &mut State,
        args: &[Operand],
        stores: &[Store],
    ) -> (Vec<CallStore>, bool) {
        let mut changed = false;
        loop {
            let groups = self.call_stores(st, args, stores);
            let mut again = false;
            for g in &groups {
                let (contents, state) = self.grow(st, &g.ptr, &g.val, true);
                changed |= contents;
                again |= contents | state;
            }
            if !again {
                return (groups, changed);
            }
        }
    }

    /// A store of a value with origins `val` through a pointer with origins `ptr`: every local
    /// view object the pointer may point to or reach gains `val` (module doc, **Stores**).
    /// `call`: the store of a call, which also reaches what its sources reach. Whether a loan's
    /// contents changed, and whether a local's set did.
    fn grow(&mut self, st: &mut State, ptr: &Bits, val: &Bits, call: bool) -> (bool, bool) {
        let body = self.body;
        let mut targets = self.reach(st, ptr);
        if call {
            // the callee may keep a pointer it is given as a source and store through it too
            targets.union_with(&self.reach(st, val));
        }
        let (mut contents, mut state) = (false, false);
        for a in targets.iter() {
            let Some(l) = body.loans.get(a) else { continue };
            // a place behind a dereference is read with the sets of the pointers it was
            // borrowed through, which are targets with it
            if !l.mutable
                || l.place.has_deref()
                || !self.is_ref_like(l.place.ty)
                || body.locals[l.place.local as usize].capture.is_some()
            {
                continue;
            }
            let mut add = val.clone();
            for m in val.iter() {
                if self.cannot_hold(a, m, val) {
                    add.remove(m);
                }
            }
            if self.contents[a].union_with(&add) {
                contents = true;
                self.grown = true;
            }
            if let Some(&t) = self.tracked.get(l.place.local as usize)
                && t != u32::MAX
            {
                state |= st.o[t as usize].union_with(&add);
            }
        }
        (contents, state)
    }

    /// Whether the local written through loan `target` cannot come to hold loan `m` of a value
    /// with origins `val`. `m` must be another loan of the same local, without a dereference,
    /// of a place that overlaps the target's (a loan of a disjoint part is kept: the local then
    /// borrows from itself, as after `o.a = &o.b`): both are used by the store, so both are
    /// live there, and two live loans of overlapping places, one of them `&mut`, are a conflict
    /// where the later one is made. In a body without that error the value holds no reference
    /// made by `m`: it is in the set because sets are unions.
    ///
    /// The target loan itself is always kept. Pointers into disjoint parts of the local can
    /// share it (a callee returned both `from` one `&mut`; a callee given the borrow as a
    /// source moved it into a helper that split it), and one may be stored through the other.
    ///
    /// And never when `val` holds a loan made through a pointer with `m` in its set
    /// ([`Flow::made`]), other than the reborrow of the temporary that holds exactly `m`: such
    /// a reference may point into the local, beside the place written, and only `m` keeps the
    /// local from being used directly while it is held.
    fn cannot_hold(&self, target: usize, m: usize, val: &Bits) -> bool {
        let loans = &self.body.loans;
        let (Some(l), Some(x)) = (loans.get(target), loans.get(m)) else { return false };
        if m == target
            || x.place.local != l.place.local
            || x.place.has_deref()
            || !places_conflict(&x.place, &l.place, Access::MutBorrow)
        {
            return false;
        }
        !val.iter().any(|d| {
            d != m
                && loans.get(d).is_some_and(|y| y.place.has_deref() && y.whole != Some(m as LoanId))
                && self.made[d].contains(m)
        })
    }

    /// The origins of the pointers a borrow of `p` goes through (module doc, **Contents**): of
    /// every dereference after the last `&` one, and of that one.
    fn through(&self, st: &State, p: &Place) -> Bits {
        let mut cur = self.local_o(st, p.local);
        let mut out = self.empty();
        for pr in &p.proj {
            if let Proj::Deref(k) = pr {
                match k {
                    DerefKind::Raw => return self.empty(),
                    DerefKind::Shared => out = cur.clone(),
                    DerefKind::Mut => {
                        out.union_with(&cur);
                    }
                }
                cur = self.contents_of(st, &cur);
            }
        }
        out
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
                let through = if place.derefs() > 1 { Some(self.through(st, place)) } else { ptr };
                if let Some(p) = through {
                    s.union_with(&p);
                    if let Some(m) = self.made.get_mut(*loan as usize) {
                        changed |= m.union_with(&p);
                    }
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
            Rvalue::Call { args, flows, .. } => match flows {
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
                            s.union_with(&self.contents_of(st, &x));
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
                // a call's stores come first: its result, like the callee's reads, sees them
                if let Rvalue::Call { args, stores, .. } = rv
                    && !stores.is_empty()
                {
                    changed |= self.settle_call(st, args, stores).1;
                }
                let (s, ch) = self.rv_origins(st, rv, dest.ty);
                changed |= ch;
                for_each_op(rv, |o| {
                    if o.kind == OpKind::Move {
                        st.moved.insert(o.place.local as usize);
                    }
                });
                // re-initialized: moves of the place or below it are undone
                self.reinit(st, dest);
                if dest.has_deref() {
                    if !s.is_empty()
                        && let (_, Some(ptr)) = self.place_origins(st, dest)
                    {
                        changed |= self.grow(st, &ptr, &s, false).0;
                    }
                } else if let Some(&t) = self.tracked.get(dest.local as usize)
                    && t != u32::MAX
                {
                    if dest.proj.is_empty() {
                        st.o[t as usize] = s;
                    } else {
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
                // the activation loan is the reservation's borrow, made through the same pointers
                let m = self.made.get(*reserved as usize).cloned();
                if let (Some(m), Some(d)) = (m, self.made.get_mut(*active as usize)) {
                    changed |= d.union_with(&m);
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
                    st.moved.insert(o.place.local as usize);
                }
            }
            StmtKind::Drop(_) => {}
        }
        changed
    }

    /// A write of a whole local initializes it again; a write of a part of a moved local is a
    /// use of it (E0502).
    fn reinit(&self, st: &mut State, dest: &Place) {
        if dest.proj.is_empty() {
            st.moved.remove(dest.local as usize);
        }
    }

    /// The loans the given live locals reach: those in their origin sets, and after a call's
    /// store those in the contents of such loans.
    pub fn live_loans(&self, st: &State, live: &Bits) -> Bits {
        let mut out = self.empty();
        for t in live.iter() {
            if let Some(o) = st.o.get(t) {
                out.union_with(o);
            }
        }
        if self.grown { self.reach(st, &out) } else { out }
    }

    /// Whether local `l` reaches loan `loan` ([`Flow::live_loans`]).
    pub fn holds(&self, st: &State, l: LocalId, loan: LoanId) -> bool {
        let o = self.local_o(st, l);
        o.contains(loan as usize) || (self.grown && self.reach(st, &o).contains(loan as usize))
    }

    pub fn loan(&self, l: LoanId) -> &crate::cfg::Loan {
        &self.body.loans[l as usize]
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
