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
//! and `contents(Pc(i)) = {Pc(i)}`. A borrow `&P` carries its loan, the origins of the pointers
//! it reborrows through (a reborrow keeps the outer loans live), and the origins of the borrowed
//! value (a view object borrowed keeps its own origins live). The pointers are those of every
//! dereference of `P` after its last `&` one, and that one: what a `&mut` points to is reached
//! only through the place holding the `&mut`, so the borrow must keep that place borrowed too
//! (else a second borrow made through it would alias the first); a `&` can be copied out of its
//! place, so nothing before it is needed.
//!
//! **Stores.** A store puts a ref-like value with origins `S` into a view object behind a
//! pointer with origins `P`: an assignment through a dereference, or a call whose callee has a
//! `from` clause on a `&mut` param ([`Store`]: `P` is the target argument's set, `S` the union
//! over the clause's entries of the source argument's set, or of its `contents` for `from
//! copy`). The objects the pointer may point to or reach are named by `reach(P)`
//! ([`Flow::reach`]: `P` closed under what the places of its loans hold at the store), and for
//! a call by `reach(S)` too: the callee may keep a pointer it is given as a source and store
//! through it. Each of them must end up with `S` in the sets its readers use
//! ([`Flow::store_fits`]). `S` empty: nothing to do. Else, for every such atom:
//! - a `&mut` loan `L` of a ref-like place without a dereference, at a call: the set of the
//!   borrowed local gains `S` from the call on, and so does `contents(L)` ([`Flow::apply`];
//!   without the loans of that local itself: it is written through `L`, so no other loan of it
//!   is live and it holds no borrow of itself). The local is read later with that set, and
//!   through a pointer with what the local's set says where it is read. A place behind a
//!   dereference is read with the sets of the pointers it was borrowed through, which are in
//!   the pointer's set with it: nothing is added for it. A captured local of a nested body
//!   belongs to the enclosing fn, which learns nothing from the closure: it is not grown, and
//!   takes the store only if `S ⊆ contents(L)`;
//! - the same loan, at an assignment: `S ⊆ contents(L)` is required, and the borrowed local's
//!   set gains `S`: `contents(L)` may hold what a call stored on another path;
//! - `Pb(i)`, param `i` a reference whose type can be stored through ([`Tys::holds_mut`]; a
//!   param of another type is in `P` only as the origins of a value): `S ⊆ decl(i) ∪ {Pc(i)}`,
//!   with `decl(i)` the atoms the `from` clause of param `i` covers (`from q`: `Pb(q)` and
//!   `Pc(q)`; `from copy q`: `Pc(q)`). The caller adds the declared origins to the object
//!   (the rule above, applied to its call), and a value with origin `Pc(i)` was read from that
//!   object or from what it reaches. A by-value view or a capture that can be stored through
//!   may hold pointers to several objects of the caller under the one atom and takes no clause:
//!   nothing with origins is stored through it;
//! - if the type of param `i` holds a `&mut` to a ref-like type below a reference
//!   ([`Tys::holds_inner_mut`]): for `Pc(i)`, and for `Pb(i)` unless the place written is `*i`
//!   itself with no other dereference (a call's argument never counts as that), `S ⊆ decl(i)`.
//!   A pointer found inside what a param reaches points to a caller object whose own set the
//!   caller keeps: it may be smaller than the set of the object it was read from, so `Pc(i)`
//!   does not pass. Such a pointer has `Pc(i)`, or `Pb(i)` alone when a call returned it `from`
//!   an argument that was the param moved. The caller grows every `&mut` loan in `reach` of its
//!   argument, so a declared origin may land in any of them. Without such a `&mut` in the type,
//!   `Pc(i)` is in `P` only as the origins of the value pointed to;
//! - any param `i`, whatever its type, when the clause of some `&mut` param `t` names `i` and
//!   `S ⊆ decl(t)`: the caller's store through the argument of `t` has the argument of `i`
//!   among its sources, so it grows every view that argument reaches, with at least `S`.
//!
//! In the param rules `S` is taken without its passive loans ([`Flow::store_fits`]): a `&`
//! loan of a place reached last through a `&` (`&xs[a..b]`, `&other.f` with `xs`, `other` of
//! `&` type) blocks no access of the body, and what it borrows from is the origin set of that
//! `&` pointer, which is in `S` with it and is held to the rule. A borrow through a `&mut`
//! stays a loan the caller cannot account for: the body could go on using the `&mut` beside
//! the stored reference.
//!
//! A store needs an atom of `reach(P)` that one of the rules above applies to: a pointer the
//! rules know nothing about (a raw dereference first, the result of a call `from` a shared
//! anchor) takes only values without origins, or, in an assignment, values whose origins lie
//! in `contents(reach(P))`.
//!
//! After a call's store a local may hold origins that the locals holding a loan of it do
//! not: a loan is live while a live local reaches it ([`Flow::live_loans`] closes under
//! [`Flow::reach`]), and a value read through a pointer to a local gets the local's set at
//! that point ([`Flow::contents_of`]). A call through a fn value stores nothing: its params
//! take no clause, so its body is held to the rules above with `decl` empty.
//!
//! **Fixpoint.** Forward dataflow in reverse postorder to a fixpoint (union at joins). The
//! usual bound (loop depth + 1 rounds) holds for gen/kill problems; copies between locals
//! inside a loop can need more, so the iteration runs to the fixpoint (monotone, finite) and the
//! round count is reported ([`Flow::rounds`]).

use rustc_hash::FxHashMap;

use crate::bits::Bits;
use crate::cfg::{Body, LoanId, LocalId, OpKind, Operand, Place, Proj, Rvalue, StmtKind, Store};
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

/// Why a store is rejected ([`Flow::store_fits`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Misfit {
    /// A view object the pointer may point to cannot take the value's origins.
    Object,
    /// Only what param `param` points to or reaches lacks them, and a `from` clause on it with
    /// the entries `add` (param, `copy`) declares the store.
    Clause { param: u32, add: Vec<(u32, bool)> },
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

    /// The pointer and value origins of store `s` of a call with operands `args` (`None`: the
    /// value has no origins).
    pub fn store_sets(&self, st: &State, args: &[Operand], s: &Store) -> Option<(Bits, Bits)> {
        let (t, q) = (args.get(s.target as usize)?, args.get(s.source as usize)?);
        let mut val = self.op_origins(st, q);
        if s.copy {
            val = self.contents_of(st, &val);
        }
        if val.is_empty() {
            return None;
        }
        Some((self.op_origins(st, t), val))
    }

    /// Whether a value with origins `val` may be stored through a pointer with origins `ptr`
    /// (module doc, **Stores**). `direct`: the param whose own dereference is the only one of
    /// the place an assignment writes. `call`: the store of a call, which grows the local
    /// objects ([`Flow::apply`]); an assignment's value must be in them already.
    pub fn store_fits(
        &self,
        st: &State,
        ptr: &Bits,
        val: &Bits,
        direct: Option<u32>,
        call: bool,
    ) -> Result<(), Misfit> {
        if val.is_empty() {
            return Ok(());
        }
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
                if (!call || captured) && !val.is_subset(&self.contents[a]) {
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
        let unplaced = !placed && (call || !held.is_subset(&allowed));
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

    /// The store of a call: a value with origins `val` through a pointer with origins `ptr`.
    /// Every local view object the pointer may point to or reach gains `val` (module doc,
    /// **Stores**); whether a loan's contents changed.
    fn grow(&mut self, st: &mut State, ptr: &Bits, val: &Bits) -> bool {
        let body = self.body;
        // the callee may keep a pointer it is given as a source and store through it too
        let mut targets = self.reach(st, ptr);
        targets.union_with(&self.reach(st, val));
        let mut changed = false;
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
            // a local holds no borrow of itself: while it is written through this loan, no
            // other loan of it is live
            let mut add = val.clone();
            for m in val.iter() {
                if body.loans.get(m).is_some_and(|x| x.place.local == l.place.local && !x.place.has_deref()) {
                    add.remove(m);
                }
            }
            if self.contents[a].union_with(&add) {
                changed = true;
                self.grown = true;
            }
            if let Some(&t) = self.tracked.get(l.place.local as usize)
                && t != u32::MAX
            {
                st.o[t as usize].union_with(&add);
            }
        }
        changed
    }

    /// An assignment of a value with origins `val` through the dereferences of `dest`: every
    /// borrowed local it may land in has `val` from here on (module doc, **Stores**).
    fn settle(&self, st: &mut State, dest: &Place, val: &Bits) {
        let (_, Some(ptr)) = self.place_origins(st, dest) else { return };
        for a in self.reach(st, &ptr).iter() {
            let Some(l) = self.body.loans.get(a) else { continue };
            if !l.mutable
                || l.place.has_deref()
                || !self.is_ref_like(l.place.ty)
                || !val.is_subset(&self.contents[a])
            {
                continue;
            }
            if let Some(&t) = self.tracked.get(l.place.local as usize)
                && t != u32::MAX
            {
                st.o[t as usize].union_with(val);
            }
        }
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
                if place.derefs() > 1 {
                    s.union_with(&self.through(st, place));
                } else if let Some(p) = ptr {
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
                let (s, ch) = self.rv_origins(st, rv, dest.ty);
                changed |= ch;
                if let Rvalue::Call { args, stores, .. } = rv {
                    for x in stores {
                        if let Some((ptr, val)) = self.store_sets(st, args, x) {
                            changed |= self.grow(st, &ptr, &val);
                        }
                    }
                }
                for_each_op(rv, |o| {
                    if o.kind == OpKind::Move {
                        st.moved.insert(o.site as usize);
                    }
                });
                // re-initialized: moves of the place or below it are undone
                self.reinit(st, dest);
                if dest.has_deref() {
                    if !s.is_empty() {
                        self.settle(st, dest, &s);
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
