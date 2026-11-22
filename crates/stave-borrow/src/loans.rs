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
//! - a `&mut` loan `L` of a ref-like place without a dereference: the set of the borrowed
//!   local gains `S` from the store on, and so does `contents(L)` ([`Flow::apply`]). Left out
//!   are only loans of that local that no reference in the value can have: another live loan
//!   of a place overlapping `L`'s would be a conflict. A loan of another part of the local
//!   stays (the local then borrows from itself), and so does `L` itself: a value made from
//!   the pointer written through may point into the local (`q.cur = &q.n`; a callee given
//!   `&mut b` as a source can split it and store one part through the other), and the local
//!   is then borrowed for as long as it is used. The local is read later with that set, and
//!   through a pointer with what the local's set says where it is read. A pointer that may
//!   point to several locals grows each of them. A place behind a dereference is read with
//!   the sets of the pointers it was borrowed through, which are in the pointer's set with
//!   it: nothing is added for it. A captured local of a nested body belongs to the enclosing
//!   fn, which learns nothing from the closure: it is not grown, and takes the store only if
//!   `S ⊆ contents(L)`;
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
//!   among its sources, so it grows every view that argument reaches, with at least `S`. The
//!   caller does so also where the target of `t` holds no borrow at its instance (a generic
//!   callee checked with a `T: view`): the store then has no pointer atoms, only its sources.
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
//! The stores of one call are made together, to a fixpoint ([`Flow::settle_call`]): a store
//! into one target feeds the `from copy` entry of another, and the callee reads a `from copy`
//! source, like the caller a `from copy` result, after its own stores. The values of the
//! stores and the result of the call are those of the state after all of them.
//!
//! After a call's store a local may hold origins that the locals holding a loan of it do
//! not: a loan is live while a live local reaches it (the live loans are closed under
//! [`Flow::reach`]), and a value read through a pointer to a local gets the local's set at
//! that point ([`Flow::contents_of`]). A call through a fn value stores nothing: its params
//! take no clause, so its body is held to the rules above with `decl` empty.
//!
//! **Fixpoint.** Forward dataflow in reverse postorder to a fixpoint (union at joins). The
//! usual bound (loop depth + 1 rounds) holds for gen/kill problems; copies between locals
//! inside a loop can need more, so the iteration runs to the fixpoint (finite) and the round
//! count is reported ([`Flow::rounds`]).
//!
//! **Storage.** One [`State`] is walked through the blocks and changed in place by
//! [`Flow::apply`]; an origin set is an ascending list of atoms ([`Set`]) that a state names by
//! its id in [`Flow::sets`]. A block has an entry state of its own ([`Saved`]) only when it
//! cannot take the state the walk reached: every block but one whose single predecessor comes
//! right before it in the reverse postorder (then the walk is at that predecessor's end when it
//! gets there). An entry state keeps only what can still be read from the block's entry on
//! ([`crate::live`]): the sets of the tracked locals live there and the moved marks of the
//! locals some access may still test. A set that is not read before its local is redefined
//! cannot change a set that is, so dropping it changes nothing that is read. The mark of a
//! local that no access can test after one of its moves ([`Flow::tested_after_move`]) is never
//! kept: every test of it finds it unset. Where a block has several successors the walked
//! state drops what is dead at the block's end too ([`Flow::prune`]), so what the walk
//! carries and copies follows the live locals, not the locals of the fn.
//!
//! The entry state of a block is the union of everything that reached it in any round. For a
//! block without one, the walk has only what reaches it in the current round, which is the same
//! while later rounds can only add to earlier ones: every step is monotone until a store grows
//! a local ([`Flow::grown`]). From then on a loan's contents are read from the borrowed local's
//! set at the point of the read ([`Flow::loan_holds`]) instead of the set taken when the loan
//! was made, so in a body with a statement that can grow a local the entry states also keep
//! the sets of the borrowed locals such a read can reach ([`Flow::lent`]).
//!
//! A read from the local's set can give LESS than the set taken when the loan was made (the
//! local was written while borrowed). Then a round after the store finds less than a round
//! before it did, and the union over the rounds holds what no single round does. Where no such
//! read happens the rounds before the store found nothing the last round does not find again,
//! and the walked state is again all an entry state would hold. So the last round compares,
//! at every read of a borrowed local's set, that set with what the loan's contents were when
//! the first store grew a local ([`Flow::pre`]); if one lacks an atom, the body is solved
//! again with an entry state for every block, joined over all rounds ([`Flow::exact`]).

use std::cell::Cell;

use rustc_hash::FxHashMap;

use crate::bits::Bits;
use crate::cfg::{Body, LoanId, LocalId, OpKind, Operand, Place, Proj, Rvalue, Stmt, StmtKind, Store};
use crate::check::{Access, places_conflict, stmt_uses};
use crate::live::{Liveness, liveness};
use crate::set::{Set, SetTable};
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

/// The dataflow state at the point a walk has reached: the origin set of every tracked local, as
/// an id of [`Flow::sets`], and the maybe-moved locals (a value moves whole: one mark per local).
/// A walk has one state and changes it in place. There is no `Clone`: every whole state comes
/// from [`Flow::state`], which counts the ones it makes ([`Cost::states`]).
pub struct State {
    o: Vec<u32>,
    /// The tracked locals whose set may be non-empty, each once (`in_held`).
    held: Vec<u32>,
    in_held: Vec<bool>,
    moved: Vec<bool>,
    /// The locals whose mark may be set, each once (`in_marked`).
    marked: Vec<LocalId>,
    in_marked: Vec<bool>,
    /// While `recording`: every change of a set, as the tracked local and the id it had.
    log: Vec<(u32, u32)>,
    recording: bool,
}

impl State {
    fn new(ntracked: usize, nlocals: usize) -> State {
        State {
            o: vec![0; ntracked],
            held: Vec::new(),
            in_held: vec![false; ntracked],
            moved: vec![false; nlocals],
            marked: Vec::new(),
            in_marked: vec![false; nlocals],
            log: Vec::new(),
            recording: false,
        }
    }

    /// The id of tracked local `t`'s set.
    #[inline]
    pub fn id(&self, t: u32) -> u32 {
        self.o[t as usize]
    }

    /// The tracked locals whose set may be non-empty, each once.
    pub fn held(&self) -> &[u32] {
        &self.held
    }

    fn put(&mut self, t: u32, id: u32) {
        let old = self.o[t as usize];
        if old == id {
            return;
        }
        if self.recording {
            self.log.push((t, old));
        }
        self.write(t, id);
    }

    fn write(&mut self, t: u32, id: u32) {
        if id != 0 && !self.in_held[t as usize] {
            self.in_held[t as usize] = true;
            self.held.push(t);
        }
        self.o[t as usize] = id;
    }

    pub fn is_moved(&self, l: LocalId) -> bool {
        self.moved.get(l as usize).is_some_and(|&m| m)
    }

    fn set_moved(&mut self, l: LocalId, on: bool) {
        if let Some(m) = self.moved.get_mut(l as usize) {
            if on && !self.in_marked[l as usize] {
                self.in_marked[l as usize] = true;
                self.marked.push(l);
            }
            *m = on;
        }
    }

    fn clear(&mut self) {
        for t in self.held.drain(..) {
            self.o[t as usize] = 0;
            self.in_held[t as usize] = false;
        }
        for l in self.marked.drain(..) {
            self.moved[l as usize] = false;
            self.in_marked[l as usize] = false;
        }
        self.log.clear();
        self.recording = false;
    }

    /// Starts a record of the changes of the sets.
    pub fn record(&mut self) {
        self.log.clear();
        self.recording = true;
    }

    /// Ends the record and returns it: per change, the tracked local and the id it had.
    pub fn recorded(&mut self) -> &[(u32, u32)] {
        self.recording = false;
        &self.log
    }

    /// Ends the record and takes back the changes it lists.
    pub fn undo(&mut self) {
        self.recording = false;
        while let Some((t, old)) = self.log.pop() {
            self.write(t, old);
        }
    }
}

/// A state as a block's entry keeps it: the non-empty sets of the tracked locals that can still
/// be read there, and the moved locals among those whose mark can still be tested.
#[derive(Clone, Debug, Default)]
pub struct Saved {
    o: Vec<(u32, u32)>,
    moved: Vec<LocalId>,
}

/// What the dataflow of one body did, in counts that depend on the body alone.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cost {
    /// The statements of the reachable blocks.
    pub stmts: u64,
    /// Statements applied to a state ([`Flow::apply`]).
    pub applied: u64,
    /// Sets and moved marks copied into a block's entry state or out of one.
    pub copied: u64,
    /// The (local, block) pairs with the local's set or moved mark live at the block's entry:
    /// what the entry states can hold.
    pub live: u64,
    /// Bodies in which a store grew a local ([`Flow::grown`]).
    pub grown: u64,
    /// Bodies solved a second time, with an entry state per block ([`Flow::exact`]).
    pub exact: u64,
    /// Atoms of the sets entered into the set table or joined there.
    pub atoms: u64,
    /// Loans counted in or out of the live loans by the checks, and loans visited when the
    /// loans reached through the live ones are found again.
    pub counted: u64,
    /// Times the checks found the loans reached through the live ones again.
    pub rebuilt: u64,
    /// Whole states made ([`Flow::state`] with no spare one): each has an entry per tracked
    /// local and per local.
    pub states: u64,
}

impl Cost {
    pub fn add(&mut self, o: Cost) {
        self.stmts += o.stmts;
        self.applied += o.applied;
        self.copied += o.copied;
        self.live += o.live;
        self.grown += o.grown;
        self.exact += o.exact;
        self.atoms += o.atoms;
        self.counted += o.counted;
        self.rebuilt += o.rebuilt;
        self.states += o.states;
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

/// The store a call makes through one target argument ([`Flow::call_stores`]).
#[derive(Clone, Debug)]
pub struct CallStore {
    /// The target's operand index.
    pub target: u32,
    /// Whether the target points to a ref-like type at this instance.
    pub into_view: bool,
    /// The target pointer's origins (empty when it points to no ref-like type).
    pub ptr: Set,
    /// The value's origins: the union over the clause's entries. One set per target, because
    /// the callee may store what any source borrows through any other source it keeps.
    pub val: Set,
}

/// The solved dataflow of one body.
pub struct Flow<'b> {
    pub body: &'b Body,
    pub atoms: Atoms,
    /// Per local: its tracked index or `u32::MAX`.
    pub tracked: Vec<u32>,
    pub ntracked: usize,
    /// Per loan: the origin set of the borrowed value.
    pub contents: Vec<Set>,
    /// Per loan: the origins of the pointers it was made through (empty for a loan of a place
    /// without a dereference). While a value holds the loan, those keep the borrowed place from
    /// being reached another way.
    pub made: Vec<Set>,
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
    pub decl: Vec<Set>,
    /// Whether a call's store added origins to a loan's contents: only then can a loan's
    /// contents hold what the locals holding the loan do not.
    pub grown: bool,
    /// The distinct sets the states hold.
    pub sets: SetTable,
    /// Liveness of the tracked locals (by tracked index): where their sets can still be read.
    pub live: Liveness,
    /// Per local: its index among the locals whose moved mark an access can test after a move
    /// (`u32::MAX`: not one). Empty in a body without a move.
    mvar: Vec<u32>,
    /// Liveness of the moved marks (by `mvar` index): where an access may still test them.
    mlive: Liveness,
    /// Per tracked local: a loan's contents may be read from its set once a store has grown a
    /// local ([`Flow::loan_holds`]). Empty in a body without a statement that can grow one.
    lent: Vec<bool>,
    /// Per loan: its contents when the first store grew a local, before that store.
    pre: Vec<Set>,
    /// Whether a round of [`Flow::iterate`] is on, and whether a read of a borrowed local's
    /// set in it gave less than the loan's `pre`.
    watch: bool,
    short: Cell<bool>,
    /// Whether every reachable block has an entry state (module doc, **Storage**).
    exact: bool,
    /// Per block: it takes the state the walk reached at the end of the block before it in
    /// `rpo`, its only predecessor.
    carried: Vec<bool>,
    /// Block entry states (`None`: a carried or an unreachable block).
    entry: Vec<Option<Saved>>,
    pub preds: Vec<Vec<u32>>,
    pub rpo: Vec<u32>,
    /// Per block: its position in `rpo` (`u32::MAX`: unreachable).
    rpo_at: Vec<u32>,
    pub rounds: u32,
    pub cost: Cost,
    /// A state no walk uses now.
    spare: Option<State>,
    /// Scratch of [`Flow::save`]: what the entry keeps; per tracked local and per local, its
    /// place in the entry state being joined, plus one (0: not in it; made at the first join).
    kept: Saved,
    slot: Vec<u32>,
    mslot: Vec<u32>,
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
        let mut decl = vec![Set::new(atoms.len()); nparams];
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
        let mut rpo_at = vec![u32::MAX; n];
        for (i, &b) in rpo.iter().enumerate() {
            rpo_at[b as usize] = i as u32;
        }
        Flow {
            body,
            atoms,
            tracked,
            ntracked,
            contents: vec![Set::new(atoms.len()); body.loans.len()],
            made: vec![Set::new(atoms.len()); body.loans.len()],
            ref_like,
            ref_param,
            deep_param,
            store_param,
            clause_param,
            decl,
            grown: false,
            sets: SetTable::new(atoms.len()),
            live: Liveness::default(),
            mvar: Vec::new(),
            mlive: Liveness::default(),
            lent: Vec::new(),
            pre: Vec::new(),
            watch: false,
            short: Cell::new(false),
            exact: false,
            carried: vec![false; n],
            entry: vec![None; n],
            preds,
            rpo,
            rpo_at,
            rounds: 0,
            cost: Cost::default(),
            spare: None,
            kept: Saved::default(),
            slot: Vec::new(),
            mslot: Vec::new(),
        }
    }

    pub fn is_ref_like(&self, t: TypeId) -> bool {
        self.ref_like.get(&t).copied().unwrap_or(true)
    }

    pub fn empty(&self) -> Set {
        Set::new(self.atoms.len())
    }

    /// Whether the body has no state to track: no ref-like local, no loan, no move site. Then no
    /// statement changes the dataflow state (origin sets belong to tracked locals, loan contents
    /// to loans, moved sites to move operands), so every reachable block starts in the initial
    /// state, every origin set read from it is empty, and no access conflicts with a loan or
    /// uses a moved place.
    pub fn stateless(&self) -> bool {
        self.ntracked == 0 && self.body.loans.is_empty() && self.body.move_sites.is_empty()
    }

    /// Whether block `b` can be reached from the entry.
    pub fn reached(&self, b: usize) -> bool {
        self.rpo_at.get(b).is_some_and(|&i| i != u32::MAX)
    }

    /// Whether block `b` takes the state at the end of the block before it in `rpo`.
    pub fn carried(&self, b: usize) -> bool {
        self.carried[b]
    }

    /// What the dataflow did so far.
    pub fn cost(&self) -> Cost {
        Cost { atoms: self.sets.work(), ..self.cost }
    }

    /// A cleared state for a walk; [`Flow::recycle`] takes it back.
    pub fn state(&mut self) -> State {
        match self.spare.take() {
            Some(mut st) => {
                st.clear();
                st
            }
            None => {
                self.cost.states += 1;
                State::new(self.ntracked, self.body.locals.len())
            }
        }
    }

    pub fn recycle(&mut self, st: State) {
        self.spare = Some(st);
    }

    /// Solves the forward dataflow.
    pub fn solve(&mut self) {
        let body = self.body;
        if body.blocks.is_empty() {
            return;
        }
        self.cost.stmts = self.rpo.iter().map(|&b| body.blocks[b as usize].stmts.len() as u64).sum();
        if self.stateless() {
            // The fixpoint without iterating: the first round reaches every block of the reverse
            // postorder, and a second one finds nothing new if it set any entry but block 0's.
            for &b in &self.rpo {
                self.entry[b as usize] = Some(Saved::default());
            }
            self.rounds = if self.rpo.len() > 1 { 2 } else { 1 };
            return;
        }
        self.live =
            liveness(body, &self.preds, &self.tracked, self.ntracked, |s, f| stmt_uses(s, |l, _| f(l)));
        if !body.move_sites.is_empty() {
            self.mvar = self.tested_after_move();
            let nm = self.mvar.iter().filter(|&&v| v != u32::MAX).count();
            self.mlive = liveness(body, &self.preds, &self.mvar, nm, |s, f| moved_tests(body, s, f));
        }
        self.cost.live = self.live.size() + self.mlive.size();
        for i in 1..self.rpo.len() {
            let b = self.rpo[i] as usize;
            self.carried[b] = self.preds[b].len() == 1 && self.preds[b][0] == self.rpo[i - 1];
        }
        // a statement that can grow a local: a store through a dereference, a call that stores
        let can_grow = body.blocks.iter().flat_map(|b| &b.stmts).any(|s| match &s.kind {
            StmtKind::Assign { dest, rv, .. } => {
                (dest.has_deref() && self.is_ref_like(dest.ty))
                    || matches!(rv, Rvalue::Call { stores, .. } if !stores.is_empty())
            }
            _ => false,
        });
        if can_grow {
            self.lent = vec![false; self.ntracked];
            for l in &body.loans {
                if !l.place.has_deref()
                    && self.is_ref_like(l.place.ty)
                    && let Some(&t) = self.tracked.get(l.place.local as usize)
                    && t != u32::MAX
                {
                    self.lent[t as usize] = true;
                }
            }
        }
        self.iterate();
        self.cost.grown = u64::from(self.grown);
        if !(self.grown && self.short.get()) {
            return;
        }
        // a round after the store read less than one before it found: only an entry state per
        // block keeps what the earlier rounds found
        self.exact = true;
        self.cost.exact = 1;
        self.grown = false;
        self.pre = Vec::new();
        self.carried.fill(false);
        self.entry.fill(None);
        let empty = self.empty();
        self.contents.fill(empty.clone());
        self.made.fill(empty);
        self.iterate();
    }

    /// Whether a loan's contents may be read from the set of tracked local `t`.
    pub fn is_lent(&self, t: u32) -> bool {
        self.lent.get(t as usize).is_some_and(|&x| x)
    }

    /// Numbers the locals whose moved mark must be followed: those with an access that tests
    /// the mark ([`moved_tests`]) and can run after one of their moves. The mark of any other
    /// local is unset wherever it is tested.
    ///
    /// A path leads to later positions of the reverse postorder except over an edge that goes
    /// back; so from a move, a test is reached only if it comes later in that order, or at
    /// any position from the earliest one the edges going back from there on lead to.
    fn tested_after_move(&self) -> Vec<u32> {
        let body = self.body;
        let n = self.rpo.len();
        // per position: the earliest position an edge going back from it or a later one leads to
        let mut back = vec![u32::MAX; n + 1];
        for (i, &b) in self.rpo.iter().enumerate().rev() {
            back[i] = back[i + 1];
            for &succ in body.blocks[b as usize].term.succs() {
                let to = self.rpo_at[succ as usize];
                if to <= i as u32 {
                    back[i] = back[i].min(to);
                }
            }
        }
        // per local: its first move and its last test, as (position, statement)
        let none = (u32::MAX, u32::MAX);
        let mut first_move = vec![none; body.locals.len()];
        let mut last_test: Vec<Option<(u32, u32)>> = vec![None; body.locals.len()];
        for (i, &b) in self.rpo.iter().enumerate() {
            for (k, s) in body.blocks[b as usize].stmts.iter().enumerate() {
                let at = (i as u32, k as u32);
                stmt_moves(s, |l| first_move[l as usize] = first_move[l as usize].min(at));
                moved_tests(body, s, &mut |l| last_test[l as usize] = last_test[l as usize].max(Some(at)));
            }
        }
        let mut mvar = vec![u32::MAX; body.locals.len()];
        let mut nm = 0;
        for l in 0..body.locals.len() {
            let (m, Some(t)) = (first_move[l], last_test[l]) else { continue };
            if m == none {
                continue;
            }
            let mut from = m.0;
            while back[from as usize] < from {
                from = back[from as usize];
            }
            let again = back[m.0 as usize] <= m.0;
            if t > m || (again && t.0 >= from) {
                mvar[l] = nm;
                nm += 1;
            }
        }
        mvar
    }

    /// Runs the rounds to the fixpoint.
    fn iterate(&mut self) {
        let body = self.body;
        let mut st = self.state();
        for (i, l) in body.locals.iter().enumerate() {
            if let (Some(p), t) = (l.param, self.tracked[i])
                && t != u32::MAX
            {
                let mut s = self.empty();
                s.insert(self.atoms.pb(p));
                self.put_o(&mut st, t, &s);
            }
        }
        self.save(0, &st);
        let order = self.rpo.clone();
        self.watch = true;
        for round in 1..=100_000u32 {
            self.rounds = round;
            self.short.set(false);
            let mut changed = false;
            // whether `st` is the state at the end of the block before
            let mut have = false;
            for &b in &order {
                let b = b as usize;
                if !self.carried[b] {
                    have = self.load(b, &mut st);
                }
                if !have {
                    continue;
                }
                for s in &body.blocks[b].stmts {
                    changed |= self.apply(&s.kind, &mut st);
                }
                let succs = body.blocks[b].term.succs();
                if succs.len() > 1 && succs.iter().any(|&succ| !self.carried[succ as usize]) {
                    self.prune(b, &mut st);
                }
                for &succ in succs {
                    if !self.carried[succ as usize] {
                        changed |= self.save(succ as usize, &st);
                    }
                }
            }
            if !changed {
                break;
            }
        }
        self.watch = false;
        self.recycle(st);
    }

    /// The index of local `l` among the locals whose moved mark is followed (`u32::MAX`: not
    /// one).
    fn mvar_of(&self, l: LocalId) -> u32 {
        self.mvar.get(l as usize).copied().unwrap_or(u32::MAX)
    }

    /// Drops from `st`, the state at the end of block `b`, what cannot be read from there on.
    pub fn prune(&mut self, b: usize, st: &mut State) {
        self.cost.copied += (st.held.len() + st.marked.len()) as u64;
        let mut k = 0;
        for i in 0..st.held.len() {
            let t = st.held[i];
            let read = self.live.live_out(t, b as u32) || self.is_lent(t);
            if st.o[t as usize] != 0 && read {
                st.held[k] = t;
                k += 1;
            } else {
                st.o[t as usize] = 0;
                st.in_held[t as usize] = false;
            }
        }
        st.held.truncate(k);
        let mut k = 0;
        for i in 0..st.marked.len() {
            let l = st.marked[i];
            if st.moved[l as usize] && self.mlive.live_out(self.mvar_of(l), b as u32) {
                st.marked[k] = l;
                k += 1;
            } else {
                st.moved[l as usize] = false;
                st.in_marked[l as usize] = false;
            }
        }
        st.marked.truncate(k);
    }

    /// What the entry of block `b` keeps of state `st`.
    pub fn keep(&mut self, b: usize, st: &State, out: &mut Saved) {
        out.o.clear();
        out.moved.clear();
        for &t in &st.held {
            let id = st.o[t as usize];
            if id != 0 && (self.live.live_in(t, b as u32) || self.is_lent(t)) {
                out.o.push((t, id));
            }
        }
        for &l in &st.marked {
            if st.moved[l as usize] && self.mlive.live_in(self.mvar_of(l), b as u32) {
                out.moved.push(l);
            }
        }
        self.cost.copied += (st.held.len() + st.marked.len()) as u64;
    }

    /// Adds state `st` to the entry state of block `b`; whether that changed.
    fn save(&mut self, b: usize, st: &State) -> bool {
        let mut kept = std::mem::take(&mut self.kept);
        self.keep(b, st, &mut kept);
        let changed = match self.entry[b].take() {
            None => {
                self.entry[b] = Some(kept.clone());
                true
            }
            Some(mut e) => {
                let changed = self.join(&mut e, &kept);
                self.entry[b] = Some(e);
                changed
            }
        };
        self.kept = kept;
        changed
    }

    /// `into |= from`; whether `into` changed.
    fn join(&mut self, into: &mut Saved, from: &Saved) -> bool {
        if from.o.is_empty() && from.moved.is_empty() {
            return false;
        }
        self.cost.copied += (into.o.len() + into.moved.len()) as u64;
        if self.slot.is_empty() {
            self.slot = vec![0; self.ntracked];
            self.mslot = vec![0; self.body.locals.len()];
        }
        let mut changed = false;
        for (k, &(t, _)) in into.o.iter().enumerate() {
            self.slot[t as usize] = k as u32 + 1;
        }
        for &(t, id) in &from.o {
            let k = self.slot[t as usize];
            if k == 0 {
                into.o.push((t, id));
                self.slot[t as usize] = into.o.len() as u32;
                changed = true;
            } else {
                let have = &mut into.o[k as usize - 1].1;
                let all = self.sets.union(*have, id);
                changed |= all != *have;
                *have = all;
            }
        }
        for &(t, _) in &into.o {
            self.slot[t as usize] = 0;
        }
        for &l in &into.moved {
            self.mslot[l as usize] = 1;
        }
        for &l in &from.moved {
            if self.mslot[l as usize] == 0 {
                self.mslot[l as usize] = 1;
                into.moved.push(l);
                changed = true;
            }
        }
        for &l in &into.moved {
            self.mslot[l as usize] = 0;
        }
        changed
    }

    /// Sets `st` to the saved state `e`.
    pub fn restore(&mut self, e: &Saved, st: &mut State) {
        st.clear();
        for &(t, id) in &e.o {
            st.write(t, id);
        }
        for &l in &e.moved {
            st.set_moved(l, true);
        }
        self.cost.copied += (e.o.len() + e.moved.len()) as u64;
    }

    /// Sets `st` to the entry state of block `b`, which has one; `false`: it has none yet.
    pub fn load(&mut self, b: usize, st: &mut State) -> bool {
        let Some(e) = self.entry[b].take() else { return false };
        self.restore(&e, st);
        self.entry[b] = Some(e);
        true
    }

    /// Sets `st` to the state at the entry of block `b` of a solved body; `false`: the block is
    /// unreachable. A block without an entry state gets the state walked from the nearest block
    /// before it that has one.
    pub fn enter(&mut self, b: usize, st: &mut State) -> bool {
        if !self.reached(b) {
            return false;
        }
        let mut walk: Vec<usize> = Vec::new();
        let mut c = b;
        while self.carried[c] {
            c = self.preds[c][0] as usize;
            walk.push(c);
        }
        if !self.load(c, st) {
            return false;
        }
        let body = self.body;
        for &x in walk.iter().rev() {
            for s in &body.blocks[x].stmts {
                self.apply(&s.kind, st);
            }
        }
        true
    }

    pub fn local_o(&self, st: &State, l: LocalId) -> Set {
        match self.tracked.get(l as usize) {
            Some(&t) if t != u32::MAX => self.sets.get(st.o[t as usize]).clone(),
            _ => self.empty(),
        }
    }

    /// Makes `s` the set of tracked local `t`.
    fn put_o(&mut self, st: &mut State, t: u32, s: &Set) {
        if self.sets.get(st.o[t as usize]) == s {
            return;
        }
        let id = self.sets.id(s);
        st.put(t, id);
    }

    /// Adds `s` to the set of tracked local `t`; whether it changed.
    fn add_o(&mut self, st: &mut State, t: u32, s: &Set) -> bool {
        let cur = self.sets.get(st.o[t as usize]);
        if s.is_subset(cur) {
            return false;
        }
        let mut all = cur.clone();
        all.union_with(s);
        self.put_o(st, t, &all);
        true
    }

    /// The origins of the value in the place loan `l` borrows: the set taken when the loan was
    /// made, and once a store has grown a local, for a borrowed local what its set says in `st`
    /// (a store adds to it from the store on: what is stored later on the path is not in the
    /// place yet).
    pub fn loan_holds(&self, st: &State, l: usize) -> Set {
        if let Some(local) = self.read_local(l) {
            let now = self.local_o(st, local);
            if self.watch && !self.exact && self.pre.get(l).is_some_and(|was| !was.is_subset(&now)) {
                self.short.set(true);
            }
            now
        } else {
            self.contents[l].clone()
        }
    }

    /// The local whose set in the state is what loan `l`'s place holds ([`Flow::loan_holds`]):
    /// once a store has grown a local, a ref-like local borrowed without a dereference. `None`:
    /// the loan's contents.
    pub fn read_local(&self, l: usize) -> Option<LocalId> {
        let place = &self.body.loans[l].place;
        (self.grown && !place.has_deref() && self.is_ref_like(place.ty)).then_some(place.local)
    }

    /// `contents(S)` in state `st`.
    pub fn contents_of(&self, st: &State, s: &Set) -> Set {
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
    pub fn place_origins(&self, st: &State, p: &Place) -> (Set, Option<Set>) {
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
    pub fn reach(&self, st: &State, s: &Set) -> Set {
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
    fn covered(&self, i: u32, held: &Set) -> bool {
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
        ptr: &Set,
        val: &Set,
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
        let mut want: Option<(u32, Set)> = None;
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
    fn clause_entries(&self, missing: &Set) -> Vec<(u32, bool)> {
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
    fn grow(&mut self, st: &mut State, ptr: &Set, val: &Set, call: bool) -> (bool, bool) {
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
            if !add.is_subset(&self.contents[a]) {
                if !self.grown && !self.exact {
                    self.pre = self.contents.clone();
                }
                self.grown = true;
                self.contents[a].union_with(&add);
                contents = true;
            }
            if let Some(&t) = self.tracked.get(l.place.local as usize)
                && t != u32::MAX
            {
                state |= self.add_o(st, t, &add);
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
    fn cannot_hold(&self, target: usize, m: usize, val: &Set) -> bool {
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
    fn through(&self, st: &State, p: &Place) -> Set {
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

    pub fn op_origins(&self, st: &State, op: &Operand) -> Set {
        if op.kind == OpKind::Const || !self.is_ref_like(op.place.ty) {
            return self.empty();
        }
        self.place_origins(st, &op.place).0
    }

    /// The origin set of an rvalue; `contents` of a created loan is recorded.
    pub fn rv_origins(&mut self, st: &State, rv: &Rvalue, dest_ty: TypeId) -> (Set, bool) {
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
        self.cost.applied += 1;
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
                        st.set_moved(o.place.local, true);
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
                        self.put_o(st, t, &s);
                    } else {
                        self.add_o(st, t, &s);
                    }
                }
            }
            StmtKind::Activate { local, reserved, active } => {
                if let Some(&t) = self.tracked.get(*local as usize)
                    && t != u32::MAX
                {
                    let o = self.sets.get(st.o[t as usize]);
                    if o.contains(*reserved as usize) {
                        let mut o = o.clone();
                        o.remove(*reserved as usize);
                        o.insert(*active as usize);
                        self.put_o(st, t, &o);
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
                    st.put(t, 0);
                }
                let p = Place::local(*l, TypeId::NONE);
                self.reinit(st, &p);
            }
            StmtKind::Return(o) | StmtKind::Read(o) => {
                if o.kind == OpKind::Move {
                    st.set_moved(o.place.local, true);
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
            st.set_moved(dest.local, false);
        }
    }

    /// Whether local `l` reaches loan `loan`: it holds it, or after a call's store holds a loan
    /// whose contents reach it.
    pub fn holds(&self, st: &State, l: LocalId, loan: LoanId) -> bool {
        let o = self.local_o(st, l);
        o.contains(loan as usize) || (self.grown && self.reach(st, &o).contains(loan as usize))
    }

    pub fn loan(&self, l: LoanId) -> &crate::cfg::Loan {
        &self.body.loans[l as usize]
    }
}

/// The locals a statement's move operands move.
fn stmt_moves(s: &Stmt, mut f: impl FnMut(LocalId)) {
    match &s.kind {
        StmtKind::Assign { rv, .. } => for_each_op(rv, |o| {
            if o.kind == OpKind::Move {
                f(o.place.local)
            }
        }),
        StmtKind::Read(o) | StmtKind::Return(o) if o.kind == OpKind::Move => f(o.place.local),
        _ => {}
    }
}

/// The locals whose moved mark the accesses of a statement test ([`crate::check`]): every
/// access but a drop, a storage-dead, and the write of a whole local.
fn moved_tests(body: &Body, s: &Stmt, f: &mut dyn FnMut(LocalId)) {
    match &s.kind {
        StmtKind::Assign { dest, rv, init } => {
            for_each_op(rv, |o| {
                if o.kind != OpKind::Const {
                    f(o.place.local)
                }
            });
            if let Rvalue::Ref { place, .. } = rv {
                f(place.local);
            }
            if !*init && !dest.proj.is_empty() {
                f(dest.local);
            }
        }
        StmtKind::Activate { reserved, .. } => {
            if let Some(l) = body.loans.get(*reserved as usize) {
                f(l.place.local);
            }
        }
        StmtKind::Read(o) | StmtKind::Return(o) => {
            if o.kind != OpKind::Const {
                f(o.place.local)
            }
        }
        StmtKind::Drop(_) | StmtKind::StorageDead(_) => {}
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
