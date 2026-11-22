//! The borrow CFG: a MIR-like body per fn, built from the typed AST (not from stave-ir,
//! which is post-mono SSA).
//!
//! Places are `local.(field|deref|index)*` ([`Place`]). Every statement is one of: an assignment
//! of an rvalue to a place (a `&`/`&mut` rvalue creates a loan, [`Rvalue::Ref`]), the activation
//! of a two-phase loan ([`StmtKind::Activate`]), an explicit read (a pattern test, an f-string
//! hole, an index operand), an implicit drop, a storage-dead, or a return. Operands are copies or
//! moves of places, or constants (string literals have the static origin: no origins at all).
//! Accesses (read, write, move, drop, storage-dead) follow from the statement shapes
//! ([`crate::check`]).
//!
//! The CFG shape follows the syntax nesting (Stave has no goto): `if`, `while`, `loop`, `for`,
//! `match`, `&&`/`||`, value `if`, `?`, `break`/`continue`/`return`, and calls to `-> never`
//! fns. Evaluation order is the lowering's (stave-sema `lower_*`): left to right, the receiver
//! first, a call's arguments before the call; a by-value operand moves its place when it is
//! evaluated, before the operands after it (`Builder::hold`). Temporaries with drop glue die at the end of their
//! statement (as lowered); others live until their expression is evaluated again (it writes the
//! temporary's one slot) or the fn returns.
//!
//! Closures and `spawn` blocks are built as nested bodies ([`Body::nested`]) whose captured
//! locals are params; the closure value (or the `scope`'s join local, which accumulates every
//! spawn's loans and is read wherever the scope is left) holds one loan per captured local,
//! created at the closure's `fn` / `spawn` token. The capture list and modes are
//! stave-sema's capture analysis (`stave_sema::closures`: by reference only; `&` only when every
//! use is a proven read, else `&mut`), the same modes the thread rules read. A capture held by `&`
//! that the nested body writes, mutably borrows, or activates, or a captured local the analysis
//! does not list, fails the body as unsupported: the two must agree.
//! The closure value's origin set is the union of its capture loans, so the loans live while the
//! value (or a result of a call through it, which takes the callee's origins too) lives. Every
//! call of the value uses the same loans, so a closure's result may borrow only the captures held
//! by `&` (E0512 otherwise: [`Local::capture_mut`]). Captures are references: moving a non-Copy
//! capture out of the body is E0504.

use rustc_hash::FxHashMap;
use stave_diag::Span;
use stave_sema::itemcx::ItemCx;
use stave_sema::loops::ForKind;
use stave_sema::traits::{Adjust, World};
use stave_sema::types::{GArg, Interner, TypeId, TypeKind};
use stave_sema::{BindingRef, Coercion, DeclKind, Res, Target, TypedBody};
use stave_syntax::{NodeKind, Term};

use crate::ty::{DerefKind, Flows, Tys, call_flows, fn_value_flows};

pub type LocalId = u32;
pub type BlockId = u32;
pub type LoanId = u32;

/// Nesting limit of the builder's recursion: the type checker's (a deeper body is E0005 and is
/// skipped for its type error).
pub const MAX_DEPTH: u32 = stave_sema::typeck::MAX_DEPTH;

/// One projection step of a place.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Proj {
    Deref(DerefKind),
    /// Struct or tuple field.
    Field(u32),
    /// Field `.1` of variant `.0`.
    EnumField(u32, u32),
    /// Any element of an array or slice.
    Index,
    /// Column `.0` of a `std.soa.Soa<T>` (field `.0` of T): disjoint from the other columns,
    /// inside the whole value (so a `&mut self` method call conflicts with it).
    Column(u32),
}

/// `local.(field|deref|index)*` with the type of the whole place.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Place {
    pub local: LocalId,
    pub proj: Vec<Proj>,
    pub ty: TypeId,
}

impl Place {
    pub fn local(local: LocalId, ty: TypeId) -> Place {
        Place { local, proj: Vec::new(), ty }
    }

    pub fn derefs(&self) -> usize {
        self.proj.iter().filter(|p| matches!(p, Proj::Deref(_))).count()
    }

    pub fn has_deref(&self) -> bool {
        self.proj.iter().any(|p| matches!(p, Proj::Deref(_)))
    }

    fn with(&self, p: Proj, ty: TypeId) -> Place {
        let mut q = self.clone();
        q.proj.push(p);
        q.ty = ty;
        q
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpKind {
    Copy,
    Move,
    Const,
}

/// A use of a place's value (or a constant).
#[derive(Clone, Debug)]
pub struct Operand {
    pub kind: OpKind,
    /// Meaningless for `Const`.
    pub place: Place,
    pub span: Span,
    /// The AST node (borrow dump `N`).
    pub node: u32,
    /// The move site (`Move` only), index into [`Body::move_sites`].
    pub site: u32,
}

/// One entry of a parameter's `from` clause: a value with the origins of param (at a call:
/// argument) `source` (`copy`: with the origins of the view it points to) may be stored into
/// what `target` points to, or into a view object reached from there through `&mut` pointers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Store {
    pub target: u32,
    pub source: u32,
    pub copy: bool,
    /// At a call: whether the target argument points to a ref-like type at this instance. If
    /// not, nothing is stored into what it points to, but the callee, checked with the type as
    /// a view, may still have stored through its other sources.
    pub into_view: bool,
}

#[derive(Clone, Debug)]
pub enum Rvalue {
    Use(Operand),
    /// `&place` / `&mut place`: creates `loan`.
    Ref {
        place: Place,
        mutable: bool,
        loan: LoanId,
    },
    /// Struct, tuple, array, variant, closure values: the union of the operands' origins.
    Agg(Vec<Operand>),
    Call {
        args: Vec<Operand>,
        flows: Flows,
        /// What the callee may store through its arguments: the entries of its parameters'
        /// `from` clauses whose source argument is ref-like (a call through a fn value, a
        /// built-in, an operator: none).
        stores: Vec<Store>,
        /// The token typeck keys the call by (the call graph's call token): a call's `(`, a
        /// method's name, a user operator, the `for` token of a `next` call.
        tok: u32,
    },
    /// Operators, casts, comparisons, built-ins with a non-ref result: no origins.
    Pure(Vec<Operand>),
}

impl Rvalue {
    /// A call that stores nothing through its arguments.
    fn call(args: Vec<Operand>, flows: Flows, tok: u32) -> Rvalue {
        Rvalue::Call { args, flows, stores: Vec::new(), tok }
    }
}

#[derive(Clone, Debug)]
pub enum StmtKind {
    /// `init`: a fresh binding or temporary (no E0503, no drop of an old value).
    Assign {
        dest: Place,
        rv: Rvalue,
        init: bool,
    },
    /// A two-phase loan becomes active (the call it was made for starts).
    Activate {
        local: LocalId,
        reserved: LoanId,
        active: LoanId,
    },
    Read(Operand),
    Drop(Place),
    StorageDead(LocalId),
    Return(Operand),
}

#[derive(Clone, Debug)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
    pub node: u32,
}

#[derive(Clone, Debug)]
pub enum Terminator {
    Goto(BlockId),
    Branch(Vec<BlockId>),
    Return,
    Unreachable,
}

impl Terminator {
    pub fn succs(&self) -> &[BlockId] {
        match self {
            Terminator::Goto(b) => std::slice::from_ref(b),
            Terminator::Branch(v) => v,
            Terminator::Return | Terminator::Unreachable => &[],
        }
    }
}

#[derive(Clone, Debug)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub term: Terminator,
}

#[derive(Clone, Debug)]
pub struct Loan {
    pub place: Place,
    pub mutable: bool,
    /// The creation (`&` expression, auto-referenced receiver, closure, spawn).
    pub span: Span,
    pub node: u32,
    /// A two-phase reservation (acts as `&` until its activation loan takes over).
    pub reserved: bool,
    /// A capture loan of a closure or spawn (mutability is checked inside the nested body).
    pub capture: bool,
    /// The reborrow of a temporary that holds exactly this loan (`f(&mut x)`: the argument is
    /// `&mut *t` with `t = &mut x`): the same pointer, to the whole of that loan's place.
    pub whole: Option<LoanId>,
    /// The reborrow a `&mut` place cast to `*mut` stands for.
    pub cast: bool,
}

#[derive(Clone, Debug)]
pub struct Local {
    pub ty: TypeId,
    pub mutable: bool,
    /// The binding (or the temporary's expression) span.
    pub span: Span,
    /// Place-text name; empty: the source text of `span` (the identifier, or a temporary's
    /// expression), taken when reported.
    pub name: String,
    /// The param index (captures of a nested body are params too).
    pub param: Option<u32>,
    /// A capture of a nested body: the closure's `fn` / `spawn` token span.
    pub capture: Option<Span>,
    /// A capture the closure value holds by `&mut`: a call of the closure may write it or lend
    /// it again.
    pub capture_mut: bool,
    pub temp: bool,
    /// A binding as seen by a match guard: the guard runs after the arm's bindings are bound
    /// from the scrutinee, so inside it they are read-only views of the scrutinee (no move, no
    /// write, no `&mut`).
    pub guard: bool,
    /// The binding node (a temporary: its expression node; borrow dump `N` of drops).
    pub node: u32,
    /// The temporary standing for the storage of an immutable `static` (never moved from).
    pub static_root: bool,
}

/// What a return is checked against.
#[derive(Clone, Debug)]
pub enum RetCheck {
    /// The result carries no origins.
    None,
    /// A nested body (closure): only E0512.
    LocalsOnly,
    /// `from` params (index, copy), the `From` clause span (the return type's without a clause),
    /// and whether the signature has a `from` clause.
    From(Vec<(u32, bool)>, Span, bool),
}

/// What a match guard may not do: it only reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GuardEffect {
    /// A by-value use of a non-Copy value.
    Move(Place),
    Assign(Place),
    MutBorrow(Place),
    /// A `?`.
    Try,
}

/// Why a by-value use of a non-Copy place is E0504.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MoveOut {
    /// Behind a dereference or an index, or a guard binding.
    Behind,
    /// A capture of the closure with this (`fn` / `spawn` token) span.
    Capture(Span),
    /// A part of this place, whose type has an `impl Drop`: its destructor reads every part.
    Drop(Place),
    /// An immutable `static` or a part of it.
    Static,
    /// A part of this place (a local or a temporary): a non-Copy value is moved whole, or taken
    /// apart whole by a pattern.
    Part(Place, PartFix),
}

/// The fix an E0504 `part` names, by what is moved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PartFix {
    /// A `&mut` part: pass it on by a reborrow.
    Reborrow,
    /// Take the owner apart with this pattern (every part bound).
    Pattern(String),
    /// The owner cannot be taken apart here (an enum on the way, a field private to another
    /// module): `mem.replace`.
    Replace,
}

#[derive(Clone, Debug)]
pub struct Body {
    pub locals: Vec<Local>,
    pub blocks: Vec<Block>,
    pub loans: Vec<Loan>,
    /// Move sites: the moved place and the move's span.
    pub move_sites: Vec<(Place, Span)>,
    /// E0504: by-value uses of non-Copy places that cannot be moved from (place, span, why),
    /// kept as copies in the CFG.
    pub bad_moves: Vec<(Place, Span, MoveOut)>,
    /// Moves made by an operand that has later operands ([`Builder::hold`]): the move site and
    /// the span of the whole expression (an E0502 use inside it is case `operand`).
    pub operand_moves: Vec<(u32, Span)>,
    /// E0505: what a match guard (or a closure written in one) does besides reading.
    pub guard_effects: Vec<(Span, GuardEffect)>,
    pub ret: RetCheck,
    /// The `from` clauses of the fn's params (a nested body: none).
    pub declared: Vec<Store>,
    /// A closure or spawn body: its params and captures take no `from` clause.
    pub is_nested: bool,
    /// Bodies of closures and spawn blocks inside this one.
    pub nested: Vec<Body>,
    /// The join local of each `scope` statement with the statement's span: a read of it is the
    /// join of the threads spawned there.
    pub joins: Vec<(LocalId, Span)>,
    /// Why this body cannot be checked (nothing is reported for it).
    pub unsupported: Option<&'static str>,
    /// A disagreement with stave-sema the builder found (a compiler bug): the body is not
    /// checked, and the run fails as an internal error.
    pub internal: Option<&'static str>,
}

impl Body {
    pub fn n_params(&self) -> usize {
        self.locals.iter().filter(|l| l.param.is_some()).count()
    }
}

/// A two-phase borrow's temporary, reservation loan, and activation loan.
type Activation = (LocalId, LoanId, LoanId);

/// A value under construction.
#[derive(Clone, Debug)]
enum V {
    P(Place),
    Const,
    Never,
}

/// A postfix chain's state between steps.
enum Cur {
    /// A path with the token of its last segment.
    Path(Res, u32),
    TypeRel,
    Val(V),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Move,
    Ref(bool),
    /// Guard pre-binding: read-only guard locals ([`Local::guard`]).
    Peek,
}

struct LoopCx {
    head: BlockId,
    exit: BlockId,
    scopes: usize,
    temps: usize,
}

/// Builds one body.
pub struct Builder<'a, 'b> {
    icx: ItemCx<'a>,
    tb: &'a TypedBody,
    tys: &'b mut Tys<'a>,
    body: Body,
    cur: Option<BlockId>,
    scopes: Vec<Vec<LocalId>>,
    loops: Vec<LoopCx>,
    temps: Vec<LocalId>,
    by_node: FxHashMap<u32, LocalId>,
    /// The node spans computed so far, by node index from the body's first node (sized on first
    /// use): a node's span is taken for its temporary, its statement, and each of its operands,
    /// and finding a node's end walks down to its last token and re-lexes it.
    spans: Vec<Option<Span>>,
    depth: u32,
    /// Nested body: the closure (or spawn) node range; outer locals become captures.
    range: Option<(u32, u32)>,
    captures: Vec<(u32, LocalId)>,
    /// Nested body: the closure's `fn` / `spawn` token span (captures' E0504 label).
    closure_span: Option<Span>,
    /// The closure bodies of the fn with their capture analysis (stave-sema `closures`), shared by
    /// the nested builders.
    infos: std::rc::Rc<Vec<stave_sema::closures::ClosureInfo>>,
    /// Open `scope` statements: (scope depth of the block, hidden local accumulating the capture
    /// loans of every thread spawned into it, read where the scope is left: the join).
    joins: Vec<(usize, LocalId)>,
    /// Binding nodes of the match guards being built (in a nested body: those of the enclosing
    /// guards it is built in); see [`Local::guard`].
    guard_binds: Vec<u32>,
    /// Inside a match guard (or a closure written in one): the guard only reads.
    in_guard: u32,
    /// The pattern being bound by [`Builder::destructure`]: the projection length of its source
    /// and whether it took a non-Copy part by value.
    destr: Option<(usize, bool)>,
    /// The scrutinees of the guards being built in this body (a `&mut` of one is E0501 already).
    guard_scruts: Vec<Place>,
    /// The temporary holding the array a `for` loop iterates by value: its elements are moved
    /// into the loop binding one per iteration (not an E0504 move out of an index).
    owned_elems: Option<LocalId>,
    /// The local each `old(e)` of the `ensures` clauses names (by the call's `(` token): `e`
    /// evaluated at entry.
    olds: FxHashMap<u32, LocalId>,
    /// The local `result` names in the `ensures` clauses: the returned value, from the return
    /// to the end of the clauses.
    result: Option<LocalId>,
    /// The block temporaries of reference chunk bindings, with their store back.
    chunk_temps: FxHashMap<LocalId, ChunkWriteBack>,
}

/// A by-reference chunk binding's store back (see `Builder::chunk_ref_bind`): the chunk of the
/// sequence (through the loop's loan), `&mut`, the loop's node.
#[derive(Clone)]
struct ChunkWriteBack {
    chunk: Place,
    m: bool,
    node: u32,
}

/// Builds the body of fn decl node `fn_node` (params in `params`, return type `ret_ty`, signature
/// `from` list, the params' `from` clauses).
pub fn build_fn<'a>(
    icx: ItemCx<'a>,
    tb: &'a TypedBody,
    tys: &mut Tys<'a>,
    fn_node: u32,
    (params, ret_ty): (&[TypeId], TypeId),
    ret: RetCheck,
    declared: Vec<Store>,
) -> Body {
    let mut b = Builder::new(icx, tb, tys, None);
    let infos = {
        let ov = &b.tys.ov;
        stave_sema::closures::closures(&icx, tb, ov, fn_node)
    };
    b.infos = std::rc::Rc::new(infos);
    b.body.ret = ret;
    b.body.declared = declared;
    b.fn_body(fn_node, params, ret_ty);
    b.finish()
}

impl<'a, 'b> Builder<'a, 'b> {
    fn new(icx: ItemCx<'a>, tb: &'a TypedBody, tys: &'b mut Tys<'a>, range: Option<(u32, u32)>) -> Self {
        let body = Body {
            locals: Vec::new(),
            blocks: vec![Block { stmts: Vec::new(), term: Terminator::Unreachable }],
            loans: Vec::new(),
            move_sites: Vec::new(),
            bad_moves: Vec::new(),
            guard_effects: Vec::new(),
            operand_moves: Vec::new(),
            ret: RetCheck::None,
            declared: Vec::new(),
            is_nested: range.is_some(),
            nested: Vec::new(),
            joins: Vec::new(),
            unsupported: None,
            internal: None,
        };
        Builder {
            icx,
            tb,
            tys,
            body,
            cur: Some(0),
            scopes: Vec::new(),
            loops: Vec::new(),
            temps: Vec::new(),
            by_node: FxHashMap::default(),
            spans: Vec::new(),
            depth: 0,
            range,
            captures: Vec::new(),
            closure_span: None,
            infos: std::rc::Rc::new(Vec::new()),
            joins: Vec::new(),
            guard_binds: Vec::new(),
            in_guard: 0,
            destr: None,
            guard_scruts: Vec::new(),
            owned_elems: None,
            olds: FxHashMap::default(),
            result: None,
            chunk_temps: FxHashMap::default(),
        }
    }

    fn finish(self) -> Body {
        self.body
    }

    fn unsupported(&mut self, why: &'static str) {
        if self.body.unsupported.is_none() {
            self.body.unsupported = Some(why);
        }
    }

    /// The type of a value the type checker recorded no type for. An unknown type counts as
    /// Copy and as having no drop glue, so a value of one would be moved, taken apart, and
    /// outlived by its borrows unseen: the body is not checked, and the run fails.
    fn untyped(&mut self) -> TypeId {
        self.body.internal.get_or_insert("a value without a type");
        TypeId::NONE
    }

    // ---------- CFG primitives ----------

    fn new_block(&mut self) -> BlockId {
        self.body.blocks.push(Block { stmts: Vec::new(), term: Terminator::Unreachable });
        (self.body.blocks.len() - 1) as BlockId
    }

    fn open(&self) -> bool {
        self.cur.is_some()
    }

    fn terminate(&mut self, t: Terminator) {
        if let Some(b) = self.cur.take() {
            self.body.blocks[b as usize].term = t;
        }
    }

    fn goto(&mut self, b: BlockId) {
        self.terminate(Terminator::Goto(b));
    }

    fn switch_to(&mut self, b: BlockId) {
        self.cur = Some(b);
    }

    fn push(&mut self, kind: StmtKind, span: Span, node: u32) {
        if let Some(b) = self.cur {
            self.body.blocks[b as usize].stmts.push(Stmt { kind, span, node });
        }
    }

    // ---------- locals, temps, loans ----------

    fn span(&mut self, n: u32) -> Span {
        let lo = self.tb.lo;
        if self.spans.is_empty() {
            self.spans = vec![None; (self.tb.node + 1).saturating_sub(lo) as usize];
        }
        let slot = n.checked_sub(lo).map(|i| i as usize).filter(|&i| i < self.spans.len());
        if let Some(s) = slot.and_then(|i| self.spans[i]) {
            return s;
        }
        let s = self.icx.node_span(n);
        if let Some(i) = slot {
            self.spans[i] = Some(s);
        }
        s
    }

    fn nty(&self, n: u32) -> TypeId {
        self.tb.ty(n)
    }

    fn new_local(&mut self, ty: TypeId, mutable: bool, span: Span, name: String, temp: bool) -> LocalId {
        self.body.locals.push(Local {
            ty,
            mutable,
            span,
            name,
            param: None,
            capture: None,
            capture_mut: false,
            temp,
            guard: false,
            node: u32::MAX,
            static_root: false,
        });
        (self.body.locals.len() - 1) as LocalId
    }

    /// The local of binding node `b` (created on first sight: patterns bind in place). A
    /// binding in a later alternative of an or-pattern names the first alternative's local.
    fn binding_local(&mut self, b: u32) -> LocalId {
        let icx = self.icx;
        let b = icx.res.canonical(b);
        if let Some(&l) = self.by_node.get(&b) {
            return l;
        }
        let ty = self.tb.local(BindingRef { node: b }).unwrap_or(TypeId::NONE);
        let bind = icx.res.binding(BindingRef { node: b });
        let mutable = bind.is_some_and(|x| x.mutable);
        let span = match bind {
            Some(x) => icx.tok_span(x.token),
            None => icx.node_span(b),
        };
        let l = self.new_local(ty, mutable, span, String::new(), false);
        self.body.locals[l as usize].node = b;
        self.by_node.insert(b, l);
        if let Some(s) = self.scopes.last_mut() {
            s.push(l);
        }
        l
    }

    /// The local a resolution names (an outer local of a nested body becomes a capture).
    fn local_of(&mut self, r: Res) -> Option<LocalId> {
        let b = match r {
            Res::Local(b) | Res::SelfValue(b) => b.node,
            _ => return None,
        };
        if let Some(&l) = self.by_node.get(&b) {
            return Some(l);
        }
        if let Some((lo, hi)) = self.range
            && !(lo..=hi).contains(&b)
        {
            let icx = self.icx;
            let ty = self.tb.local(BindingRef { node: b }).unwrap_or(TypeId::NONE);
            let bind = icx.res.binding(BindingRef { node: b });
            let mutable = bind.is_some_and(|x| x.mutable && x.kind != stave_sema::BindKind::SelfParam);
            let (span, name) = match bind {
                Some(x) if x.kind == stave_sema::BindKind::SelfParam => {
                    (icx.tok_span(x.token), "self".to_string())
                }
                Some(x) => (icx.tok_span(x.token), icx.tok_str(x.token).to_string()),
                None => (icx.node_span(b), icx.node_text(b).to_string()),
            };
            let l = self.new_local(ty, mutable, span, name, false);
            let np = self.body.n_params() as u32;
            self.body.locals[l as usize].param = Some(np);
            self.body.locals[l as usize].capture = self.closure_span;
            self.body.locals[l as usize].guard = self.guard_binds.contains(&b);
            self.body.locals[l as usize].node = b;
            self.by_node.insert(b, l);
            self.captures.push((b, l));
            return Some(l);
        }
        None
    }

    /// A temporary for the value of node `n` (named by its source text when reported).
    fn temp(&mut self, ty: TypeId, n: u32) -> LocalId {
        let span = self.span(n);
        let t = self.new_local(ty, false, span, String::new(), true);
        self.body.locals[t as usize].node = n;
        t
    }

    /// Assigns `rv` to a fresh temporary of type `ty`.
    fn put_temp(&mut self, rv: Rvalue, ty: TypeId, n: u32) -> V {
        if !self.open() {
            return V::Never;
        }
        if self.tys.is_unknown(ty) {
            self.untyped();
        }
        let t = self.temp(ty, n);
        let span = self.span(n);
        self.push(StmtKind::Assign { dest: Place::local(t, ty), rv, init: true }, span, n);
        if self.tys.needs_drop(ty) {
            self.temps.push(t);
        }
        V::P(Place::local(t, ty))
    }

    fn new_loan(&mut self, place: &Place, mutable: bool, span: Span, node: u32, reserved: bool) -> LoanId {
        self.body.loans.push(Loan {
            place: place.clone(),
            mutable,
            span,
            node,
            reserved,
            capture: false,
            whole: None,
            cast: false,
        });
        (self.body.loans.len() - 1) as LoanId
    }

    /// `&place` / `&mut place` into a temporary; `two_phase`: the loan is a reservation, returned
    /// with the activation loan.
    fn borrow(
        &mut self,
        place: Place,
        mutable: bool,
        two_phase: bool,
        span: Span,
        n: u32,
    ) -> (V, Option<(LocalId, LoanId, LoanId)>) {
        if !self.open() {
            return (V::Never, None);
        }
        if mutable && self.in_guard > 0 && !self.guard_reported(&place) {
            self.body.guard_effects.push((span, GuardEffect::MutBorrow(place.clone())));
        }
        let ty = self.tys.ov.mk_ref(mutable, place.ty);
        // A `&` of an immutable static's storage (it never ends, is never moved from, and is
        // written only through interior mutability) borrows nothing: the reference is a
        // constant, as a string literal is, wherever it flows.
        if !mutable && self.body.locals.get(place.local as usize).is_some_and(|l| l.static_root) {
            let t = self.new_local(ty, false, span, String::new(), true);
            let c = Operand {
                kind: OpKind::Const,
                place: Place::local(0, TypeId::NONE),
                span,
                node: n,
                site: u32::MAX,
            };
            self.push(
                StmtKind::Assign { dest: Place::local(t, ty), rv: Rvalue::Use(c), init: true },
                span,
                n,
            );
            return (V::P(Place::local(t, ty)), None);
        }
        let res = self.new_loan(&place, mutable && !two_phase, span, n, two_phase);
        let act = two_phase.then(|| self.new_loan(&place, true, span, n, false));
        let t = self.new_local(ty, false, span, String::new(), true);
        self.push(
            StmtKind::Assign {
                dest: Place::local(t, ty),
                rv: Rvalue::Ref { place, mutable, loan: res },
                init: true,
            },
            span,
            n,
        );
        (V::P(Place::local(t, ty)), act.map(|a| (t, res, a)))
    }

    /// A write or `&mut` of `p` in a guard that is reported otherwise: of a guard binding (E0503),
    /// of the guard's scrutinee (E0501), or the reborrow of a temporary `&mut` (the borrow or
    /// call that made it is).
    fn guard_reported(&self, p: &Place) -> bool {
        let Some(l) = self.body.locals.get(p.local as usize) else { return false };
        l.guard
            || (l.temp && matches!(p.proj.as_slice(), [Proj::Deref(_)]))
            || self
                .guard_scruts
                .iter()
                .any(|s| crate::check::places_conflict(p, s, crate::check::Access::MutBorrow))
    }

    fn place_of(&mut self, v: V, n: u32) -> Option<Place> {
        match v {
            V::P(p) => Some(p),
            V::Const => {
                let ty = self.nty(n);
                let op = self.const_op(n);
                match self.put_temp(Rvalue::Use(op), ty, n) {
                    V::P(p) => Some(p),
                    _ => None,
                }
            }
            V::Never => None,
        }
    }

    fn const_op(&mut self, n: u32) -> Operand {
        Operand {
            kind: OpKind::Const,
            place: Place::local(0, TypeId::NONE),
            span: self.span(n),
            node: n,
            site: u32::MAX,
        }
    }

    /// The value of `v` as a by-value operand (moves unless Copy). Moves out of a dereference
    /// or an index are taken as copies (no code exists for them; see the crate docs).
    fn operand(&mut self, v: V, n: u32) -> Option<Operand> {
        match v {
            V::P(p) => {
                let span = self.span(n);
                Some(self.op_of_place(p, span, n, true))
            }
            V::Const => Some(self.const_op(n)),
            V::Never => None,
        }
    }

    fn op_of_place(&mut self, p: Place, span: Span, n: u32, consume: bool) -> Operand {
        let by_value = consume && !self.tys.is_copy(p.ty);
        let why = if by_value { self.move_out(&p, false) } else { None };
        // a `for` binding takes a fresh element of the owned array on every iteration
        let fresh_elem = self.owned_elem(&p);
        let movable = by_value && why.is_none() && !fresh_elem;
        // passing on a fresh `&mut` is the borrow that made it, judged there
        let fresh_ref = p.proj.is_empty()
            && self.body.locals.get(p.local as usize).is_some_and(|l| l.temp)
            && self.tys.deref(p.ty).is_some();
        if by_value && why.is_none() && self.in_guard > 0 && !fresh_ref {
            self.body.guard_effects.push((span, GuardEffect::Move(p.clone())));
        }
        if let Some(why) = why {
            self.body.bad_moves.push((p.clone(), span, why));
        }
        if movable {
            if p.proj.is_empty()
                && let Some(i) = self.temps.iter().rposition(|&t| t == p.local)
            {
                self.temps.remove(i);
            }
            self.body.move_sites.push((p.clone(), span));
            let site = (self.body.move_sites.len() - 1) as u32;
            Operand { kind: OpKind::Move, place: p, span, node: n, site }
        } else {
            Operand { kind: OpKind::Copy, place: p, span, node: n, site: u32::MAX }
        }
    }

    /// Why a by-value use of the non-Copy place `p` is E0504 (`None`: it may move). Every by-value
    /// use ends here: an operand or receiver through [`Self::op_of_place`], and a binding of a
    /// `let`, `match` (every alternative, nested sub-patterns), or `for` pattern (over a range, an
    /// owned array, an iterator) through [`Self::bind_name`] (`part`: a binding inside a pattern
    /// that takes its value apart, [`Self::destructure`]). Params of fns and closures are names.
    /// Only a whole value moves: a part of a place is E0504 unless a pattern takes the whole
    /// value apart, and not even then when the value has a destructor.
    fn move_out(&mut self, p: &Place, part: bool) -> Option<MoveOut> {
        let l = self.body.locals.get(p.local as usize)?;
        let (capture, guard, stat, lty) = (l.capture, l.guard, l.static_root, l.ty);
        let from = usize::from(self.owned_elem(p));
        let behind =
            p.proj[from..].iter().any(|x| matches!(x, Proj::Deref(_) | Proj::Index | Proj::Column(_)));
        if stat {
            Some(MoveOut::Static)
        } else if let Some(c) = capture.filter(|_| !behind) {
            Some(MoveOut::Capture(c))
        } else if behind || guard {
            Some(MoveOut::Behind)
        } else if let Some(o) = self.drop_owner(p, from) {
            Some(MoveOut::Drop(o))
        } else if !part && p.proj.len() > from {
            let owner = Place { local: p.local, proj: p.proj[..from].to_vec(), ty: TypeId::NONE };
            let fix = if matches!(self.tys.deref(p.ty), Some((_, DerefKind::Mut))) {
                PartFix::Reborrow
            } else {
                let mut t = lty;
                if from > 0 {
                    t = self.tys.elem(t);
                }
                let path = p.proj[from..].to_vec();
                match self.part_pattern(t, &path, "e", 0) {
                    Some(pat) => PartFix::Pattern(pat),
                    None => PartFix::Replace,
                }
            };
            Some(MoveOut::Part(owner, fix))
        } else {
            None
        }
    }

    /// The pattern that takes a value of type `t` apart along projections `path`: every part
    /// bound by name (a tuple element `{prefix}{i}`), the part on the path nested; `None` when a
    /// part cannot be named here (an enum, a field private to another module).
    fn part_pattern(&mut self, t: TypeId, path: &[Proj], prefix: &str, depth: u32) -> Option<String> {
        if depth > 64 {
            return None;
        }
        let next = |i: usize| match path.first() {
            Some(Proj::Field(f)) if *f as usize == i && path.len() > 1 => Some(&path[1..]),
            _ => None,
        };
        match self.tys.kind(t) {
            TypeKind::Tuple(ts) => {
                let mut parts = Vec::new();
                for (i, &et) in ts.iter().enumerate() {
                    let name = format!("{prefix}{i}");
                    parts.push(match next(i) {
                        Some(rest) => self.part_pattern(et, rest, &format!("{name}_"), depth + 1)?,
                        None => name,
                    });
                }
                let comma = if parts.len() == 1 { "," } else { "" };
                Some(format!("({}{comma})", parts.join(", ")))
            }
            TypeKind::Adt { def, .. } => {
                let prog = self.tys.prog;
                let here = prog.item(self.icx.item).map(|i| i.module);
                let decl = prog.decl(def)?;
                let adt = self.tys.sigs.adt(def)?;
                if adt.is_enum {
                    return None;
                }
                let same = Some(decl.module) == here;
                let fields: Vec<(String, bool)> = adt
                    .variants
                    .first()?
                    .fields
                    .iter()
                    .map(|f| (prog.strings.get(f.name).to_string(), f.is_pub))
                    .collect();
                let mut parts = Vec::new();
                for (i, (name, is_pub)) in fields.into_iter().enumerate() {
                    if !is_pub && !same {
                        return None;
                    }
                    parts.push(match next(i) {
                        Some(rest) => {
                            let ft = self.tys.field_ty(t, 0, i as u32);
                            let sub = self.part_pattern(ft, rest, &format!("{name}_"), depth + 1)?;
                            format!("{name}: {sub}")
                        }
                        None => name,
                    });
                }
                let tn = prog.strings.get(decl.name);
                Some(if parts.is_empty() {
                    format!("{tn}.{{}}")
                } else {
                    format!("{tn}.{{ {} }}", parts.join(", "))
                })
            }
            _ => None,
        }
    }

    /// `p` is an element (or a part of one) of the array a `for` loop owns and moves out
    /// element by element.
    fn owned_elem(&self, p: &Place) -> bool {
        self.owned_elems == Some(p.local) && p.proj.first() == Some(&Proj::Index) && !p.has_deref()
    }

    /// The longest proper prefix of `p` (from projection `from` on) whose type has an `impl Drop` (moving `p` out of it would
    /// leave its destructor a moved-out part to read), when `p` is reached by fields only.
    fn drop_owner(&mut self, p: &Place, from: usize) -> Option<Place> {
        let mut t = self.body.locals.get(p.local as usize)?.ty;
        if from > 0 {
            t = self.tys.elem(t);
        }
        let mut owner = None;
        for (i, pr) in p.proj.iter().enumerate().skip(from) {
            if self.tys.has_drop_impl(t) {
                owner = Some(Place { local: p.local, proj: p.proj[..i].to_vec(), ty: t });
            }
            t = match *pr {
                Proj::Field(f) => self.tys.field_ty(t, 0, f),
                Proj::EnumField(v, f) => self.tys.field_ty(t, v, f),
                _ => return None,
            };
        }
        owner
    }

    /// Operand `o` of an expression whose later operands are still to be evaluated. Operands are
    /// evaluated left to right and a by-value operand takes its value when it is evaluated, so
    /// a move of a named place is made here, into a temporary the expression then consumes: the
    /// later operands, and the statements they make (nested calls, borrows), see the place
    /// moved.
    fn hold(&mut self, o: Operand, whole: Span) -> Operand {
        if o.kind != OpKind::Move || !self.open() || self.body.locals[o.place.local as usize].temp {
            return o;
        }
        let ty = o.place.ty;
        let (span, node) = (o.span, o.node);
        self.body.operand_moves.push((o.site, whole));
        let t = self.new_local(ty, false, span, String::new(), true);
        self.body.locals[t as usize].node = node;
        self.push(StmtKind::Assign { dest: Place::local(t, ty), rv: Rvalue::Use(o), init: true }, span, node);
        if self.tys.needs_drop(ty) {
            self.temps.push(t);
        }
        self.op_of_place(Place::local(t, ty), span, node, true)
    }

    /// The operands of expressions `ns` of the expression at `whole`, each held
    /// ([`Self::hold`]) until the last is evaluated.
    fn operands_in_order(&mut self, ns: &[u32], whole: Span) -> Option<Vec<Operand>> {
        let mut out: Vec<Operand> = Vec::with_capacity(ns.len());
        for &c in ns {
            if let Some(prev) = out.pop() {
                let h = self.hold(prev, whole);
                out.push(h);
            }
            let v = self.expr(c);
            out.push(self.operand(v, c)?);
        }
        Some(out)
    }

    /// A non-consuming read of `v`.
    fn read(&mut self, v: V, n: u32) -> Option<Operand> {
        match v {
            V::P(p) => {
                let span = self.span(n);
                Some(self.op_of_place(p, span, n, false))
            }
            V::Const => Some(self.const_op(n)),
            V::Never => None,
        }
    }

    // ---------- scopes ----------

    fn push_scope(&mut self) {
        self.scopes.push(Vec::new());
    }

    /// Ends the innermost scope at `at` (the block's `}`).
    fn pop_scope(&mut self, at: Option<Span>) {
        let Some(s) = self.scopes.pop() else { return };
        for &l in s.iter().rev() {
            self.end_local(l, true, at);
        }
    }

    /// The drop and storage-dead of local `l` at `at` (default: its binding).
    fn end_local(&mut self, l: LocalId, dead: bool, at: Option<Span>) {
        let loc = &self.body.locals[l as usize];
        let (ty, node) = (loc.ty, loc.node);
        let span = at.unwrap_or(loc.span);
        if dead && let Some(w) = self.chunk_temps.get(&l).cloned() {
            // the block of a reference chunk binding is left (its end, `break`, `continue`,
            // `return`, `?`): the code stores the chunk back here
            self.chunk_write_back(w, span);
        }
        if self.tys.needs_drop(ty) {
            self.push(StmtKind::Drop(Place::local(l, ty)), span, node);
        }
        if dead {
            self.push(StmtKind::StorageDead(l), span, node);
        }
    }

    /// Drops the statement temporaries from `mark` on.
    fn flush_temps(&mut self, mark: usize) {
        if mark >= self.temps.len() {
            return;
        }
        let ts: Vec<LocalId> = self.temps.drain(mark..).collect();
        for &t in ts.iter().rev() {
            self.end_local(t, true, None);
        }
    }

    /// Drops on an exit path (temps and scopes above the marks), without forgetting them. A
    /// `scope` statement left on the way joins its threads before its own locals are dropped.
    fn exit_path(&mut self, scopes: usize, temps: usize, dead: bool, at: Span) {
        let ts: Vec<LocalId> = self.temps.get(temps..).unwrap_or(&[]).to_vec();
        for &t in ts.iter().rev() {
            self.end_local(t, dead, Some(at));
        }
        let ss: Vec<Vec<LocalId>> = self.scopes.get(scopes..).unwrap_or(&[]).to_vec();
        for (k, s) in ss.iter().enumerate().rev() {
            let depth = scopes + k + 1;
            let js: Vec<LocalId> = self.joins.iter().filter(|j| j.0 == depth).map(|j| j.1).collect();
            for j in js {
                self.join_read(j, at, u32::MAX);
            }
            for &l in s.iter().rev() {
                self.end_local(l, dead, Some(at));
            }
        }
    }

    /// The join of a `scope`'s threads: a read of the local holding their capture loans.
    fn join_read(&mut self, j: LocalId, at: Span, node: u32) {
        let jt = self.body.locals[j as usize].ty;
        let r = Operand { kind: OpKind::Copy, place: Place::local(j, jt), span: at, node, site: u32::MAX };
        self.push(StmtKind::Read(r), at, node);
    }

    // ---------- fn body ----------

    fn fn_body(&mut self, fn_node: u32, params: &[TypeId], ret: TypeId) {
        let icx = self.icx;
        self.push_scope();
        if let Some(ps) = icx.child(fn_node, NodeKind::Params) {
            for (i, p) in icx.kids_vec(ps).into_iter().enumerate() {
                let ty = params.get(i).copied().unwrap_or(TypeId::NONE);
                let (name, span, key) = match icx.child(p, NodeKind::SelfParam) {
                    Some(sp) => ("self".to_string(), icx.node_span(sp), Some(sp)),
                    None => (String::new(), icx.tok_span(icx.a(p)), None),
                };
                let l = self.new_local(ty, false, span, name, false);
                self.body.locals[l as usize].param = Some(i as u32);
                self.body.locals[l as usize].node = p;
                self.by_node.insert(p, l);
                if let Some(k) = key {
                    self.by_node.insert(k, l);
                }
                if let Some(s) = self.scopes.last_mut() {
                    s.push(l);
                }
            }
        }
        self.contract_entry(ret);
        if let Some(b) = icx.child(fn_node, NodeKind::Block) {
            self.block(b);
        }
        self.end_of_body(fn_node);
    }

    fn end_of_body(&mut self, n: u32) {
        if self.open() {
            let at = self.body_end(n);
            if self.result.is_some() {
                let op = self.const_op(n);
                self.leave(op, at);
                return;
            }
            self.exit_path(0, 0, false, at);
            let op = self.const_op(n);
            let span = self.span(n);
            self.push(StmtKind::Return(op), span, n);
            self.terminate(Terminator::Return);
        }
    }

    // ---------- contract clauses ----------

    /// The fn's clauses at entry, as the code they are there: each `requires` in order, then the
    /// `old(e)` values of the `ensures` clauses, each held in a local that lives to the clauses
    /// that read it (so what it borrows stays borrowed across the body).
    fn contract_entry(&mut self, ret: TypeId) {
        let tb = self.tb;
        let Some(c) = tb.contracts.as_deref() else { return };
        for cl in &c.requires {
            self.clause(cl.expr);
        }
        for &(paren, e) in &c.olds {
            if !self.open() {
                return;
            }
            let mark = self.temps.len();
            let v = self.expr(e);
            let Some(op) = self.operand(v, e) else { return };
            let (ty, span) = (self.nty(e), self.span(e));
            let l = self.new_local(ty, false, span, String::new(), true);
            self.body.locals[l as usize].node = e;
            self.push(
                StmtKind::Assign { dest: Place::local(l, ty), rv: Rvalue::Use(op), init: true },
                span,
                e,
            );
            self.flush_temps(mark);
            self.olds.insert(paren, l);
        }
        if let Some(first) = c.ensures.first() {
            let span = self.icx.tok_span(self.icx.a(first.node));
            let l = self.new_local(ret, false, span, "result".to_string(), false);
            self.body.locals[l as usize].node = c.result;
            self.by_node.insert(c.result, l);
            self.result = Some(l);
        }
    }

    /// A clause where it runs: its expression evaluated and read.
    fn clause(&mut self, e: u32) {
        if !self.open() {
            return;
        }
        let mark = self.temps.len();
        let v = self.expr(e);
        if let Some(op) = self.read(v, e)
            && op.kind != OpKind::Const
        {
            let span = op.span;
            self.push(StmtKind::Read(op), span, e);
        }
        self.flush_temps(mark);
    }

    /// Returns `op` from the fn at `at`. With `ensures` clauses the value goes to `result`, the
    /// scopes are left, the clauses run in order over the params as the body left them, and
    /// `result` is returned: it is live across the clauses, as is every `old(e)` a clause reads.
    fn leave(&mut self, op: Operand, at: Span) {
        let (span, node) = (op.span, op.node);
        let Some(res) = self.result else {
            self.push(StmtKind::Return(op), span, node);
            self.exit_path(0, 0, false, at);
            self.terminate(Terminator::Return);
            return;
        };
        let dest = Place::local(res, self.body.locals[res as usize].ty);
        self.push(StmtKind::Assign { dest: dest.clone(), rv: Rvalue::Use(op), init: true }, span, node);
        self.exit_path(0, 0, false, at);
        let tb = self.tb;
        let clauses = tb.contracts.as_deref().map_or(&[][..], |c| &c.ensures);
        // a clause is built once per return: the bindings of a `match` in it are fresh locals
        // each time
        let first = self.body.locals.len() as LocalId;
        for cl in clauses {
            self.clause(cl.expr);
            self.by_node.retain(|_, l| *l < first);
        }
        if !self.open() {
            return;
        }
        let rop = self.op_of_place(dest, span, node, true);
        self.push(StmtKind::Return(rop), span, node);
        self.terminate(Terminator::Return);
    }

    // ---------- statements ----------

    /// The span of the `}` closing block `b`.
    fn close_span(&self, b: u32) -> Span {
        self.icx.tok_span(self.icx.close(self.icx.a(b)))
    }

    /// The end of a fn or closure body: its block's `}`.
    fn body_end(&mut self, n: u32) -> Span {
        match self.icx.child(n, NodeKind::Block) {
            Some(b) => self.close_span(b),
            None => self.span(n),
        }
    }

    fn block(&mut self, b: u32) {
        self.push_scope();
        for s in self.icx.kids_vec(b) {
            if !self.open() {
                break;
            }
            self.stmt(s);
        }
        let at = self.close_span(b);
        self.pop_scope(Some(at));
    }

    fn stmt(&mut self, s: u32) {
        let icx = self.icx;
        let a = icx.a(s);
        let kids = icx.kids_vec(s);
        let mark = self.temps.len();
        let is = |t: Term| icx.is(a, t);
        if is(Term::KwLet) {
            self.let_stmt(&kids);
        } else if is(Term::KwIf) {
            if let Some(&i) = kids.first() {
                self.if_stmt(i);
            }
        } else if is(Term::KwWhile) {
            if let (Some(&c), Some(&b)) = (kids.first(), kids.get(1)) {
                self.while_stmt(c, b);
            }
        } else if is(Term::KwLoop) {
            if let Some(&b) = kids.first() {
                self.loop_stmt(b);
            }
        } else if is(Term::KwFor) {
            if let Some(&f) = kids.first() {
                self.for_stmt(f);
            }
        } else if is(Term::KwMatch) {
            self.match_(s, &kids, false);
        } else if is(Term::KwReturn) {
            self.return_stmt(s, kids.first().copied());
        } else if is(Term::KwBreak) || is(Term::KwContinue) {
            self.break_continue(s, is(Term::KwBreak));
        } else if is(Term::KwSpawn) {
            if let Some(&b) = kids.first() {
                self.spawn(s, b);
            }
        } else if is(Term::KwScope) {
            if let Some(&b) = kids.first() {
                self.scope_stmt(s, b);
            }
        } else if is(Term::KwUnsafe) {
            if let Some(&b) = kids.iter().find(|&&c| icx.kind(c) == Some(NodeKind::Block)) {
                self.block(b);
            }
        } else if is(Term::KwStrict) || is(Term::KwEmulate) || is(Term::LBrace) {
            if let Some(&b) = kids.first() {
                self.block(b);
            }
        } else if let Some(&e) = kids.first() {
            self.expr_stmt(e);
        }
        if self.open() {
            self.flush_temps(mark);
        } else {
            self.temps.truncate(mark);
        }
    }

    fn let_stmt(&mut self, kids: &[u32]) {
        let icx = self.icx;
        let Some(&pat) = kids.first() else { return };
        let Some(init) = kids.iter().copied().find(|&c| icx.is_expr(c)) else { return };
        let v = self.expr(init);
        if matches!(v, V::Never) || !self.open() {
            return;
        }
        if icx.is(icx.a(pat), Term::Underscore) {
            return; // `let _ = e;` moves nothing; a temporary is dropped at the end of the statement
        }
        let Some(p) = self.place_of(v, init) else { return };
        self.destructure(pat, p, Mode::Move, init);
    }

    fn expr_stmt(&mut self, es: u32) {
        let icx = self.icx;
        let kids = icx.kids_vec(es);
        if kids.len() >= 3 {
            let (lhs, op, rhs) = (kids[0], kids[1], kids[2]);
            let compound = !icx.is(icx.a(op), Term::Eq);
            let lv = self.expr(lhs);
            if self.in_guard > 0
                && let V::P(p) = &lv
                && !self.guard_reported(p)
            {
                let span = self.span(lhs);
                self.body.guard_effects.push((span, GuardEffect::Assign(p.clone())));
            }
            let Some(lp) = (match lv {
                V::P(p) => Some(p),
                _ => None,
            }) else {
                // an assignment to a `static mut`: the right side is consumed like any other
                let rv = self.expr(rhs);
                if let Some(op) = self.operand(rv, rhs)
                    && op.kind != OpKind::Const
                {
                    let span = op.span;
                    self.push(StmtKind::Read(op), span, rhs);
                }
                return;
            };
            // `p op= e` is `p = p op e`: the left operand is a by-value use of `p` evaluated
            // before `e` (a move when not Copy, so `e` may not use `p`, and a place that cannot
            // be moved from is E0504), and the write re-initializes `p`
            let old = if !compound {
                None
            } else if self.tys.is_copy(lp.ty) {
                self.read(V::P(lp.clone()), lhs)
            } else {
                let span = self.span(lhs);
                let o = self.op_of_place(lp.clone(), span, lhs, true);
                let whole = self.span(es);
                Some(self.hold(o, whole))
            };
            let rv = self.expr(rhs);
            let Some(r) = self.operand(rv, rhs) else { return };
            let rv = match old {
                Some(o) => {
                    let user = self.tb.target(icx.a(op)).cloned();
                    match user {
                        Some(Target::Op { .. }) => Rvalue::call(vec![o, r], Flows::None, icx.a(op)),
                        _ => Rvalue::Pure(vec![o, r]),
                    }
                }
                None => Rvalue::Use(r),
            };
            let span = self.span(lhs);
            self.push(StmtKind::Assign { dest: lp, rv, init: false }, span, lhs);
            return;
        }
        let Some(&e) = kids.first() else { return };
        let _ = self.expr(e);
    }

    fn if_stmt(&mut self, n: u32) {
        let icx = self.icx;
        let kids = icx.kids_vec(n);
        let (Some(&c), Some(&tb)) = (kids.first(), kids.get(1)) else { return };
        let els = kids.get(2).copied();
        let mark = self.temps.len();
        let cv = self.expr(c);
        let _ = self.read(cv, c);
        self.flush_temps(mark);
        if !self.open() {
            return;
        }
        let then = self.new_block();
        let join = self.new_block();
        let eb = if els.is_some() { self.new_block() } else { join };
        self.terminate(Terminator::Branch(vec![then, eb]));
        self.switch_to(then);
        self.block(tb);
        self.goto(join);
        if let Some(e) = els {
            self.switch_to(eb);
            if icx.kind(e) == Some(NodeKind::IfStmt) {
                self.if_stmt(e);
            } else {
                self.block(e);
            }
            self.goto(join);
        }
        self.switch_to(join);
    }

    fn while_stmt(&mut self, c: u32, b: u32) {
        let head = self.new_block();
        let body = self.new_block();
        let exit = self.new_block();
        self.goto(head);
        self.switch_to(head);
        let mark = self.temps.len();
        let cv = self.expr(c);
        let _ = self.read(cv, c);
        self.flush_temps(mark);
        self.terminate(Terminator::Branch(vec![body, exit]));
        self.switch_to(body);
        self.loops.push(LoopCx { head, exit, scopes: self.scopes.len(), temps: self.temps.len() });
        self.block(b);
        self.goto(head);
        self.loops.pop();
        self.switch_to(exit);
    }

    fn loop_stmt(&mut self, b: u32) {
        let body = self.new_block();
        let exit = self.new_block();
        self.goto(body);
        self.switch_to(body);
        self.loops.push(LoopCx { head: body, exit, scopes: self.scopes.len(), temps: self.temps.len() });
        self.block(b);
        self.goto(body);
        self.loops.pop();
        self.switch_to(exit);
    }

    fn break_continue(&mut self, s: u32, is_break: bool) {
        let Some(lp) = self.loops.last() else {
            self.terminate(Terminator::Unreachable);
            return;
        };
        let (target, sc, tm) = (if is_break { lp.exit } else { lp.head }, lp.scopes, lp.temps);
        let at = self.span(s);
        self.exit_path(sc, tm, true, at);
        self.goto(target);
    }

    fn return_stmt(&mut self, s: u32, e: Option<u32>) {
        let op = match e {
            Some(e) => {
                let v = self.expr(e);
                match self.operand(v, e) {
                    Some(o) => o,
                    None => return,
                }
            }
            None => self.const_op(s),
        };
        let at = self.span(s);
        self.leave(op, at);
    }

    // ---------- for ----------

    /// A `for` loop; the iterator it is given (its temporary) is dropped right after it, on
    /// every path out of it (as the lowering's loop slot).
    fn for_stmt(&mut self, f: u32) {
        self.push_scope();
        self.for_loop(f);
        let at = self.span(f);
        self.pop_scope(Some(at));
    }

    fn for_loop(&mut self, f: u32) {
        let icx = self.icx;
        let kids = icx.kids_vec(f);
        let Some(&pat) = kids.first() else { return };
        let Some(iter) = kids.iter().copied().find(|&c| icx.is_expr(c)) else { return };
        let Some(body) = kids.iter().copied().find(|&c| icx.kind(c) == Some(NodeKind::Block)) else { return };
        let tail = kids.iter().copied().find(|&c| icx.kind(c) == Some(NodeKind::TailClause));
        let bind_ty = kids.iter().copied().find(|&c| icx.kind(c) == Some(NodeKind::Type));
        let it_ty = self.nty(iter);
        let bt = self.tb.local(BindingRef { node: pat }).unwrap_or(TypeId::NONE);
        let chunk_ref = bind_ty.is_some() && matches!(self.tys.kind(bt), TypeKind::Ref { .. });
        let chunk_mut = bind_ty.is_some() && matches!(self.tys.kind(bt), TypeKind::Ref { mutable: true, .. });
        // a chunk loop: the binding is a vector or a reference to one
        let vt = match self.tys.kind(bt) {
            TypeKind::Ref { inner, .. } => inner,
            _ => bt,
        };
        let chunk = bind_ty.is_some() && matches!(self.tys.kind(vt), TypeKind::Vector { .. });
        enum Src {
            Range,
            Elems(Place, Option<DerefKind>),
            Iter(Place, Flows, TypeId),
        }
        let iv = self.expr(iter);
        let src = match self.tb.for_kind(f) {
            Some(ForKind::Range) => {
                let _ = self.read(iv, iter);
                Src::Range
            }
            Some(ForKind::Seq { mutable }) => {
                let Some(p) = self.place_of(iv, iter) else { return };
                let k = if mutable { DerefKind::Mut } else { DerefKind::Shared };
                Src::Elems(p, Some(k))
            }
            // an array value (a place is E0219): the loop's temporary
            Some(ForKind::Array) => {
                let Some(op) = self.operand(iv, iter) else { return };
                let V::P(p) = self.put_temp(Rvalue::Use(op), it_ty, iter) else { return };
                Src::Elems(p, None)
            }
            // an iterator value or a `&mut` to one
            Some(ForKind::Iter | ForKind::IterMut) => {
                let Some(op) = self.operand(iv, iter) else { return };
                let t = self.temp(it_ty, iter);
                self.body.locals[t as usize].mutable = true;
                let span = self.span(iter);
                self.push(
                    StmtKind::Assign { dest: Place::local(t, it_ty), rv: Rvalue::Use(op), init: true },
                    span,
                    iter,
                );
                if let Some(s) = self.scopes.last_mut() {
                    s.push(t);
                }
                let (flows, item) = self.next_of(f);
                match flows {
                    Some(fl) => Src::Iter(Place::local(t, it_ty), fl, item),
                    None => {
                        self.unsupported("`for` over a type without `next`");
                        return;
                    }
                }
            }
            // no source: the type checker reported it, and a body with an error is not built
            _ => {
                self.unsupported("`for` over a type that is not a `for` source");
                return;
            }
        };
        if !self.open() {
            return;
        }
        let head = self.new_block();
        let bb = self.new_block();
        let exit = self.new_block();
        self.goto(head);
        self.switch_to(head);
        // per-iteration item
        let item: Option<Place> = match &src {
            Src::Range => None,
            Src::Elems(p, dk) => {
                let base = match dk {
                    Some(k) => {
                        let inner = self.tys.deref(p.ty).map_or(TypeId::NONE, |x| x.0);
                        p.with(Proj::Deref(*k), inner)
                    }
                    None => p.clone(),
                };
                let et = self.tys.elem(base.ty);
                Some(base.with(Proj::Index, et))
            }
            Src::Iter(p, flows, item_ty) => {
                let span = self.span(iter);
                let (rv, _) = self.borrow(p.clone(), true, false, span, iter);
                let Some(rop) = self.operand(rv, iter) else { return };
                let opt_ty = self.option_of(*item_ty);
                let call = Rvalue::call(vec![rop], flows.clone(), icx.a(f));
                let res = self.put_temp(call, opt_ty, iter);
                let V::P(rp) = res else { return };
                let some = self.some_index();
                Some(rp.with(Proj::EnumField(some, 0), *item_ty))
            }
        };
        self.terminate(Terminator::Branch(vec![bb, exit]));
        self.switch_to(bb);
        self.loops.push(LoopCx { head, exit, scopes: self.scopes.len(), temps: self.temps.len() });
        self.push_scope();
        match (&src, item) {
            (Src::Range, _) => {
                let op = self.const_op(iter);
                let bt = self.tb.local(BindingRef { node: pat }).unwrap_or(TypeId::NONE);
                if let V::P(p) = self.put_temp(Rvalue::Use(op), bt, iter) {
                    self.destructure(pat, p, Mode::Move, iter);
                }
            }
            (Src::Elems(_, _), Some(ep)) if chunk && chunk_ref => {
                let span = self.span(pat);
                if let Some(v) = self.chunk_ref_bind(ep, chunk_mut, vt, span, iter) {
                    self.bind_pattern(pat, v, Mode::Move, iter);
                }
            }
            (Src::Elems(_, dk), Some(ep)) => {
                let by_ref = dk.is_some() && (bind_ty.is_none() || chunk_ref);
                if by_ref {
                    let m = matches!(dk, Some(DerefKind::Mut));
                    let span = self.span(iter);
                    let (rv, _) = self.borrow(ep, m, false, span, iter);
                    if let V::P(p) = rv {
                        self.destructure(pat, p, Mode::Move, iter);
                    }
                } else if bind_ty.is_some() {
                    // a chunk by value: read the elements, bind a fresh vector
                    let span = self.span(iter);
                    let r = self.op_of_place(ep, span, iter, false);
                    self.push(StmtKind::Read(r), span, iter);
                    let bt = self.tb.local(BindingRef { node: pat }).unwrap_or(TypeId::NONE);
                    let op = self.const_op(iter);
                    if let V::P(p) = self.put_temp(Rvalue::Use(op), bt, iter) {
                        self.destructure(pat, p, Mode::Move, iter);
                    }
                } else {
                    // by value over an owned array (the loop's temporary)
                    let saved = self.owned_elems.replace(ep.local);
                    self.destructure(pat, ep, Mode::Move, iter);
                    self.owned_elems = saved;
                }
            }
            (Src::Iter(..), Some(ip)) => {
                // the item `next` returned is a value of its own, taken out of its `Option`
                let it = self.temp(ip.ty, iter);
                let span = self.span(iter);
                let op = Operand { kind: OpKind::Copy, place: ip.clone(), span, node: iter, site: u32::MAX };
                self.push(
                    StmtKind::Assign { dest: Place::local(it, ip.ty), rv: Rvalue::Use(op), init: true },
                    span,
                    iter,
                );
                self.destructure(pat, Place::local(it, ip.ty), Mode::Move, iter);
            }
            _ => {}
        }
        self.block(body);
        let at = self.close_span(body);
        self.pop_scope(Some(at));
        self.goto(head);
        self.loops.pop();
        self.switch_to(exit);
        if let (Some(tc), Src::Elems(p, dk)) = (tail, &src) {
            self.tail_clause(tc, p.clone(), *dk, iter);
        }
    }

    /// `tail masked (v, m) { }` / `tail scalar (x) { }`: one pass over the remaining elements.
    fn tail_clause(&mut self, tc: u32, src: Place, dk: Option<DerefKind>, iter: u32) {
        let icx = self.icx;
        let kids = icx.kids_vec(tc);
        let Some(b) = kids.iter().copied().find(|&c| icx.kind(c) == Some(NodeKind::Block)) else { return };
        let binds: Vec<u32> =
            kids.iter().copied().filter(|&c| icx.kind(c) == Some(NodeKind::TailBind)).collect();
        self.push_scope();
        let base = match dk {
            Some(k) => {
                let inner = self.tys.deref(src.ty).map_or(TypeId::NONE, |x| x.0);
                src.with(Proj::Deref(k), inner)
            }
            None => src,
        };
        let et = self.tys.elem(base.ty);
        let ep = base.with(Proj::Index, et);
        for (i, &tb) in binds.iter().enumerate() {
            let l = self.binding_local(tb);
            let lt = self.body.locals[l as usize].ty;
            let span = self.span(tb);
            if i == 0
                && let TypeKind::Ref { mutable, inner } = self.tys.kind(lt)
            {
                let Some(v) = self.chunk_ref_bind(ep.clone(), mutable, inner, span, iter) else { continue };
                if let Some(op) = self.operand(V::P(v), iter) {
                    self.push(
                        StmtKind::Assign { dest: Place::local(l, lt), rv: Rvalue::Use(op), init: true },
                        span,
                        tb,
                    );
                }
            } else {
                if i == 0 {
                    let r = self.op_of_place(ep.clone(), span, iter, false);
                    self.push(StmtKind::Read(r), span, iter);
                }
                let op = self.const_op(tb);
                self.push(
                    StmtKind::Assign { dest: Place::local(l, lt), rv: Rvalue::Use(op), init: true },
                    span,
                    tb,
                );
            }
        }
        self.block(b);
        let at = self.close_span(b);
        self.pop_scope(Some(at));
    }

    /// A by-reference chunk binding over element place `ep` (`m`: `&mut`; `vt`: the vector
    /// type), in the binding's block (the innermost scope): the loop's code holds the chunk in a
    /// temporary of that block, loaded from the sequence and, under `&mut`, stored back wherever
    /// the block is left. So the binding borrows that temporary (it dies with the block: the
    /// binding cannot leave it), and the chunk of the sequence stays borrowed (`&mut` under
    /// `&mut`) until the store ([`Builder::end_local`]). Returns the binding's value.
    fn chunk_ref_bind(&mut self, ep: Place, m: bool, vt: TypeId, span: Span, n: u32) -> Option<Place> {
        let (rv, _) = self.borrow(ep, m, false, span, n);
        let V::P(src) = rv else { return None };
        let k = if m { DerefKind::Mut } else { DerefKind::Shared };
        let inner = self.tys.deref(src.ty).map_or(TypeId::NONE, |x| x.0);
        let chunk = src.with(Proj::Deref(k), inner);
        let c = self.new_local(vt, m, span, String::new(), false);
        if let Some(s) = self.scopes.last_mut() {
            s.push(c);
        }
        let op = self.op_of_place(chunk.clone(), span, n, false);
        self.push(StmtKind::Assign { dest: Place::local(c, vt), rv: Rvalue::Use(op), init: true }, span, n);
        self.chunk_temps.insert(c, ChunkWriteBack { chunk, m, node: n });
        let (bv, _) = self.borrow(Place::local(c, vt), m, false, span, n);
        let V::P(b) = bv else { return None };
        Some(b)
    }

    /// Where a reference chunk binding's block is left: a `&mut` chunk is stored back into the
    /// sequence, a `&` one is read there (its loan lasts the block).
    fn chunk_write_back(&mut self, w: ChunkWriteBack, at: Span) {
        if !self.open() {
            return;
        }
        if w.m {
            // the stored lanes hold no borrow: the store is a write through the chunk's loan; the
            // temporary's end (right after) is what conflicts with a binding kept past the block
            let op = self.const_op(w.node);
            self.push(StmtKind::Assign { dest: w.chunk, rv: Rvalue::Use(op), init: false }, at, w.node);
        } else {
            let r = self.op_of_place(w.chunk, at, w.node, false);
            self.push(StmtKind::Read(r), at, w.node);
        }
    }

    fn option_of(&mut self, t: TypeId) -> TypeId {
        let o = self.tys.sigs.lang.option;
        if o.is_none() {
            return TypeId::NONE;
        }
        self.tys.ov.intern(TypeKind::Adt { def: o, args: vec![GArg::Type(t)].into_boxed_slice() })
    }

    fn some_index(&self) -> u32 {
        let sigs = self.tys.sigs;
        sigs.adt(sigs.lang.option)
            .and_then(|a| a.variants.iter().position(|v| self.tys.prog.strings.get(v.name) == "Some"))
            .unwrap_or(1) as u32
    }

    /// The flows and item type of the `next` the type checker resolved for `for` loop `f` (never
    /// a lookup by name here: it would see the impls of this body's package only).
    fn next_of(&mut self, f: u32) -> (Option<Flows>, TypeId) {
        let prog = self.tys.prog;
        let Some(Target::Method { method: m, args, .. }) = self.tb.implicit_at(self.icx.a(f)).cloned() else {
            return (None, TypeId::NONE);
        };
        let w = World { prog, sigs: self.tys.sigs };
        let Some((sig, _, self_ty)) = w.method_sig(&m) else { return (None, TypeId::NONE) };
        let ret = self.tys.subst(sig.ret, &args, self_ty);
        let item = self.tys.payload(ret);
        let flows =
            if self.tys.ref_like(ret) { call_flows(self.tys.prog, &self.tys.ov, sig) } else { Flows::None };
        (Some(flows), item)
    }

    // ---------- match ----------

    fn match_(&mut self, n: u32, kids: &[u32], is_expr: bool) -> V {
        let icx = self.icx;
        let Some(&scrut) = kids.first() else { return V::Never };
        let arms = &kids[1..];
        let sv = self.expr(scrut);
        let Some(sp) = self.place_of(sv, scrut) else { return V::Never };
        let rty = if is_expr { self.nty(n) } else { TypeId::NONE };
        let result = if is_expr {
            let t = self.temp(rty, n);
            Some(t)
        } else {
            None
        };
        let join = self.new_block();
        let mut tests: Vec<BlockId> = Vec::new();
        for _ in arms {
            tests.push(self.new_block());
        }
        if tests.is_empty() {
            self.terminate(Terminator::Unreachable);
        } else {
            self.terminate(Terminator::Branch(tests.clone()));
        }
        let mut reached = false;
        for (k, &arm) in arms.iter().enumerate() {
            // [ArmPattern, guard?, body]
            let ak = icx.kids_vec(arm);
            let (Some(&ap), Some(&body)) = (ak.first(), ak.last()) else { continue };
            let alts = icx.kids_vec(ap);
            let guard = if ak.len() >= 3 { Some(ak[1]) } else { None };
            self.switch_to(tests[k]);
            self.push_scope();
            if let Some(g) = guard {
                // the arm's bindings are bound before the guard runs: the scrutinee stays
                // borrowed (shared) for the whole guard, and the bindings are read-only there
                let hold = self.guard_hold(&sp, scrut);
                let gmark = self.guard_binds.len();
                self.bind_alts(&alts, &sp, Mode::Peek, scrut);
                let mark = self.temps.len();
                self.in_guard += 1;
                self.guard_scruts.push(sp.clone());
                let gv = self.expr(g);
                let _ = self.read(gv, g);
                self.guard_scruts.pop();
                self.in_guard -= 1;
                self.flush_temps(mark);
                if let Some(h) = hold {
                    let span = self.span(g);
                    let r = self.op_of_place(h, span, u32::MAX, false);
                    self.push(StmtKind::Read(r), span, u32::MAX);
                }
                // the guard's view of the bindings ends with it; the arm binds them again. A
                // guard only reads (E0505), so nothing it made can hold a borrow of them
                for b in self.guard_binds.split_off(gmark) {
                    self.by_node.remove(&b);
                }
                let ok = self.new_block();
                let mut next: Vec<BlockId> = vec![ok];
                next.extend_from_slice(&tests[k + 1..]);
                self.terminate(Terminator::Branch(next));
                self.switch_to(ok);
            }
            self.bind_alts(&alts, &sp, Mode::Move, scrut);
            if is_expr {
                let mark = self.temps.len();
                let v = self.expr(body);
                if let (Some(op), Some(t)) = (self.operand(v, body), result) {
                    let span = self.span(body);
                    self.push(
                        StmtKind::Assign { dest: Place::local(t, rty), rv: Rvalue::Use(op), init: true },
                        span,
                        body,
                    );
                }
                self.flush_temps(mark);
            } else {
                self.block(body);
            }
            let at = self.span(arm);
            self.pop_scope_or_truncate(at);
            if self.open() {
                reached = true;
            }
            self.goto(join);
        }
        self.switch_to(join);
        if !reached {
            self.terminate(Terminator::Unreachable);
            return V::Never;
        }
        match result {
            Some(t) => {
                if self.tys.needs_drop(rty) {
                    self.temps.push(t);
                }
                V::P(Place::local(t, rty))
            }
            None => V::Const,
        }
    }

    fn pop_scope_or_truncate(&mut self, at: Span) {
        if self.open() {
            self.pop_scope(Some(at));
        } else {
            self.scopes.pop();
        }
    }

    /// Binds the alternatives `alts` of an arm's pattern against `src`. An or-pattern is a
    /// branch to one block per alternative: each binds the arm's locals, then they meet again.
    fn bind_alts(&mut self, alts: &[u32], src: &Place, mode: Mode, n: u32) {
        if let [p] = alts {
            self.destructure(*p, src.clone(), mode, n);
            return;
        }
        if !self.open() {
            return;
        }
        let blocks: Vec<BlockId> = alts.iter().map(|_| self.new_block()).collect();
        let meet = self.new_block();
        self.terminate(Terminator::Branch(blocks.clone()));
        for (&p, &b) in alts.iter().zip(&blocks) {
            self.switch_to(b);
            self.destructure(p, src.clone(), mode, n);
            self.goto(meet);
        }
        self.switch_to(meet);
    }

    /// Binds a whole pattern `p` against place `src`. A pattern that binds a non-Copy part of
    /// `src` by value takes the whole value apart: its parts are read into the bindings, then
    /// `src` moves as one value (the parts it does not bind end there), so `src` itself must be
    /// a value that may move whole.
    fn destructure(&mut self, p: u32, src: Place, mode: Mode, n: u32) {
        let saved = self.destr.replace((src.proj.len(), false));
        self.bind_pattern(p, src.clone(), mode, n);
        let took = self.destr.take().is_some_and(|(_, took)| took);
        self.destr = saved;
        if took && self.open() {
            let span = self.span(n);
            // a temporary scrutinee stays on the statement's drop list: another arm may leave it
            // whole (the drop of a moved value does nothing)
            let pos = self.temps.iter().rposition(|&t| t == src.local && src.proj.is_empty());
            let op = self.op_of_place(src, span, n, true);
            if let Some(i) = pos
                && !self.temps.contains(&op.place.local)
            {
                self.temps.insert(i, op.place.local);
            }
            self.push(StmtKind::Read(op), span, n);
        }
    }

    /// Binds pattern `p` against place `src` (by move, by reference after peeling references,
    /// or reads only); refutable sub-patterns read the tested place.
    fn bind_pattern(&mut self, p: u32, src: Place, mode: Mode, n: u32) {
        if !self.open() {
            return;
        }
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            self.unsupported("pattern nesting too deep");
            self.depth -= 1;
            return;
        }
        self.bind_pattern_in(p, src, mode, n);
        self.depth -= 1;
    }

    fn peel(&mut self, mut p: Place, mut mode: Mode) -> (Place, Mode) {
        for _ in 0..1000 {
            match self.tys.kind(p.ty) {
                TypeKind::Ref { mutable, inner } => {
                    let k = if mutable { DerefKind::Mut } else { DerefKind::Shared };
                    p = p.with(Proj::Deref(k), inner);
                    mode = match mode {
                        Mode::Move => Mode::Ref(mutable),
                        Mode::Ref(m) => Mode::Ref(m && mutable),
                        Mode::Peek => Mode::Peek,
                    };
                }
                _ => break,
            }
        }
        (p, mode)
    }

    fn test_read(&mut self, p: &Place, n: u32) {
        let span = self.span(n);
        let r = self.op_of_place(p.clone(), span, n, false);
        self.push(StmtKind::Read(r), span, n);
    }

    fn bind_pattern_in(&mut self, p: u32, src: Place, mode: Mode, n: u32) {
        let icx = self.icx;
        let a = icx.a(p);
        if icx.is(a, Term::Underscore) {
            return;
        }
        if icx.is(a, Term::KwMut) {
            self.bind_name(p, src, mode);
            return;
        }
        let kids = icx.kids_vec(p);
        if let Some(&l0) = kids.first()
            && icx.kind(l0) == Some(NodeKind::Literal)
        {
            let (dp, _) = self.peel(src, mode);
            self.test_read(&dp, p);
            return;
        }
        if icx.is(a, Term::LParen) {
            let close = icx.close(a);
            if kids.len() == 1 && !icx.is(close - 1, Term::Comma) {
                self.bind_pattern(kids[0], src, mode, n);
                return;
            }
            let (dp, pm) = self.peel(src, mode);
            for (i, &k) in kids.iter().enumerate() {
                let et = self.tys.field_ty(dp.ty, 0, i as u32);
                let sub = dp.with(Proj::Field(i as u32), et);
                self.bind_pattern(k, sub, pm, n);
            }
            return;
        }
        if !icx.is(a, Term::Ident) {
            return;
        }
        let e = icx.chain_end(a);
        let has_paren = icx.is(e + 1, Term::LParen);
        let has_rec = icx.is(e + 1, Term::DotLBrace);
        if e == a && !has_paren && !has_rec {
            self.bind_name(p, src, mode);
            return;
        }
        let Some(u) = icx.uses(p).last() else { return };
        let Res::Decl(d) = u.res else { return };
        let (dp, pm) = self.peel(src, mode);
        let dk = self.tys.prog.decl(d).map(|x| x.kind);
        match dk {
            Some(DeclKind::Variant | DeclKind::Struct) => {
                let is_variant = dk == Some(DeclKind::Variant);
                let vi = if is_variant { self.tys.prog.decl(d).map_or(0, |x| x.index) } else { 0 };
                if is_variant {
                    self.test_read(&dp, p);
                }
                let sub = |s: &mut Self, fi: u32| -> Place {
                    let ft = s.tys.field_ty(dp.ty, vi, fi);
                    if is_variant {
                        dp.with(Proj::EnumField(vi, fi), ft)
                    } else {
                        dp.with(Proj::Field(fi), ft)
                    }
                };
                if has_rec {
                    let TypeKind::Adt { def, .. } = self.tys.kind(dp.ty) else { return };
                    let names: Vec<String> = self
                        .tys
                        .sigs
                        .adt(def)
                        .and_then(|ad| ad.variants.get(vi as usize))
                        .map(|v| {
                            v.fields.iter().map(|f| self.tys.prog.strings.get(f.name).to_string()).collect()
                        })
                        .unwrap_or_default();
                    for fp in kids {
                        if icx.kind(fp) != Some(NodeKind::FieldPat) {
                            continue;
                        }
                        let name = icx.tok_str(icx.a(fp));
                        let Some(fi) = names.iter().position(|x| x == name) else { continue };
                        let sp = sub(self, fi as u32);
                        match icx.child(fp, NodeKind::Pattern) {
                            Some(sub_p) => self.bind_pattern(sub_p, sp, pm, n),
                            None => self.bind_name(fp, sp, pm),
                        }
                    }
                } else if has_paren {
                    for (fi, &k) in kids.iter().enumerate() {
                        let sp = sub(self, fi as u32);
                        self.bind_pattern(k, sp, pm, n);
                    }
                }
            }
            _ => self.test_read(&dp, p),
        }
    }

    /// A shared loan of the scrutinee `sp` for a guard, in a temporary (read at the guard's end).
    fn guard_hold(&mut self, sp: &Place, scrut: u32) -> Option<Place> {
        if !self.open() {
            return None;
        }
        let span = self.span(scrut);
        let ty = self.tys.ov.mk_ref(false, sp.ty);
        let loan = self.new_loan(sp, false, span, scrut, false);
        let t = self.new_local(ty, false, span, String::new(), true);
        let rv = Rvalue::Ref { place: sp.clone(), mutable: false, loan };
        self.push(StmtKind::Assign { dest: Place::local(t, ty), rv, init: true }, span, u32::MAX);
        Some(Place::local(t, ty))
    }

    /// A guard's view of binding `b`: a read-only local (not in any scope: it is never dropped,
    /// and its storage ends with the guard) holding the bound value, or a shared borrow for a binding by reference. Every alternative
    /// of an or-pattern binds the one guard local of the first alternative's binding.
    fn bind_guard_name(&mut self, b: u32, src: Place) {
        let icx = self.icx;
        let span = match icx.res.binding(BindingRef { node: b }) {
            Some(x) => icx.tok_span(x.token),
            None => icx.node_span(b),
        };
        let first = icx.res.canonical(b);
        let ty = self.tb.local(BindingRef { node: first }).unwrap_or(TypeId::NONE);
        let g = match self.by_node.get(&first) {
            Some(&g) if first != b && self.guard_binds.contains(&first) => g,
            _ => {
                let g = self.new_local(ty, false, span, String::new(), false);
                self.body.locals[g as usize].guard = true;
                self.body.locals[g as usize].node = first;
                self.by_node.insert(first, g);
                self.guard_binds.push(first);
                g
            }
        };
        let by_ref = ty != src.ty && matches!(self.tys.kind(ty), TypeKind::Ref { .. });
        let rv = if by_ref {
            let loan = self.new_loan(&src, false, span, b, false);
            Rvalue::Ref { place: src, mutable: false, loan }
        } else {
            Rvalue::Use(self.op_of_place(src, span, b, false))
        };
        self.push(StmtKind::Assign { dest: Place::local(g, ty), rv, init: true }, span, b);
    }

    fn bind_name(&mut self, b: u32, src: Place, mode: Mode) {
        if mode == Mode::Peek {
            self.bind_guard_name(b, src);
            return;
        }
        let l = self.binding_local(b);
        let lt = self.body.locals[l as usize].ty;
        let span = match self.icx.res.alias(b).and(self.icx.res.binding(BindingRef { node: b })) {
            Some(x) => self.icx.tok_span(x.token),
            None => self.body.locals[l as usize].span,
        };
        match mode {
            Mode::Peek => {}
            Mode::Move => {
                let top = self.destr.as_ref().map(|d| d.0);
                let op = match top {
                    // a part of the value a pattern takes apart: read here, the whole value
                    // moves once the pattern is bound
                    Some(top) if src.proj.len() > top => {
                        // the type checker's answer, which the lowering reads too
                        let by_value = self.tb.binds_by_move(b);
                        if by_value == self.tys.is_copy(src.ty) {
                            self.body
                                .internal
                                .get_or_insert("a binding whose value is Copy for one stage only");
                        }
                        let why = if by_value { self.move_out(&src, true) } else { None };
                        match why {
                            Some(why) => self.body.bad_moves.push((src.clone(), span, why)),
                            // a pattern names each part once, so this binding is its one owner
                            None if by_value => {
                                if let Some(d) = self.destr.as_mut() {
                                    d.1 = true;
                                }
                            }
                            None => {}
                        }
                        Operand { kind: OpKind::Copy, place: src, span, node: b, site: u32::MAX }
                    }
                    _ => self.op_of_place(src, span, b, true),
                };
                self.push(
                    StmtKind::Assign { dest: Place::local(l, lt), rv: Rvalue::Use(op), init: true },
                    span,
                    b,
                );
            }
            Mode::Ref(m) => {
                let loan = self.new_loan(&src, m, span, b, false);
                self.push(
                    StmtKind::Assign {
                        dest: Place::local(l, lt),
                        rv: Rvalue::Ref { place: src, mutable: m, loan },
                        init: true,
                    },
                    span,
                    b,
                );
            }
        }
    }

    // ---------- closures and spawn ----------

    /// Builds a nested body over node range of `n` (params `params`, body block `blk`); returns the
    /// capture list (outer local, mutated).
    fn nested(&mut self, n: u32, blk: u32, params: Option<u32>, is_closure: bool) -> Vec<(LocalId, bool)> {
        let lo = self.icx.ast.descendants(n).start;
        let icx = self.icx;
        let tb = self.tb;
        let mut nb = Builder::new(icx, tb, self.tys, Some((lo, n)));
        nb.body.ret = if is_closure { RetCheck::LocalsOnly } else { RetCheck::None };
        nb.depth = self.depth;
        nb.closure_span = Some(icx.tok_span(icx.a(n)));
        nb.infos = self.infos.clone();
        nb.guard_binds = self.guard_binds.clone();
        nb.in_guard = self.in_guard;
        nb.push_scope();
        if let Some(ps) = params {
            for (i, p) in icx.kids_vec(ps).into_iter().enumerate() {
                let ty = tb.local(BindingRef { node: p }).unwrap_or(TypeId::NONE);
                let l = nb.new_local(ty, false, icx.tok_span(icx.a(p)), String::new(), false);
                nb.body.locals[l as usize].param = Some(i as u32);
                nb.body.locals[l as usize].node = p;
                nb.by_node.insert(p, l);
                if let Some(s) = nb.scopes.last_mut() {
                    s.push(l);
                }
            }
        }
        nb.block(blk);
        nb.end_of_body(n);
        let caps = std::mem::take(&mut nb.captures);
        let mut inner = nb.finish();
        // the capture analysis (stave-sema) decides the list and the modes, which the thread rules
        // read too: a capture the CFG writes, mutably borrows, or activates while the analysis
        // holds it by `&`, or one the analysis does not list, means the two disagree, and the
        // body is not known to be checked
        let info = self.infos.iter().find(|c| c.node == n).cloned();
        let mut out = Vec::new();
        let mut seen: Vec<u32> = Vec::new();
        let mut disagree: Option<&'static str> = None;
        if let Some(info) = &info {
            for cap in &info.captures {
                let cl = caps.iter().find(|c| c.0 == cap.binding).map(|c| c.1);
                let mutated = cap.mode == stave_sema::closures::CaptureMode::RefMut;
                if let Some(cl) = cl {
                    if !mutated && mutates(&inner, cl) {
                        disagree.get_or_insert("a capture held by `&` is mutated");
                    }
                    inner.locals[cl as usize].capture_mut = mutated;
                }
                if let Some(ol) = self.local_of(Res::Local(BindingRef { node: cap.binding })) {
                    out.push((ol, mutated));
                    seen.push(cap.binding);
                }
            }
        }
        for (bnode, cl) in caps {
            if seen.contains(&bnode) {
                continue;
            }
            disagree.get_or_insert("a capture missing from the capture analysis");
            inner.locals[cl as usize].capture_mut = true;
            if let Some(ol) = self.local_of(Res::Local(BindingRef { node: bnode })) {
                out.push((ol, true));
            }
        }
        if let Some(why) = disagree.or(inner.internal) {
            self.body.internal.get_or_insert(why);
        }
        if inner.unsupported.is_some() {
            self.unsupported("nested body");
        }
        self.body.nested.push(inner);
        out
    }

    /// The capture loans of a closure or spawn at node `n`, aggregated into one temporary.
    fn capture_value(&mut self, caps: Vec<(LocalId, bool)>, ty: TypeId, n: u32) -> V {
        let ops = self.capture_ops(caps, n);
        self.put_temp(Rvalue::Agg(ops), ty, n)
    }

    /// One capture loan per captured local, created at node `n`.
    fn capture_ops(&mut self, caps: Vec<(LocalId, bool)>, n: u32) -> Vec<Operand> {
        let span = self.span(n);
        let mut ops = Vec::new();
        for (l, m) in caps {
            let lt = self.body.locals[l as usize].ty;
            // a capture's mutation is reported where the closure body makes it
            let effects = self.body.guard_effects.len();
            let (v, _) = self.borrow(Place::local(l, lt), m, false, span, n);
            if let Some(last) = self.body.loans.last_mut() {
                last.capture = true;
            }
            if let Some(op) = self.operand(v, n) {
                ops.push(op);
            }
            self.body.guard_effects.truncate(effects);
        }
        ops
    }

    fn closure(&mut self, n: u32) -> V {
        let icx = self.icx;
        let Some(blk) = icx.child(n, NodeKind::Block) else { return V::Const };
        let caps = self.nested(n, blk, icx.child(n, NodeKind::Params), true);
        let ty = self.nty(n);
        self.capture_value(caps, ty, n)
    }

    fn scope_stmt(&mut self, s: u32, b: u32) {
        self.push_scope();
        // the join local starts empty on every entry; each spawn adds its capture loans
        let sspan = self.span(s);
        let j = self.new_local(TypeId::NONE, false, sspan, String::new(), true);
        self.body.locals[j as usize].node = s;
        self.body.joins.push((j, sspan));
        let init =
            StmtKind::Assign { dest: Place::local(j, TypeId::NONE), rv: Rvalue::Agg(Vec::new()), init: true };
        self.push(init, sspan, u32::MAX);
        self.joins.push((self.scopes.len(), j));
        for st in self.icx.kids_vec(b) {
            if !self.open() {
                break;
            }
            self.stmt(st);
        }
        self.joins.pop();
        // the join point: every spawn's captures stay borrowed until here
        let span = self.close_span(b);
        self.join_read(j, span, s);
        self.pop_scope(Some(span));
    }

    fn spawn(&mut self, s: u32, b: u32) {
        let caps = self.nested(s, b, None, false);
        let Some(&(_, j)) = self.joins.last() else {
            let _ = self.capture_value(caps, TypeId::NONE, s);
            return;
        };
        // a field write unions into the join local's origins: the threads of earlier spawns
        // (earlier iterations of a loop too) keep their loans
        let ops = self.capture_ops(caps, s);
        let dest = Place { local: j, proj: vec![Proj::Field(0)], ty: TypeId::NONE };
        let span = self.span(s);
        self.push(StmtKind::Assign { dest, rv: Rvalue::Agg(ops), init: true }, span, s);
    }

    // ---------- expressions ----------

    fn coercion_of(&self, n: u32) -> Option<Coercion> {
        self.tb.coercion(n).cloned()
    }

    fn expr(&mut self, n: u32) -> V {
        if !self.open() {
            return V::Never;
        }
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            self.depth -= 1;
            self.unsupported("expression nesting too deep");
            self.terminate(Terminator::Unreachable);
            return V::Never;
        }
        let v = self.expr_node(n);
        let v = match self.coercion_of(n) {
            Some(co) => self.coerce(v, &co, n),
            None => v,
        };
        self.depth -= 1;
        if self.open() && self.tys.is_never(self.nty(n)) {
            self.terminate(Terminator::Unreachable);
            return V::Never;
        }
        v
    }

    fn coerce(&mut self, v: V, c: &Coercion, n: u32) -> V {
        match c {
            Coercion::MutToShared | Coercion::Reborrow => {
                let Some(p) = self.place_of(v, n) else { return V::Never };
                let Some((inner, k)) = self.tys.deref(p.ty) else { return V::P(p) };
                if k == DerefKind::Raw {
                    return V::P(p);
                }
                let span = self.span(n);
                let m = matches!(c, Coercion::Reborrow);
                // the value is a temporary just made by a borrow expression
                let whole = match self.cur.and_then(|b| self.body.blocks[b as usize].stmts.last()) {
                    Some(Stmt {
                        kind: StmtKind::Assign { dest, rv: Rvalue::Ref { loan, .. }, .. }, ..
                    }) if p.proj.is_empty()
                        && dest.proj.is_empty()
                        && dest.local == p.local
                        && self.body.locals[p.local as usize].temp =>
                    {
                        Some(*loan)
                    }
                    _ => None,
                };
                let before = self.body.loans.len();
                let v = self.borrow(p.with(Proj::Deref(k), inner), m, false, span, n).0;
                if let Some(l) = self.body.loans.get_mut(before) {
                    l.whole = whole;
                }
                v
            }
            Coercion::FmtToStr => {
                let Some(p) = self.place_of(v, n) else { return V::Never };
                let span = self.span(n);
                self.borrow(p, false, false, span, n).0
            }
            // a fn item is already a constant; a closure or fn value whose type changes keeps
            // what it borrows
            Coercion::FnItem
            | Coercion::NeverToAny
            | Coercion::ArrayToSlice
            | Coercion::Unalign
            | Coercion::StringToStr => v,
        }
    }

    fn expr_node(&mut self, n: u32) -> V {
        let Some(k) = self.icx.kind(n) else { return V::Const };
        match k {
            NodeKind::IfExpr => self.if_expr(n),
            NodeKind::MatchExpr => {
                let kids = self.icx.kids_vec(n);
                self.match_(n, &kids, true)
            }
            NodeKind::RangeExpr | NodeKind::CmpExpr => self.pure_kids(n),
            NodeKind::OrExpr | NodeKind::AndExpr => self.logic(n),
            NodeKind::BitOr
            | NodeKind::BitXor
            | NodeKind::BitAnd
            | NodeKind::Shift
            | NodeKind::Add
            | NodeKind::Mul => self.binary(n),
            NodeKind::Cast => {
                let kids = self.icx.kids_vec(n);
                let Some(&first) = kids.first() else { return V::Const };
                let v = self.expr(first);
                let ty = self.nty(n);
                // a `&mut T` place cast to `*mut` is `&mut *place` (a fresh `&mut` temporary is
                // judged where it was made): behind a `&` it is E0503, as the reborrow is
                if let V::P(p) = &v
                    && matches!(self.tys.kind(ty), TypeKind::Ptr { mutable: true, .. })
                    && let Some((inner, DerefKind::Mut)) = self.tys.deref(p.ty)
                    && !(p.proj.is_empty() && self.body.locals.get(p.local as usize).is_some_and(|l| l.temp))
                {
                    let span = self.span(n);
                    let (r, _) =
                        self.borrow(p.clone().with(Proj::Deref(DerefKind::Mut), inner), true, false, span, n);
                    if let Some(l) = self.body.loans.last_mut() {
                        l.cast = true;
                    }
                    let Some(op) = self.read(r, first) else { return V::Never };
                    return self.put_temp(Rvalue::Pure(vec![op]), ty, n);
                }
                // a cast whose result can hold borrows or is not Copy is a use of the operand:
                // the result borrows what the operand borrows, and the operand moves
                if self.tys.ref_like(ty) || !self.tys.is_copy(ty) {
                    let Some(op) = self.operand(v, first) else { return V::Never };
                    return self.put_temp(Rvalue::Use(op), ty, n);
                }
                let Some(op) = self.read(v, first) else { return V::Never };
                self.put_temp(Rvalue::Pure(vec![op]), ty, n)
            }
            NodeKind::Unary => self.unary(n),
            NodeKind::Postfix => self.postfix(n),
            // a bare name is a path with no steps
            NodeKind::Primary if self.icx.is_path_head(n) => self.postfix(n),
            NodeKind::Primary => self.primary(n),
            _ => V::Const,
        }
    }

    /// Operands read, a result without origins.
    fn pure_kids(&mut self, n: u32) -> V {
        let mut ops = Vec::new();
        for c in self.icx.kids_vec(n) {
            if self.icx.kind(c) == Some(NodeKind::CmpOp) {
                continue;
            }
            let v = self.expr(c);
            match self.read(v, c) {
                Some(o) => ops.push(o),
                None => return V::Never,
            }
        }
        let ty = self.nty(n);
        self.put_temp(Rvalue::Pure(ops), ty, n)
    }

    fn binary(&mut self, n: u32) -> V {
        let icx = self.icx;
        let kids = icx.kids_vec(n);
        let Some(&first) = kids.first() else { return V::Const };
        let v = self.expr(first);
        let Some(mut acc) = self.operand(v, first) else { return V::Never };
        let ty = self.nty(n);
        let whole = self.span(n);
        for &c in &kids[1..] {
            let optok = icx.a(c).saturating_sub(1);
            acc = self.hold(acc, whole);
            let v = self.expr(c);
            let Some(r) = self.operand(v, c) else { return V::Never };
            let rv = match self.tb.target(optok) {
                Some(Target::Op { .. }) => Rvalue::call(vec![acc, r], Flows::None, optok),
                _ => Rvalue::Pure(vec![acc, r]),
            };
            let V::P(p) = self.put_temp(rv, ty, n) else { return V::Never };
            let sp = self.span(n);
            acc = self.op_of_place(p, sp, n, true);
        }
        self.put_temp(Rvalue::Use(acc), ty, n)
    }

    fn logic(&mut self, n: u32) -> V {
        let kids = self.icx.kids_vec(n);
        let ty = self.nty(n);
        let join = self.new_block();
        for (i, &c) in kids.iter().enumerate() {
            let v = self.expr(c);
            let _ = self.read(v, c);
            if !self.open() {
                break;
            }
            if i + 1 < kids.len() {
                let next = self.new_block();
                self.terminate(Terminator::Branch(vec![next, join]));
                self.switch_to(next);
            } else {
                self.goto(join);
            }
        }
        self.switch_to(join);
        self.put_temp(Rvalue::Pure(Vec::new()), ty, n)
    }

    fn if_expr(&mut self, n: u32) -> V {
        let kids = self.icx.kids_vec(n);
        let (Some(&c), Some(&a), Some(&b)) = (kids.first(), kids.get(1), kids.get(2)) else {
            return V::Const;
        };
        let ty = self.nty(n);
        let cv = self.expr(c);
        let _ = self.read(cv, c);
        if !self.open() {
            return V::Never;
        }
        let t = self.temp(ty, n);
        let (then, els, join) = (self.new_block(), self.new_block(), self.new_block());
        self.terminate(Terminator::Branch(vec![then, els]));
        let mut reached = false;
        for (blk, e) in [(then, a), (els, b)] {
            self.switch_to(blk);
            let v = self.expr(e);
            if let Some(op) = self.operand(v, e) {
                let span = self.span(e);
                self.push(
                    StmtKind::Assign { dest: Place::local(t, ty), rv: Rvalue::Use(op), init: true },
                    span,
                    e,
                );
            }
            reached |= self.open();
            self.goto(join);
        }
        self.switch_to(join);
        if !reached {
            self.terminate(Terminator::Unreachable);
            return V::Never;
        }
        if self.tys.needs_drop(ty) {
            self.temps.push(t);
        }
        V::P(Place::local(t, ty))
    }

    fn unary(&mut self, n: u32) -> V {
        let icx = self.icx;
        let a = icx.a(n);
        let Some(c) = icx.last_child(n) else { return V::Const };
        if icx.is(a, Term::KwUnsafe) {
            return self.expr(c);
        }
        if icx.is(a, Term::Star) {
            let v = self.expr(c);
            let Some(p) = self.place_of(v, c) else { return V::Never };
            let Some((inner, k)) = self.tys.deref(p.ty) else { return V::P(p) };
            return V::P(p.with(Proj::Deref(k), inner));
        }
        if icx.is(a, Term::Amp) || icx.is(a, Term::AmpAmp) {
            let rt = self.nty(n);
            let mutable = match self.tys.kind(rt) {
                TypeKind::Ref { mutable, .. } => mutable,
                _ => icx.is(a + 1, Term::KwMut) && icx.is(a, Term::Amp),
            };
            let v = self.expr(c);
            let Some(p) = self.place_of(v, c) else { return V::Never };
            let span = self.span(n);
            return self.borrow(p, mutable, false, span, n).0;
        }
        // `-` `!` `~`: a value without origins (user operators are calls by value)
        let v = self.expr(c);
        let Some(op) = self.operand(v, c) else { return V::Never };
        let ty = self.nty(n);
        let rv = match self.tb.target(a) {
            Some(Target::Op { .. }) => Rvalue::call(vec![op], Flows::None, a),
            _ => Rvalue::Pure(vec![op]),
        };
        self.put_temp(rv, ty, n)
    }

    fn primary(&mut self, n: u32) -> V {
        let icx = self.icx;
        let a = icx.a(n);
        let t = icx.tk(a);
        let is = |x: Term| t == x as u8;
        if is(Term::IntLit)
            || is(Term::FloatLit)
            || is(Term::KwTrue)
            || is(Term::KwFalse)
            || is(Term::CharLit)
            || is(Term::StrLit)
        {
            return V::Const;
        }
        if is(Term::FmtLit) {
            return self.fstring(n, a);
        }
        if is(Term::KwSelf) {
            let r = icx.uses(n).first().map(|u| u.res);
            return match r.and_then(|r| self.local_of(r)) {
                Some(l) => V::P(Place::local(l, self.body.locals[l as usize].ty)),
                None => V::Const,
            };
        }
        if is(Term::LParen) {
            let close = icx.close(a);
            let kids = icx.kids_vec(n);
            if kids.is_empty() {
                return V::Const;
            }
            if kids.len() == 1 && !icx.is(close - 1, Term::Comma) {
                return self.expr(kids[0]);
            }
            return self.agg(n, &kids);
        }
        if is(Term::LBracket) {
            let kids = icx.kids_vec(n);
            if kids.len() == 2 && icx.is(icx.end_tok(kids[0]) + 1, Term::Semi) {
                return self.agg(n, &kids[..1]);
            }
            return self.agg(n, &kids);
        }
        if is(Term::KwFn) {
            return self.closure(n);
        }
        match icx.use_at(n, a) {
            Some(u) => {
                let r = u.res;
                self.path_val(r, u.token, n, icx.end_tok(n))
            }
            None => V::Const,
        }
    }

    fn agg(&mut self, n: u32, kids: &[u32]) -> V {
        let whole = self.span(n);
        let Some(ops) = self.operands_in_order(kids, whole) else { return V::Never };
        let ty = self.nty(n);
        self.put_temp(Rvalue::Agg(ops), ty, n)
    }

    /// The value of path `r` (last segment at token `tok`). An immutable `static` is a place
    /// that is never moved from: a temporary standing for its storage ([`Local::static_root`]).
    /// The value of the path resolved to `r` whose last segment is at token `tok`, in the chain
    /// of node `n`; `at`: the token where the type checker records the target of a path value
    /// that ends there.
    fn path_val(&mut self, r: Res, tok: u32, n: u32, at: u32) -> V {
        if let Some(l) = self.local_of(r) {
            return V::P(Place::local(l, self.body.locals[l as usize].ty));
        }
        let Res::Decl(d) = r else { return V::Const };
        let Some(decl) = self.tys.prog.decl(d) else { return V::Const };
        if decl.kind == DeclKind::Variant {
            // a variant without payload is a value of its enum: when the enum has drop glue it
            // is a temporary that ends with its statement, as a constructor's value is
            let ty = match self.tb.target(at) {
                Some(Target::Variant { ty, .. }) if !self.tys.is_unknown(*ty) => *ty,
                _ => self.untyped(),
            };
            if !self.tys.needs_drop(ty) {
                return V::Const;
            }
            let v = self.put_temp(Rvalue::Agg(Vec::new()), ty, n);
            if let V::P(tp) = &v {
                self.body.locals[tp.local as usize].span = self.icx.range_span(self.icx.a(n), tok);
            }
            return v;
        }
        if decl.kind != DeclKind::Static || decl.flags.has(stave_sema::DeclFlags::MUT) || !self.open() {
            return V::Const;
        }
        let Some(ty) = self.tys.sigs.konst(d).map(|c| c.ty) else { return V::Const };
        let span = self.icx.tok_span(tok);
        let t = self.new_local(ty, false, span, String::new(), true);
        self.body.locals[t as usize].static_root = true;
        let c = Operand {
            kind: OpKind::Const,
            place: Place::local(0, TypeId::NONE),
            span,
            node: u32::MAX,
            site: u32::MAX,
        };
        self.push(
            StmtKind::Assign { dest: Place::local(t, ty), rv: Rvalue::Use(c), init: true },
            span,
            u32::MAX,
        );
        V::P(Place::local(t, ty))
    }

    /// An f-string: reads of its holes (names and field paths), an owned `String` result.
    fn fstring(&mut self, n: u32, tok: u32) -> V {
        let icx = self.icx;
        let text = icx.tok_str(tok);
        let base = icx.tok_span(tok).offset;
        let bytes = text.as_bytes();
        let mut j = 2usize;
        let mut holes: Vec<(u32, Vec<String>)> = Vec::new();
        while j < bytes.len() {
            match bytes[j] {
                b'\\' => j += 2,
                b'{' if bytes.get(j + 1) == Some(&b'{') => j += 2,
                b'{' => {
                    let start = j + 1;
                    let mut k = start;
                    while k < bytes.len() && bytes[k] != b'}' {
                        k += 1;
                    }
                    let inner = &text[start..k.min(text.len())];
                    let segs: Vec<String> = inner.split('.').map(|s| s.trim().to_string()).collect();
                    holes.push((base + start as u32, segs));
                    j = k + 1;
                }
                _ => j += 1,
            }
        }
        for (off, segs) in holes {
            let Some(u) = icx.use_at_offset(n, off) else { continue };
            let r = u.res;
            let Some(l) = self.local_of(r) else { continue };
            let mut p = Place::local(l, self.body.locals[l as usize].ty);
            for s in segs.iter().skip(1) {
                for _ in 0..64 {
                    match self.tys.deref(p.ty) {
                        Some((inner, k)) if k != DerefKind::Raw => p = p.with(Proj::Deref(k), inner),
                        _ => break,
                    }
                }
                match self.tys.field_by_name(p.ty, s) {
                    Some((i, ft)) => p = p.with(Proj::Field(i), ft),
                    None => break,
                }
            }
            let len = segs.first().map_or(1, |s| s.len() as u32);
            let span = Span::new(icx.fid(), off, len);
            let r = Operand { kind: OpKind::Copy, place: p, span, node: n, site: u32::MAX };
            self.push(StmtKind::Read(r), span, n);
        }
        let ty = self.nty(n);
        self.put_temp(Rvalue::Pure(Vec::new()), ty, n)
    }

    // ---------- postfix chains ----------

    /// A postfix chain: a `Postfix`, or a bare `Primary` path (no steps).
    fn postfix(&mut self, n: u32) -> V {
        let icx = self.icx;
        let kids = icx.kids_vec(n);
        let Some(prim) = icx.chain_head(n) else { return V::Const };
        let pa = icx.a(prim);
        let pend = icx.postfix_end(n);
        let prim_end = icx.primary_end(prim);
        let mut kid = 1usize;
        let mut cur = if icx.is(pa, Term::Ident) {
            match icx.use_at(prim, pa) {
                Some(u) => Cur::Path(u.res, u.token),
                None => Cur::TypeRel,
            }
        } else if icx.is(pa, Term::KwSelfType) {
            Cur::Path(Res::SelfType, pa)
        } else {
            Cur::Val(self.expr(prim))
        };
        let mut p = prim_end + 1;
        for _ in 0..10_000_000 {
            if !self.open() {
                return V::Never;
            }
            let t = icx.tk(p);
            if t == Term::Dot as u8 && icx.is(p + 1, Term::Ident) {
                let seg = p + 1;
                let mut q = p + 2;
                if icx.is(q, Term::ColonColon) {
                    kid += 1;
                    let (e, partial) = icx.angle_end(q + 1);
                    q = if partial { e - 1 } else { e };
                }
                if !matches!(cur, Cur::Val(_))
                    && let Some(u) = icx.use_at(prim, seg)
                {
                    cur = Cur::Path(u.res, u.token);
                    p = q;
                    continue;
                }
                if !matches!(cur, Cur::Val(_))
                    && let Some(d) = icx.deferred_at(prim, seg)
                    && d.kind == stave_sema::DeferKind::TypeRelative
                {
                    cur = Cur::TypeRel;
                    p = q;
                    continue;
                }
                let recv = self.cur_value(cur, n);
                if matches!(recv, V::Never) {
                    return V::Never;
                }
                if icx.is(q, Term::LParen) {
                    let close = icx.close(q);
                    let args = if close == q + 1 {
                        stave_sema::itemcx::Kids::default()
                    } else {
                        let a = kids.get(kid).copied();
                        kid += 1;
                        a.map(|a| icx.kids_vec(a)).unwrap_or_default()
                    };
                    let last = close == pend;
                    cur = Cur::Val(self.method_call(n, recv, seg, close, &args, last));
                    p = close + 1;
                } else {
                    cur = Cur::Val(self.field_step(recv, seg, n));
                    p = q;
                }
                continue;
            }
            if t == Term::Dot as u8 && icx.is(p + 1, Term::IntLit) {
                let recv = self.cur_value(cur, n);
                cur = Cur::Val(self.field_step(recv, p + 1, n));
                p += 2;
                continue;
            }
            if t == Term::DotLBrace as u8 {
                let close = icx.close(p);
                let sl = kids.get(kid).copied();
                kid += 1;
                let last = close == pend;
                let fields = sl.map(|s| icx.kids_vec(s)).unwrap_or_default();
                let ty = self.literal_ty(p);
                cur = Cur::Val(self.agg_ty(n, &fields, ty));
                // the value is named by the literal's text, not by the rest of the chain
                if !last && let Cur::Val(V::P(tp)) = &cur {
                    self.body.locals[tp.local as usize].span = icx.range_span(icx.a(n), close);
                }
                p = close + 1;
                continue;
            }
            if t == Term::LParen as u8 {
                let close = icx.close(p);
                let args = if close == p + 1 {
                    stave_sema::itemcx::Kids::default()
                } else {
                    let a = kids.get(kid).copied();
                    kid += 1;
                    a.map(|a| icx.kids_vec(a)).unwrap_or_default()
                };
                let last = close == pend;
                cur = Cur::Val(self.call(n, cur, p, &args, last));
                p = close + 1;
                continue;
            }
            if t == Term::LBracket as u8 {
                let close = icx.close(p);
                let ie = kids.get(kid).copied();
                kid += 1;
                let base = self.cur_value(cur, n);
                cur = Cur::Val(self.index_step(base, ie, p, n));
                p = close + 1;
                continue;
            }
            if t == Term::Question as u8 {
                let base = self.cur_value(cur, n);
                let last = p == pend;
                cur = Cur::Val(self.try_step(n, base, p, last));
                p += 1;
                continue;
            }
            break;
        }
        match cur {
            Cur::Val(v) => v,
            Cur::Path(r, tok) => self.path_val(r, tok, n, pend),
            Cur::TypeRel => V::Const,
        }
    }

    /// The value of the chain of node `n` so far, used by its next step.
    fn cur_value(&mut self, cur: Cur, n: u32) -> V {
        match cur {
            Cur::Val(v) => v,
            Cur::Path(r, tok) => {
                let at = self.icx.chain_head(n).map_or(tok, |prim| self.icx.primary_end(prim));
                self.path_val(r, tok, n, at)
            }
            Cur::TypeRel => V::Const,
        }
    }

    /// The type of the struct literal or variant constructor whose target is at token `tok`:
    /// the one the type checker gave the value it makes (the chain's node has the type of the
    /// last step only). Nothing is inferred here: a second answer could differ from the first.
    fn literal_ty(&mut self, tok: u32) -> TypeId {
        match self.tb.target(tok) {
            Some(Target::Struct { ty, .. } | Target::Variant { ty, .. }) if !self.tys.is_unknown(*ty) => *ty,
            _ => self.untyped(),
        }
    }

    fn agg_ty(&mut self, n: u32, kids: &[u32], ty: TypeId) -> V {
        let whole = self.span(n);
        let Some(ops) = self.operands_in_order(kids, whole) else { return V::Never };
        self.put_temp(Rvalue::Agg(ops), ty, n)
    }

    fn apply_derefs(&mut self, mut p: Place, k: u32) -> Place {
        for _ in 0..k {
            match self.tys.deref(p.ty) {
                Some((inner, dk)) => p = p.with(Proj::Deref(dk), inner),
                None => {
                    p = p.with(Proj::Deref(DerefKind::Shared), TypeId::NONE);
                }
            }
        }
        p
    }

    fn field_step(&mut self, recv: V, tok: u32, n: u32) -> V {
        let (index, derefs) = match self.tb.target(tok) {
            Some(Target::Field { index, derefs, .. }) => (*index, *derefs),
            Some(Target::TupleField { index, derefs }) => (*index, *derefs),
            Some(Target::Column { index, derefs }) => {
                let (index, derefs) = (*index, *derefs);
                let Some(p) = self.place_of(recv, n) else { return V::Never };
                let p = self.apply_derefs(p, derefs);
                let row = match self.tys.kind(p.ty) {
                    TypeKind::Adt { args, .. } => match args.first() {
                        Some(GArg::Type(t)) => *t,
                        _ => TypeId::NONE,
                    },
                    _ => TypeId::NONE,
                };
                let et = self.tys.field_ty(row, 0, index);
                let st = self.tys.ov.intern(TypeKind::Slice { elem: et, align: 0 });
                return V::P(p.with(Proj::Column(index), st));
            }
            _ => return V::Const,
        };
        let Some(p) = self.place_of(recv, n) else { return V::Never };
        let p = self.apply_derefs(p, derefs);
        let ft = self.tys.field_ty(p.ty, 0, index);
        V::P(p.with(Proj::Field(index), ft))
    }

    fn index_step(&mut self, base: V, ie: Option<u32>, tok: u32, n: u32) -> V {
        let (derefs, slice) = match self.tb.target(tok) {
            Some(Target::Index { derefs }) => (*derefs, false),
            Some(Target::Slice { derefs }) => (*derefs, true),
            _ => (0, false),
        };
        let Some(p) = self.place_of(base, n) else { return V::Never };
        if let Some(ie) = ie {
            let iv = self.expr(ie);
            match self.read(iv, ie) {
                Some(op) if op.kind != OpKind::Const => {
                    let span = op.span;
                    self.push(StmtKind::Read(op), span, ie);
                }
                Some(_) => {}
                None => return V::Never,
            }
        }
        let p = self.apply_derefs(p, derefs);
        let et = self.tys.elem(p.ty);
        // a sub-slice is any element (no range reasoning), typed as the slice: never Copy
        let t = if slice { self.tys.ov.intern(TypeKind::Slice { elem: et, align: 0 }) } else { et };
        V::P(p.with(Proj::Index, t))
    }

    fn try_step(&mut self, n: u32, base: V, tok: u32, last: bool) -> V {
        let Some(bp) = self.place_of(base, n) else { return V::Never };
        let bt = bp.ty;
        let span = self.icx.tok_span(tok);
        if self.in_guard > 0 {
            self.body.guard_effects.push((span, GuardEffect::Try));
        }
        let op = self.op_of_place(bp, span, n, true);
        let V::P(hold) = self.put_temp(Rvalue::Use(op), bt, n) else { return V::Never };
        let (err, cont) = (self.new_block(), self.new_block());
        self.terminate(Terminator::Branch(vec![cont, err]));
        self.switch_to(err);
        let rop = self.op_of_place(hold.clone(), span, n, true);
        self.leave(rop, span);
        self.switch_to(cont);
        let pt = if last { self.nty(n) } else { self.tys.payload(bt) };
        let op = self.op_of_place(hold, span, n, true);
        self.put_temp(Rvalue::Use(op), pt, n)
    }

    fn args(&mut self, args: &[u32], whole: Span) -> Option<Vec<Operand>> {
        self.operands_in_order(args, whole)
    }

    fn sig_flows(&mut self, ret: TypeId, sig: &stave_sema::sigs::FnSig) -> Flows {
        if self.tys.ref_like(ret) { call_flows(self.tys.prog, &self.tys.ov, sig) } else { Flows::None }
    }

    fn flows_if(&mut self, ret: TypeId, f: impl FnOnce() -> Flows) -> Flows {
        if self.tys.ref_like(ret) { f() } else { Flows::None }
    }

    /// The stores of a direct call of `sig` with operands `ops` (the receiver first): the
    /// entries of its params' `from` clauses that can carry an origin at this instance.
    fn sig_stores(&mut self, sig: &stave_sema::sigs::FnSig, ops: &[Operand]) -> Vec<Store> {
        let mut out = Vec::new();
        for o in &sig.param_origins {
            let (target, source) = (u32::from(o.target), u32::from(o.source));
            let (Some(t), Some(s)) = (ops.get(target as usize), ops.get(source as usize)) else { continue };
            if t.kind == OpKind::Const || s.kind == OpKind::Const {
                continue;
            }
            let tt = t.place.ty;
            let into_view = match self.tys.deref(tt) {
                _ if self.tys.is_unknown(tt) => true,
                Some((inner, _)) => self.tys.ref_like(inner),
                None => false,
            };
            if self.tys.ref_like(s.place.ty) {
                out.push(Store { target, source, copy: o.copy, into_view });
            }
        }
        out
    }

    fn call(&mut self, n: u32, cur: Cur, paren: u32, args: &[u32], last: bool) -> V {
        let icx = self.icx;
        let tgt = self.tb.target(paren).cloned();
        let span = icx.range_span(icx.a(n), icx.close(paren));
        let node_ty = if last { self.nty(n) } else { TypeId::NONE };
        match tgt {
            Some(Target::Fn { decl, args: ga }) => {
                self.direct_call(decl, &ga, None, args, last, n, (span, paren))
            }
            Some(Target::Intrinsic { decl }) => {
                self.direct_call(decl, &[], None, args, last, n, (span, paren))
            }
            Some(Target::Variant { .. }) => {
                let Some(ops) = self.args(args, span) else { return V::Never };
                let ty = self.literal_ty(paren);
                let v = self.put_temp(Rvalue::Agg(ops), ty, n);
                if !last && let V::P(tp) = &v {
                    self.body.locals[tp.local as usize].span = span;
                }
                v
            }
            Some(Target::TraitMethod { method, self_ty, args: ga }) => {
                self.direct_call(method, &ga, Some(self_ty), args, last, n, (span, paren))
            }
            Some(Target::FnValue { .. }) => {
                let fv = self.cur_value(cur, n);
                let Some(fp) = self.place_of(fv, n) else { return V::Never };
                let ft = fp.ty;
                let fop = self.op_of_place(fp, span, n, false);
                let ret = if last {
                    node_ty
                } else {
                    match self.tys.kind(ft) {
                        TypeKind::Fn { ret, .. } => ret,
                        _ => TypeId::NONE,
                    }
                };
                let Some(mut ops) = self.args(args, span) else { return V::Never };
                ops.insert(0, fop);
                // The result takes the origins of every ref/view argument, and of the fn
                // value itself (operand 0): a closure may return a borrow of what it captured
                let flows = match self.tys.kind(ft) {
                    TypeKind::Fn { params, ret: fr, .. } if self.tys.ref_like(ret) => {
                        let t = &*self.tys;
                        let mut v = match fn_value_flows(t.prog, &t.ov, &params, fr, t.env, 1) {
                            Flows::Named(v) => v,
                            _ => Vec::new(),
                        };
                        v.insert(0, (0, false));
                        Flows::Named(v)
                    }
                    _ => Flows::None,
                };
                self.call_result(Rvalue::call(ops, flows, paren), ret, n, span)
            }
            Some(Target::SizeOf { .. }) => V::Const,
            Some(Target::Old) => match self.olds.get(&paren) {
                Some(&l) => V::P(Place::local(l, self.body.locals[l as usize].ty)),
                None => {
                    self.unsupported("`old` outside an `ensures` clause");
                    V::Never
                }
            },
            _ => {
                let Some(ops) = self.args(args, span) else { return V::Never };
                // a type-relative built-in (`f32x4.splat(x)`) has its target at its name token
                let ty = if last {
                    node_ty
                } else {
                    let mut name = paren.saturating_sub(1);
                    if icx.is(name, Term::Gt) || icx.is(name, Term::GtGt) {
                        while name > icx.a(n) && !icx.is(name, Term::ColonColon) {
                            name -= 1;
                        }
                        name = name.saturating_sub(1);
                    }
                    match self.tb.target(name) {
                        Some(Target::Builtin { ret, .. }) => *ret,
                        _ => TypeId::NONE,
                    }
                };
                let flows = self.flows_if(ty, || Flows::AllArgs);
                self.call_result(Rvalue::call(ops, flows, paren), ty, n, span)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn direct_call(
        &mut self,
        decl: stave_sema::DeclId,
        ga: &[GArg],
        self_ty: Option<TypeId>,
        args: &[u32],
        last: bool,
        n: u32,
        (span, tok): (Span, u32),
    ) -> V {
        let sigs = self.tys.sigs;
        let node_ty = if last { self.nty(n) } else { TypeId::NONE };
        let Some(ops) = self.args(args, span) else { return V::Never };
        let Some(sig) = sigs.fn_sig(decl) else {
            let flows = self.flows_if(node_ty, || Flows::AllArgs);
            return self.call_result(Rvalue::call(ops, flows, tok), node_ty, n, span);
        };
        let ret = if last { node_ty } else { self.tys.subst(sig.ret, ga, self_ty) };
        let flows = self.sig_flows(ret, sig);
        let stores = self.sig_stores(sig, &ops);
        self.call_result(Rvalue::Call { args: ops, flows, stores, tok }, ret, n, span)
    }

    fn call_result(&mut self, rv: Rvalue, ret: TypeId, n: u32, span: Span) -> V {
        if !self.open() {
            return V::Never;
        }
        let v = self.put_temp(rv, ret, n);
        if let Some(b) = self.cur
            && let Some(s) = self.body.blocks[b as usize].stmts.last_mut()
        {
            s.span = span;
        }
        // the result is named by the call's text, not by the rest of the chain it starts
        if let V::P(p) = &v {
            self.body.locals[p.local as usize].span = span;
        }
        if self.tys.is_never(ret) {
            self.terminate(Terminator::Unreachable);
            return V::Never;
        }
        v
    }

    /// The receiver operand after typeck's adjustment, and a two-phase activation.
    fn receiver(
        &mut self,
        recv: V,
        adj: Adjust,
        n: u32,
        rspan: Span,
    ) -> Option<(Operand, Option<Activation>)> {
        let p = self.place_of(recv, n)?;
        let p = self.apply_derefs(p, adj.derefs);
        match adj.autoref {
            Some(m) => {
                let (v, act) = self.borrow(p, m, m, rspan, n);
                let V::P(tp) = v else { return None };
                Some((self.op_of_place(tp, rspan, n, true), act))
            }
            None if adj.reborrow => match self.tys.deref(p.ty) {
                Some((inner, k)) if k == DerefKind::Mut => {
                    let (v, act) = self.borrow(p.with(Proj::Deref(k), inner), true, true, rspan, n);
                    let V::P(tp) = v else { return None };
                    Some((self.op_of_place(tp, rspan, n, true), act))
                }
                _ => Some((self.op_of_place(p, rspan, n, true), None)),
            },
            None => Some((self.op_of_place(p, rspan, n, true), None)),
        }
    }

    fn method_call(&mut self, n: u32, recv: V, seg: u32, close: u32, args: &[u32], last: bool) -> V {
        let icx = self.icx;
        let rspan = icx.range_span(icx.a(n), seg.saturating_sub(2));
        let span = icx.range_span(icx.a(n), close);
        let node_ty = if last { self.nty(n) } else { TypeId::NONE };
        let tgt = self.tb.target(seg).cloned();
        let (adj, sig, ret) = match &tgt {
            Some(Target::Method { method, args: ga, adjust }) => {
                let w = World { prog: self.tys.prog, sigs: self.tys.sigs };
                match w.method_sig(method) {
                    Some((sig, _, self_ty)) => {
                        let ret = if last { node_ty } else { self.tys.subst(sig.ret, ga, self_ty) };
                        (*adjust, Some(sig), ret)
                    }
                    None => (*adjust, None, node_ty),
                }
            }
            Some(Target::Builtin { adjust, ret, .. }) => (*adjust, None, if last { node_ty } else { *ret }),
            _ => (Adjust::default(), None, node_ty),
        };
        let Some((rop, act)) = self.receiver(recv, adj, n, rspan) else { return V::Never };
        let rop = if args.is_empty() { rop } else { self.hold(rop, span) };
        let Some(mut ops) = self.args(args, span) else { return V::Never };
        ops.insert(0, rop);
        if let Some((t, res, a)) = act {
            self.push(StmtKind::Activate { local: t, reserved: res, active: a }, rspan, n);
        }
        let (flows, stores) = match sig {
            Some(s) => (self.sig_flows(ret, s), self.sig_stores(s, &ops)),
            None => (self.flows_if(ret, || Flows::AllArgs), Vec::new()),
        };
        self.call_result(Rvalue::Call { args: ops, flows, stores, tok: seg }, ret, n, span)
    }
}

/// Whether a nested body writes, mutably borrows, or moves (a place rooted at) local `l`.
fn mutates(b: &Body, l: LocalId) -> bool {
    for blk in &b.blocks {
        for s in &blk.stmts {
            match &s.kind {
                StmtKind::Assign { dest, rv, init } => {
                    if dest.local == l && !*init {
                        return true;
                    }
                    let ops: &[Operand] = match rv {
                        Rvalue::Use(o) => std::slice::from_ref(o),
                        Rvalue::Agg(v) | Rvalue::Call { args: v, .. } | Rvalue::Pure(v) => v,
                        Rvalue::Ref { place, mutable, .. } => {
                            if place.local == l && *mutable {
                                return true;
                            }
                            &[]
                        }
                    };
                    if ops.iter().any(|o| o.kind == OpKind::Move && o.place.local == l) {
                        return true;
                    }
                }
                StmtKind::Activate { reserved, .. }
                    if b.loans.get(*reserved as usize).is_some_and(|x| x.place.local == l) =>
                {
                    return true;
                }
                _ => {}
            }
        }
    }
    false
}
