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
//! first, a call's arguments before the call. Temporaries with drop glue die at the end of their
//! statement (as lowered); others live to the end of the fn.
//!
//! Closures and `spawn` blocks are built as nested bodies ([`Body::nested`]) whose captured
//! locals are params; the closure value (or the `scope`'s join local, which accumulates every
//! spawn's loans and is read wherever the scope is left) holds one loan per captured local,
//! created at the closure's `fn` / `spawn` token. The capture list and modes are
//! stave-sema's capture analysis (`stave_sema::closures`: by reference only; `&mut` if the
//! body mutates the local or the local is a `&mut T`, else `&`); a capture the nested body writes,
//! mutably borrows, or reborrows is `&mut` too (a safety net, never weaker than the analysis).
//! The closure value's origin set is the union of its capture loans, so the loans live while the
//! value (or a result of a call through it, which takes the callee's origins too) lives. Captures
//! are references: moving a non-Copy capture out of the body is E0504.

use rustc_hash::FxHashMap;
use stave_diag::Span;
use stave_sema::itemcx::ItemCx;
use stave_sema::traits::{Adjust, Lookup, World};
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
    },
    /// Operators, casts, comparisons, built-ins with a non-ref result: no origins.
    Pure(Vec<Operand>),
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
    pub temp: bool,
    /// A binding as seen by a match guard: the guard runs after the arm's bindings are bound
    /// from the scrutinee, so inside it they are read-only views of the scrutinee (no move, no
    /// write, no `&mut`).
    pub guard: bool,
    /// The binding node (a temporary: its expression node; borrow dump `N` of drops).
    pub node: u32,
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

#[derive(Clone, Debug)]
pub struct Body {
    pub locals: Vec<Local>,
    pub blocks: Vec<Block>,
    pub loans: Vec<Loan>,
    /// Move sites: the moved place and the move's span.
    pub move_sites: Vec<(Place, Span)>,
    /// E0504: by-value uses of non-Copy places behind a dereference, an index, or a capture
    /// (place, span, the closure's span for a capture), kept as copies in the CFG.
    pub bad_moves: Vec<(Place, Span, Option<Span>)>,
    pub ret: RetCheck,
    /// Bodies of closures and spawn blocks inside this one.
    pub nested: Vec<Body>,
    /// Why this body cannot be checked (nothing is reported for it).
    pub unsupported: Option<&'static str>,
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
    Path(Res),
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
    /// The temporary holding the array a `for` loop iterates by value: its elements are moved
    /// into the loop binding one per iteration (not an E0504 move out of an index).
    owned_elems: Option<LocalId>,
}

fn passthrough_kind(k: NodeKind) -> bool {
    matches!(
        k,
        NodeKind::Expr
            | NodeKind::RangeExpr
            | NodeKind::OrExpr
            | NodeKind::AndExpr
            | NodeKind::CmpExpr
            | NodeKind::BitOr
            | NodeKind::BitXor
            | NodeKind::BitAnd
            | NodeKind::Shift
            | NodeKind::Add
            | NodeKind::Mul
            | NodeKind::Cast
    )
}

/// Builds the body of fn decl node `fn_node` (params in `params`, signature `from` list).
pub fn build_fn<'a>(
    icx: ItemCx<'a>,
    tb: &'a TypedBody,
    tys: &mut Tys<'a>,
    fn_node: u32,
    params: &[TypeId],
    ret: RetCheck,
) -> Body {
    let mut b = Builder::new(icx, tb, tys, None);
    let infos = {
        let ov = &b.tys.ov;
        stave_sema::closures::closures(&icx, tb, |t| stave_sema::closures::is_mut_ref(ov, t), fn_node)
    };
    b.infos = std::rc::Rc::new(infos);
    b.body.ret = ret;
    b.fn_body(fn_node, params);
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
            ret: RetCheck::None,
            nested: Vec::new(),
            unsupported: None,
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
            owned_elems: None,
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
            temp,
            guard: false,
            node: u32::MAX,
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
        let t = self.temp(ty, n);
        let span = self.span(n);
        self.push(StmtKind::Assign { dest: Place::local(t, ty), rv, init: true }, span, n);
        if self.tys.needs_drop(ty) {
            self.temps.push(t);
        }
        V::P(Place::local(t, ty))
    }

    fn new_loan(&mut self, place: &Place, mutable: bool, span: Span, node: u32, reserved: bool) -> LoanId {
        self.body.loans.push(Loan { place: place.clone(), mutable, span, node, reserved, capture: false });
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
        let ty = self.tys.ov.mk_ref(mutable, place.ty);
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

    fn place_of(&mut self, v: V, n: u32) -> Option<Place> {
        match v {
            V::P(p) => Some(p),
            V::Const => {
                let ty = self.nty(self.core(n));
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
        if by_value && self.owned_elems == Some(p.local) && !p.has_deref() {
            // a `for` binding takes a fresh element of the owned array on every iteration
            return Operand { kind: OpKind::Copy, place: p, span, node: n, site: u32::MAX };
        }
        let behind = p.proj.iter().any(|x| matches!(x, Proj::Deref(_) | Proj::Index));
        let (capture, guard) =
            self.body.locals.get(p.local as usize).map_or((None, false), |l| (l.capture, l.guard));
        if by_value && (behind || capture.is_some() || guard) {
            // E0504: a move out of a borrow, an index, a capture, or a guard binding; checked
            // as a copy
            self.body.bad_moves.push((p.clone(), span, capture.filter(|_| !behind)));
        }
        let movable = by_value && !behind && capture.is_none() && !guard;
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

    fn fn_body(&mut self, fn_node: u32, params: &[TypeId]) {
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
        if let Some(b) = icx.child(fn_node, NodeKind::Block) {
            self.block(b);
        }
        self.end_of_body(fn_node);
    }

    fn end_of_body(&mut self, n: u32) {
        if self.open() {
            let at = self.body_end(n);
            self.exit_path(0, 0, false, at);
            let op = self.const_op(n);
            let span = self.span(n);
            self.push(StmtKind::Return(op), span, n);
            self.terminate(Terminator::Return);
        }
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
        let Some(init) = kids.iter().copied().find(|&c| icx.kind(c) == Some(NodeKind::Expr)) else { return };
        let v = self.expr(init);
        if matches!(v, V::Never) || !self.open() {
            return;
        }
        if icx.is(icx.a(pat), Term::Underscore) {
            return; // `let _ = e;` moves nothing; a temporary is dropped at the end of the statement
        }
        let Some(p) = self.place_of(v, init) else { return };
        self.bind_pattern(pat, p, Mode::Move, init);
    }

    fn expr_stmt(&mut self, es: u32) {
        let icx = self.icx;
        let kids = icx.kids_vec(es);
        if kids.len() >= 3 {
            let (lhs, op, rhs) = (kids[0], kids[1], kids[2]);
            let compound = !icx.is(icx.a(op), Term::Eq);
            let lv = self.expr(lhs);
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
            let old = if compound { self.read(V::P(lp.clone()), lhs) } else { None };
            let rv = self.expr(rhs);
            let Some(r) = self.operand(rv, rhs) else { return };
            let rv = match old {
                Some(o) => {
                    let user = self.tb.target(icx.a(op)).cloned();
                    match user {
                        Some(Target::Op { .. }) => Rvalue::Call { args: vec![o, r], flows: Flows::None },
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
        let (span, node) = (op.span, op.node);
        self.push(StmtKind::Return(op), span, node);
        let at = self.span(s);
        self.exit_path(0, 0, false, at);
        self.terminate(Terminator::Return);
    }

    // ---------- for ----------

    fn for_stmt(&mut self, f: u32) {
        let icx = self.icx;
        let kids = icx.kids_vec(f);
        let Some(&pat) = kids.first() else { return };
        let Some(iter) = kids.iter().copied().find(|&c| icx.kind(c) == Some(NodeKind::Expr)) else { return };
        let Some(body) = kids.iter().copied().find(|&c| icx.kind(c) == Some(NodeKind::Block)) else { return };
        let tail = kids.iter().copied().find(|&c| icx.kind(c) == Some(NodeKind::TailClause));
        let bind_ty = kids.iter().copied().find(|&c| icx.kind(c) == Some(NodeKind::Type));
        let it_ty = self.nty(iter);
        let chunk_ref = bind_ty.is_some_and(|_| {
            let bt = self.tb.local(BindingRef { node: pat }).unwrap_or(TypeId::NONE);
            matches!(self.tys.kind(bt), TypeKind::Ref { .. })
        });
        enum Src {
            Range,
            Elems(Place, Option<DerefKind>),
            Iter(Place, Flows, TypeId),
        }
        let iv = self.expr(iter);
        let src = match self.tys.kind(it_ty) {
            TypeKind::Range { .. } => {
                let _ = self.read(iv, iter);
                Src::Range
            }
            TypeKind::Ref { mutable, inner }
                if matches!(self.tys.kind(inner), TypeKind::Array { .. } | TypeKind::Slice { .. }) =>
            {
                let Some(p) = self.place_of(iv, iter) else { return };
                let k = if mutable { DerefKind::Mut } else { DerefKind::Shared };
                Src::Elems(p, Some(k))
            }
            TypeKind::Array { .. } => {
                let Some(op) = self.operand(iv, iter) else { return };
                let V::P(p) = self.put_temp(Rvalue::Use(op), it_ty, iter) else { return };
                Src::Elems(p, None)
            }
            _ => {
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
                let (flows, item) = self.next_of(it_ty);
                match flows {
                    Some(fl) => Src::Iter(Place::local(t, it_ty), fl, item),
                    None => {
                        self.unsupported("`for` over a type without `next`");
                        return;
                    }
                }
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
                let res = self.put_temp(Rvalue::Call { args: vec![rop], flows: flows.clone() }, opt_ty, iter);
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
                    self.bind_pattern(pat, p, Mode::Move, iter);
                }
            }
            (Src::Elems(_, dk), Some(ep)) => {
                let by_ref = dk.is_some() && (bind_ty.is_none() || chunk_ref);
                if by_ref {
                    let m = matches!(dk, Some(DerefKind::Mut));
                    let span = self.span(iter);
                    let (rv, _) = self.borrow(ep, m, false, span, iter);
                    if let V::P(p) = rv {
                        self.bind_pattern(pat, p, Mode::Move, iter);
                    }
                } else if bind_ty.is_some() {
                    // a chunk by value: read the elements, bind a fresh vector
                    let span = self.span(iter);
                    let r = self.op_of_place(ep, span, iter, false);
                    self.push(StmtKind::Read(r), span, iter);
                    let bt = self.tb.local(BindingRef { node: pat }).unwrap_or(TypeId::NONE);
                    let op = self.const_op(iter);
                    if let V::P(p) = self.put_temp(Rvalue::Use(op), bt, iter) {
                        self.bind_pattern(pat, p, Mode::Move, iter);
                    }
                } else {
                    // by value over an owned array (the loop's temporary)
                    let saved = self.owned_elems.replace(ep.local);
                    self.bind_pattern(pat, ep, Mode::Move, iter);
                    self.owned_elems = saved;
                }
            }
            (Src::Iter(..), Some(ip)) => self.bind_pattern(pat, ip, Mode::Move, iter),
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
            if i == 0 && matches!(self.tys.kind(lt), TypeKind::Ref { mutable: true, .. }) {
                let (rv, _) = self.borrow(ep.clone(), true, false, span, iter);
                if let Some(op) = self.operand(rv, iter) {
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

    /// The flows and item type of `it.next()` for an iterator type.
    fn next_of(&mut self, it: TypeId) -> (Option<Flows>, TypeId) {
        let prog = self.tys.prog;
        let Some(next) = prog.strings.lookup("next") else { return (None, TypeId::NONE) };
        let w = World { prog, sigs: self.tys.sigs };
        let env = self.tys.env;
        let Lookup::Found { m, .. } = w.lookup_method(&mut self.tys.ov, it, next, env) else {
            return (None, TypeId::NONE);
        };
        let Some((sig, args, self_ty)) = w.method_sig(&m) else { return (None, TypeId::NONE) };
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
                let gv = self.expr(g);
                let _ = self.read(gv, g);
                self.flush_temps(mark);
                if let Some(h) = hold {
                    let span = self.span(g);
                    let r = self.op_of_place(h, span, u32::MAX, false);
                    self.push(StmtKind::Read(r), span, u32::MAX);
                }
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
            self.bind_pattern(*p, src.clone(), mode, n);
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
            self.bind_pattern(p, src.clone(), mode, n);
            self.goto(meet);
        }
        self.switch_to(meet);
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

    /// A guard's view of binding `b`: a read-only local (not in any scope: it is never dropped)
    /// holding the bound value, or a shared borrow for a binding by reference. Every alternative
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
                let op = self.op_of_place(src, span, b, true);
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
        let inner = nb.finish();
        // the capture analysis (stave-sema) decides the modes; a capture the CFG shows mutated is
        // `&mut` in any case (never weaker than the analysis)
        let info = self.infos.iter().find(|c| c.node == n).cloned();
        let mut out = Vec::new();
        let mut seen: Vec<u32> = Vec::new();
        if let Some(info) = &info {
            for cap in &info.captures {
                let cl = caps.iter().find(|c| c.0 == cap.binding).map(|c| c.1);
                let mutated = cap.mode == stave_sema::closures::CaptureMode::RefMut
                    || cl.is_some_and(|cl| mutates(&inner, cl));
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
            let mutated = mutates(&inner, cl);
            if let Some(ol) = self.local_of(Res::Local(BindingRef { node: bnode })) {
                out.push((ol, mutated));
            }
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
            let (v, _) = self.borrow(Place::local(l, lt), m, false, span, n);
            if let Some(last) = self.body.loans.last_mut() {
                last.capture = true;
            }
            if let Some(op) = self.operand(v, n) {
                ops.push(op);
            }
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

    fn passthrough(&self, n: u32) -> Option<u32> {
        let icx = self.icx;
        let k = icx.kind(n)?;
        if passthrough_kind(k) {
            return icx.only_child(n);
        }
        if k == NodeKind::Unary {
            let c = icx.first_child(n)?;
            if icx.kind(c) == Some(NodeKind::Postfix) && icx.a(c) == icx.a(n) {
                return Some(c);
            }
        }
        None
    }

    fn core(&self, n: u32) -> u32 {
        let mut cur = n;
        for _ in 0..1_000_000 {
            match self.passthrough(cur) {
                Some(c) => cur = c,
                None => break,
            }
        }
        cur
    }

    fn coercion_of(&self, n: u32) -> Option<Coercion> {
        let tb = self.tb;
        // a coercion of `n`'s chain is keyed at `n` or a descendant: none in that range, none at all
        let lo = self.icx.ast.descendants(n).start;
        let k = tb.coercions.partition_point(|c| c.0 < lo);
        if tb.coercions.get(k).is_none_or(|c| c.0 > n) {
            return None;
        }
        let mut cur = n;
        for _ in 0..1_000_000 {
            if let Some(c) = tb.coercion(cur) {
                return Some(c.clone());
            }
            match self.passthrough(cur) {
                Some(c) => cur = c,
                None => break,
            }
        }
        let icx = self.icx;
        if icx.kind(cur) == Some(NodeKind::Postfix)
            && let Some(p) = icx.first_child(cur)
            && icx.postfix_end(cur) == icx.primary_end(p)
            && let Some(c) = tb.coercion(p)
        {
            return Some(c.clone());
        }
        None
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
        let c = self.core(n);
        let v = self.expr_node(c);
        let v = match self.coercion_of(n) {
            Some(co) => self.coerce(v, &co, n),
            None => v,
        };
        self.depth -= 1;
        if self.open() && self.tys.is_never(self.nty(c)) {
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
                self.borrow(p.with(Proj::Deref(k), inner), m, false, span, n).0
            }
            Coercion::FmtToStr => {
                let Some(p) = self.place_of(v, n) else { return V::Never };
                let span = self.span(n);
                self.borrow(p, false, false, span, n).0
            }
            Coercion::FnItem => V::Const,
            Coercion::NeverToAny | Coercion::ArrayToSlice | Coercion::Unalign | Coercion::StringToStr => v,
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
                let Some(op) = self.read(v, first) else { return V::Never };
                let ty = self.nty(n);
                self.put_temp(Rvalue::Pure(vec![op]), ty, n)
            }
            NodeKind::Unary => self.unary(n),
            NodeKind::Postfix => self.postfix(n),
            NodeKind::Primary => self.primary(n),
            NodeKind::Expr => match self.icx.first_child(n) {
                Some(c) => self.expr(c),
                None => V::Const,
            },
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
        for &c in &kids[1..] {
            let optok = icx.a(c).saturating_sub(1);
            let v = self.expr(c);
            let Some(r) = self.operand(v, c) else { return V::Never };
            let rv = match self.tb.target(optok) {
                Some(Target::Op { .. }) => Rvalue::Call { args: vec![acc, r], flows: Flows::None },
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
            Some(Target::Op { .. }) => Rvalue::Call { args: vec![op], flows: Flows::None },
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
                self.path_val(r)
            }
            None => V::Const,
        }
    }

    fn agg(&mut self, n: u32, kids: &[u32]) -> V {
        let mut ops = Vec::with_capacity(kids.len());
        for &c in kids {
            let v = self.expr(c);
            match self.operand(v, c) {
                Some(o) => ops.push(o),
                None => return V::Never,
            }
        }
        let ty = self.nty(n);
        self.put_temp(Rvalue::Agg(ops), ty, n)
    }

    fn path_val(&mut self, r: Res) -> V {
        match self.local_of(r) {
            Some(l) => V::P(Place::local(l, self.body.locals[l as usize].ty)),
            None => V::Const,
        }
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

    fn postfix(&mut self, n: u32) -> V {
        let icx = self.icx;
        let kids = icx.kids_vec(n);
        let Some(&prim) = kids.first() else { return V::Const };
        let pa = icx.a(prim);
        let pend = icx.postfix_end(n);
        let prim_end = icx.primary_end(prim);
        let mut kid = 1usize;
        let mut cur = if icx.is(pa, Term::Ident) {
            match icx.use_at(prim, pa) {
                Some(u) => Cur::Path(u.res),
                None => Cur::TypeRel,
            }
        } else if icx.is(pa, Term::KwSelfType) {
            Cur::Path(Res::SelfType)
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
                    cur = Cur::Path(u.res);
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
                let recv = self.cur_value(cur);
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
                let recv = self.cur_value(cur);
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
                let ty = if last { self.nty(n) } else { TypeId::NONE };
                cur = Cur::Val(self.agg_ty(n, &fields, ty));
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
                let base = self.cur_value(cur);
                cur = Cur::Val(self.index_step(base, ie, p, n));
                p = close + 1;
                continue;
            }
            if t == Term::Question as u8 {
                let base = self.cur_value(cur);
                let last = p == pend;
                cur = Cur::Val(self.try_step(n, base, p, last));
                p += 1;
                continue;
            }
            break;
        }
        match cur {
            Cur::Val(v) => v,
            Cur::Path(r) => self.path_val(r),
            Cur::TypeRel => V::Const,
        }
    }

    fn cur_value(&mut self, cur: Cur) -> V {
        match cur {
            Cur::Val(v) => v,
            Cur::Path(r) => self.path_val(r),
            Cur::TypeRel => V::Const,
        }
    }

    fn agg_ty(&mut self, n: u32, kids: &[u32], ty: TypeId) -> V {
        let mut ops = Vec::with_capacity(kids.len());
        for &c in kids {
            let v = self.expr(c);
            match self.operand(v, c) {
                Some(o) => ops.push(o),
                None => return V::Never,
            }
        }
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
            _ => return V::Const,
        };
        let Some(p) = self.place_of(recv, n) else { return V::Never };
        let p = self.apply_derefs(p, derefs);
        let ft = self.tys.field_ty(p.ty, 0, index);
        V::P(p.with(Proj::Field(index), ft))
    }

    fn index_step(&mut self, base: V, ie: Option<u32>, tok: u32, n: u32) -> V {
        let derefs = match self.tb.target(tok) {
            Some(Target::Index { derefs }) => *derefs,
            _ => 0,
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
        V::P(p.with(Proj::Index, et))
    }

    fn try_step(&mut self, n: u32, base: V, tok: u32, last: bool) -> V {
        let Some(bp) = self.place_of(base, n) else { return V::Never };
        let bt = bp.ty;
        let span = self.icx.tok_span(tok);
        let op = self.op_of_place(bp, span, n, true);
        let V::P(hold) = self.put_temp(Rvalue::Use(op), bt, n) else { return V::Never };
        let (err, cont) = (self.new_block(), self.new_block());
        self.terminate(Terminator::Branch(vec![cont, err]));
        self.switch_to(err);
        let rop = self.op_of_place(hold.clone(), span, n, true);
        self.push(StmtKind::Return(rop), span, n);
        self.exit_path(0, 0, false, span);
        self.terminate(Terminator::Return);
        self.switch_to(cont);
        let pt = if last { self.nty(n) } else { self.tys.payload(bt) };
        let op = self.op_of_place(hold, span, n, true);
        self.put_temp(Rvalue::Use(op), pt, n)
    }

    fn args(&mut self, args: &[u32]) -> Option<Vec<Operand>> {
        let mut out = Vec::with_capacity(args.len());
        for &a in args {
            let v = self.expr(a);
            out.push(self.operand(v, a)?);
        }
        Some(out)
    }

    fn sig_flows(&mut self, ret: TypeId, sig: &stave_sema::sigs::FnSig) -> Flows {
        if self.tys.ref_like(ret) { call_flows(self.tys.prog, &self.tys.ov, sig) } else { Flows::None }
    }

    fn flows_if(&mut self, ret: TypeId, f: impl FnOnce() -> Flows) -> Flows {
        if self.tys.ref_like(ret) { f() } else { Flows::None }
    }

    fn call(&mut self, n: u32, cur: Cur, paren: u32, args: &[u32], last: bool) -> V {
        let icx = self.icx;
        let tgt = self.tb.target(paren).cloned();
        let span = icx.range_span(icx.a(n), icx.close(paren));
        let node_ty = if last { self.nty(n) } else { TypeId::NONE };
        match tgt {
            Some(Target::Fn { decl, args: ga }) => self.direct_call(decl, &ga, None, args, last, n, span),
            Some(Target::Intrinsic { decl }) => self.direct_call(decl, &[], None, args, last, n, span),
            Some(Target::Variant { .. }) => {
                let Some(ops) = self.args(args) else { return V::Never };
                self.put_temp(Rvalue::Agg(ops), node_ty, n)
            }
            Some(Target::TraitMethod { method, self_ty, args: ga }) => {
                self.direct_call(method, &ga, Some(self_ty), args, last, n, span)
            }
            Some(Target::FnValue) => {
                let fv = self.cur_value(cur);
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
                let Some(mut ops) = self.args(args) else { return V::Never };
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
                self.call_result(Rvalue::Call { args: ops, flows }, ret, n, span)
            }
            Some(Target::SizeOf { .. }) => V::Const,
            _ => {
                let Some(ops) = self.args(args) else { return V::Never };
                let flows = self.flows_if(node_ty, || Flows::AllArgs);
                self.call_result(Rvalue::Call { args: ops, flows }, node_ty, n, span)
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
        span: Span,
    ) -> V {
        let sigs = self.tys.sigs;
        let node_ty = if last { self.nty(n) } else { TypeId::NONE };
        let Some(ops) = self.args(args) else { return V::Never };
        let Some(sig) = sigs.fn_sig(decl) else {
            let flows = self.flows_if(node_ty, || Flows::AllArgs);
            return self.call_result(Rvalue::Call { args: ops, flows }, node_ty, n, span);
        };
        let ret = if last { node_ty } else { self.tys.subst(sig.ret, ga, self_ty) };
        let flows = self.sig_flows(ret, sig);
        self.call_result(Rvalue::Call { args: ops, flows }, ret, n, span)
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
            Some(Target::Builtin { adjust, .. }) => (*adjust, None, node_ty),
            _ => (Adjust::default(), None, node_ty),
        };
        let Some((rop, act)) = self.receiver(recv, adj, n, rspan) else { return V::Never };
        let Some(mut ops) = self.args(args) else { return V::Never };
        ops.insert(0, rop);
        if let Some((t, res, a)) = act {
            self.push(StmtKind::Activate { local: t, reserved: res, active: a }, rspan, n);
        }
        let flows = match sig {
            Some(s) => self.sig_flows(ret, s),
            None => self.flows_if(ret, || Flows::AllArgs),
        };
        self.call_result(Rvalue::Call { args: ops, flows }, ret, n, span)
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
