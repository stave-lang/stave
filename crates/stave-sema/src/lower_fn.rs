//! Per-function lowering state: locals, the SSA environment, joins and their
//! block params, places (locals, memory, rvalues) and their projections, drops, requests, sites.
//! Expressions are in [`crate::lower_expr`], statements and patterns in [`crate::lower_stmt`].
//!
//! **Locals.** Every binding node of the checked body is a local, numbered by source position
//! (fn params first). A local whose address is taken (`&x`, an auto-referenced method receiver
//! rooted at it) or whose type needs drop lives in a `slot` created at the front of `b0`, in local
//! order, before any temporary slot; every other local is an SSA value held in the environment.
//!
//! **Joins.** A join block gets one param per SSA local declared before its construct and
//! assigned inside it (a syntactic pre-pass over assignment statements), in local order, after
//! the construct's own leading params (an expression value, a `for` counter). Dead params are
//! removed afterwards ([`crate::lower_post`]), which leaves exactly the live ones.
//!
//! **Owners and drops.** An owner is a droppable local or a temporary with drop glue; temporaries
//! are numbered after the locals, as they are made. The environment holds, per owner, whether it
//! holds its value ([`Init`]): a by-value read of the owner moves it out, a binding or an
//! assignment of the whole brings it back. A value moves whole: a pattern that binds a part by
//! value takes the owner apart where the pattern is committed and drops the parts it ignores
//! there; a part moved by an expression does not reach the lowering. The state flows along
//! every edge the lowering builds: a join merges the states of its edges, and an owner that
//! holds its value on one edge only makes the job be lowered again with a dynamic drop flag for
//! it (a hidden `bool`: an SSA local for a local, a slot cleared at entry for a temporary). An
//! owner is dropped where it ends: a local at the end of its scope (reverse declaration order),
//! a temporary at the end of its statement (reverse creation order), both on every `break`,
//! `continue`, `return`, and `?` that leaves them; a place is dropped before it is overwritten.

use rustc_hash::{FxHashMap, FxHashSet};
use stave_ir::ir::ConstId;
use stave_ir::{Block, CastKind, FuncBuilder, FuncId, IntrinsicId, Op, Operand, Site, StrId, Value};
use stave_syntax::NodeKind;

use crate::Program;
use crate::decls::DeclKind;
use crate::ids::DeclId;
use crate::itemcx::{Brackets, ItemCx};

use crate::mono::{InstKey, LangFns, Req, Resolved};
use crate::resolve::Res;
use crate::sigs::Sigs;
use crate::traits::{MethodRef, World};
use crate::typeck::{Target, Typeck, TypedBody};
use crate::types::{ConstArg, GArg, Interner, Overlay, TypeId, TypeKind, TypeTable};

pub(crate) type Ty = TypeId;

/// Program-wide read-only inputs of one lowering round.
pub(crate) struct G<'a> {
    pub prog: &'a Program,
    pub tc: &'a Typeck,
    pub sigs: &'a Sigs,
    /// typeck's canonical table plus the types interned by earlier rounds.
    pub table: &'a TypeTable,
    pub brackets: &'a [Brackets],
    pub lang: &'a LangFns,
    /// `static` / `static mut` decl → index in `Module::statics`.
    pub statics: &'a FxHashMap<DeclId, u32>,
    /// Per package: built-in (std, runtime, arch).
    pub builtin_pkg: &'a [bool],
    /// Variant indices (`Option.None/Some`, `Result.Ok/Err`).
    pub opt_none: u32,
    pub opt_some: u32,
    pub res_ok: u32,
    pub res_err: u32,
    /// Some fn checks a `requires`: calls that can reach one store the user site first.
    pub any_requires: bool,
    /// Borrow expressions get an instruction of their own and a row in `Body::borrows`.
    pub borrows: bool,
    /// The features generic instances inherit from their type arguments, beyond their `needs`.
    pub inherited: &'a FxHashMap<InstKey, u64>,
}

impl<'a> G<'a> {
    pub fn world(&self) -> World<'a> {
        World { prog: self.prog, sigs: self.sigs }
    }

    /// Whether decl `d` belongs to a built-in package.
    pub fn builtin_decl(&self, d: DeclId) -> bool {
        self.prog
            .decl(d)
            .and_then(|x| self.prog.module(x.module))
            .is_some_and(|m| self.builtin_pkg.get(m.package.idx()).copied().unwrap_or(false))
    }

    /// Whether a call of decl `d` can run a `requires` check, which reports at the user site:
    /// `d` checks one, or `d` is a dispatch fn (its arm may).
    pub fn reaches_requires(&self, d: DeclId) -> bool {
        if !self.any_requires {
            return false;
        }
        if self.prog.decl(d).is_some_and(|x| x.flags.has(crate::decls::DeclFlags::DISPATCH)) {
            return true;
        }
        self.tc.body(d).and_then(|b| b.contracts.as_deref()).is_some_and(|c| !c.checked_requires().is_empty())
    }
}

/// What a [`LowerError`] means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LowerErrorKind {
    /// A valid construct stage0 does not lower yet (vector code, f16 literals, ...).
    Unsupported,
    /// An inconsistency between the checked program and lowering: a compiler bug.
    Internal,
}

/// A lowering error (unsupported construct or internal inconsistency).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LowerError {
    pub kind: LowerErrorKind,
    /// Checked-IR path of the fn being lowered.
    pub func: String,
    /// `file:line:col` of the construct, if known.
    pub at: String,
    pub msg: String,
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: in {}: {}", self.at, self.func, self.msg)
    }
}

impl std::error::Error for LowerError {}

/// The value of an expression.
#[derive(Clone, Debug)]
pub(crate) enum E {
    V(Operand, Ty),
    Unit,
    /// The current block is terminated (a diverging expression).
    Never,
    P(Place),
}

#[derive(Clone, Debug)]
pub(crate) enum Base {
    /// An SSA local.
    Local(usize),
    /// Memory at a pointer (thin, or fat for a slice / `str` place): the pointer and its type.
    Mem(Operand, Ty),
    /// An rvalue aggregate.
    Val(Operand, Ty),
}

#[derive(Clone, Debug)]
pub(crate) enum Proj {
    /// Field `i` (struct or tuple), result type.
    Field(u32, Ty),
    /// Element (array or slice), index operand, result type, site token of the `[`.
    Elem(Operand, Ty, u32),
    /// Field `f` of variant `v`, result type.
    EnumField(u32, u32, Ty),
}

#[derive(Clone, Debug)]
pub(crate) struct Place {
    pub base: Base,
    pub proj: Vec<Proj>,
    pub ty: Ty,
    /// The local the place is rooted at without a dereference (moves).
    pub root: Option<usize>,
}

/// A local of the job's body.
#[derive(Clone, Debug)]
pub(crate) struct LocalInfo {
    pub node: u32,
    pub ty: Ty,
    /// Lives in a slot.
    pub mem: bool,
    pub drop: bool,
    /// The dynamic drop flag of an owner.
    pub flag: Option<Flag>,
    /// A temporary.
    pub temp: Option<Temp>,
}

/// Whether an owner holds its value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Init {
    /// No: not bound yet, moved out, or dropped.
    #[default]
    Gone,
    Full,
}

/// A dynamic drop flag: set while its owner holds its value.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Flag {
    /// A hidden `bool` local.
    Ssa(usize),
    /// A `bool` slot, cleared at fn entry.
    Mem(Operand),
}

/// Where a temporary's value is.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Home {
    Slot(Operand),
    /// An SSA value, put into a slot only where a part of it is dropped.
    Val(Operand),
    /// The array of a `for` loop over an array value, in a slot: the loop takes its elements one
    /// by one, and the owner holds the ones from the index in the second slot (a `usize`) on.
    Rest(Operand, Operand),
}

/// What ends a temporary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Ends {
    /// Its statement: the index in [`FnState::temps`].
    Stmt(usize),
    /// The scope with this index in [`FnState::scopes`].
    Scope(usize),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Temp {
    /// Creation number in the job: the same in every lowering of it.
    pub seq: u32,
    pub home: Home,
    pub ends: Ends,
}

/// An owner that needs a dynamic drop flag: a local's binding node, or a temporary's creation
/// number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct FlagKey {
    pub temp: bool,
    pub id: u32,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Env {
    pub vals: Vec<Option<Operand>>,
    /// Per owner. Shorter than the owner list when temporaries were made since: those are `Gone`.
    pub st: Vec<Init>,
}

/// A join block under construction.
#[derive(Clone, Debug)]
pub(crate) struct Join {
    pub block: Block,
    pub prefix: Vec<Value>,
    pub locals: Vec<usize>,
    pub params: Vec<Value>,
    /// Merged owner states of the edges so far (`None`: no edge yet).
    pub st: Option<Vec<Init>>,
    /// Locals declared before the construct (their states merge).
    pub outer: usize,
    /// The statements and scopes open when the join was made: their temporaries outlive the
    /// construct (their states merge).
    pub stmts: usize,
    pub scopes: usize,
}

#[derive(Clone, Debug)]
pub(crate) enum LoopKind {
    /// `while` / `loop`: `continue` goes to the head with no prefix.
    Plain,
    /// `for` over a range or a sequence: `continue` steps the counter (`i + step`: 1, or the
    /// lane count of a chunk loop); inclusive ranges also carry the `more` flag (`i != end`).
    Count { i: Operand, ty: Ty, end: Option<Operand>, step: u32 },
}

#[derive(Clone, Debug)]
pub(crate) struct LoopCx {
    pub head: usize,
    pub exit: usize,
    pub scope: usize,
    pub kind: LoopKind,
}

/// One function being built (the instance, a closure, or drop glue).
pub(crate) struct FnState {
    pub b: FuncBuilder,
    pub slot: Vec<Option<Operand>>,
    pub env: Env,
    pub scopes: Vec<Vec<usize>>,
    pub loops: Vec<LoopCx>,
    pub joins: Vec<Join>,
    /// The temporaries of the open statements, in creation order.
    pub temps: Vec<usize>,
    /// Per open statement (or condition, guard, value branch or arm, clause): where its
    /// temporaries start in `temps`.
    pub stmts: Vec<usize>,
    /// The node of each open statement ([`crate::temps`]), innermost last.
    pub open: Vec<u32>,
    /// The node being lowered: a temporary made now is made by it.
    pub cur: u32,
    /// Per scope depth: the length of `stmts` when the scope at that depth was last pushed.
    pub scope_stmts: Vec<usize>,
    /// The owners from this number on are this fn's own temporaries; those below belong to the
    /// fns around a closure body (and are the locals).
    pub own_temps: usize,
    pub ret: Ty,
    pub site_consts: Vec<ConstId>,
    /// The values of the int constants made by [`FnCx::int`], by constant index.
    pub kvals: Vec<Option<i128>>,
    /// The user site stored last in the current block (no store between: std code stores none).
    pub last_site: Option<(Block, u32)>,
    /// Open `scope` statements: (scope depth of their block, the `&mut usize` thread-list
    /// handle); their spawns are joined when that scope ends or is left.
    pub scope_handles: Vec<(usize, Operand)>,
    /// The fn has `ensures` clauses (never set for a closure): each return checks them, and the
    /// end of a unit body (this token, the body's `}`) does too.
    pub ensures_end: Option<u32>,
    /// `&mut` chunk bindings: (scope depth of the chunk body, the write-back of the binding's
    /// slot); written back when that scope ends or is left.
    pub write_backs: Vec<(usize, WriteBack)>,
    /// The borrow side table of the body so far: borrows, and the other events.
    pub borrows: Vec<stave_ir::BorrowMark>,
    pub events: Vec<stave_ir::EventMark>,
    /// While the pattern of a `match` arm with a guard binds: the moves it makes (pointer to
    /// the place, its type). They happen when the guard holds, not when the names are bound.
    pub held: Option<Vec<(Operand, Ty)>>,
    /// With the borrow side table on: the slots of the rvalues whose address the open
    /// statements took (pointer, its type), in order, and per open statement where its slots
    /// start. Each ends with its statement ([`FnCx::end_temps`]).
    pub stmt_slots: Vec<(Operand, Ty)>,
    pub slot_marks: Vec<usize>,
    /// The site of an instruction that has none of its own: the statement's, with the borrow
    /// side table on (a report names source lines); else none.
    pub stmt_site: Site,
}

/// The write-back of a `&mut` chunk binding: the vector in `slot` goes to `s[i..i+N]`, only the
/// active lanes of `mask` in a masked tail.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WriteBack {
    pub slot: Operand,
    pub vt: Ty,
    pub s: Operand,
    pub i: Operand,
    pub mask: Option<Operand>,
    pub tok: u32,
}

/// The lowered function.
pub(crate) struct FnOut {
    /// Closure index (`None`: the job's own fn).
    pub closure: Option<u32>,
    pub params: Vec<Ty>,
    pub ret: Option<Ty>,
    pub body: Option<stave_ir::Body>,
    pub site_consts: Vec<ConstId>,
    /// A closure's `fn` token (its PC-table site).
    pub name_off: Option<(stave_diag::FileId, u32)>,
}

/// Everything a job produced (one instance with its closures, or one drop glue).
pub(crate) struct JobOut {
    pub funcs: Vec<FnOut>,
    pub reqs: Vec<Req>,
    pub strs: Vec<Vec<u8>>,
    pub intrs: Vec<String>,
    pub sites: Vec<(stave_diag::FileId, u32)>,
    pub local_types: Vec<TypeKind>,
    pub err: Option<LowerError>,
    /// The fn's name token.
    pub name: Option<(stave_diag::FileId, u32)>,
}

/// The lowering context of one job.
pub(crate) struct FnCx<'a> {
    pub g: &'a G<'a>,
    pub icx: ItemCx<'a>,
    /// Lowering a checked body (false: drop glue).
    pub has_body: bool,
    pub tb: Option<&'a TypedBody>,
    pub tys: Overlay<'a>,
    pub args: Vec<GArg>,
    pub self_ty: Option<Ty>,
    /// The caller is not in a built-in package (user sites, runtime/README.md).
    pub user: bool,
    pub path: String,
    pub locals: Vec<LocalInfo>,
    pub by_node: FxHashMap<u32, usize>,
    /// Assignment statements: (statement node, local), sorted.
    pub assigns: Vec<(u32, usize)>,
    /// Closure-body nodes (literals and `spawn` bodies) in `@<enclosing>.<k>` order.
    pub closure_nodes: Vec<u32>,
    /// Their captures (crate::closures), same order.
    pub closure_info: Vec<crate::closures::ClosureInfo>,
    pub reqs: Vec<Req>,
    req_map: FxHashMap<Req, u32>,
    pub strs: Vec<Vec<u8>>,
    str_map: FxHashMap<Vec<u8>, u32>,
    pub intrs: Vec<String>,
    pub sites: Vec<(stave_diag::FileId, u32)>,
    site_map: FxHashMap<u32, u32>,
    pub drop_memo: FxHashMap<Ty, bool>,
    subst_memo: FxHashMap<Ty, Ty>,
    /// The owners that get a dynamic drop flag.
    pub flagged: FxHashSet<FlagKey>,
    /// Discovered during this attempt: owners that need a flag.
    pub need_flag: Vec<FlagKey>,
    /// Temporaries made so far.
    pub temp_seq: u32,
    /// The `?` tokens of the body, sorted.
    pub try_toks: Vec<u32>,
    /// While a match guard is lowered: the owner states when it started, and whether an owner
    /// that held a value then was moved out of since.
    pub guard: Option<(Vec<Init>, bool)>,
    /// With the borrow side table on: binding nodes of the locals that live in a slot because a
    /// value is moved out of them (a move is an event on the bytes of its place).
    pub slotted: FxHashSet<u32>,
    /// Discovered during this attempt: locals that need such a slot.
    pub need_slot: Vec<u32>,
    copy_memo: FxHashMap<Ty, bool>,
    pub err: Option<LowerError>,
    pub f: FnState,
    pub outs: Vec<FnOut>,
    pub depth: u32,
    /// The token of the call being lowered (site of calls the lowering adds, e.g. `as_str`).
    pub cur_tok: u32,
    /// `n_parent` of the body's signature (generic numbering of cast types).
    pub cur_n_parent: u32,
    /// The unsubstituted impl self type of a method body.
    pub raw_self_ty: Option<Ty>,
    /// The needs of the fn being lowered (its closures have the same).
    pub needs: stave_target::FeatureSet,
    /// The `old(e)` snapshots taken at entry, by the call's `(` token: (value, type).
    pub old_vals: FxHashMap<u32, (Operand, Ty)>,
    /// The local `result` names in the `ensures` clauses.
    pub result_local: Option<usize>,
    /// Lowering a `requires` clause: no user-site stores, so a false clause reports the site
    /// the caller stored.
    pub hold_site: bool,
}

impl<'a> FnCx<'a> {
    pub fn new(g: &'a G<'a>, icx: ItemCx<'a>, path: String) -> FnCx<'a> {
        FnCx {
            g,
            icx,
            has_body: false,
            tb: None,
            tys: Overlay::new(g.table),
            args: Vec::new(),
            self_ty: None,
            user: false,
            path,
            locals: Vec::new(),
            by_node: FxHashMap::default(),
            assigns: Vec::new(),
            closure_nodes: Vec::new(),
            closure_info: Vec::new(),
            reqs: Vec::new(),
            req_map: FxHashMap::default(),
            strs: Vec::new(),
            str_map: FxHashMap::default(),
            intrs: Vec::new(),
            sites: Vec::new(),
            site_map: FxHashMap::default(),
            drop_memo: FxHashMap::default(),
            subst_memo: FxHashMap::default(),
            flagged: FxHashSet::default(),
            need_flag: Vec::new(),
            temp_seq: 0,
            try_toks: Vec::new(),
            guard: None,
            slotted: FxHashSet::default(),
            need_slot: Vec::new(),
            copy_memo: FxHashMap::default(),
            err: None,
            f: FnState::new(&[], TypeId::NONE),
            outs: Vec::new(),
            depth: 0,
            cur_tok: 0,
            cur_n_parent: 0,
            raw_self_ty: None,
            needs: stave_target::FeatureSet::EMPTY,
            old_vals: FxHashMap::default(),
            result_local: None,
            hold_site: false,
        }
    }

    pub fn icx(&self) -> ItemCx<'a> {
        self.icx
    }

    pub fn tb(&self) -> &'a TypedBody {
        match self.tb {
            Some(t) => t,
            None => empty_body(),
        }
    }

    // ---------- errors ----------

    /// An internal error at token `tok` (a compiler bug); the first error of the fn is kept.
    pub fn fail_at(&mut self, tok: Option<u32>, msg: impl Into<String>) -> E {
        self.error_at(LowerErrorKind::Internal, tok, msg.into())
    }

    pub fn fail(&mut self, n: u32, msg: impl Into<String>) -> E {
        let tok = Some(self.icx.a(n));
        self.fail_at(tok, msg)
    }

    /// A valid construct stage0 does not lower yet, at token `tok`.
    pub fn unsupported_at(&mut self, tok: Option<u32>, msg: impl Into<String>) -> E {
        self.error_at(LowerErrorKind::Unsupported, tok, msg.into())
    }

    /// Vector code, which stage0 does not generate yet; `what` names the construct.
    pub fn vector_unsupported(&mut self, tok: Option<u32>, what: &str) -> E {
        self.unsupported_at(tok, format!("vector code generation is not implemented yet: {what}"))
    }

    fn error_at(&mut self, kind: LowerErrorKind, tok: Option<u32>, msg: String) -> E {
        if self.err.is_none() {
            let at = match (tok, self.has_body) {
                (Some(t), true) => {
                    let sp = self.icx.tok_span(t);
                    match self.g.prog.source_map.locate(sp) {
                        Ok(l) => format!("{}:{}:{}", l.path, l.line, l.col),
                        Err(_) => String::from("?"),
                    }
                }
                _ => String::from("?"),
            };
            self.err = Some(LowerError { kind, func: self.path.clone(), at, msg });
        }
        if self.f.b.is_open() {
            self.f.b.unreachable();
        }
        E::Never
    }

    // ---------- types ----------

    pub fn sub(&mut self, t: Ty) -> Ty {
        if t.is_none() || (self.args.is_empty() && self.self_ty.is_none()) {
            return t;
        }
        if let Some(&r) = self.subst_memo.get(&t) {
            return r;
        }
        let args = std::mem::take(&mut self.args);
        let r = self.tys.subst(t, &args, self.self_ty);
        self.args = args;
        self.subst_memo.insert(t, r);
        r
    }

    /// The substituted type of node `n` (`NONE` if it has none).
    pub fn nty(&mut self, n: u32) -> Ty {
        let t = self.tb().ty(n);
        self.sub(t)
    }

    pub fn kind(&self, t: Ty) -> TypeKind {
        self.tys.kind(t).clone()
    }

    pub fn t_bool(&mut self) -> Ty {
        self.tys.intern(TypeKind::Bool)
    }

    pub fn t_unit(&mut self) -> Ty {
        self.tys.intern(TypeKind::Tuple(Box::new([])))
    }

    pub fn t_int(&mut self, signed: bool, bits: crate::types::IntBits) -> Ty {
        self.tys.intern(TypeKind::Int { signed, bits })
    }

    pub fn t_usize(&mut self) -> Ty {
        self.t_int(false, crate::types::IntBits::Size)
    }

    pub fn t_u32(&mut self) -> Ty {
        self.t_int(false, crate::types::IntBits::B32)
    }

    pub fn t_ref(&mut self, mutable: bool, t: Ty) -> Ty {
        self.tys.intern(TypeKind::Ref { mutable, inner: t })
    }

    pub fn t_ptr(&mut self, mutable: bool, t: Ty) -> Ty {
        self.tys.intern(TypeKind::Ptr { mutable, inner: t })
    }

    pub fn is_unit(&self, t: Ty) -> bool {
        matches!(self.tys.kind(t), TypeKind::Tuple(ts) if ts.is_empty())
    }

    pub fn is_never(&self, t: Ty) -> bool {
        matches!(self.tys.kind(t), TypeKind::Never)
    }

    /// IR return type: `None` for unit and `never`.
    pub fn ret_of(&self, t: Ty) -> Option<Ty> {
        if t.is_none() || self.is_unit(t) || self.is_never(t) { None } else { Some(t) }
    }

    pub fn pointee(&self, t: Ty) -> Option<(Ty, bool, bool)> {
        match self.tys.kind(t) {
            TypeKind::Ref { mutable, inner } => Some((*inner, *mutable, false)),
            TypeKind::Ptr { mutable, inner } => Some((*inner, *mutable, true)),
            _ => None,
        }
    }

    /// A pointer type like `like` (ref / raw, same mutability) to `inner`.
    pub fn ptr_like(&mut self, like: Ty, inner: Ty) -> Ty {
        match self.tys.kind(like).clone() {
            TypeKind::Ptr { mutable, .. } => self.t_ptr(mutable, inner),
            TypeKind::Ref { mutable, .. } => self.t_ref(mutable, inner),
            _ => self.t_ref(true, inner),
        }
    }

    pub fn needs_drop(&mut self, t: Ty) -> bool {
        if t.is_none() {
            return false;
        }
        let w = self.g.world();
        crate::mono::needs_drop(&w, &mut self.tys, t, &mut self.drop_memo)
    }

    /// Whether a by-value use of a place of type `t` copies it (else it moves the value out).
    pub fn is_copy(&mut self, t: Ty) -> bool {
        if t.is_none() {
            return true;
        }
        if let Some(&c) = self.copy_memo.get(&t) {
            return c;
        }
        let w = self.g.world();
        let c = w.is_copy(&mut self.tys, t, &[]);
        self.copy_memo.insert(t, c);
        c
    }

    /// With the borrow side table on: an event of `kind` on the place the thin pointer `p` of
    /// type `pt` names. Its row is on an instruction of its own, an identity cast of `p`.
    pub fn event(&mut self, kind: stave_ir::EventKind, p: Operand, pt: Ty) {
        if !self.g.borrows || !self.open() {
            return;
        }
        let thin = self
            .pointee(pt)
            .is_some_and(|(i, _, _)| !matches!(self.kind(i), TypeKind::Slice { .. } | TypeKind::Str));
        if !thin {
            return;
        }
        if let Operand::Value(value) = self.op1(Op::Cast(CastKind::Exact), pt, p) {
            self.f.events.push(stave_ir::EventMark { value, kind });
        }
    }

    /// [`FnCx::event`] on the slot of local `l`, if it has one.
    fn local_event(&mut self, kind: stave_ir::EventKind, l: usize) {
        if !self.g.borrows {
            return;
        }
        if let Some(p) = self.f.slot.get(l).copied().flatten() {
            let pt = self.t_ref(true, self.locals[l].ty);
            self.event(kind, p, pt);
        }
    }

    /// The locals of a scope end, last declared first (after their drops).
    pub fn end_locals(&mut self, ls: &[usize]) {
        if !self.g.borrows {
            return;
        }
        for &l in ls.iter().rev() {
            self.local_event(stave_ir::EventKind::End, l);
        }
    }

    pub fn adt_fields(&mut self, t: Ty) -> Option<Vec<Vec<Ty>>> {
        crate::mono::adt_fields(self.g.sigs, &mut self.tys, t)
    }

    /// (size, align) of a concrete type (equal to stave-ir's layout rules).
    pub fn layout(&mut self, t: Ty) -> Option<(u64, u64)> {
        crate::traits::layout_of(&mut self.tys, self.g.sigs, t, 0)
    }

    pub fn ir(t: Ty) -> stave_ir::TypeId {
        stave_ir::TypeId(t.0)
    }

    // ---------- ids ----------

    pub fn req(&mut self, r: Req) -> FuncId {
        if let Some(&i) = self.req_map.get(&r) {
            return FuncId(i);
        }
        let i = self.reqs.len() as u32;
        self.reqs.push(r.clone());
        // Only start code's `cpu_init` intrinsic calls `runtime.cpu.init`: a program gets it
        // exactly when it can read the feature word.
        if let Req::Inst(k) = &r
            && Some(k.decl) == self.g.lang.cpu_features
            && let Some(init) = self.g.lang.cpu_init
        {
            self.req(Req::Inst(InstKey { decl: init, args: Vec::new() }));
        }
        self.req_map.insert(r, i);
        FuncId(i)
    }

    pub fn string(&mut self, bytes: &[u8]) -> StrId {
        if let Some(&i) = self.str_map.get(bytes) {
            return StrId(i);
        }
        let i = self.strs.len() as u32;
        self.strs.push(bytes.to_vec());
        self.str_map.insert(bytes.to_vec(), i);
        StrId(i)
    }

    pub fn intrinsic(&mut self, name: &str) -> IntrinsicId {
        if let Some(i) = self.intrs.iter().position(|n| n == name) {
            return IntrinsicId(i as u32);
        }
        self.intrs.push(name.to_string());
        IntrinsicId(self.intrs.len() as u32 - 1)
    }

    /// The site of token `tok` of the current item.
    pub fn site(&mut self, tok: u32) -> Site {
        if !self.has_body {
            return Site::NONE;
        }
        let icx = self.icx;
        let off = icx.tok_start(tok);
        if let Some(&i) = self.site_map.get(&off) {
            return Site(i);
        }
        let i = self.sites.len() as u32;
        self.sites.push((icx.fid(), off));
        self.site_map.insert(off, i);
        Site(i)
    }

    // ---------- emission ----------

    pub fn int(&mut self, v: i128) -> Operand {
        let o = self.f.b.int(v);
        if let Operand::Const(c) = o {
            let i = c.index();
            if self.f.kvals.len() <= i {
                self.f.kvals.resize(i + 1, None);
            }
            self.f.kvals[i] = Some(v);
        }
        o
    }

    /// The int value of a constant operand made by [`FnCx::int`].
    pub fn kval(&self, o: Operand) -> Option<i128> {
        match o {
            Operand::Const(c) => self.f.kvals.get(c.index()).copied().flatten(),
            _ => None,
        }
    }

    pub fn op1(&mut self, op: Op, t: Ty, a: Operand) -> Operand {
        Operand::Value(self.f.b.op(op, Self::ir(t), &[a]))
    }

    pub fn op2(&mut self, op: Op, t: Ty, a: Operand, b: Operand) -> Operand {
        Operand::Value(self.f.b.op(op, Self::ir(t), &[a, b]))
    }

    pub fn opn(&mut self, op: Op, t: Ty, args: &[Operand]) -> Operand {
        Operand::Value(self.f.b.op(op, Self::ir(t), args))
    }

    /// Emits a trapping instruction at the site of `tok`.
    pub fn trapping(&mut self, tok: u32, op: Op, t: Option<Ty>, args: &[Operand]) -> Option<Operand> {
        let s = self.site(tok);
        self.f.b.set_site(s);
        let r = self.f.b.inst(op, t.map(Self::ir), args);
        self.f.b.set_site(self.f.stmt_site);
        r.map(Operand::Value)
    }

    pub fn unit_val(&mut self) -> Operand {
        let u = self.t_unit();
        self.opn(Op::Make, u, &[])
    }

    pub fn open(&self) -> bool {
        self.f.b.is_open()
    }

    pub fn new_slot(&mut self, t: Ty) -> Operand {
        let p = self.t_ref(true, t);
        Operand::Value(self.f.b.slot(Self::ir(p)))
    }

    /// Converts a pointer value to pointer type `want` (thin: `cast`; fat: rebuilt).
    pub fn convert_ptr(&mut self, v: Operand, from: Ty, want: Ty) -> Operand {
        if from == want {
            return v;
        }
        let Some((inner, _, _)) = self.pointee(want) else { return v };
        match self.kind(inner) {
            TypeKind::Slice { elem, .. } => {
                let us = self.t_usize();
                let ep = self.t_ref(false, elem);
                let p = self.op1(Op::SlicePtr, ep, v);
                let n = self.op1(Op::SliceLen, us, v);
                self.op2(Op::SliceMake, want, p, n)
            }
            TypeKind::Str => {
                let us = self.t_usize();
                let u8t = self.t_int(false, crate::types::IntBits::B8);
                let ep = self.t_ref(false, u8t);
                let p = self.op1(Op::SlicePtr, ep, v);
                let n = self.op1(Op::SliceLen, us, v);
                self.op2(Op::SliceMake, want, p, n)
            }
            _ => self.op1(Op::Cast(CastKind::Exact), want, v),
        }
    }

    // ---------- locals ----------

    pub fn local_of(&self, res: Res) -> Option<usize> {
        match res {
            Res::Local(b) | Res::SelfValue(b) => self.by_node.get(&b.node).copied(),
            _ => None,
        }
    }

    pub fn local_place(&mut self, l: usize) -> Place {
        let ty = self.locals[l].ty;
        if let Some(t) = self.locals[l].temp {
            return match t.home {
                Home::Slot(p) | Home::Rest(p, _) => {
                    let pt = self.t_ref(true, ty);
                    Place { base: Base::Mem(p, pt), proj: Vec::new(), ty, root: Some(l) }
                }
                Home::Val(v) => Place { base: Base::Val(v, ty), proj: Vec::new(), ty, root: Some(l) },
            };
        }
        match self.f.slot.get(l).copied().flatten() {
            Some(p) => {
                let pt = self.t_ref(true, ty);
                Place { base: Base::Mem(p, pt), proj: Vec::new(), ty, root: Some(l) }
            }
            None => Place { base: Base::Local(l), proj: Vec::new(), ty, root: Some(l) },
        }
    }

    /// Declares local `l` with its initial value (moves it into the slot, if any).
    pub fn bind(&mut self, l: usize, v: Option<Operand>) {
        let slot = self.f.slot.get(l).copied().flatten();
        if let Some(p) = slot {
            self.local_event(stave_ir::EventKind::Begin, l);
            if let Some(v) = v {
                self.f.b.store(p, v);
            }
        } else {
            let v = match v {
                Some(v) => v,
                None => self.unit_val(),
            };
            if let Some(x) = self.f.env.vals.get_mut(l) {
                *x = Some(v);
            }
        }
        if self.locals[l].drop {
            self.set_flag(l, true);
            self.set_st(l, Init::Full);
        }
        // a scope lists the locals it drops and, with the borrow side table on, the ones whose
        // slot ends with it
        if (self.locals[l].drop || (self.g.borrows && slot.is_some()))
            && let Some(s) = self.f.scopes.last_mut()
        {
            s.push(l);
        }
    }

    // ---------- owners ----------

    /// Whether owner `l` holds its value.
    pub fn st(&self, l: usize) -> Init {
        self.f.env.st.get(l).copied().unwrap_or(Init::Gone)
    }

    pub fn set_st(&mut self, l: usize, s: Init) {
        if self.f.env.st.len() <= l {
            self.f.env.st.resize(l + 1, Init::Gone);
        }
        self.f.env.st[l] = s;
    }

    /// Sets the flag of owner `l`, if it has one.
    fn set_flag(&mut self, l: usize, on: bool) {
        let Some(f) = self.locals[l].flag else { return };
        let v = self.int(i128::from(on));
        match f {
            Flag::Ssa(fl) => self.f.env.vals[fl] = Some(v),
            Flag::Mem(p) => {
                if self.open() {
                    self.f.b.store(p, v);
                }
            }
        }
    }

    /// The value of a flag (`None`: an SSA flag whose local is not bound here).
    fn flag_val(&mut self, f: Flag) -> Option<Operand> {
        match f {
            Flag::Ssa(fl) => self.f.env.vals.get(fl).copied().flatten(),
            Flag::Mem(p) => {
                let bt = self.t_bool();
                Some(Operand::Value(self.f.b.load(Self::ir(bt), p)))
            }
        }
    }

    /// Owner `l` is moved out.
    pub fn mark_moved(&mut self, l: usize) {
        if let Some((st, moved)) = &mut self.guard
            && st.get(l).is_some_and(|s| *s != Init::Gone)
        {
            *moved = true;
        }
        self.set_st(l, Init::Gone);
        self.set_flag(l, false);
    }

    /// Whether place `p` is its owner as a whole, or an element of it (which stands for its
    /// whole array): the places a value with drop glue is moved out of.
    pub fn whole(p: &Place) -> bool {
        !matches!(p.proj.first(), Some(Proj::Field(..) | Proj::EnumField(..)))
    }

    /// A by-value read of place `p` moves the owner it is rooted at (droppable values only). A
    /// value moves whole: the checker rejects a move of a part (E0504).
    pub fn note_move(&mut self, p: &Place) {
        let Some(l) = p.root else { return };
        if !self.locals[l].drop || !self.needs_drop(p.ty) {
            return;
        }
        if !Self::whole(p) {
            self.fail_at(None, "a part of a value is moved out");
            return;
        }
        self.mark_moved(l);
    }

    /// A temporary that owns a value of droppable type `ty`: it ends with the innermost open
    /// statement, or with the innermost scope when `scoped`.
    pub fn new_temp(&mut self, ty: Ty, home: Home, scoped: bool) -> usize {
        let seq = self.temp_seq;
        self.temp_seq += 1;
        let ends = if scoped {
            Ends::Scope(self.f.scopes.len().saturating_sub(1))
        } else {
            Ends::Stmt(self.f.temps.len())
        };
        let mut flag = None;
        if self.flagged.contains(&FlagKey { temp: true, id: seq }) {
            let bt = self.t_bool();
            let fs = self.new_slot(bt);
            let (zero, one) = (self.int(0), self.int(1));
            self.f.b.entry_store(fs, zero);
            self.f.b.store(fs, one);
            flag = Some(Flag::Mem(fs));
        }
        let l = self.locals.len();
        self.locals.push(LocalInfo {
            node: 0,
            ty,
            mem: true,
            drop: true,
            flag,
            temp: Some(Temp { seq, home, ends }),
        });
        self.set_st(l, Init::Full);
        match ends {
            Ends::Scope(d) => {
                if let Some(s) = self.f.scopes.get_mut(d) {
                    s.push(l);
                }
            }
            Ends::Stmt(_) => {
                self.made_at();
                self.f.temps.push(l);
            }
        }
        l
    }

    /// A temporary of the innermost open statement is made: that must be the statement the
    /// table gives the node being lowered, the one whose end ends it.
    fn made_at(&mut self) {
        let open = self.f.open.last().copied().unwrap_or(crate::temps::NONE);
        if self.tb().stmt_ends.of(self.f.cur) != open {
            self.fail_at(None, "a temporary outside the statement that ends it");
        }
    }

    /// A temporary for rvalue `v` in a slot of its own: the place.
    pub fn slot_temp(&mut self, v: Operand, t: Ty) -> Place {
        let (s, st, l) = self.slot_temp_at(v, t);
        Place { base: Base::Mem(s, st), proj: Vec::new(), ty: t, root: Some(l) }
    }

    /// [`FnCx::slot_temp`] as the slot, its pointer type, and the owner.
    pub fn slot_temp_at(&mut self, v: Operand, t: Ty) -> (Operand, Ty, usize) {
        let (s, st) = self.spill(v, t);
        let l = self.new_temp(t, Home::Slot(s), false);
        (s, st, l)
    }

    /// A temporary for rvalue `v`, read as the value it is: the place. With a dynamic flag the
    /// value lives in a slot, since its drop is then not dominated by `v`.
    pub fn value_temp(&mut self, v: Operand, t: Ty, scoped: bool) -> Place {
        if self.flagged.contains(&FlagKey { temp: true, id: self.temp_seq }) {
            let (s, st) = self.spill(v, t);
            let l = self.new_temp(t, Home::Slot(s), scoped);
            return Place { base: Base::Mem(s, st), proj: Vec::new(), ty: t, root: Some(l) };
        }
        let l = self.new_temp(t, Home::Val(v), scoped);
        Place { base: Base::Val(v, t), proj: Vec::new(), ty: t, root: Some(l) }
    }

    /// Forgets the temporaries at the end of the owner list that have ended: their numbers are
    /// used again, so the list stays as long as the owners alive at once.
    fn trim_temps(&mut self) {
        while self.locals.len() > self.f.own_temps
            && let Some(li) = self.locals.last()
        {
            let Some(t) = li.temp else { break };
            let l = self.locals.len() - 1;
            let alive = match t.ends {
                Ends::Stmt(i) => self.f.temps.get(i) == Some(&l),
                Ends::Scope(d) => self.f.scopes.get(d).is_some_and(|s| s.contains(&l)),
            };
            if alive {
                break;
            }
            self.locals.pop();
            if let Some(s) = self.f.env.st.get_mut(l) {
                *s = Init::Gone;
            }
        }
    }

    /// Whether a `?` is written in tokens `a..=b`.
    pub fn try_between(&self, a: u32, b: u32) -> bool {
        let i = self.try_toks.partition_point(|&t| t < a);
        self.try_toks.get(i).is_some_and(|&t| t <= b)
    }

    /// Whether one of the operands `rest` may leave the fn by `?`.
    pub fn try_in(&self, rest: &[u32]) -> bool {
        match (rest.first(), rest.last()) {
            (Some(&a), Some(&b)) if !self.try_toks.is_empty() => {
                self.try_between(self.icx.a(a), self.icx.end_tok(b))
            }
            _ => false,
        }
    }

    /// Operand value `v`, already evaluated, becomes a temporary until [`FnCx::release`]: a `?`
    /// in a later operand drops it.
    pub fn hold(&mut self, v: Operand, t: Ty, held: &mut Vec<usize>) {
        if !t.is_none() && self.needs_drop(t) {
            held.push(self.slot_temp_at(v, t).2);
        }
    }

    /// The held operands were consumed.
    pub fn release(&mut self, held: &[usize]) {
        for &l in held {
            self.mark_moved(l);
        }
    }

    // ---------- joins ----------

    /// SSA locals declared before construct `n` and assigned inside it (local order).
    pub fn assigned_in(&self, n: u32) -> Vec<usize> {
        if !self.has_body {
            return Vec::new();
        }
        let lo = self.icx.ast.descendants(n).start;
        let a = self.assigns.partition_point(|x| x.0 < lo);
        let b = self.assigns.partition_point(|x| x.0 < n);
        let mut out: Vec<usize> = Vec::new();
        for &(_, l) in &self.assigns[a..b] {
            let li = &self.locals[l];
            if li.node < lo && !li.mem && self.f.env.vals.get(l).is_some_and(|v| v.is_some()) {
                out.push(l);
            }
        }
        // drop flags of droppable locals in scope
        out.extend(self.flag_locals(lo));
        out.sort_unstable();
        out.dedup();
        out.retain(|&l| l < self.locals.len());
        out
    }

    /// The SSA drop flags, bound here, of the locals declared before node `lo`.
    pub fn flag_locals(&self, lo: u32) -> Vec<usize> {
        let mut out = Vec::new();
        for li in &self.locals {
            if li.node >= lo {
                continue;
            }
            if let Some(Flag::Ssa(fl)) = li.flag
                && self.f.env.vals.get(fl).is_some_and(|v| v.is_some())
            {
                out.push(fl);
            }
        }
        out
    }

    /// Creates a join block for construct `n` with leading params of types `prefix`.
    pub fn join(&mut self, n: u32, prefix: &[Ty]) -> usize {
        let locals = self.assigned_in(n);
        let block = self.f.b.create_block();
        let prefix: Vec<Value> = prefix.iter().map(|&t| self.f.b.add_param(block, Self::ir(t))).collect();
        let mut params = Vec::with_capacity(locals.len());
        for &l in &locals {
            let t = self.locals[l].ty;
            params.push(self.f.b.add_param(block, Self::ir(t)));
        }
        let outer = if self.has_body { self.icx.ast.descendants(n).start as usize } else { 0 };
        let (stmts, scopes) = (self.f.stmts.len(), self.f.scopes.len());
        self.f.joins.push(Join { block, prefix, locals, params, st: None, outer, stmts, scopes });
        self.f.joins.len() - 1
    }

    /// The branch args to join `j` from the current env (records the edge's moved states).
    pub fn edge(&mut self, j: usize, prefix: &[Operand]) -> Vec<Operand> {
        let mut args: Vec<Operand> = prefix.to_vec();
        let locals = self.f.joins[j].locals.clone();
        for &l in &locals {
            let v = match self.f.env.vals.get(l).copied().flatten() {
                Some(v) => v,
                None => {
                    let t = self.locals[l].ty;
                    self.dummy(t)
                }
            };
            args.push(v);
        }
        let st = self.edge_state(j);
        match self.f.joins[j].st.take() {
            None => self.f.joins[j].st = Some(st),
            Some(mut merged) => {
                self.merge_states(&mut merged, &st);
                self.f.joins[j].st = Some(merged);
            }
        }
        args
    }

    /// Merges the owner states `st` of one more edge into `merged`. An owner that holds its
    /// value on one edge only is told by its flag, asked for here if it has none.
    pub fn merge_states(&mut self, merged: &mut Vec<Init>, st: &[Init]) {
        if merged.len() < st.len() {
            merged.resize(st.len(), Init::Gone);
        }
        for (l, b) in st.iter().enumerate() {
            if merged[l] == *b || l >= self.locals.len() {
                continue;
            }
            let li = &self.locals[l];
            if li.flag.is_none() {
                self.need_flag.push(match li.temp {
                    Some(t) => FlagKey { temp: true, id: t.seq },
                    None => FlagKey { temp: false, id: li.node },
                });
            }
            merged[l] = Init::Full;
        }
    }

    /// The owner states an edge to join `j` carries from here: those of the owners that outlive
    /// the construct (a local declared before it, a temporary of a statement or scope that was
    /// open when the join was made); every other owner has ended.
    fn edge_state(&self, j: usize) -> Vec<Init> {
        let jn = &self.f.joins[j];
        let stmt_temps = self.f.stmts.get(jn.stmts).copied().unwrap_or(self.f.temps.len());
        let mut st = Vec::with_capacity(self.locals.len());
        for (l, li) in self.locals.iter().enumerate() {
            let outlives = li.drop
                && match li.temp {
                    Some(_) if l < self.f.own_temps => false,
                    Some(t) => match t.ends {
                        Ends::Stmt(i) => i < stmt_temps,
                        Ends::Scope(d) => d < jn.scopes,
                    },
                    None => (li.node as usize) < jn.outer,
                };
            st.push(if outlives { self.st(l) } else { Init::Gone });
        }
        st
    }

    pub fn br_join(&mut self, j: usize, prefix: &[Operand]) {
        if !self.open() {
            return;
        }
        let args = self.edge(j, prefix);
        let b = self.f.joins[j].block;
        self.f.b.br(b, &args);
    }

    /// Continues in join `j`'s block with its params as the env.
    pub fn enter(&mut self, j: usize) {
        let jn = self.f.joins[j].clone();
        self.f.b.switch_to(jn.block);
        for (&l, &p) in jn.locals.iter().zip(&jn.params) {
            self.f.env.vals[l] = Some(Operand::Value(p));
        }
        if let Some(st) = jn.st {
            self.f.env.st = st;
        }
        let outer = jn.outer;
        for (l, li) in self.locals.iter().enumerate() {
            if (li.node as usize) >= outer
                && li.drop
                && li.temp.is_none()
                && let Some(s) = self.f.env.st.get_mut(l)
            {
                *s = Init::Gone;
            }
        }
    }

    /// A value of type `t` for a path where the local has none (dead or erroneous code).
    pub fn dummy(&mut self, t: Ty) -> Operand {
        match self.kind(t) {
            TypeKind::Int { .. } | TypeKind::Bool | TypeKind::Char => self.int(0),
            TypeKind::Float(crate::types::FloatBits::F32) => self.f.b.f32(0.0),
            TypeKind::Float(_) => self.f.b.f64(0.0),
            _ => {
                let pt = self.t_ptr(false, t);
                let z = self.int(0);
                let p = self.op1(Op::Cast(CastKind::Exact), pt, z);
                Operand::Value(self.f.b.load(Self::ir(t), p))
            }
        }
    }

    // ---------- places ----------

    /// `*v` for a pointer value of type `pt`.
    pub fn deref(&mut self, v: Operand, pt: Ty) -> Place {
        let inner = self.pointee(pt).map_or(TypeId::NONE, |x| x.0);
        Place { base: Base::Mem(v, pt), proj: Vec::new(), ty: inner, root: None }
    }

    /// Copies an rvalue into a fresh slot, which owns nothing.
    pub fn spill(&mut self, v: Operand, t: Ty) -> (Operand, Ty) {
        let p = self.new_slot(t);
        let pt = self.t_ref(true, t);
        // each evaluation is a new temporary
        self.event(stave_ir::EventKind::Begin, p, pt);
        self.f.b.store(p, v);
        (p, pt)
    }

    /// The address of a place, as pointer type `want` when given.
    pub fn addr(&mut self, p: &Place, want: Option<Ty>) -> Option<Operand> {
        self.addr_t(p, want).map(|a| a.0)
    }

    /// [`FnCx::addr`] with the pointer's type.
    pub fn addr_t(&mut self, p: &Place, want: Option<Ty>) -> Option<(Operand, Ty)> {
        let (mut cur, mut cur_t) = match &p.base {
            Base::Mem(v, t) => (*v, *t),
            Base::Val(v, t) => {
                let (s, st) = self.spill(*v, *t);
                if self.g.borrows && !self.f.stmts.is_empty() {
                    self.made_at();
                    self.f.stmt_slots.push((s, st));
                }
                (s, st)
            }
            Base::Local(l) => {
                let l = *l;
                // an SSA local has no address: copy it (read-only uses)
                let v = self.f.env.vals.get(l).copied().flatten()?;
                let t = self.locals[l].ty;
                self.spill(v, t)
            }
        };
        let n = p.proj.len();
        for (k, pr) in p.proj.iter().enumerate() {
            let last = k + 1 == n;
            match pr {
                Proj::Field(i, ft) => {
                    let nt = match (last, want) {
                        (true, Some(w)) => w,
                        _ => self.ptr_like(cur_t, *ft),
                    };
                    let i = self.int(*i as i128);
                    cur = self.op2(Op::FieldAddr, nt, cur, i);
                    cur_t = nt;
                }
                Proj::EnumField(v, f, ft) => {
                    let nt = match (last, want) {
                        (true, Some(w)) => w,
                        _ => self.ptr_like(cur_t, *ft),
                    };
                    let (v, f) = (self.int(*v as i128), self.int(*f as i128));
                    cur = self.opn(Op::EnumFieldAddr, nt, &[cur, v, f]);
                    cur_t = nt;
                }
                Proj::Elem(idx, et, tok) => {
                    let base_t = self.pointee(cur_t).map_or(TypeId::NONE, |x| x.0);
                    let n = match self.kind(base_t) {
                        TypeKind::Array { len: ConstArg::Value(n), .. } => self.int(n as i128),
                        _ => {
                            let us = self.t_usize();
                            self.op1(Op::SliceLen, us, cur)
                        }
                    };
                    self.trapping(*tok, Op::BoundsCheck, None, &[*idx, n]);
                    let nt = match (last, want) {
                        (true, Some(w)) => w,
                        _ => self.ptr_like(cur_t, *et),
                    };
                    cur = self.op2(Op::ElemAddr, nt, cur, *idx);
                    cur_t = nt;
                }
            }
        }
        if let Some(w) = want
            && n == 0
        {
            cur = self.convert_ptr(cur, cur_t, w);
            cur_t = w;
        }
        Some((cur, cur_t))
    }

    /// Reads a place by value (a move of its owner, when droppable).
    pub fn read(&mut self, p: &Place) -> Option<Operand> {
        let r = self.take(p);
        self.note_move(p);
        r
    }

    /// The value of a place that is moved out of it, without the move of its owner: a part a
    /// pattern binds by value (its owner is moved where the pattern is committed).
    pub fn take(&mut self, p: &Place) -> Option<Operand> {
        let sized = !matches!(self.tys.kind(p.ty), TypeKind::Slice { .. } | TypeKind::Str);
        if self.g.borrows
            && sized
            && !matches!(p.base, Base::Val(..))
            && !self.is_unit(p.ty)
            && !self.is_copy(p.ty)
        {
            match &p.base {
                // a move is an event on the bytes of its place: the local gets a slot
                Base::Local(l) => self.need_slot.push(self.locals[*l].node),
                _ => {
                    let (a, at) = self.addr_t(p, None)?;
                    let v = Operand::Value(self.f.b.load(Self::ir(p.ty), a));
                    match &mut self.f.held {
                        Some(h) => h.push((a, at)),
                        None => self.event(stave_ir::EventKind::Gone, a, at),
                    }
                    return Some(v);
                }
            }
        }
        self.peek(p)
    }

    /// Reads a place without moving out of it (discriminants, copies of borrowed values).
    pub fn peek(&mut self, p: &Place) -> Option<Operand> {
        if self.is_unit(p.ty) {
            return Some(self.unit_val());
        }
        match &p.base {
            Base::Local(l) => {
                let v = self.f.env.vals.get(*l).copied().flatten();
                let v = match v {
                    Some(v) => v,
                    None => {
                        let t = self.locals[*l].ty;
                        self.dummy(t)
                    }
                };
                self.read_proj(v, &p.proj)
            }
            Base::Val(v, _) => self.read_proj(*v, &p.proj),
            Base::Mem(..) => {
                if matches!(self.kind(p.ty), TypeKind::Slice { .. } | TypeKind::Str) {
                    self.fail_at(None, "read of an unsized place");
                    return None;
                }
                let a = self.addr(p, None)?;
                Some(Operand::Value(self.f.b.load(Self::ir(p.ty), a)))
            }
        }
    }

    fn read_proj(&mut self, mut v: Operand, proj: &[Proj]) -> Option<Operand> {
        for pr in proj {
            v = match pr {
                Proj::Field(i, t) => {
                    let i = self.int(*i as i128);
                    self.op2(Op::Get, *t, v, i)
                }
                Proj::Elem(idx, t, tok) => self.trapping(*tok, Op::Get, Some(*t), &[v, *idx])?,
                Proj::EnumField(var, f, t) => {
                    let (a, b) = (self.int(*var as i128), self.int(*f as i128));
                    self.opn(Op::EnumGet, *t, &[v, a, b])
                }
            };
        }
        Some(v)
    }

    /// Writes `v` into a place (no drop of the old value; see [`FnCx::assign`]).
    pub fn write(&mut self, p: &Place, v: Operand) {
        match &p.base {
            Base::Mem(..) => {
                if let Some(a) = self.addr(p, None) {
                    self.f.b.store(a, v);
                }
            }
            Base::Local(l) => {
                let l = *l;
                if p.proj.is_empty() {
                    self.f.env.vals[l] = Some(v);
                    return;
                }
                let Some(root) = self.f.env.vals.get(l).copied().flatten() else { return };
                // aggregates along the path
                let mut chain = vec![(root, self.locals[l].ty)];
                for pr in &p.proj[..p.proj.len() - 1] {
                    let (cur, _) = *chain.last().unwrap_or(&(root, TypeId::NONE));
                    let next = match pr {
                        Proj::Field(i, t) => {
                            let i = self.int(*i as i128);
                            (self.op2(Op::Get, *t, cur, i), *t)
                        }
                        Proj::Elem(idx, t, tok) => match self.trapping(*tok, Op::Get, Some(*t), &[cur, *idx])
                        {
                            Some(x) => (x, *t),
                            None => return,
                        },
                        Proj::EnumField(..) => {
                            self.fail_at(None, "assignment to an enum payload of an SSA value");
                            return;
                        }
                    };
                    chain.push(next);
                }
                let mut nv = v;
                for (pr, &(agg, at)) in p.proj.iter().zip(chain.iter()).rev() {
                    nv = match pr {
                        Proj::Field(i, _) => {
                            let i = self.int(*i as i128);
                            self.opn(Op::Set, at, &[agg, i, nv])
                        }
                        Proj::Elem(idx, _, tok) => {
                            match self.trapping(*tok, Op::Set, Some(at), &[agg, *idx, nv]) {
                                Some(x) => x,
                                None => return,
                            }
                        }
                        Proj::EnumField(..) => return,
                    };
                }
                self.f.env.vals[l] = Some(nv);
            }
            Base::Val(..) => {
                self.fail_at(None, "assignment to a temporary");
            }
        }
    }

    /// `place = v`: drops the old value of the place where it holds one, then writes it. An
    /// owner assigned as a whole holds its value afterwards.
    pub fn assign(&mut self, p: &Place, v: Operand) {
        match p.root {
            Some(l) if self.locals[l].drop => {
                self.drop_if_held(l, p);
                self.write(p, v);
                if p.proj.is_empty() {
                    self.set_st(l, Init::Full);
                    self.set_flag(l, true);
                }
            }
            Some(_) => self.write(p, v),
            None => {
                self.drop_place(p);
                self.write(p, v);
            }
        }
    }

    /// Drops the value in place `p`, which is owner `l` or a part of it, if the owner holds its
    /// value: its flag decides when it has one (a constant flag is folded), else the state.
    fn drop_if_held(&mut self, l: usize, p: &Place) {
        if !self.needs_drop(p.ty) || !self.open() {
            return;
        }
        let Some(f) = self.locals[l].flag else {
            if self.st(l) == Init::Full {
                self.drop_held(l, p);
            }
            return;
        };
        let Some(fv) = self.flag_val(f) else { return };
        match self.kval(fv) {
            Some(0) => {}
            Some(_) => self.drop_held(l, p),
            None => {
                let yes = self.f.b.create_block();
                let cont = self.f.b.create_block();
                self.f.b.cbr(fv, yes, &[], cont, &[]);
                self.f.b.switch_to(yes);
                self.drop_held(l, p);
                if self.open() {
                    self.f.b.br(cont, &[]);
                }
                self.f.b.switch_to(cont);
            }
        }
    }

    /// Drops what owner `l` holds of its value in place `p`: all of it, or, of the array a `for`
    /// loop takes element by element, the elements the loop has not taken.
    fn drop_held(&mut self, l: usize, p: &Place) {
        match self.locals[l].temp {
            Some(Temp { home: Home::Rest(s, taken), .. }) => self.drop_rest(s, taken, p.ty),
            _ => self.drop_place(p),
        }
    }

    /// Drops the elements of the array of type `t` in slot `s` from the index in slot `taken`
    /// on, in order.
    fn drop_rest(&mut self, s: Operand, taken: Operand, t: Ty) {
        let TypeKind::Array { elem, len } = self.kind(t) else {
            self.fail_at(None, "a `for` loop owns a value that is no array");
            return;
        };
        let us = self.t_usize();
        let bt = self.t_bool();
        let (head, hv) = self.f.b.create_block_with(&[Self::ir(us)]);
        let body = self.f.b.create_block();
        let exit = self.f.b.create_block();
        let first = Operand::Value(self.f.b.load(Self::ir(us), taken));
        self.f.b.br(head, &[first]);
        self.f.b.switch_to(head);
        let i = Operand::Value(hv[0]);
        let n = self.int(len.value().unwrap_or(0) as i128);
        let c = self.op2(Op::Lt, bt, i, n);
        self.f.b.cbr(c, body, &[], exit, &[]);
        self.f.b.switch_to(body);
        let ep = self.t_ref(true, elem);
        let a = self.op2(Op::ElemAddr, ep, s, i);
        self.drop_ptr(a, elem);
        let one = self.int(1);
        let next = self.op2(Op::AddWrap, us, i, one);
        self.f.b.br(head, &[next]);
        self.f.b.switch_to(exit);
    }

    // ---------- drops ----------

    /// Drops the value in a place (memory).
    pub fn drop_place(&mut self, p: &Place) {
        if !self.needs_drop(p.ty) {
            return;
        }
        let pt = self.t_ref(true, p.ty);
        let Some(a) = self.addr(p, Some(pt)) else { return };
        self.drop_ptr(a, p.ty);
    }

    /// `drop(*a)` for a `&mut T` pointer `a`.
    pub fn drop_ptr(&mut self, a: Operand, t: Ty) {
        if !self.needs_drop(t) {
            return;
        }
        let w = self.g.world();
        let direct = match crate::mono::drop_method(&w, &mut self.tys, t) {
            Some(k) => {
                let fields = self.adt_fields(t).unwrap_or_default();
                let mut any = false;
                for f in fields.into_iter().flatten() {
                    any |= self.needs_drop(f);
                }
                if any { None } else { Some(k) }
            }
            None => None,
        };
        let f = match direct {
            Some(k) => self.req(Req::Inst(k)),
            None => self.req(Req::Glue(t)),
        };
        self.f.last_site = None;
        // the glue runs on a `&mut` borrow of the place; then the value is gone from it
        let pt = self.t_ref(true, t);
        let arg = self.borrowed(a, pt);
        self.f.b.call(f, None, &[arg]);
        self.event(stave_ir::EventKind::Gone, a, pt);
    }

    /// Drops owner `l` where it ends, if it holds its value (its state is kept: the caller knows
    /// whether the owner is left on this path only).
    pub fn drop_owner(&mut self, l: usize) {
        if !self.locals[l].drop || !self.open() {
            return;
        }
        if self.locals[l].flag.is_none() && self.st(l) == Init::Gone {
            return;
        }
        let p = match self.local_place(l) {
            // the value is dropped in memory
            Place { base: Base::Val(v, t), .. } => {
                let (s, st) = self.spill(v, t);
                Place { base: Base::Mem(s, st), proj: Vec::new(), ty: t, root: Some(l) }
            }
            p => p,
        };
        self.drop_if_held(l, &p);
        if let Some(Temp { home: Home::Slot(s) | Home::Rest(s, _), .. }) = self.locals[l].temp {
            // the slot of a temporary ends with it
            let pt = self.t_ref(true, self.locals[l].ty);
            self.event(stave_ir::EventKind::End, s, pt);
        }
        // a temporary's flag says "made and not dropped" each time its statement runs
        if let Some(Flag::Mem(fs)) = self.locals[l].flag
            && self.open()
        {
            let zero = self.int(0);
            self.f.b.store(fs, zero);
        }
    }

    /// Drops the droppable locals of scopes `from..` (innermost first), keeping their states.
    pub fn drop_scopes(&mut self, from: usize) {
        let n = self.f.scopes.len();
        for s in (from..n).rev() {
            let wbs: Vec<WriteBack> =
                self.f.write_backs.iter().rev().filter(|w| w.0 == s + 1).map(|w| w.1).collect();
            for w in wbs {
                self.write_back(w);
            }
            // threads spawned in a `scope` block are joined before its locals are dropped
            let hs: Vec<Operand> =
                self.f.scope_handles.iter().rev().filter(|h| h.0 == s + 1).map(|h| h.1).collect();
            for h in hs {
                self.scope_join(h);
            }
            let ls = self.f.scopes[s].clone();
            for &l in ls.iter().rev() {
                self.drop_owner(l);
            }
        }
    }

    pub fn push_scope(&mut self) {
        let d = self.f.scopes.len();
        self.f.scope_stmts.truncate(d);
        self.f.scope_stmts.push(self.f.stmts.len());
        self.f.scopes.push(Vec::new());
    }

    /// Ends the innermost scope: drops its locals (if the block is still open).
    pub fn pop_scope(&mut self) {
        self.pop_scope_at(false);
    }

    /// [`FnCx::pop_scope`]; `frame_exit`: the fn returns next, which ends every slot of the
    /// frame, so the locals get no end of their own.
    pub fn pop_scope_at(&mut self, frame_exit: bool) {
        let d = self.f.scopes.len();
        while let Some(&(wd, w)) = self.f.write_backs.last()
            && wd == d
        {
            self.f.write_backs.pop();
            self.write_back(w);
        }
        while let Some(&(hd, h)) = self.f.scope_handles.last()
            && hd == d
        {
            self.f.scope_handles.pop();
            self.scope_join(h);
        }
        let Some(ls) = self.f.scopes.pop() else { return };
        for &l in ls.iter().rev() {
            self.drop_owner(l);
            self.set_st(l, Init::Gone);
        }
        if !frame_exit {
            self.end_locals(&ls);
        }
        self.trim_temps();
    }

    /// Stores a `&mut` chunk binding's vector back to its slice (active lanes only when masked).
    pub fn write_back(&mut self, w: WriteBack) {
        if !self.open() {
            return;
        }
        let v = Operand::Value(self.f.b.load(Self::ir(w.vt), w.slot));
        match w.mask {
            Some(m) => self.vop(w.tok, stave_ir::VecOp::StoreMasked, None, &[w.s, w.i, v, m]),
            None => self.vop(w.tok, stave_ir::VecOp::Store, None, &[w.s, w.i, v]),
        };
    }

    /// `runtime.thread.scope_join(h)`: waits for the threads spawned into the `scope` handle `h`.
    pub fn scope_join(&mut self, h: Operand) {
        if !self.open() {
            return;
        }
        let Some(d) = self.g.prog.known.decl(stave_ir::Known::ScopeJoin) else {
            self.fail_at(None, "`runtime.thread.scope_join` not found");
            return;
        };
        let f = self.req(Req::Inst(InstKey { decl: d, args: Vec::new() }));
        self.f.last_site = None;
        self.f.b.call(f, None, &[h]);
    }

    /// Opens statement `b` (a node [`crate::temps`] names): the temporaries made from here on
    /// end at the matching [`FnCx::end_temps`].
    pub fn begin_temps(&mut self, b: u32) {
        if !self.tb().stmt_ends.is_stmt(b) {
            self.fail(b, "a statement the temporaries table does not name");
        }
        self.f.stmts.push(self.f.temps.len());
        self.f.slot_marks.push(self.f.stmt_slots.len());
        self.f.open.push(b);
    }

    /// Ends the innermost open statement: drops its temporaries, last made first, then ends
    /// the slots of the rvalues it took the address of (every temporary ends with its
    /// statement, M14).
    pub fn end_temps(&mut self) {
        let Some(from) = self.f.stmts.pop() else { return };
        let ts: Vec<usize> = self.f.temps.get(from..).unwrap_or(&[]).to_vec();
        for &l in ts.iter().rev() {
            self.drop_owner(l);
            self.set_st(l, Init::Gone);
        }
        self.f.temps.truncate(from);
        self.trim_temps();
        let at = self.f.slot_marks.pop().unwrap_or(0);
        self.end_slots(at);
        self.f.stmt_slots.truncate(at);
        self.f.open.pop();
    }

    /// Drops, on a path that leaves them, the temporaries of the open statements from index
    /// `from` on (0: all, a `return` or `?`), keeping them for the paths that stay.
    pub fn exit_temps(&mut self, from: usize) {
        let at = if from == 0 { 0 } else { self.f.stmts.get(from).copied().unwrap_or(self.f.temps.len()) };
        let ts: Vec<usize> = self.f.temps.get(at..).unwrap_or(&[]).to_vec();
        for &l in ts.iter().rev() {
            self.drop_owner(l);
        }
        let at = if from == 0 {
            0
        } else {
            self.f.slot_marks.get(from).copied().unwrap_or(self.f.stmt_slots.len())
        };
        self.end_slots(at);
    }

    /// The end of the rvalue slots of the open statements from `stmt_slots[at]` on, last
    /// taken first.
    fn end_slots(&mut self, at: usize) {
        let ss: Vec<(Operand, Ty)> = self.f.stmt_slots.get(at..).unwrap_or(&[]).to_vec();
        for &(s, st) in ss.iter().rev() {
            self.event(stave_ir::EventKind::End, s, st);
        }
    }

    // ---------- calls ----------

    /// Signature of an instance (params, ret), substituted.
    pub fn inst_sig(&mut self, k: &InstKey) -> Option<(Vec<Ty>, Ty)> {
        let sig = self.g.sigs.fn_sig(k.decl)?;
        let mut ps = Vec::with_capacity(sig.params.len());
        for &p in &sig.params {
            ps.push(self.tys.subst(p, &k.args, None));
        }
        let r = self.tys.subst(sig.ret, &k.args, None);
        Some((ps, r))
    }

    /// Emits a direct call of instance `k` (with a user-site store first when a user fn calls
    /// into a built-in package). Returns the result.
    pub fn call_inst(&mut self, k: InstKey, args: &[Operand], tok: u32) -> E {
        let (_, ret) = match self.inst_sig(&k) {
            Some(s) => s,
            None => return self.fail_at(Some(tok), "call of a fn without a signature"),
        };
        let builtin = self.g.builtin_decl(k.decl);
        let requires = self.g.reaches_requires(k.decl);
        if self.user && (builtin || requires) {
            self.user_site(tok);
        } else {
            self.f.last_site = None;
        }
        let f = self.req(Req::Inst(k));
        let r = self.emit_call(f, args, ret, tok);
        if requires && !builtin {
            // user code ran: it may have stored other sites
            self.f.last_site = None;
        }
        r
    }

    pub fn user_site(&mut self, tok: u32) {
        if self.hold_site {
            return;
        }
        let Some(sus) = self.g.lang.set_user_site else { return };
        let s = self.site(tok);
        let here = self.f.b.current().map(|b| (b, s.0));
        if here.is_some() && self.f.last_site == here {
            return;
        }
        self.f.last_site = here;
        let c = self.f.b.int(s.0 as i128);
        if let Operand::Const(ci) = c {
            self.f.site_consts.push(ci);
        }
        let f = self.req(Req::Inst(InstKey { decl: sus, args: Vec::new() }));
        self.f.b.call(f, None, &[c]);
    }

    /// In a user fn, a call of error-trace hook `hook` (`runtime.errtrace.push` / `restart`) with
    /// the site of token `tok`. Like the user-site store it is no call-graph edge, so it adds no
    /// effect; the callee neither traps nor stores a user site.
    pub fn errtrace_hook(&mut self, hook: Option<DeclId>, tok: u32) {
        let Some(d) = hook else { return };
        if !self.user || !self.open() {
            return;
        }
        let s = self.site(tok);
        let c = self.f.b.int(s.0 as i128);
        if let Operand::Const(ci) = c {
            self.f.site_consts.push(ci);
        }
        let f = self.req(Req::Inst(InstKey { decl: d, args: Vec::new() }));
        self.f.b.call(f, None, &[c]);
    }

    pub fn emit_call(&mut self, f: FuncId, args: &[Operand], ret: Ty, tok: u32) -> E {
        let rt = self.ret_of(ret);
        let s = self.site(tok);
        self.f.b.set_site(s);
        let r = self.f.b.call(f, rt.map(Self::ir), args);
        self.f.b.set_site(self.f.stmt_site);
        self.call_result(r, ret)
    }

    pub fn call_result(&mut self, r: Option<Value>, ret: Ty) -> E {
        if self.is_never(ret) {
            self.f.b.unreachable();
            return E::Never;
        }
        match r {
            Some(v) => E::V(Operand::Value(v), ret),
            None => E::Unit,
        }
    }

    /// Resolves a method reference with flat args to a callable.
    pub fn resolve_method(&mut self, m: &MethodRef, args: &[GArg]) -> Option<Resolved> {
        let args: Vec<GArg> = args.iter().map(|a| self.sub_garg(*a)).collect();
        match m {
            MethodRef::Impl { method, .. } => Some(Resolved::Inst(InstKey { decl: *method, args })),
            MethodRef::Bound { method, trait_args, self_ty } => {
                let st = self.sub(*self_ty);
                let np = self.g.sigs.fn_sig(*method).map_or(0, |s| s.n_parent as usize);
                let ta: Vec<GArg> = trait_args.iter().map(|a| self.sub_garg(*a)).collect();
                let own: Vec<GArg> = args.get(np..).unwrap_or(&[]).to_vec();
                let w = self.g.world();
                crate::mono::resolve_trait_method(&w, &mut self.tys, *method, st, &ta, &own)
            }
            MethodRef::Builtin(_) => None,
        }
    }

    pub fn sub_garg(&mut self, a: GArg) -> GArg {
        match a {
            GArg::Type(t) => GArg::Type(self.sub(t)),
            GArg::Const(ConstArg::Param(i)) => match self.args.get(i as usize) {
                Some(GArg::Const(c)) => GArg::Const(*c),
                _ => a,
            },
            c => c,
        }
    }

    /// The target recorded at token `t`.
    pub fn target(&self, t: u32) -> Option<&'a Target> {
        self.tb().target(t)
    }

    pub fn decl_kind(&self, d: DeclId) -> Option<DeclKind> {
        self.g.prog.decl(d).map(|x| x.kind)
    }

    pub fn kids(&self, n: u32) -> crate::itemcx::Kids {
        self.icx().kids_vec(n)
    }

    pub fn nkind(&self, n: u32) -> Option<NodeKind> {
        self.icx().kind(n)
    }
}

impl FnState {
    pub fn new(params: &[Ty], ret: Ty) -> FnState {
        let ps: Vec<stave_ir::TypeId> = params.iter().map(|&t| FnCx::ir(t)).collect();
        FnState {
            b: FuncBuilder::new(&ps),
            slot: Vec::new(),
            env: Env::default(),
            scopes: Vec::new(),
            loops: Vec::new(),
            joins: Vec::new(),
            temps: Vec::new(),
            stmts: Vec::new(),
            open: Vec::new(),
            cur: crate::temps::NONE,
            scope_stmts: Vec::new(),
            own_temps: 0,
            ret,
            site_consts: Vec::new(),
            kvals: Vec::new(),
            last_site: None,
            scope_handles: Vec::new(),
            write_backs: Vec::new(),
            borrows: Vec::new(),
            events: Vec::new(),
            held: None,
            stmt_slots: Vec::new(),
            slot_marks: Vec::new(),
            stmt_site: Site::NONE,
            ensures_end: None,
        }
    }
}

fn empty_body() -> &'static TypedBody {
    static B: std::sync::OnceLock<TypedBody> = std::sync::OnceLock::new();
    B.get_or_init(|| TypedBody {
        decl: DeclId::NONE,
        item: crate::ids::ItemId::NONE,
        node: 0,
        lo: 0,
        node_types: Vec::new(),
        locals: Vec::new(),
        targets: Vec::new(),
        tries: Vec::new(),
        coercions: Vec::new(),
        sites: Vec::new(),
        implicit: Vec::new(),
        for_kinds: Vec::new(),
        moves: Vec::new(),
        drops: Vec::new(),
        denied: Vec::new(),
        valued: Vec::new(),
        has_error: true,
        emulated: Vec::new(),
        contracts: None,
        test_domain: None,
        stmt_ends: crate::temps::StmtEnds::default(),
    })
}
