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
//! **Drops.** A droppable local is dropped at the end of its scope (reverse declaration order),
//! at `break`/`continue`/`return` for the scopes they leave, and before it is overwritten, unless
//! it was moved out. The moved state is tracked statically; when two paths disagree at a join,
//! the job is lowered again with a dynamic drop flag (a hidden `bool` SSA local) for that local.

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
    /// The hidden flag local of a flagged droppable local.
    pub flag: Option<usize>,
}

/// Static moved state of a droppable local.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MState {
    Uninit,
    Init,
    Moved,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Env {
    pub vals: Vec<Option<Operand>>,
    pub st: Vec<MState>,
}

/// A join block under construction.
#[derive(Clone, Debug)]
pub(crate) struct Join {
    pub block: Block,
    pub prefix: Vec<Value>,
    pub locals: Vec<usize>,
    pub params: Vec<Value>,
    /// Merged moved states of the edges so far (`None`: no edge yet).
    pub st: Option<Vec<MState>>,
    /// Locals declared before the construct (their states merge).
    pub outer: usize,
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
    /// Statement temporaries to drop at the end of the statement: (pointer, type).
    pub temps: Vec<(Operand, Ty)>,
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
    /// Binding nodes of locals that get a dynamic drop flag.
    pub flagged: FxHashSet<u32>,
    /// Discovered during this attempt: locals that need a flag.
    pub need_flag: Vec<u32>,
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

    pub fn unsupported(&mut self, n: u32, msg: impl Into<String>) -> E {
        let tok = Some(self.icx.a(n));
        self.unsupported_at(tok, msg)
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
        self.f.b.set_site(Site::NONE);
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
        if let Some(p) = self.f.slot.get(l).copied().flatten() {
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
            if let Some(fl) = self.locals[l].flag {
                let one = self.int(1);
                self.f.env.vals[fl] = Some(one);
            }
            self.f.env.st[l] = MState::Init;
            if let Some(s) = self.f.scopes.last_mut() {
                s.push(l);
            }
        }
    }

    /// Marks the root local of a place read by value as moved (droppable parts only).
    pub fn note_move(&mut self, root: Option<usize>, ty: Ty) {
        let Some(l) = root else { return };
        if !self.locals[l].drop || !self.needs_drop(ty) {
            return;
        }
        if let Some(fl) = self.locals[l].flag {
            let z = self.int(0);
            self.f.env.vals[fl] = Some(z);
        }
        self.f.env.st[l] = MState::Moved;
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
        for li in &self.locals {
            if let Some(fl) = li.flag
                && li.node < lo
                && self.f.env.vals.get(fl).is_some_and(|v| v.is_some())
            {
                out.push(fl);
            }
        }
        out.sort_unstable();
        out.dedup();
        out.retain(|&l| l < self.locals.len());
        out
    }

    /// Creates a join block for construct `n` with leading params of types `prefix`.
    pub fn join(&mut self, n: u32, prefix: &[Ty]) -> usize {
        let locals = self.assigned_in(n);
        let block = self.f.b.create_block();
        let pre: Vec<Value> = prefix.iter().map(|&t| self.f.b.add_param(block, Self::ir(t))).collect();
        let mut params = Vec::with_capacity(locals.len());
        for &l in &locals {
            let t = self.locals[l].ty;
            params.push(self.f.b.add_param(block, Self::ir(t)));
        }
        let outer = if self.has_body { self.icx.ast.descendants(n).start as usize } else { 0 };
        self.f.joins.push(Join { block, prefix: pre, locals, params, st: None, outer });
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
        let outer = self.f.joins[j].outer;
        let st = self.f.env.st.clone();
        match self.f.joins[j].st.take() {
            None => self.f.joins[j].st = Some(st),
            Some(prev) => {
                let mut merged = prev.clone();
                for (l, (a, b)) in prev.iter().zip(&st).enumerate() {
                    if a != b {
                        if (self.locals[l].node as usize) < outer
                            && self.locals[l].drop
                            && self.locals[l].flag.is_none()
                            && *a != MState::Uninit
                            && *b != MState::Uninit
                        {
                            self.need_flag.push(self.locals[l].node);
                        }
                        merged[l] = MState::Moved;
                    }
                }
                self.f.joins[j].st = Some(merged);
            }
        }
        args
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
            if (li.node as usize) >= outer && li.drop {
                self.f.env.st[l] = MState::Uninit;
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

    /// Spills an rvalue into a fresh temporary slot (dropped at the end of the statement when
    /// `own` and droppable).
    pub fn spill(&mut self, v: Operand, t: Ty, own: bool) -> (Operand, Ty) {
        let p = self.new_slot(t);
        self.f.b.store(p, v);
        if own && self.needs_drop(t) {
            self.f.temps.push((p, t));
        }
        let pt = self.t_ref(true, t);
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
                let (s, st) = self.spill(*v, *t, false);
                (s, st)
            }
            Base::Local(l) => {
                let l = *l;
                // an SSA local has no address: copy it (read-only uses)
                let v = self.f.env.vals.get(l).copied().flatten()?;
                let t = self.locals[l].ty;
                self.spill(v, t, false)
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

    /// Reads a place by value (a move of its droppable root).
    pub fn read(&mut self, p: &Place) -> Option<Operand> {
        let r = self.peek(p);
        self.note_move(p.root, p.ty);
        r
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

    /// `place = v`: drops the old value when the place holds an initialized droppable value.
    pub fn assign(&mut self, p: &Place, v: Operand) {
        let whole_local = p.proj.is_empty() && p.root.is_some();
        if whole_local && let Some(l) = p.root {
            if self.locals[l].drop {
                if let Some(fl) = self.locals[l].flag {
                    let f = self.f.env.vals[fl];
                    if let Some(f) = f {
                        self.drop_if(f, p.clone());
                    }
                    let one = self.int(1);
                    self.f.env.vals[fl] = Some(one);
                } else if self.f.env.st[l] == MState::Init {
                    self.drop_place(p);
                }
                self.f.env.st[l] = MState::Init;
            }
            self.write(p, v);
            return;
        }
        if self.needs_drop(p.ty) {
            let init = match p.root {
                Some(l) => {
                    self.f.env.st.get(l).copied() == Some(MState::Init) || self.locals[l].flag.is_some()
                }
                None => true,
            };
            if init {
                self.drop_place(p);
            }
        }
        self.write(p, v);
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
        self.f.b.call(f, None, &[a]);
    }

    /// Drops `p` if the flag value `flag` is set.
    pub fn drop_if(&mut self, flag: Operand, p: Place) {
        let yes = self.f.b.create_block();
        let cont = self.f.b.create_block();
        self.f.b.cbr(flag, yes, &[], cont, &[]);
        self.f.b.switch_to(yes);
        self.drop_place(&p);
        self.f.b.br(cont, &[]);
        self.f.b.switch_to(cont);
    }

    /// Drops local `l` at a scope exit (if initialized).
    pub fn drop_local(&mut self, l: usize) {
        if !self.locals[l].drop || !self.open() {
            return;
        }
        let p = self.local_place(l);
        if let Some(fl) = self.locals[l].flag {
            let Some(f) = self.f.env.vals.get(fl).copied().flatten() else { return };
            if self.kval(f) == Some(0) {
                return;
            }
            if self.kval(f) == Some(1) {
                self.drop_place(&p);
                return;
            }
            self.drop_if(f, p);
        } else if self.f.env.st[l] == MState::Init {
            self.drop_place(&p);
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
                self.drop_local(l);
            }
        }
    }

    pub fn push_scope(&mut self) {
        self.f.scopes.push(Vec::new());
    }

    /// Ends the innermost scope: drops its locals (if the block is still open).
    pub fn pop_scope(&mut self) {
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
            self.drop_local(l);
            self.f.env.st[l] = MState::Uninit;
        }
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
        let Some(d) = crate::mono::item_at(self.g.prog, "runtime.thread", "scope_join") else {
            self.fail_at(None, "`runtime.thread.scope_join` not found");
            return;
        };
        let f = self.req(Req::Inst(InstKey { decl: d, args: Vec::new() }));
        self.f.last_site = None;
        self.f.b.call(f, None, &[h]);
    }

    /// Drops the statement temporaries.
    pub fn drop_temps(&mut self) {
        let ts = std::mem::take(&mut self.f.temps);
        if !self.open() {
            return;
        }
        for &(p, t) in ts.iter().rev() {
            self.drop_ptr(p, t);
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
        self.f.b.set_site(Site::NONE);
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
            ret,
            site_consts: Vec::new(),
            kvals: Vec::new(),
            last_site: None,
            scope_handles: Vec::new(),
            write_backs: Vec::new(),
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
        coercions: Vec::new(),
        sites: Vec::new(),
        implicit: Vec::new(),
        drops: Vec::new(),
        denied: Vec::new(),
        has_error: true,
        emulated: Vec::new(),
        contracts: None,
        test_domain: None,
    })
}
