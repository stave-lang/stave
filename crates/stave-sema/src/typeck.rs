//! The type checker (non-mono parts): one function body at a time, from the frozen
//! signature table ([`crate::sigs`]), in parallel (rayon), with deterministic output.
//!
//! [`check`] runs, in order: the signature build (sequential, demand-driven, canonical decl
//! order), impl conformance and the intrinsic table, every body (fns, methods, exports, const and
//! `static mut` initializers) in parallel on big-stack workers (as many as the caller's rayon pool
//! has; module `pool`), the merge of the bodies' type overlays into the global table, the
//! canonical renumbering of every type id, and the instantiation depth check (E0902,
//! [`crate::mono::depth_check`]). Diagnostics are in canonical order.
//!
//! **What a body records** ([`TypedBody`]): the type of every expression node (`TypeId::NONE`
//! for nodes that denote no value, such as `io` in `io.print(..)`, and for the error type), the
//! type of every local binding, the resolved target of every call, method, operator, field,
//! variant, and path ([`Target`]), the implicit coercions made explicit ([`Coercion`]: auto-ref
//! and auto-deref of receivers, `&mut` reborrows, `never` to any type, `&mut T` to `&T`, `&[T; N]`
//! to `&[T]`, `&String` and f-strings to `&str`, fn items to fn types), and one instantiation
//! record per call of a generic fn or method ([`CallSite`]: caller, node, decl, canonical args),
//! for the mono stage. A body with any type error (including an unresolved name) has
//! `has_error`: borrowck, lowering, and mono skip it (no cascades).
//!
//! **No cascades**: the error type is compatible with everything; a check whose
//! operand has the error type reports nothing.
//!
//! **Rules LANG.md leaves open**: an int literal never becomes a
//! float (E0201 found `{integer}`); matching a reference against a constructor pattern peels the
//! reference and binds by reference (`match self { Option.Some(x) => .. }` with `self: &Option<T>`
//! binds `x: &T`); `for x in xs` over `&[T]` / `&[T; N]` binds `x: &T` (`&mut`: `&mut T`), over an
//! array value binds `T`, over a type with `fn next(&mut self) -> Option<X>` binds `X`; `&[T; N]`
//! coerces to `&[T]` and `&mut T` to `&T` at coercion sites; fn values are Copy; an f-string hole
//! whose type has no `Display` impl is E0208 (`fmt`); a bound not satisfied at a call is E0901
//! case `unsatisfied`; view arguments for a param without `view` are E0414 (case `container`
//! when the generic ADT is in the std package). Ranges have the internal type
//! `Range<T>` (see [`crate::types`]). Expressions, blocks, and patterns nested deeper than
//! [`MAX_DEPTH`] are E0005 case `expr` (once per body, at the first node over the limit) and get
//! the error type.

use std::time::{Duration, Instant};

use rayon::prelude::*;
use rustc_hash::FxHashMap;
use stave_diag::{Diagnostic, Span};
use stave_syntax::{NodeKind, Term};

use crate::Program;
use crate::callgraph::CallGraph;
use crate::decls::{DeclFlags, DeclKind};
use crate::diag::mk;
use crate::ids::{BindingRef, DeclId, ItemId};
use crate::itemcx::{Brackets, ItemCx};
use crate::sigs::{self, DeclSig, FnSig, GenericParam, NameCx, SigBuilder, Sigs, TyEnv};
use crate::traits::{Adjust, MethodRef, World};
use crate::types::{self, ConstArg, GArg, Interner, Overlay, TypeId, TypeKind, TypeTable, ids};

/// Maximum expression/statement/pattern nesting the checker descends (beyond: E0005 case `expr`).
/// The one limit of every recursive pass over a body (typeck, borrowck, lowering): each counts
/// at most one level per level typeck counts, so a body typeck accepts never hits a later limit.
pub const MAX_DEPTH: u32 = 20_000;

/// What a node resolved to (for lowering, borrowck, mono).
#[derive(Clone, Debug, PartialEq)]
pub enum Target {
    /// A call or value of fn item `decl` with its generic args (flat: container's, then own).
    Fn { decl: DeclId, args: Vec<GArg> },
    /// A method call: the method, its full generic args, the receiver adjustment.
    Method { method: MethodRef, args: Vec<GArg>, adjust: Adjust },
    /// A trait method called through a bound or a qualified `Trait.m(x)` with `Self` = `self_ty`.
    TraitMethod { method: DeclId, self_ty: TypeId, args: Vec<GArg> },
    /// A built-in method of a built-in type, or a type-relative built-in (`f32x4.splat`);
    /// `ret`: the call's result type; `consts`: its constants (the `div_const` divisor's bits,
    /// two's complement for signed lanes; the `permute` lane indices).
    Builtin { name: &'static str, recv: TypeId, adjust: Adjust, ret: TypeId, consts: Vec<u64> },
    /// A user-type operator: the trait method (`Add.add`) through the impl or a bound.
    Op { method: MethodRef, args: Vec<GArg> },
    /// An enum variant constructor or value; `ty`: the enum type of the value made.
    Variant { decl: DeclId, index: u32, ty: TypeId },
    /// A struct literal; `ty`: the struct type of the value made.
    Struct { decl: DeclId, ty: TypeId },
    /// Field `index` of the struct (after `derefs` auto-derefs).
    Field { decl: DeclId, index: u32, derefs: u32 },
    /// Tuple element.
    TupleField { index: u32, derefs: u32 },
    /// Column `index` of a `std.soa.Soa<T>` (field `index` of T): a `[F]` place (after `derefs`
    /// auto-derefs).
    Column { index: u32, derefs: u32 },
    /// A row built-in of `std.soa` (`soa_row_size`, `soa_read`, `soa_write`) on row type `ty`.
    SoaRow { op: crate::prim::Builtin, ty: TypeId },
    /// A const or `static mut` use.
    Global { decl: DeclId },
    /// `size_of::<T>()` / `align_of::<T>()`.
    SizeOf { ty: TypeId, align: bool },
    /// `fence_stream()`.
    FenceStream,
    /// An `extern "stave-intrinsic"` call (checked IR `intrinsic.<name>`).
    Intrinsic { decl: DeclId },
    /// A call through a fn value of fn type `ty`.
    FnValue { ty: TypeId },
    /// Element access (after `derefs` auto-derefs).
    Index { derefs: u32 },
    /// Sub-slice `xs[a..b]` / `xs[a..=b]` of an array or slice (after `derefs` auto-derefs).
    Slice { derefs: u32 },
    /// `old(e)` in an `ensures` clause: the value of `e` at entry
    /// ([`crate::contracts::Contracts::olds`]).
    Old,
}

/// An implicit conversion made explicit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Coercion {
    /// `never` to the expected type.
    NeverToAny,
    /// `&mut T` passed as `&T`.
    MutToShared,
    /// `&mut T` argument to a `&mut T` parameter: `&mut *a`.
    Reborrow,
    /// `&[T; N]` (or `&mut`) to `&[T]`.
    ArrayToSlice,
    /// `&[T align A]` (or `&mut`) to `&[T]`.
    Unalign,
    /// `&String` to `&str` (`String.as_str`).
    StringToStr,
    /// An f-string temporary borrowed as `&str` (`String.as_str`, lives to the end of the statement).
    FmtToStr,
    /// A fn item used as a fn value.
    FnItem,
}

/// A generic instantiation (mono input): decl and canonical args.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Instance {
    pub decl: DeclId,
    pub args: Vec<GArg>,
}

/// One instantiation site.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallSite {
    pub caller: DeclId,
    pub node: u32,
    pub inst: Instance,
}

/// The checked body of one fn, method, const, or `static mut` initializer.
#[derive(Clone, Debug)]
pub struct TypedBody {
    pub decl: DeclId,
    pub item: ItemId,
    /// The decl's node (`FnDecl`, `ConstDecl`, ...).
    pub node: u32,
    /// First node index covered (`node_types[i]` is node `lo + i`).
    pub lo: u32,
    pub node_types: Vec<TypeId>,
    /// Local bindings: (binding node, type), sorted by node.
    pub locals: Vec<(u32, TypeId)>,
    /// Resolved targets keyed by the token of their step (see [`crate::expr`] docs: the name
    /// token of a field, method, or path step, the `(` of a call, the `[` of an index, the `.{`
    /// of a struct literal, the operator token of a user-type operator), sorted by token.
    pub targets: Vec<(u32, Target)>,
    /// The operand type of each `?`, keyed by its token, sorted: a value the step reads whole that
    /// has no node of its own when `?` follows another step.
    pub tries: Vec<(u32, TypeId)>,
    /// Implicit coercions keyed by the node of the coerced expression, sorted.
    pub coercions: Vec<(u32, Coercion)>,
    pub sites: Vec<CallSite>,
    /// Calls the source does not spell, keyed by the token of their construct, sorted: a `for`
    /// loop's `next` (a [`Target::Method`] at the `for` token) and an f-string hole's
    /// `Display.fmt` (a [`Target::TraitMethod`] with the writer type as its own arg, at the
    /// f-string token; one per hole).
    pub implicit: Vec<(u32, Target)>,
    /// The kind of each `for` loop (by its node), decided once from the source's type as written
    /// here ([`crate::loops::ForKind`]), sorted; absent for a loop whose source is no source.
    pub for_kinds: Vec<(u32, crate::loops::ForKind)>,
    /// The pattern bindings that take a value that is not Copy by value: binding nodes, sorted.
    /// Decided on the types of the definition (a type parameter without a `Copy` bound is not
    /// Copy, whatever an instance gives it), so a pattern moves its value in every instance or in
    /// none.
    pub moves: Vec<u32>,
    /// The value types this body may drop, each with the token where such a value appears, sorted
    /// and without duplicates: its locals (params and pattern bindings, at their binding's first
    /// token) and the values it makes (call results, struct and variant literals). A value the
    /// body drops is one of these or a part of one; types that cannot own a `Drop` value (scalars,
    /// references, pointers, fn values) are left out.
    pub drops: Vec<(u32, TypeId)>,
    /// Fn values made (a closure literal, a fn item used as a value) where a fn type with a deny
    /// clause their own type lacks is wanted, so the effect pass checks them on their effects
    /// ([`crate::effects`], E1403): (the closure's `fn` token or the token of the fn item's value
    /// target, the effects that must be absent, the coerced node), sorted by token.
    pub denied: Vec<(u32, u8, u32)>,
    /// Fn items used as values whose fn type, where the value is made, has a deny clause (the
    /// item's own, or that of the fn type it is coerced to there): (the token of the value target,
    /// that clause, the node), sorted by token. A value of such a type is called without proving
    /// the item's `requires`, so the effect pass checks the item's entry effect against the clause
    /// ([`crate::effects`], E1403).
    pub valued: Vec<(u32, u8, u32)>,
    pub has_error: bool,
    /// Class U ops inside `emulate { }`, in source order ([`crate::gates`]).
    pub emulated: Vec<crate::gates::Emulated>,
    /// The fn's `requires` / `ensures` clauses, if it has any.
    pub contracts: Option<Box<crate::contracts::Contracts>>,
    /// A `test fn` with parameters: the domain its inputs come from.
    pub test_domain: Option<Box<crate::testgen::Domain>>,
    /// Where each temporary of the body ends: the one answer checker and lowering read.
    pub stmt_ends: crate::temps::StmtEnds,
    /// Internal errors found while checking the body (const evaluation).
    pub internal: Vec<String>,
}

impl TypedBody {
    /// The type of node `n` (`NONE` for non-values, error, and nodes outside the body).
    pub fn ty(&self, n: u32) -> TypeId {
        n.checked_sub(self.lo).and_then(|i| self.node_types.get(i as usize)).copied().unwrap_or(TypeId::NONE)
    }

    pub fn local(&self, b: BindingRef) -> Option<TypeId> {
        self.locals.binary_search_by_key(&b.node, |x| x.0).ok().map(|i| self.locals[i].1)
    }

    /// The target recorded at token `tok`.
    pub fn target(&self, tok: u32) -> Option<&Target> {
        self.targets.binary_search_by_key(&tok, |x| x.0).ok().map(|i| &self.targets[i].1)
    }

    /// The kind of the `for` loop at node `f` (`None`: its source is no source, an error).
    pub fn for_kind(&self, f: u32) -> Option<crate::loops::ForKind> {
        self.for_kinds.binary_search_by_key(&f, |x| x.0).ok().map(|i| self.for_kinds[i].1)
    }

    /// The implicit call recorded at token `tok` (a `for` loop's `next`).
    pub fn implicit_at(&self, tok: u32) -> Option<&Target> {
        self.implicit.binary_search_by_key(&tok, |x| x.0).ok().map(|i| &self.implicit[i].1)
    }

    /// Whether the pattern binding at node `b` takes a value that is not Copy by value
    /// ([`TypedBody::moves`]).
    pub fn binds_by_move(&self, b: u32) -> bool {
        self.moves.binary_search(&b).is_ok()
    }

    pub fn coercion(&self, n: u32) -> Option<&Coercion> {
        self.coercions.binary_search_by_key(&n, |x| x.0).ok().map(|i| &self.coercions[i].1)
    }
}

/// Wall times of the checker's phases.
#[derive(Clone, Copy, Debug, Default)]
pub struct TypeckTimings {
    pub sigs: Duration,
    pub bodies: Duration,
    pub canon: Duration,
    /// [`crate::callgraph`] and the recursion check on it.
    pub graph: Duration,
}

/// The type-checked program.
#[derive(Debug)]
pub struct Typeck {
    /// Canonical type table (ids = canonical type ids; the error type is last).
    pub types: TypeTable,
    pub sigs: Sigs,
    /// Sorted by decl.
    pub bodies: Vec<TypedBody>,
    /// Unique instances, sorted (canonical).
    pub instances: Vec<Instance>,
    /// The whole-program call graph ([`crate::callgraph`]).
    pub call_graph: CallGraph,
    /// Per generic instance that inherits features from its type arguments: those beyond its
    /// declared `needs` and the target baseline (keys in the call graph's type ids;
    /// [`crate::inherit`]).
    pub inst_features: rustc_hash::FxHashMap<crate::mono::InstKey, u64>,
    /// Per type whose drop glue loads a vector wider than the target baseline holds: the
    /// features the glue is compiled with (keys in the call graph's type ids; [`crate::inherit`]).
    pub glue_features: rustc_hash::FxHashMap<TypeId, u64>,
    /// Type-checking diagnostics (canonical order). [`Typeck::all_diagnostics`] merges resolution's.
    pub diagnostics: Vec<Diagnostic>,
    pub timings: TypeckTimings,
    pub error_type: TypeId,
    /// Internal errors of the checker (const evaluation): the compilation stops with the first.
    pub internal: Vec<String>,
}

impl Typeck {
    pub fn body(&self, d: DeclId) -> Option<&TypedBody> {
        self.bodies.binary_search_by_key(&d, |b| b.decl).ok().map(|i| &self.bodies[i])
    }

    /// Every class U op inside `emulate { }`, sorted by span.
    pub fn emulated(&self) -> Vec<&crate::gates::Emulated> {
        let mut v: Vec<&crate::gates::Emulated> = self.bodies.iter().flat_map(|b| &b.emulated).collect();
        v.sort_by_key(|e| e.span);
        v
    }

    /// Resolution and type diagnostics together, canonical order (duplicates removed).
    pub fn all_diagnostics(&self, prog: &Program) -> Vec<Diagnostic> {
        let mut v = prog.diagnostics.clone();
        v.extend(self.diagnostics.iter().cloned());
        stave_diag::canonical_order(v, &prog.source_map)
    }

    /// Whether fn `d` has a type error (borrowck and lowering skip it).
    pub fn has_error(&self, d: DeclId) -> bool {
        self.body(d).is_none_or(|b| b.has_error)
    }

    /// The checked-IR TypeKey text of type `t`.
    pub fn type_key(&self, t: TypeId) -> String {
        types::type_key_text(&self.types, t)
    }
}

/// Whether a node kind is an expression node (one `types` dump line each, stage-gates).
pub fn is_expr_kind(k: NodeKind) -> bool {
    matches!(
        k,
        NodeKind::IfExpr
            | NodeKind::MatchExpr
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
            | NodeKind::Unary
            | NodeKind::Postfix
            | NodeKind::Primary
    )
}

/// The `types` dump (stage-gates): the type table `t<T> key` (every
/// type a line names, closed over key components, by T), then `I N t<T>` / `I N -` for every
/// expression node of every item not dropped, by (I, N).
pub fn dump_types(prog: &Program, tc: &Typeck) -> String {
    let mut lines: Vec<(u32, u32, TypeId)> = Vec::new();
    let mut by_item: FxHashMap<ItemId, Vec<&TypedBody>> = FxHashMap::default();
    for b in &tc.bodies {
        by_item.entry(b.item).or_default().push(b);
    }
    for (i, it) in prog.items.iter().enumerate() {
        if !it.ok {
            continue;
        }
        let item = ItemId(i as u32);
        let ast = prog.item_ast(item);
        let bodies = by_item.get(&item);
        for n in 0..ast.len() as u32 {
            let Some(k) = ast.kind(n) else { continue };
            if !is_expr_kind(k) {
                continue;
            }
            let t = bodies
                .and_then(|bs| bs.iter().find(|b| n >= b.lo && n <= b.node))
                .map_or(TypeId::NONE, |b| b.ty(n));
            lines.push((i as u32, n, t));
        }
    }
    // the type table: named types, closed over components
    let mut used = vec![false; tc.types.len()];
    let mut stack: Vec<TypeId> = lines.iter().map(|l| l.2).filter(|t| !t.is_none()).collect();
    let mut kids = Vec::new();
    while let Some(t) = stack.pop() {
        let Some(u) = used.get_mut(t.idx()) else { continue };
        if *u {
            continue;
        }
        *u = true;
        kids.clear();
        tc.types.kind(t).children(&mut kids);
        stack.extend_from_slice(&kids);
    }
    let mut out = String::new();
    for (i, &u) in used.iter().enumerate() {
        if u && TypeId(i as u32) != tc.error_type {
            out.push_str(&format!("t{i} {}\n", tc.type_key(TypeId(i as u32))));
        }
    }
    for (i, n, t) in lines {
        if t.is_none() {
            out.push_str(&format!("{i} {n} -\n"));
        } else {
            out.push_str(&format!("{i} {n} t{}\n", t.0));
        }
    }
    out
}

/// Type-checks the program. Runs with the caller's thread count (the current rayon pool's) on
/// big-stack workers, so no nesting can overflow (no panics; see module `pool`).
pub fn check(prog: &Program) -> Typeck {
    crate::pool::on_big_stacks(|| check_in_pool(prog))
}

/// A body to check.
#[derive(Clone, Copy)]
struct Job {
    decl: DeclId,
    item: ItemId,
    node: u32,
    kind: JobKind,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum JobKind {
    Fn,
    Const,
}

fn check_in_pool(prog: &Program) -> Typeck {
    let t0 = Instant::now();
    let brackets: Vec<Brackets> = prog
        .items
        .par_iter()
        .map(|it| {
            match prog.file(it.file).and_then(|f| f.parsed.items.get(it.index as usize).map(|x| (f, x))) {
                Some((f, info)) if it.ok => Brackets::new(&f.parsed.tokens, info.start, info.end),
                _ => Brackets::default(),
            }
        })
        .collect();
    let mut sb = SigBuilder::new(prog, &brackets);
    sb.build_all();
    let mut internal = std::mem::take(&mut sb.internal);
    let (mut tys, sigs_table, mut diags) = sb.finish();
    diags.extend(crate::traits::check_impls(prog, &sigs_table, &mut tys, &brackets));
    diags.extend(crate::builtins::check_intrinsics(prog, &sigs_table, &tys, &brackets));
    diags.extend(crate::ffi::check(prog, &sigs_table, &tys));
    diags.extend(crate::dispatch::check_dispatches(prog, &sigs_table, &mut tys, &brackets));
    let t_sigs = t0.elapsed();

    // Bodies.
    let t1 = Instant::now();
    let target_arch = prog.target.arch();
    let mut jobs: Vec<Job> = Vec::new();
    for (k, d) in prog.decls.iter().enumerate() {
        let decl = DeclId(k as u32);
        let Some(it) = prog.item(d.item) else { continue };
        if !it.ok || d.node == u32::MAX {
            continue;
        }
        let kind = match d.kind {
            DeclKind::Fn | DeclKind::Method => {
                if d.flags.has(DeclFlags::EXTERN_IMPORT) || d.flags.has(DeclFlags::DISPATCH) {
                    continue;
                }
                // A fn needing another arch's features is not compiled on this arch.
                if let Some(sig) = sigs_table.fn_sig(decl)
                    && let Some(fs) = stave_target::FeatureSet::from_bits(sig.features)
                    && !fs.is_empty()
                    && !fs.is_owned_by(target_arch)
                {
                    continue;
                }
                JobKind::Fn
            }
            DeclKind::Const | DeclKind::Static => JobKind::Const,
            _ => continue,
        };
        jobs.push(Job { decl, item: d.item, node: d.node, kind });
    }
    let w = World { prog, sigs: &sigs_table };
    let results: Vec<(TypedBody, Vec<TypeKind>, Vec<Diagnostic>)> = jobs
        .par_iter()
        .map(|j| {
            let br = &brackets[j.item.idx()];
            match ItemCx::new(prog, j.item, br) {
                Some(icx) => check_body(w, &tys, icx, *j),
                None => (empty_body(*j), Vec::new(), Vec::new()),
            }
        })
        .collect();
    let t_bodies = t1.elapsed();

    // Merge overlays (decl order), then canonical ids.
    let t2 = Instant::now();
    let mut bodies = Vec::with_capacity(results.len());
    for (mut b, local, ds) in results {
        diags.extend(ds);
        internal.append(&mut b.internal);
        if !local.is_empty() {
            let m = tys.merge(&local);
            remap_body(&mut b, &|t: TypeId| {
                if t.is_local() { m.get(t.idx()).copied().unwrap_or(ids::ERROR) } else { t }
            });
        }
        bodies.push(b);
    }
    let remap = tys.canonicalize();
    let err = tys.error_ty();
    let map = |t: TypeId| -> TypeId {
        if t.is_none() {
            return t;
        }
        let n = remap.get(t.idx()).copied().unwrap_or(err);
        if n == err { TypeId::NONE } else { n }
    };
    let map_sig =
        |t: TypeId| -> TypeId { if t.is_none() { t } else { remap.get(t.idx()).copied().unwrap_or(err) } };
    bodies.par_iter_mut().for_each(|b| remap_body(b, &map));
    let mut sigs_table = sigs_table;
    remap_sigs(&mut sigs_table, &map_sig);
    let gated = crate::inherit::gate_bodies(prog, &sigs_table, &tys, &bodies, &brackets);
    diags.extend(gated.diagnostics);
    let mut instances: Vec<Instance> =
        bodies.iter().flat_map(|b| b.sites.iter().map(|s| s.inst.clone())).collect();
    instances.sort();
    instances.dedup();
    let deep = crate::mono::depth_check(prog, &sigs_table, &tys, &bodies, &brackets);
    // polymorphic recursion (E0902) has no finite instance graph
    let too_deep = !deep.is_empty();
    diags.extend(deep);
    let t_canon = t2.elapsed();
    let t3 = Instant::now();
    let call_graph = if too_deep {
        CallGraph { truncated: true, ..CallGraph::default() }
    } else {
        crate::callgraph::build(prog, &sigs_table, &tys, &bodies, &brackets)
    };
    let mut inst_features = rustc_hash::FxHashMap::default();
    let mut glue_features = rustc_hash::FxHashMap::default();
    if !call_graph.truncated {
        diags.extend(crate::recursion::check(prog, &sigs_table, &tys, &call_graph));
        let inh =
            crate::inherit::check(prog, &sigs_table, &tys, &bodies, &call_graph, &brackets, &gated.needs);
        diags.extend(inh.diagnostics);
        inst_features = inh.features;
        glue_features = inh.glue;
    } else if !too_deep && let Some((span, what)) = call_graph.truncated_at {
        // Every check over the graph (recursion, effects, ranks, contract purity, stack) would
        // miss the left-out instances: the program is rejected.
        let generic = what.map_or_else(|| "drop glue".to_string(), |d| prog.decl_path(d));
        diags.extend(mk("E0902", Some("graph"), &[("generic", &generic)], span, &[]));
    }
    diags.extend(crate::typedepth::check(prog, &sigs_table, &tys, &bodies, &call_graph, &brackets));
    let t_graph = t3.elapsed();
    let diagnostics = stave_diag::canonical_order(diags, &prog.source_map);
    Typeck {
        types: tys,
        sigs: sigs_table,
        bodies,
        instances,
        call_graph,
        inst_features,
        glue_features,
        diagnostics,
        timings: TypeckTimings { sigs: t_sigs, bodies: t_bodies, canon: t_canon, graph: t_graph },
        error_type: err,
        internal,
    }
}

fn empty_body(j: Job) -> TypedBody {
    TypedBody {
        decl: j.decl,
        item: j.item,
        node: j.node,
        lo: j.node,
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
        internal: Vec::new(),
    }
}

fn map_garg(a: &mut GArg, f: &dyn Fn(TypeId) -> TypeId) {
    if let GArg::Type(t) = a {
        *t = f(*t);
    }
}

fn map_gargs(v: &mut [GArg], f: &dyn Fn(TypeId) -> TypeId) {
    for a in v {
        map_garg(a, f);
    }
}

fn map_method(m: &mut MethodRef, f: &dyn Fn(TypeId) -> TypeId) {
    match m {
        MethodRef::Impl { impl_args, .. } => map_gargs(impl_args, f),
        MethodRef::Bound { trait_args, self_ty, .. } => {
            map_gargs(trait_args, f);
            *self_ty = f(*self_ty);
        }
        MethodRef::Builtin(_) => {}
    }
}

fn remap_body(b: &mut TypedBody, f: &dyn Fn(TypeId) -> TypeId) {
    for t in &mut b.node_types {
        *t = f(*t);
    }
    for (_, t) in &mut b.locals {
        *t = f(*t);
    }
    for (_, t) in b.targets.iter_mut().chain(b.implicit.iter_mut()) {
        match t {
            Target::Fn { args, .. } => map_gargs(args, f),
            Target::Method { method, args, .. } | Target::Op { method, args } => {
                map_method(method, f);
                map_gargs(args, f);
            }
            Target::TraitMethod { self_ty, args, .. } => {
                *self_ty = f(*self_ty);
                map_gargs(args, f);
            }
            Target::Builtin { recv, ret, .. } => {
                *recv = f(*recv);
                *ret = f(*ret);
            }
            Target::SizeOf { ty, .. }
            | Target::FnValue { ty }
            | Target::SoaRow { ty, .. }
            | Target::Variant { ty, .. }
            | Target::Struct { ty, .. } => *ty = f(*ty),
            _ => {}
        }
    }
    for (_, t) in &mut b.tries {
        *t = f(*t);
    }
    for s in &mut b.sites {
        map_gargs(&mut s.inst.args, f);
    }
    for (_, t) in &mut b.drops {
        *t = f(*t);
    }
}

fn remap_generics(g: &mut [GenericParam], f: &dyn Fn(TypeId) -> TypeId) {
    for p in g {
        p.const_ty = f(p.const_ty);
        for b in &mut p.bounds {
            map_gargs(&mut b.args, f);
        }
    }
}

fn remap_sigs(s: &mut Sigs, f: &dyn Fn(TypeId) -> TypeId) {
    for d in &mut s.decls {
        match d {
            DeclSig::Fn(x) => {
                remap_generics(&mut x.generics, f);
                for t in &mut x.params {
                    *t = f(*t);
                }
                x.ret = f(x.ret);
            }
            DeclSig::Adt(a) => {
                remap_generics(&mut a.generics, f);
                for v in &mut a.variants {
                    for fd in &mut v.fields {
                        fd.ty = f(fd.ty);
                    }
                }
            }
            DeclSig::Impl(i) => {
                remap_generics(&mut i.generics, f);
                i.self_ty = f(i.self_ty);
                if let Some(t) = &mut i.trait_ref {
                    map_gargs(&mut t.args, f);
                }
            }
            DeclSig::Trait(t) => remap_generics(&mut t.generics, f),
            DeclSig::Const(c) | DeclSig::Static(c) => {
                c.ty = f(c.ty);
                // the checker compared them before the renumbering
                c.types = Vec::new();
            }
            DeclSig::Alias(a) => {
                remap_generics(&mut a.generics, f);
                a.target = f(a.target);
            }
            DeclSig::None => {}
        }
    }
}

/// A pending-literal-aware type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Ty {
    T(TypeId),
    /// An unsuffixed literal group still without a type.
    Lit(u32),
}

impl Ty {
    pub const ERR: Ty = Ty::T(ids::ERROR);
    pub const NONE: Ty = Ty::T(TypeId::NONE);
}

#[derive(Clone, Debug)]
pub(crate) struct LitMember {
    pub node: u32,
    pub mag: u128,
    pub neg: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum PendingCheck {
    /// Unary `-` on the group's value: the type must be signed or a float.
    Neg(u32),
    /// Constant shift amount: must be below the bit width.
    Shift(u32, u128),
}

#[derive(Clone, Debug)]
pub(crate) struct LitGroup {
    pub parent: u32,
    pub float: bool,
    pub fixed: Option<TypeId>,
    pub members: Vec<LitMember>,
    pub checks: Vec<PendingCheck>,
}

/// Loop context (break targets).
#[derive(Clone, Copy, Debug)]
pub(crate) struct LoopCx {
    pub has_break: bool,
}

/// Return context (fn body or closure body).
#[derive(Clone, Debug)]
pub(crate) struct RetCx {
    pub ty: TypeId,
    pub span: Option<Span>,
    /// Loops of this context.
    pub loops: Vec<LoopCx>,
    pub is_never: bool,
}

/// The per-body checker.
pub(crate) struct Ck<'a> {
    pub w: World<'a>,
    pub prog: &'a Program,
    pub sigs: &'a Sigs,
    pub icx: ItemCx<'a>,
    pub tys: Overlay<'a>,
    pub decl: DeclId,
    pub generics: &'a [GenericParam],
    pub n_parent: u32,
    pub self_ty: Option<TypeId>,
    pub features: u64,
    pub lo: u32,
    pub node_ty: Vec<Ty>,
    pub locals: FxHashMap<u32, TypeId>,
    pub lits: Vec<LitGroup>,
    pub rets: Vec<RetCx>,
    pub unsafe_depth: u32,
    pub scope_depth: u32,
    /// Inside a closure literal that is (transitively) inside a `scope` block: its `fn` token
    /// (a `spawn` there is E0808, not E0801).
    pub closure_in_scope: Option<u32>,
    pub diags: Vec<Diagnostic>,
    pub has_error: bool,
    pub targets: Vec<(u32, Target)>,
    /// The operand type of each `?` (by its token).
    pub tries: Vec<(u32, TypeId)>,
    pub coercions: Vec<(u32, Coercion)>,
    pub sites: Vec<CallSite>,
    pub implicit: Vec<(u32, Target)>,
    pub for_kinds: Vec<(u32, crate::loops::ForKind)>,
    /// `c.select(a, b)` on a `bool`: the token of its target and the type of its operands, which
    /// is its result type once the literals among them are fixed.
    pub selects: Vec<(u32, Ty)>,
    /// [`TypedBody::moves`], unsorted.
    pub moves: Vec<u32>,
    pub drops: Vec<(u32, TypeId)>,
    pub denied: Vec<(u32, u8, u32)>,
    pub valued: Vec<(u32, u8, u32)>,
    pub depth: u32,
    /// E0005 case `expr` was reported for this body.
    pub depth_reported: bool,
    /// The current statement always calls a `-> never` fn (path rule).
    pub diverges: bool,
    /// The current position is reachable (no earlier diverging statement).
    pub reachable: bool,
    pub kids_buf: Vec<Vec<u32>>,
    /// While the alternatives of an or-pattern are typed: the bindings of the current
    /// alternative (binding node, type, mode), in source order.
    pub or_binds: Option<Vec<(u32, TypeId, crate::patterns::Mode)>>,
    pub gates: crate::gates::Gates,
    /// Inside a contract clause: which kind, and the `old(..)` nesting.
    pub contract: Option<crate::contracts::ClauseCx>,
    /// The `old(e)` calls typed so far: (`(` token, `e`'s node).
    pub olds: Vec<(u32, u32)>,
    pub contracts: Option<Box<crate::contracts::Contracts>>,
    /// The input domain of a `test fn` with parameters ([`crate::testgen`]).
    pub domain: Option<Box<crate::testgen::Domain>>,
    /// [`TypedBody::internal`].
    pub internal: Vec<String>,
}

fn check_body(
    w: World<'_>,
    global: &TypeTable,
    icx: ItemCx<'_>,
    j: Job,
) -> (TypedBody, Vec<TypeKind>, Vec<Diagnostic>) {
    let prog = w.prog;
    let sigs = w.sigs;
    let lo = icx.ast.descendants(j.node).start;
    let n_nodes = (j.node + 1).saturating_sub(lo) as usize;
    let decl = prog.decl(j.decl).copied();
    let (generics, n_parent, self_ty, features): (&[GenericParam], u32, Option<TypeId>, u64) =
        match sigs.fn_sig(j.decl) {
            Some(s) => {
                let self_ty = match decl {
                    Some(d) if d.kind == DeclKind::Method => sigs.impl_def(d.parent).map(|i| i.self_ty),
                    _ => None,
                };
                (&s.generics[..], s.n_parent, self_ty, s.features)
            }
            None => (&[][..], 0, None, 0),
        };
    let mut ck = Ck {
        w,
        prog,
        sigs,
        icx,
        tys: Overlay::new(global),
        decl: j.decl,
        generics,
        n_parent,
        self_ty,
        features,
        lo,
        node_ty: vec![Ty::NONE; n_nodes],
        locals: FxHashMap::default(),
        lits: Vec::new(),
        rets: Vec::new(),
        unsafe_depth: 0,
        scope_depth: 0,
        closure_in_scope: None,
        diags: Vec::new(),
        has_error: false,
        targets: Vec::new(),
        tries: Vec::new(),
        coercions: Vec::new(),
        sites: Vec::new(),
        implicit: Vec::new(),
        for_kinds: Vec::new(),
        selects: Vec::new(),
        moves: Vec::new(),
        drops: Vec::new(),
        denied: Vec::new(),
        valued: Vec::new(),
        depth: 0,
        depth_reported: false,
        diverges: false,
        reachable: true,
        kids_buf: Vec::new(),
        or_binds: None,
        gates: crate::gates::Gates::default(),
        contract: None,
        olds: Vec::new(),
        contracts: None,
        domain: None,
        internal: Vec::new(),
    };
    match j.kind {
        JobKind::Fn => {
            if let Some(sig) = sigs.fn_sig(j.decl) {
                ck.fn_body(sig, j.node);
            }
        }
        JobKind::Const => ck.const_body(j.node),
    }
    ck.finish(j)
}

impl<'a> Ck<'a> {
    // ---------- helpers ----------

    pub fn names(&self) -> NameCx<'a> {
        NameCx { prog: self.prog, generics: self.generics }
    }

    pub fn show(&self, t: TypeId) -> String {
        types::display(&self.tys, &self.names(), t)
    }

    pub fn show_ty(&self, t: Ty) -> String {
        match t {
            Ty::T(t) => self.show(t),
            Ty::Lit(g) => {
                if self.lit_root(g).1.float {
                    "{float}".to_string()
                } else {
                    "{integer}".to_string()
                }
            }
        }
    }

    pub fn report(&mut self, d: Option<Diagnostic>) {
        if d.is_some() {
            self.has_error = true;
        }
        self.diags.extend(d);
    }

    pub fn set(&mut self, n: u32, t: Ty) {
        if let Some(slot) = n.checked_sub(self.lo).and_then(|i| self.node_ty.get_mut(i as usize)) {
            *slot = t;
        }
        if t == Ty::ERR {
            self.has_error = true;
        }
    }

    pub fn target(&mut self, n: u32, t: Target) {
        self.targets.push((n, t));
    }

    pub fn coerce_note(&mut self, n: u32, c: Coercion) {
        self.coercions.push((n, c));
    }

    pub fn kids(&mut self, n: u32) -> Vec<u32> {
        let mut v = self.kids_buf.pop().unwrap_or_default();
        self.icx.kids(n, &mut v);
        v
    }

    pub fn free_kids(&mut self, v: Vec<u32>) {
        if self.kids_buf.len() < 64 {
            self.kids_buf.push(v);
        }
    }

    pub fn env(&self) -> TyEnv<'a> {
        TyEnv { icx: self.icx, n_parent: self.n_parent, self_ty: self.self_ty }
    }

    pub fn lower_ty(&mut self, n: u32, ret_pos: bool) -> TypeId {
        let env = self.env();
        let t = sigs::lower_type(self, &env, n, ret_pos);
        if t == ids::ERROR {
            self.has_error = true;
        }
        t
    }

    /// Generic-param environment of this body.
    pub fn genv(&self) -> &'a [GenericParam] {
        self.generics
    }

    // ---------- literal groups ----------

    pub fn new_lit(&mut self, node: u32, mag: u128, neg: bool, float: bool) -> u32 {
        let g = self.lits.len() as u32;
        self.lits.push(LitGroup {
            parent: g,
            float,
            fixed: None,
            members: vec![LitMember { node, mag, neg }],
            checks: Vec::new(),
        });
        g
    }

    pub fn find(&mut self, g: u32) -> u32 {
        let mut r = g;
        for _ in 0..1_000_000 {
            let p = self.lits.get(r as usize).map_or(r, |x| x.parent);
            if p == r {
                break;
            }
            r = p;
        }
        // path compression
        let mut x = g;
        for _ in 0..1_000_000 {
            let p = self.lits.get(x as usize).map_or(x, |l| l.parent);
            if p == x {
                break;
            }
            if let Some(l) = self.lits.get_mut(x as usize) {
                l.parent = r;
            }
            x = p;
        }
        r
    }

    pub fn lit_root(&self, g: u32) -> (u32, &LitGroup) {
        let mut r = g;
        for _ in 0..1_000_000 {
            let p = self.lits[r as usize].parent;
            if p == r {
                break;
            }
            r = p;
        }
        (r, &self.lits[r as usize])
    }

    /// Resolves a `Ty` whose literal group is fixed.
    pub fn resolve(&mut self, t: Ty) -> Ty {
        match t {
            Ty::Lit(g) => {
                let r = self.find(g);
                match self.lits[r as usize].fixed {
                    Some(t) => Ty::T(t),
                    None => Ty::Lit(r),
                }
            }
            t => t,
        }
    }

    pub fn add_check(&mut self, g: u32, c: PendingCheck) {
        let r = self.find(g);
        if let Some(t) = self.lits[r as usize].fixed {
            self.run_check(c, t);
        } else {
            self.lits[r as usize].checks.push(c);
        }
    }

    /// Unites two pending groups (`false`: int vs float).
    pub fn unite(&mut self, a: u32, b: u32) -> bool {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return true;
        }
        if self.lits[ra as usize].float != self.lits[rb as usize].float {
            return false;
        }
        let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
        let members = std::mem::take(&mut self.lits[hi as usize].members);
        let checks = std::mem::take(&mut self.lits[hi as usize].checks);
        self.lits[hi as usize].parent = lo;
        self.lits[lo as usize].members.extend(members);
        self.lits[lo as usize].checks.extend(checks);
        true
    }

    /// Fixes group `g` to type `t`. `false`: `t` is not a type of that literal kind (the caller
    /// reports E0201); the group is then fixed to the error type.
    pub fn fix(&mut self, g: u32, t: TypeId) -> bool {
        let r = self.find(g);
        if let Some(prev) = self.lits[r as usize].fixed {
            return prev == t || prev == ids::ERROR || t == ids::ERROR;
        }
        let float = self.lits[r as usize].float;
        let ok = t == ids::ERROR || (if float { self.tys.is_float(t) } else { self.tys.int(t).is_some() });
        let t = if ok { t } else { ids::ERROR };
        self.lits[r as usize].fixed = Some(t);
        if t == ids::ERROR {
            return ok;
        }
        let members = self.lits[r as usize].members.clone();
        for m in members {
            self.range_check(&m, t, float);
        }
        let checks = std::mem::take(&mut self.lits[r as usize].checks);
        for c in checks {
            self.run_check(c, t);
        }
        true
    }

    fn range_check(&mut self, m: &LitMember, t: TypeId, float: bool) {
        if float {
            // value re-parsed from the token
            let text = self.icx.node_text(m.node).to_string();
            let body = text.trim_start_matches('-').trim();
            let fits = crate::literals::parse_float(body)
                .map(|(v, _)| crate::literals::float_fits(v, t))
                .unwrap_or(true);
            if !fits {
                let tn = self.show(t);
                let d = mk(
                    "E0203",
                    Some("float"),
                    &[("literal", &text), ("type", &tn)],
                    self.icx.node_span(m.node),
                    &[],
                );
                self.report(d);
            }
            return;
        }
        if !crate::literals::int_fits(m.mag, m.neg, t) {
            let text = self.icx.node_text(m.node).to_string();
            let tn = self.show(t);
            let (lo, hi) = crate::literals::int_range(t).unwrap_or((0, 0));
            let d = mk(
                "E0203",
                None,
                &[("literal", &text), ("type", &tn), ("min", &lo.to_string()), ("max", &hi.to_string())],
                self.icx.node_span(m.node),
                &[],
            );
            self.report(d);
        }
    }

    pub fn run_check(&mut self, c: PendingCheck, t: TypeId) {
        if t == ids::ERROR {
            return;
        }
        match c {
            PendingCheck::Neg(n) => {
                if matches!(self.tys.int(t), Some((false, _))) {
                    let tn = self.show(t);
                    let d = mk(
                        "E0206",
                        None,
                        &[("op", "-"), ("type", &tn)],
                        self.icx.tok_span(self.icx.a(n)),
                        &[],
                    );
                    self.report(d);
                }
            }
            PendingCheck::Shift(n, amount) => {
                if let Some((_, bits)) = self.tys.int(t) {
                    self.e0205(n, amount, bits.bits(), t);
                }
            }
        }
    }

    pub fn e0205(&mut self, n: u32, amount: u128, bits: u32, t: TypeId) {
        if amount >= bits as u128 {
            let tn = self.show(t);
            let d = mk(
                "E0205",
                None,
                &[
                    ("amount", &amount.min(u64::MAX as u128).to_string()),
                    ("bits", &bits.to_string()),
                    ("type", &tn),
                ],
                self.icx.node_span(n),
                &[],
            );
            self.report(d);
        }
    }

    /// Forces a concrete type: a pending literal here has no context (E0202).
    pub fn force(&mut self, t: Ty) -> TypeId {
        match self.resolve(t) {
            Ty::T(t) => t,
            Ty::Lit(g) => {
                self.e0202(g);
                ids::ERROR
            }
        }
    }

    pub fn e0202(&mut self, g: u32) {
        let r = self.find(g);
        let grp = &self.lits[r as usize];
        if grp.fixed.is_some() {
            return;
        }
        let float = grp.float;
        let first = grp.members.iter().map(|m| (self.icx.a(m.node), m.node)).min().map(|x| x.1);
        self.lits[r as usize].fixed = Some(ids::ERROR);
        if let Some(n) = first {
            let text = self.icx.node_text(n).to_string();
            let d = mk(
                "E0202",
                if float { Some("float") } else { None },
                &[("literal", &text)],
                self.icx.node_span(n),
                &[],
            );
            self.report(d);
        }
    }

    // ---------- coercion ----------

    /// Checks that the value of node `n` (type `got`) can be used where `want` is required,
    /// recording implicit coercions. `why`: the span requiring `want` (E0201 related label).
    pub fn coerce(&mut self, n: u32, got: Ty, want: TypeId, why: Option<Span>) -> TypeId {
        let got = self.resolve(got);
        if want == ids::ERROR || want.is_none() {
            if let Ty::Lit(g) = got {
                let _ = self.fix(g, ids::ERROR);
            }
            return want;
        }
        match got {
            Ty::Lit(g) => {
                if self.fix(g, want) {
                    self.set_lit_nodes(n, want);
                    return want;
                }
                self.mismatch(n, got, want, why);
                ids::ERROR
            }
            Ty::T(t) => {
                if t == want || t == ids::ERROR || self.tys.has_error(t) {
                    return want;
                }
                if t.is_none() {
                    return ids::ERROR;
                }
                if t == ids::NEVER {
                    self.coerce_note(n, Coercion::NeverToAny);
                    return want;
                }
                if want == ids::REF_STR && self.is_fstring_string(n, t) {
                    self.coerce_note(n, Coercion::FmtToStr);
                    return want;
                }
                if let Some(c) = self.coercible(t, want) {
                    if c == Coercion::FnItem {
                        self.fn_value_as(n, want);
                    }
                    self.coerce_note(n, c);
                    return want;
                }
                if self.tys.has_error(want) {
                    return want;
                }
                if let Some(widened) = self.deny_widened(t, want) {
                    if let Some(tok) = self.made_fn_value(n) {
                        self.fn_value_as(n, want);
                        self.denied.push((tok, widened, n));
                        self.coerce_note(n, Coercion::FnItem);
                        return want;
                    }
                    let (from, to) = (self.show(t), self.show(want));
                    let effect = crate::effects::Effects(widened).names()[0];
                    let holes = [("from", from.as_str()), ("to", to.as_str()), ("effect", effect)];
                    let d = mk("E1403", None, &holes, self.icx.node_span(n), &[]);
                    self.report(d);
                    return ids::ERROR;
                }
                let drops_features = matches!(
                    (self.tys.kind(t), self.tys.kind(want)),
                    (TypeKind::Fn { params: p1, ret: r1, .. }, TypeKind::Fn { params: p2, ret: r2, .. })
                        if p1 == p2 && r1 == r2
                );
                if drops_features {
                    let (from, to) = (self.show(t), self.show(want));
                    let d = mk("E0707", None, &[("from", &from), ("to", &to)], self.icx.node_span(n), &[]);
                    self.report(d);
                    return ids::ERROR;
                }
                self.mismatch(n, got, want, why);
                ids::ERROR
            }
        }
    }

    /// The effects fn type `want` denies beyond those fn type `t` denies, when that is the only
    /// reason `t` is not coercible to `want` (equal params and return type, features allowed).
    fn deny_widened(&self, t: TypeId, want: TypeId) -> Option<u8> {
        let (
            TypeKind::Fn { params: p1, ret: r1, features: f1, deny: d1 },
            TypeKind::Fn { params: p2, ret: r2, features: f2, deny: d2 },
        ) = (self.tys.kind(t), self.tys.kind(want))
        else {
            return None;
        };
        let widened = d2 & !d1;
        (p1 == p2 && r1 == r2 && (f1 & !(f2 | self.proven_features())) == 0 && widened != 0)
            .then_some(widened)
    }

    /// A fn item made into a value by node `n` is coerced to fn type `want` right there: its
    /// value has the deny clause of `want` from then on ([`TypedBody::valued`]).
    fn fn_value_as(&mut self, n: u32, want: TypeId) {
        let (Some(tok), TypeKind::Fn { deny, .. }) = (self.made_fn_value(n), self.tys.kind(want)) else {
            return;
        };
        let deny = *deny;
        for v in self.valued.iter_mut().filter(|v| v.0 == tok) {
            v.1 = deny;
        }
    }

    /// The token a fn value made by node `n` is known by: a closure literal's `fn` token, or the
    /// token of a fn item's value target (`None` for any other expression).
    fn made_fn_value(&self, n: u32) -> Option<u32> {
        if crate::closures::closure_kind(&self.icx, n) == Some(crate::closures::ClosureKind::Literal) {
            return Some(self.icx.a(n));
        }
        let end = self.icx.end_tok(n);
        self.targets.iter().rev().any(|(tok, t)| *tok == end && matches!(t, Target::Fn { .. })).then_some(end)
    }

    /// The features every value made in this body may assume: its `needs` and the target
    /// baseline.
    pub fn proven_features(&self) -> u64 {
        self.features | self.prog.target.baseline().bits()
    }

    /// Re-types a node chain whose value was a pending literal (the group is resolved at the
    /// end).
    fn set_lit_nodes(&mut self, _n: u32, _t: TypeId) {}

    /// Whether node `n` is an f-string (through parentheses) of type `t` = `std.string.String`.
    fn is_fstring_string(&self, n: u32, t: TypeId) -> bool {
        let s = self.sigs.lang.string;
        if s.is_none() || !matches!(self.tys.kind(t), TypeKind::Adt { def, .. } if *def == s) {
            return false;
        }
        self.fstring_node(n)
    }

    /// An implicit coercion from `t` to `want`, if one applies.
    pub fn coercible(&mut self, t: TypeId, want: TypeId) -> Option<Coercion> {
        let (tk, wk) = (self.tys.kind(t).clone(), self.tys.kind(want).clone());
        match (&tk, &wk) {
            (TypeKind::Ref { mutable: tm, inner: ti }, TypeKind::Ref { mutable: false, inner: wi }) => {
                if ti == wi && *tm {
                    return Some(Coercion::MutToShared);
                }
                // &[T; N] -> &[T]
                if let (TypeKind::Array { elem, .. }, TypeKind::Slice { elem: we, align: 0 }) =
                    (self.tys.kind(*ti), self.tys.kind(*wi))
                    && elem == we
                {
                    return Some(Coercion::ArrayToSlice);
                }
                // &[T align A] -> &[T]
                if let (TypeKind::Slice { elem, align }, TypeKind::Slice { elem: we, align: 0 }) =
                    (self.tys.kind(*ti), self.tys.kind(*wi))
                    && elem == we
                    && *align != 0
                {
                    return Some(Coercion::Unalign);
                }
                // &String -> &str
                if *wi == ids::STR
                    && let TypeKind::Adt { def, .. } = self.tys.kind(*ti)
                    && *def == self.sigs.lang.string
                    && def.is_some()
                {
                    return Some(Coercion::StringToStr);
                }
                None
            }
            (TypeKind::Ref { mutable: true, inner: ti }, TypeKind::Ref { mutable: true, inner: wi }) => {
                if let (TypeKind::Array { elem, .. }, TypeKind::Slice { elem: we, align: 0 }) =
                    (self.tys.kind(*ti), self.tys.kind(*wi))
                    && elem == we
                {
                    return Some(Coercion::ArrayToSlice);
                }
                if let (TypeKind::Slice { elem, align }, TypeKind::Slice { elem: we, align: 0 }) =
                    (self.tys.kind(*ti), self.tys.kind(*wi))
                    && elem == we
                    && *align != 0
                {
                    return Some(Coercion::Unalign);
                }
                None
            }
            (
                TypeKind::Fn { params: p1, ret: r1, features: f1, deny: d1 },
                TypeKind::Fn { params: p2, ret: r2, features: f2, deny: d2 },
            ) => {
                // Features may drop to those proven where the coercion happens: CPU features
                // never change while a program runs, so the value stays callable (else E0707).
                // The deny clause may only shrink: the value keeps every guarantee the target
                // type states (else E1403).
                if p1 == p2 && r1 == r2 && (f1 & !(f2 | self.proven_features())) == 0 && (d2 & !d1) == 0 {
                    return Some(Coercion::FnItem);
                }
                None
            }
            _ => None,
        }
    }

    /// E0201 at node `n`.
    pub fn mismatch(&mut self, n: u32, got: Ty, want: TypeId, why: Option<Span>) {
        if let Ty::T(t) = got
            && (t == ids::ERROR || self.tys.has_error(t))
        {
            return;
        }
        if self.tys.has_error(want) {
            return;
        }
        let expected = self.show(want);
        let found = self.show_ty(got);
        let operand = self.icx.node_text(n).to_string();
        let roles: Vec<(&str, Span)> = why.map(|s| ("expected", s)).into_iter().collect();
        let d = mk(
            "E0201",
            None,
            &[("expected", &expected), ("found", &found), ("operand", &operand)],
            self.icx.node_span(n),
            &roles,
        );
        self.report(d);
    }

    /// Checks node `n` against `want` (hint + coercion).
    pub fn check_expr(&mut self, n: u32, want: TypeId, why: Option<Span>) -> TypeId {
        let t = self.expr(n, Some(want));
        self.coerce(n, t, want, why)
    }

    // ---------- bodies ----------

    fn fn_body(&mut self, sig: &'a FnSig, n: u32) {
        let icx = self.icx;
        // parameters
        let mut params = crate::itemcx::Kids::default();
        if let Some(p) = icx.child(n, NodeKind::Params) {
            params = icx.kids_vec(p);
        }
        for (i, &p) in params.iter().enumerate() {
            let t = sig.params.get(i).copied().unwrap_or(ids::ERROR);
            if let Some(sp) = icx.child(p, NodeKind::SelfParam) {
                self.locals.insert(sp, t);
            }
            self.locals.insert(p, t);
        }
        let Some(block) = icx.child(n, NodeKind::Block) else { return };
        let ret = sig.ret;
        self.rets.push(RetCx { ty: ret, span: sig.ret_span, loops: Vec::new(), is_never: ret == ids::NEVER });
        self.contract_clauses(n, ret);
        self.reachable = true;
        let div = self.block(block);
        self.rets.pop();
        self.end_of_body(block, div, ret);
        self.contract_rules(n);
        self.test_domain(sig, n);
    }

    /// E0207 at the end of a body that can reach it.
    pub fn end_of_body(&mut self, block: u32, diverges: bool, ret: TypeId) {
        if diverges || ret == ids::UNIT || ret == ids::ERROR {
            return;
        }
        let close = self.icx.close(self.icx.a(block));
        let name = self.prog.decl(self.decl).map_or("_", |d| self.prog.strings.get(d.name)).to_string();
        let name = if name.is_empty() { "_".to_string() } else { name };
        if ret == ids::NEVER {
            let d = mk("E0207", Some("never"), &[("name", &name)], self.icx.tok_span(close), &[]);
            self.report(d);
            return;
        }
        let tn = self.show(ret);
        let d = mk("E0207", None, &[("name", &name), ("type", &tn)], self.icx.tok_span(close), &[]);
        self.report(d);
    }

    fn const_body(&mut self, n: u32) {
        let icx = self.icx;
        let Some(c) = self.sigs.konst(self.decl) else { return };
        let ty = c.ty;
        let kids = icx.kids_vec(n);
        let Some(&e) = kids.iter().find(|&&c| icx.is_expr(c)) else { return };
        self.rets.push(RetCx { ty: ids::ERROR, span: None, loops: Vec::new(), is_never: false });
        let tspan = kids.iter().find(|&&c| icx.kind(c) == Some(NodeKind::Type)).map(|&t| icx.node_span(t));
        self.check_expr(e, ty, tspan);
        self.rets.pop();
        self.static_rules(n, ty, tspan);
    }

    /// The rules of a `static` item: an immutable one is shared by every thread, so its type
    /// must be Sync (E0417); `static mut` belongs to the built-in packages only (E0809).
    fn static_rules(&mut self, n: u32, ty: TypeId, tspan: Option<Span>) {
        let icx = self.icx;
        let Some(decl) = self.prog.decl(self.decl).copied() else { return };
        if decl.kind != DeclKind::Static {
            return;
        }
        if decl.flags.has(DeclFlags::MUT) {
            let pkg = self.prog.files.get(icx.fid().0 as usize).map(|f| f.package);
            let builtin = pkg
                .and_then(|p| self.prog.packages.get(p.idx()))
                .is_some_and(|p| p.kind == crate::package::PackageKind::Builtin);
            if !builtin {
                let a = icx.a(n);
                let name = self.prog.strings.get(decl.name).to_string();
                let d = mk("E0809", None, &[("name", &name)], icx.range_span(a, a + 1), &[]);
                self.report(d);
            }
        } else if ty != ids::ERROR
            && !self.tys.has_error(ty)
            && !crate::kinds::is_view_concrete(self.prog, &self.tys, ty)
        {
            // (a reference or view type is E0413, from the signature phase)
            let sync = sigs::TraitRef { trait_decl: self.sigs.lang.sync, args: Vec::new() };
            if sync.trait_decl.is_some() && !self.w.implements(&mut self.tys, ty, &sync, &[]) {
                let tn = self.show(ty);
                let d = mk("E0417", None, &[("type", &tn)], tspan.unwrap_or(decl.name_span), &[]);
                self.report(d);
            }
        }
    }

    /// Records that the body may own (and drop) a value of type `t`, at token `tok`.
    pub fn owned(&mut self, tok: u32, t: TypeId) {
        if !t.is_none()
            && matches!(
                self.tys.kind(t),
                TypeKind::Adt { .. } | TypeKind::Tuple(_) | TypeKind::Array { .. } | TypeKind::Param(_)
            )
            && !self.tys.has_error(t)
        {
            self.drops.push((tok, t));
        }
    }

    /// [`TypedBody::drops`]: the locals and the call results (literals are recorded where they
    /// are checked).
    fn record_drops(&mut self) {
        let icx = self.icx;
        let locals: Vec<(u32, TypeId)> = self.locals.iter().map(|(&b, &t)| (icx.a(b), t)).collect();
        for (tok, t) in locals {
            self.owned(tok, t);
        }
        let targets = std::mem::take(&mut self.targets);
        let implicit = std::mem::take(&mut self.implicit);
        for (tok, t) in targets.iter().chain(implicit.iter()) {
            if let Some(r) = self.call_result(*tok, t) {
                self.owned(*tok, r);
            }
        }
        self.targets = targets;
        self.implicit = implicit;
        self.drops.sort_unstable();
        self.drops.dedup();
    }

    /// The result type of the call recorded as `t` at `tok`, if it is a call.
    fn call_result(&mut self, tok: u32, t: &Target) -> Option<TypeId> {
        let sigs = self.sigs;
        match t {
            Target::Fn { decl, args } if self.icx.is(tok, Term::LParen) => {
                let r = sigs.fn_sig(*decl)?.ret;
                Some(self.tys.subst(r, args, None))
            }
            Target::Method { method, args, .. } | Target::Op { method, args } => {
                let (m, self_ty) = match method {
                    MethodRef::Impl { method, .. } => (*method, None),
                    MethodRef::Bound { method, self_ty, .. } => (*method, Some(*self_ty)),
                    MethodRef::Builtin(_) => return None,
                };
                let r = sigs.fn_sig(m)?.ret;
                Some(self.tys.subst(r, args, self_ty))
            }
            Target::TraitMethod { method, self_ty, args } => {
                let r = sigs.fn_sig(*method)?.ret;
                Some(self.tys.subst(r, args, Some(*self_ty)))
            }
            Target::FnValue { ty } => match self.tys.kind(*ty) {
                TypeKind::Fn { ret, .. } => Some(*ret),
                _ => None,
            },
            Target::Builtin { ret, .. } => Some(*ret),
            // an intrinsic's own generic params are not the body's: such a result is left out
            // (std's wrappers return it, so their callers own it)
            Target::Intrinsic { decl } => {
                sigs.fn_sig(*decl).map(|s| s.ret).filter(|&r| !self.tys.has_params(r))
            }
            _ => None,
        }
    }

    fn finish(mut self, j: Job) -> (TypedBody, Vec<TypeKind>, Vec<Diagnostic>) {
        // Pending literal groups left: no context anywhere (E0202).
        for g in 0..self.lits.len() as u32 {
            let r = self.find(g);
            if r == g && self.lits[r as usize].fixed.is_none() {
                self.e0202(g);
            }
        }
        // A const initializer must be a const-expr: the signature phase computes its value, and
        // the diagnostic it deferred is reported unless this body reported an error (a failed
        // const without a deferred one has its diagnostic elsewhere: the evaluation's own, or
        // that of the const or type it failed on).
        let reported = self.diags.iter().any(|d| d.severity() == stave_diag::Severity::Error);
        if j.kind == JobKind::Const
            && let Some(c) = self.sigs.konst(self.decl)
        {
            if c.value.is_none() {
                let d = if reported { None } else { c.deferred.clone() };
                self.has_error = true;
                self.report(d);
            } else if !reported && !self.has_error {
                self.same_const_types(&c.types);
            }
        }
        if j.kind == JobKind::Fn {
            self.value_sizes(j);
        }
        let mut node_types = Vec::with_capacity(self.node_ty.len());
        for i in 0..self.node_ty.len() {
            let t = self.node_ty[i];
            let t = match self.resolve(t) {
                Ty::T(t) => t,
                Ty::Lit(_) => ids::ERROR,
            };
            node_types.push(if t == ids::ERROR { TypeId::NONE } else { t });
        }
        let mut locals: Vec<(u32, TypeId)> = self.locals.iter().map(|(&k, &v)| (k, v)).collect();
        locals.sort_by_key(|x| x.0);
        let mut moves = std::mem::take(&mut self.moves);
        moves.sort_unstable();
        moves.dedup();
        for (tok, out) in std::mem::take(&mut self.selects) {
            let t = match self.resolve(out) {
                Ty::T(t) => t,
                Ty::Lit(_) => ids::ERROR,
            };
            if let Some((_, Target::Builtin { ret, .. })) = self.targets.iter_mut().rev().find(|x| x.0 == tok)
            {
                *ret = t;
            }
        }
        self.targets.sort_by_key(|x| x.0);
        self.implicit.sort_by_key(|x| x.0);
        self.record_drops();
        self.coercions.sort_by_key(|x| x.0);
        self.coercions.dedup();
        self.denied.sort();
        self.denied.dedup();
        self.valued.retain(|v| v.1 != 0);
        self.valued.sort();
        self.valued.dedup();
        let has_error =
            self.has_error || self.diags.iter().any(|d| d.severity() == stave_diag::Severity::Error);
        let stmt_ends = crate::temps::StmtEnds::new(&self.icx, j.node, self.lo, self.contracts.as_deref());
        let body = TypedBody {
            decl: j.decl,
            item: j.item,
            node: j.node,
            lo: self.lo,
            node_types,
            locals,
            targets: std::mem::take(&mut self.targets),
            tries: {
                let mut t = std::mem::take(&mut self.tries);
                t.sort_by_key(|x| x.0);
                t
            },
            coercions: std::mem::take(&mut self.coercions),
            sites: std::mem::take(&mut self.sites),
            implicit: std::mem::take(&mut self.implicit),
            for_kinds: {
                let mut k = std::mem::take(&mut self.for_kinds);
                k.sort_by_key(|x| x.0);
                k
            },
            moves,
            drops: std::mem::take(&mut self.drops),
            denied: std::mem::take(&mut self.denied),
            valued: std::mem::take(&mut self.valued),
            has_error,
            emulated: std::mem::take(&mut self.gates.emulated),
            contracts: self.contracts.take(),
            test_domain: self.domain.take(),
            stmt_ends,
            internal: std::mem::take(&mut self.internal),
        };
        // thread-crossing capture checks (E0802, E0806; crate::threads)
        let td = self.thread_checks(&body);
        self.diags.extend(td);
        (body, self.tys.local, self.diags)
    }

    /// E0911 case `size`: a value of this fn (a parameter, a binding, the return value) whose
    /// type has more than [`crate::consteval::MAX_VALUE_BYTES`]: a back end keeps each value in
    /// one frame slot. Once per type, at its first binding; a generic type is left to its
    /// instances.
    fn value_sizes(&mut self, j: Job) {
        let mut held: Vec<(u32, TypeId)> = self.locals.iter().map(|(&n, &t)| (n, t)).collect();
        held.sort_unstable_by_key(|x| x.0);
        let mut seen = rustc_hash::FxHashSet::default();
        let name_span = self.prog.decl(j.decl).map(|d| d.name_span);
        let ret = self.sigs.fn_sig(j.decl).map(|s| s.ret);
        let spans: Vec<(Span, TypeId)> =
            held.into_iter().map(|(n, t)| (self.icx.node_span(n), t)).chain(name_span.zip(ret)).collect();
        for (sp, t) in spans {
            if t.is_none() || !seen.insert(t) || self.tys.has_error(t) || self.tys.has_params(t) {
                continue;
            }
            let Some((bytes, _)) = crate::traits::layout_of(&mut self.tys, self.sigs, t, 0) else { continue };
            if bytes > crate::consteval::MAX_VALUE_BYTES {
                let tn = self.show(t);
                let d = mk("E0911", Some("size"), &[("type", &tn), ("bytes", &bytes.to_string())], sp, &[]);
                self.report(d);
            }
        }
    }

    /// The types const evaluation decided for the scalar nodes of this const initializer are
    /// the ones this checker gave them: else the value is not the one the source means.
    fn same_const_types(&mut self, types: &[(u32, TypeId)]) {
        for &(n, t) in types {
            let Some(&got) = n.checked_sub(self.lo).and_then(|i| self.node_ty.get(i as usize)) else {
                continue;
            };
            let got = match self.resolve(got) {
                Ty::T(g) => g,
                Ty::Lit(_) => TypeId::NONE,
            };
            if got != t {
                let text = self.icx.node_text(n).to_string();
                let g = if got.is_none() { "no type".to_string() } else { self.show(got) };
                let e = self.show(t);
                self.has_error = true;
                self.internal.push(format!("const evaluation typed `{text}` as `{e}`, the checker as `{g}`"));
                return;
            }
        }
    }

    // ---------- statements ----------

    /// Checks a block: whether it diverges (path rule).
    pub fn block(&mut self, b: u32) -> bool {
        if !self.enter(b) {
            return false;
        }
        let entry_reach = self.reachable;
        let kids = self.kids(b);
        let mut div = false;
        for &s in &kids {
            if div {
                self.reachable = false;
            }
            if self.stmt(s) {
                div = true;
            }
        }
        self.free_kids(kids);
        self.reachable = entry_reach;
        self.leave();
        div
    }

    /// Descends into node `n`; false (and E0005 case `expr`, once per body) past [`MAX_DEPTH`].
    pub fn enter(&mut self, n: u32) -> bool {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            self.depth -= 1;
            self.has_error = true;
            if !self.depth_reported {
                self.depth_reported = true;
                let at = self.icx.tok_span(self.icx.a(n));
                let d = mk("E0005", Some("expr"), &[], at, &[]);
                self.diags.extend(d);
            }
            return false;
        }
        true
    }

    pub fn leave(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    /// A statement: whether it diverges.
    fn stmt(&mut self, s: u32) -> bool {
        let icx = self.icx;
        let a = icx.a(s);
        let t = icx.tk(a);
        let kids = self.kids(s);
        let r = if t == Term::KwLet as u8 {
            self.let_stmt(&kids)
        } else if t == Term::KwIf as u8 {
            match kids.first() {
                Some(&i) => self.if_stmt(i),
                None => false,
            }
        } else if t == Term::KwWhile as u8 {
            if let (Some(&c), Some(&b)) = (kids.first(), kids.get(1)) {
                self.cond(c);
                self.loop_body(b);
            }
            false
        } else if t == Term::KwLoop as u8 {
            match kids.first() {
                Some(&b) => !self.loop_body(b),
                None => false,
            }
        } else if t == Term::KwFor as u8 {
            if let Some(&f) = kids.first() {
                self.for_stmt(f);
            }
            false
        } else if t == Term::KwMatch as u8 {
            self.match_stmt(s, &kids)
        } else if t == Term::KwReturn as u8 {
            self.return_stmt(s, kids.first().copied());
            true
        } else if t == Term::KwBreak as u8 || t == Term::KwContinue as u8 {
            let is_break = t == Term::KwBreak as u8;
            match self.rets.last_mut().and_then(|r| r.loops.last_mut()) {
                Some(l) => {
                    if is_break {
                        l.has_break = true;
                    }
                }
                None => {
                    let text = self.icx.tok_str(a).to_string();
                    let d =
                        mk("E0201", Some("not-in-loop"), &[("operand", &text)], self.icx.tok_span(a), &[]);
                    self.report(d);
                }
            }
            // `break`/`continue` end the block's path but do not make the enclosing statement
            // diverge for E0207 purposes beyond this block (the loop rule handles `loop`).
            true
        } else if t == Term::KwUnsafe as u8 {
            self.unsafe_depth += 1;
            let r =
                kids.iter().find(|&&c| icx.kind(c) == Some(NodeKind::Block)).is_some_and(|&b| self.block(b));
            self.unsafe_depth -= 1;
            r
        } else if t == Term::KwScope as u8 {
            self.scope_depth += 1;
            let r = kids.first().is_some_and(|&b| self.block(b));
            self.scope_depth -= 1;
            r
        } else if t == Term::KwSpawn as u8 {
            if self.scope_depth == 0 {
                // a closure inside the scope is not part of it (its spawn would have no scope to
                // join): E0808, with the closure as related span
                let d = match self.closure_in_scope {
                    Some(f) => mk("E0808", None, &[], icx.tok_span(a), &[("closure", icx.tok_span(f))]),
                    None => mk("E0801", None, &[], icx.tok_span(a), &[]),
                };
                self.report(d);
            }
            // a spawn body is a separate thread: its paths do not end the enclosing fn's path
            let saved = self.scope_depth;
            let saved_closure = self.closure_in_scope.take();
            self.scope_depth = 0;
            self.rets.push(RetCx { ty: ids::UNIT, span: None, loops: Vec::new(), is_never: false });
            if let Some(&b) = kids.first() {
                self.block(b);
            }
            self.rets.pop();
            self.scope_depth = saved;
            self.closure_in_scope = saved_closure;
            false
        } else if t == Term::KwStrict as u8 || t == Term::KwEmulate as u8 {
            let saved = self.gates.enter(a, t == Term::KwStrict as u8);
            let r = kids.first().is_some_and(|&b| self.block(b));
            self.gates.leave(saved);
            r
        } else if t == Term::LBrace as u8 {
            kids.first().is_some_and(|&b| self.block(b))
        } else {
            match kids.first() {
                Some(&e) => self.expr_stmt(e),
                None => false,
            }
        };
        self.free_kids(kids);
        r
    }

    fn cond(&mut self, c: u32) {
        let t = self.expr(c, Some(ids::BOOL));
        self.coerce(c, t, ids::BOOL, None);
    }

    /// A loop body: whether a `break` targets this loop.
    fn loop_body(&mut self, b: u32) -> bool {
        if let Some(r) = self.rets.last_mut() {
            r.loops.push(LoopCx { has_break: false });
        }
        let reach = self.reachable;
        self.block(b);
        self.reachable = reach;
        self.rets.last_mut().and_then(|r| r.loops.pop()).is_some_and(|l| l.has_break)
    }

    fn let_stmt(&mut self, kids: &[u32]) -> bool {
        let icx = self.icx;
        let Some(&pat) = kids.first() else { return false };
        let ty_node = kids.iter().copied().find(|&c| icx.kind(c) == Some(NodeKind::Type));
        let init = kids.iter().copied().find(|&c| icx.is_expr(c));
        self.diverges = false;
        let t = match (ty_node, init) {
            (Some(tn), Some(e)) => {
                let want = self.lower_ty(tn, false);
                let sp = icx.node_span(tn);
                self.check_expr(e, want, Some(sp));
                want
            }
            (None, Some(e)) => {
                let t = self.expr(e, None);
                self.force(t)
            }
            (Some(tn), None) => self.lower_ty(tn, false),
            (None, None) => ids::ERROR,
        };
        let div = std::mem::take(&mut self.diverges);
        let p = self.pattern(pat, t, crate::patterns::Mode::Move);
        crate::patterns::irrefutable(self, pat, &p, t);
        div
    }

    fn expr_stmt(&mut self, es: u32) -> bool {
        let icx = self.icx;
        let kids = self.kids(es);
        self.diverges = false;
        if kids.len() >= 3 {
            // OrExpr AssignOp Expr
            let (lhs, op, rhs) = (kids[0], kids[1], kids[2]);
            let lt = self.expr(lhs, None);
            let lt = self.force(lt);
            self.assign_target(lhs);
            let opt = icx.tk(icx.a(op));
            if opt == Term::Eq as u8 {
                self.check_expr(rhs, lt, Some(icx.node_span(lhs)));
            } else {
                self.compound_assign(op, opt, lhs, lt, rhs);
            }
        } else if let Some(&e) = kids.first() {
            let t = self.expr(e, None);
            let t = self.force(t);
            // a type holding the error type already has its diagnostic
            if t != ids::UNIT && t != ids::NEVER && !t.is_none() && !self.tys.has_error(t) {
                let at = icx.node_span(e);
                let ty = self.show(t);
                let insert = Span::new(at.file, at.offset, 0);
                let d = mk("E0210", None, &[("type", &ty)], at, &[("start", insert)]);
                self.report(d);
            }
        }
        self.free_kids(kids);
        std::mem::take(&mut self.diverges)
    }

    /// E0803 for a `static mut` written through an assignment is reported by the path use itself;
    /// an assignment into an immutable `static` is E0503.
    fn assign_target(&mut self, lhs: u32) {
        let at = self.icx.node_span(lhs);
        self.static_mutation(lhs, at);
    }

    fn return_stmt(&mut self, s: u32, e: Option<u32>) {
        let icx = self.icx;
        let Some(rc) = self.rets.last().cloned() else { return };
        if rc.is_never {
            if let Some(e) = e {
                let _ = self.expr(e, None);
            }
            if self.reachable {
                let name =
                    self.prog.decl(self.decl).map_or("_", |d| self.prog.strings.get(d.name)).to_string();
                let d = mk("E0207", Some("never"), &[("name", &name)], icx.tok_span(icx.a(s)), &[]);
                self.report(d);
            }
            return;
        }
        match e {
            Some(e) => {
                self.check_expr(e, rc.ty, rc.span);
            }
            None => {
                if rc.ty != ids::UNIT && rc.ty != ids::ERROR {
                    let expected = self.show(rc.ty);
                    let roles: Vec<(&str, Span)> = rc.span.map(|s| ("expected", s)).into_iter().collect();
                    let d = mk(
                        "E0201",
                        None,
                        &[("expected", &expected), ("found", "()"), ("operand", "return")],
                        icx.tok_span(icx.a(s)),
                        &roles,
                    );
                    self.report(d);
                }
            }
        }
    }

    /// An `if` statement chain (iterative over `else if`): whether it diverges.
    fn if_stmt(&mut self, first: u32) -> bool {
        let icx = self.icx;
        let mut n = first;
        let mut all_div = true;
        let reach = self.reachable;
        for _ in 0..10_000_000 {
            let kids = self.kids(n);
            if let Some(&c) = kids.first() {
                self.cond(c);
            }
            let then_div = kids.get(1).is_some_and(|&b| self.block(b));
            self.reachable = reach;
            all_div &= then_div;
            match kids.get(2).copied() {
                Some(e) if icx.kind(e) == Some(NodeKind::IfStmt) => {
                    self.free_kids(kids);
                    n = e;
                    continue;
                }
                Some(b) => {
                    let d = self.block(b);
                    self.reachable = reach;
                    all_div &= d;
                }
                None => all_div = false,
            }
            self.free_kids(kids);
            return all_div;
        }
        false
    }

    fn compound_assign(&mut self, op_node: u32, opt: u8, lhs: u32, lt: TypeId, rhs: u32) {
        // `a op= b`: the operator's rules with both operands of a's type (a shift amount too).
        let bin = match opt {
            x if x == Term::PlusEq as u8 => Term::Plus,
            x if x == Term::MinusEq as u8 => Term::Minus,
            x if x == Term::StarEq as u8 => Term::Star,
            x if x == Term::SlashEq as u8 => Term::Slash,
            x if x == Term::PercentEq as u8 => Term::Percent,
            x if x == Term::PlusPercentEq as u8 => Term::PlusPercent,
            x if x == Term::MinusPercentEq as u8 => Term::MinusPercent,
            x if x == Term::StarPercentEq as u8 => Term::StarPercent,
            x if x == Term::PlusPipeEq as u8 => Term::PlusPipe,
            x if x == Term::MinusPipeEq as u8 => Term::MinusPipe,
            x if x == Term::StarPipeEq as u8 => Term::StarPipe,
            x if x == Term::AmpEq as u8 => Term::Amp,
            x if x == Term::PipeEq as u8 => Term::Pipe,
            x if x == Term::CaretEq as u8 => Term::Caret,
            x if x == Term::LtLtEq as u8 => Term::LtLt,
            x if x == Term::GtGtEq as u8 => Term::GtGt,
            _ => Term::LtLtPercent,
        };
        let rt = self.expr(rhs, Some(lt));
        let is_shift = matches!(bin, Term::LtLt | Term::GtGt | Term::LtLtPercent);
        let rt = self.coerce(rhs, rt, lt, Some(self.icx.node_span(lhs)));
        if is_shift && bin != Term::LtLtPercent {
            self.const_shift_amount(rhs, lt);
        }
        self.binop_types(op_node, bin as u8, lhs, Ty::T(lt), rhs, Ty::T(rt));
    }

    /// `let`/`for`/arm patterns: lowered for exhaustiveness, bindings typed.
    pub fn pattern(&mut self, p: u32, t: TypeId, mode: crate::patterns::Mode) -> crate::patterns::Pat {
        crate::patterns::check_pattern(self, p, t, mode)
    }

    // ---------- misc ----------

    /// The `TyEnv` and checked turbofish args of a `GenericArgs` node for params `shape`.
    pub fn turbofish(&mut self, ga: u32, shape: &[Option<TypeId>], at: u32) -> Option<Vec<GArg>> {
        let env = self.env();
        sigs::lower_gargs(self, &env, Some(ga), shape, at)
    }

    pub fn is_unsafe_ctx(&self) -> bool {
        self.unsafe_depth > 0
    }

    /// E0803 at `sp` unless inside `unsafe(..)` (the rule: [`crate::unsafety`]).
    pub fn unsafe_op(&mut self, sp: Span) {
        if !self.is_unsafe_ctx() {
            let d = crate::unsafety::e0803(sp);
            self.report(d);
        }
    }

    /// Records an instantiation when the callee is generic. Every instantiation of a fn is
    /// recorded here, so this is where the bounds and the view-kind rule of all its generics (the
    /// parent impl's or trait's included) are checked against `args`, reported at `at`.
    pub fn site(&mut self, node: u32, at: Span, decl: DeclId, args: &[GArg]) {
        if args.is_empty() {
            return;
        }
        self.check_instance(at, decl, args);
        self.sites.push(CallSite { caller: self.decl, node, inst: Instance { decl, args: args.to_vec() } });
    }

    /// E0704: callee features must be a subset of the caller's and the target baseline.
    pub fn features_ok(&mut self, at: Span, callee: DeclId, callee_features: u64) {
        let missing = callee_features & !self.proven_features();
        if missing == 0 {
            return;
        }
        // The names to add to the caller's `needs`: those of the callee's own list that it does
        // not cover, in that order; for a fn value, the features of `missing` no other implies.
        let mut names: Vec<&str> = Vec::new();
        if let Some(sig) = self.sigs.fn_sig(callee) {
            for (n, bits) in &sig.needs_named {
                if bits & !self.proven_features() != 0 && !names.contains(&n.as_str()) {
                    names.push(n);
                }
            }
        }
        if names.is_empty() {
            names = crate::types::feature_names(missing);
        }
        if names.is_empty() {
            return;
        }
        let feature = names.join(", ");
        let path = crate::lower::fn_path(self.prog, self.sigs, callee);
        let roles: Vec<(&str, Span)> =
            self.prog.decl(callee).map(|d| ("callee", d.name_span)).into_iter().collect();
        let d = mk("E0704", None, &[("callee", &path), ("feature", &feature)], at, &roles);
        self.report(d);
    }
}

impl sigs::LowerHost for Ck<'_> {
    fn prog(&self) -> &Program {
        self.prog
    }

    fn sigs_generics_len(&mut self, d: DeclId) -> Option<Vec<Option<TypeId>>> {
        Some(self.sigs.generics(d).iter().map(|g| g.is_const.then_some(g.const_ty)).collect())
    }

    fn adt_generics(&mut self, d: DeclId) -> Vec<GenericParam> {
        self.sigs.generics(d).to_vec()
    }

    fn alias_target(&mut self, d: DeclId) -> Option<(Vec<GenericParam>, TypeId)> {
        match self.sigs.get(d) {
            DeclSig::Alias(a) => Some((a.generics.clone(), a.target)),
            _ => None,
        }
    }

    fn const_u64(&mut self, env: &TyEnv<'_>, n: u32, ty: TypeId) -> ConstArg {
        crate::consteval::eval_u64(self, env, n, ty)
    }

    fn report(&mut self, d: Option<Diagnostic>) {
        Ck::report(self, d);
    }

    fn view_env(&self) -> &[GenericParam] {
        self.generics
    }

    fn const_param_ty(&self, idx: u32) -> Option<TypeId> {
        self.generics.get(idx as usize).filter(|g| g.is_const).map(|g| g.const_ty)
    }

    fn vector_width(&mut self, span: Span, ty: &str, bits: u32) {
        self.gate_width_of(span, ty, bits);
    }

    fn sig_width(&mut self, span: Span, t: TypeId, features: u64) {
        self.sig_width_at(span, t, features);
    }
}

impl crate::consteval::CEnv for Ck<'_> {
    fn const_value(&mut self, d: DeclId) -> Option<(TypeId, crate::consteval::Val)> {
        let c = self.sigs.konst(d)?;
        if !matches!(self.sigs.get(d), DeclSig::Const(_)) {
            return None;
        }
        Some((c.ty, c.value.clone()?))
    }

    fn adt_def(&mut self, d: DeclId) -> Option<sigs::AdtDef> {
        self.sigs.adt(d).cloned()
    }

    fn layout(&mut self, t: TypeId) -> Option<(u64, u64)> {
        crate::traits::layout_of(&mut self.tys, self.sigs, t, 0)
    }

    fn lang(&self) -> &sigs::LangItems {
        &self.sigs.lang
    }

    fn fn_types(&mut self, d: DeclId) -> Option<(Vec<TypeId>, TypeId)> {
        self.sigs.fn_sig(d).map(|f| (f.params.clone(), f.ret))
    }

    fn internal(&mut self, msg: String) {
        self.has_error = true;
        self.internal.push(msg);
    }
}

impl Interner for Ck<'_> {
    fn kind(&self, t: TypeId) -> &TypeKind {
        self.tys.kind(t)
    }

    fn intern(&mut self, k: TypeKind) -> TypeId {
        self.tys.intern(k)
    }

    fn fixed_ty(&self, f: TypeId) -> TypeId {
        self.tys.fixed_ty(f)
    }
}
