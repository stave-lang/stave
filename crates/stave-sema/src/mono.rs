//! Monomorphization: instance keys, the instance root set of
//! contracts/checked-ir.md "Instance root set", and the mono-time resolution of calls that typeck
//! left generic (trait methods through a bound, operators through a bound, `Drop` impls).
//!
//! **Instances.** An instance is a fn or method decl with its flat generic args (container's first,
//! then the fn's own), keyed by [`InstKey`] = canonical (decl, args): args are interned type ids of
//! the lowering's type table (typeck's canonical table, extended with the instances' types; equal
//! keys = equal types), so the instance cache is a hash map on the key.
//! Lowering requests instances round by round: round 0 is the root set, each later
//! round lowers the instances first requested by the previous one, until no new instance appears.
//!
//! **Root set** (contracts/checked-ir.md "Instance root set"). debug: every
//! non-generic fn and method with a checked body, of the root package, its dependencies, and the
//! built-in packages (fns dropped for another arch have none: typeck makes no body for them;
//! `test fn`s only under `stave test`, [`crate::lower::Mode::Test`] /
//! [`crate::lower::Mode::ReleaseTest`]); release: `main`
//! of the root package, the `extern "C"` exports, the `allocator` module's `alloc`/`free`/`realloc`,
//! `test fn`s under `stave test`, and the runtime entry points the compiler's own stubs call
//! (`runtime.start.entry`, `runtime.panic.trap`, `runtime.panic.trap_msg`,
//! `runtime.panic.on_signal`; contracts/intrinsics.md "Paths fixed by the compiler"). Both modes
//! then add every instance a lowered body requests (calls, fn-value uses, drop glue), round by
//! round; a lowered `dispatch fn` requests its arm targets ([`dispatch_arms`]), so a reachable
//! dispatch makes them reachable.
//!
//! **Instantiation depth** (E0902, [`depth_check`], run by the type checker): from every
//! non-generic body, the generic instances its call sites request, depth first, substituting
//! each instance's args into its own body's sites (trait method sites resolve to the impl
//! method). An instance that would be the [`INST_DEPTH`]+1-th instance of its generic on the
//! current chain is E0902 at the requesting call, once per generic, and is not expanded. So
//! polymorphic recursion never reaches lowering; lowering keeps a higher limit of its own.

use rustc_hash::{FxHashMap, FxHashSet};
use stave_diag::Diagnostic;

use crate::Program;
use crate::decls::{DeclFlags, DeclKind};
use crate::ids::{DeclId, Sym};
use crate::itemcx::{Brackets, ItemCx};
use crate::sigs::{Sigs, TraitRef};
use crate::traits::World;
use crate::typeck::{Typeck, TypedBody};
use crate::types::{ConstArg, GArg, Interner, Overlay, TypeId, TypeKind, TypeTable};

/// A canonical instance key: a fn or method decl and its flat generic args.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InstKey {
    pub decl: DeclId,
    pub args: Vec<GArg>,
}

/// What a body references by a local fn id (`Operand::Func(FuncId(i))` = request `i`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Req {
    Inst(InstKey),
    /// The drop glue of a type.
    Glue(TypeId),
    /// Closure `k` of the same job.
    Closure(u32),
}

/// The std/runtime items the lowering calls by path (contracts/intrinsics.md "Paths fixed by the
/// compiler").
#[derive(Clone, Debug, Default)]
pub(crate) struct LangFns {
    pub string_new: Option<DeclId>,
    pub string_push_str: Option<DeclId>,
    pub string_as_str: Option<DeclId>,
    pub set_user_site: Option<DeclId>,
}

fn sym(prog: &Program, s: &str) -> Option<Sym> {
    prog.strings.lookup(s)
}

/// The module whose path text is `path`.
pub(crate) fn module_decl(prog: &Program, path: &str) -> Option<crate::ids::ModuleId> {
    prog.modules.iter().find(|m| m.path_text == path).map(|m| m.id)
}

/// Item `name` of module `path`.
pub(crate) fn item_at(prog: &Program, path: &str, name: &str) -> Option<DeclId> {
    let m = module_decl(prog, path)?;
    prog.module_item(m, sym(prog, name)?)
}

/// The inherent method `name` of ADT `adt` (first match, impl order).
pub(crate) fn inherent_method(prog: &Program, sigs: &Sigs, adt: DeclId, name: &str) -> Option<DeclId> {
    let s = sym(prog, name)?;
    let list = sigs.impls_by_head.get(&crate::resolve::TypeHead::Adt(adt))?;
    for &i in list {
        let Some(imp) = sigs.impl_def(i) else { continue };
        if imp.trait_ref.is_some() {
            continue;
        }
        if let Some(m) = imp.methods.iter().copied().find(|&m| prog.decl(m).is_some_and(|d| d.name == s)) {
            return Some(m);
        }
    }
    None
}

impl LangFns {
    pub fn find(prog: &Program, sigs: &Sigs) -> LangFns {
        let s = sigs.lang.string;
        let m = |n: &str| if s.is_some() { inherent_method(prog, sigs, s, n) } else { None };
        LangFns {
            string_new: m("new"),
            string_push_str: m("push_str"),
            string_as_str: m("as_str"),
            set_user_site: item_at(prog, "runtime.panic", "set_user_site"),
        }
    }
}

/// Which build the root set is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootMode {
    Debug,
    Release,
}

/// The instance root set, in decl order.
pub(crate) fn roots(prog: &Program, tc: &Typeck, mode: RootMode, test: bool) -> Vec<InstKey> {
    let mut out: Vec<InstKey> = Vec::new();
    let is_fn = |d: DeclId| {
        prog.decl(d).is_some_and(|x| matches!(x.kind, DeclKind::Fn | DeclKind::Method))
            && tc.body(d).is_some_and(|b| !b.has_error)
            && tc.sigs.fn_sig(d).is_some_and(|s| s.generics.is_empty())
    };
    match mode {
        RootMode::Debug => {
            for b in &tc.bodies {
                let Some(d) = prog.decl(b.decl) else { continue };
                if d.flags.has(DeclFlags::TEST) && !test {
                    continue;
                }
                if is_fn(b.decl) {
                    out.push(InstKey { decl: b.decl, args: Vec::new() });
                }
            }
        }
        RootMode::Release => {
            let mut add = |d: Option<DeclId>| {
                if let Some(d) = d
                    && is_fn(d)
                    && !out.iter().any(|k| k.decl == d)
                {
                    out.push(InstKey { decl: d, args: Vec::new() });
                }
            };
            for p in &prog.packages {
                if p.is_root
                    && let Some(m) = p.root_module.get()
                    && let Some(s) = sym(prog, "main")
                {
                    add(prog.module_item(m, s));
                }
            }
            for (k, d) in prog.decls.iter().enumerate() {
                let id = DeclId(k as u32);
                if d.flags.has(DeclFlags::EXTERN_EXPORT) || (test && d.flags.has(DeclFlags::TEST)) {
                    add(Some(id));
                }
            }
            for m in allocator_module(prog) {
                for n in ["alloc", "free", "realloc"] {
                    add(sym(prog, n).and_then(|s| prog.module_item(m, s)));
                }
            }
            for (m, n) in [
                ("runtime.start", "entry"),
                ("runtime.panic", "trap"),
                ("runtime.panic", "trap_msg"),
                ("runtime.panic", "on_signal"),
            ] {
                add(item_at(prog, m, n));
            }
            out.sort();
        }
    }
    out
}

/// The arm targets of `dispatch fn` `d` in arm order (the `_` arm last), as root instances: only
/// targets with a checked body (an arm naming a fn of another arch is dropped with it).
pub(crate) fn dispatch_arms(prog: &Program, tc: &Typeck, d: DeclId) -> Vec<InstKey> {
    let mut out: Vec<InstKey> = Vec::new();
    let Some(decl) = prog.decl(d) else { return out };
    if !decl.flags.has(DeclFlags::DISPATCH) || decl.node == u32::MAX {
        return out;
    }
    let ast = prog.item_ast(decl.item);
    for n in ast.descendants(decl.node) {
        if ast.kind(n) != Some(stave_syntax::NodeKind::Name) {
            continue;
        }
        if let Some(u) = prog.uses(decl.item, n).last()
            && let crate::resolve::Res::Decl(t) = u.res
            && prog.decl(t).is_some_and(|x| matches!(x.kind, DeclKind::Fn | DeclKind::Method))
            && tc.body(t).is_some_and(|b| !b.has_error)
            && tc.sigs.fn_sig(t).is_some_and(|s| s.generics.is_empty())
        {
            let k = InstKey { decl: t, args: Vec::new() };
            if !out.contains(&k) {
                out.push(k);
            }
        }
    }
    out
}

/// The module named by the `allocator` declaration, if any.
pub(crate) fn allocator_module(prog: &Program) -> Vec<crate::ids::ModuleId> {
    let mut out = Vec::new();
    for (i, it) in prog.items.iter().enumerate() {
        if !it.ok || it.decl.is_some() {
            continue;
        }
        let ast = prog.item_ast(crate::ids::ItemId(i as u32));
        let Some(r) = ast.root() else { continue };
        let Some(node) =
            ast.descendants(r).find(|&n| ast.kind(n) == Some(stave_syntax::NodeKind::AllocatorDecl))
        else {
            continue;
        };
        let uses = prog.uses(crate::ids::ItemId(i as u32), node);
        if let Some(u) = uses.last()
            && let crate::resolve::Res::Decl(d) = u.res
            && let Some(dd) = prog.decl(d)
            && dd.kind == DeclKind::Module
        {
            out.push(dd.module);
        }
    }
    out
}

/// A call resolved at mono time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Resolved {
    Inst(InstKey),
    /// A built-in scalar operator (the trait name: `Add`, `Neg`, ...).
    BuiltinOp(&'static str),
}

/// A trait method `method` for `Self` = `self_ty` with the trait's args and the method's own args:
/// the impl method instance, or a built-in operator.
pub(crate) fn resolve_trait_method<I: Interner + ?Sized>(
    w: &World<'_>,
    cx: &mut I,
    method: DeclId,
    self_ty: TypeId,
    trait_args: &[GArg],
    own_args: &[GArg],
) -> Option<Resolved> {
    let md = w.prog.decl(method)?;
    let trait_decl = md.parent;
    let name = md.name;
    let tr = TraitRef { trait_decl, args: trait_args.to_vec() };
    if let Some((imp, impl_args)) = w.find_impl(cx, self_ty, &tr, &[], 0)
        && let Some(idef) = w.sigs.impl_def(imp)
        && let Some(m) =
            idef.methods.iter().copied().find(|&m| w.prog.decl(m).is_some_and(|d| d.name == name))
    {
        let mut args = impl_args;
        args.extend_from_slice(own_args);
        return Some(Resolved::Inst(InstKey { decl: m, args }));
    }
    let tname = w.prog.strings.get(w.prog.decl(trait_decl)?.name);
    let op = crate::sigs::OP_TRAITS.iter().find(|(t, _)| *t == tname)?.0;
    if w.sigs.lang.ops.get(op) == Some(&trait_decl) && crate::traits::builtin_op_impl(cx, op, self_ty) {
        return Some(Resolved::BuiltinOp(op));
    }
    None
}

/// The `Drop.drop` impl method instance of `t`, if `t` implements `Drop`.
pub(crate) fn drop_method<I: Interner + ?Sized>(w: &World<'_>, cx: &mut I, t: TypeId) -> Option<InstKey> {
    let drop = w.sigs.lang.drop;
    if drop.is_none() || !matches!(cx.kind(t), TypeKind::Adt { .. }) {
        return None;
    }
    let (imp, args) = w.find_impl(cx, t, &TraitRef { trait_decl: drop, args: Vec::new() }, &[], 0)?;
    let m = *w.sigs.impl_def(imp)?.methods.first()?;
    Some(InstKey { decl: m, args })
}

/// Field types of the variants of a concrete ADT type (struct: one variant).
pub(crate) fn adt_fields<I: Interner + ?Sized>(
    sigs: &Sigs,
    cx: &mut I,
    t: TypeId,
) -> Option<Vec<Vec<TypeId>>> {
    let TypeKind::Adt { def, args } = cx.kind(t).clone() else { return None };
    let adt = sigs.adt(def)?;
    let mut out = Vec::with_capacity(adt.variants.len());
    for v in &adt.variants {
        let mut fs = Vec::with_capacity(v.fields.len());
        for f in &v.fields {
            fs.push(cx.subst(f.ty, &args, None));
        }
        out.push(fs);
    }
    Some(out)
}

/// Whether values of the concrete type `t` need drop glue (a `Drop` impl anywhere inside, not
/// behind a pointer). Memoized in `memo`.
///
/// This is reachability over containment edges (ADT fields, tuple and array elements), walked
/// with an explicit stack, so the answer is exact for a type of any finite nesting depth.
pub(crate) fn needs_drop<I: Interner + ?Sized>(
    w: &World<'_>,
    cx: &mut I,
    t: TypeId,
    memo: &mut FxHashMap<TypeId, bool>,
) -> bool {
    if let Some(&b) = memo.get(&t) {
        return b;
    }
    let mut seen: FxHashSet<TypeId> = FxHashSet::default();
    seen.insert(t);
    let mut stack = vec![t];
    while let Some(u) = stack.pop() {
        // memo `false`: nothing reachable from `u` has a Drop impl; `true`: something does
        match memo.get(&u) {
            Some(false) => continue,
            Some(true) => {
                memo.insert(t, true);
                return true;
            }
            None => {}
        }
        let inner: Vec<TypeId> = match cx.kind(u).clone() {
            TypeKind::Adt { .. } => {
                if drop_method(w, cx, u).is_some() {
                    memo.insert(u, true);
                    memo.insert(t, true);
                    return true;
                }
                adt_fields(w.sigs, cx, u).map_or_else(Vec::new, |vs| vs.into_iter().flatten().collect())
            }
            TypeKind::Tuple(ts) => ts.to_vec(),
            TypeKind::Array { elem, .. } => vec![elem],
            _ => Vec::new(),
        };
        for x in inner {
            if seen.insert(x) {
                stack.push(x);
            }
        }
    }
    // The search found no Drop impl: none is reachable from any type it visited.
    for u in seen {
        memo.insert(u, false);
    }
    false
}

/// Most instances of one generic on one instantiation chain (E0902).
pub(crate) const INST_DEPTH: u32 = 64;

/// Most distinct instances [`depth_check`] visits.
const INST_BUDGET: usize = 1 << 20;

/// `a` with the generic params replaced by `args`.
pub(crate) fn subst_garg<I: Interner + ?Sized>(cx: &mut I, a: GArg, args: &[GArg]) -> GArg {
    match a {
        GArg::Type(t) => GArg::Type(cx.subst(t, args, None)),
        GArg::Const(ConstArg::Param(i)) => match args.get(i as usize) {
            Some(GArg::Const(c)) => GArg::Const(*c),
            _ => a,
        },
        c => c,
    }
}

fn concrete<I: Interner + ?Sized>(cx: &I, a: &GArg) -> bool {
    match a {
        GArg::Type(t) => !cx.has_params(*t),
        GArg::Const(c) => !matches!(c, ConstArg::Param(_)),
    }
}

/// The instance a call site of `decl` with concrete `args` requests (a trait method: the impl
/// method; `Self` is the last arg).
fn site_instance<I: Interner + ?Sized>(
    w: &World<'_>,
    cx: &mut I,
    decl: DeclId,
    args: Vec<GArg>,
) -> Option<InstKey> {
    if w.prog.decl(decl)?.kind != DeclKind::TraitMethod {
        return Some(InstKey { decl, args });
    }
    let (&last, rest) = args.split_last()?;
    let GArg::Type(self_ty) = last else { return None };
    let np = (w.sigs.fn_sig(decl)?.n_parent as usize).min(rest.len());
    match resolve_trait_method(w, cx, decl, self_ty, &rest[..np], &rest[np..])? {
        Resolved::Inst(k) => Some(k),
        Resolved::BuiltinOp(_) => None,
    }
}

/// E0902 diagnostics of the checked program (module docs). `bodies` sorted by decl.
pub(crate) fn depth_check(
    prog: &Program,
    sigs: &Sigs,
    table: &TypeTable,
    bodies: &[TypedBody],
    brackets: &[Brackets],
) -> Vec<Diagnostic> {
    struct Frame<'b> {
        decl: DeclId,
        args: Vec<GArg>,
        body: &'b TypedBody,
        next: usize,
    }
    let body_of = |d: DeclId| {
        bodies.binary_search_by_key(&d, |b| b.decl).ok().map(|i| &bodies[i]).filter(|b| !b.has_error)
    };
    let w = World { prog, sigs };
    let mut cx = Overlay::new(table);
    let mut seen: FxHashSet<InstKey> = FxHashSet::default();
    let mut on_chain: FxHashMap<DeclId, u32> = FxHashMap::default();
    let mut reported: FxHashSet<DeclId> = FxHashSet::default();
    let mut out = Vec::new();
    for root in bodies {
        if root.has_error
            || root.sites.is_empty()
            || !sigs.fn_sig(root.decl).is_some_and(|s| s.generics.is_empty())
        {
            continue;
        }
        let mut stack = vec![Frame { decl: root.decl, args: Vec::new(), body: root, next: 0 }];
        while let Some(top) = stack.last_mut() {
            let caller: &TypedBody = top.body;
            let Some(site) = caller.sites.get(top.next) else {
                if let Some(f) = stack.pop()
                    && !f.args.is_empty()
                    && let Some(n) = on_chain.get_mut(&f.decl)
                {
                    *n -= 1;
                }
                continue;
            };
            top.next += 1;
            let args: Vec<GArg> = site.inst.args.iter().map(|&a| subst_garg(&mut cx, a, &top.args)).collect();
            if !args.iter().all(|a| concrete(&cx, a)) {
                continue;
            }
            let Some(key) = site_instance(&w, &mut cx, site.inst.decl, args) else { continue };
            if key.args.is_empty() {
                continue;
            }
            let Some(body) = body_of(key.decl) else { continue };
            if seen.len() >= INST_BUDGET {
                return out;
            }
            if !seen.insert(key.clone()) {
                continue;
            }
            let n = on_chain.entry(key.decl).or_default();
            if *n >= INST_DEPTH {
                if reported.insert(key.decl)
                    && let Some(icx) =
                        brackets.get(caller.item.idx()).and_then(|b| ItemCx::new(prog, caller.item, b))
                {
                    let path = prog.decl_path(key.decl);
                    out.extend(crate::diag::mk(
                        "E0902",
                        None,
                        &[("generic", &path)],
                        icx.tok_span(site.node),
                        &[],
                    ));
                }
                continue;
            }
            *n += 1;
            stack.push(Frame { decl: key.decl, args: key.args, body, next: 0 });
        }
    }
    out
}
