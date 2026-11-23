//! Lowering to the checked SSA IR (contracts/checked-ir.md):
//! [`lower`] turns a type-checked program into a [`stave_ir::Module`].
//!
//! **Pipeline.** Instances are lowered round by round ([`crate::mono`]): round 0 is the instance
//! root set of the [`Mode`]; every body of a round is lowered in parallel (rayon), each into its
//! own type overlay over the frozen type table (typeck's canonical one, extended by earlier
//! rounds); then, sequentially and in job order, the overlays are merged and the requested
//! instances (calls, fn values, drop glue) not seen before become the next round's jobs. After the last round the module is assembled:
//!
//! - **types**: every type a printed fn uses (signature, type args, values, results), closed over
//!   key components, gets the canonical rank among them as its id, so
//!   the printed `type` lines are dense; types needed only by ADT shapes, statics, or bodiless
//!   fns follow, ranked among themselves (not printed). `Range<T>` is the tuple `(T, T)`.
//! - **instances**: ranked by (decl, args by canonical type id); drop glue after every decl
//!   instance, by its type; a closure is `@<enclosing>.<k>`. `Module::funcs` is in instance order.
//! - **strings**: deduplicated, ranked by (xxHash32 seed 0, bytes), like interned strings S.
//! - **sites**: `Module::sites` rows by (fn instance, source offset) (contracts/intrinsics.md site
//!   table); every trapping instruction, call, and user-site store carries its row; `Func::name_site`
//!   is the fn's name token (PC table).
//! - **statics**: `static` and `static mut` items by decl, with their initial bytes; `intrinsic.static_addr <i>`.
//!
//! **Paths** (`fn @i <path>`): a fn's decl path; a method `<module>.<Type>.<name>` (the impl's
//! self-type head: an ADT path or a built-in type name); drop glue `drop_glue`.

use std::time::{Duration, Instant};

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use stave_ir::types::{GenericArg, Lanes as IrLanes, TypeKey, TypeTable as IrTypes};
use stave_ir::{AdtShape, DeclId as IrDecl, FuncId, InstanceId, Module, Site, SiteInfo, Static};
use stave_target::FeatureSet;

use crate::Program;
use crate::decls::{DeclFlags, DeclKind};
use crate::ids::{DeclId, ItemId};
use crate::itemcx::{Brackets, ItemCx};
use crate::lower_fn::{FnCx, FnOut, G, JobOut};
pub use crate::lower_fn::{LowerError, LowerErrorKind};
use crate::mono::{InstKey, LangFns, Req, RootMode};
use crate::typeck::Typeck;
use crate::types::{ConstArg, FloatBits, GArg, IntBits, Interner, Overlay, TypeId, TypeKind, TypeTable};

/// Which build to lower for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Every non-generic fn of the program's own packages, the release roots, and the instances
    /// they request (contracts/checked-ir.md).
    Debug,
    /// [`Mode::Debug`] with every fn of the built-in packages a root too (covers all of `std` and
    /// `runtime`).
    DebugAll,
    /// Only what is reachable from `main`, exports, the allocator, and the runtime entry points.
    Release,
    /// Debug plus the `test fn`s (`stave test`).
    Test,
    /// Release plus the `test fn`s as roots (`stave test --release`).
    ReleaseTest,
}

/// Wall times of [`lower_timed`].
#[derive(Clone, Copy, Debug, Default)]
pub struct LowerTimings {
    /// Mono rounds (parallel lowering + merges).
    pub rounds: Duration,
    pub assemble: Duration,
    pub n_rounds: u32,
    pub n_funcs: usize,
}

/// A unit of lowering work.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Job {
    Inst(InstKey),
    Glue(TypeId),
}

/// Lowers a checked program (no error diagnostics) to the checked IR.
pub fn lower(prog: &Program, tc: &Typeck, mode: Mode) -> Result<Module, Vec<LowerError>> {
    lower_timed(prog, tc, mode).0
}

/// What a lowering starts from.
enum Roots<'a> {
    Mode(Mode),
    /// The fn instances of these call-graph nodes ([`Typeck::call_graph`]; a closure: its fn's).
    Graph(&'a [u32]),
}

/// Where a site row is: (file, byte offset of its token).
pub type SitePos = (stave_diag::FileId, u32);

/// The checked IR of a set of call-graph nodes ([`lower_graph_nodes`]).
pub struct GraphLowering {
    pub module: Module,
    /// The call-graph nodes of each function (none: drop glue, or an instance the graph does not
    /// have).
    pub nodes_of_func: Vec<Vec<u32>>,
    /// Where each site row is: (file, byte offset of its token).
    pub site_pos: Vec<SitePos>,
}

/// Lowers the fn instances of call-graph nodes `nodes` and everything they request, as every
/// build mode lowers each of them (lowering an instance does not depend on the mode), and maps
/// the functions back to graph nodes. `drops`: as for [`lower_mapped`].
pub fn lower_graph_nodes(
    prog: &Program,
    tc: &Typeck,
    nodes: &[u32],
    drops: Option<&crate::drops::Schedules>,
) -> (Result<GraphLowering, Vec<LowerError>>, LowerTimings) {
    crate::pool::on_big_stacks(|| {
        let (r, tm) = lower_in_pool(prog, tc, Roots::Graph(nodes), false, drops);
        let r = r.map(|a| {
            let nodes_of_func = graph_nodes_of(tc, &a);
            GraphLowering { module: a.module, nodes_of_func, site_pos: a.site_pos }
        });
        (r, tm)
    })
}

/// A lowered module with what lowering knew about it.
struct Assembled {
    module: Module,
    jobs: Vec<Job>,
    /// The job of each instance rank (the first number of a [`stave_ir::InstanceId`]).
    order: Vec<usize>,
    table: TypeTable,
    site_pos: Vec<SitePos>,
}

/// Graph type `t` (typeck's table plus the graph's local types, read through `cx`) in the
/// lowering's table.
fn graph_to_table(
    cx: &Overlay<'_>,
    table: &mut TypeTable,
    memo: &mut FxHashMap<TypeId, TypeId>,
    t: TypeId,
) -> TypeId {
    if !t.is_local() {
        return t;
    }
    if let Some(&x) = memo.get(&t) {
        return x;
    }
    let k = cx.kind(t).clone().map(|c| graph_to_table(cx, table, memo, c));
    let x = table.intern(k);
    memo.insert(t, x);
    x
}

/// The features instance `k` is compiled with: its generic's `needs` and those it inherits from
/// its type arguments ([`crate::inherit`]).
fn inst_features(sigs: &crate::sigs::Sigs, inherited: &FxHashMap<InstKey, u64>, k: &InstKey) -> FeatureSet {
    let declared = sigs.fn_sig(k.decl).map_or(0, |s| s.features);
    let more = if k.args.is_empty() { 0 } else { inherited.get(k).copied().unwrap_or(0) };
    FeatureSet::from_bits(declared | more).unwrap_or(FeatureSet::EMPTY)
}

/// Lowering type `t` as a graph type (typeck's ids, below `n_global`, are shared).
fn table_to_graph(
    table: &TypeTable,
    n_global: usize,
    cx: &mut Overlay<'_>,
    memo: &mut FxHashMap<TypeId, TypeId>,
    t: TypeId,
) -> TypeId {
    if t.is_local() || t.idx() < n_global {
        return t;
    }
    if let Some(&x) = memo.get(&t) {
        return x;
    }
    let k = table.kind(t).clone().map(|c| table_to_graph(table, n_global, cx, memo, c));
    let x = cx.intern(k);
    memo.insert(t, x);
    x
}

/// The graph nodes of each function of `a`: every node of its instance (the plain node and one
/// per fn-value specialization share one body), or of the closure in each of them.
fn graph_nodes_of(tc: &Typeck, a: &Assembled) -> Vec<Vec<u32>> {
    use crate::callgraph::NodeKind;
    let g = &tc.call_graph;
    let mut cx = g.types(&tc.types);
    let mut memo = FxHashMap::default();
    let mut key_of: FxHashMap<&InstKey, Vec<u32>> = FxHashMap::default();
    let mut first_closure = vec![u32::MAX; g.nodes.len()];
    for (i, nd) in g.nodes.iter().enumerate() {
        match &nd.kind {
            NodeKind::Fn { inst, .. } => key_of.entry(inst).or_default().push(i as u32),
            NodeKind::Closure { parent, index: 0, .. } => first_closure[*parent as usize] = i as u32,
            _ => {}
        }
    }
    let n_global = tc.types.len();
    let mut nodes_of_rank: Vec<Vec<u32>> = Vec::with_capacity(a.order.len());
    for &j in &a.order {
        let nodes = match &a.jobs[j] {
            Job::Inst(k) => {
                let args: Vec<GArg> = k
                    .args
                    .iter()
                    .map(|x| match x {
                        GArg::Type(t) => {
                            GArg::Type(table_to_graph(&a.table, n_global, &mut cx, &mut memo, *t))
                        }
                        c => *c,
                    })
                    .collect();
                key_of.get(&InstKey { decl: k.decl, args }).cloned().unwrap_or_default()
            }
            Job::Glue(_) => Vec::new(),
        };
        nodes_of_rank.push(nodes);
    }
    a.module
        .funcs
        .iter()
        .map(|f| {
            let nodes: &[u32] =
                f.instance.0.first().and_then(|&r| nodes_of_rank.get(r as usize)).map_or(&[], |v| v);
            let Some(&c) = f.instance.0.get(1) else { return nodes.to_vec() };
            nodes
                .iter()
                .filter_map(|&node| {
                    let cn = first_closure.get(node as usize).copied().unwrap_or(u32::MAX).saturating_add(c);
                    match g.nodes.get(cn as usize).map(|x| &x.kind) {
                        Some(NodeKind::Closure { parent, index, .. }) if *parent == node && *index == c => {
                            Some(cn)
                        }
                        _ => None,
                    }
                })
                .collect()
        })
        .collect()
}

/// [`lower`] with phase timings.
/// Runs with the caller's thread count (the current rayon pool's) on big-stack workers
/// (module `pool`).
pub fn lower_timed(
    prog: &Program,
    tc: &Typeck,
    mode: Mode,
) -> (Result<Module, Vec<LowerError>>, LowerTimings) {
    crate::pool::on_big_stacks(|| {
        let (r, tm) = lower_in_pool(prog, tc, Roots::Mode(mode), false, None);
        (r.map(|a| a.module), tm)
    })
}

/// A lowered module and the call-graph nodes of each function ([`lower_mapped`]).
pub type Mapped = (Module, Vec<Vec<u32>>);

/// [`lower_timed`] plus the call-graph nodes of each function, as [`GraphLowering::nodes_of_func`].
/// `borrows`: each borrow expression gets an instruction of its own (an identity cast) and a row
/// in its body's side table (`Body::borrows`), for the interpreter's borrow tracking; the
/// module is then not the canonical checked IR. `drops`: the drop schedules of the borrow-checked
/// bodies ([`crate::drops`]); every drop the lowering emits in such a body must be on its
/// schedule, else the lowering fails with an internal error.
pub fn lower_mapped(
    prog: &Program,
    tc: &Typeck,
    mode: Mode,
    borrows: bool,
    drops: Option<&crate::drops::Schedules>,
) -> (Result<Mapped, Vec<LowerError>>, LowerTimings) {
    crate::pool::on_big_stacks(|| {
        let (r, tm) = lower_in_pool(prog, tc, Roots::Mode(mode), borrows, drops);
        let r = r.map(|a| {
            let nodes_of_func = graph_nodes_of(tc, &a);
            (a.module, nodes_of_func)
        });
        (r, tm)
    })
}

fn lower_in_pool(
    prog: &Program,
    tc: &Typeck,
    roots: Roots<'_>,
    borrows: bool,
    drops: Option<&crate::drops::Schedules>,
) -> (Result<Assembled, Vec<LowerError>>, LowerTimings) {
    let mut tm = LowerTimings::default();
    let err = |msg: &str| LowerError {
        kind: LowerErrorKind::Internal,
        func: String::new(),
        at: String::new(),
        msg: msg.to_string(),
    };
    if prog.has_errors() || tc.diagnostics.iter().any(|d| d.severity() == stave_diag::Severity::Error) {
        return (Err(vec![err("the program has error diagnostics")]), tm);
    }
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
    // typeck's canonical table, signatures, and bodies are used as they are: the type helpers
    // look the well-known types up in the table (`Interner::fixed_ty`). Instances' types are
    // interned after the canonical ones; the module's type ids are re-ranked at assembly.
    let mut table = tc.types.clone();
    let sigs = &tc.sigs;
    let lang = LangFns::find(prog);
    let mut statics_list: Vec<DeclId> = Vec::new();
    {
        let mut ov = crate::types::Overlay::new(&table);
        let mut h = crate::lower_expr::TyHost { tys: &mut ov, prog, sigs };
        for (k, d) in prog.decls.iter().enumerate() {
            let id = DeclId(k as u32);
            let Some(c) = sigs.konst(id).filter(|c| c.value.is_some()) else { continue };
            let image = d.kind == DeclKind::Const
                && !h.has_params(c.ty)
                && crate::consteval::cells(&mut h, c.ty).is_some_and(|n| n > CONST_INLINE_CELLS);
            if d.kind == DeclKind::Static || image {
                statics_list.push(id);
            }
        }
    }
    let statics: FxHashMap<DeclId, u32> =
        statics_list.iter().enumerate().map(|(i, &d)| (d, i as u32)).collect();
    let builtin_pkg: Vec<bool> =
        prog.packages.iter().map(|p| p.kind == crate::package::PackageKind::Builtin).collect();
    let vidx = |adt: DeclId, name: &str| -> u32 {
        let s = prog.strings.lookup(name);
        sigs.adt(adt).and_then(|a| a.variants.iter().position(|v| Some(v.name) == s)).map_or(0, |i| i as u32)
    };
    let (opt, res) = (sigs.lang.option, sigs.lang.result);
    let (opt_none, opt_some) = (vidx(opt, "None"), vidx(opt, "Some"));
    let (res_ok, res_err) = (vidx(res, "Ok"), vidx(res, "Err"));
    let any_item = prog.items.iter().position(|it| it.ok).map(|i| ItemId(i as u32));
    let any_requires =
        tc.bodies.iter().any(|b| b.contracts.as_deref().is_some_and(|c| !c.checked_requires().is_empty()));

    let mut jobs: Vec<Job> = match roots {
        Roots::Mode(mode) => {
            let (rmode, test) = match mode {
                Mode::Debug => (RootMode::Debug, false),
                Mode::DebugAll => (RootMode::DebugAll, false),
                Mode::Release => (RootMode::Release, false),
                Mode::Test => (RootMode::Debug, true),
                Mode::ReleaseTest => (RootMode::Release, true),
            };
            crate::mono::roots(prog, tc, rmode, test).into_iter().map(Job::Inst).collect()
        }
        Roots::Graph(nodes) => {
            let g = &tc.call_graph;
            let cx = g.types(&tc.types);
            let mut memo = FxHashMap::default();
            let mut seen: FxHashSet<InstKey> = FxHashSet::default();
            let mut out = Vec::new();
            for &n in nodes {
                let Some(f) = g.fn_of(n) else { continue };
                let crate::callgraph::NodeKind::Fn { inst, body: true, .. } = &g.nodes[f as usize].kind
                else {
                    continue;
                };
                let args: Vec<GArg> = inst
                    .args
                    .iter()
                    .map(|x| match x {
                        GArg::Type(t) => GArg::Type(graph_to_table(&cx, &mut table, &mut memo, *t)),
                        c => *c,
                    })
                    .collect();
                let k = InstKey { decl: inst.decl, args };
                if seen.insert(k.clone()) {
                    out.push(Job::Inst(k));
                }
            }
            out
        }
    };
    // the features generic instances inherit from their type arguments, keyed in this table
    let inherited: FxHashMap<InstKey, u64> = if tc.inst_features.is_empty() {
        FxHashMap::default()
    } else {
        let cx = tc.call_graph.types(&tc.types);
        let mut memo = FxHashMap::default();
        tc.inst_features
            .iter()
            .map(|(k, &f)| {
                let args = k
                    .args
                    .iter()
                    .map(|x| match x {
                        GArg::Type(t) => GArg::Type(graph_to_table(&cx, &mut table, &mut memo, *t)),
                        c => *c,
                    })
                    .collect();
                (InstKey { decl: k.decl, args }, f)
            })
            .collect()
    };
    // the features drop glue is compiled with, keyed in this table
    let glue_features: FxHashMap<TypeId, u64> = if tc.glue_features.is_empty() {
        FxHashMap::default()
    } else {
        let cx = tc.call_graph.types(&tc.types);
        let mut memo = FxHashMap::default();
        tc.glue_features.iter().map(|(&t, &f)| (graph_to_table(&cx, &mut table, &mut memo, t), f)).collect()
    };
    let mut cache: FxHashMap<Job, usize> = jobs.iter().cloned().enumerate().map(|(i, j)| (j, i)).collect();
    // the job that first requested each job (roots: none)
    let mut parent: Vec<Option<usize>> = vec![None; jobs.len()];
    let mut outs: Vec<Option<JobOut>> = Vec::new();
    let mut done = 0usize;
    let mut errors: Vec<LowerError> = Vec::new();
    while done < jobs.len() {
        tm.n_rounds += 1;
        let pending: Vec<Job> = jobs[done..].to_vec();
        let results: Vec<JobOut> = {
            let g = G {
                prog,
                tc,
                sigs,
                table: &table,
                brackets: &brackets,
                lang: &lang,
                statics: &statics,
                builtin_pkg: &builtin_pkg,
                opt_none,
                opt_some,
                res_ok,
                res_err,
                any_requires,
                borrows,
                inherited: &inherited,
                drops,
            };
            pending.par_iter().map(|j| lower_job(&g, any_item, j)).collect()
        };
        let first = done;
        done = jobs.len();
        for (ri, mut r) in results.into_iter().enumerate() {
            if let Some(e) = r.err.take() {
                errors.push(e);
            }
            let m = table.merge(&r.local_types);
            let map = |t: TypeId| -> TypeId {
                if t.is_local() { m.get(t.idx()).copied().unwrap_or(TypeId::NONE) } else { t }
            };
            remap_job(&mut r, &map);
            for q in &r.reqs {
                let job = match q {
                    Req::Inst(k) => Job::Inst(k.clone()),
                    Req::Glue(t) => Job::Glue(*t),
                    Req::Closure(_) => continue,
                };
                if !cache.contains_key(&job) {
                    if let Job::Inst(k) = &job
                        && !k.args.is_empty()
                        && same_decl_depth(&jobs, &parent, first + ri, k.decl) >= LOWER_INST_DEPTH
                    {
                        let path = fn_path(prog, sigs, k.decl);
                        let msg = format!("instantiation depth limit exceeded at `{path}`");
                        let kind = LowerErrorKind::Internal;
                        errors.push(LowerError { kind, func: path, at: String::new(), msg });
                        continue;
                    }
                    cache.insert(job.clone(), jobs.len());
                    jobs.push(job);
                    parent.push(Some(first + ri));
                }
            }
            outs.push(Some(r));
        }
    }
    tm.rounds = t0.elapsed();
    if !errors.is_empty() {
        errors.sort_by(|a, b| (&a.func, &a.at, &a.msg).cmp(&(&b.func, &b.at, &b.msg)));
        errors.dedup();
        return (Err(errors), tm);
    }
    let t1 = Instant::now();
    let outs: Vec<JobOut> = outs.into_iter().flatten().collect();
    let m = assemble(prog, sigs, &mut table, &jobs, &cache, outs, &statics_list, &inherited, &glue_features);
    tm.assemble = t1.elapsed();
    tm.n_funcs = m.as_ref().map_or(0, |m| m.0.funcs.len());
    let r = m.map(|(mut module, order, site_pos)| {
        // Also without the borrow side table: the bounds pass reads a field behind a `&` as one
        // value only outside an `UnsafeCell`.
        if !sigs.lang.unsafe_cell.is_none() {
            module.cell = Some(IrDecl(sigs.lang.unsafe_cell.0));
        }
        if borrows {
            let frozen = |d: &DeclId| prog.decl(*d).is_some_and(|x| !x.flags.has(DeclFlags::MUT));
            module.frozen = statics_list.iter().map(frozen).collect();
        } else {
            module.clauses = Vec::new();
        }
        Assembled { module, jobs, order, table, site_pos }
    });
    (r.map_err(|e| vec![e]), tm)
}

/// Most instances of one generic on a request chain in lowering. The type checker's E0902
/// ([`crate::mono::depth_check`]) stops polymorphic recursion at
/// [`crate::mono::INST_DEPTH`]; this higher limit only keeps lowering finite for a request chain
/// that check does not follow.
const LOWER_INST_DEPTH: u32 = 4 * crate::mono::INST_DEPTH;

/// How many instances of `decl` are on the request chain ending at job `j`.
fn same_decl_depth(jobs: &[Job], parent: &[Option<usize>], j: usize, decl: DeclId) -> u32 {
    let mut n = 0;
    let mut cur = Some(j);
    while let Some(i) = cur {
        if let Some(Job::Inst(k)) = jobs.get(i)
            && k.decl == decl
        {
            n += 1;
        }
        cur = parent.get(i).copied().flatten();
    }
    n
}

fn remap_job(r: &mut JobOut, map: &dyn Fn(TypeId) -> TypeId) {
    for q in &mut r.reqs {
        match q {
            Req::Inst(k) => {
                for a in &mut k.args {
                    if let GArg::Type(t) = a {
                        *t = map(*t);
                    }
                }
            }
            Req::Glue(t) => *t = map(*t),
            Req::Closure(_) => {}
        }
    }
    let mt = |t: stave_ir::TypeId| stave_ir::TypeId(map(TypeId(t.0)).0);
    for f in &mut r.funcs {
        f.params.iter_mut().for_each(|t| *t = map(*t));
        f.ret = f.ret.map(map);
        if let Some(b) = &mut f.body {
            b.values.iter_mut().for_each(|v| v.ty = mt(v.ty));
            b.insts.ty.iter_mut().flatten().for_each(|t| *t = mt(*t));
            for o in &mut b.operands {
                if let stave_ir::Operand::Type(t) = o {
                    *t = mt(*t);
                }
            }
        }
    }
}

/// The link symbol of an `extern "C"` import or export `d`: its name, unmangled.
fn c_symbol(prog: &Program, sigs: &crate::sigs::Sigs, d: DeclId) -> Option<String> {
    let decl = prog.decl(d)?;
    let c = decl.flags.has(DeclFlags::EXTERN_IMPORT | DeclFlags::EXTERN_EXPORT)
        && sigs.fn_sig(d).is_some_and(|s| s.abi == crate::sigs::Abi::C);
    c.then(|| prog.strings.get(decl.name).to_string())
}

/// The checked-IR path of fn or method decl `d`.
pub(crate) fn fn_path(prog: &Program, sigs: &crate::sigs::Sigs, d: DeclId) -> String {
    let Some(decl) = prog.decl(d) else { return String::new() };
    if decl.kind != DeclKind::Method {
        return prog.decl_path(d);
    }
    let name = prog.strings.get(decl.name);
    let module = prog.module(decl.module).map_or(String::new(), |m| m.path_text.clone());
    let head = match sigs.impl_def(decl.parent).and_then(|i| i.head) {
        Some(crate::resolve::TypeHead::Adt(a)) => prog.decl_path(a),
        Some(crate::resolve::TypeHead::Prim(p)) => {
            if module.is_empty() {
                p.text()
            } else {
                format!("{module}.{}", p.text())
            }
        }
        None => {
            if module.is_empty() {
                "impl".to_string()
            } else {
                format!("{module}.impl")
            }
        }
    };
    format!("{head}.{name}")
}

fn lower_job(g: &G<'_>, any_item: Option<ItemId>, job: &Job) -> JobOut {
    let empty = |e: Option<LowerError>| JobOut {
        funcs: Vec::new(),
        reqs: Vec::new(),
        strs: Vec::new(),
        intrs: Vec::new(),
        sites: Vec::new(),
        local_types: Vec::new(),
        err: e,
        name: None,
    };
    let icx_of = |item: ItemId| g.brackets.get(item.idx()).and_then(|b| ItemCx::new(g.prog, item, b));
    match job {
        Job::Glue(t) => {
            let Some(icx) = any_item.and_then(icx_of) else {
                return empty(Some(LowerError {
                    kind: LowerErrorKind::Internal,
                    func: "drop_glue".into(),
                    at: String::new(),
                    msg: "no items".into(),
                }));
            };
            let mut cx = FnCx::new(g, icx, "drop_glue".into());
            let out = cx.lower_glue(*t);
            finish_job(cx, out, None)
        }
        Job::Inst(k) => {
            let prog = g.prog;
            let path = fn_path(prog, g.sigs, k.decl);
            let Some(decl) = prog.decl(k.decl).copied() else { return empty(None) };
            let Some(sig) = g.sigs.fn_sig(k.decl) else { return empty(None) };
            let Some(icx) = icx_of(decl.item).or_else(|| any_item.and_then(icx_of)) else {
                return empty(None);
            };
            let name = Some((icx.fid(), decl.name_span.offset));
            if g.tc.body(k.decl).is_some_and(|b| b.has_error) {
                // lowering runs only without error diagnostics: a flagged body is a checker bug,
                // never a silent declaration
                return empty(Some(LowerError {
                    kind: LowerErrorKind::Internal,
                    func: path,
                    at: String::new(),
                    msg: "body flagged with a type error but no error diagnostic".into(),
                }));
            }
            let needs = inst_features(g.sigs, g.inherited, k);
            if decl.flags.has(DeclFlags::DISPATCH) {
                let mut cx = FnCx::new(g, icx, path);
                cx.needs = needs;
                let (ps, ret) = cx.inst_sig(k).unwrap_or((Vec::new(), TypeId::NONE));
                let out = match crate::dispatch::plan(prog, g.tc, &icx, k.decl) {
                    Some(plan) => cx.lower_dispatch(&plan, &ps, ret),
                    None => {
                        cx.fail_at(None, "dispatch without a resolved arm chain");
                        FnOut {
                            closure: None,
                            params: ps,
                            ret: cx.ret_of(ret),
                            body: None,
                            site_consts: Vec::new(),
                            name_off: None,
                        }
                    }
                };
                return finish_job(cx, out, name);
            }
            let body = g.tc.body(k.decl).filter(|b| !b.has_error);
            let declared_only = body.is_none()
                || decl.flags.has(DeclFlags::EXTERN_IMPORT)
                || sig.abi == crate::sigs::Abi::Intrinsic;
            if declared_only {
                let mut cx = FnCx::new(g, icx, path);
                let (ps, ret) = cx.inst_sig(k).unwrap_or((Vec::new(), TypeId::NONE));
                let ret = cx.ret_of(ret);
                let out = FnOut {
                    closure: None,
                    params: ps,
                    ret,
                    body: None,
                    site_consts: Vec::new(),
                    name_off: None,
                };
                return finish_job(cx, out, name);
            }
            let Some(tb) = body else { return empty(None) };
            let mut flagged: FxHashSet<crate::lower_fn::FlagKey> = FxHashSet::default();
            let mut slotted: FxHashSet<u32> = FxHashSet::default();
            for _ in 0..64 {
                let mut cx = FnCx::new(g, icx, path.clone());
                cx.has_body = true;
                cx.tb = Some(tb);
                cx.sched = g.drops.and_then(|m| m.get(&tb.decl));
                cx.args = k.args.clone();
                cx.cur_n_parent = sig.n_parent;
                cx.raw_self_ty = if decl.kind == DeclKind::Method {
                    g.sigs.impl_def(decl.parent).map(|i| i.self_ty)
                } else {
                    None
                };
                cx.user = !g.builtin_decl(k.decl);
                cx.needs = needs;
                cx.flagged = flagged.clone();
                cx.slotted = slotted.clone();
                cx.setup(decl.node);
                let (ps, ret) = cx.inst_sig(k).unwrap_or((Vec::new(), TypeId::NONE));
                let mut out = cx.lower_body(decl.node, &ps, ret);
                cx.check_asked();
                out.name_off = name;
                let new: Vec<crate::lower_fn::FlagKey> =
                    cx.need_flag.iter().filter(|n| !flagged.contains(*n)).cloned().collect();
                let slots: Vec<u32> = cx.need_slot.iter().copied().filter(|n| !slotted.contains(n)).collect();
                if (new.is_empty() && slots.is_empty()) || cx.err.is_some() {
                    return finish_job(cx, out, name);
                }
                flagged.extend(new);
                slotted.extend(slots);
            }
            empty(Some(LowerError {
                kind: LowerErrorKind::Internal,
                func: path,
                at: String::new(),
                msg: "drop flags did not converge".into(),
            }))
        }
    }
}

fn finish_job(mut cx: FnCx<'_>, out: FnOut, name: Option<(stave_diag::FileId, u32)>) -> JobOut {
    let mut funcs = vec![out];
    let mut cl = std::mem::take(&mut cx.outs);
    cl.sort_by_key(|f| f.closure);
    funcs.extend(cl);
    JobOut {
        funcs,
        reqs: cx.reqs,
        strs: cx.strs,
        intrs: cx.intrs,
        sites: cx.sites,
        local_types: cx.tys.local,
        err: cx.err,
        name,
    }
}

// ---------- assembly ----------

/// Sema → IR type translation into a temporary IR table.
struct Tr {
    tmp: IrTypes,
    memo: FxHashMap<TypeId, stave_ir::TypeId>,
    /// IR temp id → sema type (ADTs, for shapes).
    adt_sema: FxHashMap<stave_ir::TypeId, TypeId>,
}

impl Tr {
    fn tr(&mut self, table: &TypeTable, t: TypeId, depth: u32) -> stave_ir::TypeId {
        if let Some(&r) = self.memo.get(&t) {
            return r;
        }
        let d = depth + 1;
        let key = match table.kind(t).clone() {
            TypeKind::Int { signed, bits } => TypeKey::Int {
                signed,
                bits: match bits {
                    IntBits::B8 => stave_ir::IntBits::B8,
                    IntBits::B16 => stave_ir::IntBits::B16,
                    IntBits::B32 => stave_ir::IntBits::B32,
                    IntBits::B64 => stave_ir::IntBits::B64,
                    IntBits::Size => stave_ir::IntBits::Size,
                },
            },
            TypeKind::Float(f) => TypeKey::Float(match f {
                FloatBits::F16 => stave_ir::FloatBits::F16,
                FloatBits::F32 => stave_ir::FloatBits::F32,
                FloatBits::F64 => stave_ir::FloatBits::F64,
                FloatBits::B16 => stave_ir::FloatBits::B16,
            }),
            TypeKind::Bool => TypeKey::Bool,
            TypeKind::Char => TypeKey::Char,
            TypeKind::Str => TypeKey::Str,
            TypeKind::Ref { mutable, inner } => TypeKey::Ref { mutable, inner: self.tr(table, inner, d) },
            TypeKind::Ptr { mutable, inner } => TypeKey::RawPtr { mutable, inner: self.tr(table, inner, d) },
            TypeKind::Array { elem, len } => {
                TypeKey::Array { elem: self.tr(table, elem, d), len: len.value().unwrap_or(0) }
            }
            TypeKind::Slice { elem, align } => {
                TypeKey::Slice { elem: self.tr(table, elem, d), align: (align != 0).then_some(align) }
            }
            TypeKind::Tuple(ts) => TypeKey::Tuple(ts.iter().map(|&x| self.tr(table, x, d)).collect()),
            TypeKind::Adt { def, args } => TypeKey::Adt {
                def: IrDecl(def.0),
                args: args
                    .iter()
                    .map(|a| match a {
                        GArg::Type(x) => GenericArg::Type(self.tr(table, *x, d)),
                        GArg::Const(ConstArg::Value(v)) => GenericArg::Const(*v as i128),
                        GArg::Const(_) => GenericArg::Const(0),
                    })
                    .collect(),
            },
            TypeKind::Fn { params, ret, features, .. } => {
                let ps = params.iter().map(|&x| self.tr(table, x, d)).collect();
                let r = if matches!(table.kind(ret), TypeKind::Never) {
                    self.tmp.unit()
                } else {
                    self.tr(table, ret, d)
                };
                TypeKey::FnType {
                    params: ps,
                    ret: r,
                    features: FeatureSet::from_bits(features).unwrap_or(FeatureSet::EMPTY),
                }
            }
            TypeKind::Vector { lane, n } => TypeKey::Vec { lane: self.tr(table, lane, d), n: lanes(n) },
            TypeKind::Mask { width, n } => TypeKey::Mask { width, n: lanes(n) },
            TypeKind::Param(i) => TypeKey::Param(i),
            TypeKind::ConstParam(i) => TypeKey::ConstParam(i),
            TypeKind::Range { elem, .. } => {
                let e = self.tr(table, elem, d);
                TypeKey::Tuple(vec![e, e])
            }
            TypeKind::Never | TypeKind::SelfTy | TypeKind::Error => TypeKey::Tuple(Vec::new()),
        };
        let is_adt = matches!(key, TypeKey::Adt { .. });
        let r = self.tmp.intern(key);
        if is_adt {
            self.adt_sema.entry(r).or_insert(t);
        }
        self.memo.insert(t, r);
        r
    }
}

fn lanes(n: crate::prim::Lanes) -> IrLanes {
    match n {
        crate::prim::Lanes::Fixed(k) => IrLanes::Fixed(k),
        crate::prim::Lanes::Native => IrLanes::Native,
    }
}

/// Canonical order of instance args (by final type id; types before consts).
fn cmp_args(a: &[GenericArg], b: &[GenericArg]) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    for (x, y) in a.iter().zip(b) {
        let o = match (x, y) {
            (GenericArg::Type(p), GenericArg::Type(q)) => p.cmp(q),
            (GenericArg::Const(p), GenericArg::Const(q)) => p.cmp(q),
            (GenericArg::Type(_), GenericArg::Const(_)) => Ordering::Less,
            (GenericArg::Const(_), GenericArg::Type(_)) => Ordering::Greater,
        };
        if o != Ordering::Equal {
            return o;
        }
    }
    a.len().cmp(&b.len())
}

#[allow(clippy::too_many_arguments)]
fn assemble(
    prog: &Program,
    sigs: &crate::sigs::Sigs,
    table: &mut TypeTable,
    jobs: &[Job],
    cache: &FxHashMap<Job, usize>,
    outs: Vec<JobOut>,
    statics_list: &[DeclId],
    inherited: &FxHashMap<InstKey, u64>,
    glue_features: &FxHashMap<TypeId, u64>,
) -> Result<(Module, Vec<usize>, Vec<SitePos>), LowerError> {
    let mut tr = Tr { tmp: IrTypes::new(), memo: FxHashMap::default(), adt_sema: FxHashMap::default() };
    // 1. translate every used type; the printed set P
    let mut used: Vec<stave_ir::TypeId> = Vec::new();
    // the types of the values and results of the bodies (a handful, on every value): each is
    // translated and added to `used` once
    let mut seen = vec![false; table.len()];
    let mut job_targs: Vec<Vec<GenericArg>> = Vec::with_capacity(jobs.len());
    for (ji, job) in jobs.iter().enumerate() {
        let targs: Vec<GenericArg> = match job {
            Job::Inst(k) => k
                .args
                .iter()
                .map(|a| match a {
                    GArg::Type(t) => GenericArg::Type(tr.tr(table, *t, 0)),
                    GArg::Const(ConstArg::Value(v)) => GenericArg::Const(*v as i128),
                    GArg::Const(_) => GenericArg::Const(0),
                })
                .collect(),
            Job::Glue(t) => vec![GenericArg::Type(tr.tr(table, *t, 0))],
        };
        job_targs.push(targs.clone());
        let Some(o) = outs.get(ji) else { continue };
        for f in &o.funcs {
            if let Some(b) = &f.body {
                used.extend(targs.iter().filter_map(|a| match a {
                    GenericArg::Type(t) => Some(*t),
                    GenericArg::Const(_) => None,
                }));
                for &p in &f.params {
                    used.push(tr.tr(table, p, 0));
                }
                if let Some(r) = f.ret {
                    used.push(tr.tr(table, r, 0));
                }
                for v in &b.values {
                    let t = TypeId(v.ty.0);
                    if first_use(&mut seen, t) {
                        used.push(tr.tr(table, t, 0));
                    }
                }
                for t in b.insts.ty.iter().flatten() {
                    let t = TypeId(t.0);
                    if first_use(&mut seen, t) {
                        used.push(tr.tr(table, t, 0));
                    }
                }
            }
        }
    }
    let close = |tmp: &IrTypes, seeds: &mut Vec<stave_ir::TypeId>, set: &mut Vec<bool>| {
        while let Some(t) = seeds.pop() {
            if set.len() <= t.index() {
                set.resize(tmp.len().max(t.index() + 1), false);
            }
            if set[t.index()] {
                continue;
            }
            set[t.index()] = true;
            if let Some(k) = tmp.key(t) {
                k.for_each_component(|c| seeds.push(c));
            }
        }
    };
    let mut in_p: Vec<bool> = Vec::new();
    close(&tr.tmp, &mut used, &mut in_p);
    // 2. extras S: shapes, statics, bodiless signatures, bool/usize/u8
    let mut extra: Vec<stave_ir::TypeId> = Vec::new();
    {
        let b = table.intern(TypeKind::Bool);
        let us = table.intern(TypeKind::Int { signed: false, bits: IntBits::Size });
        let u8t = table.intern(TypeKind::Int { signed: false, bits: IntBits::B8 });
        for t in [b, us, u8t] {
            extra.push(tr.tr(table, t, 0));
        }
        for &d in statics_list {
            if let Some(c) = sigs.konst(d) {
                extra.push(tr.tr(table, c.ty, 0));
            }
        }
        for o in &outs {
            for f in &o.funcs {
                if f.body.is_none() {
                    for &p in &f.params {
                        extra.push(tr.tr(table, p, 0));
                    }
                    if let Some(r) = f.ret {
                        extra.push(tr.tr(table, r, 0));
                    }
                }
            }
        }
    }
    let mut in_all = in_p.clone();
    let mut shapes: FxHashMap<stave_ir::TypeId, AdtShape> = FxHashMap::default();
    let mut seeds = extra;
    loop {
        close(&tr.tmp, &mut seeds, &mut in_all);
        // shapes of every ADT in the set
        let adts: Vec<(stave_ir::TypeId, TypeId)> = tr
            .adt_sema
            .iter()
            .filter(|(i, _)| in_all.get(i.index()).copied().unwrap_or(false) && !shapes.contains_key(i))
            .map(|(&i, &s)| (i, s))
            .collect();
        if adts.is_empty() {
            break;
        }
        let mut adts = adts;
        adts.sort();
        for (i, s) in adts {
            let is_enum = match table.kind(s) {
                TypeKind::Adt { def, .. } => sigs.adt(*def).is_some_and(|a| a.is_enum),
                _ => false,
            };
            let vs = crate::mono::adt_fields(sigs, table, s).unwrap_or_default();
            let mut ivs = Vec::with_capacity(vs.len());
            for v in vs {
                let fs: Vec<stave_ir::TypeId> = v.iter().map(|&f| tr.tr(table, f, 0)).collect();
                seeds.extend(fs.iter().copied());
                ivs.push(fs);
            }
            let sh = if is_enum {
                AdtShape::enumeration(ivs)
            } else {
                AdtShape::structure(ivs.into_iter().next().unwrap_or_default())
            };
            shapes.insert(i, sh);
        }
    }
    // 3. canonical order: P, then the rest
    let n_tmp = tr.tmp.len();
    in_p.resize(n_tmp, false);
    in_all.resize(n_tmp, false);
    let cmp = |a: &stave_ir::TypeId, b: &stave_ir::TypeId| {
        tr.tmp.canonical_cmp(*a, *b).unwrap_or(std::cmp::Ordering::Equal)
    };
    let mut p_list: Vec<stave_ir::TypeId> =
        (0..n_tmp).filter(|&i| in_p[i]).map(|i| stave_ir::TypeId(i as u32)).collect();
    p_list.sort_by(cmp);
    let mut s_list: Vec<stave_ir::TypeId> =
        (0..n_tmp).filter(|&i| in_all[i] && !in_p[i]).map(|i| stave_ir::TypeId(i as u32)).collect();
    s_list.sort_by(cmp);
    let mut fin = vec![stave_ir::TypeId(u32::MAX); n_tmp];
    for (k, t) in p_list.iter().chain(s_list.iter()).enumerate() {
        fin[t.index()] = stave_ir::TypeId(k as u32);
    }
    let mut types = IrTypes::new();
    for t in p_list.iter().chain(s_list.iter()) {
        let mut key = tr.tmp.key(*t).cloned().unwrap_or(TypeKey::Bool);
        key.map_components(|c| fin.get(c.index()).copied().unwrap_or(c));
        let id = types.intern(key);
        if let Some(sh) = shapes.get(t) {
            let mut sh = sh.clone();
            for v in &mut sh.variants {
                for f in v.iter_mut() {
                    *f = fin.get(f.index()).copied().unwrap_or(*f);
                }
            }
            types.set_shape(id, sh);
        }
    }
    let ty_final = |t: TypeId, tr: &mut Tr, table: &TypeTable| -> stave_ir::TypeId {
        let x = tr.tr(table, t, 0);
        fin.get(x.index()).copied().unwrap_or(stave_ir::TypeId(u32::MAX))
    };
    // 4. instance order
    let mut inst_order: Vec<usize> = (0..jobs.len()).filter(|&i| matches!(jobs[i], Job::Inst(_))).collect();
    let fin_args = |i: usize| -> Vec<GenericArg> {
        job_targs[i]
            .iter()
            .map(|a| match a {
                GenericArg::Type(t) => GenericArg::Type(fin.get(t.index()).copied().unwrap_or(*t)),
                c => *c,
            })
            .collect()
    };
    let decl_of = |i: usize| match &jobs[i] {
        Job::Inst(k) => k.decl,
        Job::Glue(_) => DeclId::NONE,
    };
    inst_order
        .sort_by(|&a, &b| decl_of(a).cmp(&decl_of(b)).then_with(|| cmp_args(&fin_args(a), &fin_args(b))));
    let mut glue_order: Vec<usize> = (0..jobs.len()).filter(|&i| matches!(jobs[i], Job::Glue(_))).collect();
    glue_order.sort_by(|&a, &b| cmp_args(&fin_args(a), &fin_args(b)));
    let order: Vec<usize> = inst_order.into_iter().chain(glue_order).collect();
    // FuncIds: per job in order: its fn, then its closures
    let mut first_func = vec![u32::MAX; jobs.len()];
    let mut nf = 0u32;
    for &j in &order {
        first_func[j] = nf;
        nf += outs.get(j).map_or(1, |o| o.funcs.len().max(1)) as u32;
    }
    // 5. strings and intrinsics
    let mut all_strs: Vec<Vec<u8>> = outs.iter().flat_map(|o| o.strs.iter().cloned()).collect();
    for &d in statics_list {
        if let Some(v) = sigs.konst(d).and_then(|c| c.value.as_ref()) {
            str_bytes(v, &mut all_strs, &mut FxHashSet::default());
        }
    }
    all_strs.sort_by(|a, b| {
        stave_intern::hash::xxh32(a, 0).cmp(&stave_intern::hash::xxh32(b, 0)).then_with(|| a.cmp(b))
    });
    all_strs.dedup();
    let str_index: FxHashMap<&[u8], u32> =
        all_strs.iter().enumerate().map(|(i, s)| (s.as_slice(), i as u32)).collect();
    let mut all_intrs: Vec<String> = outs.iter().flat_map(|o| o.intrs.iter().cloned()).collect();
    all_intrs.sort();
    all_intrs.dedup();
    // the final id of every sema type translated so far, by sema id (`u32::MAX`: none yet, ask
    // `tr`): the bodies map the type of every value and result
    let mut fin_of = vec![stave_ir::TypeId(u32::MAX); table.len()];
    for (s, x) in &tr.memo {
        if !s.is_local()
            && let Some(slot) = fin_of.get_mut(s.idx())
        {
            *slot = fin.get(x.index()).copied().unwrap_or(stave_ir::TypeId(u32::MAX));
        }
    }
    // 6. functions
    let mut module = Module::new();
    module.strings = all_strs.clone();
    module.intrinsics = all_intrs.clone();
    let mut funcs: Vec<stave_ir::Func> = Vec::with_capacity(nf as usize);
    let mut clauses: Vec<stave_ir::Clauses> = Vec::with_capacity(nf as usize);
    let mut sites: Vec<SiteInfo> = Vec::new();
    let mut site_pos: Vec<SitePos> = Vec::new();
    // per type: the widest vector a value of it holds (0: none)
    let mut held_bits: FxHashMap<TypeId, u32> = FxHashMap::default();
    // Files of built-in packages are named `<package>/<path in package>` (`std/vec.stv`) in the
    // site table, so a binary does not depend on where the library is installed.
    let site_names: FxHashMap<u32, String> = prog
        .files
        .iter()
        .filter_map(|f| {
            let p = prog.packages.get(f.package.0 as usize)?;
            (p.kind == crate::package::PackageKind::Builtin)
                .then(|| (f.id.0, format!("{}/{}", p.name, f.rel_path)))
        })
        .collect();
    let mut outs = outs;
    for (rank, &j) in order.iter().enumerate() {
        let job = &jobs[j];
        let o = outs.get_mut(j);
        let Some(o) = o else { continue };
        let path = match job {
            Job::Inst(k) => fn_path(prog, sigs, k.decl),
            Job::Glue(_) => "drop_glue".to_string(),
        };
        let features = match job {
            Job::Inst(k) => inst_features(sigs, inherited, k),
            Job::Glue(t) => {
                FeatureSet::from_bits(glue_features.get(t).copied().unwrap_or(0)).unwrap_or(FeatureSet::EMPTY)
            }
        };
        let symbol = match job {
            Job::Inst(k) => c_symbol(prog, sigs, k.decl),
            Job::Glue(_) => None,
        };
        let known = match job {
            Job::Inst(k) => prog.known.of(k.decl),
            Job::Glue(_) => None,
        };
        let type_args = fin_args(j);
        let reqs = o.reqs.clone();
        let job_sites = o.sites.clone();
        let strs = o.strs.clone();
        let intrs = o.intrs.clone();
        let job_name = o.name;
        for (k, f) in o.funcs.iter_mut().enumerate() {
            let instance = match f.closure {
                Some(c) => InstanceId(vec![rank as u32, c]),
                None => InstanceId(vec![rank as u32]),
            };
            let params: Vec<stave_ir::TypeId> =
                f.params.iter().map(|&p| ty_final(p, &mut tr, table)).collect();
            let ret = f.ret.map(|r| ty_final(r, &mut tr, table));
            // sites of this fn: rows by source offset
            let mut local_sites: Vec<u32> = Vec::new();
            if let Some(b) = &f.body {
                local_sites.extend(b.insts.site.iter().filter(|s| **s != Site::NONE).map(|s| s.0));
                for &c in &f.site_consts {
                    if let Some(stave_ir::Const::Int(v)) = b.consts.get(c.index()) {
                        local_sites.push(*v as u32);
                    }
                }
            }
            let name_at = match f.closure {
                None => job_name,
                Some(_) => f.name_off,
            };
            local_sites.sort_by_key(|&s| job_sites.get(s as usize).map(|x| (x.0.0, x.1)));
            local_sites.dedup();
            let mut site_map: FxHashMap<u32, u32> = FxHashMap::default();
            let mut rows: Vec<((u32, u32), Option<u32>)> = local_sites
                .iter()
                .filter_map(|&s| job_sites.get(s as usize).map(|x| ((x.0.0, x.1), Some(s))))
                .collect();
            if let Some((file, off)) = name_at
                && !rows.iter().any(|r| r.0 == (file.0, off))
            {
                rows.push(((file.0, off), None));
            }
            rows.sort();
            let mut name_site = Site::NONE;
            for ((file, off), s) in rows {
                let row = sites.len() as u32;
                let (fp, line, col) =
                    match prog.source_map.locate(stave_diag::Span::new(stave_diag::FileId(file), off, 0)) {
                        Ok(l) => (
                            site_names.get(&file).cloned().unwrap_or_else(|| l.path.to_string()),
                            l.line,
                            l.col,
                        ),
                        Err(_) => (String::from("<unknown>"), 0, 0),
                    };
                sites.push(SiteInfo { file: fp, line, col, func: path.clone() });
                site_pos.push((stave_diag::FileId(file), off));
                if let Some(s) = s {
                    site_map.insert(s, row);
                }
                if name_at == Some((stave_diag::FileId(file), off)) {
                    name_site = Site(row);
                }
            }
            // A fn holds by value only vectors its features hold: the type checker gated each
            // body, decided a generic instance's features from its type arguments, and drop
            // glue's from its type ([`crate::inherit`]). Anything else is a checker bug: an
            // internal error in every build, never code left to the backend.
            if let Some(b) = &f.body {
                let width = crate::gates::feature_vector_bits(prog.target.baseline().union(features));
                let held =
                    f.params.iter().copied().chain(f.ret).chain(b.values.iter().map(|v| TypeId(v.ty.0)));
                for t in held.filter(|t| !t.is_none() && !t.is_local()) {
                    let bits = *held_bits.entry(t).or_insert_with(|| {
                        let mut ov = Overlay::new(&*table);
                        crate::inherit::held_wide(&mut ov, sigs, t).map_or(0, |w| w.1)
                    });
                    if bits > width {
                        return Err(LowerError {
                            kind: LowerErrorKind::Internal,
                            func: path.clone(),
                            at: String::new(),
                            msg: format!(
                                "holds a {bits}-bit vector outside its features ({width}-bit registers)"
                            ),
                        });
                    }
                }
            }
            let body = f.body.take().map(|mut b| {
                for &c in &f.site_consts {
                    if let Some(stave_ir::Const::Int(v)) = b.consts.get_mut(c.index()) {
                        *v = site_map.get(&(*v as u32)).copied().unwrap_or(u32::MAX) as i128;
                    }
                }
                let tyf = |t: u32| match fin_of.get(TypeId(t).idx()) {
                    Some(&f) if f.0 != u32::MAX && !TypeId(t).is_local() => f,
                    _ => ty_final_pure(&tr, &fin, t),
                };
                let func = |i: u32| -> FuncId {
                    match reqs.get(i as usize) {
                        Some(Req::Inst(k)) => {
                            FuncId(cache.get(&Job::Inst(k.clone())).map_or(u32::MAX, |&jj| first_func[jj]))
                        }
                        Some(Req::Glue(t)) => {
                            FuncId(cache.get(&Job::Glue(*t)).map_or(u32::MAX, |&jj| first_func[jj]))
                        }
                        Some(Req::Closure(c)) => FuncId(first_func[j] + 1 + *c),
                        None => FuncId(u32::MAX),
                    }
                };
                let string = |i: u32| -> stave_ir::StrId {
                    stave_ir::StrId(
                        strs.get(i as usize)
                            .and_then(|s| str_index.get(s.as_slice()))
                            .copied()
                            .unwrap_or(u32::MAX),
                    )
                };
                let intrinsic = |i: u32| -> stave_ir::IntrinsicId {
                    stave_ir::IntrinsicId(
                        intrs
                            .get(i as usize)
                            .and_then(|n| all_intrs.iter().position(|x| x == n))
                            .map_or(u32::MAX, |p| p as u32),
                    )
                };
                let site = |s: u32| -> Site { site_map.get(&s).map_or(Site::NONE, |&r| Site(r)) };
                let rm = crate::lower_post::Remap {
                    ty: &tyf,
                    func: &func,
                    string: &string,
                    intrinsic: &intrinsic,
                    site: &site,
                };
                crate::lower_post::remap(&mut b, &rm);
                b
            });
            let _ = k;
            clauses.push(match (job, f.closure) {
                (Job::Inst(_), Some(_)) => {
                    stave_ir::Clauses { kind: stave_ir::ClauseKind::Closure, ..Default::default() }
                }
                (Job::Inst(key), None) => match sigs.fn_sig(key.decl) {
                    Some(s) => stave_ir::Clauses {
                        kind: stave_ir::ClauseKind::Item,
                        result: s.origins.iter().map(|o| (u32::from(o.param), o.copy)).collect(),
                        params: s
                            .param_origins
                            .iter()
                            .map(|p| (u32::from(p.target), u32::from(p.source), p.copy))
                            .collect(),
                    },
                    None => stave_ir::Clauses::default(),
                },
                (Job::Glue(_), _) => stave_ir::Clauses::default(),
            });
            funcs.push(stave_ir::Func {
                instance,
                path: path.clone(),
                type_args: if f.closure.is_some() { Vec::new() } else { type_args.clone() },
                params,
                ret,
                features,
                body,
                name_site,
                symbol: if f.closure.is_some() { None } else { symbol.clone() },
                known: if f.closure.is_some() { None } else { known },
            });
        }
    }
    module.funcs = funcs;
    module.clauses = clauses;
    module.types = types;
    module.sites = sites;
    // 7. statics
    let layouts = stave_ir::layout::Layouts::compute(&module.types);
    for &d in statics_list {
        let Some(c) = sigs.konst(d) else { continue };
        let ty = ty_final(c.ty, &mut tr, table);
        let size = layouts.size_of(ty).unwrap_or(0) as usize;
        let mut init = vec![0u8; size];
        let mut refs = Vec::new();
        if let Some(v) = &c.value {
            let mut out = ValOut { bytes: &mut init, refs: &mut refs, strs: &str_index };
            write_val(&module.types, &layouts, ty, v, &mut out, 0);
        }
        refs.sort_by_key(|r: &stave_ir::DataRef| r.offset);
        module.statics.push(Static { path: prog.decl_path(d), ty, init, refs });
    }
    Ok((module, order, site_pos))
}

/// Whether sema type `t` is met for the first time (always for an id outside `seen`).
fn first_use(seen: &mut [bool], t: TypeId) -> bool {
    if t.is_local() || t.is_none() {
        return true;
    }
    match seen.get_mut(t.idx()) {
        Some(s) => !std::mem::replace(s, true),
        None => true,
    }
}

fn ty_final_pure(tr: &Tr, fin: &[stave_ir::TypeId], t: u32) -> stave_ir::TypeId {
    match tr.memo.get(&TypeId(t)) {
        Some(x) => fin.get(x.index()).copied().unwrap_or(stave_ir::TypeId(u32::MAX)),
        None => stave_ir::TypeId(u32::MAX),
    }
}

/// The most value cells a const's value is built from in the code where it is used; a larger
/// const is read from its image, a read-only static ([`crate::lower_expr`] `const_use`).
pub(crate) const CONST_INLINE_CELLS: u64 = 64;

/// The string bytes a const value holds, appended to `out`. A shared part (`seen`, by address)
/// is visited once: the value's distinct parts, not its leaves.
fn str_bytes(v: &crate::consteval::Val, out: &mut Vec<Vec<u8>>, seen: &mut FxHashSet<usize>) {
    use crate::consteval::Val;
    match v {
        Val::Str(s) => out.push(s.to_vec()),
        Val::Agg(vs) | Val::Variant(_, vs) if seen.insert(vs.as_ptr() as usize) => {
            vs.iter().for_each(|x| str_bytes(x, out, seen));
        }
        Val::Rep(v, _) if seen.insert(std::sync::Arc::as_ptr(v) as usize) => str_bytes(v, out, seen),
        _ => {}
    }
}

/// Where [`write_val`] writes a static's image: its bytes, its address words, and the string
/// table ([`str_bytes`] put every string of the value in it).
struct ValOut<'a> {
    bytes: &'a mut [u8],
    refs: &'a mut Vec<stave_ir::DataRef>,
    strs: &'a FxHashMap<&'a [u8], u32>,
}

/// Writes a const value's bytes at `off` (stave-ir layout); a string is an address word (to
/// its bytes in the string table) and its length.
fn write_val(
    types: &IrTypes,
    lay: &stave_ir::layout::Layouts,
    t: stave_ir::TypeId,
    v: &crate::consteval::Val,
    out: &mut ValOut<'_>,
    off: usize,
) {
    use crate::consteval::Val;
    use stave_ir::layout::Shape;
    let Ok(l) = lay.get_layout(t) else { return };
    // nothing to write; and a zero-size value may have any number of leaves
    if l.size == 0 {
        return;
    }
    let put = |out: &mut ValOut<'_>, at: usize, n: usize, bits: u64| {
        for (k, byte) in bits.to_le_bytes().iter().enumerate().take(n.min(8)) {
            if let Some(x) = out.bytes.get_mut(at + k) {
                *x = *byte;
            }
        }
    };
    match (&l.shape, v) {
        (Shape::Scalar, Val::Int(i)) => match types.key(t) {
            Some(TypeKey::Float(stave_ir::FloatBits::F32)) => put(out, off, 4, (*i as f32).to_bits() as u64),
            Some(TypeKey::Float(stave_ir::FloatBits::F64)) => put(out, off, 8, (*i as f64).to_bits()),
            _ => put(out, off, l.size as usize, *i as u64),
        },
        (Shape::Scalar, Val::Float(x)) => match types.key(t) {
            Some(TypeKey::Float(stave_ir::FloatBits::F32)) => put(out, off, 4, (*x as f32).to_bits() as u64),
            _ => put(out, off, 8, x.to_bits()),
        },
        (Shape::Scalar, Val::Bool(b)) => put(out, off, 1, *b as u64),
        (Shape::Scalar, Val::Char(c)) => put(out, off, 4, *c as u64),
        (Shape::Fat, Val::Str(s)) => {
            if let Some(&sid) = out.strs.get(&s[..]) {
                let target = stave_ir::DataTarget::Str(stave_ir::StrId(sid));
                out.refs.push(stave_ir::DataRef { offset: off as u32, target, addend: 0 });
            }
            put(out, off + 8, 8, s.len() as u64);
        }
        (Shape::Fields(fs), Val::Agg(vs)) => {
            for ((fo, ft), fv) in fs.iter().zip(vs.iter()) {
                write_val(types, lay, *ft, fv, out, off + *fo as usize);
            }
        }
        (Shape::Array { elem, stride, .. }, Val::Agg(vs)) => {
            for (k, fv) in vs.iter().enumerate() {
                write_val(types, lay, *elem, fv, out, off + k * *stride as usize);
            }
        }
        (Shape::Array { elem, stride, .. }, Val::Rep(v, n)) => {
            // the element once, then its bytes (and address words) copied: the buffer starts
            // zeroed, so an all-zero element needs nothing more
            let (stride, n) = (*stride as usize, *n as usize);
            let refs0 = out.refs.len();
            write_val(types, lay, *elem, v, out, off);
            let new_refs: Vec<stave_ir::DataRef> = out.refs[refs0..].to_vec();
            let end = (off + stride).min(out.bytes.len());
            if n < 2
                || stride == 0
                || (new_refs.is_empty() && out.bytes[off.min(end)..end].iter().all(|b| *b == 0))
            {
                return;
            }
            let mut done = 1;
            while done < n {
                let k = done.min(n - done);
                let (src, dst) = (off, off + done * stride);
                if dst + k * stride > out.bytes.len() {
                    break;
                }
                out.bytes.copy_within(src..src + k * stride, dst);
                done += k;
            }
            for i in 1..n {
                for r in &new_refs {
                    out.refs.push(stave_ir::DataRef { offset: r.offset + (i * stride) as u32, ..*r });
                }
            }
        }
        (Shape::Enum(vs), Val::Variant(vi, fs)) => {
            put(out, off, 4, *vi as u64);
            if let Some(fl) = vs.get(*vi as usize) {
                for ((fo, ft), fv) in fl.iter().zip(fs.iter()) {
                    write_val(types, lay, *ft, fv, out, off + *fo as usize);
                }
            }
        }
        _ => {}
    }
}

/// Adds the interpreter's stand-in for the compiler-emitted `__stave_main` stub
/// (contracts/intrinsics.md `main_addr`): `fn(arg: usize) -> usize` calling `main_path` and
/// returning its status (`i32` sign-extended, 0 for a unit `main`). Not part of the checked IR.
pub fn interp_main_stub(m: &mut Module, main_path: &str) -> Option<FuncId> {
    let main = m.find_func(main_path)?;
    let mret = m.func(main)?.ret;
    let us = m.types.usize();
    let mut b = stave_ir::FuncBuilder::new(&[us]);
    let r = b.call(main, mret, &[]);
    match r {
        Some(v) => {
            let x = b.cast(stave_ir::CastKind::Wrap, us, v);
            b.ret(Some(stave_ir::Operand::Value(x)));
        }
        None => {
            let z = b.int(0);
            b.ret(Some(z));
        }
    }
    let body = b.finish().ok()?;
    let f = m.declare(InstanceId(vec![u32::MAX]), "__stave_main", vec![us], Some(us));
    m.define(f, body);
    Some(f)
}

/// The site table in the layout of contracts/intrinsics.md for a load at address `base`: the
/// count, the 32-byte rows, then the UTF-8 file and fn paths the rows point to (for the
/// interpreter; the backends emit `__stave_site_table` themselves).
pub fn site_table_bytes(m: &Module, base: u64) -> Vec<u8> {
    let n = m.sites.len();
    let head = 8 + 32 * n;
    let mut out = vec![0u8; head];
    out[..8].copy_from_slice(&(n as u64).to_le_bytes());
    let mut strs: Vec<u8> = Vec::new();
    let mut interned: FxHashMap<String, u64> = FxHashMap::default();
    for (i, r) in m.sites.iter().enumerate() {
        let mut addr_of = |s: &str| -> u64 {
            if let Some(&a) = interned.get(s) {
                return a;
            }
            let a = base + head as u64 + strs.len() as u64;
            strs.extend_from_slice(s.as_bytes());
            interned.insert(s.to_string(), a);
            a
        };
        let fa = addr_of(&r.file);
        let na = addr_of(&r.func);
        let o = 8 + 32 * i;
        out[o..o + 8].copy_from_slice(&fa.to_le_bytes());
        out[o + 8..o + 12].copy_from_slice(&(r.file.len() as u32).to_le_bytes());
        out[o + 12..o + 16].copy_from_slice(&r.line.to_le_bytes());
        out[o + 16..o + 24].copy_from_slice(&na.to_le_bytes());
        out[o + 24..o + 28].copy_from_slice(&(r.func.len() as u32).to_le_bytes());
        out[o + 28..o + 32].copy_from_slice(&r.col.to_le_bytes());
    }
    out.extend_from_slice(&strs);
    out
}
