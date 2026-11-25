//! The stage0 back end: lowering to the checked IR (mono + lower, stave-sema),
//! the debug code generators (x86-64: stave-codegen-x64, AArch64: stave-codegen-a64), and the static
//! ELF link (stave-obj).
//!
//! - [`lower`]: [`stave_sema::lower`] plus the IR verifier. Lowering errors that are all of kind
//!   [`LowerErrorKind::Unsupported`] (a construct stage0 does not lower yet) are
//!   [`Failure::Unsupported`]; any other lowering or verifier error is a compiler bug
//!   ([`Failure::Internal`]).
//! - [`lower`] then decides which bounds and contract checks the program keeps, the same way in
//!   every build mode, on the canonical module ([`canonical`]; level `lowered`): per function, the
//!   bounds audit (an element access without a check or proof is a compiler bug) and the proofs
//!   of removable checks ([`stave_ir::bounds::analyze_func`]); the witness checker on every
//!   proof ([`stave_ir::witness::check`]; a refused witness is a compiler bug); then the removal
//!   of the accepted checks, which renumbers the bodies it changed so that the module stays
//!   canonical ([`stave_ir::canon::restore_by`]); then the IR verifier again, in every compiler
//!   profile ([`checked_module`]). Loading IR text runs the same steps. The module it returns is
//!   the checked IR: what the `ir` dump prints, the interpreter runs, and both code generators
//!   get.
//! - [`codegen_link`]: release builds first run the IR passes of stave-opt and renumber their
//!   result canonically, then the IR verifier
//!   and the bounds audit again on their result (a failure is a compiler bug); then per-function
//!   code generation in parallel (rayon on the caller's pool for AArch64; stave-codegen-x64 uses
//!   its own scoped threads), then the link. Debug builds use stave-obj's debug layout (padded
//!   slots, jump table), release builds the packed one. Backend errors on valid IR (vectors,
//!   f16) are [`Failure::Unsupported`]. Given the call graph, it also computes the stack bound
//!   from the frame sizes the backend reports ([`crate::stack`]) and writes the stack table.
//! - [`main_path`]: the checked-IR path of the root package's `main`.

use std::fmt;
use std::time::Instant;

use rayon::prelude::*;
use stave_ir::bounds::{
    Analysis, CallFinding, Claims, Finding, Proved, Unguarded, analyze_func_in, call_findings, claim_func,
    findings,
};
use stave_ir::canon::Edits;
use stave_ir::text::Level;
use stave_ir::witness::FnWitness;
use stave_ir::witness::check::{Removal, check_module_by, remove, remove_whole};
use stave_ir::{FuncId, Module};
use stave_obj::{FuncCode, ImageBuilder, Machine, ObjectImage, StackUse};
use stave_sema::{DeclKind, LowerErrorKind, Mode, Program, Typeck};
use stave_target::{Arch, Triple};

use crate::pipeline::Stage;
use crate::{prof, stack};

/// Why the back end produced no executable for a program without error diagnostics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Failure {
    /// The program uses something stage0 cannot compile yet (exit 2, "not implemented yet";
    /// conformance: not run, needs `stage`).
    Unsupported { stage: Stage, message: String },
    /// A compiler bug (exit 3).
    Internal { stage: Stage, message: String },
    /// Nothing to build: the root package has no `main` (exit 2).
    NoMain,
    /// The target has no stage0 back end (`gpu-sm86`).
    Target(Triple),
    /// No target was given and the host is not a stage0 target (neither x86-64 nor AArch64).
    NoHostTarget,
}

impl Failure {
    pub fn stage(&self) -> Stage {
        match self {
            Failure::Unsupported { stage, .. } | Failure::Internal { stage, .. } => *stage,
            Failure::NoMain => Stage::Link,
            Failure::Target(_) | Failure::NoHostTarget => Stage::Codegen,
        }
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Failure::Unsupported { stage, message } => {
                write!(f, "`{}` is not implemented yet for this program: {message}", stage.name())
            }
            Failure::Internal { stage, message } => write!(f, "{} failed: {message}", stage.name()),
            Failure::NoMain => {
                f.write_str("the program has no `fn main` in its root module: nothing to build")
            }
            Failure::Target(t) => write!(f, "stage0 has no code generator for target {t} yet"),
            Failure::NoHostTarget => write!(
                f,
                "the host ({}) is not a stage0 target; give one with --target",
                std::env::consts::ARCH
            ),
        }
    }
}

/// Most errors listed in a [`Failure`] message.
const MAX_LISTED: usize = 8;

fn listed<T: fmt::Display>(errs: &[T]) -> String {
    let mut s: Vec<String> = errs.iter().take(MAX_LISTED).map(T::to_string).collect();
    if errs.len() > MAX_LISTED {
        s.push(format!("... and {} more", errs.len() - MAX_LISTED));
    }
    s.join("; ")
}

/// The checked-IR path of `main` in the root package's root module, if it has one.
pub fn main_path(prog: &Program) -> Option<String> {
    let pkg = prog.packages.iter().find(|p| p.is_root)?;
    let name = prog.strings.lookup("main")?;
    let d = prog.module_item(pkg.root_module, name)?;
    let decl = prog.decl(d)?;
    (decl.kind == DeclKind::Fn).then(|| prog.decl_path(d))
}

/// E0108: a program (`bin` package or single file) whose root module has no `fn main`,
/// at offset 0 of the root module. `None` for libraries and when `main` exists.
pub fn no_main(prog: &Program) -> Option<stave_diag::Diagnostic> {
    use stave_sema::package::PackageKind;
    let pkg = prog.packages.iter().find(|p| p.is_root)?;
    if matches!(pkg.kind, PackageKind::Lib | PackageKind::Builtin) || main_path(prog).is_some() {
        return None;
    }
    let file = prog.module(pkg.root_module)?.file;
    stave_diag::Diagnostic::new("E0108", None, &[], Some(stave_diag::Span::new(file, 0, 0)), &[]).ok()
}

/// The checked IR of a program and the proofs of the bounds checks it no longer has.
pub struct Lowered {
    pub module: Module,
    /// Per function that lost a check: the witness the checker accepted and the rule behind
    /// each row (function order; instruction indices are those before the removal).
    pub removed: Vec<Proved>,
    /// Per function: the call-graph nodes it stands for ([`stave_sema::lower_mapped`]).
    pub nodes_of_func: Vec<Vec<u32>>,
    /// The canonical module before the removal of proven checks (level `lowered`), when asked
    /// for.
    pub lowered: Option<Module>,
}

/// Lowers (mono + lower), verifies, and removes the bounds checks that have a checked witness.
/// The program must have no error diagnostics. `drops`: the borrow checker's drop schedules; a
/// drop the lowering emits that its body's schedule lacks fails the build as a compiler bug.
pub fn lower(
    prog: &Program,
    tc: &Typeck,
    mode: Mode,
    borrows: bool,
    drops: Option<&stave_sema::drops::Schedules>,
) -> Result<Lowered, Failure> {
    lower_keeping(prog, tc, mode, borrows, drops, false)
}

/// [`lower`], keeping with `keep_lowered` the canonical module as it stands after the verifier
/// and before the removal of proven checks ([`Lowered::lowered`]).
pub fn lower_keeping(
    prog: &Program,
    tc: &Typeck,
    mode: Mode,
    borrows: bool,
    drops: Option<&stave_sema::drops::Schedules>,
    keep_lowered: bool,
) -> Result<Lowered, Failure> {
    let (lowered, tm) = stave_sema::lower_mapped(prog, tc, mode, borrows, drops);
    prof::part(".rounds", tm.rounds);
    prof::part(".assemble", tm.assemble);
    prof::mark("lower");
    let (mut m, mut nodes_of_func) = lowered.map_err(|es| lower_failure(&es))?;
    m.main = main_path(prog).and_then(|p| m.find_func(&p));
    verified(&m, Stage::Lower, Level::Lowered)?;
    prof::mark("verify");
    // The removal reads the canonical module, level `lowered`: the proofs read a constant by its
    // pool entry, so they must see the pool the text carries (one entry per distinct constant),
    // or loading the `lowered` text would remove other checks than the build does.
    canonical_moving(&mut m, Some(&mut nodes_of_func));
    let lowered = keep_lowered.then(|| m.clone());
    let r = checked_module(&mut m, Some(&mut nodes_of_func))?;
    let remap = r.remap;
    let mut removed: Vec<Proved> = r.proved.into_iter().filter(|p| !p.witness.removed.is_empty()).collect();
    for p in &mut removed {
        p.witness.func = remap.get(p.witness.func.index()).copied().unwrap_or(p.witness.func);
    }
    removed.sort_by_key(|p| p.witness.func);
    Ok(Lowered { module: m, removed, nodes_of_func, lowered })
}

/// [`canonical`], the rows of `nodes_of_func` moving with their fns. A build and a load of IR
/// text run it on the module of level `lowered`, the load on the module of any level.
pub(crate) fn canonical_moving(m: &mut Module, nodes_of_func: Option<&mut Vec<Vec<u32>>>) {
    let remap = canonical(m);
    if let Some(n) = nodes_of_func {
        stave_ir::canon::permute_by_func(n, &remap);
    }
    prof::mark("canonical");
}

/// The IR verifier by the rules of `level` on the module a build or a load hands on (the checked
/// module, or loaded text of level `checked` or `opt` after its canonical numbering), in every
/// profile of the compiler.
pub(crate) fn verified_final(m: &Module, level: Level) -> Result<(), Failure> {
    verified(m, Stage::Lower, level)?;
    prof::mark("checked-verify");
    Ok(())
}

/// From the canonical module of level `lowered` to the checked module, for a build and for a load
/// of `lowered` text alike: the removal of proven checks ([`remove_proven_checks`] on the whole
/// program), which keeps the module canonical, the unchecked entries' call-graph nodes, then
/// [`verified_final`]. Returns the removal's records (ids before the renumbering) with the
/// renumbering ([`CheckRemoval::remap`]).
pub(crate) fn checked_module(
    m: &mut Module,
    nodes_of_func: Option<&mut Vec<Vec<u32>>>,
) -> Result<CheckRemoval, Failure> {
    let r = remove_proven_checks(m, true)?;
    if let Some(n) = nodes_of_func {
        extend_nodes(n, &r.removal);
        stave_ir::canon::permute_by_func(n, &r.remap);
    }
    verified_final(m, Level::Checked)?;
    Ok(r)
}

/// [`stave_ir::canon::canonicalize`] with the bodies rewritten in parallel. Every module that is
/// printed, interpreted or given to a code generator is canonical: lowering's result, and again
/// the release passes' result.
pub fn canonical(m: &mut Module) -> Vec<FuncId> {
    stave_ir::canon::canonicalize_by(m, |funcs, one| funcs.par_iter_mut().for_each(one))
}

/// Gives every unchecked entry that is a function of its own the call-graph nodes of the
/// function it is an entry of. The new functions follow the lowered ones in the order of
/// [`Removal::unchecked`].
pub(crate) fn extend_nodes(nodes_of_func: &mut Vec<Vec<u32>>, removal: &Removal) {
    for &(f, entry) in &removal.unchecked {
        if entry != f && entry.index() == nodes_of_func.len() {
            let nodes = nodes_of_func.get(f.index()).cloned().unwrap_or_default();
            nodes_of_func.push(nodes);
        }
    }
}

/// The failure for lowering errors: all [`LowerErrorKind::Unsupported`] is
/// [`Failure::Unsupported`], anything else a compiler bug.
pub(crate) fn lower_failure(es: &[stave_sema::LowerError]) -> Failure {
    let message = listed(es);
    if es.iter().all(|e| e.kind == LowerErrorKind::Unsupported) {
        Failure::Unsupported { stage: Stage::Lower, message }
    } else {
        Failure::Internal { stage: Stage::Lower, message }
    }
}

/// The IR verifier on `m` by the rules of `level`; an error is a compiler bug of `stage`.
pub(crate) fn verified(m: &Module, stage: Stage, level: Level) -> Result<(), Failure> {
    verify(m, level).map_err(|es| {
        let named: Vec<String> = es
            .iter()
            .map(|e| {
                let f = e.func.and_then(|f| m.func(f)).map_or(String::new(), |f| format!("{}: ", f.path));
                format!("{f}{e}")
            })
            .collect();
        Failure::Internal { stage, message: format!("IR verifier: {}", listed(&named)) }
    })
}

/// The bounds audit on `m`; an unguarded element access is a compiler bug of `stage`.
fn audited(m: &Module, stage: Stage) -> Result<(), Failure> {
    let per_fn: Vec<Vec<Unguarded>> = (0..m.funcs.len() as u32)
        .into_par_iter()
        .map(|f| stave_ir::bounds::audit_func(m, FuncId(f)).1)
        .collect();
    let bad: Vec<Unguarded> = per_fn.into_iter().flatten().collect();
    if bad.is_empty() {
        Ok(())
    } else {
        Err(Failure::Internal { stage, message: format!("bounds audit: {}", listed(&bad)) })
    }
}

/// What [`remove_proven_checks`] decided.
pub(crate) struct CheckRemoval {
    /// The accepted witnesses, per function that has one.
    pub proved: Vec<Proved>,
    /// The contract checks proved or refuted (the refuted ones stay).
    pub found: Vec<Finding>,
    /// The calls that always fail a `requires` of their callee.
    pub calls: Vec<CallFinding>,
    /// The unchecked entries made and the calls that go to them.
    pub removal: Removal,
    /// Where each function went (`old -> new`, the new unchecked entries after the others):
    /// with `whole` the removal renumbers the changed bodies so that a canonical module stays
    /// canonical; else the identity.
    pub remap: Vec<FuncId>,
}

/// The one step that decides which bounds and contract checks a build keeps and which calls
/// enter their callee past its `requires` checks; it does not look at the build mode. Per
/// function, in parallel: the summary it states to its callers, then the audit that every
/// element access has its check or a proof, and the witnesses of the checks that are provably
/// redundant, of the contract checks that always fail, and of the calls that prove or refute
/// their callee's `requires`. Then the witness checker, which shares no code with the proof
/// search, validates every witness, and only the checks it accepted are deleted and only the
/// calls it accepted are retargeted. `whole`: `m` is the whole program of a build, not the part
/// of it the effect pass lowers; only then may a function lose its checked entry. When `whole`,
/// `m` must be canonical on entry (the removal renumbers only what it edited); it is canonical on
/// return.
pub(crate) fn remove_proven_checks(m: &mut Module, whole: bool) -> Result<CheckRemoval, Failure> {
    let internal = |message: String| Failure::Internal { stage: Stage::Lower, message };
    let stated = Claims::from_rows(
        (0..m.funcs.len() as u32).into_par_iter().map(|f| claim_func(m, FuncId(f))).collect(),
    );
    let found: Vec<Analysis> = (0..m.funcs.len() as u32)
        .into_par_iter()
        .map(|f| analyze_func_in(m, FuncId(f), Some(&stated)))
        .collect();
    let mut proved = Vec::new();
    let mut bad = Vec::new();
    for a in found {
        bad.extend(a.unguarded);
        proved.extend(a.proved);
    }
    if !bad.is_empty() {
        return Err(internal(format!("bounds audit: {}", listed(&bad))));
    }
    prof::mark("bounds-analyze");
    remove_checked(m, proved, whole)
}

/// Has the witness checker validate every proof (functions in parallel, each reading the
/// summaries of all; it accepts all or nothing), then deletes the checks and retargets the calls
/// it accepted. A witness it refuses is a compiler bug, and then nothing is changed. The contract
/// checks proved or refuted and the calls refuted are read before the removal. When `whole`, `m`
/// must be canonical on entry, as for [`remove_proven_checks`].
fn remove_checked(m: &mut Module, proved: Vec<Proved>, whole: bool) -> Result<CheckRemoval, Failure> {
    let internal = |message: String| Failure::Internal { stage: Stage::Lower, message };
    if cfg!(debug_assertions) && whole {
        assert!(
            stave_ir::canon::is_canonical(m),
            "the removal of proven checks of a whole program needs a canonical module on entry"
        );
    }
    let witnesses: Vec<&FnWitness> = proved.iter().map(|p| &p.witness).collect();
    let checked = {
        let m: &Module = m;
        check_module_by(m, &witnesses, |one| (0..witnesses.len()).into_par_iter().map(one).collect())
            .map_err(|refused| internal(format!("bounds witness: {}", listed(&refused))))?
    };
    prof::mark("witness-check");
    // A whole program stays canonical: the removal changes only the bodies of the functions with
    // an accepted check (a contract check's is its failure call), entry or call, and adds the
    // unchecked entries, so only those are renumbered.
    let edits = whole.then(|| {
        let mut edited = vec![false; m.funcs.len()];
        for c in &checked {
            if (!c.insts().is_empty() || !c.calls().is_empty() || c.has_entry())
                && let Some(x) = edited.get_mut(c.func().index())
            {
                *x = true;
            }
        }
        let e = Edits::new(m, edited);
        prof::mark("bounds-uses");
        e
    });
    let found: Vec<Finding> = proved.iter().flat_map(|p| findings(m, p)).collect();
    let calls: Vec<CallFinding> = proved.iter().flat_map(|p| call_findings(m, p)).collect();
    let removal = if whole { remove_whole(m, &checked) } else { remove(m, &checked) };
    if let Some(f) = removal.refused {
        return Err(internal(format!("bounds witness: fn #{}: not one accepted set of rows", f.0)));
    }
    prof::mark("bounds-remove");
    let remap = match edits {
        Some(e) => {
            // The partial renumbering must give exactly what the full one gives, module and map.
            let full = cfg!(debug_assertions).then(|| {
                let mut full_m = m.clone();
                let map = canonical(&mut full_m);
                (full_m, map)
            });
            let remap = stave_ir::canon::restore_by(m, &e, |funcs, one| {
                funcs.par_iter_mut().enumerate().for_each(|(i, f)| one(i, f))
            });
            if let Some((full_m, map)) = full {
                assert!(
                    remap == map && *m == full_m,
                    "the partial renumbering after the removal of proven checks differs from the full one"
                );
            }
            prof::mark("bounds-renumber");
            remap
        }
        None => (0..m.funcs.len() as u32).map(FuncId).collect(),
    };
    Ok(CheckRemoval { proved, found, calls, removal, remap })
}

/// [`stave_ir::verify_module_at`] with the functions verified in parallel (same errors, same
/// order).
fn verify(m: &Module, level: Level) -> Result<(), Vec<stave_ir::VerifyError>> {
    let mut errs = stave_ir::verify::verify_types(m)?;
    let per_fn: Vec<Vec<stave_ir::VerifyError>> = (0..m.funcs.len() as u32)
        .into_par_iter()
        .map(|f| stave_ir::verify::verify_func_body(m, FuncId(f), level))
        .collect();
    errs.extend(per_fn.into_iter().flatten());
    if errs.is_empty() { Ok(()) } else { Err(errs) }
}

/// A linked executable and the back-end timings.
pub struct Built {
    pub image: ObjectImage,
    pub codegen_ms: f64,
    pub link_ms: f64,
    /// Functions with a body (compiled).
    pub funcs: u64,
    /// The stack bound (when the caller gave the call graph).
    pub stack: Option<stack::Report>,
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// The image layout of a debug or release build for `machine`.
pub fn image_options(machine: Machine, release: bool) -> stave_obj::Options {
    if release { stave_obj::Options::release(machine) } else { stave_obj::Options::debug(machine) }
}

/// Generates code for every function of `m` and links the static executable whose `main` is
/// [`Module::main`] (`__stave_main`; contracts/intrinsics.md; none: [`Failure::NoMain`]). With
/// `stack`, the stack table holds the computed bound; without, it is empty (every stack gets the
/// `[run] stack` default). `m` is canonical ([`canonical`]), as [`lower`] returns it: the code
/// generators read functions, blocks and values in id order, so only the canonical numbering
/// makes the image a function of the module's content. A module whose functions are out of instance order is refused.
pub fn codegen_link(
    m: &Module,
    triple: Triple,
    release: bool,
    threads: Option<usize>,
    stack: Option<&stack::Input>,
) -> Result<Built, Failure> {
    generate(m, triple, if release { Passes::Run } else { Passes::Debug }, threads, stack)
}

/// [`codegen_link`] of a release build whose module is already the output of the release passes
/// (level `opt`, verified): the passes do not run again.
pub fn codegen_link_optimized(
    m: &Module,
    triple: Triple,
    threads: Option<usize>,
    stack: Option<&stack::Input>,
) -> Result<Built, Failure> {
    generate(m, triple, Passes::Done, threads, stack)
}

/// The build mode of [`generate`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum Passes {
    /// A debug build: no release passes, the debug layout.
    Debug,
    /// A release build of a checked module: the release passes first.
    Run,
    /// A release build of their output.
    Done,
}

fn generate(
    m: &Module,
    triple: Triple,
    passes: Passes,
    threads: Option<usize>,
    stack: Option<&stack::Input>,
) -> Result<Built, Failure> {
    let release = passes != Passes::Debug;
    // A backend without vector code generation: a vector value or op stops the build here.
    if !generates_vector_code(triple)
        && let Some(f) = stave_ir::simd::first_vector_func(m)
    {
        let path = m.func(f).map_or("?", |f| f.path.as_str());
        return Err(Failure::Unsupported {
            stage: Stage::Codegen,
            message: format!("vector code generation (in `{path}`)"),
        });
    }
    let main = m.main.ok_or(Failure::NoMain)?;
    let funcs = m.funcs.iter().filter(|f| f.body.is_some()).count() as u64;
    if !m.funcs.windows(2).all(|w| w[0].instance <= w[1].instance) {
        let message = "the module is not canonical: functions out of instance order".to_string();
        return Err(Failure::Internal { stage: Stage::Codegen, message });
    }
    let owned;
    let m = if passes == Passes::Run {
        owned = release_passes(m)?;
        &owned
    } else {
        m
    };
    match triple.arch() {
        Arch::X86_64 => x64(m, main, triple.baseline(), release, threads, funcs, stack),
        Arch::Aarch64 => a64(m, main, release, funcs, stack),
        Arch::Gpu => Err(Failure::Target(triple)),
    }
}

/// The release passes on the canonical module `m`: element pointer types, the optimizer on
/// every body, then the canonical numbering again (the passes keep every function), the IR
/// verifier and the bounds audit on the result. The module the release backends compile and
/// the `opt` dump prints.
pub fn release_passes(m: &Module) -> Result<Module, Failure> {
    let mut c = m.clone();
    stave_opt::slices::declare_elem_pointers(&mut c.types);
    let types = &c.types;
    c.funcs.par_iter_mut().filter_map(|f| f.body.as_mut()).for_each(|b| stave_opt::optimize(types, b));
    prof::mark("opt");
    let remap = canonical(&mut c);
    if !stave_ir::canon::is_identity(&remap) {
        let message = "the release passes moved a function".to_string();
        return Err(Failure::Internal { stage: Stage::Opt, message });
    }
    prof::mark("opt-canonical");
    // The optimizer's output must be valid IR at level `opt`. The audit after it is stage0's
    // self-check of its base passes; level `opt` itself does not require it, and loading `opt`
    // text skips it. It sees only the accesses it counts that the passes leave in reference
    // form: `split_elem_addrs` has put a slice's element accesses behind `slice.ptr`, which it
    // does not count.
    verified(&c, Stage::Opt, Level::Opt)?;
    prof::mark("opt-verify");
    audited(&c, Stage::Opt)?;
    prof::mark("opt-audit");
    Ok(c)
}

/// Whether the backend of `triple` generates code for vector values and ops.
fn generates_vector_code(triple: Triple) -> bool {
    match triple.arch() {
        Arch::X86_64 | Arch::Aarch64 => true,
        Arch::Gpu => false,
    }
}

fn x64(
    m: &Module,
    main: FuncId,
    baseline: stave_target::FeatureSet,
    release: bool,
    threads: Option<usize>,
    funcs: u64,
    stack: Option<&stack::Input>,
) -> Result<Built, Failure> {
    use stave_codegen_x64::Error as E;
    let classify = |stage: Stage, e: E| match e {
        E::Unsupported { .. } => Failure::Unsupported { stage, message: e.to_string() },
        _ => Failure::Internal { stage, message: e.to_string() },
    };
    let image = image_options(Machine::X86_64, release);
    let graph = stack.map(|s| stack::graph(m, s));
    let o = stave_codegen_x64::Options {
        image: image.clone(),
        main: Some(main),
        // The driver verified the module after lowering and after the release passes.
        verify: None,
        threads: threads.unwrap_or(0),
        release,
        features: baseline,
        reserve_slots: graph.as_ref().map_or_else(Vec::new, |g| g.slots().to_vec()),
        ..stave_codegen_x64::Options::default()
    };
    let t0 = Instant::now();
    let mut b = ImageBuilder::new(Machine::X86_64, image);
    let e = stave_codegen_x64::emit_module(m, &o, &mut b).map_err(|e| classify(Stage::Codegen, e))?;
    let codegen_ms = ms(t0);
    prof::mark("codegen");
    let report = stack.zip(graph.as_ref()).map(|(s, g)| stack::compute(g, m, s, &e.stack, main));
    if let Some(r) = &report {
        let link_err =
            |e: stave_obj::ObjError| Failure::Internal { stage: Stage::Link, message: e.to_string() };
        b.define_rodata(e.stack_table, r.table(&e.func_syms)).map_err(link_err)?;
        b.define_rodata(e.reserve_table, r.reserve_table()).map_err(link_err)?;
        prof::mark("stack");
    }
    let t1 = Instant::now();
    let img = stave_codegen_x64::link(b, &e).map_err(|e| classify(Stage::Link, e))?;
    prof::mark("link");
    Ok(Built { image: img, codegen_ms, link_ms: ms(t1), funcs, stack: report })
}

fn a64(
    m: &Module,
    main: FuncId,
    release: bool,
    funcs: u64,
    stack: Option<&stack::Input>,
) -> Result<Built, Failure> {
    use stave_codegen_a64::debug::{Codegen, Error, ErrorKind, FuncScratch, prepare};
    let classify = |stage: Stage, e: Error| match e.kind {
        ErrorKind::Unsupported(_) => Failure::Unsupported { stage, message: e.to_string() },
        _ => Failure::Internal { stage, message: e.to_string() },
    };
    let t0 = Instant::now();
    let mut b = ImageBuilder::new(Machine::Aarch64, image_options(Machine::Aarch64, release));
    let syms = prepare(&mut b, m, Some(main)).map_err(|e| classify(Stage::Codegen, e))?;
    let graph = stack.map(|s| stack::graph(m, s));
    let slots = graph.as_ref().map_or_else(Vec::new, |g| g.slots().to_vec());
    let cg = Codegen::new(m, &syms).release(release).reserve_slots(&slots);
    let ids: Vec<FuncId> =
        (0..m.funcs.len()).filter(|&i| m.funcs[i].body.is_some()).map(|i| FuncId(i as u32)).collect();
    let codes: Vec<Result<(FuncCode, StackUse), Error>> =
        ids.par_iter().map_init(FuncScratch::default, |s, &f| cg.compile_stack(f, s)).collect();
    let mut uses = vec![StackUse::default(); m.funcs.len()];
    for (&f, code) in ids.iter().zip(codes) {
        let (code, used) = code.map_err(|e| classify(Stage::Codegen, e))?;
        uses[f.index()] = used;
        let sym = syms.funcs.get(f.index()).copied().flatten().ok_or_else(|| Failure::Internal {
            stage: Stage::Codegen,
            message: format!("fn #{} has no symbol", f.0),
        })?;
        b.define_func(sym, code)
            .map_err(|e| Failure::Internal { stage: Stage::Codegen, message: e.to_string() })?;
    }
    let codegen_ms = ms(t0);
    prof::mark("codegen");
    let report = stack.zip(graph.as_ref()).map(|(s, g)| stack::compute(g, m, s, &uses, main));
    let (table, reserve) = match &report {
        Some(r) => (r.table(&syms.funcs), r.reserve_table()),
        None => (stave_obj::stub::stack_table(0, 0, &[]), stave_obj::stub::reserve_table(&[])),
    };
    let link_err = |e: stave_obj::ObjError| Failure::Internal { stage: Stage::Link, message: e.to_string() };
    b.define_rodata(syms.stack_table, table).map_err(link_err)?;
    b.define_rodata(syms.reserve_table, reserve).map_err(link_err)?;
    prof::mark("stack");
    let t1 = Instant::now();
    let img = b.link().map_err(|e| Failure::Internal { stage: Stage::Link, message: e.to_string() })?;
    prof::mark("link");
    Ok(Built { image: img, codegen_ms, link_ms: ms(t1), funcs, stack: report })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `app.at(a: &[u32; 4], i: usize)`: `check.bounds 2 4` (provable), then `check.bounds i 4`
    /// (not provable), each before its `addr.elem`.
    fn two_checks() -> (Module, FuncId) {
        use stave_ir::{FuncBuilder, InstanceId, Op};
        let mut m = Module::new();
        let (us, u32t) = (m.types.usize(), m.types.u32());
        let arr = m.types.array(u32t, 4);
        let (pa, pe) = (m.types.ref_(false, arr), m.types.ref_(false, u32t));
        let f = m.declare(InstanceId(vec![0]), "app.at", vec![pa, us], Some(u32t));
        let mut b = FuncBuilder::new(&[pa, us]);
        let (a, i) = (b.param(0).unwrap(), b.param(1).unwrap());
        let (two, four) = (b.int(2), b.int(4));
        b.bounds_check(two, four);
        let p = b.op(Op::ElemAddr, pe, &[a.into(), two]);
        let x = b.load(u32t, p);
        b.bounds_check(i, four);
        let q = b.op(Op::ElemAddr, pe, &[a.into(), i.into()]);
        let y = b.load(u32t, q);
        let s = b.binary(Op::AddWrap, u32t, x, y);
        b.ret(Some(s.into()));
        m.define(f, b.finish().unwrap());
        canonical(&mut m);
        (m, f)
    }

    fn checks(m: &Module, f: FuncId) -> usize {
        let b = m.func(f).unwrap().body.as_ref().unwrap();
        b.insts.op.iter().filter(|op| **op == stave_ir::Op::BoundsCheck).count()
    }

    /// The step removes the check that has a proof and keeps the other. A proof the witness
    /// checker refuses stops the build as a compiler bug, and no check is removed.
    #[test]
    fn a_refused_witness_is_an_internal_error() {
        let (mut m, f) = two_checks();
        let proved = remove_proven_checks(&mut m, true).unwrap_or_else(|e| panic!("{e}")).proved;
        assert_eq!((proved.len(), proved[0].proofs.len(), checks(&m, f)), (1, 1, 1));

        let (mut m, f) = two_checks();
        let mut forged = stave_ir::bounds::prove_func(&m, f).unwrap();
        // the row of the provable check, pointed at the other one
        let ops = &m.func(f).unwrap().body.as_ref().unwrap().insts.op;
        let second = ops.iter().rposition(|op| *op == stave_ir::Op::BoundsCheck).unwrap();
        forged.witness.removed[0].inst = second as u32;
        let Err(e) = remove_checked(&mut m, vec![forged], true) else {
            panic!("a forged witness was accepted")
        };
        assert_eq!(e.stage(), Stage::Lower);
        assert!(matches!(e, Failure::Internal { .. }), "{e:?}");
        let text = e.to_string();
        assert!(text.starts_with("lower failed: bounds witness: @0 app.at: inst "), "{text}");
        assert_eq!(checks(&m, f), 2);
    }

    #[test]
    fn listed_messages() {
        assert_eq!(listed(&["a", "b"]), "a; b");
        let many: Vec<String> = (0..10).map(|i| i.to_string()).collect();
        assert!(listed(&many).ends_with("; ... and 2 more"));
    }

    #[test]
    fn no_host_target_names_the_host() {
        let m = Failure::NoHostTarget.to_string();
        assert!(m.contains(std::env::consts::ARCH) && m.contains("--target"), "{m}");
        assert!(!m.contains("gpu-sm86"), "{m}");
        assert_eq!(Failure::NoHostTarget.stage(), Stage::Codegen);
    }
}
