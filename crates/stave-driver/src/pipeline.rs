//! The stage0 driver pipeline.
//!
//! read → lex → items → parse → intern → resolve (+ canonical renumbering) → typeck (+ canonical
//! type renumbering) → borrowck (stave-borrow, per body) → effects (deny clauses; [`crate::effects`],
//! no `--timings` line) → lock ranks ([`stave_sema::ranks`], no `--timings` line) → mono + lower (checked IR) → codegen → link (static ELF with DWARF line
//! tables).
//!
//! The front end through name resolution is [`stave_sema::Program::load`] (packages and manifests,
//! the built-in `std`/`runtime` of a [`Library`], per-file parallel read + lex + items + parse,
//! interning, modules and imports, declarations, resolution, canonical ids); type checking is
//! [`stave_sema::check`] (per body, parallel). Diagnostics of every stage are put into the
//! canonical order once, at the end ([`Output::diagnostics`]).
//!
//! `--timings` stages (schema names, `contracts/diagnostics.schema.json`), in execution order:
//! `read` (manifests, file discovery, file reads), `lex`, `items`, `parse` (stave-sema's
//! [`stave_sema::Timings::stages`]), `intern`, `resolve`, `typeck`, `borrowck` (items = bodies
//! checked), then for builds `lower` (mono + lowering + IR verification + the removal of proven
//! bounds checks with its witness check, fused; no separate `mono` line),
//! `codegen` (per function, parallel), `link` (layout, relocations, DWARF line tables, ELF; no
//! separate `dwarf` line).
//!
//! Goals: [`Goal::Check`] stops after the effect pass ([`Output::borrowck`], [`Output::effects`]); a body the borrow
//! checker could not check ([`stave_borrow::Borrowck::unchecked`]) is [`Output::failure`] for every
//! goal but [`Goal::Parse`]. [`Goal::Build`] goes on
//! through `link` when there is no error diagnostic; the back end's result is [`Output::module`] (the checked IR, also
//! with [`Options::lower`] on a check, for the `ir` dump), [`Output::image`], or
//! [`Output::failure`] ([`crate::build`]).
//!
//! Determinism: every id is canonical (stave-sema), every parallel step collects in index order,
//! and nothing visible depends on the thread count or the shuffle knob [`stave_sema::SHUFFLE_ENV`]
//! (tests: `dump_determinism.rs`). The thread count comes from [`Options::threads`] (the
//! CLI reads `STAVE_THREADS`); each run uses its own rayon pool for loading and resolution.
//! `stave_sema::check` runs bodies on stave-sema's big-stack pool, with as many workers as the
//! calling pool.

use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;
use stave_diag::{Diagnostic, Severity, SourceFiles};
use stave_ir::{FuncId, Module};
use stave_obj::ObjectImage;
use stave_sema::{DeclKind, Mode, Program, Typeck};
use stave_target::Triple;

use crate::build::{self, Failure};
use crate::prof;
use crate::shipped::Library;

/// A pipeline stage, named as in the `timings` enum of `contracts/diagnostics.schema.json`
/// (stage0 subset, schema order).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stage {
    Read,
    Lex,
    Items,
    Parse,
    Resolve,
    Intern,
    Typeck,
    Borrowck,
    Mono,
    Lower,
    Opt,
    Codegen,
    Dwarf,
    Link,
    Incremental,
}

impl Stage {
    pub const ALL: [Stage; 15] = [
        Stage::Read,
        Stage::Lex,
        Stage::Items,
        Stage::Parse,
        Stage::Resolve,
        Stage::Intern,
        Stage::Typeck,
        Stage::Borrowck,
        Stage::Mono,
        Stage::Lower,
        Stage::Opt,
        Stage::Codegen,
        Stage::Dwarf,
        Stage::Link,
        Stage::Incremental,
    ];

    /// The schema name.
    pub fn name(self) -> &'static str {
        match self {
            Stage::Read => "read",
            Stage::Lex => "lex",
            Stage::Items => "items",
            Stage::Parse => "parse",
            Stage::Resolve => "resolve",
            Stage::Intern => "intern",
            Stage::Typeck => "typeck",
            Stage::Borrowck => "borrowck",
            Stage::Mono => "mono",
            Stage::Lower => "lower",
            Stage::Opt => "opt",
            Stage::Codegen => "codegen",
            Stage::Dwarf => "dwarf",
            Stage::Link => "link",
            Stage::Incremental => "incremental",
        }
    }

    /// What the `items` count of a timing means (schema: lines for read/lex/parse, items for
    /// items, strings for intern, functions otherwise).
    pub fn unit(self) -> &'static str {
        match self {
            Stage::Read | Stage::Lex | Stage::Parse => "lines",
            Stage::Items => "items",
            Stage::Intern => "strings",
            _ => "fns",
        }
    }

    /// Whether the diagnostics of this stage are complete in stage0: the codes of every stage
    /// through `borrowck`.
    pub fn reports_all_codes(self) -> bool {
        self <= Stage::Borrowck
    }

    /// The stage that reports diagnostic `code`, or `None` for codes from outside the pipeline
    /// (`stave test`/`stave run` E1001, the stage1 probe E1301–E1304, the GPU tables E11xx/E12xx).
    /// A code's set of diagnostics is final once its stage ran:
    ///
    /// | Stage | Codes |
    /// |---|---|
    /// | read | E1002 (manifests), E1003 (unreadable source) |
    /// | lex | E0001 E0002 E0003 E0006 E0015 E0017 E0018 E0024 E0029 |
    /// | items | E0016 |
    /// | parse | E0004 E0011–E0014 E0019–E0023 E0025–E0028 |
    /// | resolve | E0101–E0107 |
    /// | typeck | E0005 (bracket depth from the item cut, cases `expr` and `type` from typeck),E02xx E03xx (case `limit` included) E04xx E06xx, E0700–E0718 (the lowering-class gates E0700 E0701 included), E08xx (E0807 case `thread` included), E09xx (E0902: the instantiation depth check runs with typeck), E1404 (deny clause names); the signature's `from` clauses E0514 E0515 E0520 E0521; E0503 case `static` (writes to a `static`) |
    /// | borrowck | E05xx except E0514 E0515 E0520 E0521 (accesses, loans, returned origins) |
    /// | lower | E0108 (no `fn main`, builds only, reported before lowering); E1401 E1403 (the effect pass), E1405 E1406 (the rank pass), after borrowck on programs without errors, in checks and builds |
    /// | (none) | retired codes: E0511 |
    pub fn of_code(code: &str) -> Option<Stage> {
        match code {
            "E0001" | "E0002" | "E0003" | "E0006" | "E0015" | "E0017" | "E0018" | "E0024" | "E0029" => {
                return Some(Stage::Lex);
            }
            "E0016" => return Some(Stage::Items),
            // The bracket-depth case comes from the item cut, `expr` and `type` from typeck: the set
            // is final after typeck.
            "E0005" => return Some(Stage::Typeck),
            // E0004 also comes from the item cut (pre-item run, late import): both are front end.
            "E0004" | "E0011" | "E0012" | "E0013" | "E0014" => return Some(Stage::Parse),
            "E0019" | "E0020" | "E0021" | "E0022" | "E0023" | "E0025" | "E0026" | "E0027" | "E0028" => {
                return Some(Stage::Parse);
            }
            "E1002" | "E1003" => return Some(Stage::Read),
            // a build without `main` (stave build/run/test only; reported before lowering)
            "E0108" => return Some(Stage::Lower),
            // the effect and rank passes run after borrowck on programs without errors, before
            // lowering
            "E1401" | "E1403" | "E1405" | "E1406" => return Some(Stage::Lower),
            "E1404" => return Some(Stage::Typeck),
            // `from` clauses: checked with the signature (stave-sema origins), never from a body.
            "E0514" | "E0515" | "E0520" | "E0521" => return Some(Stage::Typeck),
            // retired: a missing `from` declares the empty origin set
            "E0511" => return None,
            _ => {}
        }
        let digits =
            code.strip_prefix('E').filter(|d| d.len() == 4 && d.bytes().all(|b| b.is_ascii_digit()))?;
        let n: u16 = digits.parse().ok()?;
        match n {
            101..=199 => Some(Stage::Resolve),
            501..=599 => Some(Stage::Borrowck),
            200..=999 => Some(Stage::Typeck),
            _ => None,
        }
    }
}

/// How far a run goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Goal {
    /// Lex and parse only (conformance `PARSE-ONLY`): no built-in packages, and only the
    /// diagnostics of the read, lex, items, and parse stages are kept.
    Parse,
    /// `stave check`: through borrowck, no codegen.
    Check,
    /// `stave build`: through link.
    Build,
}

impl Goal {
    /// The last stage the goal needs.
    pub fn last(self) -> Stage {
        match self {
            Goal::Parse => Stage::Parse,
            Goal::Check => Stage::Borrowck,
            Goal::Build => Stage::Link,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    pub goal: Goal,
    /// Worker threads; `None` = rayon's default (available parallelism).
    pub threads: Option<usize>,
    /// The build target (dependency selection, intrinsic arch checks, code generation). `None`:
    /// host baseline.
    pub target: Option<Triple>,
    /// The shuffle test knob ([`stave_sema::SHUFFLE_ENV`]); `None` reads the environment.
    pub shuffle_seed: Option<u64>,
    /// `--release`: lower only what `main` reaches ([`Mode::Release`]) and link the packed image.
    pub release: bool,
    /// `stave test`: lower the `test fn`s too ([`Mode::Test`], with `release`
    /// [`Mode::ReleaseTest`]).
    pub tests: bool,
    /// Lower even when the goal is [`Goal::Check`] (the `ir` dump).
    pub lower: bool,
    /// Collect the borrow checker's per-access lines (the `borrow` dump).
    pub borrow_dump: bool,
    /// Lower with the borrow side table (`Body::borrows`), for the interpreter's borrow
    /// tracking. The module is then not the canonical checked IR.
    pub borrow_marks: bool,
    /// Lower a second module with the borrow side table, beside the canonical one
    /// ([`Output::tracked`]): dumps and the backend get the canonical module, the interpreter the
    /// second. Default: whether `STAVE_INTERP_CHECK=borrow` is set.
    pub tracked: bool,
    /// Lower although the borrow checker rejected the program (its diagnostics stay in
    /// [`Output::diagnostics`]) or could not check a body: the borrow-tracking interpreter runs
    /// such a program to see what the dynamic model says. Checks only: a build ignores it.
    pub unchecked_borrows: bool,
    /// `stave effects`: compute `trap` from the checked IR of every fn, not only of what the deny
    /// clauses reach ([`crate::effects::run`]).
    pub effects_all: bool,
    /// Contents that stand in for source files on disk ([`stave_sema::Config::sources`]): a
    /// check of edits that are not written yet.
    pub sources: std::collections::BTreeMap<std::path::PathBuf, Vec<u8>>,
}

impl Options {
    pub fn new(goal: Goal) -> Options {
        Options {
            goal,
            threads: None,
            target: None,
            shuffle_seed: None,
            release: false,
            tests: false,
            lower: false,
            borrow_dump: false,
            borrow_marks: false,
            tracked: crate::interp::borrow_from_env(),
            unchecked_borrows: false,
            effects_all: false,
            sources: std::collections::BTreeMap::new(),
        }
    }

    /// The lowering mode.
    pub fn mode(&self) -> Mode {
        if self.tests && self.release {
            Mode::ReleaseTest
        } else if self.tests {
            Mode::Test
        } else if self.release {
            Mode::Release
        } else {
            Mode::Debug
        }
    }

    pub fn threads(mut self, threads: Option<usize>) -> Options {
        self.threads = threads;
        self
    }
}

/// Wall time of one stage (`--timings`, `stats` dump).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StageTiming {
    pub stage: Stage,
    pub ms: f64,
    /// Lines, items, strings, or fns per [`Stage::unit`].
    pub items: u64,
    /// Parallel rounds (stage1); stage0 has none.
    pub rounds: Option<u64>,
    /// Times the stage left its parallel path (stage1); stage0 has none.
    pub fallbacks: Option<u64>,
}

/// Everything a run produced.
pub struct Output {
    /// The resolved program: files (`FileId` = rank of (package, path)), items, strings,
    /// decls, resolutions, all with canonical ids.
    pub program: Program,
    /// The type-checked program (`None` when the goal is [`Goal::Parse`]).
    pub typeck: Option<Typeck>,
    /// The borrow checker's result (`None` when the goal is [`Goal::Parse`]).
    pub borrowck: Option<stave_borrow::Borrowck>,
    /// The effect sets of the call graph's nodes ([`crate::effects`]; `None` with error
    /// diagnostics before the effect pass).
    pub effects: Option<stave_sema::effects::EffectSets>,
    /// All diagnostics in canonical order, deduplicated.
    pub diagnostics: Vec<Diagnostic>,
    /// The stages that ran, in execution order.
    pub timings: Vec<StageTiming>,
    /// The checked IR (builds, and checks with [`Options::lower`]; only without error
    /// diagnostics): the lowered program after the removal of the bounds checks that have a
    /// checked witness, the same in every build mode.
    pub module: Option<Module>,
    /// [`Output::module`] lowered again with the borrow side table ([`Options::tracked`]): what
    /// the borrow-tracking interpreter runs. Not the canonical checked IR.
    pub tracked: Option<Module>,
    /// The bounds checks [`Output::module`] no longer has, per function: the witness the
    /// checker accepted and the rule behind each row.
    pub removed_checks: Vec<stave_ir::bounds::Proved>,
    /// The checked-IR function of the root package's `main`, if any.
    pub main: Option<FuncId>,
    /// The executable ([`Goal::Build`] without errors).
    pub image: Option<ObjectImage>,
    /// The stack bound of the executable ([`crate::stack`]).
    pub stack: Option<crate::stack::Report>,
    /// Why a run without error diagnostics failed: a body borrowck could not check (check and
    /// build), or no image (build).
    pub failure: Option<Failure>,
    /// Source lines of every file of the program.
    pub lines: u64,
}

impl Output {
    /// The diagnostic file table: the source files (ids = `program.files`), then the manifests.
    pub fn files(&self) -> &SourceFiles {
        &self.program.source_map
    }

    pub fn error_count(&self) -> usize {
        self.diagnostics.iter().filter(|d| d.severity() == Severity::Error).count()
    }

    pub fn has_errors(&self) -> bool {
        self.error_count() > 0
    }

    /// The module the interpreter runs: the one with the borrow side table when there is one
    /// ([`Options::tracked`]), else the checked IR.
    pub fn for_interp(&self) -> Option<&Module> {
        self.tracked.as_ref().or(self.module.as_ref())
    }

    /// The build summary's stack lines (empty without a computed bound).
    pub fn stack_lines(&self) -> Vec<String> {
        let (Some(r), Some(tc), Some(m)) = (&self.stack, &self.typeck, &self.module) else {
            return Vec::new();
        };
        crate::stack::summary(r, &self.program, tc, m, |sp| match self.files().locate(sp) {
            Ok(l) => format!("{}:{}:{}", l.path, l.line, l.col),
            Err(_) => "?".to_string(),
        })
    }
}

#[derive(Debug)]
pub enum PipelineError {
    /// The thread pool cannot be built.
    Threads(String),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::Threads(e) => write!(f, "cannot start worker threads: {e}"),
        }
    }
}

impl std::error::Error for PipelineError {}

/// A rayon pool of `threads` workers (`None`: the default count).
pub fn thread_pool(threads: Option<usize>) -> Result<rayon::ThreadPool, PipelineError> {
    let mut b = rayon::ThreadPoolBuilder::new();
    if let Some(n) = threads {
        b = b.num_threads(n.max(1));
    }
    b.build().map_err(|e| PipelineError::Threads(e.to_string()))
}

/// Runs the pipeline on `root` (a package directory or a `.stv` file, see
/// [`crate::package::check_root`]) on its own pool of [`Options::threads`] workers.
pub fn run(root: &Path, lib: &Library, opts: &Options) -> Result<Output, PipelineError> {
    let pool = thread_pool(opts.threads)?;
    Ok(run_on(&pool, root, lib, opts))
}

/// Runs the pipeline on `pool` (reused across runs by `stave test`).
///
/// Called from a worker of another pool, it waits on a plain thread: a worker waiting in
/// `install` runs its own pool's other jobs on its stack meanwhile (rayon's cross-pool wait), so
/// runs started from the items of a `par_iter` would nest on one stack without bound.
pub fn run_on(pool: &rayon::ThreadPool, root: &Path, lib: &Library, opts: &Options) -> Output {
    if rayon::current_thread_index().is_some() && pool.current_thread_index().is_none() {
        return std::thread::scope(|s| s.spawn(|| pool.install(|| run_inner(root, lib, opts))).join())
            .unwrap_or_else(|e| std::panic::resume_unwind(e));
    }
    pool.install(|| run_inner(root, lib, opts))
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn timing(stage: Stage, ms: f64, items: u64) -> StageTiming {
    StageTiming { stage, ms, items, rounds: None, fallbacks: None }
}

/// Lines of a text: newlines, plus one for a last line without one.
fn line_count(b: &[u8]) -> u64 {
    let nl = b.iter().filter(|&&c| c == b'\n').count() as u64;
    nl + u64::from(b.last().is_some_and(|&c| c != b'\n'))
}

fn run_inner(root: &Path, lib: &Library, opts: &Options) -> Output {
    let dropped = dropped_diagnostics();
    let mut out = run_stages(root, lib, opts);
    if dropped_diagnostics() != dropped && !matches!(out.failure, Some(Failure::Internal { .. })) {
        out.failure = Some(dropped_failure());
    }
    out
}

/// Diagnostics the lexer, parser, checkers could not build from their registry templates since
/// process start: each is a compiler bug that would drop a diagnostic.
fn dropped_diagnostics() -> u64 {
    stave_syntax::diagnostic_build_failures()
        + stave_sema::diagnostic_build_failures()
        + stave_borrow::diagnostic_build_failures()
}

/// A run in which a diagnostic could not be built fails as an internal error (exit 3), whatever
/// else it reports: the dropped diagnostic may be its only error.
fn dropped_failure() -> Failure {
    let what = stave_sema::first_diagnostic_build_failure()
        .or_else(stave_syntax::first_diagnostic_build_failure)
        .unwrap_or("a borrow checker diagnostic");
    Failure::Internal {
        stage: Stage::Typeck,
        message: format!("a diagnostic could not be built from its registry template ({what})"),
    }
}

fn run_stages(root: &Path, lib: &Library, opts: &Options) -> Output {
    let parse_only = opts.goal == Goal::Parse;
    let mut cfg = lib.config(root);
    if parse_only {
        // PARSE-ONLY: no built-in packages, no semantic stage (stave-sema's parse-only load).
        cfg = cfg.parse_only();
    }
    cfg.target = opts.target;
    cfg.shuffle_seed = opts.shuffle_seed;
    cfg.sources = opts.sources.clone();

    // ---- read, lex, items, parse, intern, resolve (+ canonical ids): stave-sema's loader.
    let program = Program::load(&cfg);
    let lines: u64 = program.files.par_iter().map(|f| line_count(&f.bytes)).sum();
    let items = program.items.len() as u64;
    let fns =
        program.decls.iter().filter(|d| matches!(d.kind, DeclKind::Fn | DeclKind::Method)).count() as u64;
    let mut timings = Vec::new();
    for (name, d) in program.timings.stages() {
        let Some(stage) = Stage::ALL.into_iter().find(|s| s.name() == name) else { continue };
        if parse_only && stage > Stage::Parse {
            continue;
        }
        let n = match stage {
            Stage::Items => items,
            Stage::Intern => program.strings.len() as u64,
            Stage::Resolve => fns,
            _ => lines,
        };
        timings.push(timing(stage, ms(d), n));
    }
    if prof::enabled() {
        let t = &program.timings;
        prof::part(".discover", t.load);
        prof::part(".read", t.read);
        prof::part(".lex", t.lex);
        prof::part(".items", t.items);
        prof::part(".parse", t.parse_items);
        prof::part(".intern", t.intern);
        prof::part(".resolve", t.resolve);
        prof::mark("load");
    }

    if parse_only {
        let diagnostics = stave_diag::canonical_order(program.diagnostics.clone(), &program.source_map);
        return Output {
            program,
            typeck: None,
            borrowck: None,
            effects: None,
            diagnostics,
            timings,
            module: None,
            tracked: None,
            removed_checks: Vec::new(),
            main: None,
            image: None,
            stack: None,
            failure: None,
            lines,
        };
    }

    // Syntax without semantics yet: its checks would be wrong, so nothing after resolve runs.
    if !program.has_errors()
        && let Some(what) = program.unimplemented()
    {
        let diagnostics = stave_diag::canonical_order(program.diagnostics.clone(), &program.source_map);
        return Output {
            program,
            typeck: None,
            borrowck: None,
            effects: None,
            diagnostics,
            timings,
            module: None,
            tracked: None,
            removed_checks: Vec::new(),
            main: None,
            image: None,
            stack: None,
            failure: Some(Failure::Unsupported { stage: Stage::Typeck, message: what }),
            lines,
        };
    }

    // ---- typeck (per body, parallel; canonical type ids).
    let t0 = Instant::now();
    let tc = stave_sema::check(&program);
    timings.push(timing(Stage::Typeck, ms(t0.elapsed()), tc.bodies.len() as u64));
    prof::part(".sigs", tc.timings.sigs);
    prof::part(".bodies", tc.timings.bodies);
    prof::part(".canon", tc.timings.canon);
    prof::part(".call-graph", tc.timings.graph);
    prof::mark("typeck");
    let mut diagnostics = tc.all_diagnostics(&program);
    prof::mark("diag-order");

    // ---- borrowck (per body, parallel; bodies with a type error are skipped).
    let t0 = Instant::now();
    let bc = stave_borrow::check(&program, &tc, stave_borrow::Options { dump: opts.borrow_dump });
    timings.push(timing(Stage::Borrowck, ms(t0.elapsed()), bc.checked));
    prof::part(".cfg-cpu", bc.phases[0]);
    prof::part(".dataflow-cpu", bc.phases[1]);
    prof::part(".check-cpu", bc.phases[2]);
    prof::mark("borrowck");
    if !bc.diagnostics.is_empty() {
        diagnostics.extend(bc.diagnostics.iter().cloned());
        diagnostics = stave_diag::canonical_order(diagnostics, &program.source_map);
    }

    let mut has_errors = diagnostics.iter().any(|d| d.severity() == Severity::Error);
    // E0108: a build needs `main` (reported only without other errors: a dropped or invalid item
    // may be `main`)
    if opts.goal == Goal::Build
        && !has_errors
        && let Some(d) = build::no_main(&program)
    {
        diagnostics.push(d);
        diagnostics = stave_diag::canonical_order(diagnostics, &program.source_map);
        has_errors = true;
    }
    // A body the borrow checker could not model is not known to be sound: the run fails (check
    // and build alike) instead of accepting it without its E05xx diagnostics.
    let mut failure = if has_errors {
        None
    } else {
        silent_error(&tc, |d| program.decl_path(d))
            .or_else(|| unchecked_failure(&bc, |d| program.decl_path(d)))
    };

    // ---- effects: deny clauses against the effect sets of the call graph.
    let mut effect_sets = None;
    if !has_errors && failure.is_none() {
        let t0 = Instant::now();
        match crate::effects::run(&program, &tc, opts.effects_all) {
            Ok(r) => {
                if !r.diagnostics.is_empty() {
                    has_errors |= r.diagnostics.iter().any(|d| d.severity() == Severity::Error);
                    diagnostics.extend(r.diagnostics);
                    diagnostics = stave_diag::canonical_order(diagnostics, &program.source_map);
                }
                effect_sets = Some(r.sets);
            }
            Err(f) => failure = Some(f),
        }
        prof::part(".effects", t0.elapsed());
        prof::mark("effects");
    }
    // ---- lock ranks: every acquisition above the ranks held, every join too.
    if !has_errors && failure.is_none() {
        let t0 = Instant::now();
        let ds = stave_sema::ranks::check(&program, &tc, &bc.locks, &bc.drops);
        if !ds.is_empty() {
            diagnostics.extend(ds);
            diagnostics = stave_diag::canonical_order(diagnostics, &program.source_map);
            has_errors = true;
        }
        prof::part(".ranks", t0.elapsed());
    }
    let mut out = Output {
        program,
        typeck: None,
        borrowck: Some(bc),
        effects: effect_sets,
        diagnostics,
        timings,
        module: None,
        tracked: None,
        removed_checks: Vec::new(),
        main: None,
        image: None,
        stack: None,
        failure,
        lines,
    };

    let lower = opts.goal == Goal::Build || opts.lower;
    // the errors that stop lowering: with `unchecked_borrows`, all but the borrow checker's
    let stopped = if opts.unchecked_borrows && opts.goal == Goal::Check {
        let borrowck = |d: &Diagnostic| Stage::of_code(d.code()) == Some(Stage::Borrowck);
        out.diagnostics.iter().any(|d| d.severity() == Severity::Error && !borrowck(d))
            || out.failure.as_ref().is_some_and(|f| f.stage() != Stage::Borrowck)
    } else {
        has_errors || out.failure.is_some()
    };
    if !lower || stopped {
        out.typeck = Some(tc);
        return out;
    }

    // ---- mono + lower + verify + removal of the bounds checks with a checked witness.
    let t0 = Instant::now();
    let lowered = build::lower(&out.program, &tc, opts.mode(), opts.borrow_marks);
    let funcs = lowered.as_ref().map_or(0, |l| l.module.funcs.iter().filter(|f| f.body.is_some()).count());
    out.timings.push(timing(Stage::Lower, ms(t0.elapsed()), funcs as u64));
    out.typeck = Some(tc);
    let (module, nodes_of_func) = match lowered {
        Ok(l) => {
            out.removed_checks = l.removed;
            (l.module, l.nodes_of_func)
        }
        Err(f) => {
            out.failure = Some(f);
            return out;
        }
    };
    out.main = build::main_path(&out.program).and_then(|p| module.find_func(&p));
    prof::mark("find-main");
    if opts.tracked
        && !opts.borrow_marks
        && let Some(tc) = &out.typeck
    {
        match build::lower(&out.program, tc, opts.mode(), true) {
            Ok(l) => out.tracked = Some(l.module),
            Err(f) => {
                out.failure = Some(f);
                return out;
            }
        }
    }
    if opts.goal == Goal::Build {
        // ---- codegen + link.
        let triple = opts.target.or_else(Triple::host);
        match (triple, out.main) {
            (None, _) => out.failure = Some(Failure::NoHostTarget),
            (Some(_), None) => out.failure = Some(Failure::NoMain),
            (Some(t), Some(main)) => {
                let run_stack = out.program.packages.iter().find(|p| p.is_root).and_then(|p| p.run_stack);
                let built = match &out.typeck {
                    Some(tc) => {
                        let input = crate::stack::Input {
                            graph: &tc.call_graph,
                            nodes_of_func: &nodes_of_func,
                            run_stack,
                        };
                        build::codegen_link(&module, main, t, opts.release, opts.threads, Some(&input))
                    }
                    None => build::codegen_link(&module, main, t, opts.release, opts.threads, None),
                };
                match built {
                    Ok(b) => {
                        out.timings.push(timing(Stage::Codegen, b.codegen_ms, b.funcs));
                        out.timings.push(timing(Stage::Link, b.link_ms, b.funcs));
                        out.image = Some(b.image);
                        out.stack = b.stack;
                    }
                    Err(f) => out.failure = Some(f),
                }
            }
        }
    }
    out.module = Some(module);
    out
}

/// The failure for the first body borrowck could not check (its decl path from `path`), if any.
/// A body the checker gave an error type while no error diagnostic was reported: the checker
/// lost an error, so the program is not known to be well typed (exit 3, check and build alike).
fn silent_error(tc: &stave_sema::Typeck, path: impl Fn(stave_sema::DeclId) -> String) -> Option<Failure> {
    let b = tc.bodies.iter().find(|b| b.has_error)?;
    Some(Failure::Internal {
        stage: Stage::Typeck,
        message: format!("`{}` has a type error but no error diagnostic", path(b.decl)),
    })
}

fn unchecked_failure(
    bc: &stave_borrow::Borrowck,
    path: impl Fn(stave_sema::DeclId) -> String,
) -> Option<Failure> {
    if let Some(&(d, why)) = bc.internal.first() {
        return Some(Failure::Internal {
            stage: Stage::Borrowck,
            message: format!(
                "the borrow checker disagrees with the stages before it on `{}` ({why})",
                path(d)
            ),
        });
    }
    let (d, why) = bc.unchecked()?;
    Some(Failure::Unsupported {
        stage: Stage::Borrowck,
        message: format!("the borrow checker cannot check `{}` ({why})", path(d)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs started from the items of a `par_iter` run one at a time on each caller's stack: a
    /// caller waiting for its run does not take up the next item.
    #[test]
    fn parallel_runs_do_not_nest() {
        use std::cell::Cell;
        thread_local! {
            static DEPTH: Cell<u32> = const { Cell::new(0) };
        }
        let root =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/conformance/syntax/rec_fn_check.stv");
        let lib = Library::in_dir(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."));
        let opts = Options::new(Goal::Parse).threads(Some(1));
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let most = pool.install(|| {
            (0..16u32)
                .into_par_iter()
                .map(|_| {
                    let d = DEPTH.with(|c| {
                        c.set(c.get() + 1);
                        c.get()
                    });
                    let out = run(&root, &lib, &opts).unwrap();
                    assert!(!out.has_errors());
                    DEPTH.with(|c| c.set(c.get() - 1));
                    d
                })
                .max()
        });
        assert_eq!(most, Some(1));
    }

    /// An unchecked body fails the run as unsupported in `borrowck`; a fully checked program
    /// does not.
    #[test]
    fn unchecked_body_fails() {
        let mut bc = stave_borrow::Borrowck::default();
        assert_eq!(unchecked_failure(&bc, |_| unreachable!()), None);
        bc.unsupported.push((stave_sema::DeclId(7), "expression nesting too deep"));
        bc.unsupported.push((stave_sema::DeclId(9), "nested body"));
        let f = unchecked_failure(&bc, |d| format!("pkg.f{}", d.0)).unwrap();
        assert_eq!(f.stage(), Stage::Borrowck);
        assert!(matches!(&f, Failure::Unsupported { .. }));
        assert_eq!(
            f.to_string(),
            "`borrowck` is not implemented yet for this program: the borrow checker cannot check \
             `pkg.f7` (expression nesting too deep)"
        );
        // a disagreement with an earlier stage is a compiler bug, whatever else is unchecked
        bc.internal.push((stave_sema::DeclId(8), "a capture held by `&` is mutated"));
        let f = unchecked_failure(&bc, |d| format!("pkg.f{}", d.0)).unwrap();
        assert!(matches!(&f, Failure::Internal { stage: Stage::Borrowck, .. }));
        assert_eq!(
            f.to_string(),
            "borrowck failed: the borrow checker disagrees with the stages before it on `pkg.f8` \
             (a capture held by `&` is mutated)"
        );
    }

    #[test]
    fn stage_names_and_codes() {
        let names: Vec<&str> = Stage::ALL.iter().map(|s| s.name()).collect();
        assert_eq!(names[..7], ["read", "lex", "items", "parse", "resolve", "intern", "typeck"]);
        assert!(Stage::ALL.windows(2).all(|w| w[0] < w[1]));
        let of = |c| Stage::of_code(c);
        assert_eq!(of("E0016"), Some(Stage::Items));
        assert_eq!(of("E0004"), Some(Stage::Parse));
        assert_eq!(of("E0021"), Some(Stage::Parse));
        assert_eq!(of("E0024"), Some(Stage::Lex));
        assert_eq!(of("E1002"), Some(Stage::Read));
        assert_eq!(of("E0101"), Some(Stage::Resolve));
        assert_eq!(of("E0107"), Some(Stage::Resolve));
        assert_eq!(of("E0108"), Some(Stage::Lower));
        assert_eq!(of("E1401"), Some(Stage::Lower));
        assert_eq!(of("E1403"), Some(Stage::Lower));
        assert_eq!(of("E1404"), Some(Stage::Typeck));
        assert_eq!(of("E0201"), Some(Stage::Typeck));
        assert_eq!(of("E0909"), Some(Stage::Typeck));
        assert_eq!(of("E0803"), Some(Stage::Typeck));
        assert_eq!(of("E0501"), Some(Stage::Borrowck));
        assert_eq!(of("E0510"), Some(Stage::Borrowck));
        assert_eq!(of("E0513"), Some(Stage::Borrowck));
        assert_eq!(of("E0511"), None, "retired");
        assert_eq!(of("E0807"), Some(Stage::Typeck));
        for c in ["E0514", "E0515", "E0520", "E0521"] {
            assert_eq!(of(c), Some(Stage::Typeck), "{c}");
        }
        assert_eq!(of("E0700"), Some(Stage::Typeck));
        assert_eq!(of("E0701"), Some(Stage::Typeck));
        assert_eq!(of("E0902"), Some(Stage::Typeck));
        assert_eq!(of("E0005"), Some(Stage::Typeck));
        assert_eq!(of("E0300"), Some(Stage::Typeck));
        for c in ["E1001", "E1205", "E1302", "E9999", "X0201", "E201", "E02x1"] {
            assert_eq!(of(c), None, "{c}");
        }
        assert!(Stage::Borrowck.reports_all_codes() && !Stage::Lower.reports_all_codes());
        assert_eq!(Goal::Check.last(), Stage::Borrowck);
    }

    #[test]
    fn lines() {
        assert_eq!(line_count(b""), 0);
        assert_eq!(line_count(b"a"), 1);
        assert_eq!(line_count(b"a\n"), 1);
        assert_eq!(line_count(b"a\nb"), 2);
    }
}
