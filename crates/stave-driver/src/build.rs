//! The stage0 back end: lowering to the checked IR (mono + lower, stave-sema),
//! the debug code generators (x86-64: stave-codegen-x64, AArch64: stave-codegen-a64), and the static
//! ELF link (stave-obj).
//!
//! - [`lower`]: [`stave_sema::lower`] plus the IR verifier. Lowering errors that are all of kind
//!   [`LowerErrorKind::Unsupported`] (a construct stage0 does not lower yet) are
//!   [`Failure::Unsupported`]; any other lowering or verifier error is a compiler bug
//!   ([`Failure::Internal`]).
//! - [`lower`] also runs the bounds audit ([`stave_ir::bounds::audit`]): an element access without a
//!   check or proof is a compiler bug.
//! - [`codegen_link`]: release builds first run the IR passes of stave-opt around the elision of
//!   the bounds checks that have a recorded proof ([`stave_ir::bounds::elide`], records in
//!   [`Built::elided`]); then per-function code generation in parallel (rayon on the caller's pool for
//!   AArch64; stave-codegen-x64 uses its own scoped threads), then the link. Debug builds use
//!   stave-obj's debug layout (padded slots, jump table), release builds the packed one. Backend
//!   errors on valid IR (vectors, f16) are [`Failure::Unsupported`].
//! - [`main_path`]: the checked-IR path of the root package's `main`.

use std::fmt;
use std::time::Instant;

use rayon::prelude::*;
use stave_ir::{FuncId, Module};
use stave_obj::{FuncCode, ImageBuilder, Machine, ObjectImage};
use stave_sema::{DeclKind, LowerErrorKind, Mode, Program, Typeck};
use stave_target::{Arch, Triple};

use crate::pipeline::Stage;
use crate::prof;

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

/// Lowers (mono + lower) and verifies. The program must have no error diagnostics.
pub fn lower(prog: &Program, tc: &Typeck, mode: Mode) -> Result<Module, Failure> {
    let (lowered, tm) = stave_sema::lower_timed(prog, tc, mode);
    prof::part(".rounds", tm.rounds);
    prof::part(".assemble", tm.assemble);
    prof::mark("lower");
    let m = lowered.map_err(|es| {
        let message = listed(&es);
        if es.iter().all(|e| e.kind == LowerErrorKind::Unsupported) {
            Failure::Unsupported { stage: Stage::Lower, message }
        } else {
            Failure::Internal { stage: Stage::Lower, message }
        }
    })?;
    verify(&m).map_err(|es| {
        let named: Vec<String> = es
            .iter()
            .map(|e| {
                let f = e.func.and_then(|f| m.func(f)).map_or(String::new(), |f| format!("{}: ", f.path));
                format!("{f}{e}")
            })
            .collect();
        Failure::Internal { stage: Stage::Lower, message: format!("IR verifier: {}", listed(&named)) }
    })?;
    prof::mark("verify");
    // Every element access keeps its bounds check or a proof.
    let bad = unguarded(&m);
    if !bad.is_empty() {
        return Err(Failure::Internal {
            stage: Stage::Lower,
            message: format!("bounds audit: {}", listed(&bad)),
        });
    }
    prof::mark("bounds-audit");
    Ok(m)
}

/// [`stave_ir::verify_module`] with the functions verified in parallel (same errors, same order).
fn verify(m: &Module) -> Result<(), Vec<stave_ir::VerifyError>> {
    let mut errs = stave_ir::verify::verify_types(m)?;
    let per_fn: Vec<Vec<stave_ir::VerifyError>> = (0..m.funcs.len() as u32)
        .into_par_iter()
        .map(|f| stave_ir::verify::verify_func_body(m, FuncId(f)))
        .collect();
    errs.extend(per_fn.into_iter().flatten());
    if errs.is_empty() { Ok(()) } else { Err(errs) }
}

/// The unguarded element accesses of [`stave_ir::bounds::audit`], the functions audited in
/// parallel (same list, same order).
fn unguarded(m: &Module) -> Vec<stave_ir::bounds::Unguarded> {
    let per_fn: Vec<Vec<stave_ir::bounds::Unguarded>> = (0..m.funcs.len() as u32)
        .into_par_iter()
        .map(|f| stave_ir::bounds::audit_func(m, FuncId(f)).1)
        .collect();
    per_fn.into_iter().flatten().collect()
}

/// A linked executable and the back-end timings.
pub struct Built {
    pub image: ObjectImage,
    pub codegen_ms: f64,
    pub link_ms: f64,
    /// Functions with a body (compiled).
    pub funcs: u64,
    /// Bounds checks removed with their proofs (release builds only; stave_ir::bounds).
    pub elided: Vec<stave_ir::bounds::Elided>,
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// The image layout of a debug or release build for `machine`.
pub fn image_options(machine: Machine, release: bool) -> stave_obj::Options {
    if release { stave_obj::Options::release(machine) } else { stave_obj::Options::debug(machine) }
}

/// Generates code for every function of `m` and links the static executable whose `main` is
/// `main` (`__stave_main`; contracts/intrinsics.md).
pub fn codegen_link(
    m: &Module,
    main: FuncId,
    triple: Triple,
    release: bool,
    threads: Option<usize>,
) -> Result<Built, Failure> {
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
    let funcs = m.funcs.iter().filter(|f| f.body.is_some()).count() as u64;
    // Release: drop the bounds checks that have a recorded proof (the checked IR keeps them).
    let mut elided = Vec::new();
    let owned;
    let m = if release {
        let mut c = m.clone();
        c.funcs.par_iter_mut().filter_map(|f| f.body.as_mut()).for_each(stave_opt::before_elision);
        elided = stave_ir::bounds::elide(&mut c);
        stave_opt::slices::declare_elem_pointers(&mut c.types);
        let types = &c.types;
        c.funcs
            .par_iter_mut()
            .filter_map(|f| f.body.as_mut())
            .for_each(|b| stave_opt::after_elision(types, b));
        owned = c;
        &owned
    } else {
        m
    };
    let mut built = match triple.arch() {
        Arch::X86_64 => x64(m, main, triple.baseline(), release, threads, funcs),
        Arch::Aarch64 => a64(m, main, release, funcs),
        Arch::Gpu => Err(Failure::Target(triple)),
    }?;
    built.elided = elided;
    Ok(built)
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
) -> Result<Built, Failure> {
    use stave_codegen_x64::Error as E;
    let classify = |stage: Stage, e: E| match e {
        E::Unsupported { .. } => Failure::Unsupported { stage, message: e.to_string() },
        _ => Failure::Internal { stage, message: e.to_string() },
    };
    let image = image_options(Machine::X86_64, release);
    let o = stave_codegen_x64::Options {
        image: image.clone(),
        main: Some(main),
        // The driver verified the module after lowering.
        verify: false,
        threads: threads.unwrap_or(0),
        release,
        features: baseline,
        ..stave_codegen_x64::Options::default()
    };
    let t0 = Instant::now();
    let mut b = ImageBuilder::new(Machine::X86_64, image);
    let e = stave_codegen_x64::emit_module(m, &o, &mut b).map_err(|e| classify(Stage::Codegen, e))?;
    let codegen_ms = ms(t0);
    prof::mark("codegen");
    let t1 = Instant::now();
    let img = stave_codegen_x64::link(b, &e).map_err(|e| classify(Stage::Link, e))?;
    prof::mark("link");
    Ok(Built { image: img, codegen_ms, link_ms: ms(t1), funcs, elided: Vec::new() })
}

fn a64(m: &Module, main: FuncId, release: bool, funcs: u64) -> Result<Built, Failure> {
    use stave_codegen_a64::debug::{Codegen, Error, ErrorKind, FuncScratch, prepare};
    let classify = |stage: Stage, e: Error| match e.kind {
        ErrorKind::Unsupported(_) => Failure::Unsupported { stage, message: e.to_string() },
        _ => Failure::Internal { stage, message: e.to_string() },
    };
    let t0 = Instant::now();
    let mut b = ImageBuilder::new(Machine::Aarch64, image_options(Machine::Aarch64, release));
    let syms = prepare(&mut b, m, Some(main)).map_err(|e| classify(Stage::Codegen, e))?;
    let cg = Codegen::new(m, &syms).release(release);
    let ids: Vec<FuncId> =
        (0..m.funcs.len()).filter(|&i| m.funcs[i].body.is_some()).map(|i| FuncId(i as u32)).collect();
    let codes: Vec<Result<FuncCode, Error>> =
        ids.par_iter().map_init(FuncScratch::default, |s, &f| cg.compile(f, s)).collect();
    for (&f, code) in ids.iter().zip(codes) {
        let code = code.map_err(|e| classify(Stage::Codegen, e))?;
        let sym = syms.funcs.get(f.index()).copied().flatten().ok_or_else(|| Failure::Internal {
            stage: Stage::Codegen,
            message: format!("fn #{} has no symbol", f.0),
        })?;
        b.define_func(sym, code)
            .map_err(|e| Failure::Internal { stage: Stage::Codegen, message: e.to_string() })?;
    }
    let codegen_ms = ms(t0);
    prof::mark("codegen");
    let t1 = Instant::now();
    let img = b.link().map_err(|e| Failure::Internal { stage: Stage::Link, message: e.to_string() })?;
    prof::mark("link");
    Ok(Built { image: img, codegen_ms, link_ms: ms(t1), funcs, elided: Vec::new() })
}

#[cfg(test)]
mod tests {
    use super::*;

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
