//! `stave ir VERB FILE` (contracts/cli.md): IR text in, through the passes a build of a program
//! runs, as glue over [`stave_ir::text::read`], [`build`] and [`interp`].
//!
//! - [`load`]: contracts/checked-ir.md "Loading": the reader (lines, structure, verification,
//!   audit), then at level `lowered` the removal of proven checks (step 4) and the canonical
//!   numbering, as after lowering.
//! - [`fmt`], [`opt`]: the canonical text of the loaded module, or of the release passes' output.
//! - [`build`]: the image, as `stave build` makes it from the same module.
//! - [`run`]: the loaded module's `main` in the interpreter.

use stave_ir::text::{self, Header, Level, Mode, StackInput};
use stave_ir::{FuncId, Module, Site};

use crate::build::{self, Built, Failure};
use crate::effects;
use crate::interp::{self, Interpreted, Limits};

/// Why a verb produced nothing. The exit code is [`IrError::exit`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IrError {
    /// FILE cannot be read (exit 2).
    Io(String),
    /// The input is rejected (exit 1): one rendered line per fault.
    Input(Vec<String>),
    /// A compiler bug (exit 3): a witness the checker refused, or a pass that failed on a module
    /// the loader accepted.
    Bug(String),
    /// The program uses something stage0 cannot compile yet (exit 2).
    Unsupported(String),
    /// The interpreted run failed (exit 1): a memory error of the checking mode, a run that did
    /// not exit.
    Run(String),
}

impl IrError {
    pub fn exit(&self) -> i32 {
        match self {
            IrError::Input(_) | IrError::Run(_) => 1,
            IrError::Io(_) | IrError::Unsupported(_) => 2,
            IrError::Bug(_) => 3,
        }
    }

    /// The lines to print on stderr.
    pub fn lines(&self) -> Vec<String> {
        match self {
            IrError::Input(ls) => ls.clone(),
            IrError::Io(m) | IrError::Bug(m) | IrError::Unsupported(m) | IrError::Run(m) => {
                vec![format!("stave: {m}")]
            }
        }
    }

    fn failure(f: Failure) -> IrError {
        match f {
            Failure::Unsupported { .. } | Failure::Target(_) => IrError::Unsupported(f.to_string()),
            f => IrError::Bug(f.to_string()),
        }
    }
}

/// A loaded file.
#[derive(Clone, Debug)]
pub struct LoadedIr {
    /// The name errors are located by.
    pub file: String,
    pub header: Header,
    /// The module as the file holds it (its level), canonical.
    pub read: Module,
    /// The module the passes after loading get: at level `lowered` after the removal of proven
    /// checks and the canonical numbering, else [`LoadedIr::read`].
    pub module: Module,
    /// The stack section of [`LoadedIr::read`], and of [`LoadedIr::module`] (an unchecked entry
    /// stands for the nodes of the fn it is an entry of).
    pub read_stack: Option<StackInput>,
    pub stack: Option<StackInput>,
}

/// Reads and loads `src` (checked-ir.md "Loading", steps 1 to 5); `file` names it in errors.
pub fn load(file: &str, src: &[u8]) -> Result<LoadedIr, IrError> {
    let loaded = text::read(src).map_err(|es| IrError::Input(es.iter().map(|e| e.render(file)).collect()))?;
    let text::Loaded { header, mut module, mut stack, lines } = loaded;
    // The reader keeps the text's type numbering; the module the passes get is canonical. Fns,
    // blocks and instructions keep their places (the reader numbered them canonically), so the
    // line map still locates them.
    let remap = build::canonical(&mut module);
    if let Some(s) = &mut stack {
        stave_ir::canon::permute_by_func(&mut s.nodes_of_func, &remap);
    }
    let read = module.clone();
    let read_stack = stack.clone();
    if header.level != Level::Lowered {
        return Ok(LoadedIr { file: file.to_string(), header, read, module, read_stack, stack });
    }
    let mut m = module;
    let r = build::remove_proven_checks(&mut m, true).map_err(IrError::failure)?;
    let mut errs = Vec::new();
    let line_of = |f: FuncId, site: Site| -> u32 {
        let body = read.func(f).and_then(|x| x.body.as_ref());
        let inst = body.and_then(|b| b.insts.site.iter().position(|s| *s == site));
        let fl = lines.funcs.get(f.index());
        inst.and_then(|i| fl?.insts.get(i).copied()).or(fl.map(|x| x.header)).unwrap_or(lines.header)
    };
    // The faults a build reports as errors (E0215, E0217, E0218), by the same rules: none in an
    // instance of a generic fn.
    let none = |_| stave_diag::Span::new(stave_diag::FileId(0), 0, 0);
    for x in &r.found {
        let generic = read.func(x.func).is_some_and(|f| !f.type_args.is_empty());
        if effects::contract_diag(x, generic, none).is_some_and(|d| d.0 != "E0216") {
            errs.push(format!(
                "{file}:{}: verify: a contract check that fails whenever it is reached",
                line_of(x.func, x.site)
            ));
        }
    }
    let uncallable = effects::uncallable(&read, &r.found);
    for c in r.calls.iter().filter(|c| effects::call_reported(&read, &uncallable, c)) {
        let callee = read.func(c.callee).map_or("?", |g| g.path.as_str());
        errs.push(format!(
            "{file}:{}: verify: the call fails a `requires` of `{callee}` whenever it is reached",
            line_of(c.func, c.site)
        ));
    }
    if !errs.is_empty() {
        return Err(IrError::Input(errs));
    }
    if let Some(s) = &mut stack {
        build::extend_nodes(&mut s.nodes_of_func, &r.removal);
    }
    let remap = build::canonical(&mut m);
    if let Some(s) = &mut stack {
        stave_ir::canon::permute_by_func(&mut s.nodes_of_func, &remap);
    }
    Ok(LoadedIr { file: file.to_string(), header, read, module: m, read_stack, stack })
}

/// `fmt`: the canonical text of the loaded module at the file's level, with the file's stack
/// section.
pub fn fmt(l: &LoadedIr) -> Result<String, IrError> {
    print(&l.read, &l.header, l.read_stack.as_ref())
}

/// `opt`: the release passes on the loaded module, their verifier and audit, printed at level
/// `opt`. Only a `release` file at level `lowered` or `checked`.
pub fn opt(l: &LoadedIr) -> Result<String, IrError> {
    if l.header.mode != Mode::Release || l.header.level == Level::Opt {
        return Err(IrError::Input(vec![format!(
            "{}:1: error: `stave ir opt` needs a `release` file at level `lowered` or `checked`",
            l.file
        )]));
    }
    let m = build::release_passes(&l.module).map_err(IrError::failure)?;
    print(&m, &Header { level: Level::Opt, ..l.header }, None)
}

/// `build`: the image for the header's target, as `stave build` makes it: a `release` file at
/// level `opt` skips the release passes (the loader verified and audited it). The stack table
/// comes from the stack section, empty without one.
pub fn build(l: &LoadedIr, threads: Option<usize>) -> Result<Built, IrError> {
    if l.module.main.is_none() {
        return Err(IrError::Input(vec![format!(
            "{}:1: error: the file has no `main` line: nothing to build",
            l.file
        )]));
    }
    let (m, t, stack) = (&l.module, l.header.target, l.stack.as_ref());
    let built = match (l.header.mode, l.header.level) {
        (Mode::Release, Level::Opt) => build::codegen_link_optimized(m, t, threads, stack),
        (mode, _) => build::codegen_link(m, t, mode == Mode::Release, threads, stack),
    };
    built.map_err(IrError::failure)
}

/// `run`: the loaded module's `main` in the interpreter with `args` (argv after the program
/// name), on the module the build would compile (no borrow side table: the text has none).
pub fn run(l: &LoadedIr, args: &[&[u8]]) -> Result<Interpreted, IrError> {
    let Some(main) = l.module.main.and_then(|f| l.module.func(f)) else {
        return Err(IrError::Input(vec![format!(
            "{}:1: error: the file has no `main` line: nothing to run",
            l.file
        )]));
    };
    let limits = Limits { borrow: false, ..Limits::default() };
    interp::run(&l.module, &main.path.clone(), l.header.target, args, &[], limits).map_err(IrError::Run)
}

fn print(m: &Module, h: &Header, stack: Option<&StackInput>) -> Result<String, IrError> {
    stave_ir::print_module(m, h, stack).map_err(|es| {
        IrError::Bug(format!("IR verifier: {}", es.first().map_or_else(String::new, ToString::to_string)))
    })
}
