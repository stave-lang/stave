//! The IR text (contracts/ir-text.ebnf, contracts/checked-ir.md): the one printer of a module and
//! the one reader of the text.
//!
//! - [`print`]: [`print_module`] prints a module as the gate form of the text, or as the emit
//!   form when it is given the stack section; [`print_checked`] prints a bare module, for callers
//!   that have no target, mode, or stack section (tests, debugging). [`spell`]: what the printer
//!   refuses, a module whose text would not load.
//! - [`lex`]: one line to tokens (bytes, any input).
//! - [`line`]: one line parsed on the generated LL(1) table and typed; no line looks at another,
//!   so lines can be parsed in parallel.
//! - [`build`]: the structure pass over the parsed lines and the module build.
//! - [`read`]: lines, structure, then verification (type table, statics, every fn) and the
//!   bounds audit. A module that fails any step is never returned.

mod build;
mod lex;
mod line;
mod print;
mod spell;

use std::fmt;

pub use print::{print_checked, print_module};
use stave_target::Triple;

use crate::bounds;
use crate::ir::{FuncId, Module};
use crate::layout::Layouts;
use crate::types::CanonError;
use crate::verify::{VerifyError, verify_func_body, verify_target, verify_types};

/// At most this many errors are reported.
pub const MAX_ERRORS: usize = 100;

/// The largest static the reader lays out, in bytes.
pub const MAX_STATIC_BYTES: u64 = 1 << 30;

/// The contract version of the text; the header's number.
pub const VERSION: u32 = 4;

/// The build mode of the header.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    Debug,
    Release,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Debug => "debug",
            Mode::Release => "release",
        }
    }
}

/// The IR level of the header (contracts/checked-ir.md "Levels and forms").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Level {
    /// After lowering and the verifier, before the removal of proven checks (`--dump=lir`).
    Lowered,
    /// After the removal of the checks that have a witness (`--dump=ir`).
    Checked,
    /// After the release passes (`--dump=opt`).
    Opt,
}

impl Level {
    pub fn name(self) -> &'static str {
        match self {
            Level::Lowered => "lowered",
            Level::Checked => "checked",
            Level::Opt => "opt",
        }
    }
}

/// The header line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub target: Triple,
    pub mode: Mode,
    pub level: Level,
}

/// The kind of a call-graph edge of the stack section.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    Call,
    Indirect,
    Spawn,
}

impl EdgeKind {
    pub fn name(self) -> &'static str {
        match self {
            EdgeKind::Call => "call",
            EdgeKind::Indirect => "indirect",
            EdgeKind::Spawn => "spawn",
        }
    }
}

/// A call-graph edge of the stack section.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StackEdge {
    pub from: u32,
    pub to: u32,
    pub kind: EdgeKind,
}

/// The stack section (emit form only): what the stack bound reads besides the module.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StackInput {
    /// `runstack`: the root package's `[run] stack` in bytes.
    pub run_stack: Option<u64>,
    /// `truncated`: the call graph is incomplete.
    pub truncated: bool,
    /// Per call-graph node: whether it is a `rec fn`.
    pub nodes: Vec<bool>,
    /// Ascending by (from, to), one per pair.
    pub edges: Vec<StackEdge>,
    /// Per fn (by [`FuncId`]): the nodes it stands for, ascending.
    pub nodes_of_func: Vec<Vec<u32>>,
}

impl StackInput {
    /// The out-edges of node `n`, ascending by target.
    pub fn succ(&self, n: u32) -> &[StackEdge] {
        let a = self.edges.partition_point(|e| e.from < n);
        let b = self.edges.partition_point(|e| e.from <= n);
        &self.edges[a..b.max(a)]
    }
}

/// The lines of one fn's parts, 1-based.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FnLines {
    pub header: u32,
    /// Per block: its header line.
    pub blocks: Vec<u32>,
    /// Per block: its terminator line.
    pub terms: Vec<u32>,
    /// Per instruction (body-wide index): its line.
    pub insts: Vec<u32>,
}

/// Where each part of the module stands in the text: verifier and audit errors, and any later
/// report about the module, are located through it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LineMap {
    pub header: u32,
    /// Per type: its `type` line.
    pub types: Vec<u32>,
    /// Per static: its `static` line.
    pub statics: Vec<u32>,
    /// Per fn (by [`FuncId`]).
    pub funcs: Vec<FnLines>,
}

/// A loaded module and the rest of the text's data.
#[derive(Clone, Debug)]
pub struct Loaded {
    pub header: Header,
    /// The module; its `main` line is [`Module::main`].
    pub module: Module,
    /// The stack section (`None`: the text has none).
    pub stack: Option<StackInput>,
    pub lines: LineMap,
}

/// The step of "Loading" an error comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// Step 1: a line that does not lex or parse.
    Syntax,
    /// Step 2: the structure pass and the build.
    Structure,
    /// Step 3: the IR verifier.
    Verify,
    /// Step 5: the bounds audit.
    Audit,
}

/// One fault of the input, at a 1-based line (and column for steps 1 and 2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadError {
    pub line: u32,
    pub col: Option<u32>,
    pub kind: ErrorKind,
    pub msg: String,
}

impl ReadError {
    /// `FILE:LINE:COL: error: MSG`, or `FILE:LINE: verify: MSG` for the verifier and the audit.
    pub fn render(&self, file: &str) -> String {
        match (self.kind, self.col) {
            (ErrorKind::Syntax | ErrorKind::Structure, Some(c)) => {
                format!("{file}:{}:{c}: error: {}", self.line, self.msg)
            }
            (ErrorKind::Syntax | ErrorKind::Structure, None) => {
                format!("{file}:{}: error: {}", self.line, self.msg)
            }
            _ => format!("{file}:{}: verify: {}", self.line, self.msg),
        }
    }
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render("<ir>"))
    }
}

/// Reads IR text: every step of contracts/checked-ir.md "Loading" the reader owns (lines,
/// structure, verification, audit), each only when the one before had no error.
pub fn read(src: &[u8]) -> Result<Loaded, Vec<ReadError>> {
    let mut errs = Vec::new();
    let mut lines = Vec::new();
    for (k, raw) in src.split(|&c| c == b'\n').enumerate() {
        let no = u32::try_from(k + 1).unwrap_or(u32::MAX);
        match line::parse_line(raw) {
            Ok(Some(p)) => lines.push(build::Ln { no, p }),
            Ok(None) => {}
            Err(e) => {
                errs.push(ReadError { line: no, col: Some(e.col), kind: ErrorKind::Syntax, msg: e.msg });
                if errs.len() >= MAX_ERRORS {
                    break;
                }
            }
        }
    }
    if !errs.is_empty() {
        return Err(errs);
    }
    let built = build::structure(lines)?;
    let loaded = lay_out_statics(built)?;
    verify(&loaded)?;
    audit(&loaded)?;
    Ok(loaded)
}

fn verify_err(line: u32, msg: impl Into<String>) -> ReadError {
    ReadError { line, col: None, kind: ErrorKind::Verify, msg: msg.into() }
}

/// Step 3, first part: the type table is well formed, then each static gets its bytes (its type's
/// size, zero but where `data` lines say otherwise) and its address words.
fn lay_out_statics(b: build::Built) -> Result<Loaded, Vec<ReadError>> {
    let build::Built { header, mut module, statics, stack, lines } = b;
    if let Err(e) = module.types.check_well_formed() {
        let (t, msg) = match e {
            CanonError::BadComponent { ty, component } => {
                (ty, format!("t{}: component t{} out of range", ty.0, component.0))
            }
            CanonError::Cycle(t) => (t, format!("t{}: the type is a component of itself", t.0)),
            CanonError::TooDeep(t) => {
                (t, format!("t{}: nested deeper than {}", t.0, crate::types::MAX_TYPE_DEPTH))
            }
        };
        return Err(vec![verify_err(lines.types.get(t.index()).copied().unwrap_or(lines.header), msg)]);
    }
    let layouts = Layouts::compute(&module.types);
    let mut errs = Vec::new();
    for (i, s) in statics.into_iter().enumerate() {
        let mut init = Vec::new();
        if let Ok(l) = layouts.get_layout(s.ty) {
            if l.size > MAX_STATIC_BYTES {
                errs.push(verify_err(
                    s.line,
                    format!(
                        "static {i}: {} bytes, more than this reader lays out ({MAX_STATIC_BYTES})",
                        l.size
                    ),
                ));
            } else {
                init = vec![0u8; l.size as usize];
                check_data_inside_statics(&mut errs, i, &s, &mut init);
            }
        }
        module.statics.push(crate::ir::Static { path: s.path, ty: s.ty, init, refs: s.refs });
    }
    if !errs.is_empty() {
        return Err(errs);
    }
    Ok(Loaded { header, module, stack, lines })
}

/// The `data` bytes of static `i` lie inside its type's size (checked-ir.md "Loading" step 2,
/// which needs the layout of the verified type table); they are copied into `init`.
fn check_data_inside_statics(errs: &mut Vec<ReadError>, i: usize, s: &build::StaticG, init: &mut [u8]) {
    for (line, _, offset, bytes) in &s.data {
        let at = *offset as usize;
        match init.get_mut(at..at.saturating_add(bytes.len())) {
            Some(dst) => dst.copy_from_slice(bytes),
            None => errs.push(verify_err(
                *line,
                format!(
                    "static {i}: `data` bytes at {at}..{} lie past its size {}",
                    at + bytes.len(),
                    init.len()
                ),
            )),
        }
    }
}

/// Step 3: the verifier on the type table, shapes and statics, then on every fn, then against
/// the header's target.
fn verify(l: &Loaded) -> Result<(), Vec<ReadError>> {
    let m = &l.module;
    let mut errs: Vec<VerifyError> = match verify_types(m) {
        Ok(es) => es,
        Err(es) => return Err(es.iter().map(|e| locate(l, e)).take(MAX_ERRORS).collect()),
    };
    for f in 0..m.funcs.len() {
        errs.extend(verify_func_body(m, FuncId(f as u32)));
        if errs.len() >= MAX_ERRORS {
            break;
        }
    }
    if errs.is_empty() {
        errs = verify_target(m, l.header.target.arch());
    }
    if errs.is_empty() { Ok(()) } else { Err(errs.iter().map(|e| locate(l, e)).take(MAX_ERRORS).collect()) }
}

/// Step 5: the bounds audit of every fn, reading calls by the loaded `claim` rows (levels
/// `checked` and `opt`) or by the summaries computed from the bodies (`lowered`).
fn audit(l: &Loaded) -> Result<(), Vec<ReadError>> {
    let m = &l.module;
    let mut errs = Vec::new();
    for f in 0..m.funcs.len() {
        for u in bounds::audit_func(m, FuncId(f as u32)).1 {
            let line = l.lines.funcs.get(f).and_then(|x| x.insts.get(u.inst as usize).copied());
            let line = line.or_else(|| l.lines.funcs.get(f).map(|x| x.header)).unwrap_or(l.lines.header);
            errs.push(ReadError {
                line,
                col: None,
                kind: ErrorKind::Audit,
                msg: format!("{:?} access without a bounds check or proof", u.kind),
            });
        }
        if errs.len() >= MAX_ERRORS {
            break;
        }
    }
    if errs.is_empty() { Ok(()) } else { Err(errs) }
}

/// A verifier error at the line of what it is about.
fn locate(l: &Loaded, e: &VerifyError) -> ReadError {
    verify_err(verify_error_line(&l.lines, e), e.msg.clone())
}

/// The line a verifier error is about: its instruction; for a block error, the block's
/// terminator (the block header for the entry-params check); its fn's header; for a module
/// error, the `type` or `static` line its message names, else the header.
fn verify_error_line(map: &LineMap, e: &VerifyError) -> u32 {
    let Some(f) = e.func.and_then(|f| map.funcs.get(f.index())) else {
        return module_error_line(map, &e.msg);
    };
    match (e.block, e.inst) {
        (_, Some(i)) => f.insts.get(i as usize).copied().unwrap_or(f.header),
        (Some(b), None) => {
            let lines = if e.msg.starts_with("b0 params") { &f.blocks } else { &f.terms };
            lines.get(b.index()).copied().unwrap_or(f.header)
        }
        (None, None) => f.header,
    }
}

/// The line a module-level verifier message (`t<N>: ...`, `static <N>: ...`) names.
fn module_error_line(map: &LineMap, msg: &str) -> u32 {
    let num = |s: &str| -> Option<usize> {
        let d: String = s.chars().take_while(char::is_ascii_digit).collect();
        if s[d.len()..].starts_with(':') { d.parse().ok() } else { None }
    };
    if let Some(rest) = msg.strip_prefix("static ")
        && let Some(i) = num(rest)
    {
        return map.statics.get(i).copied().unwrap_or(map.header);
    }
    if let Some(rest) = msg.strip_prefix('t')
        && let Some(i) = num(rest)
    {
        return map.types.get(i).copied().unwrap_or(map.header);
    }
    map.header
}
