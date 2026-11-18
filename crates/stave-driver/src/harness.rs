//! `test fn` harness of `stave test PKG_DIR` (LANG.md §2: "`test fn name() { }`
//! runs under `stave test` and is excluded from builds").
//!
//! The package is lowered once in [`stave_sema::Mode::Test`] (every `test fn` is a root); the
//! harness entry table is the list of the root package's `test fn`s in canonical decl order
//! ([`crate::exec::test_fns`]). Each entry is linked as its own executable whose `__stave_main`
//! calls that test (one process per test, so a panic, exit 101, fails only that test), and run
//! from memory ([`crate::exec::Executable`]) with the package directory as the working
//! directory. A test passes iff it exits 0.
//!
//! A `test fn` with parameters is called by a generated fn instead ([`add_caller`]): one call per
//! input of [`stave_sema::testgen::inputs`], in order, all in one process. Scalars are constants;
//! an array or tuple value and the elements of a slice are statics. The test fn writes a
//! stdout marker as each input starts and another when its `requires` skips the input
//! (contracts/cli.md "Test fn inputs"): their counts give the inputs run and skipped, and on a
//! failure the input that failed, which `--filter T --exact --input N` runs alone.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

use stave_ir::layout::{Layouts, Shape};
use stave_ir::{Const, FuncBuilder, FuncId, InstanceId, Module, Op, Operand, Static, TypeId, TypeKey};
use stave_sema::testgen::{Domain, Gen, Value};

use crate::build::{self, Failure};
use crate::exec::{self, Executable, RunSpec};
use crate::pipeline::{self, Goal, Options};
use crate::shipped::Library;

/// The stdout line a `test fn` with parameters writes as an input starts
/// (`runtime.panic.test_input`).
pub const INPUT_MARKER: &[u8] = b"\x01stave-test input\n";
/// The stdout line it writes when a `requires` clause skips the input (`runtime.panic.test_skip`).
pub const SKIP_MARKER: &[u8] = b"\x01stave-test skip\n";

/// One `test fn` run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TestFnResult {
    /// Checked-IR path (`main.adds_up`).
    pub name: String,
    pub outcome: TestFnOutcome,
    /// A `test fn` with parameters that ran: its input counts.
    pub inputs: Option<InputReport>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TestFnOutcome {
    Passed,
    /// Exit status and stderr of the test process.
    Failed {
        exit: i32,
        stderr: String,
    },
    /// Built but not runnable here (E1001).
    NotRun,
    /// A `test fn` with parameters that ran no input, or whose process stopped before its last
    /// input with status 0 (the reason): a failure.
    Incomplete(String),
}

/// What happened to the inputs of a `test fn` with parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputReport {
    /// Inputs passed to the test fn.
    pub total: usize,
    /// Inputs it started (all of them unless one failed).
    pub started: usize,
    /// Inputs its `requires` clauses skipped.
    pub skipped: usize,
    /// The input that failed: its index and its parameters as Stave literals.
    pub failing: Option<(usize, String)>,
    /// The command that runs the failing input alone.
    pub replay: Option<String>,
}

/// Everything the harness did.
#[derive(Debug, Default)]
pub struct TestFnReport {
    pub results: Vec<TestFnResult>,
}

impl TestFnReport {
    pub fn passed(&self) -> usize {
        self.results.iter().filter(|r| r.outcome == TestFnOutcome::Passed).count()
    }

    pub fn failed(&self) -> usize {
        self.results
            .iter()
            .filter(|r| matches!(r.outcome, TestFnOutcome::Failed { .. } | TestFnOutcome::Incomplete(_)))
            .count()
    }

    pub fn not_run(&self) -> usize {
        self.results.iter().filter(|r| r.outcome == TestFnOutcome::NotRun).count()
    }

    pub fn render(&self) -> String {
        let mut s = String::new();
        for r in &self.results {
            match &r.outcome {
                TestFnOutcome::Passed => match &r.inputs {
                    Some(i) => {
                        let run = i.started - i.skipped;
                        let n = if run == 1 { "input" } else { "inputs" };
                        let _ =
                            writeln!(s, "ok        test {}: {run} {n} run, {} skipped", r.name, i.skipped);
                        // Fewer than one input in ten ran: the ranges hardly narrow the domain.
                        if run * 10 < i.total {
                            let _ = writeln!(
                                s,
                                "    note: `requires` skipped {} of {} inputs; only comparisons of a parameter or `p.len()` with a constant narrow the inputs",
                                i.skipped, i.total
                            );
                        }
                    }
                    None => {
                        let _ = writeln!(s, "ok        test {}", r.name);
                    }
                },
                TestFnOutcome::NotRun => {
                    let _ = writeln!(s, "not run   test {}: E1001: target not runnable on this host", r.name);
                }
                TestFnOutcome::Incomplete(why) => {
                    let _ = writeln!(s, "FAILED    test {}: {why}", r.name);
                }
                TestFnOutcome::Failed { exit, stderr } => {
                    match r.inputs.as_ref().and_then(|i| i.failing.as_ref().map(|f| (i, f))) {
                        Some((i, (k, _))) => {
                            let _ = writeln!(
                                s,
                                "FAILED    test {} (exit {exit}) at input {k} ({} of {} started)",
                                r.name, i.started, i.total
                            );
                        }
                        None => {
                            let _ = writeln!(s, "FAILED    test {} (exit {exit})", r.name);
                        }
                    }
                    for l in stderr.lines() {
                        let _ = writeln!(s, "    {l}");
                    }
                    if let Some(i) = &r.inputs {
                        if let Some((k, shown)) = &i.failing {
                            let _ = writeln!(s, "    input {k}: {shown}");
                        }
                        if let Some(c) = &i.replay {
                            let _ = writeln!(s, "    replay: {c}");
                        }
                    }
                }
            }
        }
        let _ = writeln!(
            s,
            "test fns: {} passed, {} failed, {} not run",
            self.passed(),
            self.failed(),
            self.not_run()
        );
        s
    }
}

/// Why the harness could not run.
#[derive(Debug)]
pub enum HarnessError {
    /// The package has error diagnostics (`stave test` reports them through the conformance run).
    Diagnostics,
    Failure(Failure),
    Io(String),
    /// `--input N` with a selection of no `test fn`, of several, or of one without parameters
    /// (the reason): a usage error.
    Selection(String),
}

/// Which tests and inputs to run.
#[derive(Clone, Copy, Debug, Default)]
pub struct Selection<'a> {
    /// A substring of the test's path, or (`exact`) the whole path.
    pub filter: Option<&'a str>,
    /// `--exact`: `filter` must equal the test's path.
    pub exact: bool,
    /// `--runner CMD`.
    pub runner: Option<&'a str>,
    /// `--input N`: only input N of the one selected `test fn`, which must have parameters.
    pub input: Option<usize>,
}

impl Selection<'_> {
    /// Whether the test with checked-IR path `name` is selected.
    pub fn selects(&self, name: &str) -> bool {
        match self.filter {
            None => true,
            Some(f) if self.exact => name == f,
            Some(f) => name.contains(f),
        }
    }
}

/// Builds and runs the `test fn`s of the package at `root` selected by `sel`.
/// `opts.goal`/`opts.tests`/`opts.lower` are set here.
pub fn run(
    pool: &rayon::ThreadPool,
    root: &Path,
    lib: &Library,
    opts: &Options,
    sel: Selection<'_>,
) -> Result<TestFnReport, HarnessError> {
    let mut o = opts.clone();
    o.goal = Goal::Check;
    o.tests = true;
    o.lower = true;
    let mut out = pipeline::run_on(pool, root, lib, &o);
    if out.has_errors() {
        return Err(HarnessError::Diagnostics);
    }
    if let Some(f) = out.failure.take() {
        return Err(HarnessError::Failure(f));
    }
    let Some(mut module) = out.module.take() else { return Ok(TestFnReport::default()) };
    let triple = opts
        .target
        .or_else(stave_target::Triple::host)
        .ok_or(HarnessError::Failure(Failure::NoHostTarget))?;
    let runner = sel.runner;
    let runnable = runner.is_some_and(|r| !r.trim().is_empty()) || exec::can_run(triple);
    let cwd = if root.is_dir() { root.to_path_buf() } else { root.parent().unwrap_or(root).to_path_buf() };
    let mut report = TestFnReport::default();
    let domain_of =
        |decl| out.typeck.as_ref().and_then(|tc| tc.body(decl)).and_then(|b| b.test_domain.as_deref());
    let tests: Vec<(usize, (String, FuncId, stave_sema::DeclId))> = exec::test_fns(&out.program, &module)
        .into_iter()
        .enumerate()
        .filter(|(_, (name, ..))| sel.selects(name))
        .collect();
    if let Some(i) = sel.input {
        let names: Vec<&str> = tests.iter().map(|(_, (n, ..))| n.as_str()).collect();
        let what = match (sel.filter, sel.exact) {
            (None, _) => "the package".to_string(),
            (Some(f), true) => format!("--filter {f} --exact"),
            (Some(f), false) => format!("--filter {f}"),
        };
        let why = match tests.as_slice() {
            [] => Some(format!("--input {i}: {what} selects no test fn")),
            [(_, (name, _, decl))] if domain_of(*decl).is_none() => {
                Some(format!("--input {i}: test fn {name} has no parameters, so it has no input {i}"))
            }
            [_] => None,
            _ => Some(format!(
                "--input {i}: {what} selects {} test fns ({}); select one: --filter NAME --exact",
                names.len(),
                names.join(", ")
            )),
        };
        if let Some(why) = why {
            return Err(HarnessError::Selection(why));
        }
    }
    for (k, (name, f, decl)) in tests {
        let domain = domain_of(decl);
        let Some(domain) = domain else {
            let built = pool
                .install(|| build::codegen_link(&module, f, triple, opts.release, opts.threads, None))
                .map_err(HarnessError::Failure)?;
            let outcome = if !runnable {
                TestFnOutcome::NotRun
            } else {
                let spec = RunSpec { args: &[], runner, cwd: Some(&cwd), inherit: false };
                let r = Executable::new(&format!("test{k}"), built.image.bytes())
                    .and_then(|mut x| x.run(&spec))
                    .map_err(|e| HarnessError::Io(format!("test {name}: {e}")))?;
                if r.exit == 0 {
                    TestFnOutcome::Passed
                } else {
                    TestFnOutcome::Failed {
                        exit: r.exit,
                        stderr: String::from_utf8_lossy(&r.stderr).into_owned(),
                    }
                }
            };
            report.results.push(TestFnResult { name, outcome, inputs: None });
            continue;
        };
        let all = stave_sema::testgen::inputs(domain);
        let chosen: Vec<usize> = match sel.input {
            Some(i) if i < all.len() => vec![i],
            Some(i) => {
                let why = format!(
                    "no input {i}: the test has {} inputs (0 to {})",
                    all.len(),
                    all.len().max(1) - 1
                );
                report.results.push(TestFnResult {
                    name,
                    outcome: TestFnOutcome::Incomplete(why),
                    inputs: None,
                });
                continue;
            }
            None => (0..all.len()).collect(),
        };
        if chosen.is_empty() {
            let why = "its `requires` ranges admit no input".to_string();
            report.results.push(TestFnResult { name, outcome: TestFnOutcome::Incomplete(why), inputs: None });
            continue;
        }
        let picked: Vec<&[Value]> = chosen.iter().map(|&i| all[i].as_slice()).collect();
        let caller = add_caller(&mut module, f, domain, &picked, k).map_err(|message| {
            HarnessError::Failure(Failure::Internal { stage: pipeline::Stage::Lower, message })
        })?;
        let built = pool
            .install(|| build::codegen_link(&module, caller, triple, opts.release, opts.threads, None))
            .map_err(HarnessError::Failure)?;
        if !runnable {
            report.results.push(TestFnResult { name, outcome: TestFnOutcome::NotRun, inputs: None });
            continue;
        }
        let spec = RunSpec { args: &[], runner, cwd: Some(&cwd), inherit: false };
        let r = Executable::new(&format!("test{k}"), built.image.bytes())
            .and_then(|mut x| x.run(&spec))
            .map_err(|e| HarnessError::Io(format!("test {name}: {e}")))?;
        let (started, skipped) = (count(&r.stdout, INPUT_MARKER), count(&r.stdout, SKIP_MARKER));
        let mut inputs = InputReport { total: chosen.len(), started, skipped, failing: None, replay: None };
        let outcome = if r.exit == 0 {
            if started != chosen.len() {
                // The process ended with status 0 before its last input (an explicit exit).
                TestFnOutcome::Incomplete(format!("exited after {started} of {} inputs", chosen.len()))
            } else if started == skipped {
                TestFnOutcome::Incomplete(format!("`requires` skipped all {skipped} inputs"))
            } else {
                TestFnOutcome::Passed
            }
        } else {
            if let Some(&i) = started.checked_sub(1).and_then(|s| chosen.get(s)) {
                inputs.failing = Some((i, domain.show(&all[i])));
                inputs.replay = Some(replay(root, &name, i, opts, triple, runner));
            }
            TestFnOutcome::Failed { exit: r.exit, stderr: String::from_utf8_lossy(&r.stderr).into_owned() }
        };
        report.results.push(TestFnResult { name, outcome, inputs: Some(inputs) });
    }
    Ok(report)
}

/// How often `pat` occurs in `hay` (non-overlapping).
pub fn count(hay: &[u8], pat: &[u8]) -> usize {
    let mut n = 0;
    let mut i = 0;
    while i + pat.len() <= hay.len() {
        if &hay[i..i + pat.len()] == pat {
            n += 1;
            i += pat.len();
        } else {
            i += 1;
        }
    }
    n
}

/// The `stave test` command that runs input `i` of test `name` alone, in the same build (`--exact`:
/// no other test whose path contains `name` is selected).
fn replay(
    root: &Path,
    name: &str,
    i: usize,
    opts: &Options,
    triple: stave_target::Triple,
    runner: Option<&str>,
) -> String {
    let mut c = format!(
        "stave test {} --filter {} --exact --input {i}",
        word(&root.display().to_string()),
        word(name)
    );
    if opts.release {
        c.push_str(" --release");
    }
    let _ = write!(c, " --target {triple}");
    if let Some(r) = runner {
        let _ = write!(c, " --runner {}", word(r));
    }
    c
}

/// `s` as one shell word.
fn word(s: &str) -> String {
    if !s.is_empty() && s.bytes().all(|b| b.is_ascii_alphanumeric() || b"_./=+:,-".contains(&b)) {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', "'\\''"))
    }
}

/// Adds `__stave_test_inputs.<tag>`, a fn without parameters that calls `test` once per input of
/// `inputs` (values of `d`'s parameters), in order, and returns. An array or tuple argument is
/// loaded from a static holding its bytes; a slice argument is made from a static `[T; n]` of its
/// elements (equal statics shared).
pub fn add_caller(
    m: &mut Module,
    test: FuncId,
    d: &Domain,
    inputs: &[&[Value]],
    tag: usize,
) -> Result<FuncId, String> {
    let f = m.func(test).ok_or("no test fn")?;
    let (params, ret) = (f.params.clone(), f.ret);
    if ret.is_some() || params.len() != d.params.len() {
        return Err(format!("`{}`: not a `test fn` with the domain's parameters", f.path));
    }
    let lay = Layouts::compute(&m.types);
    let sa = m.intrinsic("static_addr");
    let mut cx = Caller { m, lay, sa, tag, statics: HashMap::new() };
    let mut b = FuncBuilder::new(&[]);
    for input in inputs {
        let mut args = Vec::with_capacity(params.len());
        for ((p, &t), v) in d.params.iter().zip(&params).zip(input.iter()) {
            args.push(cx.arg(&mut b, &p.values, t, v)?);
        }
        b.call(test, None, &args);
    }
    b.ret(None);
    let body = b.finish().map_err(|e| e.to_string())?;
    let f = cx.m.declare(
        InstanceId(vec![u32::MAX, tag as u32]),
        format!("__stave_test_inputs.{tag}"),
        Vec::new(),
        None,
    );
    cx.m.define(f, body);
    Ok(f)
}

struct Caller<'m> {
    m: &'m mut Module,
    /// Layouts of the types before the caller was added (the parameter and element types).
    lay: Layouts,
    sa: stave_ir::IntrinsicId,
    tag: usize,
    statics: HashMap<(TypeId, Vec<u8>), usize>,
}

impl Caller<'_> {
    /// The argument for value `v` of generator `g` and type `t`.
    fn arg(&mut self, b: &mut FuncBuilder, g: &Gen, t: TypeId, v: &Value) -> Result<Operand, String> {
        Ok(match (g, v) {
            (Gen::Int { .. }, Value::Int(i)) => b.int(*i),
            (Gen::Bool, Value::Bool(x)) => b.int(i128::from(*x)),
            (Gen::Char, Value::Char(c)) => b.int(i128::from(*c)),
            (Gen::F32, Value::F32(x)) => b.konst(Const::F32(*x)),
            (Gen::F64, Value::F64(x)) => b.konst(Const::F64(*x)),
            (Gen::Array(..) | Gen::Tuple(_), _) => {
                let size = self.size(t)?;
                let mut bytes = vec![0u8; size];
                encode(&self.lay, t, v, &mut bytes, 0)?;
                let p = self.static_addr(b, t, bytes);
                Operand::Value(b.load(t, p))
            }
            (Gen::Slice(..), Value::List(vs)) => {
                let Some(TypeKey::Ref { inner, .. }) = self.m.types.key(t).cloned() else {
                    return Err("slice parameter type".into());
                };
                let Some(TypeKey::Slice { elem, .. }) = self.m.types.key(inner).cloned() else {
                    return Err("slice parameter type".into());
                };
                let es = self.size(elem)?;
                let mut bytes = vec![0u8; es * vs.len()];
                for (j, e) in vs.iter().enumerate() {
                    encode(&self.lay, elem, e, &mut bytes, j * es)?;
                }
                let at = self.m.types.array(elem, vs.len() as u64);
                let p = self.static_addr(b, at, bytes);
                let ep = self.m.types.ref_(false, elem);
                let z = b.int(0);
                let e0 = b.op(Op::ElemAddr, ep, &[p.into(), z]);
                let n = b.int(vs.len() as i128);
                Operand::Value(b.op(Op::SliceMake, t, &[e0.into(), n]))
            }
            _ => return Err(format!("a value {v:?} for {g:?}")),
        })
    }

    fn size(&self, t: TypeId) -> Result<usize, String> {
        self.lay.size_of(t).map(|s| s as usize).ok_or_else(|| "no layout".to_string())
    }

    /// The address of a static of type `t` holding `bytes` (one static per distinct pair).
    fn static_addr(&mut self, b: &mut FuncBuilder, t: TypeId, bytes: Vec<u8>) -> stave_ir::Value {
        let next = self.m.statics.len();
        let idx = *self.statics.entry((t, bytes.clone())).or_insert(next);
        if idx == next {
            let path = format!("__stave_test_inputs.{}.{}", self.tag, idx);
            self.m.statics.push(Static { path, ty: t, init: bytes });
        }
        let pt = self.m.types.ptr(true, t);
        let i = b.int(idx as i128);
        b.op(Op::Intrinsic(self.sa), pt, &[i])
    }
}

/// Writes value `v` of type `t` into `out` at `off` in the type's memory layout.
fn encode(lay: &Layouts, t: TypeId, v: &Value, out: &mut [u8], off: usize) -> Result<(), String> {
    let l = lay.get_layout(t).map_err(|e| e.to_string())?;
    match (&l.shape, v) {
        (Shape::Scalar, _) => {
            let bits: u64 = match v {
                Value::Int(i) => *i as u64,
                Value::Bool(x) => u64::from(*x),
                Value::Char(c) => u64::from(*c),
                Value::F32(x) => u64::from(*x),
                Value::F64(x) => *x,
                Value::List(_) => return Err("a list for a scalar".into()),
            };
            let n = l.size as usize;
            let dst = out.get_mut(off..off + n).ok_or("value outside its static")?;
            dst.copy_from_slice(&bits.to_le_bytes()[..n]);
        }
        (Shape::Fields(fs), Value::List(vs)) if fs.len() == vs.len() => {
            for ((fo, ft), fv) in fs.iter().zip(vs) {
                encode(lay, *ft, fv, out, off + *fo as usize)?;
            }
        }
        (Shape::Array { elem, stride, len }, Value::List(vs)) if *len as usize == vs.len() => {
            for (j, ev) in vs.iter().enumerate() {
                encode(lay, *elem, ev, out, off + j * *stride as usize)?;
            }
        }
        _ => return Err(format!("no encoding of {v:?} in {:?}", l.shape)),
    }
    Ok(())
}
