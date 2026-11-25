//! The IR text round trip and build equality on real programs (contracts/checked-ir.md
//! "Canonical module"): every program of `tests/conformance/run`, `tests/conformance/simd` and
//! `bench/{kernels,simd}` that lowers, for x86-64 and AArch64, debug and release, at the levels
//! `lowered`, `checked` and `opt` (release):
//!
//! - `read(print(m)) == m`, and printing the read module gives the same bytes (the emit form at
//!   `checked`, so the stack section round-trips too);
//! - loading the `lowered` text (the removal of proven checks, step 4) gives the `checked` module;
//! - build equality: the image `stave ir build` makes from the emit form of each level is the
//!   image the build of the program makes, byte for byte.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rayon::prelude::*;
use stave_driver::build::{self, Failure};
use stave_driver::ir_cmd;
use stave_driver::pipeline::{self, Goal, Options, Output};
use stave_driver::shipped::Library;
use stave_ir::text::{Header, Level, Mode, StackInput, read};
use stave_ir::{Module, print_module};
use stave_target::Triple;

fn repo() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

/// The programs: `.stv` files and package directories.
fn programs() -> Vec<PathBuf> {
    let mut v = Vec::new();
    for dir in ["tests/conformance/run", "tests/conformance/simd", "bench/kernels", "bench/simd"] {
        for e in std::fs::read_dir(repo().join(dir)).unwrap() {
            let p = e.unwrap().path();
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            if name.starts_with(['_', '.']) {
                continue;
            }
            let pkg = p.is_dir() && p.join("stave.toml").is_file();
            if pkg || p.extension().is_some_and(|x| x == "stv") {
                v.push(p);
            }
        }
    }
    v.sort();
    v
}

fn lowered(root: &Path, target: Triple, release: bool) -> Option<Output> {
    let mut opts = Options::new(Goal::Check).threads(Some(2));
    opts.lower = true;
    opts.keep_lowered = true;
    opts.release = release;
    opts.target = Some(target);
    let out = pipeline::run(root, &Library::in_dir(repo()), &opts).ok()?;
    (!out.has_errors() && out.failure.is_none() && out.module.is_some() && out.lowered.is_some())
        .then_some(out)
}

/// The first part of `a` and `b` that differs, for the failure message.
fn first_difference(a: &Module, b: &Module) -> String {
    if a.types != b.types {
        return "types".into();
    }
    for (i, (f, g)) in a.funcs.iter().zip(&b.funcs).enumerate() {
        if f != g {
            let what = if f.body != g.body { "body" } else { "signature or attributes" };
            return format!("fn #{i} {} ({what})", f.path);
        }
    }
    let parts = [
        ("funcs", a.funcs.len() == b.funcs.len()),
        ("strings", a.strings == b.strings),
        ("intrinsics", a.intrinsics == b.intrinsics),
        ("sites", a.sites == b.sites),
        ("statics", a.statics == b.statics),
        ("main", a.main == b.main),
        ("cell", a.cell == b.cell),
        ("decls", a.decls == b.decls),
    ];
    parts.iter().find(|p| !p.1).map_or_else(|| "stated (claim rows)".into(), |p| p.0.to_string())
}

/// `read(print(m)) == m` and `print(read(print(m))) == print(m)`; returns the text.
fn round_trip(what: &str, m: &Module, h: &Header, stack: Option<&StackInput>) -> Result<String, String> {
    let text = print_module(m, h, stack).map_err(|e| format!("{what}: not printed: {e:?}"))?;
    let l = read(text.as_bytes()).map_err(|es| {
        let shown: Vec<String> = es.iter().take(5).map(|e| e.render(what)).collect();
        format!("{what}: the printed text does not load:\n{}", shown.join("\n"))
    })?;
    if l.header != *h {
        return Err(format!("{what}: header {:?} read back as {:?}", h, l.header));
    }
    if l.module != *m {
        return Err(format!("{what}: read(print(m)) != m: {}", first_difference(&l.module, m)));
    }
    if l.stack.as_ref() != stack {
        return Err(format!("{what}: the stack section does not read back"));
    }
    let again = print_module(&l.module, h, l.stack.as_ref()).map_err(|e| format!("{what}: {e:?}"))?;
    if again != text {
        return Err(format!("{what}: print(read(t)) != t"));
    }
    Ok(text)
}

/// The stack section of the `lowered` module: each fn stands for the nodes of the fn with its
/// instance in the checked module (the unchecked entries the removal makes are not in it yet).
fn lowered_stack(lowered: &Module, checked: &Module, s: &StackInput) -> StackInput {
    let nodes_of_func = lowered
        .funcs
        .iter()
        .map(|f| {
            let k = checked.funcs.binary_search_by(|g| g.instance.cmp(&f.instance)).ok();
            k.and_then(|k| s.nodes_of_func.get(k).cloned()).unwrap_or_default()
        })
        .collect();
    StackInput { nodes_of_func, ..s.clone() }
}

#[derive(Default)]
struct Tally {
    programs: usize,
    not_lowered: usize,
    trips: [usize; 3],
    builds: [usize; 3],
    not_built: usize,
    failures: Vec<String>,
}

fn one(root: &Path, target: Triple, release: bool, t: &Mutex<Tally>) {
    let name = root.strip_prefix(repo()).unwrap_or(root).display().to_string();
    let Some(out) = lowered(root, target, release) else {
        t.lock().unwrap().not_lowered += 1;
        return;
    };
    let mode = if release { Mode::Release } else { Mode::Debug };
    let h = |level| Header { target: out.program.target, mode, level };
    let checked = out.module.as_ref().unwrap();
    let low = out.lowered.as_ref().unwrap();
    let tc = out.typeck.as_ref().unwrap();
    let run_stack = out.program.packages.iter().find(|p| p.is_root).and_then(|p| p.run_stack);
    let stack = stave_driver::stack::input(&tc.call_graph, &out.nodes_of_func, run_stack);
    let low_stack = lowered_stack(low, checked, &stack);
    let mut fails = Vec::new();
    let mut texts: Vec<(usize, String)> = Vec::new();
    let mut trips = [0usize; 3];
    let mut check = |k: usize, r: Result<String, String>| match r {
        Ok(text) => {
            trips[k] += 1;
            texts.push((k, text));
        }
        Err(e) => fails.push(format!("{name} {target} {mode:?}: {e}")),
    };
    check(0, round_trip("lowered", low, &h(Level::Lowered), Some(&low_stack)));
    check(1, round_trip("checked", checked, &h(Level::Checked), Some(&stack)));
    if release {
        match build::release_passes(checked) {
            Ok(o) => check(2, round_trip("opt", &o, &h(Level::Opt), Some(&stack))),
            Err(e) => fails.push(format!("{name} {target} release: release passes: {e}")),
        }
    }
    // loading the `lowered` text runs the removal of proven checks: the `checked` module
    if let Some((_, text)) = texts.iter().find(|x| x.0 == 0) {
        match ir_cmd::load("lowered", text.as_bytes()) {
            Ok(l) if l.module == *checked => {}
            Ok(l) => fails.push(format!(
                "{name} {target} {mode:?}: loading `lowered` does not give `checked`: {}",
                first_difference(&l.module, checked)
            )),
            Err(e) => fails.push(format!("{name} {target} {mode:?}: loading `lowered`: {:?}", e.lines())),
        }
    }
    // build equality: the build's image against `stave ir build` of each level's emit form
    let mut builds = [0usize; 3];
    let mut not_built = 0;
    if checked.main.is_some() {
        match build::codegen_link(checked, target, release, Some(2), Some(&stack)) {
            Ok(direct) => {
                for (k, text) in &texts {
                    let level = ["lowered", "checked", "opt"][*k];
                    let image = ir_cmd::load(level, text.as_bytes()).and_then(|l| ir_cmd::build(&l, Some(2)));
                    match image {
                        Ok(b) if b.image.bytes() == direct.image.bytes() => builds[*k] += 1,
                        Ok(_) => fails.push(format!("{name} {target} {mode:?} {level}: the image differs")),
                        Err(e) => fails.push(format!("{name} {target} {mode:?} {level}: {:?}", e.lines())),
                    }
                }
            }
            Err(Failure::Unsupported { .. }) => not_built += 1,
            Err(e) => fails.push(format!("{name} {target} {mode:?}: the build fails: {e}")),
        }
    } else {
        not_built += 1;
    }
    let mut t = t.lock().unwrap();
    t.programs += 1;
    for k in 0..3 {
        t.trips[k] += trips[k];
        t.builds[k] += builds[k];
    }
    t.not_built += not_built;
    t.failures.extend(fails);
}

fn corpus(target: &str, release: bool) {
    let target = Triple::parse(target).unwrap();
    let t = Mutex::new(Tally::default());
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    pool.install(|| programs().par_iter().for_each(|p| one(p, target, release, &t)));
    let t = t.into_inner().unwrap();
    println!(
        "{target} release={release}: {} programs lowered ({} not), round trips lowered/checked/opt {:?}, \
         build equality {:?}, {} not built",
        t.programs, t.not_lowered, t.trips, t.builds, t.not_built
    );
    assert!(t.failures.is_empty(), "{} failures:\n{}", t.failures.len(), t.failures.join("\n"));
    assert!(t.programs > 100 && t.trips[1] == t.programs, "too few programs: {}", t.programs);
}

#[test]
fn x86_64_debug() {
    corpus("x86_64-linux", false);
}

#[test]
fn x86_64_release() {
    corpus("x86_64-linux", true);
}

#[test]
fn aarch64_debug() {
    corpus("aarch64-linux", false);
}

#[test]
fn aarch64_release() {
    corpus("aarch64-linux", true);
}
