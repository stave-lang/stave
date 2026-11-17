//! Effect pass tests ([`stave_sema::effects`]): own effects and their propagation over the call
//! graph, the runtime boundary fns, deny clauses on fn types and on indirect edges, the checks a
//! program gets, on small programs checked against the shipped std and runtime. `trap` from the
//! checked IR is covered by the driver's tests (`crates/stave-driver/tests/effects.rs`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use stave_sema::callgraph::NodeKind;
use stave_sema::effects::{self, BOUNDARY, Effects, Why};
use stave_sema::{Config, Fs, Program, Typeck};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The program `src` with the shipped std and runtime, and its error diagnostics (code and
/// message).
fn load_any(src: &str, target: &str) -> (Program, Typeck, Vec<String>) {
    let mut m = BTreeMap::new();
    m.insert(PathBuf::from("/m/stave.toml"), b"name = \"m\"\nkind = \"bin\"\n".to_vec());
    m.insert(PathBuf::from("/m/src/main.stv"), src.as_bytes().to_vec());
    for e in std::fs::read_dir(repo().join("std")).unwrap() {
        let e = e.unwrap().path();
        if e.extension().is_some_and(|x| x == "stv") {
            m.insert(PathBuf::from("/std").join(e.file_name().unwrap()), std::fs::read(&e).unwrap());
        }
    }
    for sub in ["", "x86_64", "aarch64"] {
        for e in std::fs::read_dir(repo().join("runtime").join(sub)).unwrap() {
            let e = e.unwrap().path();
            if e.is_file() {
                let rel = if sub.is_empty() {
                    PathBuf::from(e.file_name().unwrap())
                } else {
                    Path::new(sub).join(e.file_name().unwrap())
                };
                m.insert(PathBuf::from("/runtime").join(rel), std::fs::read(&e).unwrap());
            }
        }
    }
    let mut c = Config::new("/m").with_shipped("/std", "/runtime");
    c.fs = Fs::Mem(m);
    c.target = stave_target::Triple::parse(target).ok();
    let p = Program::load(&c);
    let tc = stave_sema::check(&p);
    let errs: Vec<String> = tc
        .all_diagnostics(&p)
        .iter()
        .filter(|d| d.severity() == stave_diag::Severity::Error)
        .map(|d| format!("{} {}", d.code(), d.message()))
        .collect();
    (p, tc, errs)
}

fn load(src: &str) -> (Program, Typeck) {
    let (p, tc, errs) = load_any(src, "x86_64-linux");
    assert!(errs.is_empty(), "{}", errs.join("\n"));
    (p, tc)
}

/// Every fn and closure node's effect set (without the checked IR), by node name.
fn sets(p: &Program, tc: &Typeck) -> BTreeMap<String, Effects> {
    let s = effects::solve(p, tc, None);
    let g = &tc.call_graph;
    let ns: Vec<u32> = (0..g.nodes.len() as u32)
        .filter(|&n| !matches!(g.nodes[n as usize].kind, NodeKind::Table { .. }))
        .collect();
    let names = g.node_names(p, tc, &ns);
    names.into_iter().zip(ns).map(|(name, n)| (name, s.set[n as usize])).collect()
}

#[track_caller]
fn is(m: &BTreeMap<String, Effects>, name: &str, want: &[&str]) {
    let got = m.get(name).unwrap_or_else(|| panic!("no node `{name}`"));
    assert_eq!(got.names(), want, "{name}");
}

#[test]
fn effect_names() {
    for (i, n) in Effects::NAMES.iter().enumerate() {
        assert_eq!(Effects::of_name(n), Some(Effects(1 << i)));
    }
    assert_eq!(Effects::of_name("gc"), None);
    assert_eq!(Effects::ALL.names(), Effects::NAMES);
    assert_eq!((Effects::TRAP | Effects::ALLOC).names(), ["alloc", "trap"]);
}

/// Every runtime boundary fn exists on both arches: a renamed one would silently let the
/// allocator's lock and system calls through as `lock` and `io`.
#[test]
fn boundary_fns_exist() {
    for target in ["x86_64-linux", "aarch64-linux"] {
        let (p, tc, errs) = load_any("fn main() {}\n", target);
        assert!(errs.is_empty(), "{errs:?}");
        let g = &tc.call_graph;
        let names = g.node_names(&p, &tc, &(0..g.nodes.len() as u32).collect::<Vec<_>>());
        for (m, f, _, _) in BOUNDARY {
            let path = format!("{m}.{f}");
            assert!(names.contains(&path), "{target}: no node `{path}`");
        }
    }
}

/// Own effects and their propagation; the boundary fns keep what they call to themselves.
#[test]
fn sources_and_propagation() {
    let (p, tc) = load(
        "import std.vec;
import std.io;
import std.sync;
import std.thread;
extern \"C\" {
    fn puts(s: *const u8) -> i32;
}
fn push(v: &mut vec.Vec<u32>) { v.push(1u32); }
fn print() { io.print(\"x\\n\"); }
fn locked(m: &sync.Mutex<u32>) -> u32 { let g = m.lock(); return *g.get(); }
fn once(o: &sync.Once<u32>) -> u32 { return *o.get_or_init(fn() -> u32 { return 1u32; }); }
fn par() -> u32 {
    let (a, b) = thread.join(fn() -> u32 { return 1u32; }, fn() -> u32 { return 2u32; });
    return a +% b;
}
fn scoped() { scope { spawn { } } }
fn raw(p: *const u32) -> u32 { return unsafe(\"p is valid\") *p; }
fn ext(p: *const u8) -> i32 { return unsafe(\"p is a C string\") puts(p); }
rec fn walk(n: u32) -> u32 { return if n == 0u32 then 0u32 else walk(n -% 1u32); }
fn calls_walk() -> u32 { return walk(3u32); }
fn plain(x: u32) -> u32 { return x +% 1u32; }
fn main() {}
",
    );
    let m = sets(&p, &tc);
    is(&m, "main.push", &["alloc", "trap"]);
    // no checked IR here: `trap` only from call cycles and the runtime's panic fns
    is(&m, "main.print", &["io"]);
    is(&m, "main.locked", &["lock"]);
    assert!(m["main.once"].contains(Effects::LOCK), "{:?}", m["main.once"].names());
    // join starts a thread (alloc io trap) and waits for it
    is(&m, "main.par", &["alloc", "trap", "io", "block"]);
    is(&m, "main.scoped", &["alloc", "trap", "io", "block"]);
    is(&m, "main.scoped.0", &[]);
    is(&m, "main.raw", &["unsafe"]);
    assert_eq!(m["main.ext"], Effects::ALL);
    is(&m, "main.walk", &["trap", "recursive"]);
    is(&m, "main.calls_walk", &["trap", "recursive"]);
    is(&m, "main.plain", &[]);
    is(&m, "runtime.alloc.alloc", &["alloc", "trap"]);
    is(&m, "runtime.os.futex", &["lock"]);
    is(&m, "runtime.panic.trap", &["trap"]);
    // the builtin packages' own unsafe blocks do not count
    let outside: Vec<&String> =
        m.iter().filter(|(n, e)| !n.starts_with("main.") && e.meets(Effects::UNSAFE)).map(|x| x.0).collect();
    // the start code runs `main` through the `call_on_stack` intrinsic: every effect
    assert_eq!(outside, ["runtime.start.entry"]);
}

/// Effects of a spawned body count for the fn that spawns it: `std.thread.join` runs `b` in a
/// closure passed to `runtime.thread.start`, which has a spawn edge.
#[test]
fn a_started_closure_is_spawned() {
    let (p, tc) = load(
        "import std.io;
import std.thread;
fn loud() -> u32 { io.print(\"b\\n\"); return 2u32; }
fn quiet() -> u32 { return 1u32; }
fn main() {
    let (a, b) = thread.join(quiet, loud);
}
",
    );
    let d = tc.call_graph.dump(&p, &tc);
    assert!(
        d.lines()
            .any(|l| l.starts_with("edge std.thread.join<u32, u32> -> std.thread.join<u32, u32>.0 spawn")),
        "{d}"
    );
    let m = sets(&p, &tc);
    assert!(m["std.thread.join<u32, u32>"].contains(Effects::IO | Effects::BLOCK));
}

/// A fn-value call's edge carries the deny clause of the value's type (two calls into one
/// table: what both deny); those effects do not flow through it.
#[test]
fn indirect_edges_carry_the_deny_clause() {
    let (p, tc) = load(
        "import std.debug;
fn trapping(x: u32) -> u32 { debug.assert(x != 0u32, \"x\"); return x; }
fn one(f: fn(u32) -> u32 without(trap), x: u32) -> u32 { return f(x); }
fn two(f: fn(u32) -> u32 without(trap, alloc), g: fn(u32) -> u32 without(trap), x: u32) -> u32 {
    return f(g(x));
}
fn open(f: fn(u32) -> u32, x: u32) -> u32 { return f(x); }
fn main() {
    let a = open(trapping, 1u32);
}
",
    );
    let d = tc.call_graph.dump(&p, &tc);
    let edge = |from: &str| -> String {
        d.lines()
            .find(|l| l.starts_with(&format!("edge {from} -> table fn(u32) -> u32 ")))
            .unwrap_or_else(|| panic!("no edge from {from}:\n{d}"))
            .to_string()
    };
    assert!(edge("main.one").contains(" indirect without(trap) @"), "{}", edge("main.one"));
    assert!(edge("main.two").contains(" indirect without(trap) @"), "{}", edge("main.two"));
    assert!(edge("main.open").contains(" indirect @"), "{}", edge("main.open"));
    let m = sets(&p, &tc);
    is(&m, "main.open", &["trap"]);
    is(&m, "main.one", &[]);
    is(&m, "main.two", &[]);
}

/// The fn type text names the deny clause; widening it is E1403, narrowing it is a coercion.
#[test]
fn fn_types_carry_the_deny_clause() {
    let (_, _, errs) = load_any(
        "fn widen(f: fn(u32) -> u32 without(alloc)) {
    let g: fn(u32) -> u32 without(alloc, trap) = f;
}
fn narrow(f: fn(u32) -> u32 without(alloc, trap)) {
    let g: fn(u32) -> u32 without(trap) = f;
    let h: fn(u32) -> u32 = g;
}
fn main() {}
",
        "x86_64-linux",
    );
    assert_eq!(
        errs,
        [
            "E1403 `fn(u32) -> u32 without(alloc)` cannot be used as `fn(u32) -> u32 without(alloc, trap)`: it does not deny `trap`"
        ]
    );
}

/// The checks: one per instance of a generic fn with a deny clause, one per instance of the
/// body making a closure or fn item value where a deny clause is wanted.
#[test]
fn checks_of_instances_and_made_values() {
    let (p, tc) = load(
        "fn id<T>(x: T) -> T without(io) { return x; }
fn apply(f: fn(u32) -> u32 without(trap), x: u32) -> u32 { return f(x); }
fn inc(x: u32) -> u32 { return x +% 1u32; }
fn via<T>(x: T) -> u32 { return apply(fn(y: u32) -> u32 { return y; }, 1u32); }
fn main() {
    let a = id(1u32);
    let b = id(true);
    let c = apply(inc, 1u32);
    let d = via(1u8);
    let e = via(2u16);
}
",
    );
    let plan = effects::plan(&p, &tc);
    let g = &tc.call_graph;
    let named = |c: &effects::Check| g.node_names(&p, &tc, &[c.node]).remove(0);
    let mut clause: Vec<String> =
        plan.checks.iter().filter(|c| matches!(c.why, Why::Clause)).map(&named).collect();
    clause.sort();
    assert_eq!(clause, ["main.id<bool>", "main.id<u32>"]);
    let mut made: Vec<String> =
        plan.checks.iter().filter(|c| matches!(c.why, Why::Made { .. })).map(&named).collect();
    made.sort();
    assert_eq!(made, ["main.inc", "main.via<u16>.0", "main.via<u8>.0"]);
    assert!(plan.needs_ir());
    assert!(plan.checks.iter().all(|c| c.deny == Effects::TRAP || c.deny == Effects::IO));
}

/// The sets do not depend on the worker count (per-level steps collect in SCC order).
#[test]
fn sets_do_not_depend_on_the_thread_count() {
    let (p, tc) = load(
        "import std.vec;
import std.io;
fn a(v: &mut vec.Vec<u32>) { v.push(1u32); io.print(\"x\\n\"); }
fn main() {}
",
    );
    let run = |n: usize| {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap();
        pool.install(|| effects::solve(&p, &tc, None).set)
    };
    let one = run(1);
    assert_eq!(one, run(8));
    assert!(one.iter().any(|e| e.meets(Effects::IO)));
}
