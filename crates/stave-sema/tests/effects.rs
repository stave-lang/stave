//! Effect pass tests ([`stave_sema::effects`]): own effects and their propagation over the call
//! graph, the runtime boundary fns, deny clauses on fn types and on indirect edges, the checks a
//! program gets, on small programs checked against the shipped std and runtime. `trap` from the
//! checked IR is covered by the driver's tests (`crates/stave-driver/tests/effects.rs`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use stave_sema::callgraph::{EdgeKind, NodeKind};
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

/// Every fn and closure's effect set (without the checked IR) by source name, as `stave effects`
/// reports it: the union over the call-graph nodes the program uses.
fn sets(p: &Program, tc: &Typeck) -> BTreeMap<String, Effects> {
    let s = effects::solve(p, tc, None);
    let g = &tc.call_graph;
    let groups = effects::by_source(g, &s);
    let firsts: Vec<u32> = groups.iter().map(|x| x.0[0]).collect();
    let names = g.source_names(p, tc, &firsts);
    names.into_iter().zip(groups).map(|(name, (_, e))| (name, e)).collect()
}

/// `name` without the fn values a call-graph node is specialized on (`a{0=b}.0` -> `a.0`).
fn unspec(name: &str) -> String {
    let (mut out, mut depth) = (String::new(), 0);
    for c in name.chars() {
        match c {
            '{' => depth += 1,
            '}' => depth -= 1,
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out
}

/// The dump line of the edge `from -> to` (source names; any specialization), if there is one.
fn edge_line(d: &str, from: &str, to: &str) -> Option<String> {
    d.lines()
        .filter_map(|l| l.strip_prefix("edge "))
        .find(|l| {
            let Some((f, rest)) = l.split_once(" -> ") else { return false };
            unspec(f) == from && unspec(rest).starts_with(&format!("{to} "))
        })
        .map(str::to_string)
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
    // no source has `park`
    assert_eq!(Effects::ALL.names(), Effects::NAMES[..7]);
    assert!(!Effects::ALL.meets(Effects::PARK));
    assert_eq!(Effects::fn_type_deny((Effects::BLOCK | Effects::PARK).0), Effects::BLOCK.0);
    assert_eq!((Effects::TRAP | Effects::ALLOC).names(), ["alloc", "trap"]);
    // denying `block` denies its part `park`; a deny clause is shown without the implied part
    assert_eq!(Effects::deny_of_name("block"), Some(Effects::BLOCK | Effects::PARK));
    assert_eq!(Effects::deny_of_name("park"), Some(Effects::PARK));
    assert_eq!(Effects::deny_of_name("io"), Some(Effects::IO));
    assert_eq!((Effects::BLOCK | Effects::PARK | Effects::IO).minimal().names(), ["io", "block"]);
    assert_eq!(Effects::PARK.minimal(), Effects::PARK);
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
        for (k, _, _) in BOUNDARY {
            let path = k.path();
            assert!(p.known.decl(*k).is_some(), "{target}: `{path}` not resolved");
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
unsafe(\"puts as declared in stdio.h\") extern \"C\" {
    fn puts(s: *const u8) -> i32;
}
fn push(v: &mut vec.Vec<u32>) { v.push(1u32); }
fn print() { io.print(\"x\\n\"); }
fn locked(m: &sync.Mutex<u32, 1>) -> u32 { let g = m.lock(); return *g.get(); }
fn once(o: &sync.Once<u32, 1>) -> u32 { return *o.get_or_init(fn() -> u32 { return 1u32; }); }
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
    // a write may wait on a pipe: `block` besides `io`
    is(&m, "main.print", &["io", "block"]);
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
    let e = edge_line(&d, "std.thread.join<u32, u32>", "std.thread.join<u32, u32>.0").unwrap_or_default();
    assert!(e.contains(" spawn @"), "{d}");
    let m = sets(&p, &tc);
    assert!(m["std.thread.join<u32, u32>"].contains(Effects::IO | Effects::BLOCK));
}

/// A fn-value call's edges carry the deny clause of the value's type, both the flow-refined
/// edges to the values that reach the call and the edge to the whole table where an untracked
/// value does (a fn value returned by a call); the denied effects do not flow through them.
#[test]
fn indirect_edges_carry_the_deny_clause() {
    let (p, tc) = load(
        "import std.debug;
fn trapping(x: u32) -> u32 { debug.assert(x != 0u32, \"x\"); return x; }
fn safe(x: u32) -> u32 without(trap) { return x +% 1u32; }
fn safe_both(x: u32) -> u32 without(trap, alloc) { return x +% 2u32; }
fn one(f: fn(u32) -> u32 without(trap), x: u32) -> u32 { return f(x); }
fn two(f: fn(u32) -> u32 without(trap, alloc), g: fn(u32) -> u32 without(trap), x: u32) -> u32 {
    return f(g(x));
}
fn pick() -> fn(u32) -> u32 without(trap) { return safe; }
fn returned(x: u32) -> u32 {
    let f = pick();
    return f(x);
}
fn open(f: fn(u32) -> u32, x: u32) -> u32 { return f(x); }
fn main() {
    let a = open(trapping, 1u32);
    let b = one(safe, 1u32);
    let c = two(safe_both, safe, 1u32);
    let d = returned(1u32);
}
",
    );
    let d = tc.call_graph.dump(&p, &tc);
    let edge = |from: &str, to: &str| -> String {
        edge_line(&d, from, to).unwrap_or_else(|| panic!("no edge {from} -> {to}:\n{d}"))
    };
    let e = edge("main.one", "main.safe");
    assert!(e.contains(" indirect without(trap) @"), "{e}");
    let e = edge("main.two", "main.safe");
    assert!(e.contains(" indirect without(trap) @"), "{e}");
    let e = edge("main.two", "main.safe_both");
    assert!(e.contains(" indirect without(alloc, trap) @"), "{e}");
    let e = edge("main.returned", "table fn(u32) -> u32");
    assert!(e.contains(" indirect without(trap) @"), "{e}");
    let e = edge("main.open", "main.trapping");
    assert!(e.contains(" indirect @"), "{e}");
    let m = sets(&p, &tc);
    is(&m, "main.open", &["trap"]);
    is(&m, "main.one", &[]);
    is(&m, "main.two", &[]);
    // the table holds `trapping`; the edge's deny clause keeps `trap` out
    is(&m, "main.returned", &[]);
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

/// Implicit drops count: a fn that only lets a `Vec` go out of scope frees its buffer (`alloc`),
/// through a drop edge to the glue and `Vec.drop`; the glue itself has no effect of its own, and
/// the chain of `without(alloc)` on it starts with that drop edge. A `MutexGuard` going out of
/// scope unlocks, which is no effect.
#[test]
fn implicit_drops_count() {
    let (p, tc) = load(
        "import std.vec;
import std.sync;
fn sink(v: vec.Vec<u32>) {}
fn sink_denied(v: vec.Vec<u32>) without(alloc) {}
fn guard(m: &sync.Mutex<u32, 1>) -> u32 { let g = m.lock(); return *g.get(); }
fn main() {}
",
    );
    let m = sets(&p, &tc);
    is(&m, "main.sink", &["alloc"]);
    is(&m, "main.guard", &["lock"]);
    let s = effects::solve(&p, &tc, None);
    let g = &tc.call_graph;
    let names = g.source_names(&p, &tc, &(0..g.nodes.len() as u32).collect::<Vec<_>>());
    let at = |name: &str| names.iter().position(|n| n == name).unwrap() as u32;
    let glue = g.nodes.iter().position(|n| matches!(n.kind, NodeKind::Glue { .. })).unwrap();
    assert_eq!(s.local[glue], Effects::NONE);
    let (edges, end) = s.chain(g, at("main.sink_denied"), Effects::ALLOC).unwrap();
    assert_eq!(g.out[edges[0] as usize].kind, EdgeKind::Drop);
    assert_eq!(names[end as usize], "runtime.alloc.free");
    let plan = effects::plan(&p, &tc);
    let d = effects::check(&p, &tc, &plan, &s);
    assert_eq!(d.len(), 1);
    assert_eq!(d[0].code(), "E1401");
    assert!(d[0].message().contains("`runtime.alloc.free`, which frees memory"), "{}", d[0].message());
}

/// The nodes of fn `name` (every specialization and the plain node) and which are used.
fn nodes_of(p: &Program, tc: &Typeck, s: &effects::EffectSets, name: &str) -> Vec<(String, bool)> {
    let g = &tc.call_graph;
    let all: Vec<u32> = (0..g.nodes.len() as u32).collect();
    let src = g.source_names(p, tc, &all);
    let full = g.node_names(p, tc, &all);
    all.iter()
        .filter(|&&n| src[n as usize] == name && matches!(g.nodes[n as usize].kind, NodeKind::Fn { .. }))
        .map(|&n| (full[n as usize].clone(), s.used[n as usize]))
        .collect()
}

/// E1401 / E1403 diagnostics of the program (code, message).
fn effect_errors(p: &Program, tc: &Typeck) -> Vec<(String, String)> {
    let s = effects::solve(p, tc, None);
    let plan = effects::plan(p, tc);
    effects::check(p, tc, &plan, &s).iter().map(|d| (d.code().to_string(), d.message().to_string())).collect()
}

/// A helper called with two closures, one trapping and one not, has a node for each: its line
/// is the union, and `without(trap)` on it is E1401, once, in source names.
#[test]
fn a_helper_called_with_two_closures_has_their_union() {
    let src = |deny: &str| {
        format!(
            "import std.debug;
fn apply(f: fn(u32) -> u32, x: u32) -> u32 {deny} {{ return f(x); }}
fn main() {{
    let a = apply(fn(x: u32) -> u32 {{ debug.assert(x != 0u32, \"x\"); return x; }}, 1u32);
    let b = apply(fn(x: u32) -> u32 {{ return x +% 1u32; }}, 2u32);
}}
"
        )
    };
    let (p, tc) = load(&src(""));
    let s = effects::solve(&p, &tc, None);
    let nodes = nodes_of(&p, &tc, &s, "main.apply");
    assert_eq!(nodes.iter().filter(|x| x.1).count(), 2, "{nodes:?}");
    is(&sets(&p, &tc), "main.apply", &["trap"]);
    let (p, tc) = load(&src("without(trap)"));
    let errs = effect_errors(&p, &tc);
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert_eq!(errs[0].0, "E1401");
    assert_eq!(
        errs[0].1,
        "`main.apply` denies `trap`, but it reaches `runtime.panic.trap_msg`, which may trap"
    );
}

/// A helper only ever called with one closure: its plain node runs nowhere and counts for
/// nothing, while a helper no call reaches keeps its plain node.
#[test]
fn an_unused_plain_node_does_not_count() {
    let (p, tc) = load(
        "fn apply(f: fn(u32) -> u32, x: u32) -> u32 without(trap) { return f(x); }
fn lonely(f: fn(u32) -> u32, x: u32) -> u32 { return f(x); }
fn main() {
    let a = apply(fn(x: u32) -> u32 { return x +% 1u32; }, 1u32);
}
",
    );
    let s = effects::solve(&p, &tc, None);
    let apply = nodes_of(&p, &tc, &s, "main.apply");
    assert_eq!(apply.len(), 2, "{apply:?}");
    assert!(apply.iter().any(|(n, used)| n == "main.apply" && !used), "{apply:?}");
    assert!(apply.iter().any(|(n, used)| n != "main.apply" && *used), "{apply:?}");
    assert_eq!(nodes_of(&p, &tc, &s, "main.lonely"), [("main.lonely".to_string(), true)]);
    is(&sets(&p, &tc), "main.apply", &[]);
    assert!(effect_errors(&p, &tc).is_empty());
}

/// Past 16 known values the calls share the plain node, which then holds the rest: the effects
/// stay sound (the trapping value is the 17th).
#[test]
fn past_the_specialization_limit_the_plain_node_holds_the_rest() {
    let mut src =
        String::from("import std.debug;\nfn apply(f: fn(u32) -> u32, x: u32) -> u32 DENY { return f(x); }\n");
    for i in 0..16 {
        src.push_str(&format!("fn v{i}(x: u32) -> u32 {{ return x +% {i}u32; }}\n"));
    }
    src.push_str("fn v16(x: u32) -> u32 { debug.assert(x != 0u32, \"x\"); return x; }\nfn main() {\n");
    for i in 0..17 {
        src.push_str(&format!("    let a{i} = apply(v{i}, 1u32);\n"));
    }
    src.push_str("}\n");
    let (p, tc) = load(&src.replace("DENY", ""));
    let s = effects::solve(&p, &tc, None);
    let apply = nodes_of(&p, &tc, &s, "main.apply");
    assert_eq!(apply.len(), 17, "{apply:?}");
    assert!(apply.iter().all(|x| x.1), "{apply:?}");
    assert!(apply.iter().any(|(n, _)| n == "main.apply"));
    is(&sets(&p, &tc), "main.apply", &["trap"]);
    let (p, tc) = load(&src.replace("DENY", "without(trap)"));
    let errs = effect_errors(&p, &tc);
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert_eq!(
        errs[0].1,
        "`main.apply` denies `trap`, but it reaches `runtime.panic.trap_msg`, which may trap"
    );
}

/// The recursion check and the effect pass name a cycle through a specialized helper the same
/// way: source names, no specialization.
#[test]
fn recursion_and_effect_diagnostics_use_source_names() {
    let (p, tc, errs) = load_any(
        "fn apply(f: fn(u32) -> u32, x: u32) -> u32 { return f(x); }
fn spin(n: u32) -> u32 { return apply(spin, n -% 1u32); }
fn g(n: u32) -> u32 without(recursive) { return spin(n); }
fn main() {
    let a = g(3u32);
}
",
        "x86_64-linux",
    );
    let rec: Vec<&String> = errs.iter().filter(|e| e.starts_with("E1402")).collect();
    assert_eq!(rec.len(), 1, "{errs:?}");
    let eff = effect_errors(&p, &tc);
    assert_eq!(eff.len(), 1, "{eff:?}");
    assert_eq!(eff[0].0, "E1401");
    for m in rec.iter().map(|s| s.as_str()).chain([eff[0].1.as_str()]) {
        assert!(!m.contains('{'), "{m}");
        assert!(m.contains("main.spin") || m.contains("main.apply"), "{m}");
    }
    assert!(eff[0].1.contains("`main.g` denies `recursive`, but it reaches `main."), "{eff:?}");
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
