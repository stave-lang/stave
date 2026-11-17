//! Whole-program call graph tests: edges (direct, generic instances, methods, trait-bound calls,
//! implied calls, dispatch arms, fn values), fn-value target tables, roots, SCCs, and levels, on
//! small programs checked against the shipped std and runtime.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use stave_sema::callgraph::{CallGraph, EdgeKind, NodeKind, RootKind};
use stave_sema::{Config, Fs, Program, Typeck};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn load(src: &str, target: &str) -> (Program, Typeck) {
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
        .filter(|d| d.severity() == stave_diag::Severity::Error && !["E1402", "E1409"].contains(&d.code()))
        .map(|d| format!("{} {}", d.code(), d.message()))
        .collect();
    assert!(errs.is_empty(), "{}", errs.join("\n"));
    (p, tc)
}

/// The dump lines of the program's own package (`main.` paths), plus cycles and roots.
fn dump(src: &str) -> (Program, Typeck, String) {
    let (p, tc) = load(src, "x86_64-linux");
    let d = plain(&tc.call_graph.dump(&p, &tc));
    (p, tc, d)
}

/// The dump with the specializations (`{0=main.f}` after a fn name) left out, for tests that are
/// not about them.
fn plain(d: &str) -> String {
    let mut out = String::with_capacity(d.len());
    let mut depth = 0u32;
    let mut prev = ' ';
    for c in d.chars() {
        if c == '{' && (depth > 0 || (prev != ' ' && prev != '\n')) {
            depth += 1;
        } else if c == '}' && depth > 0 {
            depth -= 1;
        } else if depth == 0 {
            out.push(c);
        }
        prev = c;
    }
    out
}

#[track_caller]
fn has(d: &str, line: &str) {
    assert!(d.lines().any(|l| l.starts_with(line)), "no line starting `{line}` in:\n{}", own(d));
}

#[track_caller]
fn lacks(d: &str, line: &str) {
    assert!(!d.lines().any(|l| l.starts_with(line)), "unexpected line `{line}` in:\n{}", own(d));
}

/// The lines that mention the root package (for failure messages).
fn own(d: &str) -> String {
    d.lines().filter(|l| l.contains("main.") || l.starts_with("levels")).collect::<Vec<_>>().join("\n")
}

/// The node named `name` in the dump.
fn node(g: &CallGraph, d: &str, name: &str) -> u32 {
    let l = d
        .lines()
        .find(|l| {
            l.strip_prefix("node ")
                .and_then(|r| r.split_once(' '))
                .is_some_and(|(_, n)| n == name || n.starts_with(&format!("{name} ")))
        })
        .unwrap_or_else(|| panic!("no node `{name}` in:\n{}", own(d)));
    let id: u32 = l["node ".len()..].split(' ').next().unwrap().parse().unwrap();
    assert!((id as usize) < g.nodes.len());
    id
}

/// Every edge between two SCCs goes to a lower level; SCC ids ascend with the level; the CSR
/// arrays agree with each other.
fn check_invariants(g: &CallGraph) {
    let n = g.nodes.len();
    assert_eq!(g.out_start.len(), n + 1);
    assert_eq!(g.in_start.len(), n + 1);
    assert_eq!(g.in_edges.len(), g.out.len());
    assert_eq!(g.scc_of.len(), n);
    for v in 0..n as u32 {
        let mut last = None;
        for e in g.succ(v) {
            assert!(last.is_none_or(|l| l < e.to), "out-edges sorted and unique");
            last = Some(e.to);
            let (a, b) = (g.scc_of[v as usize], g.scc_of[e.to as usize]);
            if a != b {
                assert!(g.scc_level[a as usize] > g.scc_level[b as usize], "edge {v} -> {} goes up", e.to);
            }
            assert!(g.pred(e.to).any(|(c, x)| c == v && x == e), "in-edge of {v} -> {}", e.to);
        }
    }
    for s in 1..g.scc_count() {
        assert!(g.scc_level[s - 1] <= g.scc_level[s]);
    }
    for l in 0..g.level_count() {
        for s in g.level(l) {
            assert_eq!(g.scc_level[s as usize] as usize, l);
        }
    }
    for s in 0..g.scc_count() as u32 {
        for &v in g.scc(s) {
            assert_eq!(g.scc_of[v as usize], s);
        }
    }
    assert!(!g.truncated);
}

#[test]
fn direct_generic_method_and_bound_calls() {
    let (_p, tc, d) = dump(
        "trait Shape {
    fn area(&self) -> u32;
}
struct Sq { s: u32 }
impl Shape for Sq {
    fn area(&self) -> u32 { return self.s * self.s; }
}
impl Sq {
    fn new(s: u32) -> Sq { return Sq.{ s: s }; }
    fn grow(&mut self) { self.s = self.s + 1u32; }
}
struct V { x: u32 }
impl Add for V {
    fn add(self, o: V) -> V { return V.{ x: self.x + o.x }; }
}
fn total<T: Shape>(x: &T) -> u32 { return x.area() + leaf(); }
fn id<T>(x: T) -> T { return x; }
fn leaf() -> u32 { return 1u32; }
fn unused<T>(x: T) -> T { return x; }
fn main() {
    let mut q = Sq.new(2u32);
    q.grow();
    let a = total(&q);
    let b = id(a);
    let c = id(true);
    let v = V.{ x: 1u32 } + V.{ x: 2u32 };
}
",
    );
    let g = &tc.call_graph;
    has(&d, "edge main.main -> main.Sq.new call @21:20");
    has(&d, "edge main.main -> main.Sq.grow call @22:7");
    has(&d, "edge main.main -> main.total<Sq> call @23:13");
    has(&d, "edge main.total<Sq> -> main.Sq.area call @16:45");
    has(&d, "edge main.total<Sq> -> main.leaf call");
    has(&d, "edge main.main -> main.id<u32> call @24:13");
    has(&d, "edge main.main -> main.id<bool> call @25:13");
    has(&d, "edge main.main -> main.V.add call @26:27");
    // generic fns exist only as instances; an uncalled generic has none
    lacks(&d, "node 0 main.total ");
    assert!(!d.contains("main.unused"), "{}", own(&d));
    has(&d, "root main.main main");
    // leaf < total<Sq> < main by level
    let lv = |n: &str| g.scc_level[g.scc_of[node(g, &d, n) as usize] as usize];
    assert!(lv("main.leaf") < lv("main.total<Sq>"));
    assert!(lv("main.total<Sq>") < lv("main.main"));
    assert!(!d.lines().any(|l| l.starts_with("cycle") && l.contains("main.")), "{}", own(&d));
    check_invariants(g);
}

#[test]
fn fn_values_tables_and_a_cycle_through_a_param() {
    let (_p, tc, d) = dump(
        "fn twice(f: fn(u32) -> u32, x: u32) -> u32 { return f(f(x)); }
fn inc(x: u32) -> u32 { return x + 1u32; }
fn dec(x: u32) -> u32 { return x - 1u32; }
fn walk(f: fn(u32) -> u32, n: u32) -> u32 {
    if n == 0u32 { return 0u32; }
    return f(n - 1u32);
}
fn step(n: u32) -> u32 { return walk(step, n); }
fn other(x: u32, y: u32) -> u32 { return x; }
fn main() {
    let a = twice(inc, 3u32);
    let k: u32 = 5;
    let b = twice(fn(x: u32) -> u32 { return x + k; }, a);
    let c = step(a);
    let d = dec(b);
    let o = other;
    let e = o(c, d);
}
",
    );
    let g = &tc.call_graph;
    let t = "table fn(u32) -> u32";
    // each call through a param reaches what its callers pass there
    has(&d, "edge main.twice -> main.inc indirect @1:53");
    has(&d, "edge main.twice -> main.main.0 indirect @1:53");
    lacks(&d, "edge main.twice -> main.step");
    has(&d, "edge main.walk -> main.step indirect @6:12");
    lacks(&d, "edge main.walk -> main.inc");
    lacks(&d, &format!("edge main.twice -> {t}"));
    // the table keeps every value of its type (the upper bound)
    has(&d, &format!("edge {t} -> main.inc member @11:19"));
    has(&d, &format!("edge {t} -> main.step member @8:38"));
    has(&d, &format!("edge {t} -> main.main.0 member @13:19"));
    // called directly only: not in any table
    lacks(&d, &format!("edge {t} -> main.dec"));
    // another fn type, another table
    has(&d, "edge table fn(u32, u32) -> u32 -> main.other member @16:13");
    // through a local
    has(&d, "edge main.main -> main.other indirect @17:13");
    lacks(&d, "edge table fn(u32, u32) -> u32 -> main.inc");
    // step -> walk -> step through walk's param
    has(&d, "cycle level 0 {main.step, main.walk}");
    assert!(!d.lines().any(|l| l.starts_with("cycle") && l.contains("main.twice")), "{}", own(&d));
    let tn = node(g, &d, t);
    let ti = match g.nodes[tn as usize].kind {
        NodeKind::Table { table } => table as usize,
        _ => panic!("not a table"),
    };
    assert_eq!(g.tables[ti].node, tn);
    assert!(g.succ(tn).iter().all(|e| e.kind == EdgeKind::Member));
    check_invariants(g);
}

#[test]
fn closures_of_generic_fns_are_per_instance() {
    let (_p, tc, d) = dump(
        "fn via<T>(x: T, f: fn(T) -> T) -> T {
    let g = fn(y: T) -> T { return f(y); };
    return g(x);
}
fn inc(x: u32) -> u32 { return x + 1u32; }
fn not(x: bool) -> bool { return !x; }
fn main() {
    let a = via(1u32, inc);
    let b = via(true, not);
}
",
    );
    has(&d, "edge main.via<u32> -> main.via<u32>.0 indirect @3:12");
    has(&d, "edge main.via<u32>.0 -> main.inc indirect @2:36");
    has(&d, "edge main.via<bool>.0 -> main.not indirect @2:36");
    lacks(&d, "edge main.via<u32>.0 -> main.not");
    has(&d, "edge table fn(u32) -> u32 -> main.via<u32>.0 member @2:13");
    has(&d, "edge table fn(u32) -> u32 -> main.inc member @8:23");
    has(&d, "edge table fn(bool) -> bool -> main.via<bool>.0 member @2:13");
    lacks(&d, "edge table fn(bool) -> bool -> main.inc");
    // g calls f, which holds only what main passes
    assert!(!d.lines().any(|l| l.starts_with("cycle") && l.contains("main.")), "{}", own(&d));
    check_invariants(&tc.call_graph);
}

#[test]
fn mutual_recursion_with_and_without_rec() {
    let (_p, tc, d) = dump(
        "rec fn even(n: u32) -> bool {
    if n == 0u32 { return true; }
    return odd(n - 1u32);
}
rec fn odd(n: u32) -> bool {
    if n == 0u32 { return false; }
    return even(n - 1u32);
}
fn ping(n: u32) -> u32 {
    if n == 0u32 { return 0u32; }
    return pong(n - 1u32);
}
fn pong(n: u32) -> u32 { return ping(n); }
struct T { v: u32 }
impl T {
    rec fn down(&self, n: u32) -> u32 {
        if n == 0u32 { return self.v; }
        return self.down(n - 1u32);
    }
}
fn main() {
    let a = even(4u32);
    let b = ping(3u32);
    let t = T.{ v: 1u32 };
    let c = t.down(2u32);
}
",
    );
    let g = &tc.call_graph;
    for n in ["main.even", "main.odd", "main.T.down"] {
        assert!(g.nodes[node(g, &d, n) as usize].rec(), "{n}");
    }
    for n in ["main.ping", "main.pong", "main.main"] {
        assert!(!g.nodes[node(g, &d, n) as usize].rec(), "{n}");
    }
    has(&d, "node 0 ");
    assert!(d.contains("main.even rec"), "{}", own(&d));
    has(&d, "cycle level 0 {main.even, main.odd}");
    has(&d, "cycle level 0 {main.ping, main.pong}");
    has(&d, "cycle level 0 {main.T.down}");
    has(&d, "edge main.T.down -> main.T.down call @18:21");
    let s = g.scc_of[node(g, &d, "main.even") as usize];
    assert!(g.is_cycle(s));
    assert!(!g.is_cycle(g.scc_of[node(g, &d, "main.main") as usize]));
    check_invariants(g);
}

#[test]
fn dispatch_calls_stand_for_their_arms() {
    let src = "fn sum_v3(xs: &[f32]) -> f32 needs(avx2) { return 0.0f32; }
fn sum_sve(xs: &[f32]) -> f32 needs(sve) { return 2.0f32; }
fn sum_base(xs: &[f32]) -> f32 { return 1.0f32; }
dispatch fn sum(xs: &[f32]) -> f32 { avx2 => sum_v3, sve => sum_sve, _ => sum_base, }
fn apply(f: fn(&[f32]) -> f32, xs: &[f32]) -> f32 { return f(xs); }
fn main() {
    let xs: [f32; 2] = [1.0f32, 2.0f32];
    let s = sum(&xs);
    let t = apply(sum, &xs);
}
";
    let (p, tc, d) = dump(src);
    let g = &tc.call_graph;
    has(&d, "edge main.main -> main.sum_v3 dispatch main.sum @8:13");
    has(&d, "edge main.main -> main.sum_base dispatch main.sum @8:13");
    // a dispatch used as a fn value stands for its arms too
    has(&d, "edge table fn(&[f32]) -> f32 -> main.sum_v3 member @9:19");
    has(&d, "edge table fn(&[f32]) -> f32 -> main.sum_base member @9:19");
    assert!(!d.lines().any(|l| l.starts_with("node") && l.ends_with(" main.sum")), "{}", own(&d));
    // another arch's arm is not built here
    assert!(!d.contains("main.sum_sve"), "{}", own(&d));
    check_invariants(g);
    let (p2, tc2) = load(src, "aarch64-linux");
    let d2 = tc2.call_graph.dump(&p2, &tc2);
    has(&d2, "edge main.main -> main.sum_sve dispatch main.sum");
    has(&d2, "edge main.main -> main.sum_base dispatch main.sum");
    assert!(!d2.contains("main.sum_v3"), "{}", own(&d2));
    let _ = p;
}

#[test]
fn spawn_join_and_par_for_bodies_are_roots() {
    let (_p, tc, d) = dump(
        "import std.thread;
fn work(i: usize) { }
fn one() -> u64 { return 1u64; }
fn main() {
    let mut a: u64 = 0;
    scope {
        spawn { a = 1u64; }
    }
    let (x, y) = thread.join(one, fn() -> u64 { return 2u64; });
    thread.par_for(0usize, 4usize, work);
}
",
    );
    let g = &tc.call_graph;
    has(&d, "root main.main main @4:4");
    has(&d, "root main.main.0 spawn @7:9");
    has(&d, "root main.one std.thread.join arg 0 @9:30");
    has(&d, "root main.main.1 std.thread.join arg 1 @9:35");
    has(&d, "root main.work std.thread.par_for arg 2 @10:36");
    has(&d, "node 0 ");
    has(&d, "edge main.main -> main.main.0 spawn @7:9");
    assert!(d.contains("main.main.0 spawn\n"), "{}", own(&d));
    // a spawn body runs as a `fn()` value: runtime.thread.entry calls it through the table
    has(&d, "edge table fn() -> main.main.0 member");
    has(&d, "edge runtime.thread.entry -> table fn() indirect");
    let kinds: Vec<RootKind> = g.roots.iter().map(|r| r.kind).collect();
    assert_eq!(kinds.iter().filter(|k| **k == RootKind::Main).count(), 1);
    assert_eq!(kinds.iter().filter(|k| **k == RootKind::Spawn).count(), 1, "{}", own(&d));
    check_invariants(g);
}

#[test]
fn implied_calls_for_next_fstring_display_and_as_str() {
    let (_p, tc, d) = dump(
        "import std.io;
import std.string;
struct Count { n: u32 }
impl Count {
    fn next(&mut self) -> Option<u32> {
        if self.n == 0u32 { return Option.None; }
        self.n = self.n - 1u32;
        return Option.Some(self.n);
    }
}
struct P { x: u32 }
impl Display for P {
    fn fmt<W: Write>(&self, w: &mut W) { w.write_str(\"p\"); }
}
fn show(s: &str) { }
fn main() {
    let c = Count.{ n: 3u32 };
    for v in c { }
    let p = P.{ x: 1u32 };
    io.print(f\"{p}\\n\");
    let s = string.String.new();
    show(&s);
}
",
    );
    let g = &tc.call_graph;
    has(&d, "edge main.main -> main.Count.next call @18:5");
    has(&d, "edge main.main -> main.P.fmt<String> call @20:14");
    has(&d, "edge main.P.fmt<String> -> std.string.String.write_str call @13:44");
    has(&d, "edge main.main -> std.string.String.new call @20:14");
    has(&d, "edge main.main -> std.string.String.push_str call @20:14");
    has(&d, "edge main.main -> std.string.String.as_str call @20:14");
    check_invariants(g);
}

#[test]
fn thread_bodies_reach_only_what_flows_to_them() {
    // join's own `fn()` closure calls `b()`, which holds only the closure main passes
    let (_p, tc, d) = dump(
        "import std.thread;
fn tick() { }
fn main() {
    let (x, y) = thread.join(fn() { }, fn() { });
    thread.par_for(0usize, 4usize, fn(i: usize) {
        let g = tick;
        g();
    });
}
",
    );
    let cycles: Vec<&str> = d.lines().filter(|l| l.starts_with("cycle") && !l.ends_with(" drop")).collect();
    assert!(cycles.is_empty(), "{}", cycles.join("\n"));
    has(&d, "edge std.thread.join<(), ()>.0 -> main.main.1 indirect");
    has(&d, "edge std.thread.join<(), ()> -> main.main.0 indirect");
    has(&d, "edge std.thread.run.0 -> main.main.2 indirect");
    has(&d, "edge main.main.2 -> main.tick indirect @7:9");
    check_invariants(&tc.call_graph);
}

#[test]
fn a_closure_that_reaches_itself_through_a_param_is_a_cycle() {
    let (_p, tc, d) = dump(
        "import std.thread;
fn call(f: fn(u32) -> u32, n: u32) -> u32 { return f(n); }
fn again(n: u32) -> u32 { return call(fn(x: u32) -> u32 { return again(x); }, n); }
fn twice() {
    let (x, y) = thread.join(fn() { twice(); }, fn() { });
}
fn main() {
    let a = again(1u32);
    twice();
}
",
    );
    has(&d, "cycle level 0 {main.again, main.again.0, main.call}");
    let cyc = d.lines().find(|l| l.starts_with("cycle") && l.contains("main.twice")).expect("join cycle");
    assert!(cyc.contains("main.twice.0") && cyc.contains("std.thread.join<(), ()>"), "{cyc}");
    assert!(!cyc.contains("main.twice.1") && !cyc.ends_with(" drop"), "{cyc}");
    check_invariants(&tc.call_graph);
}

#[test]
fn fn_values_through_two_calls_and_a_view_field() {
    let (_p, tc, d) = dump(
        "fn leaf() { }
fn other() { }
fn inner(f: fn()) { f(); }
fn outer(f: fn()) { inner(f); }
view struct Cb { f: fn() }
fn fire(c: &Cb) {
    let f = c.f;
    f();
}
fn main() {
    outer(leaf);
    let c = Cb.{ f: other };
    fire(&c);
}
",
    );
    has(&d, "edge main.inner -> main.leaf indirect @3:21");
    lacks(&d, "edge main.inner -> main.other");
    has(&d, "edge main.fire -> main.other indirect @8:5");
    lacks(&d, "edge main.fire -> main.leaf");
    lacks(&d, "edge main.inner -> table");
    lacks(&d, "edge main.fire -> table");
    check_invariants(&tc.call_graph);
}

#[test]
fn drop_cycles_of_recursive_types_are_marked() {
    let (_p, tc, d) = dump(
        "import std.vec;
import std.mem;
struct N { kids: vec.Vec<N> }
struct M { kids: vec.Vec<M> }
impl Drop for M {
    fn drop(&mut self) {
        let k = mem.replace(&mut self.kids, vec.Vec::<M>.new());
        release(k);
    }
}
fn release(k: vec.Vec<M>) { }
fn main() {
    let n = N.{ kids: vec.Vec::<N>.new() };
    let m = M.{ kids: vec.Vec::<M>.new() };
}
",
    );
    let g = &tc.call_graph;
    has(&d, "edge main.main -> drop_glue N drop @13:9");
    has(&d, "edge drop_glue N -> drop_glue Vec<N> drop");
    has(&d, "edge drop_glue Vec<N> -> std.vec.Vec.drop<N> drop");
    // dropping an N drops its kids: a drop cycle through std only
    let n = d.lines().find(|l| l.starts_with("cycle") && l.contains("drop_glue N")).expect("N cycle");
    assert!(n.ends_with(" drop") && n.contains("std.vec.Vec.clear<N>"), "{n}");
    // M's drop reaches a user fn that drops the kids: not a drop cycle
    let m = d.lines().find(|l| l.starts_with("cycle") && l.contains("drop_glue M")).expect("M cycle");
    assert!(m.contains("main.release") && m.contains("main.M.drop") && !m.ends_with(" drop"), "{m}");
    let s = g.scc_of[node(g, &d, "drop_glue N") as usize];
    assert!(g.is_cycle(s) && g.is_drop_cycle(s));
    assert!(!g.is_drop_cycle(g.scc_of[node(g, &d, "main.release") as usize]));
    check_invariants(g);
}

#[test]
fn drops_of_temporaries_moved_values_and_fields() {
    let (_p, tc, d) = dump(
        "struct G { v: u32 }
impl Drop for G {
    fn drop(&mut self) { }
}
struct H { g: G }
fn make() -> G { return G.{ v: 1u32 }; }
fn consume(g: G) { }
fn temp() -> u32 { return make().v; }
fn moved() {
    let g = make();
    consume(g);
}
fn field() {
    let h = H.{ g: G.{ v: 2u32 } };
}
fn plain(x: u32, r: &G) -> u32 { return x + r.v; }
fn main() {
    let a = temp();
    moved();
    field();
    let b = plain(1u32, &make());
}
",
    );
    has(&d, "edge main.temp -> drop_glue G drop @8:27");
    has(&d, "edge main.consume -> drop_glue G drop @7:12");
    has(&d, "edge main.moved -> drop_glue G drop @10:9");
    has(&d, "edge main.field -> drop_glue H drop @14:9");
    has(&d, "edge drop_glue H -> drop_glue G drop");
    has(&d, "edge drop_glue G -> main.G.drop drop");
    // a borrowed value is not dropped here
    lacks(&d, "edge main.plain -> drop_glue");
    has(&d, "edge main.main -> drop_glue G drop");
    check_invariants(&tc.call_graph);
}

#[test]
fn shipped_std_graph_is_consistent_and_independent_of_load_order() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/std_root/main.stv");
    for t in ["x86_64-linux", "aarch64-linux"] {
        let mut c = Config::new(&root).with_shipped(repo().join("std"), repo().join("runtime"));
        c.target = stave_target::Triple::parse(t).ok();
        let p = Program::load(&c);
        let tc = stave_sema::check(&p);
        check_invariants(&tc.call_graph);
        let d = tc.call_graph.dump(&p, &tc);
        // std and runtime have no call cycle but drop cycles
        let cycles: Vec<&str> =
            d.lines().filter(|l| l.starts_with("cycle") && !l.ends_with(" drop")).collect();
        assert!(cycles.is_empty(), "{t}: {}", cycles.join("\n"));
        c.shuffle_seed = Some(7);
        let p2 = Program::load(&c);
        let tc2 = stave_sema::check(&p2);
        assert_eq!(d, tc2.call_graph.dump(&p2, &tc2), "{t}: graph depends on the load order");
    }
}

#[test]
fn corpus_graph_is_independent_of_load_order_and_threads() {
    let out = std::env::temp_dir().join(format!("stave-call-graph-corpus-{}", std::process::id()));
    let o = gen_corpus::Options {
        lines: 6_000,
        seed: 3,
        out: out.clone(),
        mix: gen_corpus::Mix::default_mix().unwrap(),
        threads: 0,
        force: true,
    };
    gen_corpus::generate(&o).unwrap();
    let load = |shuffle: Option<u64>| {
        let mut c = Config::new(&out).with_shipped(repo().join("std"), repo().join("runtime"));
        c.target = stave_target::Triple::parse("x86_64-linux").ok();
        c.shuffle_seed = shuffle;
        let p = Program::load(&c);
        let tc = stave_sema::check(&p);
        let d = tc.call_graph.dump(&p, &tc);
        (tc, d)
    };
    let (tc, base) = load(None);
    check_invariants(&tc.call_graph);
    assert!(tc.call_graph.nodes.len() > 500 && tc.call_graph.level_count() > 3, "{}", own(&base));
    assert_eq!(load(Some(11)).1, base, "shuffled load order");
    for threads in [1usize, 5] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        assert!(pool.install(|| load(None).1) == base, "threads {threads}");
    }
    let _ = std::fs::remove_dir_all(&out);
}

/// The E1402 / E1409 diagnostics of a program: (code, message, line of the primary span, the
/// related labels as `line:label`).
fn recursion_errors(src: &str) -> Vec<(String, String, u32, Vec<String>)> {
    let (p, tc) = load(src, "x86_64-linux");
    let line = |s: stave_diag::Span| p.source_map.locate(s).map_or(0, |l| l.line);
    tc.all_diagnostics(&p)
        .iter()
        .filter(|d| ["E1402", "E1409"].contains(&d.code()))
        .map(|d| {
            let related = d.related().iter().map(|r| format!("{}:{}", line(r.span), r.label)).collect();
            (d.code().to_string(), d.message().to_string(), d.primary().map_or(0, line), related)
        })
        .collect()
}

#[test]
fn unmarked_recursion_names_the_cycle() {
    let errs = recursion_errors(
        "fn a(n: u32) -> u32 {
    if n == 0 { return 0; }
    return b(n - 1);
}
fn b(n: u32) -> u32 { return c(n); }
fn c(n: u32) -> u32 { return a(n) + b(n); }
rec fn d(n: u32) -> u32 {
    if n == 0 { return 0; }
    return e(n - 1);
}
fn e(n: u32) -> u32 { return d(n); }
fn main() {
    let x = a(3) + d(3);
}
",
    );
    assert_eq!(errs.len(), 2, "{errs:?}");
    let (code, msg, at, rel) = &errs[0];
    assert_eq!((code.as_str(), *at), ("E1402", 3));
    assert_eq!(
        msg,
        "recursive call cycle `main.a -> main.b -> main.c -> main.a`: `main.a` is not a `rec fn`"
    );
    assert_eq!(
        rel,
        &[
            "3:`main.a` calls the next fn of the cycle here".to_string(),
            "5:next call of the cycle".to_string(),
            "6:next call of the cycle".to_string(),
            "5:also in this recursion, not a `rec fn`".to_string(),
            "6:also in this recursion, not a `rec fn`".to_string(),
        ]
    );
    let (code, msg, at, _) = &errs[1];
    assert_eq!((code.as_str(), *at), ("E1402", 11));
    assert_eq!(msg, "recursive call cycle `main.e -> main.d -> main.e`: `main.e` is not a `rec fn`");
}

#[test]
fn a_cycle_through_a_closure_is_its_own_error() {
    let errs = recursion_errors(
        "fn call(f: fn(u32) -> u32, n: u32) -> u32 { return f(n); }
fn again(n: u32) -> u32 {
    return call(fn(x: u32) -> u32 { return again(x); }, n);
}
fn main() {
    let x = again(1);
}
",
    );
    assert_eq!(errs.len(), 1, "{errs:?}");
    let (code, msg, at, _) = &errs[0];
    assert_eq!((code.as_str(), *at), ("E1409", 3));
    assert_eq!(
        msg,
        "recursive call cycle `closure in main.again -> main.again -> main.call -> closure in main.again` goes through a closure literal in `main.again`"
    );
}

#[test]
fn generic_recursion_is_reported_once_with_the_instance() {
    let errs = recursion_errors(
        "fn deep<X>(x: X, n: u32) -> u32 {
    if n == 0 { return 0; }
    return deep(x, n - 1);
}
fn main() {
    let a = deep(7u8, 2) + deep(true, 2);
}
",
    );
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert!(errs[0].1.starts_with("recursive call cycle `main.deep<"), "{errs:?}");
    assert!(errs[0].1.ends_with("`main.deep` is not a `rec fn`"), "{errs:?}");
}

/// A fn taking a fn value gets a node per known value it is called with, so a helper called with
/// `twice` and, inside `twice`, with a closure is no cycle; values that are not known at the call
/// share the plain node.
#[test]
fn fn_params_are_specialized_per_known_value() {
    let src = "fn twice(x: u32) -> u32 {
    let f = fn(y: u32) -> u32 { return y * 2; };
    return apply(f, x);
}
fn apply(f: fn(u32) -> u32, x: u32) -> u32 { return f(x); }
fn pick(k: bool) -> u32 {
    let g = if k then twice else twice;
    return apply(g, 1);
}
fn main() {
    let a = apply(twice, 5) + pick(true);
}
";
    let (p, tc) = load(src, "x86_64-linux");
    let d = tc.call_graph.dump(&p, &tc);
    let has_raw = |l: &str| assert!(d.lines().any(|x| x.starts_with(l)), "no `{l}` in:\n{}", own(&d));
    has_raw("edge main.main -> main.apply{0=main.twice} call");
    has_raw("edge main.apply{0=main.twice} -> main.twice indirect");
    has_raw("edge main.twice -> main.apply{0=main.twice.0} call");
    has_raw("edge main.apply{0=main.twice.0} -> main.twice.0 indirect");
    // an untracked value goes to the plain node, which calls through its fn-type table
    has_raw("edge main.pick -> main.apply call");
    has_raw("edge main.apply -> table fn(u32) -> u32 indirect");
    assert!(!d.lines().any(|l| l.starts_with("cycle") && !l.ends_with(" drop")), "{}", own(&d));
    check_invariants(&tc.call_graph);
}
