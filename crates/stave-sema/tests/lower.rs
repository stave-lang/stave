//! Lowering to the checked IR: the shipped std and runtime plus
//! the run-test programs of `tests/conformance/run/` lower and verify, and run in the stave-ir
//! interpreter from `runtime.start.entry` (the `_start` path: argv/auxv, signal setup, main's
//! stack, `call_on_stack` into `__stave_main`, `exit_group`); traps go to `runtime.panic.trap`,
//! which prints the panic line from the site table. The `ir` dump is deterministic across
//! thread counts and the shuffle knob.

use std::path::{Path, PathBuf};

use stave_ir::interp::{Config as IConfig, Interp, Outcome, Val};
use stave_ir::{BufferHost, Module, PanicKind, Site, SyscallAbi};
use stave_sema::{Config, Mode, Program};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn config(root: &Path, target: &str, shuffle: Option<u64>) -> Config {
    let mut c = Config::new(root).with_shipped(repo().join("std"), repo().join("runtime"));
    c.target = stave_target::Triple::parse(target).ok();
    c.shuffle_seed = shuffle;
    c
}

/// Loads, checks, lowers, and verifies; panics with the reasons on failure.
fn lowered(root: &Path, target: &str, mode: Mode, shuffle: Option<u64>) -> Module {
    let p = Program::load(&config(root, target, shuffle));
    let tc = stave_sema::check(&p);
    let d = tc.all_diagnostics(&p);
    if !d.is_empty() {
        let text = stave_diag::emit_human(&d, &p.source_map).unwrap_or_default();
        panic!("{}: diagnostics:\n{text}", root.display());
    }
    let m = match stave_sema::lower(&p, &tc, mode) {
        Ok(m) => m,
        Err(es) => {
            let v: Vec<String> = es.iter().take(40).map(|e| e.to_string()).collect();
            panic!("{}: {} lowering errors:\n{}", root.display(), es.len(), v.join("\n"));
        }
    };
    if let Err(es) = stave_ir::verify_module(&m) {
        let v: Vec<String> = es
            .iter()
            .take(30)
            .map(|e| {
                let f = e
                    .func
                    .and_then(|f| m.func(f))
                    .map_or(String::new(), |f| format!("{} @{}", f.path, f.instance));
                format!("{f}: {e}")
            })
            .collect();
        panic!("{}: {} verifier errors:\n{}", root.display(), es.len(), v.join("\n"));
    }
    m
}

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
    steps: u64,
}

const KINDS: [PanicKind; 7] = PanicKind::ALL;

/// Runs a lowered program from `runtime.start.entry` in the interpreter.
fn run(m: &mut Module, abi: SyscallAbi, files: &[(String, Vec<u8>)]) -> Run {
    let stub = stave_sema::lower::interp_main_stub(m, "main.main").expect("main.main");
    let entry = m.find_func("runtime.start.entry").expect("runtime.start.entry");
    let trap = m.find_func("runtime.panic.trap").expect("runtime.panic.trap");
    let mut host = BufferHost::new(abi);
    host.values.insert("main_addr".into(), stub.0 as u64);
    host.values.insert("signal_handler_addr".into(), 0);
    for (k, v) in files {
        host.files.insert(k.as_bytes().to_vec(), v.clone());
    }
    let cfg = IConfig { max_steps: 20_000_000_000, max_depth: 20_000, ..IConfig::default() };
    let m: &Module = m;
    let mut it = Interp::new(m, host, cfg).unwrap_or_else(|e| panic!("interp: {e:?}"));
    let n = stave_sema::lower::site_table_bytes(m, 0).len() as u64;
    let st = it.mem.alloc(n.max(8), 8).expect("site table");
    let bytes = stave_sema::lower::site_table_bytes(m, st);
    it.mem.write(st, &bytes).expect("site table write");
    let pc = it.mem.alloc(8, 8).expect("pc table");
    it.host.values.insert("site_table".into(), st);
    it.host.values.insert("pc_table".into(), pc);
    let sp = it.process_stack(&[b"prog"], &[]).expect("process stack");
    let mut out = it.run(entry, vec![Val::S(sp)]).unwrap_or_else(|e| panic!("run: {e:?}"));
    if let Outcome::Trap(t) = out {
        let kind = KINDS.iter().position(|k| *k == t.kind).unwrap_or(0) as u64;
        let site = if t.kind == PanicKind::StackOverflow {
            m.func(t.func).map_or(Site::NONE, |f| f.name_site)
        } else {
            t.site
        };
        out =
            it.run(trap, vec![Val::S(kind), Val::S(site.0 as u64)]).unwrap_or_else(|e| panic!("trap: {e:?}"));
    }
    let code = match out {
        Outcome::Exit(c) => c,
        o => panic!("program did not exit: {o:?}"),
    };
    let steps = it.steps();
    let h = it.into_host();
    Run {
        code,
        stdout: String::from_utf8_lossy(&h.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&h.stderr).into_owned(),
        steps,
    }
}

fn data_files(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut v = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_file() {
                let name = p.file_name().unwrap().to_string_lossy().into_owned();
                v.push((name, std::fs::read(&p).unwrap()));
            }
        }
    }
    v
}

/// Expectations of a run test: (stdout, exit, stderr-contains).
fn expectations(p: &Path) -> (Option<String>, i32, Vec<String>) {
    if p.is_dir() {
        let out = std::fs::read_to_string(p.join("expected.stdout")).ok();
        let exit = std::fs::read_to_string(p.join("expected.exit")).ok().and_then(|s| s.trim().parse().ok());
        return (out, exit.unwrap_or(0), Vec::new());
    }
    let text = std::fs::read_to_string(p).unwrap();
    let mut out = None;
    let mut exit = 0;
    let mut errs = Vec::new();
    for line in text.lines() {
        let Some(i) = line.find("//~ ") else { continue };
        let rest = line[i + 4..].trim();
        if let Some(f) = rest.strip_prefix("STDOUT ") {
            out = std::fs::read_to_string(p.parent().unwrap().join(f.trim())).ok();
        } else if let Some(n) = rest.strip_prefix("EXIT ") {
            exit = n.trim().parse().unwrap_or(0);
        } else if let Some(s) = rest.strip_prefix("STDERR-CONTAINS ") {
            let s = s.trim().trim_matches('"');
            errs.push(s.replace("\\n", "\n").replace("\\\"", "\"").replace("\\\\", "\\"));
        }
    }
    (out, exit, errs)
}

fn run_programs() -> Vec<PathBuf> {
    let dir = repo().join("tests/conformance/run");
    let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("stave.toml").is_file() || p.extension().is_some_and(|x| x == "stv"))
        .collect();
    v.sort();
    v
}

fn check_run(p: &Path, target: &str, abi: SyscallAbi, mode: Mode) -> Result<String, String> {
    let mut m = lowered(p, target, mode, None);
    let files = if p.is_dir() { data_files(p) } else { Vec::new() };
    let r = run(&mut m, abi, &files);
    let (out, exit, errs) = expectations(p);
    let name = p.file_name().unwrap().to_string_lossy().into_owned();
    let mut bad = Vec::new();
    if r.code != exit {
        bad.push(format!("exit {} (want {exit})", r.code));
    }
    if let Some(o) = out
        && o != r.stdout
    {
        bad.push(format!("stdout {:?}\n  want {:?}", r.stdout, o));
    }
    for e in &errs {
        if !r.stderr.contains(e.as_str()) {
            bad.push(format!("stderr {:?} lacks {:?}", r.stderr, e));
        }
    }
    if bad.is_empty() {
        Ok(format!("{name}: ok, exit {}, {} steps", r.code, r.steps))
    } else {
        Err(format!("{name}: {}\n  stderr: {:?}", bad.join("; "), r.stderr))
    }
}

#[test]
fn hello_prints_and_exits_zero() {
    let p = repo().join("tests/conformance/run/hello");
    let mut m = lowered(&p, "x86_64-linux", Mode::Debug, None);
    let r = run(&mut m, SyscallAbi::X86_64, &[]);
    assert_eq!(r.stdout, "hello\n", "stderr: {}", r.stderr);
    assert_eq!(r.code, 0);
}

#[test]
fn run_programs_in_the_interpreter() {
    let mut fails = Vec::new();
    let mut oks = Vec::new();
    for (p, mode) in run_programs().into_iter().flat_map(|p| [(p.clone(), Mode::Debug), (p, Mode::Release)]) {
        match std::panic::catch_unwind(|| check_run(&p, "x86_64-linux", SyscallAbi::X86_64, mode)) {
            Ok(Ok(s)) => oks.push(format!("{s} ({mode:?})")),
            Ok(Err(e)) => fails.push(e),
            Err(e) => {
                let msg = e
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default();
                fails.push(format!("{}: panicked: {msg}", p.display()));
            }
        }
    }
    eprintln!("{}", oks.join("\n"));
    assert!(fails.is_empty(), "{} failed:\n{}", fails.len(), fails.join("\n"));
}

#[test]
fn release_and_aarch64_runs() {
    for name in ["hello", "fib", "panic_unwrap.stv"] {
        let p = repo().join("tests/conformance/run").join(name);
        let a = check_run(&p, "x86_64-linux", SyscallAbi::X86_64, Mode::Release);
        assert!(a.is_ok(), "{a:?}");
        let b = check_run(&p, "aarch64-linux", SyscallAbi::Aarch64, Mode::Debug);
        assert!(b.is_ok(), "{b:?}");
    }
}

#[test]
fn ir_dump_is_deterministic() {
    for name in ["fib", "json_lite", "wordcount"] {
        let p = repo().join("tests/conformance/run").join(name);
        let base = stave_ir::print_checked(&lowered(&p, "x86_64-linux", Mode::Debug, None));
        assert!(base.contains("fn @"));
        for threads in [1usize, 3, 12] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            let t = pool.install(|| stave_ir::print_checked(&lowered(&p, "x86_64-linux", Mode::Debug, None)));
            assert!(t == base, "{name}: threads {threads} differ");
        }
        for seed in [1u64, 7, 99] {
            let t = stave_ir::print_checked(&lowered(&p, "x86_64-linux", Mode::Debug, Some(seed)));
            assert!(t == base, "{name}: shuffle {seed} differs");
        }
    }
}

#[test]
fn std_and_runtime_lower_and_verify_on_both_arches() {
    let p = repo().join("tests/conformance/run/hello");
    for t in ["x86_64-linux", "aarch64-linux"] {
        let m = lowered(&p, t, Mode::Debug, None);
        assert!(m.funcs.len() > 100, "{t}: {} funcs", m.funcs.len());
        let text = stave_ir::print_checked(&m);
        assert!(!text.contains(" ?") && !text.contains("%?") && !text.contains("b?"), "{t}: unresolved ids");
        assert!(!text.contains("\n\n") && !text.contains(" \n"));
    }
}

/// A golden for the construct table and SSA rule on a small fn (contracts/checked-ir.md).
#[test]
fn checked_ir_of_small_fns() {
    let dir = std::env::temp_dir().join(format!("stave-lower-small-{}", std::process::id()));
    let _ = std::fs::create_dir_all(dir.join("src"));
    std::fs::write(dir.join("stave.toml"), "name = \"app\"\nkind = \"bin\"\n").unwrap();
    std::fs::write(
        dir.join("src/main.stv"),
        "fn max(a: i32, b: i32) -> i32 {\n    if a > b { return a; }\n    return b;\n}\n\n\
         fn sum(n: u32) -> u32 {\n    let mut s: u32 = 0;\n    let mut i: u32 = 0;\n    while i < n {\n        s = s + i;\n        i = i + 1;\n    }\n    return s;\n}\n\n\
         fn main() -> i32 {\n    let x = max(3, 4) + 1;\n    let t = sum(4);\n    return x;\n}\n",
    )
    .unwrap();
    let m = lowered(&dir, "x86_64-linux", Mode::Release, None);
    let text = stave_ir::print_checked(&m);
    let f = |name: &str| -> String {
        let start = text.find(&format!(" {name} {{\n")).map(|i| text[..i].rfind("fn @").unwrap_or(i));
        let s = start.unwrap_or_else(|| panic!("no {name} in\n{text}"));
        let e = text[s..].find("}\n").map_or(text.len(), |e| s + e + 2);
        text[s..e].to_string()
    };
    let max = f("main.max");
    let max_body: String = max.lines().skip(1).collect::<Vec<_>>().join("\n");
    assert_eq!(
        max_body,
        "b0(%0: t3, %1: t3):\n  %2 = gt t5 %0 %1\n  cbr %2 b1() b2()\nb1():\n  ret %0\nb2():\n  ret %1\n}"
            .replace("t3", &ty_of(&text, "int s 32"))
            .replace("t5", &ty_of(&text, "bool")),
        "\n{text}"
    );
    let sum = f("main.sum");
    // while: head (s, i), body, exit; loop-carried params, no copies
    assert!(sum.contains("b1(%1: "), "{sum}");
    assert!(sum.contains("lt "), "{sum}");
    assert!(sum.contains("br b1("), "{sum}");
    let _ = std::fs::remove_dir_all(&dir);
}

fn ty_of(text: &str, key: &str) -> String {
    for l in text.lines() {
        if let Some(rest) = l.strip_prefix("type ")
            && let Some((t, k)) = rest.split_once(' ')
            && k == key
        {
            return t.to_string();
        }
    }
    String::from("t?")
}

#[test]
fn generated_corpus_lowers_and_verifies() {
    let lines: u64 =
        std::env::var("STAVE_LOWER_CORPUS_LINES").ok().and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let out = std::env::temp_dir().join(format!("stave-lower-corpus-{lines}-{}", std::process::id()));
    let o = gen_corpus::Options {
        lines,
        seed: 1,
        out: out.clone(),
        mix: gen_corpus::Mix::default_mix().unwrap(),
        threads: 0,
        force: true,
    };
    gen_corpus::generate(&o).unwrap();
    let p = Program::load(&config(&out, "x86_64-linux", None));
    let tc = stave_sema::check(&p);
    assert!(tc.all_diagnostics(&p).is_empty());
    let pool = rayon::ThreadPoolBuilder::new().num_threads(12).build().unwrap();
    let mut best = std::time::Duration::MAX;
    let mut module = None;
    for _ in 0..5 {
        let t0 = std::time::Instant::now();
        let r = pool.install(|| stave_sema::lower(&p, &tc, Mode::Debug));
        let dt = t0.elapsed();
        best = best.min(dt);
        match r {
            Ok(m) => module = Some(m),
            Err(es) => {
                let v: Vec<String> = es.iter().take(30).map(|e| e.to_string()).collect();
                panic!("{} lowering errors:\n{}", es.len(), v.join("\n"));
            }
        }
    }
    let m = module.unwrap();
    if let Err(es) = stave_ir::verify_module(&m) {
        let v: Vec<String> = es
            .iter()
            .take(30)
            .map(|e| {
                let f = e.func.and_then(|f| m.func(f)).map_or(String::new(), |f| f.path.clone());
                format!("{f}: {e}")
            })
            .collect();
        panic!("{} verifier errors:\n{}", es.len(), v.join("\n"));
    }
    let src_lines: usize = p.files.iter().map(|f| f.bytes.iter().filter(|&&b| b == b'\n').count()).sum();
    eprintln!(
        "corpus {lines} lines ({src_lines} with std/runtime): {} funcs, mono+lower best {:.1} ms on 12 threads = {:.1} ms per 100k lines",
        m.funcs.len(),
        best.as_secs_f64() * 1e3,
        best.as_secs_f64() * 1e3 * 100_000.0 / src_lines as f64
    );
    let a = stave_ir::print_checked(&m);
    let m1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| stave_sema::lower(&p, &tc, Mode::Debug));
    assert!(stave_ir::print_checked(&m1.unwrap()) == a, "thread count changes the ir dump");
    let _ = std::fs::remove_dir_all(&out);
}

/// `STAVE_LOWER_DUMP=<program dir or .stv> cargo test -p stave-sema --test lower dump -- --nocapture`
/// prints the checked IR and the interpreter run of one program.
#[test]
fn dump() {
    let Ok(path) = std::env::var("STAVE_LOWER_DUMP") else { return };
    let p = PathBuf::from(path);
    let mode = if std::env::var("STAVE_LOWER_RELEASE").is_ok() { Mode::Release } else { Mode::Debug };
    let mut m = lowered(&p, "x86_64-linux", mode, None);
    if std::env::var("STAVE_LOWER_IR").is_ok() {
        println!("{}", stave_ir::print_checked(&m));
    }
    let files = if p.is_dir() { data_files(&p) } else { Vec::new() };
    let r = run(&mut m, SyscallAbi::X86_64, &files);
    println!("exit {} steps {}\n--- stdout\n{}--- stderr\n{}", r.code, r.steps, r.stdout, r.stderr);
}

fn temp_pkg(name: &str, src: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("stave-lower-{name}-{}", std::process::id()));
    let _ = std::fs::create_dir_all(dir.join("src"));
    std::fs::write(dir.join("stave.toml"), format!("name = \"{name}\"\nkind = \"bin\"\n")).unwrap();
    std::fs::write(dir.join("src/main.stv"), src).unwrap();
    dir
}

/// Drop: scope end in reverse order, overwrite, `mem.drop`, moved values (a dynamic
/// drop flag for a conditional move), by-value params, f-string temporaries, Vec of String.
#[test]
fn drops_run_at_scope_end_on_overwrite_and_not_after_moves() {
    let src = r#"import std.io;
import std.mem;
import std.vec;
import std.string;

struct Guard { pub id: u32 }

impl Drop for Guard {
    fn drop(&mut self) { io.print(f"drop {self.id}\n"); }
}

fn take(g: Guard) {
    io.print(f"take {g.id}\n");
}

fn cond(c: bool) {
    let g = Guard.{ id: 10 };
    if c {
        take(g);
    }
    io.print("end cond\n");
}

fn pick(c: bool) -> Guard {
    let a = Guard.{ id: 20 };
    let b = Guard.{ id: 21 };
    if c { return a; }
    return b;
}

fn main() -> i32 {
    let a = Guard.{ id: 1 };
    let b = Guard.{ id: 2 };
    mem.drop(a);
    let mut c = Guard.{ id: 3 };
    c = Guard.{ id: 4 };
    cond(true);
    cond(false);
    let p = pick(true);
    io.print(f"picked {p.id}\n");
    let mut v = vec.Vec::<string.String>.new();
    v.push(string.String.from_str("x"));
    v.push(f"y{b.id}");
    let n = v.len();
    io.print(f"n {n}\n");
    for i in 0u32..3 {
        let g = Guard.{ id: 30 + i };
        if i == 1 { continue; }
        io.print(f"loop {i}\n");
    }
    return 0;
}
"#;
    let dir = temp_pkg("drops", src);
    let mut m = lowered(&dir, "x86_64-linux", Mode::Debug, None);
    let r = run(&mut m, SyscallAbi::X86_64, &[]);
    let want = "drop 1\ndrop 3\ntake 10\ndrop 10\nend cond\nend cond\ndrop 10\ndrop 21\npicked 20\nn 2\n\
                loop 0\ndrop 30\ndrop 31\nloop 2\ndrop 32\ndrop 20\ndrop 4\ndrop 2\n";
    assert_eq!(r.stdout, want, "stderr: {}", r.stderr);
    assert_eq!(r.code, 0);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A tour of the lowered constructs: user operator impls, trait methods through a bound,
/// matches by reference, guards and ranges (chain mode), `?`, fn items and closures as fn
/// values, `loop`/`break`, arrays (SSA `set`), `for` over `&[T; N]`, casts, tuples, consts,
/// `static mut`, `size_of`, chars, floats.
#[test]
fn language_tour_runs() {
    let src = r#"import std.io;

struct P { pub x: i32, pub y: i32 }

impl Copy for P {}

impl Add for P {
    fn add(self, rhs: Self) -> Self { return Self.{ x: self.x + rhs.x, y: self.y + rhs.y }; }
}

enum Shape { Circle(i32), Rect { w: i32, h: i32 }, Empty }

trait Area { fn area(&self) -> i32; }

impl Area for Shape {
    fn area(&self) -> i32 {
        return match self {
            Shape.Circle(r) => 3 * *r * *r,
            Shape.Rect.{ w, h } => *w * *h,
            Shape.Empty => 0,
        };
    }
}

fn total<T: Area>(xs: &[T]) -> i32 {
    let mut s: i32 = 0;
    for i in 0..xs.len() {
        s = s + xs[i].area();
    }
    return s;
}

fn classify(n: i32) -> i32 {
    return match n {
        0 => 100,
        x if x < 0 => -1,
        1..=9 => 1,
        _ => 2,
    };
}

fn first_even(xs: &[i32]) -> Option<i32> {
    for i in 0..xs.len() {
        if xs[i] % 2 == 0 { return Option.Some(xs[i]); }
    }
    return Option.None;
}

fn sum_two(xs: &[i32]) -> Option<i32> {
    let a = first_even(xs)?;
    return Option.Some(a + 1);
}

fn apply(f: fn(i32) -> i32, x: i32) -> i32 { return f(x); }

fn double(x: i32) -> i32 { return x * 2; }

const K: i32 = 7;
static mut COUNT: u32 = 0;

fn bump() -> u32 {
    unsafe("single thread") { COUNT = COUNT + 1; }
    return unsafe("single thread") COUNT;
}

fn main() -> i32 {
    let p = P.{ x: 1, y: 2 } + P.{ x: 10, y: 20 };
    io.print(f"p {p.x} {p.y}\n");
    let shapes: [Shape; 3] = [Shape.Circle(2), Shape.Rect.{ w: 3, h: 4 }, Shape.Empty];
    let t = total(&shapes);
    io.print(f"total {t}\n");
    let c0 = classify(0);
    let c1 = classify(-5);
    let c2 = classify(7);
    let c3 = classify(70);
    io.print(f"classify {c0} {c1} {c2} {c3}\n");
    let xs: [i32; 4] = [1, 3, 8, 5];
    match sum_two(&xs) {
        Option.Some(v) => { io.print(f"some {v}\n"); }
        Option.None => { io.print("none\n"); }
    }
    let odd: [i32; 2] = [1, 3];
    let r = sum_two(&odd).is_none();
    io.print(f"none {r}\n");
    let a = apply(double, 21);
    let b = apply(fn(x: i32) -> i32 { return x + K; }, 1);
    io.print(f"apply {a} {b}\n");
    let mut arr: [u8; 4] = [0u8; 4];
    let mut i: usize = 0;
    loop {
        if i == 4 { break; }
        arr[i] = (i as% u8) * 3;
        i = i + 1;
    }
    let mut acc: u32 = 0;
    for v in &arr { acc = acc + (*v as u32); }
    io.print(f"acc {acc}\n");
    let big: i64 = 300;
    let s: u8 = big as| u8;
    let o: Option<u8> = big as? u8;
    let n = o.is_none();
    io.print(f"sat {s} {n}\n");
    let tup = (3u8, true);
    let (q, f) = tup;
    io.print(f"tup {q} {f}\n");
    let x1 = bump();
    let x2 = bump();
    io.print(f"count {x2}\n");
    let sz = size_of::<P>();
    io.print(f"size {sz}\n");
    let ch: char = 'z';
    io.print(f"char {ch}\n");
    let fl: f64 = 0.1 + 0.2;
    io.print(f"float {fl}\n");
    return 0;
}
"#;
    let dir = temp_pkg("tour", src);
    let mut m = lowered(&dir, "x86_64-linux", Mode::Debug, None);
    let text = stave_ir::print_checked(&m);
    assert!(text.lines().any(|l| l.starts_with("fn @") && l.contains('.') && l.ends_with("main.main {")));
    assert!(
        text.lines().any(|l| l.starts_with("fn @") && l.split(' ').nth(1).is_some_and(|i| i.contains('.'))),
        "a closure fn @k.0"
    );
    let r = run(&mut m, SyscallAbi::X86_64, &[]);
    let want = "p 11 22\ntotal 24\nclassify 100 -1 1 2\nsome 9\nnone true\napply 42 8\nacc 18\nsat 255 true\n\
                tup 3 true\ncount 2\nsize 8\nchar z\nfloat 0.30000000000000004\n";
    assert_eq!(r.stdout, want, "stderr: {}", r.stderr);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Generated programs (seeds 1..4, 6k lines) run to an exit in the interpreter: no interpreter
/// error (a malformed or unsupported lowering), whatever their exit status.
#[test]
fn generated_programs_run_in_the_interpreter() {
    for seed in 1u64..=4 {
        let out = std::env::temp_dir().join(format!("stave-lower-run-{seed}-{}", std::process::id()));
        let o = gen_corpus::Options {
            lines: 6_000,
            seed,
            out: out.clone(),
            mix: gen_corpus::Mix::default_mix().unwrap(),
            threads: 0,
            force: true,
        };
        gen_corpus::generate(&o).unwrap();
        let mut m = lowered(&out, "x86_64-linux", Mode::Release, None);
        let r = run(&mut m, SyscallAbi::X86_64, &[]);
        eprintln!(
            "seed {seed}: exit {} after {} steps, stdout {:?}, stderr {:?}",
            r.code, r.steps, r.stdout, r.stderr
        );
        assert!(r.code == 0 || r.code == 101, "seed {seed}: exit {}", r.code);
        let _ = std::fs::remove_dir_all(&out);
    }
}
