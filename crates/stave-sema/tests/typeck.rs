//! Type-checker tests: the shipped std and runtime check clean on both CPU
//! architectures; generated corpora check clean and canonically (same `types` dump whatever the
//! internal order); recorded types, targets, coercions, and instances; no panics on mangled
//! input.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use stave_sema::types::Interner;
use stave_sema::{Config, Fs, Program, Target, TypeKind, Typeck};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn shipped(root: &Path, target: Option<&str>, shuffle: Option<u64>) -> Config {
    let mut c = Config::new(root).with_shipped(repo().join("std"), repo().join("runtime"));
    c.target = target.and_then(|t| stave_target::Triple::parse(t).ok());
    c.shuffle_seed = shuffle;
    c
}

fn describe(p: &Program, tc: &Typeck) -> Vec<String> {
    tc.all_diagnostics(p)
        .iter()
        .take(30)
        .map(|d| {
            let l = d.primary().and_then(|s| p.source_map.locate(s).ok());
            format!("{} {:?} {}", d.code(), l.map(|l| (l.path.to_string(), l.line, l.col)), d.message())
        })
        .collect()
}

#[test]
fn shipped_std_and_runtime_check_clean_on_both_arches() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/std_root/main.stv");
    for t in ["x86_64-linux", "aarch64-linux"] {
        let p = Program::load(&shipped(&root, Some(t), None));
        let tc = stave_sema::check(&p);
        let d = describe(&p, &tc);
        assert!(d.is_empty(), "{t}:\n{}", d.join("\n"));
        assert!(tc.bodies.iter().all(|b| !b.has_error), "{t}: a body is flagged");
        // every std and runtime fn was checked
        assert!(tc.bodies.len() > 300, "{t}: {} bodies", tc.bodies.len());
    }
    assert_eq!(stave_sema::diagnostic_build_failures(), 0);
}

fn make_corpus(lines: u64, seed: u64) -> PathBuf {
    let out = std::env::temp_dir().join(format!("stave-typeck-corpus-{lines}-{seed}-{}", std::process::id()));
    let o = gen_corpus::Options {
        lines,
        seed,
        out: out.clone(),
        mix: gen_corpus::Mix::default_mix().unwrap(),
        threads: 0,
        force: true,
    };
    gen_corpus::generate(&o).unwrap();
    out
}

#[test]
fn generated_corpus_checks_clean_and_canonically() {
    for seed in [1u64, 2, 3] {
        let dir = make_corpus(12_000, seed);
        let p = Program::load(&shipped(&dir, None, None));
        let tc = stave_sema::check(&p);
        let d = describe(&p, &tc);
        assert!(d.is_empty(), "seed {seed}:\n{}", d.join("\n"));
        let base = stave_sema::dump_types(&p, &tc);
        assert!(base.lines().count() > 10_000);
        // shuffled internal order (strings, decls) and a second run: identical dump
        let p2 = Program::load(&shipped(&dir, None, Some(seed + 7)));
        let tc2 = stave_sema::check(&p2);
        assert_eq!(stave_sema::dump_types(&p2, &tc2), base, "seed {seed}");
        assert_eq!(tc2.instances, tc.instances, "seed {seed}");
        // the checker runs with the caller's thread count: any count, same dump
        if seed == 1 {
            for threads in [1usize, 5] {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                let (d, inst) = pool.install(|| {
                    let p3 = Program::load(&shipped(&dir, None, None));
                    let tc3 = stave_sema::check(&p3);
                    (stave_sema::dump_types(&p3, &tc3), tc3.instances)
                });
                assert!(d == base, "threads {threads}: types dump differs");
                assert_eq!(inst, tc.instances, "threads {threads}");
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

fn mem(src: &str) -> (Program, Typeck) {
    mem_with(src, &[])
}

/// [`mem`] with extra files (absolute path, text), e.g. a module added to the built-in `std`.
fn mem_with(src: &str, extra: &[(&str, &str)]) -> (Program, Typeck) {
    let mut m = BTreeMap::new();
    m.insert(PathBuf::from("/m/stave.toml"), b"name = \"m\"\nkind = \"bin\"\n".to_vec());
    m.insert(PathBuf::from("/m/src/main.stv"), src.as_bytes().to_vec());
    for (path, text) in extra {
        m.insert(PathBuf::from(path), text.as_bytes().to_vec());
    }
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
    c.target = stave_target::Triple::parse("x86_64-linux").ok();
    let p = Program::load(&c);
    let tc = stave_sema::check(&p);
    (p, tc)
}

fn main_body<'a>(p: &Program, tc: &'a Typeck, name: &str) -> &'a stave_sema::TypedBody {
    let d = (0..p.decls.len() as u32)
        .map(stave_sema::DeclId)
        .find(|&d| {
            p.decl(d).is_some_and(|x| {
                x.kind == stave_sema::DeclKind::Fn
                    && p.strings.get(x.name) == name
                    && p.module(x.module).is_some_and(|m| p.packages[m.package.idx()].is_root)
            })
        })
        .expect("decl");
    tc.body(d).expect("body")
}

#[test]
fn records_types_targets_coercions_and_instances() {
    let (p, tc) = mem("import std.io;\n\
         fn id<T>(x: T) -> T { return x; }\n\
         struct S { pub v: u32 }\n\
         impl S { pub fn get(&self) -> u32 { return self.v; } }\n\
         fn main() -> i32 {\n\
             let s = S.{ v: 1 };\n\
             let a: u32 = s.get() + id(2u32);\n\
             let b: u8 = 7;\n\
             io.print(f\"{a}\\n\");\n\
             return 0;\n\
         }\n");
    let d = describe(&p, &tc);
    assert!(d.is_empty(), "{}", d.join("\n"));
    let b = main_body(&p, &tc, "main");
    assert!(!b.has_error);
    // a method call with an auto-ref receiver
    let m = b.targets.iter().find_map(|(_, t)| match t {
        Target::Method { adjust, .. } => Some(*adjust),
        _ => None,
    });
    assert_eq!(m.map(|a| a.autoref), Some(Some(false)));
    // the f-string borrowed as `&str`
    assert!(b.coercions.iter().any(|(_, c)| *c == stave_sema::Coercion::FmtToStr));
    // `id::<u32>` instance
    assert!(tc.instances.iter().any(|i| p.strings.get(p.decl(i.decl).unwrap().name) == "id"
        && matches!(i.args.as_slice(), [stave_sema::types::GArg::Type(t)] if tc.types.kind(*t) == &TypeKind::Int { signed: false, bits: stave_sema::types::IntBits::B32 })));
    // locals are typed; the literal 7 got `u8`
    assert!(b.locals.iter().any(|(_, t)| tc.type_key(*t) == "int u 8"));
    let dump = stave_sema::dump_types(&p, &tc);
    assert!(dump.lines().any(|l| l.ends_with(" -")), "module paths print `-`");
}

#[test]
fn reports_core_errors() {
    let (_, tc) = mem("fn f(a: i32, b: i64) -> i64 { return a + b; }\n\
         fn g() -> u32 { let k = 1 + 2; }\n\
         fn main() -> i32 { return 0; }\n");
    let codes: Vec<&str> = tc.diagnostics.iter().map(|d| d.code()).collect();
    assert_eq!(codes, ["E0201", "E0202", "E0207"]);
}

/// Or-pattern diagnostics are located at the alternative: E0302 names the binding and has the
/// first alternative as related span; E0301 (case `alternative`) has the covering pattern. Only
/// the first alternative's bindings are locals, with a resolver alias from each later one.
#[test]
fn or_pattern_diagnostics_point_at_the_alternative() {
    let src = "enum T { A(u32), B(u64), C }\n\
         fn f(t: T) -> u32 {\n\
         \x20   return match t {\n\
         \x20       T.A(n) | T.B(n) => 1,\n\
         \x20       T.C | T.A(_) => 2,\n\
         \x20   };\n\
         }\n\
         fn g(t: T) -> u32 {\n\
         \x20   return match t { T.A(n) | T.B(_) | T.C if true => 1, T.A(n) => n, _ => 0 };\n\
         }\n\
         fn ok(p: (u32, u32, bool)) -> u32 {\n\
         \x20   return match p { (a, b, true) | (b, a, false) => a - b };\n\
         }\n\
         fn main() -> i32 { return 0; }\n";
    let (p, tc) = mem(src);
    let at = |s: stave_diag::Span| {
        let l = p.source_map.locate(s).unwrap();
        (l.line, l.col, l.len)
    };
    let got: Vec<_> = tc
        .diagnostics
        .iter()
        .map(|d| {
            // the labelled spans: the primary one, then the related one
            let rel: Vec<_> = d.related().iter().map(|r| (r.label.as_str(), at(r.span))).collect();
            (d.code(), d.message(), rel)
        })
        .collect();
    let e0302 = "`n` is not bound the same way in every alternative of this pattern";
    let differs = "alternative that differs";
    let want = [
        ("E0302", e0302, vec![(differs, (4, 18, 6)), ("first alternative", (4, 9, 6))]),
        (
            "E0301",
            "unreachable alternative of a match arm",
            vec![
                ("this alternative never matches", (5, 15, 6)),
                ("earlier pattern that already covers it", (4, 9, 6)),
            ],
        ),
        ("E0302", e0302, vec![(differs, (9, 31, 6)), ("first alternative", (9, 22, 6))]),
        ("E0302", e0302, vec![(differs, (9, 40, 3)), ("first alternative", (9, 22, 6))]),
    ];
    assert_eq!(got, want);
    assert_eq!(stave_sema::diagnostic_build_failures(), 0);
    let body = main_body(&p, &tc, "ok");
    assert_eq!(body.locals.len(), 3, "the param and the first alternative's `a` and `b`");
    let res = &p.res[body.item.idx()];
    assert_eq!(res.aliases.len(), 2);
    assert!(
        res.aliases
            .iter()
            .all(|&(alt, first)| alt > first && body.local(stave_sema::BindingRef { node: first }).is_some())
    );
}

/// `static mut` in a built-in package: no E0809, and every use outside `unsafe(..)` (read, write,
/// borrow, f-string hole) is E0803. In the root package the declaration itself is E0809.
#[test]
fn static_mut_rules_by_package() {
    let fixture = "static mut HITS: u32 = 0;\n\
         pub fn bump() -> u32 {\n\
             HITS = 1;\n\
             let a: u32 = HITS;\n\
             let p = &mut HITS as *mut u32;\n\
             unsafe(\"test fixture: one thread\") { HITS = HITS + a; }\n\
             return unsafe(\"test fixture: one thread\") HITS;\n\
         }\n";
    let main = "import std.zz_fixture;\n\
         static mut LOCAL: u32 = 0;\n\
         fn main() -> i32 {\n\
             let n = zz_fixture.bump();\n\
             return unsafe(\"test: one thread\") LOCAL as% i32;\n\
         }\n";
    let (p, tc) = mem_with(main, &[("/std/zz_fixture.stv", fixture)]);
    let mut got: Vec<(&str, bool, u32)> = tc
        .diagnostics
        .iter()
        .map(|d| {
            let sp = d.primary().unwrap();
            let f = p.file(sp.file).unwrap();
            let root = p.packages[f.package.idx()].is_root;
            let line = 1 + f.bytes[..sp.offset as usize].iter().filter(|&&b| b == b'\n').count() as u32;
            (d.code(), root, line)
        })
        .collect();
    got.sort();
    assert_eq!(got, [("E0803", false, 3), ("E0803", false, 4), ("E0803", false, 5), ("E0809", true, 2)]);
}

#[test]
fn no_panics_on_mangled_sources() {
    let base =
        std::fs::read_to_string(repo().join("tests/conformance/types/generic_core_traits.stv")).unwrap();
    let toks: Vec<&str> =
        base.split_inclusive(|c: char| c.is_whitespace() || "(){}[];,.".contains(c)).collect();
    let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
    for round in 0..120u32 {
        let mut out = String::new();
        for (i, t) in toks.iter().enumerate() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            match seed % 23 {
                0 => {}
                1 => out.push_str(toks[(seed as usize / 7) % toks.len()]),
                2 => out.push_str(t.trim()),
                _ => out.push_str(t),
            }
            if i % 97 == 0 && round % 5 == 0 {
                out.push_str("((&*-1 as u8)) ");
            }
        }
        let (_, tc) = mem(&out);
        let _ = tc.diagnostics.len();
    }
    // deep nesting
    let deep = format!(
        "fn main() -> i32 {{ let x: i32 = {}1{}; return {}x; }}\n",
        "(".repeat(3000),
        ")".repeat(3000),
        "-".repeat(5000)
    );
    let (_, tc) = mem(&deep);
    let _ = tc.diagnostics.len();
    let elif = format!(
        "fn f(x: u32) -> u32 {{ return {}0; }}\nfn main() -> i32 {{ return 0; }}\n",
        (0..3000).map(|i| format!("if x == {i} then {i} else ")).collect::<String>()
    );
    let (p, tc) = mem(&elif);
    assert!(tc.all_diagnostics(&p).is_empty());
}

/// The type helpers work on typeck's canonical table (canonical `u8` is id 0, the fresh table's
/// error id): substitution keeps `W<u8>` an ADT, impl matching binds `T = u8`, layouts and
/// lowering-time helpers see the right types (regression: they compared against fixed ids).
#[test]
fn type_helpers_on_the_canonical_table() {
    use stave_sema::Scalar;
    use stave_sema::sigs::TraitRef;
    use stave_sema::traits::World;
    use stave_sema::types::GArg;
    let (p, tc) = mem("pub trait Tr {}\n\
         pub struct W<T> { pub x: Option<T>, pub y: T }\n\
         impl<T> Tr for W<T> {}\n\
         fn main() -> i32 { let w: W<u8> = W.{ x: Option.None, y: 1 }; return 0; }\n");
    assert!(describe(&p, &tc).is_empty(), "{:?}", describe(&p, &tc));
    let mut t = tc.types.clone();
    let u8t = t.scalar_ty(Scalar::U8);
    assert_eq!(u8t, stave_sema::TypeId(0), "canonical u8 must be id 0 for this test to bite");
    assert_eq!(t.error_ty(), tc.error_type);
    let find = |name: &str, kind: stave_sema::DeclKind| {
        (0..p.decls.len() as u32)
            .map(stave_sema::DeclId)
            .find(|&d| p.decl(d).is_some_and(|x| x.kind == kind && p.strings.get(x.name) == name))
            .expect(name)
    };
    let w = find("W", stave_sema::DeclKind::Struct);
    let tr = find("Tr", stave_sema::DeclKind::Trait);
    let opt = tc.sigs.lang.option;
    let adt = tc.sigs.adt(w).expect("W sig");
    let (fx, fy) = (adt.variants[0].fields[0].ty, adt.variants[0].fields[1].ty);
    let args = [GArg::Type(u8t)];
    let sx = t.subst(fx, &args, None);
    assert_eq!(t.kind(sx), &TypeKind::Adt { def: opt, args: Box::new([GArg::Type(u8t)]) });
    assert_eq!(t.subst(fy, &args, None), u8t);
    let wu8 = t.mk_adt(w, vec![GArg::Type(u8t)]);
    assert!(matches!(t.kind(wu8), TypeKind::Adt { def, .. } if *def == w));
    let r = t.mk_ref(false, wu8);
    assert_eq!(t.kind(r), &TypeKind::Ref { mutable: false, inner: wu8 });
    // the body's `w` has exactly this type
    let b = main_body(&p, &tc, "main");
    assert!(b.locals.iter().any(|&(_, lt)| lt == wu8));
    let world = World { prog: &p, sigs: &tc.sigs };
    let (imp, iargs) =
        world.find_impl(&mut t, wu8, &TraitRef { trait_decl: tr, args: Vec::new() }, &[]).expect("impl");
    assert_eq!(p.decl(imp).map(|d| d.kind), Some(stave_sema::DeclKind::Impl));
    assert_eq!(iargs, vec![GArg::Type(u8t)]);
    let imp_def = tc.sigs.impl_def(imp).expect("impl sig");
    assert_eq!(world.impl_applies(&mut t, imp_def, wu8, &[]), Some(vec![GArg::Type(u8t)]));
    assert_eq!(world.impl_applies(&mut t, imp_def, u8t, &[]), None);
    // layout: Option<u8> (tag + payload) then u8
    let lx = stave_sema::traits::layout_of(&mut t, &tc.sigs, sx).expect("layout");
    let lw = stave_sema::traits::layout_of(&mut t, &tc.sigs, wu8).expect("layout");
    assert!(lw.0 > lx.0, "{lx:?} {lw:?}");
}

/// `check` runs with the caller's thread count on big-stack workers: inside a one-thread pool
/// with small stacks, deep nesting still checks (no overflow) and the result is identical.
#[test]
fn check_runs_on_the_callers_thread_count() {
    let deep = format!(
        "fn main() -> i32 {{ let x: i32 = {}1{}; return {}x; }}\n",
        "(".repeat(3000),
        ")".repeat(3000),
        "-".repeat(5000)
    );
    let (p, tc) = mem(&deep);
    let base = stave_sema::dump_types(&p, &tc);
    let one = rayon::ThreadPoolBuilder::new().num_threads(1).stack_size(1 << 20).build().unwrap();
    let (p1, tc1) = one.install(|| mem(&deep));
    assert_eq!(stave_sema::dump_types(&p1, &tc1), base);
}

/// E1407 names the first non-flat component and the path to it; E1408 rejects any impl.
#[test]
fn flat_diagnostics_name_the_first_non_flat_component() {
    let src = "import std.cell;\n\
         struct Raw { pub p: *const u8 }\n\
         struct Outer { pub n: u32, pub inner: Raw }\n\
         enum Msg { Quit, Data(u32, Raw), Named { len: u32, at: Outer } }\n\
         struct Tag<T> { pub raw: u32 }\n\
         struct Handle { pub id: u32 }\n\
         impl Drop for Handle { fn drop(&mut self) {} }\n\
         view struct Cursor { pub at: &u32 }\n\
         impl Flat for Raw {}\n\
         fn need<T: Flat>() {}\n\
         fn fwd<T>() { need::<T>(); }\n\
         fn main() -> i32 {\n\
         \x20   need::<&u32>();\n\
         \x20   need::<fn(u32) -> u32>();\n\
         \x20   need::<Cursor>();\n\
         \x20   need::<Handle>();\n\
         \x20   need::<[[Outer; 2]; 3]>();\n\
         \x20   need::<(u32, Raw)>();\n\
         \x20   need::<Msg>();\n\
         \x20   need::<Option<&u32>>();\n\
         \x20   need::<Tag<*const u8>>();\n\
         \x20   need::<cell.Cell<u32>>();\n\
         \x20   need::<str>();\n\
         \x20   return 0;\n\
         }\n";
    let (p, tc) = mem(src);
    let got: Vec<String> = tc
        .all_diagnostics(&p)
        .iter()
        .map(|d| {
            let line = d.primary().map_or(0, |s| p.source_map.locate(s).map_or(0, |l| l.line));
            format!("{line}: {} {}", d.code(), d.message())
        })
        .collect();
    let want = [
        "9: E1408 `impl Flat for Raw`: `Flat` is derived by the compiler, never implemented",
        "11: E1407 `T` is not `Flat`: it is a generic parameter without a `Flat` bound",
        "13: E1407 `&u32` is not `Flat`: it is a reference",
        "14: E1407 `fn(u32) -> u32` is not `Flat`: it is a fn value",
        "15: E1407 `Cursor` is not `Flat`: it is a view type",
        "16: E1407 `Handle` is not `Flat`: it is a type with `Drop`",
        "17: E1407 `[[Outer; 2]; 3]` is not `Flat`: component `[_][_].inner.p` has type `*const u8`, a raw pointer",
        "18: E1407 `(u32, Raw)` is not `Flat`: component `1.p` has type `*const u8`, a raw pointer",
        "19: E1407 `Msg` is not `Flat`: component `Data.1.p` has type `*const u8`, a raw pointer",
        "20: E1407 `Option<&u32>` is not `Flat`: component `Some.0` has type `&u32`, a reference",
        "21: E1407 `Tag<*const u8>` is not `Flat`: component `<T>` has type `*const u8`, a raw pointer",
        "22: E1407 `Cell<u32>` is not `Flat`: component `v` has type `UnsafeCell<u32>`, `UnsafeCell` (interior mutability)",
        "23: E1407 `str` is not `Flat`: it is unsized",
    ];
    assert_eq!(got, want);
    assert_eq!(stave_sema::diagnostic_build_failures(), 0);
}

/// E1410 names the first component that is not Frozen (`*` for a pointee); views, `Drop` types,
/// and raw pointers to Frozen data are Frozen; a `Flat` bound implies `Frozen`; E1408 rejects an
/// impl of Frozen.
#[test]
fn frozen_diagnostics_name_the_first_unfrozen_component() {
    let src = "import std.atomic;\n\
         import std.cell;\n\
         import std.sync;\n\
         import std.vec;\n\
         struct Plain { pub n: u32 }\n\
         struct Counter { pub n: u32, pub hits: atomic.U64 }\n\
         view struct Cursor { pub at: &Counter }\n\
         impl Frozen for Plain {}\n\
         fn need<T: Frozen + view>() {}\n\
         fn fwd<T>() { need::<T>(); }\n\
         fn via_flat<T: Flat>() { need::<T>(); }\n\
         fn main() -> i32 {\n\
         \x20   need::<&u32>();\n\
         \x20   need::<*const vec.Vec<u32>>();\n\
         \x20   need::<&[u32]>();\n\
         \x20   need::<cell.Cell<u32>>();\n\
         \x20   need::<&atomic.U64>();\n\
         \x20   need::<Cursor>();\n\
         \x20   need::<fn(u32) -> u32>();\n\
         \x20   need::<(u32, sync.Mutex<u32, 1>)>();\n\
         \x20   need::<Option<&[cell.Cell<u32>]>>();\n\
         \x20   return 0;\n\
         }\n";
    let (p, tc) = mem(src);
    let got: Vec<String> = tc
        .all_diagnostics(&p)
        .iter()
        .map(|d| {
            let line = d.primary().map_or(0, |s| p.source_map.locate(s).map_or(0, |l| l.line));
            format!("{line}: {} {}", d.code(), d.message())
        })
        .collect();
    let want = [
        "8: E1408 `impl Frozen for Plain`: `Frozen` is derived by the compiler, never implemented",
        "10: E1410 `T` is not `Frozen`: it is a generic parameter without a `Frozen` bound",
        "16: E1410 `Cell<u32>` is not `Frozen`: component `v` has type `UnsafeCell<u32>`, `UnsafeCell` (interior mutability)",
        "17: E1410 `&U64` is not `Frozen`: component `*.v` has type `UnsafeCell<u64>`, `UnsafeCell` (interior mutability)",
        "18: E1410 `Cursor` is not `Frozen`: component `at.*.hits.v` has type `UnsafeCell<u64>`, `UnsafeCell` (interior mutability)",
        "19: E1410 `fn(u32) -> u32` is not `Frozen`: it is a fn value (its captures are not in its type)",
        "20: E1410 `(u32, Mutex<u32, 1>)` is not `Frozen`: component `1.word` has type `UnsafeCell<u32>`, `UnsafeCell` (interior mutability)",
        "21: E1410 `Option<&[Cell<u32>]>` is not `Frozen`: component `Some.0.*[_].v` has type `UnsafeCell<u32>`, `UnsafeCell` (interior mutability)",
    ];
    assert_eq!(got, want);
    assert_eq!(stave_sema::diagnostic_build_failures(), 0);
}

/// E0211 cases `frozen` / `frozen-self` name the touched value and the path to its `UnsafeCell`
/// or fn value; case `static` names the chain to a static that is not Frozen.
#[test]
fn contract_diagnostics_name_the_shared_state() {
    let src = "import std.atomic;\n\
         static HITS: atomic.U64 = atomic.U64.new(0);\n\
         fn bump() -> bool { return HITS.fetch_add(1) >= 0; }\n\
         fn via(x: u32) -> bool { return x > 0 && bump(); }\n\
         fn tick(c: &atomic.U64) requires(c.fetch_add(1u64) >= 0u64) { }\n\
         fn each(f: fn(u32) -> bool, x: u32) requires(f(x)) { }\n\
         fn any<T>(xs: &[T]) requires(xs.len() > 0) { }\n\
         fn g(x: u32) requires(via(x)) { }\n\
         fn get(v: &mut [u32]) -> &mut u32 from v ensures(v.len() > 0) { return &mut v[0]; }\n\
         fn main() { }\n";
    let (p, tc) = mem(src);
    let show = |ds: &[stave_diag::Diagnostic]| -> Vec<String> {
        ds.iter()
            .map(|d| {
                let line = d.primary().map_or(0, |s| p.source_map.locate(s).map_or(0, |l| l.line));
                format!("{line}: {} {}", d.code(), d.message())
            })
            .collect()
    };
    let want = [
        "5: E0211 `c` cannot be used in a contract: `&U64` is not `Frozen`: component `*.v` has type `UnsafeCell<u64>`, `UnsafeCell` (interior mutability)",
        "6: E0211 `f` cannot be used in a contract: `fn(u32) -> bool` is not `Frozen`: it is a fn value (its captures are not in its type)",
        "7: E0211 `xs` cannot be used in a contract: `&[T]` is not `Frozen`: component `*[_]` has type `T`, a generic parameter without a `Frozen` bound",
        "9: E0211 `ensures` cannot read `v`: `result` holds a `&mut` borrowed from it",
    ];
    assert_eq!(show(&tc.all_diagnostics(&p)), want);
    let want = [
        "8: E0211 `main.via` cannot be called in a contract: it reaches `main.bump`, which uses `main.HITS` of type `U64`, not `Frozen`",
    ];
    assert_eq!(show(&stave_sema::contracts::shared_statics(&p, &tc)), want);
    assert_eq!(stave_sema::diagnostic_build_failures(), 0);
}

/// A type past the type height limit is never created: E0005 case `type` where it is written,
/// and no `Flat` check runs on it.
#[test]
fn flat_depth_cutoff_fails_closed() {
    let nest = |n: usize| format!("{}u32{}", "W<".repeat(n), ">".repeat(n));
    let src = format!(
        "struct W<T> {{ pub v: T }}\nfn need<T: Flat>() {{}}\n\
         fn main() -> i32 {{ need::<{}>(); need::<{}>(); return 0; }}\n",
        nest(999),
        nest(1100)
    );
    let (p, tc) = mem(&src);
    let mut got: Vec<String> =
        tc.all_diagnostics(&p).iter().map(|d| format!("{} {}", d.code(), d.message())).collect();
    got.sort();
    assert_eq!(got, vec!["E0005 type nested deeper than 1000 levels".to_string()]);
    assert_eq!(stave_sema::diagnostic_build_failures(), 0);
}

/// One shared type depth limit (E0005 case `type`): 1,000 levels pass, 1,001 do not, written,
/// through references, by fields, or built by an instance; a pointer to another type adds nothing.
#[test]
fn type_depth_limit() {
    let w = |n: usize, leaf: &str| format!("{}{leaf}{}", "W<".repeat(n), ">".repeat(n));
    let codes = |body: &str| -> Vec<String> {
        let src = format!("struct W<T> {{ pub v: T }}\n{body}\nfn main() -> i32 {{ return 0; }}\n");
        let (p, tc) = mem(&src);
        tc.all_diagnostics(&p).iter().map(|d| d.code().to_string()).collect()
    };
    let none: Vec<String> = Vec::new();
    let one = vec!["E0005".to_string()];
    assert_eq!(codes(&format!("fn f(x: {}) {{}}", w(999, "u32"))), none);
    assert_eq!(codes(&format!("fn f(x: {}) {{}}", w(1000, "u32"))), one);
    assert_eq!(codes(&format!("fn f(x: {}u32) {{}}", "&".repeat(999))), none);
    assert_eq!(codes(&format!("fn f(x: {}u32) {{}}", "&".repeat(1000))), one);
    assert_eq!(codes(&format!("fn f(x: &{}) {{}}", w(998, "u32"))), none);
    assert_eq!(codes(&format!("fn f(x: &{}) {{}}", w(999, "u32"))), one);
    // fields: `S0` holds `S1` ... holds a `u32`
    let chain = |n: usize, ptr: bool| -> String {
        let mut s = String::new();
        for i in 0..n {
            let inner = if i + 1 == n { "u32".to_string() } else { format!("S{}", i + 1) };
            let inner = if ptr && i + 1 < n { format!("*const {inner}") } else { inner };
            s += &format!("struct S{i} {{ pub x: {inner} }}\n");
        }
        s
    };
    assert_eq!(codes(&chain(999, false)), none);
    assert_eq!(codes(&chain(1000, false)), one);
    assert_eq!(codes(&chain(1100, true)), none);
    // an instance: `hold::<W^999<u32>>` makes a `W^1000<u32>`
    let hold = |n: usize| {
        format!("fn hold<T>(x: T) {{ let y: W<T> = W.{{ v: x }}; }}\nfn g(a: {}) {{ hold(a); }}", w(n, "u32"))
    };
    assert_eq!(codes(&hold(998)), none);
    assert_eq!(codes(&hold(999)), one);
    assert_eq!(stave_sema::diagnostic_build_failures(), 0);
}

/// The E0908 fixes are code that compiles once the body is filled in: `impls-static` writes the
/// bound with one impl's trait args, the method's params and result, and the call with its args;
/// `for` names the trait as the module does.
#[test]
fn e0908_fixes_write_the_signature_and_the_trait() {
    let src = "struct A { n: u32 }\n\
         trait Mk<T> { fn mk(x: u32, k: u8) -> T; }\n\
         impl Mk<u32> for A { fn mk(x: u32, k: u8) -> u32 { return x; } }\n\
         impl Mk<u64> for A { fn mk(x: u32, k: u8) -> u64 { return 0; } }\n\
         trait Z<T> { fn z(); }\n\
         impl Z<u8> for A { fn z() { } }\n\
         impl Z<u16> for A { fn z() { } }\n\
         trait Next { fn next(&mut self) -> Option<u64>; }\n\
         trait Back { fn next(&mut self) -> Option<u64>; }\n\
         fn g<T: Next + Back>(t: T) { for x in t { } }\n\
         fn main() -> i32 { let m: u32 = A.mk(1u32, 2u8); A.z(); return 0; }\n";
    let (_, tc) = mem(src);
    let fixes: Vec<(&str, &str)> =
        tc.diagnostics.iter().map(|d| (d.code(), d.fix().map_or("", |f| f.description.as_str()))).collect();
    let has = |needle: &str| fixes.iter().any(|(c, f)| *c == "E0908" && f.contains(needle));
    assert!(
        has(
            "`fn pick<X: Mk<u32>>(x: u32, k: u8) -> u32 { return X.mk(x, k); }`, called as `pick::<A>(x, k)`"
        ),
        "{fixes:?}"
    );
    assert!(has("`fn pick<X: Z<u8>>() { return X.z(); }`, called as `pick::<A>()`"), "{fixes:?}");
    assert!(has("`let mut it = t; loop { match Next.next(&mut it) {"), "{fixes:?}");
    assert_eq!(fixes.iter().filter(|(c, _)| *c == "E0908").count(), 3, "{fixes:?}");
    assert!(fixes.iter().all(|(_, f)| !f.contains("<..>")), "{fixes:?}");
}

/// A field named twice in a struct pattern is E0105 `pattern-field`: no field of a pattern is
/// renamed, so the text speaks of naming the field once.
#[test]
fn e0105_in_a_pattern_says_name_the_field_once() {
    let src = "struct H { n: u32, d: u32 }\n\
         fn main() -> i32 { let h = H.{ n: 1u32, d: 2u32 }; let H.{ n, n: m, d: _ } = h; return 0; }\n";
    let (_, tc) = mem(src);
    let got: Vec<(&str, &str)> = tc
        .diagnostics
        .iter()
        .filter(|d| d.code() == "E0105")
        .map(|d| (d.message(), d.fix().map_or("", |f| f.description.as_str())))
        .collect();
    assert_eq!(got.len(), 1, "{got:?}");
    assert_eq!(got[0].0, "field `n` of `main.H` is named twice in one pattern");
    assert!(got[0].1.starts_with("name `n` once") && !got[0].1.contains("rename"), "{got:?}");
}

/// Polymorphic recursion at `(T, T)` doubles the type argument per level (2^k leaves, k + 1
/// distinct types): it reaches E0902 as promptly as linear growth, as the type walks visit each
/// distinct type once. The bound is generous for a loaded CI host; a tree walk needs 2^64 steps.
#[test]
fn doubling_instance_chain_is_e0902_promptly() {
    let src = "rec fn f<T: Copy>(x: T, n: u32) -> u32 {\n\
         if n == 0 { return 0; }\n\
         return f::<(T, T)>((x, x), n - 1);\n}\n\
         fn main() -> i32 { let a = f(1u32, 3); return 0; }\n";
    let start = std::time::Instant::now();
    let (_, tc) = mem(src);
    let codes: Vec<&str> = tc.diagnostics.iter().map(|d| d.code()).collect();
    assert_eq!(codes, ["E0902"], "{codes:?}");
    assert!(start.elapsed() < std::time::Duration::from_secs(60), "{:?}", start.elapsed());
}

/// The (code, message, fix) of every error of `src`.
fn texts(src: &str) -> Vec<(&'static str, String, String)> {
    let (_, tc) = mem(src);
    tc.diagnostics
        .iter()
        .filter(|d| d.severity() == stave_diag::Severity::Error)
        .map(|d| {
            (d.code(), d.message().to_string(), d.fix().map_or(String::new(), |f| f.description.clone()))
        })
        .collect()
}

/// Each fix can be followed: the `&str` pattern fix compares bytes (a `&str` has no `==`), a
/// `let` pattern and a match ending in `_` are not told to add a `_` arm, a generic `size_of`
/// length says why it has no value, an empty array has no index range, and the shift range is
/// a number.
#[test]
fn fixes_that_apply() {
    let t = texts(
        "fn f(s: &str) -> u32 { match s { \"a\" => { return 1; } _ => { return 2; } } }\nfn main() -> i32 { return f(\"a\") as% i32; }\n",
    );
    assert_eq!(t.len(), 1, "{t:?}");
    assert!(t[0].0 == "E0206" && t[0].2.contains("mem.str_bytes(s).eq(mem.str_bytes(\"a\"))"), "{t:?}");
    let t = texts(
        "fn f(o: Option<u32>) -> u32 { let Option.Some(x) = o; return x; }\nfn main() -> i32 { return f(Option.None) as% i32; }\n",
    );
    assert_eq!(t.len(), 1, "{t:?}");
    assert!(t[0].0 == "E0300" && !t[0].2.contains("`_` arm"), "{t:?}");
    let t = texts(
        "fn f<T>() -> usize { let a = [0u8; size_of::<T>()]; return a.len(); }\nfn main() -> i32 { return f::<u32>() as% i32; }\n",
    );
    assert_eq!(t.len(), 1, "{t:?}");
    assert!(t[0].0 == "E0904" && t[0].1.contains("type parameter") && !t[0].1.contains("outside"), "{t:?}");
    let t = texts("const E: [u8; 0] = [];\nconst X: u8 = E[0];\nfn main() -> i32 { return X as i32; }\n");
    assert_eq!(t.len(), 1, "{t:?}");
    assert!(t[0].0 == "E0203" && !t[0].2.contains("-1"), "{t:?}");
    let t = texts("const K: i32 = -1;\nconst S: i32 = 1i32 >> K;\nfn main() -> i32 { return S; }\n");
    assert_eq!(t.len(), 1, "{t:?}");
    assert!(t[0].0 == "E0205" && t[0].2.contains("by 0 to 31,"), "{t:?}");
}

/// A field given twice in a struct literal is E0105 `literal-field`: a literal renames nothing,
/// so the text says to remove one of the two values.
#[test]
fn e0105_in_a_literal_says_remove_one_value() {
    let src = "struct H { n: u32, d: u32 }\n\
         fn main() -> i32 { let h = H.{ n: 1u32, n: 3u32, d: 2u32 }; return h.d as% i32; }\n";
    let (_, tc) = mem(src);
    let got: Vec<(&str, &str)> = tc
        .diagnostics
        .iter()
        .filter(|d| d.code() == "E0105")
        .map(|d| (d.message(), d.fix().map_or("", |f| f.description.as_str())))
        .collect();
    assert_eq!(got.len(), 1, "{got:?}");
    assert_eq!(got[0].0, "field `n` of `main.H` is given twice in one struct literal");
    assert!(got[0].1.starts_with("give `n` once") && !got[0].1.contains("rename"), "{got:?}");
}

/// The E0908 fixes name each trait instance as the calling module does: a trait of this module
/// by its name, a generic one with its args (`Nx::<u64>`), a trait of an imported module through
/// the import (`geo.X1`), one of a module not imported with the `import` to add; `impls-static`
/// writes the instance's types, not the impl's params. Of two `next`s the `for` fix names the
/// one with the iterator shape. `main.X` would name `fn main`, so no fix starts with it.
#[test]
fn e0908_fixes_name_instances_that_resolve() {
    let geo = "pub struct It { pub i: u64 }\n\
         pub struct Jt { pub i: u64 }\n\
         pub trait X1 { fn m(&self) -> u64; }\n\
         pub trait X2 { fn m(&self) -> u64; }\n\
         impl X1 for It { fn m(&self) -> u64 { return 1; } }\n\
         impl X2 for It { fn m(&self) -> u64 { return 2; } }\n";
    let tr = "import geo;\n\
         pub trait Z1 { fn n(&self) -> u64; }\n\
         pub trait Z2 { fn n(&self) -> u64; }\n\
         impl Z1 for geo.Jt { fn n(&self) -> u64 { return 3; } }\n\
         impl Z2 for geo.Jt { fn n(&self) -> u64 { return 4; } }\n";
    let src = "import geo;\n\
         struct It { i: u64 }\n\
         struct W<T> { v: T }\n\
         trait X1 { fn m(&self) -> u64; }\n\
         trait X2 { fn m(&self) -> u64; }\n\
         impl X1 for It { fn m(&self) -> u64 { return 1; } }\n\
         impl X2 for It { fn m(&self) -> u64 { return 2; } }\n\
         trait Nx<X> { fn next(&mut self) -> Option<X>; }\n\
         trait Back { fn next(&mut self) -> Option<u64>; }\n\
         trait GNext { fn next<U>(&mut self) -> Option<u64>; }\n\
         impl Nx<u8> for W<u8> { fn next(&mut self) -> Option<u8> { return Option.None; } }\n\
         impl Nx<u16> for W<u8> { fn next(&mut self) -> Option<u16> { return Option.None; } }\n\
         trait Mk<T> { fn mk(x: T) -> T; }\n\
         impl<T: Copy> Mk<[T; 1]> for W<T> { fn mk(x: [T; 1]) -> [T; 1] { return x; } }\n\
         impl Mk<u64> for W<u64> { fn mk(x: u64) -> u64 { return x; } }\n\
         fn f<T: Nx<u64> + Back>(t: T) { for x in t { } }\n\
         fn g<T: GNext + Back>(t: T) { for x in t { } }\n\
         fn h<T: Nx<u8> + Nx<u16>>(t: T) { for x in t { } }\n\
         fn main() -> i32 {\n\
           let it = It.{ i: 0 }; let a = it.m();\n\
           let mut w = W::<u8>.{ v: 0 }; let b = w.next();\n\
           let c: u64 = W::<u64>.mk(1u64);\n\
           let gi = geo.It.{ i: 0 }; let d = gi.m();\n\
           let gj = geo.Jt.{ i: 0 }; let e = gj.n();\n\
           return 0; }\n";
    let (_, tc) = mem_with(src, &[("/m/src/geo.stv", geo), ("/m/src/tr.stv", tr)]);
    let fixes: Vec<&str> = tc
        .diagnostics
        .iter()
        .filter(|d| d.code() == "E0908")
        .map(|d| d.fix().map_or("", |f| f.description.as_str()))
        .collect();
    let has = |needle: &str| fixes.iter().any(|f| f.contains(needle));
    assert!(has("`X1.m(x)` or `X2.m(x)`"), "{fixes:?}");
    assert!(has("`Nx::<u8>.next(x)` or `Nx::<u16>.next(x)`"), "{fixes:?}");
    assert!(has("`fn pick<X: Mk<[u64; 1]>>(x: [u64; 1]) -> [u64; 1] { return X.mk(x); }`"), "{fixes:?}");
    assert!(has("match Nx::<u64>.next(&mut it)"), "{fixes:?}");
    assert!(has("match Back.next(&mut it)"), "{fixes:?}");
    // two instances of one trait under a loop: the loop-shaped fix too
    assert!(has("let mut it = t; loop { match Nx::<u8>.next(&mut it)"), "{fixes:?}");
    assert!(has("`geo.X1.m(x)` or `geo.X2.m(x)`"), "{fixes:?}");
    assert!(has("`tr.Z1.n(x)`") && has("after adding `import tr;` to the imports"), "{fixes:?}");
    assert_eq!(fixes.len(), 8, "{fixes:?}");
    assert!(
        fixes.iter().all(|f| !f.contains("main.") && !f.contains("T0") && !f.contains("<..>")),
        "{fixes:?}"
    );
}

/// E0219 `next-shape` says what keeps a `next` from being an iterator's, at that `next`; E0609
/// `lane-type` names the lane type no vector has.
#[test]
fn e0219_next_shape_and_e0609_lane_type_say_why() {
    let src = "struct It { i: u64 }\n\
         struct G { i: u64 }\n\
         struct P { a: u32 }\n\
         impl It { fn next(&mut self, k: u64) -> Option<u64> { return Option.None; } }\n\
         impl G { fn next<U>(&mut self) -> Option<u64> { return Option.None; } }\n\
         fn v(x: vec<P, 4>) { }\n\
         fn main() -> i32 { for x in It.{ i: 0 } { } for y in G.{ i: 0 } { } return 0; }\n";
    let (_, tc) = mem(src);
    let got: Vec<(&str, &str, Vec<&str>)> = tc
        .diagnostics
        .iter()
        .map(|d| (d.code(), d.message(), d.related().iter().map(|r| r.label.as_str()).collect()))
        .collect();
    let shape = |why: &str| {
        got.iter().any(|(c, m, l)| {
            *c == "E0219"
                && m.ends_with("its `next` is not `fn next(&mut self) -> Option<T>`")
                && l.contains(&why)
        })
    };
    assert!(shape("it takes params besides `&mut self`"), "{got:?}");
    assert!(shape("it has generic params of its own"), "{got:?}");
    let lane = "`vec<P, 4>` has the lane type `P`, which no vector has";
    assert!(got.iter().any(|(c, m, _)| *c == "E0609" && *m == lane), "{got:?}");
    assert_eq!(got.len(), 3, "{got:?}");
}

/// E0704 case `value` names the missing features no other missing one implies, as for a direct
/// call: `avx2` (not the baseline's `sse42` or the implied `avx`), and both of two.
#[test]
fn e0704_value_names_the_missing_features() {
    let src = "fn one(f: fn(u32) -> u32 needs(avx2)) -> u32 { return f(1u32); }\n\
         fn two(f: fn(u32) -> u32 needs(avx2, fma)) -> u32 needs(avx) { return f(1u32); }\n\
         fn main() -> i32 { return 0; }\n";
    let (_, tc) = mem(src);
    let got: Vec<&str> = tc.diagnostics.iter().filter(|d| d.code() == "E0704").map(|d| d.message()).collect();
    let want = [
        "a call through a value of type `fn(u32) -> u32 needs(avx2)` needs `avx2`, which the caller does not have",
        "a call through a value of type `fn(u32) -> u32 needs(avx2, fma)` needs `avx2, fma`, which the caller \
         does not have",
    ];
    assert_eq!(got, want, "{got:?}");
}

/// E0709 case `test` and E0705 case `dispatch` say what the rule is and how to write the code.
#[test]
fn e0709_test_and_e0705_dispatch_texts() {
    let src = "fn base(x: u32) -> u32 { return x; }\n\
         dispatch fn inner(x: u32) -> u32 { _ => base, }\n\
         dispatch fn outer(x: u32) -> u32 { _ => inner, }\n\
         test fn t() needs(avx2) { }\n\
         fn main() -> i32 { return 0; }\n";
    let (_, tc) = mem(src);
    let got: Vec<(&str, &str, &str)> = tc
        .diagnostics
        .iter()
        .map(|d| (d.code(), d.message(), d.fix().map_or("", |f| f.description.as_str())))
        .collect();
    assert_eq!(got.len(), 2, "{got:?}");
    assert!(
        got.iter().any(|(c, m, f)| *c == "E0705"
            && *m == "dispatch arm names `dispatch fn` `main.inner`: an arm target is a fn with a body"
            && f.starts_with("name the arms of `main.inner` in this dispatch directly")),
        "{got:?}"
    );
    assert!(
        got.iter().any(|(c, m, f)| *c == "E0709"
            && *m == "a `test fn` cannot declare `needs`"
            && f.contains("through a `dispatch` fn whose `_` arm")),
        "{got:?}"
    );
}

/// `TxN.load_first_fault` is reserved (E0208 `reserved`): the fix writes the masked load of the
/// lanes in bounds with the vector's own mask type, also inside `emulate { }`.
#[test]
fn e0208_reserved_first_fault_load_names_the_masked_load() {
    let src = "fn f(s: &[f32]) -> f32x4 { return f32x4.load_first_fault(s, 0); }\n\
         fn g(s: &[u16]) { emulate { let v = u16x8.load_first_fault(s, 1); } }\n\
         fn main() -> i32 { return 0; }\n";
    let (_, tc) = mem(src);
    let got: Vec<(&str, &str, &str)> = tc
        .diagnostics
        .iter()
        .map(|d| (d.code(), d.message(), d.fix().map_or("", |f| f.description.as_str())))
        .collect();
    let has = |ty: &str, mask: &str| {
        got.iter().any(|(c, m, f)| {
            *c == "E0208"
                && *m
                    == format!("`{ty}.load_first_fault` is reserved: first-fault loads are not available yet")
                && f.contains(&format!("`{ty}.load_masked(s, i, {mask}.first_n(s.len() -| i))`"))
        })
    };
    assert!(has("f32x4", "mask32x4"), "{got:?}");
    assert!(has("u16x8", "mask16x8"), "{got:?}");
    assert_eq!(got.len(), 2, "{got:?}");
}

/// E0201 for value arms of fn types that differ only in `needs` / `without` names a change that
/// compiles, never an `as` (E0204 for fn types): with no wanted type the fn type both arms
/// convert to; with one, the arm not usable as it, converted alone first (in a `match`, also an
/// earlier arm).
#[test]
fn e0201_fn_arms_fixes_name_what_compiles() {
    let src = "fn twice(x: u64) -> u64 { return x * 2u64; }\n\
         fn stop(x: u64) -> u64 without(io) { return x; }\n\
         fn wide(x: u64) -> u64 needs(avx2) { return x; }\n\
         fn apply(f: fn(u64) -> u64 without(io), x: u64) -> u64 without(io) { return f(x); }\n\
         fn main() -> i32 {\n\
         let c = true;\n\
         let k = 1u32;\n\
         let a = if c then stop else twice;\n\
         let b = if c then wide else twice;\n\
         let r = apply(if c then twice else stop, 3u64);\n\
         let s = apply(match k { 0u32 => twice, _ => stop }, 3u64);\n\
         return 0; }\n";
    let (_, tc) = mem(src);
    let got: Vec<(&str, &str)> =
        tc.diagnostics.iter().map(|d| (d.code(), d.fix().map_or("", |f| f.description.as_str()))).collect();
    let has = |needle: &str| got.iter().filter(|(c, f)| *c == "E0201" && f.contains(needle)).count();
    assert_eq!(has("`let f: fn(u64) -> u64 = ..;`"), 1, "{got:?}");
    assert_eq!(has("`let f: fn(u64) -> u64 needs(avx2) = ..;`"), 1, "{got:?}");
    assert_eq!(has("`let g: fn(u64) -> u64 without(io) = twice;`"), 2, "{got:?}");
    assert!(got.iter().all(|(_, f)| !f.contains(" as fn")), "{got:?}");
    assert_eq!(got.len(), 4, "{got:?}");
}

/// With a wanted fn type, an arm needing features neither that type nor the enclosing fn has is
/// E0707, as the same arm converted alone (converting it first compiles no better); inside a fn
/// with those features the arms convert.
#[test]
fn e0707_fn_arm_needing_features_the_wanted_type_lacks() {
    let src = "fn plain(x: u64) -> u64 without(trap) { return x; }\n\
         fn wide(x: u64) -> u64 needs(avx2) { return x; }\n\
         fn pick(k: u32) -> u64 needs(avx2) { let f: fn(u64) -> u64 = match k { 0u32 => plain, _ => wide }; return f(1u64); }\n\
         fn main() -> i32 {\n\
         let k = 1u32;\n\
         let f: fn(u64) -> u64 = match k { 0u32 => plain, _ => wide };\n\
         let g: fn(u64) -> u64 = if k == 0u32 then wide else plain;\n\
         return 0; }\n";
    let (_, tc) = mem(src);
    let got: Vec<(&str, &str)> = tc.diagnostics.iter().map(|d| (d.code(), d.message())).collect();
    let e0707 = "`fn(u64) -> u64 needs(avx2)` cannot be used as `fn(u64) -> u64`";
    assert_eq!(got.iter().filter(|(c, m)| *c == "E0707" && m.contains(e0707)).count(), 2, "{got:?}");
    assert_eq!(got.len(), 2, "{got:?}");
}

/// E0201 offers plain `as` only where the cast table admits it; otherwise the fix names the form
/// that gives the expected type (a marked cast, `Option.Some` / `Result.Ok`, taking the payload
/// out, a borrow, a copy through `*`), and with none of them no conversion at all.
#[test]
fn e0201_fixes_name_a_conversion_that_exists() {
    let src = "import std.vec;\n\
         fn some(v: &vec.Vec<u32>) -> Option<&u32> from v { return v.get_ref(0); }\n\
         fn lit() -> Option<u32> { return 5; }\n\
         fn ok(x: u32) -> Result<u32, u8> { return x; }\n\
         fn un(x: Option<u32>) -> u32 { return x; }\n\
         fn unr(x: Result<u32, u8>) -> u32 { return x; }\n\
         fn wide(x: u32) -> u64 { return x; }\n\
         fn narrow(x: u64) -> u32 { return x; }\n\
         fn sat(x: f64) -> i32 { return x; }\n\
         fn round(x: u32) -> f32 { return x; }\n\
         fn take(xs: &[u32]) -> usize { return xs.len(); }\n\
         fn bump(x: &mut u32) { *x = *x + 1u32; }\n\
         fn read(x: &u32) -> u32 { return x; }\n\
         fn none(x: u32) -> bool { return x; }\n\
         fn float() -> f64 { return 2; }\n\
         fn main() -> i32 {\n\
         let a: [u32; 2] = [1u32; 2];\n\
         let n = take(a);\n\
         let mut k: u32 = 0;\n\
         bump(k);\n\
         return 0; }\n";
    let (_, tc) = mem(src);
    let got: Vec<(&str, &str)> =
        tc.diagnostics.iter().map(|d| (d.code(), d.fix().map_or("", |f| f.description.as_str()))).collect();
    let has = |needle: &str| got.iter().filter(|(c, f)| *c == "E0201" && f.contains(needle)).count();
    assert_eq!(has("`Option.Some(v.get_ref(0))`"), 1, "{got:?}");
    assert_eq!(has("`Option.Some(5)`"), 1, "{got:?}");
    assert_eq!(has("`Result.Ok(x)`"), 1, "{got:?}");
    assert_eq!(has("`x.unwrap()` (panics `Unwrap` on `Option.None`)"), 1, "{got:?}");
    assert_eq!(has("`x.unwrap()` (panics `Unwrap` on `Result.Err`)"), 1, "{got:?}");
    assert_eq!(has("`x as u64`"), 1, "{got:?}");
    assert_eq!(has("`x as% u32`"), 1, "{got:?}");
    assert_eq!(has("`x as| i32`"), 1, "{got:?}");
    assert_eq!(has("`x as~ f32`"), 1, "{got:?}");
    assert_eq!(has("borrow it: `&a`"), 1, "{got:?}");
    assert_eq!(has("borrow it mutably: `&mut k`"), 1, "{got:?}");
    assert_eq!(has("copy the value out: `*x`"), 1, "{got:?}");
    assert_eq!(has("write a value of type `bool` here"), 1, "{got:?}");
    assert_eq!(has("write a value of type `f64` here"), 1, "{got:?}");
    assert!(got.iter().all(|(_, f)| !f.contains(" as Option") && !f.contains(" as Result")), "{got:?}");
    assert!(got.iter().all(|(_, f)| !f.contains("`x as bool`") && !f.contains("`2 as f64`")), "{got:?}");
    assert_eq!(got.len(), 14, "{got:?}");
}

/// E0206 for unary `*` on a value that is no reference names what the type offers: its first
/// `&self -> &U` and `&mut self -> &mut U` methods (a lock guard's `get` / `get_mut`), one of
/// them, or else the value itself; an operator on a reference says to apply it to the value
/// behind it.
#[test]
fn e0206_deref_names_the_reference_accessors() {
    let src = "import std.sync;\n\
         struct R { v: u64 }\n\
         impl R { pub fn get(&self) -> &u64 from self { return &self.v; } }\n\
         struct W { v: u64 }\n\
         impl W { pub fn put(&mut self, x: u64) { self.v = x; } pub fn slot(&mut self) -> &mut u64 from self { return &mut self.v; } }\n\
         fn guard(m: &sync.Mutex<u64, 1>) -> u64 { let g = m.lock(); return *g; }\n\
         fn read(r: R) -> u64 { return *r; }\n\
         fn write(w: W) -> u64 { return *w; }\n\
         fn plain(x: u64) -> u64 { return *x; }\n\
         fn refs(a: &u64, b: &u64) -> u64 { return a + b; }\n\
         fn main() -> i32 { return 0; }\n";
    let (_, tc) = mem(src);
    let got: Vec<(&str, &str, &str)> = tc
        .diagnostics
        .iter()
        .map(|d| (d.code(), d.message(), d.fix().map_or("", |f| f.description.as_str())))
        .collect();
    let has = |needle: &str| got.iter().filter(|(c, _, f)| *c == "E0206" && f.contains(needle)).count();
    assert_eq!(has("read `*g.get()`; to change the value, `let r = g.get_mut();`"), 1, "{got:?}");
    assert_eq!(has("`R` gives a `&` from its method `get`: read `*r.get()`"), 1, "{got:?}");
    assert_eq!(has("`W` gives a `&mut` from its method `slot`: `let r = w.slot();`"), 1, "{got:?}");
    assert_eq!(has("`x` is a value already: use it without `*`"), 1, "{got:?}");
    assert_eq!(has("`&u64` is a reference: apply `+` to the value behind it"), 1, "{got:?}");
    let deref = "`*` cannot dereference `MutexGuard<u64, 1>`: it is no reference or raw pointer";
    assert!(got.iter().any(|(_, m, _)| *m == deref), "{got:?}");
    assert_eq!(got.len(), 5, "{got:?}");
}

/// Trait queries are answered once per (type, trait) and checking run: along a chain of 20,000
/// structs linked by raw pointers, a `Frozen` query at the head (failing at the far end, or
/// holding) and a `Copy` query over 20,000 structs with impls of their own take time linear in
/// the chain. Answers found inside a cycle are kept only once the cycle's goal is settled: a
/// mutual cycle is Sync when neither holds a `Cell` and is not when one does, whichever type is
/// asked first.
#[test]
fn trait_queries_are_answered_once_per_type() {
    let codes = |src: &str| -> Vec<String> {
        let (p, tc) = mem(src);
        let mut c: Vec<String> = tc.all_diagnostics(&p).iter().map(|d| d.code().to_string()).collect();
        c.sort();
        c
    };
    let chain = |n: usize, last: &str, copy: bool| -> String {
        let mut s = String::from("import std.cell;\n");
        for k in 1..n {
            s += &format!("struct S{k} {{ pub p: *const S{}, pub b: u8 }}\n", k + 1);
            if copy {
                s += &format!("impl Copy for S{k} {{}}\n");
            }
        }
        s += &format!("struct S{n} {{ pub {last} }}\n");
        s
    };
    let none: Vec<String> = Vec::new();
    let frozen = "fn need<T: Frozen>(t: &T) -> u8 { return 1u8; }\n\
                  fn f(p: &S1) -> u8 { return need(p); }\nfn main() -> i32 { return 0; }\n";
    assert_eq!(codes(&(chain(20_000, "v: u32", false) + frozen)), none);
    assert_eq!(codes(&(chain(20_000, "c: cell.Cell<u32>", false) + frozen)), vec!["E1410".to_string()]);
    let copy = "fn need<T: Copy>(t: &T) -> u8 { return 1u8; }\n\
                fn f(p: &S1) -> u8 { return need(p); }\nfn main() -> i32 { return 0; }\n";
    assert_eq!(codes(&(chain(20_000, "v: u32", true) + copy)), none);
    let mutual = |cell: &str, a_first: bool| -> String {
        let calls = if a_first { "need(&a) + need(&v)" } else { "need(&v) + need(&a)" };
        format!(
            "import std.vec;\nimport std.cell;\nstruct A {{ pub bs: vec.Vec<B> }}\n\
             struct B {{ pub as_: vec.Vec<A>{cell} }}\nfn need<T: Sync>(t: &T) -> u8 {{ return 1u8; }}\n\
             fn main() -> i32 {{\n    let a = A.{{ bs: vec.Vec::<B>.new() }};\n    \
             let v = vec.Vec::<A>.new();\n    return (({calls}) as u8) as i32;\n}}\n"
        )
    };
    let two = vec!["E0901".to_string(), "E0901".to_string()];
    assert_eq!(codes(&mutual(", pub c: cell.Cell<u32>", true)), two);
    assert_eq!(codes(&mutual(", pub c: cell.Cell<u32>", false)), two);
    assert_eq!(codes(&mutual("", true)), none);
    assert_eq!(codes(&mutual("", false)), none);
    assert_eq!(stave_sema::diagnostic_build_failures(), 0);
}
