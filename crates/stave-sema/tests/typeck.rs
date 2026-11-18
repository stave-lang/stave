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
        world.find_impl(&mut t, wu8, &TraitRef { trait_decl: tr, args: Vec::new() }, &[], 0).expect("impl");
    assert_eq!(p.decl(imp).map(|d| d.kind), Some(stave_sema::DeclKind::Impl));
    assert_eq!(iargs, vec![GArg::Type(u8t)]);
    let imp_def = tc.sigs.impl_def(imp).expect("impl sig");
    assert_eq!(world.impl_applies(&mut t, imp_def, wu8, &[], 0), Some(vec![GArg::Type(u8t)]));
    assert_eq!(world.impl_applies(&mut t, imp_def, u8t, &[], 0), None);
    // layout: Option<u8> (tag + payload) then u8
    let lx = stave_sema::traits::layout_of(&mut t, &tc.sigs, sx, 0).expect("layout");
    let lw = stave_sema::traits::layout_of(&mut t, &tc.sigs, wu8, 0).expect("layout");
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

/// Past its depth cutoff the Flat check fails closed (E1407 case `depth`).
#[test]
fn flat_depth_cutoff_fails_closed() {
    let nest = |n: usize| format!("{}u32{}", "W<".repeat(n), ">".repeat(n));
    let src = format!(
        "struct W<T> {{ pub v: T }}\nfn need<T: Flat>() {{}}\n\
         fn main() -> i32 {{ need::<{}>(); need::<{}>(); return 0; }}\n",
        nest(1000),
        nest(1100)
    );
    let (p, tc) = mem(&src);
    let got: Vec<String> =
        tc.all_diagnostics(&p).iter().map(|d| format!("{} {}", d.code(), d.message())).collect();
    // the type text is elided in the middle (`W<W<..>>`)
    assert_eq!(got.len(), 1, "{got:?}");
    assert!(got[0].starts_with("E1407 `W<W<"), "{}", got[0]);
    let tail = ">>` is not `Flat`: it nests deeper than 1024 levels, the limit of the `Flat` check";
    assert!(got[0].ends_with(tail), "{}", got[0]);
    assert_eq!(stave_sema::diagnostic_build_failures(), 0);
}
