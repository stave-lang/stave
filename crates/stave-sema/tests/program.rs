//! Program-level tests: canonical ids under shuffled internal order, generated corpora
//! resolve with no diagnostics, resolution facts on small programs, dump shapes.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use stave_sema::{BuiltinPackage, Config, DeclKind, Fs, Program, Res};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture_std() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/std")
}

/// `strings` dump (stage-gates): `S hash32 "escaped"`.
fn strings_dump(p: &Program) -> String {
    let mut s = String::new();
    for (id, text) in p.strings.iter() {
        let mut esc = String::new();
        for b in text.bytes() {
            match b {
                b'\\' => esc.push_str("\\\\"),
                b'"' => esc.push_str("\\\""),
                b'\n' => esc.push_str("\\n"),
                b'\t' => esc.push_str("\\t"),
                0x20..=0x7e => esc.push(b as char),
                _ => esc.push_str(&format!("\\x{b:02x}")),
            }
        }
        s += &format!("{} {:08x} \"{esc}\"\n", id.0, p.strings.hash(id));
    }
    s
}

/// `names` dump (stage-gates): `I N -> decl D` / `I N -> local I N2`, by (I, N, offset).
fn names_dump(p: &Program) -> String {
    let mut s = String::new();
    for (i, r) in p.res.iter().enumerate() {
        for u in &r.uses {
            match u.res {
                Res::Decl(d) => s += &format!("{i} {} -> decl {}\n", u.node, d.0),
                Res::Local(b) => s += &format!("{i} {} -> local {i} {}\n", u.node, b.node),
                Res::Generic(g) => s += &format!("{i} {} -> local {i} {}\n", u.node, g.node),
                _ => {}
            }
        }
    }
    s
}

/// Everything id-dependent the API exposes, as text.
fn snapshot(p: &Program) -> String {
    let mut s = strings_dump(p);
    for (k, d) in p.decls.iter().enumerate() {
        s += &format!(
            "D{k} {:?} {} m{} p{} i{} n{} x{} f{}\n",
            d.kind,
            p.strings.get(d.name),
            d.module.0,
            d.parent.0,
            d.item.0,
            d.node,
            d.index,
            d.flags.0
        );
    }
    for m in &p.modules {
        s += &format!("M{} {} d{} f{}\n", m.id.0, m.path_text, m.decl.0, m.file.0);
        for i in &m.imports {
            s += &format!("  import {} -> {}\n", p.strings.get(i.bound), i.target.0);
        }
    }
    for i in &p.impls {
        s += &format!("impl d{} {:?} t{} {:?}\n", i.decl.0, i.head, i.trait_decl.0, i.methods);
    }
    for (k, it) in p.items.iter().enumerate() {
        s += &format!("I{k} f{} {} d{}\n", it.file.0, it.index, it.decl.0);
    }
    s += &names_dump(p);
    for r in &p.res {
        for d in &r.deferred {
            s += &format!("defer {} {} {:?} {:?}\n", d.node, d.offset, d.kind, d.base);
        }
        for b in &r.bindings {
            s += &format!("bind {:?} {} {:?}\n", b.at, p.strings.get(b.name), b.kind);
        }
    }
    for d in &p.diagnostics {
        s += &format!("{} {:?} {}\n", d.code(), d.primary(), d.message());
    }
    s
}

fn load(root: &Path, seed: Option<u64>) -> Program {
    let mut c = Config::new(root).with_std(fixture_std());
    c.shuffle_seed = seed;
    Program::load(&c)
}

#[test]
fn canonical_ids_ignore_internal_order() {
    let suite = repo().join("tests/conformance/resolve");
    let mut roots: Vec<PathBuf> = std::fs::read_dir(&suite).unwrap().map(|e| e.unwrap().path()).collect();
    roots.sort();
    for root in roots {
        let base = snapshot(&load(&root, None));
        for seed in [1u64, 7, 0xdead_beef] {
            let other = snapshot(&load(&root, Some(seed)));
            assert_eq!(other, base, "{} differs with shuffle seed {seed}", root.display());
        }
    }
}

fn make_corpus(lines: u64, seed: u64) -> PathBuf {
    let out = std::env::temp_dir().join(format!("stave-sema-corpus-{lines}-{seed}-{}", std::process::id()));
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
fn generated_corpus_resolves_cleanly_and_canonically() {
    for seed in [1u64, 2, 3] {
        let dir = make_corpus(8000, seed);
        let p = load(&dir, None);
        let diags: Vec<String> = p
            .diagnostics
            .iter()
            .take(20)
            .map(|d| {
                let l = d.primary().and_then(|s| p.source_map.locate(s).ok());
                format!("{} {:?} {}", d.code(), l.map(|l| (l.path.to_string(), l.line, l.col)), d.message())
            })
            .collect();
        assert!(diags.is_empty(), "seed {seed}:\n{}", diags.join("\n"));
        assert!(p.res.iter().map(|r| r.uses.len()).sum::<usize>() > 1000);
        let base = snapshot(&p);
        assert_eq!(snapshot(&load(&dir, Some(seed + 100))), base, "seed {seed}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

fn mem(files: &[(&str, &str)]) -> Program {
    let mut m = BTreeMap::new();
    for (p, t) in files {
        m.insert(PathBuf::from("/m").join(p), t.as_bytes().to_vec());
    }
    for (p, t) in [
        ("core.stv", "pub enum Option<T> { None, Some(T) }\npub trait Copy {}\n"),
        ("io.stv", "pub fn print(s: &str) {\n}\n"),
    ] {
        m.insert(PathBuf::from("/std").join(p), t.as_bytes().to_vec());
    }
    let mut c = Config::new("/m").with_std("/std");
    c.fs = Fs::Mem(m);
    Program::load(&c)
}

fn codes(p: &Program) -> Vec<&'static str> {
    p.diagnostics.iter().map(|d| d.code()).collect()
}

const TOML: &str = "name = \"m\"\nkind = \"bin\"\nversion = \"0.1.0\"\n";

#[test]
fn locals_params_and_chains() {
    let p = mem(&[
        ("stave.toml", TOML),
        (
            "src/main.stv",
            "import std.io;\nenum E { A(i32), B }\nfn f(x: i32) -> i32 {\n    let y: i32 = x;\n    let e = E.A(y);\n    io.print(f\"{y}\");\n    return y;\n}\n",
        ),
    ]);
    assert_eq!(codes(&p), Vec::<&str>::new());
    let fitem =
        p.items.iter().position(|it| p.decl(it.decl).is_some_and(|d| d.kind == DeclKind::Fn)).unwrap();
    let r = &p.res[fitem];
    // `x` → the param; `y` (twice + the f-string hole) → the let pattern; `E`, `A`, `io`, `print`.
    let locals = r.uses.iter().filter(|u| matches!(u.res, Res::Local(_))).count();
    assert_eq!(locals, 4);
    assert_eq!(r.uses.iter().filter(|u| u.in_fmt).count(), 1);
    let decl_names: Vec<String> = r
        .uses
        .iter()
        .filter_map(|u| match u.res {
            Res::Decl(d) => Some(p.decl_path(d)),
            _ => None,
        })
        .collect();
    assert!(decl_names.contains(&"main.E".to_string()), "{decl_names:?}");
    assert!(decl_names.contains(&"main.E.A".to_string()), "{decl_names:?}");
    assert!(decl_names.contains(&"std.io".to_string()), "{decl_names:?}");
    assert!(decl_names.contains(&"std.io.print".to_string()), "{decl_names:?}");
    // Builtin types have no names line; `i32` uses are Prim.
    assert!(r.uses.iter().any(|u| matches!(u.res, Res::Prim(_))));
}

#[test]
fn deferred_steps_and_members() {
    let p = mem(&[
        ("stave.toml", TOML),
        (
            "src/main.stv",
            "struct S { pub a: i32, b: u8 }\nimpl S {\n    pub fn new() -> S { return S.{ a: 1, b: 2 }; }\n    fn get(&self) -> i32 { return self.a; }\n}\nfn main() -> i32 {\n    let s = S.new();\n    return s.get() + s.a + f32x4.splat(1.0).reduce_add() as i32;\n}\n",
        ),
    ]);
    assert_eq!(codes(&p), Vec::<&str>::new());
    let s = p.decls.iter().position(|d| d.kind == DeclKind::Struct).unwrap();
    let fields: Vec<&str> = p
        .members(stave_sema::DeclId(s as u32))
        .iter()
        .map(|&f| p.strings.get(p.decls[f.idx()].name))
        .collect();
    assert_eq!(fields, ["a", "b"]);
    let main = p
        .items
        .iter()
        .position(|it| p.decl(it.decl).is_some_and(|d| p.strings.get(d.name) == "main"))
        .unwrap();
    let kinds: Vec<_> = p.res[main].deferred.iter().map(|d| d.kind).collect();
    assert_eq!(
        kinds,
        [stave_sema::DeferKind::Member, stave_sema::DeferKind::Member, stave_sema::DeferKind::TypeRelative]
    );
    let imp = &p.impls[0];
    assert_eq!(imp.head, Some(stave_sema::TypeHead::Adt(stave_sema::DeclId(s as u32))));
    assert_eq!(p.impls_of(imp.head.unwrap()).count(), 1);
}

#[test]
fn generic_type_param_paths_are_type_relative() {
    let p = mem(&[
        ("stave.toml", TOML),
        (
            "src/main.stv",
            "trait Mk { fn mk() -> u8; }\nfn f<T: Mk>() -> u8 {\n    return T.mk();\n}\nimpl Mk for u8 {\n    fn mk() -> u8 { return Self.max(); }\n}\n",
        ),
    ]);
    assert_eq!(codes(&p), Vec::<&str>::new());
    let kinds: Vec<_> = p.res.iter().flat_map(|r| r.deferred.iter().map(|d| (d.kind, d.base))).collect();
    assert!(kinds.iter().all(|(k, _)| *k == stave_sema::DeferKind::TypeRelative), "{kinds:?}");
    assert_eq!(kinds.len(), 2);
    assert!(matches!(kinds[0].1, Res::Generic(_)));
    assert_eq!(kinds[1].1, Res::SelfType);
    let imp = &p.impls[0];
    assert_eq!(imp.head, Some(stave_sema::TypeHead::Prim(stave_sema::Prim::Scalar(stave_sema::Scalar::U8))));
}

#[test]
fn origins_and_self() {
    let p = mem(&[
        ("stave.toml", TOML),
        (
            "src/main.stv",
            "fn first(xs: &[u8], ys: &[u8]) -> &u8 from xs + zs {\n    return &xs[0];\n}\nfn main() -> i32 {\n    let v: u8 = self;\n    return 0;\n}\n",
        ),
    ]);
    assert_eq!(codes(&p), ["E0101", "E0101"]);
}

#[test]
fn unbalanced_file_and_bom_do_not_cascade() {
    let p = mem(&[
        ("stave.toml", TOML),
        ("src/main.stv", "fn main() -> i32 { return x; \n"),
        ("src/b.stv", "\u{feff}fn g() {}\n"),
    ]);
    assert_eq!(codes(&p), ["E0017", "E0016"]);
}

#[test]
fn single_file_root() {
    let dir = std::env::temp_dir().join(format!("stave-sema-single-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("t.stv");
    std::fs::write(&f, "import std.io;\nfn main() -> i32 {\n    io.print(\"hi\");\n    return zz;\n}\n")
        .unwrap();
    let p = load(&f, None);
    let d: Vec<_> = p
        .diagnostics
        .iter()
        .map(|d| {
            let l = p.source_map.locate(d.primary().unwrap()).unwrap();
            (d.code(), l.path.to_string(), l.line)
        })
        .collect();
    assert_eq!(d, [("E0101", "t.stv".to_string(), 4)]);
    assert_eq!(
        p.modules
            .iter()
            .find(|m| m.file.0 == p.files.iter().position(|x| x.abs_path == f).unwrap() as u32)
            .unwrap()
            .path_text,
        "main"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn never_panics_on_odd_inputs() {
    let odd = [
        "",
        "import;",
        "import a.b as;",
        "fn f(",
        "impl for {}",
        "enum E { A(), }",
        "struct S { a: Self }",
        "fn f() { let a = b.c.d.e::<T>>= 1; }",
        "fn f<const N: usize>() -> [u8; N] { return [0u8; N]; }",
        "allocator a.b;",
        "allocator main;\nallocator main;",
        "dispatch fn d(x: u8) -> u8 { avx2 => nope, _ => d, }",
        "unsafe(\"x\") extern \"C\" { fn puts(s: *const u8) -> i32 without(io); fn printf(f: *const u8, ...) -> i32; }",
        "fn f() { let x = fn(a: i32) -> i32 { return a + y; }; }",
        "fn f() { match 1u8 { m.X(a, b) => { } Y.{ a, b: c } => { } } }",
        "trait T { fn f(&self) -> Self; }",
        "fn f() -> u8 { return f\"{self.x}{a.b}\"; }",
        "\u{0}\u{1}@@@",
    ];
    for (k, src) in odd.iter().enumerate() {
        let p = mem(&[("stave.toml", TOML), ("src/main.stv", src)]);
        let _ = snapshot(&p);
        let _ = k;
    }
    // Deep nesting: no stack overflow (iterative walk).
    let deep = format!("fn f() -> i32 {{ return {}1{}; }}", "(".repeat(5000), ")".repeat(5000));
    let p = mem(&[("stave.toml", TOML), ("src/main.stv", &deep)]);
    assert!(p.diagnostics.is_empty(), "{:?}", codes(&p));
    let unary = format!("fn f() -> i32 {{ return {}1; }}", "-".repeat(200_000));
    let _ = mem(&[("stave.toml", TOML), ("src/main.stv", &unary)]);
}

mod fuzz {
    use super::*;
    use proptest::prelude::*;

    const WORDS: [&str; 48] = [
        "fn", "let", "mut", "struct", "enum", "impl", "trait", "for", "in", "match", "return", "import",
        "pub", "Self", "self", "as", "if", "then", "else", "x", "y", "S", "E", "T", "m", "std", "io", "(",
        ")", "{", "}", "[", "]", ".", ".{", ",", ";", ":", "::", "<", ">", "=", "=>", "->", "1u8",
        "f\"{x}\"", "tail", "_",
    ];

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]
        #[test]
        fn token_soup_never_panics(ws in proptest::collection::vec(0usize..WORDS.len(), 0..120)) {
            let src: Vec<&str> = ws.iter().map(|&k| WORDS[k]).collect();
            let src = src.join(" ");
            let p = mem(&[("stave.toml", TOML), ("src/main.stv", &src), ("src/m.stv", "pub fn f() {\n}\n")]);
            for (k, r) in p.res.iter().enumerate() {
                let ast = p.item_ast(stave_sema::ItemId(k as u32));
                for u in &r.uses {
                    prop_assert!((u.node as usize) < ast.len());
                }
            }
            let _ = snapshot(&p);
        }

        #[test]
        fn manifest_soup_never_panics(s in "[a-z\\[\\]=\"{}., #\n0-9_-]{0,80}") {
            let p = mem(&[("stave.toml", &s), ("src/main.stv", "fn main() -> i32 {\n    return 0;\n}\n")]);
            prop_assert!(p.diagnostics.iter().all(|d| d.code() == "E1002"));
        }
    }
}

fn diag_lines(p: &Program) -> Vec<String> {
    p.diagnostics
        .iter()
        .map(|d| {
            let l = d.primary().and_then(|s| p.source_map.locate(s).ok());
            format!("{} {:?} {}", d.code(), l.map(|l| (l.path.to_string(), l.line, l.col)), d.message())
        })
        .collect()
}

/// The shipped `std/` and `runtime/` resolve cleanly for every CPU target, and the runtime's
/// per-target dependency (runtime/stave.toml) selects exactly one arch package.
#[test]
fn shipped_std_and_runtime_resolve_per_target() {
    let dir = std::env::temp_dir().join(format!("stave-sema-shipped-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("hello.stv");
    std::fs::write(
        &f,
        "import std.io;\nimport std.vec;\nimport std.string;\n\nfn main() -> i32 {\n    let n: u32 = 7;\n    io.print(f\"n = {n}\\n\");\n    let v = vec.Vec::<u8>.new();\n    let s = string.String.new();\n    return 0;\n}\n",
    )
    .unwrap();
    let builtins = vec![
        BuiltinPackage {
            name: "std".into(),
            dir: repo().join("std"),
            deps: vec!["runtime".into()],
            implicit: true,
        },
        BuiltinPackage {
            name: "runtime".into(),
            dir: repo().join("runtime"),
            deps: Vec::new(),
            implicit: false,
        },
    ];
    for (target, arch, other) in [
        ("x86_64-linux-v2", "runtime_x86_64", "runtime_aarch64"),
        ("x86_64-linux-v4", "runtime_x86_64", "runtime_aarch64"),
        ("aarch64-linux", "runtime_aarch64", "runtime_x86_64"),
        ("aarch64-linux+sve2", "runtime_aarch64", "runtime_x86_64"),
    ] {
        let mut c = Config::new(&f);
        c.target = Some(stave_target::Triple::parse(target).unwrap());
        c.builtins = builtins.clone();
        let p = Program::load(&c);
        let d = diag_lines(&p);
        assert!(d.is_empty(), "{target}:\n{}", d.join("\n"));
        let mods: Vec<&str> = p.modules.iter().map(|m| m.path_text.as_str()).collect();
        assert!(mods.contains(&format!("{arch}.sys").as_str()), "{target}: {mods:?}");
        assert!(mods.contains(&"runtime.os") && mods.contains(&"std.core"), "{mods:?}");
        assert!(!mods.iter().any(|m| m.starts_with(other) || m.starts_with("runtime.x86_64")), "{mods:?}");
        let rt = p.packages.iter().find(|p| p.name == "runtime").unwrap();
        assert!(rt.manifest.is_some());
        assert_eq!(rt.deps.iter().map(|d| d.0.as_str()).collect::<Vec<_>>(), ["arch"]);
        let prelude: Vec<&str> = p.prelude.iter().map(|&(s, _)| p.strings.get(s)).collect();
        for name in ["Option", "Result", "Copy", "Drop", "Display", "Add", "Not"] {
            assert!(prelude.contains(&name), "{name} not in {prelude:?}");
        }
        assert!(!prelude.contains(&"Vec") && !prelude.contains(&"String"), "{prelude:?}");
    }
    // Without a target table for the arch (gpu-sm86 uses runtime-gpu), `import arch.sys` fails.
    let mut c = Config::new(&f);
    c.target = Some(stave_target::Triple::parse("gpu-sm86").unwrap());
    c.builtins = builtins;
    let p = Program::load(&c);
    assert!(codes(&p).contains(&"E0101"), "{:?}", diag_lines(&p));
    let _ = std::fs::remove_dir_all(&dir);
}

/// The runtime does not depend on std: its modules see no prelude.
#[test]
fn prelude_only_for_packages_depending_on_std() {
    let mut m = BTreeMap::new();
    for (p, t) in [
        ("/std/core.stv", "pub enum Option<T> { None, Some(T) }\n"),
        ("/rt/x.stv", "pub fn f(o: Option<u8>) -> never {\n    loop { }\n}\n"),
        ("/m/stave.toml", TOML),
        ("/m/src/main.stv", "fn main(o: Option<u8>) -> i32 {\n    return 0;\n}\n"),
    ] {
        m.insert(PathBuf::from(p), t.as_bytes().to_vec());
    }
    let mut c = Config::new("/m");
    c.builtins = vec![
        BuiltinPackage { name: "std".into(), dir: "/std".into(), deps: vec!["rt".into()], implicit: true },
        BuiltinPackage { name: "rt".into(), dir: "/rt".into(), deps: Vec::new(), implicit: false },
    ];
    c.fs = Fs::Mem(m);
    let p = Program::load(&c);
    let d = diag_lines(&p);
    // rt.x: `Option` unresolved (no prelude); main sees it.
    assert_eq!(codes(&p), ["E0101"], "{d:?}");
    assert!(d[0].contains("x.stv"), "{d:?}");
}

/// An unreadable source file is E1003 at its offset 0, never a crash.
#[test]
fn unreadable_source_file_is_e1003() {
    use std::os::unix::fs::PermissionsExt;
    let dir = std::env::temp_dir().join(format!("stave-sema-unreadable-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("src")).unwrap();
    std::fs::write(dir.join("stave.toml"), TOML).unwrap();
    std::fs::write(dir.join("src/main.stv"), "fn main() -> i32 {\n    return 0;\n}\n").unwrap();
    let bad = dir.join("src/bad.stv");
    std::fs::write(&bad, "pub fn f() {\n}\n").unwrap();
    std::fs::set_permissions(&bad, std::fs::Permissions::from_mode(0o000)).unwrap();
    let readable = std::fs::read(&bad).is_ok(); // true when the tests run as root
    let p = load(&dir, None);
    let _ = std::fs::set_permissions(&bad, std::fs::Permissions::from_mode(0o644));
    let _ = std::fs::remove_dir_all(&dir);
    if readable {
        return;
    }
    let d = diag_lines(&p);
    assert_eq!(codes(&p), ["E1003"], "{d:?}");
    assert!(d[0].contains("permission denied") && d[0].contains("src/bad.stv"), "{d:?}");
}

/// The exported `strings` / `names` printers produce the stage-gates formats (the test's own
/// printers above are the reference).
#[test]
fn exported_dump_printers() {
    let root = repo().join("tests/conformance/resolve");
    let mut roots: Vec<PathBuf> = std::fs::read_dir(&root).unwrap().map(|e| e.unwrap().path()).collect();
    roots.sort();
    assert!(!roots.is_empty());
    for r in roots {
        let p = load(&r, None);
        assert_eq!(stave_sema::dump_strings(&p), strings_dump(&p), "{}", r.display());
        // the reference printer covers every item; the dump skips dropped ones
        let want: String = names_dump(&p)
            .lines()
            .filter(|l| {
                let i: usize = l.split(' ').next().unwrap().parse().unwrap();
                p.items[i].ok
            })
            .map(|l| format!("{l}\n"))
            .collect();
        assert_eq!(stave_sema::dump_names(&p), want, "{}", r.display());
        let mut per_file = Vec::new();
        for f in 0..p.files.len() {
            stave_sema::names_file_into(&p, f, &mut per_file);
        }
        assert_eq!(per_file, want.as_bytes());
        let mut sb = Vec::new();
        stave_sema::strings_into(&p, &mut sb);
        assert_eq!(sb, strings_dump(&p).as_bytes());
    }
}

/// `Config::parse_only`: read, lex, items, parse; no built-in packages, no interning or
/// resolution (unresolved names are not reported), syntax errors are.
#[test]
fn parse_only_load() {
    let mut m = BTreeMap::new();
    m.insert(PathBuf::from("/m/stave.toml"), TOML.as_bytes().to_vec());
    m.insert(
        PathBuf::from("/m/src/main.stv"),
        b"import std.io;\nkernel fn k() { nope(); }\nfn main() -> i32 { return zz; }\nfn bad() { let = ; }\n"
            .to_vec(),
    );
    m.insert(PathBuf::from("/std/core.stv"), b"pub trait Copy {}\n".to_vec());
    let mut c = Config::new("/m").with_std("/std").parse_only();
    c.fs = Fs::Mem(m.clone());
    let p = Program::load(&c);
    assert_eq!(p.files.len(), 1, "no built-in packages");
    assert_eq!(p.packages.len(), 1);
    assert!(p.modules.is_empty() && p.decls.is_empty() && p.res.is_empty() && p.strings.is_empty());
    assert_eq!(p.items.len(), p.files[0].parsed.items.len());
    assert_eq!(p.items.len(), 4);
    assert_eq!(p.items.iter().filter(|it| it.ok).count(), 3);
    assert!(p.items.iter().all(|it| it.module.is_none() && it.decl.is_none()));
    let cs = codes(&p);
    assert!(!cs.is_empty() && cs.iter().all(|c| c.starts_with("E00")), "{cs:?}");
    let t = p.timings;
    assert_eq!((t.intern, t.resolve), (Default::default(), Default::default()));
    // the full load of the same program resolves (and reports more)
    let mut c = Config::new("/m").with_std("/std");
    c.fs = Fs::Mem(m);
    let full = Program::load(&c);
    assert!(full.files.len() > 1 && !full.decls.is_empty());
    assert!(codes(&full).iter().any(|c| c.starts_with("E01")), "{:?}", codes(&full));
    // same front end either way
    let pos = full.files.iter().position(|f| f.rel_path == p.files[0].rel_path).unwrap();
    assert_eq!(full.files[pos].parsed.tokens, p.files[0].parsed.tokens);
    assert_eq!(full.files[pos].parsed.diagnostics, p.files[0].parsed.diagnostics);
}

/// Loader timings under the diagnostics-schema stage names.
#[test]
fn stage_timings() {
    let p = load(&repo().join("tests/conformance/resolve/chains.stv"), None);
    let names: Vec<&str> = p.timings.stages().iter().map(|s| s.0).collect();
    assert_eq!(names, ["read", "lex", "items", "parse", "intern", "resolve"]);
    let t = p.timings;
    assert!(t.read + t.lex + t.items + t.parse_items <= t.parse);
}
