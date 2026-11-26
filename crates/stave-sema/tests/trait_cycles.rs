//! Trait queries on cycles of types: answers held while the goals they rest on are in progress
//! give the same answers as recomputing them, in any query order and on any thread count, and
//! the work grows linearly with the length of a cycle (counted in goals, not seconds).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use stave_sema::sigs::TraitRef;
use stave_sema::traits::{Answers, World};
use stave_sema::types::{GArg, Interner};
use stave_sema::{Config, DeclId, DeclKind, Fs, Program, TypeId, TypeKind, Typeck};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn mem(src: &str) -> (Program, Typeck) {
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
    c.target = stave_target::Triple::parse("x86_64-linux").ok();
    let p = Program::load(&c);
    let tc = stave_sema::check(&p);
    (p, tc)
}

/// The diagnostics as `line code message`.
fn diags(p: &Program, tc: &Typeck) -> Vec<String> {
    tc.all_diagnostics(p)
        .iter()
        .map(|d| {
            let line = d.primary().map_or(0, |s| p.source_map.locate(s).map_or(0, |l| l.line));
            format!("{line} {} {}", d.code(), d.message())
        })
        .collect()
}

/// Deterministic pseudo-random numbers (no dependency, same sequence everywhere).
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: u64) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) % n
    }
}

/// The program family: structs `S0..Sk` whose fields reach each other through raw pointers,
/// `Vec`s, a generic wrapper holding its argument as a field and as a type argument, options,
/// arrays, and tuples; some fields are not Frozen / Send / Sync / Flat (`Cell`, a raw pointer, a
/// `Drop` type). `view`: every struct is a view and fields may be references and fn values.
/// `open`: a Send impl bounded by a user trait whose impl is bounded by Send (Send and Sync are
/// then not closed marker traits; a Send goal can reach an inductive goal and back).
fn family(seed: u64, view: bool, open: bool) -> (String, usize) {
    let mut r = Lcg(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let k = 2 + r.below(6) as usize;
    let mut s = String::from(
        "import std.cell;\nimport std.vec;\nstruct W<T: view> { pub t: T }\n\
         struct P<T: view> { pub p: *const T }\n\
         unsafe(\"test\") impl<T: Send + view> Send for P<T> {}\n\
         unsafe(\"test\") impl<T: Sync + Frozen + view> Sync for P<T> {}\n\
         struct D { pub n: u32 }\nimpl Drop for D { fn drop(&mut self) {} }\n",
    );
    if open {
        s.push_str(
            "trait Tr {}\nstruct Q<T: view> { pub p: *const T }\nimpl<T: Send + view> Tr for Q<T> {}\n\
             struct R<T: view> { pub p: *const T }\nunsafe(\"test\") impl<T: Tr + view> Send for R<T> {}\n",
        );
    }
    let kw = if view { "view struct" } else { "struct" };
    for i in 0..k {
        let nf = 1 + r.below(3);
        let mut fields = Vec::new();
        for f in 0..nf {
            let j = r.below(k as u64);
            let opts = if open { 15 } else { 14 } + u64::from(view);
            let ty = match r.below(opts) {
                0 => format!("*const S{j}"),
                1 => format!("W<(u32, *const S{j})>"),
                2 => format!("vec.Vec<S{j}>"),
                3 => format!("W<vec.Vec<S{j}>>"),
                4 => format!("Option<*mut S{j}>"),
                5 => format!("[*const S{j}; 2]"),
                6 => format!("(vec.Vec<S{j}>, vec.Vec<S{j}>)"),
                7 => format!("P<S{j}>"),
                8 => format!("W<P<W<S{j}>>>"),
                9 => "cell.Cell<u32>".to_string(),
                10 if view => "fn(u32) -> u32".to_string(),
                10 => "u64".to_string(),
                11 => "D".to_string(),
                12 => "u32".to_string(),
                13 => "*const u8".to_string(),
                14 if open => format!("R<Q<S{j}>>"),
                _ => format!("&S{j}"),
            };
            fields.push(format!("pub f{f}: {ty}"));
        }
        s.push_str(&format!("{kw} S{i} {{ {} }}\n", fields.join(", ")));
    }
    s.push_str("fn main() -> i32 { return 0; }\n");
    (s, k)
}

/// The types reachable from `roots` through components and fields.
fn reach(tc: &Typeck, cx: &mut stave_sema::TypeTable, roots: &[TypeId]) -> Vec<TypeId> {
    let mut seen = std::collections::BTreeSet::new();
    let mut work = roots.to_vec();
    let mut out = Vec::new();
    while let Some(t) = work.pop() {
        if !seen.insert(t) {
            continue;
        }
        out.push(t);
        let k = cx.kind(t).clone();
        k.children(&mut work);
        if let TypeKind::Adt { def, args } = k
            && let Some(adt) = tc.sigs.adt(def)
        {
            for f in adt.variants.iter().flat_map(|v| v.fields.iter()) {
                work.push(cx.subst(f.ty, &args, None));
            }
        }
    }
    out
}

fn struct_named(p: &Program, name: &str) -> DeclId {
    (0..p.decls.len() as u32)
        .map(DeclId)
        .find(|&d| p.decl(d).is_some_and(|x| x.kind == DeclKind::Struct && p.strings.get(x.name) == name))
        .expect(name)
}

/// On every program of the family: each (type, trait) answer of a fresh held query equals the
/// one recomputed without held answers; answers kept across queries asked in a shuffled order
/// equal, held or not, the fresh ones (in the open family: held equals not held, same order).
#[test]
fn held_answers_equal_recomputed_ones_on_generated_cycles() {
    let mut programs = 0;
    let mut checked = 0;
    for seed in 0..36u64 {
        let (view, open) = (seed % 3 == 1, seed % 3 == 2);
        let (src, k) = family(seed, view, open);
        let (p, tc) = mem(&src);
        let errs: Vec<String> = diags(&p, &tc).into_iter().filter(|d| d.contains(" E")).collect();
        assert!(errs.is_empty(), "seed {seed}:\n{src}\n{errs:#?}");
        assert_eq!(tc.sigs.markers_closed, !open, "seed {seed}");
        // nothing from the checking run: every answer below is computed by its own query
        tc.sigs.queries.arm(0);
        let w = World { prog: &p, sigs: &tc.sigs };
        let mut cx = tc.types.clone();
        let roots: Vec<TypeId> =
            (0..k).map(|i| cx.mk_adt(struct_named(&p, &format!("S{i}")), Vec::new())).collect();
        let types = reach(&tc, &mut cx, &roots);
        let l = &tc.sigs.lang;
        let traits = [l.copy, l.send, l.sync, l.flat, l.frozen];
        let mut goals = Vec::new();
        for &t in &types {
            for &tr in &traits {
                goals.push((t, TraitRef { trait_decl: tr, args: Vec::new() }));
            }
        }
        let reference: Vec<bool> =
            goals.iter().map(|(t, tr)| w.implements_unheld(&mut cx, *t, tr, &[])).collect();
        for (i, (t, tr)) in goals.iter().enumerate() {
            assert_eq!(w.implements(&mut cx, *t, tr, &[]), reference[i], "seed {seed} goal {i}\n{src}");
        }
        for round in 0..3u64 {
            let mut order: Vec<usize> = (0..goals.len()).collect();
            let mut r = Lcg(seed * 7 + round + 1);
            for i in (1..order.len()).rev() {
                order.swap(i, r.below(i as u64 + 1) as usize);
            }
            let (mut held, mut unheld) = (Answers::default(), Answers::unheld());
            for &i in &order {
                let (t, tr) = &goals[i];
                let a = w.implements_kept(&mut cx, *t, tr, &[], &mut held);
                let b = w.implements_kept(&mut cx, *t, tr, &[], &mut unheld);
                assert_eq!(a, b, "seed {seed} round {round} goal {i}\n{src}");
                if !open {
                    assert_eq!(a, reference[i], "seed {seed} round {round} goal {i}\n{src}");
                }
            }
        }
        programs += 1;
        checked += goals.len();
    }
    assert_eq!(programs, 36);
    assert!(checked > 1000, "{checked} goals");
}

/// `struct W<T> { t: T }` holds its argument twice (a field and a type argument), so a cycle of
/// `S_i { a: W<(u32, *const S_{i+1})> }` reaches each member by two paths at each step: the
/// goals computed grow linearly with the cycle, for a check (one Frozen bound), the kinds dump,
/// and the same through `Vec`s for Send and Sync. A 5000-long cycle checks on the big-stack
/// workers whatever the caller's stack.
#[test]
fn work_on_a_cycle_is_linear_in_its_length() {
    let w_cycle = |n: usize, need: &str| {
        let mut s = String::from("struct W<T> { t: T }\n");
        for i in 0..n {
            s.push_str(&format!("struct S{i} {{ a: W<(u32, *const S{})> }}\n", (i + 1) % n));
        }
        s.push_str(&format!("fn need<T: {need}>() {{}}\nfn main() -> i32 {{ need::<S0>(); return 0; }}\n"));
        s
    };
    let vec_cycle = |n: usize| {
        let mut s = String::from("import std.vec;\n");
        for i in 0..n {
            let j = (i + 1) % n;
            s.push_str(&format!("struct S{i} {{ a: vec.Vec<S{j}>, b: vec.Vec<S{j}> }}\n"));
        }
        s.push_str("fn need<T: Send + Sync>() {}\nfn main() -> i32 { need::<S0>(); return 0; }\n");
        s
    };
    let goals = |src: &str| {
        let (p, tc) = mem(src);
        assert_eq!(diags(&p, &tc), Vec::<String>::new());
        let checked = tc.sigs.queries.goals_computed();
        stave_sema::dump_kinds(&p, &tc);
        (checked, tc.sigs.queries.goals_computed() - checked)
    };
    // per member: S_i, W<..>, the tuple, the pointer, for each trait asked; generous
    let per = 64;
    for (name, make) in [
        ("frozen", &(|n| w_cycle(n, "Frozen")) as &dyn Fn(usize) -> String),
        ("send-sync", &vec_cycle as &dyn Fn(usize) -> String),
    ] {
        let (c2, d2) = goals(&make(2));
        for n in [16, 1000] {
            let (c, d) = goals(&make(n));
            let bound = per * (n as u64 - 2);
            assert!(c - c2 <= bound, "{name} n={n}: check computed {} goals over n=2, bound {bound}", c - c2);
            assert!(d.saturating_sub(d2) <= bound, "{name} n={n}: dump computed {} goals over n=2", d - d2);
        }
    }
    let one = rayon::ThreadPoolBuilder::new().num_threads(1).stack_size(1 << 20).build().unwrap();
    let (c2, _) = goals(&w_cycle(2, "Frozen"));
    let (c, _) = one.install(|| goals(&w_cycle(5000, "Frozen")));
    assert!(c - c2 <= per * 4998, "n=5000: {} goals", c - c2);
}

/// Cycles whose answers are known: one that holds, one false deep inside (a `Cell` three steps
/// in), two interlocking cycles of which one reaches the false one, a Send/Sync cycle through
/// `Vec`s with a raw pointer member. Each bound is asked in an order that enters the cycles at
/// different members; the diagnostics are the same on one thread, two, and the default count.
#[test]
fn cycle_answers_and_diagnostics_do_not_depend_on_order_or_threads() {
    let src = "import std.cell;\nimport std.vec;\n\
         struct W<T> { t: T }\n\
         struct A0 { a: W<(u32, *const A1)> }\n\
         struct A1 { a: W<(u32, *const A2)>, b: W<(u32, *const A0)> }\n\
         struct A2 { a: W<(u32, *const A0)> }\n\
         struct B0 { a: W<(u32, *const B1)> }\n\
         struct B1 { a: W<(u32, *const B2)> }\n\
         struct B2 { a: W<(u32, *const B3)> }\n\
         struct B3 { a: W<(u32, *const B0)>, c: cell.Cell<u32> }\n\
         struct C0 { a: W<(u32, *const C1)> }\n\
         struct C1 { a: W<(u32, *const C0)>, d: W<(u32, *const D0)> }\n\
         struct D0 { a: W<(u32, *const D1)> }\n\
         struct D1 { a: W<(u32, *const C1)>, b: W<(u32, *const B2)> }\n\
         struct E0 { a: W<(u32, *const E1)> }\n\
         struct E1 { a: W<(u32, *const E0)>, d: W<(u32, *const A1)> }\n\
         struct V0 { a: vec.Vec<V1>, b: vec.Vec<V1> }\n\
         struct V1 { a: vec.Vec<V2>, b: vec.Vec<V0> }\n\
         struct V2 { a: vec.Vec<V0>, p: *const u8 }\n\
         struct U0 { a: vec.Vec<U1> }\n\
         struct U1 { a: vec.Vec<U0>, b: vec.Vec<U1> }\n\
         fn frozen<T: Frozen>() {}\n\
         fn send<T: Send>() {}\n\
         fn sync<T: Sync>() {}\n\
         fn main() -> i32 {\n\
         \x20   frozen::<A1>();\n\
         \x20   frozen::<B2>();\n\
         \x20   frozen::<D0>();\n\
         \x20   frozen::<C0>();\n\
         \x20   frozen::<E0>();\n\
         \x20   frozen::<B0>();\n\
         \x20   frozen::<A0>();\n\
         \x20   send::<V1>();\n\
         \x20   sync::<V0>();\n\
         \x20   send::<U1>();\n\
         \x20   sync::<U0>();\n\
         \x20   frozen::<V2>();\n\
         \x20   return 0;\n\
         }\n";
    let (p, tc) = mem(src);
    let got = diags(&p, &tc);
    let lines: Vec<u32> = got.iter().map(|d| d.split(' ').next().unwrap().parse().unwrap()).collect();
    // B2 (the Cell two steps in), D0 and C0 (they reach B2), B0; V1 and V0 (a raw pointer)
    assert_eq!(lines, [27, 28, 29, 31, 33, 34], "{got:#?}");
    let w = World { prog: &p, sigs: &tc.sigs };
    let mut cx = tc.types.clone();
    let frozen = TraitRef { trait_decl: tc.sigs.lang.frozen, args: Vec::new() };
    for (name, want) in
        [("A0", true), ("A1", true), ("E1", true), ("B0", false), ("B3", false), ("C1", false), ("D1", false)]
    {
        let t = cx.mk_adt(struct_named(&p, name), Vec::<GArg>::new());
        assert_eq!(w.implements_unheld(&mut cx, t, &frozen, &[]), want, "{name}");
    }
    for threads in [1, 2] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let (p2, tc2) = pool.install(|| mem(src));
        assert_eq!(diags(&p2, &tc2), got, "{threads} threads");
    }
}
