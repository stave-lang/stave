//! Trait queries on cycles of types: answers held while the goals they rest on are in progress
//! give the same answers as recomputing them, in any query order and on any thread count, and
//! the work grows linearly with the length of a cycle (counted in goals, not seconds). A cycle
//! through an inductive goal (a user trait, Copy) proves nothing, whichever goal is asked first
//! (contracts/lang-details.md "Trait cycles"): checked against an oracle that computes that
//! meaning on the program's goal graph, apart from the solver.

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
    decl_named(p, DeclKind::Struct, name)
}

fn decl_named(p: &Program, kind: DeclKind, name: &str) -> DeclId {
    (0..p.decls.len() as u32)
        .map(DeclId)
        .find(|&d| p.decl(d).is_some_and(|x| x.kind == kind && p.strings.get(x.name) == name))
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

/// The traits of the mixed family, by index: two coinductive, three inductive.
const MIXED: [&str; 5] = ["Send", "Sync", "Tr0", "Tr1", "Copy"];

/// A type a goal of the mixed family asks about: `S<j>`, `W<x><S<j>>`, `W<x><W<y><S<j>>>`.
#[derive(Clone, Copy)]
enum Ask {
    S(usize),
    W(usize, usize),
    Ww(usize, usize, usize),
}

impl Ask {
    fn text(self) -> String {
        match self {
            Ask::S(j) => format!("S{j}"),
            Ask::W(x, j) => format!("W{x}<S{j}>"),
            Ask::Ww(x, y, j) => format!("W{x}<W{y}<S{j}>>"),
        }
    }

    fn ty(self, p: &Program, cx: &mut stave_sema::TypeTable) -> TypeId {
        let s = |cx: &mut stave_sema::TypeTable, j: usize| {
            cx.mk_adt(struct_named(p, &format!("S{j}")), Vec::new())
        };
        let w = |cx: &mut stave_sema::TypeTable, x: usize, t: TypeId| {
            cx.mk_adt(struct_named(p, &format!("W{x}")), vec![GArg::Type(t)])
        };
        match self {
            Ask::S(j) => s(cx, j),
            Ask::W(x, j) => {
                let t = s(cx, j);
                w(cx, x, t)
            }
            Ask::Ww(x, y, j) => {
                let t = s(cx, j);
                let t = w(cx, y, t);
                w(cx, x, t)
            }
        }
    }
}

/// A program of the mixed family: wrappers `W<n><T> { p: *const T }`, each with impls of some of
/// [`MIXED`] whose param is bounded by one of them or by none (Send / Sync impls `unsafe`), and
/// structs `S0..Sk` holding wrappers of each other, `u32`, and raw pointers, some with impls of
/// the user traits. So Send / Sync goals reach user trait and Copy goals and back: cycles of
/// coinductive goals only, of inductive goals only, and mixed ones. `or`: also impls of Send,
/// Sync, Tr0, Tr1 for every type (`impl<T: B> Tr0 for T`), so a goal has two rules (rejected,
/// E0420 / E0903, but still in the solver's impl lists): the shape a wrong answer of one goal can
/// hide behind another.
#[derive(Clone)]
struct Mixed {
    src: String,
    k: usize,
    /// Per wrapper, per trait of [`MIXED`]: whether it has an impl, and the impl's bounds.
    wrap: Vec<[Option<Vec<usize>>; 5]>,
    /// Per struct: whether it has an impl of Tr0, of Tr1.
    own: Vec<[bool; 2]>,
    /// Per trait of [`MIXED`] but Copy: the bounds of its impl for every type, if any.
    blanket: [Option<Vec<usize>>; 4],
    /// Each struct and each wrapper type a field names.
    asks: Vec<Ask>,
}

/// The bounds of one impl's param: none, one, or (`two`) two of [`MIXED`], few of them Copy (a
/// struct is never Copy), so that many goals reach no failing one and only their cycles decide
/// them.
fn mixed_bounds(r: &mut Lcg, two: bool) -> Vec<usize> {
    let pick = |r: &mut Lcg| match r.below(17) {
        0..=5 => 0,
        6..=9 => 1,
        10..=12 => 2,
        13..=15 => 3,
        _ => 4,
    };
    if r.below(7) == 0 {
        return Vec::new();
    }
    let mut b = vec![pick(r)];
    if two && r.below(3) == 0 {
        let c = pick(r);
        if c != b[0] {
            b.push(c);
        }
    }
    b
}

fn bounds_text(b: &[usize]) -> String {
    if b.is_empty() {
        return String::new();
    }
    format!(": {}", b.iter().map(|&x| MIXED[x]).collect::<Vec<_>>().join(" + "))
}

fn mixed_family(seed: u64, or: bool) -> Mixed {
    let mut r = Lcg(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let k = 2 + r.below(5) as usize;
    let m = 2 + r.below(3) as usize;
    let mut src = String::from("trait Tr0 {}\ntrait Tr1 {}\n");
    let mut blanket: [Option<Vec<usize>>; 4] = Default::default();
    if or {
        for (a, slot) in blanket.iter_mut().enumerate() {
            if r.below(2) == 0 {
                let mut b = mixed_bounds(&mut r, true);
                if b.is_empty() {
                    b.push(r.below(4) as usize);
                }
                let un = if a < 2 { "unsafe(\"test\") " } else { "" };
                src.push_str(&format!("{un}impl<T{}> {} for T {{}}\n", bounds_text(&b), MIXED[a]));
                *slot = Some(b);
            }
        }
    }
    let mut wrap = Vec::new();
    for n in 0..m {
        src.push_str(&format!("struct W{n}<T> {{ pub p: *const T }}\n"));
        let mut row: [Option<Vec<usize>>; 5] = Default::default();
        // few goals without an impl
        for (a, slot) in row.iter_mut().enumerate() {
            if r.below(10) < [1, 1, 2, 2, 5][a] {
                continue;
            }
            let b = mixed_bounds(&mut r, or);
            let un = if a < 2 { "unsafe(\"test\") " } else { "" };
            src.push_str(&format!("{un}impl<T{}> {} for W{n}<T> {{}}\n", bounds_text(&b), MIXED[a]));
            *slot = Some(b);
        }
        wrap.push(row);
    }
    let mut own = Vec::new();
    let mut asks = Vec::new();
    for i in 0..k {
        let o = [r.below(2) == 0, r.below(2) == 0];
        for (a, &yes) in o.iter().enumerate() {
            if yes {
                src.push_str(&format!("impl Tr{a} for S{i} {{}}\n"));
            }
        }
        own.push(o);
        asks.push(Ask::S(i));
        // a cycle through an inductive goal needs a wrapper of a wrapper: the bound of the outer
        // one's impl asks the inner one, whose impl's bound asks the struct
        let nf = 1 + r.below(2);
        let mut fields = Vec::new();
        for f in 0..nf {
            let j = r.below(k as u64) as usize;
            let (x, y) = (r.below(m as u64) as usize, r.below(m as u64) as usize);
            let ty = match r.below(20) {
                0..=9 => {
                    asks.push(Ask::Ww(x, y, j));
                    Ask::Ww(x, y, j).text()
                }
                10..=14 => {
                    asks.push(Ask::W(x, j));
                    Ask::W(x, j).text()
                }
                15 | 16 => format!("W{x}<u32>"),
                17 | 18 => "u32".to_string(),
                _ => format!("*const S{j}"),
            };
            fields.push(format!("pub f{f}: {ty}"));
        }
        src.push_str(&format!("struct S{i} {{ {} }}\n", fields.join(", ")));
    }
    Mixed { src, k, wrap, own, blanket, asks }
}

/// The rule of goal (t, MIXED[a]) from the family's own record of its impls: the goal holds when
/// every goal of one of these clauses does (none: it does not; an empty one: it does).
fn mixed_rule(
    p: &Program,
    tc: &Typeck,
    cx: &mut stave_sema::TypeTable,
    m: &Mixed,
    t: TypeId,
    a: usize,
) -> Vec<Vec<(TypeId, usize)>> {
    let yes = |b: bool| if b { vec![Vec::new()] } else { Vec::new() };
    // an impl for every type: its bound, on the same type
    let mut out: Vec<Vec<(TypeId, usize)>> = m
        .blanket
        .get(a)
        .cloned()
        .flatten()
        .map(|b| b.into_iter().map(|b| (t, b)).collect())
        .into_iter()
        .collect();
    let own = match cx.kind(t).clone() {
        TypeKind::Int { .. } => yes(a != 2 && a != 3),
        TypeKind::Ptr { .. } => yes(a == 4),
        TypeKind::Adt { def, args } => {
            let name = p.strings.get(p.decl(def).unwrap().name).to_string();
            let i: usize = name[1..].parse().unwrap();
            let adt = tc.sigs.adt(def).unwrap();
            let fs: Vec<TypeId> = adt.variants.iter().flat_map(|v| v.fields.iter()).map(|f| f.ty).collect();
            let fields: Vec<(TypeId, usize)> =
                fs.into_iter().map(|f| (cx.subst(f, &args, None), a)).collect();
            if name.starts_with('S') {
                match a {
                    0 | 1 => vec![fields],
                    2 | 3 => yes(m.own[i][a - 2]),
                    _ => yes(false),
                }
            } else {
                let GArg::Type(x) = args[0] else { panic!("{name}") };
                match &m.wrap[i][a] {
                    Some(b) => vec![b.iter().map(|&b| (x, b)).collect()],
                    // no impl for the head: Send / Sync are structural, a user trait or Copy
                    // does not hold
                    None if a < 2 => vec![fields],
                    None => yes(false),
                }
            }
        }
        k => panic!("{k:?}"),
    };
    out.extend(own);
    out
}

/// The answers the language gives the goals `asked` and every goal they reach
/// (contracts/lang-details.md "Trait cycles"): the least fixpoint over the inductive goals of
/// the greatest fixpoint over the coinductive ones, by plain iteration on the goal graph.
/// `all_co`: every goal taken as coinductive instead (what holds if a cycle through an
/// inductive goal held).
fn mixed_oracle(
    p: &Program,
    tc: &Typeck,
    cx: &mut stave_sema::TypeTable,
    m: &Mixed,
    asked: &[(TypeId, usize)],
    all_co: bool,
) -> std::collections::HashMap<(TypeId, usize), bool> {
    let mut index = std::collections::HashMap::new();
    let mut nodes: Vec<(TypeId, usize)> = Vec::new();
    let mut rules: Vec<Vec<Vec<(TypeId, usize)>>> = Vec::new();
    let mut work = asked.to_vec();
    while let Some(g) = work.pop() {
        if index.contains_key(&g) {
            continue;
        }
        index.insert(g, nodes.len());
        nodes.push(g);
        let rule = mixed_rule(p, tc, cx, m, g.0, g.1);
        work.extend(rule.iter().flatten().copied());
        rules.push(rule);
    }
    let co: Vec<bool> = nodes.iter().map(|g| all_co || g.1 < 2).collect();
    let eval = |val: &[bool], i: usize| rules[i].iter().any(|c| c.iter().all(|g| val[index[g]]));
    let mut x = vec![false; nodes.len()];
    loop {
        let mut y: Vec<bool> = (0..nodes.len()).map(|i| co[i] || eval(&x, i)).collect();
        loop {
            let next: Vec<bool> = (0..nodes.len()).map(|i| if co[i] { eval(&y, i) } else { y[i] }).collect();
            if next == y {
                break;
            }
            y = next;
        }
        if y == x {
            break;
        }
        x = y;
    }
    nodes.into_iter().zip(x).collect()
}

fn mixed_traits(p: &Program, tc: &Typeck) -> [DeclId; 5] {
    let l = &tc.sigs.lang;
    [l.send, l.sync, decl_named(p, DeclKind::Trait, "Tr0"), decl_named(p, DeclKind::Trait, "Tr1"), l.copy]
}

fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    let mut r = Lcg(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    for i in (1..order.len()).rev() {
        order.swap(i, r.below(i as u64 + 1) as usize);
    }
    order
}

/// On every program of the mixed family (a third with impls for every type), every goal on every
/// type the structs reach equals the oracle's answer: asked fresh (held or not), kept across
/// queries in shuffled orders, and through the run-wide cache from 1, 2, and 16 threads, each
/// asking in its own shuffled order.
#[test]
fn mixed_cycles_answer_as_the_oracle_in_every_order() {
    let mut programs = 0;
    let mut checked = 0;
    let mut through_inductive = 0;
    let mut by_blanket = 0;
    for seed in 0..180u64 {
        let or = seed % 3 == 2;
        let m = mixed_family(seed, or);
        let (p, tc) = mem(&format!("{}fn main() -> i32 {{ return 0; }}\n", m.src));
        let errs: Vec<String> = diags(&p, &tc)
            .into_iter()
            .filter(|d| d.contains(" E") && !(or && (d.contains(" E0420 ") || d.contains(" E0903 "))))
            .collect();
        assert!(errs.is_empty(), "seed {seed}:\n{}\n{errs:#?}", m.src);
        tc.sigs.queries.arm(0);
        let w = World { prog: &p, sigs: &tc.sigs };
        let mut cx = tc.types.clone();
        let roots: Vec<TypeId> = (0..m.k).map(|i| Ask::S(i).ty(&p, &mut cx)).collect();
        let types = reach(&tc, &mut cx, &roots);
        let traits = mixed_traits(&p, &tc);
        let goals: Vec<(TypeId, usize)> = types.iter().flat_map(|&t| (0..5).map(move |a| (t, a))).collect();
        let want = mixed_oracle(&p, &tc, &mut cx, &m, &goals, false);
        let co = mixed_oracle(&p, &tc, &mut cx, &m, &goals, true);
        if or {
            let plain = Mixed { blanket: Default::default(), ..m.clone() };
            let without = mixed_oracle(&p, &tc, &mut cx, &plain, &goals, false);
            by_blanket += goals.iter().filter(|g| want[g] != without[g]).count();
        }
        let tref = |a: usize| TraitRef { trait_decl: traits[a], args: Vec::new() };
        let show =
            |g: &(TypeId, usize), cx: &stave_sema::TypeTable| format!("{:?}: {}", cx.kind(g.0), MIXED[g.1]);
        for g in &goals {
            let fresh = w.implements(&mut cx, g.0, &tref(g.1), &[]);
            assert_eq!(fresh, want[g], "seed {seed}: {}\n{}", show(g, &cx), m.src);
            assert_eq!(w.implements_unheld(&mut cx, g.0, &tref(g.1), &[]), want[g], "seed {seed} unheld");
        }
        for round in 0..4u64 {
            let (mut held, mut unheld) = (Answers::default(), Answers::unheld());
            for i in shuffled(goals.len(), seed * 7 + round + 1) {
                let g = &goals[i];
                let a = w.implements_kept(&mut cx, g.0, &tref(g.1), &[], &mut held);
                let b = w.implements_kept(&mut cx, g.0, &tref(g.1), &[], &mut unheld);
                assert_eq!(
                    (a, b),
                    (want[g], want[g]),
                    "seed {seed} round {round}: {}\n{}",
                    show(g, &cx),
                    m.src
                );
            }
        }
        for threads in [1u64, 2, 16] {
            tc.sigs.queries.arm(cx.len());
            std::thread::scope(|sc| {
                for th in 0..threads {
                    let (w, goals, want, tref) = (&w, &goals, &want, &tref);
                    let mut cx = cx.clone();
                    sc.spawn(move || {
                        for i in shuffled(goals.len(), seed * 131 + th * 17 + threads) {
                            let g = &goals[i];
                            let got = w.implements(&mut cx, g.0, &tref(g.1), &[]);
                            assert_eq!(got, want[g], "seed {seed}, {threads} threads, thread {th}, goal {i}");
                        }
                    });
                }
            });
        }
        tc.sigs.queries.arm(0);
        programs += 1;
        checked += goals.len();
        through_inductive += goals.iter().filter(|g| want[g] != co[g]).count();
    }
    assert_eq!(programs, 180);
    assert!(checked > 3000, "{checked} goals");
    // goals that hold only if a cycle through an inductive goal held
    assert!(
        through_inductive > 100,
        "{through_inductive} goals refused by a cycle through an inductive goal"
    );
    assert!(by_blanket > 100, "{by_blanket} goals decided by an impl for every type");
}

/// The checker's answers on the mixed family: each goal asked by a bound in a fn of its own, the
/// fns in a shuffled order, checked on 1, 2, and 16 threads, twice each: E0901 exactly at
/// the bounds the oracle refuses, in every run. The programs checked are 24 of the family, most
/// with a bound refused by a cycle through an inductive goal.
#[test]
fn mixed_cycle_diagnostics_do_not_depend_on_order_or_threads() {
    let pools: Vec<rayon::ThreadPool> =
        [1, 2, 16].iter().map(|&n| rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap()).collect();
    let mut refused = 0;
    let mut through_inductive = 0;
    let mut programs = 0;
    for seed in 100..1000u64 {
        if programs == 24 {
            break;
        }
        let m = mixed_family(seed, false);
        let all: Vec<(Ask, usize)> = m.asks.iter().flat_map(|&t| (0..5).map(move |a| (t, a))).collect();
        let asks: Vec<(Ask, usize)> = shuffled(all.len(), seed).into_iter().map(|i| all[i]).collect();
        let mut src = m.src.clone();
        for (a, name) in MIXED.iter().enumerate() {
            src.push_str(&format!("fn need{a}<T: {name}>() {{}}\n"));
        }
        let first = src.lines().count() as u32 + 1;
        for (n, (t, a)) in asks.iter().enumerate() {
            src.push_str(&format!("fn ask{n}() {{ need{a}::<{}>(); }}\n", t.text()));
        }
        src.push_str("fn main() -> i32 { return 0; }\n");
        let (p, tc) = mem(&src);
        let mut cx = tc.types.clone();
        let goals: Vec<(TypeId, usize)> = asks.iter().map(|&(t, a)| (t.ty(&p, &mut cx), a)).collect();
        let want = mixed_oracle(&p, &tc, &mut cx, &m, &goals, false);
        let co = mixed_oracle(&p, &tc, &mut cx, &m, &goals, true);
        let expect: Vec<u32> =
            (0..goals.len()).filter(|&n| !want[&goals[n]]).map(|n| first + n as u32).collect();
        let mixed = goals.iter().filter(|g| want[g] != co[g]).count();
        if mixed == 0 && programs >= 4 {
            continue;
        }
        programs += 1;
        refused += expect.len();
        through_inductive += mixed;
        for (pi, pool) in pools.iter().enumerate() {
            for run in 0..2 {
                let (p, tc) = pool.install(|| mem(&src));
                let got = diags(&p, &tc);
                assert!(got.iter().all(|d| d.contains(" E0901 ")), "seed {seed}:\n{src}\n{got:#?}");
                let lines: Vec<u32> =
                    got.iter().map(|d| d.split(' ').next().unwrap().parse().unwrap()).collect();
                assert_eq!(lines, expect, "seed {seed}, pool {pi}, run {run}:\n{src}\n{got:#?}");
            }
        }
    }
    assert_eq!(programs, 24);
    assert!(refused > 100, "{refused} refused bounds");
    assert!(
        through_inductive > 30,
        "{through_inductive} bounds refused by a cycle through an inductive goal"
    );
}

/// Many independent mixed cycles (`S: Send` through `Q<S>: Tr` and back), each asked from its
/// coinductive end in one fn and from its inductive end in another, checked on 1, 2, and 16
/// threads ten times each: every bound is refused in every run.
#[test]
fn many_mixed_cycles_are_refused_in_every_run() {
    let n = 40;
    let mut src = String::from(
        "trait Tr {}\nstruct Q<T> { pub p: *const T }\nimpl<T: Send> Tr for Q<T> {}\n\
         struct R<T> { pub p: *const T }\nunsafe(\"test\") impl<T: Tr> Send for R<T> {}\n\
         fn need_send<T: Send>() {}\nfn need_tr<T: Tr>() {}\n",
    );
    for i in 0..n {
        src.push_str(&format!("struct S{i} {{ pub f: R<Q<S{i}>> }}\n"));
    }
    for i in 0..n {
        src.push_str(&format!(
            "fn a{i}() {{ need_send::<S{i}>(); }}\nfn b{i}() {{ need_tr::<Q<S{i}>>(); }}\n"
        ));
    }
    src.push_str("fn main() -> i32 { return 0; }\n");
    for threads in [1, 2, 16] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        for run in 0..10 {
            let (p, tc) = pool.install(|| mem(&src));
            let got = diags(&p, &tc);
            assert_eq!(got.len(), 2 * n, "{threads} threads, run {run}: {got:#?}");
            assert!(got.iter().all(|d| d.contains(" E0901 ")), "{got:#?}");
        }
    }
}

/// Mixed cycles of length n, every member asked from both ends: through one path
/// (`S_i { f: R<Q<S_i+1>> }`), two paths (two such fields), and a cycle of Send goals that holds
/// with a user trait goal off the cycle at each member. The goals computed by the check and by
/// the kinds dump grow linearly with n, on one thread and on the default count, and so do those
/// of kept queries asking the members in cycle order (each query reaches the members asked
/// before it only at the end of the cycle, so each would recompute the cycle if the answers of
/// the first one were not all kept).
#[test]
fn work_on_a_mixed_cycle_is_linear_in_its_length() {
    let head = "trait Tr {}\nstruct Qc<T> { pub p: *const T }\nimpl<T: Send> Tr for Qc<T> {}\n\
                struct R<T> { pub p: *const T }\nunsafe(\"test\") impl<T: Tr> Send for R<T> {}\n\
                struct P<T> { pub p: *const T }\nunsafe(\"test\") impl<T: Send> Send for P<T> {}\n\
                fn need_send<T: Send>() {}\nfn need_tr<T: Tr>() {}\n";
    let make = |n: usize, form: usize| {
        let mut s = String::from(head);
        for i in 0..n {
            let j = (i + 1) % n;
            let fields = match form {
                0 => format!("pub f: R<Qc<S{j}>>"),
                1 => format!("pub f: R<Qc<S{j}>>, pub g: R<Qc<S{j}>>"),
                _ => format!("pub next: P<S{j}>, pub r: R<Qc<u32>>"),
            };
            s.push_str(&format!("struct S{i} {{ {fields} }}\n"));
            s.push_str(&format!(
                "fn a{i}() {{ need_send::<S{i}>(); }}\nfn b{i}() {{ need_tr::<Qc<S{i}>>(); }}\n"
            ));
        }
        s.push_str("fn main() -> i32 { return 0; }\n");
        s
    };
    let goals = |n: usize, form: usize| {
        let (p, tc) = mem(&make(n, form));
        let errs = diags(&p, &tc);
        assert_eq!(errs.len(), if form == 2 { 0 } else { 2 * n }, "form {form} n={n}: {errs:#?}");
        let checked = tc.sigs.queries.goals_computed();
        stave_sema::dump_kinds(&p, &tc);
        let dumped = tc.sigs.queries.goals_computed() - checked;
        // nothing from the run: only what the kept answers hold answers the later goals
        tc.sigs.queries.arm(0);
        let w = World { prog: &p, sigs: &tc.sigs };
        let mut cx = tc.types.clone();
        let send = TraitRef { trait_decl: tc.sigs.lang.send, args: Vec::new() };
        let tr = TraitRef { trait_decl: decl_named(&p, DeclKind::Trait, "Tr"), args: Vec::new() };
        let qc = struct_named(&p, "Qc");
        let before = tc.sigs.queries.goals_computed();
        // a query recurses once per goal on its path: a stack like the checker's workers
        std::thread::scope(|sc| {
            let body = || {
                let mut kept = Answers::default();
                for i in 0..n {
                    let s = cx.mk_adt(struct_named(&p, &format!("S{i}")), Vec::new());
                    let q = cx.mk_adt(qc, vec![GArg::Type(s)]);
                    assert_eq!(
                        w.implements_kept(&mut cx, s, &send, &[], &mut kept),
                        form == 2,
                        "form {form} S{i}"
                    );
                    assert_eq!(
                        w.implements_kept(&mut cx, q, &tr, &[], &mut kept),
                        form == 2,
                        "form {form} Qc<S{i}>"
                    );
                }
            };
            std::thread::Builder::new().stack_size(1 << 29).spawn_scoped(sc, body).unwrap().join().unwrap();
        });
        (checked, dumped, tc.sigs.queries.goals_computed() - before)
    };
    let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let per = 64;
    for form in 0..3 {
        for single in [true, false] {
            let run = |n: usize| if single { one.install(|| goals(n, form)) } else { goals(n, form) };
            let (c2, d2, k2) = run(2);
            for n in [16, 400] {
                let (c, d, k) = run(n);
                let bound = per * (n as u64 - 2);
                assert!(
                    c - c2 <= bound,
                    "form {form} n={n} one thread {single}: check computed {} goals over n=2",
                    c - c2
                );
                assert!(
                    d.saturating_sub(d2) <= bound,
                    "form {form} n={n}: dump computed {} goals over n=2",
                    d - d2
                );
                assert!(
                    k - k2 <= bound,
                    "form {form} n={n}: kept queries computed {} goals over n=2",
                    k - k2
                );
            }
        }
    }
}
