//! Program families around the borrow-checker constructs that once hid soundness holes: one
//! single-file program per `(class, seed)`, the same text for the same pair on every host.
//!
//! A program is a shared prelude (views, helpers, a storing trait, a `dispatch fn`, a type with
//! `Drop`) plus one to three `case` fns that `main` calls and prints. Each case puts one holder of
//! a borrow (a reference, a view, a closure, an array of references, a view stored into through a
//! `from` clause, ...) of a referent, then uses and invalidates the referent (a write, a move, a
//! drop, the end of its block) in some control-flow frame. The verdict is known by construction:
//! a case is bad when a use of the holder may follow the invalidation on some path, and a program
//! with a bad case must be rejected; a program whose cases are all good must be accepted. A few
//! shapes are bad for a fixed rule instead (`[e; n]` of a non-Copy `e`, a closure returning a
//! borrow of a capture it changes, a store a `from` clause does not declare).
//!
//! The `generic-sites` class (module `sites`) has its own prelude and cases: a bounded generic
//! item instantiated at one site kind, bad iff the type argument does not meet the bound.

use crate::rng::Rng;
use std::fmt::Write as _;

mod sites;

pub use sites::RULE_CODES;

/// One family of programs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Class {
    /// Closures capturing by `&` / `&mut`, coerced to other fn types, passed and held in views.
    ClosureCoerce,
    /// Casts of references and views to their own type.
    RefCast,
    /// Borrows of temporaries whose expression is evaluated again in a loop.
    TempLoop,
    /// `[e; n]` with Copy and non-Copy elements, arrays of references.
    ArrayRepeat,
    /// Views built `from` / `from copy` several origins, reassigned in branches and loops.
    ViewOrigins,
    /// Calls of fns with parameter `from` clauses: plain, generic, trait method, `dispatch`.
    ParamFrom,
    /// Moves and drops interleaved with live borrows.
    MoveDrop,
    /// Statement-level loan liveness: uses before / after writes, branches, `break`,
    /// `continue`, early `return`, straight-line sequences of several holders.
    Liveness,
    /// A generic item with a bound instantiated at one site kind with a type argument that
    /// fits the bound or not.
    GenericSites,
}

impl Class {
    pub const ALL: [Class; 9] = [
        Class::ClosureCoerce,
        Class::RefCast,
        Class::TempLoop,
        Class::ArrayRepeat,
        Class::ViewOrigins,
        Class::ParamFrom,
        Class::MoveDrop,
        Class::Liveness,
        Class::GenericSites,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Class::ClosureCoerce => "closure-coerce",
            Class::RefCast => "ref-cast",
            Class::TempLoop => "temp-loop",
            Class::ArrayRepeat => "array-repeat",
            Class::ViewOrigins => "view-origins",
            Class::ParamFrom => "param-from",
            Class::MoveDrop => "move-drop",
            Class::Liveness => "liveness",
            Class::GenericSites => "generic-sites",
        }
    }

    pub fn parse(s: &str) -> Option<Class> {
        Class::ALL.into_iter().find(|c| c.name() == s)
    }
}

/// The verdict a program must get from the checker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Accept,
    Reject,
}

/// One generated program.
#[derive(Clone, Debug)]
pub struct Program {
    pub class: Class,
    pub seed: u64,
    pub text: String,
    pub expect: Verdict,
    /// Diagnostic codes a rejection is expected to carry (one of them), for a bad program.
    pub codes: Vec<&'static str>,
    /// One label per case: shape, frame, `good` / `bad`.
    pub cases: Vec<String>,
}

const PRELUDE: &str = "import std.io;
import std.mem;
import std.vec;

view struct Pick { best: &u32, count: u32 }

view struct Hold { f: fn() -> u32 }

view struct Two { a: &u32, b: &u32 }

view struct Outer { inner: Pick, n: u32 }

view struct Slot { r: &mut Pick }

struct Pair { a: u32, b: u32 }

struct G { id: u32 }

impl Drop for G {
    fn drop(&mut self) {
        io.print(f\"drop {self.id}\\n\");
    }
}

impl Pair {
    fn first(&self) -> &u32 from self {
        return &self.a;
    }
}

trait Sink {
    fn put(&mut self from x, x: &u32);
}

impl Sink for Pick {
    fn put(&mut self from x, x: &u32) {
        self.best = x;
        self.count = self.count +% 1;
    }
}

fn next(i: u32) -> u32 {
    return i +% 1;
}

fn pair(i: u32) -> Pair {
    return Pair.{ a: i, b: i +% 1 };
}

fn arr(i: u32) -> [u32; 2] {
    return [i, i +% 1];
}

fn tup(i: u32) -> (u32, u32) {
    return (i, i +% 2);
}

fn mkg(i: u32) -> G {
    return G.{ id: i };
}

fn read(p: &u32) -> u32 {
    return *p;
}

fn bump(p: &mut u32) {
    *p = *p +% 1;
}

fn consume(g: G) -> u32 {
    return g.id;
}

fn apply(f: fn() -> u32) -> u32 {
    return f();
}

fn keep(a: &u32, b: &u32) -> &u32 from a + b {
    return if *a > *b then a else b;
}

fn mk(a: &u32, b: &u32, c: bool) -> Pick from a + b {
    return Pick.{ best: if c then a else b, count: 0 };
}

fn best(p: &Pick) -> &u32 from copy p {
    return p.best;
}

fn set(dst: &mut Pick from a, a: &u32) {
    dst.best = a;
}

fn set2(dst: &mut Pick from a + copy b, a: &u32, b: &Pick) {
    if *a > *b.best {
        dst.best = a;
    } else {
        dst.best = b.best;
    }
}

fn feed<V: view + Sink>(t: &mut V from x, x: &u32) {
    t.put(x);
}

fn set_t(dst: &mut (Pick, u32) from a, a: &u32) {
    dst.0.best = a;
    dst.1 = dst.1 +% 1;
}

fn set_a(dst: &mut [Pick; 2] from a, a: &u32) {
    dst[1].best = a;
}

fn set_o(dst: &mut Option<&u32> from a, a: &u32) {
    *dst = Option.Some(a);
}

fn set_c(dst: &mut Pick from copy b, b: &Pick) {
    dst.best = b.best;
}

fn set_get(dst: &mut Pick from a, a: &u32) -> &u32 from copy dst {
    dst.best = a;
    return dst.best;
}

fn bump_slot(s: Slot) {
    s.r.count = s.r.count +% 1;
}

fn callr(f: fn() -> &u32) -> &u32 from f {
    return f();
}

fn opt(o: Option<&u32>) -> u32 {
    return match o { Option.Some(q) => *q, Option.None => 0u32 };
}

fn put_base(dst: &mut Pick from x, x: &u32) {
    dst.best = x;
}

dispatch fn put_any(dst: &mut Pick from x, x: &u32) { _ => put_base, }
";

/// The referent a holder borrows; every case declares all three.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Referent {
    /// `x: u32`
    X,
    /// `g: G` (has `Drop`)
    G,
    /// `v: vec.Vec<u32>`
    V,
}

/// What ends the referent's value for the holder.
#[derive(Clone)]
struct Inval {
    text: String,
    /// A move: the referent is gone after it, so it may not run in a loop body that goes on,
    /// nor before the holder borrows it.
    moves: bool,
    /// Targets an unrelated local: never conflicts.
    unrelated: bool,
}

/// A holder of a borrow of a referent.
struct Holder {
    name: &'static str,
    /// Statements that make the holder.
    setup: Vec<String>,
    /// A `u32` expression that uses the holder.
    use_: String,
    referent: Referent,
    /// For the scope frame: the holder declared with a borrow of `zero`, and statements that
    /// make it borrow `w`, a local of an inner block.
    scoped: Option<(Vec<String>, Vec<String>)>,
}

/// Indented text of one fn body.
struct Body {
    out: String,
    ind: usize,
}

impl Body {
    fn new() -> Body {
        Body { out: String::new(), ind: 1 }
    }

    fn line(&mut self, s: &str) {
        for _ in 0..self.ind {
            self.out.push_str("    ");
        }
        self.out.push_str(s);
        self.out.push('\n');
    }

    fn lines(&mut self, ls: &[String]) {
        for l in ls {
            self.line(l);
        }
    }

    fn open(&mut self, s: &str) {
        self.line(&format!("{s} {{"));
        self.ind += 1;
    }

    fn close(&mut self) {
        self.ind -= 1;
        self.line("}");
    }

    /// `s = s +% <e>;`
    fn add(&mut self, e: &str) {
        self.line(&format!("s = s +% {e};"));
    }
}

/// One generated case fn.
struct Case {
    items: String,
    body: String,
    label: String,
    bad: bool,
    codes: &'static [&'static str],
}

fn v(r: &mut Rng) -> u32 {
    r.range(1, 40)
}

fn pick<'a, T>(r: &mut Rng, xs: &'a [T]) -> &'a T {
    &xs[r.below(xs.len() as u32) as usize]
}

fn s(x: &str) -> String {
    x.to_string()
}

fn inval(r: &mut Rng, referent: Referent, allow_move: bool) -> Inval {
    let k = v(r);
    if r.chance(150) {
        let text = match r.below(2) {
            0 => format!("z = z +% {k};"),
            _ => s("bump(&mut z);"),
        };
        return Inval { text, moves: false, unrelated: true };
    }
    let (text, moves) = match referent {
        Referent::X => (
            match r.below(4) {
                0 => format!("x = seed +% {k};"),
                1 => format!("x +%= {k};"),
                2 => s("bump(&mut x);"),
                _ => format!("{{ let m: &mut u32 = &mut x; *m = {k}; }}"),
            },
            false,
        ),
        Referent::G => match r.below(if allow_move { 5 } else { 2 }) {
            0 => (format!("g = G.{{ id: {k} }};"), false),
            1 => (format!("g.id = {k};"), false),
            2 => (s("mem.drop(g);"), true),
            3 => (s("s = s +% consume(g);"), true),
            _ => (s("let moved = g;"), true),
        },
        Referent::V => match r.below(if allow_move { 3 } else { 1 }) {
            0 => (format!("v.push({k});"), false),
            1 => (s("mem.drop(v);"), true),
            _ => (s("let moved = v;"), true),
        },
    };
    Inval { text, moves, unrelated: false }
}

fn holder(class: Class, r: &mut Rng) -> Holder {
    let k = v(r);
    let c = format!("seed > {}", v(r));
    let wa = if r.chance(500) { " without(alloc)" } else { "" };
    let n = r.range(2, 5);
    let i = r.below(n);
    let x = Referent::X;
    let h = |name, setup: Vec<String>, use_: &str, referent, scoped| Holder {
        name,
        setup,
        use_: s(use_),
        referent,
        scoped,
    };
    match class {
        Class::ClosureCoerce => match r.below(8) {
            6 => h(
                "closure-through-fn",
                vec![s("let r: &u32 = callr(fn() -> &u32 { return &x; });")],
                pick(r, &["*r", "read(r)"]),
                x,
                None,
            ),
            7 => h(
                "closure-of-view",
                vec![
                    format!("let p = mk(&x, &y, {c});"),
                    s("let f: fn() -> &u32 = fn() -> &u32 { return p.best; };"),
                    s("let r: &u32 = f();"),
                ],
                "*r",
                x,
                None,
            ),
            0 => h(
                "closure",
                vec![format!("let f: fn() -> u32{wa} = fn() -> u32 {{ return x +% {k}; }};")],
                pick(r, &["f()", "apply(f)"]),
                x,
                Some((
                    vec![s("let mut f: fn() -> u32 = fn() -> u32 { return zero; };")],
                    vec![format!("f = fn() -> u32 {{ return w +% {k}; }};")],
                )),
            ),
            1 => h(
                "closure-narrowed",
                vec![
                    format!("let f0: fn() -> u32 without(alloc) = fn() -> u32 {{ return x +% {k}; }};"),
                    s("let f: fn() -> u32 = f0;"),
                ],
                pick(r, &["f()", "apply(f)"]),
                x,
                Some((
                    vec![s("let mut f: fn() -> u32 = fn() -> u32 { return zero; };")],
                    vec![s("let f0: fn() -> u32 without(alloc) = fn() -> u32 { return w; };"), s("f = f0;")],
                )),
            ),
            2 => h(
                "closure-in-view",
                vec![format!("let h = Hold.{{ f: fn() -> u32 {{ return x +% {k}; }} }};")],
                "apply(h.f)",
                x,
                Some((
                    vec![s("let mut h = Hold.{ f: fn() -> u32 { return zero; } };")],
                    vec![s("h = Hold.{ f: fn() -> u32 { return w; } };")],
                )),
            ),
            3 => h(
                "closure-returns-ref",
                vec![s("let f: fn() -> &u32 = fn() -> &u32 { return &x; };"), s("let r: &u32 = f();")],
                "*r",
                x,
                None,
            ),
            4 => h(
                "closure-mut-capture",
                vec![format!("let f: fn() -> u32 = fn() -> u32 {{ x = x +% {k}; return x; }};")],
                pick(r, &["f()", "apply(f)"]),
                x,
                None,
            ),
            _ => h(
                "closure-of-g",
                vec![s("let f: fn() -> u32 = fn() -> u32 { return g.id; };")],
                pick(r, &["f()", "apply(f)"]),
                Referent::G,
                None,
            ),
        },
        Class::RefCast => match r.below(6) {
            0 => h(
                "ref-cast",
                vec![s("let r0: &u32 = &x;"), s("let r: &u32 = r0 as &u32;")],
                "*r",
                x,
                Some((
                    vec![s("let mut r: &u32 = &zero;")],
                    vec![s("let a: &u32 = &w;"), s("r = a as &u32;")],
                )),
            ),
            1 => h(
                "ref-cast-direct",
                vec![s("let r: &u32 = &x as &u32;")],
                pick(r, &["*r", "read(r)"]),
                x,
                Some((vec![s("let mut r: &u32 = &zero as &u32;")], vec![s("r = &w as &u32;")])),
            ),
            2 => h(
                "ref-cast-into-view",
                vec![s("let r0: &u32 = &x;"), format!("let p = Pick.{{ best: r0 as &u32, count: {k} }};")],
                "*p.best",
                x,
                Some((
                    vec![s("let mut p = Pick.{ best: &zero, count: 0 };")],
                    vec![s("let a: &u32 = &w;"), s("p.best = a as &u32;")],
                )),
            ),
            3 => h(
                "mut-cast",
                vec![s("let m0: &mut u32 = &mut x;"), s("let m: &mut u32 = m0 as &mut u32;")],
                "*m",
                x,
                None,
            ),
            4 => h("mut-cast-direct", vec![s("let m: &mut u32 = &mut x as &mut u32;")], "*m", x, None),
            _ => h(
                "ref-cast-g",
                vec![s("let r0: &u32 = &g.id;"), s("let r: &u32 = r0 as &u32;")],
                "*r",
                Referent::G,
                None,
            ),
        },
        Class::ArrayRepeat => match r.below(3) {
            0 => h(
                "repeat-ref",
                vec![format!("let a: [&u32; {n}] = [&x; {n}];")],
                &format!("*a[{i}]"),
                x,
                Some((
                    vec![format!("let mut a: [&u32; {n}] = [&zero; {n}];")],
                    vec![format!("a = [&w; {n}];")],
                )),
            ),
            1 => h(
                "repeat-tuple",
                vec![format!("let a: [(&u32, u32); {n}] = [(&x, {k}); {n}];")],
                &format!("*a[{i}].0"),
                x,
                None,
            ),
            _ => h(
                "repeat-ref-g",
                vec![format!("let a: [&u32; {n}] = [&g.id; {n}];")],
                &format!("*a[{i}]"),
                Referent::G,
                None,
            ),
        },
        Class::ViewOrigins => match r.below(8) {
            5 => h(
                "two-swap",
                vec![
                    s("let mut t = Two.{ a: &zero, b: &zero };"),
                    s("t.b = &x;"),
                    s("t = Two.{ a: t.b, b: t.a };"),
                ],
                pick(r, &["*t.a", "*t.a +% *t.b"]),
                x,
                Some((vec![s("let mut t = Two.{ a: &zero, b: &zero };")], vec![s("t.a = &w;")])),
            ),
            6 => h(
                "nested-view",
                vec![format!("let o = Outer.{{ inner: mk(&y, &x, {c}), n: {k} }};")],
                "*o.inner.best",
                x,
                Some((
                    vec![s("let mut o = Outer.{ inner: Pick.{ best: &zero, count: 0 }, n: 0 };")],
                    vec![s("o.inner = Pick.{ best: &w, count: 1 };")],
                )),
            ),
            7 => h(
                "through-pointer",
                vec![
                    s("let mut p = Pick.{ best: &zero, count: 0 };"),
                    s("let mut q = Pick.{ best: &zero, count: 0 };"),
                    format!("{{ let pp: &mut Pick = if {c} then &mut p else &mut q; pp.best = &x; }}"),
                ],
                pick(r, &["*p.best", "*q.best", "*best(&q)"]),
                x,
                Some((
                    vec![
                        s("let mut p = Pick.{ best: &zero, count: 0 };"),
                        s("let mut q = Pick.{ best: &zero, count: 0 };"),
                    ],
                    vec![
                        format!("let pp: &mut Pick = if {c} then &mut q else &mut p;"),
                        s("*pp = Pick.{ best: &w, count: 3 };"),
                    ],
                )),
            ),
            0 => h("mk", vec![format!("let p = mk(&x, &y, {c});")], "*p.best", x, None),
            1 => h(
                "mk-copy",
                vec![format!("let p = mk(&y, &x, {c});"), s("let q: &u32 = best(&p);")],
                "*q",
                x,
                Some((
                    vec![s("let mut q: &u32 = &zero;")],
                    vec![s("let p = mk(&w, &zero, true);"), s("q = best(&p);")],
                )),
            ),
            2 => h(
                "branch-reassign",
                vec![
                    s("let mut p = Pick.{ best: &zero, count: 0 };"),
                    format!(
                        "if {c} {{ p = Pick.{{ best: &x, count: 1 }}; }} else {{ p = mk(&y, &zero, true); }}"
                    ),
                ],
                "*p.best",
                x,
                None,
            ),
            3 => h(
                "loop-reassign",
                vec![
                    s("let mut p = Pick.{ best: &zero, count: 0 };"),
                    s("let mut j: u32 = 0;"),
                    format!("while j < {n} {{ if j == {i} {{ p = mk(&y, &x, j > 0); }} j = j +% 1; }}"),
                ],
                "*p.best",
                x,
                Some((vec![s("let mut p = Pick.{ best: &zero, count: 0 };")], vec![s("p.best = &w;")])),
            ),
            _ => h(
                "field-store",
                vec![s("let mut p = Pick.{ best: &zero, count: 0 };"), s("p.best = &x;")],
                pick(r, &["*p.best", "*best(&p)"]),
                x,
                None,
            ),
        },
        Class::ParamFrom => {
            let base = s("let mut p = Pick.{ best: &zero, count: 0 };");
            let use_ = pick(r, &["*p.best", "*best(&p)", "read(p.best)"]);
            let (name, store): (&'static str, Vec<String>) = match r.below(13) {
                7 => (
                    "tuple",
                    vec![
                        s("let mut t = (Pick.{ best: &zero, count: 0 }, 0u32);"),
                        s("set_t(&mut t, &x);"),
                        s("let (tp, _) = t;"),
                        s("p = tp;"),
                    ],
                ),
                8 => (
                    "array",
                    vec![
                        s("let mut a = [Pick.{ best: &zero, count: 0 }, Pick.{ best: &zero, count: 1 }];"),
                        s("set_a(&mut a, &x);"),
                        s("p.best = a[1].best;"),
                    ],
                ),
                9 => (
                    "option",
                    vec![
                        s("let mut o: Option<&u32> = Option.None;"),
                        s("set_o(&mut o, &x);"),
                        s("p.best = match o { Option.Some(q) => q, Option.None => &zero };"),
                    ],
                ),
                10 => (
                    "copy-source",
                    vec![s("let q0 = Pick.{ best: &x, count: 0 };"), s("set_c(&mut p, &q0);")],
                ),
                11 => ("storing-result", vec![s("let q: &u32 = set_get(&mut p, &x);"), s("p.best = q;")]),
                12 => ("slot", vec![s("set(&mut p, &x);"), s("bump_slot(Slot.{ r: &mut p });")]),
                0 => ("set", vec![s("set(&mut p, &x);")]),
                1 => {
                    ("set2-a", vec![s("let q0 = Pick.{ best: &y, count: 0 };"), s("set2(&mut p, &x, &q0);")])
                }
                2 => (
                    "set2-copy",
                    vec![s("let q0 = Pick.{ best: &x, count: 0 };"), s("set2(&mut p, &y, &q0);")],
                ),
                3 => ("trait-method", vec![s("p.put(&x);")]),
                4 => ("generic", vec![s("feed(&mut p, &x);")]),
                5 => ("dispatch", vec![s("put_any(&mut p, &x);")]),
                _ => ("two-level", vec![s("{ let pp: &mut Pick = &mut p; set(pp, &x); }")]),
            };
            let mut setup = vec![base.clone()];
            setup.extend(store);
            Holder {
                name,
                setup,
                use_: s(use_),
                referent: x,
                scoped: Some((vec![base], vec![s("set(&mut p, &w);")])),
            }
        }
        Class::MoveDrop => match r.below(4) {
            0 => {
                h("ref-of-g", vec![s("let r: &u32 = &g.id;")], pick(r, &["*r", "read(r)"]), Referent::G, None)
            }
            1 => h(
                "view-of-g",
                vec![format!("let p = Pick.{{ best: &g.id, count: {k} }};")],
                "*p.best",
                Referent::G,
                None,
            ),
            2 => h("vec-get-ref", vec![s("let r: &u32 = v.get_ref(0);")], "*r", Referent::V, None),
            _ => h(
                "closure-of-g",
                vec![s("let f: fn() -> u32 = fn() -> u32 { return g.id; };")],
                "f()",
                Referent::G,
                None,
            ),
        },
        // `generic-sites` makes its cases without holders.
        Class::Liveness | Class::TempLoop | Class::GenericSites => match r.below(3) {
            0 => h(
                "ref",
                vec![s("let r: &u32 = &x;")],
                pick(r, &["*r", "read(r)"]),
                x,
                Some((vec![s("let mut r: &u32 = &zero;")], vec![s("r = &w;")])),
            ),
            1 => h("mut-ref", vec![s("let m: &mut u32 = &mut x;")], "*m", x, None),
            _ => h(
                "view",
                vec![format!("let p = Pick.{{ best: &x, count: {k} }};")],
                "*p.best",
                x,
                Some((
                    vec![s("let mut p = Pick.{ best: &zero, count: 0 };")],
                    vec![s("p = Pick.{ best: &w, count: 2 };")],
                )),
            ),
        },
    }
}

/// The case fn's head: the referents, an unrelated local, the accumulator.
fn head(b: &mut Body, k: u32) {
    b.line("let zero: u32 = 0;");
    b.line("let mut s: u32 = 0;");
    b.line(&format!("let mut x: u32 = seed +% {k};"));
    b.line("let y: u32 = seed +% 2;");
    b.line("let mut z: u32 = seed;");
    b.line(&format!("let mut g = G.{{ id: seed +% {k} }};"));
    b.line("let mut v = vec.Vec::<u32>.new();");
    b.line("v.push(seed);");
}

/// Places `h` and an invalidation in a control-flow frame; `bad` asks for a use that may follow
/// the invalidation. Returns the frame's name and whether the case really is bad.
fn frame(r: &mut Rng, b: &mut Body, h: &Holder, want_bad: bool) -> (String, bool) {
    let n = r.range(2, 5);
    let k = r.below(n);
    let c = format!("seed > {}", v(r));
    let mut kinds: Vec<&str> = vec!["straight", "branch", "break", "return"];
    if h.scoped.is_some() {
        kinds.push("scope");
    }
    let kind = *pick(r, &kinds);
    let kind = if !want_bad && r.chance(150) { "write-first" } else { kind };
    let kind = if r.chance(150) { "continue" } else { kind };
    let w = inval(r, h.referent, !matches!(kind, "continue" | "write-first"));
    let bad = want_bad && !w.unrelated;
    let u = h.use_.as_str();
    match kind {
        "straight" => {
            b.lines(&h.setup);
            let pre = !bad || r.chance(500);
            if pre {
                b.add(u);
            }
            b.line(&w.text);
            if want_bad {
                b.add(u);
            }
        }
        "branch" => {
            b.lines(&h.setup);
            b.open(&format!("if {c}"));
            b.line(&w.text);
            b.ind -= 1;
            b.open("} else");
            b.add(u);
            b.close();
            if want_bad {
                b.add(u);
            }
        }
        "break" => {
            b.lines(&h.setup);
            b.line("let mut i: u32 = 0;");
            b.open("loop");
            b.open(&format!("if i == {k}"));
            b.line(&w.text);
            b.line("break;");
            b.close();
            b.add(u);
            b.line("i = i +% 1;");
            b.close();
            if want_bad {
                b.add(u);
            }
        }
        "return" => {
            b.lines(&h.setup);
            b.open(&format!("if {c}"));
            b.line(&w.text);
            if want_bad {
                b.add(u);
            }
            b.line("return s;");
            b.close();
            b.add(u);
        }
        "continue" => {
            if want_bad {
                b.lines(&h.setup);
            }
            b.line("let mut i: u32 = 0;");
            b.open(&format!("while i < {n}"));
            b.line("i = i +% 1;");
            if want_bad {
                b.add(u);
            } else {
                b.lines(&h.setup);
            }
            b.open(&format!("if i == {k}"));
            b.line(&w.text);
            b.line("continue;");
            b.close();
            if !want_bad {
                b.add(u);
            }
            b.close();
        }
        "write-first" => {
            b.line(&w.text);
            b.lines(&h.setup);
            b.add(u);
        }
        _ => {
            let (decl, repoint) = h.scoped.as_ref().expect("scope frame needs a scoped holder");
            b.lines(decl);
            b.line("{");
            b.ind += 1;
            b.line(&format!("let w: u32 = seed +% {};", v(r)));
            b.lines(repoint);
            if !want_bad || r.chance(500) {
                b.add(u);
            }
            b.close();
            if want_bad {
                b.add(u);
            }
            return (s("scope"), want_bad);
        }
    }
    let target = if w.unrelated {
        "-unrelated"
    } else if w.moves {
        "-move"
    } else {
        ""
    };
    (format!("{kind}{target}"), bad)
}

/// A case of `class` built from a holder in a frame.
fn framed(class: Class, r: &mut Rng, idx: usize, want_bad: bool) -> Case {
    let mut b = Body::new();
    head(&mut b, r.range(1, 50));
    let mut h = holder(class, r);
    let mut name = h.name.to_string();
    if r.chance(200) {
        // a closure over the holder: it keeps every borrow the holder keeps
        h.setup.push(format!("let fw: fn() -> u32 = fn() -> u32 {{ return {}; }};", h.use_));
        h.use_ = s(if r.chance(500) { "fw()" } else { "apply(fw)" });
        h.scoped = None;
        name.push_str("+closure");
    }
    let (f, bad) = frame(r, &mut b, &h, want_bad);
    let codes: &'static [&'static str] = if bad { &["E0501", "E0502"] } else { &[] };
    finish(idx, b, String::new(), format!("{name}/{f}"), bad, codes)
}

fn finish(
    idx: usize,
    mut b: Body,
    items: String,
    label: String,
    bad: bool,
    codes: &'static [&'static str],
) -> Case {
    b.line("return s;");
    let body = format!("fn case{idx}(seed: u32) -> u32 {{\n{}}}\n", b.out);
    Case { items, body, label: format!("{label}/{}", if bad { "bad" } else { "good" }), bad, codes }
}

/// Straight-line sequences of setups, uses, and writes over several holders; bad iff, between
/// the setup of a holder and its last use, an origin of it is written, or, for a `&mut` holder,
/// another holder of a common origin is set up or used, or a `&mut` holder of a common origin
/// is set up while a shared one lives.
fn sequence(r: &mut Rng, idx: usize, want_bad: bool) -> Case {
    for _ in 0..64 {
        let mut r2 = Rng::derive(r.next_u64(), 1, 1);
        let r = &mut r2;
        // (setup, origins, use, exclusive)
        let holders: Vec<(String, Vec<&str>, String, bool)> = (0..r.range(2, 4))
            .map(|j| match r.below(6) {
                0 => (format!("let r{j}: &u32 = &x;"), vec!["x"], format!("*r{j}"), false),
                1 => (
                    format!("let p{j} = mk(&x, &y, seed > 9);"),
                    vec!["x", "y"],
                    format!("*p{j}.best"),
                    false,
                ),
                2 => (
                    format!("let f{j}: fn() -> u32 = fn() -> u32 {{ return z +% 1; }};"),
                    vec!["z"],
                    format!("f{j}()"),
                    false,
                ),
                3 => (format!("let a{j}: [&u32; 2] = [&z; 2];"), vec!["z"], format!("*a{j}[1]"), false),
                4 => (format!("let m{j}: &mut u32 = &mut z;"), vec!["z"], format!("*m{j}"), true),
                _ => (format!("let n{j}: &mut u32 = &mut x;"), vec!["x"], format!("*n{j}"), true),
            })
            .collect();
        let mut events: Vec<(u8, usize)> = (0..holders.len()).map(|j| (0, j)).collect();
        for _ in 0..r.range(3, 7) {
            match r.below(2) {
                0 => events.push((1, r.below(holders.len() as u32) as usize)),
                _ => events.push((2, r.below(2) as usize)),
            }
        }
        // shuffle, then move each setup before the first use of its holder
        for i in (1..events.len()).rev() {
            let j = r.below(i as u32 + 1) as usize;
            events.swap(i, j);
        }
        let mut ordered: Vec<(u8, usize)> = Vec::new();
        for &(e, j) in &events {
            if e == 1 && !ordered.contains(&(0, j)) {
                ordered.push((0, j));
            }
            if !(e == 0 && ordered.contains(&(0, j))) {
                ordered.push((e, j));
            }
        }
        let wnames = ["x", "z"];
        let mut bad = false;
        for (h, hd) in holders.iter().enumerate() {
            let set = ordered.iter().position(|&ev| ev == (0, h));
            let last = ordered.iter().rposition(|&ev| ev == (1, h));
            let (Some(set), Some(last)) = (set, last) else { continue };
            for &(e, t) in &ordered[set + 1..last] {
                let common = |o: &Vec<&str>| o.iter().any(|n| hd.1.contains(n));
                bad |= match e {
                    2 => hd.1.contains(&wnames[t]),
                    _ => t != h && common(&holders[t].1) && (hd.3 || (e == 0 && holders[t].3)),
                };
            }
        }
        if bad != want_bad {
            continue;
        }
        let mut b = Body::new();
        head(&mut b, r.range(1, 50));
        let mut labels = String::new();
        for &(e, j) in &ordered {
            match e {
                0 => b.line(&holders[j].0),
                1 => b.add(&holders[j].2),
                _ => b.line(&format!("{} = {} +% {};", wnames[j], wnames[j], v(r))),
            }
            labels.push(['s', 'u', 'w'][e as usize]);
        }
        let codes: &'static [&'static str] = if bad { &["E0501"] } else { &[] };
        return finish(idx, b, String::new(), format!("sequence-{labels}/straight"), bad, codes);
    }
    framed(Class::Liveness, r, idx, want_bad)
}

/// Borrows of temporaries in loops.
fn temp_loop(r: &mut Rng, idx: usize, want_bad: bool) -> Case {
    let n = r.range(2, 5);
    let c = format!("seed > {}", v(r));
    let mut b = Body::new();
    head(&mut b, r.range(1, 50));
    let shape = r.below(4);
    if shape == 3 {
        // a temporary with a drop lives until its statement ends
        if want_bad {
            b.line("let r: &u32 = &mkg(seed).id;");
            b.add("*r");
        } else {
            b.add(pick(r, &["read(&mkg(seed).id)", "mkg(seed).id"]));
        }
        return finish(idx, b, String::new(), s("drop-temp"), want_bad, &["E0501"]);
    }
    if shape == 2 && want_bad {
        b.line("let mut best: &u32 = &zero;");
        b.line("let mut i: u32 = 0;");
        b.open(&format!("while i < {n}"));
        b.line("best = keep(best, &next(i));");
        b.line("i = i +% 1;");
        b.close();
        b.add("*best");
        return finish(idx, b, String::new(), s("call-arg"), true, &["E0501"]);
    }
    let temp = pick(
        r,
        &[
            "&next(i)",
            "pair(i).first()",
            "&pair(i).b",
            "&tup(i).1",
            "&arr(i)[1]",
            "&(i +% 1)",
            "&(if seed > 3 then i else 7u32)",
        ],
    );
    b.line("let mut r: &u32 = &zero;");
    b.line("let mut i: u32 = 0;");
    let looph = if r.chance(500) { format!("while i < {n}") } else { s("loop") };
    let is_loop = looph == "loop";
    b.open(&looph);
    if is_loop {
        b.line(&format!("if i == {n} {{ break; }}"));
    }
    b.line("let before: &u32 = r;");
    if !want_bad {
        b.add("*before");
    }
    if r.chance(300) {
        b.line(&format!("if {c} {{ r = {temp}; }}"));
    } else {
        b.line(&format!("r = {temp};"));
    }
    if want_bad {
        b.add("*before");
    } else if r.chance(500) {
        b.add("*r");
    }
    b.line("i = i +% 1;");
    b.close();
    b.add("*r");
    finish(idx, b, String::new(), format!("temp {temp}"), want_bad, &["E0501"])
}

/// Shapes bad or good for a fixed rule of the class.
fn fixed(class: Class, r: &mut Rng, idx: usize, want_bad: bool) -> Option<Case> {
    let mut b = Body::new();
    head(&mut b, r.range(1, 50));
    let k = v(r);
    let n = r.range(2, 4);
    let (label, items, codes): (&str, String, &'static [&'static str]) = match class {
        Class::ArrayRepeat => {
            if want_bad {
                match r.below(4) {
                    0 => b.line(&format!("let a = [&mut x; {n}];")),
                    1 => b.line(&format!("let a = [g; {n}];")),
                    2 => b.line(&format!("let a = [v; {n}];")),
                    _ => b.line("let a = [g; 0];"),
                }
                b.add("a.len() as% u32");
                ("repeat-non-copy", String::new(), &["E0418"])
            } else {
                match r.below(3) {
                    0 => {
                        b.line("let a = [g; 1];");
                        b.add("a[0].id");
                    }
                    1 => {
                        b.line(&format!("let a: [u32; {n}] = [seed +% {k}; {n}];"));
                        b.line(&format!("for i in 0usize..{n}usize {{ s = s +% a[i]; }}"));
                    }
                    _ => {
                        b.line(&format!("let a: [(u32, u32); {n}] = [(seed, {k}); {n}];"));
                        b.add("a[1].0 +% a[0].1");
                    }
                }
                ("repeat-copy", String::new(), &[])
            }
        }
        Class::ClosureCoerce => {
            if want_bad {
                b.line(&format!("let f = fn() -> &u32 {{ x = x +% {k}; return &x; }};"));
                b.line("let r: &u32 = f();");
                b.add("*r");
                ("closure-ref-of-mut-capture", String::new(), &["E0512"])
            } else {
                b.line(&format!("let f = fn() -> u32 {{ x = x +% {k}; return x; }};"));
                b.add("f()");
                b.add("f()");
                b.add("x");
                ("closure-mut-then-read", String::new(), &[])
            }
        }
        Class::RefCast if r.chance(500) => {
            // `as` to its own type exists for scalars, vectors, masks, references, raw pointers
            if want_bad {
                match r.below(5) {
                    0 => b.line("let p = Pick.{ best: &x, count: 1 } as Pick;"),
                    1 => b.line(&format!("let p: (u32, u32) = (seed, {k}u32) as (u32, u32);")),
                    2 => b.line("let p: Option<&u32> = Option.Some(&x) as Option<&u32>;"),
                    3 => b.line("let p = G.{ id: seed } as G;"),
                    _ => b.line("let f0: fn(u32) -> u32 = next; let p = f0 as fn(u32) -> u32;"),
                }
            } else {
                b.line(&format!("let n: u32 = (seed +% {k}) as u32;"));
                b.line("let r: &u32 = &x as &u32;");
                b.add("n +% *r");
            }
            ("identity-cast-kind", String::new(), &["E0204"])
        }
        Class::RefCast => {
            b.line("let m0: &mut u32 = &mut x;");
            b.line("let m: &mut u32 = m0 as &mut u32;");
            b.line(&format!("*m = {k};"));
            if want_bad {
                b.line("*m0 = 1;");
            }
            b.add("x");
            ("mut-cast-source", String::new(), &["E0502"])
        }
        Class::ParamFrom if r.chance(400) => {
            b.line("let mut p = Pick.{ best: &zero, count: 0 };");
            let items = if want_bad {
                format!("fn via_slot{idx}(s: Slot, a: &u32) {{\n    s.r.best = a;\n}}\n")
            } else {
                format!("fn via_slot{idx}(s: Slot, a: &u32) {{\n    s.r.count = s.r.count +% *a;\n}}\n")
            };
            b.line(&format!("via_slot{idx}(Slot.{{ r: &mut p }}, &x);"));
            b.add("*p.best");
            ("by-value-view-store", items, &["E0513"])
        }
        Class::ParamFrom => {
            let items = if want_bad {
                format!("fn store{idx}(dst: &mut Pick, a: &u32) {{\n    dst.best = a;\n}}\n")
            } else {
                format!(
                    "fn store{idx}(dst: &mut Pick from a, a: &u32) {{\n    if *a > {k} {{\n        dst.best = a;\n    }}\n    dst.count = dst.count +% 1;\n}}\n"
                )
            };
            b.line("let mut p = Pick.{ best: &zero, count: 0 };");
            b.line(&format!("store{idx}(&mut p, &x);"));
            b.add("*p.best");
            ("callee-store", items, &["E0513"])
        }
        _ => return None,
    };
    Some(finish(idx, b, items, s(label), want_bad, codes))
}

/// The case `idx` of a program of `class`.
fn case(class: Class, r: &mut Rng, idx: usize, want_bad: bool) -> Case {
    match class {
        Class::GenericSites => sites::case(r, idx, want_bad),
        Class::TempLoop => {
            if r.chance(700) {
                temp_loop(r, idx, want_bad)
            } else {
                framed(Class::TempLoop, r, idx, want_bad)
            }
        }
        Class::Liveness => {
            if r.chance(400) {
                sequence(r, idx, want_bad)
            } else {
                framed(class, r, idx, want_bad)
            }
        }
        _ => {
            if r.chance(250)
                && let Some(c) = fixed(class, r, idx, want_bad)
            {
                return c;
            }
            framed(class, r, idx, want_bad)
        }
    }
}

/// The program of `(class, seed)`.
pub fn generate(class: Class, seed: u64) -> Program {
    let mut r = Rng::derive(seed, 0xB0_0C, class as u64);
    let ncases = r.range(1, 3) as usize;
    let bad_case = if r.chance(450) { Some(r.below(ncases as u32) as usize) } else { None };
    let cases: Vec<Case> = (0..ncases).map(|i| case(class, &mut r, i, bad_case == Some(i))).collect();
    let mut text = format!("// fuzz-classes {} seed {seed}\n", class.name());
    text.push_str(if class == Class::GenericSites { sites::PRELUDE } else { PRELUDE });
    for c in &cases {
        text.push('\n');
        if !c.items.is_empty() {
            text.push_str(&c.items);
            text.push('\n');
        }
        text.push_str(&c.body);
    }
    text.push_str("\nfn main() -> i32 {\n");
    let _ = writeln!(text, "    let seed: u32 = {};", r.range(0, 60));
    for i in 0..ncases {
        let _ = writeln!(text, "    let c{i}: u32 = case{i}(seed);");
        let _ = writeln!(text, "    io.print(f\"case{i} {{c{i}}}\\n\");");
    }
    text.push_str("    return 0;\n}\n");
    let bad = cases.iter().any(|c| c.bad);
    let codes = cases.iter().filter(|c| c.bad).flat_map(|c| c.codes.iter().copied()).collect();
    Program {
        class,
        seed,
        text,
        expect: if bad { Verdict::Reject } else { Verdict::Accept },
        codes,
        cases: cases.into_iter().map(|c| c.label).collect(),
    }
}
