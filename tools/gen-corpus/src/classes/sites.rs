//! The `generic-sites` class: one generic item per case (a fn, a struct with an inherent impl,
//! a trait impl for a generic struct, a trait method with its own generics, a generic struct
//! built in a literal or held in a field, or a generic item of std: `mem.forget`, `Cell.new`,
//! `[T].copy_from`) with a bound, instantiated at one site kind with one type argument. The
//! verdict is known by construction: the case is bad iff the argument lacks what the bound asks
//! (a view for a param without `view`, or a trait or kind the type does not have). Every site is
//! an instantiation, so every site must check the bound. One shape is a declaration rule
//! instead: an `impl Drop` for a type whose param is `view` is bad iff its own param is not.

use super::{Body, Case, finish};
use crate::rng::Rng;

/// The class's own prelude: the argument types and the user trait.
pub(super) const PRELUDE: &str = "import std.cell;
import std.io;
import std.mem;
import std.vec;

view struct Pick { best: &u32, count: u32 }

struct S { a: u32 }

struct P { a: u32 }

impl Copy for P {}

struct D { id: u32 }

impl Drop for D {
    fn drop(&mut self) {
        io.print(f\"drop {self.id}\\n\");
    }
}

trait Tr {
    fn val(&self) -> u32;
}

impl Tr for u32 {
    fn val(&self) -> u32 {
        return *self;
    }
}

impl Tr for P {
    fn val(&self) -> u32 {
        return self.a;
    }
}

impl Tr for Pick {
    fn val(&self) -> u32 {
        return *self.best +% self.count;
    }
}

fn next(i: u32) -> u32 {
    return i +% 1;
}

fn vec1(x: u32) -> vec.Vec<u32> {
    let mut v = vec.Vec::<u32>.new();
    v.push(x);
    return v;
}
";

/// Codes a rejection for an unmet bound or view kind may carry: the view kind (E0411 for a std
/// ADT, E0412 for `forget`, else E0414), an unsatisfied trait (E0901), no method or operator
/// impl applies (E0208, E0206), not Flat (E1407), not Frozen (E1410).
pub const RULE_CODES: &[&str] = &["E0414", "E0411", "E0412", "E0901", "E0208", "E0206", "E1407", "E1410"];

/// What a bound asks besides the view kind.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Req {
    None,
    /// The prelude's user trait `Tr`.
    Tr,
    Copy,
    Frozen,
    Flat,
    Sync,
    Send,
}

#[derive(Clone, Copy, Debug)]
struct Bound {
    view: bool,
    req: Req,
}

const REQS: [Req; 7] = [Req::None, Req::Tr, Req::Copy, Req::Frozen, Req::Flat, Req::Sync, Req::Send];

impl Bound {
    /// `<T: ..>` text after the param name: empty for no bound.
    fn text(self) -> String {
        let req = match self.req {
            Req::None => "",
            Req::Tr => "Tr",
            Req::Copy => "Copy",
            Req::Frozen => "Frozen",
            Req::Flat => "Flat",
            Req::Sync => "Sync",
            Req::Send => "Send",
        };
        match (self.view, req) {
            (false, "") => String::new(),
            (true, "") => ": view".to_string(),
            (false, r) => format!(": {r}"),
            (true, r) => format!(": view + {r}"),
        }
    }

    fn label(self) -> String {
        let t = self.text();
        if t.is_empty() { "none".to_string() } else { t[2..].replace(" + ", "+") }
    }
}

/// A type argument with no type arguments of its own.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Base {
    U32,
    /// A struct, not Copy, no trait.
    S,
    /// A Copy struct that has `Tr`.
    P,
    /// A struct with `Drop`.
    D,
    Ref,
    Mut,
    /// A view struct that has `Tr`.
    Pick,
    /// `(&u32, u32)`
    Tup,
    /// `Option<&u32>`
    Opt,
    /// `fn(u32) -> u32`
    Fn,
    Cell,
    Vec,
}

const BASES: [Base; 12] = [
    Base::U32,
    Base::S,
    Base::P,
    Base::D,
    Base::Ref,
    Base::Mut,
    Base::Pick,
    Base::Tup,
    Base::Opt,
    Base::Fn,
    Base::Cell,
    Base::Vec,
];

/// The kinds of a type, as LANG.md §1-§3 and std.core define them.
#[derive(Clone, Copy)]
struct Kinds {
    view: bool,
    copy: bool,
    frozen: bool,
    flat: bool,
    sync: bool,
    send: bool,
    tr: bool,
}

impl Base {
    fn ty(self) -> &'static str {
        match self {
            Base::U32 => "u32",
            Base::S => "S",
            Base::P => "P",
            Base::D => "D",
            Base::Ref => "&u32",
            Base::Mut => "&mut u32",
            Base::Pick => "Pick",
            Base::Tup => "(&u32, u32)",
            Base::Opt => "Option<&u32>",
            Base::Fn => "fn(u32) -> u32",
            Base::Cell => "cell.Cell<u32>",
            Base::Vec => "vec.Vec<u32>",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Base::U32 => "u32",
            Base::S => "struct",
            Base::P => "copy-struct",
            Base::D => "drop-struct",
            Base::Ref => "ref",
            Base::Mut => "mut-ref",
            Base::Pick => "view-struct",
            Base::Tup => "tuple-ref",
            Base::Opt => "option-ref",
            Base::Fn => "fn-value",
            Base::Cell => "cell",
            Base::Vec => "vec",
        }
    }

    /// A fresh value; `j` picks the referents (`x{j}`, `y{j}`, `fv{j}`).
    fn value(self, j: u32) -> String {
        match self {
            Base::U32 => format!("x{j}"),
            Base::S => format!("S.{{ a: x{j} }}"),
            Base::P => format!("P.{{ a: x{j} }}"),
            Base::D => format!("D.{{ id: x{j} }}"),
            Base::Ref => format!("&x{j}"),
            Base::Mut => format!("&mut y{j}"),
            Base::Pick => format!("Pick.{{ best: &x{j}, count: {j} }}"),
            Base::Tup => format!("(&x{j}, x{j})"),
            Base::Opt => format!("Option.Some(&x{j})"),
            Base::Fn => format!("fv{j}"),
            Base::Cell => format!("cell.Cell::<u32>.new(x{j})"),
            Base::Vec => format!("vec1(x{j})"),
        }
    }

    /// A reference, a fn type, a view struct, or a tuple / `Option` holding one is a view.
    /// Copy: scalars, `&T`, tuples of Copy, `impl Copy`, fn types; never `&mut T`. Frozen: no
    /// `UnsafeCell` (`Cell`) and no fn value reachable. Flat: no reference, view, fn value,
    /// `UnsafeCell`, or `Drop` type. A fn type is neither Send nor Sync; `Cell` is Send, not Sync.
    fn kinds(self) -> Kinds {
        let k = |view, copy, frozen, flat, sync, send, tr| Kinds { view, copy, frozen, flat, sync, send, tr };
        match self {
            Base::U32 => k(false, true, true, true, true, true, true),
            Base::S => k(false, false, true, true, true, true, false),
            Base::P => k(false, true, true, true, true, true, true),
            Base::D => k(false, false, true, false, true, true, false),
            Base::Ref => k(true, true, true, false, true, true, false),
            Base::Mut => k(true, false, true, false, true, true, false),
            Base::Pick => k(true, false, true, false, true, true, true),
            Base::Tup => k(true, true, true, false, true, true, false),
            Base::Opt => k(true, true, true, false, true, true, false),
            Base::Fn => k(true, true, false, false, false, false, false),
            Base::Cell => k(false, false, false, false, false, true, false),
            Base::Vec => k(false, false, true, false, true, true, false),
        }
    }
}

/// The type argument: a base type, or one nested in a tuple or an `Option` (the nested generic
/// argument: its kinds come from the inner type; neither has `Tr`).
#[derive(Clone, Copy, Debug)]
enum Arg {
    Base(Base),
    /// `(B, u32)`
    Pair(Base),
    /// `Option<B>`
    Some(Base),
}

impl Arg {
    fn ty(self) -> String {
        match self {
            Arg::Base(b) => b.ty().to_string(),
            Arg::Pair(b) => format!("({}, u32)", b.ty()),
            Arg::Some(b) => format!("Option<{}>", b.ty()),
        }
    }

    fn label(self) -> String {
        match self {
            Arg::Base(b) => b.label().to_string(),
            Arg::Pair(b) => format!("pair({})", b.label()),
            Arg::Some(b) => format!("some({})", b.label()),
        }
    }

    fn value(self, j: u32) -> String {
        match self {
            Arg::Base(b) => b.value(j),
            Arg::Pair(b) => format!("({}, x{j})", b.value(j)),
            Arg::Some(b) => format!("Option.Some({})", b.value(j)),
        }
    }

    fn kinds(self) -> Kinds {
        match self {
            Arg::Base(b) => b.kinds(),
            Arg::Pair(b) | Arg::Some(b) => Kinds { tr: false, ..b.kinds() },
        }
    }

    /// Whether `T` bound by `b` may be this type.
    fn fits(self, b: Bound) -> bool {
        let k = self.kinds();
        (b.view || !k.view)
            && match b.req {
                Req::None => true,
                Req::Tr => k.tr,
                Req::Copy => k.copy,
                Req::Frozen => k.frozen,
                Req::Flat => k.flat,
                Req::Sync => k.sync,
                Req::Send => k.send,
            }
    }
}

/// Where the generic item is instantiated.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Site {
    /// `gen(a)`: args inferred.
    Call,
    /// `gen::<A>(a)`
    Turbofish,
    /// `let f: fn(A) -> u32 = gen;`: args inferred from the expected fn type.
    FnValue,
    /// `let f = gen::<A>;`
    FnValueArgs,
    /// `apply(gen, a)`: a fn item as an argument of fn type.
    FnArg,
    /// `w.get()` on `impl<T: B> W<T>`: impl selection by method call.
    ImplMethod,
    /// `W::<A>.get(&w)`
    ImplPath,
    /// `let g = W::<A>.get;`: an impl method by path as a fn value.
    ImplMethodValue,
    /// `for e in w`: the implied `next` of `impl<T: B> W<T>`.
    ForNext,
    /// `w.q()` on `impl<T: B> Tq for W<T>`.
    TraitMethod,
    /// `Tq.q(&w)`
    TraitPath,
    /// `need(&w)` with `need<X: Tq + view>`: bound satisfaction selects the impl.
    BoundSat,
    /// `w + w2` on `impl<T: B> Add for W<T>`.
    Operator,
    /// `f"{w}"` on `impl<T: B> Display for W<T>`.
    FString,
    /// `u.m(a)`, `m<T: B>` a trait method's own generic param.
    OwnMethod,
    /// `u.m::<A>(a)`
    OwnTurbofish,
    /// `q.m(x)` in `tb<Q: Tm>`: through a bound.
    OwnBound,
    /// `q.m::<A>(x)` in `tb<Q: Tm>`.
    OwnBoundArgs,
    /// `Tm.m(&u, a)`
    OwnTraitPath,
    /// `G::<A>.{ v: a }` with `struct G<T: B>`.
    StructLit,
    /// `G<A>` as the type of a field of a view struct that is built.
    FieldType,
    /// `G<A>` in a type only (`let o: Option<G<A>> = Option.None;`): the view kind is checked
    /// there; a trait bound of an ADT param in a type position is not (no value of the type can be
    /// built), so only arguments that fit or fail the view kind are drawn.
    TypeOnly,
    /// `impl<T: B> Drop for W<T>`, `W<T: view>`, and a `W<A>` dropped: a declaration rule (a
    /// destructor must take a view wherever its type does), bad iff `B` lacks `view`, whatever `A`.
    DropImpl,
    /// `let f: fn(A) = mem.forget;`: std's `forget<T>` (no `view`) as a fn value.
    StdForget,
    /// `let mk = cell.Cell::<A>.new;`: a constructor of std's `Cell<T: Copy>` as a fn value.
    StdCellNew,
    /// `long[0..1].copy_from(&short[0..1])` on `[A; 1]`: std's `impl<T: Copy> [T]`.
    StdCopyFrom,
}

const SITES: [Site; 26] = [
    Site::Call,
    Site::Turbofish,
    Site::FnValue,
    Site::FnValueArgs,
    Site::FnArg,
    Site::ImplMethod,
    Site::ImplPath,
    Site::ImplMethodValue,
    Site::ForNext,
    Site::TraitMethod,
    Site::TraitPath,
    Site::BoundSat,
    Site::Operator,
    Site::FString,
    Site::OwnMethod,
    Site::OwnTurbofish,
    Site::OwnBound,
    Site::OwnBoundArgs,
    Site::OwnTraitPath,
    Site::StructLit,
    Site::FieldType,
    Site::TypeOnly,
    Site::DropImpl,
    Site::StdForget,
    Site::StdCellNew,
    Site::StdCopyFrom,
];

impl Site {
    fn label(self) -> &'static str {
        match self {
            Site::Call => "call",
            Site::Turbofish => "turbofish",
            Site::FnValue => "fn-value",
            Site::FnValueArgs => "fn-value-args",
            Site::FnArg => "fn-arg",
            Site::ImplMethod => "impl-method",
            Site::ImplPath => "impl-path",
            Site::ImplMethodValue => "impl-method-value",
            Site::ForNext => "for-next",
            Site::TraitMethod => "trait-method",
            Site::TraitPath => "trait-path",
            Site::BoundSat => "bound-sat",
            Site::Operator => "operator",
            Site::FString => "fstring",
            Site::OwnMethod => "own-method",
            Site::OwnTurbofish => "own-turbofish",
            Site::OwnBound => "own-bound",
            Site::OwnBoundArgs => "own-bound-args",
            Site::OwnTraitPath => "own-trait-path",
            Site::StructLit => "struct-lit",
            Site::FieldType => "field-type",
            Site::TypeOnly => "type-only",
            Site::DropImpl => "drop-impl",
            Site::StdForget => "std-forget",
            Site::StdCellNew => "std-cell-new",
            Site::StdCopyFrom => "std-copy-from",
        }
    }
}

/// The body of a fn taking `x: T`: it uses what the bound gives (a copy, a `Tr` method) and
/// returns a `u32` that borrows nothing.
fn fn_body(b: Bound, k: u32, ind: &str) -> String {
    match b.req {
        Req::Copy => format!("{ind}let c = x;\n{ind}let d = x;\n{ind}return {k};\n"),
        Req::Tr => format!("{ind}return x.val() +% {k};\n"),
        _ => format!("{ind}let y = x;\n{ind}return {k};\n"),
    }
}

/// The body of a `&self` method of `W<T> { v: T, n: u32 }`, like [`fn_body`].
fn self_body(b: Bound, ind: &str) -> String {
    match b.req {
        Req::Copy => format!("{ind}let c = self.v;\n{ind}let d = self.v;\n{ind}return self.n;\n"),
        Req::Tr => format!("{ind}return self.v.val() +% self.n;\n"),
        _ => format!("{ind}return self.n;\n"),
    }
}

/// The case `idx` of a `generic-sites` program.
pub(super) fn case(r: &mut Rng, idx: usize, want_bad: bool) -> Case {
    let site = SITES[r.below(SITES.len() as u32) as usize];
    let draw = |r: &mut Rng| {
        let mut b = Bound { view: r.chance(450), req: REQS[r.below(REQS.len() as u32) as usize] };
        match site {
            // `impl Add` returns `Self`: at `T: view` a view result without a `from` clause.
            Site::Operator => b.view = false,
            // The std items bring their own bounds.
            Site::StdForget => b = Bound { view: false, req: Req::None },
            Site::StdCellNew | Site::StdCopyFrom => b = Bound { view: false, req: Req::Copy },
            _ => {}
        }
        let base = BASES[r.below(BASES.len() as u32) as usize];
        let arg = match r.below(8) {
            0 => Arg::Pair(base),
            1 => Arg::Some(base),
            _ => Arg::Base(base),
        };
        (b, arg)
    };
    let good = |bound: Bound, arg: Arg| if site == Site::DropImpl { bound.view } else { arg.fits(bound) };
    let drawable = |bound: Bound, arg: Arg| {
        site != Site::TypeOnly || arg.fits(bound) || (arg.kinds().view && !bound.view)
    };
    let (mut bound, mut arg) = draw(r);
    for _ in 0..64 {
        if drawable(bound, arg) && good(bound, arg) != want_bad {
            break;
        }
        (bound, arg) = draw(r);
    }
    if !drawable(bound, arg) {
        (bound, arg) = (Bound { view: false, req: Req::None }, Arg::Base(Base::U32));
    }
    let bad = !good(bound, arg);
    let k = r.range(1, 40);
    let (bt, at, i) = (bound.text(), arg.ty(), idx);
    let mut items = String::new();
    let mut b = Body::new();
    b.line(&format!("let x0: u32 = seed +% {k};"));
    b.line(&format!("let x1: u32 = seed +% {};", k + 1));
    b.line("let mut y0: u32 = seed;");
    b.line("let mut y1: u32 = seed +% 1;");
    b.line("let fv0: fn(u32) -> u32 = next;");
    b.line("let fv1: fn(u32) -> u32 = next;");
    b.line("let mut s: u32 = 0;");
    b.line(&format!("let a0: {at} = {};", arg.value(0)));
    let wstruct = format!("struct W{i}<T: view> {{ v: T, n: u32 }}\n");
    let make_w = |b: &mut Body, name: &str, j: u32| {
        b.line(&format!("let mut {name} = W{i}::<{at}>.{{ v: a{j}, n: {} }};", k % 4 + 1));
    };
    match site {
        Site::Call | Site::Turbofish | Site::FnValue | Site::FnValueArgs | Site::FnArg => {
            items.push_str(&format!("fn gen{i}<T{bt}>(x: T) -> u32 {{\n{}}}\n", fn_body(bound, k, "    ")));
            match site {
                Site::Call => b.add(&format!("gen{i}(a0)")),
                Site::Turbofish => b.add(&format!("gen{i}::<{at}>(a0)")),
                Site::FnValue => {
                    b.line(&format!("let f: fn({at}) -> u32 = gen{i};"));
                    b.add("f(a0)");
                }
                Site::FnValueArgs => {
                    b.line(&format!("let f = gen{i}::<{at}>;"));
                    b.add("f(a0)");
                }
                _ => {
                    items.push_str(&format!(
                        "\nfn apply{i}(f: fn({at}) -> u32, a: {at}) -> u32 {{\n    return f(a);\n}}\n"
                    ));
                    b.add(&format!("apply{i}(gen{i}, a0)"));
                }
            }
        }
        Site::ImplMethod | Site::ImplPath | Site::ImplMethodValue => {
            items.push_str(&wstruct);
            items.push_str(&format!(
                "\nimpl<T{bt}> W{i}<T> {{\n    fn get(&self) -> u32 {{\n{}    }}\n}}\n",
                self_body(bound, "        ")
            ));
            make_w(&mut b, "w", 0);
            match site {
                Site::ImplMethod => b.add("w.get()"),
                Site::ImplPath => b.add(&format!("W{i}::<{at}>.get(&w)")),
                _ => {
                    b.line(&format!("let g = W{i}::<{at}>.get;"));
                    b.add("g(&w)");
                }
            }
        }
        Site::ForNext => {
            items.push_str(&wstruct);
            items.push_str(&format!(
                "\nimpl<T{bt}> W{i}<T> {{\n    fn next(&mut self) -> Option<u32> {{\n        if self.n == 0 {{\n            return Option.None;\n        }}\n        self.n = self.n - 1;\n        return Option.Some(self.n);\n    }}\n}}\n"
            ));
            make_w(&mut b, "w", 0);
            b.open("for e in w");
            b.add("e");
            b.close();
        }
        Site::TraitMethod | Site::TraitPath | Site::BoundSat => {
            items.push_str(&wstruct);
            items.push_str(&format!("\ntrait Tq{i} {{\n    fn q(&self) -> u32;\n}}\n"));
            items.push_str(&format!(
                "\nimpl<T{bt}> Tq{i} for W{i}<T> {{\n    fn q(&self) -> u32 {{\n{}    }}\n}}\n",
                self_body(bound, "        ")
            ));
            make_w(&mut b, "w", 0);
            match site {
                Site::TraitMethod => b.add("w.q()"),
                Site::TraitPath => b.add(&format!("Tq{i}.q(&w)")),
                _ => {
                    items.push_str(&format!(
                        "\nfn need{i}<X: Tq{i} + view>(x: &X) -> u32 {{\n    return x.q();\n}}\n"
                    ));
                    b.add(&format!("need{i}(&w)"));
                }
            }
        }
        Site::Operator => {
            items.push_str(&wstruct);
            items.push_str(&format!(
                "\nimpl<T{bt}> Add for W{i}<T> {{\n    fn add(self, rhs: Self) -> Self {{\n        return rhs;\n    }}\n}}\n"
            ));
            b.line(&format!("let a1: {at} = {};", arg.value(1)));
            make_w(&mut b, "w", 0);
            make_w(&mut b, "w2", 1);
            b.line("let w3 = w + w2;");
            b.add("w3.n");
        }
        Site::FString => {
            items.push_str(&wstruct);
            items.push_str(&format!(
                "\nimpl<T{bt}> Display for W{i}<T> {{\n    fn fmt<V: Write>(&self, out: &mut V) {{\n        out.write_str(\"w{i}\");\n    }}\n}}\n"
            ));
            make_w(&mut b, "w", 0);
            b.line("io.print(f\"{w}\\n\");");
        }
        Site::OwnMethod | Site::OwnTurbofish | Site::OwnBound | Site::OwnBoundArgs | Site::OwnTraitPath => {
            items.push_str(&format!("trait Tm{i} {{\n    fn m<T{bt}>(&self, x: T) -> u32;\n}}\n"));
            items.push_str(&format!("\nstruct U{i} {{}}\n"));
            items.push_str(&format!(
                "\nimpl Tm{i} for U{i} {{\n    fn m<T{bt}>(&self, x: T) -> u32 {{\n{}    }}\n}}\n",
                fn_body(bound, k, "        ")
            ));
            b.line(&format!("let u = U{i}.{{}};"));
            match site {
                Site::OwnMethod => b.add("u.m(a0)"),
                Site::OwnTurbofish => b.add(&format!("u.m::<{at}>(a0)")),
                Site::OwnTraitPath => b.add(&format!("Tm{i}.m(&u, a0)")),
                _ => {
                    let call =
                        if site == Site::OwnBound { "q.m(x)".to_string() } else { format!("q.m::<{at}>(x)") };
                    items.push_str(&format!(
                        "\nfn tb{i}<Q: Tm{i}>(q: &Q, x: {at}) -> u32 {{\n    return {call};\n}}\n"
                    ));
                    b.add(&format!("tb{i}(&u, a0)"));
                }
            }
        }
        Site::TypeOnly => {
            items.push_str(&format!("struct G{i}<T{bt}> {{ v: T }}\n"));
            b.line(&format!("let o: Option<G{i}<{at}>> = Option.None;"));
        }
        Site::StructLit | Site::FieldType => {
            items.push_str(&format!("struct G{i}<T{bt}> {{ v: T }}\n"));
            if site == Site::StructLit {
                b.line(&format!("let g = G{i}::<{at}>.{{ v: a0 }};"));
                b.add(&format!("{k}"));
            } else {
                items.push_str(&format!("\nview struct H{i} {{ g: G{i}<{at}>, k: u32 }}\n"));
                b.line(&format!("let h = H{i}.{{ g: G{i}.{{ v: a0 }}, k: {k} }};"));
                b.add("h.k");
            }
        }
        Site::DropImpl => {
            items.push_str(&wstruct);
            items.push_str(&format!(
                "\nimpl<T{bt}> Drop for W{i}<T> {{\n    fn drop(&mut self) {{\n        io.print(f\"drop w{i} {{self.n}}\\n\");\n    }}\n}}\n"
            ));
            make_w(&mut b, "w", 0);
            b.add("w.n");
        }
        Site::StdForget => {
            b.line(&format!("let f: fn({at}) = mem.forget;"));
            b.line("f(a0);");
        }
        Site::StdCellNew => {
            b.line(&format!("let mk = cell.Cell::<{at}>.new;"));
            b.line("let c = mk(a0);");
        }
        Site::StdCopyFrom => {
            b.line(&format!("let a1: {at} = {};", arg.value(1)));
            b.line(&format!("let mut long: [{at}; 1] = [a0];"));
            b.line(&format!("let short: [{at}; 1] = [a1];"));
            b.line("long[0..1].copy_from(&short[0..1]);");
        }
    }
    b.add("x0 +% x1");
    let label = format!("{}/{}/{}", site.label(), bound.label(), arg.label());
    let codes = if bad { RULE_CODES } else { &[][..] };
    finish(idx, b, items, label, bad, codes)
}
