//! The duplicate form (`stave_ir::form`): equal exactly when the code is the same, whatever the
//! function's own name, instance id, and site rows; a different callee, literal, or type gives
//! another form; self calls and closures are named relative to the function; the form up to
//! `Self` renames one `Self` type and its methods.

use stave_ir::form::{Callees, MethodOf, SelfTypes, closures_of, form, same_code};
use stave_ir::text::read;
use stave_ir::{FuncId, Module, TypeId};

fn load(src: &str) -> Module {
    match read(src.as_bytes()) {
        Ok(l) => l.module,
        Err(es) => {
            let r: Vec<String> = es.iter().map(|e| e.render("t.ir")).collect();
            panic!("rejected:\n{}\n--- input:\n{src}", r.join("\n"))
        }
    }
}

fn text(m: &Module, f: u32) -> String {
    form(m, FuncId(f), None, &Callees::default()).expect("a body").text
}

const CALLS: &str = "stave.ir 4 \"x86_64-linux-v2\" debug checked
type t0 int s 64
site 0 \"src/a.stv\" 1 1 app.a
site 1 \"src/b.stv\" 7 3 app.b
fn @0 app.a(t0) -> t0 site 0 {
b0(%0: t0):
  %1 = add t0 %0 1 !0
  %2 = call t0 @2 %1 !0
  ret %2
}
fn @1 app.b(t0) -> t0 site 1 {
b0(%0: t0):
  %1 = add t0 %0 1 !1
  %2 = call t0 @2 %1 !1
  ret %2
}
fn @2 app.c(t0) -> t0 {
b0(%0: t0):
  %1 = add t0 %0 2 !0
  %2 = call t0 @2 %1 !0
  ret %2
}
fn @3 app.d(t0) -> t0 {
b0(%0: t0):
  %1 = add t0 %0 1 !0
  %2 = call t0 @3 %1 !0
  ret %2
}
fn @4 app.e(t0) -> t0 {
b0(%0: t0):
  %1 = add t0 %0 1 !0
  %2 = call t0 @4 %1 !0
  ret %2
}
fn @5 app.f(t0) -> t0 {
b0(%0: t0):
  %1 = add t0 %0 1 !0
  %2 = call t0 @0 %1 !0
  ret %2
}
";

#[test]
fn name_instance_and_sites_are_not_part_of_the_form() {
    let m = load(CALLS);
    assert_eq!(text(&m, 0), text(&m, 1));
    let t = text(&m, 0);
    assert!(!t.contains("app.a") && !t.contains('!'), "{t}");
    assert!(t.starts_with("fn(t0) -> t0\n"), "{t}");
}

#[test]
fn a_different_literal_or_callee_is_another_form() {
    let m = load(CALLS);
    // `app.c` adds 2, not 1 (and calls itself: `@self`, not `@2`).
    assert_ne!(text(&m, 0), text(&m, 2));
    // `app.f` calls `app.a`, `app.a` calls `app.c`.
    assert_ne!(text(&m, 0), text(&m, 5));
}

#[test]
fn a_self_call_is_named_relative_to_the_function() {
    let m = load(CALLS);
    assert_eq!(text(&m, 3), text(&m, 4));
    assert!(text(&m, 3).contains("call t0 @self %1"), "{}", text(&m, 3));
}

const CLOSURES: &str = "stave.ir 4 \"x86_64-linux-v2\" debug checked
type t0 int s 64
fn @0 app.a(t0) -> t0 {
b0(%0: t0):
  %1 = call t0 @0.0 %0
  ret %1
}
fn @0.0 app.a(t0) -> t0 {
b0(%0: t0):
  %1 = mul.wrap t0 %0 2
  ret %1
}
fn @1 app.b(t0) -> t0 {
b0(%0: t0):
  %1 = call t0 @1.0 %0
  ret %1
}
fn @1.0 app.b(t0) -> t0 {
b0(%0: t0):
  %1 = mul.wrap t0 %0 2
  ret %1
}
fn @2 app.c(t0) -> t0 {
b0(%0: t0):
  %1 = call t0 @2.0 %0
  ret %1
}
fn @2.0 app.c(t0) -> t0 {
b0(%0: t0):
  %1 = mul.wrap t0 %0 3
  ret %1
}
";

#[test]
fn closures_are_part_of_their_function() {
    let m = load(CLOSURES);
    assert_eq!(closures_of(&m, FuncId(0)), vec![FuncId(1)]);
    assert_eq!(text(&m, 0), text(&m, 2));
    assert!(text(&m, 0).contains("@self.0") && text(&m, 0).contains("closure 0(t0) -> t0"));
    // The closure of `app.c` multiplies by 3.
    assert_ne!(text(&m, 0), text(&m, 4));
    // Size: the function's two lines and block, and the closure's.
    assert_eq!(form(&m, FuncId(0), None, &Callees::default()).unwrap().size, 4);
}

const SELF: &str = "stave.ir 4 \"x86_64-linux-v2\" debug checked
decl d0 struct app.A
decl d1 struct app.B
type t0 int s 64
type t1 adt d0
type t2 adt d1
shape t1 struct t0
shape t2 struct t0
fn @0 app.A.get(t1) -> t0 {
b0(%0: t1):
  %1 = get t0 %0 0
  ret %1
}
fn @1 app.A.twice(t1) -> t0 {
b0(%0: t1):
  %1 = call t0 @0 %0
  %2 = add t0 %1 %1
  ret %2
}
fn @2 app.B.get(t2) -> t0 {
b0(%0: t2):
  %1 = get t0 %0 0
  ret %1
}
fn @3 app.B.twice(t2) -> t0 {
b0(%0: t2):
  %1 = call t0 @2 %0
  %2 = add t0 %1 %1
  ret %2
}
fn @4 app.B.other(t1) -> t0 {
b0(%0: t1):
  %1 = call t0 @0 %0
  %2 = add t0 %1 %1
  ret %2
}
";

#[test]
fn the_form_up_to_self_renames_the_self_type_and_its_methods() {
    let m = load(SELF);
    let (a, b) = (TypeId(1), TypeId(2));
    let method =
        |t: TypeId, name: &str| Some(MethodOf { self_ty: t, label: format!("-.{name}"), impl_args: 0 });
    let methods =
        vec![method(a, "get"), method(a, "twice"), method(b, "get"), method(b, "twice"), method(b, "other")];
    let cx = Callees { methods, same: Vec::new() };
    // Concrete forms differ: the param types differ.
    assert_ne!(text(&m, 0), text(&m, 2));
    let (sa, sb) = (SelfTypes::new(&m, a), SelfTypes::new(&m, b));
    let up = |f: u32, s: &SelfTypes| form(&m, FuncId(f), Some(s), &cx).unwrap().text;
    assert_eq!(up(0, &sa), up(2, &sb));
    assert!(up(0, &sa).starts_with("fn(Self) -> t0\n"), "{}", up(0, &sa));
    // `A.twice` calls `A.get`, `B.twice` calls `B.get`: both `@Self.-.get`.
    assert_eq!(up(1, &sa), up(3, &sb));
    assert!(up(1, &sa).contains("@Self.-.get"), "{}", up(1, &sa));
    // `B.other` takes an `A` and calls `A.get`, a method of another type.
    assert_ne!(up(1, &sa), up(4, &sb));
}

const CHAIN: &str = "stave.ir 4 \"x86_64-linux-v2\" debug checked
type t0 int s 64
fn @0 app.f1(t0) -> t0 {
b0(%0: t0):
  %1 = call t0 @2 %0
  ret %1
}
fn @1 app.f2(t0) -> t0 {
b0(%0: t0):
  %1 = call t0 @3 %0
  ret %1
}
fn @2 app.g1(t0) -> t0 {
b0(%0: t0):
  %1 = mul t0 %0 3
  ret %1
}
fn @3 app.g2(t0) -> t0 {
b0(%0: t0):
  %1 = mul t0 %0 3
  ret %1
}
fn @4 app.f3(t0) -> t0 {
b0(%0: t0):
  %1 = call t0 @5 %0
  ret %1
}
fn @5 app.g3(t0) -> t0 {
b0(%0: t0):
  %1 = mul t0 %0 4
  ret %1
}
";

#[test]
fn calls_of_functions_with_the_same_code_are_the_same_code() {
    let m = load(CHAIN);
    // `f1` calls `g1`, `f2` calls `g2`: different callees, so different plain forms.
    assert_ne!(text(&m, 0), text(&m, 1));
    let all: Vec<FuncId> = (0..6).map(FuncId).collect();
    let same = same_code(&m, &all, &[], |fs, one| fs.iter().map(|&f| one(f)).collect());
    assert_eq!(same, [0, 0, 2, 2, 4, 5].map(FuncId));
    let cx = Callees { methods: Vec::new(), same };
    let with = |f: u32| form(&m, FuncId(f), None, &cx).unwrap().text;
    assert_eq!(with(0), with(1));
    // `g3` multiplies by 4.
    assert_ne!(with(0), with(4));
}
