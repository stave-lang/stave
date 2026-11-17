//! Cross-check of the generated `stave-syntax` tables: they equal a fresh build from
//! `contracts/grammar.ebnf`, and a minimal table-driven PDA over them (split actions: D3 and
//! unary `&&`) accepts the `stave` blocks of LANG.md, a few embedded
//! programs and the generated stage1 `.stv` files, and rejects invalid programs.

use ll1gen::{Config, Sym, build_tables};
use stave_syntax::generated::ll1_table as lt;
use stave_syntax::generated::node_kinds::{KIND_NAME, NKIND, NodeKind};
use stave_syntax::generated::terminals::{NTERM, SPLIT_FIRST, SPLIT_REST, TERM_IS_CLASS, TERM_TEXT, Term};
use std::path::{Path, PathBuf};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn term(text: &str) -> u16 {
    TERM_TEXT.iter().position(|t| *t == text).unwrap_or_else(|| panic!("no terminal {text:?}")) as u16
}

fn is_idstart(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_' || b >= 0x80
}

fn is_idcont(b: u8) -> bool {
    is_idstart(b) || b.is_ascii_digit()
}

/// Minimal Stave lexer producing terminal ids (no spans, no diagnostics).
fn lex(src: &str) -> Result<Vec<u16>, String> {
    let b = src.as_bytes();
    let n = b.len();
    let at = |j: usize| b.get(j).copied().unwrap_or(0);
    let mut ops: Vec<(&str, u16)> = TERM_TEXT
        .iter()
        .enumerate()
        .filter(|&(i, t)| !TERM_IS_CLASS[i] && !t.bytes().all(is_idcont))
        .map(|(i, t)| (*t, i as u16))
        .collect();
    ops.sort_by_key(|&(t, _)| std::cmp::Reverse(t.len()));
    let mut out: Vec<u16> = Vec::new();
    let mut i = 0;
    let skip_quoted = |mut j: usize, q: u8| -> Result<usize, String> {
        while j < n && b[j] != q && b[j] != b'\n' {
            j += if b[j] == b'\\' { 2 } else { 1 };
        }
        if j < n && b[j] == q { Ok(j + 1) } else { Err(format!("unterminated literal at {j}")) }
    };
    while i < n {
        let c = b[i];
        let prev_dot = out.last() == Some(&term("."));
        if c.is_ascii_whitespace() {
            i += 1;
        } else if c == b'/' && at(i + 1) == b'/' {
            while i < n && b[i] != b'\n' {
                i += 1;
            }
        } else if c == b'f' && at(i + 1) == b'"' {
            i = skip_quoted(i + 2, b'"')?;
            out.push(Term::FmtLit.id());
        } else if is_idstart(c) {
            let s = i;
            while i < n && is_idcont(b[i]) {
                i += 1;
            }
            let word = &src[s..i];
            if word == "as" && matches!(at(i), b'%' | b'|' | b'?' | b'~') {
                out.push(term(&src[s..i + 1]));
                i += 1;
            } else if let Some(k) = TERM_TEXT.iter().position(|t| *t == word) {
                out.push(if TERM_IS_CLASS[k] { Term::Ident.id() } else { k as u16 });
            } else {
                out.push(Term::Ident.id());
            }
        } else if c.is_ascii_digit() {
            let mut float = false;
            if prev_dot || (c == b'0' && matches!(at(i + 1), b'x' | b'b' | b'o')) {
                i += 1;
            } else {
                while at(i).is_ascii_digit() || at(i) == b'_' {
                    i += 1;
                }
                if at(i) == b'.' && at(i + 1).is_ascii_digit() {
                    float = true;
                    i += 1;
                    while at(i).is_ascii_digit() || at(i) == b'_' {
                        i += 1;
                    }
                }
                if matches!(at(i), b'e' | b'E') {
                    let d = if matches!(at(i + 1), b'+' | b'-') { 2 } else { 1 };
                    if at(i + d).is_ascii_digit() {
                        float = true;
                        i += d;
                        while at(i).is_ascii_digit() || at(i) == b'_' {
                            i += 1;
                        }
                    }
                }
            }
            while i < n && is_idcont(b[i]) {
                i += 1;
            }
            out.push(if float { Term::FloatLit.id() } else { Term::IntLit.id() });
        } else if c == b'"' {
            i = skip_quoted(i + 1, b'"')?;
            out.push(Term::StrLit.id());
        } else if c == b'\'' {
            i = skip_quoted(i + 1, b'\'')?;
            out.push(Term::CharLit.id());
        } else if let Some(&(t, id)) = ops.iter().find(|(t, _)| b[i..].starts_with(t.as_bytes())) {
            out.push(id);
            i += t.len();
        } else {
            return Err(format!("unexpected byte {c:#x} at {i}"));
        }
    }
    Ok(out)
}

/// Table-driven PDA over the generated tables. Returns the number of expansions.
fn parse(src: &str) -> Result<usize, String> {
    let mut toks = lex(src)?;
    toks.push(Term::Eof.id());
    let mut stack = vec![lt::NT_BIT | lt::START];
    let (mut i, mut expansions) = (0, 0);
    let name = |t: u16| TERM_TEXT[t as usize];
    while let Some(top) = stack.pop() {
        let tok = *toks.get(i).ok_or("input ended before the stack")?;
        let first = SPLIT_FIRST[tok as usize];
        if top & lt::NT_BIT == 0 {
            if top == tok {
                i += 1;
            } else if first >= 0 && top == first as u16 {
                toks[i] = SPLIT_REST[tok as usize] as u16;
            } else {
                return Err(format!("token {i}: expected {:?}, got {:?}", name(top), name(tok)));
            }
            continue;
        }
        let nt = top & !lt::NT_BIT;
        let mut p = lt::action(nt, tok);
        if p < 0 && first >= 0 {
            p = lt::action(nt, first as u16);
        }
        if p < 0 {
            return Err(format!("token {i}: {} cannot start with {:?}", lt::NT_NAME[nt as usize], name(tok)));
        }
        expansions += 1;
        stack.extend_from_slice(lt::rhs_rev(p as u16));
    }
    if i != toks.len() {
        return Err(format!("trailing tokens at {i}"));
    }
    Ok(expansions)
}

#[test]
fn generated_tables_equal_a_fresh_build() {
    let src = std::fs::read_to_string(repo().join("contracts/grammar.ebnf")).unwrap();
    let t = build_tables(&src, &Config::contract()).expect("build");
    assert_eq!((lt::NNT, NTERM, lt::NPROD, NKIND), (t.nts.len(), t.nterm(), t.prods.len(), t.nkind));
    assert!(lt::TABLE[..] == t.table[..], "TABLE differs");
    for (i, p) in t.prods.iter().enumerate() {
        assert_eq!(lt::PROD_LHS[i], p.lhs);
        let rev: Vec<u16> = p
            .rhs
            .iter()
            .rev()
            .map(|s| match *s {
                Sym::T(x) => x,
                Sym::N(n) => lt::NT_BIT | n,
            })
            .collect();
        assert_eq!(lt::rhs_rev(i as u16), &rev[..], "production {i}");
    }
    for (i, n) in t.nts.iter().enumerate() {
        assert_eq!((lt::NT_NAME[i], lt::NT_ORIGIN[i], lt::NT_NAMED[i]), (n.name.as_str(), n.origin, n.named));
        assert_eq!(lt::NT_PASS[i], n.pass, "{}", n.name);
    }
    for (i, k) in NodeKind::ALL.iter().enumerate() {
        assert_eq!((k.id() as usize, k.nt() as usize, k.name()), (i, i, KIND_NAME[i]));
        assert_eq!(lt::NT_NAME[i], k.name());
    }
    for (i, x) in Term::ALL.iter().enumerate() {
        assert_eq!((x.id() as usize, x.text()), (i, t.terms[i].as_str()));
        assert_eq!(Term::from_id(i as u16), Some(*x));
    }
    assert_eq!(Term::from_id(NTERM as u16), None);
    assert_eq!(lt::NOVERRIDE, 4);
    let d2 = lt::OVERRIDE_TYPE_BASE_LT;
    assert_eq!(lt::NT_ORIGIN[lt::OVERRIDE_NT[d2] as usize], NodeKind::TypeBase.nt());
    assert_eq!(lt::OVERRIDE_ORIGIN[d2], NodeKind::TypeBase.nt());
    assert_eq!(lt::OVERRIDE_TERM[d2], Term::Lt.id());
    assert_eq!(lt::OVERRIDE_ORIGIN[lt::OVERRIDE_STMT_KW_UNSAFE], NodeKind::Stmt.nt());
    assert_eq!(lt::OVERRIDE_TERM[lt::OVERRIDE_TYPE_BASE_KW_NEEDS], Term::KwNeeds.id());
    assert_eq!(lt::OVERRIDE_TERM[lt::OVERRIDE_TYPE_BASE_KW_WITHOUT], Term::KwWithout.id());
    for ((&nt, &tm), &p) in lt::OVERRIDE_NT.iter().zip(&lt::OVERRIDE_TERM).zip(&lt::OVERRIDE_PROD) {
        assert_eq!(lt::action(nt, tm), p as i16);
    }
    for (tok, first, rest) in [
        (Term::GtGt, Term::Gt, Term::Gt),
        (Term::GtEq, Term::Gt, Term::Eq),
        (Term::GtGtEq, Term::Gt, Term::GtEq),
        (Term::AmpAmp, Term::Amp, Term::Amp),
    ] {
        assert_eq!(SPLIT_FIRST[tok as usize], first as i16, "{tok:?}");
        assert_eq!(SPLIT_REST[tok as usize], rest as i16, "{tok:?}");
    }
    assert_eq!(SPLIT_FIRST.iter().filter(|&&r| r >= 0).count(), 4);
    assert_eq!(SPLIT_REST.iter().filter(|&&r| r >= 0).count(), 4);
}

const VALID: [&str; 4] = [
    r#"import std.io;
import a.b as c;

struct Point { x: i32, y: i32 }
view struct Lexer { src: &str, pos: usize }
enum Shape { Circle(f32), Rect { w: f32, h: f32 }, }

fn area(s: Shape) -> f32 {
    match s {
        Shape.Circle(r) => { return 3.14f32 * r * r; }
        Shape.Rect.{ w, h: hh } if w > 0.0 => { return w * hh; }
        _ => { return 0.0; }
    }
}

pub fn main() {
    let p: Point = Point.{ x: 1, y: 2, };
    let mut n: i32 = p.x + p.y;
    while n > 0 { n -= 1; }
    if n == 0 { io.print(f"n = {n} {p.x}\n"); } else if n < 0 { return; } else { loop { break; } }
    let v = if n < 3 then 1i32 else 2i32;
    let t = (1, 2u8,);
    let a = t.0.1;
    let m = match v { 1 => 10, -2..=5 => 20, _ => 30, };
    let ch: char = '\n';
    let ok = !(a == 1) && ~v != 0 || v <<% 3 >= 1e-6;
    let rr = &&a;
    let rm = &&mut a;
    let c = Cell::<u32>.{ v: 1u32 };
    let d = m.Cell::<Option<u32>>.{ v: Option.None };
}
impl<T> Cell<T> { fn new(v: T) -> Self { return Self.{ v: v }; } }
fn refs(x: &&i32) -> bool { return x == x && x != x; }
"#,
    r#"fn max<T: Ord + Copy>(a: T, b: T) -> T { return if a > b then a else b; }
fn f(x: u64) -> bool {
    let v: Vec<Vec<i32>> = Vec.new();
    let w: Vec<i32>= make();
    let z: A<B<C>>= g::<[1, 2], {N + 1}, 4>();
    x >>= 2;
    x +%= 1;
    let k = (x as usize) < 3;
    let p = &mut v as *mut Vec<Vec<i32>>;
    unsafe("p is valid") { *p = v; }
    let q = unsafe("p is valid") *p;
    let c = fn(y: i32) -> i32 { return y + 1; };
    let s = c(1)?;
    scope { spawn { work(); } spawn { work(); } }
    strict { emulate { } }
    return x as% u8 == 1u8 && x as? u16 != y as~ f32;
}
"#,
    r#"pub unsafe("callers pass valid pointers") fn raw(p: *const u8) -> u8 { return 0u8; }
trait Area { fn area(&self) -> f32; unsafe("x") fn g(self); }
impl Area for Point { pub fn area(&self) -> f32 { return 0.0; } }
impl<T: Copy, const N: usize> Wrap<T> { fn get(&self) -> &T from self { return &self.v; } }
unsafe("Send is sound here") impl Send for Wrap<i32> { }
const N: usize = 4 * 2;
type Bytes = [u8; 16];
type Aligned = [f32 align 64];
extern "C" { fn puts(s: *const u8) -> i32; fn printf(f: *const u8, ...) -> i32; }
extern "C" fn add(a: i32, b: i32) -> i32 { return a + b; }
dispatch fn sum(xs: &[f32]) -> f32 needs(sse42) { avx2 + fma => sum_avx2, _ => sum_scalar, }
fn sum_avx2(xs: &[f32]) -> f32 needs(avx2, fma) {
    let mut acc: f32x8 = f32x8.splat(0.0);
    for v: f32x8 in xs { acc = acc + v; } tail masked (r, m) { acc = acc + r; }
    for x in xs { } tail scalar (y) { }
    for i in 0..n { }
    return acc.sum();
}
fn h() -> fn(i32, &T) -> &T needs(avx2) { return g; }
fn k() -> (fn()) needs(avx2) { }
test fn t1() { let a: [i32; 3] = [1, 2, 3]; let b = [0u8; 16]; let c = a[0]; }
kernel fn kk(xs: &mut [f32]) { }
allocator my.alloc;
"#,
    "",
];

const INVALID: [&str; 9] = [
    "fn f() { let x = a as usize < b; }",
    "fn f() { let s = S { a: 1 }; }",
    "fn f() { let b = a < b < c; }",
    "fn f() { unsafe *p = 1; }",
    "fn f() -> i32 { return 1 }",
    "fn f() { } import a;",
    "fn f() { x = if c { 1 } else { 2 }; }",
    "fn f(mut x: i32) { }",
    "fn f() { let v = a..; }",
];

#[test]
fn pda_accepts_valid_programs() {
    for (i, src) in VALID.iter().enumerate() {
        if let Err(e) = parse(src) {
            panic!("valid program {i}: {e}");
        }
    }
}

#[test]
fn pda_rejects_invalid_programs() {
    for src in INVALID {
        assert!(parse(src).is_err(), "accepted: {src}");
    }
}

#[test]
fn pda_accepts_lang_md_stave_blocks() {
    let md = std::fs::read_to_string(repo().join("LANG.md")).unwrap();
    let mut blocks = 0;
    let mut rest = md.as_str();
    while let Some(s) = rest.find("```stave\n") {
        let body = &rest[s + "```stave\n".len()..];
        let e = body.find("```").expect("unclosed block");
        if let Err(err) = parse(&body[..e]) {
            panic!("LANG.md Stave block {blocks}: {err}\n{}", &body[..e]);
        }
        blocks += 1;
        rest = &body[e + 3..];
    }
    assert!(blocks > 0);
}

#[test]
fn pda_accepts_generated_stage1_files() {
    for f in ["ll1_table.stv", "terminals.stv", "node_kinds.stv"] {
        let src = std::fs::read_to_string(repo().join(ll1gen::STAVE_DIR).join(f)).unwrap();
        if let Err(e) = parse(&src) {
            panic!("{f}: {e}");
        }
    }
}
