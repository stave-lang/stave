//! Round trip and determinism: parse → print → parse gives the same tokens and
//! AST (and printing is idempotent), over random programs from the grammar, the LANG.md blocks,
//! and the syntax conformance files; item-parallel parsing with 1, 2, 12 threads gives identical
//! ASTs and diagnostics.
//!
//! `tools/gen-corpus` is still a stub, so the random programs come from the generator below
//! (every grammar rule, D2/D3 shapes, glued and spaced token forms, comments, ASCII names).

mod common;

use common::{F, lang_md_blocks, relayout, repo, stave_files};
use proptest::prelude::*;
use stave_syntax::{ParsedFile, format_source, parse_file, parse_file_stepwise, print_file};

/// A deterministic random program generator over the grammar.
struct Gen {
    s: u64,
    out: String,
}

const IDENTS: [&str; 12] = ["a", "b", "c", "x", "y", "n", "foo", "bar", "_q", "q1", "v2", "Zed"];
const TYPES: [&str; 8] = ["i32", "u8", "f64", "bool", "T", "Foo", "usize", "f32x4"];
const LEVELS: [&[&str]; 6] = [
    &["|"],
    &["^"],
    &["&"],
    &["<<", ">>", "<<%"],
    &["+", "-", "+%", "-%", "+|", "-|"],
    &["*", "/", "%", "*%", "*|"],
];
const CMP: [&str; 6] = ["==", "!=", "<", "<=", ">", ">="];
const ASSIGN: [&str; 18] = [
    "=", "+=", "-=", "*=", "/=", "%=", "+%=", "-%=", "*%=", "+|=", "-|=", "*|=", "&=", "|=", "^=", "<<=",
    ">>=", "<<%=",
];

impl Gen {
    fn new(seed: u64) -> Gen {
        Gen { s: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1, out: String::new() }
    }

    fn next(&mut self) -> u64 {
        self.s ^= self.s << 13;
        self.s ^= self.s >> 7;
        self.s ^= self.s << 17;
        self.s
    }

    fn pick(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn chance(&mut self, pct: u64) -> bool {
        self.next() % 100 < pct
    }

    fn w(&mut self, s: &str) {
        self.out.push_str(s);
        self.out.push(' ');
    }

    fn either(&mut self, pct: u64, a: &str, b: &str) {
        let c = self.chance(pct);
        self.w(if c { a } else { b });
    }

    fn any(&mut self, xs: &[&str]) {
        let i = self.pick(xs.len());
        self.w(xs[i]);
    }

    fn ident(&mut self) {
        let i = IDENTS[self.pick(IDENTS.len())];
        self.w(i);
    }

    fn nl(&mut self) {
        if self.chance(10) {
            self.out.push_str("// note\n");
        } else {
            self.out.push('\n');
        }
    }

    fn ty(&mut self, d: u32) {
        if d == 0 {
            let t = TYPES[self.pick(TYPES.len())];
            return self.w(t);
        }
        match self.pick(14) {
            0 => {
                self.w("&");
                self.ty(d - 1);
            }
            1 => {
                self.w("&mut");
                self.ty(d - 1);
            }
            2 => {
                self.either(50, "*const", "*mut");
                self.ty(d - 1);
            }
            3 => {
                self.w("[");
                self.ty(d - 1);
                self.w("; 4 ]");
            }
            4 => {
                self.w("[");
                self.ty(d - 1);
                if self.chance(30) {
                    self.w("align 32");
                }
                self.w("]");
            }
            5 => self.w("()"),
            6 => {
                self.w("(");
                self.ty(d - 1);
                self.w(", )");
            }
            7 => {
                self.w("(");
                self.ty(d - 1);
                self.w(",");
                self.ty(d - 1);
                self.w(")");
            }
            8 => {
                self.w("fn(");
                self.ty(d - 1);
                self.w(") ->");
                self.ty(d - 1);
                if self.chance(30) {
                    self.w("needs(avx2)");
                }
                if self.chance(20) {
                    self.w("without(alloc, unsafe)");
                }
            }
            9 => {
                self.w("Vec <");
                self.ty(d - 1);
                self.w(">");
            }
            10 => self.either(50, "Vec<Vec<i32>>", "Map<u8, Option<Vec<u8> > >"),
            11 => self.w("std.vec.Vec<i32>"),
            12 => self.w("V<4, {N + 1}, [0, 1], [u8; 2]>"),
            _ => {
                let t = TYPES[self.pick(TYPES.len())];
                self.w(t);
            }
        }
    }

    fn expr(&mut self, d: u32) {
        if d > 0 && self.chance(6) {
            self.w("if");
            self.expr(d - 1);
            self.w("then");
            self.expr(d - 1);
            self.w("else");
            return self.expr(d - 1);
        }
        if d > 0 && self.chance(5) {
            self.w("match");
            self.expr(d - 1);
            self.w("{");
            let arms = 1 + self.pick(3);
            for k in 0..arms {
                self.arm_pattern(d - 1);
                if self.chance(20) {
                    self.w("if");
                    self.expr(d - 1);
                }
                self.w("=>");
                self.expr(d - 1);
                if k + 1 < arms || self.chance(50) {
                    self.w(",");
                }
            }
            return self.w("}");
        }
        self.or(d);
    }

    fn or(&mut self, d: u32) {
        self.and(d);
        while d > 0 && self.chance(15) {
            self.w("||");
            self.and(d - 1);
        }
    }

    fn and(&mut self, d: u32) {
        self.cmp(d);
        while d > 0 && self.chance(15) {
            self.w("&&");
            self.cmp(d - 1);
        }
    }

    fn cmp(&mut self, d: u32) {
        self.level(0, d);
        if d > 0 && self.chance(20) {
            let op = CMP[self.pick(CMP.len())];
            self.w(op);
            self.level(0, d - 1);
        }
    }

    fn level(&mut self, k: usize, d: u32) {
        if k == LEVELS.len() {
            return self.cast(d);
        }
        self.level(k + 1, d);
        while d > 0 && self.chance(12) {
            let ops = LEVELS[k];
            let op = ops[self.pick(ops.len())];
            self.w(op);
            self.level(k + 1, d - 1);
        }
    }

    fn cast(&mut self, d: u32) {
        if d > 0 && self.chance(8) {
            // parenthesized: a bare cast before `<` would be D2 (E0011)
            self.w("(");
            self.unary(d - 1);
            let op = ["as", "as%", "as|", "as?", "as~"][self.pick(5)];
            self.w(op);
            self.ty(1);
            return self.w(")");
        }
        self.unary(d);
    }

    fn unary(&mut self, d: u32) {
        if d > 0 && self.chance(15) {
            match self.pick(8) {
                0 => self.w("-"),
                1 => self.w("!"),
                2 => self.w("~"),
                3 => self.w("*"),
                4 => self.w("&"),
                5 => self.w("&mut"),
                6 => self.w("&&"),
                _ => self.w("unsafe(\"ok\")"),
            }
            return self.unary(d - 1);
        }
        self.postfix(d);
    }

    fn postfix(&mut self, d: u32) {
        let named = self.primary(d);
        let steps = if d == 0 { 0 } else { self.pick(3) };
        for _ in 0..steps {
            match self.pick(7) {
                0 => {
                    self.w(".");
                    self.ident();
                }
                1 => self.w(".0"),
                2 => {
                    self.w(".");
                    self.ident();
                    if self.chance(30) {
                        self.w("::<i32, 4>");
                    }
                    self.args(d - 1);
                }
                3 if named => self.args(d - 1),
                4 => {
                    self.w("[");
                    self.expr(d - 1);
                    self.w("]");
                }
                5 => self.w("?"),
                6 if named => {
                    self.w(".{");
                    if self.chance(70) {
                        self.ident();
                        self.w(":");
                        self.expr(d - 1);
                        if self.chance(50) {
                            self.w(",");
                        }
                    }
                    self.w("}");
                }
                _ => self.w(".1"),
            }
        }
    }

    fn args(&mut self, d: u32) {
        self.w("(");
        let n = self.pick(3);
        for k in 0..n {
            if k > 0 {
                self.w(",");
            }
            self.expr(d);
        }
        self.w(")");
    }

    /// Returns true when the primary is a plain name (calls and struct literals may follow).
    fn primary(&mut self, d: u32) -> bool {
        let lit = ["1", "0xff", "1_000u32", "2.5", "1e-6", "1.0f32", "\"s\\n\"", "'c'", "true", "false"];
        match if d == 0 { self.pick(3) } else { self.pick(12) } {
            0 => {
                self.ident();
                true
            }
            1 => {
                let l = lit[self.pick(lit.len())];
                self.w(l);
                false
            }
            2 => {
                self.w("f\"v = {x} {self.n}\"");
                false
            }
            3 => {
                self.w("(");
                self.expr(d - 1);
                self.w(")");
                false
            }
            4 => {
                self.w("(");
                self.expr(d - 1);
                self.w(",");
                if self.chance(50) {
                    self.expr(d - 1);
                }
                self.w(")");
                false
            }
            5 => {
                self.w("[");
                self.expr(d - 1);
                self.either(50, ";", ",");
                self.expr(d - 1);
                self.w("]");
                false
            }
            6 => {
                self.w("fn(x: i32) ->");
                self.ty(0);
                self.block(d - 1);
                false
            }
            7 => {
                self.w("g::<");
                self.ty(1);
                self.w(">");
                true
            }
            8 => {
                self.either(50, "self", "Self");
                false
            }
            9 => {
                self.w("t.0.1");
                false
            }
            10 => {
                self.w("()");
                false
            }
            _ => {
                self.ident();
                self.w(".");
                self.ident();
                true
            }
        }
    }

    fn arm_pattern(&mut self, d: u32) {
        self.pattern(d);
        if self.chance(25) {
            for _ in 0..1 + self.pick(2) {
                self.w("|");
                self.pattern(d);
            }
        }
    }

    fn pattern(&mut self, d: u32) {
        match if d == 0 { self.pick(3) } else { self.pick(9) } {
            0 => self.w("_"),
            1 => self.ident(),
            2 => self.any(&["1", "-1", "'a'", "\"s\"", "true", "2.5"]),
            3 => self.any(&["0..=9", "-5..-1", "'a'..='z'"]),
            4 => {
                self.w("mut");
                self.ident();
            }
            5 => {
                self.w("E.V(");
                self.pattern(d - 1);
                self.w(",");
                self.pattern(d - 1);
                self.w(")");
            }
            6 => {
                self.w("E.S.{");
                self.ident();
                self.w(",");
                self.ident();
                self.w(":");
                self.pattern(d - 1);
                self.w("}");
            }
            7 => {
                self.w("(");
                self.pattern(d - 1);
                self.w(",");
                self.w(")");
            }
            _ => {
                self.w("(");
                self.pattern(d - 1);
                self.w(",");
                self.pattern(d - 1);
                self.w(")");
            }
        }
    }

    fn block(&mut self, d: u32) {
        self.w("{");
        self.nl();
        let n = if d == 0 { self.pick(2) } else { self.pick(4) };
        for _ in 0..n {
            self.stmt(d);
            self.nl();
        }
        self.w("}");
    }

    fn stmt(&mut self, d: u32) {
        let d1 = d.saturating_sub(1);
        match self.pick(16) {
            0 => {
                self.w("let");
                self.pattern(d1);
                if self.chance(50) {
                    self.w(":");
                    self.ty(2);
                }
                self.w("=");
                if self.chance(20) {
                    self.or(d1);
                    self.either(50, "..", "..=");
                    self.or(d1);
                } else {
                    self.expr(d1);
                }
                self.w(";");
            }
            1 => {
                self.w("if");
                self.expr(d1);
                self.block(d1);
                if self.chance(50) {
                    self.w("else");
                    if self.chance(50) {
                        self.w("if");
                        self.expr(d1);
                        self.block(d1);
                    } else {
                        self.block(d1);
                    }
                }
            }
            2 => {
                self.w("while");
                self.expr(d1);
                self.block(d1);
            }
            3 => {
                self.w("loop");
                self.block(d1);
            }
            4 => {
                self.w("for");
                self.pattern(d1);
                if self.chance(50) {
                    self.w(": f32x4");
                }
                self.w("in");
                self.expr(d1);
                self.block(d1);
                match self.pick(4) {
                    0 => {
                        self.w("tail masked (v, m)");
                        self.block(d1);
                    }
                    1 => {
                        self.w("tail scalar (x)");
                        self.block(d1);
                    }
                    2 => self.w("tail exact ;"),
                    _ => {}
                }
            }
            5 => {
                self.w("match");
                self.expr(d1);
                self.w("{");
                for _ in 0..self.pick(3) {
                    self.arm_pattern(d1);
                    if self.chance(20) {
                        self.w("if");
                        self.expr(d1);
                    }
                    self.w("=>");
                    self.block(d1);
                }
                self.w("}");
            }
            6 => {
                self.w("return");
                if self.chance(70) {
                    self.expr(d1);
                }
                self.w(";");
            }
            7 => self.either(50, "break ;", "continue ;"),
            8 => {
                self.w("unsafe(\"sound\")");
                self.block(d1);
            }
            9 => {
                let kw = ["strict", "emulate", "scope", "spawn"][self.pick(4)];
                self.w(kw);
                self.block(d1);
            }
            10 => self.block(d1),
            11 | 12 => {
                // not `unary`: a statement starting with `unsafe` opens a block
                self.postfix(d1);
                let op = ASSIGN[self.pick(ASSIGN.len())];
                self.w(op);
                self.expr(d1);
                self.w(";");
            }
            13 => self.w("x>>=1;"),
            _ => {
                self.ident();
                self.args(d1);
                self.w(";");
            }
        }
    }

    fn generics(&mut self) {
        if self.chance(40) {
            self.w("<T: Copy + std.ops.Add<T> + view, const N: usize>");
        }
    }

    fn params(&mut self, with_self: bool) {
        self.w("(");
        if with_self {
            self.any(&["self", "&self", "&mut self"]);
            self.param_from();
            self.w(",");
        }
        for _ in 0..self.pick(3) {
            self.ident();
            self.w(":");
            self.ty(2);
            self.param_from();
            self.w(",");
        }
        self.w(")");
    }

    fn param_from(&mut self) {
        if self.chance(10) {
            self.either(50, "from x", "from copy x + self + y");
        }
    }

    fn contracts(&mut self, d: u32) {
        if self.chance(15) {
            for _ in 0..self.pick(3) {
                self.w("requires(");
                self.expr(d);
                self.w(")");
            }
            for _ in 0..self.pick(3) {
                self.w("ensures(");
                self.expr(d);
                self.w(")");
            }
        }
    }

    fn fn_sig(&mut self, with_self: bool) {
        self.w("fn");
        self.ident();
        self.generics();
        self.params(with_self);
        if self.chance(60) {
            self.w("->");
            self.ty(2);
            if self.chance(20) {
                self.w("from a + copy self");
            }
        }
        if self.chance(15) {
            self.w("needs(avx2, fma)");
        }
    }

    /// A fn item or method signature: `fn_sig` plus the optional `rec` and `without(..)`.
    fn decl_sig(&mut self, with_self: bool) {
        if self.chance(15) {
            self.w("rec");
        }
        self.fn_sig(with_self);
        if self.chance(15) {
            self.w("without(trap, lock)");
        }
    }

    fn item(&mut self, d: u32) {
        match self.pick(15) {
            0..=3 => {
                if self.chance(30) {
                    self.w("pub");
                }
                if self.chance(10) {
                    self.w("unsafe(\"why\")");
                }
                self.decl_sig(false);
                self.contracts(d.min(1));
                self.block(d);
            }
            4 => {
                self.either(30, "view struct", "struct");
                self.ident();
                self.generics();
                self.w("{");
                for _ in 0..self.pick(3) {
                    if self.chance(30) {
                        self.w("pub");
                    }
                    self.ident();
                    self.w(":");
                    self.ty(2);
                    self.w(",");
                }
                self.w("}");
            }
            5 => {
                self.w("enum E {");
                self.w("A, B(i32, u8), C { w: f64, h: f64 }");
                if self.chance(50) {
                    self.w(",");
                }
                self.w("}");
            }
            6 => {
                self.w("const K :");
                self.ty(1);
                self.w("=");
                self.expr(d);
                self.w(";");
            }
            7 => {
                self.w("type Al");
                self.generics();
                self.w("=");
                self.ty(2);
                self.w(";");
            }
            8 => {
                self.w("trait Tr {");
                self.nl();
                for _ in 0..self.pick(3) {
                    if self.chance(20) {
                        self.w("unsafe(\"x\")");
                    }
                    self.fn_sig(true);
                    self.w(";");
                    self.nl();
                }
                self.w("}");
            }
            9 => {
                self.w("impl");
                self.generics();
                self.ty(1);
                if self.chance(50) {
                    self.w("for");
                    self.ty(1);
                }
                self.w("{");
                self.nl();
                for _ in 0..self.pick(3) {
                    if self.chance(40) {
                        self.w("pub");
                    }
                    if self.chance(20) {
                        self.w("unsafe(\"x\")");
                    }
                    self.decl_sig(true);
                    self.contracts(d.min(1));
                    self.block(d.saturating_sub(1));
                    self.nl();
                }
                self.w("}");
            }
            10 => {
                self.w("extern \"C\" { fn puts(s: *const u8) -> i32; fn printf(f: *const u8, ...) -> i32; }");
                self.nl();
                self.w("extern \"C\" fn add(a: i32, b: i32) -> i32");
                self.block(d.saturating_sub(1));
            }
            11 => self.w("dispatch fn sum(xs: &[f32]) -> f32 needs(sse42) { avx2 + fma => m.a, _ => b, }"),
            12 => {
                self.either(50, "test fn t()", "kernel fn k(xs: &mut [f32])");
                self.block(d);
            }
            13 => {
                self.either(50, "static S :", "pub static mut S :");
                self.ty(1);
                self.w("=");
                self.expr(d);
                self.w(";");
            }
            _ => self.w("allocator my.alloc;"),
        }
    }

    fn program(seed: u64, items: usize, depth: u32) -> String {
        let mut g = Gen::new(seed);
        for _ in 0..g.pick(3) {
            g.w("import std.io");
            if g.chance(50) {
                g.w("as io2");
            }
            g.w(";");
            g.nl();
        }
        for _ in 0..items {
            g.item(depth);
            g.nl();
            g.nl();
        }
        g.out
    }
}

/// The macro-expanding parser equals the step-by-step PDA.
fn same_as_stepwise(src: &[u8], pf: &ParsedFile) {
    let r = parse_file_stepwise(src, F);
    assert_eq!(pf.tokens, r.tokens);
    assert_eq!(pf.items, r.items, "{}", String::from_utf8_lossy(src));
    assert!(pf.ast == r.ast, "AST differs from the stepwise PDA:\n{}", String::from_utf8_lossy(src));
    assert_eq!(pf.diagnostics, r.diagnostics, "{}", String::from_utf8_lossy(src));
}

/// parse → print → parse: same items, token kinds and AST; the printer never falls back; printing
/// the printed text gives the same text.
fn check_roundtrip(name: &str, src: &[u8]) -> ParsedFile {
    let pf = parse_file(src, F);
    same_as_stepwise(src, &pf);
    let p1 = print_file(&pf, src);
    assert_eq!(p1.fallbacks, 0, "{name}: printer fell back\n{}", String::from_utf8_lossy(src));
    let pf2 = parse_file(p1.text.as_bytes(), F);
    let kinds = |p: &ParsedFile| p.tokens.iter().map(|t| t.kind).collect::<Vec<_>>();
    assert_eq!(kinds(&pf), kinds(&pf2), "{name}: tokens differ\n--- printed:\n{}", p1.text);
    let st = |p: &ParsedFile| p.items.iter().map(|i| (i.kind, i.start, i.end, i.ok)).collect::<Vec<_>>();
    assert_eq!(st(&pf), st(&pf2), "{name}: items differ");
    assert!(pf.ast == pf2.ast, "{name}: AST differs\n--- printed:\n{}", p1.text);
    let p2 = print_file(&pf2, p1.text.as_bytes());
    assert_eq!(p1.text, p2.text, "{name}: printing is not idempotent");
    pf
}

#[test]
fn generated_programs_parse_clean() {
    for seed in 0..300 {
        let src = Gen::program(seed, 6, 4);
        let pf = parse_file(src.as_bytes(), F);
        assert!(pf.diagnostics.is_empty(), "seed {seed}: {:#?}\n{src}", pf.diagnostics.first());
        if std::env::var("STAVE_SHOW").is_ok_and(|v| v == seed.to_string()) {
            eprintln!("{src}\n-----\n{}", print_file(&pf, src.as_bytes()).text);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]
    #[test]
    fn roundtrip_generated(seed in any::<u64>(), items in 1usize..8, depth in 0u32..6) {
        let src = Gen::program(seed, items, depth);
        let pf = check_roundtrip(&format!("seed {seed}"), src.as_bytes());
        prop_assert!(pf.diagnostics.is_empty(), "{:?}\n{}", pf.diagnostics.first(), src);
    }

    /// Generated programs with other whitespace and comments between their tokens: the round
    /// trip holds, and the formatter's own check of its output passes.
    #[test]
    fn roundtrip_generated_any_layout(seed in any::<u64>(), layout in any::<u64>(), items in 1usize..8, depth in 0u32..6) {
        let src = relayout(Gen::program(seed, items, depth).as_bytes(), layout);
        let pf = check_roundtrip(&format!("seed {seed} layout {layout}"), src.as_bytes());
        prop_assert!(pf.diagnostics.is_empty(), "{:?}\n{}", pf.diagnostics.first(), src);
        let formatted = format_source(src.as_bytes(), F);
        prop_assert!(formatted.is_ok(), "{:?}\n{}", formatted, src);
    }

    /// Arbitrary bytes: no panic, and the round trip holds whenever the file has items.
    #[test]
    fn roundtrip_arbitrary(bytes in proptest::collection::vec(any::<u8>(), 0..400)) {
        let pf = parse_file(&bytes, F);
        same_as_stepwise(&bytes, &pf);
        if pf.stop.is_none() {
            check_roundtrip("bytes", &bytes);
        }
    }

    /// Generated programs with a few bytes flipped: broken items, still no panic.
    #[test]
    fn roundtrip_mutated(seed in any::<u64>(), flips in proptest::collection::vec((any::<usize>(), any::<u8>()), 1..4)) {
        let mut src = Gen::program(seed, 4, 3).into_bytes();
        for (at, b) in flips {
            if !src.is_empty() {
                let n = src.len();
                let pool = b"(){}[];.,<>=&|+-*/ \n\"'f@_x0";
                src[at % n] = pool[b as usize % pool.len()];
            }
        }
        let pf = parse_file(&src, F);
        same_as_stepwise(&src, &pf);
        if pf.stop.is_none() {
            check_roundtrip("mutated", &src);
        }
    }
}

#[test]
fn roundtrip_lang_md_and_conformance_files() {
    for (i, b) in lang_md_blocks().iter().enumerate() {
        check_roundtrip(&format!("LANG.md block {i}"), b.as_bytes());
    }
    for f in stave_files(&repo().join("tests/conformance")) {
        let src = std::fs::read(&f).unwrap();
        let pf = parse_file(&src, F);
        if pf.stop.is_none() {
            check_roundtrip(&f.display().to_string(), &src);
        }
    }
}

#[test]
fn printer_keeps_a_leading_feff_identifier() {
    check_roundtrip("feff", "\n\u{FEFF}\n".as_bytes());
    check_roundtrip("feff fn", "x\u{FEFF} fn f() { }".as_bytes());
}

#[test]
fn printer_keeps_the_shebang_line() {
    let src = "#!/usr/bin/env -S stave run\n// c\nimport std.io;\nfn main()->i32{return 0;}";
    let pf = check_roundtrip("shebang", src.as_bytes());
    assert!(pf.diagnostics.is_empty());
    assert_eq!(
        print_file(&pf, src.as_bytes()).text,
        "#!/usr/bin/env -S stave run\n// c\nimport std.io;\nfn main() -> i32 { return 0; }\n"
    );
    for src in
        ["#!", "#!x", "#!x\n", "#!x\r\nfn f() { }", "#! \"a\nfn f() { let = 1; }\n", "#!a\n#!b\nfn f() { }"]
    {
        let pf = check_roundtrip(src, src.as_bytes());
        let text = print_file(&pf, src.as_bytes()).text;
        // trailing whitespace of the line goes, as in a comment
        let line = src.split('\n').next().unwrap().trim_end();
        assert!(text.starts_with(&format!("{line}\n")), "{src:?}: {text:?}");
    }
    // a dropped first item that starts with `#!` (not at offset 0) must not become a shebang line
    for src in ["\n#!x y\nfn f() { }", " #!", "x\n#!y", "\n\n#!/bin/x\nfn f() { let = 1; }"] {
        let pf = check_roundtrip(src, src.as_bytes());
        assert!(!print_file(&pf, src.as_bytes()).text.starts_with("#!"), "{src:?}");
    }
}

#[test]
fn printer_layout_example() {
    let src = "fn f(x:i32)->i32{let v:Vec<Vec<i32>> =g::<u8>(x);if x<1{return -x;}else{loop{break;}}\
               match x{1=>{}_=>{}}return t.0.1+a[i]?;}";
    let pf = parse_file(src.as_bytes(), F);
    let p = print_file(&pf, src.as_bytes());
    let want = "fn f(x: i32) -> i32 {\n    let v: Vec<Vec<i32>> = g::<u8>(x);\n    \
                if x < 1 { return -x; } else { loop { break; } }\n    match x {\n        1 => {}\n        _ => {}\n    }\n    \
                return t.0.1 + a[i]?;\n}\n";
    assert_eq!(p.text, want);
}

/// A large file mixing good items, broken items, and lexical errors.
fn big_file() -> String {
    let mut s = String::new();
    for seed in 0..400 {
        s.push_str(&Gen::program(seed, 5, 3)); // later imports are E0004 (LateImport)
        match seed % 7 {
            0 => s.push_str("fn broken() { let = 1; }\n"),
            1 => s.push_str("fn lexbad() { let s = f\"{a+b}\"; }\n"),
            2 => s.push_str("fn cmp() { let b = a < b < c; }\n"),
            3 => s.push_str("fn lit() { let s = S { a: 1 }; }\n"),
            _ => {}
        }
    }
    for b in lang_md_blocks() {
        s.push_str(&b);
    }
    s
}

#[test]
fn item_parallel_parsing_is_deterministic() {
    let src = big_file();
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| parse_file(src.as_bytes(), F))
    };
    let base = run(1);
    same_as_stepwise(src.as_bytes(), &base);
    assert!(base.items.len() > 1000);
    assert!(base.diagnostics.len() > 100);
    for threads in [2, 12] {
        let pf = run(threads);
        assert_eq!(pf.tokens, base.tokens, "{threads} threads");
        assert_eq!(pf.items, base.items, "{threads} threads");
        assert!(pf.ast == base.ast, "{threads} threads: AST differs");
        assert_eq!(pf.diagnostics, base.diagnostics, "{threads} threads");
    }
}
