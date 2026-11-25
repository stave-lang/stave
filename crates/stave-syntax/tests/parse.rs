//! Parser, item finder, and refinement tests.

mod common;

use common::{F, codes, one, one_at, repo};
use stave_diag::{SourceFiles, emit_json};
use stave_syntax::{ItemKind, NodeKind, Term, items, lex, parse_file, token_text};

fn ok(src: &str) {
    let pf = parse_file(src.as_bytes(), F);
    assert!(pf.diagnostics.is_empty(), "{src}\n{:#?}", pf.diagnostics);
    assert!(pf.items.iter().all(|i| i.ok), "{src}");
}

#[test]
fn lang_md_blocks_parse() {
    for (i, b) in common::lang_md_blocks().iter().enumerate() {
        let pf = parse_file(b.as_bytes(), F);
        assert!(pf.diagnostics.is_empty(), "LANG.md block {i}: {:#?}", pf.diagnostics);
        assert!(pf.items.iter().all(|i| i.ok));
        assert!(!pf.items.is_empty());
    }
}

#[test]
fn stage1_generated_files_parse() {
    for f in [
        "src/generated/ll1_table.stv",
        "src/generated/terminals.stv",
        "src/generated/node_kinds.stv",
        "ir/src/generated/ir_text_ll1_table.stv",
        "ir/src/generated/ir_text_terminals.stv",
        "ir/src/generated/ir_text_node_kinds.stv",
        "src/generated/diag_templates.stv",
    ] {
        let p = repo().join("stage1").join(f);
        let Ok(src) = std::fs::read(&p) else { continue };
        let pf = parse_file(&src, F);
        assert!(pf.diagnostics.is_empty(), "{f}: {:#?}", pf.diagnostics.first());
    }
}

#[test]
fn ast_shape_postorder() {
    let src = "fn f() {}";
    let pf = parse_file(src.as_bytes(), F);
    let ast = pf.item_ast(0);
    let got: Vec<(NodeKind, u32, u32)> = ast.nodes.iter().map(|n| (n.kind(), n.a, n.b)).collect();
    assert_eq!(
        got,
        vec![
            (NodeKind::Block, 4, 0),
            (NodeKind::FnDecl, 0, 1),
            (NodeKind::ItemBody, 0, 2),
            (NodeKind::Item, 0, 3)
        ]
    );
    assert_eq!(ast.root(), Some(3));
    assert_eq!(ast.children(3), vec![2]);
    assert_eq!(ast.children(1), vec![0]);
    assert_eq!(ast.parents(), vec![1, 2, 3, u32::MAX]);
}

#[test]
fn ast_children_in_source_order() {
    let src = "const X: i32 = a + b * c;";
    let pf = parse_file(src.as_bytes(), F);
    let ast = pf.item_ast(0);
    let add = (0..ast.len() as u32).find(|&n| ast.kind(n) == Some(NodeKind::Add)).unwrap();
    let kids = ast.children(add);
    assert_eq!(kids.len(), 2);
    let a: Vec<u32> = kids.iter().map(|&k| ast.get(k).unwrap().a).collect();
    assert_eq!(a, vec![5, 7], "operands start at `a` and `b`");
    assert_eq!(ast.kind(kids[0]), Some(NodeKind::Primary));
    assert_eq!(ast.child(add, NodeKind::Mul), Some(kids[1]));
    // a node matching no token points at the next token (ArrOrIdx in `[]` permute args)
    let pf = parse_file(b"fn f() { let v = x.p::<[]>(); }", F);
    let ast = pf.item_ast(0);
    let n = (0..ast.len() as u32).find(|&n| ast.kind(n) == Some(NodeKind::ArrOrIdx)).unwrap();
    let a = ast.get(n).unwrap().a as usize;
    assert_eq!(pf.tokens[a].kind, Term::RBracket as u8);
    assert_eq!(ast.get(n).unwrap().b, 0);
}

#[test]
fn pass_through_levels_make_no_node() {
    use NodeKind as K;
    // `const X: i32 = e;` is TypeBase, Type, e's nodes, ConstDecl, ItemBody, Item
    let expr = |e: &str| -> Vec<K> {
        let src = format!("const X: i32 = {e};");
        let pf = parse_file(src.as_bytes(), F);
        assert!(pf.diagnostics.is_empty() && pf.items[0].ok, "{src}");
        let kinds: Vec<K> = pf.item_ast(0).nodes.iter().map(|n| n.kind()).collect();
        assert_eq!(kinds[..2], [K::TypeBase, K::Type], "{src}");
        assert_eq!(kinds[kinds.len() - 3..], [K::ConstDecl, K::ItemBody, K::Item], "{src}");
        kinds[2..kinds.len() - 3].to_vec()
    };
    assert_eq!(expr("x"), [K::Primary]);
    assert_eq!(expr("(x)"), [K::Primary, K::Primary]);
    assert_eq!(expr("-x"), [K::Primary, K::Unary]);
    assert_eq!(expr("&&x"), [K::Primary, K::Unary, K::Unary]);
    assert_eq!(expr("unsafe(\"ok\") *p"), [K::Unsafe, K::Primary, K::Unary, K::Unary]);
    assert_eq!(expr("x?"), [K::Primary, K::Postfix]);
    assert_eq!(expr("f()"), [K::Primary, K::Postfix]);
    assert_eq!(expr("f(a)"), [K::Primary, K::Primary, K::Args, K::Postfix]);
    assert_eq!(expr("S.{ a: 1 }"), [K::Primary, K::Primary, K::StructLit, K::Postfix]);
    assert_eq!(expr("x as u8"), [K::Primary, K::TypeBase, K::Type, K::Cast]);
    assert_eq!(expr("a + b * c"), [K::Primary, K::Primary, K::Primary, K::Mul, K::Add]);
    assert_eq!(expr("a < b"), [K::Primary, K::CmpOp, K::Primary, K::CmpExpr]);
    assert_eq!(expr("a || b && c"), [K::Primary, K::Primary, K::Primary, K::AndExpr, K::OrExpr]);
    assert_eq!(expr("0..n"), [K::Primary, K::Primary, K::RangeExpr]);
    assert_eq!(expr("if c then a else b"), [K::Primary, K::Primary, K::Primary, K::IfExpr]);
    // the stepwise PDA omits the same nodes
    let src = b"fn f() { let v = -a.b(c, d?)[0] as u8 + (x) * y; g(h); }";
    assert_eq!(parse_file(src, F).ast, stave_syntax::parse_file_stepwise(src, F).ast);
}

#[test]
fn d3_split_inside_type_args() {
    ok("fn f() { let v: Vec<Vec<i32>> = x; let w: Vec<i32>= x; let z: A<B<C>>= g::<[1, 2], {N + 1}, 4>(); }");
    ok("fn f() { let o: Option<Option<i32>> = y; }");
    ok("fn f(x: u64) { x >>= 2; let y = a >> b; let c = a >= b; }");
    // `a >> b` is a shift: a Shift node with two Add children
    let pf = parse_file(b"const X: u32 = a >> b;", F);
    let ast = pf.item_ast(0);
    let sh = (0..ast.len() as u32).find(|&n| ast.kind(n) == Some(NodeKind::Shift)).unwrap();
    assert_eq!(ast.children(sh).len(), 2);
    // `>>` closing two lists: no CmpOp node anywhere
    let pf = parse_file(b"type T = A<B<C>>;", F);
    assert!(pf.item_ast(0).nodes.iter().all(|n| n.kind() != NodeKind::CmpOp));
}

#[test]
fn ampamp_in_unary_position_is_two_refs() {
    ok("fn f() { let r = &&x; let s = &&mut x; let t = a && b; }");
    let pf = parse_file(b"const R: &&i32 = &&X;", F);
    let ast = pf.item_ast(0);
    let unary_amp = ast
        .nodes
        .iter()
        .filter(|n| n.kind() == NodeKind::Unary && pf.tokens[n.a as usize].kind == Term::AmpAmp as u8)
        .count();
    assert_eq!(unary_amp, 2, "both Unary nodes start at the `&&` token");
}

#[test]
fn refine_e0013() {
    assert_eq!(one_at("fn f() { unsafe { } }"), ("E0013", "unsafe".into()));
    assert_eq!(one_at("fn f() { unsafe(x) { } }"), ("E0013", "unsafe".into()));
    assert_eq!(one_at("unsafe fn f() { }"), ("E0013", "unsafe".into()));
    assert_eq!(one_at("fn f() { let a = unsafe *p; }"), ("E0013", "unsafe".into()));
    ok("fn f() { unsafe(\"ok\") { *p = 1; } let b = unsafe(\"ok\") *p; }");
    // statement-start unsafe always opens a block
    assert_eq!(one_at("fn f() { unsafe(\"x\") *p = 1; }"), ("E0004", "*".into()));
}

#[test]
fn refine_e0011() {
    assert_eq!(one_at("fn f() { let b = x as usize < y; }"), ("E0011", "<".into()));
    assert_eq!(one_at("fn f() { let b = x as Vec<u8 ; }"), ("E0011", "<".into()));
    assert_eq!(one_at("fn f() { let b = x as &usize < y; }"), ("E0011", "<".into()));
    assert_eq!(one_at("fn f() { let b = x as fn() -> u8 < y; }"), ("E0011", "<".into()));
    ok("fn f() { let b = (x as usize) < y; let v = x as Vec<u8>; let w = x as% u8 == 1u8; }");
    // a later error after the D2 list was closed is not E0011
    assert_eq!(one_at("fn f() { let v = x as Vec<u8> < ; }").0, "E0004");
}

#[test]
fn refine_e0014() {
    let d = one("fn f() { let s = S { a: 1 }; }");
    assert_eq!(d.code(), "E0014");
    assert_eq!(d.message(), "struct literal or pattern after `S` needs `.{`");
    assert_eq!(one_at("fn f() { let s = S {}; }"), ("E0014", "{".into()));
    assert_eq!(one_at("fn f() { let s = S { a, b }; }"), ("E0014", "{".into()));
    assert_eq!(one_at("fn f() { match x { S { a } => { } } }"), ("E0014", "{".into()));
    let d = one("fn f() { if x == Shape.Rect { w: 1 } { } }");
    assert_eq!(d.code(), "E0014");
    assert_eq!(d.message(), "struct literal or pattern after `Shape.Rect` needs `.{`");
    ok("fn f() { if x == S.{ a: 1 } { } let p = Shape.Rect.{ w: 1.0, h: 2.0 }; }");
    // a `{` after a name that is a plain block stays fine
    ok("fn f() { while x { a = 1; } }");
}

#[test]
fn refine_e0012() {
    assert_eq!(one_at("fn f() { let b = a < b < c; }"), ("E0012", "<".into()));
    assert_eq!(one_at("fn f() { let b = a == b != c; }"), ("E0012", "!=".into()));
    // CmpOp outside an expression stays E0004
    assert_eq!(one_at("fn f() { let x < 3; }"), ("E0004", "<".into()));
}

#[test]
fn e0004_expected_sets_in_terminal_order() {
    let d = one("fn f() -> i32 { return 1 }");
    assert_eq!(d.message(), "unexpected }, expected one of: ;");
    let d = one("fn f() { let x = ; }");
    assert_eq!(
        d.message(),
        "unexpected ;, expected one of: !, &, (, *, -, CharLit, FloatLit, FmtLit, Ident, IntLit, Self, \
         StrLit, [, false, fn, if, match, self, true, unsafe, ~"
    );
    // the found token is spelled by kind
    assert!(one("fn f() { let x = 1 2; }").message().starts_with("unexpected IntLit, expected one of: "));
    // a byte that starts no token is E0018 (lexical), and its item gets no E0004
    assert_eq!(one_at("fn f() { let x = @; }"), ("E0018", "@".into()));
    // fn types have no `from`
    assert_eq!(one_at("struct S { g: fn(&T) -> &T from x }").0, "E0004");
}

/// The parent kind of every `Without` node of item 0, and the effect names it lists.
fn withouts(src: &str) -> Vec<(NodeKind, Vec<String>)> {
    let pf = parse_file(src.as_bytes(), F);
    assert!(pf.diagnostics.is_empty(), "{src}\n{:#?}", pf.diagnostics);
    let ast = pf.item_ast(0);
    let parents = ast.parents();
    let text = |n: u32| {
        let a = ast.get(n).unwrap().a as usize;
        String::from_utf8_lossy(token_text(src.as_bytes(), &pf.tokens, a)).into_owned()
    };
    (0..ast.len() as u32)
        .filter(|&n| ast.kind(n) == Some(NodeKind::Without))
        .map(|n| {
            let effects =
                ast.children(n).into_iter().filter(|&c| ast.kind(c) == Some(NodeKind::Effect)).map(text);
            (ast.kind(parents[n as usize]).unwrap(), effects.collect())
        })
        .collect()
}

#[test]
fn rec_and_without_in_the_tree() {
    let src = "rec fn f(n: u32) -> u32 without(alloc, unsafe) { return n; }";
    let pf = parse_file(src.as_bytes(), F);
    let ast = pf.item_ast(0);
    let f = (0..ast.len() as u32).find(|&n| ast.kind(n) == Some(NodeKind::FnDecl)).unwrap();
    assert_eq!(pf.tokens[ast.get(f).unwrap().a as usize].kind, Term::KwRec as u8, "FnDecl starts at `rec`");
    assert_eq!(withouts(src), vec![(NodeKind::FnDecl, vec!["alloc".to_string(), "unsafe".to_string()])]);
    // D5: `without` directly after a fn type belongs to that fn type
    let w = |e: &str| vec![e.to_string()];
    assert_eq!(withouts("fn f() -> fn() without(io) { }"), vec![(NodeKind::TypeBase, w("io"))]);
    assert_eq!(withouts("fn f() -> (fn()) without(io) { }"), vec![(NodeKind::FnDecl, w("io"))]);
    assert_eq!(
        withouts("fn f(g: fn(u8) -> u8 needs(avx2) without(trap)) -> u8 needs(avx2) without(lock) { }"),
        vec![(NodeKind::TypeBase, w("trap")), (NodeKind::FnDecl, w("lock"))]
    );
    // the parser takes any name; the closed effect set is checked after parsing
    assert_eq!(withouts("fn f() without(frobnicate) { }"), vec![(NodeKind::FnDecl, w("frobnicate"))]);
    // `rec` starts an item
    let l = lex(b"fn a() { } rec fn b() { }", F);
    assert_eq!(items(&l.tokens, F).unwrap().items.len(), 2);
    ok(
        "test rec fn t() { }\nimpl S { pub unsafe(\"x\") rec fn m(&self) { } }\npub unsafe(\"x\") rec fn u() { }",
    );
    // the reserved words are keywords, never names
    for w in [
        "rec",
        "without",
        "invariant",
        "decreases",
        "ghost",
        "pure",
        "spec",
        "forall",
        "exists",
        "derive",
        "async",
        "await",
        "yield",
    ] {
        assert_eq!(one_at(&format!("fn f() {{ let {w} = 1; }}")), ("E0004", w.into()), "{w}");
    }
}

#[test]
fn item_cut() {
    let src =
        "import a;\nimport b.c as d;\npub fn f() { }\nstruct S { a: i32 }\nconst X: i32 = 1; type A = B;";
    let l = lex(src.as_bytes(), F);
    let t = items(&l.tokens, F).unwrap();
    let kinds: Vec<ItemKind> = t.items.iter().map(|i| i.kind).collect();
    use ItemKind::*;
    assert_eq!(kinds, vec![Import, Import, Item, Item, Item, Item]);
    // `fn` after `pub` and a keyword not after `;`/`}` start no item
    let l = lex(b"const X: i32 = 1 fn g() { }", F);
    assert_eq!(items(&l.tokens, F).unwrap().items.len(), 1);
    assert_eq!(one_at("const X: i32 = 1 fn g() { }"), ("E0004", "fn".into()));
}

#[test]
fn recovery_per_item() {
    // two broken items: two E0004, first error each; the good item between them is ok
    let pf = parse_file(b"fn a() { let = 1; let = 2; }\nfn b() { }\nfn c() { return }", F);
    assert_eq!(pf.diagnostics.iter().map(|d| d.code()).collect::<Vec<_>>(), vec!["E0004", "E0004"]);
    assert_eq!(pf.items.iter().map(|i| i.ok).collect::<Vec<_>>(), vec![false, true, false]);
    // tokens before the first item
    assert_eq!(one_at("x y fn f() { }"), ("E0004", "x".into()));
    let d = one("x y fn f() { }");
    assert!(
        d.message().starts_with("unexpected Ident, expected one of: EOF, allocator, const"),
        "{}",
        d.message()
    );
    // an import after an item
    assert_eq!(one_at("fn f() { } import a;"), ("E0004", "import".into()));
    let d = one("fn f() { } import a;");
    assert!(!d.message().contains(", import"), "{}", d.message());
    // an item holding an error token: lexical diagnostics only
    assert_eq!(codes("fn f() { let s = f\"{a+b}\"; let = ; }"), vec!["E0006"]);
    assert_eq!(codes("\"abc\nfn f() { }"), vec!["E0015"]);
}

#[test]
fn e0016_locations() {
    assert_eq!(one_at("fn f() { let a = ( ]; }"), ("E0016", "]".into()));
    assert_eq!(one("fn f() { let a = ( ]; }").message(), "] closes (");
    assert_eq!(one_at("fn f() { } )"), ("E0016", ")".into()));
    assert_eq!(one("fn f() { } )").message(), "unmatched closing )");
    assert_eq!(one_at("fn f() { let a = S.{ b: [1, 2 }"), ("E0016", "}".into()));
    assert_eq!(one_at("fn f() { if a { let s = S.{ b: 1 "), ("E0016", ".{".into()));
    assert_eq!(one("fn f() { if a { let s = S.{ b: 1 ").message(), ".{ is never closed");
    // exactly one E0016 and no E0004; lexical diagnostics still reported
    let pf = parse_file(b"fn f() { let = ; let s = f\"{1}\"; let c = '+\n}}\nfn g() { let", F);
    let mut c: Vec<&str> = pf.diagnostics.iter().map(|d| d.code()).collect();
    c.sort();
    assert_eq!(c, vec!["E0006", "E0015", "E0016"]);
}

#[test]
fn golden_json_e0004() {
    let src = "fn f() {\n    let x: [i32; 3] = [1, 2 3];\n}\n";
    let pf = parse_file(src.as_bytes(), F);
    let mut files = SourceFiles::new();
    files.add("p", "src/main.stv", src.as_bytes()).unwrap();
    let json = emit_json(&pf.diagnostics, &files).unwrap();
    let golden = repo().join("crates/stave-syntax/tests/golden/e0004_expected.json");
    if std::env::var_os("STAVE_BLESS").is_some() {
        std::fs::create_dir_all(golden.parent().unwrap()).unwrap();
        std::fs::write(&golden, &json).unwrap();
    }
    let want = std::fs::read_to_string(&golden).expect("golden file (run with STAVE_BLESS=1)");
    assert_eq!(json, want);
}

#[test]
fn diagnostics_canonically_sorted() {
    let pf = parse_file(b"fn b() { let = ; }\nfn a() { let s = f\"{1}\"; }\nfn c() { return }", F);
    let offs: Vec<u32> = pf.diagnostics.iter().map(|d| d.primary().unwrap().offset).collect();
    let mut sorted = offs.clone();
    sorted.sort();
    assert_eq!(offs, sorted);
}
