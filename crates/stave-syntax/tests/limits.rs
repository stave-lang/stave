//! Limits: > 4 GiB files, bracket depth 10,000 / 10,001,
//! E0016 over E0005, and long bracket-free chains on a small stack (the parser is iterative).

mod common;

use common::F;
use stave_syntax::{FileStop, MAX_DEPTH, map_file, parse_file, print_file};

/// Runs `f` in a one-thread rayon pool with a small stack (the parser's item tasks run there too):
/// recursion over nesting depth would overflow it.
fn small_stack<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).stack_size(256 * 1024).build().expect("pool");
    pool.install(f)
}

#[test]
fn sparse_file_over_4gib_is_one_e0002() {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let path = dir.join("stave_syntax_e0002_sparse.stv");
    let f = std::fs::File::create(&path).expect("create");
    let len = (1u64 << 32) + 1;
    f.set_len(len).expect("sparse set_len");
    drop(f);
    let bytes = map_file(&path).expect("map");
    assert_eq!(bytes.len() as u64, len);
    let pf = parse_file(&bytes, F);
    let _ = std::fs::remove_file(&path);
    assert_eq!(pf.stop, Some(FileStop::TooLarge));
    assert_eq!(pf.diagnostics.len(), 1);
    let d = &pf.diagnostics[0];
    assert_eq!(d.code(), "E0002");
    assert_eq!(d.primary().map(|s| s.offset), Some(0));
    assert_eq!(d.message(), "source file is 4294967297 bytes, over the 4 GiB limit");
    assert!(pf.items.is_empty());
    assert_eq!(pf.tokens.len(), 1, "only EOF");
}

/// `fn f() { let x = ((…(1)…)); }` with total bracket depth `depth` (the body `{` is level 1).
fn nested(depth: usize) -> String {
    let n = depth - 1;
    format!("fn f() {{ let x = {}1{}; }}\n", "(".repeat(n), ")".repeat(n))
}

#[test]
fn depth_10000_is_accepted() {
    let src = nested(MAX_DEPTH);
    let pf = small_stack(move || parse_file(src.as_bytes(), F));
    assert!(pf.diagnostics.is_empty(), "{:?}", pf.diagnostics.first());
    assert_eq!(pf.items.len(), 1);
    assert!(pf.items[0].ok);
}

#[test]
fn depth_10001_is_one_e0005_at_the_first_opener_reaching_it() {
    // A lexical error is still reported; the syntax error of the other item is not.
    let src = format!("fn g() {{ let s = f\"{{a+b}}\"; }}\nfn h() {{ let = ; }}\n{}", nested(MAX_DEPTH + 1));
    let pf = small_stack({
        let src = src.clone();
        move || parse_file(src.as_bytes(), F)
    });
    let codes: Vec<&str> = pf.diagnostics.iter().map(|d| d.code()).collect();
    assert_eq!(codes, vec!["E0006", "E0005"], "{:#?}", pf.diagnostics);
    assert_eq!(pf.stop, Some(FileStop::TooDeep));
    assert!(pf.items.is_empty());
    let off = pf.diagnostics[1].primary().unwrap().offset as usize;
    let body = src.rfind("fn f() {").unwrap() + "fn f() {".len() - 1;
    // the body `{` is depth 1, so the 10,000th `(` reaches 10,001
    let parens = &src[body..=off];
    assert_eq!(parens.matches('(').count(), MAX_DEPTH);
    assert_eq!(&src[off..off + 1], "(");
}

#[test]
fn unbalanced_and_too_deep_is_e0016_only() {
    let mut src = nested(MAX_DEPTH + 5);
    let last_paren = src.rfind(')').unwrap();
    src.remove(last_paren);
    let pf = small_stack(move || parse_file(src.as_bytes(), F));
    let codes: Vec<&str> = pf.diagnostics.iter().map(|d| d.code()).collect();
    assert_eq!(codes, vec!["E0016"]);
    assert_eq!(pf.stop, Some(FileStop::Unbalanced));
}

#[test]
fn long_unary_chain_parses_iteratively() {
    let src = format!("fn f(y: bool) -> bool {{ return {}y; }}\n", "!".repeat(100_000));
    let pf = small_stack({
        let src = src.clone();
        move || {
            let pf = parse_file(src.as_bytes(), F);
            let printed = print_file(&pf, src.as_bytes());
            (pf, printed)
        }
    });
    let (pf, printed) = pf;
    assert!(pf.diagnostics.is_empty());
    assert!(pf.items[0].ok);
    assert_eq!(printed.fallbacks, 0);
    let unary = pf.item_ast(0).nodes.iter().filter(|n| n.kind() == stave_syntax::NodeKind::Unary).count();
    assert_eq!(unary, 100_001);
}

#[test]
fn long_else_if_chain_parses_iteratively() {
    let mut src = String::from("fn f(a: bool) {\n    if a { }");
    for _ in 0..100_000 {
        src.push_str(" else if a { }");
    }
    src.push_str(" else { }\n}\n");
    let pf = small_stack(move || parse_file(src.as_bytes(), F));
    assert!(pf.diagnostics.is_empty());
    let ifs = pf.item_ast(0).nodes.iter().filter(|n| n.kind() == stave_syntax::NodeKind::IfStmt).count();
    assert_eq!(ifs, 100_001);
}
