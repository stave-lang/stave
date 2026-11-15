//! The canonical layout (`format_source`, what `stave fmt` writes): golden layouts, comments
//! kept in place, files with syntax errors refused; and over the conformance files, `std`,
//! `runtime`, and the LANG.md blocks, as written and with random whitespace and comments between
//! their tokens: formatting twice changes nothing more, and the AST and the comments stay the
//! same.

mod common;

use common::{F, lang_md_blocks, relayout, repo, stave_files};
use proptest::prelude::*;
use stave_syntax::{FormatError, ParsedFile, format_source, parse_file, token_end};

fn fmt(src: &str) -> String {
    match format_source(src.as_bytes(), F) {
        Ok(t) => t,
        Err(e) => panic!("{e:?}\n--- source:\n{src}"),
    }
}

/// `src` formats to `want`, and `want` is a fixed point.
fn golden(src: &str, want: &str) {
    assert_eq!(fmt(src), want, "\n--- source:\n{src}");
    assert_eq!(fmt(want), want, "not a fixed point:\n{want}");
}

/// `text` is already canonical.
fn fixed(text: &str) {
    assert_eq!(fmt(text), text);
}

/// The `//` comments of `src` in order, without trailing whitespace: the non-blank lines of the
/// text between tokens.
fn comments(src: &[u8], pf: &ParsedFile) -> Vec<String> {
    let mut out = Vec::new();
    let mut from = 0;
    for (i, t) in pf.tokens.iter().enumerate() {
        let gap = String::from_utf8_lossy(&src[from..t.start as usize]).into_owned();
        out.extend(gap.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_string));
        from = token_end(src, &pf.tokens, i) as usize;
    }
    out
}

#[test]
fn spacing_and_one_statement_per_line() {
    golden(
        "fn f(x:i32)->i32{let a=x+1;let b=a*2;return b;}",
        "fn f(x: i32) -> i32 {\n    let a = x + 1;\n    let b = a * 2;\n    return b;\n}\n",
    );
    golden(
        "fn  f ( p : * const S , v : & mut [ u8 ] ) -> u8 {\n\treturn unsafe ( \"p is valid\" ) ( * p ) . x +% v [ 0 ] as% u8 ;\n}",
        "fn f(p: *const S, v: &mut [u8]) -> u8 {\n    return unsafe(\"p is valid\") (*p).x +% v[0] as% u8;\n}\n",
    );
    golden("import  a . b  as c ;import d;", "import a.b as c;\nimport d;\n");
    golden("fn f() {}", "fn f() {}\n");
    golden("fn f() {}\r\nfn g() {}\r\n", "fn f() {}\nfn g() {}\n");
    golden(
        "fn f() {} fn g() {} const N: u8 = 1; struct S {}",
        "fn f() {}\nfn g() {}\nconst N: u8 = 1;\nstruct S {}\n",
    );
    fixed(
        "fn f<T: Add<T>>(a: Vec<Vec<T>>) -> Option<Vec<T>> {\n    let n = g::<Vec<u8>>(a) >> 1;\n    return h(n);\n}\n",
    );
    fixed(
        "fn f(t: ((u8, u8), u8), xs: &[i32]) -> u8 {\n    for i in 0..xs.len() { g(-xs[i], !b, ~m, &mut *p, 1..=2); }\n    return t.0.1;\n}\n",
    );
}

#[test]
fn blocks_on_one_line() {
    // kept: on one line in the source, at most one statement
    fixed("fn f(c: bool) -> i32 {\n    if c { return 1; }\n    return 0;\n}\n");
    fixed("fn f() -> i32 { return 1; }\n");
    fixed("impl Area for Unit { fn area(self) -> f64 { return 1.0; } }\n");
    fixed("trait Area { fn area(self) -> f64; }\n");
    fixed(
        "fn f(c: bool) {\n    for v: f32x4 in xs { acc = acc + v; } tail masked (v, m) { acc = acc + v; }\n}\n",
    );
    // broken in the source: stays broken
    fixed("fn f() -> i32 {\n    return 1;\n}\n");
    golden("fn f() -> i32 { return 1;\n}\n", "fn f() -> i32 {\n    return 1;\n}\n");
    // two statements or members: broken, and so is every block around them
    golden("fn f() { a(); b(); }\n", "fn f() {\n    a();\n    b();\n}\n");
    golden(
        "fn f(c: bool) { if c { a(); b(); } }\n",
        "fn f(c: bool) {\n    if c {\n        a();\n        b();\n    }\n}\n",
    );
    golden("trait T { fn a(self); fn b(self); }\n", "trait T {\n    fn a(self);\n    fn b(self);\n}\n");
    // a closure body in an argument list
    golden(
        "fn f() { g(fn(x: i32) -> i32 { let y = x; return y; }, 1); }\n",
        "fn f() {\n    g(fn(x: i32) -> i32 {\n        let y = x;\n        return y;\n    }, 1);\n}\n",
    );
    fixed("fn f() {\n    let h = fn(x: i32) -> i32 { return x; };\n    return h(1);\n}\n");
    // empty
    golden(
        "fn f() {\n}\nstruct S { }\nimpl Copy for S {\n\n}\n",
        "fn f() {}\nstruct S {}\nimpl Copy for S {}\n",
    );
    golden(
        "fn f() { g( ); let a: [u8; 0] = [ ]; let s = S.{ }; loop { } }\n",
        "fn f() {\n    g();\n    let a: [u8; 0] = [];\n    let s = S.{};\n    loop {}\n}\n",
    );
}

#[test]
fn else_and_tail() {
    golden(
        "fn f(c: bool) {\n    if c {\n        a();\n    }\n    else if d {\n        b();\n    }\n    else\n    {\n        e();\n    }\n}\n",
        "fn f(c: bool) {\n    if c {\n        a();\n    } else if d {\n        b();\n    } else {\n        e();\n    }\n}\n",
    );
    // after a one-line block the source decides
    fixed("fn f(c: bool) {\n    if c { a(); }\n    else if d { b(); }\n    else { e(); }\n}\n");
    fixed("fn f(c: bool) {\n    if c { a(); } else { b(); }\n}\n");
    // the blocks of one statement are laid out alike
    golden(
        "fn f(c: bool) {\n    if c { a(); }\n    else if d { b(); e(); }\n    else {}\n}\n",
        "fn f(c: bool) {\n    if c {\n        a();\n    } else if d {\n        b();\n        e();\n    } else {}\n}\n",
    );
    golden(
        "fn f(xs: &[f32]) {\n    for v: f32x4 in xs { a(); b(); } tail scalar (x) { c(); }\n    for v: f32x4 in xs {\n        a();\n    }\n    tail exact;\n}\n",
        "fn f(xs: &[f32]) {\n    for v: f32x4 in xs {\n        a();\n        b();\n    } tail scalar (x) {\n        c();\n    }\n    for v: f32x4 in xs {\n        a();\n    } tail exact;\n}\n",
    );
}

#[test]
fn lists_follow_the_source() {
    // broken after the opener: the closer gets its own line, the elements one level deeper
    golden(
        "fn f() {\n    g(\n      a, b,\n      c);\n}\n",
        "fn f() {\n    g(\n        a, b,\n        c\n    );\n}\n",
    );
    // not broken after the opener: the closer stays, later lines are continuations
    golden("fn f() {\n    g(a,\n      b\n    );\n}\n", "fn f() {\n    g(a,\n        b);\n}\n");
    golden("struct S {\n  a: i32,\n      pub b: u8,\n}\n", "struct S {\n    a: i32,\n    pub b: u8,\n}\n");
    fixed("struct S { a: i32, pub b: u8 }\n");
    fixed("enum E { A, B(u8), C { x: i32 } }\n");
    fixed("enum E {\n    A,\n    B(u8),\n    C { x: i32 },\n}\n");
    fixed("const T: [u64; 4] = [\n    1, 2,\n    3, 4,\n];\n");
    fixed(
        "fn f() -> S {\n    return S.{\n        a: 1, b: 2,\n        c: g(\n            3,\n        ),\n    };\n}\n",
    );
    fixed("dispatch fn sum(xs: &[f32]) -> f32 { avx2 => sum_v3, _ => sum_base, }\n");
    fixed("dispatch fn sum(xs: &[f32]) -> f32 {\n    avx2 => sum_v3,\n    _ => sum_base,\n}\n");
    fixed("fn long(\n    a: i32,\n    b: i32,\n) -> i32 {\n    return a + b;\n}\n");
}

#[test]
fn continuation_lines() {
    golden(
        "fn f() -> bool {\n    return a\n    && b\n            || c;\n}\n",
        "fn f() -> bool {\n    return a\n        && b\n        || c;\n}\n",
    );
    golden(
        "fn f() {\n    let x = a\n    .b()\n    .c(\n    1,\n    );\n    let y =\n    if c then 1\n    else 2;\n}\n",
        "fn f() {\n    let x = a\n        .b()\n        .c(\n            1,\n        );\n    let y =\n        if c then 1\n        else 2;\n}\n",
    );
    // a statement block opened on a continuation line is indented from the statement
    golden(
        "fn f() {\n    if a\n    && b {\n    g();\n    h();\n    }\n}\n",
        "fn f() {\n    if a\n        && b {\n        g();\n        h();\n    }\n}\n",
    );
    golden(
        "fn f(a: i32,\n  b: i32)\n-> i32 {\n    return a;\n}\n",
        "fn f(a: i32,\n    b: i32)\n    -> i32 {\n    return a;\n}\n",
    );
    // no break where the tokens are glued, and none before `{`
    golden("fn f\n(x\n: i32\n)\n{\n    g\n    (x\n    )\n    ;\n}\n", "fn f(x: i32) {\n    g(x);\n}\n");
    // the line after an item's or member's `unsafe(..)` is not a continuation
    fixed("pub unsafe(\"p is valid\")\nfn f(p: *const u8) {}\n");
    fixed("impl S {\n    pub unsafe(\"p is valid\")\n    fn f(p: *const u8) {}\n}\n");
    fixed("fn f(p: *const u8) -> u8 {\n    return unsafe(\"p is valid\")\n        *p;\n}\n");
}

#[test]
fn match_statements_and_values() {
    golden(
        "fn f(x: i32) -> i32 {\n    match x {\n    1 => { a(); }\n    2 | 3 => {\n    b(); }\n    n if n > 3 => {}\n    _ => {}\n    }\n    return match x { 1 => 2, _ => 3 };\n}\n",
        "fn f(x: i32) -> i32 {\n    match x {\n        1 => { a(); }\n        2 | 3 => {\n            b();\n        }\n        n if n > 3 => {}\n        _ => {}\n    }\n    return match x { 1 => 2, _ => 3 };\n}\n",
    );
    golden(
        "fn f(x: i32) -> i32 {\n    return match x {\n    1 => 2,\n      _ => 3,\n    };\n}\n",
        "fn f(x: i32) -> i32 {\n    return match x {\n        1 => 2,\n        _ => 3,\n    };\n}\n",
    );
    fixed("fn f(x: i32) {\n    match x { _ => {} }\n}\n");
}

#[test]
fn blank_lines() {
    golden(
        "\n\nimport a;\n\n\n\nimport b;\nfn f() {\n\n    a();\n\n\n    b();\n\n}\n\n\n\nfn g() {}\n\n\n",
        "import a;\n\nimport b;\nfn f() {\n    a();\n\n    b();\n}\n\nfn g() {}\n",
    );
    golden("struct S {\n\n    a: i32,\n\n    b: i32,\n\n}\n", "struct S {\n    a: i32,\n\n    b: i32,\n}\n");
    // none inside a statement
    golden(
        "fn f() {\n    let x = a\n\n        + b;\n\n    g(a,\n\n        b);\n}\n",
        "fn f() {\n    let x = a\n        + b;\n\n    g(a,\n        b);\n}\n",
    );
    golden("", "");
    golden("\n \n\t\n", "");
}

#[test]
fn trailing_comments() {
    // on the line of the code they follow; a run of consecutive lines is aligned
    golden(
        "fn f() {\n    a();// one\n    let x = b();        // two\n\n    c();   // alone\n}   // end\n",
        "fn f() {\n    a();         // one\n    let x = b(); // two\n\n    c(); // alone\n}        // end\n",
    );
    golden("fn f() { // why\n    a(); }\n", "fn f() { // why\n    a();\n}\n");
    golden("fn f() {} // x  \t\n", "fn f() {} // x\n");
    golden("fn f() {} // c\r\n", "fn f() {} // c\n");
    fixed("const T: [u64; 4] = [\n    1, 2, // first\n    3, 4, // second\n];\n");
    fixed("fn f() -> i32 {\n    return a // left\n        + b; // right\n}\n");
}

#[test]
fn comments_between_items() {
    golden(
        "// head\n//\n// more\n\n\nimport a; // the import\n   // about f\nfn f() {}\n\n\n// tail\n",
        "// head\n//\n// more\n\nimport a; // the import\n// about f\nfn f() {}\n\n// tail\n",
    );
    fixed("// only a comment\n");
    golden("// no newline at the end", "// no newline at the end\n");
    fixed("fn f() {}\n// between\nfn g() {}\n");
}

#[test]
fn comments_in_empty_blocks() {
    golden("fn f() {\n// nothing to do\n}\n", "fn f() {\n    // nothing to do\n}\n");
    golden("fn f() { // nothing\n}\n", "fn f() { // nothing\n}\n");
    golden(
        "fn f(x: i32) {\n    match x {\n        1 => {\n            // skip\n\n        }\n        _ => { }\n    }\n    g(\n        // no arguments\n    );\n}\nstruct S {\n    // no fields\n}\n",
        "fn f(x: i32) {\n    match x {\n        1 => {\n            // skip\n        }\n        _ => {}\n    }\n    g(\n        // no arguments\n    );\n}\nstruct S {\n    // no fields\n}\n",
    );
}

#[test]
fn comments_before_else() {
    fixed(
        "fn f(c: bool) {\n    if c {\n        a();\n    }\n    // otherwise\n    else {\n        b();\n    }\n}\n",
    );
    golden(
        "fn f(c: bool) {\n    if c {\n        a();\n    } // done\n    else {\n        b();\n        // last\n    }\n}\n",
        "fn f(c: bool) {\n    if c {\n        a();\n    } // done\n    else {\n        b();\n        // last\n    }\n}\n",
    );
}

#[test]
fn comments_in_match_arms() {
    fixed(
        "fn f(x: i32) -> i32 {\n    match x {\n        // one\n        1 => { a(); } // done\n        // the rest\n        _ => {\n            // nothing\n        }\n    }\n    return match x {\n        1 => 2, // two\n        // other\n        _ => 3,\n    };\n}\n",
    );
}

#[test]
fn comments_in_argument_lists() {
    golden(
        "fn f() {\n    g(a, // first\n      b);\n    g(\n        a,\n          // second\n        b, // last\n    );\n    g(a,\n      b // closing\n    );\n}\n",
        "fn f() {\n    g(a, // first\n        b);\n    g(\n        a,\n        // second\n        b, // last\n    );\n    g(\n        a,\n        b // closing\n    );\n}\n",
    );
    fixed("fn f(\n    a: i32, // left\n    // right\n    b: i32,\n) {}\n");
}

#[test]
fn files_with_syntax_errors_are_refused() {
    for src in [
        "fn f( {",
        "fn f() { let = 1; }\n",
        "fn f() {}\nfn g() { let s = S { a: 1 }; }\n",
        "fn f() { let c = 'a; }\n",
        "fn f() { /* no */ }\n",
        "\u{FEFF}fn f() {}\n",
        "fn f() {}\nimport a;\n",
        "x fn f() {}\n",
    ] {
        match format_source(src.as_bytes(), F) {
            Err(FormatError::Diagnostics(d)) => assert!(!d.is_empty(), "{src}"),
            other => panic!("{src}: {other:?}"),
        }
    }
    assert!(matches!(format_source(b"fn f() {} // \xff\n", F), Err(FormatError::Diagnostics(_))));
}

/// Formats `src` if it has no lexical or syntax error and checks the result. Returns whether it
/// was formatted.
fn check(name: &str, src: &[u8]) -> bool {
    let pf = parse_file(src, F);
    if !pf.diagnostics.is_empty() {
        assert!(matches!(format_source(src, F), Err(FormatError::Diagnostics(_))), "{name}");
        return false;
    }
    let once = match format_source(src, F) {
        Ok(t) => t,
        Err(e) => panic!("{name}: {e:?}"),
    };
    let twice = match format_source(once.as_bytes(), F) {
        Ok(t) => t,
        Err(e) => panic!("{name}: formatted text: {e:?}\n{once}"),
    };
    assert!(once == twice, "{name}: formatting twice differs\n--- once:\n{once}\n--- twice:\n{twice}");
    let pf2 = parse_file(once.as_bytes(), F);
    // the same tree: node kinds and subtree sizes, positions aside
    let shape = |p: &ParsedFile| p.ast.nodes().map(|n| (n.kind, n.b)).collect::<Vec<_>>();
    assert!(shape(&pf) == shape(&pf2), "{name}: the AST changed\n{once}");
    assert_eq!(pf.items.len(), pf2.items.len(), "{name}");
    assert_eq!(comments(src, &pf), comments(once.as_bytes(), &pf2), "{name}: the comments changed");
    assert!(once.is_empty() || once.ends_with('\n') && !once.ends_with("\n\n"), "{name}: file end");
    assert!(!once.starts_with('\n') && !once.contains("\n\n\n"), "{name}: blank lines");
    for line in once.lines() {
        assert!(!line.ends_with([' ', '\t', '\r']), "{name}: trailing whitespace in {line:?}");
        let indent = line.len() - line.trim_start().len();
        assert!(
            line[..indent].bytes().all(|b| b == b' ') && indent % 4 == 0,
            "{name}: indentation of {line:?}"
        );
    }
    true
}

#[test]
fn conformance_files_are_stable() {
    let files = stave_files(&repo().join("tests/conformance"));
    let mut formatted = 0;
    for f in &files {
        let src = std::fs::read(f).unwrap();
        formatted += usize::from(check(&f.display().to_string(), &src));
    }
    // most conformance files have no syntax error (negative tests of later stages included)
    assert!(formatted >= 250 && formatted < files.len(), "{formatted} of {} files formatted", files.len());
}

#[test]
fn std_and_runtime_files_are_stable() {
    for dir in ["std", "runtime"] {
        let files = stave_files(&repo().join(dir));
        assert!(files.len() >= 8, "{dir}: {} files", files.len());
        for f in files {
            let src = std::fs::read(&f).unwrap();
            assert!(check(&f.display().to_string(), &src), "{}: has syntax errors", f.display());
        }
    }
}

#[test]
fn lang_md_blocks_are_stable() {
    for (i, b) in lang_md_blocks().iter().enumerate() {
        assert!(check(&format!("LANG.md block {i}"), b.as_bytes()), "LANG.md block {i}: has syntax errors");
    }
}

/// Every source without a syntax error: conformance files, `std`, `runtime`, LANG.md blocks.
fn corpus() -> &'static [Vec<u8>] {
    static CORPUS: std::sync::OnceLock<Vec<Vec<u8>>> = std::sync::OnceLock::new();
    CORPUS.get_or_init(|| {
        let mut all: Vec<Vec<u8>> = lang_md_blocks().into_iter().map(String::into_bytes).collect();
        for dir in ["tests/conformance", "std", "runtime"] {
            all.extend(stave_files(&repo().join(dir)).iter().map(|f| std::fs::read(f).unwrap()));
        }
        all.retain(|src| parse_file(src, F).diagnostics.is_empty());
        all
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]
    /// Whatever the whitespace and comments between the tokens: the layout is a fixed point and
    /// keeps the tokens, the AST, and the comments.
    #[test]
    fn any_layout_formats_to_a_fixed_point(file in any::<usize>(), seed in any::<u64>()) {
        let all = corpus();
        let src = relayout(&all[file % all.len()], seed);
        prop_assert!(check("relayout", src.as_bytes()), "{}", src);
    }
}
