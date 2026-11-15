//! Lexer tests: single tokens, numbers, f-strings, literal errors,
//! file-level errors, token ends.

mod common;

use common::{F, kinds, lex_diags, toks};
use stave_syntax::{KIND_ERROR, LexStop, Term, lex, token_end};

#[test]
fn single_tokens_fr002a() {
    let ops = [
        "as%", "as|", "as?", "as~", "...", "..", "..=", ".{", "+%", "-%", "*%", "<<%", "+|", "-|", "*|",
        "+%=", "-%=", "*%=", "+|=", "-|=", "*|=", "<<%=", "->", "=>", "::", "<<=", ">>=", "&&", "||",
    ];
    for op in ops {
        assert_eq!(kinds(op), vec![op], "{op}");
        assert_eq!(kinds(&format!("x {op} y")), vec!["Ident", op, "Ident"], "{op}");
    }
    assert_eq!(kinds("x as% u8"), vec!["Ident", "as%", "Ident"]);
    assert_eq!(kinds("x as % u8"), vec!["Ident", "as", "%", "Ident"]);
    assert_eq!(kinds("x as |u8"), vec!["Ident", "as", "|", "Ident"]);
    assert_eq!(kinds("has%"), vec!["Ident", "%"]);
    assert_eq!(kinds("as%="), vec!["as%", "="]);
    assert_eq!(kinds("...."), vec!["...", "."]);
    assert_eq!(kinds("<<%=="), vec!["<<%=", "="]);
    assert_eq!(kinds("S.{a"), vec!["Ident", ".{", "Ident"]);
    assert_eq!(kinds("S. {a"), vec!["Ident", ".", "{", "Ident"]);
}

#[test]
fn numbers_and_dots() {
    assert_eq!(toks("t.0.1"), vec!["Ident:t", ".:.", "IntLit:0", ".:.", "IntLit:1"]);
    assert_eq!(toks("t.0e5"), vec!["Ident:t", ".:.", "IntLit:0e5"]);
    assert_eq!(toks("0..n"), vec!["IntLit:0", "..:..", "Ident:n"]);
    assert_eq!(toks("0..=9"), vec!["IntLit:0", "..=:..=", "IntLit:9"]);
    assert_eq!(toks("1."), vec!["IntLit:1", ".:."]);
    assert_eq!(toks("1.x"), vec!["IntLit:1", ".:.", "Ident:x"]);
    assert_eq!(toks("1.0"), vec!["FloatLit:1.0"]);
    assert_eq!(toks("1.0f32"), vec!["FloatLit:1.0f32"]);
    assert_eq!(toks("1e-6"), vec!["FloatLit:1e-6"]);
    assert_eq!(toks("1.5e+7"), vec!["FloatLit:1.5e+7"]);
    assert_eq!(toks("1e+x"), vec!["IntLit:1e", "+:+", "Ident:x"]);
    assert_eq!(toks("1_000u32"), vec!["IntLit:1_000u32"]);
    assert_eq!(toks("0xff_u8 0b1010 0o17"), vec!["IntLit:0xff_u8", "IntLit:0b1010", "IntLit:0o17"]);
    assert_eq!(toks("1.5.sqrt()"), vec!["FloatLit:1.5", ".:.", "Ident:sqrt", "(:(", "):)"]);
    assert_eq!(toks("x. 1.5"), vec!["Ident:x", ".:.", "FloatLit:1.5"]);
    assert_eq!(toks("-128i8"), vec!["-:-", "IntLit:128i8"]);
}

#[test]
fn identifiers_keywords_underscore() {
    assert_eq!(kinds("_ _x __ x_ _0"), vec!["_", "Ident", "Ident", "Ident", "Ident"]);
    assert_eq!(kinds("fn Self self selfish"), vec!["fn", "Self", "self", "Ident"]);
    assert_eq!(kinds("kernel allocator emulate"), vec!["kernel", "allocator", "emulate"]);
    // identifiers are ASCII only: a non-ASCII byte starts no token (E0018)
    assert_eq!(kinds("a_Z9 Zz_0"), vec!["Ident", "Ident"]);
    assert_eq!(kinds("é"), vec!["ERROR"]);
    assert_eq!(kinds("naïve"), vec!["Ident", "ERROR", "Ident"]);
}

#[test]
fn comments_and_whitespace() {
    assert_eq!(kinds(""), Vec::<&str>::new());
    assert_eq!(kinds("// only a comment"), Vec::<&str>::new());
    assert_eq!(kinds("a // c\nb"), vec!["Ident", "Ident"]);
    assert_eq!(kinds("a /* b */"), vec!["Ident", "ERROR"]);
    assert_eq!(kinds("a / * b * /"), vec!["Ident", "/", "*", "Ident", "*", "/"]);
    assert_eq!(kinds("\t\r\n x"), vec!["Ident"]);
    let l = lex(b"", F);
    assert_eq!(l.tokens.len(), 1);
    assert!(l.diagnostics.is_empty());
}

#[test]
fn literals() {
    assert_eq!(toks(r#""a\"b" 'c' '\n'"#), vec![r#"StrLit:"a\"b""#, "CharLit:'c'", r"CharLit:'\n'"]);
    assert_eq!(toks(r#"f"n = {n} {self.x} {p.x.y}\n""#), vec![r#"FmtLit:f"n = {n} {self.x} {p.x.y}\n""#]);
    assert_eq!(kinds(r#"f"{self}" f"" f"}" if"x""#), vec!["FmtLit", "FmtLit", "FmtLit", "if", "StrLit"]);
    assert_eq!(kinds(r#"xf"s""#), vec!["Ident", "StrLit"]);
}

#[test]
fn fstring_holes_e0006() {
    for bad in ["{t.0}", "{a+b}", "{ x }", "{}", "{fn}", "{x.self}", "{_}", "{Self}", "{a.}", "{.a}", "{a b}"]
    {
        let src = format!("f\"v {bad} w\"");
        let l = lex(src.as_bytes(), F);
        assert_eq!(l.tokens[0].kind, KIND_ERROR, "{bad}");
        assert_eq!(l.tokens.len(), 2, "{bad}: one token");
        let d: Vec<_> = l.diagnostics.iter().map(|d| d.code()).collect();
        assert_eq!(d, vec!["E0006"], "{bad}");
        let s = l.diagnostics[0].primary().unwrap();
        assert_eq!(&src[s.offset as usize..(s.offset + s.len) as usize], bad);
    }
    let d = &lex(br#"f"{t.0}""#, F).diagnostics[0];
    assert_eq!(d.message(), "f-string hole `{t.0}` is not a name or field path");
    // every bad hole is reported; good ones are not
    assert_eq!(lex_diags(br#"f"{a} {t.0} {b} {1}""#), vec![("E0006", 6), ("E0006", 16)]);
    // an unclosed hole ends at the quote
    assert_eq!(lex_diags(br#"f"{a+" x"#), vec![("E0006", 2)]);
}

#[test]
fn unterminated_e0003_vs_newline_e0015() {
    for (src, what) in [("\"abc", "string"), ("'+", "char"), ("f\"abc", "f-string"), ("x = \"a\\", "string")]
    {
        let l = lex(src.as_bytes(), F);
        assert_eq!(l.diagnostics.len(), 1, "{src}");
        let d = &l.diagnostics[0];
        assert_eq!(d.code(), "E0003", "{src}");
        assert_eq!(d.message(), format!("unterminated {what} literal"));
        assert_eq!(l.tokens[l.tokens.len() - 2].kind, KIND_ERROR);
    }
    for (src, what) in
        [("\"abc\nx", "string"), ("'+\nx", "char"), ("f\"a{b}\nx", "f-string"), ("\"a\\\nx", "string")]
    {
        let l = lex(src.as_bytes(), F);
        assert_eq!(l.diagnostics.len(), 1, "{src:?}");
        let d = &l.diagnostics[0];
        assert_eq!(d.code(), "E0015", "{src:?}");
        assert_eq!(d.message(), format!("raw newline inside a {what} literal"));
        let s = d.primary().unwrap();
        assert_eq!(s.offset, 0);
        assert_eq!(&src[(s.offset + s.len) as usize..], "\nx", "ends at the newline");
        // lexing resumes on the next line
        assert_eq!(
            l.tokens.iter().map(|t| t.kind).collect::<Vec<_>>(),
            vec![KIND_ERROR, Term::Ident as u8, Term::Eof as u8]
        );
    }
    // f-string with a bad hole and a newline: both reported
    assert_eq!(lex_diags(b"f\"{a+b}\n"), vec![("E0015", 0), ("E0006", 2)]);
}

#[test]
fn invalid_utf8_e0001_once_and_stop() {
    let src = b"fn a() {}\nfn b\xff() {}\n\xfe\"unterminated";
    let l = lex(src, F);
    assert_eq!(l.stop, Some(LexStop::InvalidUtf8 { offset: 14 }));
    assert_eq!(lex_diags(src), vec![("E0001", 14)]);
    // the tokens before the invalid byte only (`b` is ASCII and ends before it)
    let k: Vec<u8> = l.tokens.iter().map(|t| t.kind).collect();
    assert_eq!(k.len(), 9);
    assert_eq!(k[7], Term::Ident as u8);
    assert_eq!(*k.last().unwrap(), Term::Eof as u8);
    assert_eq!(l.tokens.last().unwrap().start, 14);
    // a lexical error after the invalid byte is not reported, one before it is
    let src2 = b"\"a\n\xc3x\"b";
    assert_eq!(lex_diags(src2), vec![("E0015", 0), ("E0001", 3)]);
    // invalid byte inside a comment
    let src3 = b"a // \xff\nb";
    assert_eq!(lex_diags(src3), vec![("E0001", 5)]);
    assert_eq!(lex(src3, F).tokens.len(), 2);
}

#[test]
fn bom_e0017_resumes() {
    let src = b"\xEF\xBB\xBFfn f() {}";
    let l = lex(src, F);
    assert_eq!(lex_diags(src), vec![("E0017", 0)]);
    assert_eq!(l.tokens[0].start, 3);
    assert_eq!(l.tokens.len(), 7);
    assert!(l.stop.is_none());
    // a BOM later in the file is not E0017: three non-ASCII bytes, one E0018
    assert_eq!(lex_diags("a\u{FEFF}b".as_bytes()), vec![("E0018", 1)]);
}

#[test]
fn unexpected_characters_e0018() {
    // each byte that starts no token, alone: one error token, one E0018 at it
    for bad in [b"@" as &[u8], b"$", b"\\", b"`", b"#", b"\x00", b"\x01", b"\x1f", b"\x7f", b"\x0b", b"\x0c"]
    {
        let mut src = b"a ".to_vec();
        src.extend_from_slice(bad);
        src.extend_from_slice(b" b");
        let l = lex(&src, F);
        let k: Vec<u8> = l.tokens.iter().map(|t| t.kind).collect();
        assert_eq!(k, vec![Term::Ident as u8, KIND_ERROR, Term::Ident as u8, Term::Eof as u8], "{bad:?}");
        assert_eq!(lex_diags(&src), vec![("E0018", 2)], "{bad:?}");
        let d = &l.diagnostics[0];
        assert_eq!(d.message(), "unexpected character");
        assert_eq!(d.primary().unwrap().len, 1, "{bad:?}");
    }
    // one E0018 per run of consecutive bad bytes, at its start; the run is one token
    let src = "x @$\\` y é ü2 €€ \x01@";
    let l = lex(src.as_bytes(), F);
    assert_eq!(
        lex_diags(src.as_bytes()),
        vec![("E0018", 2), ("E0018", 9), ("E0018", 12), ("E0018", 16), ("E0018", 23)]
    );
    let lens: Vec<u32> = l.diagnostics.iter().map(|d| d.primary().unwrap().len).collect();
    assert_eq!(lens, vec![4, 2, 2, 6, 2]);
    assert_eq!(kinds(src), vec!["Ident", "ERROR", "Ident", "ERROR", "ERROR", "IntLit", "ERROR", "ERROR"]);
    // non-ASCII is fine in literals and comments
    assert!(lex_diags("\"é€\" 'ü' f\"{x} ñ\" // ß\nx".as_bytes()).is_empty());
    // a run that reaches an invalid UTF-8 byte ends there; E0001 follows
    assert_eq!(lex_diags(b"a \xc3\xa9\xff b"), vec![("E0018", 2), ("E0001", 4)]);
    // an item with an E0018 token gets no E0004
    let pf = stave_syntax::parse_file(b"fn f() { let a = 1 @ 2; }\nfn g() { }\n", F);
    let codes: Vec<&str> = pf.diagnostics.iter().map(|d| d.code()).collect();
    assert_eq!(codes, vec!["E0018"]);
    assert_eq!(pf.items.iter().map(|i| i.ok).collect::<Vec<_>>(), vec![false, true]);
}

#[test]
fn lifetimes_e0024_and_block_comments_e0029() {
    // a `'` and an identifier with no `'` directly after it: one error token
    assert_eq!(toks("&'a str"), vec!["&:&", "ERROR:'a", "Ident:str"]);
    assert_eq!(toks("'static"), vec!["ERROR:'static"]);
    assert_eq!(
        lex_diags(b"fn f<'a>(x: &'a u8, c: char) { let c = 'a'; let d = '_'; }"),
        vec![("E0024", 5), ("E0024", 13)]
    );
    assert_eq!(kinds("'a' 'ab' '\\n' ' '"), vec!["CharLit", "CharLit", "CharLit", "CharLit"]);
    let l = lex(b"&'long x", F);
    assert_eq!(l.diagnostics[0].message(), "lifetime `'long` is not Stave: references have no lifetimes");
    // `/*`: an error token through the first `*/` on its line, else the end of the line
    assert_eq!(toks("a /* c */ b"), vec!["Ident:a", "ERROR:/* c */", "Ident:b"]);
    assert_eq!(toks("a /* c\nb */"), vec!["Ident:a", "ERROR:/* c", "Ident:b", "*:*", "/:/"]);
    assert_eq!(toks("a/*p"), vec!["Ident:a", "ERROR:/*p"]);
    assert_eq!(toks("a / *p"), vec!["Ident:a", "/:/", "*:*", "Ident:p"]);
    assert_eq!(toks("/*/ x */ y"), vec!["ERROR:/*/ x */", "Ident:y"]);
    assert_eq!(lex_diags(b"x /* a */ /* b"), vec![("E0029", 2), ("E0029", 10)]);
    assert_eq!(kinds("// /* in a line comment"), Vec::<&str>::new());
    assert_eq!(kinds("\"/* in a string */\""), vec!["StrLit"]);
}

#[test]
fn token_end_matches_lexing() {
    let src = r#"fn f(t: (i32, i32)) -> f32 { let a = t.0.1e2; let s = "x\"y"; let c = 'c';
        let f = f"{a.b}"; x as% u8 >>= 1.5e-3 ... .{ } "unterminated
        @ f"{bad hole}" 'z"#;
    let b = src.as_bytes();
    let l = lex(b, F);
    for i in 0..l.tokens.len() {
        let e = token_end(b, &l.tokens, i) as usize;
        let s = l.tokens[i].start as usize;
        assert!(e >= s && e <= b.len());
        if let Some(next) = l.tokens.get(i + 1) {
            assert!(e <= next.start as usize, "token {i} overlaps the next");
            assert!(b[e..next.start as usize].iter().all(|c| c.is_ascii_whitespace()), "gap after {i}");
        }
    }
}
