//! The IR text grammar (`contracts/ir-text.ebnf`) under the `IR_TEXT` profile: it builds with
//! no conflict and no override, `AnyWord` acts only where the contract says, the checked-in
//! Rust tables equal a fresh build, and a table-driven parse over them (the token classification
//! of the `IR_TEXT` doc comment) accepts a line of every kind and rejects malformed lines.

use ll1gen::{Config, IR_TEXT, Sym, Tables, build_tables};
use stave_ir::generated::ir_text_ll1_table as lt;
use stave_ir::generated::ir_text_node_kinds::{KIND_NAME, NKIND};
use stave_ir::generated::ir_text_terminals::{NTERM, TERM_IS_CLASS, TERM_IS_WORD, TERM_TEXT, Term};
use std::path::{Path, PathBuf};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn ir_src() -> String {
    std::fs::read_to_string(repo().join(IR_TEXT.grammar)).expect("read contracts/ir-text.ebnf")
}

fn ir() -> Tables {
    match build_tables(&ir_src(), &IR_TEXT.config) {
        Ok(t) => t,
        Err(e) => panic!("{}: {e:?}", IR_TEXT.grammar),
    }
}

#[test]
fn builds_with_no_conflict_and_no_override() {
    let t = ir();
    assert!(IR_TEXT.config.overrides.is_empty() && IR_TEXT.config.required_overrides.is_empty());
    assert!(t.overrides.is_empty());
    assert!(t.split.is_empty());
    assert_eq!(t.nts[0].name, "Line");
    assert!(t.nts.iter().all(|n| !n.pass));
    assert_eq!(t.table.len(), t.nts.len() * t.nterm());
    for &p in &t.table {
        assert!(p == -1 || (0..t.prods.len() as i16).contains(&p));
    }
}

/// The same grammar under a profile that lists an override fails as stale: nothing to resolve.
#[test]
fn an_override_on_the_ir_grammar_is_stale() {
    let ov = [ll1gen::Override { rule: "Key", token: "int", why: "none" }];
    let cfg = Config { overrides: &ov, ..Config::ir_text() };
    let Err(errs) = build_tables(&ir_src(), &cfg) else { panic!("expected an error") };
    assert!(errs.iter().any(|e| e.msg.contains("stale override (Key, \"int\")")), "{errs:?}");
}

#[test]
fn any_word_acts_only_in_the_path_row() {
    let t = ir();
    let any = t.any_word.expect("AnyWord is a terminal");
    assert_eq!(t.terms[any as usize], "AnyWord");
    let mut rows = Vec::new();
    for (nt, n) in t.nts.iter().enumerate() {
        if t.action(nt as u16, any) >= 0 {
            rows.push(t.nts[n.origin as usize].name.as_str());
        }
    }
    assert_eq!(rows, ["Path"]);
    let path = t.nt_id("Path").unwrap();
    let string = t.term_id("String").unwrap();
    let a = t.action(path, any);
    assert!(a >= 0 && t.action(path, string) >= 0 && a != t.action(path, string));
    // Every Word-shaped terminal holds the AnyWord action in that row, and only those do.
    for (i, text) in t.terms.iter().enumerate() {
        let word = t.term_is_word[i];
        let by_shape = !t.term_is_class[i] && ll1gen::is_word_shaped(text);
        let word_class = ["AnyWord", "Intr", "Blk", "TId", "SId", "DId", "Bytes"].contains(&text.as_str());
        assert_eq!(word, by_shape || word_class, "{text:?}");
        let cell = t.action(path, i as u16);
        assert_eq!(cell == a, word, "Path on {text:?}");
    }
    for class in ["Val", "FnRef", "Int", "Hex", "String", "EOF"] {
        assert!(!t.term_is_word[t.term_id(class).unwrap() as usize], "{class}");
    }
    for kw in ["add", "slice.len", "stave.ir", "_", "bf16", "vec.add.wrap"] {
        assert!(t.term_is_word[t.term_id(kw).unwrap() as usize], "{kw}");
    }
    for p in ["(", "->", "<=", "|", "{", "}"] {
        assert!(!t.term_is_word[t.term_id(p).unwrap() as usize], "{p}");
    }
}

/// A keyword alternative next to `AnyWord` in one row is a conflict, not a silent priority.
#[test]
fn a_keyword_next_to_any_word_is_a_conflict() {
    let src = ir_src().replace("= AnyWord | String ;", "= AnyWord | String | \"add\" ;");
    let Err(errs) = build_tables(&src, &IR_TEXT.config) else { panic!("expected a conflict") };
    let all = errs.iter().map(ToString::to_string).collect::<Vec<_>>().join("\n");
    assert!(all.contains("(Path, \"add\")") && all.contains("Path -> \"add\""), "{all}");
}

#[test]
fn the_start_rule_must_come_first() {
    let src =
        ir_src().replace("\nLine         = HeaderLine", "\nEntry        = Line ;\nLine         = HeaderLine");
    let Err(errs) = build_tables(&src, &IR_TEXT.config) else { panic!("expected an error") };
    assert!(errs[0].msg.contains("start rule `Line` must be the first rule"), "{errs:?}");
}

#[test]
fn checked_in_rust_tables_equal_a_fresh_build() {
    let t = ir();
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
        assert!(!lt::NT_PASS[i]);
    }
    for (i, k) in KIND_NAME.iter().enumerate() {
        assert_eq!(lt::NT_NAME[i], *k);
    }
    for (i, x) in Term::ALL.iter().enumerate() {
        assert_eq!((x.id() as usize, x.text()), (i, t.terms[i].as_str()));
        assert_eq!(TERM_IS_CLASS[i], t.term_is_class[i]);
        assert_eq!(TERM_IS_WORD[i], t.term_is_word[i]);
        assert_eq!(TERM_TEXT[i], t.terms[i]);
    }
    assert_eq!(Some(Term::AnyWord.id()), t.any_word);
    assert_eq!(lt::START, 0);
}

/// The IR terminal ids are `u16`: the profile has more terminals than a `u8` kind column holds.
#[test]
fn terminal_ids_are_u16_and_exceed_the_u8_limit() {
    let (t, out) = ll1gen::generate(&ir_src(), IR_TEXT.grammar, &IR_TEXT).expect("generate ir-text");
    assert!(t.nterm() > ll1gen::LANGUAGE.terminal_ids.max_terminals(), "{}", t.nterm());
    let terminals = out.iter().find(|o| o.path.ends_with("ir_text_terminals.stv")).expect("terminals file");
    let ids: Vec<&str> = terminals.content.lines().filter(|l| l.starts_with("pub const IR_T_")).collect();
    assert_eq!(ids.len(), t.nterm());
    assert!(ids.iter().all(|l| l.contains(": u16 = ")), "{ids:?}");
}

/// The IR tables share nothing with the language tables: every Stave name carries the prefix,
/// the file stems carry theirs, and the output directories do not overlap in file names.
#[test]
fn stave_names_are_prefixed_and_files_do_not_collide() {
    let (_, ir_out) = ll1gen::generate(&ir_src(), IR_TEXT.grammar, &IR_TEXT).expect("generate ir-text");
    let lang_src = std::fs::read_to_string(repo().join(ll1gen::LANGUAGE.grammar)).unwrap();
    let (_, lang_out) =
        ll1gen::generate(&lang_src, ll1gen::LANGUAGE.grammar, &ll1gen::LANGUAGE).expect("generate language");
    let paths: Vec<&str> = ir_out.iter().chain(&lang_out).map(|o| o.path.as_str()).collect();
    let mut sorted = paths.clone();
    sorted.sort_unstable();
    sorted.dedup();
    // `mod.rs` is per directory; the two Rust directories differ.
    assert_eq!(sorted.len(), paths.len(), "{paths:?}");
    let mut stave_names: Vec<(String, &str)> = Vec::new();
    for o in ir_out.iter().chain(&lang_out).filter(|o| o.path.ends_with(".stv")) {
        for line in o.content.lines() {
            if let Some(rest) = line.strip_prefix("pub const ") {
                let name = rest.split(':').next().unwrap().to_string();
                if o.path.contains("ir_text_") {
                    assert!(name.starts_with("IR_"), "{}: {name}", o.path);
                } else {
                    assert!(!name.starts_with("IR_"), "{}: {name}", o.path);
                }
                stave_names.push((name, &o.path));
            }
        }
    }
    let mut names: Vec<&str> = stave_names.iter().map(|(n, _)| n.as_str()).collect();
    names.sort_unstable();
    let total = names.len();
    names.dedup();
    assert_eq!(names.len(), total, "a top-level Stave name is defined twice");
}

// ---- the token classification (doc comment of `ll1gen::IR_TEXT`) and a table-driven parse ----

fn num_end(b: &[u8], mut i: usize) -> Result<usize, String> {
    let s = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    if i == s {
        return Err(format!("digits expected at {s}"));
    }
    if b[s] == b'0' && i - s > 1 {
        return Err(format!("leading zero at {s}"));
    }
    Ok(i)
}

fn is_num(w: &str) -> bool {
    !w.is_empty() && w.bytes().all(|c| c.is_ascii_digit()) && (w == "0" || !w.starts_with('0'))
}

fn is_lower_hex(b: u8) -> bool {
    b.is_ascii_digit() || (b'a'..=b'f').contains(&b)
}

/// One line to terminal ids, without the final `EOF`.
fn lex(t: &Tables, line: &str) -> Result<Vec<u16>, String> {
    let class = |n: &str| t.term_id(n).unwrap_or_else(|| panic!("no class {n}"));
    let b = line.as_bytes();
    let (mut i, mut out) = (0, Vec::new());
    while i < b.len() {
        let c = b[i];
        let s = i;
        match c {
            b' ' | b'\t' => i += 1,
            b'\r' => return Err("carriage return".into()),
            b'#' => break,
            b'"' => {
                i += 1;
                loop {
                    match b.get(i) {
                        None => return Err("unterminated string".into()),
                        Some(b'"') => break,
                        Some(b'\\') => {
                            match b.get(i + 1) {
                                Some(b'\\' | b'"' | b'n' | b't') => i += 2,
                                Some(b'x')
                                    if b.get(i + 2..i + 4)
                                        .is_some_and(|h| h.iter().all(|&d| is_lower_hex(d))) =>
                                {
                                    i += 4;
                                }
                                _ => return Err(format!("bad escape at {i}")),
                            };
                        }
                        Some(0x20..=0x7e) => i += 1,
                        Some(x) => return Err(format!("byte {x:#x} in string")),
                    }
                }
                i += 1;
                out.push(class("String"));
            }
            b'%' => {
                i = num_end(b, i + 1)?;
                out.push(class("Val"));
            }
            b'@' => {
                i = num_end(b, i + 1)?;
                while b.get(i) == Some(&b'.') && b.get(i + 1).is_some_and(u8::is_ascii_digit) {
                    i = num_end(b, i + 1)?;
                }
                if b[i..].starts_with(b".u") {
                    i += 2;
                }
                out.push(class("FnRef"));
            }
            b'-' if b.get(i + 1) == Some(&b'>') => {
                i += 2;
                out.push(t.term_id("->").unwrap());
            }
            b'-' | b'0'..=b'9' => {
                if c == b'0' && b.get(i + 1) == Some(&b'x') {
                    let mut j = i + 2;
                    while b.get(j).is_some_and(|&d| is_lower_hex(d)) {
                        j += 1;
                    }
                    if ![4, 8, 16].contains(&(j - i - 2)) {
                        return Err(format!("hex constant at {i} has {} digits", j - i - 2));
                    }
                    i = j;
                    out.push(class("Hex"));
                } else {
                    let d = if c == b'-' { i + 1 } else { i };
                    if c == b'-' && b.get(d) == Some(&b'0') && !b.get(d + 1).is_some_and(u8::is_ascii_digit) {
                        return Err("-0".into());
                    }
                    i = num_end(b, d)?;
                    out.push(class("Int"));
                }
            }
            b'<' if b.get(i + 1) == Some(&b'=') => {
                i += 2;
                out.push(t.term_id("<=").unwrap());
            }
            b'(' | b')' | b'<' | b'>' | b',' | b':' | b'=' | b'!' | b'|' | b'{' | b'}' => {
                i += 1;
                out.push(t.term_id(&line[s..i]).unwrap());
            }
            b'A'..=b'Z' | b'a'..=b'z' | b'_' => {
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b'.') {
                    i += 1;
                }
                let w = &line[s..i];
                let shaped = |p: char| w.strip_prefix(p).is_some_and(is_num);
                let id = if w.strip_prefix("intrinsic.").is_some_and(|r| {
                    !r.is_empty()
                        && r.bytes().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'_')
                }) {
                    class("Intr")
                } else if shaped('b') {
                    class("Blk")
                } else if shaped('t') {
                    class("TId")
                } else if shaped('s') {
                    class("SId")
                } else if shaped('d') {
                    class("DId")
                } else if w.strip_prefix('x').is_some_and(|h| {
                    (2..=128).contains(&h.len()) && h.len() % 2 == 0 && h.bytes().all(is_lower_hex)
                }) {
                    class("Bytes")
                } else {
                    match t.term_id(w) {
                        Some(k) if !t.term_is_class[k as usize] => k,
                        _ => t.any_word.unwrap(),
                    }
                };
                out.push(id);
            }
            _ => return Err(format!("unexpected byte {c:#x} at {i}")),
        }
    }
    Ok(out)
}

/// Table-driven PDA over a fresh build; returns the number of expansions.
fn parse(t: &Tables, line: &str) -> Result<usize, String> {
    let mut toks = lex(t, line)?;
    toks.push(t.term_id("EOF").unwrap());
    let mut stack = vec![Sym::N(0)];
    let (mut i, mut expansions) = (0, 0);
    let name = |x: u16| t.terms[x as usize].as_str();
    while let Some(top) = stack.pop() {
        let tok = toks[i];
        match top {
            Sym::T(x) => {
                let any = t.any_word == Some(x) && t.term_is_word[tok as usize];
                if x != tok && !any {
                    return Err(format!("token {i}: expected {:?}, got {:?}", name(x), name(tok)));
                }
                i += 1;
            }
            Sym::N(n) => {
                let p = t.action(n, tok);
                if p < 0 {
                    return Err(format!(
                        "token {i}: {} cannot start with {:?}",
                        t.nts[n as usize].name,
                        name(tok)
                    ));
                }
                expansions += 1;
                stack.extend(t.prods[p as usize].rhs.iter().rev());
            }
        }
    }
    if i != toks.len() - 1 {
        return Err(format!("trailing tokens at {i}"));
    }
    Ok(expansions)
}

/// One accepted line per line kind (and per alternative of the larger ones), with the rule
/// whose first token decides it.
const VALID: &[(&str, &str)] = &[
    ("HeaderLine", "stave.ir 4 \"x86_64-linux-v2\" release checked"),
    ("HeaderLine", "stave.ir 4 \"aarch64-linux+sve\" debug lowered"),
    ("DeclLine", "decl d0 struct main.Point"),
    ("DeclLine", "decl d1 enum slice.len"),
    ("DeclLine", "decl d2 struct \"my-pkg.T\""),
    ("TypeLine", "type t0 int s 32"),
    ("TypeLine", "type t1 int u size"),
    ("TypeLine", "type t2 float b16"),
    ("TypeLine", "type t3 float 64"),
    ("TypeLine", "type t4 bool"),
    ("TypeLine", "type t5 char"),
    ("TypeLine", "type t6 str"),
    ("TypeLine", "type t7 ref mut t0"),
    ("TypeLine", "type t8 ref t0"),
    ("TypeLine", "type t9 ptr const t0"),
    ("TypeLine", "type t10 array t0 16"),
    ("TypeLine", "type t11 slice t0 align 64"),
    ("TypeLine", "type t12 slice t0"),
    ("TypeLine", "type t13 tuple t0 t1"),
    ("TypeLine", "type t14 tuple"),
    ("TypeLine", "type t15 adt d0 t0 3"),
    ("TypeLine", "type t16 fntype t0 t1 -> t2 avx2 bf16"),
    ("TypeLine", "type t17 fntype -> t0"),
    ("TypeLine", "type t18 vec t3 4"),
    ("TypeLine", "type t19 vec t3 n"),
    ("TypeLine", "type t20 mask 32 8"),
    ("ShapeLine", "shape t15 struct t0 t1"),
    ("ShapeLine", "shape t21 enum | t0 | | t1 t2"),
    ("StrLine", "str s0 \"hi\\n\\x00\\\"\""),
    ("SiteLine", "site 0 \"src/main.stv\" 12 5 main.f"),
    ("StaticLine", "static 0 t0 main.COUNTER"),
    ("DataLine", "data 0 16 x00ff10"),
    ("RefLine", "ref 0 8 s0 0"),
    ("RefLine", "ref 0 8 static 1 -4"),
    ("MainLine", "main @0"),
    ("CellLine", "cell d1"),
    ("RunStackLine", "runstack 4096"),
    ("TruncatedLine", "truncated"),
    ("NodeLine", "node 0 rec"),
    ("NodeLine", "node 1"),
    ("EdgeLine", "edge 0 1 call"),
    ("EdgeLine", "edge 1 0 indirect"),
    ("EdgeLine", "edge 1 2 spawn"),
    ("NodesLine", "nodes @0 0 1 2"),
    ("FnLine", "fn @0 main.main () {"),
    (
        "FnLine",
        "fn @1.2.u std.add<t0, 3>(t0, t1) -> t2 needs (avx2, fma) symbol \"f\" variadic known altstack unbuilt site 4 {",
    ),
    ("FnLine", "fn @3 libc.puts (t0)"),
    ("FnLine", "fn @4 \"odd-pkg.f\" () -> t1 {"),
    ("ClaimLine", "claim @0 pre checked p 0 <= lenp 1 -1"),
    ("ClaimLine", "claim @0 post result <= zero 0"),
    ("BlockLine", "b0():"),
    ("BlockLine", "b1(%0: t0, %1: t2):"),
    ("ValueInst", "%2 = add t0 %0 %1"),
    ("ValueInst", "%3 = neg t0 7 !5"),
    ("ValueInst", "%4 = select t0 %0 0x3f800000 -3"),
    ("ValueInst", "%5 = make t3"),
    ("ValueInst", "%6 = enum.make t0 1 %0 %1"),
    ("ValueInst", "%7 = enum.get t0 %0 1 2"),
    ("ValueInst", "%8 = addr.field t0 %0 3"),
    ("ValueInst", "%9 = fn.make t0 @2 %0"),
    ("ValueInst", "%10 = call t0 @1.u %0 s1 bf16 0x3f80"),
    ("ValueInst", "%11 = call.ind t0 %0 %1"),
    ("ValueInst", "%12 = intrinsic.write t0 %0 %1"),
    ("ValueInst", "%13 = vec.add.wrap t5 %0 %1"),
    ("ValueInst", "%14 = vec.permute t5 %0 3 2 1 0"),
    ("ValueInst", "%15 = mask.all t5"),
    ("EffectInst", "store () %0 %1 !2"),
    ("EffectInst", "mem.fence_stream ()"),
    ("EffectInst", "call () @1 t4 %0"),
    ("EffectInst", "intrinsic.exit () 0"),
    ("EffectInst", "vec.store_masked () %0 %1 %2 %3"),
    ("TermLine", "switch %0 1: b1() 2: b2(%0, 5) _: b3()"),
];

const VALID_TERMS: &[&str] = &["br b1(%0, %1)", "cbr %0 b1() b2(%1)", "ret", "ret %0", "unreachable", "}"];

/// (what is wrong, the line, a part of the error).
const INVALID: &[(&str, &str, &str)] = &[
    ("operand where the result type belongs", "%1 = add %0 %0 %0", "expected \"TId\", got \"Val\""),
    ("a line kind that is no keyword", "nope 1 2", "Line cannot start with \"AnyWord\""),
    ("a bare Name where a TId belongs", "type t0 ref mut foo", "expected \"TId\", got \"AnyWord\""),
    ("missing closing parenthesis", "b0(%0: t0:", "cannot start with \":\""),
    ("an Int where a path belongs", "decl d0 struct 5", "Path cannot start with \"Int\""),
    ("trailing token", "main @0 @1", "trailing tokens at 2"),
];

#[test]
fn parse_accepts_a_line_of_every_kind() {
    let t = ir();
    let mut kinds: Vec<&str> = VALID.iter().map(|(k, _)| *k).collect();
    for (what, line) in VALID.iter().copied().chain(VALID_TERMS.iter().map(|l| ("TermLine", *l))) {
        if let Err(e) = parse(&t, line) {
            panic!("{what}: {line:?}: {e}");
        }
    }
    kinds.extend(["TermLine", "TermLine"]);
    kinds.sort_unstable();
    kinds.dedup();
    // Every alternative of `Line` is exercised: the rule behind each is among the kinds above,
    // or reached through `ValueInst`/`EffectInst`/`TermLine`.
    let line = t.nt_id("Line").unwrap();
    let mut expected: Vec<String> = Vec::new();
    for pr in t.prods.iter().filter(|p| p.lhs == line) {
        expected.push(match pr.rhs.as_slice() {
            [Sym::N(n)] => t.nts[*n as usize].name.clone(),
            [Sym::T(x)] => format!("{:?}", t.terms[*x as usize]),
            other => panic!("unexpected shape of a Line alternative: {other:?}"),
        });
    }
    expected.retain(|k| k != "\"}\"");
    let mut got: Vec<String> = kinds.iter().map(|k| (*k).to_string()).collect();
    got.sort_unstable();
    expected.sort_unstable();
    assert_eq!(got, expected, "VALID must hold one line per alternative of Line");
}

#[test]
fn parse_rejects_malformed_lines() {
    let t = ir();
    for (what, line, part) in INVALID {
        match parse(&t, line) {
            Ok(_) => panic!("accepted ({what}): {line}"),
            Err(e) => assert!(e.contains(part), "({what}): {line}: {e}"),
        }
    }
}

/// The lexer and the PDA also reject on lexical grounds the table cannot see.
#[test]
fn lexical_errors_are_not_tokens() {
    let t = ir();
    for line in
        ["main @", "ref 0 8 s0 -0", "static 0 t0 \"unterminated", "data 0 1 0x123", "cell d01", "main @0\r"]
    {
        assert!(parse(&t, line).is_err(), "accepted: {line}");
    }
}

/// Word shape comes before keyword: `b16` is a block id, never a keyword; a name equal to a
/// class name is a plain Name.
#[test]
fn classification_order() {
    let t = ir();
    let one = |s: &str| lex(&t, s).unwrap();
    let id = |n: &str| t.term_id(n).unwrap();
    assert_eq!(one("b16"), [id("Blk")]);
    assert_eq!(one("Val"), [t.any_word.unwrap()]);
    assert_eq!(one("x00ff"), [id("Bytes")]);
    assert_eq!(one("x0"), [t.any_word.unwrap()]);
    assert_eq!(one("t05"), [t.any_word.unwrap()]);
    assert_eq!(one("intrinsic.x86_cpuid"), [id("Intr")]);
    assert_eq!(one("intrinsic."), [t.any_word.unwrap()]);
    assert_eq!(one("add.wrap"), [id("add.wrap")]);
    assert_eq!(one("@1.2.u"), [id("FnRef")]);
    assert_eq!(one("-> <= ! 0x00ff"), [id("->"), id("<="), id("!"), id("Hex")]);
    assert_eq!(one("add # a comment"), [id("add")]);
}

/// `gen` (no `--profile`) runs both grammars: a conflict in the IR grammar fails it, names the
/// rule and the token, and still processes the other grammar.
#[test]
fn a_conflict_in_the_ir_grammar_fails_the_run() {
    let tmp = Path::new(env!("CARGO_TARGET_TMPDIR")).join("ll1gen-ir-conflict");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(tmp.join("contracts")).unwrap();
    std::fs::copy(repo().join(ll1gen::LANGUAGE.grammar), tmp.join(ll1gen::LANGUAGE.grammar)).unwrap();
    let broken = ir_src().replace("= AnyWord | String ;", "= AnyWord | String | \"add\" ;");
    std::fs::write(tmp.join(IR_TEXT.grammar), broken).unwrap();
    let run = |check: bool| {
        let mut c = std::process::Command::new(env!("CARGO_BIN_EXE_ll1gen"));
        c.arg("--root").arg(&tmp);
        if check {
            c.arg("--check");
        }
        c.output().expect("run ll1gen")
    };
    for check in [false, true] {
        let out = run(check);
        assert_eq!(out.status.code(), Some(1), "check={check}");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains("ir-text:") && err.contains("(Path, \"add\")"), "{err}");
        assert!(err.contains("Path -> \"add\""), "{err}");
    }
    assert!(!tmp.join(IR_TEXT.rust_dir).exists(), "nothing written for the failing grammar");
    assert!(tmp.join(ll1gen::LANGUAGE.rust_dir).join("ll1_table.rs").is_file());
}
