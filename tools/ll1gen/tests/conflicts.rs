//! The LL(1) conflict set of `contracts/grammar.ebnf`, unlisted and stale
//! overrides, the D3 split map, deterministic output, and grammar errors with line numbers.

use ll1gen::{Config, Error, LANGUAGE, OVERRIDES, Sym, Tables, build_tables, generate};
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn contract_src() -> String {
    std::fs::read_to_string(repo().join("contracts/grammar.ebnf")).expect("read contracts/grammar.ebnf")
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn fixture_src(name: &str) -> String {
    std::fs::read_to_string(fixture(name)).expect("read fixture")
}

fn contract() -> Tables {
    match build_tables(&contract_src(), &Config::contract()) {
        Ok(t) => t,
        Err(e) => panic!("contracts/grammar.ebnf: {e:?}"),
    }
}

fn errors(src: &str) -> Vec<Error> {
    match build_tables(src, &Config::contract()) {
        Ok(_) => panic!("expected an error"),
        Err(e) => e,
    }
}

fn joined(errs: &[Error]) -> String {
    errs.iter().map(ToString::to_string).collect::<Vec<_>>().join("\n")
}

/// The four overrides, each resolved to the first listed alternative.
fn check_overrides(t: &Tables) {
    let keys: Vec<(&str, &str)> = t.overrides.iter().map(|o| (o.rule.as_str(), o.token.as_str())).collect();
    assert_eq!(keys, [("Stmt", "unsafe"), ("TypeBase", "<"), ("TypeBase", "needs"), ("TypeBase", "without")]);
    for o in &t.overrides {
        assert!(o.alternatives.len() >= 2, "{o:?}");
        assert_eq!(o.prod, o.alternatives[0], "first listed alternative wins");
        assert_eq!(t.action(o.nt, o.term), o.prod as i16);
        assert_eq!(t.nts[o.origin as usize].name, o.rule);
        assert_eq!(Some(o.origin), t.nt_id(&o.rule));
        assert_eq!(Some(o.term), t.term_id(&o.token));
        let rhs = &t.prods[o.prod as usize].rhs;
        let first = match rhs.first() {
            Some(Sym::N(n)) => t.nts[*n as usize].name.as_str(),
            other => panic!("{o:?}: chosen rhs starts with {other:?}"),
        };
        match o.token.as_str() {
            "unsafe" => assert_eq!(first, "Unsafe", "Stmt -> Unsafe Block"),
            "<" => assert_eq!((first, rhs.len()), ("GenericArgs", 1)),
            "needs" => assert_eq!((first, rhs.len()), ("Needs", 1)),
            _ => assert_eq!((first, rhs.len()), ("Without", 1)),
        }
        // The losing alternative of the two TypeBase helpers is ε.
        if o.rule == "TypeBase" {
            assert!(t.prods[o.alternatives[1] as usize].rhs.is_empty());
        }
    }
}

#[test]
fn contract_grammar_has_exactly_the_four_overrides() {
    let t = contract();
    check_overrides(&t);
    assert_eq!(OVERRIDES.len(), 4);
}

#[test]
fn contract_tables_are_consistent() {
    let t = contract();
    let nterm = t.nterm();
    assert_eq!(t.nts[0].name, "File");
    assert_eq!(t.table.len(), t.nts.len() * nterm);
    let mut sorted = t.terms.clone();
    sorted.sort();
    assert_eq!(sorted, t.terms, "terminal ids in sorted order");
    for (i, n) in t.nts.iter().enumerate() {
        assert_eq!(n.named, i < t.nkind, "named rules first: {}", n.name);
        assert!((n.origin as usize) < t.nkind);
        if n.named {
            assert_eq!(n.origin as usize, i);
            assert!(!n.name.contains('#'));
        } else {
            assert!(n.name.starts_with(&format!("{}#", t.nts[n.origin as usize].name)), "{}", n.name);
        }
    }
    for r in ["Unsafe", "AndExpr", "CmpOp", "Stmt", "TypeBase", "GenericArgs"] {
        assert!(t.nt_id(r).is_some_and(|i| (i as usize) < t.nkind), "{r}");
    }
    for &p in &t.table {
        assert!(p == -1 || (0..t.prods.len() as i16).contains(&p));
    }
    for (i, p) in t.prods.iter().enumerate() {
        for s in &p.rhs {
            match *s {
                Sym::T(x) => assert!((x as usize) < nterm, "prod {i}"),
                Sym::N(n) => assert!((n as usize) < t.nts.len(), "prod {i}"),
            }
        }
    }
}

#[test]
fn split_map() {
    let t = contract();
    let s: Vec<(&str, &str, &str)> = t
        .split
        .iter()
        .map(|&(a, b, c)| {
            (t.terms[a as usize].as_str(), t.terms[b as usize].as_str(), t.terms[c as usize].as_str())
        })
        .collect();
    assert_eq!(s, [(">>", ">", ">"), (">=", ">", "="), (">>=", ">", ">="), ("&&", "&", "&")]);
}

#[test]
fn fixture_with_the_four_overrides_passes() {
    let t = build_tables(&fixture_src("base.ebnf"), &Config::contract()).expect("base fixture");
    check_overrides(&t);
}

#[test]
fn unlisted_conflict_fails_naming_rule_token_alternatives() {
    let errs = errors(&fixture_src("extra_conflict.ebnf"));
    assert_eq!(errs.len(), 1, "{}", joined(&errs));
    let e = &errs[0];
    assert!(e.msg.contains("not in OVERRIDES"), "{e}");
    assert!(e.msg.contains("(Unary, \"Ident\")"), "{e}");
    assert!(e.msg.contains("Unary -> \"Ident\" | Unary -> \"Ident\" \"!\""), "{e}");
    let line = fixture_src("extra_conflict.ebnf").lines().position(|l| l.starts_with("Unary")).unwrap() + 1;
    assert_eq!(e.line, Some(line), "points at the origin rule");
}

#[test]
fn listed_but_unused_override_is_stale() {
    let errs = errors(&fixture_src("stale_override.ebnf"));
    assert_eq!(errs.len(), 1, "{}", joined(&errs));
    assert!(errs[0].msg.contains("stale override (TypeBase, \"needs\")"), "{}", errs[0]);
}

#[test]
fn rules_named_by_parse_errors_contract_are_required() {
    let src = fixture_src("base.ebnf").replace("AndExpr", "Conj");
    let all = joined(&errors(&src));
    assert!(all.contains("rule `AndExpr`"), "{all}");
    let cfg = Config {
        overrides: &OVERRIDES[..],
        required_rules: &[],
        required_overrides: &[("TypeBase", "<")],
        pass_through: &[],
        ..Config::contract()
    };
    let no_d2 = Config { overrides: &OVERRIDES[..1], ..cfg };
    let Err(errs) = build_tables(&fixture_src("base.ebnf"), &no_d2) else { panic!("expected errors") };
    let all = joined(&errs);
    assert!(all.contains("(TypeBase, \"<\") named by contracts/parse-errors.md"), "{all}");
    assert!(all.contains("not in OVERRIDES: (TypeBase, \"<\")"), "{all}");
}

#[test]
fn pass_through_rules_must_exist_and_not_derive_epsilon() {
    let base = fixture_src("base.ebnf");
    let cfg = Config { pass_through: &["Cast", "AndExpr"], ..Config::contract() };
    let t = build_tables(&base, &cfg).expect("base");
    let pass: Vec<&str> = t.nts.iter().filter(|n| n.pass).map(|n| n.name.as_str()).collect();
    assert_eq!(pass, ["AndExpr", "Cast"]);
    let t = build_tables(&base, &Config::contract()).expect("base");
    assert_eq!(t.nts.iter().filter(|n| n.pass).count(), ll1gen::PASS_THROUGH.len());
    let cfg = Config { pass_through: &["Cast", "Nope"], ..Config::contract() };
    let Err(errs) = build_tables(&base, &cfg) else { panic!("expected errors") };
    assert_eq!(errs.len(), 1, "{}", joined(&errs));
    assert!(errs[0].msg.contains("pass-through rule `Nope` is missing"), "{}", errs[0]);
    let src = format!("{base}Opt = Ident? ;\n");
    let Err(errs) = build_tables(&src, &Config { pass_through: &["Opt"], ..Config::contract() }) else {
        panic!("expected errors")
    };
    assert_eq!(errs.len(), 1, "{}", joined(&errs));
    assert!(errs[0].msg.contains("pass-through rule `Opt` derives ε"), "{}", errs[0]);
}

#[test]
fn too_many_terminals_for_the_u8_kind_column_are_refused() {
    let mut t = contract();
    let mut i = 0;
    while t.terms.len() <= LANGUAGE.terminal_ids.max_terminals() {
        t.terms.push(format!("synthetic{i}"));
        i += 1;
    }
    let err = ll1gen::render(&t, LANGUAGE.grammar, &LANGUAGE).err().expect("too many terminals").to_string();
    assert!(err.contains("255 terminals, at most 254"), "{err}");
    // At the limit the count check passes (the synthetic terminals then fail naming instead).
    t.terms.pop();
    let err =
        ll1gen::render(&t, LANGUAGE.grammar, &LANGUAGE).err().map(|e| e.to_string()).unwrap_or_default();
    assert!(!err.contains("at most 254"), "{err}");
}

#[test]
fn output_is_byte_identical_across_runs() {
    let src = contract_src();
    let (_, a) = generate(&src, "contracts/grammar.ebnf", &LANGUAGE).expect("generate");
    let (_, b) = generate(&src, "contracts/grammar.ebnf", &LANGUAGE).expect("generate");
    assert_eq!(a.len(), 7);
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.path, y.path);
        assert!(x.content == y.content, "{} differs between runs", x.path);
        assert!(
            x.content.starts_with("// generated by tools/ll1gen from contracts/grammar.ebnf; do not edit\n")
        );
    }
}

#[test]
fn malformed_grammars_report_line_numbers() {
    let m = "(* ---------- items *)\n";
    let cases: [(&str, String, usize); 9] = [
        ("missing `;`", format!("{m}A = \"a\"\nB = \"b\" ;\n"), 3),
        ("unterminated literal", format!("{m}A = \"a ;\n"), 2),
        ("empty literal", format!("{m}\nA = \"\" ;\n"), 3),
        ("undefined nonterminal", format!("{m}A = B ;\n"), 2),
        ("unexpected character", format!("{m}A = \"a\" ~ ;\n"), 2),
        ("unclosed group", format!("{m}A = ( \"a\" ;\n"), 2),
        ("duplicate rule", format!("{m}A = \"a\" ;\nA = \"b\" ;\n"), 3),
        ("unterminated comment", format!("{m}A = \"a\" ;\n(* x\n"), 3),
        ("symbol expected", format!("{m}A = \"a\" | ? ;\n"), 2),
    ];
    for (what, src, line) in cases {
        let errs = errors(&src);
        assert_eq!(errs[0].line, Some(line), "{what}: {}", joined(&errs));
    }
    let deep = format!("{m}A = {}\"a\"{} ;\n", "(".repeat(5000), ")".repeat(5000));
    assert!(errors(&deep)[0].msg.contains("nested too deeply"));
    assert!(errors("A = \"a\" ;")[0].msg.contains("items marker"));
    assert!(errors(&format!("{m}Ident = \"a\" ;\n"))[0].msg.contains("lexical class"));
    assert!(!errors(m)[0].msg.is_empty());
}

fn ll1gen() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ll1gen"))
}

#[test]
fn cli_exits_1_on_unlisted_conflict() {
    let tmp = Path::new(env!("CARGO_TARGET_TMPDIR")).join("ll1gen-conflict");
    let out = ll1gen()
        .arg("--profile")
        .arg("language")
        .arg("--root")
        .arg(&tmp)
        .arg("--grammar")
        .arg(fixture("extra_conflict.ebnf"))
        .output()
        .expect("run ll1gen");
    assert_eq!(out.status.code(), Some(1));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("(Unary, \"Ident\")") && err.contains("Unary -> \"Ident\" \"!\""), "{err}");
    assert!(!tmp.join(ll1gen::RUST_DIR).exists(), "nothing written on error");
}

#[test]
fn cli_check_detects_stale_files() {
    let tmp = Path::new(env!("CARGO_TARGET_TMPDIR")).join("ll1gen-check");
    let _ = std::fs::remove_dir_all(&tmp);
    let g = fixture("base.ebnf");
    let run = |check: bool| {
        let mut c = ll1gen();
        c.args(["--profile", "language", "--root"]).arg(&tmp).arg("--grammar").arg(&g);
        if check {
            c.arg("--check");
        }
        c.output().expect("run ll1gen")
    };
    assert_eq!(run(true).status.code(), Some(1), "missing files are stale");
    assert_eq!(run(false).status.code(), Some(0));
    assert_eq!(run(true).status.code(), Some(0));
    let f = tmp.join(ll1gen::STAVE_DIR).join("terminals.stv");
    let mut s = std::fs::read_to_string(&f).unwrap();
    s.push('\n');
    std::fs::write(&f, s).unwrap();
    let out = run(true);
    assert_eq!(out.status.code(), Some(1));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("stage1/src/generated/terminals.stv"), "{err}");
    assert!(!err.contains("ll1_table.stv"), "{err}");
}

#[test]
fn checked_in_files_are_fresh() {
    let out = ll1gen().arg("--check").output().expect("run ll1gen");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
}
