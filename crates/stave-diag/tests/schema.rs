//! Every emitted JSON line validates against contracts/diagnostics.schema.json (oneOf
//! diagnostic | timings | parallelism); `fix` on every error; `primary` absent only for the
//! location-less codes; `col` 1-based in UTF-8 bytes; `message` one line, no trailing period.

mod common;

use serde_json::Value;
use stave_diag::{Diagnostic, SourceFiles, Span, emit_json};

fn schema() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../contracts/diagnostics.schema.json");
    let text = std::fs::read_to_string(path).expect("read diagnostics.schema.json");
    serde_json::from_str(&text).expect("schema is JSON")
}

/// A JSON Schema validator for the keywords diagnostics.schema.json uses. An unknown keyword is
/// an error, so the subset cannot silently grow.
struct Validator<'a> {
    root: &'a Value,
}

impl Validator<'_> {
    fn resolve(&self, r: &str) -> Result<&Value, String> {
        let p = r.strip_prefix("#/").ok_or_else(|| format!("unsupported $ref {r}"))?;
        let mut v = self.root;
        for part in p.split('/') {
            v = v.get(part).ok_or_else(|| format!("dangling $ref {r}"))?;
        }
        Ok(v)
    }

    fn type_ok(t: &str, v: &Value) -> bool {
        match t {
            "object" => v.is_object(),
            "array" => v.is_array(),
            "string" => v.is_string(),
            "integer" => v.is_i64() || v.is_u64(),
            "number" => v.is_number(),
            "boolean" => v.is_boolean(),
            "null" => v.is_null(),
            _ => false,
        }
    }

    fn pattern_ok(p: &str, s: &str) -> Result<bool, String> {
        match p {
            "^E[0-9]{4}$" => {
                Ok(s.len() == 5 && s.starts_with('E') && s[1..].bytes().all(|b| b.is_ascii_digit()))
            }
            _ => Err(format!("unsupported pattern {p}")),
        }
    }

    fn validate(&self, schema: &Value, v: &Value, at: &str, errs: &mut Vec<String>) {
        let Some(obj) = schema.as_object() else {
            errs.push(format!("{at}: schema is not an object"));
            return;
        };
        for (k, s) in obj {
            match k.as_str() {
                "$schema" | "$id" | "title" | "description" | "$defs" => {}
                "then" | "else" => {}
                "$ref" => match s.as_str().map(|r| self.resolve(r)) {
                    Some(Ok(target)) => self.validate(target, v, at, errs),
                    Some(Err(e)) => errs.push(e),
                    None => errs.push(format!("{at}: $ref not a string")),
                },
                "oneOf" => {
                    let n = s
                        .as_array()
                        .map(|alts| alts.iter().filter(|alt| self.check(alt, v)).count())
                        .unwrap_or(0);
                    if n != 1 {
                        errs.push(format!("{at}: {n} oneOf alternatives match, want 1"));
                    }
                }
                "allOf" => {
                    for alt in s.as_array().into_iter().flatten() {
                        self.validate(alt, v, at, errs);
                    }
                }
                "if" => {
                    let branch = if self.check(s, v) { obj.get("then") } else { obj.get("else") };
                    if let Some(b) = branch {
                        self.validate(b, v, at, errs);
                    }
                }
                "type" => {
                    let ok = match s {
                        Value::String(t) => Self::type_ok(t, v),
                        Value::Array(ts) => {
                            ts.iter().any(|t| t.as_str().is_some_and(|t| Self::type_ok(t, v)))
                        }
                        _ => false,
                    };
                    if !ok {
                        errs.push(format!("{at}: not of type {s}"));
                    }
                }
                "const" => {
                    if v != s {
                        errs.push(format!("{at}: {v} is not the const {s}"));
                    }
                }
                "enum" => {
                    if !s.as_array().is_some_and(|vals| vals.contains(v)) {
                        errs.push(format!("{at}: {v} not in enum {s}"));
                    }
                }
                "pattern" => {
                    if let (Some(p), Some(str_v)) = (s.as_str(), v.as_str()) {
                        match Self::pattern_ok(p, str_v) {
                            Ok(true) => {}
                            Ok(false) => errs.push(format!("{at}: {str_v:?} does not match {p}")),
                            Err(e) => errs.push(e),
                        }
                    }
                }
                "minimum" => {
                    if let (Some(min), Some(x)) = (s.as_f64(), v.as_f64())
                        && x < min
                    {
                        errs.push(format!("{at}: {x} < minimum {min}"));
                    }
                }
                "required" => {
                    if let Some(o) = v.as_object() {
                        for r in s.as_array().into_iter().flatten().filter_map(Value::as_str) {
                            if !o.contains_key(r) {
                                errs.push(format!("{at}: required key {r:?} missing"));
                            }
                        }
                    }
                }
                "properties" => {
                    if let (Some(props), Some(o)) = (s.as_object(), v.as_object()) {
                        for (pk, pv) in o {
                            if let Some(ps) = props.get(pk) {
                                self.validate(ps, pv, &format!("{at}.{pk}"), errs);
                            }
                        }
                    }
                }
                "additionalProperties" => {
                    if let Some(o) = v.as_object() {
                        let props = obj.get("properties").and_then(Value::as_object);
                        for (pk, pv) in o {
                            if props.is_some_and(|p| p.contains_key(pk)) {
                                continue;
                            }
                            match s {
                                Value::Bool(false) => errs.push(format!("{at}: unexpected key {pk:?}")),
                                Value::Bool(true) => {}
                                other => self.validate(other, pv, &format!("{at}.{pk}"), errs),
                            }
                        }
                    }
                }
                "items" => {
                    if let Some(a) = v.as_array() {
                        for (i, item) in a.iter().enumerate() {
                            self.validate(s, item, &format!("{at}[{i}]"), errs);
                        }
                    }
                }
                other => errs.push(format!("{at}: unsupported schema keyword {other:?}")),
            }
        }
    }

    fn check(&self, schema: &Value, v: &Value) -> bool {
        let mut errs = Vec::new();
        self.validate(schema, v, "$", &mut errs);
        errs.is_empty()
    }
}

fn validate_line(schema: &Value, line: &str) -> Result<Value, String> {
    let v: Value = serde_json::from_str(line).map_err(|e| format!("not JSON: {e}: {line}"))?;
    let mut errs = Vec::new();
    Validator { root: schema }.validate(schema, &v, "$", &mut errs);
    if errs.is_empty() { Ok(v) } else { Err(format!("{line}\n  {}", errs.join("\n  "))) }
}

fn json_lines(diags: &[Diagnostic], files: &SourceFiles) -> Vec<String> {
    let out = emit_json(diags, files).expect("spans resolve");
    assert!(out.ends_with('\n'));
    out.lines().map(str::to_string).collect()
}

#[test]
fn every_code_and_case_validates() {
    let schema = schema();
    let (files, file) = common::sample_files();
    let diags = common::all_diagnostics(file);
    assert!(diags.len() >= 100, "one diagnostic per non-retired (code, shape)");
    let lines = json_lines(&diags, &files);
    assert_eq!(lines.len(), diags.len(), "one object per line");
    let mut failures = Vec::new();
    for line in &lines {
        if let Err(e) = validate_line(&schema, line) {
            failures.push(e);
        }
    }
    assert!(failures.is_empty(), "invalid lines:\n{}", failures.join("\n"));
}

#[test]
fn errors_carry_a_fix_and_messages_are_one_line() {
    let (files, file) = common::sample_files();
    for line in json_lines(&common::all_diagnostics(file), &files) {
        let v: Value = serde_json::from_str(&line).expect("JSON");
        let message = v["message"].as_str().expect("message");
        assert!(!message.contains('\n') && !message.ends_with('.'), "{line}");
        assert!(!message.is_empty(), "{line}");
        if v["severity"] == "error" {
            let fix = v["fix"]["description"].as_str().expect("fix on an error");
            assert!(!fix.is_empty(), "{line}");
        }
        let fix_ok =
            v.get("fix").is_none_or(|f| f["description"].as_str().is_some_and(|d| !d.contains('\n')));
        assert!(fix_ok, "{line}");
    }
}

#[test]
fn every_fix_states_its_applicability() {
    let (files, file) = common::sample_files();
    let mut exact = std::collections::BTreeSet::new();
    for line in json_lines(&common::all_diagnostics(file), &files) {
        let v: Value = serde_json::from_str(&line).expect("JSON");
        let Some(fix) = v.get("fix") else { continue };
        let a = fix["applicability"].as_str().expect("applicability on every fix");
        let edits = fix["edits"].as_array().expect("edits");
        assert!(matches!(a, "exact" | "maybe" | "placeholder"), "{line}");
        assert!(!edits.is_empty() || a == "placeholder", "a fix without edits is a placeholder: {line}");
        if a == "exact" {
            exact.insert(v["code"].as_str().expect("code").to_string());
        }
    }
    assert_eq!(
        exact.into_iter().collect::<Vec<_>>(),
        ["E0017", "E0023", "E0024", "E0026", "E0027", "E0210", "E0907"]
    );
    // an edit whose role span is not supplied is left out: the fix has nothing to apply
    let d =
        Diagnostic::new("E0210", None, &[("type", "u32")], Some(Span::new(file, 0, 1)), &[]).expect("E0210");
    let fix = d.fix().expect("fix");
    assert!(fix.edits.is_empty());
    assert_eq!(fix.applicability, stave_diag::Applicability::Placeholder);
    let line = &json_lines(std::slice::from_ref(&d), &files)[0];
    assert!(line.contains(r#""applicability":"placeholder","edits":[]"#), "{line}");
}

#[test]
fn primary_absent_only_for_locationless_codes() {
    let locationless: Vec<&str> =
        stave_diag::entries().iter().filter(|e| !e.retired && !e.located).map(|e| e.code).collect();
    assert_eq!(locationless, ["E1001", "E1301", "E1302", "E1303"], "NOLOC codes of the schema description");
    let (files, file) = common::sample_files();
    for line in json_lines(&common::all_diagnostics(file), &files) {
        let v: Value = serde_json::from_str(&line).expect("JSON");
        let code = v["code"].as_str().expect("code");
        assert_eq!(v.get("primary").is_none(), locationless.contains(&code), "{line}");
    }
    // A located code refuses a missing primary span, a location-less one refuses a given one.
    let span = Span::new(file, 0, 1);
    assert!(Diagnostic::new("E0101", None, &[("name", "x")], None, &[]).is_err());
    assert!(Diagnostic::new("E1302", None, &[("reason", "no device")], Some(span), &[]).is_err());
}

#[test]
fn col_is_one_based_utf8_bytes() {
    let (files, file) = common::sample_files();
    // Offset 29 is the `:` after `é` (2 bytes) on line 2, which starts at offset 19.
    let d =
        Diagnostic::new("E0101", None, &[("name", "é")], Some(Span::new(file, 29, 1)), &[]).expect("E0101");
    let line = &json_lines(&[d], &files)[0];
    let v: Value = serde_json::from_str(line).expect("JSON");
    assert_eq!(v["primary"]["line"], 2);
    assert_eq!(v["primary"]["col"], 11, "bytes, not chars: {line}");
    assert_eq!(v["primary"]["file"], "src/main.stv");
    // First byte of the file.
    let d = Diagnostic::new("E0017", None, &[], Some(Span::new(file, 0, 3)), &[]).expect("E0017");
    let v: Value = serde_json::from_str(&json_lines(&[d], &files)[0]).expect("JSON");
    assert_eq!((v["primary"]["line"].as_u64(), v["primary"]["col"].as_u64()), (Some(1), Some(1)));
}

#[test]
fn strings_are_escaped() {
    let schema = schema();
    let (files, file) = common::sample_files();
    let tricky = "a\"b\\c\u{1}d";
    let d =
        Diagnostic::new("E0006", None, &[("hole", tricky)], Some(Span::new(file, 0, 1)), &[]).expect("E0006");
    let v = validate_line(&schema, &json_lines(std::slice::from_ref(&d), &files)[0]).expect("valid");
    assert_eq!(v["message"].as_str(), Some(d.message()));
    assert!(d.message().contains(tricky));
}

#[test]
fn validator_rejects_bad_lines() {
    let schema = schema();
    let bad = [
        // error without fix
        r#"{"kind":"diagnostic","code":"E0101","severity":"error","message":"m"}"#,
        // fix without applicability
        r#"{"kind":"diagnostic","code":"E0101","severity":"error","message":"m","fix":{"description":"d","edits":[]}}"#,
        // unknown applicability
        r#"{"kind":"diagnostic","code":"E0101","severity":"error","message":"m","fix":{"description":"d","applicability":"sure","edits":[]}}"#,
        // bad code
        r#"{"kind":"diagnostic","code":"X0101","severity":"warning","message":"m"}"#,
        // unknown key
        r#"{"kind":"diagnostic","code":"E0101","severity":"warning","message":"m","extra":1}"#,
        // col 0
        r#"{"kind":"diagnostic","code":"E1205","severity":"warning","message":"m","primary":{"file":"a","line":1,"col":0,"len":0}}"#,
    ];
    for line in bad {
        assert!(validate_line(&schema, line).is_err(), "accepted: {line}");
    }
    let good = [
        r#"{"kind":"diagnostic","code":"E1205","severity":"warning","message":"m"}"#,
        r#"{"kind":"diagnostic","code":"E0101","severity":"error","message":"m","fix":{"description":"d","applicability":"placeholder","edits":[]}}"#,
        r#"{"kind":"timings","stages":[{"stage":"lex","ms":1.5,"items":10}]}"#,
    ];
    for line in good {
        validate_line(&schema, line).expect("valid");
    }
}
