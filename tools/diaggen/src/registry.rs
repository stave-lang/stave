//! Loads and checks `contracts/diagnostics.toml` (format in its header) against the human tables.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;

use crate::md::{self, Row};
use crate::template;

pub const HOLE_KINDS: [&str; 8] =
    ["name", "path", "type", "token-kinds", "int", "place", "source", "text-literal"];

/// Code range prefix (first three characters) → area name.
pub const AREAS: [(&str, &str); 15] = [
    ("E00", "lex-parse"),
    ("E01", "resolve"),
    ("E02", "types"),
    ("E03", "patterns"),
    ("E04", "kinds"),
    ("E05", "borrow"),
    ("E06", "simd"),
    ("E07", "features"),
    ("E08", "threads-unsafe-ffi"),
    ("E09", "generics"),
    ("E10", "driver"),
    ("E11", "gpu-kernel"),
    ("E12", "gpu-program"),
    ("E13", "gpu-compiler"),
    ("E14", "effects"),
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFile {
    #[serde(default)]
    diag: Vec<RawEntry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawEntry {
    code: String,
    severity: Option<String>,
    area: Option<String>,
    rule: Option<String>,
    retired: Option<bool>,
    located: Option<bool>,
    holes: Option<BTreeMap<String, String>>,
    choices: Option<BTreeMap<String, Vec<String>>>,
    message: Option<String>,
    labels: Option<Vec<RawLabel>>,
    fix: Option<RawFix>,
    cases: Option<BTreeMap<String, RawCase>>,
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct RawLabel {
    at: String,
    text: String,
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct RawFix {
    description: String,
    applicability: Option<String>,
    #[serde(default)]
    edits: Vec<RawEdit>,
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct RawEdit {
    at: String,
    replacement: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCase {
    message: Option<String>,
    labels: Option<Vec<RawLabel>>,
    fix: Option<RawFix>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hole {
    pub name: String,
    pub kind: String,
    pub choices: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Label {
    pub at: String,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Edit {
    pub at: String,
    pub replacement: String,
}

/// How far a fix can be trusted (the registry header defines the three values).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Applicability {
    Exact,
    Maybe,
    Placeholder,
}

impl Applicability {
    pub const ALL: [Applicability; 3] =
        [Applicability::Exact, Applicability::Maybe, Applicability::Placeholder];

    /// The registry spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Applicability::Exact => "exact",
            Applicability::Maybe => "maybe",
            Applicability::Placeholder => "placeholder",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fix {
    pub description: String,
    /// Stated or defaulted; `Placeholder` whenever `edits` is empty.
    pub applicability: Applicability,
    pub edits: Vec<Edit>,
}

/// A complete shape: the base (`case == ""`) or a case with the base's missing keys filled in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    pub case: String,
    pub message: String,
    pub labels: Vec<Label>,
    pub fix: Option<Fix>,
}

/// A checked entry. Retired entries have no holes or shapes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub code: String,
    /// `E0004` → 4.
    pub number: u16,
    pub severity: Severity,
    pub area: String,
    pub rule: String,
    pub retired: bool,
    pub located: bool,
    /// Sorted by name.
    pub holes: Vec<Hole>,
    /// Base first, then cases sorted by name.
    pub shapes: Vec<Shape>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Registry {
    /// Sorted by code.
    pub entries: Vec<Entry>,
}

/// All problems found, one line each.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Errors(pub Vec<String>);

impl std::fmt::Display for Errors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for e in &self.0 {
            writeln!(f, "{e}")?;
        }
        Ok(())
    }
}

impl std::error::Error for Errors {}

fn is_role(s: &str) -> bool {
    let mut b = s.bytes();
    matches!(b.next(), Some(c) if c.is_ascii_lowercase())
        && b.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-')
}

fn one_line_text(s: &str) -> bool {
    !s.chars().any(|c| c.is_control())
}

struct Checker<'a> {
    code: &'a str,
    errs: &'a mut Vec<String>,
}

impl Checker<'_> {
    fn err(&mut self, msg: String) {
        self.errs.push(format!("{}: {msg}", self.code));
    }

    /// Checks one template and adds its holes to `used`.
    fn template(
        &mut self,
        what: &str,
        t: &str,
        declared: &BTreeMap<String, String>,
        used: &mut BTreeSet<String>,
    ) {
        if !one_line_text(t) {
            self.err(format!("{what}: control character or newline in {t:?}"));
        }
        match template::holes(t) {
            Err(e) => self.err(format!("{what}: {e}")),
            Ok(hs) => {
                for h in hs {
                    if !declared.contains_key(&h) {
                        self.err(format!("{what}: hole {{{h}}} is not declared in `holes`"));
                    }
                    used.insert(h);
                }
            }
        }
    }

    fn shape(
        &mut self,
        name: &str,
        shape: &Shape,
        located: bool,
        declared: &BTreeMap<String, String>,
        used: &mut BTreeSet<String>,
    ) {
        let what = |k: &str| if name.is_empty() { k.to_string() } else { format!("case {name}: {k}") };
        if shape.message.is_empty() {
            self.err(what("message is empty"));
        }
        if shape.message.ends_with('.') {
            self.err(what("message ends with a period"));
        }
        if shape.message.trim() != shape.message {
            self.err(what("message has leading or trailing spaces"));
        }
        self.template(&what("message"), &shape.message, declared, used);
        for l in &shape.labels {
            self.at(&what("label"), &l.at, located);
            if l.text.is_empty() {
                self.err(what("label text is empty"));
            }
            self.template(&what("label"), &l.text, declared, used);
        }
        if let Some(f) = &shape.fix {
            if f.description.is_empty() {
                self.err(what("fix description is empty"));
            }
            self.template(&what("fix"), &f.description, declared, used);
            for e in &f.edits {
                self.at(&what("edit"), &e.at, located);
                self.template(&what("edit"), &e.replacement, declared, used);
            }
        }
    }

    fn at(&mut self, what: &str, at: &str, located: bool) {
        if at == "primary" {
            if !located {
                self.err(format!("{what}: at = \"primary\" in a location-less entry"));
            }
        } else if !is_role(at) {
            self.err(format!("{what}: bad role {at:?} (lower-case letters, digits, `-`)"));
        }
    }
}

fn convert_labels(ls: &[RawLabel]) -> Vec<Label> {
    ls.iter().map(|l| Label { at: l.at.clone(), text: l.text.clone() }).collect()
}

/// The applicability of a fix: the stated one, else `maybe` with edits and `placeholder`
/// without. A fix without edits has nothing to apply, so it cannot state another value.
fn applicability(f: &RawFix) -> Result<Applicability, String> {
    let default = if f.edits.is_empty() { Applicability::Placeholder } else { Applicability::Maybe };
    let Some(stated) = f.applicability.as_deref() else { return Ok(default) };
    let Some(a) = Applicability::ALL.into_iter().find(|a| a.as_str() == stated) else {
        return Err(format!("applicability {stated:?} is not \"exact\", \"maybe\", or \"placeholder\""));
    };
    if f.edits.is_empty() && a != Applicability::Placeholder {
        return Err(format!("applicability {stated:?} on a fix without edits (always \"placeholder\")"));
    }
    Ok(a)
}

/// Converts a fix. A bad `applicability` is reported for `case` (`""`: the base) and replaced by
/// the default.
fn convert_fix(f: &RawFix, case: &str, c: &mut Checker) -> Fix {
    let applicability = applicability(f).unwrap_or_else(|e| {
        c.err(if case.is_empty() { format!("fix: {e}") } else { format!("case {case}: fix: {e}") });
        if f.edits.is_empty() { Applicability::Placeholder } else { Applicability::Maybe }
    });
    Fix {
        description: f.description.clone(),
        applicability,
        edits: f
            .edits
            .iter()
            .map(|e| Edit { at: e.at.clone(), replacement: e.replacement.clone() })
            .collect(),
    }
}

fn check_entry(raw: RawEntry, errs: &mut Vec<String>) -> Option<Entry> {
    let code = raw.code.clone();
    let mut c = Checker { code: &code, errs };
    if !md::is_code(&code) {
        c.err("code does not match ^E[0-9]{4}$".to_string());
        return None;
    }
    let number: u16 = code[1..].parse().ok()?;
    let rule = raw.rule.clone().unwrap_or_default();
    if rule.is_empty() {
        c.err("`rule` missing".to_string());
    } else if !one_line_text(&rule) {
        c.err("`rule` is not one line".to_string());
    }
    let retired = raw.retired.unwrap_or(false);
    if retired {
        let extra = [
            ("severity", raw.severity.is_some()),
            ("area", raw.area.is_some()),
            ("located", raw.located.is_some()),
            ("holes", raw.holes.is_some()),
            ("choices", raw.choices.is_some()),
            ("message", raw.message.is_some()),
            ("labels", raw.labels.is_some()),
            ("fix", raw.fix.is_some()),
            ("cases", raw.cases.is_some()),
        ];
        for (k, present) in extra {
            if present {
                c.err(format!("retired entry has `{k}` (only code, rule, retired)"));
            }
        }
        return Some(Entry {
            code: code.clone(),
            number,
            severity: Severity::Error,
            area: String::new(),
            rule,
            retired: true,
            located: true,
            holes: Vec::new(),
            shapes: Vec::new(),
        });
    }
    let severity = match raw.severity.as_deref() {
        Some("error") => Severity::Error,
        Some("warning") => Severity::Warning,
        Some(s) => {
            c.err(format!("severity {s:?} is not \"error\" or \"warning\""));
            Severity::Error
        }
        None => {
            c.err("`severity` missing".to_string());
            Severity::Error
        }
    };
    let area = raw.area.clone().unwrap_or_default();
    match AREAS.iter().find(|(p, _)| code.starts_with(p)) {
        None => c.err("code is in no range of diagnostic-codes.md".to_string()),
        Some((_, want)) if *want != area => c.err(format!("area {area:?}, the code's range is {want:?}")),
        Some(_) => {}
    }
    let located = raw.located.unwrap_or(true);
    let declared = raw.holes.clone().unwrap_or_default();
    for (name, kind) in &declared {
        if !template::is_hole_name(name) {
            c.err(format!("bad hole name {name:?}"));
        }
        if !HOLE_KINDS.contains(&kind.as_str()) {
            c.err(format!("hole {name}: unknown kind {kind:?}"));
        }
    }
    let choices = raw.choices.clone().unwrap_or_default();
    for (name, list) in &choices {
        match declared.get(name).map(String::as_str) {
            Some("text-literal") => {}
            Some(k) => c.err(format!("choices for hole {name} of kind {k:?} (only text-literal)")),
            None => c.err(format!("choices for undeclared hole {name}")),
        }
        if list.is_empty() {
            c.err(format!("hole {name}: empty choices"));
        }
        let mut seen = BTreeSet::new();
        for ch in list {
            if !seen.insert(ch) {
                c.err(format!("hole {name}: choice {ch:?} listed twice"));
            }
            if !one_line_text(ch) {
                c.err(format!("hole {name}: choice {ch:?} is not one line"));
            }
        }
    }
    for (name, kind) in &declared {
        if kind == "text-literal" && !choices.contains_key(name) {
            c.err(format!("text-literal hole {name} has no `choices`"));
        }
    }
    let Some(message) = raw.message.clone() else {
        c.err("`message` missing".to_string());
        return None;
    };
    let base_labels = convert_labels(raw.labels.as_deref().unwrap_or_default());
    let base_fix = raw.fix.as_ref().map(|f| convert_fix(f, "", &mut c));
    if severity == Severity::Error && base_fix.is_none() {
        c.err("error entry has no `fix`".to_string());
    }
    let base = Shape { case: String::new(), message, labels: base_labels, fix: base_fix };
    let mut used = BTreeSet::new();
    c.shape("", &base, located, &declared, &mut used);
    let mut shapes = vec![base.clone()];
    for (name, case) in raw.cases.iter().flatten() {
        if !is_role(name) {
            c.err(format!("bad case name {name:?} (lower-case letters, digits, `-`)"));
        }
        if case.message.is_none() && case.labels.is_none() && case.fix.is_none() {
            c.err(format!("case {name} gives no key"));
        }
        let shape = Shape {
            case: name.clone(),
            message: case.message.clone().unwrap_or_else(|| base.message.clone()),
            labels: case.labels.as_deref().map(convert_labels).unwrap_or_else(|| base.labels.clone()),
            fix: match &case.fix {
                Some(f) => Some(convert_fix(f, name, &mut c)),
                None => base.fix.clone(),
            },
        };
        c.shape(name, &shape, located, &declared, &mut used);
        shapes.push(shape);
    }
    for name in declared.keys() {
        if !used.contains(name) {
            c.err(format!("hole {name} is declared but used by no template of the entry or its cases"));
        }
    }
    let holes = declared
        .iter()
        .map(|(name, kind)| Hole {
            name: name.clone(),
            kind: kind.clone(),
            choices: choices.get(name).cloned().unwrap_or_default(),
        })
        .collect();
    Some(Entry { code: code.clone(), number, severity, area, rule, retired: false, located, holes, shapes })
}

/// Parses and checks the registry text alone (no cross-check with the human tables).
pub fn parse_registry(toml_text: &str) -> Result<Registry, Errors> {
    let raw: RawFile =
        toml::from_str(toml_text).map_err(|e| Errors(vec![format!("diagnostics.toml: {e}")]))?;
    let mut errs = Vec::new();
    let mut entries = Vec::new();
    let mut prev: Option<String> = None;
    let mut seen = BTreeSet::new();
    for r in raw.diag {
        let code = r.code.clone();
        if !seen.insert(code.clone()) {
            errs.push(format!("{code}: code appears twice"));
        }
        if let Some(p) = prev.as_ref().filter(|p| **p >= code) {
            errs.push(format!("{code}: entries not sorted by code ({code} after {p})"));
        }
        prev = Some(code);
        if let Some(e) = check_entry(r, &mut errs) {
            entries.push(e);
        }
    }
    if !errs.is_empty() {
        return Err(Errors(errs));
    }
    entries.sort_by(|a, b| a.code.cmp(&b.code));
    Ok(Registry { entries })
}

/// Parses the registry and checks it against the rows of the human tables: the same codes,
/// retired flags, and (where the table has the column) severities.
pub fn check(toml_text: &str, tables: &[(&str, &str)]) -> Result<Registry, Errors> {
    let reg = parse_registry(toml_text)?;
    let mut rows: BTreeMap<String, Row> = BTreeMap::new();
    let mut errs = Vec::new();
    for (source, text) in tables {
        for row in md::rows(source, text) {
            if let Some(first) = rows.get(&row.code) {
                errs.push(format!("{}: listed twice ({} and {})", row.code, first.source, row.source));
                continue;
            }
            rows.insert(row.code.clone(), row);
        }
    }
    for e in &reg.entries {
        match rows.get(&e.code) {
            None => errs
                .push(format!("{}: has an entry in diagnostics.toml but no row in the human tables", e.code)),
            Some(row) => {
                if row.retired != e.retired {
                    errs.push(format!(
                        "{}: retired is {} in diagnostics.toml but {} in {}",
                        e.code, e.retired, row.retired, row.source
                    ));
                }
                if let Some(sev) = &row.severity {
                    let want = match e.severity {
                        Severity::Error => "error",
                        Severity::Warning => "warning",
                    };
                    if !e.retired && sev != want {
                        errs.push(format!(
                            "{}: severity {want} in diagnostics.toml, {sev} in {}",
                            e.code, row.source
                        ));
                    }
                }
            }
        }
    }
    for (code, row) in &rows {
        if reg.entries.binary_search_by(|e| e.code.as_str().cmp(code)).is_err() {
            errs.push(format!("{code}: row in {} has no entry in diagnostics.toml", row.source));
        }
    }
    if errs.is_empty() { Ok(reg) } else { Err(Errors(errs)) }
}
