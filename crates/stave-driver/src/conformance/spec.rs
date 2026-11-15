//! Test discovery and per-test expectations (contracts/conformance-annotations.md).

use std::collections::BTreeSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use stave_target::Triple;

use super::annot::{Annot, Code, Directive, Severity, Syntax, scan};

/// One conformance test: a `.stv` file, or a package directory (a directory with `stave.toml`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TestCase {
    /// The file or package directory.
    pub path: PathBuf,
    /// Display name: the path relative to the suite root, `/`-separated.
    pub name: String,
    pub package: bool,
}

impl TestCase {
    /// The directory diagnostic file paths are relative to: the package directory, or the
    /// directory holding the single file.
    pub fn base_dir(&self) -> PathBuf {
        if self.package {
            self.path.clone()
        } else {
            self.path.parent().map(Path::to_path_buf).unwrap_or_default()
        }
    }
}

fn file_name(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// `p` relative to `base`, `/`-separated, without `.` components; `p` itself (normalized) if it
/// is not under `base`.
pub fn rel_key(base: &Path, p: &Path) -> String {
    let rel = p.strip_prefix(base).unwrap_or(p);
    let parts: Vec<String> = rel
        .components()
        .filter(|c| !matches!(c, std::path::Component::CurDir))
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    parts.join("/")
}

/// Find every test under `root`, sorted by name. `root` may itself be a `.stv` file or a package.
/// Entries whose name starts with `.` or `_` are skipped below the root (`_selftest/` holds the
/// runner's own fixtures, which only make sense with a fake compiler). A package's subdirectories
/// belong to it (path dependencies), so discovery does not descend into packages.
pub fn discover(root: &Path) -> io::Result<Vec<TestCase>> {
    let meta = fs::metadata(root)?;
    if meta.is_file() {
        return Ok(vec![TestCase { path: root.to_path_buf(), name: file_name(root), package: false }]);
    }
    if root.join("stave.toml").is_file() {
        return Ok(vec![TestCase { path: root.to_path_buf(), name: file_name(root), package: true }]);
    }
    let mut out = Vec::new();
    walk(root, root, &mut out)?;
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

fn sorted_entries(dir: &Path) -> io::Result<Vec<(String, PathBuf, fs::FileType)>> {
    let mut v = Vec::new();
    for e in fs::read_dir(dir)? {
        let e = e?;
        v.push((e.file_name().to_string_lossy().into_owned(), e.path(), e.file_type()?));
    }
    v.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(v)
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<TestCase>) -> io::Result<()> {
    for (name, path, ty) in sorted_entries(dir)? {
        if name.starts_with('.') || name.starts_with('_') {
            continue;
        }
        if ty.is_dir() {
            if path.join("stave.toml").is_file() {
                out.push(TestCase { name: rel_key(root, &path), path, package: true });
            } else {
                walk(root, &path, out)?;
            }
        } else if ty.is_file() && name.ends_with(".stv") {
            out.push(TestCase { name: rel_key(root, &path), path, package: false });
        }
    }
    Ok(())
}

/// An expected diagnostic: `loc` is `(file key, line)`, or `None` for `NOLOC`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExpectedDiag {
    pub severity: Severity,
    pub code: Code,
    pub loc: Option<(String, u32)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TestKind {
    /// At least one `ERROR` annotation (located, `NOLOC`, or in `stave.toml`).
    Negative,
    Positive,
}

/// Everything a test's annotations and `expected.*` files say.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TestSpec {
    pub expected: Vec<ExpectedDiag>,
    /// Resolved path of the golden stdout (`STDOUT` or `expected.stdout`).
    pub stdout: Option<PathBuf>,
    pub stderr_contains: Vec<String>,
    /// `EXIT` or `expected.exit`.
    pub exit: Option<u8>,
    /// `TARGET`s in annotation order; empty = the host baseline.
    pub targets: Vec<Triple>,
    pub args: Vec<String>,
    pub check_only: bool,
    /// `PARSE-ONLY`: lex and parse only.
    pub parse_only: bool,
    /// `RESOURCE-LIMIT`: no effect in `stave test` (for the `gpu-sm86` differential tool).
    pub resource_limit: bool,
}

impl TestSpec {
    pub fn kind(&self) -> TestKind {
        if self.expected.iter().any(|d| d.severity == Severity::Error) {
            TestKind::Negative
        } else {
            TestKind::Positive
        }
    }

    /// A positive test with run expectations: the built binary runs.
    pub fn is_run_test(&self) -> bool {
        self.stdout.is_some() || !self.stderr_contains.is_empty() || self.exit.is_some()
    }
}

fn read_text(p: &Path, errs: &mut Vec<String>) -> Option<String> {
    match fs::read(p) {
        // Invalid UTF-8 is legal test input (E0001); line structure survives lossy decoding.
        Ok(b) => Some(String::from_utf8_lossy(&b).into_owned()),
        Err(e) => {
            errs.push(format!("{}: cannot read: {e}", p.display()));
            None
        }
    }
}

/// Files of a package test: every `.stv` and every `stave.toml` below it, as (key, path, syntax).
fn package_files(dir: &Path) -> io::Result<Vec<(String, PathBuf, Syntax)>> {
    fn go(base: &Path, dir: &Path, out: &mut Vec<(String, PathBuf, Syntax)>) -> io::Result<()> {
        for (name, path, ty) in sorted_entries(dir)? {
            if name.starts_with('.') {
                continue;
            }
            if ty.is_dir() {
                go(base, &path, out)?;
            } else if ty.is_file() && name.ends_with(".stv") {
                out.push((rel_key(base, &path), path, Syntax::Stave));
            } else if ty.is_file() && name == "stave.toml" {
                out.push((rel_key(base, &path), path, Syntax::Toml));
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    go(dir, dir, &mut out)?;
    Ok(out)
}

/// Read a test's annotations. `Err` lists why it is malformed. `known_codes`: if given, every
/// annotated code must be in it.
pub fn load_spec(case: &TestCase, known_codes: Option<&BTreeSet<Code>>) -> Result<TestSpec, Vec<String>> {
    let mut errs = Vec::new();
    let base = case.base_dir();
    let files = if case.package {
        package_files(&case.path).map_err(|e| vec![format!("{}: cannot list: {e}", case.path.display())])?
    } else {
        vec![(file_name(&case.path), case.path.clone(), Syntax::Stave)]
    };
    let mut spec = TestSpec::default();
    let mut seen: BTreeSet<&'static str> = BTreeSet::new();
    for (key, path, syntax) in &files {
        let Some(text) = read_text(path, &mut errs) else { continue };
        // an empty single-file test cannot hold an annotation: it is a PARSE-ONLY positive
        // (contracts/conformance-annotations.md "Syntax")
        if !case.package && text.is_empty() {
            spec.parse_only = true;
        }
        let (annots, bad) = scan(&text, *syntax);
        for b in bad {
            errs.push(format!("{key}:{}: malformed annotation: {}", b.line, b.message));
        }
        let is_root = !case.package || key == "src/main.stv" || key == "src/lib.stv";
        for a in annots {
            let at = format!("{key}:{}", a.comment_line);
            match a.annot {
                Annot::Diag { severity, code, line } => {
                    if let Some(known) = known_codes
                        && !known.contains(&code)
                    {
                        errs.push(format!("{at}: {code} is not in the diagnostic registry"));
                    }
                    spec.expected.push(ExpectedDiag { severity, code, loc: line.map(|l| (key.clone(), l)) });
                }
                Annot::Directive(d) => {
                    let kw = d.keyword();
                    if !is_root {
                        errs.push(format!(
                            "{at}: {kw} is allowed only in the root module (src/main.stv or src/lib.stv)"
                        ));
                        continue;
                    }
                    let repeatable = matches!(d, Directive::Target(_) | Directive::StderrContains(_));
                    if !repeatable && !seen.insert(kw) {
                        errs.push(format!("{at}: {kw} given twice"));
                        continue;
                    }
                    let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
                    match d {
                        Directive::Stdout(f) => spec.stdout = Some(dir.join(f)),
                        Directive::StderrContains(s) => spec.stderr_contains.push(s),
                        Directive::Exit(n) => spec.exit = Some(n),
                        Directive::Target(t) => {
                            if spec.targets.contains(&t) {
                                errs.push(format!("{at}: TARGET {t} given twice"));
                            }
                            spec.targets.push(t);
                        }
                        Directive::Args(a) => spec.args = a,
                        Directive::CheckOnly => spec.check_only = true,
                        Directive::ParseOnly => spec.parse_only = true,
                        Directive::ResourceLimit => spec.resource_limit = true,
                    }
                }
            }
        }
    }
    if case.package {
        let golden = base.join("expected.stdout");
        if golden.is_file() {
            if spec.stdout.is_some() {
                errs.push("both expected.stdout and a STDOUT annotation".to_string());
            }
            spec.stdout = Some(golden);
        }
        let exit_file = base.join("expected.exit");
        if exit_file.is_file()
            && let Some(text) = read_text(&exit_file, &mut errs)
        {
            let t = text.trim();
            match t.parse::<u8>() {
                Ok(n) if t.bytes().all(|b| b.is_ascii_digit()) => {
                    if spec.exit.is_some() {
                        errs.push("both expected.exit and an EXIT annotation".to_string());
                    }
                    spec.exit = Some(n);
                }
                _ => errs.push(format!("expected.exit: one decimal exit status 0..=255, found `{t}`")),
            }
        }
    }
    if spec.kind() == TestKind::Negative && spec.is_run_test() {
        errs.push("a negative test (ERROR annotations) cannot have run expectations".to_string());
    }
    if spec.check_only && spec.is_run_test() {
        errs.push("CHECK-ONLY cannot be combined with run expectations".to_string());
    }
    if spec.parse_only && spec.is_run_test() {
        errs.push("PARSE-ONLY cannot be combined with run expectations".to_string());
    }
    if spec.parse_only && spec.check_only {
        errs.push("PARSE-ONLY and CHECK-ONLY cannot be combined".to_string());
    }
    if errs.is_empty() { Ok(spec) } else { Err(errs) }
}

/// Every code with a row in a contract table (`| EXXXX | ...`), e.g. diagnostic-codes.md and the
/// GPU diagnostic tables.
pub fn codes_from_registry(text: &str) -> BTreeSet<Code> {
    text.lines()
        .filter_map(|l| {
            let cell = l.trim_start().strip_prefix('|')?.split('|').next()?.trim();
            Code::parse(cell)
        })
        .collect()
}
