//! Packages: `stave.toml`, path dependencies, per-target dependencies,
//! built-in packages (`std`, `runtime`), and module file discovery.
//!
//! - A package is a directory with `stave.toml` and `src/main.stv` (`kind = "bin"`) or `src/lib.stv`
//!   (`kind = "lib"`). Its modules are the `.stv` files under `src/`: `src/a/b.stv` is `a.b`.
//! - A built-in package (passed by the driver: `std/`, `runtime/`) has its modules directly under
//!   its directory (no `src/`), so `std/io.stv` is `std.io`. It may have a `stave.toml`:
//!   then its `[dependencies]` and `[target.<t>.dependencies]` apply as for any package, and each
//!   path dependency is a built-in package too (same layout and privileges, e.g. intrinsics). A
//!   subdirectory holding a `stave.toml` is a nested package, not part of the enclosing package's
//!   modules (`runtime/x86_64/` is the package `runtime_x86_64`, bound as `arch` in
//!   `runtime/stave.toml` for x86-64 targets).
//! - A single `.stv` file (conformance tests, LANG.md examples) is a package of one module,
//!   `main`.
//! - Dependencies for target T: `[dependencies]`, then `[target.x86_64-linux.dependencies]` (any
//!   x86-64 level) or `[target.aarch64-linux.dependencies]` (any AArch64 target), then
//!   `[target.T.dependencies]`; a later table overrides a name.
//! - Invalid manifests are E1002, located in `stave.toml` (cases: unknown key, `wrong-type`,
//!   `non-path-dependency`, `unknown-target`, `syntax`, `duplicate-key`, `bad-kind`,
//!   `missing-name`, `bad-name`, `missing-root`, `missing-dependency`, `unreadable`). A source file that
//!   cannot be read is E1003, at offset 0 of that file.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use stave_target::{Arch, Triple};

use crate::toml::{self, TEntry, TSpan, TTable, TValue};

/// A built-in package supplied by the driver (no manifest).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuiltinPackage {
    pub name: String,
    pub dir: PathBuf,
    /// Names of other built-in packages it depends on (`std` → `runtime`). Its own `stave.toml`,
    /// if any, adds path dependencies.
    pub deps: Vec<String>,
    /// Every manifest package depends on it implicitly (`std`).
    pub implicit: bool,
}

/// What a package is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PackageKind {
    Bin,
    Lib,
    /// A built-in package, or a dependency of one (modules at the directory root, no root
    /// module; the manifest is optional).
    Builtin,
    /// One `.stv` file given as the build root.
    SingleFile,
}

/// A package of the program (canonical order: (name, directory)).
#[derive(Clone, Debug)]
pub struct Package {
    pub name: String,
    pub kind: PackageKind,
    pub version: Option<String>,
    pub dir: PathBuf,
    /// The manifest's file id (a file of [`crate::Program::source_map`]).
    pub manifest: Option<stave_diag::FileId>,
    /// Effective dependencies for the build target (binding name, package), sorted by name.
    pub deps: Vec<(String, crate::PackageId)>,
    /// `src/main.stv` or `src/lib.stv` (single file: the file).
    pub root_module: crate::ModuleId,
    /// The package being built.
    pub is_root: bool,
}

/// Read access to sources: the disk, or an in-memory tree (tests).
#[derive(Clone, Debug, Default)]
pub enum Fs {
    #[default]
    Disk,
    /// Normalized path → contents.
    Mem(BTreeMap<PathBuf, Vec<u8>>),
}

/// Lexical normalization: drops `.`, resolves `..` against earlier components.
pub fn normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// File contents.
pub enum FileBytes {
    Owned(Vec<u8>),
    Mapped(stave_syntax::SourceBytes),
}

impl std::ops::Deref for FileBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            FileBytes::Owned(v) => v,
            FileBytes::Mapped(m) => m,
        }
    }
}

impl std::fmt::Debug for FileBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FileBytes({} bytes)", self.len())
    }
}

/// Deepest directory nesting searched for modules (guards against symlink loops).
const MAX_DIR_DEPTH: usize = 64;

impl Fs {
    pub fn is_file(&self, p: &Path) -> bool {
        match self {
            Fs::Disk => p.is_file(),
            Fs::Mem(m) => m.contains_key(&normalize(p)),
        }
    }

    pub fn read(&self, p: &Path) -> std::io::Result<FileBytes> {
        match self {
            Fs::Disk => stave_syntax::map_file(p).map(FileBytes::Mapped),
            Fs::Mem(m) => m
                .get(&normalize(p))
                .map(|v| FileBytes::Owned(v.clone()))
                .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::NotFound)),
        }
    }

    fn read_vec(&self, p: &Path) -> std::io::Result<Vec<u8>> {
        match self {
            Fs::Disk => std::fs::read(p),
            Fs::Mem(m) => m
                .get(&normalize(p))
                .cloned()
                .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::NotFound)),
        }
    }

    /// A stable identity for a package directory.
    pub fn canonical(&self, p: &Path) -> PathBuf {
        match self {
            Fs::Disk => std::fs::canonicalize(p).unwrap_or_else(|_| normalize(p)),
            Fs::Mem(_) => normalize(p),
        }
    }

    /// The `.stv` files of a built-in package directory: [`Fs::stave_files`] minus the files inside
    /// a subdirectory that holds its own `stave.toml` (a nested package).
    pub fn builtin_files(&self, dir: &Path) -> Vec<String> {
        let mut nested: BTreeMap<String, bool> = BTreeMap::new();
        let mut out = Vec::new();
        for rel in self.stave_files(dir) {
            let parts: Vec<&str> = rel.split('/').collect();
            let mut inside = false;
            for k in 1..parts.len() {
                let d = parts[..k].join("/");
                let is_pkg = *nested
                    .entry(d.clone())
                    .or_insert_with(|| self.is_file(&dir.join(&d).join("stave.toml")));
                if is_pkg {
                    inside = true;
                    break;
                }
            }
            if !inside {
                out.push(rel);
            }
        }
        out
    }

    /// Every `.stv` file below `dir`, as `/`-separated relative paths, sorted by bytes.
    pub fn stave_files(&self, dir: &Path) -> Vec<String> {
        let mut out = Vec::new();
        match self {
            Fs::Disk => walk(dir, String::new(), 0, &mut out),
            Fs::Mem(m) => {
                let dir = normalize(dir);
                for p in m.keys() {
                    if let Ok(rel) = p.strip_prefix(&dir)
                        && rel.extension().is_some_and(|e| e == "stv")
                    {
                        let parts: Vec<String> =
                            rel.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
                        out.push(parts.join("/"));
                    }
                }
            }
        }
        out.sort();
        out
    }
}

fn walk(dir: &Path, prefix: String, depth: usize, out: &mut Vec<String>) {
    if depth > MAX_DIR_DEPTH {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<(String, PathBuf)> =
        rd.filter_map(|e| e.ok()).map(|e| (e.file_name().to_string_lossy().into_owned(), e.path())).collect();
    entries.sort();
    for (name, path) in entries {
        let rel = if prefix.is_empty() { name.clone() } else { format!("{prefix}/{name}") };
        if path.is_dir() {
            walk(&path, rel, depth + 1, out);
        } else if name.ends_with(".stv") && path.is_file() {
            out.push(rel);
        }
    }
}

/// An E1002 found while reading a manifest (converted to a diagnostic once file ids exist).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ManifestIssue {
    /// `None` (unknown key) or the E1002 case name.
    pub case: Option<&'static str>,
    /// The case's holes (`key`, `expected`, `root`).
    pub holes: Vec<(&'static str, String)>,
    pub span: TSpan,
}

impl ManifestIssue {
    fn new(case: Option<&'static str>, holes: &[(&'static str, &str)], span: TSpan) -> ManifestIssue {
        ManifestIssue { case, holes: holes.iter().map(|(k, v)| (*k, v.to_string())).collect(), span }
    }

    /// The key text of the issue (tests).
    #[cfg(test)]
    fn key(&self) -> String {
        self.holes.iter().find(|h| h.0 == "key").map_or(String::new(), |h| h.1.clone())
    }
}

/// A manifest file read during loading.
#[derive(Debug)]
pub(crate) struct ManifestFile {
    pub package: usize,
    pub display_path: String,
    pub text: Vec<u8>,
    pub issues: Vec<ManifestIssue>,
}

/// A source file found during loading (not yet read).
#[derive(Clone, Debug)]
pub(crate) struct PendingFile {
    pub package: usize,
    /// Relative to the package directory (`src/a/b.stv`; built-in: `io.stv`).
    pub rel_path: String,
    /// Path for diagnostics: relative to the root package directory when inside it.
    pub display_path: String,
    pub abs_path: PathBuf,
    /// Module path relative to the package (`["a", "b"]`; `["main"]`, `["lib"]`).
    pub module_rel: Vec<String>,
}

/// A package during loading (indices into `Loaded::packages`).
#[derive(Debug)]
pub(crate) struct RawPackage {
    pub name: String,
    pub kind: PackageKind,
    pub version: Option<String>,
    pub dir: PathBuf,
    pub canonical: PathBuf,
    pub deps: Vec<(String, usize)>,
    pub manifest: Option<usize>,
    pub is_root: bool,
    /// Root module relative path (`["main"]`, `["lib"]`), if any.
    pub root_rel: Option<Vec<String>>,
}

#[derive(Debug, Default)]
pub(crate) struct Loaded {
    pub packages: Vec<RawPackage>,
    pub manifests: Vec<ManifestFile>,
    pub files: Vec<PendingFile>,
}

/// One dependency entry of a manifest table.
#[derive(Clone, Debug)]
struct DepSpec {
    name: String,
    path: Option<String>,
    /// The `path` value (E1002 `missing-dependency`).
    path_span: TSpan,
}

#[derive(Debug, Default)]
struct ManifestData {
    name: Option<String>,
    kind: Option<PackageKind>,
    /// The `kind` value (E1002 `missing-root`).
    kind_span: Option<TSpan>,
    version: Option<String>,
    deps: Vec<DepSpec>,
    /// `[target.<t>.dependencies]` by target table name.
    target_deps: Vec<(String, Vec<DepSpec>)>,
}

const TARGET_ALIAS_X86: &str = "x86_64-linux";
/// Also the exact name of the AArch64 base target; its table applies to `+sve` and `+sve2` too.
const TARGET_ALIAS_A64: &str = "aarch64-linux";

fn known_target(name: &str) -> bool {
    name == TARGET_ALIAS_X86 || Triple::ALL.iter().any(|t| t.name() == name)
}

struct Check<'a> {
    text: &'a [u8],
    issues: Vec<ManifestIssue>,
}

impl Check<'_> {
    fn src(&self, s: TSpan) -> String {
        String::from_utf8_lossy(self.text.get(s.lo as usize..s.hi as usize).unwrap_or_default()).into_owned()
    }

    fn unknown(&mut self, e: &TEntry) {
        let key = self.src(e.key_span);
        self.issues.push(ManifestIssue::new(None, &[("key", &key)], e.key_span));
    }

    fn wrong_type(&mut self, e: &TEntry, expected: &'static str) {
        let key = self.src(e.key_span);
        let span = e.value.span();
        self.issues.push(ManifestIssue::new(
            Some("wrong-type"),
            &[("key", &key), ("expected", expected)],
            span,
        ));
    }

    fn non_path(&mut self, e: &TEntry) {
        let key = self.src(e.key_span);
        self.issues.push(ManifestIssue::new(Some("non-path-dependency"), &[("key", &key)], e.key_span));
    }

    fn string(&mut self, e: &TEntry) -> Option<String> {
        match &e.value {
            TValue::Str(s, _) => Some(s.clone()),
            _ => {
                self.wrong_type(e, "string");
                None
            }
        }
    }

    fn table<'t>(&mut self, e: &'t TEntry) -> Option<&'t TTable> {
        match &e.value {
            TValue::Table(t) => Some(t),
            _ => {
                self.wrong_type(e, "table");
                None
            }
        }
    }

    fn deps(&mut self, t: &TTable) -> Vec<DepSpec> {
        let mut out = Vec::new();
        for e in &t.entries {
            let mut spec = DepSpec { name: e.key.clone(), path: None, path_span: e.key_span };
            match &e.value {
                TValue::Table(d) => {
                    let has_path = d.get("path").is_some();
                    if !has_path {
                        self.non_path(e);
                    }
                    for k in &d.entries {
                        if k.key == "path" {
                            spec.path = self.string(k);
                            spec.path_span = k.value.span();
                        } else if has_path {
                            self.unknown(k);
                        }
                    }
                }
                _ => self.non_path(e),
            }
            out.push(spec);
        }
        out
    }

    fn manifest(&mut self, root: &TTable) -> ManifestData {
        let mut m = ManifestData::default();
        for e in &root.entries {
            match e.key.as_str() {
                "name" => {
                    m.name = self.string(e);
                    if m.name.as_deref().is_some_and(|n| !valid_name(n)) {
                        let (v, at) = (self.src(e.value.span()), e.value.span());
                        self.issues.push(ManifestIssue::new(Some("bad-name"), &[("key", &v)], at));
                    }
                }
                "version" => m.version = self.string(e),
                "kind" => {
                    if let Some(k) = self.string(e) {
                        m.kind_span = Some(e.value.span());
                        match k.as_str() {
                            "bin" => m.kind = Some(PackageKind::Bin),
                            "lib" => m.kind = Some(PackageKind::Lib),
                            _ => {
                                let v = self.src(e.value.span());
                                self.issues.push(ManifestIssue::new(
                                    Some("bad-kind"),
                                    &[("key", &v)],
                                    e.value.span(),
                                ));
                            }
                        }
                    }
                }
                "dependencies" => {
                    if let Some(t) = self.table(e) {
                        m.deps = self.deps(t);
                    }
                }
                "target" => {
                    let Some(t) = self.table(e) else { continue };
                    for te in &t.entries {
                        if !known_target(&te.key) {
                            let key = self.src(te.key_span);
                            let key = key.trim_matches(['"', '\'']).to_string();
                            self.issues.push(ManifestIssue::new(
                                Some("unknown-target"),
                                &[("key", &key)],
                                te.key_span,
                            ));
                            continue;
                        }
                        let Some(tt) = self.table(te) else { continue };
                        for de in &tt.entries {
                            if de.key == "dependencies" {
                                if let Some(dt) = self.table(de) {
                                    let deps = self.deps(dt);
                                    m.target_deps.push((te.key.clone(), deps));
                                }
                            } else {
                                self.unknown(de);
                            }
                        }
                    }
                }
                // GPU program (`[run]`, `[gpu]`) and stage1 compiler (`[compiler]`) keys: checked there.
                "run" | "gpu" | "compiler" => {
                    let _ = self.table(e);
                }
                _ => self.unknown(e),
            }
        }
        m
    }
}

/// Whether `name` can be a package name: one plain path component (it names the default
/// executable), so not empty, `.`, or `..`, no `/`, `\`, or NUL, no leading `-` (an option).
fn valid_name(name: &str) -> bool {
    !(name.is_empty()
        || name == "."
        || name == ".."
        || name.starts_with('-')
        || name.contains(['/', '\\', '\0']))
}

/// Reads and checks one manifest: (data, issues). `name` is required (E1002 `missing-name`, at
/// offset 0) unless the manifest stopped at a syntax error.
fn read_manifest(text: &[u8]) -> (ManifestData, Vec<ManifestIssue>) {
    let (root, err) = toml::parse(text);
    let mut c = Check { text, issues: Vec::new() };
    let data = c.manifest(&root);
    match err {
        Some(e) if e.duplicate => {
            // The redefined key: from the error offset to the end of the key.
            let lo = e.offset as usize;
            let end = text.get(lo..).map_or(lo, |r| {
                lo + r
                    .iter()
                    .position(|&b| matches!(b, b'=' | b']' | b'\n' | b' ' | b'\t' | b'.'))
                    .unwrap_or(r.len())
            });
            let key = String::from_utf8_lossy(text.get(lo..end).unwrap_or_default()).into_owned();
            let span = TSpan { lo: e.offset, hi: end.min(u32::MAX as usize) as u32 };
            c.issues.push(ManifestIssue::new(Some("duplicate-key"), &[("key", &key)], span));
        }
        Some(e) => {
            // The rest of the line from the error offset.
            let lo = e.offset as usize;
            let end =
                text.get(lo..).map_or(lo, |r| lo + r.iter().position(|&b| b == b'\n').unwrap_or(r.len()));
            let key = String::from_utf8_lossy(text.get(lo..end).unwrap_or_default()).trim().to_string();
            let key = if key.is_empty() { "<end of line>".to_string() } else { key };
            let span = TSpan { lo: e.offset, hi: end.min(u32::MAX as usize) as u32 };
            c.issues.push(ManifestIssue::new(Some("syntax"), &[("key", &key)], span));
        }
        None => {}
    }
    if root.get("name").is_none() && err.is_none() {
        c.issues.push(ManifestIssue::new(Some("missing-name"), &[], TSpan { lo: 0, hi: 0 }));
    }
    (data, c.issues)
}

/// The effective dependency list for `target`: base, then the arch table (`x86_64-linux` for
/// every x86-64 level, `aarch64-linux` for every AArch64 target), then the exact one.
fn effective_deps(m: &ManifestData, target: Triple) -> Vec<DepSpec> {
    let mut out: Vec<DepSpec> = Vec::new();
    let mut apply = |deps: &[DepSpec]| {
        for d in deps {
            match out.iter_mut().find(|o| o.name == d.name) {
                Some(o) => *o = d.clone(),
                None => out.push(d.clone()),
            }
        }
    };
    apply(&m.deps);
    let alias = match target.arch() {
        Arch::X86_64 => Some(TARGET_ALIAS_X86),
        Arch::Aarch64 => Some(TARGET_ALIAS_A64),
        _ => None,
    };
    if let Some(alias) = alias {
        for (t, deps) in &m.target_deps {
            if t == alias {
                apply(deps);
            }
        }
    }
    for (t, deps) in &m.target_deps {
        if t == target.name() && Some(t.as_str()) != alias {
            apply(deps);
        }
    }
    out
}

/// Path of `p` for diagnostics: relative to `root` when inside it, else as given.
fn display(root: &Path, p: &Path) -> String {
    let rel = p.strip_prefix(root).unwrap_or(p);
    let parts: Vec<String> = rel
        .components()
        .filter(|c| !matches!(c, Component::CurDir))
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    parts.join("/")
}

fn module_rel(rel_in_dir: &str) -> Vec<String> {
    let stem = rel_in_dir.strip_suffix(".stv").unwrap_or(rel_in_dir);
    stem.split('/').map(str::to_string).collect()
}

/// Loads the package graph from `root` (a package directory or a `.stv` file).
pub(crate) fn load(fs: &Fs, root: &Path, target: Triple, builtins: &[BuiltinPackage]) -> Loaded {
    let mut l = Loaded::default();
    let single = root.extension().is_some_and(|e| e == "stv") && fs.is_file(root);
    let root_dir: PathBuf =
        if single { root.parent().map(Path::to_path_buf).unwrap_or_default() } else { root.into() };
    // specs[p] = the effective dependency specs of package p, consumed once.
    let mut specs: Vec<Vec<DepSpec>> = Vec::new();
    let mut by_dir: BTreeMap<PathBuf, usize> = BTreeMap::new();
    let mut queue: Vec<usize> = Vec::new();
    if single {
        let fname = root.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        l.packages.push(RawPackage {
            name: "main".into(),
            kind: PackageKind::SingleFile,
            version: None,
            dir: root_dir.clone(),
            canonical: fs.canonical(root),
            deps: Vec::new(),
            manifest: None,
            is_root: true,
            root_rel: Some(vec!["main".into()]),
        });
        specs.push(Vec::new());
        l.files.push(PendingFile {
            package: 0,
            rel_path: fname.clone(),
            display_path: fname,
            abs_path: root.to_path_buf(),
            module_rel: vec!["main".into()],
        });
    } else {
        let first = add_package(fs, &mut l, &mut specs, &root_dir, &root_dir, target, true, false);
        by_dir.insert(l.packages[first].canonical.clone(), first);
        queue.push(first);
    }
    // Built-in packages (their manifests, if any, are read like any other).
    let first_builtin = l.packages.len();
    for b in builtins {
        let id = add_package(fs, &mut l, &mut specs, &root_dir, &b.dir, target, false, true);
        l.packages[id].name = b.name.clone();
        by_dir.insert(l.packages[id].canonical.clone(), id);
        queue.push(id);
    }
    for (k, b) in builtins.iter().enumerate() {
        for dn in &b.deps {
            if let Some(j) = builtins.iter().position(|x| &x.name == dn) {
                l.packages[first_builtin + k].deps.push((dn.clone(), first_builtin + j));
            }
        }
    }
    // Path dependencies, transitively. A dependency of a built-in package is built-in too.
    while let Some(p) = queue.pop() {
        let deps = std::mem::take(&mut specs[p]);
        let builtin = l.packages[p].kind == PackageKind::Builtin;
        for d in deps {
            let Some(path) = &d.path else { continue };
            let dir = normalize(&l.packages[p].dir.join(path));
            if !fs.is_file(&dir.join("stave.toml")) {
                // E1002 `missing-dependency`, at the `path` value; imports of it fail (E0101).
                if let Some(m) = l.packages[p].manifest {
                    l.manifests[m].issues.push(ManifestIssue::new(
                        Some("missing-dependency"),
                        &[("key", &d.name)],
                        d.path_span,
                    ));
                }
                continue;
            }
            let canon = fs.canonical(&dir);
            let id = match by_dir.get(&canon) {
                Some(&id) => id,
                None => {
                    let id = add_package(fs, &mut l, &mut specs, &root_dir, &dir, target, false, builtin);
                    by_dir.insert(canon, id);
                    queue.push(id);
                    id
                }
            };
            l.packages[p].deps.push((d.name.clone(), id));
        }
    }
    // Every non-built-in package depends on the implicit built-ins (`std`).
    for p in l.packages.iter_mut().filter(|p| p.kind != PackageKind::Builtin) {
        for (k, b) in builtins.iter().enumerate() {
            if b.implicit && !p.deps.iter().any(|(n, _)| n == &b.name) {
                p.deps.push((b.name.clone(), first_builtin + k));
            }
        }
    }
    for p in &mut l.packages {
        p.deps.sort();
        p.deps.dedup_by(|a, b| a.0 == b.0);
    }
    l
}

/// Adds the package in `dir`: its manifest (required unless `builtin`), the effective
/// dependency specs, and its module files (`src/**` for a manifest package, the directory itself
/// minus nested packages for a built-in one). Returns the package index.
#[allow(clippy::too_many_arguments)]
fn add_package(
    fs: &Fs,
    l: &mut Loaded,
    specs: &mut Vec<Vec<DepSpec>>,
    root_dir: &Path,
    dir: &Path,
    target: Triple,
    is_root: bool,
    builtin: bool,
) -> usize {
    let id = l.packages.len();
    let mpath = dir.join("stave.toml");
    let has_manifest = !builtin || fs.is_file(&mpath);
    let mut data = ManifestData::default();
    let mut manifest = None;
    if has_manifest {
        let (text, issues) = match fs.read_vec(&mpath) {
            Ok(text) => {
                let (d, is) = read_manifest(&text);
                data = d;
                (text, is)
            }
            Err(_) => (Vec::new(), vec![ManifestIssue::new(Some("unreadable"), &[], TSpan { lo: 0, hi: 0 })]),
        };
        manifest = Some(l.manifests.len());
        l.manifests.push(ManifestFile { package: id, display_path: display(root_dir, &mpath), text, issues });
    }
    specs.resize_with(id + 1, Vec::new);
    specs[id] = effective_deps(&data, target);
    let dirname = dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "main".into());
    let name = data.name.clone().unwrap_or(dirname);
    if builtin {
        l.packages.push(RawPackage {
            name,
            kind: PackageKind::Builtin,
            version: data.version.clone(),
            dir: dir.to_path_buf(),
            canonical: fs.canonical(dir),
            deps: Vec::new(),
            manifest,
            is_root: false,
            root_rel: None,
        });
        for rel in fs.builtin_files(dir) {
            let abs = dir.join(&rel);
            l.files.push(PendingFile {
                package: id,
                display_path: display(root_dir, &abs),
                module_rel: module_rel(&rel),
                rel_path: rel,
                abs_path: abs,
            });
        }
        return id;
    }
    let files = fs.stave_files(&dir.join("src"));
    let has = |f: &str| files.iter().any(|x| x == f);
    let kind = data.kind.unwrap_or(if has("lib.stv") && !has("main.stv") {
        PackageKind::Lib
    } else {
        PackageKind::Bin
    });
    let root_file = match kind {
        PackageKind::Lib => "lib",
        _ => "main",
    };
    let unreadable =
        manifest.is_some_and(|m| l.manifests[m].issues.iter().any(|i| i.case == Some("unreadable")));
    if !has(&format!("{root_file}.stv"))
        && !unreadable
        && let Some(m) = manifest
    {
        let span = data.kind_span.unwrap_or(TSpan { lo: 0, hi: 0 });
        let root = format!("src/{root_file}.stv");
        l.manifests[m].issues.push(ManifestIssue::new(Some("missing-root"), &[("root", &root)], span));
    }
    l.packages.push(RawPackage {
        name,
        kind,
        version: data.version.clone(),
        dir: dir.to_path_buf(),
        canonical: fs.canonical(dir),
        deps: Vec::new(),
        manifest,
        is_root,
        root_rel: Some(vec![root_file.to_string()]),
    });
    for rel in files {
        let abs = dir.join("src").join(&rel);
        l.files.push(PendingFile {
            package: id,
            rel_path: format!("src/{rel}"),
            display_path: display(root_dir, &abs),
            module_rel: module_rel(&rel),
            abs_path: abs,
        });
    }
    id
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Issues other than `missing-name` (most samples have no `name`).
    fn issues(text: &str) -> Vec<(Option<&'static str>, String)> {
        let (_, is) = read_manifest(text.as_bytes());
        is.into_iter().filter(|i| i.case != Some("missing-name")).map(|i| (i.case, i.key())).collect()
    }

    #[test]
    fn manifest_issues() {
        assert_eq!(issues("name = \"a\"\nkind = \"bin\"\nversion = \"0.1.0\"\n"), vec![]);
        assert_eq!(
            issues("name = \"a\"\ncolour = \"blue\"\n[dependencies]\nprims = { git = \"x\" }\n"),
            vec![(None, "colour".into()), (Some("non-path-dependency"), "prims".into())]
        );
        assert_eq!(issues("name = 3\n"), vec![(Some("wrong-type"), "name".into())]);
        assert_eq!(issues("[dependencies]\np = \"1.0\"\n"), vec![(Some("non-path-dependency"), "p".into())]);
        assert_eq!(
            issues("[target.foo.dependencies]\n[target.gpu-sm86.dependencies]\nx = { path = \"x\" }\n"),
            vec![(Some("unknown-target"), "foo".into())]
        );
        assert_eq!(issues("[target.x86_64-linux.deps]\n"), vec![(None, "deps".into())]);
        assert_eq!(issues("[run]\nstack = \"8MiB\"\n[gpu]\n[compiler]\n"), vec![]);
        assert_eq!(issues("run = 1\n"), vec![(Some("wrong-type"), "run".into())]);
        assert_eq!(issues("name = \n"), vec![(Some("syntax"), "<end of line>".into())]);
        assert_eq!(issues("name = \"a\"\nname = \"b\"\n"), vec![(Some("duplicate-key"), "name".into())]);
        assert_eq!(issues("name = \"a\"\nkind = \"exe\"\n"), vec![(Some("bad-kind"), "\"exe\"".into())]);
        for bad in ["", ".", "..", "a/b", "a\\\\b", "a\\u0000b", "-o"] {
            let text = format!("name = \"{bad}\"\n");
            assert_eq!(issues(&text), vec![(Some("bad-name"), format!("\"{bad}\""))], "{bad:?}");
        }
        assert_eq!(issues("name = \"my-pkg.2\"\n"), vec![]);
        let (_, is) = read_manifest(b"kind = \"bin\"\n");
        assert_eq!(is.iter().map(|i| i.case).collect::<Vec<_>>(), vec![Some("missing-name")]);
        let (_, is) = read_manifest(b"name = 3\n");
        assert_eq!(is.iter().map(|i| i.case).collect::<Vec<_>>(), vec![Some("wrong-type")]);
    }

    #[test]
    fn per_target_merge() {
        let text = "[dependencies]\nprims = { path = \"cpu\" }\nu = { path = \"u\" }\n\
                    [target.x86_64-linux.dependencies]\nprims = { path = \"x86\" }\n\
                    [target.x86_64-linux-v3.dependencies]\nprims = { path = \"v3\" }\n\
                    [target.gpu-sm86.dependencies]\nprims = { path = \"gpu\" }\n";
        let (m, is) = read_manifest(text.as_bytes());
        assert!(is.iter().all(|i| i.case == Some("missing-name")));
        let get = |t: &str| {
            let d = effective_deps(&m, Triple::parse(t).unwrap());
            d.into_iter().map(|d| (d.name, d.path.unwrap_or_default())).collect::<Vec<_>>()
        };
        assert_eq!(get("x86_64-linux-v2"), vec![("prims".into(), "x86".into()), ("u".into(), "u".into())]);
        assert_eq!(get("x86_64-linux-v3")[0].1, "v3");
        assert_eq!(get("gpu-sm86")[0].1, "gpu");
        assert_eq!(get("aarch64-linux")[0].1, "cpu");
        // `aarch64-linux` applies to every AArch64 target; `+sve` is merged after it.
        let text = "[dependencies]\na = { path = \"base\" }\n[target.aarch64-linux.dependencies]\na = { path = \"a64\" }\n\
                    [target.\"aarch64-linux+sve\".dependencies]\na = { path = \"sve\" }\n";
        let (m, is) = read_manifest(text.as_bytes());
        assert_eq!(is.iter().filter(|i| i.case != Some("missing-name")).count(), 0);
        let get = |t: &str| effective_deps(&m, Triple::parse(t).unwrap())[0].path.clone().unwrap_or_default();
        assert_eq!(get("aarch64-linux"), "a64");
        assert_eq!(get("aarch64-linux+sve"), "sve");
        assert_eq!(get("aarch64-linux+sve2"), "a64");
        assert_eq!(get("x86_64-linux-v3"), "base");
    }

    #[test]
    fn builtin_files_skip_nested_packages() {
        let mut m = BTreeMap::new();
        for p in
            ["/rt/stave.toml", "/rt/os.stv", "/rt/sub/x.stv", "/rt/x86_64/stave.toml", "/rt/x86_64/sys.stv"]
        {
            m.insert(PathBuf::from(p), Vec::new());
        }
        let fs = Fs::Mem(m);
        assert_eq!(fs.builtin_files(Path::new("/rt")), vec!["os.stv".to_string(), "sub/x.stv".to_string()]);
    }
}
