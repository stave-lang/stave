//! One rule for every golden below `tests/`, whatever reads it (the harnesses, the CLI tests,
//! the gates): it holds no path of the machine that wrote it. A diagnostic's span in a file of
//! the package under test is relative to the package; one in a library file (`std`, a stand-in
//! library) carries the library's absolute path. So no golden case has a diagnostic pointing
//! into a library file: such a case is checked without a golden. A harness refuses to compare
//! or write (also under `STAVE_BLESS`) a golden that names a host path, and a test scans every
//! file below `tests/`.

use std::path::Path;

/// The repository root as a path prefix with its trailing `/`: as given and canonical.
fn prefixes(repo: &Path) -> Vec<String> {
    let mut v = vec![format!("{}/", repo.display())];
    if let Ok(c) = std::fs::canonicalize(repo) {
        v.push(format!("{}/", c.display()));
    }
    v.dedup();
    v
}

/// Why `text` may not be a golden: it names the repository root, or a diagnostic file by an
/// absolute path (a span in a library file).
pub fn host_path(text: &str, repo: &Path) -> Option<String> {
    for p in prefixes(repo) {
        if text.contains(&p) {
            return Some(format!("holds the host path {p}: no golden may point into a library file"));
        }
    }
    text.contains("\"file\":\"/")
        .then(|| "holds an absolute \"file\" path: no golden may point into a library file".to_string())
}
