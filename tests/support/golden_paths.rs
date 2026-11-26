//! Goldens hold no path of the machine that wrote them. A diagnostic's span in a file of the
//! package under test is relative to the package; one in a library file (`std`, a stand-in
//! library) carries the library's absolute path, which the harness rewrites relative to the
//! repository root before it compares or writes a golden. A golden that would still hold an
//! absolute path is refused, also under `STAVE_BLESS`.

use std::path::Path;

/// The repository root as a path prefix with its trailing `/`: as given and canonical.
fn prefixes(repo: &Path) -> Vec<String> {
    let mut v = vec![format!("{}/", repo.display())];
    if let Ok(c) = std::fs::canonicalize(repo) {
        v.push(format!("{}/", c.display()));
    }
    // the longer one first: one form may be a prefix of the other
    v.sort_by_key(|p| std::cmp::Reverse(p.len()));
    v.dedup();
    v
}

/// `text` with every path below the repository root written relative to it.
pub fn neutral(text: &str, repo: &Path) -> String {
    let mut s = text.to_string();
    for p in prefixes(repo) {
        s = s.replace(&p, "");
    }
    s
}

/// Why `text` may not be a golden: it names the repository root, or a diagnostic file by an
/// absolute path.
pub fn host_path(text: &str, repo: &Path) -> Option<String> {
    for p in prefixes(repo) {
        if text.contains(&p) {
            return Some(format!("holds the repository path {p}"));
        }
    }
    text.contains("\"file\":\"/").then(|| "holds an absolute \"file\" path".to_string())
}
