//! The shipped built-in packages `std` and `runtime` (contracts/cli.md "Shipped library").
//!
//! A *library directory* `D` holds `D/std/` (with `core.stv`, the prelude module) and
//! `D/runtime/`. [`locate`] picks the first that exists, in this order:
//!
//! 1. `$STAVE_STD_DIR`, when set and not empty: it must be a library directory (otherwise
//!    [`LocateError::BadEnv`]; no fallback, so a typo is never silently ignored);
//! 2. the install layout `<exe dir>/../lib/stave/`;
//! 3. the development layout: the nearest ancestor of the `stave` executable's directory that holds a
//!    `Cargo.toml` and is a library directory (the repository root, for `target/<profile>/stave`
//!    and for test binaries in `target/<profile>/deps/`).
//!
//! Nothing found is [`LocateError::NotFound`] (the CLI reports it as a usage error, exit 2: the
//! installation cannot serve the request; not a compiler bug).

use std::ffi::OsStr;
use std::fmt;
use std::path::{Path, PathBuf};

use stave_sema::Config;

/// Environment variable overriding the library directory.
pub const STD_DIR_ENV: &str = "STAVE_STD_DIR";

/// The built-in packages a program is built with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Library {
    /// The shipped `std` (implicit dependency of every package; depends on `runtime`) and
    /// `runtime` (its `stave.toml` binds the per-arch package).
    Shipped { std: PathBuf, runtime: PathBuf },
    /// No built-in packages.
    None,
}

impl Library {
    /// The shipped packages of library directory `dir` (not checked).
    pub fn in_dir(dir: &Path) -> Library {
        Library::Shipped { std: dir.join("std"), runtime: dir.join("runtime") }
    }

    /// A stave-sema [`Config`] building `root` with these packages.
    pub fn config(&self, root: &Path) -> Config {
        let c = Config::new(root);
        match self {
            Library::Shipped { std, runtime } => c.with_shipped(std, runtime),
            Library::None => c,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LocateError {
    /// `$STAVE_STD_DIR` is set but is not a library directory.
    BadEnv(PathBuf),
    /// No candidate is a library directory (the candidates tried, in order).
    NotFound(Vec<String>),
}

impl fmt::Display for LocateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LocateError::BadEnv(p) => write!(
                f,
                "{STD_DIR_ENV}={}: not a Stave library directory (needs std/core.stv and runtime/)",
                p.display()
            ),
            LocateError::NotFound(tried) => {
                write!(
                    f,
                    "cannot find the shipped std and runtime (set {STD_DIR_ENV}=DIR with DIR/std and DIR/runtime)"
                )?;
                if !tried.is_empty() {
                    write!(f, "; tried: {}", tried.join(", "))?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for LocateError {}

/// Whether `dir` is a library directory.
pub fn is_library_dir(dir: &Path) -> bool {
    dir.join("std").join("core.stv").is_file() && dir.join("runtime").is_dir()
}

/// The library directory for `env` (the value of `$STAVE_STD_DIR`) and `exe` (the `stave`
/// executable), per the module docs.
pub fn locate_dir(env: Option<&OsStr>, exe: Option<&Path>) -> Result<PathBuf, LocateError> {
    if let Some(v) = env.filter(|v| !v.is_empty()) {
        let d = PathBuf::from(v);
        return if is_library_dir(&d) { Ok(d) } else { Err(LocateError::BadEnv(d)) };
    }
    let mut tried = Vec::new();
    let Some(exe_dir) = exe.and_then(Path::parent) else { return Err(LocateError::NotFound(tried)) };
    if let Some(prefix) = exe_dir.parent() {
        let d = prefix.join("lib").join("stave");
        if is_library_dir(&d) {
            return Ok(d);
        }
        tried.push(d.display().to_string());
    }
    for a in exe_dir.ancestors() {
        if a.join("Cargo.toml").is_file() && is_library_dir(a) {
            return Ok(a.to_path_buf());
        }
    }
    tried.push(format!("a repository root above {}", exe_dir.display()));
    Err(LocateError::NotFound(tried))
}

/// [`locate_dir`] for this process (`$STAVE_STD_DIR`, `std::env::current_exe`).
pub fn locate() -> Result<Library, LocateError> {
    let env = std::env::var_os(STD_DIR_ENV);
    let exe = std::env::current_exe().ok();
    let exe = exe.map(|p| p.canonicalize().unwrap_or(p));
    locate_dir(env.as_deref(), exe.as_deref()).map(|d| Library::in_dir(&d))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    #[test]
    fn lookup_order() {
        let repo = repo();
        assert!(is_library_dir(&repo), "the repository root is a library directory");
        // Env override wins, and must be valid.
        assert_eq!(locate_dir(Some(repo.as_os_str()), None), Ok(repo.clone()));
        let bad = repo.join("crates");
        assert_eq!(locate_dir(Some(bad.as_os_str()), None), Err(LocateError::BadEnv(bad)));
        // Empty env = unset.
        let exe = repo.join("target/release/stave");
        assert_eq!(locate_dir(Some(OsStr::new("")), Some(&exe)), Ok(repo.clone()));
        // Development layout: an ancestor with Cargo.toml.
        assert_eq!(locate_dir(None, Some(&repo.join("target/debug/deps/stave-0123"))), Ok(repo.clone()));
        // Nothing.
        let e = locate_dir(None, Some(Path::new("/nonexistent/bin/stave"))).unwrap_err();
        assert!(e.to_string().contains("/nonexistent/lib/stave"), "{e}");
        assert!(matches!(locate_dir(None, None), Err(LocateError::NotFound(_))));
    }

    #[test]
    fn install_layout() {
        let tmp = std::env::temp_dir().join(format!("stave-shipped-{}", std::process::id()));
        let lib = tmp.join("lib/stave");
        std::fs::create_dir_all(lib.join("std")).unwrap();
        std::fs::create_dir_all(lib.join("runtime")).unwrap();
        std::fs::write(lib.join("std/core.stv"), "").unwrap();
        assert_eq!(locate_dir(None, Some(&tmp.join("bin/stave"))), Ok(lib.clone()));
        assert_eq!(
            Library::in_dir(&lib),
            Library::Shipped { std: lib.join("std"), runtime: lib.join("runtime") }
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
