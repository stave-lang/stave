//! The build root argument (`stave check [PKG_DIR]`): a package directory (with `stave.toml`) or one
//! `.stv` file.
//!
//! Packages themselves are loaded by stave-sema ([`stave_sema::Program::load`]: manifests with E1002,
//! path and per-target dependencies, the built-in packages, module discovery). This module only
//! rejects roots that are not a package at all, before anything runs: those are usage errors
//! (exit 2), not diagnostics.

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum RootError {
    /// The path does not exist or cannot be inspected.
    NotFound(PathBuf, io::Error),
    /// A directory without `stave.toml`.
    NoManifest(PathBuf),
    /// A file argument that is not `.stv`.
    NotStave(PathBuf),
}

impl fmt::Display for RootError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RootError::NotFound(p, e) => write!(f, "{}: {e}", p.display()),
            RootError::NoManifest(p) => {
                write!(f, "{}: not a package (no stave.toml) and not a .stv file", p.display())
            }
            RootError::NotStave(p) => write!(f, "{}: not a .stv file or package directory", p.display()),
        }
    }
}

impl std::error::Error for RootError {}

/// Checks that `path` is a package directory (with `stave.toml`) or a `.stv` file.
pub fn check_root(path: &Path) -> Result<(), RootError> {
    let meta = fs::metadata(path).map_err(|e| RootError::NotFound(path.to_path_buf(), e))?;
    if meta.is_file() {
        if path.extension().is_none_or(|e| e != "stv") {
            return Err(RootError::NotStave(path.to_path_buf()));
        }
        return Ok(());
    }
    if !path.join("stave.toml").is_file() {
        return Err(RootError::NoManifest(path.to_path_buf()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roots() {
        let here = Path::new(env!("CARGO_MANIFEST_DIR"));
        assert!(check_root(&here.join("../../tests/conformance/dumps/valid_small/input")).is_ok());
        assert!(check_root(&here.join("../../tests/conformance/syntax/lex_empty.stv")).is_ok());
        let e = check_root(here).unwrap_err();
        assert!(e.to_string().contains("not a package (no stave.toml)"), "{e}");
        assert!(matches!(check_root(&here.join("Cargo.toml")), Err(RootError::NotStave(_))));
        assert!(matches!(check_root(Path::new("/nonexistent/x")), Err(RootError::NotFound(..))));
    }
}
