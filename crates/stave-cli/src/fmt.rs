//! `stave fmt [PKG_DIR | FILE...] [--check]` (contracts/cli.md): rewrites source files in the
//! canonical layout of [`stave_syntax::format_source`].
//!
//! Each argument is a file, or a directory standing for every `.stv` file below it (hidden
//! directories and symbolic links are skipped); no argument means the current directory. Files
//! are handled in sorted order, each on its own:
//!
//! - a file with lexical or syntax diagnostics is left unchanged, its diagnostics go to stderr,
//!   and the exit status is 1;
//! - a file already in the canonical layout is not written;
//! - `--check` writes nothing: it prints the path of each file that would change on stdout, one
//!   per line, and exits 1 if there is any.
//!
//! A path that cannot be read or written is a usage error (exit 2).

use std::io::Write;
use std::path::{Path, PathBuf};

use stave_diag::SourceFiles;
use stave_syntax::FormatError;

use crate::args::Args;
use crate::{EXIT_OK, EXIT_USAGE, EXIT_USER, display, internal_error};

pub(crate) fn run(a: &Args, out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    let roots = if a.paths.is_empty() { vec![PathBuf::from(".")] } else { a.paths.clone() };
    let mut files = Vec::new();
    for root in &roots {
        if let Err(e) = collect(root, &mut files) {
            let _ = writeln!(err, "stave: {}: {e}", display(root));
            return EXIT_USAGE;
        }
    }
    files.sort();
    files.dedup();
    let mut code = EXIT_OK;
    for path in &files {
        let src = match std::fs::read(path) {
            Ok(s) => s,
            Err(e) => {
                let _ = writeln!(err, "stave: {}: {e}", display(path));
                return EXIT_USAGE;
            }
        };
        let name = display(path);
        let mut table = SourceFiles::new();
        let file = match table.add("", &name, &src) {
            Ok(f) => f,
            Err(e) => return internal_error(&e.to_string(), err),
        };
        match stave_syntax::format_source(&src, file) {
            Ok(text) if text.as_bytes() == src => {}
            Ok(_) if a.check => {
                let _ = writeln!(out, "{name}");
                code = EXIT_USER;
            }
            Ok(text) => {
                if let Err(e) = std::fs::write(path, text) {
                    let _ = writeln!(err, "stave: {name}: {e}");
                    return EXIT_USAGE;
                }
            }
            Err(FormatError::Diagnostics(diags)) => match stave_diag::emit_human(&diags, &table) {
                Ok(t) => {
                    let _ = err.write_all(t.as_bytes());
                    code = EXIT_USER;
                }
                Err(e) => return internal_error(&e.to_string(), err),
            },
            Err(FormatError::Internal(what)) => {
                return internal_error(&format!("fmt: {name}: {what}"), err);
            }
        }
    }
    code
}

/// Appends `path` if it is a file, or every `.stv` file below it if it is a directory.
fn collect(path: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if !std::fs::metadata(path)?.is_dir() {
        files.push(path.to_path_buf());
        return Ok(());
    }
    let mut todo = vec![path.to_path_buf()];
    while let Some(dir) = todo.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let (kind, p) = (entry.file_type()?, entry.path());
            if kind.is_dir() {
                if !entry.file_name().to_string_lossy().starts_with('.') {
                    todo.push(p);
                }
            } else if kind.is_file() && p.extension().is_some_and(|x| x == "stv") {
                files.push(p);
            }
        }
    }
    Ok(())
}
