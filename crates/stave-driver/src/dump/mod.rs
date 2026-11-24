//! Stage dumps `--dump=STAGE,... --dump-dir DIR`: one file
//! `DIR/STAGE.txt` per stage, in the formats, record orders, and field rules of
//! `contracts/stage-gates.md`. Line-oriented, `\n`-terminated, no
//! trailing spaces; nothing depends on the thread count (files are rendered in parallel and
//! written in file order).
//!
//! Every dump covers the whole program: the root package, its dependencies, and the built-in
//! packages (`F` = rank of (package, path), `I` numbered file after file, `S`/`D`/`T` the
//! canonical stave-sema ids).
//!
//! Implemented: `tokens items ast strings names sigs types borrow ir diags stats` (`sigs` is
//! [`stave_sema::dump_sigs`], the printer of the decl table and signature types; `types` is
//! [`stave_sema::dump_types`], the one printer of the expression-node set and TypeKey text; `ir` is
//! [`stave_ir::print_checked`] of the lowered module (module `ir`), and no `ir.txt` exists when the
//! build has error diagnostics). `borrow` is stave-borrow's per-access lines (module `borrow`).

mod ast;
mod borrow;
mod diags;
mod ir;
mod items;
mod names;
mod sigs;
mod stats;
mod strings;
mod tokens;
mod types;

use std::fmt;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::pipeline::Output;

/// A dump stage, in contract order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DumpStage {
    Tokens,
    Items,
    Ast,
    Strings,
    Names,
    Sigs,
    Types,
    Borrow,
    Ir,
    Diags,
    Stats,
}

impl DumpStage {
    pub const ALL: [DumpStage; 11] = [
        DumpStage::Tokens,
        DumpStage::Items,
        DumpStage::Ast,
        DumpStage::Strings,
        DumpStage::Names,
        DumpStage::Sigs,
        DumpStage::Types,
        DumpStage::Borrow,
        DumpStage::Ir,
        DumpStage::Diags,
        DumpStage::Stats,
    ];

    pub fn name(self) -> &'static str {
        match self {
            DumpStage::Tokens => "tokens",
            DumpStage::Items => "items",
            DumpStage::Ast => "ast",
            DumpStage::Strings => "strings",
            DumpStage::Names => "names",
            DumpStage::Sigs => "sigs",
            DumpStage::Types => "types",
            DumpStage::Borrow => "borrow",
            DumpStage::Ir => "ir",
            DumpStage::Diags => "diags",
            DumpStage::Stats => "stats",
        }
    }

    pub fn parse(s: &str) -> Option<DumpStage> {
        DumpStage::ALL.into_iter().find(|d| d.name() == s)
    }

    /// The file name in the dump directory.
    pub fn file_name(self) -> String {
        format!("{}.txt", self.name())
    }
}

#[derive(Debug)]
pub enum DumpError {
    Io(PathBuf, io::Error),
    /// The run stopped before the stage this dump prints (e.g. `types` of a parse-only run).
    NotRun(DumpStage),
    /// A diagnostic span outside its file (a compiler bug).
    Locate(stave_diag::LocateError),
}

impl fmt::Display for DumpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DumpError::Io(p, e) => write!(f, "cannot write {}: {e}", p.display()),
            DumpError::NotRun(s) => write!(f, "dump `{}`: the run stopped before its stage", s.name()),
            DumpError::Locate(e) => write!(f, "internal: {e}"),
        }
    }
}

impl std::error::Error for DumpError {}

/// Renders one dump into `w`.
pub fn render(stage: DumpStage, out: &Output, w: &mut dyn Write) -> Result<(), DumpError> {
    let io = |e| DumpError::Io(PathBuf::from(stage.file_name()), e);
    match stage {
        DumpStage::Tokens => per_file(out, w, tokens::file).map_err(io),
        DumpStage::Items => per_file(out, w, items::file).map_err(io),
        DumpStage::Ast => per_file(out, w, ast::file).map_err(io),
        DumpStage::Strings => strings::write(out, w).map_err(io),
        DumpStage::Names => per_file(out, w, names::file).map_err(io),
        DumpStage::Sigs => match &out.typeck {
            Some(tc) => sigs::write(out, tc, w).map_err(io),
            None => Err(DumpError::NotRun(stage)),
        },
        DumpStage::Types => match &out.typeck {
            Some(tc) => types::write(out, tc, w).map_err(io),
            None => Err(DumpError::NotRun(stage)),
        },
        DumpStage::Diags => diags::write(out, w),
        DumpStage::Stats => stats::write(out, w).map_err(io),
        DumpStage::Borrow => borrow::write(out, w),
        // Callers skip `ir` when there are error diagnostics ([`write_dumps`]).
        DumpStage::Ir => ir::write(out, w),
    }
}

/// Renders one dump into a byte vector.
pub fn render_to_vec(stage: DumpStage, out: &Output) -> Result<Vec<u8>, DumpError> {
    let mut v = Vec::new();
    render(stage, out, &mut v)?;
    Ok(v)
}

/// Writes `DIR/STAGE.txt` for every stage (the directory is created). `ir.txt` is not written
/// (an old one is removed) when the build has error diagnostics (stage-gates.md).
pub fn write_dumps(out: &Output, stages: &[DumpStage], dir: &Path) -> Result<(), DumpError> {
    std::fs::create_dir_all(dir).map_err(|e| DumpError::Io(dir.to_path_buf(), e))?;
    for &s in stages {
        let path = dir.join(s.file_name());
        if s == DumpStage::Ir && out.has_errors() {
            match std::fs::remove_file(&path) {
                Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(DumpError::Io(path, e)),
                _ => continue,
            }
        }
        let f = std::fs::File::create(&path).map_err(|e| DumpError::Io(path.clone(), e))?;
        let mut w = io::BufWriter::with_capacity(1 << 20, f);
        render(s, out, &mut w).map_err(|e| match e {
            DumpError::Io(_, e) => DumpError::Io(path.clone(), e),
            other => other,
        })?;
        w.flush().map_err(|e| DumpError::Io(path.clone(), e))?;
    }
    Ok(())
}

/// Files rendered per parallel batch (bounds memory on large corpora).
const BATCH: usize = 64;

/// Renders every file with `f` in parallel batches and writes them in file order.
fn per_file<F>(out: &Output, w: &mut dyn Write, f: F) -> io::Result<()>
where
    F: Fn(&Output, usize, &mut Vec<u8>) + Sync,
{
    let n = out.program.files.len();
    let mut lo = 0;
    while lo < n {
        let hi = (lo + BATCH).min(n);
        let bufs: Vec<Vec<u8>> = (lo..hi)
            .into_par_iter()
            .map(|k| {
                let mut b = Vec::new();
                f(out, k, &mut b);
                b
            })
            .collect();
        for b in bufs {
            w.write_all(&b)?;
        }
        lo = hi;
    }
    Ok(())
}

/// Appends the decimal form of `n`.
fn push_u32(buf: &mut Vec<u8>, n: u32) {
    let mut tmp = [0u8; 10];
    let mut i = tmp.len();
    let mut v = n;
    loop {
        i -= 1;
        tmp[i] = b'0' + (v % 10) as u8;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    buf.extend_from_slice(&tmp[i..]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimal() {
        let mut b = Vec::new();
        for n in [0, 7, 10, 4_294_967_295] {
            push_u32(&mut b, n);
            b.push(b' ');
        }
        assert_eq!(b, b"0 7 10 4294967295 ");
    }

    #[test]
    fn names() {
        for s in DumpStage::ALL {
            assert_eq!(DumpStage::parse(s.name()), Some(s));
        }
        let all: Vec<&str> = DumpStage::ALL.iter().map(|s| s.name()).collect();
        assert_eq!(
            all,
            ["tokens", "items", "ast", "strings", "names", "sigs", "types", "borrow", "ir", "diags", "stats"]
        );
    }
}
