//! `ir`: the canonical checked IR of the lowered module (contracts/checked-ir.md, stage-gates.md
//! `ir`): [`stave_ir::print_checked`], the one printer of this dump.
//!
//! Printed only for a build with no error diagnostics: with errors there is no checked IR, this
//! refuses ([`DumpError::NotRun`]), and [`super::write_dumps`] writes no `ir.txt` (an old one is
//! removed). A run that did not lower (a plain `stave check` without `--dump=ir`, i.e. without
//! `Options::lower`) or whose lowering failed has no module and refuses too.

use std::io::Write;
use std::path::PathBuf;

use super::{DumpError, DumpStage};
use crate::pipeline::Output;

pub(super) fn write(out: &Output, w: &mut dyn Write) -> Result<(), DumpError> {
    match &out.module {
        Some(m) if !out.has_errors() => w
            .write_all(stave_ir::print_checked(m).as_bytes())
            .map_err(|e| DumpError::Io(PathBuf::from(DumpStage::Ir.file_name()), e)),
        _ => Err(DumpError::NotRun(DumpStage::Ir)),
    }
}
