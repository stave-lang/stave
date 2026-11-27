//! `mir`: the borrow CFG of every borrow-checkable body, one section per body by (D, k)
//! (stage-gates.md). The printer is stave-borrow's [`stave_borrow::dump`]: each body's section is
//! printed by the worker that built it, while the body's type overlay lives. Needs
//! [`crate::pipeline::Options::mir_dump`].

use std::io::Write;
use std::path::PathBuf;

use super::{DumpError, DumpStage};
use crate::pipeline::Output;

pub(super) fn write(out: &Output, w: &mut dyn Write) -> Result<(), DumpError> {
    let Some(mir) = out.borrowck.as_ref().and_then(|bc| bc.mir.as_ref()) else {
        return Err(DumpError::NotRun(DumpStage::Mir));
    };
    let io = |e| DumpError::Io(PathBuf::from(DumpStage::Mir.file_name()), e);
    for (_, s) in mir {
        w.write_all(s.as_bytes()).map_err(io)?;
    }
    Ok(())
}
