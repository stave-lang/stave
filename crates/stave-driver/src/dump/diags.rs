//! `diags`: one `diagnostics.schema.json` diagnostic line per diagnostic, in canonical order (the
//! same lines `--diagnostics json` prints; never the timings or summary lines).

use std::io::Write;
use std::path::PathBuf;

use super::DumpError;
use crate::pipeline::Output;

pub(super) fn write(out: &Output, w: &mut dyn Write) -> Result<(), DumpError> {
    let text = stave_diag::emit_json(&out.diagnostics, out.files()).map_err(DumpError::Locate)?;
    w.write_all(text.as_bytes()).map_err(|e| DumpError::Io(PathBuf::from("diags.txt"), e))
}
