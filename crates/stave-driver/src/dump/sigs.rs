//! `sigs`: the type table of the types the declarations name, `t<T> key`, then one line per decl
//! with its signature facts, by D (stage-gates.md).
//!
//! The printer is stave-sema's [`stave_sema::dump_sigs`]: it reads the signature table and the
//! canonical type table the type checker owns, never a body.

use std::io::{self, Write};

use stave_sema::Typeck;

use crate::pipeline::Output;

pub(super) fn write(out: &Output, tc: &Typeck, w: &mut dyn Write) -> io::Result<()> {
    w.write_all(stave_sema::dump_sigs(&out.program, tc).as_bytes())
}
