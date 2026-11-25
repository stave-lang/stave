//! `types`: the type table `t<T> key`, then `I N t<T>` / `I N -` per expression node of every
//! item not dropped (stage-gates.md).
//!
//! The printer is stave-sema's [`stave_sema::dump_types`]: stave-sema owns the expression-node set
//! (`typeck::is_expr_kind`), the table numbering, and the checked-IR TypeKey text, so the dump
//! and the type checker cannot disagree.

use std::io::{self, Write};

use stave_sema::Typeck;

use crate::pipeline::Output;

pub(super) fn write(out: &Output, tc: &Typeck, w: &mut dyn Write) -> io::Result<()> {
    w.write_all(stave_sema::dump_types(&out.program, tc).as_bytes())
}
