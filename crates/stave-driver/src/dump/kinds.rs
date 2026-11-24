//! `kinds`: per type of the `sigs` table its kind flags, depth, and layout, then the impl table
//! and the impls that apply to each type (stage-gates.md).
//!
//! The printer is stave-sema's [`stave_sema::dump_kinds`]: it reads the signature table, the
//! canonical type table, and the trait and layout rules, never a body.

use std::io::{self, Write};

use stave_sema::Typeck;

use crate::pipeline::Output;

pub(super) fn write(out: &Output, tc: &Typeck, w: &mut dyn Write) -> io::Result<()> {
    w.write_all(stave_sema::dump_kinds(&out.program, tc).as_bytes())
}
