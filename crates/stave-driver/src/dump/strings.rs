//! `strings`: `S hash32 "escaped-bytes"` per interned string, ordered by S (stage-gates.md):
//! [`stave_sema::strings_into`], the one printer of this dump.

use std::io::{self, Write};

use crate::pipeline::Output;

pub(super) fn write(out: &Output, w: &mut dyn Write) -> io::Result<()> {
    let mut buf = Vec::with_capacity(1 << 16);
    stave_sema::strings_into(&out.program, &mut buf);
    w.write_all(&buf)
}
