//! `names`: `I N -> decl D` / `I N -> local I N2` per resolving Ident use, ordered by (I, N) and
//! then by the identifier's byte offset (stage-gates.md field rules): one file at a time through
//! [`stave_sema::names_file_into`], the one printer of this dump.

use crate::pipeline::Output;

pub(super) fn file(out: &Output, f: usize, buf: &mut Vec<u8>) {
    stave_sema::names_file_into(&out.program, f, buf);
}
