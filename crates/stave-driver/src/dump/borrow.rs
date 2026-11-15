//! `borrow`: `I N access-kind verdict loan` per borrow-checker access (stage-gates.md), for every
//! body that was borrow checked (bodies with a type error are skipped, `Typeck::has_error`).
//! N = the place expression (`read write move`) or the local's binding node (`drop
//! storage-dead`); `loan` = the creation node of the reported loan (stage0's choice: the
//! stage-gates format leaves the loan id open), or `-`. Lines sorted by (I, N), kinds in the
//! order `read write move drop storage-dead` per node. Needs [`crate::pipeline::Options::borrow_dump`].

use std::io::Write;
use std::path::PathBuf;

use super::{DumpError, DumpStage};
use crate::pipeline::Output;

pub(super) fn write(out: &Output, w: &mut dyn Write) -> Result<(), DumpError> {
    let Some(bc) = &out.borrowck else { return Err(DumpError::NotRun(DumpStage::Borrow)) };
    let io = |e| DumpError::Io(PathBuf::from(DumpStage::Borrow.file_name()), e);
    let rank = |k: &str| ["read", "write", "move", "drop", "storage-dead"].iter().position(|x| *x == k);
    let mut lines: Vec<_> = bc.dump.iter().collect();
    lines.sort_by_key(|(i, d)| (*i, d.node, rank(d.kind), d.verdict, d.loan));
    for (i, d) in lines {
        let loan = d.loan.map_or_else(|| "-".to_string(), |l| l.to_string());
        writeln!(w, "{i} {} {} {} {loan}", d.node, d.kind, d.verdict.unwrap_or("ok")).map_err(io)?;
    }
    Ok(())
}
