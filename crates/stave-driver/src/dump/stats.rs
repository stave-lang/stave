//! `stats`: `STAGE ms items rounds fallbacks peak_bytes` per stage that ran, in pipeline order
//! (stage-gates.md; informational, never compared). stage0 has no parallel rounds, fallbacks, or
//! device arenas: those fields are `-`.

use std::io::{self, Write};

use crate::pipeline::Output;

pub(super) fn write(out: &Output, w: &mut dyn Write) -> io::Result<()> {
    let opt = |v: Option<u64>| v.map_or_else(|| "-".to_string(), |n| n.to_string());
    for t in &out.timings {
        writeln!(w, "{} {:.3} {} {} {} -", t.stage.name(), t.ms, t.items, opt(t.rounds), opt(t.fallbacks))?;
    }
    Ok(())
}
