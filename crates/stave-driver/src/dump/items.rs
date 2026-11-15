//! `items`: `I F O_start O_end ok|dropped` per item, ordered by I = rank of (F, start offset)
//! (stage-gates.md): items are numbered file after file (`File::first_item`). `O_end` is the
//! offset just after the item's last token. The run of tokens before the first item start is an
//! item (always `dropped`); a file with E0001, E0002, E0005, or E0016 has no items.

use super::push_u32;
use crate::pipeline::Output;

pub(super) fn file(out: &Output, f: usize, buf: &mut Vec<u8>) {
    let Some(file) = out.program.files.get(f) else { return };
    let src: &[u8] = &file.bytes;
    for (i, it) in file.parsed.items.iter().enumerate() {
        let (start, end) = file.parsed.item_offsets(src, i).unwrap_or_else(|| {
            let o = file.parsed.tokens.get(it.start as usize).map_or(0, |t| t.start);
            (o, o)
        });
        push_u32(buf, file.first_item.saturating_add(i as u32));
        buf.push(b' ');
        push_u32(buf, f as u32);
        buf.push(b' ');
        push_u32(buf, start);
        buf.push(b' ');
        push_u32(buf, end);
        buf.extend_from_slice(if it.ok { b" ok\n" } else { b" dropped\n" });
    }
}
