//! `ast`: `I N KIND a b` per node of every item that was not dropped, ordered by (I, N)
//! (stage-gates.md). N is the per-item postorder index; `KIND` the grammar rule name; `a` the
//! file token index of the node's first token (the next token for a node that matched none);
//! `b` the subtree size excluding the node.

use super::push_u32;
use crate::pipeline::Output;

pub(super) fn file(out: &Output, f: usize, buf: &mut Vec<u8>) {
    let Some(file) = out.program.files.get(f) else { return };
    for i in 0..file.parsed.items.len() {
        let item = file.first_item.saturating_add(i as u32);
        for (n, node) in file.parsed.item_ast(i).nodes.iter().enumerate() {
            push_u32(buf, item);
            buf.push(b' ');
            push_u32(buf, n as u32);
            buf.push(b' ');
            buf.extend_from_slice(node.kind().name().as_bytes());
            buf.push(b' ');
            push_u32(buf, node.a);
            buf.push(b' ');
            push_u32(buf, node.b);
            buf.push(b'\n');
        }
    }
}
