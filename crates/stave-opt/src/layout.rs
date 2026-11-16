//! Block layout for release code generation.
//!
//! - A loop's blocks follow its header, and the blocks after the loop come after all of them:
//!   the order is a reverse postorder whose depth-first search visits the successor of the
//!   lowest loop depth first (the successor visited last lands right after the block). One
//!   iteration then takes one taken jump (the back edge).
//! - [`rotated_order`] also puts a loop's test after its body (one conditional jump per
//!   iteration), for a backend that can emit a block before the block that dominates it.
//! - Cold blocks go after the hot ones. A block is cold when it cannot return: it ends in
//!   `unreachable` (after a `panic.*` or a call of a `-> never` fn), or every successor is cold.
//!   Hot code then reaches a cold block only by a forward conditional jump that is not taken on
//!   the hot path.

use stave_ir::{Block, Body, Operand, Term, Value};

use crate::cfg::{Edges, for_targets, term_operand};

/// The emission order of a release build: every reachable block, the entry first, a definition
/// before the blocks it dominates.
pub fn order(b: &Body) -> Vec<u32> {
    let nb = b.blocks.len();
    if nb == 0 {
        return Vec::new();
    }
    let mut edges = Edges::new(b);
    let depth = edges.loop_depths();
    for s in &mut edges.succ {
        // stable: equal depths keep the terminator's order
        s.sort_by_key(|t| depth[*t as usize]);
    }
    let mut post: Vec<u32> = Vec::with_capacity(nb);
    let mut seen = vec![false; nb];
    let mut stack: Vec<(u32, usize)> = vec![(0, 0)];
    seen[0] = true;
    while let Some(top) = stack.last_mut() {
        let node = top.0;
        if let Some(&s) = edges.succ[node as usize].get(top.1) {
            top.1 += 1;
            if !seen[s as usize] {
                seen[s as usize] = true;
                stack.push((s, 0));
            }
        } else {
            post.push(node);
            stack.pop();
        }
    }
    post.reverse();
    cold_last(b, &post)
}

/// [`order`] with every loop whose header tests (a `cbr` with one target in the loop and one out
/// of it) rotated: the header moves after the loop's last block when that block branches back to
/// it. The last block then falls into the test and the test branches back to the loop's first
/// block, so an iteration takes one conditional jump; the loop is entered by a jump to the test.
///
/// The header now comes after blocks it dominates, so its values are used before they are
/// defined in emission order: a backend must keep each such value in a register for its whole
/// life (or emit with [`order`]). A loop is left as it is when one of them fails `in_register`.
pub fn rotated_order(b: &Body, in_register: impl Fn(Value) -> bool) -> Vec<u32> {
    let mut order = order(b);
    let loops = Edges::new(b).natural_loops();
    if loops.is_empty() {
        return order;
    }
    let mut pos = vec![u32::MAX; b.blocks.len()];
    for (k, &x) in order.iter().enumerate() {
        pos[x as usize] = k as u32;
    }
    // `stamp[x] == k + 1`: block `x` is in loop `k`
    let mut stamp = vec![0u32; b.blocks.len()];
    let mut outside: Option<Vec<bool>> = None;
    for (k, (header, blocks)) in loops.iter().enumerate() {
        let id = k as u32 + 1;
        for &x in blocks {
            stamp[x as usize] = id;
        }
        let member = |x: usize| stamp[x] == id;
        let h = *header as usize;
        if h == 0 {
            continue;
        }
        let Term::Cbr { then, els, .. } = b.blocks[h].term else { continue };
        let (t, e) = (then.block.index(), els.block.index());
        if t == h || e == h || member(t) == member(e) {
            continue;
        }
        let p = pos[h] as usize;
        if p >= order.len() {
            continue;
        }
        let mut end = p + 1;
        while end < order.len() && member(order[end] as usize) {
            end += 1;
        }
        let last = order[end - 1] as usize;
        if end == p + 1 || !matches!(b.blocks[last].term, Term::Br(tg) if tg.block.index() == h) {
            continue;
        }
        // At the bottom the header's params are live, for the register assignment, over the
        // whole loop: a join inside it (an `if` that updates a loop variable) could no longer
        // share their registers and would cost a copy per iteration.
        if order[p + 1..end].iter().any(|&x| !b.params(Block(x)).is_empty()) {
            continue;
        }
        let outside = outside.get_or_insert_with(|| used_outside_def(b));
        let bd = &b.blocks[h];
        let defined = b
            .params(Block(*header))
            .iter()
            .copied()
            .chain(bd.insts.indices().filter_map(|i| b.insts.result[i as usize]));
        if defined.filter(|v| outside[v.index()]).any(|v| !in_register(v)) {
            continue;
        }
        order[p..end].rotate_left(1);
        for (q, &x) in order[p..end].iter().enumerate() {
            pos[x as usize] = (p + q) as u32;
        }
    }
    order
}

/// Per value: read in a block other than the one defining it.
fn used_outside_def(b: &Body) -> Vec<bool> {
    let nv = b.values.len();
    let mut def = vec![u32::MAX; nv];
    for (bi, bd) in b.blocks.iter().enumerate() {
        for &v in b.params(Block(bi as u32)) {
            def[v.index()] = bi as u32;
        }
        for i in bd.insts.indices() {
            if let Some(v) = b.insts.result[i as usize] {
                def[v.index()] = bi as u32;
            }
        }
    }
    let mut out = vec![false; nv];
    for (bi, bd) in b.blocks.iter().enumerate() {
        let mut note = |o: Operand| {
            if let Operand::Value(v) = o
                && def.get(v.index()).is_some_and(|&d| d != bi as u32)
            {
                out[v.index()] = true;
            }
        };
        for i in bd.insts.indices() {
            b.args(i).iter().for_each(|&o| note(o));
        }
        if let Some(o) = term_operand(&bd.term) {
            note(o);
        }
        for_targets(b, &bd.term, |t| b.target_args(t).iter().for_each(|&o| note(o)));
    }
    out
}

/// Per block: cold. `rpo`: the reachable blocks in reverse postorder.
pub fn cold_blocks(b: &Body, rpo: &[u32]) -> Vec<bool> {
    let mut cold = vec![false; b.blocks.len()];
    // Successors come first in reverse RPO, except along back edges: repeat until stable.
    loop {
        let mut changed = false;
        for &bi in rpo.iter().rev() {
            let Some(bd) = b.blocks.get(bi as usize) else { continue };
            if cold[bi as usize] {
                continue;
            }
            let all_cold = |ts: &mut dyn Iterator<Item = u32>| {
                let mut any = false;
                for t in ts {
                    any = true;
                    if !cold.get(t as usize).copied().unwrap_or(false) {
                        return false;
                    }
                }
                any
            };
            let c = match bd.term {
                Term::Unreachable => true,
                Term::Ret(_) => false,
                Term::Br(t) => all_cold(&mut [t.block.0].into_iter()),
                Term::Cbr { then, els, .. } => all_cold(&mut [then.block.0, els.block.0].into_iter()),
                Term::Switch { cases, default, .. } => all_cold(
                    &mut b.switch_cases(cases).iter().map(|c| c.target.block.0).chain([default.block.0]),
                ),
            };
            if c {
                cold[bi as usize] = true;
                changed = true;
            }
        }
        if !changed {
            return cold;
        }
    }
}

/// The emission order: `rpo` with the cold blocks moved to the end (both parts keep their
/// order; the entry stays first).
pub fn cold_last(b: &Body, rpo: &[u32]) -> Vec<u32> {
    let cold = cold_blocks(b, rpo);
    let is_cold = |bi: &u32| *bi != 0 && cold.get(*bi as usize).copied().unwrap_or(false);
    if !rpo.iter().any(is_cold) {
        return rpo.to_vec();
    }
    let mut order: Vec<u32> = rpo.iter().copied().filter(|bi| !is_cold(bi)).collect();
    order.extend(rpo.iter().copied().filter(is_cold));
    order
}
