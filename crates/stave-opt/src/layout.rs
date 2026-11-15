//! Block layout for release code generation.
//!
//! - A loop's blocks follow its header, and the blocks after the loop come after all of them:
//!   the order is a reverse postorder whose depth-first search visits the successor of the
//!   lowest loop depth first (the successor visited last lands right after the block). One
//!   iteration then takes one taken jump (the back edge).
//! - Cold blocks go after the hot ones. A block is cold when it cannot return: it ends in
//!   `unreachable` (after a `panic.*` or a call of a `-> never` fn), or every successor is cold.
//!   Hot code then reaches a cold block only by a forward conditional jump that is not taken on
//!   the hot path.

use stave_ir::{Body, Term};

use crate::cfg::Edges;

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
