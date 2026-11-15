//! Control-flow facts the passes share: successors of a terminator, loop nesting depth, the
//! block of every instruction.

use stave_ir::ir::Target;
use stave_ir::{Body, Operand, Term};

pub(crate) const NONE: u32 = u32::MAX;

/// Calls `f` on each target of a terminator (switch: cases, then the default).
pub(crate) fn for_targets(b: &Body, t: &Term, mut f: impl FnMut(&Target)) {
    match t {
        Term::Br(t) => f(t),
        Term::Cbr { then, els, .. } => {
            f(then);
            f(els);
        }
        Term::Switch { cases, default, .. } => {
            for c in b.switch_cases(*cases) {
                f(&c.target);
            }
            f(default);
        }
        Term::Ret(_) | Term::Unreachable => {}
    }
}

/// The operand a terminator reads besides its target arguments.
pub(crate) fn term_operand(t: &Term) -> Option<Operand> {
    match t {
        Term::Cbr { cond: o, .. } | Term::Switch { scrut: o, .. } | Term::Ret(Some(o)) => Some(*o),
        _ => None,
    }
}

/// Uses per value, saturating at 255: the operands of every instruction and terminator (the
/// operand pool may hold entries no instruction owns any more, which do not count).
pub(crate) fn use_counts(b: &Body) -> Vec<u8> {
    let mut n = vec![0u8; b.values.len()];
    let mut add = |o: Operand| {
        if let Operand::Value(v) = o
            && let Some(c) = n.get_mut(v.index())
        {
            *c = c.saturating_add(1);
        }
    };
    for bd in &b.blocks {
        for i in bd.insts.indices() {
            b.args(i).iter().for_each(|o| add(*o));
        }
        if let Some(o) = term_operand(&bd.term) {
            add(o);
        }
        for_targets(b, &bd.term, |t| b.target_args(t).iter().for_each(|o| add(*o)));
    }
    n
}

/// Successor and predecessor lists over the blocks reachable from the entry.
pub(crate) struct Edges {
    pub succ: Vec<Vec<u32>>,
    pub preds: Vec<Vec<u32>>,
    pub reachable: Vec<bool>,
}

impl Edges {
    pub(crate) fn new(b: &Body) -> Edges {
        let nb = b.blocks.len();
        let reachable = b.reachable();
        let mut succ: Vec<Vec<u32>> = vec![Vec::new(); nb];
        let mut preds: Vec<Vec<u32>> = vec![Vec::new(); nb];
        for (bi, bd) in b.blocks.iter().enumerate() {
            if !reachable[bi] {
                continue;
            }
            for_targets(b, &bd.term, |t| {
                if t.block.index() < nb {
                    succ[bi].push(t.block.0);
                    preds[t.block.index()].push(bi as u32);
                }
            });
        }
        Edges { succ, preds, reachable }
    }

    /// Loop nesting depth per block (natural loops of the DFS back edges; 0 outside loops).
    pub(crate) fn loop_depths(&self) -> Vec<u8> {
        let nb = self.succ.len();
        let mut depth = vec![0u8; nb];
        if nb == 0 {
            return depth;
        }
        let mut state = vec![0u8; nb];
        let mut back: Vec<(u32, u32)> = Vec::new();
        let mut stack: Vec<(u32, usize)> = vec![(0, 0)];
        state[0] = 1;
        while let Some(top) = stack.last_mut() {
            let node = top.0;
            if let Some(&s) = self.succ[node as usize].get(top.1) {
                top.1 += 1;
                match state[s as usize] {
                    0 => {
                        state[s as usize] = 1;
                        stack.push((s, 0));
                    }
                    1 => back.push((node, s)),
                    _ => {}
                }
            } else {
                state[node as usize] = 2;
                stack.pop();
            }
        }
        let mut mark = vec![NONE; nb];
        let mut work = Vec::new();
        for (li, &(latch, header)) in back.iter().enumerate() {
            let li = li as u32;
            mark[header as usize] = li;
            depth[header as usize] = depth[header as usize].saturating_add(1);
            work.clear();
            work.push(latch);
            while let Some(x) = work.pop() {
                if mark[x as usize] == li {
                    continue;
                }
                mark[x as usize] = li;
                depth[x as usize] = depth[x as usize].saturating_add(1);
                work.extend(self.preds[x as usize].iter().copied());
            }
        }
        depth
    }
}

/// The block of every instruction (`NONE` for one outside every block's span).
pub(crate) fn block_of_insts(b: &Body) -> Vec<u32> {
    let mut of = vec![NONE; b.insts.len()];
    for (bi, bd) in b.blocks.iter().enumerate() {
        for i in bd.insts.indices() {
            if let Some(s) = of.get_mut(i as usize) {
                *s = bi as u32;
            }
        }
    }
    of
}
