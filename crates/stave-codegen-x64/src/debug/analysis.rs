//! The one analysis pass of the debug backend: reverse postorder of the
//! reachable blocks, loop nesting depth, and value liveness at the granularity the greedy local
//! allocator needs: which values are used outside their defining block (they live in their
//! frame home across block boundaries) and which values are used at all. Per-block last uses
//! are computed by the code generator when it enters a block.

use stave_ir::ir::{Target, ValueDef};
use stave_ir::{Body, Operand, Term};

pub const NONE: u32 = u32::MAX;

pub struct Analysis {
    /// Reachable blocks in reverse postorder (`b0` first).
    pub rpo: Vec<u32>,
    /// Loop nesting depth per block (0 = not in a loop; unreachable blocks 0).
    pub loop_depth: Vec<u16>,
    /// Defining block per value (`NONE` if it has no definition in a reachable block).
    pub def_block: Vec<u32>,
    /// Used in a reachable block other than its defining block.
    pub escapes: Vec<bool>,
    /// Used anywhere (reachable).
    pub used: Vec<bool>,
}

/// Calls `f` on each target of a terminator.
pub fn for_targets(b: &Body, t: &Term, mut f: impl FnMut(&Target)) {
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

/// Calls `f` on every operand of a terminator (condition / scrutinee / return value, then every
/// target argument).
pub fn for_term_operands(b: &Body, t: &Term, mut f: impl FnMut(Operand)) {
    match t {
        Term::Cbr { cond, .. } => f(*cond),
        Term::Switch { scrut, .. } => f(*scrut),
        Term::Ret(Some(o)) => f(*o),
        _ => {}
    }
    for_targets(b, t, |tg| {
        for &o in b.target_args(tg) {
            f(o);
        }
    });
}

pub fn analyze(b: &Body) -> Analysis {
    let nb = b.blocks.len();
    let nv = b.values.len();
    // Successor lists (block indices in range only), flat: `succ[start[i]..start[i + 1]]`.
    let mut start: Vec<u32> = Vec::with_capacity(nb + 1);
    let mut flat: Vec<u32> = Vec::with_capacity(nb * 2);
    for bd in &b.blocks {
        start.push(flat.len() as u32);
        for_targets(b, &bd.term, |t| {
            if t.block.index() < nb {
                flat.push(t.block.0);
            }
        });
    }
    start.push(flat.len() as u32);
    let succ_of = |i: usize| &flat[start[i] as usize..start[i + 1] as usize];
    // Iterative DFS: postorder and back edges (target on the DFS stack).
    let mut state = vec![0u8; nb]; // 0 new, 1 on stack, 2 done
    let mut post = Vec::with_capacity(nb);
    let mut back: Vec<(u32, u32)> = Vec::new();
    if nb > 0 {
        let mut stack: Vec<(u32, usize)> = vec![(0, 0)];
        state[0] = 1;
        while let Some(top) = stack.last_mut() {
            let node = top.0;
            if let Some(&s) = succ_of(node as usize).get(top.1) {
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
                post.push(node);
                stack.pop();
            }
        }
    }
    let rpo: Vec<u32> = post.iter().rev().copied().collect();
    let reachable: Vec<bool> = state.iter().map(|&s| s == 2).collect();

    // Natural loops of the back edges: the header plus every block that reaches the latch
    // without passing through the header.
    let mut loop_depth = vec![0u16; nb];
    if !back.is_empty() {
        let mut preds: Vec<Vec<u32>> = vec![Vec::new(); nb];
        for (i, _) in reachable.iter().enumerate().filter(|x| *x.1) {
            for &s in succ_of(i) {
                preds[s as usize].push(i as u32);
            }
        }
        let mut mark = vec![u32::MAX; nb];
        for (li, &(latch, header)) in back.iter().enumerate() {
            let li = li as u32;
            mark[header as usize] = li;
            loop_depth[header as usize] = loop_depth[header as usize].saturating_add(1);
            let mut work = vec![latch];
            while let Some(x) = work.pop() {
                if mark[x as usize] == li {
                    continue;
                }
                mark[x as usize] = li;
                loop_depth[x as usize] = loop_depth[x as usize].saturating_add(1);
                work.extend(preds[x as usize].iter().copied());
            }
        }
    }

    // Definitions.
    let mut def_block = vec![NONE; nv];
    for (bi, bd) in b.blocks.iter().enumerate() {
        if !reachable[bi] {
            continue;
        }
        for &v in bd.params.of(&b.block_params) {
            if let Some(d) = def_block.get_mut(v.index()) {
                *d = bi as u32;
            }
        }
        for i in bd.insts.indices() {
            if let Some(Some(v)) = b.insts.result.get(i as usize)
                && let Some(d) = def_block.get_mut(v.index())
            {
                *d = bi as u32;
            }
        }
    }
    // b0 params are defined by the entry even if the verifier recorded them oddly.
    for (v, vd) in b.values.iter().enumerate() {
        if let ValueDef::Param { block, .. } = vd.def
            && block.0 == 0
            && def_block[v] == NONE
            && nb > 0
        {
            def_block[v] = 0;
        }
    }

    // Uses.
    let mut escapes = vec![false; nv];
    let mut used = vec![false; nv];
    let mut note = |o: Operand, bi: u32| {
        if let Operand::Value(v) = o
            && v.index() < nv
        {
            used[v.index()] = true;
            if def_block[v.index()] != bi {
                escapes[v.index()] = true;
            }
        }
    };
    for &bi in &rpo {
        let Some(bd) = b.blocks.get(bi as usize) else { continue };
        for i in bd.insts.indices() {
            for &o in b.args(i) {
                note(o, bi);
            }
        }
        for_term_operands(b, &bd.term, |o| note(o, bi));
    }
    Analysis { rpo, loop_depth, def_block, escapes, used }
}
