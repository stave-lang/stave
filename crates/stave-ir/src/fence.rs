//! Placement of `mem.fence_stream` after streaming stores (contracts/simd-ops.md "Memory hints"):
//! on every path from a streaming store (`mem.store_stream`, `vec.store_stream`) to a return or a
//! call (`call`, `call.ind`, `intrinsic.*`), a fence follows the last streaming store before that
//! return or call. A fence goes where no further streaming store can come before the return or
//! call: at the start of the block a streaming loop exits to, right before the call, or right
//! after the last store of straight-line code. One fence serves every store before it, so a
//! fence the program already has (a `fence_stream()` call) is never repeated.
//!
//! At a point between two instructions three facts decide (each over some path, through the
//! CFG): P, a streaming store ran since the last fence; S, a streaming store can still run before
//! the next fence, call, or return; O, a call or return can come before the next fence. A point
//! needs a fence iff P, not S, and O. Fences go at the first points of each needing region: at a
//! block start when a predecessor reaches it with P and S (the edge out of a streaming loop), else
//! right after the needing region starts inside a block. The facts are then recomputed until no
//! point needs a fence.

use crate::ir::{Block, Body, Op, Site, Span, Term, ValueDef};
use crate::simd::VecOp;

fn ops(b: &Body, k: usize) -> &[Op] {
    b.blocks[k].insts.of(&b.insts.op)
}

fn is_stream_store(op: Op) -> bool {
    matches!(op, Op::StoreStream | Op::Vec(VecOp::StoreStream))
}

/// A call: code that may read the streamed data.
fn observes(op: Op) -> bool {
    matches!(op, Op::Call | Op::CallInd | Op::Intrinsic(_))
}

/// Rounds of placement before the fallback (a fence before every call or return reached with P).
const ROUNDS: usize = 16;

/// Places the fences of streaming stores in `b` (a body whose blocks are all reachable). Returns
/// how many it inserted.
pub fn place_stream_fences(b: &mut Body) -> u32 {
    if !b.insts.op.iter().any(|&op| is_stream_store(op)) {
        return 0;
    }
    let mut total = 0;
    for _ in 0..ROUNDS {
        let f = Facts::new(b);
        let mut at = f.loop_exits(b);
        if at.is_empty() {
            at = f.region_starts(b);
        }
        if at.is_empty() {
            return total;
        }
        total += insert(b, &at);
    }
    let f = Facts::new(b);
    let at = f.unfenced_observers(b);
    total + insert(b, &at)
}

/// The facts of every block: P at entry and exit, S and O at entry and exit.
struct Facts {
    preds: Vec<Vec<u32>>,
    p_in: Vec<bool>,
    p_out: Vec<bool>,
    s_out: Vec<bool>,
    o_out: Vec<bool>,
}

impl Facts {
    fn new(b: &Body) -> Facts {
        let nb = b.blocks.len();
        let mut preds = vec![Vec::new(); nb];
        let mut succs = vec![Vec::new(); nb];
        for (k, out) in succs.iter_mut().enumerate() {
            b.for_each_successor(Block(k as u32), |t| {
                if t.block.index() < nb {
                    preds[t.block.index()].push(k as u32);
                    out.push(t.block.0);
                }
            });
        }
        // P, forward
        let (mut p_in, mut p_out) = (vec![false; nb], vec![false; nb]);
        let mut changed = true;
        while changed {
            changed = false;
            for k in 0..nb {
                let i = preds[k].iter().any(|&q| p_out[q as usize]);
                let o = ops(b, k).iter().fold(i, |p, &op| p_fwd(op, p));
                if i != p_in[k] || o != p_out[k] {
                    (p_in[k], p_out[k], changed) = (i, o, true);
                }
            }
        }
        // S and O, backward
        let ret = |k: usize| matches!(b.blocks[k].term, Term::Ret(_));
        let (mut s_in, mut s_out) = (vec![false; nb], vec![false; nb]);
        let (mut o_in, mut o_out) = (vec![false; nb], vec![false; nb]);
        let mut changed = true;
        while changed {
            changed = false;
            for k in (0..nb).rev() {
                let so = succs[k].iter().any(|&q| s_in[q as usize]);
                let oo = ret(k) || succs[k].iter().any(|&q| o_in[q as usize]);
                let (si, oi) = ops(b, k).iter().rev().fold((so, oo), |(s, o), &op| back(op, s, o));
                if (so, oo, si, oi) != (s_out[k], o_out[k], s_in[k], o_in[k]) {
                    (s_out[k], o_out[k], s_in[k], o_in[k]) = (so, oo, si, oi);
                    changed = true;
                }
            }
        }
        Facts { preds, p_in, p_out, s_out, o_out }
    }

    /// Whether each point of block `k` needs a fence: entry `0`, before instruction `j` at `j`,
    /// before the terminator last.
    fn needs(&self, b: &Body, k: usize) -> Vec<bool> {
        let ops = ops(b, k);
        let n = ops.len();
        let mut p = vec![false; n + 1];
        p[0] = self.p_in[k];
        for (j, &op) in ops.iter().enumerate() {
            p[j + 1] = p_fwd(op, p[j]);
        }
        let mut need = vec![false; n + 1];
        let (mut s, mut o) = (self.s_out[k], self.o_out[k]);
        need[n] = p[n] && !s && o;
        for j in (0..n).rev() {
            (s, o) = back(ops[j], s, o);
            need[j] = p[j] && !s && o;
        }
        need
    }

    /// Block starts reached from a predecessor that leaves with P and S: the exits of streaming
    /// loops (and of branches that store on one side only).
    fn loop_exits(&self, b: &Body) -> Vec<(u32, u32)> {
        let mut at = Vec::new();
        for k in 1..b.blocks.len() {
            if !self.needs(b, k)[0] {
                continue;
            }
            if self.preds[k].iter().any(|&q| self.p_out[q as usize] && self.s_out[q as usize]) {
                at.push((k as u32, 0));
            }
        }
        at
    }

    /// The first point of each needing run inside a block.
    fn region_starts(&self, b: &Body) -> Vec<(u32, u32)> {
        let mut at = Vec::new();
        for k in 0..b.blocks.len() {
            for (j, w) in self.needs(b, k).windows(2).enumerate() {
                if w[1] && !w[0] {
                    at.push((k as u32, j as u32 + 1));
                }
            }
        }
        at
    }

    /// Every call and return reached with P.
    fn unfenced_observers(&self, b: &Body) -> Vec<(u32, u32)> {
        let mut at = Vec::new();
        for k in 0..b.blocks.len() {
            let ops = ops(b, k);
            let mut p = self.p_in[k];
            for (j, &op) in ops.iter().enumerate() {
                if p && observes(op) {
                    at.push((k as u32, j as u32));
                    p = false;
                }
                p = p_fwd(op, p);
            }
            if p && matches!(b.blocks[k].term, Term::Ret(_)) {
                at.push((k as u32, ops.len() as u32));
            }
        }
        at
    }
}

/// P after `op`, from P before it.
fn p_fwd(op: Op, p: bool) -> bool {
    if is_stream_store(op) {
        true
    } else if op == Op::FenceStream {
        false
    } else {
        p
    }
}

/// (S, O) before `op`, from (S, O) after it.
fn back(op: Op, s: bool, o: bool) -> (bool, bool) {
    if is_stream_store(op) {
        (true, o)
    } else if op == Op::FenceStream {
        (false, false)
    } else if observes(op) {
        (false, true)
    } else {
        (s, o)
    }
}

/// Inserts a fence at each (block, point) of `at`; returns how many.
fn insert(b: &mut Body, at: &[(u32, u32)]) -> u32 {
    if at.is_empty() {
        return 0;
    }
    let mut at = at.to_vec();
    at.sort_unstable();
    at.dedup();
    let old = std::mem::take(&mut b.insts);
    let n = old.op.len() + at.len();
    let mut new = crate::ir::Insts::default();
    new.op.reserve(n);
    new.ty.reserve(n);
    new.result.reserve(n);
    new.args.reserve(n);
    new.site.reserve(n);
    let mut next = at.iter().peekable();
    for (k, bd) in b.blocks.iter_mut().enumerate() {
        let start = new.op.len() as u32;
        let len = bd.insts.len;
        for j in 0..=len {
            while next.peek().is_some_and(|&&(bk, bj)| bk as usize == k && bj == j) {
                next.next();
                new.op.push(Op::FenceStream);
                new.ty.push(None);
                new.result.push(None);
                new.args.push(Span::default());
                new.site.push(Site::NONE);
            }
            if j == len {
                break;
            }
            let i = (bd.insts.start + j) as usize;
            if let Some(v) = old.result[i]
                && let Some(vd) = b.values.get_mut(v.index())
            {
                vd.def = ValueDef::Inst(new.op.len() as u32);
            }
            new.op.push(old.op[i]);
            new.ty.push(old.ty[i]);
            new.result.push(old.result[i]);
            new.args.push(old.args[i]);
            new.site.push(old.site[i]);
        }
        bd.insts = Span { start, len: new.op.len() as u32 - start };
    }
    b.insts = new;
    at.len() as u32
}
