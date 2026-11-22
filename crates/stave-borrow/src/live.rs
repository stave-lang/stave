//! Liveness of locals, stored as runs of blocks.
//!
//! A local is live at a point when some path from there reaches a use of it before a statement
//! that redefines it entirely ([`crate::check::stmt_def`]). Each local is walked on its own,
//! backward from its uses. Blocks are numbered in source order, so the blocks at whose entry
//! (or end) a local is live are a few runs of consecutive ones: the tables grow with the
//! locals, not with blocks times locals.

use crate::cfg::{Body, LocalId, Stmt};
use crate::check::stmt_def;

/// The blocks where each variable has some property, as maximal runs of consecutive blocks.
#[derive(Default)]
struct Runs {
    /// Per variable: where its runs start in `runs` (one more entry closes the last).
    at: Vec<u32>,
    /// (first, last), ascending per variable.
    runs: Vec<(u32, u32)>,
}

impl Runs {
    fn has(&self, v: u32, b: u32) -> bool {
        let (Some(&lo), Some(&hi)) = (self.at.get(v as usize), self.at.get(v as usize + 1)) else {
            return false;
        };
        let runs = &self.runs[lo as usize..hi as usize];
        let k = runs.partition_point(|r| r.1 < b);
        runs.get(k).is_some_and(|r| r.0 <= b)
    }

    /// Starts the runs of variable `x`: the variables before it that have none get none.
    fn start(&mut self, x: u32) {
        while self.at.len() <= x as usize {
            self.at.push(self.runs.len() as u32);
        }
    }

    /// Adds the runs of the blocks from `first` to `last` that `mark` has `x` for.
    fn add(&mut self, x: u32, mark: &[u32], first: u32, last: u32) {
        let mut b = first;
        while b <= last {
            if mark[b as usize] != x {
                b += 1;
                continue;
            }
            let start = b;
            while b < last && mark[b as usize + 1] == x {
                b += 1;
            }
            self.runs.push((start, b));
            b += 1;
        }
    }
}

/// Where the variables of one kind are live.
#[derive(Default)]
pub struct Liveness {
    ins: Runs,
    outs: Runs,
}

impl Liveness {
    /// Whether variable `v` is live at the entry of block `b`.
    pub fn live_in(&self, v: u32, b: u32) -> bool {
        self.ins.has(v, b)
    }

    /// Whether variable `v` is live at the end of block `b`: at the entry of a successor.
    pub fn live_out(&self, v: u32, b: u32) -> bool {
        self.outs.has(v, b)
    }

    /// The (variable, block) pairs with the variable live at the block's entry.
    pub fn size(&self) -> u64 {
        self.ins.runs.iter().map(|r| u64::from(r.1 - r.0) + 1).sum()
    }
}

/// Liveness of the locals that are variables (`var[local]`: the variable's index, `u32::MAX`:
/// none). `uses` lists the locals a statement uses.
pub fn liveness(
    body: &Body,
    preds: &[Vec<u32>],
    var: &[u32],
    nvars: usize,
    uses: impl Fn(&Stmt, &mut dyn FnMut(LocalId)),
) -> Liveness {
    let nb = body.blocks.len();
    let mut out = Liveness::default();
    if nvars == 0 {
        return out;
    }
    // (variable, block): the block uses the variable before it redefines it; it redefines it
    let mut exposed: Vec<(u32, u32)> = Vec::new();
    let mut kills: Vec<(u32, u32)> = Vec::new();
    // per variable: the last block seen to use it, to redefine it
    let mut used_in = vec![u32::MAX; nvars];
    let mut killed_in = vec![u32::MAX; nvars];
    for (b, blk) in body.blocks.iter().enumerate() {
        let b = b as u32;
        for s in &blk.stmts {
            uses(s, &mut |l| {
                let v = var[l as usize];
                if v != u32::MAX && killed_in[v as usize] != b && used_in[v as usize] != b {
                    used_in[v as usize] = b;
                    exposed.push((v, b));
                }
            });
            if let Some(d) = stmt_def(s) {
                let v = var[d as usize];
                if v != u32::MAX && killed_in[v as usize] != b {
                    killed_in[v as usize] = b;
                    kills.push((v, b));
                }
            }
        }
    }
    exposed.sort_unstable();
    kills.sort_unstable();
    // per block: the last variable found to be redefined there, live at its end, live at its entry
    let mut kill_at = vec![u32::MAX; nb];
    let mut out_at = vec![u32::MAX; nb];
    let mut in_at = vec![u32::MAX; nb];
    let mut work: Vec<u32> = Vec::new();
    let (mut e, mut k) = (0, 0);
    while e < exposed.len() {
        let x = exposed[e].0;
        out.ins.start(x);
        out.outs.start(x);
        while k < kills.len() && kills[k].0 < x {
            k += 1;
        }
        while k < kills.len() && kills[k].0 == x {
            kill_at[kills[k].1 as usize] = x;
            k += 1;
        }
        // the first and the last block at whose entry `x` is live, at whose end it is
        let (mut first, mut last) = (u32::MAX, 0);
        let (mut ofirst, mut olast) = (u32::MAX, 0);
        while e < exposed.len() && exposed[e].0 == x {
            let b = exposed[e].1;
            in_at[b as usize] = x;
            (first, last) = (first.min(b), last.max(b));
            work.push(b);
            e += 1;
        }
        while let Some(b) = work.pop() {
            for &p in &preds[b as usize] {
                if out_at[p as usize] == x {
                    continue;
                }
                out_at[p as usize] = x;
                (ofirst, olast) = (ofirst.min(p), olast.max(p));
                if kill_at[p as usize] != x && in_at[p as usize] != x {
                    in_at[p as usize] = x;
                    (first, last) = (first.min(p), last.max(p));
                    work.push(p);
                }
            }
        }
        out.ins.add(x, &in_at, first, last);
        if ofirst != u32::MAX {
            out.outs.add(x, &out_at, ofirst, olast);
        }
    }
    out.ins.start(nvars as u32);
    out.outs.start(nvars as u32);
    out
}

#[cfg(test)]
mod tests {
    use super::{Liveness, Runs};

    #[test]
    fn runs_answer_per_block() {
        let mut ins = Runs::default();
        // variable 0 at blocks 1 2 3 7, variable 1 nowhere, variable 2 everywhere
        let mark = [9, 0, 0, 0, 9, 9, 9, 0, 9, 9];
        ins.start(0);
        ins.add(0, &mark, 1, 7);
        ins.start(2);
        ins.add(2, &[2; 10], 0, 9);
        ins.start(3);
        let l = Liveness { ins, outs: Runs::default() };
        let live: Vec<u32> = (0..10).filter(|&b| l.live_in(0, b)).collect();
        assert_eq!(live, vec![1, 2, 3, 7]);
        assert!((0..10).all(|b| !l.live_in(1, b) && l.live_in(2, b)));
        assert!(!l.live_in(2, 10) && !l.live_in(3, 0) && !l.live_out(0, 1));
        assert_eq!(l.size(), 14);
    }
}
