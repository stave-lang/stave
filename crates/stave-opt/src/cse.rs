//! Repeated loop steps: an integer arithmetic instruction whose result only feeds branch arguments
//! (the next value of a loop variable, `i = i + 8` after the test `i + 8 <= n`) and whose op,
//! type, and operands equal those of an instruction that dominates it reads that instruction's
//! result instead.
//!
//! The ops include trapping ones (`add`, `sub`, `mul`, `shl`, `shr`: `Overflow`). The dominating
//! instruction runs first on every path to the second, on the same operand values: when it traps,
//! the second is never reached, and when it does not, the second would neither trap nor compute
//! another result. Removing the second changes no result, no trap, and no trap order.
//!
//! Only a repeat in a loop that also holds the first instruction is shared, and only when
//! nothing but branch arguments read it: sharing more would keep the first result in a register
//! across blocks (or a whole function) where recomputing it is as cheap. Compares are left alone
//! (a backend fuses a compare with the branch that is its only reader), and so is address
//! arithmetic (a backend folds a single-use address into the access).

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hash, Hasher};

use stave_ir::verify::Dominators;
use stave_ir::{Body, Op, Operand, TypeKey, TypeTable};

use crate::cfg::{Edges, term_operand};
use crate::edit::Edit;

/// An operand as compared: equal constants are equal whatever their pool entries.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Arg {
    None,
    Value(u32),
    Int(i128),
    Other(Operand),
}

type Key = (Op, u32, [Arg; 2]);

/// A small multiplicative hasher: keys are a few words, and most functions only need to learn
/// that no key repeats.
#[derive(Default)]
struct Fx(u64);

impl Fx {
    fn mix(&mut self, x: u64) {
        self.0 = (self.0.rotate_left(5) ^ x).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

impl Hasher for Fx {
    fn write(&mut self, bytes: &[u8]) {
        for &x in bytes {
            self.mix(u64::from(x));
        }
    }

    fn write_u8(&mut self, x: u8) {
        self.mix(u64::from(x));
    }

    fn write_u16(&mut self, x: u16) {
        self.mix(u64::from(x));
    }

    fn write_u32(&mut self, x: u32) {
        self.mix(u64::from(x));
    }

    fn write_u64(&mut self, x: u64) {
        self.mix(x);
    }

    fn write_u128(&mut self, x: u128) {
        self.mix(x as u64);
        self.mix((x >> 64) as u64);
    }

    fn write_usize(&mut self, x: usize) {
        self.mix(x as u64);
    }

    fn write_i128(&mut self, x: i128) {
        self.write_u128(x as u128);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

fn hash_of(k: &Key) -> u64 {
    let mut h = Fx::default();
    k.hash(&mut h);
    h.finish()
}

fn candidate(op: Op) -> bool {
    matches!(
        op,
        Op::Add
            | Op::Sub
            | Op::Mul
            | Op::Neg
            | Op::AddWrap
            | Op::SubWrap
            | Op::MulWrap
            | Op::ShlWrap
            | Op::AddSat
            | Op::SubSat
            | Op::MulSat
            | Op::Shl
            | Op::Shr
            | Op::And
            | Op::Or
            | Op::Xor
    )
}

/// Blocks a search from one block visits before it gives up (and the repeat is kept).
const REACH_LIMIT: usize = 4096;

/// Whether block `to` is reachable from block `from` (within [`REACH_LIMIT`] blocks). With `to`
/// dominating `from`, that puts both in one loop. `seen[x] == stamp` marks the visited blocks.
fn reaches(
    succ: &[Vec<u32>],
    from: usize,
    to: usize,
    seen: &mut [u32],
    stamp: u32,
    work: &mut Vec<u32>,
) -> bool {
    work.clear();
    work.extend(succ[from].iter().copied());
    let mut visited = 0;
    while let Some(x) = work.pop() {
        let x = x as usize;
        if x == to {
            return true;
        }
        if seen[x] == stamp {
            continue;
        }
        seen[x] = stamp;
        visited += 1;
        if visited > REACH_LIMIT {
            return false;
        }
        work.extend(succ[x].iter().copied());
    }
    false
}

/// Replaces the dominated repeats of the integer ops above that only feed branch arguments.
/// Returns whether the body changed.
pub fn remove_redundant(types: &TypeTable, b: &mut Body) -> bool {
    let nb = b.blocks.len();
    let nv = b.values.len();
    // Per value: read by nothing but branch arguments.
    let mut edge_only = vec![true; nv];
    let mut not_edge = |o: Operand| {
        if let Operand::Value(v) = o
            && let Some(e) = edge_only.get_mut(v.index())
        {
            *e = false;
        }
    };
    for bd in &b.blocks {
        for i in bd.insts.indices() {
            b.args(i).iter().for_each(|&o| not_edge(o));
        }
        if let Some(o) = term_operand(&bd.term) {
            not_edge(o);
        }
    }
    let int = |t| matches!(types.key(t), Some(TypeKey::Int { .. }));
    let repeat = |i: usize| {
        candidate(b.insts.op[i])
            && b.args(i as u32).len() <= 2
            && b.insts.ty[i].is_some_and(int)
            && b.insts.result[i].is_some_and(|v| edge_only.get(v.index()).copied().unwrap_or(false))
    };
    if nb == 0 || !(0..b.insts.len()).any(repeat) {
        return false;
    }
    let key = |i: u32| -> Key {
        let mut args = [Arg::None; 2];
        for (a, &o) in args.iter_mut().zip(b.args(i)) {
            *a = match (o, b.int_const(o)) {
                (Operand::Value(v), _) => Arg::Value(v.0),
                (_, Some(k)) => Arg::Int(k),
                _ => Arg::Other(o),
            };
        }
        let t = b.insts.ty[i as usize].map_or(u32::MAX, |t| t.0);
        (b.insts.op[i as usize], t, args)
    };
    // Without two equal keys, one of them a repeat, nothing is shared: skip the dominator tree.
    let mut hashes: Vec<(u64, bool)> = (0..b.insts.len())
        .filter(|&i| {
            candidate(b.insts.op[i]) && b.args(i as u32).len() <= 2 && b.insts.ty[i].is_some_and(int)
        })
        .map(|i| (hash_of(&key(i as u32)), repeat(i)))
        .collect();
    hashes.sort_unstable();
    if !hashes.windows(2).any(|w| w[0].0 == w[1].0 && (w[0].1 || w[1].1)) {
        return false;
    }
    let dom = Dominators::compute(b);
    let succ = Edges::new(b).succ;
    let mut seen_at = vec![0u32; nb];
    let mut stamp = 0u32;
    let mut work: Vec<u32> = Vec::new();
    let mut children: Vec<Vec<u32>> = vec![Vec::new(); nb];
    for bi in 1..nb {
        if dom.reachable(bi)
            && let Some(p) = dom.idom(bi)
            && p != bi
        {
            children[p].push(bi as u32);
        }
    }
    let mut edit = Edit::new(b);
    // The instructions of the open blocks of a preorder walk of the dominator tree: key -> result.
    // Each stack entry (block, next child, keys it added) drops its keys when the walk leaves the
    // block's subtree.
    let mut seen: HashMap<Key, (Operand, usize), BuildHasherDefault<Fx>> = HashMap::default();
    let mut stack: Vec<(u32, usize, Vec<Key>)> = vec![(0, 0, Vec::new())];
    let mut entered = vec![false; nb];
    while let Some(top) = stack.last_mut() {
        let bi = top.0 as usize;
        if !entered[bi] {
            entered[bi] = true;
            let mut added = Vec::new();
            for i in b.blocks[bi].insts.indices() {
                let iu = i as usize;
                let (Some(v), Some(t)) = (b.insts.result[iu], b.insts.ty[iu]) else { continue };
                if !candidate(b.insts.op[iu]) || !int(t) || b.args(i).len() > 2 {
                    continue;
                }
                let k = key(i);
                match seen.get(&k) {
                    Some(&(prev, at))
                        if repeat(iu) && {
                            stamp += 1;
                            reaches(&succ, bi, at, &mut seen_at, stamp, &mut work)
                        } =>
                    {
                        edit.remove(i);
                        edit.replace(v, prev);
                    }
                    Some(_) => {}
                    None => {
                        seen.insert(k, (Operand::Value(v), bi));
                        added.push(k);
                    }
                }
            }
            top.2 = added;
        }
        let top = stack.last_mut().expect("non-empty");
        if let Some(&c) = children[top.0 as usize].get(top.1) {
            top.1 += 1;
            stack.push((c, 0, Vec::new()));
        } else {
            let (_, _, added) = stack.pop().expect("non-empty");
            for k in added {
                seen.remove(&k);
            }
        }
    }
    let changed = !edit.is_empty();
    edit.apply(b);
    changed
}
