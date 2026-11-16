//! Global register assignment for release code generation, shared by both backends.
//!
//! A value that is used outside its defining block, or is a param of a block other than the
//! entry (the branches into the block write it), gets one register for its whole life or none
//! (it then lives in its frame home, as in a debug build). Other values used only in their
//! defining block are left to the backend's block-local register cache.
//!
//! 1. Liveness: for every candidate value, the set of blocks where it is live-in, found by
//!    walking predecessors from each use up to the defining block. Its *interval* is the hull of
//!    those blocks and the defining block in emission order.
//! 2. Linear scan over the intervals at block granularity: two values share a register only if
//!    their intervals have no block in common. So inside one block a register assigned here has
//!    one owner, and a backend may write a value's register anywhere in the defining block before
//!    the definition.
//! 3. When registers run out, the interval with the lowest weight (uses and the definition, each
//!    counted `8^loop depth`) stays in memory.
//!
//! Registers come in two kinds per class: *saved* (preserved by calls) and *scratch* (clobbered by
//! calls). An interval takes a scratch register only if no block of it holds a clobbering
//! instruction after the definition.
//!
//! Vector values ([`CLASS_VEC`]) take the float registers: a vector register and the float
//! register of the same number are one register, so the two classes share one pool and only
//! differ in which of its registers survive a call ([`Pools::vec_saved`]). A value without a
//! register lives in its frame home, which the backend sizes and aligns for the value (16 or 32
//! bytes for a vector).
//!
//! On an edge `P -> T`, the register of a param of `T` is written at the end of `P`. Any other
//! value holding that register at the end of `P` is not live into `T` (it would overlap the param
//! at `T`'s first block), so the write is safe on the edge.

use stave_ir::ir::ValueDef;
use stave_ir::{Body, Operand, Term};

use crate::cfg::{Edges, NONE, for_targets, term_operand};

/// No register: the value lives in the block-local cache or its frame home.
pub const NO_REG: u8 = u8::MAX;
pub const CLASS_NONE: u8 = 0;
pub const CLASS_INT: u8 = 1;
pub const CLASS_FLOAT: u8 = 2;
/// A vector or mask value: a register of the float pool, whole.
pub const CLASS_VEC: u8 = 3;

/// Blocks visited by the liveness walk before a function is left unassigned.
const WALK_LIMIT: u64 = 50_000_000;

/// How many registers of each kind the backend dedicates. At most 32 per class.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pools {
    pub int_saved: u8,
    pub int_scratch: u8,
    pub float_saved: u8,
    pub float_scratch: u8,
    /// How many of the float registers, from index 0, keep a whole vector across a clobbering
    /// instruction (at most `float_saved`). The other float registers hold a vector only where
    /// no clobber intervenes.
    pub vec_saved: u8,
}

/// The result: per value, a register index within its class or [`NO_REG`]. Indices below the
/// class's saved count are saved registers, the rest scratch registers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Assignment {
    pub reg: Vec<u8>,
    /// Per value with a register: its interval, as the first and last index into `order`. No
    /// other value has the register in a block of that range.
    pub first: Vec<u32>,
    pub last: Vec<u32>,
    /// Bit `i`: int register index `i` is assigned to some value.
    pub int_used: u32,
    /// The same for the float registers, whether a float or a vector has them.
    pub float_used: u32,
}

impl Assignment {
    pub fn is_empty(&self) -> bool {
        self.int_used == 0 && self.float_used == 0
    }
}

struct Interval {
    value: u32,
    lo: u32,
    hi: u32,
    weight: u64,
    class: u8,
    clobbered: bool,
}

fn for_term_values(b: &Body, t: &Term, mut f: impl FnMut(u32)) {
    let mut one = |o: Operand| {
        if let Operand::Value(v) = o {
            f(v.0);
        }
    };
    if let Some(o) = term_operand(t) {
        one(o);
    }
    for_targets(b, t, |tg| {
        for &o in b.target_args(tg) {
            one(o);
        }
    });
}

/// Calls `f(value, block)` for every value operand of the reachable blocks, in emission order.
fn for_uses(b: &Body, order: &[u32], mut f: impl FnMut(u32, u32)) {
    for &bi in order {
        let Some(bd) = b.blocks.get(bi as usize) else { continue };
        for i in bd.insts.indices() {
            for &o in b.args(i) {
                if let Operand::Value(v) = o {
                    f(v.0, bi);
                }
            }
        }
        for_term_values(b, &bd.term, |v| f(v, bi));
    }
}

/// Assigns registers to the values of `b`.
///
/// - `order`: the reachable blocks in emission order, entry first.
/// - `class`: per value, [`CLASS_INT`] / [`CLASS_FLOAT`] / [`CLASS_VEC`] for values that can live
///   in a register of that class, [`CLASS_NONE`] for the rest.
/// - `clobbers`: per instruction, whether it destroys the scratch registers (a call).
pub fn assign(b: &Body, order: &[u32], class: &[u8], clobbers: &[bool], pools: Pools) -> Assignment {
    let nb = b.blocks.len();
    let nv = b.values.len();
    let mut out = Assignment {
        reg: vec![NO_REG; nv],
        first: vec![0; nv],
        last: vec![0; nv],
        int_used: 0,
        float_used: 0,
    };
    let ints = usize::from(pools.int_saved) + usize::from(pools.int_scratch);
    let floats = usize::from(pools.float_saved) + usize::from(pools.float_scratch);
    if order.len() < 2 || (ints == 0 && floats == 0) || ints > 32 || floats > 32 || class.len() < nv {
        return out;
    }
    let mut pos = vec![NONE; nb];
    for (k, &bi) in order.iter().enumerate() {
        if let Some(p) = pos.get_mut(bi as usize) {
            *p = k as u32;
        }
    }
    let edges = Edges::new(b);
    // `order` must hold every reachable block: the liveness walk follows predecessors.
    if (0..nb).any(|bi| edges.reachable[bi] != (pos[bi] != NONE)) {
        return out;
    }
    let preds = &edges.preds;
    let depth = edges.loop_depths();
    let block_weight = |bi: u32| 1u64 << (3 * u32::from(depth[bi as usize].min(8)));

    // Definitions: block and instruction (`NONE` for a block param) per candidate.
    let mut def_block = vec![NONE; nv];
    let mut def_inst = vec![NONE; nv];
    let mut weight = vec![0u64; nv];
    let mut last_clobber = vec![NONE; nb];
    for &bi in order {
        let Some(bd) = b.blocks.get(bi as usize) else { continue };
        for &p in bd.params.of(&b.block_params) {
            if p.index() < nv && class[p.index()] != CLASS_NONE {
                def_block[p.index()] = bi;
                weight[p.index()] = block_weight(bi);
            }
        }
        for i in bd.insts.indices() {
            if clobbers.get(i as usize).copied().unwrap_or(true) {
                last_clobber[bi as usize] = i;
            }
            if let Some(Some(v)) = b.insts.result.get(i as usize)
                && v.index() < nv
                && class[v.index()] != CLASS_NONE
            {
                def_block[v.index()] = bi;
                def_inst[v.index()] = i;
                weight[v.index()] = block_weight(bi);
            }
        }
    }
    // The entry's params are defined there whatever their recorded definition says.
    for (v, vd) in b.values.iter().enumerate() {
        if let ValueDef::Param { block, .. } = vd.def
            && block.0 == 0
            && class[v] != CLASS_NONE
            && def_block[v] == NONE
        {
            def_block[v] = 0;
            weight[v] = 1;
        }
    }

    // Uses in other blocks than the definition, grouped by value.
    let mut start = vec![0u32; nv + 1];
    let mut last = vec![NONE; nv];
    let mut used = vec![false; nv];
    for_uses(b, order, |v, bi| {
        let vi = v as usize;
        if vi < nv && def_block[vi] != NONE {
            used[vi] = true;
            weight[vi] = weight[vi].saturating_add(block_weight(bi));
            if def_block[vi] != bi && last[vi] != bi {
                last[vi] = bi;
                start[vi + 1] += 1;
            }
        }
    });
    for v in 0..nv {
        start[v + 1] += start[v];
    }
    let mut fill = start.clone();
    let mut use_blocks = vec![0u32; start[nv] as usize];
    last.fill(NONE);
    for_uses(b, order, |v, bi| {
        let vi = v as usize;
        if vi < nv && def_block[vi] != NONE && def_block[vi] != bi && last[vi] != bi {
            last[vi] = bi;
            use_blocks[fill[vi] as usize] = bi;
            fill[vi] += 1;
        }
    });

    // Blocks with a clobber, as a prefix count over the emission order.
    let mut clob_prefix = vec![0u32; order.len() + 1];
    for (k, &bi) in order.iter().enumerate() {
        clob_prefix[k + 1] = clob_prefix[k] + u32::from(last_clobber[bi as usize] != NONE);
    }

    // Intervals.
    let mut intervals: Vec<Interval> = Vec::new();
    let mut stamp = vec![NONE; nb];
    let mut stack: Vec<u32> = Vec::new();
    let mut walked = 0u64;
    for v in 0..nv {
        let uses = &use_blocks[start[v] as usize..start[v + 1] as usize];
        let d = def_block[v];
        // A param of a later block is written by the branches into it: it crosses blocks even
        // when every read is in its own block.
        let param = d != NONE && d != 0 && def_inst[v] == NONE && used[v];
        if uses.is_empty() && !param {
            continue;
        }
        let (mut lo, mut hi) = (pos[d as usize], pos[d as usize]);
        stack.clear();
        stack.extend_from_slice(uses);
        while let Some(x) = stack.pop() {
            if x == d || stamp[x as usize] == v as u32 {
                continue;
            }
            stamp[x as usize] = v as u32;
            walked += 1;
            let p = pos[x as usize];
            lo = lo.min(p);
            hi = hi.max(p);
            stack.extend(preds[x as usize].iter().copied());
        }
        if walked > WALK_LIMIT {
            return out;
        }
        let in_def = last_clobber[d as usize] != NONE
            && (def_inst[v] == NONE || last_clobber[d as usize] > def_inst[v]);
        let dp = pos[d as usize] as usize;
        // A use reached only through a back edge puts blocks before the definition in the hull.
        let before = clob_prefix[dp] - clob_prefix[lo as usize];
        let after = clob_prefix[hi as usize + 1] - clob_prefix[dp + 1];
        intervals.push(Interval {
            value: v as u32,
            lo,
            hi,
            weight: weight[v],
            class: class[v],
            clobbered: in_def || before + after > 0,
        });
    }
    intervals.sort_by_key(|i| (i.lo, i.value));

    // Linear scan. `owner[pool][reg]` = index into `intervals`; pool 0 is the int registers,
    // pool 1 the float registers.
    let mut owner = [[NONE; 32]; 2];
    for k in 0..intervals.len() {
        let (lo, cls, clobbered, w) = {
            let i = &intervals[k];
            (i.lo, i.class, i.clobbered, i.weight)
        };
        // `saved`: the registers that keep a value of this class across a clobber; `pushed`:
        // the ones the prologue saves when used.
        let (saved, pushed, total) = match cls {
            CLASS_INT => (usize::from(pools.int_saved), usize::from(pools.int_saved), ints),
            CLASS_VEC => {
                (usize::from(pools.vec_saved.min(pools.float_saved)), usize::from(pools.float_saved), floats)
            }
            _ => (usize::from(pools.float_saved), usize::from(pools.float_saved), floats),
        };
        let own = &mut owner[usize::from(cls != CLASS_INT)];
        for o in own.iter_mut().take(total) {
            if *o != NONE && intervals[*o as usize].hi < lo {
                *o = NONE;
            }
        }
        // Scratch registers first (no save in the prologue), saved ones for clobbered intervals:
        // a float leaves the ones that keep a vector for last.
        let candidates = if clobbered { 0..saved } else { 0..total };
        let vsaved = if cls == CLASS_FLOAT { usize::from(pools.vec_saved).min(saved) } else { 0 };
        let pick = |own: &[u32; 32]| {
            let free = |r: &usize| own[*r] == NONE;
            if clobbered {
                (vsaved..saved).find(free).or_else(|| (0..vsaved).find(free))
            } else {
                (pushed..total).find(free).or_else(|| (0..pushed).find(free))
            }
        };
        let r = match pick(own) {
            Some(r) => Some(r),
            None => {
                let victim = candidates.clone().min_by_key(|&r| intervals[own[r] as usize].weight);
                match victim {
                    Some(r) if intervals[own[r] as usize].weight < w => {
                        out.reg[intervals[own[r] as usize].value as usize] = NO_REG;
                        Some(r)
                    }
                    _ => None,
                }
            }
        };
        if let Some(r) = r {
            own[r] = k as u32;
            out.reg[intervals[k].value as usize] = r as u8;
        }
    }
    for i in &intervals {
        let r = out.reg[i.value as usize];
        if r != NO_REG {
            out.first[i.value as usize] = i.lo;
            out.last[i.value as usize] = i.hi;
            if i.class == CLASS_INT {
                out.int_used |= 1 << r;
            } else {
                out.float_used |= 1 << r;
            }
        }
    }
    out
}
