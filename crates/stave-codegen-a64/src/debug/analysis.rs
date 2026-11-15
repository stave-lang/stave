//! The analysis pass: reverse postorder, natural loops, liveness intervals, last uses, and the
//! frame layout.
//!
//! Liveness is TPDE's interval approximation: a value is live over the RPO block range from its
//! definition (for a block param: also every predecessor, where it is written) to its last use,
//! where a use inside a loop that does not contain the definition extends the interval to the
//! loop's last block. Home slots are assigned by a linear scan over these intervals at block
//! granularity, so values that are never live at the same time share a slot. An irreducible CFG
//! falls back to whole-function intervals (no sharing).

use stave_ir::ir::{Target, ValueDef};
use stave_ir::{Body, Func, Op, Operand, Term};
use stave_opt::regalloc;

use super::abi;
use super::emit::NOREG;
use super::support::{Codegen, Intr};
use super::ty::{Sc, Ty};
use super::{R, bad, unsupported};

pub(crate) const NONE: u32 = u32::MAX;

/// Storage class of a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Vk {
    /// Not in a reachable block.
    Dead,
    Gpr(Sc),
    Fpr(Sc),
    /// Fat pointer or aggregate, always in its home slot.
    Mem(Ty),
    /// The address of a `slot` (rematerialized from `sp`, never stored).
    Slot,
}

/// Number of successors of a terminator.
pub(crate) fn succ_count(b: &Body, blk: usize) -> usize {
    match b.blocks.get(blk).map(|d| d.term) {
        Some(Term::Br(_)) => 1,
        Some(Term::Cbr { .. }) => 2,
        Some(Term::Switch { cases, .. }) => cases.len as usize + 1,
        _ => 0,
    }
}

/// Successor `k` (switch: cases, then default).
pub(crate) fn succ(b: &Body, blk: usize, k: usize) -> Option<Target> {
    match b.blocks.get(blk)?.term {
        Term::Br(t) if k == 0 => Some(t),
        Term::Cbr { then, .. } if k == 0 => Some(then),
        Term::Cbr { els, .. } if k == 1 => Some(els),
        Term::Switch { cases, default, .. } => {
            if k < cases.len as usize {
                b.switch_cases(cases).get(k).map(|c| c.target)
            } else if k == cases.len as usize {
                Some(default)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Per-function analysis results (buffers reused across functions).
#[derive(Default)]
pub struct Analysis {
    /// Reachable blocks in reverse postorder.
    pub(crate) rpo: Vec<u32>,
    /// Block → RPO index, `NONE` if unreachable.
    pub(crate) rpo_of: Vec<u32>,
    /// The reachable blocks in emission order: `rpo`, in a release build with the cold blocks
    /// last (stave-opt `layout`). Liveness stays in RPO indices whatever this order is.
    pub(crate) emit: Vec<u32>,
    /// Per value: RPO index of the defining block.
    pub(crate) def_rpo: Vec<u32>,
    pub(crate) start: Vec<u32>,
    pub(crate) end: Vec<u32>,
    /// Per operand-pool entry: the last use of that value in its block.
    pub(crate) last_use: Vec<bool>,
    /// Per value: no use in its defining block (after the definition).
    pub(crate) unused: Vec<bool>,
    pub(crate) kind: Vec<Vk>,
    /// Per value: home slot (or slot area) offset from `sp`.
    pub(crate) home: Vec<u32>,
    pub(crate) frame: u32,
    pub(crate) scratch_off: u32,
    pub(crate) x8_off: u32,
    pub(crate) args: Vec<abi::Arg>,
    /// Release: per value, the callee-saved register it owns from its definition to its last
    /// use (`x19`–`x28` for an integer-class value, `v8`–`v15` for a float), `NOREG` for none.
    /// Empty in a debug build.
    pub(crate) own: Vec<u8>,
    /// Per owned value: the first and last block (index into `emit`) of its interval. No other
    /// value has its register in those blocks.
    pub(crate) own_first: Vec<u32>,
    pub(crate) own_last: Vec<u32>,
    /// Release: uses per value, saturating. Empty in a debug build.
    pub(crate) uses: Vec<u8>,
    /// The owned registers as (is FP, number); entry `k` is saved at `[x29 - 8 * (k + 1)]`.
    pub(crate) saved: Vec<(bool, u8)>,
    /// Release: per block, the compare instruction its `cbr` emits itself (`cmp` + `b.cond`, no
    /// bool value); `NONE` otherwise. Empty in a debug build.
    pub(crate) fused: Vec<u32>,
    // scratch
    tin: Vec<u32>,
    tout: Vec<u32>,
    stack: Vec<(u32, u32)>,
    pred_start: Vec<u32>,
    preds: Vec<u32>,
    block_loop: Vec<u32>,
    loop_hdr: Vec<u32>,
    loop_end: Vec<u32>,
    loop_parent: Vec<u32>,
    work: Vec<u32>,
    seen: Vec<u32>,
    bucket_start: Vec<u32>,
    bucket_end: Vec<u32>,
    order: Vec<u32>,
    order_end: Vec<u32>,
    free8: Vec<u32>,
    free_big: Vec<(u32, u32, u32)>,
}

fn align_up(x: u32, a: u32) -> u32 {
    let a = a.max(1);
    x.div_ceil(a) * a
}

/// Frame offsets are computed in `u64` and checked against [`FRAME_MAX`] before narrowing, so no
/// sum of slot or home sizes can wrap.
fn align_up64(x: u64, a: u64) -> u64 {
    let a = a.max(1);
    x.div_ceil(a) * a
}

/// Frames must stay below 16 MiB: every SP offset is then reachable with two `add` immediates.
const FRAME_MAX: u64 = 1 << 24;

fn frame_too_large() -> super::Error {
    unsupported("stack frame of 16 MiB or more")
}

impl Analysis {
    fn ancestor(&self, a: usize, b: usize) -> bool {
        self.tin[a] <= self.tin[b] && self.tout[b] <= self.tout[a]
    }

    /// End of the interval for a use at RPO `u` of a value defined at RPO `d`.
    fn ext(&self, u: u32, d: u32) -> u32 {
        let mut e = u;
        let mut l = self.block_loop.get(u as usize).copied().unwrap_or(NONE);
        while l != NONE && self.loop_hdr[l as usize] > d {
            e = e.max(self.loop_end[l as usize]);
            l = self.loop_parent[l as usize];
        }
        e
    }

    pub(crate) fn run(&mut self, cg: &Codegen<'_>, f: &Func, body: &Body) -> R<()> {
        let nb = body.blocks.len();
        let nv = body.values.len();
        if nb == 0 {
            return Err(bad("body has no blocks"));
        }
        self.dfs(body, nb);
        let nr = self.rpo.len();
        let irreducible = self.loops(body, nb);
        self.defs(cg, body, nv)?;
        // Intervals.
        self.start.clear();
        self.start.extend_from_slice(&self.def_rpo);
        self.end.clear();
        self.end.extend_from_slice(&self.def_rpo);
        let mut scratch_need = 0u64;
        for i in 0..nr {
            let blk = self.rpo[i] as usize;
            let bd = &body.blocks[blk];
            let i = i as u32;
            for k in bd.insts.indices() {
                for o in body.args(k) {
                    self.use_at(*o, i)?;
                }
            }
            match bd.term {
                Term::Cbr { cond: o, .. } | Term::Switch { scrut: o, .. } | Term::Ret(Some(o)) => {
                    self.use_at(o, i)?
                }
                _ => {}
            }
            for k in 0..succ_count(body, blk) {
                let Some(t) = succ(body, blk, k) else { continue };
                let tr = self.rpo_of.get(t.block.index()).copied().unwrap_or(NONE);
                if tr == NONE {
                    return Err(bad("branch to an unknown block"));
                }
                let mut need = 0u64;
                for (pi, o) in body.target_args(&t).iter().enumerate() {
                    self.use_at(*o, i)?;
                    if let Operand::Value(v) = o
                        && let Some(vd) = body.values.get(v.index())
                        && matches!(vd.def, ValueDef::Param { block, index } if block == t.block && index as usize != pi)
                    {
                        need += align_up64(u64::from(cg.tys[vd.ty.index()].size().max(8)), 8);
                    }
                }
                scratch_need = scratch_need.max(need);
                let ext = self.ext(i, tr);
                for &p in body.params(t.block) {
                    let p = p.index();
                    self.start[p] = self.start[p].min(i);
                    self.end[p] = self.end[p].max(ext);
                }
            }
        }
        // RPO does not keep loop bodies contiguous (an exit may come between the header and the
        // body), so a value defined before a loop and live across it (used after the loop) must
        // cover the whole loop: extend every interval that contains a loop header it does not
        // start at. Loops were recorded by decreasing header index.
        if !irreducible && !self.loop_hdr.is_empty() {
            for v in 0..nv {
                if self.def_rpo[v] == NONE {
                    continue;
                }
                let (st, mut en) = (self.start[v], self.end[v]);
                for l in (0..self.loop_hdr.len()).rev() {
                    let h = self.loop_hdr[l];
                    if h > en {
                        break;
                    }
                    if h > st {
                        en = en.max(self.loop_end[l]);
                    }
                }
                self.end[v] = en;
            }
        }
        if irreducible {
            for v in 0..nv {
                if self.def_rpo[v] != NONE {
                    self.start[v] = 0;
                    self.end[v] = nr as u32 - 1;
                }
            }
        }
        self.last_uses(body, nr);
        self.frame(cg, f, body, nr, scratch_need)?;
        self.own.clear();
        self.own_first.clear();
        self.own_last.clear();
        self.uses.clear();
        self.saved.clear();
        self.fused.clear();
        self.emit.clear();
        if cg.release {
            self.emit = stave_opt::layout::order(body);
            self.ownership(body)?;
            self.fusion(cg, body);
        } else {
            self.emit.extend_from_slice(&self.rpo);
        }
        Ok(())
    }

    /// Release: finds the compares a `cbr` can emit itself: defined in the branch's block, used
    /// by nothing else, on scalars. Their operands stay bound up to the terminator (no last-use
    /// mark in the block).
    fn fusion(&mut self, cg: &Codegen<'_>, body: &Body) {
        self.uses.resize(body.values.len(), 0);
        let uses = &mut self.uses;
        let mut add = |o: Operand| {
            if let Operand::Value(v) = o
                && let Some(c) = uses.get_mut(v.index())
            {
                *c = c.saturating_add(1);
            }
        };
        body.operands.iter().for_each(|o| add(*o));
        for bd in &body.blocks {
            match bd.term {
                Term::Cbr { cond: o, .. } | Term::Switch { scrut: o, .. } | Term::Ret(Some(o)) => add(o),
                _ => {}
            }
        }
        self.fused.resize(body.blocks.len(), NONE);
        for &blk in &self.rpo {
            let bd = &body.blocks[blk as usize];
            let Term::Cbr { cond: Operand::Value(c), .. } = bd.term else { continue };
            let Some(ValueDef::Inst(j)) = body.values.get(c.index()).map(|v| v.def) else { continue };
            if !bd.insts.indices().contains(&j) || self.uses[c.index()] != 1 {
                continue;
            }
            if !matches!(body.insts.op[j as usize], Op::Eq | Op::Ne | Op::Lt | Op::Le | Op::Gt | Op::Ge) {
                continue;
            }
            let scalar = |o: &Operand| match o {
                Operand::Value(v) => {
                    body.value_ty(*v).is_some_and(|t| matches!(cg.ty(t), Ty::Sc(_)))
                        && !matches!(self.kind[v.index()], Vk::Slot)
                }
                Operand::Const(_) => true,
                _ => false,
            };
            let args = body.args(j);
            if args.len() != 2
                || !args.iter().all(scalar)
                || args.iter().all(|o| matches!(o, Operand::Const(_)))
            {
                continue;
            }
            self.fused[blk as usize] = j;
            for oi in body.insts.args[j as usize].indices() {
                self.last_use[oi as usize] = false;
            }
        }
    }

    /// Release: which values own a register, and the save area for those registers at the top
    /// of the frame.
    fn ownership(&mut self, body: &Body) -> R<()> {
        const INTS: u8 = 10;
        const FLOATS: u8 = 8;
        let class: Vec<u8> = self
            .kind
            .iter()
            .map(|k| match k {
                Vk::Gpr(_) => regalloc::CLASS_INT,
                Vk::Fpr(_) => regalloc::CLASS_FLOAT,
                _ => regalloc::CLASS_NONE,
            })
            .collect();
        let pools =
            regalloc::Pools { int_saved: INTS, int_scratch: 0, float_saved: FLOATS, float_scratch: 0 };
        let asg = regalloc::assign(body, &self.emit, &class, &[], pools);
        if asg.is_empty() {
            return Ok(());
        }
        self.own_first.extend_from_slice(&asg.first);
        self.own_last.extend_from_slice(&asg.last);
        self.own.extend(asg.reg.iter().zip(&class).map(|(&r, &c)| {
            if r == regalloc::NO_REG {
                NOREG
            } else if c == regalloc::CLASS_FLOAT {
                8 + r % FLOATS
            } else {
                19 + r % INTS
            }
        }));
        self.saved.extend((0..INTS).filter(|r| asg.int_used & (1 << r) != 0).map(|r| (false, 19 + r)));
        self.saved.extend((0..FLOATS).filter(|r| asg.float_used & (1 << r) != 0).map(|r| (true, 8 + r)));
        let frame = align_up64(u64::from(self.frame) + 8 * self.saved.len() as u64, 16);
        if frame >= FRAME_MAX {
            return Err(frame_too_large());
        }
        self.frame = frame as u32;
        Ok(())
    }

    fn use_at(&mut self, o: Operand, i: u32) -> R<()> {
        if let Operand::Value(v) = o {
            let d = self.def_rpo.get(v.index()).copied().unwrap_or(NONE);
            if d == NONE {
                return Err(bad(format!("use of %{} defined in no reachable block", v.0)));
            }
            let e = self.ext(i, d);
            let end = &mut self.end[v.index()];
            *end = (*end).max(e);
        }
        Ok(())
    }

    /// Iterative DFS from b0: preorder/postorder stamps, RPO.
    fn dfs(&mut self, body: &Body, nb: usize) {
        self.tin.clear();
        self.tin.resize(nb, NONE);
        self.tout.clear();
        self.tout.resize(nb, NONE);
        self.rpo.clear();
        self.stack.clear();
        let mut clock = 0u32;
        self.tin[0] = clock;
        clock += 1;
        self.stack.push((0, 0));
        while let Some(&(bk, k)) = self.stack.last() {
            if let Some(t) = succ(body, bk as usize, k as usize) {
                if let Some(top) = self.stack.last_mut() {
                    top.1 += 1;
                }
                let s = t.block.index();
                if s < nb && self.tin[s] == NONE {
                    self.tin[s] = clock;
                    clock += 1;
                    self.stack.push((s as u32, 0));
                }
            } else {
                self.tout[bk as usize] = clock;
                clock += 1;
                self.rpo.push(bk);
                self.stack.pop();
            }
        }
        self.rpo.reverse();
        self.rpo_of.clear();
        self.rpo_of.resize(nb, NONE);
        for (i, &b) in self.rpo.iter().enumerate() {
            self.rpo_of[b as usize] = i as u32;
        }
    }

    /// Natural loops (innermost loop per RPO index, loop ends, nesting). Returns true for an
    /// irreducible CFG.
    fn loops(&mut self, body: &Body, nb: usize) -> bool {
        let nr = self.rpo.len();
        // Predecessors (reachable blocks only), CSR.
        self.pred_start.clear();
        self.pred_start.resize(nb + 1, 0);
        for &b in &self.rpo {
            for k in 0..succ_count(body, b as usize) {
                if let Some(t) = succ(body, b as usize, k) {
                    self.pred_start[t.block.index() + 1] += 1;
                }
            }
        }
        for i in 0..nb {
            self.pred_start[i + 1] += self.pred_start[i];
        }
        self.preds.clear();
        self.preds.resize(self.pred_start[nb] as usize, 0);
        self.work.clear();
        self.work.extend_from_slice(&self.pred_start[..nb]);
        let mut irreducible = false;
        let mut any_loop = false;
        for &b in &self.rpo {
            for k in 0..succ_count(body, b as usize) {
                if let Some(t) = succ(body, b as usize, k) {
                    let s = t.block.index();
                    let at = &mut self.work[s];
                    self.preds[*at as usize] = b;
                    *at += 1;
                    if self.rpo_of[s] <= self.rpo_of[b as usize] {
                        if self.ancestor(s, b as usize) {
                            any_loop = true;
                        } else {
                            irreducible = true;
                        }
                    }
                }
            }
        }
        self.block_loop.clear();
        self.block_loop.resize(nr, NONE);
        self.loop_hdr.clear();
        self.loop_end.clear();
        self.loop_parent.clear();
        if irreducible || !any_loop {
            return irreducible;
        }
        for hr in (0..nr).rev() {
            let h = self.rpo[hr] as usize;
            let (ps, pe) = (self.pred_start[h] as usize, self.pred_start[h + 1] as usize);
            let is_header = self.preds[ps..pe].iter().any(|&p| self.ancestor(h, p as usize));
            if !is_header {
                continue;
            }
            let l = self.loop_hdr.len() as u32;
            self.loop_hdr.push(hr as u32);
            self.loop_end.push(hr as u32);
            self.loop_parent.push(NONE);
            self.block_loop[hr] = l;
            let mut end = hr as u32;
            self.work.clear();
            for k in ps..pe {
                let p = self.preds[k];
                if self.ancestor(h, p as usize) {
                    self.work.push(p);
                }
            }
            while let Some(x) = self.work.pop() {
                let x = x as usize;
                if x == h {
                    continue;
                }
                let xr = self.rpo_of[x] as usize;
                let inner = self.block_loop[xr];
                let from = if inner == NONE {
                    self.block_loop[xr] = l;
                    end = end.max(xr as u32);
                    x
                } else {
                    let mut top = inner;
                    while self.loop_parent[top as usize] != NONE {
                        top = self.loop_parent[top as usize];
                    }
                    if top == l {
                        continue;
                    }
                    self.loop_parent[top as usize] = l;
                    end = end.max(self.loop_end[top as usize]);
                    self.rpo[self.loop_hdr[top as usize] as usize] as usize
                };
                let (s, e) = (self.pred_start[from] as usize, self.pred_start[from + 1] as usize);
                for k in s..e {
                    let p = self.preds[k];
                    if p as usize != h {
                        self.work.push(p);
                    }
                }
            }
            self.loop_end[l as usize] = end;
        }
        false
    }

    /// Defining blocks and storage classes.
    fn defs(&mut self, cg: &Codegen<'_>, body: &Body, nv: usize) -> R<()> {
        self.def_rpo.clear();
        self.def_rpo.resize(nv, NONE);
        self.kind.clear();
        self.kind.resize(nv, Vk::Dead);
        for (i, &blk) in self.rpo.iter().enumerate() {
            let bd = &body.blocks[blk as usize];
            for &p in bd.params.of(&body.block_params) {
                if let Some(d) = self.def_rpo.get_mut(p.index()) {
                    *d = i as u32;
                }
            }
            for k in bd.insts.indices() {
                if let Some(Some(r)) = body.insts.result.get(k as usize) {
                    if let Some(d) = self.def_rpo.get_mut(r.index()) {
                        *d = i as u32;
                    }
                    if body.insts.op[k as usize] == Op::Slot {
                        self.kind[r.index()] = Vk::Slot;
                    }
                }
            }
        }
        for v in 0..nv {
            if self.def_rpo[v] == NONE || self.kind[v] == Vk::Slot {
                continue;
            }
            let t = body.values[v].ty;
            self.kind[v] = match cg.tys.get(t.index()).copied().unwrap_or(Ty::Bad("type id out of range")) {
                Ty::Sc(s) if s.is_fp() => Vk::Fpr(s),
                Ty::Sc(s) => Vk::Gpr(s),
                t @ (Ty::Fat | Ty::Mem { .. }) => Vk::Mem(t),
                Ty::Bad(why) => return Err(unsupported(format!("%{v}: {why}"))),
            };
        }
        Ok(())
    }

    fn last_uses(&mut self, body: &Body, nr: usize) {
        self.last_use.clear();
        self.last_use.resize(body.operands.len(), false);
        self.unused.clear();
        self.unused.resize(body.values.len(), false);
        self.seen.clear();
        self.seen.resize(body.values.len(), NONE);
        for i in 0..nr {
            let blk = self.rpo[i] as usize;
            let bd = &body.blocks[blk];
            let stamp = i as u32;
            let mark = |o: &Operand, seen: &mut Vec<u32>| {
                if let Operand::Value(v) = o
                    && let Some(s) = seen.get_mut(v.index())
                {
                    *s = stamp;
                }
            };
            match bd.term {
                Term::Cbr { cond: o, .. } | Term::Switch { scrut: o, .. } | Term::Ret(Some(o)) => {
                    mark(&o, &mut self.seen)
                }
                _ => {}
            }
            for k in 0..succ_count(body, blk) {
                if let Some(t) = succ(body, blk, k) {
                    for o in body.target_args(&t) {
                        mark(o, &mut self.seen);
                    }
                }
            }
            for k in bd.insts.indices().rev() {
                if let Some(Some(r)) = body.insts.result.get(k as usize)
                    && let Some(u) = self.unused.get_mut(r.index())
                {
                    *u = self.seen[r.index()] != stamp;
                }
                let span = body.insts.args[k as usize];
                for oi in span.indices().rev() {
                    if let Some(Operand::Value(v)) = body.operands.get(oi as usize)
                        && let Some(s) = self.seen.get_mut(v.index())
                    {
                        self.last_use[oi as usize] = *s != stamp;
                        *s = stamp;
                    }
                }
            }
        }
    }

    fn frame(&mut self, cg: &Codegen<'_>, f: &Func, body: &Body, nr: usize, scratch_need: u64) -> R<()> {
        // Outgoing stack arguments and slots.
        let mut out = 0u32;
        let mut slots = 0u64;
        let mut slot_list: Vec<(usize, u64)> = Vec::new();
        for &blk in &self.rpo {
            for k in body.blocks[blk as usize].insts.indices() {
                let args = body.args(k);
                match body.insts.op[k as usize] {
                    Op::Call => {
                        if let Some(Operand::Func(g)) = args.first()
                            && let Some(g) = cg.m.func(*g)
                        {
                            let n = abi::classify(g.params.iter().map(|t| cg.ty(*t)), &mut self.args);
                            out = out.max(n);
                        }
                    }
                    Op::CallInd => {
                        // the declared params, then the environment word
                        let env = std::iter::once(super::ty::Ty::Sc(super::ty::Sc::USIZE));
                        if let Some(Operand::Value(fv)) = args.first()
                            && let Some(vd) = body.values.get(fv.index())
                            && let Some(stave_ir::TypeKey::FnType { params, .. }) = cg.m.types.key(vd.ty)
                        {
                            let n =
                                abi::classify(params.iter().map(|t| cg.ty(*t)).chain(env), &mut self.args);
                            out = out.max(n);
                        } else if let Some(Operand::Func(g)) = args.first()
                            && let Some(g) = cg.m.func(*g)
                        {
                            let n =
                                abi::classify(g.params.iter().map(|t| cg.ty(*t)).chain(env), &mut self.args);
                            out = out.max(n);
                        }
                    }
                    Op::Intrinsic(id) => {
                        if let Some(Intr::Ccall(n)) = cg.intr.get(id.index()) {
                            out = out.max(u32::from(n.saturating_sub(8)) * 8);
                        }
                    }
                    Op::Slot => {
                        let Some(Some(r)) = body.insts.result.get(k as usize) else { continue };
                        let t = body.insts.ty[k as usize].ok_or_else(|| bad("slot without a type"))?;
                        let p =
                            super::ty::pointee(&cg.m.types, t).ok_or_else(|| bad("slot of a non-pointer"))?;
                        let l = cg.layouts.get_layout(p).map_err(|e| unsupported(format!("slot: {e}")))?;
                        if l.align > 16 || l.size > (1 << 30) {
                            return Err(unsupported("slot too large or over-aligned"));
                        }
                        slots = align_up64(slots, l.align);
                        slot_list.push((r.index(), slots));
                        slots += l.size;
                        if slots >= FRAME_MAX {
                            return Err(frame_too_large());
                        }
                    }
                    _ => {}
                }
            }
        }
        let out = align_up64(u64::from(out), 16);
        let slot_base = out;
        let scratch_off = align_up64(slot_base + slots, 16);
        let x8_off = align_up64(scratch_off + scratch_need, 16);
        let ret_mem = matches!(f.ret.map(|t| cg.ty(t)), Some(Ty::Mem { .. }));
        let homes_base = align_up64(x8_off + if ret_mem { 8 } else { 0 }, 16);
        if homes_base >= FRAME_MAX {
            return Err(frame_too_large());
        }
        self.scratch_off = scratch_off as u32;
        self.x8_off = x8_off as u32;
        let homes_base = homes_base as u32;
        let nv = body.values.len();
        self.home.clear();
        self.home.resize(nv, 0);
        for (v, off) in slot_list {
            self.home[v] = (slot_base + off) as u32;
        }
        // Linear scan over the intervals.
        self.bucket_start.clear();
        self.bucket_start.resize(nr + 1, 0);
        self.bucket_end.clear();
        self.bucket_end.resize(nr + 1, 0);
        let needs_home = |k: Vk| matches!(k, Vk::Gpr(_) | Vk::Fpr(_) | Vk::Mem(_));
        for v in 0..nv {
            if needs_home(self.kind[v]) {
                self.bucket_start[self.start[v] as usize + 1] += 1;
                self.bucket_end[self.end[v] as usize + 1] += 1;
            }
        }
        for i in 0..nr {
            self.bucket_start[i + 1] += self.bucket_start[i];
            self.bucket_end[i + 1] += self.bucket_end[i];
        }
        let total = self.bucket_start[nr] as usize;
        self.order.clear();
        self.order.resize(total, 0);
        self.order_end.clear();
        self.order_end.resize(total, 0);
        self.work.clear();
        self.work.extend_from_slice(&self.bucket_start[..nr]);
        self.seen.clear();
        self.seen.extend_from_slice(&self.bucket_end[..nr]);
        for v in 0..nv {
            if needs_home(self.kind[v]) {
                let s = &mut self.work[self.start[v] as usize];
                self.order[*s as usize] = v as u32;
                *s += 1;
                let e = &mut self.seen[self.end[v] as usize];
                self.order_end[*e as usize] = v as u32;
                *e += 1;
            }
        }
        self.free8.clear();
        self.free_big.clear();
        let mut top = 0u64;
        for i in 0..nr {
            for k in self.bucket_start[i]..self.bucket_start[i + 1] {
                let v = self.order[k as usize] as usize;
                let (size, align) = self.home_class(v);
                let off = if size == 0 {
                    0
                } else if size == 8 && align <= 8 {
                    match self.free8.pop() {
                        Some(o) => u64::from(o),
                        None => {
                            let o = top;
                            top += 8;
                            o
                        }
                    }
                } else {
                    let n = self.free_big.len();
                    let lo = n.saturating_sub(32);
                    match (lo..n).rev().find(|&j| self.free_big[j].0 == size && self.free_big[j].1 == align) {
                        Some(j) => u64::from(self.free_big.swap_remove(j).2),
                        None => {
                            let o = align_up64(top, u64::from(align));
                            top = o + u64::from(size);
                            o
                        }
                    }
                };
                if u64::from(homes_base) + top >= FRAME_MAX {
                    return Err(frame_too_large());
                }
                self.home[v] = homes_base + off as u32;
            }
            for k in self.bucket_end[i]..self.bucket_end[i + 1] {
                let v = self.order_end[k as usize] as usize;
                let (size, align) = self.home_class(v);
                let off = self.home[v] - homes_base;
                if size == 8 && align <= 8 {
                    self.free8.push(off);
                } else if size > 0 {
                    self.free_big.push((size, align, off));
                }
            }
        }
        let frame = align_up64(u64::from(homes_base) + top, 16);
        if frame >= FRAME_MAX {
            return Err(frame_too_large());
        }
        self.frame = frame as u32;
        Ok(())
    }

    fn home_class(&self, v: usize) -> (u32, u32) {
        match self.kind[v] {
            Vk::Mem(t) => match t {
                Ty::Mem { size, align } => (align_up(size, 8), align.max(8)),
                _ => (16, 8),
            },
            _ => (8, 8),
        }
    }
}
