//! Post-pass of one lowered body (contracts/checked-ir.md "SSA rule", "Order"): blocks unreachable
//! from `b0` are deleted, and join params whose value is never used (pruned SSA: a param is live
//! iff an instruction or terminator uses it, or it flows into a live param) are removed with their
//! branch arguments. Blocks keep their creation order (the construct-table preorder); values are
//! renumbered densely in block order (params, then results), as the printer numbers them.
//!
//! Also the id remapping of the final assembly ([`remap`]): types, fn ids, strings, intrinsics,
//! sites.

use stave_ir::ir::{BlockData, Insts, Span, SwitchCase, Target, ValueData, ValueDef};
use stave_ir::{Block, Body, Op, Operand, Site, Term, Value};

fn term_uses(t: &Term, out: &mut Vec<Operand>) {
    match t {
        Term::Cbr { cond, .. } => out.push(*cond),
        Term::Switch { scrut, .. } => out.push(*scrut),
        Term::Ret(Some(o)) => out.push(*o),
        _ => {}
    }
}

/// Deletes unreachable blocks and dead join params; renumbers values.
pub(crate) fn finalize(b: Body) -> Body {
    let nb = b.blocks.len();
    if nb == 0 {
        return b;
    }
    let reach = b.reachable();
    let nv = b.values.len();
    let mut useful = vec![false; nv];
    let mut ops = Vec::new();
    // direct uses
    for (bi, bd) in b.blocks.iter().enumerate() {
        if !reach[bi] {
            continue;
        }
        for i in bd.insts.indices() {
            for o in b.args(i) {
                if let Operand::Value(v) = o
                    && let Some(u) = useful.get_mut(v.index())
                {
                    *u = true;
                }
            }
        }
        ops.clear();
        term_uses(&bd.term, &mut ops);
        for o in &ops {
            if let Operand::Value(v) = o
                && let Some(u) = useful.get_mut(v.index())
            {
                *u = true;
            }
        }
    }
    // fn params are always kept
    for &p in b.params(Block(0)) {
        if let Some(u) = useful.get_mut(p.index()) {
            *u = true;
        }
    }
    // edges: (target param, operand)
    let mut edges: Vec<(Value, Value)> = Vec::new();
    for (bi, _) in b.blocks.iter().enumerate() {
        if !reach[bi] {
            continue;
        }
        b.for_each_successor(Block(bi as u32), |t| {
            let ps = b.params(t.block);
            for (k, o) in b.target_args(&t).iter().enumerate() {
                if let (Some(&p), Operand::Value(v)) = (ps.get(k), o) {
                    edges.push((p, *v));
                }
            }
        });
    }
    loop {
        let mut changed = false;
        for &(p, v) in &edges {
            if useful.get(p.index()).copied().unwrap_or(false)
                && let Some(u) = useful.get_mut(v.index())
                && !*u
            {
                *u = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // keep: reachable blocks; params that are useful
    let mut new_block = vec![u32::MAX; nb];
    let mut order = Vec::new();
    for (bi, r) in reach.iter().enumerate() {
        if *r {
            new_block[bi] = order.len() as u32;
            order.push(bi);
        }
    }
    let keep_param = |v: Value| useful.get(v.index()).copied().unwrap_or(false);
    // value renumbering
    let mut vmap = vec![u32::MAX; nv];
    // the new body is at most as large as the old one
    let mut values: Vec<ValueData> = Vec::with_capacity(nv);
    let mut block_params: Vec<Value> = Vec::with_capacity(b.block_params.len());
    let mut blocks: Vec<BlockData> = Vec::with_capacity(nb);
    let mut insts = Insts::default();
    let ni = b.insts.op.len();
    insts.op.reserve(ni);
    insts.ty.reserve(ni);
    insts.result.reserve(ni);
    insts.args.reserve(ni);
    insts.site.reserve(ni);
    let mut operands: Vec<Operand> = Vec::with_capacity(b.operands.len());
    let mut cases: Vec<SwitchCase> = Vec::new();
    // first pass: number values
    for &bi in &order {
        let bd = &b.blocks[bi];
        let mut k = 0u32;
        for &p in bd.params.of(&b.block_params) {
            if keep_param(p) {
                vmap[p.index()] = values.len() as u32;
                values.push(ValueData {
                    ty: b.values[p.index()].ty,
                    def: ValueDef::Param { block: Block(new_block[bi]), index: k },
                });
                k += 1;
            }
        }
        for i in bd.insts.indices() {
            if let Some(Some(v)) = b.insts.result.get(i as usize) {
                vmap[v.index()] = values.len() as u32;
                values.push(ValueData { ty: b.values[v.index()].ty, def: ValueDef::Inst(0) });
            }
        }
    }
    let mv = |o: Operand| -> Operand {
        match o {
            Operand::Value(v) => Operand::Value(Value(vmap.get(v.index()).copied().unwrap_or(u32::MAX))),
            o => o,
        }
    };
    let push_ops = |operands: &mut Vec<Operand>, xs: &[Operand]| -> Span {
        let start = operands.len() as u32;
        operands.extend(xs.iter().map(|&o| mv(o)));
        Span { start, len: xs.len() as u32 }
    };
    let target = |operands: &mut Vec<Operand>, t: &Target| -> Target {
        let ps = b.params(t.block);
        let start = operands.len() as u32;
        for (k, o) in b.target_args(t).iter().enumerate() {
            if ps.get(k).is_some_and(|&p| keep_param(p)) {
                operands.push(mv(*o));
            }
        }
        Target {
            block: Block(new_block.get(t.block.index()).copied().unwrap_or(u32::MAX)),
            args: Span { start, len: operands.len() as u32 - start },
        }
    };
    for &bi in &order {
        let bd = &b.blocks[bi];
        let pstart = block_params.len() as u32;
        for &p in bd.params.of(&b.block_params) {
            if keep_param(p) {
                block_params.push(Value(vmap[p.index()]));
            }
        }
        let params = Span { start: pstart, len: block_params.len() as u32 - pstart };
        let istart = insts.op.len() as u32;
        for i in bd.insts.indices() {
            let iu = i as usize;
            let idx = insts.op.len() as u32;
            let res = b.insts.result[iu].map(|v| Value(vmap[v.index()]));
            if let Some(r) = res
                && let Some(vd) = values.get_mut(r.index())
            {
                vd.def = ValueDef::Inst(idx);
            }
            let args = push_ops(&mut operands, b.args(i));
            insts.op.push(b.insts.op[iu]);
            insts.ty.push(b.insts.ty[iu]);
            insts.result.push(res);
            insts.args.push(args);
            insts.site.push(b.insts.site[iu]);
        }
        let ispan = Span { start: istart, len: insts.op.len() as u32 - istart };
        let term = match bd.term {
            Term::Br(t) => Term::Br(target(&mut operands, &t)),
            Term::Cbr { cond, then, els } => {
                let t = target(&mut operands, &then);
                let e = target(&mut operands, &els);
                Term::Cbr { cond: mv(cond), then: t, els: e }
            }
            Term::Switch { scrut, cases: cs, default } => {
                let old: Vec<SwitchCase> = b.switch_cases(cs).to_vec();
                let mut nc = Vec::with_capacity(old.len());
                for c in &old {
                    nc.push(SwitchCase { value: c.value, target: target(&mut operands, &c.target) });
                }
                let d = target(&mut operands, &default);
                let span = Span { start: cases.len() as u32, len: nc.len() as u32 };
                cases.extend(nc);
                Term::Switch { scrut: mv(scrut), cases: span, default: d }
            }
            Term::Ret(o) => Term::Ret(o.map(mv)),
            Term::Unreachable => Term::Unreachable,
        };
        blocks.push(BlockData { params, insts: ispan, term });
    }
    let borrows = b
        .borrows
        .iter()
        .filter_map(|m| {
            let v = vmap.get(m.value.index()).copied().filter(|&v| v != u32::MAX)?;
            Some(stave_ir::BorrowMark { value: Value(v), ..*m })
        })
        .collect();
    let events = b
        .events
        .iter()
        .filter_map(|m| {
            let v = vmap.get(m.value.index()).copied().filter(|&v| v != u32::MAX)?;
            Some(stave_ir::EventMark { value: Value(v), ..*m })
        })
        .collect();
    Body { blocks, block_params, insts, operands, consts: b.consts, cases, values, borrows, events }
}

/// Id maps applied to a body at assembly.
pub(crate) struct Remap<'a> {
    pub ty: &'a dyn Fn(u32) -> stave_ir::TypeId,
    pub func: &'a dyn Fn(u32) -> stave_ir::FuncId,
    pub string: &'a dyn Fn(u32) -> stave_ir::StrId,
    pub intrinsic: &'a dyn Fn(u32) -> stave_ir::IntrinsicId,
    pub site: &'a dyn Fn(u32) -> Site,
}

/// Rewrites every id of `b` through `m`.
pub(crate) fn remap(b: &mut Body, m: &Remap<'_>) {
    for v in &mut b.values {
        v.ty = (m.ty)(v.ty.0);
    }
    for t in b.insts.ty.iter_mut().flatten() {
        *t = (m.ty)(t.0);
    }
    for op in &mut b.insts.op {
        if let Op::Intrinsic(i) = op {
            *i = (m.intrinsic)(i.0);
        }
    }
    for s in &mut b.insts.site {
        if *s != Site::NONE {
            *s = (m.site)(s.0);
        }
    }
    let one = |o: &mut Operand| match o {
        Operand::Type(t) => *t = (m.ty)(t.0),
        Operand::Func(f) => *f = (m.func)(f.0),
        Operand::Str(s) => *s = (m.string)(s.0),
        _ => {}
    };
    for o in &mut b.operands {
        one(o);
    }
    for bd in &mut b.blocks {
        match &mut bd.term {
            Term::Cbr { cond: o, .. } | Term::Switch { scrut: o, .. } | Term::Ret(Some(o)) => one(o),
            _ => {}
        }
    }
}
