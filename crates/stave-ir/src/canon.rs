//! The canonical form of a module: one numbering of functions, blocks, values, constants,
//! operands and intrinsic names, decided by a stated rule ([`canonicalize`]) and not by the order
//! in which lowering, check removal or the optimizer happened to build them. Two modules that
//! print the same text have, once canonical, equal [`Module`]s; code generation reads only
//! canonical modules, so its output does not depend on construction order.

use rustc_hash::FxHashMap;

use crate::ir::{
    Block, BlockData, Body, BorrowMark, Const, ConstId, EventMark, Func, FuncId, Insts, IntrinsicId, Module,
    Op, Operand, Site, Span, SwitchCase, Target, Term, Value, ValueData, ValueDef,
};
use crate::types::{DeclId, TypeId, TypeKey, TypeTable};

/// Rewrites `m` into canonical form and returns where each function went (`old -> new`).
///
/// The rule:
///
/// 1. **Functions** are in ascending [`crate::InstanceId`] order (lexicographic: a closure
///    `41.k` follows `41`, the unchecked entry `41.u` follows the closures of `41`), bodiless
///    functions included. Equal instance ids, which a lowered module does not have, keep their
///    stored order. Every `Operand::Func` is rewritten.
/// 2. **Per-function tables** of the module (`stated`, `clauses`) and `main` move with their
///    function. A table that is not empty is first extended to one row per function with the
///    row that a missing one means (`None`, the default clauses).
/// 3. **Intrinsic names** are the names the bodies use, sorted by bytes, without duplicates.
/// 4. **Blocks** of a body: those reachable from `b0`, in stored order, numbered densely from 0.
///    Unreachable blocks and their instructions are dropped.
/// 5. **Values**: dense from 0 in block order, per block its params, then its instruction
///    results (the printed `%N`). A value no kept block defines is dropped; one that is still
///    used (malformed input only) is numbered after the defined ones, in order of first use, and
///    has no definition.
/// 6. **Pools** are rebuilt in print order: block params per block; instructions per block;
///    operands per instruction, then the terminator's (condition or scrutinee, then each target's
///    args: `then`, `els`; the cases, then `default`); switch cases per terminator; and the
///    constant pool with one entry per distinct constant (kind and bits), in order of its first
///    operand in that same order (an inline condition, scrutinee, or return operand included).
///    Operands with equal constants share the entry: the bounds audit reads two constant
///    operands as the same integer when they name the same entry, so sharing must follow the
///    value, not which operands lowering happened to build from one constant.
/// 7. **Borrow side table** (`borrows`, `events`): stored order, values renumbered, marks of
///    dropped values dropped.
/// 8. **Types** (checked-ir.md "Ids"): the *P* set, the closure over key components of the
///    types the functions with a body use after rule 4 (signatures and type arguments, values,
///    instruction results, type operands), ranked in canonical type-key order
///    ([`crate::types::TypeTable::canonical_cmp`]); then the *S* set, the other types that the
///    shapes of kept ADT types, the statics, and the bodiless functions' signatures reach over
///    key components and shapes, ranked the same way. No other type is kept (the verifier names
///    the scalar type of a constant operand by key, so it needs no type no value has), and every
///    type id of the module is rewritten. `decls` keeps the decls a kept `adt`
///    key or `cell` names. A type table that is not well formed is left as it is.
///
/// Strings, sites and statics are canonical as lowered (ranked by contracted keys) and are not
/// changed. Callers that hold other per-function tables move them with the result
/// ([`permute_by_func`]); nothing outside the module holds a type id across this pass. The pass
/// is idempotent.
pub fn canonicalize(m: &mut Module) -> Vec<FuncId> {
    canonicalize_by(m, |funcs, one| funcs.iter_mut().for_each(one))
}

/// [`canonicalize`] with the functions rewritten by `each`: each call must call the given
/// function once on every element of the slice, in any order or in parallel (functions are
/// independent).
pub fn canonicalize_by<E>(m: &mut Module, each: E) -> Vec<FuncId>
where
    E: Fn(&mut [Func], &(dyn Fn(&mut Func) + Sync)),
{
    let n = m.funcs.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| m.funcs[a].instance.cmp(&m.funcs[b].instance));
    let mut remap = vec![FuncId(0); n];
    for (new, &old) in order.iter().enumerate() {
        remap[old] = FuncId(new as u32);
    }

    // Only reachable blocks count: the body rewrite drops the others, so a name that only dead
    // code uses would be kept by one pass and dropped by the next.
    let mut used: Vec<&str> = Vec::new();
    for b in m.funcs.iter().filter_map(|f| f.body.as_ref()) {
        let reach = b.reachable();
        let kept = b.blocks.iter().zip(&reach).filter(|&(_, &r)| r).map(|(bd, _)| bd);
        for i in kept.flat_map(|bd| bd.insts.indices()) {
            if let Some(Op::Intrinsic(id)) = b.insts.op.get(i as usize)
                && let Some(name) = m.intrinsic_name(*id)
            {
                used.push(name);
            }
        }
    }
    used.sort_unstable();
    used.dedup();
    let names: Vec<String> = used.into_iter().map(str::to_string).collect();
    let intrinsics: Vec<IntrinsicId> = m
        .intrinsics
        .iter()
        .map(|s| names.binary_search(s).map_or(IntrinsicId(u32::MAX), |i| IntrinsicId(i as u32)))
        .collect();

    let mut old: Vec<Option<Func>> = std::mem::take(&mut m.funcs).into_iter().map(Some).collect();
    m.funcs = order.iter().filter_map(|&i| old[i].take()).collect();
    if !m.stated.is_empty() {
        permute_by_func(&mut m.stated, &remap);
    }
    if !m.clauses.is_empty() {
        permute_by_func(&mut m.clauses, &remap);
    }
    m.intrinsics = names;
    m.main = m.main.map(|f| remap.get(f.index()).copied().unwrap_or(FuncId(u32::MAX)));

    let maps = Maps { funcs: &remap, intrinsics: &intrinsics };
    each(&mut m.funcs, &|f: &mut Func| {
        if let Some(b) = &f.body {
            f.body = Some(canonical_body(b, &maps));
        }
    });
    canonical_types(m, &each);
    remap
}

/// Rule 8 of [`canonicalize`], on a module whose bodies are already canonical (only reachable
/// blocks count).
fn canonical_types<E>(m: &mut Module, each: &E)
where
    E: Fn(&mut [Func], &(dyn Fn(&mut Func) + Sync)),
{
    if m.types.check_well_formed().is_err() {
        return;
    }
    let n = m.types.len();
    let mut seeds: Vec<TypeId> = Vec::new();
    for f in &m.funcs {
        let Some(b) = &f.body else { continue };
        seeds.extend(f.signature_types());
        seeds.extend(b.values.iter().map(|v| v.ty));
        seeds.extend(b.insts.ty.iter().flatten());
        seeds.extend(b.operands.iter().filter_map(|o| match o {
            Operand::Type(t) => Some(*t),
            _ => None,
        }));
    }
    let mut in_p = vec![false; n];
    close_types(&m.types, &mut seeds, &mut in_p, false);
    let mut in_all = vec![false; n];
    seeds.extend((0..n as u32).map(TypeId).filter(|t| in_p[t.index()]));
    seeds.extend(m.statics.iter().map(|s| s.ty));
    for f in m.funcs.iter().filter(|f| f.body.is_none()) {
        seeds.extend(f.signature_types());
    }
    close_types(&m.types, &mut seeds, &mut in_all, true);

    let types = &m.types;
    let cmp = |a: &TypeId, b: &TypeId| types.canonical_cmp(*a, *b).unwrap_or(std::cmp::Ordering::Equal);
    let mut order: Vec<TypeId> = (0..n as u32).map(TypeId).filter(|t| in_p[t.index()]).collect();
    order.sort_by(cmp);
    let mut s: Vec<TypeId> =
        (0..n as u32).map(TypeId).filter(|t| in_all[t.index()] && !in_p[t.index()]).collect();
    s.sort_by(cmp);
    order.extend(s);
    if order.len() != n || order.iter().enumerate().any(|(k, t)| t.index() != k) {
        let remap = m.types.keep_in_order(&order);
        let map = |t: TypeId| remap.get(t.index()).copied().unwrap_or(TypeId(u32::MAX));
        each(&mut m.funcs, &|f: &mut Func| f.map_types(map));
        for st in &mut m.statics {
            st.ty = map(st.ty);
        }
    }
    let mut named: Vec<DeclId> = m.cell.into_iter().collect();
    for k in m.types.keys() {
        if let TypeKey::Adt { def, .. } = k {
            named.push(*def);
        }
    }
    named.sort_unstable();
    m.decls.retain(|d| named.binary_search(&d.id).is_ok());
}

/// Marks in `set` every type `seeds` reach over key components, and with `shapes` over the
/// fields of ADT shapes too (empties `seeds`). Ids out of range are ignored.
fn close_types(types: &TypeTable, seeds: &mut Vec<TypeId>, set: &mut [bool], shapes: bool) {
    while let Some(t) = seeds.pop() {
        match set.get_mut(t.index()) {
            Some(seen) if !*seen => *seen = true,
            _ => continue,
        }
        if let Some(k) = types.key(t) {
            k.for_each_component(|c| seeds.push(c));
        }
        if shapes && let Some(sh) = types.shape(t) {
            seeds.extend(sh.variants.iter().flatten().copied());
        }
    }
}

/// Whether `m` is its own canonical form ([`canonicalize`] changes nothing).
pub fn is_canonical(m: &Module) -> bool {
    let mut c = m.clone();
    canonicalize(&mut c);
    c == *m
}

/// Moves the rows of a per-function table with their functions: row `i` goes to `remap[i]`.
/// The table is first extended to `remap.len()` rows with `T::default()` (the row a missing one
/// means); rows past `remap.len()` are dropped.
pub fn permute_by_func<T: Default>(table: &mut Vec<T>, remap: &[FuncId]) {
    table.resize_with(remap.len().max(table.len()), T::default);
    table.truncate(remap.len());
    let mut old: Vec<Option<T>> = std::mem::take(table).into_iter().map(Some).collect();
    let mut new: Vec<Option<T>> = (0..remap.len()).map(|_| None).collect();
    for (i, row) in old.iter_mut().enumerate() {
        if let Some(slot) = remap.get(i).and_then(|f| new.get_mut(f.index())) {
            *slot = row.take();
        }
    }
    *table = new.into_iter().map(Option::unwrap_or_default).collect();
}

/// Whether `remap` (of [`canonicalize`]) left every function where it was.
pub fn is_identity(remap: &[FuncId]) -> bool {
    remap.iter().enumerate().all(|(i, f)| f.index() == i)
}

struct Maps<'a> {
    funcs: &'a [FuncId],
    intrinsics: &'a [IntrinsicId],
}

/// The canonical rewrite of one body (rules 4 to 7 of [`canonicalize`]).
fn canonical_body(b: &Body, maps: &Maps<'_>) -> Body {
    let reach = b.reachable();
    let kept: Vec<usize> = (0..b.blocks.len()).filter(|&i| reach[i]).collect();
    let mut block_num = vec![u32::MAX; b.blocks.len()];
    for (n, &bi) in kept.iter().enumerate() {
        block_num[bi] = n as u32;
    }

    // Values defined by kept blocks, in print order.
    let mut value_num = vec![u32::MAX; b.values.len()];
    let mut values: Vec<ValueData> = Vec::new();
    for &bi in &kept {
        let bd = &b.blocks[bi];
        let results = bd.insts.indices().filter_map(|i| b.insts.result.get(i as usize).copied().flatten());
        for v in bd.params.of(&b.block_params).iter().copied().chain(results) {
            if let Some(slot) = value_num.get_mut(v.index())
                && *slot == u32::MAX
            {
                *slot = values.len() as u32;
                values.push(ValueData { ty: b.values[v.index()].ty, def: ValueDef::Inst(u32::MAX) });
            }
        }
    }

    let mut cx = Rewrite {
        b,
        maps,
        block_num: &block_num,
        value_num,
        out: Body::default(),
        values,
        const_ids: FxHashMap::default(),
    };
    let mut blocks = Vec::with_capacity(kept.len());
    for &bi in &kept {
        let bd = b.blocks[bi];
        let nb = blocks.len() as u32;
        let params = bd.params.of(&b.block_params);
        let pspan = Span { start: cx.out.block_params.len() as u32, len: params.len() as u32 };
        for (j, &v) in params.iter().enumerate() {
            let nv = cx.value(v);
            cx.out.block_params.push(nv);
            cx.define(nv, ValueDef::Param { block: Block(nb), index: j as u32 });
        }
        let start = cx.out.insts.len() as u32;
        for i in bd.insts.indices() {
            cx.inst(i as usize);
        }
        let ispan = Span { start, len: cx.out.insts.len() as u32 - start };
        let term = cx.term(bd.term);
        blocks.push(BlockData { params: pspan, insts: ispan, term });
    }
    let Rewrite { value_num, mut out, values, .. } = cx;
    out.blocks = blocks;
    out.values = values;
    let num = |v: Value| value_num.get(v.index()).copied().filter(|&n| n != u32::MAX).map(Value);
    out.borrows =
        b.borrows.iter().filter_map(|x| num(x.value).map(|value| BorrowMark { value, ..*x })).collect();
    out.events =
        b.events.iter().filter_map(|x| num(x.value).map(|value| EventMark { value, ..*x })).collect();
    out
}

struct Rewrite<'a> {
    b: &'a Body,
    maps: &'a Maps<'a>,
    block_num: &'a [u32],
    value_num: Vec<u32>,
    /// The body being built; its `values` are kept apart until the end.
    out: Body,
    values: Vec<ValueData>,
    /// The new pool entry of each constant (by [`const_key`]).
    const_ids: FxHashMap<(u8, u128), u32>,
}

/// A constant's kind and bits: equal keys are equal constants.
fn const_key(c: Const) -> (u8, u128) {
    match c {
        Const::Int(x) => (0, x as u128),
        Const::F16(x) => (1, u128::from(x)),
        Const::BF16(x) => (2, u128::from(x)),
        Const::F32(x) => (3, u128::from(x)),
        Const::F64(x) => (4, u128::from(x)),
    }
}

impl Rewrite<'_> {
    /// The new number of `v`; a value no kept block defines gets the next one, without a
    /// definition.
    fn value(&mut self, v: Value) -> Value {
        match self.value_num.get(v.index()).copied() {
            Some(n) if n != u32::MAX => Value(n),
            known => {
                let n = self.values.len() as u32;
                let ty = self.b.values.get(v.index()).map_or(TypeId(u32::MAX), |d| d.ty);
                self.values.push(ValueData { ty, def: ValueDef::Inst(u32::MAX) });
                if known.is_some() {
                    self.value_num[v.index()] = n;
                }
                Value(n)
            }
        }
    }

    /// Records the first definition of new value `v`.
    fn define(&mut self, v: Value, def: ValueDef) {
        if let Some(d) = self.values.get_mut(v.index())
            && d.def == ValueDef::Inst(u32::MAX)
        {
            d.def = def;
        }
    }

    fn operand(&mut self, o: Operand) -> Operand {
        match o {
            Operand::Value(v) => Operand::Value(self.value(v)),
            Operand::Const(c) => match self.b.konst(c) {
                Some(k) => {
                    let next = self.out.consts.len() as u32;
                    let id = *self.const_ids.entry(const_key(k)).or_insert(next);
                    if id == next {
                        self.out.consts.push(k);
                    }
                    Operand::Const(ConstId(id))
                }
                None => Operand::Const(ConstId(u32::MAX)),
            },
            Operand::Func(f) => {
                Operand::Func(self.maps.funcs.get(f.index()).copied().unwrap_or(FuncId(u32::MAX)))
            }
            Operand::Str(_) | Operand::Type(_) => o,
        }
    }

    fn operands(&mut self, span: Span) -> Span {
        let start = self.out.operands.len() as u32;
        for &o in span.of(&self.b.operands) {
            let n = self.operand(o);
            self.out.operands.push(n);
        }
        Span { start, len: self.out.operands.len() as u32 - start }
    }

    fn inst(&mut self, i: usize) {
        let b = self.b;
        // A span past the instruction arrays (malformed input only) names no instruction.
        let op = match b.insts.op.get(i).copied() {
            Some(Op::Intrinsic(id)) => {
                Op::Intrinsic(self.maps.intrinsics.get(id.index()).copied().unwrap_or(IntrinsicId(u32::MAX)))
            }
            Some(op) => op,
            None => return,
        };
        let args = self.operands(b.insts.args.get(i).copied().unwrap_or_default());
        let result = b.insts.result.get(i).copied().flatten().map(|v| self.value(v));
        let idx = self.out.insts.len() as u32;
        if let Some(v) = result {
            self.define(v, ValueDef::Inst(idx));
        }
        let insts: &mut Insts = &mut self.out.insts;
        insts.op.push(op);
        insts.ty.push(b.insts.ty.get(i).copied().flatten());
        insts.result.push(result);
        insts.args.push(args);
        insts.site.push(b.insts.site.get(i).copied().unwrap_or(Site::NONE));
    }

    fn target(&mut self, t: Target) -> Target {
        let block = Block(self.block_num.get(t.block.index()).copied().unwrap_or(u32::MAX));
        Target { block, args: self.operands(t.args) }
    }

    fn term(&mut self, t: Term) -> Term {
        match t {
            Term::Br(t) => Term::Br(self.target(t)),
            Term::Cbr { cond, then, els } => {
                let cond = self.operand(cond);
                let then = self.target(then);
                Term::Cbr { cond, then, els: self.target(els) }
            }
            Term::Switch { scrut, cases, default } => {
                let scrut = self.operand(scrut);
                let mut new: Vec<SwitchCase> = Vec::with_capacity(cases.len as usize);
                for c in cases.of(&self.b.cases) {
                    new.push(SwitchCase { value: c.value, target: self.target(c.target) });
                }
                let default = self.target(default);
                let span = Span { start: self.out.cases.len() as u32, len: new.len() as u32 };
                self.out.cases.extend(new);
                Term::Switch { scrut, cases: span, default }
            }
            Term::Ret(o) => Term::Ret(o.map(|o| self.operand(o))),
            Term::Unreachable => Term::Unreachable,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::canonicalize;
    use crate::{FuncBuilder, InstanceId, Module, Op};

    #[test]
    fn a_name_only_dead_code_uses_is_dropped_by_the_first_pass() {
        let mut m = Module::new();
        let i32t = m.types.i32();
        let us = m.types.usize();
        let live = m.intrinsic("thread_start_addr");
        let dead_only = m.intrinsic("stack_table");
        let f = m.declare(InstanceId(vec![0]), "app.f", vec![i32t], Some(i32t));
        let mut b = FuncBuilder::new(&[i32t]);
        let x = b.param(0).unwrap();
        let dead = b.create_block();
        b.intrinsic(live, Some(us), &[]);
        b.ret(Some(x.into()));
        b.switch_to(dead);
        b.intrinsic(dead_only, Some(us), &[]);
        b.ret(Some(x.into()));
        m.define(f, b.finish().unwrap());

        let mut once = m.clone();
        canonicalize(&mut once);
        assert_eq!(once.intrinsics, ["thread_start_addr"]);
        let mut twice = once.clone();
        canonicalize(&mut twice);
        assert!(once == twice);
        let body = once.funcs[0].body.as_ref().unwrap();
        assert!(body.insts.op.iter().any(|op| matches!(op, Op::Intrinsic(id) if id.0 == 0)));
    }
}
