//! Canonical checked-IR printer (contracts/checked-ir.md).
//!
//! - `type` lines: every type the printed functions use (signatures, type args, value and result
//!   types, type operands), closed over key components, by id.
//! - Functions with a body, by canonical instance id (a closure `41.k` follows `41`). Bodiless
//!   functions (extern imports, `dispatch` fns) are not printed; calls name them `@<inst>`.
//! - Ids are the module's (contracts/checked-ir.md "Ids"): the lowering assigns `t<T>` (dense
//!   ranks of the printed types), `@<inst>`, and `s<S>`; the printer prints them as stored.
//! - Blocks: unreachable blocks are dropped; the rest keep their stored order and are renumbered
//!   densely from `b0` (the lowering creates blocks in the construct-table order).
//! - Values: `%0..` dense per function in block order: block params, then instruction results.
//! - Constants inline: ints in decimal (bools 0/1, chars as code points), floats as `0x` + their
//!   IEEE bits (4, 8 or 16 lowercase hex digits), strings `s<id>`, fn items `@<instance>`.
//! - Headers: `fn @<instance> <path>` plus `<..>` generic args (`t<id>` or a decimal const).
//!
//! Output is deterministic, `\n`-terminated lines, single spaces, no blank lines. The input should
//! verify; for malformed input the printer still never panics (unknown ids print as `%?`, `b?`).

use std::fmt::Write as _;

use crate::ir::{Body, Const, Func, Module, Op, Operand, Target, Term};
use crate::types::{GenericArg, TypeId};

/// Prints the module's checked IR.
pub fn print_checked(m: &Module) -> String {
    let mut funcs: Vec<(usize, &Func)> =
        m.funcs.iter().enumerate().filter(|(_, f)| f.body.is_some()).collect();
    funcs.sort_by(|a, b| a.1.instance.cmp(&b.1.instance).then(a.0.cmp(&b.0)));

    let layouts: Vec<FnLayout> = funcs.iter().map(|(_, f)| FnLayout::new(f)).collect();

    // Used types, closed over components.
    let nt = m.types.len();
    let mut used = vec![false; nt];
    let mut work: Vec<TypeId> = Vec::new();
    for ((_, f), lay) in funcs.iter().zip(&layouts) {
        work.extend(f.params.iter().chain(f.ret.iter()).copied());
        work.extend(f.type_args.iter().filter_map(|a| match a {
            GenericArg::Type(t) => Some(*t),
            GenericArg::Const(_) => None,
        }));
        if let Some(b) = &f.body {
            work.extend(lay.value_order.iter().filter_map(|v| b.value_ty(*v)));
            for &bi in &lay.block_order {
                if let Some(bd) = b.blocks.get(bi) {
                    for i in bd.insts.indices() {
                        if let Some(Some(t)) = b.insts.ty.get(i as usize) {
                            work.push(*t);
                        }
                        for o in b.args(i) {
                            if let Operand::Type(t) = o {
                                work.push(*t);
                            }
                        }
                    }
                }
            }
        }
    }
    while let Some(t) = work.pop() {
        match used.get_mut(t.index()) {
            Some(u) if !*u => {
                *u = true;
                if let Some(k) = m.types.key(t) {
                    k.for_each_component(|c| work.push(c));
                }
            }
            _ => {}
        }
    }

    let mut out = String::new();
    for (i, u) in used.iter().enumerate() {
        if *u {
            let _ = writeln!(out, "type t{} {}", i, m.types.key_text(TypeId(i as u32)));
        }
    }
    for ((_, f), lay) in funcs.iter().zip(&layouts) {
        print_func(m, f, lay, &mut out);
    }
    out
}

/// Printed numbering of one function: reachable blocks in order, values in order.
struct FnLayout {
    /// Stored block index per printed block number.
    block_order: Vec<usize>,
    /// Printed number per stored block (`u32::MAX` = dropped).
    block_num: Vec<u32>,
    value_order: Vec<crate::ir::Value>,
    /// Printed number per value (`u32::MAX` = not printed).
    value_num: Vec<u32>,
}

impl FnLayout {
    fn new(f: &Func) -> FnLayout {
        let Some(b) = &f.body else {
            return FnLayout {
                block_order: vec![],
                block_num: vec![],
                value_order: vec![],
                value_num: vec![],
            };
        };
        let reach = b.reachable();
        let block_order: Vec<usize> = (0..b.blocks.len()).filter(|&i| reach[i]).collect();
        let mut block_num = vec![u32::MAX; b.blocks.len()];
        for (n, &bi) in block_order.iter().enumerate() {
            block_num[bi] = n as u32;
        }
        let mut value_order = Vec::new();
        let mut value_num = vec![u32::MAX; b.values.len()];
        for &bi in &block_order {
            let bd = &b.blocks[bi];
            let results =
                bd.insts.indices().filter_map(|i| b.insts.result.get(i as usize).copied().flatten());
            for v in bd.params.of(&b.block_params).iter().copied().chain(results) {
                if let Some(slot) = value_num.get_mut(v.index())
                    && *slot == u32::MAX
                {
                    *slot = value_order.len() as u32;
                    value_order.push(v);
                }
            }
        }
        FnLayout { block_order, block_num, value_order, value_num }
    }

    fn val(&self, v: crate::ir::Value) -> String {
        match self.value_num.get(v.index()) {
            Some(&n) if n != u32::MAX => format!("%{n}"),
            _ => String::from("%?"),
        }
    }

    fn blk(&self, b: crate::ir::Block) -> String {
        match self.block_num.get(b.index()) {
            Some(&n) if n != u32::MAX => format!("b{n}"),
            _ => String::from("b?"),
        }
    }
}

fn operand(m: &Module, b: &Body, lay: &FnLayout, o: Operand) -> String {
    match o {
        Operand::Value(v) => lay.val(v),
        Operand::Const(c) => match b.konst(c) {
            Some(Const::Int(x)) => x.to_string(),
            Some(Const::F16(x) | Const::BF16(x)) => format!("0x{x:04x}"),
            Some(Const::F32(x)) => format!("0x{x:08x}"),
            Some(Const::F64(x)) => format!("0x{x:016x}"),
            None => String::from("?"),
        },
        Operand::Str(s) => format!("s{}", s.0),
        Operand::Func(f) => match m.func(f) {
            Some(g) => format!("@{}", g.instance),
            None => String::from("@?"),
        },
        Operand::Type(t) => format!("t{}", t.0),
    }
}

fn target(m: &Module, b: &Body, lay: &FnLayout, t: &Target) -> String {
    let args: Vec<String> = b.target_args(t).iter().map(|o| operand(m, b, lay, *o)).collect();
    format!("{}({})", lay.blk(t.block), args.join(", "))
}

fn print_func(m: &Module, f: &Func, lay: &FnLayout, out: &mut String) {
    let Some(b) = &f.body else { return };
    let _ = write!(out, "fn @{} {}", f.instance, f.path);
    if !f.type_args.is_empty() {
        let ts: Vec<String> = f
            .type_args
            .iter()
            .map(|a| match a {
                GenericArg::Type(t) => format!("t{}", t.0),
                GenericArg::Const(c) => c.to_string(),
            })
            .collect();
        let _ = write!(out, "<{}>", ts.join(","));
    }
    out.push_str(" {\n");
    for &bi in &lay.block_order {
        let Some(bd) = b.blocks.get(bi) else { continue };
        let params: Vec<String> = bd
            .params
            .of(&b.block_params)
            .iter()
            .map(|v| format!("{}: t{}", lay.val(*v), b.value_ty(*v).map_or(u32::MAX, |t| t.0)))
            .collect();
        let _ = writeln!(out, "b{}({}):", lay.block_num[bi], params.join(", "));
        for i in bd.insts.indices() {
            let iu = i as usize;
            let op = b.insts.op.get(iu).copied();
            out.push_str("  ");
            if let Some(Some(v)) = b.insts.result.get(iu) {
                let _ = write!(out, "{} = ", lay.val(*v));
            }
            match op {
                Some(Op::Intrinsic(id)) => {
                    let _ = write!(out, "intrinsic.{}", m.intrinsic_name(id).unwrap_or("?"));
                }
                Some(op) => out.push_str(op.name().unwrap_or("?")),
                None => out.push('?'),
            }
            match b.insts.ty.get(iu).copied().flatten() {
                Some(t) => {
                    let _ = write!(out, " t{}", t.0);
                }
                None => out.push_str(" ()"),
            }
            for o in b.args(i) {
                out.push(' ');
                out.push_str(&operand(m, b, lay, *o));
            }
            out.push('\n');
        }
        out.push_str("  ");
        match bd.term {
            Term::Br(t) => {
                let _ = write!(out, "br {}", target(m, b, lay, &t));
            }
            Term::Cbr { cond, then, els } => {
                let _ = write!(
                    out,
                    "cbr {} {} {}",
                    operand(m, b, lay, cond),
                    target(m, b, lay, &then),
                    target(m, b, lay, &els)
                );
            }
            Term::Switch { scrut, cases, default } => {
                let _ = write!(out, "switch {}", operand(m, b, lay, scrut));
                for c in b.switch_cases(cases) {
                    let _ = write!(out, " {}:{}", c.value, target(m, b, lay, &c.target));
                }
                let _ = write!(out, " _:{}", target(m, b, lay, &default));
            }
            Term::Ret(None) => out.push_str("ret"),
            Term::Ret(Some(o)) => {
                let _ = write!(out, "ret {}", operand(m, b, lay, o));
            }
            Term::Unreachable => out.push_str("unreachable"),
        }
        out.push('\n');
    }
    out.push_str("}\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FuncBuilder, InstanceId};

    /// A bodiless callee is not printed but named; types only of unreachable values are not listed.
    #[test]
    fn bodiless_callee_and_unreachable_types() {
        let mut m = Module::new();
        let i32t = m.types.i32();
        let f64t = m.types.f64();
        let ext = m.declare(InstanceId(vec![0]), "app.ext", vec![i32t], Some(i32t));
        let f = m.declare(InstanceId(vec![1]), "app.f", vec![i32t], Some(i32t));
        let mut b = FuncBuilder::new(&[i32t]);
        let x = b.param(0).unwrap();
        let r = b.call(ext, Some(i32t), &[x.into()]).unwrap();
        b.ret(Some(r.into()));
        let dead = b.create_block();
        b.switch_to(dead);
        let one = b.f64(1.0);
        let y = b.op(Op::Add, f64t, &[one, one]);
        let _ = y;
        b.unreachable();
        m.define(f, b.finish().unwrap());
        let got = print_checked(&m);
        assert_eq!(got, "type t0 int s 32\nfn @1 app.f {\nb0(%0: t0):\n  %1 = call t0 @0 %0\n  ret %1\n}\n");
    }
}
