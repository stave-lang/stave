//! The canonical form (`stave_ir::canon`): idempotent, independent of the stored function order,
//! invisible to the printer and the interpreter, and valid IR.

mod common;

use std::sync::Arc;

use common::*;
use stave_ir::canon::{canonicalize, is_identity, permute_by_func};
use stave_ir::interp::Val;
use stave_ir::ir::ValueDef;
use stave_ir::witness::Summary;
use stave_ir::{Block, ClauseKind, Clauses, FuncId, Module, Op, Operand, Term, print_checked, verify_module};

/// Four functions stored out of instance order (`@2`, `@0`, `@0.u`, `@1`), with bodies whose
/// stored numbering is not the printed one: a block param made before the entry's results, an
/// unreachable block, a constant shared by two operands, equal constants built separately, a
/// switch, an intrinsic name no body
/// uses before the used one, and the per-function tables set.
fn module() -> (Module, FuncId) {
    let mut m = Module::new();
    let (i32t, bt) = (m.types.i32(), m.types.bool());
    m.intrinsic("unused");
    let tsa = m.intrinsic("thread_start_addr");
    let us = m.types.usize();
    let g = define(&mut m, &[2], "app.g", &[i32t], Some(i32t), |b, _| {
        let x = b.param(0).unwrap();
        let (join, r) = b.create_block_with(&[i32t]);
        let (other, _) = b.create_block_with(&[i32t]);
        let dead = b.create_block();
        let seven = b.int(7);
        let y = b.binary(Op::AddWrap, i32t, x, seven);
        b.switch(y, &[(8, join, &[seven]), (9, other, &[y.into()])], join, &[y.into()]);
        b.switch_to(other);
        b.intrinsic(tsa, Some(us), &[]);
        let again = b.int(7);
        b.br(join, &[again]);
        b.switch_to(dead);
        let z = b.binary(Op::MulWrap, i32t, x, x);
        b.br(join, &[z.into()]);
        b.switch_to(join);
        b.ret(Some(r[0].into()));
    });
    let f = define(&mut m, &[0], "app.f", &[i32t], Some(i32t), |b, _| {
        let x = b.param(0).unwrap();
        let (then, t) = b.create_block_with(&[i32t]);
        let els = b.create_block();
        let one = b.int(1);
        let c = b.cmp(Op::Lt, bt, x, one);
        b.cbr(c, then, &[one], els, &[]);
        b.switch_to(els);
        let r = b.call(g, Some(i32t), &[x.into()]).unwrap();
        b.br(then, &[r.into()]);
        b.switch_to(then);
        b.ret(Some(t[0].into()));
    });
    let u = define(&mut m, &[0, u32::MAX], "app.f", &[i32t], Some(i32t), |b, _| {
        let x = b.param(0).unwrap();
        let r = b.call(g, Some(i32t), &[x.into()]).unwrap();
        b.ret(Some(r.into()));
    });
    let main = define(&mut m, &[1], "app.main", &[], Some(i32t), |b, _| {
        let a = b.int(3);
        let r = b.call(f, Some(i32t), &[a]).unwrap();
        let s = b.call(u, Some(i32t), &[r.into()]).unwrap();
        b.ret(Some(s.into()));
    });
    m.stated = (0..m.funcs.len()).map(|i| (i % 2 == 0).then(|| Arc::new(Summary::default()))).collect();
    m.clauses = vec![Clauses { kind: ClauseKind::Item, ..Clauses::default() }];
    (m, main)
}

fn canonical(mut m: Module) -> Module {
    canonicalize(&mut m);
    m
}

/// `m` with its functions stored in `order` (new position -> old id), every reference moved.
fn permuted(m: &Module, order: &[usize]) -> Module {
    let mut remap = vec![FuncId(0); order.len()];
    for (new, &old) in order.iter().enumerate() {
        remap[old] = FuncId(new as u32);
    }
    let mut p = m.clone();
    p.funcs = order.iter().map(|&i| m.funcs[i].clone()).collect();
    for f in &mut p.funcs {
        for o in f.body.iter_mut().flat_map(|b| &mut b.operands) {
            if let Operand::Func(g) = o {
                *g = remap[g.index()];
            }
        }
    }
    permute_by_func(&mut p.stated, &remap);
    permute_by_func(&mut p.clauses, &remap);
    p
}

#[test]
fn canonicalize_is_idempotent() {
    let (m, _) = module();
    let once = canonical(m);
    let mut twice = once.clone();
    assert!(is_identity(&canonicalize(&mut twice)));
    assert_eq!(once, twice);
}

#[test]
fn function_order_does_not_matter() {
    let (m, _) = module();
    let want = canonical(m.clone());
    for order in [[3, 2, 1, 0], [1, 0, 3, 2], [2, 3, 0, 1]] {
        assert_eq!(canonical(permuted(&m, &order)), want, "{order:?}");
    }
    let order: Vec<String> = want.funcs.iter().map(|f| f.instance.to_string()).collect();
    assert_eq!(order, ["0", "0.u", "1", "2"]);
}

#[test]
fn canonical_form_is_what_the_printer_shows() {
    let (m, main) = module();
    let mut c = m.clone();
    let remap = canonicalize(&mut c);
    verify_module(&m).unwrap_or_else(|e| panic!("{e:?}"));
    verify_module(&c).unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(print_checked(&c).unwrap(), print_checked(&m).unwrap());
    let before = ret_val(run(&m, main, vec![]));
    assert_eq!(ret_val(run(&c, remap[main.index()], vec![])), before);
    assert_eq!(before, Val::S(17));
    // the tables moved with their functions
    let row = |m: &Module, inst: &str| m.funcs.iter().position(|f| f.instance.to_string() == inst).unwrap();
    for inst in ["0", "0.u", "1", "2"] {
        assert_eq!(c.stated[row(&c, inst)], m.stated[row(&m, inst)], "{inst}");
    }
    assert_eq!(c.clauses.len(), c.funcs.len());
    assert_eq!(c.clauses[row(&c, "2")].kind, ClauseKind::Item);
    assert_eq!(c.intrinsics, ["thread_start_addr"]);
}

#[test]
fn bodies_are_numbered_in_print_order() {
    let (m, _) = module();
    let c = canonical(m);
    for f in &c.funcs {
        let b = f.body.as_ref().unwrap();
        assert!(b.reachable().iter().all(|&r| r), "{}: unreachable block kept", f.instance);
        let (mut values, mut params, mut ops) = (0u32, 0u32, 0u32);
        // every operand in print order: per block the instructions' args, then the terminator's
        let mut printed: Vec<Operand> = Vec::new();
        for (bi, bd) in b.blocks.iter().enumerate() {
            let blk = Block(bi as u32);
            assert_eq!(bd.params.start, params);
            params += bd.params.len;
            for (j, v) in bd.params.of(&b.block_params).iter().enumerate() {
                assert_eq!(
                    (v.0, b.values[v.index()].def),
                    (values, ValueDef::Param { block: blk, index: j as u32 })
                );
                values += 1;
            }
            for i in bd.insts.indices() {
                if let Some(v) = b.insts.result[i as usize] {
                    assert_eq!((v.0, b.values[v.index()].def), (values, ValueDef::Inst(i)));
                    values += 1;
                }
                assert_eq!(b.insts.args[i as usize].start, ops);
                ops += b.insts.args[i as usize].len;
                printed.extend_from_slice(b.args(i));
            }
            match bd.term {
                Term::Cbr { cond: o, .. } | Term::Switch { scrut: o, .. } | Term::Ret(Some(o)) => {
                    printed.push(o)
                }
                _ => {}
            }
            for t in b.successors(blk) {
                assert_eq!(t.args.start, ops);
                ops += t.args.len;
                printed.extend_from_slice(b.target_args(&t));
            }
        }
        assert_eq!((values as usize, ops as usize), (b.values.len(), b.operands.len()));
        let consts: Vec<u32> =
            printed.iter().filter_map(|o| if let Operand::Const(c) = o { Some(c.0) } else { None }).collect();
        // one entry per distinct constant, numbered by first use
        let mut first: Vec<u32> = Vec::new();
        for &c in &consts {
            if !first.contains(&c) {
                assert_eq!(c, first.len() as u32, "{}", f.instance);
                first.push(c);
            }
        }
        assert_eq!(first.len(), b.consts.len());
        for (i, x) in b.consts.iter().enumerate() {
            assert!(!b.consts[..i].contains(x), "{}: constant {x:?} twice", f.instance);
        }
    }
    // `@2` reads 7 three times and `@0` reads 1 twice: one entry each
    let pool = |inst: &str| {
        c.funcs.iter().find(|f| f.instance.to_string() == inst).unwrap().body.as_ref().unwrap().consts.len()
    };
    assert_eq!((pool("2"), pool("0")), (1, 1));
}

#[test]
fn a_short_table_is_extended_before_it_moves() {
    let mut t = vec![1u8, 2];
    permute_by_func(&mut t, &[FuncId(2), FuncId(0), FuncId(1)]);
    assert_eq!(t, [2, 0, 1]);
    let mut t: Vec<u8> = Vec::new();
    permute_by_func(&mut t, &[FuncId(1), FuncId(0)]);
    assert_eq!(t, [0, 0]);
}
