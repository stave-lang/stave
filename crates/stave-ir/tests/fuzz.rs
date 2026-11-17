//! Robustness: random (mostly malformed) IR never panics the verifier, printer, or interpreter,
//! and with a random witness never panics the witness checker;
//! random well-typed straight-line integer programs agree with a direct evaluation.

mod common;

use common::*;
use proptest::prelude::*;
use stave_ir::interp::{self, Config, Outcome, Val};
use stave_ir::ir::{BlockData, Insts, Span, Term, ValueData, ValueDef};
use stave_ir::witness::check::{check_func, remove};
use stave_ir::witness::{Edge, FnWitness, Node, Removed, Source};
use stave_ir::{Body, Const, FuncBuilder, Module, Op, Operand, PanicKind, TypeKey, Value};

const OPS: [Op; 24] = [
    Op::Add,
    Op::Sub,
    Op::Mul,
    Op::Div,
    Op::Rem,
    Op::Neg,
    Op::AddWrap,
    Op::SubWrap,
    Op::MulWrap,
    Op::ShlWrap,
    Op::AddSat,
    Op::SubSat,
    Op::MulSat,
    Op::Shl,
    Op::Shr,
    Op::And,
    Op::Or,
    Op::Xor,
    Op::Not,
    Op::Eq,
    Op::Lt,
    Op::Select,
    Op::Load,
    Op::Make,
];

fn arb_operand(nv: u32, nc: u32) -> impl Strategy<Value = Operand> {
    prop_oneof![
        (0..nv + 2).prop_map(|v| Operand::Value(Value(v))),
        (0..nc + 2).prop_map(|c| Operand::Const(stave_ir::ir::ConstId(c))),
        (0..3u32).prop_map(|s| Operand::Str(stave_ir::StrId(s))),
        (0..3u32).prop_map(|f| Operand::Func(stave_ir::FuncId(f))),
        (0..12u32).prop_map(|t| Operand::Type(stave_ir::TypeId(t))),
    ]
}

/// The ops a witness can name, for bodies the witness checker reads.
const WITNESS_OPS: [Op; 12] = [
    Op::BoundsCheck,
    Op::BoundsCheck,
    Op::SliceLen,
    Op::SliceMake,
    Op::Add,
    Op::Sub,
    Op::SubSat,
    Op::AddWrap,
    Op::Lt,
    Op::Eq,
    Op::Not,
    Op::Call,
];

/// A raw body with arbitrary ids and spans.
fn arb_body() -> impl Strategy<Value = Body> {
    arb_body_of(&OPS)
}

/// A witness with arbitrary rows, edges, and dominance table.
fn arb_witness() -> impl Strategy<Value = FnWitness> {
    let node = || {
        prop_oneof![
            Just(Node::Zero),
            (0..12u32).prop_map(|v| Node::Val(Value(v))),
            (0..12u32).prop_map(|v| Node::Len(Value(v))),
        ]
    };
    let src = (0..7u8, 0..14u32, 0..3u32).prop_map(|(kind, at, arg)| match kind {
        0 => Source::Branch { block: at },
        1 => Source::Assert { call: at, arg },
        2 => Source::Check { inst: at },
        3 => Source::Arith { inst: at },
        4 => Source::SliceLen { inst: at },
        5 => Source::SliceMake { inst: at },
        _ => Source::SubSat { inst: at },
    });
    let edge = (node(), node(), -3i128..4, src).prop_map(|(from, to, w, src)| Edge { from, to, w, src });
    let rows = prop::collection::vec((0..14u32, 0..6u32, 0..4u32), 0..4);
    (prop::collection::vec(0..7u32, 0..6), rows, prop::collection::vec(edge, 0..8)).prop_map(
        |(dom, rows, edges)| FnWitness {
            func: stave_ir::FuncId(0),
            dom,
            removed: rows
                .into_iter()
                .map(|(inst, start, len)| Removed {
                    inst,
                    site: stave_ir::Site::NONE,
                    edges: Span { start, len },
                })
                .collect(),
            edges,
        },
    )
}

fn arb_body_of(op_set: &'static [Op]) -> impl Strategy<Value = Body> {
    (1..5usize, 0..12usize, 0..10u32).prop_flat_map(move |(nb, ni, nv)| {
        let blocks = prop::collection::vec(
            (0..4u32, 0..3u32, 0..ni as u32 + 1, 0..3u32, 0..nb as u32 + 1, 0..5u32, any::<bool>()),
            nb,
        );
        let insts = prop::collection::vec(
            (0..op_set.len(), 0..12u32, any::<bool>(), 0..nv + 1, 0..20u32, 0..4u32),
            ni,
        );
        let ops = prop::collection::vec(arb_operand(nv, 6), 0..24);
        let consts = prop::collection::vec(any::<i64>(), 0..6);
        let values = prop::collection::vec((0..12u32, any::<bool>(), 0..12u32), nv as usize);
        (blocks, insts, ops, consts, values).prop_map(move |(blocks, insts, operands, consts, values)| Body {
            blocks: blocks
                .into_iter()
                .map(|(ps, pl, is, il, tb, ta, ret)| BlockData {
                    params: Span { start: ps, len: pl },
                    insts: Span { start: is, len: il },
                    term: if ret {
                        Term::Ret(Some(Operand::Value(Value(ta))))
                    } else {
                        Term::Br(stave_ir::ir::Target {
                            block: stave_ir::Block(tb),
                            args: Span { start: ta, len: 1 },
                        })
                    },
                })
                .collect(),
            block_params: (0..nv).map(Value).collect(),
            insts: Insts {
                op: insts.iter().map(|x| op_set[x.0]).collect(),
                ty: insts.iter().map(|x| if x.2 { Some(stave_ir::TypeId(x.1)) } else { None }).collect(),
                result: insts.iter().map(|x| Some(Value(x.3))).collect(),
                args: insts.iter().map(|x| Span { start: x.4, len: x.5 }).collect(),
                site: insts.iter().map(|_| stave_ir::Site::NONE).collect(),
            },
            operands,
            consts: consts.into_iter().map(|c| Const::Int(c as i128)).collect(),
            values: values
                .into_iter()
                .map(|(t, p, k)| ValueData {
                    ty: stave_ir::TypeId(t),
                    def: if p {
                        ValueDef::Param { block: stave_ir::Block(k % 3), index: k }
                    } else {
                        ValueDef::Inst(k)
                    },
                })
                .collect(),
            ..Body::default()
        })
    })
}

fn module_with(body: Body) -> Module {
    let mut m = Module::new();
    let t = &mut m.types;
    let i32t = t.i32();
    let u8t = t.u8();
    let bt = t.bool();
    let f64t = t.f64();
    let p = t.ref_(true, i32t);
    let pair = t.tuple(vec![i32t, u8t]);
    let _ = (bt, f64t, p, pair, t.usize(), t.unit(), t.i64(), t.str());
    m.add_string(b"x".to_vec());
    let f = m.declare(inst(&[0]), "t.f", vec![], Some(i32t));
    m.define(f, body);
    m.declare(inst(&[1]), "t.g", vec![i32t], Some(i32t));
    m
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 3000, ..ProptestConfig::default() })]

    #[test]
    fn random_bodies_never_panic(body in arb_body()) {
        let m = module_with(body);
        let _ = stave_ir::verify_module(&m);
        let _ = stave_ir::print_checked(&m);
        let cfg = Config { max_steps: 10_000, ..Config::default() };
        let _ = interp::run(&m, stave_ir::FuncId(0), vec![], host(), cfg);
    }

    #[test]
    fn random_witnesses_never_panic(body in arb_body_of(&WITNESS_OPS), w in arb_witness()) {
        let mut m = module_with(body);
        if let Ok(checked) = check_func(&m, &w) {
            remove(&mut m, &[checked]);
        }
    }
}

#[derive(Clone, Debug)]
enum Step {
    Bin(usize, usize, usize, i64),
}

fn int_key(k: usize) -> (TypeKey, u8, bool) {
    let all = [(8, true), (8, false), (16, true), (32, false), (32, true), (64, true), (64, false)];
    let (w, s) = all[k % all.len()];
    let bits = match w {
        8 => stave_ir::IntBits::B8,
        16 => stave_ir::IntBits::B16,
        32 => stave_ir::IntBits::B32,
        _ => stave_ir::IntBits::B64,
    };
    (TypeKey::Int { signed: s, bits }, w, s)
}

const BIN: [Op; 15] = [
    Op::Add,
    Op::Sub,
    Op::Mul,
    Op::Div,
    Op::Rem,
    Op::AddWrap,
    Op::SubWrap,
    Op::MulWrap,
    Op::AddSat,
    Op::SubSat,
    Op::MulSat,
    Op::And,
    Op::Or,
    Op::Xor,
    Op::ShlWrap,
];

/// Reference evaluation with Rust's own integer methods (independent of `arith`).
fn reference(op: Op, w: u8, s: bool, a: i128, b: i128) -> Result<i128, PanicKind> {
    macro_rules! go {
        ($t:ty) => {{
            let (x, y) = (a as $t, b as $t);
            let r: Option<$t> = match op {
                Op::Add => x.checked_add(y),
                Op::Sub => x.checked_sub(y),
                Op::Mul => x.checked_mul(y),
                Op::Div | Op::Rem => {
                    if y == 0 {
                        return Err(PanicKind::DivByZero);
                    }
                    if op == Op::Div { x.checked_div(y) } else { x.checked_rem(y) }
                }
                Op::AddWrap => Some(x.wrapping_add(y)),
                Op::SubWrap => Some(x.wrapping_sub(y)),
                Op::MulWrap => Some(x.wrapping_mul(y)),
                Op::AddSat => Some(x.saturating_add(y)),
                Op::SubSat => Some(x.saturating_sub(y)),
                Op::MulSat => Some(x.saturating_mul(y)),
                Op::And => Some(x & y),
                Op::Or => Some(x | y),
                Op::Xor => Some(x ^ y),
                _ => Some(x.wrapping_shl(y as u32)),
            };
            r.map(|v| v as i128).ok_or(PanicKind::Overflow)
        }};
    }
    match (w, s) {
        (8, true) => go!(i8),
        (8, false) => go!(u8),
        (16, true) => go!(i16),
        (16, false) => go!(u16),
        (32, true) => go!(i32),
        (32, false) => go!(u32),
        (64, true) => go!(i64),
        _ => go!(u64),
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 2000, ..ProptestConfig::default() })]

    /// A chain of binary ops of one int type over params and constants, checked against Rust.
    #[test]
    fn int_chain_matches_rust(
        k in 0..7usize,
        a in any::<i64>(),
        b in any::<i64>(),
        steps in prop::collection::vec((0..BIN.len(), 0..4usize, 0..4usize, any::<i64>()).prop_map(|(o, x, y, c)| Step::Bin(o, x, y, c)), 1..12),
    ) {
        let (key, w, s) = int_key(k);
        let it = stave_ir::arith::IntTy::new(w, s);
        let mut m = Module::new();
        let t = m.types.intern(key);
        let mut fb = FuncBuilder::new(&[t, t]);
        let mut vals: Vec<Operand> = vec![fb.param(0).unwrap().into(), fb.param(1).unwrap().into()];
        let mut env: Vec<i128> = vec![it.value(a as u64), it.value(b as u64)];
        let mut expect: Result<i128, PanicKind> = Ok(0);
        for Step::Bin(o, x, y, c) in &steps {
            let op = BIN[*o];
            let lhs = vals[x % vals.len()];
            let lv = env[x % env.len()];
            // operand y: a previous value or a constant of the type
            let (rhs, rv) = if y % 2 == 0 {
                (vals[y % vals.len()], env[y % env.len()])
            } else {
                let cv = it.value(*c as u64);
                (fb.int(cv), cv)
            };
            let r = fb.binary(op, t, lhs, rhs);
            let e = reference(op, w, s, lv, rv);
            match e {
                Ok(v) => {
                    vals.push(r.into());
                    env.push(it.value(v as u64));
                    expect = Ok(it.value(v as u64));
                }
                Err(kind) => {
                    expect = Err(kind);
                    vals.push(r.into());
                    break;
                }
            }
        }
        let last = *vals.last().unwrap();
        fb.ret(Some(last));
        let f = m.declare(inst(&[0]), "t.f", vec![t, t], Some(t));
        m.define(f, fb.finish().unwrap());
        let got = match run(&m, f, vec![Val::S(it.wrap(env[0])), Val::S(it.wrap(env[1]))]) {
            Outcome::Return(Some(v)) => Ok(it.value(v.bits().unwrap())),
            Outcome::Trap(t) => Err(t.kind),
            o => panic!("{o:?}"),
        };
        prop_assert_eq!(got, expect);
    }
}
