//! `read(print(m)) == m` on the values at the edges of the text: float constants (NaN payloads,
//! signed zeros, subnormals, infinities, in all four formats), the int limits of every width,
//! switch cases and claim weights at the `i128` limits, and paths, file names and symbols with
//! every byte a Rust `String` holds that the quoting rule must escape.

mod common;

use std::sync::Arc;

use common::*;
use stave_ir::text::{Header, Level, Mode, read};
use stave_ir::types::{AdtKind, DeclInfo};
use stave_ir::witness::{NO_BLOCK, Pre, Row, Summary, Term as RowTerm};
use stave_ir::{
    AdtShape, Const, DeclId, FloatBits, FuncBuilder, IntBits, Module, Op, Operand, Site, SiteInfo, TypeKey,
    canon, print_module,
};
use stave_target::{Triple, X86Level};

fn header() -> Header {
    Header { target: Triple::X86_64(X86Level::V2), mode: Mode::Release, level: Level::Checked }
}

/// Every byte 1 to 127 and a few multibyte chars: what the quoting must carry.
fn odd_path() -> String {
    let mut s: String = (1u8..=127).map(char::from).collect();
    s.push_str("\u{e9}\u{1f600}\u{0}");
    s
}

fn edge_module() -> Module {
    let mut m = Module::new();
    let b = m.types.bool();
    let ints: Vec<(stave_ir::TypeId, i128, i128)> = [
        (true, IntBits::B8),
        (true, IntBits::B16),
        (true, IntBits::B32),
        (true, IntBits::B64),
        (true, IntBits::Size),
        (false, IntBits::B8),
        (false, IntBits::B16),
        (false, IntBits::B32),
        (false, IntBits::B64),
        (false, IntBits::Size),
    ]
    .into_iter()
    .map(|(signed, bits)| {
        let w = if bits == IntBits::Size { 64 } else { bits.width() };
        let (lo, hi) =
            if signed { (-(1i128 << (w - 1)), (1i128 << (w - 1)) - 1) } else { (0, (1i128 << w) - 1) };
        (m.types.int(signed, bits), lo, hi)
    })
    .collect();
    let floats: Vec<(stave_ir::TypeId, Vec<Const>)> = vec![
        (
            m.types.intern(TypeKey::Float(FloatBits::F16)),
            [0x7e01, 0xfe00, 0x7c01, 0x8000, 0x0000, 0x0001, 0x03ff, 0x7c00, 0xfc00]
                .into_iter()
                .map(Const::F16)
                .collect(),
        ),
        (
            m.types.intern(TypeKey::Float(FloatBits::B16)),
            [0x7fc1, 0xffc0, 0x7f81, 0x8000, 0x0001, 0x007f, 0x7f80].into_iter().map(Const::BF16).collect(),
        ),
        (
            m.types.f32(),
            [
                0x7fc0_0001,
                0xffc0_0000,
                0x7f80_0001,
                0x8000_0000,
                0x0000_0001,
                0x007f_ffff,
                0x7f80_0000,
                0xff80_0000,
            ]
            .into_iter()
            .map(Const::F32)
            .collect(),
        ),
        (
            m.types.f64(),
            [
                0x7ff8_0000_0000_0001,
                0xfff8_0000_0000_0000,
                0x7ff0_0000_0000_0001,
                0x8000_0000_0000_0000,
                0x0000_0000_0000_0001,
                0x000f_ffff_ffff_ffff,
                0x7ff0_0000_0000_0000,
            ]
            .into_iter()
            .map(Const::F64)
            .collect(),
        ),
    ];
    let i64t = ints[3].0;
    let path = odd_path();
    m.decls = vec![DeclInfo { id: DeclId(u32::MAX - 1), kind: AdtKind::Struct, path: path.clone() }];
    let adt = m.types.adt(DeclId(u32::MAX - 1), vec![], AdtShape::structure(vec![i64t]));
    m.sites = vec![SiteInfo { file: path.clone(), line: u32::MAX, col: u32::MAX, func: path.clone() }];

    let f = m.declare(inst(&[0]), path.clone(), vec![i64t, adt], Some(i64t));
    {
        let fm = m.func_mut(f).unwrap();
        fm.name_site = Site(0);
    }
    let ext = m.declare(inst(&[u32::MAX - 1]), "app.ext", vec![], None);
    m.func_mut(ext).unwrap().symbol = Some(path.clone());

    let mut fb = FuncBuilder::new(&[i64t, adt]);
    let x = fb.param(0).unwrap();
    let yes = fb.int(1);
    for &(t, lo, hi) in &ints {
        let (a, c) = (fb.int(lo), fb.int(hi));
        fb.op(Op::Select, t, &[yes, a, c]);
    }
    for (t, cs) in &floats {
        for w in cs.windows(2) {
            let (a, c) = (fb.konst(w[0]), fb.konst(w[1]));
            fb.op(Op::Select, *t, &[yes, a, c]);
        }
    }
    let _ = b;
    let (b1, b2) = (fb.create_block(), fb.create_block());
    fb.switch(x, &[(i128::from(i64::MIN), b1, &[]), (i128::from(i64::MAX), b2, &[])], b2, &[]);
    fb.switch_to(b1);
    let k = fb.int(i128::from(i64::MIN));
    fb.ret(Some(k));
    fb.switch_to(b2);
    fb.ret(Some(Operand::Value(x)));
    m.define(f, fb.finish().unwrap());

    let row = |a, b, w| Row { a, b, w };
    m.stated = vec![
        Some(Arc::new(Summary {
            pre: vec![
                Pre { row: row(RowTerm::Param(0), RowTerm::Zero, i128::MAX), entry: NO_BLOCK, half: 0 },
                Pre { row: row(RowTerm::Zero, RowTerm::Param(0), i128::MIN), entry: 0, half: 1 },
            ],
            post: vec![row(RowTerm::Result, RowTerm::Param(0), i128::MIN)],
            ..Summary::default()
        })),
        None,
    ];
    canon::canonicalize(&mut m);
    m
}

#[test]
fn edge_values_round_trip() {
    let m = edge_module();
    let text = print_module(&m, &header(), None).unwrap();
    let l = read(text.as_bytes()).unwrap_or_else(|e| panic!("{e:?}\n{text}"));
    assert!(l.module == m, "read(print(m)) != m\n{text}");
    assert_eq!(print_module(&l.module, &l.header, None).unwrap(), text);
    for needle in [
        " 0x7e01",
        " bf16 0x7fc1",
        " 0x7fc00001",
        " 0x8000000000000000",
        " 0x0000000000000001",
        " -9223372036854775808",
        " 18446744073709551615",
        "170141183460469231731687303715884105727\n",
        "-170141183460469231731687303715884105728\n",
        "\\x00",
        "\\x7f",
        "\\xc3\\xa9",
        "\\xf0\\x9f\\x98\\x80",
        "\\\"",
        "\\\\",
        "\\n",
        "\\t",
        " 4294967295 4294967295 ",
    ] {
        assert!(text.contains(needle), "no `{needle}` in\n{text}");
    }
    // distinct NaN payloads and the two zeros are distinct pool entries
    let body = l.module.funcs[0].body.as_ref().unwrap();
    let f32s = body.consts.iter().filter(|c| matches!(c, Const::F32(_))).count();
    assert_eq!(f32s, 8, "{:?}", body.consts);
}

/// A path that is not UTF-8 cannot be a model path: the reader refuses it, located.
#[test]
fn a_path_that_is_not_utf8_is_refused() {
    let m = edge_module();
    let text = print_module(&m, &header(), None).unwrap();
    let bad = text.replacen("\\xc3\\xa9", "\\xc3\\x28", 1);
    assert_ne!(bad, text);
    let es = read(bad.as_bytes()).unwrap_err();
    assert!(es.iter().all(|e| e.line > 1), "{es:?}");
}
