//! The image naming and order rules both backends share.

mod common;

use common::*;
use stave_ir::link::{func_symbols, is_reserved, split_file, static_symbol, text_order};
use stave_ir::{FuncId, InstanceId, Module, Static};

#[test]
fn order_and_names() {
    let mut m = Module::new();
    let u8t = m.types.u8();
    let fns: [(&[u32], &str); 5] =
        [(&[3], "a.f"), (&[1], "a.f"), (&[2], "_start"), (&[0], "__stave_main"), (&[1, 1], "a.g")];
    let ids: Vec<FuncId> =
        fns.iter().map(|(i, p)| define(&mut m, i, p, &[], None, |b, _| b.ret(None))).collect();
    let undefined = m.declare(InstanceId(vec![9]), "a.h", vec![], None);
    let order = text_order(&m);
    assert_eq!(order, [ids[3], ids[1], ids[4], ids[2], ids[0]]);
    let names = func_symbols(&m, &order);
    assert_eq!(names[ids[0].index()].as_deref(), Some("a.f@3"));
    assert_eq!(names[ids[1].index()].as_deref(), Some("a.f@1"));
    assert_eq!(names[ids[2].index()].as_deref(), Some("_start@2"));
    assert_eq!(names[ids[3].index()].as_deref(), Some("__stave_main@0"));
    assert_eq!(names[ids[4].index()].as_deref(), Some("a.g"));
    assert_eq!(names[undefined.index()], None);
    assert!(is_reserved("__stave_fmod") && is_reserved("_start") && !is_reserved("start"));
    for (path, init) in [("a.X", vec![1]), ("", vec![0]), ("a.f", vec![0]), ("__stave_strings", vec![0])] {
        m.statics.push(Static { path: path.into(), ty: u8t, init, refs: Vec::new() });
    }
    let taken = |p: &str| p == "a.f";
    let s: Vec<String> = (0..4).map(|i| static_symbol(&m, i, taken)).collect();
    assert_eq!(s, ["a.X", "__stave_static.1", "__stave_static.2", "__stave_static.3"]);
    assert_eq!(split_file("src/a/b.stv"), ("src/a", "b.stv"));
    assert_eq!(split_file("b.stv"), (".", "b.stv"));
}
