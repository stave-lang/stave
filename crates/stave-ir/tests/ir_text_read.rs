//! The IR text reader (`stave_ir::text`): hand-written modules of every line kind and op group
//! load and verify; every rejection rule has an input and its message; mutated inputs never
//! panic.

use stave_ir::text::{EdgeKind, ErrorKind, Level, Loaded, Mode, ReadError, read};
use stave_ir::types::{AdtKind, FloatBits, GenericArg, IntBits, Lanes};
use stave_ir::{
    AdtShape, Const, DataRef, DataTarget, DeclId, FuncId, InstanceId, Op, Operand, Site, StrId, Term, TypeId,
    TypeKey, Value,
};
use stave_target::{Feature, FeatureSet, Triple};

fn render(es: &[ReadError]) -> String {
    es.iter().map(|e| e.render("t.ir")).collect::<Vec<_>>().join("\n")
}

fn load(src: &str) -> Loaded {
    match read(src.as_bytes()) {
        Ok(l) => l,
        Err(es) => panic!("rejected:\n{}\n--- input:\n{src}", render(&es)),
    }
}

/// `src` is rejected and the rendered errors contain `want`.
fn rejects(src: &str, want: &str) -> Vec<ReadError> {
    match read(src.as_bytes()) {
        Ok(_) => panic!("accepted, expected {want:?}:\n{src}"),
        Err(es) => {
            let r = render(&es);
            assert!(r.contains(want), "expected {want:?}, got:\n{r}\n--- input:\n{src}");
            es
        }
    }
}

const HEAD: &str = "stave.ir 4 \"x86_64-linux-v2\" debug checked\n";

fn with_head(body: &str) -> String {
    format!("{HEAD}{body}")
}

// ---------------------------------------------------------------------------------------------
// Accepted modules

const MAX: &str = "stave.ir 4 \"x86_64-linux-v2\" debug checked
type t0 int s 32
type t1 bool
site 0 \"src/app.stv\" 1 4 app.max
fn @0 app.max(t0, t0) -> t0 site 0 {
b0(%0: t0, %1: t0):
  %2 = gt t1 %0 %1
  cbr %2 b1() b2()
b1():
  ret %0
b2():
  ret %1
}
";

#[test]
fn the_contract_example_loads() {
    let l = load(MAX);
    assert_eq!(l.header.target, Triple::parse("x86_64-linux-v2").unwrap());
    assert_eq!((l.header.mode, l.header.level), (Mode::Debug, Level::Checked));
    let m = &l.module;
    assert_eq!(m.types.len(), 2);
    assert_eq!(m.types.key(TypeId(0)), Some(&TypeKey::Int { signed: true, bits: IntBits::B32 }));
    assert_eq!(m.sites.len(), 1);
    assert_eq!((m.sites[0].file.as_str(), m.sites[0].line, m.sites[0].col), ("src/app.stv", 1, 4));
    assert_eq!(m.sites[0].func, "app.max");
    let f = &m.funcs[0];
    assert_eq!(f.instance, InstanceId(vec![0]));
    assert_eq!(
        (f.path.as_str(), f.params.as_slice(), f.ret),
        ("app.max", &[TypeId(0), TypeId(0)][..], Some(TypeId(0)))
    );
    assert_eq!(f.name_site, Site(0));
    let b = f.body.as_ref().unwrap();
    assert_eq!(b.blocks.len(), 3);
    assert_eq!(b.values.len(), 3);
    assert_eq!(b.insts.op, vec![Op::Gt]);
    assert_eq!(b.insts.result, vec![Some(Value(2))]);
    assert!(matches!(b.blocks[0].term, Term::Cbr { cond: Operand::Value(Value(2)), .. }));
    assert_eq!(b.blocks[1].term, Term::Ret(Some(Operand::Value(Value(0)))));
    assert!(l.main.is_none() && l.stack.is_none());
    // claims are read at `checked`: one row per fn, none stated
    assert_eq!(m.stated.len(), 1);
    assert!(m.stated[0].is_none());
    assert_eq!(l.lines.funcs[0].header, 5);
    assert_eq!(l.lines.funcs[0].insts, vec![7]);
    assert_eq!(l.lines.funcs[0].terms, vec![8, 10, 12]);
}

#[test]
fn every_target_mode_and_level_of_the_header() {
    for t in Triple::ALL {
        for (mode, level) in [
            ("debug", Level::Lowered),
            ("debug", Level::Checked),
            ("release", Level::Lowered),
            ("release", Level::Checked),
            ("release", Level::Opt),
        ] {
            let lv = match level {
                Level::Lowered => "lowered",
                Level::Checked => "checked",
                Level::Opt => "opt",
            };
            let l = load(&format!("stave.ir 4 \"{}\" {mode} {lv}\n", t.name()));
            assert_eq!(l.header.target, t);
            assert_eq!(l.header.level, level);
            assert_eq!(l.header.mode, if mode == "debug" { Mode::Debug } else { Mode::Release });
            if level == Level::Lowered {
                assert!(l.module.stated.is_empty());
            }
        }
    }
}

const KEYS: &str = "stave.ir 4 \"aarch64-linux+sve\" release opt
decl d0 struct main.Point
decl d3 enum \"my-pkg.Opt\"
type t0 int s 32
type t1 int u size
type t2 float b16
type t3 float 64
type t4 bool
type t5 char
type t6 str
type t7 ref mut t0
type t8 ref t6
type t9 ptr const t0
type t10 ptr mut t0
type t11 array t0 16
type t12 slice t0 align 64
type t13 slice t0
type t14 tuple t0 t1
type t15 tuple
type t16 adt d0 t0 3
type t17 fntype t0 t1 -> t15 sve2
type t18 fntype -> t0
type t19 vec t3 4
type t20 vec t3 n
type t21 mask 32 8
type t22 mask 64 n
type t23 adt d3 t0
type t24 float 16
type t25 float 32
shape t16 struct t0 t1
shape t23 enum | | t0
";

#[test]
fn every_type_key_decl_and_shape() {
    let l = load(KEYS);
    let m = &l.module;
    let k = |i: u32| m.types.key(TypeId(i)).cloned().unwrap();
    assert_eq!(k(1), TypeKey::Int { signed: false, bits: IntBits::Size });
    assert_eq!(k(2), TypeKey::Float(FloatBits::B16));
    assert_eq!(k(3), TypeKey::Float(FloatBits::F64));
    assert_eq!(k(24), TypeKey::Float(FloatBits::F16));
    assert_eq!(k(25), TypeKey::Float(FloatBits::F32));
    assert_eq!(k(4), TypeKey::Bool);
    assert_eq!(k(5), TypeKey::Char);
    assert_eq!(k(6), TypeKey::Str);
    assert_eq!(k(7), TypeKey::Ref { mutable: true, inner: TypeId(0) });
    assert_eq!(k(8), TypeKey::Ref { mutable: false, inner: TypeId(6) });
    assert_eq!(k(9), TypeKey::RawPtr { mutable: false, inner: TypeId(0) });
    assert_eq!(k(10), TypeKey::RawPtr { mutable: true, inner: TypeId(0) });
    assert_eq!(k(11), TypeKey::Array { elem: TypeId(0), len: 16 });
    assert_eq!(k(12), TypeKey::Slice { elem: TypeId(0), align: Some(64) });
    assert_eq!(k(13), TypeKey::Slice { elem: TypeId(0), align: None });
    assert_eq!(k(14), TypeKey::Tuple(vec![TypeId(0), TypeId(1)]));
    assert_eq!(k(15), TypeKey::Tuple(vec![]));
    assert_eq!(
        k(16),
        TypeKey::Adt { def: DeclId(0), args: vec![GenericArg::Type(TypeId(0)), GenericArg::Const(3)] }
    );
    assert_eq!(
        k(17),
        TypeKey::FnType {
            params: vec![TypeId(0), TypeId(1)],
            ret: TypeId(15),
            features: FeatureSet::of(Feature::Sve2)
        }
    );
    assert_eq!(k(18), TypeKey::FnType { params: vec![], ret: TypeId(0), features: FeatureSet::EMPTY });
    assert_eq!(k(19), TypeKey::Vec { lane: TypeId(3), n: Lanes::Fixed(4) });
    assert_eq!(k(20), TypeKey::Vec { lane: TypeId(3), n: Lanes::Native });
    assert_eq!(k(21), TypeKey::Mask { width: 32, n: Lanes::Fixed(8) });
    assert_eq!(k(22), TypeKey::Mask { width: 64, n: Lanes::Native });
    assert_eq!(m.types.shape(TypeId(16)), Some(&AdtShape::structure(vec![TypeId(0), TypeId(1)])));
    assert_eq!(m.types.shape(TypeId(23)), Some(&AdtShape::enumeration(vec![vec![], vec![TypeId(0)]])));
    assert_eq!(m.decls.len(), 2);
    assert_eq!(
        (m.decls[1].id, m.decls[1].kind, m.decls[1].path.as_str()),
        (DeclId(3), AdtKind::Enum, "my-pkg.Opt")
    );
    assert_eq!(l.header.level, Level::Opt);
}

const DATA: &str = "stave.ir 4 \"x86_64-linux-v2\" debug checked
decl d0 struct std.cell.UnsafeCell
type t0 int u 8
type t1 array t0 100
type t2 int u 64
type t3 array t2 2
str s0 \"hi\\n\\x00\\\"\\\\\"
str s1 \"\"
site 0 \"src/a.stv\" 3 7 app.main
site 1 \"dir with space/b.stv\" 1 1 \"odd-pkg.f\"
static 0 t1 app.BYTES
data 0 0 x01
data 0 64 x02000003
static 1 t3 app.WORDS
ref 1 0 s0 2
ref 1 8 static 0 100
main @0
cell d0
fn @0 app.main() site 1 {
b0():
  ret
}
";

#[test]
fn strings_sites_statics_main_and_cell() {
    let l = load(DATA);
    let m = &l.module;
    assert_eq!(m.strings, vec![b"hi\n\0\"\\".to_vec(), Vec::new()]);
    assert_eq!(m.sites[1].file, "dir with space/b.stv");
    assert_eq!(m.sites[1].func, "odd-pkg.f");
    assert_eq!(m.statics.len(), 2);
    let s0 = &m.statics[0];
    assert_eq!((s0.path.as_str(), s0.ty, s0.init.len()), ("app.BYTES", TypeId(1), 100));
    let mut want = vec![0u8; 100];
    want[0] = 1;
    want[64] = 2;
    want[67] = 3;
    assert_eq!(s0.init, want);
    assert_eq!(m.statics[1].init, vec![0u8; 16]);
    assert_eq!(
        m.statics[1].refs,
        vec![
            DataRef { offset: 0, target: DataTarget::Str(StrId(0)), addend: 2 },
            DataRef { offset: 8, target: DataTarget::Static(0), addend: 100 },
        ]
    );
    assert_eq!(l.main, Some(FuncId(0)));
    assert_eq!(m.cell, Some(DeclId(0)));
    assert_eq!(m.funcs[0].name_site, Site(1));
    assert_eq!(l.lines.statics, vec![11, 14]);
}

#[test]
fn the_stack_section_of_the_emit_form() {
    let src = with_head(
        "type t0 int s 32
runstack 8388608
truncated
node 0 rec
node 1
edge 0 1 call
edge 1 0 indirect
edge 1 1 spawn
nodes @0 0 1
fn @0 app.main() {
b0():
  ret
}
fn @1 app.other() {
b0():
  ret
}
",
    );
    let l = load(&src);
    let s = l.stack.unwrap();
    assert_eq!(s.run_stack, Some(8388608));
    assert!(s.truncated);
    assert_eq!(s.nodes, vec![true, false]);
    assert_eq!(
        s.edges.iter().map(|e| (e.from, e.to, e.kind)).collect::<Vec<_>>(),
        vec![(0, 1, EdgeKind::Call), (1, 0, EdgeKind::Indirect), (1, 1, EdgeKind::Spawn)]
    );
    assert_eq!(s.nodes_of_func, vec![vec![0, 1], vec![]]);
}

const ATTRS: &str = "stave.ir 4 \"x86_64-linux-v2\" debug checked
type t0 int s 32
type t1 str
type t2 ref t1
type t3 int u size
str s0 \"x\"
site 0 \"src/app.stv\" 2 4 app.wide
fn @0.u app.wide<t0,3>(t0) -> t0 needs(avx2) site 0 {
b0(%0: t0):
  ret %0
}
fn @0 app.wide<t0,3>(t0) -> t0 needs(avx2) {
b0(%0: t0):
  %1 = call t0 @1 s0 1 %0 !0
  ret %1
}
fn @1 libc.printf(t2) -> t0 symbol \"printf\" variadic
fn @2 runtime.panic.trap(t3) known altstack
fn @3 app.neon_only() needs(neon) unbuilt
fn @0.0 app.wide() -> t0 {
b0():
  ret 0
}
";

#[test]
fn fn_header_attributes_and_instance_order() {
    let l = load(ATTRS);
    let m = &l.module;
    let ids: Vec<String> = m.funcs.iter().map(|f| f.instance.to_string()).collect();
    assert_eq!(ids, ["0", "0.0", "0.u", "1", "2", "3"]);
    let f0 = &m.funcs[0];
    assert_eq!(f0.type_args, vec![GenericArg::Type(TypeId(0)), GenericArg::Const(3)]);
    assert_eq!(f0.features, FeatureSet::of(Feature::Avx2));
    let printf = &m.funcs[3];
    assert_eq!(
        (printf.symbol.as_deref(), printf.variadic, printf.body.is_none()),
        (Some("printf"), true, true)
    );
    let trap = &m.funcs[4];
    assert_eq!(trap.known, Some(stave_ir::Known::Trap));
    assert!(trap.alt_stack && trap.ret.is_none());
    assert!(m.funcs[5].unbuilt);
    assert_eq!(m.funcs[2].instance, InstanceId(vec![0, InstanceId::UNCHECKED_ENTRY]));
    // the call names @1 by its rank
    let b = m.funcs[0].body.as_ref().unwrap();
    assert_eq!(b.args(0)[0], Operand::Func(FuncId(3)));
    assert_eq!(b.insts.site[0], Site(0));
    // line map follows the sorted fns
    assert_eq!(l.lines.funcs[0].header, 12);
}

#[test]
fn claims_become_the_stated_summary() {
    let src = with_head(
        "type t0 int s 32
type t1 int u size
fn @0 app.get(t0) -> t0 {
b0(%0: t0):
  ret %0
}
claim @0 pre checked p 0 <= zero 10
claim @0 pre zero <= p 0 0
claim @0 post result <= p 0 0
claim @0 post lenresult <= lenp 0 -3
fn @1 app.none() {
b0():
  ret
}
",
    );
    let l = load(&src);
    use stave_ir::witness::{NO_BLOCK, Term as T};
    let st = &l.module.stated;
    assert_eq!(st.len(), 2);
    let s = st[0].as_ref().unwrap();
    assert_eq!(s.pre.len(), 2);
    assert_eq!((s.pre[0].row.a, s.pre[0].row.b, s.pre[0].row.w), (T::Param(0), T::Zero, 10));
    assert_ne!(s.pre[0].entry, NO_BLOCK);
    assert_eq!(s.pre[1].entry, NO_BLOCK);
    assert_eq!(
        s.post.iter().map(|r| (r.a, r.b, r.w)).collect::<Vec<_>>(),
        vec![(T::Result, T::Param(0), 0), (T::LenResult, T::LenParam(0), -3)]
    );
    assert!(st[1].is_none());
}

/// Scalar int and float ops, compares, select and the casts.
const SCALAR: &str = "stave.ir 4 \"x86_64-linux-v2\" debug checked
type t0 int s 32
type t1 bool
type t2 int u 64
type t3 tuple t2 t2
type t4 float 64
type t5 float 32
type t6 int s 64
site 0 \"src/app.stv\" 1 1 app.ints
fn @0 app.ints(t0, t0, t2, t4) -> t0 {
b0(%0: t0, %1: t0, %2: t2, %3: t4):
  %4 = add t0 %0 %1 !0
  %5 = sub t0 %4 1 !0
  %6 = mul t0 %5 -2 !0
  %7 = div t0 %6 3 !0
  %8 = rem t0 %7 5 !0
  %9 = neg t0 %8 !0
  %10 = add.wrap t0 %9 %0
  %11 = sub.wrap t0 %10 %0
  %12 = mul.wrap t0 %11 %0
  %13 = shl.wrap t0 %12 3
  %14 = add.sat t0 %13 %1
  %15 = sub.sat t0 %14 %1
  %16 = mul.sat t0 %15 %1
  %17 = shl t0 %16 1 !0
  %18 = shr t0 %17 1 !0
  %19 = and t0 %18 %0
  %20 = or t0 %19 %0
  %21 = xor t0 %20 %0
  %22 = not t0 %21
  %23 = popcount t0 %22
  %24 = clz t0 %23
  %25 = ctz t0 %24
  %26 = mul.wide t3 %2 %2
  %27 = eq t1 %25 %0
  %28 = ne t1 %25 %0
  %29 = lt t1 %25 %0
  %30 = le t1 %25 %0
  %31 = gt t1 %25 %0
  %32 = ge t1 %25 %0
  %33 = and t1 %27 %28
  %34 = select t0 %33 %25 %0
  %35 = cast t6 %34
  %36 = cast.wrap t2 %34
  %37 = add t4 %3 0x3ff0000000000000 !0
  %38 = cast.sat t0 %37
  %39 = cast.round t5 %37
  %40 = add t5 %39 0x3f800000 !0
  ret %38
}
";

#[test]
fn scalar_ops_and_casts() {
    let l = load(SCALAR);
    let b = l.module.funcs[0].body.as_ref().unwrap();
    assert_eq!(b.insts.len(), 37);
    assert_eq!(b.insts.op[31], Op::Cast(stave_ir::CastKind::Exact));
    // one pool entry per constant operand, in print order
    assert_eq!(b.consts[0], Const::Int(1));
    assert_eq!(b.consts[1], Const::Int(-2));
    assert!(b.consts.contains(&Const::F64(0x3ff0000000000000)));
    assert!(b.consts.contains(&Const::F32(0x3f800000)));
}

/// Aggregates and enums.
const AGG: &str = "stave.ir 4 \"x86_64-linux-v2\" debug checked
decl d0 struct app.P
decl d1 enum app.Opt
type t0 int s 32
type t1 int u 32
type t2 bool
type t3 adt d0
type t4 adt d1 t0
type t5 array t0 4
type t6 tuple t0 t2
type t7 int s 64
shape t3 struct t0 t2
shape t4 enum | | t0
fn @0 app.agg(t0) -> t0 {
b0(%0: t0):
  %1 = make t3 %0 1
  %2 = get t0 %1 0
  %3 = set t3 %1 0 %2
  %4 = make t5 %0 %0 %0 %0
  %5 = get t0 %4 2
  %6 = enum.make t4 1 %5
  %7 = enum.tag t1 %6
  %8 = enum.get t0 %6 1 0
  %9 = make t6 %8 0
  %10 = get t2 %9 1
  %11 = cast t7 %8
  %12 = cast.try t4 %11
  ret %8
}
";

#[test]
fn aggregate_and_enum_ops() {
    let l = load(AGG);
    let b = l.module.funcs[0].body.as_ref().unwrap();
    assert_eq!(b.insts.op[5], Op::EnumMake);
    assert_eq!(b.int_const(b.args(5)[0]), Some(1));
}

/// Memory: slots, loads, stores, addresses, slices, checks, hints.
const MEM: &str = "stave.ir 4 \"x86_64-linux-v2\" debug checked
decl d0 enum app.E
type t0 int s 32
type t1 ref mut t0
type t2 int u size
type t3 slice t0
type t4 ref t3
type t5 ptr const t0
type t6 ref t0
type t7 tuple t0 t0
type t8 ref mut t7
type t9 adt d0
type t10 ref mut t9
shape t9 enum | t0 | t0 t0
site 0 \"src/m.stv\" 1 1 app.mem
fn @0 app.mem(t4, t8, t10, t2) -> t0 {
b0(%0: t4, %1: t8, %2: t10, %99: t2):
  %3 = slot t1
  store () %3 7 !0
  %4 = load t0 %3
  %5 = slice.len t2 %0
  %6 = slice.ptr t5 %0
  check.bounds () %99 %5 !0
  %7 = addr.elem t6 %0 %99
  %8 = load t0 %7
  %9 = addr.field t1 %1 1
  store () %9 %8
  %10 = addr.enum_field t1 %2 1 0
  store () %10 %4
  %11 = slice.make t4 %6 %5
  mem.prefetch_read_l1 () %6 0
  mem.prefetch_read_l2 () %6 0
  mem.prefetch_read_l3 () %6 0
  mem.prefetch_read_nta () %6 0
  mem.prefetch_write_l1 () %6 0
  mem.prefetch_write_l2 () %6 0
  mem.prefetch_write_l3 () %6 0
  mem.prefetch_write_nta () %6 1
  mem.store_stream () %9 %4
  mem.fence_stream ()
  ret %4
}
";

#[test]
fn memory_ops() {
    let l = load(MEM);
    let b = l.module.funcs[0].body.as_ref().unwrap();
    assert_eq!(b.insts.ty[1], None);
    assert_eq!(b.insts.result[1], None);
}

/// Calls of every kind, fn values, intrinsics, string operands.
const CALLS: &str = "stave.ir 4 \"x86_64-linux-v2\" debug checked
type t0 int s 32
type t1 int u size
type t2 int s size
type t3 fntype t0 -> t0
type t4 int u 8
type t5 ptr mut t4
type t6 ptr const t4
type t7 ref mut t0
type t8 str
type t9 ref t8
str s0 \"%d\"
site 0 \"src/c.stv\" 1 1 app.calls
fn @0 app.calls(t0, t5, t6) -> t0 {
b0(%0: t0, %1: t5, %2: t6):
  %3 = slot t7
  store () %3 %0
  %4 = call t0 @1 %0 !0
  %5 = call.ind t0 @1 %4 !0
  %6 = fn.make t3 @0.0 %3
  %7 = call.ind t0 %6 %5 !0
  %8 = intrinsic.syscall1 t2 60 0
  intrinsic.copy_bytes () %1 %2 4
  call () @2 !0
  call.ind () @2 !0
  %9 = call t0 @3 s0 1 %0 !0
  ret %7
}
fn @0.0 app.calls(t0, t7) -> t0 {
b0(%0: t0, %1: t7):
  %2 = load t0 %1
  %3 = add.wrap t0 %0 %2
  ret %3
}
fn @1 app.id(t0) -> t0 {
b0(%0: t0):
  ret %0
}
fn @2 app.ext() symbol \"ext\"
fn @3 libc.printf(t9) -> t0 symbol \"printf\" variadic
";

#[test]
fn calls_fn_values_and_intrinsics() {
    let l = load(CALLS);
    let m = &l.module;
    assert_eq!(m.intrinsics, vec!["syscall1".to_string(), "copy_bytes".to_string()]);
    let b = m.funcs[0].body.as_ref().unwrap();
    assert!(matches!(b.insts.op[7], Op::Intrinsic(i) if i.0 == 1));
    assert_eq!(b.args(10)[1], Operand::Str(StrId(0)));
}

/// Every panic kind behind a `switch`, then `unreachable`.
const PANICS: &str = "stave.ir 4 \"x86_64-linux-v2\" release checked
type t0 int s 32
fn @0 app.p(t0) {
b0(%0: t0):
  switch %0 0: b1() 1: b2() 2: b3() 3: b4() 4: b5() 5: b6() 6: b7() 7: b8() 8: b9() _: b10()
b1():
  panic.bounds ()
  unreachable
b2():
  panic.overflow ()
  unreachable
b3():
  panic.div_by_zero ()
  unreachable
b4():
  panic.assert ()
  unreachable
b5():
  panic.out_of_memory ()
  unreachable
b6():
  panic.stack_overflow ()
  unreachable
b7():
  panic.unwrap ()
  unreachable
b8():
  panic.unsupported_cpu ()
  unreachable
b9():
  panic.deadlock ()
  unreachable
b10():
  panic.contract ()
  unreachable
}
";

#[test]
fn panic_ops_and_switch() {
    let l = load(PANICS);
    let b = l.module.funcs[0].body.as_ref().unwrap();
    assert_eq!(b.blocks.len(), 11);
    assert_eq!(b.cases.len(), 9);
    assert_eq!(b.cases[8].value, 8);
    assert!(b.insts.op.iter().all(|o| matches!(o, Op::Panic(_))));
}

/// Branch targets with arguments, constants of every kind as operands.
const TERMS: &str = "stave.ir 4 \"x86_64-linux-v2\" debug checked
type t0 int s 32
type t1 bool
type t2 float 16
type t3 float b16
type t4 float 32
type t5 float 64
fn @0 app.t(t1) -> t0 {
b0(%0: t1):
  br b1(%0, 0x3c00, bf16 0x3f80, 0x3f800000, 0x3ff0000000000000)
b1(%1: t1, %2: t2, %3: t3, %4: t4, %5: t5):
  cbr %1 b2(5) b2(-5)
b2(%6: t0):
  ret %6
}
";

#[test]
fn terminators_and_constant_operands() {
    let l = load(TERMS);
    let b = l.module.funcs[0].body.as_ref().unwrap();
    assert_eq!(
        b.consts,
        vec![
            Const::F16(0x3c00),
            Const::BF16(0x3f80),
            Const::F32(0x3f800000),
            Const::F64(0x3ff0000000000000),
            Const::Int(5),
            Const::Int(-5),
        ]
    );
}

/// Vector and mask ops, with the guard of a chunk load and store.
const VEC: &str = "stave.ir 4 \"x86_64-linux-v2\" debug checked
type t0 int s 32
type t1 float 32
type t2 int u size
type t3 slice t1
type t4 ref mut t3
type t5 vec t1 4
type t6 mask 32 4
type t7 vec t0 4
type t8 bool
fn @0 app.v(t4, t1, t7, t2) -> t1 {
b0(%0: t4, %1: t1, %2: t7, %99: t2):
  %3 = vec.splat t5 %1
  %4 = vec.add t5 %3 %3
  %5 = vec.sqrt t5 %4
  %6 = vec.fma t5 %4 %5 %3
  %7 = vec.lt t6 %3 %6
  %8 = mask.not t6 %7
  %9 = mask.and t6 %7 %8
  %10 = mask.select t5 %9 %3 %6
  %11 = mask.all t6
  %12 = vec.permute t5 %10 3 2 1 0
  %13 = vec.div_const t7 %2 3
  %14 = vec.add.wrap t7 %13 %2
  %15 = vec.reduce_add_wrap t0 %14
  %16 = slice.len t2 %0
  %17 = sub.sat t2 %16 3
  check.bounds () %99 %17
  %18 = vec.load t5 %0 %99
  vec.store () %0 %99 %18
  %19 = vec.reduce_add t1 %12
  %20 = mask.none t6
  %21 = vec.load_masked t5 %0 %99 %20
  vec.store_masked () %0 %99 %21 %20
  ret %19
}
";

#[test]
fn vector_and_mask_ops() {
    let l = load(VEC);
    let b = l.module.funcs[0].body.as_ref().unwrap();
    assert!(matches!(b.insts.op[0], Op::Vec(_)));
}

#[test]
fn spacing_comments_numbering_and_fn_order_are_free() {
    let src = "# a comment line

stave.ir\t4  \"x86_64-linux-v2\" debug checked   # trailing comment
type t0 int s 32
type t1 bool
site 0 \"src/app.stv\" 1 4 app.max
fn @1 app.other() {
b0():
\tret
}
fn @0 app.max(t0, t0) -> t0 site 0 {
b0(%10: t0, %3: t0):
  %99=gt t1 %10 %3
  cbr %99 b9() b2()
b9():
  ret %10
b2():
  ret %3
}
";
    let a = load(src);
    let b = load(&format!("{MAX}fn @1 app.other() {{\nb0():\n  ret\n}}\n"));
    assert_eq!(a.module.funcs, b.module.funcs);
    assert_eq!(a.lines.header, 3);
}

// ---------------------------------------------------------------------------------------------
// Rejected inputs: one per rule

#[test]
fn lexical_rules() {
    rejects(&with_head("type t0 bool\r\n"), "t.ir:2:13: error: a carriage return is not allowed");
    rejects(&with_head("type t0 array t0 -0\n"), "t.ir:2:18: error: `-0` is not an integer");
    rejects(&with_head("type t0 array t0 01\n"), "t.ir:2:18: error: a number has no leading zero");
    rejects(&with_head("type t05 bool\n"), "t.ir:2:6: error: expected a type `tN`, found `t05`");
    rejects(
        &with_head("fn @0 a() {\nb0():\n  ret 0x3f8\n}\n"),
        "t.ir:4:7: error: a float constant is `0x` and 4, 8 or 16",
    );
    rejects(&with_head("fn @0 a() {\nb0():\n  ret 0x3F80\n}\n"), "t.ir:4:7: error: a float constant");
    rejects(&with_head("str s0 \"\\x41\"\n"), "t.ir:2:9: error: `\\x41` has the shorter form `A`");
    rejects(&with_head("str s0 \"\\x0a\"\n"), "has the shorter form `\\n`");
    rejects(&with_head("str s0 \"\\x5c\"\n"), "has the shorter form `\\\\`");
    rejects(&with_head("str s0 \"\\q\"\n"), "t.ir:2:9: error: unknown escape");
    rejects(&with_head("str s0 \"abc\n"), "t.ir:2:8: error: unterminated string");
    rejects(&with_head("str s0 \"\u{e9}\"\n"), "t.ir:2:9: error: byte 0xc3 in a string is written `\\xc3`");
    rejects(&with_head("type t0 bool \u{e9}\n"), "t.ir:2:14: error: unexpected byte 0xc3");
    rejects(
        &with_head("fn @0 a() {\nb0():\n  %1a = slot t0\n}\n"),
        "t.ir:4:5: error: `%1` must be followed by a space",
    );
    rejects(&with_head("main @1.u2\n"), "t.ir:2:10: error: `@1.u` must be followed");
    rejects(&with_head("type t0 bool $\n"), "t.ir:2:14: error: unexpected `$`");
    rejects(&with_head("data 0 0 x0\n"), "error: expected bytes `x..`, found `x0`");
}

#[test]
fn syntax_rules() {
    rejects(&with_head("frob 1 2\n"), "t.ir:2:1: error: unknown line kind `frob`");
    rejects(&with_head("fn @0 a() {\nb0():\n  %0 = frob t0\n"), "t.ir:4:8: error: unknown op `frob`");
    rejects(
        &with_head("fn @0 a() {\nb0():\n  %0 = intrinsic.vec.add t0\n"),
        "unknown op `intrinsic.vec.add`",
    );
    rejects(
        &with_head("fn @0 a() {\nb0():\n  %2 = add t0 %0\n"),
        "t.ir:4:17: error: wrong arity: `add` takes 2 operands, found the end of the line",
    );
    rejects(
        &with_head("fn @0 a() {\nb0():\n  %2 = add t0 %0 %1 %1\n"),
        "t.ir:4:21: error: wrong arity: `add` takes 2 operands, found `%1`",
    );
    rejects(
        &with_head("fn @0 a() {\nb0():\n  %2 = neg t0 %0 !3 4\n"),
        "unexpected `4` after a complete line",
    );
    rejects(&with_head("fn @0 a() {\nb0():\n  store () %0\n"), "wrong arity: `store` takes 2 operands");
    rejects(
        &with_head("fn @0 a() {\nb0():\n  %1 = slot t0 %0\n"),
        "wrong arity: `slot` takes 0 operands, found `%0`",
    );
    rejects(
        &with_head("fn @0 a() {\nb0():\n  %1 = addr.field t0 %0 %1\n"),
        "t.ir:4:25: error: expected an integer, found `%1`",
    );
    rejects(&with_head("fn @0 a() {\nb0():\n  %1 = add %0 %0\n"), "expected a type `tN`, found `%0`");
    rejects(&with_head("fn @0 a() {\nb0():\n  store t0 %0 %1\n"), "expected `(`, found `t0`");
    rejects(&with_head("fn @0 a() {\nb0():\n  %1 = store () %0 %1\n"), "unknown op `store`");
    rejects(&with_head("main @0 @1\n"), "t.ir:2:9: error: unexpected `@1` after a complete line");
    rejects(&with_head("decl d0 struct 5\n"), "t.ir:2:16: error: expected a path, found `5`");
    rejects(&with_head("type t0 ref mut foo\n"), "expected a type `tN`, found `foo`");
}

#[test]
fn header_rules() {
    rejects(
        "stave.ir 3 \"x86_64-linux-v2\" debug checked\n",
        "t.ir:1:10: error: IR text version 3 is not read here",
    );
    rejects(
        "stave.ir 4 \"x86_64-linux\" debug checked\n",
        "t.ir:1:12: error: target `x86_64-linux` is written by its canonical name `x86_64-linux-v2`",
    );
    rejects("stave.ir 4 \"riscv\" debug checked\n", "unknown target `riscv`");
    rejects(
        "stave.ir 4 \"x86_64-linux-v2\" debug opt\n",
        "t.ir:1:36: error: level `opt` exists only with mode `release`",
    );
}

#[test]
fn canonical_spelling_rules() {
    rejects(
        &with_head("decl d0 struct \"app.P\"\n"),
        "t.ir:2:16: error: path `app.P` has the Word shape: it is written bare",
    );
    rejects(
        &with_head("fn @0 a() needs(sse42, avx2)\n"),
        "t.ir:2:17: error: feature list is not canonical (implied members left out, in feature order): write `avx2`",
    );
    rejects(&with_head("fn @0 a() needs(fma, avx2)\n"), "feature list is not canonical");
    rejects(&with_head("type t0 fntype -> t0 avx2 avx2\n"), "write `avx2`");
    rejects(&with_head("type t0 int s 7\n"), "t.ir:2:15: error: an int width is 8, 16, 32, 64 or `size`");
    rejects(&with_head("type t0 float b32\n"), "t.ir:2:15: error: a float width is 16, 32, 64 or `b16`");
    rejects(&with_head("type t0 float 8\n"), "a float width is");
    rejects(
        &with_head("fn @0 a() {\nb0():\n  ret bf16 0x3f800000\n"),
        "t.ir:4:12: error: a `bf16` constant has 4 hex digits",
    );
    rejects(
        &with_head("type t0 bool\nstatic 0 t0 a.S\ndata 0 60 x0102030405\n"),
        "t.ir:4:8: error: a `data` line lies within one 64-byte block",
    );
    rejects(
        &with_head("type t0 bool\nstatic 0 t0 a.S\ndata 0 0 x0001\n"),
        "a `data` line begins and ends with a nonzero byte",
    );
    rejects(
        &with_head("main @1.4294967295\n"),
        "t.ir:2:6: error: `@1.4294967295`: the number 4294967295 is spelled `u`",
    );
    rejects(
        &with_head("type t0 array t0 99999999999999999999999999999999999999999\n"),
        "error: integer `99999999999999999999999999999999999999999` is out of range",
    );
    rejects(
        &with_head("site 4294967296 \"f\" 1 1 a\n"),
        "t.ir:2:6: error: a site row `4294967296` is out of range 0..=4294967295",
    );
    rejects(
        &with_head("type t0 bool\nstatic 0 t0 a.S\nref 0 0 static 0 -1\n"),
        "an addend `-1` is out of range",
    );
    rejects(&with_head("type t99999999999 bool\n"), "`t99999999999` is out of range");
    rejects(&with_head("type t0 mask 300 4\n"), "a mask width `300` is out of range 0..=255");
    rejects(&with_head("nodes @0 1 1\n"), "the nodes of a fn are listed ascending");
    rejects(&with_head("site 0 \"\\xff\" 1 1 a\n"), "a site file is not UTF-8");
}

#[test]
fn section_rules() {
    rejects("", "t.ir:1:1: error: the file is empty");
    rejects("type t0 bool\n", "t.ir:1:1: error: the file begins with the `stave.ir` header line");
    rejects(&with_head(HEAD), "t.ir:2:1: error: duplicate `stave.ir` header line");
    rejects(
        &with_head("str s0 \"a\"\ntype t0 bool\n"),
        "t.ir:3:1: error: `type` line out of section order: `type` lines come before `str` lines",
    );
    rejects(
        &with_head("type t1 bool\n"),
        "t.ir:2:1: error: type t1 out of order: type lines are t0, t1, ... (expected t0)",
    );
    rejects(&with_head("str s0 \"a\"\nstr s0 \"b\"\n"), "t.ir:3:1: error: str s0 out of order");
    rejects(&with_head("site 1 \"f\" 1 1 a\n"), "site 1 out of order");
    rejects(&with_head("type t0 bool\nstatic 1 t0 a.S\n"), "static 1 out of order");
    rejects(&with_head("decl d2 struct a.A\ndecl d1 struct a.B\n"), "t.ir:3:1: error: decl d1 out of order");
    rejects(&with_head("decl d2 struct a.A\ndecl d2 struct a.B\n"), "decl d2 out of order");
    rejects(
        &with_head(
            "decl d0 struct a.A\ntype t0 adt d0\ntype t1 adt d0 t0\nshape t1 struct\nshape t0 struct\n",
        ),
        "t.ir:6:1: error: shape of t0 out of order",
    );
    rejects(&with_head("fn @0 a()\nmain @0\nmain @0\n"), "`main` line out of section order");
    rejects(&with_head("main @0\nmain @0\n"), "t.ir:3:1: error: duplicate `main` line");
    rejects(&with_head("decl d0 struct a.C\ncell d0\ncell d0\n"), "duplicate `cell` line");
    rejects(&with_head("runstack 1\nruntack 2\n"), "unknown line kind `runtack`");
    rejects(
        &with_head("runstack 1\nruntack 2\n".replace("runtack", "runstack").as_str()),
        "duplicate `runstack` line",
    );
    rejects(&with_head("truncated\ntruncated\n"), "duplicate `truncated` line");
    rejects(
        &with_head("truncated\nruntack 2\n".replace("runtack", "runstack").as_str()),
        "`runstack` line out of section order",
    );
    rejects(&with_head("node 1\n"), "node 1 out of order");
    rejects(
        &with_head("node 0\nnode 1\nedge 1 0 call\nedge 0 1 call\n"),
        "t.ir:5:1: error: `edge` lines ascend by (from, to)",
    );
    rejects(
        &with_head("node 0\nnodes @1 0\nnodes @0 0\nfn @0 a()\nfn @1 b()\n"),
        "t.ir:4:1: error: `nodes` lines ascend by instance id",
    );
}

#[test]
fn static_rules() {
    rejects(
        &with_head("type t0 bool\nstatic 0 t0 a.S\ndata 1 0 x01\n"),
        "t.ir:4:1: error: `data` line of static 1 outside the lines of static 1",
    );
    rejects(
        &with_head("type t0 bool\nstatic 0 t0 a.S\nref 0 0 s0 0\ndata 0 0 x01\n"),
        "`data` line of static 0 after its `ref` lines",
    );
    rejects(
        &with_head("type t0 int u 8\ntype t1 array t0 200\nstatic 0 t1 a.S\ndata 0 64 x01\ndata 0 70 x01\n"),
        "t.ir:6:1: error: `data` lines of static 0 ascend by offset, one per 64-byte block",
    );
    rejects(
        &with_head(
            "type t0 int u 64\ntype t1 array t0 4\nstr s0 \"a\"\nstatic 0 t1 a.S\nref 0 8 s0 0\nref 0 0 s0 0\n",
        ),
        "t.ir:7:1: error: `ref` lines of static 0 ascend by offset",
    );
    rejects(
        &with_head("type t0 int u 8\ntype t1 array t0 4\nstatic 0 t1 a.S\ndata 0 2 x0102030405\n"),
        "t.ir:5: verify: static 0: `data` bytes at 2..7 lie past its size 4",
    );
    rejects(
        &with_head("type t0 int u 8\ntype t1 array t0 16\nstr s0 \"a\"\nstatic 0 t1 a.S\nref 0 0 s0 0\n"),
        "t.ir:5: verify: static 0: address words in a type aligned to 1",
    );
}

#[test]
fn id_rules() {
    rejects(&with_head("type t0 ref t9\n"), "t.ir:2:13: error: t9 names no type line");
    rejects(
        &with_head("type t0 int u 64\ntype t1 array t0 1\nstatic 0 t1 a.S\nref 0 0 s3 0\n"),
        "t.ir:5:9: error: s3 names no str line",
    );
    rejects(&with_head("fn @0 a() site 2\n"), "t.ir:2:16: error: site row 2 names no site line");
    rejects(
        &with_head("type t0 int u 64\ntype t1 array t0 1\nstatic 0 t1 a.S\nref 0 0 static 4 0\n"),
        "t.ir:5:16: error: static 4 names no static line",
    );
    rejects(&with_head("type t0 adt d5\n"), "t.ir:2:13: error: d5 names no decl line");
    rejects(&with_head("cell d1\n"), "d1 names no decl line");
    rejects(&with_head("node 0\nedge 0 3 call\n"), "t.ir:3:8: error: node 3 names no node line");
    rejects(&with_head("main @4\n"), "t.ir:2:6: error: `main` names @4, which heads no fn");
    rejects(
        &with_head("nodes @4 0\nfn @0 a()\n".replace("nodes", "node 0\nnodes").as_str()),
        "`nodes` names @4, which heads no fn",
    );
    rejects(&with_head("fn @0 a() {\nb0():\n  call () @7\n  ret\n}\n"), "t.ir:4:11: error: @7 heads no fn");
    rejects(
        &with_head("type t0 bool\nfn @0 a() {\nb0():\n  ret %3\n}\n"),
        "t.ir:5:7: error: %3 is not a value of fn @0",
    );
    rejects(&with_head("fn @0 a() {\nb0():\n  br b4()\n}\n"), "t.ir:4:6: error: b4 is not a block of fn @0");
    rejects(
        &with_head("fn @0 a() {\nb0():\n  br b1()\nb1():\n  br b1()\nb1():\n  ret\n}\n"),
        "t.ir:7:1: error: block b1 is defined twice in fn @0",
    );
    rejects(
        &with_head("type t0 bool\nfn @0 a(t0) {\nb0(%0: t0):\n  %0 = not t0 %0\n  ret\n}\n"),
        "t.ir:5:3: error: value %0 is defined twice in fn @0",
    );
    rejects(
        &with_head("decl d0 struct a.A\n"),
        "t.ir:2:1: error: decl d0 is named by no `adt` type and no `cell` line",
    );
    rejects(&with_head("type t0 bool\ntype t1 bool\n"), "t.ir:3:1: error: type t1 has the key of t0");
    rejects(
        &with_head("type t0 bool\nshape t0 struct t0\n"),
        "t.ir:3:1: error: shape of t0, which is not an `adt` type",
    );
    rejects(
        &with_head("decl d0 enum a.E\ntype t0 adt d0\nshape t0 struct\n"),
        "t.ir:4:1: error: shape of t0 is a struct, its decl d0 a enum",
    );
    rejects(
        &with_head("fn @0 a()\nfn @0 b()\n"),
        "t.ir:3:1: error: fn @0 is defined twice (first at line 2)",
    );
    rejects(
        &with_head("fn @0 app.f() known\n"),
        "t.ir:2:15: error: `known`: `app.f` is the path of no compiler-known item",
    );
}

#[test]
fn body_rules() {
    rejects(
        &with_head("fn @0 a() {\nb1():\n  ret\n}\n"),
        "t.ir:3:1: error: the body of fn @0 begins with block b0, not b1",
    );
    rejects(&with_head("fn @0 a() {\n}\n"), "t.ir:3:1: error: the body of fn @0 has no block");
    rejects(
        &with_head("fn @0 a() {\nb0():\n  ret\nb1():\n}\n"),
        "t.ir:6:1: error: `}` before the terminator of the last block",
    );
    rejects(
        &with_head("fn @0 a() {\nb0():\nb1():\n  ret\n}\n"),
        "t.ir:4:1: error: block b1 begins before block b0 has its terminator",
    );
    rejects(&with_head("fn @0 a() {\nb0():\n  ret\n  ret\n}\n"), "t.ir:5:3: error: a second terminator");
    rejects(
        &with_head("fn @0 a() {\nb0():\n  ret\n  panic.bounds ()\n}\n"),
        "t.ir:5:3: error: instruction after the terminator of its block",
    );
    rejects(&with_head("fn @0 a() {\n  ret\n"), "t.ir:3:3: error: terminator before the first block header");
    rejects(&with_head("fn @0 a() {\n  panic.bounds ()\n"), "instruction before the first block header");
    rejects(&with_head("fn @0 a()\nb0():\n"), "t.ir:3:1: error: block header outside a fn body");
    rejects(&with_head("  panic.bounds ()\n"), "t.ir:2:3: error: instruction outside a fn body");
    rejects(&with_head("  ret\n"), "terminator outside a fn body");
    rejects(&with_head("}\n"), "t.ir:2:1: error: `}` outside a fn body");
    rejects(
        &with_head("fn @0 a() {\nb0():\n  ret\n"),
        "t.ir:2:1: error: the body of fn @0 is not closed by `}`",
    );
    rejects(
        &with_head("fn @0 a() {\nb0():\n  ret\nfn @1 b()\n"),
        "t.ir:5:1: error: `fn` line inside the body of fn @0",
    );
    rejects(
        &with_head("fn @0 a() {\nb0():\n  ret\nb1():\n  ret\n}\n"),
        "t.ir:5:1: error: block b1 of fn @0 is unreachable from b0",
    );
}

#[test]
fn claim_rules() {
    let f = "fn @0 a() {\nb0():\n  ret\n}\n";
    rejects(
        &format!("stave.ir 4 \"x86_64-linux-v2\" debug lowered\n{f}claim @0 post zero <= zero 0\n"),
        "t.ir:6:1: error: no `claim` line at level `lowered`",
    );
    rejects(
        &with_head(&format!("{f}claim @1 post zero <= zero 0\n")),
        "t.ir:6:7: error: claim of @1 after fn @0",
    );
    rejects(
        &with_head("fn @0 a()\nclaim @0 post zero <= zero 0\n"),
        "t.ir:3:1: error: claim of fn @0, which has no body",
    );
    rejects(
        &with_head(&format!("{f}claim @0 post zero <= zero 0\nclaim @0 pre zero <= zero 0\n")),
        "t.ir:7:1: error: a `pre` claim after a `post` claim",
    );
    rejects(
        &with_head("claim @0 post zero <= zero 0\n"),
        "t.ir:2:1: error: `claim` line before the first fn",
    );
    rejects(
        &with_head("fn @0 a() {\nb0():\nclaim @0 post zero <= zero 0\n"),
        "t.ir:4:1: error: `claim` line inside the body of fn @0",
    );
}

#[test]
fn verifier_errors_name_the_line_they_are_about() {
    // an instruction
    let src = with_head(
        "type t0 int s 32\ntype t1 bool\nfn @0 a(t0) {\nb0(%0: t0):\n  %1 = add t1 %0 %0\n  ret\n}\n",
    );
    let es = rejects(&src, "t.ir:6: verify: ");
    assert!(es.iter().all(|e| e.kind == ErrorKind::Verify));
    // a terminator
    rejects(
        &with_head("type t0 int s 32\ntype t1 bool\nfn @0 a(t1) -> t0 {\nb0(%0: t1):\n  ret %0\n}\n"),
        "t.ir:6: verify: ",
    );
    // the entry block's params
    rejects(
        &with_head("type t0 int s 32\nfn @0 a(t0) {\nb0():\n  ret\n}\n"),
        "t.ir:4: verify: b0 params do not match the fn params",
    );
    // a type
    rejects(
        &with_head("type t0 bool\ntype t1 ref t2\ntype t2 tuple t1\n"),
        "t.ir:3: verify: t1: the type is a component of itself",
    );
}

#[test]
fn the_audit_names_the_unguarded_access() {
    let src = with_head(
        "type t0 int s 32
type t1 slice t0
type t2 ref t1
type t3 ref t0
fn @0 a(t2) -> t0 {
b0(%0: t2):
  %1 = addr.elem t3 %0 5
  %2 = load t0 %1
  ret %2
}
",
    );
    let es = rejects(&src, "t.ir:8: verify: ");
    assert_eq!(es[0].kind, ErrorKind::Audit);
    assert!(es[0].msg.contains("access without a bounds check or proof"), "{}", es[0].msg);
}

#[test]
fn one_error_per_bad_line_at_most_one_hundred() {
    let mut src = String::from(HEAD);
    for _ in 0..150 {
        src.push_str("type t0 int s 7 \"\\q\"\n");
    }
    let es = read(src.as_bytes()).unwrap_err();
    assert_eq!(es.len(), 100);
    assert!(es.windows(2).all(|w| w[0].line < w[1].line));
}

// ---------------------------------------------------------------------------------------------
// Fuzz: no input panics

const CORPUS: &[&str] = &[MAX, KEYS, DATA, SCALAR, AGG, MEM, CALLS, PANICS, TERMS, VEC, ATTRS];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

/// Bytes the IR text is made of, for insertions that make plausible tokens.
const ALPHABET: &[u8] = b" \t\n%@#\"\\-0123456789abtsdxu.(),:=!|{}<>_";

fn mutate(r: &mut Rng, src: &[u8]) -> Vec<u8> {
    let mut v = src.to_vec();
    for _ in 0..1 + r.below(4) {
        let at = r.below(v.len() + 1);
        match r.below(6) {
            0 if !v.is_empty() => {
                v.remove(at.min(v.len() - 1));
            }
            1 if !v.is_empty() => {
                let i = at.min(v.len() - 1);
                v[i] = r.next() as u8;
            }
            2 => v.insert(at, ALPHABET[r.below(ALPHABET.len())]),
            3 if !v.is_empty() => {
                // duplicate a line
                let lines: Vec<&[u8]> = src.split(|&c| c == b'\n').collect();
                let l = lines[r.below(lines.len())].to_vec();
                let mut nl = l;
                nl.push(b'\n');
                let pos = v
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| **c == b'\n')
                    .map(|(i, _)| i + 1)
                    .nth(r.below(8))
                    .unwrap_or(0);
                v.splice(pos..pos, nl);
            }
            4 if !v.is_empty() => {
                let i = at.min(v.len() - 1);
                if v[i].is_ascii_digit() {
                    v[i] = b'0' + (r.next() % 10) as u8;
                } else {
                    v.insert(i, b'1');
                }
            }
            _ => {
                let i = at.min(v.len());
                v.truncate(i);
            }
        }
    }
    v
}

#[test]
fn mutated_inputs_never_panic() {
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    let mut accepted = 0usize;
    for round in 0..20000 {
        let base = CORPUS[round % CORPUS.len()].as_bytes();
        let input = mutate(&mut r, base);
        let got = std::panic::catch_unwind(|| read(&input).is_ok());
        match got {
            Ok(ok) => accepted += ok as usize,
            Err(_) => panic!("panic on input:\n{}", String::from_utf8_lossy(&input)),
        }
    }
    assert!(accepted > 0, "no mutation kept a valid module");
}

#[test]
fn the_line_map_holds_block_lines() {
    let l = load(DATA);
    assert_eq!(l.module.funcs[0].body.as_ref().unwrap().blocks[0].term, Term::Ret(None));
    assert_eq!(l.lines.funcs[0].blocks, vec![20]);
}

#[test]
fn a_type_chain_deeper_than_the_limit_is_an_error() {
    let mut src = String::from(HEAD);
    for i in 0..1500 {
        src.push_str(&format!("type t{i} ref t{}\n", i + 1));
    }
    src.push_str("type t1500 bool\n");
    rejects(&src, "verify: t");
    rejects(&src, ": nested deeper than 1000");
}
