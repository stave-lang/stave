//! The IR text printer (contracts/checked-ir.md, contracts/ir-text.ebnf): every production of the
//! grammar is printed by the kitchen-sink module below or by the per-op test, the output is a
//! function of the module, and the spellings the contract fixes are exact.
//!
//! `tests/print_golden/kitchen_sink.ir` is the emit form of the kitchen sink (`STAVE_BLESS=1`
//! rewrites it).

mod common;

use std::collections::BTreeSet;
use std::sync::Arc;

use common::*;
use stave_ir::text::{EdgeKind, Header, Level, Mode, StackEdge, StackInput};
use stave_ir::types::{AdtKind, DeclInfo, Lanes};
use stave_ir::witness::{NO_BLOCK, Pre, Row, Summary, Term as RowTerm};
use stave_ir::{
    AdtShape, Block, CastKind, Const, DataRef, DataTarget, DeclId, FloatBits, FuncBuilder, FuncId,
    GenericArg, IntBits, Known, Module, Op, Operand, PanicKind, Prefetch, Site, SiteInfo, Static, StrId,
    TypeId, TypeKey, VecOp, canon, print_module,
};
use stave_target::{Feature, FeatureSet, Triple, X86Level};

fn repo() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn grammar() -> String {
    std::fs::read_to_string(repo().join("contracts/ir-text.ebnf")).unwrap()
}

/// The grammar without its `(* ... *)` comments.
fn grammar_rules() -> Vec<(String, String)> {
    let g = grammar();
    let mut text = String::new();
    let mut rest = g.as_str();
    while let Some(a) = rest.find("(*") {
        text.push_str(&rest[..a]);
        let b = rest[a..].find("*)").expect("unclosed comment") + a + 2;
        rest = &rest[b..];
    }
    text.push_str(rest);
    text.split(';')
        .filter_map(|r| {
            let (name, body) = r.split_once(" = ").or_else(|| r.split_once("= "))?;
            let name = name.trim();
            (!name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric()))
                .then(|| (name.to_string(), body.to_string()))
        })
        .collect()
}

/// The quoted terminals of one grammar rule.
fn terminals(rule: &str) -> Vec<String> {
    let rules = grammar_rules();
    let body = &rules.iter().find(|(n, _)| n == rule).unwrap_or_else(|| panic!("no rule {rule}")).1;
    let mut out = Vec::new();
    let mut chars = body.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '"' {
            let mut s = String::new();
            while let Some(d) = chars.next() {
                match d {
                    '\\' => s.push(chars.next().unwrap()),
                    '"' => break,
                    _ => s.push(d),
                }
            }
            out.push(s);
        }
    }
    out
}

fn header(level: Level) -> Header {
    Header { target: Triple::X86_64(X86Level::V2), mode: Mode::Debug, level }
}

/// Every key form (the canonical module renumbers them). Returns the ids by name.
struct Ty {
    i32: TypeId,
    bool: TypeId,
    f32: TypeId,
    unit: TypeId,
    u8: TypeId,
    usize: TypeId,
    slice_u8: TypeId,
    ptr_u8: TypeId,
    pair: TypeId,
    opt: TypeId,
    fnty: TypeId,
}

fn types(m: &mut Module) -> Ty {
    let t = &mut m.types;
    for signed in [true, false] {
        for bits in [IntBits::B8, IntBits::B16, IntBits::B32, IntBits::B64, IntBits::Size] {
            t.intern(TypeKey::Int { signed, bits });
        }
    }
    for f in [FloatBits::F16, FloatBits::F32, FloatBits::F64, FloatBits::B16] {
        t.intern(TypeKey::Float(f));
    }
    let (bool_, _char, _str) = (t.bool(), t.char(), t.str());
    let i32 = t.i32();
    let u8 = t.u8();
    let usize = t.usize();
    let f32 = t.f32();
    let unit = t.unit();
    t.ref_(false, i32);
    t.ref_(true, i32);
    t.ptr(false, i32);
    let ptr_u8 = t.ptr(true, u8);
    t.array(i32, 16);
    t.slice(i32);
    t.intern(TypeKey::Slice { elem: u8, align: Some(32) });
    let slice_u8 = t.slice(u8);
    t.tuple(vec![i32, bool_]);
    let pair = t.adt(DeclId(0), vec![], AdtShape::structure(vec![i32, bool_]));
    t.adt(
        DeclId(1),
        vec![GenericArg::Type(i32), GenericArg::Const(8), GenericArg::Const(-3)],
        AdtShape::enumeration(vec![vec![], vec![i32]]),
    );
    let opt = t.adt(DeclId(1), vec![GenericArg::Type(u8)], AdtShape::enumeration(vec![vec![], vec![u8]]));
    t.fn_type(vec![i32, i32], i32, FeatureSet::EMPTY);
    let fnty = t.fn_type(vec![], unit, FeatureSet::of(Feature::Avx2).union(FeatureSet::of(Feature::Fma)));
    t.intern(TypeKey::Vec { lane: f32, n: Lanes::Fixed(8) });
    t.intern(TypeKey::Vec { lane: f32, n: Lanes::Native });
    t.intern(TypeKey::Mask { width: 32, n: Lanes::Fixed(8) });
    t.intern(TypeKey::Mask { width: 8, n: Lanes::Native });
    Ty { i32, bool: bool_, f32, unit, u8, usize, slice_u8, ptr_u8, pair, opt, fnty }
}

fn row(a: RowTerm, b: RowTerm, w: i128) -> Row {
    Row { a, b, w }
}

/// A module that holds every line kind and every operand form the printer has: the kitchen sink.
/// Returns it with its stack section.
fn sink() -> (Module, StackInput) {
    let mut m = Module::new();
    let t = types(&mut m);
    m.decls = vec![
        DeclInfo { id: DeclId(0), kind: AdtKind::Struct, path: "app.Pair".into() },
        DeclInfo { id: DeclId(1), kind: AdtKind::Enum, path: "app.Opt".into() },
        DeclInfo { id: DeclId(7), kind: AdtKind::Struct, path: "my-pkg.Cell".into() },
    ];
    m.cell = Some(DeclId(7));
    let hi = m.add_string(b"hi\n".to_vec());
    let all: Vec<u8> = (0..=255u8).collect();
    let all_bytes = m.add_string(all);
    m.sites = vec![
        SiteInfo { file: "src/app.stv".into(), line: 3, col: 5, func: "app.main".into() },
        SiteInfo { file: "src/\"odd\" \u{e9}.stv".into(), line: 40, col: 1, func: "my-pkg.f".into() },
        SiteInfo { file: "src/app.stv".into(), line: 9, col: 12, func: "app.main".into() },
    ];
    let arr = m.types.array(t.u8, 200);
    let mut init = vec![0u8; 200];
    init[0] = 1;
    init[1] = 2;
    init[63] = 0xff;
    init[130] = 0xab;
    init[140] = 0xcd;
    init[199] = 7;
    m.statics = vec![
        Static { path: "app.ZERO".into(), ty: t.i32, init: vec![0; 4], refs: vec![] },
        Static {
            path: "app.TABLE".into(),
            ty: arr,
            init,
            refs: vec![
                DataRef { offset: 8, target: DataTarget::Str(hi), addend: 0 },
                DataRef { offset: 16, target: DataTarget::Static(0), addend: 4 },
                DataRef { offset: 24, target: DataTarget::Str(all_bytes), addend: 256 },
            ],
        },
    ];
    let intr = m.intrinsic("syscall3");
    let copy = m.intrinsic("copy_bytes");

    // @0: an `extern "C"` import: no body, a symbol, variadic.
    let imp = m.declare(inst(&[0]), "app.puts", vec![t.ptr_u8], Some(t.i32));
    {
        let f = m.func_mut(imp).unwrap();
        f.symbol = Some("puts".into());
        f.variadic = true;
        f.name_site = Site(1);
    }
    // @1: `main`, every terminator and operand form.
    let main = m.declare(inst(&[1]), "app.main", vec![t.i32, t.bool], Some(t.i32));
    m.func_mut(main).unwrap().name_site = Site(0);
    m.main = Some(main);
    let clo = m.declare(inst(&[1, 0]), "app.main", vec![t.i32], Some(t.i32));
    let unchecked = m.declare(inst(&[1, InstanceIdU]), "app.main", vec![t.i32, t.bool], Some(t.i32));
    // @2: a generic instance with needs.
    let gen_ = m.declare(inst(&[2]), "app.gen", vec![t.slice_u8], None);
    {
        let f = m.func_mut(gen_).unwrap();
        f.type_args = vec![GenericArg::Type(t.i32), GenericArg::Const(8), GenericArg::Const(-1)];
        f.features = FeatureSet::of(Feature::Avx2).union(FeatureSet::of(Feature::Fma));
    }
    // @3: a known item, on the alternate stack, not built by this target, with a name site.
    let known = m.declare(inst(&[3]), Known::RunQueued.path(), vec![], None);
    {
        let f = m.func_mut(known).unwrap();
        f.known = Some(Known::RunQueued);
        f.alt_stack = true;
        f.unbuilt = true;
        f.name_site = Site(2);
    }
    // @4: a bodiless fn whose path needs quotes, no params, no result.
    let quoted = m.declare(inst(&[4]), "my-pkg.f", vec![], None);
    m.func_mut(quoted).unwrap().unbuilt = true;
    // @5: a bodiless fn whose signature reaches every type: a canonical module keeps only the
    // types something reaches, those of no body after the others (the S set).
    let every: Vec<TypeId> = (0..m.types.len() as u32).map(TypeId).collect();
    m.declare(inst(&[5]), "app.every_type", every, None);

    // main's body.
    let mut b = FuncBuilder::new(&[t.i32, t.bool]);
    let (x, flag) = (b.param(0).unwrap(), b.param(1).unwrap());
    let (b1, p1) = b.create_block_with(&[t.i32, t.bool]);
    let b2 = b.create_block();
    let b3 = b.create_block();
    let (b4, p4) = b.create_block_with(&[t.i32]);
    let b5 = b.create_block();
    b.set_site(Site(0));
    let slot = b.slot(t.ptr_u8);
    let minus = b.int(-5);
    let big = b.int(i128::from(u64::MAX));
    let h16 = b.konst(Const::F16(0x3c00));
    let bf = b.konst(Const::BF16(0x3f80));
    let h32 = b.konst(Const::F32(0x3f80_0000));
    let h64 = b.konst(Const::F64(0x4000_0000_0000_0000));
    b.set_site(Site::NONE);
    let sum = b.op(Op::Add, t.i32, &[x.into(), minus]);
    b.set_site(Site(2));
    let r = b.call(imp, Some(t.i32), &[slot.into(), Operand::Str(hi), big]).unwrap();
    b.set_site(Site::NONE);
    b.inst(Op::Store, None, &[slot.into(), h16]);
    b.inst(Op::Store, None, &[slot.into(), bf]);
    b.inst(Op::Store, None, &[slot.into(), h32]);
    b.inst(Op::Store, None, &[slot.into(), h64]);
    let ty_op = b.op(Op::Make, t.pair, &[sum.into(), flag.into()]);
    let fv = b.fn_make(t.fnty, clo, slot);
    let ci = b.call_ind(fv, Some(t.i32), &[x.into()]).unwrap();
    b.call(known, None, &[]);
    b.call_ind(fv, None, &[]);
    b.intrinsic(intr, Some(t.i32), &[x.into(), Operand::Type(t.usize), r.into()]);
    b.intrinsic(copy, None, &[slot.into(), slot.into(), x.into()]);
    b.inst(Op::Panic(PanicKind::Bounds), None, &[]);
    let _ = (ty_op, ci);
    b.switch(x, &[(0, b1, &[x.into(), flag.into()]), (-1, b2, &[]), (300, b3, &[])], b4, &[r.into()]);
    b.switch_to(b1);
    b.cbr(p1[1], b2, &[], b5, &[]);
    b.switch_to(b2);
    b.br(b4, &[Operand::Func(main)]);
    b.switch_to(b3);
    b.unreachable();
    b.switch_to(b4);
    b.ret(Some(p4[0].into()));
    b.switch_to(b5);
    b.ret(Some(Operand::Func(unchecked)));
    let body = b.finish().unwrap();
    m.define(main, body);

    let mut b = FuncBuilder::new(&[t.i32]);
    let x = b.param(0).unwrap();
    b.ret(Some(x.into()));
    m.define(clo, b.finish().unwrap());
    let mut b = FuncBuilder::new(&[t.i32, t.bool]);
    let x = b.param(0).unwrap();
    b.ret(Some(x.into()));
    m.define(unchecked, b.finish().unwrap());
    let mut b = FuncBuilder::new(&[t.slice_u8]);
    b.ret(None);
    m.define(gen_, b.finish().unwrap());
    let mut b = FuncBuilder::new(&[]);
    b.ret(None);
    m.define(known, b.finish().unwrap());

    assert!(canon::canonicalize(&mut m).iter().enumerate().all(|(i, f)| f.index() == i), "declared in order");
    let _ = (t.opt, t.f32, t.unit);

    // Summaries: every row term, a `checked` pre row, a negative weight.
    let s = Summary {
        pre: vec![
            Pre { row: row(RowTerm::Param(0), RowTerm::Zero, 100), entry: NO_BLOCK, half: 0 },
            Pre { row: row(RowTerm::Zero, RowTerm::LenParam(1), -1), entry: 0, half: 0 },
        ],
        post: vec![
            row(RowTerm::Result, RowTerm::Param(0), 0),
            row(RowTerm::LenResult, RowTerm::LenParam(1), 3),
        ],
        ..Summary::default()
    };
    m.stated = vec![None, Some(Arc::new(s)), None, None, None, None, None, None];
    let stack = StackInput {
        run_stack: Some(65536),
        truncated: true,
        nodes: vec![false, true, false],
        edges: vec![
            StackEdge { from: 0, to: 1, kind: EdgeKind::Call },
            StackEdge { from: 0, to: 2, kind: EdgeKind::Indirect },
            StackEdge { from: 1, to: 2, kind: EdgeKind::Spawn },
        ],
        nodes_of_func: vec![vec![], vec![0, 1], vec![2], vec![], vec![], vec![], vec![], vec![]],
    };
    (m, stack)
}

/// The last number of an unchecked entry's instance id.
#[allow(non_upper_case_globals)]
const InstanceIdU: u32 = stave_ir::InstanceId::UNCHECKED_ENTRY;

fn emit_form() -> String {
    let (m, stack) = sink();
    print_module(&m, &header(Level::Checked), Some(&stack)).unwrap()
}

#[test]
fn kitchen_sink_golden() {
    let got = emit_form();
    let path = repo().join("crates/stave-ir/tests/print_golden/kitchen_sink.ir");
    if std::env::var_os("STAVE_BLESS").is_some() {
        std::fs::write(&path, &got).unwrap();
        return;
    }
    let want = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(got, want, "\n--- got ---\n{got}");
}

/// What each production of the grammar is printed as in the kitchen sink: a substring of the
/// emit form. Every rule of ir-text.ebnf has a row (the next test).
const WHERE: &[(&str, &[&str])] = &[
    // lexical; `Comment` is never printed (`text_discipline`)
    ("Space", &[" "]),
    ("Comment", &[]),
    ("Num", &["type t0 int u 8\n"]),
    ("Word", &["fn @1 app.main("]),
    ("Val", &["  %2 = slot t"]),
    ("FnRef", &["call t2 @0 ", "fn @1.0 app.main", "fn @1.u app.main"]),
    ("Int", &[" -5", " 18446744073709551615", "claim @1 pre checked zero <= lenp 1 -1\n"]),
    ("Hex", &[" 0x3c00", " 0x3f800000", " 0x4000000000000000"]),
    ("String", &["str s1 \"\\x00\\x01", "\\\"odd\\\" \\xc3\\xa9", "\\n"]),
    ("Punct", &["(", ")", "<", ">", ",", ":", "=", "!", "|", "{", "}", "->", "<="]),
    // module sections; `Line` is the union of the line rules below
    ("Line", &[]),
    (
        "Path",
        &[
            "decl d0 struct app.Pair\n",
            "static 0 t",
            "site 1 \"src/\\\"odd\\\" \\xc3\\xa9.stv\" 40 1 \"my-pkg.f\"\n",
        ],
    ),
    ("HeaderLine", &["stave.ir 4 \"x86_64-linux-v2\" debug checked\n"]),
    ("Mode", &[" debug ", "stave.ir 4 \"x86_64-linux-v2\" release "]),
    ("Level", &[" checked\n", "debug lowered\n", "release opt\n"]),
    (
        "DeclLine",
        &["decl d0 struct app.Pair\n", "decl d1 enum app.Opt\n", "decl d7 struct \"my-pkg.Cell\"\n"],
    ),
    ("DeclKind", &[" struct ", " enum "]),
    ("TypeLine", &["type t0 int u 8\n"]),
    (
        "Key",
        &[
            "int s 8\n",
            "int u size\n",
            "float 16\n",
            "float 64\n",
            "float b16\n",
            "type t3 bool\n",
            "char\n",
            "str\n",
            "ref t",
            "ref mut t",
            "ptr const t",
            "ptr mut t",
            "array t",
            "slice t",
            " align 32\n",
            "tuple\n",
            "tuple t",
            "adt d0\n",
            "adt d1 t",
            " 8 -3\n",
            "fntype -> t",
            "fntype t",
            "vec t",
            " n\n",
            "mask 32 8\n",
            "mask 8 n\n",
        ],
    ),
    ("Sign", &["int s ", "int u "]),
    ("IntW", &["int s 8\n", "int s 16\n", "int s 32\n", "int s 64\n", "int s size\n"]),
    ("FloatW", &["float 16\n", "float 32\n", "float 64\n", "float b16\n"]),
    ("PtrMut", &["ptr const t", "ptr mut t"]),
    ("Lanes", &["vec t", " 8\n", " n\n"]),
    ("GArg", &["adt d1 t", " 8 -3\n"]),
    ("Feature", &["needs(avx2, fma)", "avx2 fma\n"]),
    ("ShapeLine", &["shape t"]),
    ("ShapeBody", &[" struct t", " enum | | t"]),
    ("Variant", &[" enum | | t"]),
    ("StrLine", &["str s0 \"hi\\n\"\n"]),
    ("SiteLine", &["site 0 \"src/app.stv\" 3 5 app.main\n"]),
    ("StaticLine", &["static 0 t", "static 1 t"]),
    ("DataLine", &["data 1 0 x0102", "data 1 130 xab", "data 1 199 x07\n"]),
    ("RefLine", &["ref 1 8 s0 0\n", "ref 1 16 static 0 4\n", "ref 1 24 s1 256\n"]),
    ("RefTarget", &[" s0 ", " static 0 "]),
    ("MainLine", &["main @1\n"]),
    ("CellLine", &["cell d7\n"]),
    // stack section
    ("RunStackLine", &["runstack 65536\n"]),
    ("TruncatedLine", &["truncated\n"]),
    ("NodeLine", &["node 0\n", "node 1 rec\n"]),
    ("EdgeLine", &["edge 0 1 call\n"]),
    ("EdgeKind", &[" call\n", " indirect\n", " spawn\n"]),
    ("NodesLine", &["nodes @1 0 1\n", "nodes @1.0 2\n"]),
    // functions
    ("FnLine", &["fn @1 app.main(t", "fn @4 \"my-pkg.f\"() unbuilt\n"]),
    ("GenericArgs", &["app.gen<t", ",8,-1>("]),
    ("TypeList", &["app.main(t", ", t"]),
    ("FnRet", &[") -> t"]),
    ("Needs", &[" needs(avx2, fma) "]),
    ("Symbol", &[" symbol \"puts\" "]),
    ("NameSite", &[" site 0 {\n"]),
    ("ClaimLine", &["claim @1 pre checked zero <= lenp 1 -1\n", "claim @1 post lenresult <= lenp 1 3\n"]),
    ("ClaimKind", &["claim @1 pre ", "claim @1 pre checked ", "claim @1 post "]),
    ("CTerm", &[" zero ", " p 0 ", " lenp 1 ", " result ", " lenresult "]),
    // bodies
    ("BlockLine", &["b0(%0: t", "b2():\n"]),
    ("BlockParam", &["%1: t"]),
    ("ValueInst", &["  %3 = add t"]),
    ("EffectInst", &["  store () %"]),
    ("At", &[" !2\n"]),
    ("Opnd", &[" %0", " -5", " 0x3c00", " bf16 0x3f80", " s0", " @1.0", " t"]),
    ("ValueOp", &[" = add t", " = call t", " = make t", " = intrinsic.syscall3 t"]),
    ("Op0", &[" = slot t"]),
    // the op groups below are covered terminal by terminal by `every_op_of_the_grammar_prints_...`
    ("Op1", &[]),
    ("Op2", &[" = add t"]),
    ("Op3", &[]),
    ("VecOp0", &[]),
    ("VecOp1", &[]),
    ("VecOp2", &[]),
    ("VecOp3", &[]),
    (
        "EffectOp",
        &[
            "  store () ",
            "  call () @3\n",
            "  call.ind () %",
            "  intrinsic.copy_bytes () ",
            "  panic.bounds ()\n",
        ],
    ),
    ("EffOp0", &["  panic.bounds ()\n"]),
    ("EffOp2", &["  store () "]),
    ("VecStore3", &[]),
    ("TermLine", &["  switch %", "  cbr %", "  br b", "  ret ", "  ret\n", "  unreachable\n"]),
    ("Case", &[" 0:b1(", " -1:b2()", " 300:b3()"]),
    ("Target", &["b4(%"]),
];

#[test]
fn every_production_of_the_grammar_is_printed() {
    let text = emit_form();
    let lowered = {
        let (m, stack) = sink();
        let h = Header { target: Triple::X86_64(X86Level::V2), mode: Mode::Release, level: Level::Opt };
        let l = print_module(&m, &Header { level: Level::Lowered, ..header(Level::Lowered) }, None).unwrap();
        let o = print_module(&m, &h, Some(&stack)).unwrap();
        format!("{l}{o}")
    };
    let all = format!("{text}{lowered}");
    let rules: BTreeSet<String> = grammar_rules().into_iter().map(|(n, _)| n).collect();
    let covered: BTreeSet<String> = WHERE.iter().map(|(n, _)| n.to_string()).collect();
    assert_eq!(rules.difference(&covered).collect::<Vec<_>>(), Vec::<&String>::new(), "rules without a row");
    assert_eq!(covered.difference(&rules).collect::<Vec<_>>(), Vec::<&String>::new(), "rows of no rule");
    let missing: Vec<String> = WHERE
        .iter()
        .flat_map(|(rule, needles)| needles.iter().map(move |n| (rule, n)))
        .filter(|(_, n)| !all.contains(**n))
        .map(|(rule, n)| format!("{rule}: {n:?}"))
        .collect();
    assert!(missing.is_empty(), "not printed:\n{}", missing.join("\n"));
}

/// The op productions are covered op by op: every terminal of every op rule is printed in the
/// operand shape of its rule, and the other ops of `ValueOp` / `EffectOp` in the kitchen sink.
#[test]
fn every_op_of_the_grammar_prints_in_its_group_shape() {
    let mut scalar: Vec<Op> = vec![
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
        Op::Popcount,
        Op::Clz,
        Op::Ctz,
        Op::MulWide,
        Op::Eq,
        Op::Ne,
        Op::Lt,
        Op::Le,
        Op::Gt,
        Op::Ge,
        Op::Select,
        Op::Make,
        Op::Get,
        Op::Set,
        Op::EnumMake,
        Op::EnumTag,
        Op::EnumGet,
        Op::Slot,
        Op::Load,
        Op::Store,
        Op::FieldAddr,
        Op::EnumFieldAddr,
        Op::ElemAddr,
        Op::SliceMake,
        Op::SliceLen,
        Op::SlicePtr,
        Op::FnMake,
        Op::BoundsCheck,
        Op::StoreStream,
        Op::FenceStream,
        Op::Call,
        Op::CallInd,
    ];
    for k in [CastKind::Exact, CastKind::Wrap, CastKind::Sat, CastKind::Round, CastKind::Try] {
        scalar.push(Op::Cast(k));
    }
    scalar.extend(Prefetch::ALL.map(Op::Prefetch));
    scalar.extend(PanicKind::ALL.map(Op::Panic));
    scalar.extend(VecOp::ALL.map(Op::Vec));
    let by_name = |name: &str| -> Op {
        *scalar.iter().find(|o| o.name() == Some(name)).unwrap_or_else(|| panic!("no op named {name}"))
    };

    // (rule, operands after the result type, effect form)
    let groups: &[(&str, usize, bool)] = &[
        ("Op0", 0, false),
        ("Op1", 1, false),
        ("Op2", 2, false),
        ("Op3", 3, false),
        ("VecOp0", 0, false),
        ("VecOp1", 1, false),
        ("VecOp2", 2, false),
        ("VecOp3", 3, false),
        ("EffOp0", 0, true),
        ("EffOp2", 2, true),
        ("VecStore3", 3, true),
    ];
    let mut m = Module::new();
    let t0 = m.types.i32();
    let f = m.declare(inst(&[0]), "app.ops", vec![t0, t0, t0, t0], Some(t0));
    let mut b = FuncBuilder::new(&[t0, t0, t0, t0]);
    let ps: Vec<Operand> = (0..4).map(|i| b.param(i).unwrap().into()).collect();
    let mut want: Vec<String> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for (rule, n, effect) in groups {
        for name in terminals(rule) {
            let op = by_name(&name);
            let args = &ps[..*n];
            let v = b.inst(op, (!effect).then_some(t0), args);
            let ops: String = (0..*n).map(|i| format!(" %{i}")).collect();
            want.push(match v {
                Some(_) => format!("  %{{}} = {name} t0{ops}"),
                None => format!("  {name} (){ops}"),
            });
            assert!(seen.insert(name.clone()), "{name} is in two groups");
        }
    }
    // The ops with a shape of their own.
    let special: &[(&str, usize, bool)] = &[
        ("make", 2, false),
        ("enum.make", 2, false),
        ("enum.get", 3, false),
        ("addr.field", 2, false),
        ("addr.enum_field", 3, false),
        ("vec.div_const", 2, false),
        ("vec.permute", 3, false),
        ("vec.store_masked", 4, true),
    ];
    for (name, n, effect) in special {
        let op = by_name(name);
        let v = b.inst(op, (!effect).then_some(t0), &ps[..*n]);
        let ops: String = (0..*n).map(|i| format!(" %{i}")).collect();
        want.push(match v {
            Some(_) => format!("  %{{}} = {name} t0{ops}"),
            None => format!("  {name} (){ops}"),
        });
        seen.insert(name.to_string());
    }
    b.ret(Some(ps[0]));
    m.define(f, b.finish().unwrap());
    let text = print_module(&m, &header(Level::Checked), None).unwrap();
    let lines: Vec<&str> = text.lines().filter(|l| l.starts_with("  ") && !l.starts_with("  ret")).collect();
    assert_eq!(lines.len(), want.len());
    let mut next = 4usize;
    for (got, w) in lines.iter().zip(&want) {
        let expect = if w.contains("{}") {
            let s = w.replacen("{}", &next.to_string(), 1);
            next += 1;
            s
        } else {
            w.clone()
        };
        assert_eq!(*got, expect);
    }

    // Every op terminal of the grammar was reached: the groups, the special forms, and the ops
    // whose operands are not all values (`call`, `call.ind`, `fn.make`, intrinsics).
    let in_value_op: BTreeSet<String> = terminals("ValueOp")
        .into_iter()
        .chain(terminals("EffectOp"))
        .filter(|n| n.chars().all(|c| c.is_ascii_lowercase() || c == '.' || c == '_'))
        .collect();
    let rest: Vec<&String> = in_value_op.iter().filter(|n| !seen.contains(*n)).collect();
    assert_eq!(rest, ["call", "call.ind", "fn.make"], "ops of ValueOp / EffectOp not in a group");
}

/// Every Feature, Triple, mode and level spells as the contract says, and the keyword lists of
/// the grammar equal the code's.
#[test]
fn keyword_lists_equal_the_code() {
    let names: BTreeSet<String> = Feature::ALL.iter().map(|f| f.name().to_string()).collect();
    assert_eq!(names, terminals("Feature").into_iter().collect());
    let mut got = BTreeSet::new();
    for f in Feature::ALL {
        let mut m = Module::new();
        let d = m.declare(inst(&[0]), "a.f", vec![], None);
        m.func_mut(d).unwrap().features = FeatureSet::of(f);
        let text = print_module(&m, &header(Level::Checked), None).unwrap();
        let line = text.lines().find(|l| l.starts_with("fn ")).unwrap().to_string();
        let inner = line.split("needs(").nth(1).unwrap().split(')').next().unwrap().to_string();
        assert!(inner.split(", ").any(|x| x == f.name()), "{line}");
        got.extend(inner.split(", ").map(str::to_string));
    }
    assert_eq!(got, names);
    for (t, name) in Triple::ALL.iter().zip([
        "x86_64-linux-v2",
        "x86_64-linux-v3",
        "x86_64-linux-v4",
        "aarch64-linux",
        "aarch64-linux+sve",
        "aarch64-linux+sve2",
        "gpu-sm86",
    ]) {
        let h = Header { target: *t, mode: Mode::Release, level: Level::Opt };
        let text = print_module(&Module::new(), &h, None).unwrap();
        assert_eq!(text, format!("stave.ir 4 \"{name}\" release opt\n"));
    }
    assert_eq!(terminals("Mode"), ["debug", "release"]);
    assert_eq!(terminals("Level"), ["lowered", "checked", "opt"]);
    assert_eq!(terminals("DeclKind"), ["struct", "enum"]);
    assert_eq!(terminals("EdgeKind"), ["call", "indirect", "spawn"]);
}

/// The printer emits no comment, no blank line, no tab, no double or trailing space, no `\r`,
/// and only printable ASCII; instruction lines start with two spaces, every other line in
/// column 1.
#[test]
fn text_discipline() {
    let text = emit_form();
    assert!(text.is_ascii() && text.ends_with('\n'));
    assert!(!text.contains(['\r', '\t']));
    let mut in_body = false;
    for l in text.lines() {
        assert!(!l.is_empty() && !l.ends_with(' '), "{l:?}");
        let body = l.trim_start_matches(' ');
        assert!(!body.starts_with('#'), "{l:?}");
        // a string may hold two spaces; the lines without one are checked whole
        if !l.contains('"') {
            assert!(!body.contains("  "), "{l:?}");
        }
        let indent = l.len() - body.len();
        let is_block_line = in_body && (l == "}" || (l.starts_with('b') && l.ends_with(':')));
        assert_eq!(indent, if in_body && !is_block_line { 2 } else { 0 }, "{l:?}");
        if l.starts_with("fn ") {
            in_body = l.ends_with(" {");
        } else if l == "}" {
            in_body = false;
        }
    }
}

#[test]
fn the_gate_form_is_the_emit_form_without_the_stack_lines() {
    let (m, stack) = sink();
    let h = header(Level::Checked);
    let emit = print_module(&m, &h, Some(&stack)).unwrap();
    let gate = print_module(&m, &h, None).unwrap();
    let is_stack = |l: &str| {
        l == "truncated" || ["runstack ", "node ", "edge ", "nodes "].iter().any(|p| l.starts_with(p))
    };
    let filtered: String = emit.lines().filter(|l| !is_stack(l)).flat_map(|l| [l, "\n"]).collect();
    assert_eq!(filtered, gate);
    assert!(emit.len() > gate.len());
}

#[test]
fn lowered_prints_no_claims_and_levels_differ_in_the_header_only() {
    let (m, _) = sink();
    let checked = print_module(&m, &header(Level::Checked), None).unwrap();
    let lowered = print_module(&m, &header(Level::Lowered), None).unwrap();
    assert!(checked.contains("claim @1 "));
    assert!(!lowered.contains("claim "));
    let strip = |t: &str| -> String {
        t.lines()
            .filter(|l| !l.starts_with("claim ") && !l.starts_with("stave.ir"))
            .flat_map(|l| [l, "\n"])
            .collect()
    };
    assert_eq!(strip(&checked), strip(&lowered));
    assert!(lowered.starts_with("stave.ir 4 \"x86_64-linux-v2\" debug lowered\n"));
}

/// The output is a function of the module: twice the same, and for a module that is not
/// canonical (functions out of order, an unreachable block, scattered numbering) the text of
/// its canonical form.
#[test]
fn print_is_a_function_of_the_module() {
    let (m, stack) = sink();
    let h = header(Level::Checked);
    let a = print_module(&m, &h, Some(&stack)).unwrap();
    assert_eq!(a, print_module(&m, &h, Some(&stack)).unwrap());
    assert_eq!(a, print_module(&m.clone(), &h, Some(&stack)).unwrap());
    assert!(canon::is_canonical(&m));
}

/// A module whose functions, blocks and values are not in canonical order prints as its
/// canonical form: the entry point and the stack rows follow their functions.
#[test]
fn a_module_that_is_not_canonical_prints_as_its_canonical_form() {
    let build = |swapped: bool| -> (Module, StackInput) {
        let mut m = Module::new();
        let i32t = m.types.i32();
        let (callee, caller);
        if swapped {
            callee = m.declare(inst(&[5]), "app.callee", vec![i32t], Some(i32t));
            caller = m.declare(inst(&[1]), "app.main", vec![], Some(i32t));
        } else {
            caller = m.declare(inst(&[1]), "app.main", vec![], Some(i32t));
            callee = m.declare(inst(&[5]), "app.callee", vec![i32t], Some(i32t));
        }
        let mut b = FuncBuilder::new(&[i32t]);
        let x = b.param(0).unwrap();
        b.ret(Some(x.into()));
        m.define(callee, b.finish().unwrap());
        let mut b = FuncBuilder::new(&[]);
        let dead = b.create_block();
        let r = b.call(callee, Some(i32t), &[Operand::Func(callee)]).unwrap();
        b.ret(Some(r.into()));
        b.switch_to(dead);
        let y = b.int(1);
        b.ret(Some(y));
        m.define(caller, b.finish().unwrap());
        let mut nodes_of_func = vec![vec![], vec![]];
        nodes_of_func[caller.index()] = vec![0];
        nodes_of_func[callee.index()] = vec![1];
        let stack = StackInput { nodes: vec![false, false], nodes_of_func, ..StackInput::default() };
        m.main = Some(caller);
        (m, stack)
    };
    let h = header(Level::Checked);
    let (a, astack) = build(false);
    let (b, bstack) = build(true);
    assert!(!canon::is_canonical(&a), "an unreachable block");
    assert!(!canon::is_canonical(&b));
    let ta = print_module(&a, &h, Some(&astack)).unwrap();
    assert_eq!(ta, print_module(&b, &h, Some(&bstack)).unwrap());
    assert!(ta.contains("main @1\n") && ta.contains("nodes @1 0\nnodes @5 1\n"), "{ta}");
    assert!(!ta.contains("b1():"), "the unreachable block is dropped: {ta}");
}

/// A module the verifier refuses is not printed: an id that names nothing, a decl that is
/// missing, a generic parameter key, a site past the table. Never a panic.
#[test]
fn a_module_the_verifier_refuses_is_not_printed() {
    let mut m = Module::new();
    let t0 = m.types.i32();
    let f = m.declare(inst(&[0]), "app.f", vec![t0], Some(t0));
    let mut b = FuncBuilder::new(&[t0]);
    let x = b.param(0).unwrap();
    let v = b.op(Op::Add, t0, &[x.into(), Operand::Func(FuncId(77))]);
    b.ret(Some(v.into()));
    m.define(f, b.finish().unwrap());
    m.main = Some(FuncId(9));
    let refused = |m: &Module| print_module(m, &header(Level::Checked), None).unwrap_err();
    let es = refused(&m);
    assert!(es.iter().any(|e| e.msg.contains("main: fn #9 out of range")), "{es:?}");
    assert!(es.iter().any(|e| e.msg.contains("out of range")), "{es:?}");
    let _ = (StrId(0), Block(0));

    let ok = |extra: &dyn Fn(&mut Module)| {
        let mut m = Module::new();
        let t0 = m.types.i32();
        let f = m.declare(inst(&[0]), "app.f", vec![t0], Some(t0));
        let mut b = FuncBuilder::new(&[t0]);
        let x = b.param(0).unwrap();
        b.ret(Some(x.into()));
        m.define(f, b.finish().unwrap());
        extra(&mut m);
        m
    };
    assert!(print_module(&ok(&|_| {}), &header(Level::Checked), None).is_ok());
    let no_decl = ok(&|m| {
        let t = m.types.adt(DeclId(3), vec![], AdtShape::structure(vec![]));
        m.declare(inst(&[1]), "app.g", vec![t], None);
    });
    let es = refused(&no_decl);
    assert!(es.iter().any(|e| e.msg.contains("adt d3 has no decl")), "{es:?}");
    let param = ok(&|m| {
        let t = m.types.intern(TypeKey::Param(0));
        m.declare(inst(&[1]), "app.g", vec![t], None);
    });
    let es = refused(&param);
    assert!(es.iter().any(|e| e.msg.contains("a generic parameter in a monomorphized module")), "{es:?}");
    let site = ok(&|m| m.funcs[0].name_site = Site(4));
    let es = refused(&site);
    assert!(es.iter().any(|e| e.msg.contains("name site 4 out of range")), "{es:?}");
    let unnamed =
        ok(&|m| m.decls.push(DeclInfo { id: DeclId(5), kind: AdtKind::Struct, path: "app.S".into() }));
    let es = refused(&unnamed);
    assert!(es.iter().any(|e| e.msg.contains("decl d5: no type or `cell` names it")), "{es:?}");
}
