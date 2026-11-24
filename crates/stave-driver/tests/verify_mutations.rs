//! The IR verifier is the gate for modules read from text: a module that verifies must not
//! panic either back end (debug and release builds, the release passes included) or the
//! reference interpreter. Valid modules of the run corpus get one single-field mutation each;
//! every mutant is verified, and every mutant the verifier accepts goes through the whole
//! pipeline under `catch_unwind`. `STAVE_MUTANTS` sets the number of mutants per program.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

use stave_driver::build::codegen_link;
use stave_driver::interp::{self, Limits};
use stave_driver::pipeline::{self, Goal, Options};
use stave_driver::shipped::Library;
use stave_ir::ir::{ConstId, Target, ValueDef};
use stave_ir::{
    Block, CastKind, Const, FuncId, IntrinsicId, Known, Module, Op, Operand, PanicKind, Site, StrId, Term,
    TypeId, Value, VecOp,
};
use stave_target::Triple;

const PROGRAMS: [&str; 6] = [
    "tests/conformance/run/atomic_ops.stv",
    "tests/conformance/run/soa_cols.stv",
    "tests/conformance/run/orpat_arms.stv",
    "tests/conformance/run/static_str.stv",
    "tests/conformance/run/rank_threads.stv",
    "tests/conformance/simd/combinators_dot_len.stv",
];

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// xorshift64*: a fixed seed per program, so a failure replays.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform in `0..n` (`0` for `n == 0`).
    fn below(&mut self, n: usize) -> usize {
        if n == 0 { 0 } else { (self.next() % n as u64) as usize }
    }

    fn flip(&mut self) -> bool {
        self.next() & 1 == 1
    }
}

/// Each program is lowered for each target, and a target's back end builds only its own module:
/// lowering differs (vector widths, intrinsics), and a back end stops at the first fn it cannot
/// build, so a module of the other target never reaches the program's own code.
const TARGETS: [&str; 2] = ["x86_64-linux", "aarch64-linux"];

fn lowered(file: &str, target: &str) -> (Module, FuncId) {
    let mut o = Options::new(Goal::Check).threads(Some(2));
    o.lower = true;
    o.target = Some(Triple::parse(target).unwrap());
    let out = pipeline::run(&repo().join(file), &Library::in_dir(&repo()), &o).unwrap();
    assert!(!out.has_errors(), "{file}: {:?}", out.diagnostics);
    let main = out.main.unwrap_or_else(|| panic!("{file}: no main"));
    (out.module.expect("checked IR"), main)
}

/// Int constants at the edges of every type and of every shift, index, and size rule.
const EDGES: [i128; 16] = [
    0,
    1,
    -1,
    2,
    7,
    8,
    31,
    32,
    63,
    64,
    255,
    1 << 40,
    i64::MIN as i128,
    i64::MAX as i128,
    u64::MAX as i128,
    i128::MIN,
];

/// The ops a mutation may put in place of another: every scalar op and a sample of vector ops.
fn ops(m: &Module, r: &mut Rng) -> Op {
    use Op::*;
    const SCALAR: [Op; 52] = [
        Add,
        Sub,
        Mul,
        Div,
        Rem,
        Neg,
        AddWrap,
        SubWrap,
        MulWrap,
        ShlWrap,
        AddSat,
        SubSat,
        MulSat,
        Shl,
        Shr,
        And,
        Or,
        Xor,
        Not,
        Popcount,
        Clz,
        Ctz,
        MulWide,
        Eq,
        Ne,
        Lt,
        Le,
        Gt,
        Ge,
        Select,
        Cast(CastKind::Exact),
        Cast(CastKind::Wrap),
        Cast(CastKind::Sat),
        Cast(CastKind::Round),
        Cast(CastKind::Try),
        Make,
        Get,
        Set,
        EnumMake,
        EnumTag,
        EnumGet,
        Slot,
        Load,
        Store,
        FieldAddr,
        EnumFieldAddr,
        ElemAddr,
        SliceMake,
        SliceLen,
        SlicePtr,
        FnMake,
        BoundsCheck,
    ];
    match r.below(8) {
        0 => Vec(VecOp::ALL[r.below(VecOp::ALL.len())]),
        1 => Intrinsic(IntrinsicId(r.below(m.intrinsics.len() + 1) as u32)),
        2 => Panic(PanicKind::ALL[r.below(PanicKind::ALL.len())]),
        3 => [Call, CallInd, StoreStream, FenceStream][r.below(4)],
        _ => SCALAR[r.below(SCALAR.len())],
    }
}

fn some_type(m: &Module, r: &mut Rng) -> TypeId {
    TypeId(r.below(m.types.len() + 1) as u32)
}

fn operand(m: &Module, nv: usize, nc: usize, r: &mut Rng) -> Operand {
    match r.below(6) {
        0 | 1 => Operand::Value(Value(r.below(nv + 1) as u32)),
        2 => Operand::Const(ConstId(r.below(nc + 1) as u32)),
        3 => Operand::Str(StrId(r.below(m.strings.len() + 1) as u32)),
        4 => Operand::Func(FuncId(r.below(m.funcs.len() + 1) as u32)),
        _ => Operand::Type(some_type(m, r)),
    }
}

/// Intrinsic names: the table's and a few it lacks, near-miss spellings among them.
const NAMES: [&str; 28] = [
    "syscall0",
    "syscall3",
    "syscall6",
    "ccall0",
    "ccall5",
    "ccall12",
    "cpuid",
    "xgetbv",
    "thread_start_addr",
    "main_addr",
    "signal_handler_addr",
    "site_table",
    "pc_table",
    "stack_table",
    "thread_block",
    "set_thread_block",
    "call_on_stack",
    "cpu_init",
    "thread_clone",
    "atomic_cas_u32",
    "copy_bytes",
    "static_addr",
    "tid_x",
    "nope",
    "syscall03",
    "syscall+3",
    "ccall013",
    "atomic_load_u16",
];

/// Edits the block or the argument span of a jump target; returns what it changed.
fn edit_target(t: &mut Target, nblocks: usize, noperands: usize, r: &mut Rng) -> &'static str {
    match r.below(4) {
        0 => {
            t.block = Block(r.below(nblocks + 1) as u32);
            "block"
        }
        1 => {
            t.args.len = t.args.len.wrapping_add(1);
            "args len + 1"
        }
        2 => {
            t.args.len = t.args.len.saturating_sub(1);
            "args len - 1"
        }
        _ => {
            t.args.start = r.below(noperands + 1) as u32;
            "args start"
        }
    }
}

/// One single-field mutation of `m`; returns what it changed.
fn mutate(m: &mut Module, r: &mut Rng) -> String {
    let with_body: Vec<usize> = (0..m.funcs.len()).filter(|&i| m.funcs[i].body.is_some()).collect();
    let kind = r.below(28);
    // the control-flow mutations go to a fn that has the structure they edit
    let has = |i: usize| {
        let b = m.funcs[i].body.as_ref().expect("body");
        let terms = || b.blocks.iter().map(|bd| bd.term);
        match kind {
            24 => terms().any(|t| matches!(t, Term::Br(_) | Term::Cbr { .. } | Term::Switch { .. })),
            25 => !b.cases.is_empty(),
            26 => terms().any(|t| matches!(t, Term::Cbr { .. })),
            27 => terms().any(|t| matches!(t, Term::Cbr { .. } | Term::Switch { .. })),
            _ => true,
        }
    };
    let pool: Vec<usize> = with_body.iter().copied().filter(|&i| has(i)).collect();
    let pool = if pool.is_empty() { with_body } else { pool };
    let fi = pool[r.below(pool.len())];
    let ntypes = m.types.len();
    let nfuncs = m.funcs.len();
    let nsites = m.sites.len();
    if r.below(10) == 0 {
        // a huge or malformed type, as the type of a fn's param (callers then mismatch, so this
        // passes the verifier only for a fn nothing calls)
        let u8t = m.types.u8();
        let big = m.types.array(u8t, 1 << 62);
        let t = match r.below(5) {
            0 => m.types.array(u8t, 1 << 40),
            1 => m.types.array(big, 8),
            2 => m.types.intern(stave_ir::TypeKey::Vec { lane: big, n: stave_ir::Lanes::Fixed(8) }),
            3 => m.types.array(u8t, 1 << 28),
            _ => m.types.tuple(vec![big, big, big]),
        };
        let f = &mut m.funcs[fi];
        let b = f.body.as_mut().expect("body");
        let ps: Vec<Value> = b.params(Block(0)).to_vec();
        if let Some(k) = (!ps.is_empty()).then(|| r.below(ps.len())) {
            f.params[k] = t;
            b.values[ps[k].index()].ty = t;
        }
        return format!("fn {fi}: param of type {}", m.types.key_text(t));
    }
    if kind >= 14 {
        let snap = m.clone();
        let b = m.funcs[fi].body.as_mut().expect("body");
        let ni = b.insts.op.len();
        let (nv, nc) = (b.values.len(), b.consts.len());
        let i = r.below(ni);
        return match kind {
            14 | 15 if ni > 0 => {
                let op = ops(&snap, r);
                b.insts.op[i] = op;
                format!("fn {fi} inst {i}: op {op:?}")
            }
            16 if ni > 0 => {
                let t = if r.below(6) == 0 { None } else { Some(some_type(&snap, r)) };
                b.insts.ty[i] = t;
                if let Some(v) = b.insts.result[i]
                    && let (Some(t), Some(d)) = (t, b.values.get_mut(v.index()))
                {
                    d.ty = t;
                }
                format!("fn {fi} inst {i}: result type {t:?}")
            }
            17 if !b.operands.is_empty() => {
                let k = r.below(b.operands.len());
                // a same-kind operand half of the time: more mutants pass the type rules
                let o = match (b.operands[k], r.flip()) {
                    (Operand::Value(_), true) => Operand::Value(Value(r.below(nv) as u32)),
                    _ => operand(&snap, nv, nc, r),
                };
                b.operands[k] = o;
                format!("fn {fi} operand {k}: {o:?}")
            }
            18 if nc > 0 => {
                let k = r.below(nc);
                let c = match r.below(6) {
                    0 => Const::F32(r.next() as u32),
                    1 => Const::F64(r.next()),
                    2 => Const::F16(r.next() as u16),
                    _ => Const::Int(EDGES[r.below(EDGES.len())]),
                };
                b.consts[k] = c;
                format!("fn {fi} const {k}: {c:?}")
            }
            19 if ni > 0 => {
                let s = if r.below(4) == 0 { Site::NONE } else { Site(r.below(nsites + 2) as u32) };
                b.insts.site[i] = s;
                format!("fn {fi} inst {i}: site {s:?}")
            }
            20 if nv > 0 => {
                let k = r.below(nv);
                let t = some_type(&snap, r);
                b.values[k].ty = t;
                if let ValueDef::Inst(j) = b.values[k].def
                    && let Some(x) = b.insts.ty.get_mut(j as usize)
                    && x.is_some()
                {
                    *x = Some(t);
                }
                format!("fn {fi} value {k}: type {t:?}")
            }
            21 => {
                let k = r.below(b.blocks.len());
                let nb = b.blocks.len();
                let term = &mut b.blocks[k].term;
                let what = match term {
                    Term::Br(t) | Term::Cbr { then: t, .. } | Term::Switch { default: t, .. } => {
                        t.block = Block(r.below(nb + 1) as u32);
                        "target"
                    }
                    Term::Ret(o) => {
                        *o = if r.flip() { None } else { Some(operand(&snap, nv, nc, r)) };
                        "ret"
                    }
                    Term::Unreachable => {
                        *term = Term::Ret(None);
                        "unreachable"
                    }
                };
                format!("fn {fi} b{k}: terminator {what}")
            }
            24 => {
                // a jump's target: block or argument span, of `br`, `cbr`, or a switch default
                let nb = b.blocks.len();
                let nops = b.operands.len();
                let jumps: Vec<usize> = (0..nb)
                    .filter(|&k| !matches!(b.blocks[k].term, Term::Ret(_) | Term::Unreachable))
                    .collect();
                let k = jumps.get(r.below(jumps.len().max(1))).copied().unwrap_or(0);
                let t = match &mut b.blocks[k].term {
                    Term::Br(t) | Term::Switch { default: t, .. } => Some(t),
                    Term::Cbr { then, els, .. } => Some(if r.flip() { then } else { els }),
                    Term::Ret(_) | Term::Unreachable => None,
                };
                match t {
                    Some(t) => format!("fn {fi} b{k}: jump target {}", edit_target(t, nb, nops, r)),
                    None => "nothing".to_string(),
                }
            }
            25 if !b.cases.is_empty() => {
                let (nb, nops) = (b.blocks.len(), b.operands.len());
                let k = r.below(b.cases.len());
                let what = edit_target(&mut b.cases[k].target, nb, nops, r);
                format!("fn {fi} case {k}: target {what}")
            }
            26 => {
                // the else target of a conditional branch
                let (nb, nops) = (b.blocks.len(), b.operands.len());
                let cbrs: Vec<usize> =
                    (0..nb).filter(|&k| matches!(b.blocks[k].term, Term::Cbr { .. })).collect();
                match cbrs.get(r.below(cbrs.len().max(1))) {
                    Some(&k) => {
                        let Term::Cbr { els, .. } = &mut b.blocks[k].term else { unreachable!() };
                        format!("fn {fi} b{k}: else target {}", edit_target(els, nb, nops, r))
                    }
                    None => "nothing".to_string(),
                }
            }
            27 => {
                // the condition of a `cbr`, or the scrutinee of a `switch`
                let sel: Vec<usize> = (0..b.blocks.len())
                    .filter(|&k| matches!(b.blocks[k].term, Term::Cbr { .. } | Term::Switch { .. }))
                    .collect();
                match sel.get(r.below(sel.len().max(1))) {
                    Some(&k) => {
                        let o = match (r.flip(), nv) {
                            (true, 1..) => Operand::Value(Value(r.below(nv) as u32)),
                            _ => operand(&snap, nv, nc, r),
                        };
                        match &mut b.blocks[k].term {
                            Term::Cbr { cond: x, .. } | Term::Switch { scrut: x, .. } => *x = o,
                            Term::Br(_) | Term::Ret(_) | Term::Unreachable => {}
                        }
                        format!("fn {fi} b{k}: branch operand {o:?}")
                    }
                    None => "nothing".to_string(),
                }
            }
            22 if !b.cases.is_empty() => {
                let k = r.below(b.cases.len());
                b.cases[k].value = EDGES[r.below(EDGES.len())];
                format!("fn {fi} case {k}: {}", b.cases[k].value)
            }
            23 if ni > 0 => {
                let s = &mut b.insts.args[i];
                match r.below(3) {
                    0 => s.len = s.len.wrapping_add(1),
                    1 => s.len = s.len.saturating_sub(1),
                    _ => s.start = s.start.saturating_sub(1),
                }
                format!("fn {fi} inst {i}: operand span {s:?}")
            }
            _ => "nothing".to_string(),
        };
    }
    let gi = r.below(nfuncs);
    match kind {
        0 => {
            let f = &mut m.funcs[gi];
            let t = TypeId(r.below(ntypes + 1) as u32);
            if f.params.is_empty() || r.below(4) == 0 {
                f.params.push(t);
            } else {
                let k = r.below(f.params.len());
                f.params[k] = t;
            }
            format!("fn {gi}: param {t:?}")
        }
        1 => {
            let t = if r.flip() { None } else { Some(TypeId(r.below(ntypes + 1) as u32)) };
            m.funcs[gi].ret = t;
            format!("fn {gi}: ret {t:?}")
        }
        2 | 3 => {
            let k = if r.below(5) == 0 { None } else { Some(Known::ALL[r.below(Known::ALL.len())]) };
            m.funcs[gi].known = k;
            format!("fn {gi}: known {k:?}")
        }
        4 => {
            let f = &mut m.funcs[gi];
            match r.below(3) {
                0 => f.alt_stack = !f.alt_stack,
                1 => f.variadic = !f.variadic,
                _ => f.unbuilt = !f.unbuilt,
            }
            format!("fn {gi}: flags {} {} {}", f.alt_stack, f.variadic, f.unbuilt)
        }
        5 => {
            let s = Site(r.below(nsites + 2) as u32);
            m.funcs[gi].name_site = s;
            format!("fn {gi}: name site {s:?}")
        }
        6 => {
            let s = if r.flip() { None } else { Some(["main", "", "write", "x y"][r.below(4)].to_string()) };
            m.funcs[gi].symbol = s.clone();
            format!("fn {gi}: symbol {s:?}")
        }
        7 => {
            m.funcs[gi].body = None;
            format!("fn {gi}: no body")
        }
        8 if !m.intrinsics.is_empty() => {
            let k = r.below(m.intrinsics.len());
            let n = NAMES[r.below(NAMES.len())];
            m.intrinsics[k] = n.to_string();
            format!("intrinsic {k}: {n}")
        }
        9 if !m.strings.is_empty() => {
            let k = r.below(m.strings.len());
            m.strings[k].truncate(r.below(2));
            format!("string {k}: truncated")
        }
        10 if !m.statics.is_empty() => {
            let k = r.below(m.statics.len());
            let st = &mut m.statics[k];
            match r.below(3) {
                0 => st.ty = TypeId(r.below(ntypes + 1) as u32),
                1 => st.init.truncate(st.init.len().saturating_sub(1)),
                _ => st.refs.clear(),
            }
            format!("static {k}: changed")
        }
        11 if !m.sites.is_empty() => {
            m.sites.truncate(r.below(nsites));
            format!("sites: {} rows", m.sites.len())
        }
        12 => {
            m.funcs.swap(gi, r.below(nfuncs));
            format!("fn {gi}: swapped")
        }
        13 => {
            m.stated.truncate(r.below(m.stated.len() + 1));
            format!("stated: {} rows", m.stated.len())
        }
        _ => "nothing".to_string(),
    }
}

fn verifies(m: &Module) -> bool {
    match stave_ir::verify::verify_types(m) {
        Ok(e) if e.is_empty() => {
            (0..m.funcs.len()).all(|f| stave_ir::verify::verify_func_body(m, FuncId(f as u32)).is_empty())
        }
        _ => false,
    }
}

/// Everything after the verifier, on a module lowered for `target`: its back end in one build mode,
/// and the interpreter.
fn downstream(m: &Module, main: FuncId, target: &str, release: bool) {
    let triple = Triple::parse(target).unwrap();
    let _ = codegen_link(m, main, triple, release, Some(1), None);
    if let Some(f) = m.func(main) {
        let limits = Limits { max_steps: 200_000, max_depth: 200, check: false, borrow: false };
        let _ = interp::run(m, &f.path.clone(), triple, &[], &[], limits);
    }
}

/// Mutants of every program, for each target: the verifier must not panic, and an accepted mutant
/// must not panic the target's back end or the interpreter.
#[test]
fn verified_mutants_do_not_panic() {
    let per: usize = std::env::var("STAVE_MUTANTS").ok().and_then(|s| s.parse().ok()).unwrap_or(40);
    let mut panics = Vec::new();
    for (ti, target) in TARGETS.iter().enumerate() {
        for (pi, file) in PROGRAMS.iter().enumerate() {
            let (m, main) = lowered(file, target);
            assert!(verifies(&m), "{file} for {target}: the unmutated module must verify");
            // the mutants reach the program's own code only if the back end builds the whole module
            for release in [false, true] {
                let triple = Triple::parse(target).unwrap();
                let built = codegen_link(&m, main, triple, release, Some(1), None);
                assert!(built.is_ok(), "{file} for {target} (release {release}): {:?}", built.err());
            }
            let mut r = Rng(0x9e37_79b9_7f4a_7c15 ^ (pi as u64 + 1) ^ ((ti as u64) << 32));
            for k in 0..per {
                let mut x = m.clone();
                let what = mutate(&mut x, &mut r);
                let ok = match catch_unwind(AssertUnwindSafe(|| verifies(&x))) {
                    Ok(ok) => ok,
                    Err(_) => {
                        panics.push(format!("{target} {file} #{k} ({what}): the verifier panicked"));
                        continue;
                    }
                };
                if !ok {
                    continue;
                }
                let release = k % 2 == 1;
                if catch_unwind(AssertUnwindSafe(|| downstream(&x, main, target, release))).is_err() {
                    panics.push(format!(
                        "{target} {file} #{k} ({what}, release {release}): verified, then a panic"
                    ));
                }
            }
        }
    }
    assert!(panics.is_empty(), "{} panics:\n{}", panics.len(), panics.join("\n"));
}
