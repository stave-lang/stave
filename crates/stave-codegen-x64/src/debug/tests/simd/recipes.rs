//! The v4 column of the lowering table against the code K mode emits.
//!
//! For every row of `stave-target/tables/x86.toml` whose op K mode lowers, and every lane type
//! of the row, a function applying the op once to 512-bit operands is compiled as a release
//! build for the v4 baseline and disassembled. Its instructions, without what a recipe does not
//! list (register copies, constants, moves to and from the frame, the prologue and epilogue,
//! the call boundary's mask conversions, and the trap checks of the op), must be the recipe's,
//! as a multiset: a recipe may group a repeated sequence by instruction.

use std::collections::BTreeMap;

use stave_ir::{CastKind, FuncBuilder, Op, Operand, TypeId, TypeKey, VecOp};
use stave_target::lowering::{Class, Tier};

use super::super::features::{Inst, disasm};
use super::*;
use crate::debug::{bench_ctx, compile_func};

fn lane_of(key: &str) -> Option<Lane> {
    Lane::ALL.into_iter().find(|l| l.name() == key)
}

/// One function under test: its params, what it returns, and the trap-check instructions of
/// its op.
struct Probe {
    params: Vec<TypeId>,
    ret: Option<TypeId>,
    /// Extra operands after the params (constants).
    consts: Vec<i128>,
    op: VecOp,
    /// Mnemonics of the op's trap checks (prefix match).
    checks: &'static [&'static str],
    /// Mask params and a mask result that cross the call as their bits.
    mask_params: usize,
    mask_ret: bool,
}

/// The probe of `op` on the 512-bit type of `key` (`form`: the shift amount's); `None` where
/// the row does not apply to a 512-bit K-mode value.
fn probe(sp: &mut Sp, op: VecOp, key: &str, form: &str) -> Option<Probe> {
    use VecOp::*;
    let t = sp.t();
    let mut p = Probe {
        params: Vec::new(),
        ret: None,
        consts: Vec::new(),
        op,
        checks: &[],
        mask_params: 0,
        mask_ret: false,
    };
    if let Some(w) = key.strip_prefix('m').and_then(|w| w.parse::<u32>().ok()) {
        let n = 512 / w;
        let m = sp.mt(w, n);
        match op {
            MaskAnd | MaskOr | MaskXor => {
                (p.params, p.ret, p.mask_params, p.mask_ret) = (vec![m, m], Some(m), 2, true)
            }
            MaskNot => (p.params, p.ret, p.mask_params, p.mask_ret) = (vec![m], Some(m), 1, true),
            MaskAll | MaskNone => (p.ret, p.mask_ret) = (Some(m), true),
            ToVec => (p.params, p.ret, p.mask_params) = (vec![m], Some(sp.vt(sh(Lane::uint(w), n))), 1),
            ToBits => {
                let bt = Lane::uint(stave_ir::simd::to_bits_width(n));
                (p.params, p.ret, p.mask_params) = (vec![m], Some(sp.lane_t(bt)), 1);
            }
            FirstN => (p.params, p.ret, p.mask_ret) = (vec![t.usize], Some(m), true),
            Rewiden => {
                // the same lanes at the neighbouring width, one side a 512-bit mask
                let (n, w2) = if w == 8 { (32, 16) } else { (n, w / 2) };
                let (from, to) = (sp.mt(w, n), sp.mt(w2, n));
                (p.params, p.ret) = (vec![from], Some(to));
                // only the call boundary's conversions of the two masks are left
                p.checks = &["vpmov", "kmov"];
            }
            _ => return None,
        }
        return Some(p);
    }
    if let Some((f, t)) = key.split_once('>') {
        // a cast: the wider side fills 512 bits
        let (from, to) = (lane_of(f)?, lane_of(t)?);
        let n = 512 / from.bits().max(to.bits());
        let src = sp.vt(sh(from, n));
        let ret = match op {
            Cast(CastKind::Try) => sp.res_t(Res::Opt(sh(to, n))),
            Cast(_) => sp.vt(sh(to, n)),
            _ => return None,
        };
        (p.params, p.ret) = (vec![src], Some(ret));
        return Some(p);
    }
    let lane = lane_of(key)?;
    let s = sh(lane, 512 / lane.bits());
    let v = sp.vt(s);
    let m = sp.mt(lane.bits(), s.n);
    let scalar = sp.lane_t(lane);
    // an aligned access needs a slice aligned to the vector
    let align = matches!(op, LoadAligned | StoreAligned).then_some(64);
    let slice = {
        let st = sp.p.m.types.intern(TypeKey::Slice { elem: scalar, align });
        sp.p.m.types.ref_(true, st)
    };
    match op {
        Splat => (p.params, p.ret) = (vec![scalar], Some(v)),
        Add | Sub | Mul | Div | MinX86 | MaxX86 | Min | Max | MinNum | MaxNum | Copysign | AddWrap
        | SubWrap | MulWrap | AddSat | SubSat | MulSat | And | Or | Xor | Andnot => {
            (p.params, p.ret) = (vec![v, v], Some(v))
        }
        Neg | Abs | Sqrt | Floor | Ceil | Round | Trunc | AbsWrap | Not | Popcount | Clz | ScanAdd
        | ScanAddWrap => (p.params, p.ret) = (vec![v], Some(v)),
        Fma => (p.params, p.ret) = (vec![v, v, v], Some(v)),
        CheckedAdd | CheckedSub | CheckedMul => (p.params, p.ret) = (vec![v, v], Some(sp.res_t(Res::Opt(s)))),
        DivConst => (p.params, p.ret, p.consts) = (vec![v], Some(v), vec![worst_divisor(lane)]),
        // reversed lanes: bytes cross their 128-bit blocks
        Permute => (p.params, p.ret, p.consts) = (vec![v], Some(v), (0..s.n).rev().map(i128::from).collect()),
        Lookup => (p.params, p.ret) = (vec![v, v], Some(v)),
        Compress | Expand => (p.params, p.ret, p.mask_params) = (vec![v, m], Some(v), 1),
        Gather | Scatter => {
            // an array of a constant length, which 32-bit indices reach
            let at = sp.p.m.types.intern(TypeKey::Array { elem: scalar, len: 1000 });
            let arr = sp.p.m.types.ref_(op == Scatter, at);
            let iv = sp.vt(sh(Lane::uint(lane.bits()), s.n));
            p.checks = &["vpcmp", "kortest"];
            if op == Gather {
                (p.params, p.ret) = (vec![arr, iv], Some(v));
            } else {
                p.params = vec![arr, iv, v];
            }
        }
        StoreStream => {
            let st = sp.p.m.types.intern(TypeKey::Slice { elem: scalar, align: Some(64) });
            let sl = sp.p.m.types.ref_(true, st);
            (p.params, p.checks) = (vec![sl, t.usize, v], &["test"]);
        }
        Shl | Shr | ShlWrap if form == "const" => (p.params, p.ret, p.consts) = (vec![v], Some(v), vec![3]),
        Shl | Shr => (p.params, p.ret, p.checks) = (vec![v, v], Some(v), &["vpcmpu", "kortest", "vpcmpnleu"]),
        ShlWrap => (p.params, p.ret) = (vec![v, v], Some(v)),
        Eq | Ne | Lt | Le | Gt | Ge => (p.params, p.ret, p.mask_ret) = (vec![v, v], Some(m), true),
        ReduceAdd | ReduceAddWrap | ReduceMin | ReduceMax | ReduceAnd | ReduceOr => {
            (p.params, p.ret) = (vec![v], Some(scalar))
        }
        ReduceAddOrdered => (p.params, p.ret) = (vec![v, scalar], Some(scalar)),
        Bitcast => {
            let other = if lane == Lane::U8 { Lane::I8 } else { Lane::U8 };
            (p.params, p.ret) = (vec![v], Some(sp.vt(sh(other, 64))));
        }
        Split => (p.params, p.ret) = (vec![v], Some(sp.res_t(Res::Pair(sh(lane, s.n / 2))))),
        Join => {
            let h = sp.vt(sh(lane, s.n / 2));
            (p.params, p.ret) = (vec![h, h], Some(v));
        }
        Load => (p.params, p.ret) = (vec![slice, t.usize], Some(v)),
        LoadAligned => (p.params, p.ret, p.checks) = (vec![slice, t.usize], Some(v), &["test"]),
        Store => p.params = vec![slice, t.usize, v],
        StoreAligned => (p.params, p.checks) = (vec![slice, t.usize, v], &["test"]),
        LoadMasked => {
            (p.params, p.ret, p.mask_params) = (vec![slice, t.usize, m], Some(v), 1);
            p.checks = &["kmov", "test", "bsr", "sub", "cmp"];
        }
        StoreMasked => {
            (p.params, p.mask_params) = (vec![slice, t.usize, v, m], 1);
            p.checks = &["kmov", "test", "bsr", "sub", "cmp"];
        }
        Select => (p.params, p.ret, p.mask_params) = (vec![m, v, v], Some(v), 1),
        WidenLo | WidenHi => (p.params, p.ret) = (vec![v], Some(sp.vt(sh(lane.wide()?, s.n / 2)))),
        NarrowWrap | NarrowSat => {
            let half = if lane.signed() { Lane::sint(lane.bits() / 2) } else { Lane::uint(lane.bits() / 2) };
            (p.params, p.ret) = (vec![v, v], Some(sp.vt(sh(half, s.n * 2))));
        }
        _ => return None,
    }
    Some(p)
}

/// A divisor whose sequence is the longest of the lane type's: a multiplier one bit too wide
/// (unsigned), or added to the dividend and negated (signed).
fn worst_divisor(l: Lane) -> i128 {
    match (l.bits(), l.signed()) {
        (8, false) => 7,
        (8, true) => -7,
        (16, false) => 7,
        (16, true) => -15,
        (32, false) => 7,
        (32, true) => -7,
        _ => 7,
    }
}

/// The instructions of the probe's function, compiled for the v4 baseline as a release build
/// (`None`: unsupported in K mode).
fn code_of(mut sp: Sp, p: &Probe) -> Option<Vec<Inst>> {
    let f = sp.p.decl("t.probe", &p.params, p.ret);
    let mut b = FuncBuilder::new(&p.params);
    let mut args: Vec<Operand> = (0..p.params.len()).map(|i| b.param(i).unwrap().into()).collect();
    for &k in &p.consts {
        args.push(b.int(k));
    }
    match p.ret {
        Some(rt) => {
            let r = b.op(Op::Vec(p.op), rt, &args);
            b.ret(Some(r.into()));
        }
        None => {
            b.inst(Op::Vec(p.op), None, &args);
            b.ret(None);
        }
    }
    sp.p.def(f, b);
    let m = &sp.p.m;
    stave_ir::verify_module(m).unwrap_or_else(|e| panic!("{}: IR does not verify: {e:?}", p.op.name()));
    let mut cx = bench_ctx(m);
    cx.release = true;
    cx.baseline = v4();
    let code = match compile_func(&cx, f) {
        Ok(c) => c.code,
        Err(Error::Unsupported { .. }) => return None,
        Err(e) => panic!("{}: {e}", p.op.name()),
    };
    let insts = disasm(&code);
    // the constant pool follows the code and is read `rip`-relative
    let pool = insts
        .iter()
        .filter(|i| i.ops.contains("[rip"))
        .filter_map(|i| i.target)
        .filter(|t| *t < code.len())
        .min()
        .unwrap_or(code.len());
    Some(insts.into_iter().filter(|i| i.off < pool).collect())
}

const MOVES: [&str; 10] = [
    "vmovdqa",
    "vmovdqa32",
    "vmovdqa64",
    "vmovdqu",
    "vmovdqu64",
    "vmovaps",
    "vmovapd",
    "vmovups",
    "vmovq",
    "kmovq",
];

/// An instruction no recipe lists.
fn unlisted(i: &Inst) -> bool {
    let m = i.mnem.as_str();
    let ops: Vec<&str> = i.ops.split(',').collect();
    let frame = matches!(m, "sub" | "add" | "and" | "or") && (ops[0] == "rsp" || i.ops.contains("[rsp"));
    let plumbing = matches!(
        m,
        "push"
            | "pop"
            | "mov"
            | "movabs"
            | "ret"
            | "leave"
            | "lea"
            | "vzeroupper"
            | "nop"
            | "ud2"
            | "call"
            | "movzx"
            | "movsx"
            | "movsxd"
    ) || m.starts_with('j');
    let masked = i.ops.contains("{k");
    let mem = i.ops.contains("PTR");
    let mov = MOVES.contains(&m) && !masked;
    // a register copy, a move to or from the frame, a constant
    let copy = mov && !mem && ops.len() == 2 && ops[0].as_bytes()[0] == ops[1].as_bytes()[0];
    let home = mov && i.ops.contains("[rbp");
    let konst = i.ops.contains("[rip") && (mov || m.contains("broadcast"));
    let zero =
        matches!(m, "vpxor" | "vpxord" | "vpxorq") && ops.len() == 3 && ops[0] == ops[1] && ops[1] == ops[2];
    frame || plumbing || copy || home || konst || zero
}

/// The compare mnemonic without the predicate objdump folds into it.
fn plain_compare(m: &str) -> Option<String> {
    let rest = m.strip_prefix("vpcmp")?;
    for p in ["neq", "nlt", "nle", "eq", "lt", "le"] {
        if let Some(r) = rest.strip_prefix(p)
            && matches!(r, "b" | "w" | "d" | "q" | "ub" | "uw" | "ud" | "uq")
        {
            return Some(format!("vpcmp{r}"));
        }
    }
    None
}

/// The instructions of a recipe, as mnemonic counts.
fn recipe_counts(recipe: &str) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    if recipe == "none" {
        return out;
    }
    for item in recipe.split(';') {
        let item = item.trim();
        let (head, k) = match item.rsplit_once(" x") {
            Some((h, k)) if k.bytes().all(|b| b.is_ascii_digit()) => (h, k.parse().unwrap()),
            _ => (item, 1),
        };
        let m = head.split_whitespace().next().unwrap();
        *out.entry(m.to_owned()).or_insert(0) += k;
    }
    out
}

/// The listed instructions of the code, as mnemonic counts.
fn code_counts(code: &[Inst], p: &Probe, want: &BTreeMap<String, usize>) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    let (mut params, mut ret) = (p.mask_params, p.mask_ret);
    for i in code {
        let m = i.mnem.strip_suffix(".s").unwrap_or(&i.mnem);
        // the call boundary: a 512-bit mask arrives and leaves as its bits
        if m == "kmovq" && params > 0 && i.ops.starts_with('k') && i.ops.contains(",r") {
            params -= 1;
            continue;
        }
        if m.starts_with("kmov") && ret && (i.ops.starts_with("eax,k") || i.ops.starts_with("rax,k")) {
            ret = false;
            continue;
        }
        if unlisted(i) || p.checks.iter().any(|c| m.starts_with(c)) {
            continue;
        }
        let m = match plain_compare(m) {
            Some(plain) if !want.contains_key(m) => plain,
            // objdump's name of `setz`
            _ if m == "sete" => "setz".to_owned(),
            _ => m.to_owned(),
        };
        *out.entry(m).or_insert(0) += 1;
    }
    out
}

/// Rows of the v4 column backed by K mode.
const BACKED: usize = 411;

/// The v4 rows of vector ops no K-mode code backs: an unavailable cell (`div_lanes`, which has
/// no vector form at any x86 tier), and the casts on f16 lanes alone, which no backend holds in
/// vectors.
const NOT_BACKED: [&str; 3] = ["vec.div_lanes int", "vec.cast f16>f32", "vec.cast.round f32>f16"];

#[test]
fn v4_recipes_are_the_emitted_code() {
    if !super::super::features::have_objdump() {
        return;
    }
    let mut backed = Vec::new();
    let mut unbacked = Vec::new();
    let mut failures = Vec::new();
    let mut total = 0;
    for e in Tier::V4.entries() {
        let Some(cell) = e.baseline(Tier::V4) else { continue };
        total += 1;
        let Some(&op) = VecOp::ALL.iter().find(|o| o.name() == e.op) else { continue };
        let row = format!("{} {} {}", e.op, e.ty, e.form).trim_end().to_owned();
        if cell.class == Class::U {
            unbacked.push(row);
            continue;
        }
        let mut ok = true;
        // a row of f16 / bf16 lanes alone is backed by no code
        let mut checked = false;
        for (key, recipe) in e.keys.iter().zip(cell.recipes) {
            // no backend holds vectors of these lanes
            if key.split('>').any(|l| matches!(l, "f16" | "bf16")) {
                continue;
            }
            checked = true;
            let mut sp = Sp::new();
            let code = probe(&mut sp, op, key, e.form).and_then(|p| code_of(sp, &p).map(|c| (p, c)));
            let Some((p, code)) = code else {
                ok = false;
                failures.push(format!("{} {key} {}: no K-mode code", e.op, e.form));
                continue;
            };
            let want = recipe_counts(recipe);
            let got = code_counts(&code, &p, &want);
            if got != want {
                ok = false;
                let text: Vec<String> = code.iter().map(|i| format!("{} {}", i.mnem, i.ops)).collect();
                failures.push(format!(
                    "{} {key} {}: recipe `{recipe}`\n  listed: {got:?}\n  code: {}",
                    e.op,
                    e.form,
                    text.join("; ")
                ));
            }
        }
        if ok && checked {
            backed.push(row);
        } else if ok {
            unbacked.push(row);
        }
    }
    assert!(failures.is_empty(), "{} rows differ:\n{}", failures.len(), failures.join("\n"));
    if std::env::var_os("STAVE_TEST_VERBOSE").is_some() {
        eprintln!("v4 rows backed by K mode: {} of {total}", backed.len());
    }
    assert_eq!(backed.len(), BACKED, "v4 rows backed by K mode (of {total})");
    assert_eq!(unbacked, NOT_BACKED, "v4 rows no K-mode code backs");
}
