//! Per-function feature sets: every function is compiled for the module baseline plus its own
//! `needs` (`Func::features`).
//!
//! [`check_image`] disassembles every function of an image with GNU objdump and classifies each
//! instruction by its mnemonic and operands, independently of the encoder's own record: an
//! instruction outside the function's feature set fails the test (without objdump the check is
//! skipped with a message). The programs here mix baseline functions with functions of several
//! feature sets in one v2-baseline module, compare the output with scalar references and the
//! interpreter, check where `vzeroupper` sits, and run the baseline path on a CPU without AVX
//! (`qemu-x86_64 -cpu Nehalem`, skipped with a message when qemu is not installed).

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

use stave_ir::{Func, FuncBuilder, FuncId, Module, Op, Operand, TypeId, Value, VecOp};
use stave_obj::ObjectImage;
use stave_target::{Feature, FeatureSet};

use super::harness::*;
use super::simd::{Lane, Shape, Sp, sh};
use crate::debug::{Error, Options, compile_module};

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok_and(|o| o.status.success())
}

/// GNU objdump is installed (the checks parse its Intel-syntax output).
pub fn have_objdump() -> bool {
    static HAVE: OnceLock<bool> = OnceLock::new();
    *HAVE.get_or_init(|| {
        let h = Command::new("objdump")
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success() && String::from_utf8_lossy(&o.stdout).contains("GNU"));
        if !h {
            eprintln!("SKIP instruction feature checks: GNU objdump not installed");
        }
        h
    })
}

/// One disassembled instruction.
#[derive(Clone, Debug)]
pub struct Inst {
    pub off: usize,
    pub mnem: String,
    pub ops: String,
    /// The address objdump computed for a `rip`-relative operand.
    pub target: Option<usize>,
    /// The symbol the linker resolves inside this instruction (a call target, a data address).
    pub sym: Option<String>,
    /// EVEX-encoded: its first byte is `62`, which is no other instruction in 64-bit code.
    pub evex: bool,
}

impl Inst {
    pub fn text(&self) -> String {
        format!("{:#x}: {} {}", self.off, self.mnem, self.ops)
    }

    /// Writes or reads a register above its low 128 bits.
    pub fn ymm(&self) -> bool {
        self.ops.contains("ymm") || self.ops.contains("zmm")
    }
}

const PREFIXES: [&str; 9] = ["lock", "rep", "repz", "repnz", "notrack", "bnd", "data16", "addr32", "cs"];

/// Linear disassembly of raw x86-64 bytes.
pub fn disasm(code: &[u8]) -> Vec<Inst> {
    let p = tmp("code.bin");
    std::fs::write(&p, code).unwrap();
    let out = Command::new("objdump")
        .args(["-D", "-b", "binary", "-m", "i386:x86-64", "-M", "intel", "--no-show-raw-insn", "-w"])
        .arg(&p)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout);
    let mut v = Vec::new();
    for l in text.lines() {
        let Some((a, rest)) = l.split_once(":\t") else { continue };
        let Ok(off) = usize::from_str_radix(a.trim(), 16) else { continue };
        let (body, comment) = match rest.split_once('#') {
            Some((b, c)) => (b, Some(c)),
            None => (rest, None),
        };
        let mut toks = body.split_whitespace();
        let mut mnem = toks.next().unwrap_or("");
        // `{evex}` marks an EVEX instruction that also has a VEX encoding
        while PREFIXES.contains(&mnem) || mnem == "{evex}" {
            mnem = toks.next().unwrap_or("");
        }
        let ops = toks.collect::<Vec<_>>().join(" ");
        let target = comment
            .and_then(|c| c.split_whitespace().next())
            .and_then(|h| h.strip_prefix("0x"))
            .and_then(|h| usize::from_str_radix(h, 16).ok());
        let evex = code.get(off) == Some(&0x62);
        v.push(Inst { off, mnem: mnem.to_owned(), ops, target, sym: None, evex });
    }
    v
}

/// What an instruction needs beyond SSE4.2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Need {
    Baseline,
    Feat(Feature),
    /// An EVEX or mask-register instruction: only in a function compiled for the whole AVX-512
    /// set (K mode), and these features of it.
    K(FeatureSet),
    /// Outside every feature set the backend compiles for.
    Never,
}

/// The mask-register instructions (VEX-encoded).
const KOPS: [&str; 10] =
    ["kortest", "kshiftl", "kshiftr", "kandn", "kxnor", "kmov", "kand", "kxor", "knot", "kor"];

/// The AVX-512 features of an EVEX instruction, from its text: the instructions on byte and
/// word lanes and the byte shifts are `avx512bw` (not the down-conversions from doublewords and
/// quadwords), the float logic, the quadword multiply, the conversions between floats and
/// quadwords and the doubleword and quadword mask conversions `avx512dq`, the leading-zero counts
/// `avx512cd`, the rest `avx512f`; without a `zmm` operand it is an `avx512vl` form.
fn evex_need(i: &Inst) -> FeatureSet {
    let m = i.mnem.strip_suffix(".s").unwrap_or(&i.mnem);
    let dq = matches!(
        m,
        "vpmullq"
            | "vpmovm2d"
            | "vpmovm2q"
            | "vpmovd2m"
            | "vpmovq2m"
            | "vandps"
            | "vandpd"
            | "vandnps"
            | "vandnpd"
            | "vorps"
            | "vorpd"
            | "vxorps"
            | "vxorpd"
    ) || (m.starts_with("vcvt") && m.contains("qq"));
    let cd = matches!(m, "vplzcntd" | "vplzcntq");
    let down = ["vpmov", "vpmovs", "vpmovus"]
        .iter()
        .any(|p| m.strip_prefix(p).is_some_and(|r| matches!(r, "db" | "dw" | "qb" | "qw" | "qd")));
    let bw = matches!(
        m,
        "vpmovb2m"
            | "vpmovw2m"
            | "vmovdqu8"
            | "vmovdqu16"
            | "vpmaddwd"
            | "vpunpcklwd"
            | "vpslldq"
            | "vpsrldq"
    ) || (m.starts_with("vp") && (m.ends_with('b') || m.ends_with('w')) && !down);
    let mut s = FeatureSet::of(if dq {
        Feature::Avx512dq
    } else if cd {
        Feature::Avx512cd
    } else if bw {
        Feature::Avx512bw
    } else {
        Feature::Avx512f
    });
    if !i.ops.contains("zmm") {
        s.insert(Feature::Avx512vl);
    }
    s
}

/// The feature of an instruction, from its text (and, for the encoding, its first byte).
fn need(i: &Inst) -> Need {
    let m = i.mnem.as_str();
    if m == "(bad)" {
        return Need::Never;
    }
    if let Some(op) = KOPS.iter().find(|p| m.starts_with(**p)) {
        return match &m[op.len()..] {
            "w" => Need::K(FeatureSet::of(Feature::Avx512f)),
            "b" => Need::K(FeatureSet::of(Feature::Avx512dq)),
            "d" | "q" => Need::K(FeatureSet::of(Feature::Avx512bw)),
            _ => Need::Never,
        };
    }
    if i.evex {
        return Need::K(evex_need(i));
    }
    if i.ops.contains("zmm") || i.ops.contains("{k") {
        return Need::Never;
    }
    if matches!(m, "pext" | "pdep" | "bzhi" | "mulx" | "rorx" | "sarx" | "shlx" | "shrx") {
        return Need::Feat(Feature::Bmi2);
    }
    if matches!(m, "andn" | "blsi" | "blsr" | "blsmsk" | "tzcnt" | "lzcnt" | "movbe") {
        return Need::Never;
    }
    if !m.starts_with('v') {
        return Need::Baseline;
    }
    if ["vfmadd", "vfmsub", "vfnmadd", "vfnmsub"].iter().any(|p| m.starts_with(p)) {
        return Need::Feat(Feature::Fma);
    }
    if matches!(m, "vcvtph2ps" | "vcvtps2ph") {
        return Need::Feat(Feature::F16c);
    }
    let avx2_only = ["vpbroadcast", "vpsllv", "vpsrlv", "vpsrav", "vpgather", "vgather", "vpmaskmov"]
        .iter()
        .any(|p| m.starts_with(p))
        || matches!(
            m,
            "vbroadcasti128"
                | "vpermd"
                | "vpermq"
                | "vpermps"
                | "vpermpd"
                | "vperm2i128"
                | "vinserti128"
                | "vextracti128"
                | "vpblendd"
        )
        // AVX broadcasts from memory only
        || (matches!(m, "vbroadcastss" | "vbroadcastsd") && !i.ops.contains("PTR"));
    // the 256-bit forms of the integer instructions
    let int256 = i.ymm()
        && ((m.starts_with("vp") && !matches!(m, "vptest" | "vpermilps" | "vpermilpd" | "vperm2f128"))
            || matches!(m, "vmovntdqa" | "vmpsadbw"));
    Need::Feat(if avx2_only || int256 { Feature::Avx2 } else { Feature::Avx })
}

/// The code of one function inside the buffer handed to objdump.
struct Span {
    name: String,
    features: FeatureSet,
    start: usize,
    end: usize,
    /// Offsets (in the buffer) of the fields the linker fills in, and their symbols.
    relocs: Vec<(usize, Option<String>)>,
}

/// The compiler's own routines: they are baseline code in every module.
const HELPERS: [&str; 7] = [
    "_start",
    "__stave_main",
    "__stave_signal",
    "__stave_thread_start",
    "__stave_panic_fallback",
    "__stave_fma32",
    "__stave_fma64",
];

/// The instructions of every function of `img` (compiled from `m` with the module baseline
/// `baseline`), by function path, up to the function's constant pool: a function reads its
/// pool `rip`-relative, and nothing else of its own code that way.
pub fn functions(
    m: &Module,
    img: &ObjectImage,
    baseline: FeatureSet,
) -> Vec<(String, FeatureSet, Vec<Inst>)> {
    let order = stave_ir::link::text_order(m);
    let names = stave_ir::link::func_symbols(m, &order);
    let mut named: Vec<(String, String, FeatureSet)> = Vec::new();
    for f in &order {
        if let Some(n) = &names[f.index()] {
            let func = &m.funcs[f.index()];
            named.push((n.clone(), func.path.clone(), baseline.union(func.features)));
        }
    }
    named.extend(HELPERS.iter().map(|h| ((*h).to_owned(), (*h).to_owned(), baseline)));
    let mut buf = Vec::new();
    let mut spans = Vec::new();
    for (sym, path, features) in named {
        let Some(code) = img.lookup(&sym).and_then(|s| img.func_code(s)) else { continue };
        let start = buf.len();
        buf.extend_from_slice(&code.code);
        let mut relocs: Vec<(usize, Option<String>)> = code
            .relocs
            .iter()
            .map(|r| (start + r.offset as usize, img.symbol_name(r.sym).map(str::to_owned)))
            .collect();
        relocs.sort_by_key(|r| r.0);
        spans.push(Span { name: path, features, start, end: buf.len(), relocs });
        // longer than any instruction: the disassembly is in step again at the next function
        // whatever the pool bytes before it decode to
        buf.extend_from_slice(&[0x90; 32]);
    }
    let insts = disasm(&buf);
    let mut out = Vec::new();
    let mut k = 0;
    for s in spans {
        while k < insts.len() && insts[k].off < s.start {
            k += 1;
        }
        let first = k;
        while k < insts.len() && insts[k].off < s.end {
            k += 1;
        }
        let mut list = insts[first..k].to_vec();
        let mut pool = s.end;
        let mut relocs = s.relocs.iter().peekable();
        for j in 0..list.len() {
            let next = list.get(j + 1).map_or(s.end, |n| n.off);
            let i = &mut list[j];
            while relocs.next_if(|r| r.0 < i.off).is_some() {}
            match relocs.peek().filter(|r| r.0 < next) {
                Some(r) => i.sym = r.1.clone(),
                None => {
                    if let Some(t) = i.target
                        && i.ops.contains("[rip")
                        && (next..s.end).contains(&t)
                    {
                        pool = pool.min(t);
                    }
                }
            }
        }
        let code: Vec<Inst> =
            list.into_iter().filter(|i| i.off < pool).map(|i| Inst { off: i.off - s.start, ..i }).collect();
        out.push((s.name, s.features, code));
    }
    out
}

/// The features of K mode.
const V4: [Feature; 5] =
    [Feature::Avx512f, Feature::Avx512bw, Feature::Avx512vl, Feature::Avx512dq, Feature::Avx512cd];

/// Fails when a function of `img` has an instruction outside its feature set (the module
/// baseline `baseline` plus the function's own `needs`), or, in a function that adds `avx` to
/// the baseline, a vector or scalar float instruction that is not VEX-encoded. An EVEX or
/// mask-register instruction is outside the set of every function but one in K mode.
pub fn check_image(m: &Module, img: &ObjectImage, baseline: FeatureSet, what: &str) {
    if !have_objdump() {
        return;
    }
    for (name, features, code) in functions(m, img, baseline) {
        let vex_only = features.contains(Feature::Avx) && !baseline.contains(Feature::Avx);
        for i in &code {
            assert!(
                !vex_only || i.mnem.starts_with('v') || !i.ops.contains("xmm"),
                "{what}: fn `{name}` with avx has the legacy SSE `{}`",
                i.text()
            );
            let ok = match need(i) {
                Need::Baseline => true,
                Need::Feat(f) => features.contains(f),
                Need::K(s) => V4.iter().all(|f| features.contains(*f)) && s.is_subset(features),
                Need::Never => false,
            };
            if !ok {
                let names: Vec<&str> = features.iter().map(|f| f.name()).collect();
                panic!(
                    "{what}: fn `{name}` compiled for [{}] has `{}` ({:?})",
                    names.join(" "),
                    i.text(),
                    need(i)
                );
            }
        }
    }
}

/// The classifier on known instructions.
#[test]
fn classifier() {
    let inst = |mnem: &str, ops: &str, evex: bool| Inst {
        off: 0,
        mnem: mnem.into(),
        ops: ops.into(),
        target: None,
        sym: None,
        evex,
    };
    let i = |mnem: &str, ops: &str| inst(mnem, ops, false);
    use Feature::*;
    let cases = [
        ("addps", "xmm0,xmm1", Need::Baseline),
        ("popcnt", "rax,rcx", Need::Baseline),
        ("pshufb", "xmm0,xmm1", Need::Baseline),
        ("vaddps", "ymm0,ymm0,ymm1", Need::Feat(Avx)),
        ("vaddss", "xmm0,xmm0,xmm1", Need::Feat(Avx)),
        ("vpaddd", "xmm0,xmm0,xmm1", Need::Feat(Avx)),
        ("vpaddd", "ymm0,ymm0,ymm1", Need::Feat(Avx2)),
        ("vpxor", "ymm0,ymm0,ymm1", Need::Feat(Avx2)),
        ("vxorps", "ymm0,ymm0,ymm1", Need::Feat(Avx)),
        ("vmovdqu", "ymm0,YMMWORD PTR [rbp-0x40]", Need::Feat(Avx)),
        ("vptest", "ymm0,ymm1", Need::Feat(Avx)),
        ("vpmovmskb", "eax,ymm1", Need::Feat(Avx2)),
        ("vmovmskps", "eax,ymm1", Need::Feat(Avx)),
        ("vextractf128", "xmm0,ymm1,0x1", Need::Feat(Avx)),
        ("vextracti128", "xmm0,ymm1,0x1", Need::Feat(Avx2)),
        ("vpbroadcastd", "xmm0,xmm0", Need::Feat(Avx2)),
        ("vbroadcastss", "ymm0,xmm0", Need::Feat(Avx2)),
        ("vbroadcastss", "ymm0,DWORD PTR [rax]", Need::Feat(Avx)),
        ("vpsllvd", "xmm0,xmm0,xmm1", Need::Feat(Avx2)),
        ("vpmaskmovd", "xmm0,xmm1,XMMWORD PTR [rax]", Need::Feat(Avx2)),
        ("vmaskmovps", "ymm0,ymm1,YMMWORD PTR [rax]", Need::Feat(Avx)),
        ("vpblendvb", "ymm0,ymm0,ymm1,ymm2", Need::Feat(Avx2)),
        ("vblendvps", "ymm0,ymm0,ymm1,ymm2", Need::Feat(Avx)),
        ("vpgatherdd", "ymm0,DWORD PTR [rax+ymm1*4],ymm2", Need::Feat(Avx2)),
        ("vfmadd213ps", "ymm0,ymm1,ymm2", Need::Feat(Fma)),
        ("vfmadd213ss", "xmm0,xmm1,xmm2", Need::Feat(Fma)),
        ("vcvtph2ps", "ymm0,xmm1", Need::Feat(F16c)),
        ("pext", "eax,eax,ecx", Need::Feat(Bmi2)),
        ("vzeroupper", "", Need::Feat(Avx)),
        ("vpaddd", "zmm0,zmm0,zmm1", Need::Never),
        ("tzcnt", "eax,ecx", Need::Never),
    ];
    for (m, o, want) in cases {
        assert_eq!(need(&i(m, o)), want, "{m} {o}");
    }
    let k = |fs: &[Feature]| Need::K(FeatureSet::from_features(fs.iter().copied()));
    let evex = [
        ("vpaddd", "zmm0,zmm0,zmm1", k(&[Avx512f])),
        ("vpaddd", "ymm0,ymm0,ymm1", k(&[Avx512f, Avx512vl])),
        ("vpaddb", "zmm0,zmm0,zmm1", k(&[Avx512bw])),
        ("vpcmpltub", "k1,xmm0,xmm1", k(&[Avx512bw, Avx512vl])),
        ("vpmovm2d", "zmm0,k1", k(&[Avx512dq])),
        ("vpmovb2m", "k1,ymm0", k(&[Avx512bw, Avx512vl])),
        ("vxorps", "zmm0,zmm0,zmm1", k(&[Avx512dq])),
        ("vmovdqu64", "ZMMWORD PTR [rax],zmm1", k(&[Avx512f])),
        ("vmovdqu8", "zmm1{k1}{z},ZMMWORD PTR [rax]", k(&[Avx512bw])),
        ("vpmullq", "xmm0,xmm0,xmm1", k(&[Avx512dq, Avx512vl])),
        ("vextracti64x4", "ymm0,zmm1,0x1", k(&[Avx512f])),
    ];
    for (m, o, want) in evex {
        assert_eq!(need(&inst(m, o, true)), want, "{m} {o}");
    }
    let masks = [
        ("kmovw", "k1,eax", k(&[Avx512f])),
        ("kmovb", "eax,k1", k(&[Avx512dq])),
        ("kmovq", "k1,QWORD PTR [rbp-0x8]", k(&[Avx512bw])),
        ("kandnq", "k1,k2,k3", k(&[Avx512bw])),
        ("kortestb", "k1,k1", k(&[Avx512dq])),
        ("kshiftrd", "k1,k2,0x3", k(&[Avx512bw])),
    ];
    for (m, o, want) in masks {
        assert_eq!(need(&i(m, o)), want, "{m} {o}");
    }
}

/// The feature the assembler records for an instruction is the one its disassembly shows: every
/// entry of the VEX opcode tables, at both widths, with register and memory operands.
#[test]
fn recorded_features_match_the_disassembly() {
    use crate::enc::*;
    if !have_objdump() {
        return;
    }
    let (x1, x2, x3) = (Xmm::masked(1), Xmm::masked(2), Xmm::masked(3));
    let (y1, y2, y3) = (Ymm::masked(1), Ymm::masked(2), Ymm::masked(3));
    let (g, m) = (Gpr::RAX, Mem::base(Gpr::RAX, 16));
    let mut buf = Vec::new();
    let mut want: Vec<(usize, FeatureSet, String)> = Vec::new();
    // an operand shape the instruction does not have is an encoder error: skipped
    let mut add = |name: String, f: &dyn Fn(&mut Asm) -> Result<(), EncError>| {
        let mut a = Asm::new();
        if f(&mut a).is_ok() {
            want.push((buf.len(), a.features(), name));
            buf.extend_from_slice(a.bytes());
        }
    };
    for &op in VBin::ALL {
        add(format!("{op:?} x,x,x"), &|a| a.vex_bin(op, x1, x2, x3));
        add(format!("{op:?} x,x,m"), &|a| a.vex_bin(op, x1, x2, m));
        add(format!("{op:?} y,y,y"), &|a| a.vex_bin(op, y1, y2, y3));
        add(format!("{op:?} y,y,x"), &|a| a.vex_bin(op, y1, y2, x3));
        add(format!("{op:?} y,y,m"), &|a| a.vex_bin(op, y1, y2, m));
    }
    for &op in VUn::ALL {
        add(format!("{op:?} x,x"), &|a| a.vex_un(op, x1, x2));
        add(format!("{op:?} x,m"), &|a| a.vex_un(op, x1, m));
        add(format!("{op:?} y,y"), &|a| a.vex_un(op, y1, y2));
        add(format!("{op:?} y,x"), &|a| a.vex_un(op, y1, x2));
        add(format!("{op:?} y,m"), &|a| a.vex_un(op, y1, m));
    }
    for &op in VBinI::ALL {
        add(format!("{op:?} x,x,x"), &|a| a.vex_bin_i(op, x1, x2, x3, 1));
        add(format!("{op:?} x,x,m"), &|a| a.vex_bin_i(op, x1, x2, m, 1));
        add(format!("{op:?} y,y,y"), &|a| a.vex_bin_i(op, y1, y2, y3, 1));
        add(format!("{op:?} y,y,x"), &|a| a.vex_bin_i(op, y1, y2, x3, 1));
        add(format!("{op:?} y,y,m"), &|a| a.vex_bin_i(op, y1, y2, m, 1));
    }
    for &op in VUnI::ALL {
        add(format!("{op:?} x,x"), &|a| a.vex_un_i(op, x1, x2, 1));
        add(format!("{op:?} x,m"), &|a| a.vex_un_i(op, x1, m, 1));
        add(format!("{op:?} y,y"), &|a| a.vex_un_i(op, y1, y2, 1));
        add(format!("{op:?} y,m"), &|a| a.vex_un_i(op, y1, m, 1));
    }
    for &op in VStore::ALL {
        add(format!("{op:?} m,x"), &|a| a.vex_store(op, m, x1));
        add(format!("{op:?} m,y"), &|a| a.vex_store(op, m, y1));
    }
    for &op in VCvtN::ALL {
        add(format!("{op:?} x,x"), &|a| a.vex_cvt_n(op, x1, x2, VLen::X128));
        add(format!("{op:?} x,y"), &|a| a.vex_cvt_n(op, x1, y2, VLen::X256));
    }
    for &op in VMovMsk::ALL {
        add(format!("{op:?} g,x"), &|a| {
            a.vex_movmsk(op, g, x1);
            Ok(())
        });
        add(format!("{op:?} g,y"), &|a| {
            a.vex_movmsk(op, g, y1);
            Ok(())
        });
    }
    for &op in VBlendV::ALL {
        add(format!("{op:?} x"), &|a| a.vex_blendv(op, x1, x1, x2, x3));
        add(format!("{op:?} y"), &|a| a.vex_blendv(op, y1, y1, y2, y3));
    }
    for &op in VShiftI::ALL {
        add(format!("{op:?} x"), &|a| a.vex_shift_i(op, x1, x2, 3));
        add(format!("{op:?} y"), &|a| a.vex_shift_i(op, y1, y2, 3));
    }
    for &op in VExtract::ALL {
        add(format!("{op:?} x,x"), &|a| a.vex_extract(op, x1, x2, 1));
        add(format!("{op:?} x,y"), &|a| a.vex_extract(op, x1, y2, 1));
        add(format!("{op:?} m,y"), &|a| a.vex_extract(op, m, y2, 1));
    }
    for &op in VMaskStore::ALL {
        add(format!("{op:?} m,x,x"), &|a| a.vex_maskstore(op, m, x1, x2));
        add(format!("{op:?} m,y,y"), &|a| a.vex_maskstore(op, m, y1, y2));
    }
    for &op in VGather::ALL {
        let regs: [(VReg, VReg, VReg); 4] = [
            (x1.into(), x2.into(), x3.into()),
            (y1.into(), y2.into(), y3.into()),
            (y1.into(), x2.into(), y3.into()),
            (x1.into(), y2.into(), x3.into()),
        ];
        for (d, i, k) in regs {
            add(format!("{op:?} {d:?} {i:?}"), &|a| a.vex_gather(op, d, Some(g), i, Scale::S4, 0, k));
        }
    }
    for &op in PExtr::ALL {
        add(format!("{op:?}"), &|a| a.vex_pextr(op, g, x1, 1));
    }
    for &op in PInsr::ALL {
        add(format!("{op:?}"), &|a| a.vex_pinsr(op, x1, x2, g, 1));
    }
    for s in [Size::D, Size::Q] {
        add(format!("movd x,g {s:?}"), &|a| a.vex_movd_x(s, x1, g));
        add(format!("movd g,x {s:?}"), &|a| a.vex_movd_r(s, g, x1));
        add(format!("pext {s:?}"), &|a| a.pext(s, g, g, g));
        add(format!("pdep {s:?}"), &|a| a.pdep(s, g, g, g));
        add(format!("bzhi {s:?}"), &|a| a.bzhi(s, g, g, g));
    }
    add("vzeroupper".into(), &|a| {
        a.vzeroupper();
        Ok(())
    });
    // scalar float code: legacy, and VEX when the function asks for it
    for vex in [false, true] {
        add(format!("addss vex={vex}"), &|a| {
            a.set_vex_scalars(vex);
            a.sse(SseOp::Add, Prec::F32, x1, x2)
        });
        add(format!("movsd load vex={vex}"), &|a| {
            a.set_vex_scalars(vex);
            a.movs(Prec::F64, x1, m)
        });
        add(format!("cvtsi2sd vex={vex}"), &|a| {
            a.set_vex_scalars(vex);
            a.cvtsi2s(Prec::F64, x1, Size::Q, g)
        });
    }
    // legacy SSE forms record nothing
    add("paddd".into(), &|a| a.sse_bin(VBin::Paddd, x1, x2));
    add("pshufb".into(), &|a| a.sse_bin(VBin::Pshufb, x1, x2));
    add("popcnt".into(), &|a| a.popcnt(Size::Q, g, g));
    // EVEX: every entry at every length it has
    let kr = KReg::masked(1);
    for &op in EOp::ALL {
        let n = format!("{op:?}");
        let has = |ps: &[&str]| ps.iter().any(|p| n.starts_with(p));
        let imm = has(&["Rndscale", "Pternlog", "Cmpp", "Extract", "Insert", "Pshufd", "Shufi", "Align"])
            || (has(&["Pcmp"]) && !has(&["Pcmpeq", "Pcmpgt"]));
        for vl in [EVL::L128, EVL::L256, EVL::L512] {
            let halves = n.ends_with("64x4");
            let quarters = n.ends_with("32x4");
            if (vl == EVL::L128
                && (halves || quarters || matches!(op, EOp::Broadcastsd | EOp::Permq | EOp::Shufi64x2)))
                || (vl == EVL::L256 && halves)
            {
                continue;
            }
            // a non-temporal store has only the memory form
            let rm = if has(&["Movnt"]) { ERm::M(m) } else { ERm::R(3) };
            let e = Ev::new(op, vl, 1, 0, rm);
            let e = if imm { e.imm(1) } else { e };
            add(format!("{op:?} {vl:?}"), &|a| a.evex(e));
        }
    }
    for &op in EShiftI::ALL {
        for vl in [EVL::L128, EVL::L256, EVL::L512] {
            add(format!("{op:?} {vl:?}"), &|a| a.evex_shift_i(op, vl, 1, ERm::R(2), 3, kr, false));
        }
    }
    for s in KSize::ALL {
        type Emit<'a> = (&'a str, &'a dyn Fn(&mut Asm));
        let fns: [Emit; 7] = [
            ("kmov k,k", &|a| a.kmov_kk(s, kr, kr)),
            ("kmov k,g", &|a| a.kmov_kg(s, kr, g)),
            ("kmov g,k", &|a| a.kmov_gk(s, g, kr)),
            ("knot", &|a| a.knot(s, kr, kr)),
            ("kortest", &|a| a.kortest(s, kr, kr)),
            ("kshiftl", &|a| a.kshift(true, s, kr, kr, 1)),
            ("kshiftr", &|a| a.kshift(false, s, kr, kr, 1)),
        ];
        for (name, f) in fns {
            add(format!("{name} {s:?}"), &|a| {
                f(a);
                Ok(())
            });
        }
        for op in KBin::ALL {
            add(format!("{op:?} {s:?}"), &|a| {
                a.kbin(op, s, kr, kr, kr);
                Ok(())
            });
        }
        add(format!("kmov k,m {s:?}"), &|a| a.kmov_km(s, kr, m));
        add(format!("kmov m,k {s:?}"), &|a| a.kmov_mk(s, m, kr));
    }
    assert!(want.len() > 1100, "{} encodings", want.len());
    let insts = disasm(&buf);
    for (off, recorded, name) in want {
        let i = insts.iter().find(|i| i.off == off).unwrap_or_else(|| panic!("{name}: not disassembled"));
        let shown = match need(i) {
            Need::Baseline => FeatureSet::EMPTY,
            Need::Feat(f) => FeatureSet::of(f),
            Need::K(s) => s,
            Need::Never => panic!("{name}: `{}`", i.text()),
        };
        assert_eq!(recorded, shown, "{name}: `{}`", i.text());
    }
}

/// Symbol name → function, for the call targets of a disassembly.
fn by_symbol(m: &Module) -> HashMap<String, FuncId> {
    let order = stave_ir::link::text_order(m);
    let names = stave_ir::link::func_symbols(m, &order);
    order.iter().filter_map(|f| names[f.index()].clone().map(|n| (n, *f))).collect()
}

/// A 256- or 512-bit vector or a 256-bit mask type: a value of it travels in a `ymm` or `zmm`
/// register (a mask of 512 bits of lanes travels as its bits in a general register).
pub fn is_ymm(m: &Module, t: TypeId) -> bool {
    match stave_ir::simd::vec_shape(&m.types, t) {
        Some((_, lane, n)) => lane.bits() / 8 * n >= 32,
        None => stave_ir::simd::mask_shape(&m.types, t).is_some_and(|(w, n)| w / 8 * n == 32),
    }
}

/// The `ymm` upper halves are clean at every call and return of the straight-line function `f`
/// that carries no 256-bit value in a register: no instruction on a `ymm` register sits between
/// the last `vzeroupper` (or the last call that returned none) and it. `ind`: whether the
/// function's indirect calls carry a 256-bit argument and a 256-bit result. Returns the number
/// of `vzeroupper`s.
pub fn check_upper_halves(m: &Module, f: &Func, code: &[Inst], ind: (bool, bool), what: &str) -> usize {
    let syms = by_symbol(m);
    let carries = |g: &Func| (g.params.iter().any(|&t| is_ymm(m, t)), g.ret.is_some_and(|t| is_ymm(m, t)));
    let (mut dirty, ret_ymm) = carries(f);
    let mut cleaned = 0;
    for i in code {
        match i.mnem.as_str() {
            "vzeroupper" => {
                dirty = false;
                cleaned += 1;
            }
            "call" => {
                let (arg, res) = match &i.sym {
                    Some(s) => syms.get(s).map_or((false, false), |g| carries(&m.funcs[g.index()])),
                    None => ind,
                };
                assert!(!dirty || arg, "{what}: fn `{}`: dirty upper halves at `{}`", f.path, i.text());
                dirty = res;
            }
            "ret" => {
                assert!(!dirty || ret_ymm, "{what}: fn `{}`: dirty upper halves at `{}`", f.path, i.text());
                dirty = false;
            }
            _ if i.ymm() => dirty = true,
            _ => {}
        }
    }
    cleaned
}

/// One function being built, and the scalar reference of what it prints.
struct Kern<'a> {
    sp: &'a mut Sp,
    b: FuncBuilder,
    want: &'a mut String,
}

impl Kern<'_> {
    fn vec(&mut self, s: Shape, lanes: &[u64]) -> Value {
        self.sp.vec(&mut self.b, s, lanes)
    }

    fn op(&mut self, op: VecOp, s: Shape, args: &[Value]) -> Value {
        let vt = self.sp.vt(s);
        let a: Vec<Operand> = args.iter().map(|&v| v.into()).collect();
        self.b.op(Op::Vec(op), vt, &a)
    }

    /// The mask of `x < y`.
    fn lt(&mut self, s: Shape, x: Value, y: Value) -> Value {
        let mt = self.sp.mt(s.lane.bits(), s.n);
        self.b.op(Op::Vec(VecOp::Lt), mt, &[x.into(), y.into()])
    }

    /// Prints the vector `v`; `lanes`: what the scalar reference gives.
    fn out(&mut self, v: Value, s: Shape, lanes: &[u64]) {
        self.sp.out(&mut self.b, v, s);
        for w in Sp::words(s, lanes) {
            let _ = writeln!(self.want, "{w:016x}");
        }
    }

    /// Prints the scalar `v` of type `ty`; `bits`: the reference.
    fn out_scalar(&mut self, v: Value, ty: TypeId, bits: u64) {
        self.sp.p.print(&mut self.b, v.into(), ty);
        let _ = writeln!(self.want, "{bits:016x}");
    }

    fn opaque(&mut self, c: Operand, ty: TypeId) -> Value {
        self.sp.p.opaque(&mut self.b, c, ty)
    }

    fn finish(mut self, f: FuncId) {
        self.b.ret(None);
        self.sp.p.def(f, self.b);
    }
}

fn f32s(x: [f32; 8]) -> Vec<u64> {
    x.iter().map(|v| u64::from(v.to_bits())).collect()
}

fn f64s(x: [f64; 4]) -> Vec<u64> {
    x.iter().map(|v| v.to_bits()).collect()
}

fn i32s(x: [i32; 8]) -> Vec<u64> {
    x.iter().map(|v| u64::from(*v as u32)).collect()
}

fn zip<T: Copy, const N: usize>(a: [T; N], b: [T; N], f: impl Fn(T, T) -> T) -> [T; N] {
    std::array::from_fn(|i| f(a[i], b[i]))
}

const A: [f32; 8] = [1.5, -2.25, 3.0, 0.1, 1.0e10, -7.75, 0.3333, 100.0];
const B: [f32; 8] = [0.5, 4.0, -1.0, 0.2, 3.0, 2.5, 3.0, 1.0e-3];
const C: [f32; 8] = [1.0, 1.0, 0.25, -0.02, 1.0e3, 0.0, -1.0, 7.0];
const DA: [f64; 4] = [0.1, -3.5, 1.0e200, 2.0];
const DB: [f64; 4] = [0.2, 0.3, 1.0e-200, -8.0];
const DC: [f64; 4] = [-0.02, 1.05, -1.0, 16.5];
const X: [i32; 8] = [1, -2, i32::MAX, i32::MIN, 12345, -99999, 65536, 7];
const Y: [i32; 8] = [3, 5, 2, -1, 100_000, 77777, 65536, -7];
const S: [i32; 8] = [0, 1, 5, 31, 13, 7, 16, 3];

fn fma32() -> [f32; 8] {
    std::array::from_fn(|i| A[i].mul_add(B[i], C[i]))
}

fn fma64() -> [f64; 4] {
    std::array::from_fn(|i| DA[i].mul_add(DB[i], DC[i]))
}

/// A v2-baseline program of baseline functions and kernels with `needs`. Its entry runs the
/// baseline path, then, unless the command line has an argument, every kernel.
struct Mixed {
    sp: Sp,
    /// The output of the baseline path and of the kernels, from the scalar references.
    want_base: String,
    want_kernels: String,
    /// The straight-line functions with 256-bit code, and what their indirect calls carry.
    kernels: Vec<(FuncId, (bool, bool))>,
    fma: FuncId,
    soft_fma: [FuncId; 2],
    v3: FuncId,
}

fn mixed() -> Mixed {
    let mut sp = Sp::new();
    let t = sp.t();
    let (f8, d4, i8x, u8x, u16x, q4) = (
        sh(Lane::F32, 8),
        sh(Lane::F64, 4),
        sh(Lane::I32, 8),
        sh(Lane::U8, 32),
        sh(Lane::U16, 16),
        sh(Lane::I64, 4),
    );
    let f8t = sp.vt(f8);
    let (mut want_base, mut want_kernels) = (String::new(), String::new());

    // baseline: integer and scalar float code
    let base = sp.p.decl("t.base", &[t.u64], Some(t.u64));
    {
        let mut b = FuncBuilder::new(&[t.u64]);
        let x = b.param(0).unwrap();
        let three = b.int(3);
        let y = b.binary(Op::MulWrap, t.u64, x, three);
        let one = b.int(1);
        let r = b.binary(Op::AddWrap, t.u64, y, one);
        b.ret(Some(r.into()));
        sp.p.def(base, b);
    }
    let fbase = sp.p.decl("t.fbase", &[t.f32, t.f32], Some(t.f32));
    {
        let mut b = FuncBuilder::new(&[t.f32, t.f32]);
        let (x, y) = (b.param(0).unwrap(), b.param(1).unwrap());
        let m = b.binary(Op::Mul, t.f32, x, y);
        let r = b.binary(Op::Add, t.f32, m, x);
        b.ret(Some(r.into()));
        sp.p.def(fbase, b);
    }
    let fbase_bits = u64::from((1.5f32 * 2.25 + 1.5).to_bits());
    let base_ft = sp.p.m.types.fn_type(vec![t.u64], t.u64, FeatureSet::EMPTY);
    let base_path = sp.p.decl("t.base_path", &[], None);
    {
        let mut k = Kern { sp: &mut sp, b: FuncBuilder::new(&[]), want: &mut want_base };
        let (cx, cy) = (k.b.f32(1.5), k.b.f32(2.25));
        let (x, y) = (k.opaque(cx, t.f32), k.opaque(cy, t.f32));
        let r = k.b.call(fbase, Some(t.f32), &[x.into(), y.into()]).unwrap();
        k.out_scalar(r, t.f32, fbase_bits);
        let c = k.b.int(7);
        let seven = k.opaque(c, t.u64);
        let r = k.b.call(base, Some(t.u64), &[seven.into()]).unwrap();
        k.out_scalar(r, t.u64, 22);
        k.finish(base_path);
    }

    // avx: a 256-bit argument and result
    let sq = sp.p.decl("t.sq", &[f8t], Some(f8t));
    {
        let mut b = FuncBuilder::new(&[f8t]);
        let v = b.param(0).unwrap();
        let r = b.op(Op::Vec(VecOp::Mul), f8t, &[v.into(), v.into()]);
        b.ret(Some(r.into()));
        sp.p.def(sq, b);
    }
    let sq_ft = sp.p.m.types.fn_type(vec![f8t], f8t, FeatureSet::of(Feature::Avx));

    let k_avx = sp.p.decl("t.k_avx", &[], None);
    {
        let mut k = Kern { sp: &mut sp, b: FuncBuilder::new(&[]), want: &mut want_kernels };
        let (a, b, c) = (k.vec(f8, &f32s(A)), k.vec(f8, &f32s(B)), k.vec(f8, &f32s(C)));
        let sum = zip(A, B, |x, y| x + y);
        let s = k.op(VecOp::Add, f8, &[a, b]);
        k.out(s, f8, &f32s(sum));
        let prod = zip(sum, A, |x, y| x * y);
        let p = k.op(VecOp::Mul, f8, &[s, a]);
        k.out(p, f8, &f32s(prod));
        // another avx fn and back, the vector in `ymm0` both ways
        let q = k.b.call(sq, Some(f8t), &[p.into()]).unwrap();
        k.out(q, f8, &f32s(zip(prod, prod, |x, y| x * y)));
        // baseline code and back, with 256-bit values live around the calls: directly, and
        // through a fn value
        let (cx, cy) = (k.b.f32(1.5), k.b.f32(2.25));
        let (x, y) = (k.opaque(cx, t.f32), k.opaque(cy, t.f32));
        let r = k.b.call(fbase, Some(t.f32), &[x.into(), y.into()]).unwrap();
        k.out_scalar(r, t.f32, fbase_bits);
        let fo = k.b.func(base);
        let g = k.opaque(fo, base_ft);
        let seven = k.b.int(7);
        let r = k.b.call_ind(g, Some(t.u64), &[seven]).unwrap();
        k.out_scalar(r, t.u64, 22);
        // no `fma` in this fn: the software routine, lane by lane
        let f = k.op(VecOp::Fma, f8, &[a, b, c]);
        k.out(f, f8, &f32s(fma32()));
        let n = k.op(VecOp::Neg, f8, &[a]);
        k.out(n, f8, &f32s(A.map(|x| -x)));
        let m = k.lt(f8, a, b);
        let sel = k.op(VecOp::Select, f8, &[m, a, b]);
        k.out(sel, f8, &f32s(zip(A, B, |x, y| if x < y { x } else { y })));
        // the fixed tree: lane i with lane i + n/2
        let tree = ((sum[0] + sum[4]) + (sum[2] + sum[6])) + ((sum[1] + sum[5]) + (sum[3] + sum[7]));
        let red = k.b.op(Op::Vec(VecOp::ReduceAdd), t.f32, &[s.into()]);
        k.out_scalar(red, t.f32, u64::from(tree.to_bits()));
        let (da, db, dc) = (k.vec(d4, &f64s(DA)), k.vec(d4, &f64s(DB)), k.vec(d4, &f64s(DC)));
        let ds = k.op(VecOp::Add, d4, &[da, db]);
        k.out(ds, d4, &f64s(zip(DA, DB, |x, y| x + y)));
        let df = k.op(VecOp::Fma, d4, &[da, db, dc]);
        k.out(df, d4, &f64s(fma64()));
        k.finish(k_avx);
    }

    // avx: a 256-bit argument and result through a fn value
    let k_ind = sp.p.decl("t.k_ind", &[], None);
    {
        let mut k = Kern { sp: &mut sp, b: FuncBuilder::new(&[]), want: &mut want_kernels };
        let a = k.vec(f8, &f32s(A));
        let fo = k.b.func(sq);
        let g = k.opaque(fo, sq_ft);
        let q = k.b.call_ind(g, Some(f8t), &[a.into()]).unwrap();
        k.out(q, f8, &f32s(A.map(|x| x * x)));
        k.finish(k_ind);
    }

    // avx2 without fma: 256-bit integer ops, the software fma
    let k_avx2 = sp.p.decl("t.k_avx2", &[], None);
    {
        let mut k = Kern { sp: &mut sp, b: FuncBuilder::new(&[]), want: &mut want_kernels };
        let (x, y, s) = (k.vec(i8x, &i32s(X)), k.vec(i8x, &i32s(Y)), k.vec(i8x, &i32s(S)));
        let r = k.op(VecOp::AddWrap, i8x, &[x, y]);
        k.out(r, i8x, &i32s(zip(X, Y, i32::wrapping_add)));
        let r = k.op(VecOp::MulWrap, i8x, &[x, y]);
        k.out(r, i8x, &i32s(zip(X, Y, i32::wrapping_mul)));
        let r = k.op(VecOp::ShlWrap, i8x, &[x, s]);
        k.out(r, i8x, &i32s(zip(X, S, |v, n| v.wrapping_shl(n as u32))));
        let m = k.lt(i8x, x, y);
        let r = k.op(VecOp::Select, i8x, &[m, x, y]);
        k.out(r, i8x, &i32s(zip(X, Y, i32::min)));
        let u: Vec<u64> = (0..32u64).map(|i| i * 9 % 256).collect();
        let v: Vec<u64> = (0..32u64).map(|i| 250 - i * 7).collect();
        let sat: Vec<u64> = u.iter().zip(&v).map(|(p, q)| (p + q).min(255)).collect();
        let (uv, vv) = (k.vec(u8x, &u), k.vec(u8x, &v));
        let r = k.op(VecOp::AddSat, u8x, &[uv, vv]);
        k.out(r, u8x, &sat);
        let (a, b, c) = (k.vec(f8, &f32s(A)), k.vec(f8, &f32s(B)), k.vec(f8, &f32s(C)));
        let f = k.op(VecOp::Fma, f8, &[a, b, c]);
        k.out(f, f8, &f32s(fma32()));
        k.finish(k_avx2);
    }

    // avx2 and fma: the instruction
    let k_fma = sp.p.decl("t.k_fma", &[], None);
    {
        let mut k = Kern { sp: &mut sp, b: FuncBuilder::new(&[]), want: &mut want_kernels };
        let (a, b, c) = (k.vec(f8, &f32s(A)), k.vec(f8, &f32s(B)), k.vec(f8, &f32s(C)));
        let f = k.op(VecOp::Fma, f8, &[a, b, c]);
        k.out(f, f8, &f32s(fma32()));
        let (da, db, dc) = (k.vec(d4, &f64s(DA)), k.vec(d4, &f64s(DB)), k.vec(d4, &f64s(DC)));
        let df = k.op(VecOp::Fma, d4, &[da, db, dc]);
        k.out(df, d4, &f64s(fma64()));
        let (x, y) = (k.vec(i8x, &i32s(X)), k.vec(i8x, &i32s(Y)));
        let r = k.op(VecOp::SubWrap, i8x, &[x, y]);
        k.out(r, i8x, &i32s(zip(X, Y, i32::wrapping_sub)));
        k.finish(k_fma);
    }

    // the whole v3 set: `pext` for the bits of a 16-bit-lane mask
    let k_v3 = sp.p.decl("t.k_v3", &[], None);
    {
        let mut k = Kern { sp: &mut sp, b: FuncBuilder::new(&[]), want: &mut want_kernels };
        let p: Vec<u64> = (0..16u64).map(|i| i * 4099 % 65536).collect();
        let q: Vec<u64> = (0..16u64).map(|i| 20000 + (i % 3) * 20000).collect();
        let bits = (0..16).fold(0u64, |acc, i| acc | u64::from(p[i] < q[i]) << i);
        let (pv, qv) = (k.vec(u16x, &p), k.vec(u16x, &q));
        let m = k.lt(u16x, pv, qv);
        let r = k.b.op(Op::Vec(VecOp::ToBits), t.u16, &[m.into()]);
        k.out_scalar(r, t.u16, bits);
        let w: [i64; 4] = [3, -5, i64::MAX, 1 << 40];
        let z: [i64; 4] = [-7, 11, 2, 1 << 30];
        let lanes = |x: [i64; 4]| x.map(|v| v as u64).to_vec();
        let (wv, zv) = (k.vec(q4, &lanes(w)), k.vec(q4, &lanes(z)));
        let r = k.op(VecOp::MulWrap, q4, &[wv, zv]);
        k.out(r, q4, &lanes(zip(w, z, i64::wrapping_mul)));
        let (a, b, c) = (k.vec(f8, &f32s(A)), k.vec(f8, &f32s(B)), k.vec(f8, &f32s(C)));
        let f = k.op(VecOp::Fma, f8, &[a, b, c]);
        k.out(f, f8, &f32s(fma32()));
        k.finish(k_v3);
    }

    // entry: the baseline path; the kernels unless there is a command-line argument
    {
        let mut b = FuncBuilder::new(&[t.pusize]);
        let stack = b.param(0).unwrap();
        let argc = b.load(t.usize, stack);
        b.call(base_path, None, &[]);
        let (run, done) = (b.create_block(), b.create_block());
        let one = b.int(1);
        let alone = b.cmp(Op::Eq, t.bool, argc, one);
        b.cbr(alone, run, &[], done, &[]);
        b.switch_to(run);
        for f in [k_avx, k_ind, k_avx2, k_fma, k_v3] {
            b.call(f, None, &[]);
        }
        b.br(done, &[]);
        b.switch_to(done);
        let (nr, zero) = (b.int(231), b.int(0));
        b.intrinsic(sp.p.sys1, Some(t.isize), &[nr, zero]);
        b.unreachable();
        sp.p.def(sp.p.entry, b);
    }

    use Feature::*;
    let needs: [(FuncId, &[Feature]); 6] = [
        (sq, &[Avx]),
        (k_avx, &[Avx]),
        (k_ind, &[Avx]),
        (k_avx2, &[Avx2]),
        (k_fma, &[Avx2, Fma]),
        (k_v3, &[Avx2, Fma, Bmi2, F16c]),
    ];
    for (f, fs) in needs {
        sp.p.m.funcs[f.index()].features = FeatureSet::from_features(fs.iter().copied());
    }
    // the helpers that build and print the 256-bit vectors
    for f in &mut sp.p.m.funcs {
        if f.path.starts_with("t.mk") || f.path.starts_with("t.out") {
            f.features = FeatureSet::of(Avx);
        }
    }
    let kernels = vec![
        (sq, (false, false)),
        (k_avx, (false, false)),
        (k_ind, (true, true)),
        (k_avx2, (false, false)),
        (k_fma, (false, false)),
        (k_v3, (false, false)),
    ];
    Mixed { sp, want_base, want_kernels, kernels, fma: k_fma, soft_fma: [k_avx, k_avx2], v3: k_v3 }
}

/// Runs `path` (through `runner` when given) with `args`: its stdout and exit status (`None`:
/// killed by a signal).
fn run(path: &Path, runner: &[&str], args: &[&str]) -> (Vec<u8>, Option<i32>) {
    loop {
        let mut cmd = match runner.split_first() {
            Some((r, rest)) => {
                let mut c = Command::new(r);
                c.args(rest).arg(path);
                c
            }
            None => Command::new(path),
        };
        match cmd.args(args).output() {
            Err(e) if e.raw_os_error() == Some(26) => std::thread::sleep(std::time::Duration::from_millis(5)),
            r => {
                let o = r.unwrap();
                return (o.stdout, o.status.code());
            }
        }
    }
}

/// `got` is `want`, line by line.
fn same_lines(got: &[u8], want: &str, what: &str) {
    let got = String::from_utf8_lossy(got);
    let (g, w): (Vec<&str>, Vec<&str>) = (got.lines().collect(), want.lines().collect());
    for (k, (a, b)) in g.iter().zip(&w).enumerate() {
        assert_eq!(a, b, "{what}: line {}", k + 1);
    }
    assert_eq!(g.len(), w.len(), "{what}: line count");
}

/// The host CPU runs every kernel of [`mixed`].
fn host_runs_v3() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("bmi2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Functions of four feature sets in one v2-baseline module, called from baseline code and
/// calling it back: the output equals the scalar references, every function's instructions are
/// inside its set, the upper halves are clean where the ABI says, and the baseline path runs on
/// a CPU without AVX.
#[test]
fn feature_sets_in_one_module() {
    let mx = mixed();
    let p = &mx.sp.p;
    let m = &p.m;
    stave_ir::verify_module(m).unwrap_or_else(|e| panic!("IR does not verify: {e:?}"));
    let full = interp_args(p, &[b"t"]);
    let base = interp_args(p, &[b"t", b"x"]);
    same_lines(
        &full.stdout,
        &format!("{}{}", mx.want_base, mx.want_kernels),
        "interpreter against the references",
    );
    same_lines(&base.stdout, &mx.want_base, "interpreter, baseline path");
    for (build, o) in [("debug", Options::default()), ("release", release_options())] {
        let what = format!("mixed [{build}]");
        assert_eq!(o.features, v2());
        let img = compile_module(m, &o).unwrap_or_else(|e| panic!("{what}: codegen: {e}"));
        if have_objdump() {
            check_image(m, &img, o.features, &what);
            let fns = functions(m, &img, o.features);
            let code = |f: FuncId| -> &[Inst] {
                let path = &m.funcs[f.index()].path;
                &fns.iter().find(|x| &x.0 == path).unwrap_or_else(|| panic!("{what}: no code of {path}")).2
            };
            for &(f, ind) in &mx.kernels {
                let func = &m.funcs[f.index()];
                let c = code(f);
                assert!(c.iter().any(Inst::ymm), "{what}: `{}` has no 256-bit instruction", func.path);
                let cleaned = check_upper_halves(m, func, c, ind, &what);
                assert!(cleaned > 0 || func.ret.is_some(), "{what}: `{}` has no vzeroupper", func.path);
            }
            let soft =
                |i: &Inst| i.mnem == "call" && i.sym.as_deref().is_some_and(|s| s.starts_with("__stave_fma"));
            assert!(code(mx.fma).iter().any(|i| i.mnem.starts_with("vfmadd")), "{what}: no fma instruction");
            assert!(!code(mx.fma).iter().any(soft), "{what}: software fma in a fn with fma");
            for f in mx.soft_fma {
                assert!(code(f).iter().any(soft), "{what}: no software fma without the feature");
            }
            assert!(code(mx.v3).iter().any(|i| i.mnem == "pext"), "{what}: no pext with bmi2");
        }
        if !native_host() {
            eprintln!("SKIP native run of {what}: host is not x86-64 Linux");
            continue;
        }
        let path = tmp(&format!("mixed-{build}"));
        img.write_file(&path).unwrap();
        let (out, code) = run(&path, &[], &["x"]);
        same_lines(&out, &mx.want_base, &format!("{what}: baseline path"));
        assert_eq!(code, Some(0), "{what}: baseline path");
        if host_runs_v3() {
            let (out, code) = run(&path, &[], &[]);
            same_lines(&out, &String::from_utf8_lossy(&full.stdout), &format!("{what}: every kernel"));
            assert_eq!(code, Some(0), "{what}: every kernel");
        } else {
            eprintln!("SKIP kernels of {what}: the host CPU lacks avx2, fma or bmi2");
        }
        if have("qemu-x86_64") {
            // a CPU without AVX: the baseline path runs, the kernels do not
            let nehalem = ["qemu-x86_64", "-cpu", "Nehalem"];
            let (out, code) = run(&path, &nehalem, &["x"]);
            same_lines(&out, &mx.want_base, &format!("{what}: baseline path on Nehalem"));
            assert_eq!(code, Some(0), "{what}: baseline path on Nehalem");
            let (_, code) = run(&path, &nehalem, &[]);
            assert_eq!(code, None, "{what}: the kernels on a CPU without AVX end in SIGILL");
        } else {
            eprintln!("SKIP run of {what} on a CPU without AVX: qemu-x86_64 not installed");
        }
    }
}

/// An op whose instructions are outside the function's feature set is `Error::Unsupported`
/// naming the missing feature, in a function that otherwise compiles.
#[test]
fn op_outside_the_feature_set_is_unsupported() {
    let cases: [(Shape, VecOp, &[Feature], &str); 3] = [
        // 256-bit integer arithmetic is AVX2
        (sh(Lane::I32, 8), VecOp::AddWrap, &[Feature::Avx], "avx2"),
        (sh(Lane::U8, 32), VecOp::AddSat, &[Feature::Avx], "avx2"),
        // a 256-bit value needs `avx` in the function that holds it
        (sh(Lane::F32, 8), VecOp::Add, &[], "wider than the fn's 128-bit vector registers"),
    ];
    for (s, op, needs, missing) in cases {
        let mut sp = Sp::new();
        let f = sp.p.decl("t.k", &[], None);
        let mut b = FuncBuilder::new(&[]);
        let lanes: Vec<u64> = (0..u64::from(s.n)).collect();
        let (x, y) = (sp.vec(&mut b, s, &lanes), sp.vec(&mut b, s, &lanes));
        let vt = sp.vt(s);
        let r = b.op(Op::Vec(op), vt, &[x.into(), y.into()]);
        sp.out(&mut b, r, s);
        b.ret(None);
        sp.p.def(f, b);
        sp.p.main(|_, b| {
            b.call(f, None, &[]);
        });
        let mut m = sp.p.m.clone();
        for g in &mut m.funcs {
            if g.path.starts_with("t.mk") || g.path.starts_with("t.out") {
                g.features = FeatureSet::of(Feature::Avx);
            }
        }
        m.funcs[f.index()].features = FeatureSet::from_features(needs.iter().copied());
        for (build, base) in [("debug", Options::default()), ("release", release_options())] {
            let what = format!("{} {} [{build}]", op.name(), s.name());
            let o = Options { main: sp.p.main, ..base };
            match compile_module(&m, &o) {
                Err(Error::Unsupported { func, what: msg }) => {
                    assert_eq!(func, Some(f), "{what}: {msg}");
                    assert!(msg.contains(missing), "{what}: {msg}");
                }
                r => panic!("{what}: expected Unsupported, got {:?}", r.map(|_| ())),
            }
            let mut ok = m.clone();
            ok.funcs[f.index()].features = FeatureSet::of(Feature::Avx2);
            let img = compile_module(&ok, &o).unwrap_or_else(|e| panic!("{what} with avx2: {e}"));
            check_image(&ok, &img, o.features, &what);
        }
    }
}
