//! Differential test harness: IR programs built with `FuncBuilder` plus a tiny runtime written in
//! IR (`runtime.start.entry` calls `main` and exits with its status; `runtime.panic.trap` exits
//! with `100 + kind`; `t.print_u64` writes 16 hex digits and a newline). Each program runs in the
//! stave-ir interpreter (the oracle) and, compiled by the debug backend into a static ELF, natively;
//! stdout and the exit status must agree (a trap in the interpreter = exit `100 + kind` natively).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

use stave_ir::types::{AdtKind, AdtShape, DeclId, GenericArg};
use stave_ir::{
    BufferHost, CastKind, Config, FuncBuilder, FuncId, InstanceId, Interp, IntrinsicId, Module, Op, Operand,
    Outcome, PanicKind, SyscallAbi, TypeId, TypeKey, Val,
};

use stave_target::{Feature, FeatureSet};

use crate::debug::{Options, compile_module};

/// The baseline of `x86_64-linux` (v2).
pub fn v2() -> FeatureSet {
    FeatureSet::of(Feature::Sse42)
}

/// The baseline of `x86_64-linux-v3`.
pub fn v3() -> FeatureSet {
    FeatureSet::from_features([Feature::Sse42, Feature::Avx2, Feature::Fma, Feature::Bmi2, Feature::F16c])
}

/// The features of K mode: what `x86_64-linux-v4` adds to v3.
pub const V4_NEEDS: [Feature; 5] =
    [Feature::Avx512f, Feature::Avx512bw, Feature::Avx512vl, Feature::Avx512dq, Feature::Avx512cd];

/// The baseline of `x86_64-linux-v4`.
pub fn v4() -> FeatureSet {
    v3().union(FeatureSet::from_features(V4_NEEDS))
}

/// How a program with AVX-512 code runs here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum V4Leg {
    /// The CPU has the set.
    Native,
    /// Under Intel SDE: `$STAVE_SDE -icx -- <program>`.
    Sde(PathBuf),
}

/// The `v4` leg: natively on a CPU with the AVX-512 set, else under the emulator the
/// environment variable `STAVE_SDE` names (the path of `sde64`), else not at all (`None`).
/// Without it a test skips its AVX-512 runs, or fails when `STAVE_REQUIRE_LEGS` lists `v4`.
pub fn v4_leg() -> Option<V4Leg> {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("avx512dq")
            && std::arch::is_x86_feature_detected!("avx512cd")
        {
            return Some(V4Leg::Native);
        }
        if let Some(sde) = std::env::var_os("STAVE_SDE").map(PathBuf::from)
            && sde.is_file()
        {
            return Some(V4Leg::Sde(sde));
        }
    }
    let required = std::env::var("STAVE_REQUIRE_LEGS").unwrap_or_default();
    assert!(
        !required.split(',').any(|l| l.trim() == "v4"),
        "STAVE_REQUIRE_LEGS requires leg `v4` but it cannot run here: needs a CPU with avx512f, avx512bw, avx512vl, avx512dq and avx512cd, or STAVE_SDE naming Intel SDE's `sde64`"
    );
    None
}

/// The command that runs the program at `p` on the `v4` leg.
pub fn v4_command(leg: &V4Leg, p: &Path) -> Command {
    match leg {
        V4Leg::Native => Command::new(p),
        V4Leg::Sde(sde) => {
            let mut c = Command::new(sde);
            c.args(["-icx", "--"]).arg(p);
            c
        }
    }
}

/// Commonly used types.
#[derive(Clone, Copy, Debug)]
pub struct Tys {
    pub i8: TypeId,
    pub i16: TypeId,
    pub i32: TypeId,
    pub i64: TypeId,
    pub isize: TypeId,
    pub u8: TypeId,
    pub u16: TypeId,
    pub u32: TypeId,
    pub u64: TypeId,
    pub usize: TypeId,
    pub bool: TypeId,
    pub char: TypeId,
    pub f32: TypeId,
    pub f64: TypeId,
    pub str_ref: TypeId,
    pub pu8: TypeId,
    pub pusize: TypeId,
}

pub struct Prog {
    pub m: Module,
    pub t: Tys,
    pub print: FuncId,
    pub write: FuncId,
    pub entry: FuncId,
    pub trap: Option<FuncId>,
    pub main: Option<FuncId>,
    pub sys1: IntrinsicId,
    pub sys3: IntrinsicId,
    next_inst: u32,
    next_decl: u32,
    ids: Vec<(TypeId, FuncId)>,
}

pub fn int_types(t: &Tys) -> [TypeId; 10] {
    [t.i8, t.i16, t.i32, t.i64, t.isize, t.u8, t.u16, t.u32, t.u64, t.usize]
}

impl Prog {
    pub fn new() -> Prog {
        Prog::with_trap(true)
    }

    /// `trap = false`: no `runtime.panic.trap` (the backend's fallback reports traps).
    pub fn with_trap(trap: bool) -> Prog {
        let mut m = Module::new();
        let ty = &mut m.types;
        let u8_ = ty.u8();
        let usize_ = ty.usize();
        let s = ty.str();
        let t = Tys {
            i8: ty.i8(),
            i16: ty.i16(),
            i32: ty.i32(),
            i64: ty.i64(),
            isize: ty.isize(),
            u8: u8_,
            u16: ty.u16(),
            u32: ty.u32(),
            u64: ty.u64(),
            usize: usize_,
            bool: ty.bool(),
            char: ty.char(),
            f32: ty.f32(),
            f64: ty.f64(),
            str_ref: ty.ref_(false, s),
            pu8: ty.ptr(false, u8_),
            pusize: ty.ptr(false, usize_),
        };
        let sys1 = m.intrinsic("syscall1");
        let sys3 = m.intrinsic("syscall3");
        let mut p = Prog {
            m,
            t,
            print: FuncId(0),
            write: FuncId(0),
            entry: FuncId(0),
            trap: None,
            main: None,
            sys1,
            sys3,
            next_inst: 0,
            next_decl: 100,
            ids: Vec::new(),
        };
        p.entry = p.decl_known(stave_ir::Known::StartEntry, &[t.pusize], None);
        p.write = p.decl("t.write", &[t.pu8, t.usize], None);
        p.print = p.decl("t.print_u64", &[t.u64], None);
        p.def_write();
        p.def_print();
        if trap {
            let f = p.decl_known(stave_ir::Known::Trap, &[t.u32, t.u32], None);
            let mut b = FuncBuilder::new(&[t.u32, t.u32]);
            let k = b.param(0).unwrap();
            let kk = b.cast(CastKind::Exact, t.usize, k);
            let c100 = b.int(100);
            let code = b.binary(Op::AddWrap, t.usize, kk, c100);
            let nr = b.int(231);
            b.intrinsic(sys1, Some(t.isize), &[nr, code.into()]);
            b.unreachable();
            p.def(f, b);
            p.trap = Some(f);
        }
        p
    }

    pub fn decl(&mut self, path: &str, params: &[TypeId], ret: Option<TypeId>) -> FuncId {
        self.next_inst += 1;
        self.m.declare(InstanceId(vec![self.next_inst]), path, params.to_vec(), ret)
    }

    /// Declares the function of compiler-known item `k`, stamped as the front end does.
    pub fn decl_known(&mut self, k: stave_ir::Known, params: &[TypeId], ret: Option<TypeId>) -> FuncId {
        let f = self.decl(&k.path(), params, ret);
        self.m.funcs[f.index()].known = Some(k);
        f
    }

    pub fn def(&mut self, f: FuncId, b: FuncBuilder) {
        let body = b.finish().unwrap_or_else(|e| panic!("build error: {e}"));
        assert!(self.m.define(f, body));
    }

    /// `Option<t>` (variant 0 = None, 1 = Some).
    pub fn option(&mut self, t: TypeId) -> TypeId {
        self.next_decl += 1;
        self.m.record_decl(DeclId(1), AdtKind::Enum, "t.Option");
        self.m.types.adt(DeclId(1), vec![GenericArg::Type(t)], AdtShape::enumeration(vec![vec![], vec![t]]))
    }

    pub fn structure(&mut self, fields: Vec<TypeId>) -> TypeId {
        self.next_decl += 1;
        self.m.record_decl(DeclId(self.next_decl), AdtKind::Struct, format!("t.S{}", self.next_decl));
        self.m.types.adt(DeclId(self.next_decl), vec![], AdtShape::structure(fields))
    }

    pub fn enumeration(&mut self, vs: Vec<Vec<TypeId>>) -> TypeId {
        self.next_decl += 1;
        self.m.record_decl(DeclId(self.next_decl), AdtKind::Enum, format!("t.E{}", self.next_decl));
        self.m.types.adt(DeclId(self.next_decl), vec![], AdtShape::enumeration(vs))
    }

    fn def_write(&mut self) {
        let t = self.t;
        let mut b = FuncBuilder::new(&[t.pu8, t.usize]);
        let (p, n) = (b.param(0).unwrap(), b.param(1).unwrap());
        let pu = b.cast(CastKind::Exact, t.usize, p);
        let one = b.int(1);
        let one2 = b.int(1);
        b.intrinsic(self.sys3, Some(t.isize), &[one, one2, pu.into(), n.into()]);
        b.ret(None);
        self.def(self.write, b);
    }

    fn def_print(&mut self) {
        let t = self.t;
        let arr = self.m.types.array(t.u8, 17);
        let parr = self.m.types.ref_(true, arr);
        let pel = self.m.types.ref_(true, t.u8);
        let mut b = FuncBuilder::new(&[t.u64]);
        let x = b.param(0).unwrap();
        let s = b.slot(parr);
        let (head, iv) = b.create_block_with(&[t.u64]);
        let body = b.create_block();
        let exit = b.create_block();
        let z = b.int(0);
        b.br(head, &[z]);
        b.switch_to(head);
        let i = iv[0];
        let c16 = b.int(16);
        let c = b.cmp(Op::Lt, t.bool, i, c16);
        b.cbr(c, body, &[], exit, &[]);
        b.switch_to(body);
        let c4 = b.int(4);
        let i4 = b.binary(Op::MulWrap, t.u64, i, c4);
        let c60 = b.int(60);
        let sh = b.binary(Op::SubWrap, t.u64, c60, i4);
        let sx = b.binary(Op::Shr, t.u64, x, sh);
        let c15 = b.int(15);
        let nib = b.binary(Op::And, t.u64, sx, c15);
        let c10 = b.int(10);
        let isd = b.cmp(Op::Lt, t.bool, nib, c10);
        let c48 = b.int(48);
        let d1 = b.binary(Op::AddWrap, t.u64, nib, c48);
        let c87 = b.int(87);
        let d2 = b.binary(Op::AddWrap, t.u64, nib, c87);
        let d = b.select(t.u64, isd, d1, d2);
        let d8 = b.cast(CastKind::Wrap, t.u8, d);
        let p = b.op(Op::ElemAddr, pel, &[s.into(), i.into()]);
        b.store(p, d8);
        let one = b.int(1);
        let i2 = b.binary(Op::AddWrap, t.u64, i, one);
        b.br(head, &[i2.into()]);
        b.switch_to(exit);
        let c16b = b.int(16);
        let p = b.op(Op::ElemAddr, pel, &[s.into(), c16b]);
        let nl = b.int(10);
        b.store(p, nl);
        let pp = b.cast(CastKind::Exact, t.pu8, s);
        let n = b.int(17);
        b.call(self.write, None, &[pp.into(), n]);
        b.ret(None);
        self.def(self.print, b);
    }

    /// Emits `print_u64(v as u64)` for a scalar `v` of type `ty` (floats print their bits).
    pub fn print(&mut self, b: &mut FuncBuilder, v: Operand, ty: TypeId) {
        let t = self.t;
        let key = self.m.types.key(ty).cloned();
        let x: Operand = match key {
            Some(TypeKey::Int { .. }) => b.cast(CastKind::Wrap, t.u64, v).into(),
            Some(TypeKey::Bool | TypeKey::Char) => b.cast(CastKind::Exact, t.u64, v).into(),
            Some(TypeKey::Float(f)) => {
                let (bits_t, w) = match f {
                    stave_ir::FloatBits::F32 => (t.u32, ty),
                    stave_ir::FloatBits::F64 => (t.u64, ty),
                    _ => (t.u16, ty),
                };
                let ps = self.m.types.ref_(true, w);
                let pb = self.m.types.ptr(true, bits_t);
                let s = b.slot(ps);
                b.store(s, v);
                let pp = b.cast(CastKind::Exact, pb, s);
                let raw = b.load(bits_t, pp);
                b.cast(CastKind::Exact, t.u64, raw).into()
            }
            Some(TypeKey::Ref { .. } | TypeKey::RawPtr { .. }) => b.cast(CastKind::Exact, t.u64, v).into(),
            _ => panic!("print of t{}", ty.0),
        };
        b.call(self.print, None, &[x]);
    }

    /// `id_T(c)`: an opaque copy of `c` (keeps constants out of the op under test).
    pub fn opaque(&mut self, b: &mut FuncBuilder, c: Operand, ty: TypeId) -> stave_ir::Value {
        let f = match self.ids.iter().find(|x| x.0 == ty) {
            Some(x) => x.1,
            None => {
                let f = self.decl(&format!("t.id{}", ty.0), &[ty], Some(ty));
                let mut fb = FuncBuilder::new(&[ty]);
                let x = fb.param(0).unwrap();
                fb.ret(Some(x.into()));
                self.def(f, fb);
                self.ids.push((ty, f));
                f
            }
        };
        b.call(f, Some(ty), &[c]).unwrap()
    }

    /// Emits a write of a string literal.
    pub fn print_str(&mut self, b: &mut FuncBuilder, s: &str) {
        let t = self.t;
        let id = self.m.add_string(s.as_bytes().to_vec());
        let so = b.str(id);
        let p = b.op(Op::SlicePtr, t.pu8, &[so]);
        let n = b.op(Op::SliceLen, t.usize, &[so]);
        b.call(self.write, None, &[p.into(), n.into()]);
    }

    /// Defines `runtime.start.entry` calling `main` (`main` returns `i32` or unit).
    pub fn set_main(&mut self, main: FuncId) {
        let t = self.t;
        self.main = Some(main);
        let ret = self.m.funcs[main.index()].ret;
        let mut b = FuncBuilder::new(&[t.pusize]);
        let code: Operand = match ret {
            Some(r) => {
                let v = b.call(main, Some(r), &[]).unwrap();
                b.cast(CastKind::Wrap, t.usize, v).into()
            }
            None => {
                b.call(main, None, &[]);
                b.int(0)
            }
        };
        let nr = b.int(231);
        b.intrinsic(self.sys1, Some(t.isize), &[nr, code]);
        b.unreachable();
        self.def(self.entry, b);
    }

    /// Declares `main() -> i32` with the body `f` builds (it must leave the builder open: the
    /// harness adds `ret 0`).
    pub fn main(&mut self, f: impl FnOnce(&mut Prog, &mut FuncBuilder)) -> FuncId {
        let t = self.t;
        let main = self.decl("t.main", &[], Some(t.i32));
        let mut b = FuncBuilder::new(&[]);
        f(self, &mut b);
        if b.is_open() {
            let z = b.int(0);
            b.ret(Some(z));
        }
        self.def(main, b);
        self.set_main(main);
        main
    }
}

/// What a run produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunOut {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub code: i32,
}

pub fn kind_index(k: PanicKind) -> i32 {
    PanicKind::ALL.iter().position(|x| *x == k).unwrap_or(0) as i32
}

pub fn interp(p: &Prog) -> RunOut {
    interp_args(p, &[b"t"])
}

/// The interpreter's run with the command line `args` (the program name first).
pub fn interp_args(p: &Prog, args: &[&[u8]]) -> RunOut {
    let cfg = Config { max_steps: 2_000_000_000, max_depth: 100_000, ..Config::default() };
    let mut it = Interp::new(&p.m, BufferHost::new(SyscallAbi::X86_64), cfg)
        .unwrap_or_else(|e| panic!("interp setup: {e:?}"));
    let sp = it.process_stack(args, &[]).unwrap();
    let out = it.run(p.entry, vec![Val::S(sp)]).unwrap_or_else(|e| panic!("interp: {e:?}"));
    let code = match out {
        Outcome::Exit(c) => c,
        Outcome::Trap(t) => 100 + kind_index(t.kind),
        Outcome::Return(_) => panic!("entry returned"),
        Outcome::MemError(r) => panic!("{}", r.render(&p.m)),
        Outcome::BorrowError(r) => panic!("{}", r.render(&p.m)),
    };
    RunOut { stdout: it.host.stdout.clone(), stderr: it.host.stderr.clone(), code }
}

static SEQ: AtomicU32 = AtomicU32::new(0);

/// A scratch path in a directory of its own; the directory is removed when this is dropped.
pub struct Tmp(PathBuf);

pub fn tmp(name: &str) -> Tmp {
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let d = std::env::temp_dir().join(format!("stave-x64-debug-{}-{seq}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    Tmp(d.join(name))
}

impl Drop for Tmp {
    fn drop(&mut self) {
        if let Some(d) = self.0.parent() {
            let _ = std::fs::remove_dir_all(d);
        }
    }
}

impl std::ops::Deref for Tmp {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for Tmp {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<std::ffi::OsStr> for Tmp {
    fn as_ref(&self) -> &std::ffi::OsStr {
        self.0.as_os_str()
    }
}

pub fn native_host() -> bool {
    cfg!(all(target_os = "linux", target_arch = "x86_64"))
}

pub fn run_file(p: &std::path::Path) -> RunOut {
    let mut cmd = Command::new(p);
    let out = loop {
        match cmd.output() {
            Err(e) if e.raw_os_error() == Some(26) => std::thread::sleep(std::time::Duration::from_millis(5)),
            r => break r.unwrap(),
        }
    };
    use std::os::unix::process::ExitStatusExt;
    let code = match out.status.code() {
        Some(c) => c,
        None => panic!("{}: killed by signal {:?}", p.display(), out.status.signal()),
    };
    RunOut { stdout: out.stdout, stderr: out.stderr, code }
}

/// Compiles `p` and runs it natively (`None` on a non-x86-64-Linux host).
pub fn native(p: &Prog, name: &str, opts: &Options) -> Option<RunOut> {
    native_module(&p.m, p.main, name, opts)
}

/// [`native`] of the module `m`. A module with `needs` has the instructions of every function
/// checked against the function's features.
pub fn native_module(m: &Module, main: Option<FuncId>, name: &str, opts: &Options) -> Option<RunOut> {
    let opts = Options { main: opts.main.or(main), ..opts.clone() };
    let img = compile_module(m, &opts).unwrap_or_else(|e| panic!("{name}: codegen: {e}"));
    if m.funcs.iter().any(|f| !f.features.is_empty()) {
        super::features::check_image(m, &img, opts.features, name);
    }
    if !native_host() {
        return None;
    }
    let path = tmp(name);
    img.write_file(&path).unwrap();
    let r = run_file(&path);
    Some(r)
}

fn show(b: &[u8]) -> String {
    let s = String::from_utf8_lossy(b);
    if s.len() > 4000 { format!("{}…", &s[..4000]) } else { s.into_owned() }
}

/// The release configuration: packed image, release code generation.
pub fn release_options() -> Options {
    Options {
        image: stave_obj::Options::release(stave_obj::Machine::X86_64),
        release: true,
        ..Options::default()
    }
}

/// `m` with `avx` in the `needs` of every function but the runtime's: baseline code calls
/// functions with the feature, and they call it back.
pub fn with_avx(m: &Module) -> Module {
    let mut m = m.clone();
    for f in &mut m.funcs {
        if !f.path.starts_with("runtime.") {
            f.features = FeatureSet::of(Feature::Avx);
        }
    }
    m
}

/// Runs in the interpreter and natively, as a debug and as a release build, and as a debug
/// build whose functions have `avx` on top of the v2 baseline (their scalar float code is
/// VEX-encoded), and compares; returns the interpreter's result.
pub fn check(p: &Prog, name: &str) -> RunOut {
    let want = interp(p);
    let avx = with_avx(&p.m);
    let builds = [
        ("debug", Options::default(), &p.m),
        ("release", release_options(), &p.m),
        ("debug-avx", Options::default(), &avx),
    ];
    for (oname, opts, m) in builds {
        if let Some(got) = native_module(m, p.main, &format!("{name}-{oname}"), &opts) {
            if std::env::var_os("STAVE_TEST_VERBOSE").is_some() {
                eprintln!("{name} ({oname}): native exit {} stdout {} bytes", got.code, got.stdout.len());
            }
            if got.stdout != want.stdout {
                let (g, w) = (show(&got.stdout), show(&want.stdout));
                let first = g.lines().zip(w.lines()).position(|(a, b)| a != b);
                panic!(
                    "{name} ({oname}): stdout differs (first differing line {first:?})\nnative:\n{g}\ninterp:\n{w}\nnative stderr: {}",
                    show(&got.stderr)
                );
            }
            assert_eq!(got.code, want.code, "{name} ({oname}): exit status (native vs interp)");
        } else {
            eprintln!("SKIP native run of {name}: host is not x86-64 Linux");
        }
    }
    want
}
