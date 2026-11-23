//! Differential harness: an IR program (with a tiny test runtime written in IR) runs in the
//! stave-ir interpreter (the oracle) and, compiled by the AArch64 debug backend and linked by
//! stave-obj, as a static ELF under `qemu-aarch64`; stdout, stderr, and the exit status must match.
//!
//! Test runtime: `runtime.start.entry(sp)` calls `app.main()` then `exit_group(0)`;
//! `runtime.panic.trap(kind, site)` writes the two `u32`s (8 bytes, little-endian) to fd 2 and
//! exits 101. An interpreter trap is expected to look exactly like that natively.
#![allow(dead_code)]

#[path = "../../../../tests/support/legs.rs"]
pub mod legs;
pub mod rgen;
pub mod vgen;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

use stave_codegen_a64::debug::{BuildConfig, build_image};
use stave_ir::interp::{Config, Interp, Outcome, Val};
use stave_ir::{
    BufferHost, CastKind, FuncBuilder, FuncId, InstanceId, IntrinsicId, Module, Op, Operand, PanicKind,
    SyscallAbi, TypeId, Value,
};

static SEQ: AtomicU32 = AtomicU32::new(0);
static NEXT_INST: AtomicU32 = AtomicU32::new(1_000_000);

/// A scratch path in a directory of its own; the directory is removed when this is dropped.
pub struct Tmp(PathBuf);

pub fn tmp(name: &str) -> Tmp {
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let d = std::env::temp_dir().join(format!("stave-a64-debug-{}-{seq}", std::process::id()));
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

pub fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok_and(|o| o.status.success())
}

/// Common types.
#[derive(Clone, Copy, Debug)]
pub struct T {
    pub u8: TypeId,
    pub u16: TypeId,
    pub u32: TypeId,
    pub u64: TypeId,
    pub usize: TypeId,
    pub i8: TypeId,
    pub i16: TypeId,
    pub i32: TypeId,
    pub i64: TypeId,
    pub isize: TypeId,
    pub f32: TypeId,
    pub f64: TypeId,
    pub bool: TypeId,
    pub char: TypeId,
    pub unit: TypeId,
    pub str: TypeId,
    pub str_ref: TypeId,
    pub pu8: TypeId,
}

impl T {
    pub fn new(m: &mut Module) -> T {
        let u8 = m.types.u8();
        let str = m.types.str();
        T {
            u8,
            u16: m.types.u16(),
            u32: m.types.u32(),
            u64: m.types.u64(),
            usize: m.types.usize(),
            i8: m.types.i8(),
            i16: m.types.i16(),
            i32: m.types.i32(),
            i64: m.types.i64(),
            isize: m.types.isize(),
            f32: m.types.f32(),
            f64: m.types.f64(),
            bool: m.types.bool(),
            char: m.types.char(),
            unit: m.types.unit(),
            str,
            str_ref: m.types.ref_(false, str),
            pu8: m.types.ptr(false, u8),
        }
    }

    pub fn ints(&self) -> [TypeId; 10] {
        [self.u8, self.u16, self.u32, self.u64, self.usize, self.i8, self.i16, self.i32, self.i64, self.isize]
    }
}

/// A program under construction: module, types, runtime helpers, `app.main`.
pub struct Prog {
    pub m: Module,
    pub t: T,
    pub main: FuncId,
    pub out_u64: FuncId,
    pub out_f64: FuncId,
    pub out_f32: FuncId,
    pub out_str: FuncId,
    pub entry: FuncId,
    pub trap: FuncId,
    pub sys1: IntrinsicId,
    pub sys3: IntrinsicId,
}

pub fn inst() -> InstanceId {
    InstanceId(vec![NEXT_INST.fetch_add(1, Ordering::Relaxed)])
}

impl Prog {
    pub fn new() -> Prog {
        let mut m = Module::new();
        let t = T::new(&mut m);
        let sys1 = m.intrinsic("syscall1");
        let sys3 = m.intrinsic("syscall3");
        let out_u64 = m.declare(inst(), "rt.out_u64", vec![t.u64], None);
        let out_f64 = m.declare(inst(), "rt.out_f64", vec![t.f64], None);
        let out_f32 = m.declare(inst(), "rt.out_f32", vec![t.f32], None);
        let out_str = m.declare(inst(), "rt.out_str", vec![t.str_ref], None);
        let sp = m.types.ptr(false, t.usize);
        let entry = m.declare(inst(), "runtime.start.entry", vec![sp], None);
        let trap = m.declare(inst(), "runtime.panic.trap", vec![t.u32, t.u32], None);
        let main = m.declare(inst(), "app.main", vec![], None);
        m.funcs[entry.index()].known = Some(stave_ir::Known::StartEntry);
        m.funcs[trap.index()].known = Some(stave_ir::Known::Trap);
        let mut p = Prog { m, t, main, out_u64, out_f64, out_f32, out_str, entry, trap, sys1, sys3 };
        p.define_runtime();
        p
    }

    /// `write(fd, p, n)` with `p` a pointer value.
    fn write(&self, b: &mut FuncBuilder, fd: i128, p: Operand, n: Operand) {
        let t = self.t;
        let pa = b.cast(CastKind::Exact, t.usize, p);
        let nr = b.int(64);
        let fd = b.int(fd);
        b.intrinsic(self.sys3, Some(t.isize), &[nr, fd, pa.into(), n]);
    }

    fn define_runtime(&mut self) {
        let t = self.t;
        // out_u64
        let pu64 = self.m.types.ref_(true, t.u64);
        let mut b = FuncBuilder::new(&[t.u64]);
        let x = b.param(0).unwrap();
        let s = b.slot(pu64);
        b.store(s, x);
        let n = b.int(8);
        self.write(&mut b, 1, s.into(), n);
        b.ret(None);
        self.m.define(self.out_u64, b.finish().unwrap());
        // out_f64 / out_f32 with NaN canonicalized (NaN bits differ across hosts)
        for (f, ft, size) in [(self.out_f64, t.f64, 8), (self.out_f32, t.f32, 4)] {
            let pf = self.m.types.ref_(true, ft);
            let mut b = FuncBuilder::new(&[ft]);
            let x = b.param(0).unwrap();
            let c = b.cmp(Op::Ne, t.bool, x, x);
            let q = if size == 8 {
                b.konst(stave_ir::Const::F64(0x7ff8_0000_0000_0000))
            } else {
                b.konst(stave_ir::Const::F32(0x7fc0_0000))
            };
            let y = b.select(ft, c, q, x);
            let s = b.slot(pf);
            b.store(s, y);
            let n = b.int(size);
            self.write(&mut b, 1, s.into(), n);
            b.ret(None);
            self.m.define(f, b.finish().unwrap());
        }
        // out_str
        let mut b = FuncBuilder::new(&[t.str_ref]);
        let x = b.param(0).unwrap();
        let p = b.unary(Op::SlicePtr, t.pu8, x);
        let n = b.unary(Op::SliceLen, t.usize, x);
        self.write(&mut b, 1, p.into(), n.into());
        b.ret(None);
        self.m.define(self.out_str, b.finish().unwrap());
        // trap
        let pair = self.m.types.array(t.u32, 2);
        let ppair = self.m.types.ref_(true, pair);
        let mut b = FuncBuilder::new(&[t.u32, t.u32]);
        let (k, s) = (b.param(0).unwrap(), b.param(1).unwrap());
        let slot = b.slot(ppair);
        let v = b.op(Op::Make, pair, &[k.into(), s.into()]);
        b.store(slot, v);
        let n = b.int(8);
        self.write(&mut b, 2, slot.into(), n);
        let nr = b.int(94);
        let code = b.int(101);
        b.intrinsic(self.sys1, Some(t.isize), &[nr, code]);
        b.unreachable();
        self.m.define(self.trap, b.finish().unwrap());
        // entry
        let sp = self.m.funcs[self.entry.index()].params.clone();
        let mut b = FuncBuilder::new(&sp);
        b.call(self.main, None, &[]);
        let nr = b.int(94);
        let code = b.int(0);
        b.intrinsic(self.sys1, Some(t.isize), &[nr, code]);
        b.unreachable();
        self.m.define(self.entry, b.finish().unwrap());
    }

    /// Declares and defines a function.
    pub fn func(
        &mut self,
        path: &str,
        params: &[TypeId],
        ret: Option<TypeId>,
        f: impl FnOnce(&mut FuncBuilder, &Prog),
    ) -> FuncId {
        let id = self.m.declare(inst(), path, params.to_vec(), ret);
        let mut b = FuncBuilder::new(params);
        f(&mut b, self);
        self.m.define(id, b.finish().expect("builder"));
        id
    }

    /// [`Prog::func`] for compiler-known item `k`, stamped as the front end does.
    pub fn func_known(
        &mut self,
        k: stave_ir::Known,
        params: &[TypeId],
        ret: Option<TypeId>,
        f: impl FnOnce(&mut FuncBuilder, &Prog),
    ) -> FuncId {
        let id = self.func(&k.path(), params, ret, f);
        self.m.funcs[id.index()].known = Some(k);
        id
    }

    pub fn declare(&mut self, path: &str, params: &[TypeId], ret: Option<TypeId>) -> FuncId {
        self.m.declare(inst(), path, params.to_vec(), ret)
    }

    pub fn define(&mut self, f: FuncId, params: &[TypeId], body: impl FnOnce(&mut FuncBuilder, &Prog)) {
        let mut b = FuncBuilder::new(params);
        body(&mut b, self);
        self.m.define(f, b.finish().expect("builder"));
    }

    pub fn set_main(&mut self, body: impl FnOnce(&mut FuncBuilder, &Prog)) {
        let mut b = FuncBuilder::new(&[]);
        body(&mut b, self);
        b.ret(None);
        self.m.define(self.main, b.finish().expect("builder"));
    }

    /// Emits code printing scalar `v` of type `ty` (ints as u64 words, floats as raw bits).
    pub fn out(&self, b: &mut FuncBuilder, v: Value, ty: TypeId) {
        let t = self.t;
        if ty == t.f64 {
            b.call(self.out_f64, None, &[v.into()]);
        } else if ty == t.f32 {
            b.call(self.out_f32, None, &[v.into()]);
        } else if ty == t.u64 {
            b.call(self.out_u64, None, &[v.into()]);
        } else {
            let w = b.cast(CastKind::Wrap, t.u64, v);
            b.call(self.out_u64, None, &[w.into()]);
        }
    }

    /// Prints an int constant (a marker in the output).
    pub fn mark(&self, b: &mut FuncBuilder, k: u64) {
        let c = b.int(k as i128);
        b.call(self.out_u64, None, &[c]);
    }
}

/// What a run produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Run {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub code: i32,
}

pub fn trap_bytes(kind: PanicKind, site: u32) -> Vec<u8> {
    let k = PanicKind::ALL.iter().position(|x| *x == kind).unwrap() as u32;
    let mut v = k.to_le_bytes().to_vec();
    v.extend_from_slice(&site.to_le_bytes());
    v
}

/// `m` with a placeholder site row for every site it names: the tests use sites as labels the
/// trap reports, and a module that verifies names only rows of its site table.
pub fn with_sites(m: &Module) -> Module {
    let mut m = m.clone();
    let named = m.funcs.iter().flat_map(|f| {
        let insts = f.body.as_ref().map_or(&[][..], |b| &b.insts.site[..]);
        insts.iter().copied().chain([f.name_site])
    });
    let need = named.filter(|s| *s != stave_ir::Site::NONE).map(|s| s.0 as usize + 1).max().unwrap_or(0);
    while m.sites.len() < need {
        let line = m.sites.len() as u32 + 1;
        m.sites.push(stave_ir::SiteInfo { file: "t.stv".into(), line, col: 1, func: "t".into() });
    }
    m
}

pub fn interp(m: &Module, entry: FuncId) -> Run {
    let m = &with_sites(m);
    let mut it = Interp::new(m, BufferHost::new(SyscallAbi::Aarch64), Config::default())
        .unwrap_or_else(|e| panic!("verify: {e:?}"));
    let sp = it.process_stack(&[b"prog"], &[]).unwrap();
    let o = it.run(entry, vec![Val::S(sp)]).unwrap_or_else(|e| panic!("interpreter: {e:?}"));
    let h = it.into_host();
    match o {
        Outcome::Exit(c) => Run { stdout: h.stdout, stderr: h.stderr, code: c },
        Outcome::Trap(t) => {
            let mut e = h.stderr;
            e.extend(trap_bytes(t.kind, t.site.0));
            Run { stdout: h.stdout, stderr: e, code: 101 }
        }
        Outcome::Return(_) => panic!("entry returned"),
        Outcome::MemError(r) => panic!("{}", r.render(m)),
        Outcome::BorrowError(r) => panic!("{}", r.render(m)),
    }
}

/// Runs `cmd`, retrying while the kernel reports ETXTBSY.
pub fn output_retry(cmd: &mut Command) -> std::process::Output {
    for _ in 0..100 {
        match cmd.output() {
            Err(e) if e.raw_os_error() == Some(26) => {
                std::thread::sleep(std::time::Duration::from_millis(10))
            }
            r => return r.unwrap(),
        }
    }
    cmd.output().unwrap()
}

/// Whether AArch64 binaries run here; fails instead when `STAVE_REQUIRE_LEGS` lists `aarch64`.
pub fn can_run() -> bool {
    legs::gate(
        "aarch64-linux",
        cfg!(all(target_os = "linux", target_arch = "aarch64")) || have("qemu-aarch64"),
    )
}

/// Compiles (debug or release code generation), links, and runs natively; `None` without qemu.
pub fn native(m: &Module, name: &str, release: bool) -> Option<Run> {
    let cfg = BuildConfig { release, ..BuildConfig::new(None) };
    let img = build_image(m, &cfg).unwrap_or_else(|e| panic!("codegen: {e}"));
    if !can_run() {
        eprintln!("SKIP running AArch64 ELF: no qemu-aarch64");
        return None;
    }
    let p = tmp(name);
    img.write_file(&p).unwrap();
    let mut cmd = if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        Command::new(&p)
    } else {
        let mut c = Command::new("qemu-aarch64");
        c.arg(&p);
        c
    };
    let out = output_retry(&mut cmd);
    if std::env::var_os("STAVE_A64_KEEP").is_some() {
        eprintln!("kept {}", p.display());
        std::mem::forget(p);
    }
    Some(Run { stdout: out.stdout, stderr: out.stderr, code: out.status.code().unwrap_or(-1) })
}

/// The differential check, for debug and for release code generation. Returns the interpreter's
/// run.
pub fn check(p: &Prog, name: &str) -> Run {
    let want = interp(&p.m, p.entry);
    for release in [false, true] {
        check_one(p, name, release, &want);
    }
    want
}

fn check_one(p: &Prog, name: &str, release: bool, want: &Run) {
    let name = &format!("{name}{}", if release { "-release" } else { "" });
    if let Some(got) = native(&p.m, name, release)
        && got != *want
    {
        let words = |b: &[u8]| -> Vec<u64> {
            b.chunks(8).map(|c| c.iter().rev().fold(0u64, |a, x| (a << 8) | u64::from(*x))).collect()
        };
        let (gw, ww) = (words(&got.stdout), words(&want.stdout));
        let first = gw.iter().zip(&ww).position(|(a, b)| a != b);
        panic!(
            "{name}: native != interpreter\n first differing stdout word: {first:?} (native {:?}, interp {:?})\n \
                 stdout words: native {} interp {}\n stderr: native {:?} interp {:?}\n code: native {} interp {}",
            first.map(|i| gw[i]),
            first.map(|i| ww[i]),
            gw.len(),
            ww.len(),
            got.stderr,
            want.stderr,
            got.code,
            want.code
        );
    }
}
