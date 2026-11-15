//! Module-level context, symbols, compiler-emitted stubs and tables (contracts/intrinsics.md),
//! the `fmod` helper, and the all-in-one [`build_image`].

use std::collections::HashMap;

use stave_ir::layout::Layouts;
use stave_ir::{FuncId, Module, Operand, TypeId};
use stave_obj::{
    DataObj, FileId, FuncCode, ImageBuilder, Machine, ObjectImage, Options, Reloc, RelocKind, SymbolId,
};

use super::analysis::Analysis;
use super::emit::Gen;
use super::ty::{Sc, Ty, classify};
use super::{Error, ErrorKind, R, bad, unsupported};
use crate::enc::{AddSub, Addr, Asm, Cond, Ext, Fp, FpBin, Logic, Reg, Shift, VReg, Width};

/// Fixed runtime entry points (contracts/intrinsics.md "Paths fixed by the compiler").
pub const RT_ENTRY: &str = "runtime.start.entry";
pub const RT_TRAP: &str = "runtime.panic.trap";
pub const RT_ON_SIGNAL: &str = "runtime.panic.on_signal";
pub const RT_THREAD_ENTRY: &str = "runtime.thread.entry";

pub const SYM_MAIN: &str = "__stave_main";
pub const SYM_SIGNAL: &str = "__stave_signal";
pub const SYM_THREAD: &str = "__stave_thread_start";
pub const SYM_SITE_TABLE: &str = "__stave_site_table";
pub const SYM_PC_TABLE: &str = "__stave_pc_table";
pub const SYM_FMOD: &str = "__stave_fmod";
pub const SYM_STRINGS: &str = "__stave_strings";
const SYM_SITE_STRINGS: &str = "__stave_site_strings";

/// Image symbols the backend references, created by [`prepare`].
#[derive(Clone, Debug)]
pub struct Symbols {
    /// Per [`FuncId`]: the function's symbol (`None`: declared-only and never referenced).
    pub funcs: Vec<Option<SymbolId>>,
    /// `runtime.panic.trap`, if the module defines it.
    pub trap: Option<SymbolId>,
    pub fmod: SymbolId,
    /// All string literals, concatenated; `str_off[s]` is the offset of string `s`.
    pub strings: SymbolId,
    pub str_off: Vec<u32>,
    /// Per `static mut`.
    pub statics: Vec<SymbolId>,
    pub main_stub: Option<SymbolId>,
    pub signal_stub: Option<SymbolId>,
    pub thread_stub: Option<SymbolId>,
    pub site_table: SymbolId,
    pub pc_table: SymbolId,
    /// Per site: the line-table position.
    pub lines: Vec<Option<(FileId, u32, u32)>>,
}

/// What an intrinsic lowers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Intr {
    Syscall(u8),
    Ccall(u8),
    CallOnStack,
    MainAddr,
    SignalAddr,
    ThreadAddr,
    SiteTable,
    PcTable,
    StaticAddr,
    ThreadClone,
    /// `atomic_<op>_u32/u64`: op 0 load, 1 xchg, 2 add, 3 cas; wide = u64.
    Atomic {
        op: u8,
        wide: bool,
    },
    X86Only,
    Unknown,
}

fn intr_kind(name: &str) -> Intr {
    let num =
        |p: &str, max: u8| name.strip_prefix(p).and_then(|d| d.parse::<u8>().ok()).filter(|&k| k <= max);
    if let Some(k) = num("syscall", 6) {
        return Intr::Syscall(k);
    }
    if let Some(k) = num("ccall", 12) {
        return Intr::Ccall(k);
    }
    match name {
        "call_on_stack" => Intr::CallOnStack,
        "main_addr" => Intr::MainAddr,
        "signal_handler_addr" => Intr::SignalAddr,
        "thread_start_addr" => Intr::ThreadAddr,
        "site_table" => Intr::SiteTable,
        "pc_table" => Intr::PcTable,
        "static_addr" => Intr::StaticAddr,
        "thread_clone" => Intr::ThreadClone,
        "cpuid" | "xgetbv" => Intr::X86Only,
        _ if name.starts_with("atomic_") => {
            let rest = &name["atomic_".len()..];
            let (op, wide) = match rest.rsplit_once('_') {
                Some((op, "u32")) => (op, false),
                Some((op, "u64")) => (op, true),
                _ => return Intr::Unknown,
            };
            let op = match op {
                "load" => 0,
                "xchg" => 1,
                "add" => 2,
                "cas" => 3,
                _ => return Intr::Unknown,
            };
            Intr::Atomic { op, wide }
        }
        _ => Intr::Unknown,
    }
}

/// Per-module code generation context (shared by all threads; [`Codegen::compile`] is `&self`).
pub struct Codegen<'a> {
    pub(crate) m: &'a Module,
    pub(crate) syms: &'a Symbols,
    pub(crate) layouts: Layouts,
    pub(crate) tys: Vec<Ty>,
    pub(crate) intr: Vec<Intr>,
    pub(crate) release: bool,
}

/// Per-thread reusable buffers.
#[derive(Default)]
pub struct FuncScratch {
    pub(crate) an: Analysis,
    pub(crate) vreg: Vec<u8>,
}

impl<'a> Codegen<'a> {
    pub fn new(m: &'a Module, syms: &'a Symbols) -> Codegen<'a> {
        let layouts = Layouts::compute(&m.types);
        let tys = (0..m.types.len()).map(|i| classify(&m.types, &layouts, TypeId(i as u32))).collect();
        let intr = m.intrinsics.iter().map(|n| intr_kind(n)).collect();
        Codegen { m, syms, layouts, tys, intr, release: false }
    }

    /// Release code generation: values used across blocks own a callee-saved register for their
    /// whole life (`analysis`, stave-opt `regalloc`). Off: they live in their home slots.
    pub fn release(mut self, on: bool) -> Codegen<'a> {
        self.release = on;
        self
    }

    pub(crate) fn ty(&self, t: TypeId) -> Ty {
        self.tys.get(t.index()).copied().unwrap_or(Ty::Bad("type id out of range"))
    }

    pub(crate) fn func_sym(&self, f: FuncId) -> R<SymbolId> {
        self.syms
            .funcs
            .get(f.index())
            .copied()
            .flatten()
            .ok_or_else(|| bad(format!("fn #{} has no symbol", f.0)))
    }

    /// Compiles one function (it must have a body).
    pub fn compile(&self, f: FuncId, s: &mut FuncScratch) -> Result<FuncCode, Error> {
        self.compile_inner(f, s).map_err(|mut e| {
            e.func.get_or_insert(f);
            e
        })
    }

    fn compile_inner(&self, fid: FuncId, s: &mut FuncScratch) -> R<FuncCode> {
        let func = self.m.func(fid).ok_or_else(|| bad("unknown function"))?;
        let body = func.body.as_ref().ok_or_else(|| bad("function has no body"))?;
        s.an.run(self, func, body)?;
        let mut vreg = std::mem::take(&mut s.vreg);
        let mut r = Gen::new(self, func, body, &s.an, &mut vreg, false).run();
        // A function beyond the ±1 MiB reach of b.cond / cbz is redone with long branches.
        if let Err(Error { kind: super::ErrorKind::Enc(crate::enc::EncError::BranchRange(_)), .. }) = r {
            r = Gen::new(self, func, body, &s.an, &mut vreg, true).run();
        }
        s.vreg = vreg;
        r
    }
}

/// Configuration of [`build_image`].
#[derive(Clone, Debug)]
pub struct BuildConfig {
    pub opts: Options,
    /// The program's `main` (for `__stave_main`); `None`: no `__stave_main` stub.
    pub main: Option<FuncId>,
    /// [`Codegen::release`].
    pub release: bool,
}

impl BuildConfig {
    pub fn new(main: Option<FuncId>) -> BuildConfig {
        BuildConfig { opts: Options::release(Machine::Aarch64), main, release: false }
    }
}

fn intern<'s>(s: &'s str, blob: &mut Vec<u8>, map: &mut HashMap<&'s str, u32>) -> u32 {
    *map.entry(s).or_insert_with(|| {
        let o = blob.len() as u32;
        blob.extend_from_slice(s.as_bytes());
        o
    })
}

/// Declares every symbol of the module in `b` (functions in `stave_ir::link::text_order`, which
/// is the text order; names by `stave_ir::link::func_symbols`), and defines everything except the
/// function bodies: string literals, statics, the site and PC tables, the stubs (`_start`,
/// `__stave_main`, `__stave_signal`, `__stave_thread_start`) and `__stave_fmod`. Stubs are emitted when
/// their target function exists (`main` for `__stave_main`).
pub fn prepare(b: &mut ImageBuilder, m: &Module, main: Option<FuncId>) -> Result<Symbols, Error> {
    // Functions with a body, in text order (stave_ir::link).
    let order = stave_ir::link::text_order(m);
    // Calls / fn values naming a function without a body cannot be linked statically.
    for &f in &order {
        let Some(body) = &m.funcs[f.index()].body else { continue };
        for o in &body.operands {
            if let Operand::Func(g) = o
                && m.func(*g).is_none_or(|x| x.body.is_none())
            {
                let path = m.func(*g).map_or("?", |x| x.path.as_str());
                return Err(Error {
                    func: Some(f),
                    kind: ErrorKind::Unsupported(format!(
                        "reference to fn `{path}` without a body (extern imports need a C linker)"
                    )),
                });
            }
        }
    }
    let names = stave_ir::link::func_symbols(m, &order);
    let mut funcs = vec![None; m.funcs.len()];
    for f in &order {
        if let Some(name) = &names[f.index()] {
            if b.lookup(name).is_some() {
                return Err(stave_obj::ObjError::Redefined(name.clone()).into());
            }
            funcs[f.index()] = Some(b.declare(name));
        }
    }
    let text = &order[..];
    let find = |p: &str| m.find_func(p).filter(|f| m.func(*f).is_some_and(|f| f.body.is_some()));
    let sym_of = |f: Option<FuncId>| f.and_then(|f| funcs.get(f.index()).copied().flatten());
    let trap = sym_of(find(RT_TRAP));
    // Strings.
    let mut blob = Vec::new();
    let mut str_off = Vec::with_capacity(m.strings.len());
    for s in &m.strings {
        str_off.push(blob.len() as u32);
        blob.extend_from_slice(s);
    }
    let strings = b.rodata(SYM_STRINGS, DataObj::new(blob, 1))?;
    // Statics.
    let layouts = Layouts::compute(&m.types);
    let mut statics = Vec::with_capacity(m.statics.len());
    for (i, st) in m.statics.iter().enumerate() {
        let l = layouts.get_layout(st.ty).map_err(|e| unsupported(format!("static {}: {e}", st.path)))?;
        let name = stave_ir::link::static_symbol(m, i, |p| b.lookup(p).is_some());
        let size = usize::try_from(l.size).map_err(|_| unsupported("static size"))?;
        let align = u32::try_from(l.align.max(1)).map_err(|_| unsupported("static alignment"))?;
        // At least one byte (contracts/checked-ir.md): distinct statics have distinct addresses.
        let id = if st.init.iter().all(|&x| x == 0) {
            b.bss(&name, l.size.max(1), align)?
        } else {
            let mut bytes = st.init.clone();
            bytes.resize(size.max(1), 0);
            b.data(&name, DataObj::new(bytes, align))?
        };
        statics.push(id);
    }
    // Lines and the site table.
    let mut lines = Vec::with_capacity(m.sites.len());
    let mut sblob = Vec::new();
    let mut soff: HashMap<&str, u32> = HashMap::new();
    let mut rows = Vec::with_capacity(8 + 32 * m.sites.len());
    rows.extend_from_slice(&(m.sites.len() as u64).to_le_bytes());
    let mut relocs = Vec::new();
    let site_strings = b.declare(SYM_SITE_STRINGS);
    for site in &m.sites {
        let (dir, name) = stave_ir::link::split_file(&site.file);
        lines.push(Some((b.add_file(dir, name), site.line, site.col)));
        let fo = intern(&site.file, &mut sblob, &mut soff);
        let no = intern(&site.func, &mut sblob, &mut soff);
        let at = rows.len() as u32;
        relocs.push(Reloc { offset: at, kind: RelocKind::Abs64, sym: site_strings, addend: i64::from(fo) });
        relocs.push(Reloc {
            offset: at + 16,
            kind: RelocKind::Abs64,
            sym: site_strings,
            addend: i64::from(no),
        });
        rows.extend_from_slice(&0u64.to_le_bytes());
        rows.extend_from_slice(&(site.file.len() as u32).to_le_bytes());
        rows.extend_from_slice(&site.line.to_le_bytes());
        rows.extend_from_slice(&0u64.to_le_bytes());
        rows.extend_from_slice(&(site.func.len() as u32).to_le_bytes());
        rows.extend_from_slice(&site.col.to_le_bytes());
    }
    b.define_rodata(site_strings, DataObj::new(sblob, 1))?;
    let site_table = b.rodata(SYM_SITE_TABLE, DataObj { bytes: rows, align: 8, relocs })?;
    // PC table: one row per defined function, in text order.
    let mut rows = Vec::new();
    let mut relocs = Vec::new();
    let mut cnt = 0u64;
    rows.extend_from_slice(&0u64.to_le_bytes());
    for &fid in text {
        let f = &m.funcs[fid.index()];
        let Some(sym) = funcs[fid.index()] else { continue };
        // The body address, never the jump-table stub (contracts/intrinsics.md `pc_lo`).
        relocs.push(Reloc { offset: rows.len() as u32, kind: RelocKind::Abs64Body, sym, addend: 0 });
        rows.extend_from_slice(&0u64.to_le_bytes());
        rows.extend_from_slice(&f.name_site.0.to_le_bytes());
        rows.extend_from_slice(&0u32.to_le_bytes());
        cnt += 1;
    }
    rows[..8].copy_from_slice(&cnt.to_le_bytes());
    let pc_table = b.rodata(SYM_PC_TABLE, DataObj { bytes: rows, align: 8, relocs })?;
    // Stubs.
    let main_stub = match main {
        Some(mf) => {
            let msym = sym_of(Some(mf)).ok_or_else(|| bad("main has no body"))?;
            let mfn = m.func(mf).ok_or_else(|| bad("unknown main"))?;
            let ret = mfn.ret.map(|t| classify(&m.types, &layouts, t));
            if !mfn.params.is_empty() {
                return Err(unsupported("main with parameters"));
            }
            let code = main_stub(msym, ret)?;
            Some(b.func(SYM_MAIN, code)?)
        }
        None => None,
    };
    let signal_stub = match sym_of(find(RT_ON_SIGNAL)) {
        Some(t) => {
            let mut a = Asm::new();
            a.mov(Width::W, Reg::xn(0), Reg::xn(0))?;
            a.b_sym(t);
            Some(b.func(SYM_SIGNAL, a.finish()?)?)
        }
        None => None,
    };
    let thread_stub = match sym_of(find(RT_THREAD_ENTRY)) {
        Some(t) => {
            let mut a = Asm::new();
            a.b_sym(t);
            Some(b.func(SYM_THREAD, a.finish()?)?)
        }
        None => None,
    };
    let fmod = b.func(SYM_FMOD, fmod_code()?)?;
    if let Some(e) = sym_of(find(RT_ENTRY)) {
        b.add_start_stub(e)?;
    }
    Ok(Symbols {
        funcs,
        trap,
        fmod,
        strings,
        str_off,
        statics,
        main_stub,
        signal_stub,
        thread_stub,
        site_table,
        pc_table,
        lines,
    })
}

/// `__stave_main(arg) -> usize`: calls `main`, returns its status (`i32` sign-extended; 0 for unit).
fn main_stub(main: SymbolId, ret: Option<Ty>) -> R<FuncCode> {
    let mut a = Asm::new();
    let (fp, lr, sp) = (Reg::FP, Reg::LR, Reg::SP);
    a.stp(Width::X, fp, lr, Addr::Pre(sp, -16))?;
    a.mov(Width::X, fp, sp)?;
    a.bl_sym(main);
    match ret {
        None => a.mov_imm(Width::X, Reg::xn(0), 0)?,
        Some(Ty::Mem { size: 0, .. }) => a.mov_imm(Width::X, Reg::xn(0), 0)?,
        Some(Ty::Sc(Sc::Int { bits: 32, signed: true })) => {
            a.sxt(Width::X, Reg::xn(0), Reg::xn(0), Ext::W)?
        }
        Some(Ty::Sc(s)) if !s.is_fp() => {}
        _ => return Err(unsupported("main must return an integer or unit")),
    }
    a.ldp(Width::X, fp, lr, Addr::Post(sp, 16))?;
    a.ret();
    Ok(a.finish()?)
}

/// `__stave_fmod(x: f64, y: f64) -> f64` in `d0`, exact (musl's integer algorithm). Clobbers
/// `x0–x10`, `v2`. NaN results come from `(x*y)/(x*y)`.
fn fmod_code() -> R<FuncCode> {
    let mut a = Asm::new();
    let x = Reg::xn;
    let (ux, uy, ex, ey, sx) = (x(0), x(1), x(2), x(3), x(4));
    let (t5, t6, t7, t9, t10) = (x(5), x(6), x(7), x(9), x(10));
    let (d0, d1, d2) = (VReg::vn(0), VReg::vn(1), VReg::vn(2));
    let w64 = Width::X;
    let l_nan = a.new_label();
    let l_zero = a.new_label();
    let l_ret = a.new_label();
    let l_main = a.new_label();
    a.fmov_to_gpr(Fp::D, ux, d0)?;
    a.fmov_to_gpr(Fp::D, uy, d1)?;
    a.ubfm(w64, ex, ux, 52, 62)?;
    a.ubfm(w64, ey, uy, 52, 62)?;
    a.shift_imm(Shift::Lsr, w64, sx, ux, 63)?;
    a.shift_imm(Shift::Lsl, w64, t5, uy, 1)?;
    a.cbz(w64, t5, l_nan)?;
    a.mov_imm(w64, t6, 0xffe0_0000_0000_0000)?;
    a.cmp(w64, t5, t6)?;
    a.b_cond(Cond::Hi, l_nan)?;
    a.cmp_imm(w64, ex, 0x7ff)?;
    a.b_cond(Cond::Eq, l_nan)?;
    a.shift_imm(Shift::Lsl, w64, t7, ux, 1)?;
    a.cmp(w64, t7, t5)?;
    a.b_cond(Cond::Hi, l_main)?;
    a.b_cond(Cond::Eq, l_zero)?;
    a.b(l_ret)?;
    a.bind(l_main)?;
    // Normalize x and y.
    for (u, e) in [(ux, ex), (uy, ey)] {
        let l_norm = a.new_label();
        let l_ok = a.new_label();
        let l_loop = a.new_label();
        let l_done = a.new_label();
        a.cbnz(w64, e, l_norm)?;
        a.shift_imm(Shift::Lsl, w64, t9, u, 12)?;
        a.bind(l_loop)?;
        a.tbnz(t9, 63, l_done)?;
        a.addsub_imm(AddSub::Sub, w64, e, e, 1)?;
        a.shift_imm(Shift::Lsl, w64, t9, t9, 1)?;
        a.b(l_loop)?;
        a.bind(l_done)?;
        a.neg(w64, t10, e)?;
        a.addsub_imm(AddSub::Add, w64, t10, t10, 1)?;
        a.shift_reg(Shift::Lsl, w64, u, u, t10)?;
        a.b(l_ok)?;
        a.bind(l_norm)?;
        a.logic_imm(Logic::And, w64, u, u, 0x000f_ffff_ffff_ffff)?;
        a.logic_imm(Logic::Orr, w64, u, u, 0x0010_0000_0000_0000)?;
        a.bind(l_ok)?;
    }
    // x mod y.
    let l_loop = a.new_label();
    let l_after = a.new_label();
    let l_skip = a.new_label();
    a.bind(l_loop)?;
    a.cmp(w64, ex, ey)?;
    a.b_cond(Cond::Le, l_after)?;
    a.addsub_reg(AddSub::Subs, w64, t9, ux, uy, Shift::Lsl, 0)?;
    a.b_cond(Cond::Mi, l_skip)?;
    a.cbz(w64, t9, l_zero)?;
    a.mov(w64, ux, t9)?;
    a.bind(l_skip)?;
    a.shift_imm(Shift::Lsl, w64, ux, ux, 1)?;
    a.addsub_imm(AddSub::Sub, w64, ex, ex, 1)?;
    a.b(l_loop)?;
    a.bind(l_after)?;
    let l_2 = a.new_label();
    a.addsub_reg(AddSub::Subs, w64, t9, ux, uy, Shift::Lsl, 0)?;
    a.b_cond(Cond::Mi, l_2)?;
    a.cbz(w64, t9, l_zero)?;
    a.mov(w64, ux, t9)?;
    a.bind(l_2)?;
    let l_nl = a.new_label();
    let l_3 = a.new_label();
    a.bind(l_nl)?;
    a.shift_imm(Shift::Lsr, w64, t9, ux, 52)?;
    a.cbnz(w64, t9, l_3)?;
    a.shift_imm(Shift::Lsl, w64, ux, ux, 1)?;
    a.addsub_imm(AddSub::Sub, w64, ex, ex, 1)?;
    a.b(l_nl)?;
    a.bind(l_3)?;
    // Scale the result.
    let l_sub = a.new_label();
    let l_sign = a.new_label();
    a.cmp_imm(w64, ex, 0)?;
    a.b_cond(Cond::Le, l_sub)?;
    a.mov_imm(w64, t9, 1 << 52)?;
    a.sub(w64, ux, ux, t9)?;
    a.logic_reg(Logic::Orr, w64, ux, ux, ex, Shift::Lsl, 52)?;
    a.b(l_sign)?;
    a.bind(l_sub)?;
    a.neg(w64, t9, ex)?;
    a.addsub_imm(AddSub::Add, w64, t9, t9, 1)?;
    a.shift_reg(Shift::Lsr, w64, ux, ux, t9)?;
    a.bind(l_sign)?;
    a.logic_reg(Logic::Orr, w64, ux, ux, sx, Shift::Lsl, 63)?;
    a.fmov_from_gpr(Fp::D, d0, ux)?;
    a.ret();
    a.bind(l_nan)?;
    a.fp_bin(FpBin::Mul, Fp::D, d2, d0, d1);
    a.fp_bin(FpBin::Div, Fp::D, d0, d2, d2);
    a.ret();
    a.bind(l_zero)?;
    a.fmov_from_gpr(Fp::D, d2, Reg::ZR)?;
    a.fp_bin(FpBin::Mul, Fp::D, d0, d2, d0);
    a.bind(l_ret)?;
    a.ret();
    Ok(a.finish()?)
}

/// Builds a whole static AArch64 executable: [`prepare`], every function body, link.
pub fn build_image(m: &Module, cfg: &BuildConfig) -> Result<ObjectImage, Error> {
    let mut b = ImageBuilder::new(Machine::Aarch64, cfg.opts.clone());
    let syms = prepare(&mut b, m, cfg.main)?;
    let cg = Codegen::new(m, &syms).release(cfg.release);
    let mut s = FuncScratch::default();
    for fid in stave_ir::link::text_order(m) {
        let code = cg.compile(fid, &mut s)?;
        let sym = cg.func_sym(fid)?;
        b.define_func(sym, code)?;
    }
    Ok(b.link()?)
}
