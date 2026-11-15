//! Compiler-emitted symbols (contracts/intrinsics.md): `__stave_main`, `__stave_signal`,
//! `__stave_thread_start`, `__stave_site_table`, `__stave_pc_table`, and the trap fallback
//! `__stave_panic_fallback` used when the module has no `runtime.panic.trap`.

use stave_ir::{FuncId, Module, PanicKind, TypeKey};
use stave_obj::{DataObj, ImageBuilder, Reloc, RelocKind, SourceFile, SymbolId};

use super::{Res, bad, unsup};
use crate::enc::*;

pub(crate) const MAIN: &str = "__stave_main";
pub(crate) const SIGNAL: &str = "__stave_signal";
pub(crate) const THREAD: &str = "__stave_thread_start";
pub(crate) const SITE_TABLE: &str = "__stave_site_table";
pub(crate) const PC_TABLE: &str = "__stave_pc_table";
pub(crate) const FALLBACK: &str = "__stave_panic_fallback";

/// `__stave_main(arg: usize) -> usize` (C ABI): calls `main`, returns its `i32` status
/// sign-extended (Stave returns integers normalized), 0 for a unit `main`.
pub(crate) fn stave_main(
    m: &Module,
    b: &mut ImageBuilder,
    f: FuncId,
    sym: Option<SymbolId>,
) -> Res<SymbolId> {
    let func = m.func(f).ok_or_else(|| bad("main out of range"))?;
    let sym = sym.ok_or_else(|| bad("main has no symbol"))?;
    if !func.params.is_empty() {
        return Err(unsup("`main` with parameters"));
    }
    let unit = match func.ret {
        None => true,
        Some(t) => match m.types.key(t) {
            Some(TypeKey::Tuple(ts)) if ts.is_empty() => true,
            Some(TypeKey::Int { .. }) => false,
            _ => return Err(unsup("`main` must return unit or an integer")),
        },
    };
    let mut a = Asm::new();
    a.push(Gpr::RBP);
    a.mov_rr(Size::Q, Gpr::RBP, Gpr::RSP);
    a.call_sym(sym);
    if unit {
        a.alu_rr(AluOp::Xor, Size::D, Gpr::RAX, Gpr::RAX);
    }
    a.pop(Gpr::RBP);
    a.ret();
    Ok(b.func(MAIN, a.finish()?)?)
}

/// A C-ABI stub that tail-jumps to a Stave function with the same register arguments (`zext_edi`:
/// first zero-extend the C `int` in `edi`).
pub(crate) fn tail_stub(b: &mut ImageBuilder, name: &str, target: SymbolId, zext_edi: bool) -> Res<SymbolId> {
    let mut a = Asm::new();
    if zext_edi {
        a.mov_rr(Size::D, Gpr::RDI, Gpr::RDI);
    }
    a.jmp_sym(target);
    Ok(b.func(name, a.finish()?)?)
}

/// `__stave_panic_fallback(kind: u32, site: u32) -> never`: writes `panic: <Kind>\n` to fd 2 and
/// exits with status 101.
pub(crate) fn panic_fallback(b: &mut ImageBuilder) -> Res<SymbolId> {
    let mut table = Vec::with_capacity(32 * 8);
    let names: Vec<&str> = PanicKind::ALL.iter().map(|k| k.name()).chain(["Unknown"]).collect();
    for n in &names {
        let mut row = format!("panic: {n}\n").into_bytes();
        row.resize(31, 0);
        row.push((n.len() + 8) as u8);
        table.extend_from_slice(&row);
    }
    let t = b.rodata("__stave_panic_names", DataObj::new(table, 32))?;
    let mut a = Asm::new();
    let known = a.new_label();
    a.alu_ri(AluOp::Cmp, Size::D, Gpr::RDI, PanicKind::ALL.len() as i64)?;
    a.jcc(Cond::B, known)?;
    a.mov_ri(Size::D, Gpr::RDI, PanicKind::ALL.len() as i64)?;
    a.bind(known)?;
    a.mov_rr(Size::D, Gpr::RAX, Gpr::RDI);
    a.shift_ri(ShiftOp::Shl, Size::D, Gpr::RAX, 5)?;
    a.lea(Size::Q, Gpr::RSI, Mem::rip_sym(t, 0))?;
    a.alu_rr(AluOp::Add, Size::Q, Gpr::RSI, Gpr::RAX);
    a.movzx(Size::D, Gpr::RDX, Size::B, Mem::base(Gpr::RSI, 31))?;
    a.mov_ri(Size::D, Gpr::RDI, 2)?;
    a.mov_ri(Size::D, Gpr::RAX, 1)?;
    a.syscall();
    a.mov_ri(Size::D, Gpr::RDI, 101)?;
    a.mov_ri(Size::D, Gpr::RAX, 231)?;
    a.syscall();
    a.ud2();
    Ok(b.func(FALLBACK, a.finish()?)?)
}

fn file_path(dir: &str, name: &str) -> String {
    if dir.is_empty() || dir == "." { name.to_owned() } else { format!("{dir}/{name}") }
}

/// One site-table row before encoding: file index, line, column, enclosing fn path.
pub(crate) struct Row {
    pub file: u32,
    pub line: u32,
    pub col: u32,
    pub func: String,
}

/// `__stave_site_table`: `u64 count`, then 32-byte rows `{file u64, file_len u32, line u32,
/// fn u64, fn_len u32, col u32}`.
pub(crate) fn site_table(files: &[SourceFile], rows: &[Row], b: &mut ImageBuilder) -> Res<SymbolId> {
    let mut file_syms = Vec::with_capacity(files.len());
    for (i, f) in files.iter().enumerate() {
        let p = file_path(&f.dir, &f.name);
        let n = p.len() as u32;
        file_syms.push((b.rodata(&format!("__stave_file.{i}"), DataObj::new(p.into_bytes(), 1))?, n));
    }
    let mut fn_syms: rustc_hash::FxHashMap<&str, (SymbolId, u32)> = rustc_hash::FxHashMap::default();
    let mut bytes = Vec::with_capacity(8 + 32 * rows.len());
    let mut relocs = Vec::new();
    bytes.extend_from_slice(&(rows.len() as u64).to_le_bytes());
    for r in rows {
        let at = bytes.len() as u32;
        let (fsym, flen) = match file_syms.get(r.file as usize) {
            Some(&(s, n)) => (Some(s), n),
            None => (None, 0),
        };
        let fnent = if r.func.is_empty() {
            None
        } else {
            match fn_syms.get(r.func.as_str()) {
                Some(e) => Some(*e),
                None => {
                    let k = fn_syms.len();
                    let s = b.rodata(
                        &format!("__stave_fnpath.{k}"),
                        DataObj::new(r.func.clone().into_bytes(), 1),
                    )?;
                    let e = (s, r.func.len() as u32);
                    fn_syms.insert(r.func.as_str(), e);
                    Some(e)
                }
            }
        };
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&flen.to_le_bytes());
        bytes.extend_from_slice(&r.line.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&fnent.map_or(0, |e| e.1).to_le_bytes());
        bytes.extend_from_slice(&r.col.to_le_bytes());
        if let Some(s) = fsym {
            relocs.push(Reloc { offset: at, kind: RelocKind::Abs64, sym: s, addend: 0 });
        }
        if let Some((s, _)) = fnent {
            relocs.push(Reloc { offset: at + 16, kind: RelocKind::Abs64, sym: s, addend: 0 });
        }
    }
    let mut obj = DataObj::new(bytes, 8);
    obj.relocs = relocs;
    Ok(b.rodata(SITE_TABLE, obj)?)
}

/// `static mut` globals (`Module::statics`): `.bss` when zero-initialized, else `.data`.
pub(crate) fn statics(
    m: &Module,
    layouts: &stave_ir::layout::Layouts,
    b: &mut ImageBuilder,
) -> Res<Vec<SymbolId>> {
    let mut out = Vec::with_capacity(m.statics.len());
    for (i, st) in m.statics.iter().enumerate() {
        let l = layouts.get_layout(st.ty).map_err(|e| unsup(format!("static `{}`: {e}", st.path)))?;
        let align = u32::try_from(l.align.max(1)).map_err(|_| unsup("static alignment"))?;
        let name = stave_ir::link::static_symbol(m, i, |p| b.lookup(p).is_some());
        // At least one byte (contracts/checked-ir.md): distinct statics have distinct addresses.
        let sym = if st.init.iter().all(|x| *x == 0) {
            b.bss(&name, l.size.max(1), align)?
        } else {
            let mut bytes = st.init.clone();
            bytes.resize(usize::try_from(l.size).map_err(|_| unsup("static size"))?.max(1), 0);
            b.data(&name, DataObj::new(bytes, align))?
        };
        out.push(sym);
    }
    Ok(out)
}

/// `__stave_pc_table`: `u64 count`, then 16-byte rows `{pc_lo u64, site u32, pad u32}`; each
/// `pc_lo` is an `Abs64Body` relocation (the function's body, never its jump-table stub).
pub(crate) fn pc_table(rows: &[(SymbolId, u32)]) -> DataObj {
    let mut bytes = Vec::with_capacity(8 + 16 * rows.len());
    let mut relocs = Vec::with_capacity(rows.len());
    bytes.extend_from_slice(&(rows.len() as u64).to_le_bytes());
    for (s, site) in rows {
        let at = bytes.len() as u32;
        relocs.push(Reloc { offset: at, kind: RelocKind::Abs64Body, sym: *s, addend: 0 });
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&site.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
    }
    let mut obj = DataObj::new(bytes, 8);
    obj.relocs = relocs;
    obj
}
