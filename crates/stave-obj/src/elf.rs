//! Layout, relocation, and ELF64 serialization (static `ET_EXEC`, no interpreter, no libc).
//!
//! File layout (offsets contiguous; each segment's `p_vaddr ≡ p_offset (mod page)`, every segment
//! on its own virtual pages):
//!
//! ```text
//! PT_LOAD R    ELF header, program headers, .rodata
//! PT_LOAD R X  .text = [jump table] [function slots in symbol order] [reserve]
//! PT_LOAD R W  .data, then .bss (memory only)          (only when either is non-empty)
//! PT_GNU_STACK RW (non-executable stack)
//! non-alloc    .debug_abbrev .debug_info .debug_line .symtab .strtab .shstrtab, section headers
//! ```

use crate::reloc::{RelocFail, apply_reloc};
use crate::{
    FuncCode, ImageBuilder, Machine, ObjError, Options, Reloc, RelocKind, SourceFile, SymDef, SymbolId,
    dwarf, stub,
};

const EHDR: u64 = 64;
const PHDR: u64 = 56;
const SHDR: u64 = 64;
const PT_LOAD: u32 = 1;
const PT_GNU_STACK: u32 = 0x6474_e551;
const PF_X: u32 = 1;
const PF_W: u32 = 2;
const PF_R: u32 = 4;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_NOBITS: u32 = 8;
const SHF_WRITE: u64 = 1;
const SHF_ALLOC: u64 = 2;
const SHF_EXECINSTR: u64 = 4;
/// Limit on any single size or the sum of all sizes, so address arithmetic cannot overflow.
const MAX_SIZE: u64 = 1 << 32;

/// Placement of an allocated section.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SectionInfo {
    pub offset: u64,
    pub addr: u64,
    pub size: u64,
    pub align: u64,
}

/// A function's slot in `.text`: `len` bytes of code at `addr`, room for `cap`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FuncSlot {
    pub sym: SymbolId,
    pub addr: u64,
    pub len: u32,
    pub cap: u32,
}

/// Stub `i` (at `addr + i * stub_size`) jumps to `funcs[i]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JumpTable {
    pub addr: u64,
    pub stub_size: u32,
    pub funcs: Vec<SymbolId>,
}

/// What [`ObjectImage::patch_func`] rewrote.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Patch {
    /// The function moved to the reserve (its jump-table stub was retargeted).
    pub moved: bool,
    /// File byte ranges `(offset, len)` rewritten in the loaded part of the image.
    pub ranges: Vec<(u64, u64)>,
    /// Everything from this file offset to the end was rewritten (debug sections, symbols,
    /// section headers); the file length may have changed.
    pub tail_offset: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    Func(u32),
    Rodata,
    Data,
    Bss,
}

/// A linked executable image.
#[derive(Clone, Debug)]
pub struct ObjectImage {
    machine: Machine,
    opts: Options,
    names: Vec<String>,
    class: Vec<Class>,
    addr: Vec<u64>,
    size: Vec<u64>,
    slots: Vec<FuncSlot>,
    code: Vec<FuncCode>,
    jt: Option<JumpTable>,
    text: SectionInfo,
    rodata: Option<SectionInfo>,
    data: Option<SectionInfo>,
    bss: Option<SectionInfo>,
    rw_seg: Option<(u64, u64, u64, u64)>,
    reserve: (u64, u64),
    files: Vec<SourceFile>,
    entry: SymbolId,
    bytes: Vec<u8>,
    loaded_end: u64,
    nphdr: u64,
    /// Every [`RelocKind::Abs64Body`] field (the object holding it, the relocation), rewritten
    /// by [`ObjectImage::patch_func`] when the target function moves.
    body_refs: Vec<(SymbolId, Reloc)>,
}

fn align_up(v: u64, a: u64) -> u64 {
    (v + a - 1) & !(a - 1)
}

fn site(names: &[String], id: SymbolId) -> String {
    names.get(id.0 as usize).cloned().unwrap_or_default()
}

fn slot_cap(opts: &Options, len: u64) -> u64 {
    let extra = match opts.slack {
        Some(s) => u64::from(s.min).max(len.saturating_mul(u64::from(s.percent)) / 100),
        None => 0,
    };
    align_up(len.saturating_add(extra).clamp(1, MAX_SIZE), u64::from(opts.func_align))
}

fn check_opts(m: Machine, o: &Options) -> Result<(), ObjError> {
    if !o.page_size.is_power_of_two() || o.page_size < 0x1000 || o.page_size > 1 << 30 {
        return Err(ObjError::BadOption("page_size must be a power of two in 4 KiB..=1 GiB"));
    }
    if !o.base_addr.is_multiple_of(o.page_size) || o.base_addr > 1 << 46 {
        return Err(ObjError::BadOption("base_addr must be page aligned and below 2^46"));
    }
    let min_align = if m == Machine::Aarch64 { 4 } else { 1 };
    if !o.func_align.is_power_of_two() || o.func_align < min_align || u64::from(o.func_align) > o.page_size {
        return Err(ObjError::BadOption("func_align must be a power of two (≥ 4 on AArch64)"));
    }
    if u64::from(o.text_reserve) > MAX_SIZE {
        return Err(ObjError::TooLarge);
    }
    Ok(())
}

/// Checks the relocation fields of one function or data object.
fn check_code(
    m: Machine,
    names: &[String],
    id: SymbolId,
    len: usize,
    relocs: &[Reloc],
) -> Result<(), ObjError> {
    if len as u64 > MAX_SIZE {
        return Err(ObjError::TooLarge);
    }
    for r in relocs {
        if !r.kind.fits(m) {
            return Err(ObjError::RelocWrongMachine { site: site(names, id), kind: r.kind });
        }
        if r.sym.0 as usize >= names.len() {
            return Err(ObjError::UnknownSymbol(r.sym));
        }
        if r.offset as usize + r.kind.width() > len {
            return Err(ObjError::RelocOutOfBounds { site: site(names, id), offset: r.offset });
        }
    }
    Ok(())
}

fn check_lines(names: &[String], nfiles: usize, id: SymbolId, f: &mut FuncCode) -> Result<(), ObjError> {
    let max_file = nfiles.max(1);
    for r in &f.lines {
        if r.file.0 as usize >= max_file {
            return Err(ObjError::UnknownFile(r.file));
        }
        if r.offset as usize >= f.code.len() {
            return Err(ObjError::LineOutOfRange { site: site(names, id), offset: r.offset });
        }
    }
    f.lines.sort_by_key(|r| r.offset);
    Ok(())
}

pub(crate) fn link(b: ImageBuilder) -> Result<ObjectImage, ObjError> {
    let ImageBuilder { machine, opts, names, defs, files, entry, .. } = b;
    check_opts(machine, &opts)?;
    let entry = entry.ok_or(ObjError::NoEntry)?;
    let n = names.len();
    if n > (u32::MAX as usize) - 16 {
        return Err(ObjError::TooLarge);
    }
    let mut class = Vec::with_capacity(n);
    let mut size = vec![0u64; n];
    let mut code = Vec::new();
    let mut slots = Vec::new();
    let mut rodata_objs = Vec::new();
    let mut data_objs = Vec::new();
    let mut bss_objs = Vec::new();
    let mut total: u64 = 0;
    for (i, d) in defs.into_iter().enumerate() {
        let id = SymbolId(i as u32);
        match d {
            SymDef::Undef => return Err(ObjError::Undefined(names[i].clone())),
            SymDef::Func(mut f) => {
                check_code(machine, &names, id, f.code.len(), &f.relocs)?;
                check_lines(&names, files.len(), id, &mut f)?;
                size[i] = f.code.len() as u64;
                class.push(Class::Func(slots.len() as u32));
                slots.push(FuncSlot { sym: id, addr: 0, len: f.code.len() as u32, cap: 0 });
                code.push(f);
            }
            SymDef::Rodata(o) => {
                check_code(machine, &names, id, o.bytes.len(), &o.relocs)?;
                size[i] = o.bytes.len() as u64;
                class.push(Class::Rodata);
                rodata_objs.push((id, o));
            }
            SymDef::Data(o) => {
                check_code(machine, &names, id, o.bytes.len(), &o.relocs)?;
                size[i] = o.bytes.len() as u64;
                class.push(Class::Data);
                data_objs.push((id, o));
            }
            SymDef::Bss { size: s, align } => {
                if s > MAX_SIZE {
                    return Err(ObjError::TooLarge);
                }
                size[i] = s;
                class.push(Class::Bss);
                bss_objs.push((id, s, align));
            }
        }
        total += size[i] + 64;
        if total > MAX_SIZE {
            return Err(ObjError::TooLarge);
        }
    }
    if !matches!(class[entry.0 as usize], Class::Func(_)) {
        return Err(ObjError::NotAFunction(names[entry.0 as usize].clone()));
    }

    let page = opts.page_size;
    let falign = u64::from(opts.func_align);
    let has_rw = !data_objs.is_empty() || !bss_objs.is_empty();
    let nphdr = 3 + u64::from(has_rw);
    let mut addr = vec![0u64; n];

    // Segment 0: headers + .rodata, identity mapped at base.
    let mut off = EHDR + PHDR * nphdr;
    let rodata = if rodata_objs.is_empty() {
        None
    } else {
        let al = rodata_objs.iter().map(|(_, o)| u64::from(o.align)).max().unwrap_or(1);
        off = align_up(opts.base_addr + off, al) - opts.base_addr;
        let start = off;
        for (id, o) in &rodata_objs {
            // align the address (the base may be less aligned than the object)
            off = align_up(opts.base_addr + off, u64::from(o.align)) - opts.base_addr;
            addr[id.0 as usize] = opts.base_addr + off;
            off += o.bytes.len() as u64;
        }
        Some(SectionInfo { offset: start, addr: opts.base_addr + start, size: off - start, align: al })
    };
    let seg0_end = off;

    // Segment 1: .text.
    let text_align = falign.max(16);
    off = align_up(off, text_align);
    let text_off = off;
    let text_addr = align_up(opts.base_addr + seg0_end, page) + text_off % page;
    let mut cur = text_addr;
    let jt = if opts.jump_table {
        let ss = stub::jump_stub_size(machine);
        let funcs: Vec<SymbolId> = slots.iter().map(|s| s.sym).collect();
        let t = JumpTable { addr: cur, stub_size: ss, funcs };
        cur += u64::from(ss) * t.funcs.len() as u64;
        Some(t)
    } else {
        None
    };
    for s in slots.iter_mut() {
        cur = align_up(cur, falign);
        s.addr = cur;
        let cap = slot_cap(&opts, u64::from(s.len));
        s.cap = u32::try_from(cap).map_err(|_| ObjError::TooLarge)?;
        addr[s.sym.0 as usize] = cur;
        cur += cap;
        if cur - text_addr > MAX_SIZE {
            return Err(ObjError::TooLarge);
        }
    }
    let reserve_start = align_up(cur, falign);
    let reserve = if opts.jump_table {
        (reserve_start, reserve_start + u64::from(opts.text_reserve))
    } else {
        (cur, cur)
    };
    let text_end = reserve.1.max(cur);
    let text =
        SectionInfo { offset: text_off, addr: text_addr, size: text_end - text_addr, align: text_align };
    off = text_off + text.size;

    // Segment 2: .data then .bss.
    let mut data = None;
    let mut bss = None;
    let mut rw_seg = None;
    if has_rw {
        let al = data_objs
            .iter()
            .map(|(_, o)| u64::from(o.align))
            .chain(bss_objs.iter().map(|b| u64::from(b.2)))
            .max()
            .unwrap_or(1)
            .min(page);
        off = align_up(off, al);
        let seg_off = off;
        let seg_addr = align_up(text_addr + text.size, page) + seg_off % page;
        let mut a = seg_addr;
        if !data_objs.is_empty() {
            let dal = data_objs.iter().map(|(_, o)| u64::from(o.align)).max().unwrap_or(1);
            for (id, o) in &data_objs {
                a = align_up(a, u64::from(o.align));
                addr[id.0 as usize] = a;
                a += o.bytes.len() as u64;
            }
            data = Some(SectionInfo { offset: seg_off, addr: seg_addr, size: a - seg_addr, align: dal });
        }
        let filesz = a - seg_addr;
        if !bss_objs.is_empty() {
            let bal = bss_objs.iter().map(|b| u64::from(b.2)).max().unwrap_or(1);
            a = align_up(a, bal);
            let bstart = a;
            for (id, s, al) in &bss_objs {
                a = align_up(a, u64::from(*al));
                addr[id.0 as usize] = a;
                a += s;
            }
            bss = Some(SectionInfo { offset: seg_off + filesz, addr: bstart, size: a - bstart, align: bal });
        }
        rw_seg = Some((seg_off, seg_addr, filesz, a - seg_addr));
        off = seg_off + filesz;
    }
    let loaded_end = off;

    let mut img = ObjectImage {
        machine,
        opts,
        names,
        class,
        addr,
        size,
        slots,
        code: Vec::new(),
        jt,
        text,
        rodata,
        data,
        bss,
        rw_seg,
        reserve,
        files,
        entry,
        bytes: Vec::new(),
        loaded_end,
        nphdr,
        body_refs: Vec::new(),
    };
    let objs = rodata_objs.iter().chain(data_objs.iter()).map(|(id, o)| (*id, &o.relocs));
    for (id, rs) in objs.chain(img.slots.iter().zip(&code).map(|(s, f)| (s.sym, &f.relocs))) {
        img.body_refs.extend(rs.iter().filter(|r| r.kind == RelocKind::Abs64Body).map(|r| (id, *r)));
    }

    // Contents of the loaded part.
    let mut bytes = vec![0u8; loaded_end as usize];
    bytes[text_off as usize..(text_off + text.size) as usize].fill(machine.trap_fill());
    for (id, o) in rodata_objs.iter().chain(data_objs.iter()) {
        let fo = img.file_off(img.addr[id.0 as usize]) as usize;
        bytes[fo..fo + o.bytes.len()].copy_from_slice(&o.bytes);
        img.relocate(&mut bytes, *id, &o.relocs)?;
    }
    for (s, f) in img.slots.iter().zip(&code) {
        let fo = img.file_off(s.addr) as usize;
        bytes[fo..fo + f.code.len()].copy_from_slice(&f.code);
    }
    for (s, f) in img.slots.iter().zip(&code) {
        img.relocate(&mut bytes, s.sym, &f.relocs)?;
    }
    img.code = code;
    if let Some(t) = &img.jt {
        for i in 0..t.funcs.len() {
            img.write_stub(&mut bytes, i)?;
        }
    }
    img.bytes = bytes;
    img.write_tail();
    Ok(img)
}

impl ObjectImage {
    pub fn machine(&self) -> Machine {
        self.machine
    }

    pub fn options(&self) -> &Options {
        &self.opts
    }

    /// The ELF file.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Writes the file with mode 0755: to a new temporary file in the same directory, then
    /// renamed over `path`. So `path` is replaced, never written through (a symlink at `path` is
    /// replaced, not followed; a running old binary keeps its inode), and a failed write leaves
    /// `path` as it was.
    pub fn write_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        static SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let dir = match path.parent() {
            Some(d) if !d.as_os_str().is_empty() => d,
            _ => std::path::Path::new("."),
        };
        let name = path.file_name().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "output path has no file name")
        })?;
        let (tmp, mut f) = loop {
            let k = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut tname = std::ffi::OsString::from(".");
            tname.push(name);
            tname.push(format!(".tmp.{}.{k}", std::process::id()));
            let tmp = dir.join(tname);
            let mut o = std::fs::OpenOptions::new();
            o.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                o.mode(0o755);
            }
            match o.open(&tmp) {
                Ok(f) => break (tmp, f),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        };
        let r = (|| {
            f.write_all(&self.bytes)?;
            // the creation mode is filtered by the umask; the image is always 0755
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                f.set_permissions(std::fs::Permissions::from_mode(0o755))?;
            }
            drop(f);
            std::fs::rename(&tmp, path)
        })();
        if r.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        r
    }

    pub fn entry_addr(&self) -> u64 {
        self.addr[self.entry.0 as usize]
    }

    pub fn lookup(&self, name: &str) -> Option<SymbolId> {
        self.names.iter().position(|n| n == name).map(|i| SymbolId(i as u32))
    }

    pub fn symbol_name(&self, id: SymbolId) -> Option<&str> {
        self.names.get(id.0 as usize).map(String::as_str)
    }

    /// The symbol's own address (for a function: its slot, not its stub).
    pub fn symbol_addr(&self, id: SymbolId) -> Option<u64> {
        self.addr.get(id.0 as usize).copied()
    }

    pub fn symbol_size(&self, id: SymbolId) -> Option<u64> {
        self.size.get(id.0 as usize).copied()
    }

    /// The address relocations resolve to: the jump-table stub for functions when the table is
    /// enabled, otherwise the symbol address.
    pub fn ref_addr(&self, id: SymbolId) -> Option<u64> {
        match (self.class.get(id.0 as usize)?, &self.jt) {
            (Class::Func(i), Some(t)) => Some(t.addr + u64::from(*i) * u64::from(t.stub_size)),
            _ => self.symbol_addr(id),
        }
    }

    pub fn slots(&self) -> &[FuncSlot] {
        &self.slots
    }

    pub fn slot(&self, id: SymbolId) -> Option<&FuncSlot> {
        match self.class.get(id.0 as usize)? {
            Class::Func(i) => self.slots.get(*i as usize),
            _ => None,
        }
    }

    pub fn func_code(&self, id: SymbolId) -> Option<&FuncCode> {
        match self.class.get(id.0 as usize)? {
            Class::Func(i) => self.code.get(*i as usize),
            _ => None,
        }
    }

    pub fn jump_table(&self) -> Option<&JumpTable> {
        self.jt.as_ref()
    }

    pub fn text(&self) -> SectionInfo {
        self.text
    }

    pub fn rodata(&self) -> Option<SectionInfo> {
        self.rodata
    }

    pub fn data(&self) -> Option<SectionInfo> {
        self.data
    }

    pub fn bss(&self) -> Option<SectionInfo> {
        self.bss
    }

    /// Unused bytes left in the reserve.
    pub fn reserve_left(&self) -> u64 {
        self.reserve.1 - self.reserve.0
    }

    /// File offset of a loaded address (text, rodata, data; not bss).
    fn file_off(&self, a: u64) -> u64 {
        if a >= self.text.addr && a < self.text.addr + self.text.size.max(1) {
            return self.text.offset + (a - self.text.addr);
        }
        if let Some((o, va, fs, _)) = self.rw_seg
            && a >= va
            && a <= va + fs
        {
            return o + (a - va);
        }
        a.saturating_sub(self.opts.base_addr)
    }

    fn relocate(&self, bytes: &mut [u8], at: SymbolId, relocs: &[Reloc]) -> Result<(), ObjError> {
        let base = self.addr[at.0 as usize];
        let fo = self.file_off(base);
        for r in relocs {
            let s = match r.kind {
                RelocKind::Abs64Body => self.symbol_addr(r.sym),
                _ => self.ref_addr(r.sym),
            };
            let s = s.ok_or(ObjError::UnknownSymbol(r.sym))?;
            let p = base + u64::from(r.offset);
            apply_reloc(r.kind, bytes, (fo + u64::from(r.offset)) as usize, p, s, r.addend).map_err(|e| {
                match e {
                    RelocFail::OutOfBounds => {
                        ObjError::RelocOutOfBounds { site: site(&self.names, at), offset: r.offset }
                    }
                    RelocFail::Overflow => ObjError::RelocOverflow {
                        site: site(&self.names, at),
                        target: site(&self.names, r.sym),
                        kind: r.kind,
                    },
                }
            })?;
        }
        Ok(())
    }

    fn write_stub(&self, bytes: &mut [u8], i: usize) -> Result<(), ObjError> {
        let Some(t) = &self.jt else { return Ok(()) };
        let (code, at, kind, addend) = stub::jump_stub(self.machine);
        let sa = t.addr + i as u64 * u64::from(t.stub_size);
        let fo = self.file_off(sa) as usize;
        bytes[fo..fo + code.len()].copy_from_slice(code);
        let target = self.slots[i].addr;
        apply_reloc(kind, bytes, fo + at as usize, sa + u64::from(at), target, addend).map_err(|_| {
            ObjError::RelocOverflow {
                site: String::from("jump table"),
                target: site(&self.names, t.funcs[i]),
                kind,
            }
        })
    }

    /// Replaces the code of function `id` in place: within its slot, or (with the jump table) at
    /// the start of the reserve, retargeting its stub. Other functions are untouched.
    pub fn patch_func(&mut self, id: SymbolId, mut code: FuncCode) -> Result<Patch, ObjError> {
        let Some(&Class::Func(si)) = self.class.get(id.0 as usize) else {
            return Err(ObjError::NotAFunction(self.symbol_name(id).unwrap_or("").to_owned()));
        };
        let si = si as usize;
        check_code(self.machine, &self.names, id, code.code.len(), &code.relocs)?;
        check_lines(&self.names, self.files.len(), id, &mut code)?;
        let len = code.code.len() as u64;
        let old = self.slots[si];
        let mut bytes = std::mem::take(&mut self.bytes);
        let fill = self.machine.trap_fill();
        let mut ranges = Vec::new();
        let moved = len > u64::from(old.cap);
        let (new_addr, new_cap) = if !moved {
            (old.addr, u64::from(old.cap))
        } else {
            let a = align_up(self.reserve.0, u64::from(self.opts.func_align));
            let want = slot_cap(&self.opts, len);
            let cap = want.min(self.reserve.1.saturating_sub(a));
            if self.jt.is_none() || cap < len {
                self.bytes = bytes;
                return Err(ObjError::SlotOverflow {
                    name: site(&self.names, id),
                    need: len,
                    cap: u64::from(old.cap),
                });
            }
            (a, cap)
        };
        let ofo = self.file_off(old.addr) as usize;
        bytes[ofo..ofo + old.cap as usize].fill(fill);
        ranges.push((ofo as u64, u64::from(old.cap)));
        self.slots[si].addr = new_addr;
        self.slots[si].cap = new_cap as u32;
        self.slots[si].len = len as u32;
        self.addr[id.0 as usize] = new_addr;
        self.size[id.0 as usize] = len;
        let nfo = self.file_off(new_addr) as usize;
        bytes[nfo..nfo + code.code.len()].copy_from_slice(&code.code);
        if let Err(e) = self.relocate(&mut bytes, id, &code.relocs) {
            // Restore the previous state.
            self.slots[si] = old;
            self.addr[id.0 as usize] = old.addr;
            self.size[id.0 as usize] = u64::from(old.len);
            bytes[nfo..nfo + code.code.len()].fill(fill);
            let prev = &self.code[si];
            bytes[ofo..ofo + prev.code.len()].copy_from_slice(&prev.code);
            let relocs = prev.relocs.clone();
            let r = self.relocate(&mut bytes, id, &relocs);
            self.bytes = bytes;
            r?;
            return Err(e);
        }
        // Body references: drop the old code's, add the new code's.
        self.body_refs.retain(|(at, _)| *at != id);
        self.body_refs
            .extend(code.relocs.iter().filter(|r| r.kind == RelocKind::Abs64Body).map(|r| (id, *r)));
        if moved {
            ranges.push((nfo as u64, new_cap));
            self.reserve.0 = new_addr + new_cap;
            self.write_stub(&mut bytes, si)?;
            // Fields holding the moved function's body address (PC tables).
            let refs: Vec<(SymbolId, Reloc)> =
                self.body_refs.iter().filter(|(at, r)| r.sym == id && *at != id).copied().collect();
            for (at, r) in refs {
                self.relocate(&mut bytes, at, &[r])?;
                let fo = self.file_off(self.addr[at.0 as usize] + u64::from(r.offset));
                ranges.push((fo, 8));
            }
            if let Some(t) = &self.jt {
                let sa = t.addr + si as u64 * u64::from(t.stub_size);
                ranges.push((self.file_off(sa), u64::from(t.stub_size)));
            }
        }
        self.code[si] = code;
        self.bytes = bytes;
        self.write_tail();
        ranges.push((0, EHDR));
        Ok(Patch { moved, ranges, tail_offset: self.loaded_end })
    }

    /// Rebuilds the non-loaded tail (debug sections, symbols, section headers) and the headers.
    fn write_tail(&mut self) {
        let mut bytes = std::mem::take(&mut self.bytes);
        bytes.truncate(self.loaded_end as usize);
        let mut shstr = vec![0u8];
        let mut shdrs: Vec<[u64; 10]> = vec![[0; 10]];
        let mut name = |s: &str| {
            let o = shstr.len() as u64;
            shstr.extend_from_slice(s.as_bytes());
            shstr.push(0);
            o
        };
        // [name, type, flags, addr, offset, size, link, info, align, entsize]
        let mut sec_idx = [0u64; 3];
        if let Some(r) = self.rodata {
            sec_idx[0] = shdrs.len() as u64;
            shdrs.push([
                name(".rodata"),
                u64::from(SHT_PROGBITS),
                SHF_ALLOC,
                r.addr,
                r.offset,
                r.size,
                0,
                0,
                r.align,
                0,
            ]);
        }
        let text_idx = shdrs.len() as u64;
        let t = self.text;
        shdrs.push([
            name(".text"),
            u64::from(SHT_PROGBITS),
            SHF_ALLOC | SHF_EXECINSTR,
            t.addr,
            t.offset,
            t.size,
            0,
            0,
            t.align,
            0,
        ]);
        if let Some(d) = self.data {
            sec_idx[1] = shdrs.len() as u64;
            shdrs.push([
                name(".data"),
                u64::from(SHT_PROGBITS),
                SHF_ALLOC | SHF_WRITE,
                d.addr,
                d.offset,
                d.size,
                0,
                0,
                d.align,
                0,
            ]);
        }
        if let Some(b) = self.bss {
            sec_idx[2] = shdrs.len() as u64;
            shdrs.push([
                name(".bss"),
                u64::from(SHT_NOBITS),
                SHF_ALLOC | SHF_WRITE,
                b.addr,
                b.offset,
                b.size,
                0,
                0,
                b.align,
                0,
            ]);
        }
        if self.opts.debug_info {
            let seqs: Vec<dwarf::LineSeq<'_>> = self
                .slots
                .iter()
                .zip(&self.code)
                .filter(|(_, c)| !c.lines.is_empty())
                .map(|(s, c)| dwarf::LineSeq { addr: s.addr, len: u64::from(s.len), rows: &c.lines })
                .collect();
            let secs = [
                (".debug_abbrev", dwarf::debug_abbrev()),
                (".debug_info", dwarf::debug_info(&self.opts.cu_name, &self.opts.comp_dir, t.addr, t.size)),
                (
                    ".debug_line",
                    dwarf::debug_line(&self.opts.comp_dir, &self.opts.cu_name, &self.files, &seqs),
                ),
            ];
            for (n, s) in secs {
                let o = bytes.len() as u64;
                bytes.extend_from_slice(&s);
                shdrs.push([name(n), u64::from(SHT_PROGBITS), 0, 0, o, s.len() as u64, 0, 0, 1, 0]);
            }
        }
        if self.opts.symtab {
            let mut strtab = vec![0u8];
            let mut syms = vec![0u8; 24];
            for (i, nm) in self.names.iter().enumerate() {
                let so = strtab.len() as u32;
                strtab.extend(nm.bytes().filter(|&b| b != 0));
                strtab.push(0);
                let (ty, shndx) = match self.class[i] {
                    Class::Func(_) => (2u8, text_idx),
                    Class::Rodata => (1, sec_idx[0]),
                    Class::Data => (1, sec_idx[1]),
                    Class::Bss => (1, sec_idx[2]),
                };
                syms.extend_from_slice(&so.to_le_bytes());
                syms.push((1 << 4) | ty); // STB_GLOBAL
                syms.push(0); // STV_DEFAULT
                syms.extend_from_slice(&(shndx as u16).to_le_bytes());
                syms.extend_from_slice(&self.addr[i].to_le_bytes());
                syms.extend_from_slice(&self.size[i].to_le_bytes());
            }
            let o = align_up(bytes.len() as u64, 8);
            bytes.resize(o as usize, 0);
            bytes.extend_from_slice(&syms);
            let symtab_idx = shdrs.len() as u64;
            shdrs.push([
                name(".symtab"),
                u64::from(SHT_SYMTAB),
                0,
                0,
                o,
                syms.len() as u64,
                symtab_idx + 1,
                1,
                8,
                24,
            ]);
            let o = bytes.len() as u64;
            bytes.extend_from_slice(&strtab);
            shdrs.push([name(".strtab"), u64::from(SHT_STRTAB), 0, 0, o, strtab.len() as u64, 0, 0, 1, 0]);
        }
        let shstr_name = name(".shstrtab");
        let o = bytes.len() as u64;
        let shstr_len = shstr.len() as u64;
        bytes.extend_from_slice(&shstr);
        shdrs.push([shstr_name, u64::from(SHT_STRTAB), 0, 0, o, shstr_len, 0, 0, 1, 0]);
        let shoff = align_up(bytes.len() as u64, 8);
        bytes.resize(shoff as usize, 0);
        for h in &shdrs {
            bytes.extend_from_slice(&(h[0] as u32).to_le_bytes());
            bytes.extend_from_slice(&(h[1] as u32).to_le_bytes());
            for v in &h[2..6] {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            bytes.extend_from_slice(&(h[6] as u32).to_le_bytes());
            bytes.extend_from_slice(&(h[7] as u32).to_le_bytes());
            bytes.extend_from_slice(&h[8].to_le_bytes());
            bytes.extend_from_slice(&h[9].to_le_bytes());
        }
        self.write_headers(&mut bytes, shoff, shdrs.len() as u16);
        self.bytes = bytes;
    }

    fn write_headers(&self, bytes: &mut [u8], shoff: u64, shnum: u16) {
        let mut h = Vec::with_capacity((EHDR + PHDR * self.nphdr) as usize);
        h.extend_from_slice(&[0x7f, b'E', b'L', b'F', 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        h.extend_from_slice(&2u16.to_le_bytes()); // ET_EXEC
        h.extend_from_slice(&self.machine.e_machine().to_le_bytes());
        h.extend_from_slice(&1u32.to_le_bytes());
        h.extend_from_slice(&self.entry_addr().to_le_bytes());
        h.extend_from_slice(&EHDR.to_le_bytes());
        h.extend_from_slice(&shoff.to_le_bytes());
        h.extend_from_slice(&0u32.to_le_bytes());
        h.extend_from_slice(&(EHDR as u16).to_le_bytes());
        h.extend_from_slice(&(PHDR as u16).to_le_bytes());
        h.extend_from_slice(&(self.nphdr as u16).to_le_bytes());
        h.extend_from_slice(&(SHDR as u16).to_le_bytes());
        h.extend_from_slice(&shnum.to_le_bytes());
        h.extend_from_slice(&(shnum - 1).to_le_bytes());
        let page = self.opts.page_size;
        let seg0 = self.rodata.map_or(EHDR + PHDR * self.nphdr, |r| r.offset + r.size);
        let mut ph = |ty: u32, fl: u32, off: u64, va: u64, fs: u64, ms: u64, al: u64| {
            h.extend_from_slice(&ty.to_le_bytes());
            h.extend_from_slice(&fl.to_le_bytes());
            for v in [off, va, va, fs, ms, al] {
                h.extend_from_slice(&v.to_le_bytes());
            }
        };
        ph(PT_LOAD, PF_R, 0, self.opts.base_addr, seg0, seg0, page);
        let t = self.text;
        ph(PT_LOAD, PF_R | PF_X, t.offset, t.addr, t.size, t.size, page);
        if let Some((o, va, fs, ms)) = self.rw_seg {
            ph(PT_LOAD, PF_R | PF_W, o, va, fs, ms, page);
        }
        ph(PT_GNU_STACK, PF_R | PF_W, 0, 0, 0, 0, 16);
        bytes[..h.len()].copy_from_slice(&h);
    }
}
