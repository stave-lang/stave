//! Built-in static ELF64 writer.
//!
//! The backends hand over one [`FuncCode`] per function (machine code, relocations, line rows) plus
//! read-only data, writable data, and bss. [`ImageBuilder::link`] lays them out in `PT_LOAD`
//! segments, resolves every relocation, and produces an [`ObjectImage`]: a static `ET_EXEC` with no
//! `PT_INTERP`, no `PT_DYNAMIC`, no libc, entry `_start`, a symbol table, and DWARF v5 line tables.
//!
//! Debug builds give each function a padded slot and may route every function reference through a
//! jump table, so [`ObjectImage::patch_func`] can rewrite one function in place. Output bytes depend only on the inputs and the [`Options`].

pub mod dwarf;
mod elf;
mod reloc;
pub mod stub;

use std::fmt;

use rustc_hash::FxHashMap;

pub use elf::{FuncSlot, JumpTable, ObjectImage, Patch, SectionInfo};
pub use reloc::{RelocFail, apply_reloc};

/// The ELF `e_machine` of the image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Machine {
    /// `EM_X86_64` (62).
    X86_64,
    /// `EM_AARCH64` (183).
    Aarch64,
}

impl Machine {
    pub fn e_machine(self) -> u16 {
        match self {
            Machine::X86_64 => 62,
            Machine::Aarch64 => 183,
        }
    }

    /// The byte used to fill padding in `.text` (x86-64 `int3`; AArch64 zero words are `udf #0`).
    pub fn trap_fill(self) -> u8 {
        match self {
            Machine::X86_64 => 0xcc,
            Machine::Aarch64 => 0x00,
        }
    }

    /// The default segment alignment: 4 KiB on x86-64, 64 KiB on AArch64 (valid for every
    /// AArch64 Linux page size).
    pub fn default_page_size(self) -> u64 {
        match self {
            Machine::X86_64 => 0x1000,
            Machine::Aarch64 => 0x1_0000,
        }
    }
}

/// A symbol: a function, a data object, or a bss object. Ids are dense, in declaration order,
/// which is also the order of `.text`, the data sections, and `.symtab` (canonical instance order).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SymbolId(pub u32);

/// A source file in the DWARF line table (`FileId(n)` is DWARF v5 file entry `n + 1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FileId(pub u32);

/// How a relocation field is patched. `S` is the target address (the jump-table stub of a
/// function when the jump table is enabled), `B` the symbol's own address (a function's slot),
/// `A` the addend, `P` the address of the field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RelocKind {
    /// x86-64 `rel32` (call, jmp, jcc, RIP-relative disp32): 4 bytes, `S + A - P` as `i32`
    /// (`R_X86_64_PC32`). Calls use `A = -4`.
    X86Pc32,
    /// Absolute 64-bit address `S + A`, 8 bytes, either arch (`R_*_ABS64`, `movabs` imm64).
    Abs64,
    /// Absolute 64-bit address of the symbol's own body `B + A`, 8 bytes, either arch: for a
    /// function, the first instruction in its slot even when the jump table is enabled (for
    /// other symbols the same as [`RelocKind::Abs64`]). For tables that map code addresses back
    /// to functions (the PC table, contracts/intrinsics.md). [`ObjectImage::patch_func`]
    /// rewrites these fields when it moves the function.
    Abs64Body,
    /// AArch64 `B`/`BL` imm26: `(S + A - P) >> 2` (`R_AARCH64_CALL26`/`JUMP26`), ±128 MiB.
    A64Branch26,
    /// AArch64 `ADRP` imm21: `Page(S + A) - Page(P)` in 4 KiB pages (`R_AARCH64_ADR_PREL_PG_HI21`).
    A64AdrPage21,
    /// AArch64 `ADD` (immediate) imm12: `(S + A) & 0xfff` (`R_AARCH64_ADD_ABS_LO12_NC`).
    A64AddLo12,
    /// AArch64 load/store unsigned-offset imm12: `((S + A) & 0xfff) >> shift`, which must be exact
    /// (`R_AARCH64_LDST{8,16,32,64,128}_ABS_LO12_NC`, shift 0..=4).
    A64LdstLo12 { shift: u8 },
}

impl RelocKind {
    /// Width in bytes of the patched field.
    pub fn width(self) -> usize {
        match self {
            RelocKind::Abs64 | RelocKind::Abs64Body => 8,
            _ => 4,
        }
    }

    fn fits(self, m: Machine) -> bool {
        match self {
            RelocKind::Abs64 | RelocKind::Abs64Body => true,
            RelocKind::X86Pc32 => m == Machine::X86_64,
            _ => m == Machine::Aarch64,
        }
    }
}

/// One relocation inside a function or data object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Reloc {
    /// Byte offset of the field within its function or data object.
    pub offset: u32,
    pub kind: RelocKind,
    pub sym: SymbolId,
    pub addend: i64,
}

/// One row of a function's line table: the instruction at `offset` (bytes from the function
/// start) begins source position `file:line:column` (column 0 means unknown).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LineRow {
    pub offset: u32,
    pub file: FileId,
    pub line: u32,
    pub column: u32,
}

/// A function's machine code (without the frame layout).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FuncCode {
    pub code: Vec<u8>,
    pub relocs: Vec<Reloc>,
    /// One DWARF sequence per function. Rows are sorted by offset (stable) when linked.
    pub lines: Vec<LineRow>,
}

/// A read-only or writable data object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DataObj {
    pub bytes: Vec<u8>,
    /// A power of two, at least 1.
    pub align: u32,
    pub relocs: Vec<Reloc>,
}

impl DataObj {
    pub fn new(bytes: impl Into<Vec<u8>>, align: u32) -> DataObj {
        DataObj { bytes: bytes.into(), align, relocs: Vec::new() }
    }
}

/// Slack added to each function slot so a grown function can be patched in place:
/// capacity = `align_up(len + max(min, len * percent / 100), func_align)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slack {
    pub min: u32,
    pub percent: u32,
}

/// Layout and content options. [`Options::release`] and [`Options::debug`] are the two presets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Options {
    /// Virtual address of the first byte of the file (the ELF header). Page aligned.
    pub base_addr: u64,
    /// Segment alignment (`p_align`): a power of two ≥ 4096.
    pub page_size: u64,
    /// Alignment of each function slot: a power of two ≥ 4 (≥ 1 on x86-64).
    pub func_align: u32,
    /// Padding per function slot (debug builds). `None`: slots are only aligned.
    pub slack: Option<Slack>,
    /// Route every reference to a function through a per-function jump-table stub.
    pub jump_table: bool,
    /// Spare trap-filled bytes at the end of `.text` for functions that outgrow their slot
    /// (only used with the jump table).
    pub text_reserve: u32,
    /// Emit `.debug_abbrev`, `.debug_info` (one compile unit), and `.debug_line`.
    pub debug_info: bool,
    /// Emit `.symtab` and `.strtab`.
    pub symtab: bool,
    /// DWARF `DW_AT_name` of the compile unit (and file 0 when no file was added).
    pub cu_name: String,
    /// DWARF `DW_AT_comp_dir` and line-table directory 0.
    pub comp_dir: String,
}

impl Options {
    /// Packed layout, symbols and line tables, no patching support.
    pub fn release(m: Machine) -> Options {
        Options {
            base_addr: 0x40_0000,
            page_size: m.default_page_size(),
            func_align: 16,
            slack: None,
            jump_table: false,
            text_reserve: 0,
            debug_info: true,
            symtab: true,
            cu_name: String::from("main.stv"),
            comp_dir: String::from("."),
        }
    }

    /// Padded slots, a jump table, and a 64 KiB reserve for in-place patching.
    pub fn debug(m: Machine) -> Options {
        Options {
            slack: Some(Slack { min: 32, percent: 25 }),
            jump_table: true,
            text_reserve: 0x1_0000,
            ..Options::release(m)
        }
    }
}

/// A source file for the line table.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SourceFile {
    pub dir: String,
    pub name: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SymDef {
    Undef,
    Func(FuncCode),
    Rodata(DataObj),
    Data(DataObj),
    Bss { size: u64, align: u32 },
}

/// Errors from building, linking, or patching an image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ObjError {
    /// The id was not returned by this builder.
    UnknownSymbol(SymbolId),
    UnknownFile(FileId),
    /// A symbol was defined twice.
    Redefined(String),
    /// A symbol was declared and referenced but never defined.
    Undefined(String),
    /// No entry point was set.
    NoEntry,
    /// The symbol must be a function (entry point, patch target).
    NotAFunction(String),
    /// An alignment is not a power of two, or an option is out of range.
    BadOption(&'static str),
    /// A relocation field lies outside its function or data object.
    RelocOutOfBounds {
        site: String,
        offset: u32,
    },
    /// A relocation kind does not exist on this machine.
    RelocWrongMachine {
        site: String,
        kind: RelocKind,
    },
    /// The resolved value does not fit the field (or is not suitably aligned).
    RelocOverflow {
        site: String,
        target: String,
        kind: RelocKind,
    },
    /// A line row lies outside its function.
    LineOutOfRange {
        site: String,
        offset: u32,
    },
    /// The new code does not fit the slot and cannot move (no jump table or reserve exhausted).
    SlotOverflow {
        name: String,
        need: u64,
        cap: u64,
    },
    /// The image exceeds a format limit (section count, 4 GiB code, address overflow).
    TooLarge,
}

impl fmt::Display for ObjError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ObjError::UnknownSymbol(s) => write!(f, "unknown symbol id {}", s.0),
            ObjError::UnknownFile(x) => write!(f, "unknown file id {}", x.0),
            ObjError::Redefined(n) => write!(f, "symbol `{n}` defined twice"),
            ObjError::Undefined(n) => write!(f, "undefined symbol `{n}`"),
            ObjError::NoEntry => f.write_str("no entry point"),
            ObjError::NotAFunction(n) => write!(f, "`{n}` is not a function"),
            ObjError::BadOption(w) => write!(f, "invalid option: {w}"),
            ObjError::RelocOutOfBounds { site, offset } => {
                write!(f, "relocation at {site}+{offset:#x} outside its object")
            }
            ObjError::RelocWrongMachine { site, kind } => {
                write!(f, "relocation {kind:?} in {site} is not valid for this machine")
            }
            ObjError::RelocOverflow { site, target, kind } => {
                write!(f, "relocation {kind:?} in {site} to `{target}` out of range")
            }
            ObjError::LineOutOfRange { site, offset } => {
                write!(f, "line row at {site}+{offset:#x} outside the function")
            }
            ObjError::SlotOverflow { name, need, cap } => {
                write!(f, "`{name}` needs {need} bytes, slot holds {cap}")
            }
            ObjError::TooLarge => f.write_str("image too large"),
        }
    }
}

impl std::error::Error for ObjError {}

/// Collects symbols and their contents, then links them into an [`ObjectImage`].
#[derive(Clone, Debug)]
pub struct ImageBuilder {
    pub(crate) machine: Machine,
    pub(crate) opts: Options,
    pub(crate) names: Vec<String>,
    pub(crate) defs: Vec<SymDef>,
    pub(crate) by_name: FxHashMap<String, SymbolId>,
    pub(crate) files: Vec<SourceFile>,
    pub(crate) entry: Option<SymbolId>,
}

impl ImageBuilder {
    pub fn new(machine: Machine, opts: Options) -> ImageBuilder {
        ImageBuilder {
            machine,
            opts,
            names: Vec::new(),
            defs: Vec::new(),
            by_name: FxHashMap::default(),
            files: Vec::new(),
            entry: None,
        }
    }

    pub fn machine(&self) -> Machine {
        self.machine
    }

    /// Declares `name` (idempotent: an existing name returns its id). Defining it is separate, so
    /// relocations may reference symbols defined later.
    pub fn declare(&mut self, name: &str) -> SymbolId {
        if let Some(&id) = self.by_name.get(name) {
            return id;
        }
        let id = SymbolId(self.names.len() as u32);
        self.names.push(name.to_owned());
        self.defs.push(SymDef::Undef);
        self.by_name.insert(name.to_owned(), id);
        id
    }

    pub fn lookup(&self, name: &str) -> Option<SymbolId> {
        self.by_name.get(name).copied()
    }

    pub fn name(&self, id: SymbolId) -> Option<&str> {
        self.names.get(id.0 as usize).map(String::as_str)
    }

    fn define(&mut self, id: SymbolId, def: SymDef) -> Result<(), ObjError> {
        let slot = self.defs.get_mut(id.0 as usize).ok_or(ObjError::UnknownSymbol(id))?;
        if *slot != SymDef::Undef {
            return Err(ObjError::Redefined(self.names[id.0 as usize].clone()));
        }
        *slot = def;
        Ok(())
    }

    pub fn define_func(&mut self, id: SymbolId, code: FuncCode) -> Result<(), ObjError> {
        self.define(id, SymDef::Func(code))
    }

    pub fn define_rodata(&mut self, id: SymbolId, obj: DataObj) -> Result<(), ObjError> {
        check_align(obj.align)?;
        self.define(id, SymDef::Rodata(obj))
    }

    pub fn define_data(&mut self, id: SymbolId, obj: DataObj) -> Result<(), ObjError> {
        check_align(obj.align)?;
        self.define(id, SymDef::Data(obj))
    }

    pub fn define_bss(&mut self, id: SymbolId, size: u64, align: u32) -> Result<(), ObjError> {
        check_align(align)?;
        self.define(id, SymDef::Bss { size, align })
    }

    /// Declares and defines a function.
    pub fn func(&mut self, name: &str, code: FuncCode) -> Result<SymbolId, ObjError> {
        let id = self.declare(name);
        self.define_func(id, code)?;
        Ok(id)
    }

    pub fn rodata(&mut self, name: &str, obj: DataObj) -> Result<SymbolId, ObjError> {
        let id = self.declare(name);
        self.define_rodata(id, obj)?;
        Ok(id)
    }

    pub fn data(&mut self, name: &str, obj: DataObj) -> Result<SymbolId, ObjError> {
        let id = self.declare(name);
        self.define_data(id, obj)?;
        Ok(id)
    }

    pub fn bss(&mut self, name: &str, size: u64, align: u32) -> Result<SymbolId, ObjError> {
        let id = self.declare(name);
        self.define_bss(id, size, align)?;
        Ok(id)
    }

    /// Adds a source file to the line table (idempotent); the first file added is also the unit's
    /// primary file.
    pub fn add_file(&mut self, dir: &str, name: &str) -> FileId {
        let f = SourceFile { dir: dir.to_owned(), name: name.to_owned() };
        if let Some(i) = self.files.iter().position(|x| *x == f) {
            return FileId(i as u32);
        }
        self.files.push(f);
        FileId(self.files.len() as u32 - 1)
    }

    pub fn set_entry(&mut self, id: SymbolId) -> Result<(), ObjError> {
        if id.0 as usize >= self.names.len() {
            return Err(ObjError::UnknownSymbol(id));
        }
        self.entry = Some(id);
        Ok(())
    }

    /// Defines `_start` as the fixed per-arch stub ([`stub::start`]) that calls `entry(sp)`, and
    /// makes it the entry point.
    pub fn add_start_stub(&mut self, entry: SymbolId) -> Result<SymbolId, ObjError> {
        if entry.0 as usize >= self.names.len() {
            return Err(ObjError::UnknownSymbol(entry));
        }
        let start = self.func(stub::START, stub::start(self.machine, entry))?;
        self.entry = Some(start);
        Ok(start)
    }

    /// Lays out, resolves relocations, and serializes the image.
    pub fn link(self) -> Result<ObjectImage, ObjError> {
        elf::link(self)
    }
}

pub(crate) fn check_align(a: u32) -> Result<(), ObjError> {
    if a == 0 || !a.is_power_of_two() {
        return Err(ObjError::BadOption("alignment must be a power of two"));
    }
    Ok(())
}
