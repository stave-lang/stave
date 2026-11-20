//! Checking mode of the interpreter's [`crate::mem::Memory`]: every allocation (heap block of
//! the runtime allocator, stack slot, host mapping) is a separate object with its own address
//! slot, a liveness state, and per-byte initialisation bits. An access is checked against the
//! allocation its address names; a failed check is a [`MemReport`].
//!
//! Addresses: allocation `id` (≥ 1) starts at `id << 32 | 1 << 31`, so bits 32..63 of a pointer
//! are its provenance (the allocation it was derived from). Pointer arithmetic that leaves the
//! allocation stays in the same 4 GiB slot unless it moves by 2 GiB or more, so an access past the
//! end is reported against the allocation it came from even when another allocation follows it
//! in the program's view. Ids are never reused: a freed heap block or a returned frame's slot
//! keeps its record and is reported as such on any later access. The read-only data region
//! (string literals) is below `1 << 32` and is one object.

use crate::ir::{Body, FuncId, Module, Site};

/// Bits of an address below the allocation id.
pub const ID_SHIFT: u32 = 32;
/// Offset of an allocation's first byte inside its slot: room below it for underflows.
pub const SLOT_START: u64 = 1 << 31;
/// Largest allocation (and alignment) checking mode can place.
pub const MAX_SIZE: u64 = 1 << 31;
/// Frames kept in a [`Trace`].
pub const TRACE_DEPTH: usize = 8;

/// What went wrong. [`Class::name`] is the stable, machine-readable spelling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Class {
    /// Access to a freed heap block or to a slot of a returned frame.
    UseAfterFree,
    /// Free of a heap block that was already freed.
    DoubleFree,
    /// Read of a byte never written since its allocation (heap blocks and stack slots).
    UninitRead,
    /// Access outside the allocation the address belongs to (or to no allocation at all).
    OutOfBounds,
    /// Free of an address that is not the start of a live heap block: a stack slot, a static,
    /// a host mapping, an interior pointer, or no allocation.
    InvalidFree,
    /// Free or realloc with a size or alignment other than the block's.
    FreeSizeMismatch,
    /// A `copy_bytes` whose source and destination ranges share a byte.
    CopyOverlap,
}

impl Class {
    pub const ALL: [Class; 7] = [
        Class::UseAfterFree,
        Class::DoubleFree,
        Class::UninitRead,
        Class::OutOfBounds,
        Class::InvalidFree,
        Class::FreeSizeMismatch,
        Class::CopyOverlap,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Class::UseAfterFree => "use-after-free",
            Class::DoubleFree => "double-free",
            Class::UninitRead => "uninit-read",
            Class::OutOfBounds => "out-of-bounds",
            Class::InvalidFree => "invalid-free",
            Class::FreeSizeMismatch => "free-size-mismatch",
            Class::CopyOverlap => "copy-overlap",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Access {
    Read,
    Write,
    Free,
}

impl Access {
    pub fn name(self) -> &'static str {
        match self {
            Access::Read => "read",
            Access::Write => "write",
            Access::Free => "free",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AllocKind {
    /// A block of the runtime allocator (`runtime.alloc`).
    Heap,
    /// A stack slot (`slot`).
    Stack,
    /// Host memory: `mmap`, statics, the process stack, tables the embedder places.
    Host,
}

impl AllocKind {
    pub fn name(self) -> &'static str {
        match self {
            AllocKind::Heap => "heap",
            AllocKind::Stack => "stack",
            AllocKind::Host => "host",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum State {
    Live,
    /// Heap block freed / host mapping unmapped.
    Freed,
    /// Stack slot of a frame that returned.
    Returned,
}

/// The instruction executing in one frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct At {
    pub func: FuncId,
    /// Index of the instruction in its body.
    pub inst: u32,
    /// The instruction's site; an instruction without one (a `load`, a `store`) has the site of
    /// the nearest instruction before it that has one, and `exact` is false.
    pub site: Site,
    pub exact: bool,
}

/// One [`At`] per frame, innermost first.
pub type Trace = Vec<At>;

/// The allocation a report is about.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllocInfo {
    pub id: u32,
    pub kind: AllocKind,
    pub size: u64,
    pub state: State,
    pub alloc_at: Trace,
    pub freed_at: Trace,
}

/// One memory error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemReport {
    pub class: Class,
    pub access: Access,
    pub addr: u64,
    pub len: u64,
    /// The allocation `addr` names, if any.
    pub alloc: Option<AllocInfo>,
    /// `addr` minus the allocation's start.
    pub offset: i64,
    /// For [`Class::FreeSizeMismatch`]: the (size, alignment) passed.
    pub passed: Option<(u64, u64)>,
    /// Where the access happened (filled by the interpreter).
    pub at: Trace,
}

fn site_text(m: &Module, at: &At) -> String {
    let name = m.func(at.func).map_or("?", |f| f.path.as_str());
    let near = if at.exact { "" } else { "near " };
    match m.sites.get(at.site.0 as usize) {
        Some(i) => format!("{name} #{} ({near}{}:{}:{})", at.inst, i.file, i.line, i.col),
        None => format!("{name} #{}", at.inst),
    }
}

impl At {
    /// Instruction `inst` of `body` (fn `func`).
    pub fn of(func: FuncId, body: &Body, inst: u32) -> At {
        let site = |k: u32| body.insts.site.get(k as usize).copied().filter(|s| *s != Site::NONE);
        match site(inst) {
            Some(s) => At { func, inst, site: s, exact: true },
            None => {
                let near = (0..inst).rev().find_map(site).unwrap_or(Site::NONE);
                At { func, inst, site: near, exact: false }
            }
        }
    }
}

impl MemReport {
    /// A trace on one line: `f (file:l:c) <- g (file:l:c) ..`, innermost first.
    pub fn trace_text(m: &Module, t: &Trace) -> String {
        t.iter().map(|at| site_text(m, at)).collect::<Vec<_>>().join(" <- ")
    }

    /// Multi-line text: `memory error [<class>]: ...`, then the access trace, then where the
    /// allocation was made and freed.
    pub fn render(&self, m: &Module) -> String {
        use std::fmt::Write;
        let mut s = format!(
            "memory error [{}]: {} of {} byte(s) at {:#x}",
            self.class.name(),
            self.access.name(),
            self.len,
            self.addr
        );
        match &self.alloc {
            Some(a) => {
                let st = match a.state {
                    State::Live => "live",
                    State::Freed => "freed",
                    State::Returned => "frame returned",
                };
                let _ = write!(
                    s,
                    ", offset {} of allocation #{} ({}, {} bytes, {st})",
                    self.offset,
                    a.id,
                    a.kind.name(),
                    a.size
                );
            }
            None => s.push_str(", no allocation"),
        }
        if let (Some((size, align)), Some(a)) = (self.passed, &self.alloc) {
            let _ = write!(s, "; passed size {size} alignment {align}, allocated size {}", a.size);
        }
        let mut list = |label: &str, t: &Trace| {
            for (k, at) in t.iter().enumerate() {
                let _ = write!(s, "\n  {} {}", if k == 0 { label } else { "  from" }, site_text(m, at));
            }
        };
        list("at", &self.at);
        if let Some(a) = &self.alloc {
            list("allocated at", &a.alloc_at);
            list("freed at", &a.freed_at);
        }
        s
    }
}

/// One allocation's record. A borrow-tracking layer keeps its per-location state (the
/// references allowed to use each location) beside `init`, indexed the same way.
#[derive(Clone, Debug)]
pub(crate) struct Alloc {
    pub kind: AllocKind,
    pub state: State,
    pub base: u64,
    pub size: u64,
    pub align: u64,
    /// The bytes while live; emptied when freed.
    pub bytes: Vec<u8>,
    /// One bit per byte, set when written; empty when every byte is initialised.
    pub init: Vec<u64>,
    /// For a stack slot: the stack mark it was allocated at.
    pub mark: usize,
    pub alloc_at: Trace,
    pub freed_at: Trace,
}

impl Alloc {
    pub fn info(&self, id: u32) -> AllocInfo {
        AllocInfo {
            id,
            kind: self.kind,
            size: self.size,
            state: self.state,
            alloc_at: self.alloc_at.clone(),
            freed_at: self.freed_at.clone(),
        }
    }

    /// Whether bytes `off..off + n` are all initialised.
    pub fn is_init(&self, off: u64, n: u64) -> bool {
        if self.init.is_empty() {
            return true;
        }
        (off..off + n).all(|k| self.init[(k / 64) as usize] >> (k % 64) & 1 == 1)
    }

    pub fn set_init(&mut self, off: u64, n: u64) {
        if self.init.is_empty() {
            return;
        }
        for k in off..off + n {
            self.init[(k / 64) as usize] |= 1 << (k % 64);
        }
        if self.init.iter().all(|&w| w == u64::MAX) {
            self.init = Vec::new();
        }
    }

    /// Sets the initialisation of bytes `off..off + init.len()` to `init`, byte by byte.
    pub fn put_init(&mut self, off: u64, init: &[bool]) {
        if self.init.is_empty() {
            if init.iter().all(|&i| i) {
                return;
            }
            self.init = vec![u64::MAX; self.size.div_ceil(64).max(1) as usize];
        }
        for (k, &i) in init.iter().enumerate() {
            let b = off + k as u64;
            let (w, bit) = ((b / 64) as usize, 1u64 << (b % 64));
            if i {
                self.init[w] |= bit;
            } else {
                self.init[w] &= !bit;
            }
        }
        if self.init.iter().all(|&w| w == u64::MAX) {
            self.init = Vec::new();
        }
    }
}

/// The checking-mode state of a [`crate::mem::Memory`].
#[derive(Clone, Debug, Default)]
pub(crate) struct Checker {
    /// Index = allocation id; entry 0 is unused (the slot of the data region and null).
    pub allocs: Vec<Option<Alloc>>,
    /// Ids of the live stack slots, oldest first.
    pub slots: Vec<u32>,
    /// The first error found and not yet taken.
    pub report: std::cell::RefCell<Option<Box<MemReport>>>,
}

/// Bit set with `n` cleared bits, except the padding bits past `n` (set).
pub(crate) fn uninit_bits(n: u64) -> Vec<u64> {
    let words = n.div_ceil(64) as usize;
    let mut v = vec![0u64; words.max(1)];
    let tail = n % 64;
    if tail != 0 {
        let last = v.len() - 1;
        v[last] = !0u64 << tail;
    }
    if n == 0 {
        v.clear();
    }
    v
}
