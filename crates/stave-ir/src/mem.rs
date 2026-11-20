//! Simulated memory of the interpreter: a read-only data region (string literals), a stack
//! (bump allocated per frame, released on return), and heap regions (host `mmap`).
//! Every access is bounds-checked against a live region; anything else is a [`Fault`].
//! In checking mode ([`Memory::new_checked`]) memory is laid out by [`crate::memcheck`] instead,
//! and a fault that is a memory error also leaves a [`MemReport`] ([`Memory::take_report`]).

use std::collections::BTreeMap;

use crate::memcheck::{
    Access, Alloc, AllocKind, Checker, Class, ID_SHIFT, MAX_SIZE, MemReport, SLOT_START, State, Trace,
    uninit_bits,
};

/// Start of the read-only data region (string literals).
pub const DATA_BASE: u64 = 0x0010_0000;
/// Start of heap regions.
pub const HEAP_BASE: u64 = 0x1_0000_0000;
/// Start of the stack region (grows upward in this model).
pub const STACK_BASE: u64 = 0x7000_0000_0000;

/// An invalid access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fault {
    pub addr: u64,
    pub len: u64,
    pub write: bool,
}

#[derive(Clone, Debug)]
pub struct Memory {
    data: Vec<u8>,
    stack: Vec<u8>,
    stack_limit: usize,
    heap: BTreeMap<u64, Vec<u8>>,
    heap_next: u64,
    heap_bytes: usize,
    heap_limit: usize,
    chk: Option<Box<Checker>>,
}

fn align_up(x: u64, a: u64) -> Option<u64> {
    let a = a.max(1);
    x.checked_add(a - 1).map(|y| y / a * a)
}

impl Memory {
    pub fn new(stack_limit: usize, heap_limit: usize) -> Memory {
        Memory {
            data: Vec::new(),
            stack: Vec::new(),
            stack_limit,
            heap: BTreeMap::new(),
            heap_next: HEAP_BASE,
            heap_bytes: 0,
            heap_limit,
            chk: None,
        }
    }

    /// Memory in checking mode.
    pub fn new_checked(stack_limit: usize, heap_limit: usize) -> Memory {
        let chk = Checker { allocs: vec![None], ..Checker::default() };
        Memory { chk: Some(Box::new(chk)), ..Memory::new(stack_limit, heap_limit) }
    }

    pub fn is_checked(&self) -> bool {
        self.chk.is_some()
    }

    /// Appends read-only bytes; returns their address.
    pub fn add_data(&mut self, bytes: &[u8]) -> u64 {
        let a = DATA_BASE + self.data.len() as u64;
        self.data.extend_from_slice(bytes);
        // One padding byte keeps distinct literals at distinct addresses.
        self.data.push(0);
        a
    }

    fn locate(&self, addr: u64, len: u64) -> Option<(u8, u64)> {
        let end = addr.checked_add(len)?;
        if addr >= STACK_BASE {
            if end <= STACK_BASE + self.stack.len() as u64 {
                return Some((1, STACK_BASE));
            }
            return None;
        }
        if addr >= DATA_BASE && end <= DATA_BASE + self.data.len() as u64 {
            return Some((0, DATA_BASE));
        }
        let (&base, region) = self.heap.range(..=addr).next_back()?;
        if end <= base + region.len() as u64 { Some((2, base)) } else { None }
    }

    /// Reads `len` bytes at `addr`.
    pub fn read(&self, addr: u64, len: u64) -> Result<&[u8], Fault> {
        if len == 0 {
            return Ok(&[]);
        }
        if let Some(c) = &self.chk {
            return self.read_checked(c, addr, len);
        }
        let fault = Fault { addr, len, write: false };
        let (kind, base) = self.locate(addr, len).ok_or(fault)?;
        let off = (addr - base) as usize;
        let n = len as usize;
        let r = match kind {
            0 => self.data.get(off..off + n),
            1 => self.stack.get(off..off + n),
            _ => self.heap.get(&base).and_then(|r| r.get(off..off + n)),
        };
        r.ok_or(fault)
    }

    /// Writes `bytes` at `addr` (the data region is read-only).
    pub fn write(&mut self, addr: u64, bytes: &[u8]) -> Result<(), Fault> {
        if bytes.is_empty() {
            return Ok(());
        }
        let len = bytes.len() as u64;
        if self.chk.is_some() {
            return self.write_checked(addr, bytes);
        }
        let fault = Fault { addr, len, write: true };
        let (kind, base) = self.locate(addr, len).ok_or(fault)?;
        let off = (addr - base) as usize;
        let r = match kind {
            1 => self.stack.get_mut(off..off + bytes.len()),
            2 => self.heap.get_mut(&base).and_then(|r| r.get_mut(off..off + bytes.len())),
            _ => None,
        };
        r.ok_or(fault)?.copy_from_slice(bytes);
        Ok(())
    }

    /// Reads a little-endian unsigned integer of `size` (≤ 8) bytes.
    pub fn read_uint(&self, addr: u64, size: u64) -> Result<u64, Fault> {
        let b = self.read(addr, size.min(8))?;
        let mut buf = [0u8; 8];
        buf[..b.len()].copy_from_slice(b);
        Ok(u64::from_le_bytes(buf))
    }

    /// Writes the low `size` (≤ 8) bytes of `v`, little-endian.
    pub fn write_uint(&mut self, addr: u64, size: u64, v: u64) -> Result<(), Fault> {
        let b = v.to_le_bytes();
        self.write(addr, &b[..size.min(8) as usize])
    }

    /// The current stack top (a frame mark).
    pub fn stack_mark(&self) -> usize {
        self.stack.len()
    }

    /// Allocates zeroed stack bytes; `None` if the stack limit is exceeded.
    pub fn stack_alloc(&mut self, size: u64, align: u64) -> Option<u64> {
        let start = align_up(self.stack.len() as u64, align)?;
        let end = start.checked_add(size)?;
        if end > self.stack_limit as u64 {
            return None;
        }
        self.stack.resize(end as usize, 0);
        if let Some(c) = &mut self.chk {
            let a = c.place(AllocKind::Stack, size, align, false)?;
            if let Some(Some(r)) = c.allocs.last_mut() {
                r.mark = start as usize;
            }
            c.slots.push((a >> ID_SHIFT) as u32);
            return Some(a);
        }
        Some(STACK_BASE + start)
    }

    /// Releases the stack down to a mark.
    pub fn stack_release(&mut self, mark: usize) {
        self.stack.truncate(mark);
        if let Some(c) = &mut self.chk {
            while let Some(&id) = c.slots.last() {
                match c.allocs.get_mut(id as usize) {
                    Some(Some(a)) if a.mark >= mark => {
                        a.state = State::Returned;
                        a.bytes = Vec::new();
                        a.init = Vec::new();
                    }
                    _ => break,
                }
                c.slots.pop();
            }
        }
    }

    /// Allocates a zeroed heap region; `None` if the heap limit is exceeded.
    pub fn alloc(&mut self, size: u64, align: u64) -> Option<u64> {
        let total = self.heap_bytes.checked_add(usize::try_from(size).ok()?)?;
        if total > self.heap_limit {
            return None;
        }
        if let Some(c) = &mut self.chk {
            let a = c.place(AllocKind::Host, size, align, true)?;
            self.heap_bytes = total;
            return Some(a);
        }
        let base = align_up(self.heap_next, align.max(16))?;
        // A gap after each region catches small overruns.
        let next = base.checked_add(size)?.checked_add(4096)?;
        if next >= STACK_BASE {
            return None;
        }
        self.heap.insert(base, vec![0; size as usize]);
        self.heap_next = next;
        self.heap_bytes = total;
        Some(base)
    }

    /// Frees the heap region starting at `addr`; false if there is none.
    pub fn free(&mut self, addr: u64) -> bool {
        if let Some(c) = &mut self.chk {
            return match c.live_base(addr, AllocKind::Host) {
                Some(a) => {
                    a.state = State::Freed;
                    a.bytes = Vec::new();
                    a.init = Vec::new();
                    self.heap_bytes -= a.size as usize;
                    true
                }
                None => false,
            };
        }
        match self.heap.remove(&addr) {
            Some(r) => {
                self.heap_bytes -= r.len();
                true
            }
            None => false,
        }
    }

    /// Size of the heap region starting at `addr`.
    pub fn region_size(&self, addr: u64) -> Option<u64> {
        if let Some(c) = &self.chk {
            let a = c.allocs.get((addr >> ID_SHIFT) as usize)?.as_ref()?;
            return (a.kind == AllocKind::Host && a.state == State::Live && a.base == addr).then_some(a.size);
        }
        self.heap.get(&addr).map(|r| r.len() as u64)
    }

    /// Bytes currently allocated on the heap.
    pub fn heap_bytes(&self) -> usize {
        self.heap_bytes
    }

    /// The pending memory error, if any (checking mode).
    pub fn take_report(&mut self) -> Option<Box<MemReport>> {
        self.chk.as_ref().and_then(|c| c.report.borrow_mut().take())
    }

    /// Checking mode: the little-endian unsigned integer in the `size` bytes (at most 8) at
    /// `addr`, when they are bytes of a live allocation and initialised. Bytes that are not are
    /// no error here: nothing is reported.
    pub fn peek_uint(&self, addr: u64, size: u64) -> Option<u64> {
        let c = self.chk.as_ref()?;
        let a = c.allocs.get((addr >> ID_SHIFT) as usize)?.as_ref()?;
        let off = addr.checked_sub(a.base)?;
        if a.state != State::Live || off.checked_add(size)? > a.size || !a.is_init(off, size) {
            return None;
        }
        let b = a.bytes.get(off as usize..(off + size) as usize)?;
        let mut w = [0u8; 8];
        w.get_mut(..b.len())?.copy_from_slice(b);
        Some(u64::from_le_bytes(w))
    }

    pub fn has_report(&self) -> bool {
        self.chk.as_ref().is_some_and(|c| c.report.borrow().is_some())
    }

    fn read_checked<'a>(&'a self, c: &'a Checker, addr: u64, len: u64) -> Result<&'a [u8], Fault> {
        let fault = Fault { addr, len, write: false };
        if addr >> ID_SHIFT == 0 {
            let end = addr.checked_add(len).ok_or(fault)?;
            if addr >= DATA_BASE && end <= DATA_BASE + self.data.len() as u64 {
                let off = (addr - DATA_BASE) as usize;
                return self.data.get(off..off + len as usize).ok_or(fault);
            }
            c.fail(Class::OutOfBounds, Access::Read, addr, len, None);
            return Err(fault);
        }
        let (id, off) = c.check(Access::Read, addr, len).ok_or(fault)?;
        let a = c.allocs[id].as_ref().ok_or(fault)?;
        if !a.is_init(off, len) {
            c.fail(Class::UninitRead, Access::Read, addr, len, Some(id));
            return Err(fault);
        }
        a.bytes.get(off as usize..(off + len) as usize).ok_or(fault)
    }

    fn write_checked(&mut self, addr: u64, bytes: &[u8]) -> Result<(), Fault> {
        let len = bytes.len() as u64;
        let fault = Fault { addr, len, write: true };
        let Some(c) = &mut self.chk else { return Err(fault) };
        if addr >> ID_SHIFT == 0 {
            let end = addr.checked_add(len).ok_or(fault)?;
            // the data region is read-only: a fault, as without checking
            if !(addr >= DATA_BASE && end <= DATA_BASE + self.data.len() as u64) {
                c.fail(Class::OutOfBounds, Access::Write, addr, len, None);
            }
            return Err(fault);
        }
        let (id, off) = c.check(Access::Write, addr, len).ok_or(fault)?;
        let a = c.allocs[id].as_mut().ok_or(fault)?;
        a.bytes.get_mut(off as usize..(off + len) as usize).ok_or(fault)?.copy_from_slice(bytes);
        a.set_init(off, len);
        Ok(())
    }

    /// Checking mode: allocates a runtime-allocator block (uninitialised); `None` when the heap
    /// limit is exceeded or the size cannot be placed.
    pub fn heap_alloc(&mut self, size: u64, align: u64, at: Trace) -> Option<u64> {
        let total = self.heap_bytes.checked_add(usize::try_from(size).ok()?)?;
        if total > self.heap_limit {
            return None;
        }
        let c = self.chk.as_mut()?;
        let a = c.place(AllocKind::Heap, size, align, false)?;
        if let Some(Some(r)) = c.allocs.last_mut() {
            r.alloc_at = at;
        }
        self.heap_bytes = total;
        Some(a)
    }

    /// Checking mode: whether `addr` may be freed as a block of `size` and `align`; the report
    /// when not. `Ok` leaves the block live.
    pub fn heap_check_free(&self, addr: u64, size: u64, align: u64) -> Result<(), Box<MemReport>> {
        let Some(c) = &self.chk else { return Ok(()) };
        let id = (addr >> ID_SHIFT) as usize;
        let a = match c.allocs.get(id) {
            Some(Some(a)) if id != 0 => a,
            _ => return Err(c.report(Class::InvalidFree, Access::Free, addr, size, None)),
        };
        let class = if a.kind != AllocKind::Heap || addr != a.base {
            Class::InvalidFree
        } else if a.state != State::Live {
            Class::DoubleFree
        } else if a.size != size || a.align != align {
            Class::FreeSizeMismatch
        } else {
            return Ok(());
        };
        let mut r = c.report(class, Access::Free, addr, size, Some(id));
        if class == Class::FreeSizeMismatch {
            r.passed = Some((size, align));
        }
        Err(r)
    }

    /// Checking mode: frees a block [`Memory::heap_check_free`] accepted.
    pub fn heap_free(&mut self, addr: u64, at: Trace) {
        let Some(c) = &mut self.chk else { return };
        if let Some(Some(a)) = c.allocs.get_mut((addr >> ID_SHIFT) as usize) {
            a.state = State::Freed;
            a.bytes = Vec::new();
            a.init = Vec::new();
            a.freed_at = at;
            self.heap_bytes -= a.size as usize;
        }
    }

    /// Copies `n` bytes from `src` to `dst` as bytes, never as a typed value: in checking mode
    /// each destination byte is initialised exactly when its source byte was, so an
    /// uninitialised byte moves without an error. The source range is checked as a read and the
    /// destination range as a write (bounds, liveness, the read-only data region). Ranges that
    /// overlap are copied as if through a buffer.
    pub fn copy(&mut self, dst: u64, src: u64, n: u64) -> Result<(), Fault> {
        if n == 0 {
            return Ok(());
        }
        if self.chk.is_none() {
            let bytes = self.read(src, n)?.to_vec();
            return self.write(dst, &bytes);
        }
        let (bytes, init) = if src >> ID_SHIFT == 0 {
            // the data region: initialised bytes
            (self.read(src, n)?.to_vec(), None)
        } else {
            let fault = Fault { addr: src, len: n, write: false };
            let c = self.chk.as_ref().ok_or(fault)?;
            let (si, so) = c.check(Access::Read, src, n).ok_or(fault)?;
            let s = c.allocs[si].as_ref().ok_or(fault)?;
            let bytes = s.bytes.get(so as usize..(so + n) as usize).ok_or(fault)?.to_vec();
            let init: Option<Vec<bool>> =
                (!s.is_init(so, n)).then(|| (so..so + n).map(|k| s.is_init(k, 1)).collect());
            (bytes, init)
        };
        if dst >> ID_SHIFT == 0 {
            return self.write_checked(dst, &bytes);
        }
        let fault = Fault { addr: dst, len: n, write: true };
        let c = self.chk.as_mut().ok_or(fault)?;
        let (di, doff) = c.check(Access::Write, dst, n).ok_or(fault)?;
        let d = c.allocs[di].as_mut().ok_or(fault)?;
        d.bytes.get_mut(doff as usize..(doff + n) as usize).ok_or(fault)?.copy_from_slice(&bytes);
        match init {
            None => d.set_init(doff, n),
            Some(bits) => d.put_init(doff, &bits),
        }
        Ok(())
    }

    /// Checking mode: reports a copy of `n` bytes to `dst` whose source range overlaps the
    /// destination (class `copy-overlap`).
    pub fn fail_overlap(&self, dst: u64, n: u64) {
        if let Some(c) = &self.chk {
            let id = (dst >> ID_SHIFT) as usize;
            let known = matches!(c.allocs.get(id), Some(Some(_)));
            c.fail(Class::CopyOverlap, Access::Write, dst, n, known.then_some(id));
        }
    }

    /// Checking mode: records where the allocation containing `addr` was made.
    pub fn note_alloc_site(&mut self, addr: u64, at: Trace) {
        if let Some(c) = &mut self.chk
            && let Some(Some(a)) = c.allocs.get_mut((addr >> ID_SHIFT) as usize)
        {
            a.alloc_at = at;
        }
    }

    /// Checking mode: the ids of the stack slots made since `mark`.
    pub fn slots_since(&self, mark: usize) -> Vec<u32> {
        let Some(c) = &self.chk else { return Vec::new() };
        let live = |id: u32| matches!(c.allocs.get(id as usize), Some(Some(a)) if a.mark >= mark);
        c.slots.iter().rev().copied().take_while(|&id| live(id)).collect()
    }

    /// Checking mode: the allocation `addr` names, as (id, first byte, size).
    pub fn alloc_span(&self, addr: u64) -> Option<(u32, u64, u64)> {
        let id = addr >> ID_SHIFT;
        let a = self.chk.as_ref()?.allocs.get(id as usize)?.as_ref()?;
        Some((id as u32, a.base, a.size))
    }

    /// Checking mode: the first byte and the size of allocation `id`, and whether it is live.
    pub fn alloc_of(&self, id: u32) -> Option<(u64, u64, bool)> {
        let a = self.chk.as_ref()?.allocs.get(id as usize)?.as_ref()?;
        Some((a.base, a.size, a.state == State::Live))
    }

    /// Checking mode: the runtime-allocator blocks still live, as (address, size, allocation site).
    pub fn live_heap_blocks(&self) -> Vec<(u64, u64, Trace)> {
        let Some(c) = &self.chk else { return Vec::new() };
        c.allocs
            .iter()
            .flatten()
            .filter(|a| a.kind == AllocKind::Heap && a.state == State::Live)
            .map(|a| (a.base, a.size, a.alloc_at.clone()))
            .collect()
    }
}

impl Checker {
    /// A new allocation of `size` bytes; its address, or `None` if it cannot be placed.
    fn place(&mut self, kind: AllocKind, size: u64, align: u64, init: bool) -> Option<u64> {
        if size > MAX_SIZE || align > MAX_SIZE {
            return None;
        }
        let id = self.allocs.len() as u64;
        if id >= 1 << (63 - ID_SHIFT) {
            return None;
        }
        let base = id << ID_SHIFT | SLOT_START;
        let n = usize::try_from(size).ok()?;
        self.allocs.push(Some(Alloc {
            kind,
            state: State::Live,
            base,
            size,
            align,
            bytes: vec![0; n],
            init: if init { Vec::new() } else { uninit_bits(size) },
            mark: 0,
            alloc_at: Vec::new(),
            freed_at: Vec::new(),
        }));
        Some(base)
    }

    /// The live allocation of `kind` that starts at `addr`.
    fn live_base(&mut self, addr: u64, kind: AllocKind) -> Option<&mut Alloc> {
        let id = (addr >> ID_SHIFT) as usize;
        let a = self.allocs.get_mut(id)?.as_mut()?;
        (id != 0 && a.kind == kind && a.state == State::Live && a.base == addr).then_some(a)
    }

    /// Checks liveness and bounds of `addr..addr + len` (`addr` above the data region): the
    /// allocation id and offset, or `None` with the report recorded.
    fn check(&self, access: Access, addr: u64, len: u64) -> Option<(usize, u64)> {
        let id = (addr >> ID_SHIFT) as usize;
        let Some(Some(a)) = self.allocs.get(id) else {
            self.fail(Class::OutOfBounds, access, addr, len, None);
            return None;
        };
        if a.state != State::Live {
            self.fail(Class::UseAfterFree, access, addr, len, Some(id));
            return None;
        }
        let off = addr.wrapping_sub(a.base);
        if addr < a.base || off.checked_add(len).is_none_or(|e| e > a.size) {
            self.fail(Class::OutOfBounds, access, addr, len, Some(id));
            return None;
        }
        Some((id, off))
    }

    fn report(&self, class: Class, access: Access, addr: u64, len: u64, id: Option<usize>) -> Box<MemReport> {
        let a = id.and_then(|k| self.allocs.get(k)).and_then(|a| a.as_ref());
        Box::new(MemReport {
            class,
            access,
            addr,
            len,
            alloc: a.map(|a| a.info(id.unwrap_or(0) as u32)),
            offset: a.map_or(0, |a| addr.wrapping_sub(a.base) as i64),
            passed: None,
            at: Vec::new(),
        })
    }

    /// Records a report unless one is pending (the first error wins).
    fn fail(&self, class: Class, access: Access, addr: u64, len: u64, id: Option<usize>) {
        let mut r = self.report.borrow_mut();
        if r.is_none() {
            *r = Some(self.report(class, access, addr, len, id));
        }
    }
}
