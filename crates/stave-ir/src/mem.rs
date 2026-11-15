//! Simulated memory of the interpreter: a read-only data region (string literals), a stack
//! (bump allocated per frame, released on return), and heap regions (host `mmap`).
//! Every access is bounds-checked against a live region; anything else is a [`Fault`].

use std::collections::BTreeMap;

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
        }
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
        Some(STACK_BASE + start)
    }

    /// Releases the stack down to a mark.
    pub fn stack_release(&mut self, mark: usize) {
        self.stack.truncate(mark);
    }

    /// Allocates a zeroed heap region; `None` if the heap limit is exceeded.
    pub fn alloc(&mut self, size: u64, align: u64) -> Option<u64> {
        let total = self.heap_bytes.checked_add(usize::try_from(size).ok()?)?;
        if total > self.heap_limit {
            return None;
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
        self.heap.get(&addr).map(|r| r.len() as u64)
    }

    /// Bytes currently allocated on the heap.
    pub fn heap_bytes(&self) -> usize {
        self.heap_bytes
    }
}
