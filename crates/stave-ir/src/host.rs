//! Pluggable host for the interpreter: system calls (`intrinsic.syscallN`) and the other
//! machine intrinsics of contracts/intrinsics.md.

use crate::interp::Val;
use crate::mem::Memory;

/// Result of a system call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SysResult {
    /// The raw return value (negative errno as two's complement for errors).
    Ret(u64),
    /// The process exits with this status (`exit`, `exit_group`).
    Exit(i32),
}

/// The machine the interpreted program talks to.
pub trait Host {
    /// `syscallN(nr, a1..aN)`; unused args are 0.
    fn syscall(&mut self, nr: u64, args: [u64; 6], mem: &mut Memory) -> SysResult;

    /// Any intrinsic other than `syscall0..6`. `Ok(None)` for no result. `Err` aborts the run
    /// with `InterpError::Host`.
    fn intrinsic(&mut self, name: &str, args: &[Val], mem: &mut Memory) -> Result<Option<Val>, String> {
        let _ = (args, mem);
        Err(format!("intrinsic.{name} is not supported by this host"))
    }
}

/// Linux system call numbering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyscallAbi {
    X86_64,
    Aarch64,
}

struct Nrs {
    read: u64,
    write: u64,
    openat: u64,
    close: u64,
    exit: u64,
    exit_group: u64,
    mmap: u64,
    munmap: u64,
    mremap: u64,
    mprotect: u64,
    rt_sigaction: u64,
    rt_sigprocmask: u64,
    sigaltstack: u64,
    getpid: u64,
}

impl SyscallAbi {
    fn nrs(self) -> Nrs {
        match self {
            SyscallAbi::X86_64 => Nrs {
                read: 0,
                write: 1,
                openat: 257,
                close: 3,
                exit: 60,
                exit_group: 231,
                mmap: 9,
                munmap: 11,
                mremap: 25,
                mprotect: 10,
                rt_sigaction: 13,
                rt_sigprocmask: 14,
                sigaltstack: 131,
                getpid: 39,
            },
            SyscallAbi::Aarch64 => Nrs {
                read: 63,
                write: 64,
                openat: 56,
                close: 57,
                exit: 93,
                exit_group: 94,
                mmap: 222,
                munmap: 215,
                mremap: 216,
                mprotect: 226,
                rt_sigaction: 134,
                rt_sigprocmask: 135,
                sigaltstack: 132,
                getpid: 172,
            },
        }
    }
}

const ENOSYS: u64 = (-38i64) as u64;
const ENOENT: u64 = (-2i64) as u64;
const EACCES: u64 = (-13i64) as u64;
const EBADF: u64 = (-9i64) as u64;
const EFAULT: u64 = (-14i64) as u64;
const EINVAL: u64 = (-22i64) as u64;
const ENOMEM: u64 = (-12i64) as u64;
const MAP_ANONYMOUS: u64 = 0x20;
const MREMAP_MAYMOVE: u64 = 1;

/// A host that captures fd 1 and fd 2 in buffers and supports `write`, `exit`, `exit_group`,
/// anonymous `mmap`, `munmap` (whole regions), and `mremap` (moving); `read` from fd 0 (the
/// `stdin` buffer) and from read-only files of an in-memory file system (`files`, opened with
/// `openat`, relative paths and `AT_FDCWD` only; `close`); `mprotect`, `rt_sigaction`,
/// `rt_sigprocmask`, and `sigaltstack` succeed without effect; `getpid` returns 1. Other system
/// calls return `-ENOSYS`. `cpuid` and `xgetbv` return zeros (no optional features);
/// `thread_start_addr` returns 0; any other intrinsic without arguments returns its entry of
/// `values` (`main_addr`, `site_table`, `pc_table`, `signal_handler_addr`), if set.
#[derive(Clone, Debug)]
pub struct BufferHost {
    pub abi: SyscallAbi,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub stdin: Vec<u8>,
    /// The in-memory file system: path bytes → contents.
    pub files: std::collections::BTreeMap<Vec<u8>, Vec<u8>>,
    /// Results of argument-less intrinsics by name.
    pub values: std::collections::BTreeMap<String, u64>,
    /// Every system call made: (nr, args, result).
    pub log: Vec<(u64, [u64; 6], SysResult)>,
    /// Open files: fd - 3 → (contents, read position).
    open: Vec<Option<(Vec<u8>, usize)>>,
    stdin_pos: usize,
}

impl BufferHost {
    pub fn new(abi: SyscallAbi) -> BufferHost {
        BufferHost {
            abi,
            stdout: Vec::new(),
            stderr: Vec::new(),
            stdin: Vec::new(),
            files: std::collections::BTreeMap::new(),
            values: std::collections::BTreeMap::new(),
            log: Vec::new(),
            open: Vec::new(),
            stdin_pos: 0,
        }
    }

    /// A NUL-terminated string at `addr` (at most 4096 bytes).
    fn c_str(mem: &Memory, addr: u64) -> Option<Vec<u8>> {
        let mut out = Vec::new();
        for k in 0..4096u64 {
            let b = mem.read(addr.wrapping_add(k), 1).ok()?;
            if b[0] == 0 {
                return Some(out);
            }
            out.push(b[0]);
        }
        None
    }

    fn call(&mut self, nr: u64, a: [u64; 6], mem: &mut Memory) -> SysResult {
        let n = self.abi.nrs();
        if nr == n.write {
            let Ok(bytes) = mem.read(a[1], a[2]) else { return SysResult::Ret(EFAULT) };
            match a[0] {
                1 => self.stdout.extend_from_slice(bytes),
                2 => self.stderr.extend_from_slice(bytes),
                _ => return SysResult::Ret(EBADF),
            }
            SysResult::Ret(a[2])
        } else if nr == n.read {
            let fd = a[0] as i64;
            let (src, pos): (&[u8], usize) = if fd == 0 {
                (&self.stdin, self.stdin_pos)
            } else {
                match usize::try_from(fd - 3).ok().and_then(|k| self.open.get(k)).and_then(|o| o.as_ref()) {
                    Some((d, p)) => (d, *p),
                    None => return SysResult::Ret(EBADF),
                }
            };
            let k = (src.len().saturating_sub(pos) as u64).min(a[2]) as usize;
            let chunk = src[pos..pos + k].to_vec();
            if mem.write(a[1], &chunk).is_err() {
                return SysResult::Ret(EFAULT);
            }
            if fd == 0 {
                self.stdin_pos += k;
            } else if let Some(Some((_, p))) = self.open.get_mut((fd - 3) as usize) {
                *p += k;
            }
            SysResult::Ret(k as u64)
        } else if nr == n.openat {
            let Some(path) = Self::c_str(mem, a[1]) else { return SysResult::Ret(EFAULT) };
            if a[2] & 3 != 0 {
                return SysResult::Ret(EACCES);
            }
            match self.files.get(&path) {
                Some(d) => {
                    self.open.push(Some((d.clone(), 0)));
                    SysResult::Ret(self.open.len() as u64 + 2)
                }
                None => SysResult::Ret(ENOENT),
            }
        } else if nr == n.close {
            match usize::try_from((a[0] as i64) - 3).ok().and_then(|k| self.open.get_mut(k)) {
                Some(slot @ Some(_)) => {
                    *slot = None;
                    SysResult::Ret(0)
                }
                _ => SysResult::Ret(EBADF),
            }
        } else if nr == n.mprotect || nr == n.rt_sigaction || nr == n.rt_sigprocmask || nr == n.sigaltstack {
            SysResult::Ret(0)
        } else if nr == n.getpid {
            SysResult::Ret(1)
        } else if nr == n.exit || nr == n.exit_group {
            SysResult::Exit((a[0] & 0xff) as i32)
        } else if nr == n.mmap {
            if a[3] & MAP_ANONYMOUS == 0 || a[1] == 0 {
                return SysResult::Ret(EINVAL);
            }
            let len = a[1].div_ceil(4096) * 4096;
            SysResult::Ret(mem.alloc(len, 4096).unwrap_or(ENOMEM))
        } else if nr == n.munmap {
            if mem.free(a[0]) { SysResult::Ret(0) } else { SysResult::Ret(EINVAL) }
        } else if nr == n.mremap {
            let (old, new_len, flags) = (a[0], a[2], a[3]);
            let Some(old_len) = mem.region_size(old) else { return SysResult::Ret(EINVAL) };
            if flags & MREMAP_MAYMOVE == 0 || new_len == 0 {
                return SysResult::Ret(EINVAL);
            }
            let len = new_len.div_ceil(4096) * 4096;
            let Some(new) = mem.alloc(len, 4096) else { return SysResult::Ret(ENOMEM) };
            let keep = old_len.min(len);
            let bytes = mem.read(old, keep).map(|b| b.to_vec()).unwrap_or_default();
            if mem.write(new, &bytes).is_err() {
                return SysResult::Ret(EFAULT);
            }
            mem.free(old);
            SysResult::Ret(new)
        } else {
            SysResult::Ret(ENOSYS)
        }
    }
}

impl Host for BufferHost {
    fn syscall(&mut self, nr: u64, args: [u64; 6], mem: &mut Memory) -> SysResult {
        let r = self.call(nr, args, mem);
        self.log.push((nr, args, r));
        r
    }

    fn intrinsic(&mut self, name: &str, args: &[Val], mem: &mut Memory) -> Result<Option<Val>, String> {
        if let Some(r) = atomic(name, args, mem)? {
            return Ok(Some(r));
        }
        match name {
            "cpuid" => Ok(Some(Val::agg(vec![Val::S(0), Val::S(0), Val::S(0), Val::S(0)]))),
            "xgetbv" | "thread_start_addr" => Ok(Some(Val::S(0))),
            // one thread only: thread creation fails like an unsupported system call
            "thread_clone" => Ok(Some(Val::S(ENOSYS))),
            _ => match self.values.get(name) {
                Some(&v) if args.is_empty() => Ok(Some(Val::S(v))),
                _ => Err(format!("intrinsic.{name} is not supported by BufferHost")),
            },
        }
    }
}

/// The CPU atomic intrinsics (contracts/intrinsics.md) on the interpreter's memory: one thread,
/// so each is a plain read-modify-write. `None`: not an atomic intrinsic.
pub fn atomic(name: &str, args: &[Val], mem: &mut Memory) -> Result<Option<Val>, String> {
    let Some(rest) = name.strip_prefix("atomic_") else { return Ok(None) };
    let (op, size) = match rest.rsplit_once('_') {
        Some((op, "u32")) => (op, 4u64),
        Some((op, "u64")) => (op, 8u64),
        _ => return Ok(None),
    };
    let a =
        |k: usize| args.get(k).and_then(|v| v.bits()).ok_or_else(|| format!("intrinsic.{name}: operand {k}"));
    let p = a(0)?;
    let fault = |_| format!("intrinsic.{name}: fault at {p:#x}");
    let mask = if size == 8 { u64::MAX } else { u64::from(u32::MAX) };
    let old = mem.read_uint(p, size).map_err(fault)?;
    let new = match op {
        "load" => None,
        "xchg" => Some(a(1)?),
        "add" => Some(old.wrapping_add(a(1)?)),
        "cas" => (old == a(1)? & mask).then_some(a(2)?),
        _ => return Ok(None),
    };
    if let Some(n) = new {
        mem.write_uint(p, size, n & mask).map_err(fault)?;
    }
    Ok(Some(Val::S(old)))
}
