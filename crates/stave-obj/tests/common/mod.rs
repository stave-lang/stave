//! Shared helpers: a minimal ELF64 reader, hand-assembled `hello` programs, tool runners.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

use stave_obj::{DataObj, FuncCode, ImageBuilder, LineRow, Machine, ObjectImage, Options, Reloc, RelocKind};

pub fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes(b[o..o + 2].try_into().unwrap())
}
pub fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
pub fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

#[derive(Debug, Clone, Copy)]
pub struct Phdr {
    pub ty: u32,
    pub flags: u32,
    pub offset: u64,
    pub vaddr: u64,
    pub filesz: u64,
    pub memsz: u64,
    pub align: u64,
}

#[derive(Debug, Clone)]
pub struct Shdr {
    pub name: String,
    pub ty: u32,
    pub flags: u64,
    pub addr: u64,
    pub offset: u64,
    pub size: u64,
    pub link: u32,
    pub info: u32,
}

pub fn phdrs(b: &[u8]) -> Vec<Phdr> {
    let off = u64_at(b, 32) as usize;
    let n = u16_at(b, 56) as usize;
    (0..n)
        .map(|i| {
            let o = off + i * 56;
            Phdr {
                ty: u32_at(b, o),
                flags: u32_at(b, o + 4),
                offset: u64_at(b, o + 8),
                vaddr: u64_at(b, o + 16),
                filesz: u64_at(b, o + 32),
                memsz: u64_at(b, o + 40),
                align: u64_at(b, o + 48),
            }
        })
        .collect()
}

pub fn shdrs(b: &[u8]) -> Vec<Shdr> {
    let off = u64_at(b, 40) as usize;
    let n = u16_at(b, 60) as usize;
    let strndx = u16_at(b, 62) as usize;
    let raw: Vec<_> = (0..n)
        .map(|i| {
            let o = off + i * 64;
            (
                u32_at(b, o),
                u32_at(b, o + 4),
                u64_at(b, o + 8),
                u64_at(b, o + 16),
                u64_at(b, o + 24),
                u64_at(b, o + 32),
                u32_at(b, o + 40),
                u32_at(b, o + 44),
            )
        })
        .collect();
    let stroff = raw[strndx].4 as usize;
    raw.iter()
        .map(|r| {
            let s = &b[stroff + r.0 as usize..];
            let end = s.iter().position(|&c| c == 0).unwrap();
            Shdr {
                name: String::from_utf8(s[..end].to_vec()).unwrap(),
                ty: r.1,
                flags: r.2,
                addr: r.3,
                offset: r.4,
                size: r.5,
                link: r.6,
                info: r.7,
            }
        })
        .collect()
}

/// (name, value, size, info) for each `.symtab` entry after the null one.
pub fn symbols(b: &[u8]) -> Vec<(String, u64, u64, u8)> {
    let sh = shdrs(b);
    let Some(st) = sh.iter().find(|s| s.name == ".symtab") else { return Vec::new() };
    let strs = &sh[st.link as usize];
    (1..(st.size / 24) as usize)
        .map(|i| {
            let o = st.offset as usize + i * 24;
            let s = &b[strs.offset as usize + u32_at(b, o) as usize..];
            let end = s.iter().position(|&c| c == 0).unwrap();
            (String::from_utf8(s[..end].to_vec()).unwrap(), u64_at(b, o + 8), u64_at(b, o + 16), b[o + 4])
        })
        .collect()
}

/// `entry(sp)` writing `msg_len` bytes of `msg` to stdout, then `exit_group(argc)`.
pub fn hello_entry(m: Machine, msg: stave_obj::SymbolId, msg_len: u8, file: stave_obj::FileId) -> FuncCode {
    match m {
        Machine::X86_64 => {
            let code = vec![
                0x48, 0x8b, 0x1f, // mov rbx,[rdi]
                0xb8, 1, 0, 0, 0, // mov eax,1
                0xbf, 1, 0, 0, 0, // mov edi,1
                0x48, 0x8d, 0x35, 0, 0, 0, 0, // lea rsi,[rip+msg]
                0xba, msg_len, 0, 0, 0, // mov edx,len
                0x0f, 0x05, // syscall
                0xb8, 231, 0, 0, 0, // mov eax,231
                0x89, 0xdf, // mov edi,ebx
                0x0f, 0x05, // syscall
                0x0f, 0x0b, // ud2
            ];
            FuncCode {
                code,
                relocs: vec![Reloc { offset: 16, kind: RelocKind::X86Pc32, sym: msg, addend: -4 }],
                lines: vec![
                    LineRow { offset: 0, file, line: 1, column: 1 },
                    LineRow { offset: 3, file, line: 2, column: 5 },
                    LineRow { offset: 27, file, line: 3, column: 5 },
                ],
            }
        }
        Machine::Aarch64 => {
            let words: [u32; 11] = [
                0xf940_0013,                             // ldr x19,[x0]
                0xd280_0808,                             // mov x8,#64
                0xd280_0020,                             // mov x0,#1
                0x9000_0001,                             // adrp x1,msg
                0x9100_0021,                             // add x1,x1,:lo12:msg
                0xd280_0002 | (u32::from(msg_len) << 5), // mov x2,#len
                0xd400_0001,                             // svc #0
                0xd280_0bc8,                             // mov x8,#94
                0xaa13_03e0,                             // mov x0,x19
                0xd400_0001,                             // svc #0
                0xd420_0000,                             // brk #0
            ];
            FuncCode {
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
                relocs: vec![
                    Reloc { offset: 12, kind: RelocKind::A64AdrPage21, sym: msg, addend: 0 },
                    Reloc { offset: 16, kind: RelocKind::A64AddLo12, sym: msg, addend: 0 },
                ],
                lines: vec![
                    LineRow { offset: 0, file, line: 1, column: 1 },
                    LineRow { offset: 4, file, line: 2, column: 5 },
                    LineRow { offset: 28, file, line: 3, column: 5 },
                ],
            }
        }
    }
}

/// `hello`: `_start` → `runtime.start.entry(sp)` → write "hi\n", exit_group(argc).
pub fn hello(m: Machine, opts: Options) -> ObjectImage {
    let mut b = ImageBuilder::new(m, opts);
    let file = b.add_file("src", "hello.stv");
    let entry = b.declare(stave_obj::stub::RUNTIME_ENTRY);
    let msg = b.rodata("hello.msg", DataObj::new(b"hi\n".to_vec(), 1)).unwrap();
    b.add_start_stub(entry).unwrap();
    b.define_func(entry, hello_entry(m, msg, 3, file)).unwrap();
    b.link().unwrap()
}

static SEQ: AtomicU32 = AtomicU32::new(0);

/// A scratch path in a directory of its own; the directory is removed when this is dropped.
pub struct Tmp(PathBuf);

pub fn tmp(name: &str) -> Tmp {
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let d = std::env::temp_dir().join(format!("stave-obj-test-{}-{seq}", std::process::id()));
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

/// Runs the image natively (x86-64 Linux host) or under `qemu-aarch64`; `None` (with a message)
/// when neither is possible here.
pub fn run(img: &ObjectImage, name: &str, args: &[&str]) -> Option<(String, i32)> {
    let p = tmp(name);
    img.write_file(&p).unwrap();
    let mut cmd = match img.machine() {
        Machine::X86_64 if cfg!(all(target_os = "linux", target_arch = "x86_64")) => Command::new(&p),
        Machine::Aarch64 if cfg!(all(target_os = "linux", target_arch = "aarch64")) => Command::new(&p),
        Machine::Aarch64 if have("qemu-aarch64") => {
            let mut c = Command::new("qemu-aarch64");
            c.arg(&p);
            c
        }
        m => {
            eprintln!("SKIP running {name}: no way to execute {m:?} binaries on this host");
            return None;
        }
    };
    let out = output_retry(cmd.args(args));
    Some((String::from_utf8_lossy(&out.stdout).into_owned(), out.status.code().unwrap_or(-1)))
}

/// `readelf <args> file`, or `None` (with a message) when readelf is absent.
pub fn readelf(img: &ObjectImage, name: &str, args: &[&str]) -> Option<String> {
    if !have("readelf") {
        eprintln!("SKIP readelf checks: readelf not installed");
        return None;
    }
    let p = tmp(name);
    img.write_file(&p).unwrap();
    let out = Command::new("readelf").args(args).arg(&p).output().unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success() && !err.contains("Warning") && !err.contains("Error"), "readelf: {err}");
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Runs `cmd`, retrying while the kernel reports ETXTBSY: a parallel test may have forked while
/// this executable was still open for writing.
pub fn output_retry(cmd: &mut std::process::Command) -> std::process::Output {
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
