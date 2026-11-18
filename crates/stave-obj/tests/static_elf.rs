//! `hello` is a static ELF64 ET_EXEC with no PT_INTERP, no PT_DYNAMIC, no
//! DT_NEEDED; the syscall numbers of `runtime/{x86_64,aarch64}/sys.stv` are exactly
//! `runtime/syscalls.list` with the kernel ABI numbers.
//!
//! Also: segment layout and permissions, entry = `_start`, symbols, DWARF line rows, determinism,
//! in-place patching, and error paths. Programs run natively (x86-64 host) or under qemu-aarch64.

mod common;

use common::*;
use stave_obj::{DataObj, FuncCode, ImageBuilder, Machine, ObjError, Options, Reloc, RelocKind, stub};

const MACHINES: [Machine; 2] = [Machine::X86_64, Machine::Aarch64];
const PT_LOAD: u32 = 1;
const PT_DYNAMIC: u32 = 2;
const PT_INTERP: u32 = 3;
const PT_GNU_STACK: u32 = 0x6474_e551;

fn tag(m: Machine) -> &'static str {
    match m {
        Machine::X86_64 => "x64",
        Machine::Aarch64 => "a64",
    }
}

#[test]
fn hello_is_static_exec() {
    for m in MACHINES {
        for (k, opts) in [("rel", Options::release(m)), ("dbg", Options::debug(m))] {
            let img = hello(m, opts);
            let b = img.bytes();
            assert_eq!(&b[..4], b"\x7fELF");
            assert_eq!(b[4], 2, "ELFCLASS64");
            assert_eq!(b[5], 1, "little endian");
            assert_eq!(u16_at(b, 16), 2, "ET_EXEC");
            assert_eq!(u16_at(b, 18), m.e_machine());
            let ph = phdrs(b);
            assert!(ph.iter().all(|p| p.ty != PT_INTERP && p.ty != PT_DYNAMIC), "{ph:?}");
            assert!(ph.iter().any(|p| p.ty == PT_GNU_STACK && p.flags == 6), "non-exec stack");
            let sh = shdrs(b);
            for bad in [".interp", ".dynamic", ".dynsym", ".dynstr", ".got", ".plt", ".rela.dyn"] {
                assert!(sh.iter().all(|s| s.name != bad), "{bad} present");
            }
            // entry = _start
            let syms = symbols(b);
            let start = syms.iter().find(|s| s.0 == stub::START).expect("_start symbol");
            assert_eq!(u64_at(b, 24), start.1);
            assert_eq!(img.entry_addr(), start.1);
            assert!(syms.iter().any(|s| s.0 == stub::RUNTIME_ENTRY && s.3 == 0x12));
            assert!(syms.iter().any(|s| s.0 == "hello.msg" && s.3 == 0x11 && s.2 == 3));

            if let Some(out) = readelf(&img, &format!("hello-{}-{k}", tag(m)), &["-lhSsW"]) {
                assert!(out.contains("EXEC (Executable file)"), "{out}");
                assert!(!out.contains("INTERP") && !out.contains("DYNAMIC"), "{out}");
                assert!(!out.contains("Requesting program interpreter"), "{out}");
                let mach = if m == Machine::X86_64 { "X86-64" } else { "AArch64" };
                assert!(out.contains(mach), "{out}");
            }
            if let Some(out) = readelf(&img, &format!("hello-{}-{k}", tag(m)), &["-dW"]) {
                assert!(out.contains("There is no dynamic section"), "{out}");
                assert!(!out.contains("NEEDED"), "{out}");
            }
            if let Some((stdout, code)) = run(&img, &format!("hello-{}-{k}", tag(m)), &["a", "b"]) {
                assert_eq!(stdout, "hi\n");
                assert_eq!(code, 3, "exit_group(argc) with argc read through the sp passed by _start");
            }
        }
    }
}

#[test]
fn segments_and_sections() {
    for m in MACHINES {
        let mut b = ImageBuilder::new(m, Options::debug(m));
        let f = b.add_file("src", "x.stv");
        let entry = b.declare(stub::RUNTIME_ENTRY);
        let msg = b.rodata("msg", DataObj::new(b"hi\n".to_vec(), 1)).unwrap();
        b.rodata("table", DataObj::new(vec![7; 24], 8)).unwrap();
        let d = b.data("counter", DataObj::new(vec![1, 2, 3, 4, 5, 6, 7, 8], 8)).unwrap();
        b.bss("buf", 100_000, 64).unwrap();
        // a data pointer to a function and to data (Abs64)
        let mut ptrs = DataObj::new(vec![0; 16], 8);
        ptrs.relocs.push(Reloc { offset: 0, kind: RelocKind::Abs64, sym: entry, addend: 0 });
        ptrs.relocs.push(Reloc { offset: 8, kind: RelocKind::Abs64, sym: d, addend: 4 });
        let p = b.data("ptrs", ptrs).unwrap();
        b.add_start_stub(entry).unwrap();
        b.define_func(entry, hello_entry(m, msg, 3, f)).unwrap();
        let img = b.link().unwrap();
        let bytes = img.bytes();
        let ph = phdrs(bytes);
        let loads: Vec<_> = ph.iter().filter(|p| p.ty == PT_LOAD).collect();
        assert_eq!(loads.len(), 3);
        assert_eq!(loads.iter().map(|p| p.flags).collect::<Vec<_>>(), [4, 5, 6], "R, RX, RW");
        let page = m.default_page_size();
        for (i, l) in loads.iter().enumerate() {
            assert_eq!(l.align, page);
            assert_eq!(l.vaddr % page, l.offset % page, "congruent");
            assert!(l.memsz >= l.filesz);
            assert!(l.offset + l.filesz <= bytes.len() as u64);
            if i > 0 {
                let prev = loads[i - 1];
                assert!(l.vaddr >= (prev.vaddr + prev.memsz).next_multiple_of(page), "own pages");
            }
        }
        assert!(ph.iter().all(|p| p.flags & 3 != 3), "no W+X");
        assert!(loads[2].memsz - loads[2].filesz >= 100_000);

        let sh = shdrs(bytes);
        let names: Vec<_> = sh.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "",
                ".rodata",
                ".text",
                ".data",
                ".bss",
                ".debug_abbrev",
                ".debug_info",
                ".debug_line",
                ".symtab",
                ".strtab",
                ".shstrtab"
            ]
        );
        // every symbol lies inside its section
        for (n, v, sz, _) in symbols(bytes) {
            let s = sh.iter().find(|s| s.flags & 2 != 0 && v >= s.addr && v + sz <= s.addr + s.size);
            assert!(s.is_some(), "{n} at {v:#x}");
        }
        // Abs64 relocations in .data
        let pa = img.symbol_addr(p).unwrap();
        let data = sh.iter().find(|s| s.name == ".data").unwrap();
        let fo = (data.offset + (pa - data.addr)) as usize;
        assert_eq!(u64_at(bytes, fo), img.ref_addr(entry).unwrap());
        assert_eq!(u64_at(bytes, fo + 8), img.symbol_addr(d).unwrap() + 4);
        // jump table: the call in _start goes through the stub
        let jt = img.jump_table().unwrap();
        assert_eq!(jt.funcs.len(), 2);
        assert_ne!(img.ref_addr(entry), img.symbol_addr(entry));

        if let Some(out) = readelf(&img, &format!("seg-{}", tag(m)), &["-lhSsW"]) {
            assert!(out.contains(".bss"), "{out}");
        }
        if let Some((stdout, code)) = run(&img, &format!("seg-{}", tag(m)), &[]) {
            assert_eq!((stdout.as_str(), code), ("hi\n", 1));
        }
    }
}

#[test]
fn line_table_readelf() {
    for m in MACHINES {
        let img = hello(m, Options::release(m));
        let Some(out) = readelf(&img, &format!("lines-{}", tag(m)), &["--debug-dump=decodedline"]) else {
            return;
        };
        let entry = img.lookup(stub::RUNTIME_ENTRY).unwrap();
        let a = img.symbol_addr(entry).unwrap();
        let second = if m == Machine::X86_64 { 3 } else { 4 };
        assert!(out.contains("hello.stv"), "{out}");
        for (line, addr) in [(1, a), (2, a + second)] {
            let want = format!("{line:>16}{:>20}", format!("{addr:#x}"));
            let found = out.lines().any(|l| {
                let w: Vec<_> = l.split_whitespace().collect();
                w.len() >= 3
                    && w[0] == "hello.stv"
                    && w[1] == line.to_string()
                    && w[2] == format!("{addr:#x}")
            });
            assert!(found, "missing row {want}\n{out}");
        }
        let tool = if m == Machine::X86_64 { "addr2line" } else { "aarch64-linux-gnu-addr2line" };
        if !have(tool) {
            eprintln!("SKIP {tool} check: not installed");
            continue;
        }
        let p = tmp(&format!("lines-{}", tag(m)));
        img.write_file(&p).unwrap();
        let out = std::process::Command::new(tool)
            .arg("-e")
            .arg(&p)
            .args([a, a + second, a + second + 1].map(|x| format!("{x:#x}")))
            .output()
            .unwrap();
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.is_empty(), "{tool}: {err}");
        let got: Vec<_> = String::from_utf8_lossy(&out.stdout).lines().map(str::to_owned).collect();
        assert_eq!(got, ["./src/hello.stv:1", "./src/hello.stv:2", "./src/hello.stv:2"]);
    }
}

#[test]
fn deterministic() {
    for m in MACHINES {
        assert_eq!(hello(m, Options::debug(m)).bytes(), hello(m, Options::debug(m)).bytes());
        assert_eq!(hello(m, Options::release(m)).bytes(), hello(m, Options::release(m)).bytes());
    }
}

#[test]
fn patch_in_place_and_move() {
    for m in MACHINES {
        let mut img = hello(m, Options::debug(m));
        let entry = img.lookup(stub::RUNTIME_ENTRY).unwrap();
        let msg = img.lookup("hello.msg").unwrap();
        let slot0 = *img.slot(entry).unwrap();
        // shorter message: fits the slot
        let p = img.patch_func(entry, hello_entry(m, msg, 2, stave_obj::FileId(0))).unwrap();
        assert!(!p.moved);
        assert_eq!(img.slot(entry).unwrap().addr, slot0.addr);
        if let Some((stdout, _)) = run(&img, &format!("patch1-{}", tag(m)), &[]) {
            assert_eq!(stdout, "hi");
        }
        // grow past the slot: moves to the reserve, stub retargeted
        let mut big = hello_entry(m, msg, 3, stave_obj::FileId(0));
        let nop: &[u8] = if m == Machine::X86_64 { &[0x90] } else { &[0x1f, 0x20, 0x03, 0xd5] };
        let mut code = Vec::new();
        for _ in 0..(slot0.cap as usize * 2 / nop.len()) {
            code.extend_from_slice(nop);
        }
        let shift = code.len() as u32;
        code.extend_from_slice(&big.code);
        big.code = code;
        for r in &mut big.relocs {
            r.offset += shift;
        }
        for l in &mut big.lines {
            l.offset += shift;
        }
        let p = img.patch_func(entry, big.clone()).unwrap();
        assert!(p.moved);
        assert_ne!(img.slot(entry).unwrap().addr, slot0.addr);
        if let Some((stdout, code)) = run(&img, &format!("patch2-{}", tag(m)), &["x"]) {
            assert_eq!((stdout.as_str(), code), ("hi\n", 2));
        }
        // readelf still accepts the rewritten tail
        readelf(&img, &format!("patch2-{}", tag(m)), &["-lhSsW", "--debug-dump=decodedline"]);

        // without a jump table the same growth is an overflow error
        let mut img = hello(m, Options { slack: Options::debug(m).slack, ..Options::release(m) });
        let before = img.bytes().to_vec();
        let e = img.patch_func(entry, big).unwrap_err();
        assert!(matches!(e, ObjError::SlotOverflow { .. }), "{e}");
        assert_eq!(img.bytes(), &before[..], "failed patch leaves the image unchanged");
    }
}

#[test]
fn errors() {
    let m = Machine::X86_64;
    // undefined symbol
    let mut b = ImageBuilder::new(m, Options::release(m));
    let e = b.declare("e");
    let ghost = b.declare("ghost");
    b.define_func(
        e,
        FuncCode {
            code: vec![0xe8, 0, 0, 0, 0],
            relocs: vec![Reloc { offset: 1, kind: RelocKind::X86Pc32, sym: ghost, addend: -4 }],
            lines: vec![],
        },
    )
    .unwrap();
    b.set_entry(e).unwrap();
    assert_eq!(b.clone().link().unwrap_err(), ObjError::Undefined("ghost".into()));
    // redefinition
    assert!(matches!(b.define_func(e, FuncCode::default()), Err(ObjError::Redefined(_))));
    // no entry
    let mut b = ImageBuilder::new(m, Options::release(m));
    b.func("f", FuncCode { code: vec![0xc3], ..FuncCode::default() }).unwrap();
    assert_eq!(b.link().unwrap_err(), ObjError::NoEntry);
    // entry is data
    let mut b = ImageBuilder::new(m, Options::release(m));
    let d = b.rodata("d", DataObj::new(vec![1], 1)).unwrap();
    b.set_entry(d).unwrap();
    assert!(matches!(b.link(), Err(ObjError::NotAFunction(_))));
    // relocation out of bounds, wrong machine
    for (kind, off) in [(RelocKind::X86Pc32, 2), (RelocKind::A64Branch26, 0)] {
        let mut b = ImageBuilder::new(m, Options::release(m));
        let f = b.declare("f");
        b.define_func(
            f,
            FuncCode {
                code: vec![0xe8, 0, 0, 0, 0],
                relocs: vec![Reloc { offset: off, kind, sym: f, addend: 0 }],
                lines: vec![],
            },
        )
        .unwrap();
        b.set_entry(f).unwrap();
        assert!(b.link().is_err());
    }
    // bad alignment, bad options
    let mut b = ImageBuilder::new(m, Options::release(m));
    let x = b.declare("x");
    assert!(b.define_rodata(x, DataObj::new(vec![1], 3)).is_err());
    let mut b = ImageBuilder::new(m, Options { page_size: 100, ..Options::release(m) });
    let f = b.func("f", FuncCode { code: vec![0xc3], ..FuncCode::default() }).unwrap();
    b.set_entry(f).unwrap();
    assert!(matches!(b.link(), Err(ObjError::BadOption(_))));
    // line row outside the function, unknown file
    for (off, file) in [(1, 0), (0, 5)] {
        let mut b = ImageBuilder::new(m, Options::release(m));
        let f = b
            .func(
                "f",
                FuncCode {
                    code: vec![0xc3],
                    relocs: vec![],
                    lines: vec![stave_obj::LineRow {
                        offset: off,
                        file: stave_obj::FileId(file),
                        line: 1,
                        column: 0,
                    }],
                },
            )
            .unwrap();
        b.set_entry(f).unwrap();
        assert!(b.link().is_err());
    }
}

#[test]
fn edge_layouts() {
    for m in MACHINES {
        let mut b = ImageBuilder::new(m, Options::release(m));
        let f = b.func("f", FuncCode { code: vec![0; 4], ..FuncCode::default() }).unwrap();
        b.set_entry(f).unwrap();
        let big = b.rodata("big_align", DataObj::new(vec![1; 4], 1 << 23)).unwrap();
        b.data("d", DataObj::new(vec![2; 8], 8)).unwrap();
        let empty = b.data("empty_at_end", DataObj::new(Vec::new(), 1)).unwrap();
        let z = b.bss("z", 0, 1).unwrap();
        let img = b.link().unwrap();
        assert_eq!(img.symbol_addr(big).unwrap() % (1 << 23), 0);
        let d = img.data().unwrap();
        assert_eq!(img.symbol_addr(empty), Some(d.addr + 8));
        assert!(img.symbol_addr(z).is_some());
        let bytes = img.bytes();
        let sh = shdrs(bytes);
        let ro = sh.iter().find(|s| s.name == ".rodata").unwrap();
        assert_eq!(
            &bytes[ro.offset as usize..][(img.symbol_addr(big).unwrap() - ro.addr) as usize..][..4],
            &[1; 4]
        );
        // slot sizes that cannot be represented
        let mut b = ImageBuilder::new(
            m,
            Options {
                slack: Some(stave_obj::Slack { min: u32::MAX, percent: u32::MAX }),
                ..Options::release(m)
            },
        );
        let f = b.func("f", FuncCode { code: vec![0; 4], ..FuncCode::default() }).unwrap();
        b.set_entry(f).unwrap();
        assert_eq!(b.link().unwrap_err(), ObjError::TooLarge);
        let mut b = ImageBuilder::new(m, Options::release(m));
        let f = b.func("f", FuncCode { code: vec![0; 4], ..FuncCode::default() }).unwrap();
        b.set_entry(f).unwrap();
        b.bss("huge", u64::MAX, 8).unwrap();
        assert_eq!(b.link().unwrap_err(), ObjError::TooLarge);
    }
}

mod props {
    use super::*;
    use proptest::prelude::*;

    #[derive(Debug, Clone)]
    enum Obj {
        Func(usize),
        Ro(usize, u32),
        Data(usize, u32),
        Bss(u64, u32),
    }

    fn obj() -> impl Strategy<Value = Obj> {
        prop_oneof![
            (1usize..300).prop_map(Obj::Func),
            (0usize..100, 0u32..7).prop_map(|(n, a)| Obj::Ro(n, 1 << a)),
            (0usize..100, 0u32..7).prop_map(|(n, a)| Obj::Data(n, 1 << a)),
            (0u64..50_000, 0u32..13).prop_map(|(n, a)| Obj::Bss(n, 1 << a)),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        /// Any mix of objects links (or fails cleanly) into a well-formed layout.
        #[test]
        fn layout(objs in prop::collection::vec(obj(), 0..12), arm in any::<bool>(), dbg in any::<bool>(),
                  relocs in prop::collection::vec((0usize..12, 0u32..300, 0u8..6, -64i64..64), 0..6)) {
            let m = if arm { Machine::Aarch64 } else { Machine::X86_64 };
            let opts = if dbg { Options::debug(m) } else { Options::release(m) };
            let mut b = ImageBuilder::new(m, opts);
            let entry = b.func("entry", FuncCode { code: vec![0xc3; 8], ..FuncCode::default() }).unwrap();
            b.set_entry(entry).unwrap();
            let mut ids = vec![entry];
            for (i, o) in objs.iter().enumerate() {
                let n = format!("o{i}");
                let id = match o {
                    Obj::Func(len) => {
                        let mut f = FuncCode { code: vec![0; *len], ..FuncCode::default() };
                        // random relocations: must never panic
                        for &(t, off, k, a) in &relocs {
                            let kind = [RelocKind::X86Pc32, RelocKind::Abs64, RelocKind::A64Branch26,
                                RelocKind::A64AdrPage21, RelocKind::A64AddLo12,
                                RelocKind::A64LdstLo12 { shift: 3 }][k as usize];
                            let sym = stave_obj::SymbolId((t % (ids.len())) as u32);
                            f.relocs.push(Reloc { offset: off, kind, sym, addend: a * 4 });
                        }
                        b.func(&n, f).unwrap()
                    }
                    Obj::Ro(len, a) => b.rodata(&n, DataObj::new(vec![1; *len], *a)).unwrap(),
                    Obj::Data(len, a) => b.data(&n, DataObj::new(vec![2; *len], *a)).unwrap(),
                    Obj::Bss(len, a) => b.bss(&n, *len, *a).unwrap(),
                };
                ids.push(id);
            }
            let Ok(img) = b.link() else { return Ok(()) };
            let bytes = img.bytes();
            let page = m.default_page_size();
            for p in phdrs(bytes).iter().filter(|p| p.ty == PT_LOAD) {
                prop_assert_eq!(p.vaddr % page, p.offset % page);
                prop_assert!(p.offset + p.filesz <= bytes.len() as u64);
            }
            // symbols aligned, non-overlapping within each class
            let mut ranges: Vec<(u64, u64)> = Vec::new();
            for (i, o) in objs.iter().enumerate() {
                let id = ids[i + 1];
                let a = img.symbol_addr(id).unwrap();
                let sz = img.symbol_size(id).unwrap();
                let al = match o { Obj::Func(_) => 16, Obj::Ro(_, a) | Obj::Data(_, a) | Obj::Bss(_, a) => u64::from(*a) };
                prop_assert_eq!(a % al, 0);
                if sz > 0 {
                    ranges.push((a, a + sz));
                }
            }
            ranges.sort();
            for w in ranges.windows(2) {
                prop_assert!(w[0].1 <= w[1].0, "overlap {:?}", w);
            }
        }
    }
}

/// `Abs64Body` resolves to the function's slot (its first instruction) even with the jump table,
/// `Abs64` to its stub; moving the function on a patch rewrites the `Abs64Body` field.
#[test]
fn body_relocs_skip_the_jump_table() {
    for m in MACHINES {
        for opts in [Options::release(m), Options::debug(m)] {
            let jt = opts.jump_table;
            let mut b = ImageBuilder::new(m, opts);
            let file = b.add_file("src", "hello.stv");
            let entry = b.declare(stub::RUNTIME_ENTRY);
            let msg = b.rodata("hello.msg", DataObj::new(b"hi\n".to_vec(), 1)).unwrap();
            let mut t = DataObj::new(vec![0u8; 16], 8);
            t.relocs.push(Reloc { offset: 0, kind: RelocKind::Abs64, sym: entry, addend: 0 });
            t.relocs.push(Reloc { offset: 8, kind: RelocKind::Abs64Body, sym: entry, addend: 4 });
            let tab = b.rodata("table", t).unwrap();
            b.add_start_stub(entry).unwrap();
            b.define_func(entry, hello_entry(m, msg, 3, file)).unwrap();
            let mut img = b.link().unwrap();
            let read = |img: &stave_obj::ObjectImage, k: u64| {
                let r = img.rodata().unwrap();
                let a = img.symbol_addr(tab).unwrap() + k;
                u64_at(img.bytes(), (r.offset + (a - r.addr)) as usize)
            };
            let body = img.symbol_addr(entry).unwrap();
            assert_eq!(read(&img, 8), body + 4);
            assert_eq!(read(&img, 0), img.ref_addr(entry).unwrap());
            assert_eq!(read(&img, 0) != body, jt, "Abs64 goes through the stub iff the table is on");
            if !jt {
                continue;
            }
            let cap = img.slot(entry).unwrap().cap as usize;
            let mut big = hello_entry(m, msg, 3, file);
            let nop: &[u8] = if m == Machine::X86_64 { &[0x90] } else { &[0x1f, 0x20, 0x03, 0xd5] };
            let mut code: Vec<u8> =
                nop.iter().copied().cycle().take(cap * 2 / nop.len() * nop.len()).collect();
            let shift = code.len() as u32;
            code.extend_from_slice(&big.code);
            big.code = code;
            for r in &mut big.relocs {
                r.offset += shift;
            }
            for l in &mut big.lines {
                l.offset += shift;
            }
            let p = img.patch_func(entry, big).unwrap();
            assert!(p.moved);
            let moved = img.symbol_addr(entry).unwrap();
            assert_ne!(moved, body);
            assert_eq!(read(&img, 8), moved + 4, "body field follows the move");
            assert_eq!(read(&img, 0), img.ref_addr(entry).unwrap(), "stub address unchanged");
            if let Some((stdout, code)) = run(&img, &format!("bodyref-{}", tag(m)), &["x"]) {
                assert_eq!((stdout.as_str(), code), ("hi\n", 2));
            }
        }
    }
}

/// Linux syscall numbers (x86-64 `syscall_64.tbl`, AArch64 asm-generic `unistd.h`) of the runtime's syscall set.
const KERNEL_ABI: [(&str, u32, u32); 36] = [
    ("read", 0, 63),
    ("write", 1, 64),
    ("pread64", 17, 67),
    ("pwrite64", 18, 68),
    ("openat", 257, 56),
    ("close", 3, 57),
    ("fstat", 5, 80),
    ("lseek", 8, 62),
    ("fsync", 74, 82),
    ("fdatasync", 75, 83),
    ("ftruncate", 77, 46),
    ("fcntl", 72, 25),
    ("flock", 73, 32),
    ("mmap", 9, 222),
    ("munmap", 11, 215),
    ("mremap", 25, 216),
    ("mprotect", 10, 226),
    ("madvise", 28, 233),
    ("clone3", 435, 435),
    ("futex", 202, 98),
    ("exit_group", 231, 94),
    ("getrandom", 318, 278),
    ("clock_gettime", 228, 113),
    ("rt_sigaction", 13, 134),
    ("rt_sigprocmask", 14, 135),
    ("sigaltstack", 131, 132),
    ("sched_getaffinity", 204, 123),
    ("sched_yield", 24, 124),
    ("getdents64", 217, 61),
    ("unlinkat", 263, 35),
    ("mkdirat", 258, 34),
    ("renameat2", 316, 276),
    ("kill", 62, 129),
    ("getpid", 39, 172),
    ("clone", 56, 220),
    ("exit", 60, 93),
];

fn repo_file(rel: &str) -> String {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// `pub const NR_<NAME>: usize = <n>;` lines, in order.
fn nr_consts(text: &str) -> Vec<(String, u32)> {
    text.lines()
        .filter_map(|l| {
            let rest = l.trim().strip_prefix("pub const NR_")?;
            let (name, val) = rest.split_once(": usize = ")?;
            Some((name.to_ascii_lowercase(), val.trim_end_matches(';').trim().parse().ok()?))
        })
        .collect()
}

/// The entries of the `NR` array (call id order).
fn nr_array(text: &str) -> Vec<String> {
    let start = text.find("pub const NR: [usize;").expect("NR array");
    let body = &text[start..];
    let open = body.find("= [").expect("NR array init") + 3;
    let close = body[open..].find("];").expect("NR array end") + open;
    body[open..close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.strip_prefix("NR_").unwrap_or(s).to_ascii_lowercase())
        .collect()
}

/// Every syscall the runtime can make is in runtime/syscalls.list, with this arch's kernel
/// number, in list order (call id = line), and nothing else.
#[test]
fn runtime_syscalls_match_the_list() {
    let list: Vec<String> = repo_file("runtime/syscalls.list")
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(String::from)
        .collect();
    let abi: Vec<&str> = KERNEL_ABI.iter().map(|k| k.0).collect();
    assert_eq!(list, abi, "syscalls.list is the runtime's syscall set");
    for (dir, col) in [("x86_64", 0usize), ("aarch64", 1)] {
        let text = repo_file(&format!("runtime/{dir}/sys.stv"));
        let consts = nr_consts(&text);
        let names: Vec<&str> = consts.iter().map(|c| c.0.as_str()).collect();
        assert_eq!(names, abi, "{dir}: NR_* constants in list order");
        for ((name, nr), (_, x, a)) in consts.iter().zip(KERNEL_ABI) {
            let want = if col == 0 { x } else { a };
            assert_eq!(*nr, want, "{dir}: NR_{} = {nr}, kernel ABI {want}", name.to_ascii_uppercase());
        }
        assert_eq!(nr_array(&text), list, "{dir}: NR[id] follows the list order");
        assert!(text.contains(&format!("pub const NCALLS: usize = {};", list.len())), "{dir}: NCALLS");
        // Every raw syscall takes its number from the NR table, `syscallN(NR[id], ..)`, or names
        // one of its NR_* constants (`exit_group`, `sched_yield`: the panic path, which must not
        // trap, so no table lookup).
        let mut calls = 0;
        let mut direct = 0;
        for l in
            text.lines().filter(|l| !l.trim_start().starts_with("fn ") && !l.trim_start().starts_with("//"))
        {
            let mut rest = l;
            while let Some(i) = rest.find("syscall") {
                rest = &rest[i + 7..];
                let Some(d) = rest.bytes().next().filter(u8::is_ascii_digit) else { continue };
                if rest.as_bytes().get(1) != Some(&b'(') || d > b'6' {
                    continue;
                }
                let arg0 = rest[2..].split([',', ')']).next().unwrap_or("").trim();
                if arg0 == "NR[id]" {
                    calls += 1;
                    continue;
                }
                let named = arg0
                    .strip_prefix("NR_")
                    .is_some_and(|n| names.contains(&n.to_ascii_lowercase().as_str()));
                assert!(named, "{dir}: syscall number not from the table in `{}`", l.trim());
                direct += 1;
            }
        }
        assert_eq!(calls, 7, "{dir}: one wrapper per syscall0..syscall6");
        assert_eq!(direct, 2, "{dir}: exit_group and sched_yield");
    }
    // Cross-check x86-64 against the host's kernel headers when installed.
    if let Ok(h) = std::fs::read_to_string("/usr/include/x86_64-linux-gnu/asm/unistd_64.h") {
        for (name, x, _) in KERNEL_ABI {
            let def = format!("#define __NR_{name} {x}\n");
            assert!(h.contains(&def), "unistd_64.h lacks `{}`", def.trim());
        }
    }
    // The runtime calls the kernel only through `arch.sys` (no intrinsic syscall elsewhere).
    for f in ["os.stv", "start.stv", "panic.stv", "alloc.stv", "state.stv", "config.stv"] {
        let t = repo_file(&format!("runtime/{f}"));
        assert!(!t.contains("fn syscall"), "runtime/{f} declares a syscall intrinsic");
    }
}

/// `write_file` replaces the path (new inode, mode 0755) instead of writing through it: a
/// symlink there is replaced, not followed; a failed write leaves no file behind.
#[cfg(unix)]
#[test]
fn write_file_replaces_the_path() {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    let img = hello(Machine::X86_64, Options::release(Machine::X86_64));
    let p = tmp("wf-replace");
    std::fs::write(&p, b"old").unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o600)).unwrap();
    let ino = std::fs::metadata(&p).unwrap().ino();
    img.write_file(&p).unwrap();
    let md = std::fs::metadata(&p).unwrap();
    assert_ne!(md.ino(), ino, "the old file is replaced, not rewritten in place");
    assert_eq!(md.permissions().mode() & 0o777, 0o755);
    assert_eq!(std::fs::read(&p).unwrap(), img.bytes());
    // a symlink at the path
    let target = tmp("wf-link-target");
    std::fs::write(&target, b"keep").unwrap();
    let link = tmp("wf-link");
    let _ = std::fs::remove_file(&link);
    std::os::unix::fs::symlink(&target, &link).unwrap();
    img.write_file(&link).unwrap();
    assert_eq!(std::fs::read(&target).unwrap(), b"keep");
    assert!(std::fs::symlink_metadata(&link).unwrap().file_type().is_file());
    assert_eq!(std::fs::read(&link).unwrap(), img.bytes());
    // a missing directory: an error, and nothing left in the parent
    let missing = tmp("wf-missing-dir").join("out");
    assert!(img.write_file(&missing).is_err());
    let dir = p.parent().unwrap();
    let stray: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with(".wf-"))
        .collect();
    assert!(stray.is_empty(), "temporary files left: {stray:?}");
}
