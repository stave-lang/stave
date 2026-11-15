//! End to end: hand-assembled programs linked by stave-obj into static AArch64 ELF executables, run
//! under `qemu-aarch64` (or natively on an AArch64 host), and inspected with readelf.

use std::process::Command;

use stave_obj::{DataObj, ImageBuilder, Machine, ObjectImage, Options, Reloc, RelocKind, stub};

use super::{have, output_retry, tmp};
use crate::enc::*;

const X: Width = Width::X;
const SP: Reg = Reg::SP;

fn x(n: u8) -> Reg {
    Reg::x(n).unwrap()
}

/// `runtime.start.entry(sp)`: `write(1, "hi\n", 3)` directly, then through a helper (direct `bl`,
/// and `blr` through a function pointer stored in .data), keeps argc in .bss, adds a .data
/// constant with an overflow check, and exits with `argc + 10`.
fn program(opts: Options) -> ObjectImage {
    let mut b = ImageBuilder::new(Machine::Aarch64, opts);
    let file = b.add_file("src", "hi.stv");
    let entry = b.declare(stub::RUNTIME_ENTRY);
    let write_msg = b.declare("hi.write_msg");
    let msg = b.rodata("hi.msg", DataObj::new(b"hi\n".to_vec(), 1)).unwrap();
    let ten = b.data("hi.ten", DataObj::new(10u64.to_le_bytes().to_vec(), 8)).unwrap();
    let mut fp = DataObj::new(vec![0; 8], 8);
    fp.relocs.push(Reloc { offset: 0, kind: RelocKind::Abs64, sym: write_msg, addend: 0 });
    let fnptr = b.data("hi.fnptr", fp).unwrap();
    let argc = b.bss("hi.argc", 8, 8).unwrap();
    b.add_start_stub(entry).unwrap();

    let mut a = Asm::new();
    let trap = a.new_label();
    a.line(file, 1, 1);
    a.stp(X, Reg::FP, Reg::LR, Addr::Pre(SP, -32)).unwrap();
    a.mov(X, Reg::FP, SP).unwrap();
    a.str(MemSize::X, x(19), Addr::Off(SP, 16)).unwrap();
    a.line(file, 2, 5);
    a.ldr(MemSize::X, x(19), Addr::Off(x(0), 0)).unwrap();
    a.addr_of_sym(x(9), argc, 0).unwrap();
    a.str(MemSize::X, x(19), Addr::Off(x(9), 0)).unwrap();
    a.line(file, 3, 5);
    a.mov_imm(X, x(8), 64).unwrap();
    a.mov_imm(X, x(0), 1).unwrap();
    a.addr_of_sym(x(1), msg, 0).unwrap();
    a.mov_imm(X, x(2), 3).unwrap();
    a.svc(0);
    a.line(file, 4, 5);
    a.addr_of_sym(x(0), msg, 0).unwrap();
    a.mov_imm(X, x(1), 3).unwrap();
    a.bl_sym(write_msg);
    a.adrp_sym(x(16), fnptr, 0).unwrap();
    a.ldr_lo12_sym(MemSize::X, x(16), x(16), fnptr, 0).unwrap();
    a.addr_of_sym(x(0), msg, 0).unwrap();
    a.mov_imm(X, x(1), 3).unwrap();
    a.blr(x(16)).unwrap();
    a.line(file, 5, 5);
    a.adrp_sym(x(9), argc, 0).unwrap();
    a.ldr_lo12_sym(MemSize::X, x(0), x(9), argc, 0).unwrap();
    a.adrp_sym(x(10), ten, 0).unwrap();
    a.ldr_lo12_sym(MemSize::X, x(10), x(10), ten, 0).unwrap();
    a.addsub_reg(AddSub::Adds, X, x(0), x(0), x(10), Shift::Lsl, 0).unwrap();
    a.b_cond(Cond::Vs, trap).unwrap();
    a.mov_imm(X, x(8), 94).unwrap();
    a.svc(0);
    a.bind(trap).unwrap();
    a.brk(1);
    b.define_func(entry, a.finish().unwrap()).unwrap();

    let mut w = Asm::new();
    w.line(file, 8, 1);
    w.mov(X, x(2), x(1)).unwrap();
    w.mov(X, x(1), x(0)).unwrap();
    w.mov_imm(X, x(0), 1).unwrap();
    w.mov_imm(X, x(8), 64).unwrap();
    w.svc(0);
    w.ret();
    b.define_func(write_msg, w.finish().unwrap()).unwrap();
    b.link().unwrap()
}

#[test]
fn hello_runs() {
    for (name, opts) in
        [("release", Options::release(Machine::Aarch64)), ("debug", Options::debug(Machine::Aarch64))]
    {
        let img = program(opts);
        let p = tmp(&format!("hi-{name}"));
        img.write_file(&p).unwrap();
        if have("readelf") {
            let out = Command::new("readelf").arg("-lhW").arg(&p).output().unwrap();
            let h = String::from_utf8_lossy(&out.stdout);
            assert!(h.contains("EXEC (Executable file)"), "{h}");
            assert!(h.contains("AArch64"), "{h}");
            assert!(!h.contains("INTERP") && !h.contains("DYNAMIC"), "{h}");
        } else {
            eprintln!("SKIP readelf: not installed");
        }
        let mut cmd = if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
            Command::new(&p)
        } else if have("qemu-aarch64") {
            let mut c = Command::new("qemu-aarch64");
            c.arg(&p);
            c
        } else {
            eprintln!("SKIP running AArch64 ELF: no qemu-aarch64 and not an AArch64 host");
            continue;
        };
        let out = output_retry(&mut cmd);
        assert_eq!(String::from_utf8_lossy(&out.stdout), "hi\nhi\nhi\n", "{name}");
        assert_eq!(out.status.code(), Some(11), "{name}: exit_group(argc + 10)");
        let out = output_retry(cmd.args(["a", "b"]));
        assert_eq!(out.status.code(), Some(13), "{name}");
    }
}

#[test]
fn line_rows_resolve() {
    let tool = "aarch64-linux-gnu-addr2line";
    if !have(tool) {
        eprintln!("SKIP {tool}: not installed");
        return;
    }
    let img = program(Options::release(Machine::Aarch64));
    let p = tmp("hi-lines");
    img.write_file(&p).unwrap();
    let entry = img.lookup(stub::RUNTIME_ENTRY).unwrap();
    let wm = img.lookup("hi.write_msg").unwrap();
    let rows = &img.func_code(entry).unwrap().lines;
    let base = img.symbol_addr(entry).unwrap();
    let mut args: Vec<String> = rows.iter().map(|r| format!("{:#x}", base + u64::from(r.offset))).collect();
    args.push(format!("{:#x}", img.symbol_addr(wm).unwrap()));
    let out = Command::new(tool).arg("-e").arg(&p).args(&args).output().unwrap();
    assert!(out.stderr.is_empty(), "{}", String::from_utf8_lossy(&out.stderr));
    let got: Vec<String> = String::from_utf8_lossy(&out.stdout).lines().map(str::to_owned).collect();
    let mut want: Vec<String> = (1..=5).map(|l| format!("./src/hi.stv:{l}")).collect();
    want.push("./src/hi.stv:8".into());
    assert_eq!(got, want);
}
