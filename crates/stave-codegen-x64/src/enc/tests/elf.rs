//! End to end: hand-assembled programs linked by stave-obj into static ELF executables, run on this
//! host (x86-64 Linux), and inspected with readelf.

use std::process::Command;

use stave_obj::{DataObj, ImageBuilder, Machine, ObjectImage, Options, stub};

use super::{have, output_retry, tmp};
use crate::enc::*;

const RAX: Gpr = Gpr::RAX;
const RCX: Gpr = Gpr::RCX;
const RDX: Gpr = Gpr::RDX;
const RBX: Gpr = Gpr::RBX;
const RSP: Gpr = Gpr::RSP;
const RBP: Gpr = Gpr::RBP;
const RSI: Gpr = Gpr::RSI;
const RDI: Gpr = Gpr::RDI;

/// `runtime.start.entry(sp)`: `write(1, "hi\n", 3)` directly, then twice more through a helper
/// (direct call, and indirect through a `movabs` address), stores argc in .bss, adds a .data
/// constant with an overflow check, and exits with `argc + 10`.
fn program(opts: Options) -> ObjectImage {
    let mut b = ImageBuilder::new(Machine::X86_64, opts);
    let file = b.add_file("src", "hi.stv");
    let entry = b.declare(stub::RUNTIME_ENTRY);
    let write_msg = b.declare("hi.write_msg");
    let msg = b.rodata("hi.msg", DataObj::new(b"hi\n".to_vec(), 1)).unwrap();
    let ten = b.data("hi.ten", DataObj::new(10u64.to_le_bytes().to_vec(), 8)).unwrap();
    let argc = b.bss("hi.argc", 8, 8).unwrap();
    b.add_start_stub(entry).unwrap();

    let mut a = Asm::new();
    let trap = a.new_label();
    a.line(file, 1, 1);
    a.push(RBP);
    a.mov_rr(Size::Q, RBP, RSP);
    a.push(RBX);
    a.alu_ri(AluOp::Sub, Size::Q, RSP, 8).unwrap();
    a.line(file, 2, 5);
    a.mov_rm(Size::Q, RBX, Mem::base(RDI, 0)).unwrap();
    a.mov_mr(Size::Q, Mem::rip_sym(argc, 0), RBX).unwrap();
    // write(1, msg, 3)
    a.line(file, 3, 5);
    a.mov_ri(Size::Q, RAX, 1).unwrap();
    a.mov_ri(Size::D, RDI, 1).unwrap();
    a.lea(Size::Q, RSI, Mem::rip_sym(msg, 0)).unwrap();
    a.mov_ri(Size::Q, RDX, 3).unwrap();
    a.syscall();
    // write_msg(msg, 3), direct and indirect
    a.line(file, 4, 5);
    a.lea(Size::Q, RDI, Mem::rip_sym(msg, 0)).unwrap();
    a.mov_ri(Size::D, RSI, 3).unwrap();
    a.call_sym(write_msg);
    a.lea(Size::Q, RDI, Mem::rip_sym(msg, 0)).unwrap();
    a.mov_ri(Size::D, RSI, 3).unwrap();
    a.mov_r_sym(RAX, write_msg, 0);
    a.call_rm(RAX).unwrap();
    // exit_group(argc + ten), trapping on overflow
    a.line(file, 5, 5);
    a.mov_rm(Size::Q, RAX, Mem::rip_sym(argc, 0)).unwrap();
    a.mov_rm(Size::Q, RCX, Mem::rip_sym(ten, 0)).unwrap();
    a.alu_rr(AluOp::Add, Size::Q, RAX, RCX);
    a.jcc(Cond::O, trap).unwrap();
    a.mov_rr(Size::Q, RDI, RAX);
    a.mov_ri(Size::D, RAX, 231).unwrap();
    a.syscall();
    a.bind(trap).unwrap();
    a.ud2();
    b.define_func(entry, a.finish().unwrap()).unwrap();

    let mut w = Asm::new();
    w.line(file, 8, 1);
    w.mov_rr(Size::Q, RDX, RSI);
    w.mov_rr(Size::Q, RSI, RDI);
    w.mov_ri(Size::D, RDI, 1).unwrap();
    w.mov_ri(Size::D, RAX, 1).unwrap();
    w.syscall();
    w.ret();
    b.define_func(write_msg, w.finish().unwrap()).unwrap();
    b.link().unwrap()
}

fn readelf(path: &std::path::Path) -> Option<String> {
    if !have("readelf") {
        eprintln!("SKIP readelf: not installed");
        return None;
    }
    let out = Command::new("readelf").arg("-lhW").arg(path).output().unwrap();
    assert!(out.status.success());
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

#[test]
fn hello_runs() {
    for (name, opts) in
        [("release", Options::release(Machine::X86_64)), ("debug", Options::debug(Machine::X86_64))]
    {
        let img = program(opts);
        let p = tmp(&format!("hi-{name}"));
        img.write_file(&p).unwrap();
        if let Some(h) = readelf(&p) {
            assert!(h.contains("EXEC (Executable file)"), "{h}");
            assert!(h.contains("Advanced Micro Devices X86-64"), "{h}");
            assert!(!h.contains("INTERP") && !h.contains("DYNAMIC"), "{h}");
        }
        if !cfg!(all(target_os = "linux", target_arch = "x86_64")) {
            eprintln!("SKIP running x86-64 ELF: host is not x86-64 Linux");
            continue;
        }
        let out = output_retry(&mut Command::new(&p));
        assert_eq!(String::from_utf8_lossy(&out.stdout), "hi\nhi\nhi\n", "{name}");
        assert_eq!(out.status.code(), Some(11), "{name}: exit_group(argc + 10)");
        let out = output_retry(Command::new(&p).args(["a", "b"]));
        assert_eq!(out.status.code(), Some(13), "{name}");
    }
}

#[test]
fn line_rows_resolve() {
    let img = program(Options::release(Machine::X86_64));
    if !have("addr2line") {
        eprintln!("SKIP addr2line: not installed");
        return;
    }
    let p = tmp("hi-lines");
    img.write_file(&p).unwrap();
    let entry = img.lookup(stub::RUNTIME_ENTRY).unwrap();
    let wm = img.lookup("hi.write_msg").unwrap();
    let rows = &img.func_code(entry).unwrap().lines;
    let base = img.symbol_addr(entry).unwrap();
    let mut args: Vec<String> = rows.iter().map(|r| format!("{:#x}", base + u64::from(r.offset))).collect();
    args.push(format!("{:#x}", img.symbol_addr(wm).unwrap()));
    let out = Command::new("addr2line").arg("-e").arg(&p).args(&args).output().unwrap();
    assert!(out.stderr.is_empty(), "{}", String::from_utf8_lossy(&out.stderr));
    let got: Vec<String> = String::from_utf8_lossy(&out.stdout).lines().map(str::to_owned).collect();
    let mut want: Vec<String> = (1..=5).map(|l| format!("./src/hi.stv:{l}")).collect();
    want.push("./src/hi.stv:8".into());
    assert_eq!(got, want);
}
