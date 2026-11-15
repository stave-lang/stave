//! `c.select(a, b)` compiles to branch-free code on both CPU architectures, debug and release.
//! The programs are the `select_*.stv` tests of `tests/conformance/types/` and
//! `tests/conformance/run/`; their `pick*` fns (and `at_or_zero`) are straight-line code around
//! one `select`, with no loop in the source. In the machine code of each, every forward
//! conditional branch goes to a trap stub (a bounds or overflow check of an operand, ending in a
//! call of `runtime.panic.trap`), so none chooses between the operands; the only other
//! conditional branches are backward: the counted loops the backends emit to copy a value over
//! 16 words (and, in `select` itself, to select one over 64 words), whose condition is the word
//! counter. Each fn also contains a conditional move (`cmov` / `csel` / `fcsel`), so the select
//! was not folded away.
//!
//! AArch64 code is decoded here (fixed-width words); x86-64 code is disassembled with GNU
//! objdump, and without it the x86-64 half is skipped with a message.

use std::path::{Path, PathBuf};
use std::process::Command;

use stave_driver::exec::TempDir;
use stave_driver::pipeline::{self, Goal, Options, Output};
use stave_driver::shipped::Library;
use stave_ir::{Module, Op};
use stave_obj::{FuncCode, ObjectImage};
use stave_target::{Arch, Triple};

const TRAP: &str = "runtime.panic.trap";

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The buildable `select_*.stv` programs (the E0201 test is check-only).
fn programs() -> Vec<PathBuf> {
    let mut v = Vec::new();
    for dir in ["tests/conformance/types", "tests/conformance/run"] {
        let d = repo().join(dir);
        for e in std::fs::read_dir(&d).unwrap_or_else(|e| panic!("{}: {e}", d.display())) {
            let p = e.unwrap().path();
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            if name.starts_with("select_") && name.ends_with(".stv") && name != "select_mismatch.stv" {
                v.push(p);
            }
        }
    }
    v.sort();
    assert!(v.len() >= 6, "select programs: {v:?}");
    v
}

fn build(p: &Path, target: Triple, release: bool) -> Output {
    let mut o = Options::new(Goal::Build).threads(Some(4));
    o.target = Some(target);
    o.release = release;
    let out = pipeline::run(p, &Library::in_dir(&repo()), &o).unwrap();
    assert!(!out.has_errors(), "{}: diagnostics {:?}", p.display(), out.diagnostics);
    assert!(out.failure.is_none(), "{} ({target}): {}", p.display(), out.failure.as_ref().unwrap());
    out
}

/// Symbol names of the fns under test: a `pick*` or `at_or_zero` fn of the program's own
/// module whose body has a `select`.
fn checked_fns(m: &Module) -> Vec<String> {
    let order = stave_ir::link::text_order(m);
    let names = stave_ir::link::func_symbols(m, &order);
    let mut v = Vec::new();
    for f in &order {
        let func = &m.funcs[f.index()];
        let item = func.path.rsplit('.').next().unwrap_or("");
        if !(item.starts_with("pick") || item == "at_or_zero") || func.path.starts_with("std.") {
            continue;
        }
        let has_select = func.body.as_ref().is_some_and(|b| b.insts.op.contains(&Op::Select));
        assert!(has_select, "{} has no select", func.path);
        if let Some(n) = &names[f.index()] {
            v.push(n.clone());
        }
    }
    v
}

fn code<'a>(img: &'a ObjectImage, name: &str) -> &'a FuncCode {
    let s = img.lookup(name).unwrap_or_else(|| panic!("no symbol {name}"));
    img.func_code(s).unwrap_or_else(|| panic!("{name} has no code"))
}

fn calls_trap(img: &ObjectImage, c: &FuncCode, lo: u32, hi: u32) -> bool {
    c.relocs.iter().any(|r| r.offset >= lo && r.offset < hi && img.symbol_name(r.sym) == Some(TRAP))
}

// ---------------------------------------------------------------- AArch64

fn word(c: &[u8], at: usize) -> Option<u32> {
    c.get(at..at + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn sext(v: u32, bits: u32) -> i64 {
    let s = 64 - bits;
    ((u64::from(v) << s) as i64) >> s
}

/// The target offset of a conditional branch (`b.cond`, `cbz`/`cbnz`, `tbz`/`tbnz`) at `at`.
fn a64_cond_target(w: u32, at: usize) -> Option<i64> {
    let off = if w & 0xff00_0010 == 0x5400_0000 || w & 0x7e00_0000 == 0x3400_0000 {
        sext((w >> 5) & 0x7ffff, 19)
    } else if w & 0x7e00_0000 == 0x3600_0000 {
        sext((w >> 5) & 0x3fff, 14)
    } else {
        return None;
    };
    Some(at as i64 + off * 4)
}

/// Whether the code from `at` on is a trap stub: it reaches a `bl runtime.panic.trap` before
/// any return or unconditional branch.
fn a64_is_stub(img: &ObjectImage, c: &FuncCode, at: i64) -> bool {
    let Ok(mut k) = usize::try_from(at) else { return false };
    for _ in 0..8 {
        let Some(w) = word(&c.code, k) else { return false };
        if w & 0xfc00_0000 == 0x9400_0000 {
            return calls_trap(img, c, k as u32, k as u32 + 4);
        }
        if w & 0xfc00_0000 == 0x1400_0000 || w & 0xffff_fc1f == 0xd65f_0000 {
            return false;
        }
        k += 4;
    }
    false
}

fn check_a64(img: &ObjectImage, name: &str, what: &str) {
    let c = code(img, name);
    let mut moves = 0;
    for at in (0..c.code.len()).step_by(4) {
        let w = word(&c.code, at).unwrap_or(0);
        if w & 0x7fe0_0c00 == 0x1a80_0000 || w & 0xff20_0c00 == 0x1e20_0c00 {
            moves += 1;
        }
        if let Some(t) = a64_cond_target(w, at) {
            let ok = t < at as i64 || a64_is_stub(img, c, t);
            assert!(ok, "{what}: {name}: conditional branch at {at:#x} to {t:#x}");
        }
    }
    assert!(moves > 0, "{what}: {name}: no csel / fcsel");
}

// ---------------------------------------------------------------- x86-64

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok_and(|o| o.status.success())
}

/// `(offset, mnemonic, operands)` per instruction.
fn x64_disasm(code: &[u8], tmp: &Path) -> Vec<(u32, String, String)> {
    let p = tmp.join("code.bin");
    std::fs::write(&p, code).unwrap();
    let out = Command::new("objdump")
        .args(["-D", "-b", "binary", "-m", "i386:x86-64", "-M", "intel", "--no-show-raw-insn", "-w"])
        .arg(&p)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let mut v = Vec::new();
    for l in String::from_utf8(out.stdout).unwrap().lines() {
        let Some((a, rest)) = l.split_once(":\t") else { continue };
        let Ok(a) = u32::from_str_radix(a.trim(), 16) else { continue };
        let mut it = rest.split_whitespace();
        let mnem = it.next().unwrap_or("").to_string();
        let ops = it.next().unwrap_or("").to_string();
        v.push((a, mnem, ops));
    }
    v
}

fn x64_is_stub(img: &ObjectImage, c: &FuncCode, ins: &[(u32, String, String)], at: u32) -> bool {
    let Some(k) = ins.iter().position(|i| i.0 == at) else { return false };
    for j in k..(k + 8).min(ins.len()) {
        let next = ins.get(j + 1).map_or(c.code.len() as u32, |i| i.0);
        match ins[j].1.as_str() {
            "call" => return calls_trap(img, c, ins[j].0, next),
            "ret" | "jmp" => return false,
            _ => {}
        }
    }
    false
}

fn check_x64(img: &ObjectImage, name: &str, what: &str, tmp: &Path) {
    let c = code(img, name);
    let ins = x64_disasm(&c.code, tmp);
    assert!(ins.iter().any(|i| i.1.starts_with("cmov")), "{what}: {name}: no cmov");
    for (at, m, ops) in &ins {
        if !m.starts_with('j') || m == "jmp" {
            continue;
        }
        let t = ops.strip_prefix("0x").and_then(|h| u32::from_str_radix(h, 16).ok());
        let ok = t.is_some_and(|t| t < *at || x64_is_stub(img, c, &ins, t));
        assert!(ok, "{what}: {name}: conditional branch `{m} {ops}` at {at:#x}");
    }
}

#[test]
fn select_has_no_branch_on_its_condition() {
    let tmp = TempDir::new("select-branchless").unwrap();
    let objdump = have("objdump");
    if !objdump {
        eprintln!("note: objdump not installed: x86-64 code not checked");
    }
    let mut checked = 0;
    for p in programs() {
        for target in ["x86_64-linux", "aarch64-linux"].map(|t| Triple::parse(t).unwrap()) {
            if target.arch() == Arch::X86_64 && !objdump {
                continue;
            }
            for release in [false, true] {
                let out = build(&p, target, release);
                let (m, img) = (out.module.as_ref().unwrap(), out.image.as_ref().unwrap());
                let what = format!(
                    "{} ({target}{})",
                    p.file_name().unwrap().to_string_lossy(),
                    if release { ", release" } else { "" }
                );
                let fns = checked_fns(m);
                assert!(!fns.is_empty(), "{what}: no fn to check");
                for f in &fns {
                    match target.arch() {
                        Arch::X86_64 => check_x64(img, f, &what, tmp.path()),
                        _ => check_a64(img, f, &what),
                    }
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 0);
}
