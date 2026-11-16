//! Memory hints reach the machine code, debug and release (contracts/simd-ops.md "Memory
//! hints"). The programs are `tests/conformance/simd/pos_hints_stream.stv` and
//! `pos_hints_prefetch.stv`.
//!
//! - x86-64 (v2 and v3): a vector `store_stream` is `movntps`/`movntdq` (`v` forms at v3), a
//!   scalar one and a scalar `write_stream` `movnti`; each fn with streaming stores has exactly the
//!   `sfence`s the fence placement gives it (one at the loop exit of `fill`, the explicit one of
//!   `raw`); the prefetch hints are `prefetcht0/t1/t2/nta` (no `prefetchw`: v2 and v3 lack it),
//!   kept by the release optimizer although they compute nothing.
//! - AArch64: the prefetch hints are `prfm`, and streaming stores are plain stores (no `stnp`).
//!
//! x86-64 code is disassembled with GNU objdump; without it that half is skipped with a message.

use std::path::{Path, PathBuf};
use std::process::Command;

use stave_driver::exec::TempDir;
use stave_driver::pipeline::{self, Goal, Options, Output};
use stave_driver::shipped::Library;
use stave_obj::{FuncCode, ObjectImage};
use stave_target::{Arch, Triple};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn program(name: &str) -> PathBuf {
    repo().join("tests/conformance/simd").join(name)
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

/// The machine code of the fn with checked-IR path `path`.
fn code<'a>(out: &'a Output, path: &str) -> &'a FuncCode {
    let (m, img): (_, &ObjectImage) = (out.module.as_ref().unwrap(), out.image.as_ref().unwrap());
    let order = stave_ir::link::text_order(m);
    let names = stave_ir::link::func_symbols(m, &order);
    let f = m.find_func(path).unwrap_or_else(|| panic!("no fn {path}"));
    let name = names[f.index()].as_ref().unwrap_or_else(|| panic!("{path} has no symbol"));
    let s = img.lookup(name).unwrap_or_else(|| panic!("no symbol {name}"));
    img.func_code(s).unwrap_or_else(|| panic!("{name} has no code"))
}

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok_and(|o| o.status.success())
}

/// The mnemonics of `code`, in order.
fn x64_mnemonics(code: &[u8], tmp: &Path) -> Vec<String> {
    let p = tmp.join("code.bin");
    std::fs::write(&p, code).unwrap();
    let out = Command::new("objdump")
        .args(["-D", "-b", "binary", "-m", "i386:x86-64", "-M", "intel", "--no-show-raw-insn", "-w"])
        .arg(&p)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .filter_map(|l| l.split_once(":\t"))
        .filter_map(|(_, rest)| rest.split_whitespace().next().map(str::to_string))
        .collect()
}

fn count(ms: &[String], m: &str) -> usize {
    ms.iter().filter(|x| *x == m).count()
}

#[test]
fn x64_emits_non_temporal_stores_fences_and_prefetches() {
    if !have("objdump") {
        eprintln!("note: objdump not installed: x86-64 code not checked");
        return;
    }
    let tmp = TempDir::new("stream-hints").unwrap();
    for (target, v) in [("x86_64-linux", ""), ("x86_64-linux-v3", "v")] {
        let t = Triple::parse(target).unwrap();
        for release in [false, true] {
            let what = format!("{target}{}", if release { " release" } else { "" });
            let out = build(&program("pos_hints_stream.stv"), t, release);
            let ms = |path: &str| x64_mnemonics(&code(&out, path).code, tmp.path());
            let fill = ms("main.fill");
            assert_eq!(count(&fill, &format!("{v}movntps")), 1, "{what}: fill: {fill:?}");
            assert_eq!(count(&fill, "sfence"), 1, "{what}: fill: {fill:?}");
            let st = ms("main.strict_fill");
            assert_eq!(count(&st, &format!("{v}movntps")), 1, "{what}: strict_fill: {st:?}");
            assert_eq!(count(&st, "sfence"), 1, "{what}: strict_fill: {st:?}");
            let sc = ms("main.scalars");
            // u32 in the loop, then f64; the i8 store is plain
            assert_eq!(count(&sc, "movnti"), 2, "{what}: scalars: {sc:?}");
            assert_eq!(count(&sc, "sfence"), 1, "{what}: scalars: {sc:?}");
            let raw = ms("main.raw");
            assert_eq!(count(&raw, "movnti"), 1, "{what}: raw: {raw:?}");
            assert_eq!(count(&raw, &format!("{v}movntdq")), 1, "{what}: raw: {raw:?}");
            assert_eq!(count(&raw, "sfence"), 1, "{what}: raw: {raw:?}");

            let out = build(&program("pos_hints_prefetch.stv"), t, release);
            let ms = |path: &str| x64_mnemonics(&code(&out, path).code, tmp.path());
            let reads = ms("main.reads");
            for (m, n) in [("prefetcht0", 2), ("prefetcht1", 2), ("prefetcht2", 2), ("prefetchnta", 2)] {
                assert_eq!(count(&reads, m), n, "{what}: reads: {m}: {reads:?}");
            }
            let ptrs = ms("main.pointers");
            for (m, n) in [("prefetcht0", 2), ("prefetcht1", 2), ("prefetcht2", 2), ("prefetchnta", 2)] {
                assert_eq!(count(&ptrs, m), n, "{what}: pointers: {m}: {ptrs:?}");
            }
            assert!(!ptrs.iter().any(|m| m.starts_with("prefetchw")), "{what}: pointers: {ptrs:?}");
            let sum = ms("main.strict_sum");
            assert_eq!(count(&sum, "prefetchnta"), 1, "{what}: strict_sum: {sum:?}");
            assert_eq!(count(&sum, "prefetcht2"), 1, "{what}: strict_sum: {sum:?}");
        }
    }
}

fn words(c: &FuncCode) -> impl Iterator<Item = u32> + '_ {
    c.code.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b))
}

/// `prfm <op>, [xn, #imm]` (unsigned offset): the prefetch operation in bits 4:0.
fn prfm_op(w: u32) -> Option<u32> {
    (w & 0xffc0_0000 == 0xf980_0000).then_some(w & 0x1f)
}

/// `stnp` of general (32/64-bit) or SIMD&FP (s/d/q) registers.
fn is_stnp(w: u32) -> bool {
    matches!(w & 0xffc0_0000, 0x2800_0000 | 0xa800_0000 | 0x2c00_0000 | 0x6c00_0000 | 0xac00_0000)
}

#[test]
fn a64_emits_prfm_and_no_stnp() {
    let t = Triple::parse("aarch64-linux").unwrap();
    assert_eq!(t.arch(), Arch::Aarch64);
    for release in [false, true] {
        let what = if release { "release" } else { "debug" };
        let out = build(&program("pos_hints_prefetch.stv"), t, release);
        let mut ops: Vec<u32> = words(code(&out, "main.pointers")).filter_map(prfm_op).collect();
        ops.sort_unstable();
        // pldl1keep pldl2keep pldl3keep pldl1strm, pstl1keep pstl2keep pstl3keep pstl1strm
        assert_eq!(ops, [0, 1, 2, 4, 16, 17, 18, 20], "{what}: pointers");
        let reads = words(code(&out, "main.reads")).filter_map(prfm_op).count();
        assert_eq!(reads, 8, "{what}: reads");
        let out = build(&program("pos_hints_stream.stv"), t, release);
        for f in ["main.fill", "main.strict_fill", "main.scalars", "main.raw"] {
            assert!(!words(code(&out, f)).any(is_stnp), "{what}: {f} has an stnp");
        }
    }
}
