//! Encoder tests. Encodings are checked against GNU `aarch64-linux-gnu-objdump` when installed;
//! without it those tests print `SKIP` and only check that encoding succeeds.

mod cases;
mod elf;
mod props;
mod random;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

use super::{Asm, EncError};

pub(super) const OBJDUMP: &str = "aarch64-linux-gnu-objdump";

static SEQ: AtomicU32 = AtomicU32::new(0);

/// A scratch path in a directory of its own; the directory is removed when this is dropped.
pub(super) struct Tmp(PathBuf);

pub(super) fn tmp(name: &str) -> Tmp {
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let d = std::env::temp_dir().join(format!("stave-a64-test-{}-{seq}", std::process::id()));
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

pub(super) fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok_and(|o| o.status.success())
}

/// Collapses whitespace, drops `// comment` and `<symbol>` annotations.
fn norm(s: &str) -> String {
    let s = s.split("//").next().unwrap_or("");
    let mut out = String::new();
    let mut depth = 0;
    for c in s.chars() {
        match c {
            '<' => depth += 1,
            '>' if depth > 0 => depth -= 1,
            _ if depth > 0 => {}
            _ => out.push(c),
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Disassembles raw A64 words: normalized text per instruction, or `None` without objdump.
pub(super) fn objdump(code: &[u8]) -> Option<Vec<String>> {
    if !have(OBJDUMP) {
        return None;
    }
    let p = tmp("raw");
    std::fs::write(&p, code).unwrap();
    let out = Command::new(OBJDUMP)
        .args(["-D", "-b", "binary", "-m", "aarch64", "--no-show-raw-insn", "-w"])
        .arg(&p)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8(out.stdout).unwrap();
    let mut v = Vec::new();
    for l in text.lines() {
        let Some((a, rest)) = l.split_once(":\t") else { continue };
        if u32::from_str_radix(a.trim(), 16).is_err() {
            continue;
        }
        v.push(norm(rest));
    }
    Some(v)
}

/// Encodes with `f` and compares objdump's disassembly with `want`, one entry per instruction.
pub(super) fn check(f: impl FnOnce(&mut Asm) -> Result<(), EncError>, want: &[&str]) {
    let mut a = Asm::new();
    f(&mut a).unwrap();
    let code = a.finish().unwrap().code;
    check_bytes(&code, &want.iter().map(|s| norm(s)).collect::<Vec<_>>());
}

pub(super) fn check_bytes(code: &[u8], want: &[String]) {
    let Some(got) = objdump(code) else {
        eprintln!("SKIP objdump comparison: {OBJDUMP} not installed");
        return;
    };
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g, w, "instruction {i} (offset {:#x}): got `{g}`, want `{w}`", i * 4);
    }
    assert_eq!(got.len(), want.len(), "instruction count\n got: {got:?}\nwant: {want:?}");
}

/// Runs `cmd`, retrying while the kernel reports ETXTBSY: a parallel test may have forked while
/// this executable was still open for writing.
pub(super) fn output_retry(cmd: &mut std::process::Command) -> std::process::Output {
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
