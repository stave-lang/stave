//! Running built executables: `stave run`, `stave test` run tests and `test fn`s.
//!
//! - [`can_run`]: whether this host runs a target's binaries natively: the same architecture
//!   and, for `x86_64-linux-v3`/`-v4` and `aarch64-linux+sve`/`+sve2`, the CPU features of the
//!   level. Anything else needs a `--runner` (otherwise E1001).
//! - [`run`]: runs a binary, optionally through a runner command (`qemu-aarch64`, `sde64 -icx
//!   --`: split at whitespace, the binary and its arguments appended), with captured or
//!   inherited stdio. A signal-terminated program reports `128 + signal`.
//! - [`test_fns`]: the `test fn`s of the root package, in canonical decl order.

use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use stave_ir::{FuncId, Module};
use stave_sema::{DeclFlags, DeclKind, Program};
use stave_target::{A64Ext, Triple, X86Level};

/// Whether binaries for `t` run natively on this host.
pub fn can_run(t: Triple) -> bool {
    match t {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        Triple::X86_64(level) => match level {
            X86Level::V2 => std::arch::is_x86_feature_detected!("sse4.2"),
            X86Level::V3 => {
                std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
                    && std::arch::is_x86_feature_detected!("bmi2")
                    && std::arch::is_x86_feature_detected!("f16c")
            }
            X86Level::V4 => {
                std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx512bw")
                    && std::arch::is_x86_feature_detected!("avx512vl")
                    && std::arch::is_x86_feature_detected!("avx512dq")
                    && std::arch::is_x86_feature_detected!("avx512cd")
            }
        },
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        Triple::Aarch64(ext) => match ext {
            A64Ext::Base => true,
            A64Ext::Sve => std::arch::is_aarch64_feature_detected!("sve"),
            A64Ext::Sve2 => std::arch::is_aarch64_feature_detected!("sve2"),
        },
        _ => {
            let _ = (X86Level::V2, A64Ext::Base);
            false
        }
    }
}

/// What a run produced. Exit status: the program's, or `128 + signal`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ran {
    pub exit: i32,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

/// How to run one binary.
#[derive(Clone, Debug, Default)]
pub struct RunSpec<'a> {
    pub args: &'a [String],
    /// `--runner CMD`: words split at whitespace; the binary and `args` follow.
    pub runner: Option<&'a str>,
    /// Working directory (`None`: inherited).
    pub cwd: Option<&'a Path>,
    /// Inherit stdin/stdout/stderr (`stave run`) instead of capturing them (stdin is then empty).
    pub inherit: bool,
}

/// The exit status of a finished process: its code, or `128 + signal`.
pub fn status_code(s: std::process::ExitStatus) -> i32 {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = s.signal() {
            return 128 + sig;
        }
    }
    s.code().unwrap_or(1)
}

/// The command that runs `bin` under `spec`.
pub fn command(bin: &Path, spec: &RunSpec<'_>) -> io::Result<Command> {
    let mut words = spec.runner.map(|r| r.split_whitespace().collect::<Vec<_>>()).unwrap_or_default();
    let mut cmd = if words.is_empty() {
        Command::new(bin)
    } else {
        let mut c = Command::new(words.remove(0));
        c.args(words).arg(bin);
        c
    };
    cmd.args(spec.args);
    if let Some(d) = spec.cwd {
        cmd.current_dir(d);
    }
    Ok(cmd)
}

/// Runs `bin` and waits for it. ETXTBSY (the file was just written by another thread that has
/// not closed it yet in a forked child) is retried.
pub fn run(bin: &Path, spec: &RunSpec<'_>) -> io::Result<Ran> {
    let mut tries = 0;
    loop {
        let mut cmd = command(bin, spec)?;
        let r = if spec.inherit {
            cmd.status().map(|s| Ran { exit: status_code(s), ..Ran::default() })
        } else {
            cmd.stdin(Stdio::null()).output().map(|o| Ran {
                exit: status_code(o.status),
                stdout: o.stdout,
                stderr: o.stderr,
            })
        };
        match r {
            Err(e) if e.raw_os_error() == Some(26) && tries < 50 => {
                tries += 1;
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            r => return r,
        }
    }
}

/// A fresh, empty, private directory under the system temp dir, removed on drop.
///
/// The temp dir is shared with other users, so the name is unpredictable (64 random bits) and
/// the directory is created exclusively with mode 0700: an existing entry of that name (a
/// planted directory or symlink) fails the create and a new name is tried. A pre-existing path
/// is never used or removed.
pub struct TempDir(PathBuf);

/// Names tried before [`TempDir::new`] gives up.
const TEMP_TRIES: u32 = 16;

/// 64 unpredictable bits: `/dev/urandom`, else std's randomly keyed hasher (seeded from the OS)
/// over the pid, the time, and a counter.
fn random_u64() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    use std::io::Read;
    let mut b = [0u8; 8];
    if std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut b)).is_ok() {
        return u64::from_le_bytes(b);
    }
    static N: AtomicU64 = AtomicU64::new(0);
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u32(std::process::id());
    h.write_u64(N.fetch_add(1, Ordering::Relaxed));
    if let Ok(t) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        h.write_u128(t.as_nanos());
    }
    h.finish()
}

impl TempDir {
    pub fn new(tag: &str) -> io::Result<TempDir> {
        TempDir::new_in(&std::env::temp_dir(), tag)
    }

    /// A fresh directory `stave-<tag>-<16 hex digits>` in `parent`.
    pub fn new_in(parent: &Path, tag: &str) -> io::Result<TempDir> {
        TempDir::create(parent, tag, random_u64)
    }

    fn create(parent: &Path, tag: &str, mut name: impl FnMut() -> u64) -> io::Result<TempDir> {
        let mut b = std::fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            b.mode(0o700);
        }
        let mut last = None;
        for _ in 0..TEMP_TRIES {
            let d = parent.join(format!("stave-{tag}-{:016x}", name()));
            match b.create(&d) {
                Ok(()) => return Ok(TempDir(d)),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => last = Some(e),
                Err(e) => return Err(e),
            }
        }
        Err(last.unwrap_or_else(|| io::Error::from(io::ErrorKind::AlreadyExists)))
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The `test fn`s of the root package: (checked-IR path, function), in canonical decl order.
/// The module must be lowered with `stave_sema::Mode::Test`.
pub fn test_fns(prog: &Program, m: &Module) -> Vec<(String, FuncId)> {
    let mut v = Vec::new();
    for (i, d) in prog.decls.iter().enumerate() {
        if d.kind != DeclKind::Fn || !d.flags.has(DeclFlags::TEST) {
            continue;
        }
        let root = prog.module(d.module).is_some_and(|md| prog.packages[md.package.idx()].is_root);
        if !root {
            continue;
        }
        let path = prog.decl_path(stave_sema::DeclId(i as u32));
        if let Some(f) = m.find_func(&path).filter(|f| m.func(*f).is_some_and(|x| x.body.is_some())) {
            v.push((path, f));
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runner_words() {
        let args = vec!["a".to_string(), "b c".to_string()];
        let spec = RunSpec { args: &args, runner: Some("qemu-aarch64  -cpu max"), ..RunSpec::default() };
        let c = command(Path::new("/x/bin"), &spec).unwrap();
        assert_eq!(c.get_program(), "qemu-aarch64");
        let a: Vec<_> = c.get_args().map(|s| s.to_string_lossy().into_owned()).collect();
        assert_eq!(a, ["-cpu", "max", "/x/bin", "a", "b c"]);
        let c = command(Path::new("/x/bin"), &RunSpec { args: &args, ..RunSpec::default() }).unwrap();
        assert_eq!(c.get_program(), "/x/bin");
    }

    #[test]
    fn host_baseline_runs() {
        if let Some(h) = Triple::host()
            && cfg!(target_os = "linux")
        {
            assert!(can_run(h));
        }
        assert!(!can_run(Triple::GpuSm86));
    }

    #[test]
    fn temp_dirs_are_removed() {
        let d = TempDir::new("t").unwrap();
        let p = d.path().to_path_buf();
        assert!(p.is_dir());
        drop(d);
        assert!(!p.exists());
    }

    #[test]
    fn temp_dirs_are_private_and_unpredictable() {
        let a = TempDir::new("t").unwrap();
        let b = TempDir::new("t").unwrap();
        assert_ne!(a.path(), b.path());
        let name = a.path().file_name().unwrap().to_str().unwrap().to_string();
        let hex = name.strip_prefix("stave-t-").unwrap();
        assert!(hex.len() == 16 && hex.bytes().all(|c| c.is_ascii_hexdigit()), "{name}");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(a.path()).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o700);
        }
        assert_ne!(random_u64(), random_u64());
    }

    #[test]
    fn existing_temp_paths_are_never_reused() {
        let parent = TempDir::new("parent").unwrap();
        let planted = parent.path().join("planted");
        std::fs::create_dir(&planted).unwrap();
        std::fs::write(planted.join("keep"), b"x").unwrap();
        let taken = parent.path().join(format!("stave-c-{:016x}", 1));
        #[cfg(unix)]
        std::os::unix::fs::symlink(&planted, &taken).unwrap();
        #[cfg(not(unix))]
        std::fs::create_dir(&taken).unwrap();
        // The taken name is skipped (not followed, not removed); the next one is used.
        let mut names = [1u64, 2].into_iter();
        let d = TempDir::create(parent.path(), "c", || names.next().unwrap_or(3)).unwrap();
        assert_eq!(d.path(), parent.path().join(format!("stave-c-{:016x}", 2)));
        assert!(planted.join("keep").is_file() && taken.join("keep").is_file());
        // Every name taken: an error, and nothing is touched.
        let e = TempDir::create(parent.path(), "c", || 1).err().unwrap();
        assert_eq!(e.kind(), io::ErrorKind::AlreadyExists);
        assert!(planted.join("keep").is_file() && taken.join("keep").is_file());
        drop(d);
        assert!(planted.join("keep").is_file());
    }
}
