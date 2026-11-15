//! Running built executables: `stave run`, `stave test` run tests and `test fn`s.
//!
//! - [`can_run`]: whether this host runs a target's binaries natively: the same architecture
//!   and, for `x86_64-linux-v3`/`-v4` and `aarch64-linux+sve`/`+sve2`, the CPU features of the
//!   level. Anything else needs a `--runner` (otherwise E1001).
//! - [`run`]: runs a binary file, optionally through a runner command (`qemu-aarch64`, `sde64 -icx
//!   --`: split at whitespace, the binary and its arguments appended), with captured or
//!   inherited stdio. A signal-terminated program reports `128 + signal`.
//! - [`Executable`]: a linked program that runs without ever being a file. On x86-64 and AArch64
//!   Linux its ELF image lives in an anonymous memory file (`memfd_create`, close-on-exec). A
//!   native run forks and the child calls `execveat(fd, "", argv, envp, AT_EMPTY_PATH)`; the
//!   descriptor closes in the new program. A runner gets the path `/dev/fd/N`, and only its
//!   child keeps descriptor `N` open across the exec: the flag is cleared after the fork, so
//!   the parent's descriptor stays close-on-exec and no other child started meanwhile inherits
//!   the image. Both ways the child is started and awaited like a file ([`run`]): same stdio,
//!   working directory, environment, and exit status. A kernel without one of the two system
//!   calls (ENOSYS, nothing else) makes this process fall back to a temporary file per program
//!   ([`fallback_note`]); other systems always use the file.
//! - [`test_fns`]: the `test fn`s of the root package, in canonical decl order.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
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

/// The words of the runner command; none for a native run.
fn runner_words<'a>(spec: &RunSpec<'a>) -> Vec<&'a str> {
    spec.runner.map(|r| r.split_whitespace().collect()).unwrap_or_default()
}

/// The command that runs `bin` under `spec`.
pub fn command(bin: &Path, spec: &RunSpec<'_>) -> io::Result<Command> {
    let mut words = runner_words(spec);
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
    run_command(|| command(bin, spec), spec)
}

/// Starts the command `make` builds and waits for it, with the stdio of `spec`; ETXTBSY is
/// retried with a new command.
fn run_command(mut make: impl FnMut() -> io::Result<Command>, spec: &RunSpec<'_>) -> io::Result<Ran> {
    let mut tries = 0;
    loop {
        let mut cmd = make()?;
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

/// The system call whose absence (ENOSYS) made this process run programs from temporary files.
static FALLBACK: OnceLock<&'static str> = OnceLock::new();

/// Why programs of this process run from temporary files instead of memory, once that is so.
pub fn fallback_note() -> Option<String> {
    FALLBACK.get().map(|call| {
        format!("the kernel has no `{call}` (ENOSYS): programs run from temporary files, not from memory")
    })
}

/// Where an [`Executable`] holds its ELF image.
enum Image {
    /// An anonymous memory file, close-on-exec.
    Memory(File),
    /// A file (mode 0755) in a private temporary directory.
    OnDisk { path: PathBuf, _dir: TempDir },
}

/// A linked program, ready to run (module docs).
pub struct Executable {
    /// `argv[0]` of a native run from memory, and the file name on disk.
    name: String,
    image: Image,
}

impl Executable {
    /// Takes the ELF image `elf` of the program `name`.
    pub fn new(name: &str, elf: &[u8]) -> io::Result<Executable> {
        let name = name.to_string();
        if sys::IN_MEMORY && FALLBACK.get().is_none() {
            match sys::memory_file(&name, elf) {
                Ok(f) => return Ok(Executable { name, image: Image::Memory(f) }),
                Err(e) if sys::no_such_call(&e) => {
                    let _ = FALLBACK.set("memfd_create");
                }
                Err(e) => return Err(e),
            }
        }
        let image = on_disk(&name, elf)?;
        Ok(Executable { name, image })
    }

    /// Whether the image is in memory (no file was written for it).
    pub fn in_memory(&self) -> bool {
        matches!(self.image, Image::Memory(_))
    }

    /// Runs the program and waits for it, like [`run`].
    pub fn run(&mut self, spec: &RunSpec<'_>) -> io::Result<Ran> {
        let f = match &self.image {
            Image::OnDisk { path, .. } => return run(path, spec),
            Image::Memory(f) => f,
        };
        match run_command(|| sys::command(f, &self.name, spec), spec) {
            // Only a native run makes the call, in the child, which then never became the program.
            Err(e) if sys::no_such_call(&e) && runner_words(spec).is_empty() => {}
            r => return r,
        }
        let mut elf = Vec::new();
        let mut r = f;
        r.seek(SeekFrom::Start(0))?;
        r.read_to_end(&mut elf)?;
        let image = on_disk(&self.name, &elf)?;
        self.image = image;
        let _ = FALLBACK.set("execveat");
        self.run(spec)
    }
}

/// Writes `elf` as an executable file named `name` in a fresh private temporary directory.
fn on_disk(name: &str, elf: &[u8]) -> io::Result<Image> {
    let dir = TempDir::new("run")?;
    // one path component, whatever the caller's name is
    let plain = !name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\\', '\0']);
    let path = dir.path().join(if plain { name } else { "program" });
    let mut o = std::fs::OpenOptions::new();
    o.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o755);
    }
    let mut f = o.open(&path)?;
    f.write_all(elf)?;
    // the creation mode is filtered by the umask
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        f.set_permissions(std::fs::Permissions::from_mode(0o755))?;
    }
    drop(f);
    Ok(Image::OnDisk { path, _dir: dir })
}

/// The Linux system calls behind [`Executable`]. They go through libc's `syscall`: the wrappers
/// `memfd_create` and `execveat` are missing from older C libraries.
#[cfg(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
mod sys {
    use std::ffi::{CString, c_char, c_long};
    use std::fs::File;
    use std::io::{self, Write};
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStringExt;
    use std::os::unix::process::CommandExt;
    use std::path::Path;
    use std::process::Command;

    use super::{RunSpec, runner_words};

    pub const IN_MEMORY: bool = true;

    #[cfg(target_arch = "x86_64")]
    mod nr {
        pub const FCNTL: std::ffi::c_long = 72;
        pub const MEMFD_CREATE: std::ffi::c_long = 319;
        pub const EXECVEAT: std::ffi::c_long = 322;
    }

    #[cfg(target_arch = "aarch64")]
    mod nr {
        pub const FCNTL: std::ffi::c_long = 25;
        pub const MEMFD_CREATE: std::ffi::c_long = 279;
        pub const EXECVEAT: std::ffi::c_long = 281;
    }

    const EINVAL: i32 = 22;
    const ENOSYS: i32 = 38;
    const MFD_CLOEXEC: c_long = 0x1;
    const MFD_EXEC: c_long = 0x10;
    const F_SETFD: c_long = 2;
    /// The descriptor flags without FD_CLOEXEC.
    const FD_KEEP: c_long = 0;
    const AT_EMPTY_PATH: c_long = 0x1000;
    /// Bytes of the program name used as the memory file's label (the kernel allows 249).
    const MAX_LABEL: usize = 200;

    unsafe extern "C" {
        fn syscall(num: c_long, ...) -> c_long;
    }

    /// The kernel does not have the system call.
    pub fn no_such_call(e: &io::Error) -> bool {
        e.raw_os_error() == Some(ENOSYS)
    }

    /// An anonymous, close-on-exec memory file holding `elf`. `name` only labels it
    /// (`/proc/<pid>/exe` of the program reads `/memfd:<name> (deleted)`).
    pub fn memory_file(name: &str, elf: &[u8]) -> io::Result<File> {
        let label: Vec<u8> = name.bytes().filter(|&b| b != 0).take(MAX_LABEL).collect();
        let label = CString::new(label).map_err(io::Error::other)?;
        let create = |flags: c_long| {
            // SAFETY: `label` is a NUL-terminated string alive across the call, the only memory
            // the kernel reads for it.
            let fd = unsafe { syscall(nr::MEMFD_CREATE, label.as_ptr(), flags) };
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: the kernel just returned this descriptor, so nothing else owns it.
            Ok(unsafe { File::from_raw_fd(fd as i32) })
        };
        // MFD_EXEC asks for what is otherwise only the default: `vm.memfd_noexec=1` makes a
        // memory file created without it non-executable. Kernels before 6.3 reject the flag.
        let mut f = match create(MFD_CLOEXEC | MFD_EXEC) {
            Err(e) if e.raw_os_error() == Some(EINVAL) => create(MFD_CLOEXEC)?,
            r => r?,
        };
        f.write_all(elf)?;
        Ok(f)
    }

    /// A NULL-terminated `char *` array (`argv`, `envp`) and the strings it points to.
    struct CStrings {
        _strings: Vec<CString>,
        ptrs: Vec<*const c_char>,
    }

    // SAFETY: `ptrs` points into the heap buffers of `_strings`, which the value owns and never
    // changes or frees before it is dropped: sending it moves both together.
    unsafe impl Send for CStrings {}
    // SAFETY: no method writes through the pointers: sharing it shares immutable bytes.
    unsafe impl Sync for CStrings {}

    impl CStrings {
        /// A string with a NUL byte cannot be passed to a program: an error, as in `Command`.
        fn new(strings: impl Iterator<Item = Vec<u8>>) -> io::Result<CStrings> {
            let strings: Vec<CString> = strings
                .map(|s| CString::new(s).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e)))
                .collect::<io::Result<_>>()?;
            let mut ptrs: Vec<*const c_char> = strings.iter().map(|s| s.as_ptr()).collect();
            ptrs.push(std::ptr::null());
            Ok(CStrings { _strings: strings, ptrs })
        }

        fn as_ptr(&self) -> *const *const c_char {
            self.ptrs.as_ptr()
        }
    }

    /// The command that runs the image in `file` under `spec`. Natively the forked child calls
    /// `execveat` on the descriptor itself, with `name` as `argv[0]` and this process's
    /// environment; a runner gets `/dev/fd/N`, with `N` kept open across its exec.
    pub fn command(file: &File, name: &str, spec: &RunSpec<'_>) -> io::Result<Command> {
        let fd = c_long::from(file.as_raw_fd());
        if !runner_words(spec).is_empty() {
            let mut cmd = super::command(Path::new(&format!("/dev/fd/{fd}")), spec)?;
            let keep_open = move || {
                // SAFETY: one system call on the child's own descriptor table, no memory passed.
                if unsafe { syscall(nr::FCNTL, fd, F_SETFD, FD_KEEP) } < 0 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            };
            // SAFETY: the closure runs in the forked child and only makes a system call and
            // reads errno: no allocation, no lock.
            unsafe { cmd.pre_exec(keep_open) };
            return Ok(cmd);
        }
        let args = std::iter::once(name).chain(spec.args.iter().map(String::as_str));
        let argv = CStrings::new(args.map(|a| a.as_bytes().to_vec()))?;
        let envp = CStrings::new(std::env::vars_os().map(|(k, v)| {
            let mut kv = k.into_vec();
            kv.push(b'=');
            kv.extend(v.into_vec());
            kv
        }))?;
        // The program of this command is never looked up: the child replaces itself in
        // `exec_image` or reports the error.
        let mut cmd = Command::new(name);
        if let Some(d) = spec.cwd {
            cmd.current_dir(d);
        }
        let exec_image = move || {
            // SAFETY: `fd` is open in the child (inherited by the fork), the path is an empty
            // NUL-terminated string, and both arrays are NULL-terminated arrays of
            // NUL-terminated strings owned by this closure, so alive in the child.
            unsafe { syscall(nr::EXECVEAT, fd, c"".as_ptr(), argv.as_ptr(), envp.as_ptr(), AT_EMPTY_PATH) };
            // the call returns only when it failed
            Err(io::Error::last_os_error())
        };
        // SAFETY: the closure runs in the forked child and only makes a system call and reads
        // errno: no allocation, no lock (the strings were built before the fork).
        unsafe { cmd.pre_exec(exec_image) };
        Ok(cmd)
    }
}

#[cfg(not(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64"))))]
mod sys {
    use std::fs::File;
    use std::io;
    use std::process::Command;

    use super::RunSpec;

    pub const IN_MEMORY: bool = false;

    pub fn no_such_call(_: &io::Error) -> bool {
        false
    }

    pub fn memory_file(_: &str, _: &[u8]) -> io::Result<File> {
        Err(io::ErrorKind::Unsupported.into())
    }

    pub fn command(_: &File, _: &str, _: &RunSpec<'_>) -> io::Result<Command> {
        Err(io::ErrorKind::Unsupported.into())
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

    /// Any ELF serves: the shell, once from its file and once from memory.
    #[cfg(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[test]
    fn memory_run_equals_file_run() {
        let sh = Path::new("/bin/sh");
        let Ok(elf) = std::fs::read(sh) else { return };
        let mut x = Executable::new("sh", &elf).unwrap();
        assert!(x.in_memory() && fallback_note().is_none());
        let scripts = [
            "echo out; echo err >&2; exit 7",
            "kill -TERM $$",
            "kill -KILL $$",
            "pwd; echo \"$1-$2-$#\"; echo \"$HOME\"",
            // the shell's own descriptors: the image's is not among them
            "ls /proc/$$/fd",
        ];
        for script in scripts {
            let args = ["-c", script, "sh", "a b", ""].map(String::from);
            let spec = RunSpec { args: &args, cwd: Some(Path::new("/")), ..RunSpec::default() };
            let mem = x.run(&spec).unwrap();
            assert_eq!(mem, run(sh, &spec).unwrap(), "{script}");
            assert!(x.in_memory());
        }
        let args = ["-c", "echo out; exit 7"].map(String::from);
        let spec = RunSpec { args: &args, ..RunSpec::default() };
        assert_eq!(x.run(&spec).unwrap(), Ran { exit: 7, stdout: b"out\n".to_vec(), stderr: Vec::new() });
        // A runner is given `/dev/fd/N` and opens it.
        let spec = RunSpec { args: &args, runner: Some("env -u NO_SUCH_VARIABLE"), ..RunSpec::default() };
        assert_eq!(x.run(&spec).unwrap(), Ran { exit: 7, stdout: b"out\n".to_vec(), stderr: Vec::new() });
        // An argument with a NUL byte is refused, as for a file.
        let bad = ["a\0b".to_string()];
        assert!(x.run(&RunSpec { args: &bad, ..RunSpec::default() }).is_err());
        assert!(run(sh, &RunSpec { args: &bad, ..RunSpec::default() }).is_err());
    }

    #[test]
    fn on_disk_images_run_and_are_removed() {
        let image = on_disk("../x", b"#!/bin/sh\nexit 5\n").unwrap();
        let Image::OnDisk { path, .. } = &image else { panic!("not a file") };
        let path = path.clone();
        assert_eq!(path.file_name().unwrap(), "program");
        let mut x = Executable { name: "x".to_string(), image };
        assert!(!x.in_memory());
        if cfg!(target_os = "linux") {
            assert_eq!(x.run(&RunSpec::default()).unwrap().exit, 5);
        }
        drop(x);
        assert!(!path.exists());
        let Image::OnDisk { path, .. } = on_disk("prog", b"").unwrap() else { panic!("not a file") };
        assert_eq!(path.file_name().unwrap(), "prog");
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
