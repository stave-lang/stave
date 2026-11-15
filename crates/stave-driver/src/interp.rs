//! Runs a lowered program in the stave-ir reference interpreter (the oracle), the
//! way `_start` runs it natively: from `runtime.start.entry(sp)` with an initial process stack
//! (argv, empty envp and auxv), `main_addr` = a stand-in for `__stave_main`
//! ([`stave_sema::lower::interp_main_stub`]), the site table in the contracts/intrinsics.md layout,
//! and an empty PC table. A trap of the interpreter is delivered like the native trap stub does:
//! `runtime.panic.trap(kind, site)` (for `StackOverflow`, the site of the overflowing fn's name
//! token, which is what the native signal path reports through the PC table).
//!
//! Files: the in-memory file system of [`stave_ir::BufferHost`] (`files`, relative paths), so a
//! program reading `input.txt` from its working directory gets the same bytes.

use stave_ir::interp::{Config, Interp, Outcome, Val};
use stave_ir::{BufferHost, Module, PanicKind, Site, SyscallAbi};
use stave_target::{Arch, Triple};

use crate::exec::Ran;

/// Interpreter limits for one run.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_steps: u64,
    /// Call depth at which a call traps `StackOverflow`.
    pub max_depth: usize,
}

impl Default for Limits {
    fn default() -> Limits {
        Limits { max_steps: 20_000_000_000, max_depth: 20_000 }
    }
}

/// What an interpreted run produced, plus the syscall numbers it made (in order).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Interpreted {
    pub ran: Ran,
    pub syscalls: Vec<u64>,
    pub steps: u64,
}

/// Runs `m` (a module lowered for `target`) with `main_path` as `main`, argv `[prog, args..]`,
/// and `files` (relative path → bytes) readable through `openat`.
pub fn run(
    m: &Module,
    main_path: &str,
    target: Triple,
    args: &[&[u8]],
    files: &[(String, Vec<u8>)],
    limits: Limits,
) -> Result<Interpreted, String> {
    let abi = match target.arch() {
        Arch::X86_64 => SyscallAbi::X86_64,
        Arch::Aarch64 => SyscallAbi::Aarch64,
        Arch::Gpu => return Err(format!("no interpreter host for {target}")),
    };
    let mut m = m.clone();
    let stub =
        stave_sema::lower::interp_main_stub(&mut m, main_path).ok_or_else(|| format!("no `{main_path}`"))?;
    let find = |p: &str| m.find_func(p).ok_or_else(|| format!("no `{p}` in the module"));
    let entry = find("runtime.start.entry")?;
    let trap = find("runtime.panic.trap")?;
    let mut host = BufferHost::new(abi);
    host.values.insert("main_addr".into(), u64::from(stub.0));
    host.values.insert("signal_handler_addr".into(), 0);
    for (k, v) in files {
        host.files.insert(k.as_bytes().to_vec(), v.clone());
    }
    let cfg = Config { max_steps: limits.max_steps, max_depth: limits.max_depth, ..Config::default() };
    let mut it = Interp::new(&m, host, cfg).map_err(|e| format!("interp: {e:?}"))?;
    let n = stave_sema::lower::site_table_bytes(&m, 0).len() as u64;
    let st = it.mem.alloc(n.max(8), 8).ok_or("interp: no memory for the site table")?;
    let bytes = stave_sema::lower::site_table_bytes(&m, st);
    it.mem.write(st, &bytes).map_err(|e| format!("interp: site table: {e:?}"))?;
    let pc = it.mem.alloc(8, 8).ok_or("interp: no memory for the PC table")?;
    it.host.values.insert("site_table".into(), st);
    it.host.values.insert("pc_table".into(), pc);
    let mut argv: Vec<&[u8]> = vec![b"prog"];
    argv.extend_from_slice(args);
    let sp = it.process_stack(&argv, &[]).map_err(|e| format!("interp: process stack: {e:?}"))?;
    let mut out = it.run(entry, vec![Val::S(sp)]).map_err(|e| format!("interp: {e:?}"))?;
    if let Outcome::Trap(t) = out {
        let kind = PanicKind::ALL.iter().position(|k| *k == t.kind).unwrap_or(0) as u64;
        let site = if t.kind == PanicKind::StackOverflow {
            m.func(t.func).map_or(Site::NONE, |f| f.name_site)
        } else {
            t.site
        };
        out = it
            .run(trap, vec![Val::S(kind), Val::S(u64::from(site.0))])
            .map_err(|e| format!("interp: in the trap handler: {e:?}"))?;
    }
    let exit = match out {
        Outcome::Exit(c) => c,
        o => return Err(format!("interp: the program did not exit: {o:?}")),
    };
    let steps = it.steps();
    let h = it.into_host();
    Ok(Interpreted {
        ran: Ran { exit, stdout: h.stdout, stderr: h.stderr },
        syscalls: h.log.iter().map(|l| l.0).collect(),
        steps,
    })
}
